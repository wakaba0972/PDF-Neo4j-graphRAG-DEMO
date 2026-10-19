from __future__ import annotations

import base64
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import threading
from typing import Any, Callable
import urllib.error

import pymupdf

from .chunking import PageText
from .env_store import load_env
from .graph_service import (
    GPT_6_LUNA_MODEL, GPT_6_LUNA_REASONING_EFFORTS, RunCancelled, RunControl, _api_url,
    _chat_response_content, _post_json,
)


HEADER_FOOTER_MARGIN_RATIO = 0.08
DEFAULT_RENDER_SCALE = 2.0
DEFAULT_VISION_MODEL = "gpt-6-luna"
DEFAULT_VISION_CONCURRENCY = 8
# Transcription rarely benefits from reasoning, so GPT-6 Luna defaults to none.
DEFAULT_VISION_REASONING_EFFORT = "none"
# HTTP errors that will fail identically on every page (bad key, no access,
# unknown model); stop sending requests instead of retrying each page.
VISION_ABORT_HTTP_CODES = {401, 403, 404}
# Which pages are sent to the vision model: every page, or only pages with a
# detected figure/table, only tables, or only diagrams (images and vector
# illustrations). Pages without a text layer, such as scans, are always sent.
VISION_SCOPE_ALL = "all"
VISION_SCOPE_FIGURES = "figures"
VISION_SCOPE_TABLES = "tables"
VISION_SCOPE_DIAGRAMS = "diagrams"
VISION_SCOPES = (VISION_SCOPE_ALL, VISION_SCOPE_FIGURES, VISION_SCOPE_TABLES, VISION_SCOPE_DIAGRAMS)
FIGURE_KINDS = ("table", "image", "drawing")
DIAGRAM_KINDS = ("image", "drawing")
# Element kinds that send a page to the vision model, per scope.
VISION_SCOPE_KINDS: dict[str, tuple[str, ...]] = {
    VISION_SCOPE_ALL: FIGURE_KINDS,
    VISION_SCOPE_FIGURES: FIGURE_KINDS,
    VISION_SCOPE_TABLES: ("table",),
    VISION_SCOPE_DIAGRAMS: DIAGRAM_KINDS,
}
VISION_SCOPE_SKIP_LABELS = {
    VISION_SCOPE_FIGURES: "無圖表",
    VISION_SCOPE_TABLES: "無表格",
    VISION_SCOPE_DIAGRAMS: "無圖片或向量圖",
}
VISION_SYSTEM_PROMPT = """You are an expert assistant for transcribing technical document pages. Convert the supplied page image into a JSON object with page_number and content fields.
Follow these rules strictly:
1. Transcribe all visible text completely, in its original language and script, without translating, summarizing, or omitting anything.
2. Convert every table, including specification tables, into Markdown table syntax.
3. For each diagram, figure, photo, exploded view, wiring diagram, flowchart, or screen illustration, write a concise description in the same language as the page text. State what the figure shows and list its labeled components, callouts, or steps, copying every label exactly as printed; never translate a label. Do not invent labels that are not visible.
4. Add content items in visual top-to-bottom order. Each item must have a type of text, table, or diagram and a string data field.
5. Return exactly one valid JSON object and no Markdown fences or explanation."""

# Characters that differ between Traditional and Simplified Chinese, in pairs.
_TRADITIONAL_CHARS = set("這個們來時為說國後發會對過還沒樣經開關門問間見話讓認請設車機長東書業電動網線圖頁點擊選單裝")
_SIMPLIFIED_CHARS = set("这个们来时为说国后发会对过还没样经开关门问间见话让认请设车机长东书业电动网线图页点击选单装")
LANGUAGE_LABELS = {
    "Traditional Chinese": "繁體中文",
    "Simplified Chinese": "簡體中文",
    "Japanese": "日文",
    "Korean": "韓文",
    "English": "英文",
}

logger = logging.getLogger(__name__)
ProgressCallback = Callable[[float, str], None]


@dataclass
class PdfExtraction:
    pages: list[PageText]
    empty_pages: list[int]
    vision_model: str | None = None
    vision_pages: list[int] = field(default_factory=list)
    # Pages that fell back to the PDF text layer, mapped to the failure reason.
    fallback_pages: dict[int, str] = field(default_factory=dict)
    vision_abort_reason: str | None = None
    vision_scope: str = VISION_SCOPE_ALL
    # Reasoning effort sent with each vision request (None: not a GPT-6 Luna model).
    vision_reasoning_effort: str | None = None
    # Document language stated in the vision prompt (None: not detected).
    language: str | None = None
    # Pages with a detected figure or table (only checked when vision is used).
    figure_pages: list[int] = field(default_factory=list)
    # Pages kept on the text layer because they have none of the scope's elements.
    text_layer_pages: list[int] = field(default_factory=list)

    def page_source(self, page_number: int) -> str:
        """``vision``, ``fallback`` (vision failed) or ``text`` (text layer by choice)."""
        if page_number in self.fallback_pages:
            return "fallback"
        return "vision" if page_number in self.vision_pages else "text"


class _VisionAborted(Exception):
    pass


def get_pdf_page_count(path: str | Path) -> int:
    pdf_path = Path(path)
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("僅支援 PDF 檔案")

    try:
        with pymupdf.open(pdf_path) as document:
            if document.needs_pass:
                raise ValueError("目前不支援加密 PDF")
            page_count = document.page_count
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("PDF 無法開啟，可能已損毀或加密") from exc

    if page_count < 1:
        raise ValueError("PDF 沒有任何頁面")
    return page_count


def _validate_page_range(
    page_count: int, start_page: int, end_page: int | None
) -> tuple[int, int]:
    if page_count < 1:
        raise ValueError("PDF 沒有任何頁面")
    if start_page < 1:
        raise ValueError("解析起始頁必須大於等於 1")
    last_page = page_count if end_page is None else end_page
    if last_page < start_page:
        raise ValueError("解析結束頁不得小於起始頁")
    if start_page > page_count or last_page > page_count:
        raise ValueError(f"解析頁碼超出範圍；PDF 共 {page_count} 頁")
    return start_page, last_page


def _content_clip(page: pymupdf.Page) -> pymupdf.Rect:
    page_rect = page.rect
    margin = page_rect.height * HEADER_FOOTER_MARGIN_RATIO
    return pymupdf.Rect(
        page_rect.x0,
        page_rect.y0 + margin,
        page_rect.x1,
        page_rect.y1 - margin,
    )


def render_page_to_base64(
    page: pymupdf.Page, scale: float = DEFAULT_RENDER_SCALE
) -> str:
    """Render a PDF page's body area as a Base64-encoded PNG image."""
    if scale <= 0:
        raise ValueError("影像渲染倍率必須大於 0")

    pixmap = page.get_pixmap(
        matrix=pymupdf.Matrix(scale, scale),
        clip=_content_clip(page),
        alpha=False,
    )
    return base64.b64encode(pixmap.tobytes("png")).decode("ascii")


def _validate_vision_result(value: Any, page_number: int) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or type(value.get("page_number")) is not int
        or value["page_number"] != page_number
    ):
        raise ValueError("模型回傳的頁碼或 JSON 結構不正確")
    content = value.get("content")
    if not isinstance(content, list):
        raise ValueError("模型回傳的 content 必須是陣列")
    validated: list[dict[str, str]] = []
    for item in content:
        if (
            not isinstance(item, dict)
            or item.get("type") not in {"text", "table", "diagram"}
            or not isinstance(item.get("data"), str)
        ):
            raise ValueError("模型回傳的 content 項目格式不正確")
        validated.append({"type": item["type"], "data": item["data"]})
    return {"page_number": page_number, "content": validated}


def detect_document_language(texts: Any) -> str | None:
    """Guess the main language of a document from its PDF text layer.

    Pages that hold only figure labels give the vision model too little text
    to tell Traditional Chinese from Japanese or Simplified Chinese, so the
    document-wide language is stated in the prompt instead.
    """
    kana = hangul = han = latin = traditional = simplified = 0
    for text in texts:
        for char in text:
            if "\u3040" <= char <= "\u30ff":
                kana += 1
            elif "\uac00" <= char <= "\ud7a3":
                hangul += 1
            elif "\u4e00" <= char <= "\u9fff":
                han += 1
                traditional += char in _TRADITIONAL_CHARS
                simplified += char in _SIMPLIFIED_CHARS
            elif char.isascii() and char.isalpha():
                latin += 1
    if hangul > max(han, latin) * 0.5 and hangul >= 20:
        return "Korean"
    if kana >= 20 and kana > han * 0.1:
        return "Japanese"
    if han >= 20 and han * 3 >= latin:
        if traditional > simplified:
            return "Traditional Chinese"
        if simplified > traditional:
            return "Simplified Chinese"
        return None
    if latin >= 100 and not (han or kana or hangul):
        return "English"
    return None


def vision_system_prompt(language: str | None = None) -> str:
    if not language:
        return VISION_SYSTEM_PROMPT
    return (
        f"{VISION_SYSTEM_PROMPT}\n6. This document is written in {language}. Write every diagram "
        f"description in {language}, even when the page contains only figure labels; "
        "never switch to another language or script."
    )


def resolve_vision_reasoning_effort(model: str | None, effort: str | None) -> str | None:
    """The reasoning effort sent with a vision request; ``None`` for non-Luna models."""
    if str(model or "").strip().casefold() != GPT_6_LUNA_MODEL:
        return None
    effort = effort or DEFAULT_VISION_REASONING_EFFORT
    if effort not in GPT_6_LUNA_REASONING_EFFORTS:
        raise ValueError(f"不支援的 Vision 推理強度：{effort}")
    return effort


def _analyze_page_image(
    image_base64: str,
    page_number: int,
    *,
    base_url: str,
    api_key: str,
    model: str = DEFAULT_VISION_MODEL,
    language: str | None = None,
    reasoning_effort: str | None = DEFAULT_VISION_REASONING_EFFORT,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": vision_system_prompt(language)},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Analyze page {page_number}."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        },
                    },
                ],
            },
        ],
        "response_format": {"type": "json_object"},
    }
    effort = resolve_vision_reasoning_effort(model, reasoning_effort)
    if effort:
        payload["reasoning_effort"] = effort
    # GPT-6 reasoning requests reject sampling parameters unless effort is none.
    if effort in (None, "none"):
        payload["temperature"] = 0.0
    response = _post_json(_api_url(base_url, "chat/completions"), payload, api_key)
    response_text, _ = _chat_response_content(response)
    try:
        result = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise ValueError("模型回傳的不是有效 JSON") from exc
    return _validate_vision_result(result, page_number)


def _serialize_vision_result(result: dict[str, Any]) -> str:
    blocks: list[tuple[str, str]] = []
    for item in result["content"]:
        data = item["data"].strip()
        if not data:
            continue
        blocks.append((item["type"], data))
    serialized: list[str] = []
    for index, (item_type, data) in enumerate(blocks):
        if index:
            previous_type = blocks[index - 1][0]
            serialized.append("\n\n" if "table" in {previous_type, item_type} else "\n")
        serialized.append(data)
    return "".join(serialized)


def _page_text(page: pymupdf.Page) -> str:
    try:
        return page.get_text("text", clip=_content_clip(page), sort=True).strip()
    except Exception:
        return ""


def _is_vision_abort_error(exc: BaseException) -> bool:
    cause = exc.__cause__
    return isinstance(cause, urllib.error.HTTPError) and cause.code in VISION_ABORT_HTTP_CODES


def extract_pdf_with_report(
    path: str | Path,
    start_page: int = 1,
    end_page: int | None = None,
    *,
    vision_model: str | None = DEFAULT_VISION_MODEL,
    vision_scope: str = VISION_SCOPE_ALL,
    vision_reasoning_effort: str | None = DEFAULT_VISION_REASONING_EFFORT,
    max_workers: int = DEFAULT_VISION_CONCURRENCY,
    progress_callback: ProgressCallback | None = None,
    control: RunControl | None = None,
) -> PdfExtraction:
    """Extract page text, using a vision model when one is selected.

    With ``vision_scope="figures"`` only pages that have a detected figure or
    table, or no text layer at all, are sent to the vision model; ``"tables"``
    and ``"diagrams"`` narrow this to tables or to images and vector
    illustrations. The other pages keep the PDF text layer and are listed in
    ``text_layer_pages``. ``figure_pages`` always lists pages with any element.
    Pages are rendered in the calling thread (PyMuPDF documents are not
    thread-safe) and only the vision API calls run concurrently. Any page whose
    vision analysis fails keeps the PDF text layer and is listed in
    ``fallback_pages``. ``control`` pauses before new pages are sent and
    raises ``RunCancelled`` on stop; requests already in flight cannot be
    interrupted and their results are discarded.
    """
    pdf_path = Path(path)
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("僅支援 PDF 檔案")
    if max_workers < 1:
        raise ValueError("Vision 並行數必須大於 0")
    if vision_scope not in VISION_SCOPES:
        raise ValueError(f"不支援的 Vision 解析範圍：{vision_scope}")

    try:
        document = pymupdf.open(pdf_path)
    except Exception as exc:
        raise ValueError("PDF 無法開啟，可能已損毀或加密") from exc

    vision_model = (vision_model or "").strip() or None
    effort = resolve_vision_reasoning_effort(vision_model, vision_reasoning_effort)
    report = PdfExtraction(
        pages=[], empty_pages=[], vision_model=vision_model, vision_scope=vision_scope,
        vision_reasoning_effort=effort,
    )
    plain_texts: dict[int, str] = {}
    vision_texts: dict[int, str] = {}
    stop = threading.Event()
    abort_reasons: list[str] = []
    if vision_model:
        env = load_env()
        base_url = env["MODEL_OPENAI_API_BASE"]
        api_key = env["MODEL_OPENAI_API_KEY"]

    def analyze(page_number: int, image: str) -> str:
        if control:
            control.wait_if_paused()
            control.check()
        if stop.is_set():
            raise _VisionAborted()
        try:
            result = _analyze_page_image(
                image, page_number, base_url=base_url, api_key=api_key,
                model=vision_model, language=report.language, reasoning_effort=effort,
            )
        except Exception as exc:
            if _is_vision_abort_error(exc):
                stop.set()
                abort_reasons.append(str(exc))
            raise
        return _serialize_vision_result(result)

    with document:
        if document.needs_pass:
            raise ValueError("目前不支援加密 PDF")
        first_page, last_page = _validate_page_range(
            document.page_count, start_page, end_page
        )
        total = last_page - first_page + 1
        completed = 0
        if vision_model:
            report.language = _document_language(document)

        def mark_done(page_number: int) -> None:
            nonlocal completed
            completed += 1
            if progress_callback:
                progress_callback(completed / total, f"已解析 {completed} / {total} 頁")

        def mark_fallback(page_number: int, reason: str) -> None:
            report.fallback_pages[page_number] = reason
            logger.warning("PDF 第 %s 頁改用文字層擷取：%s", page_number, reason)
            mark_done(page_number)

        def collect(futures: set[Future[str]], pending: dict[Future[str], int]) -> None:
            for future in futures:
                page_number = pending.pop(future)
                try:
                    vision_texts[page_number] = future.result()
                except _VisionAborted:
                    mark_fallback(page_number, "Vision 解析已中止")
                except RunCancelled:
                    raise
                except Exception as exc:
                    mark_fallback(page_number, str(exc) or type(exc).__name__)
                else:
                    mark_done(page_number)

        def wait_for_pages(pending: dict[Future[str], int]) -> None:
            done: set[Future[str]] = set()
            while not done:
                # Poll so a stop request is noticed while requests are in flight.
                if control:
                    control.check()
                done, _ = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)
            collect(done, pending)

        pool = ThreadPoolExecutor(max_workers=max_workers)
        try:
            pending: dict[Future[str], int] = {}
            for page_number in range(first_page, last_page + 1):
                if control:
                    control.wait_if_paused()
                    control.check()
                page = document.load_page(page_number - 1)
                plain_texts[page_number] = _page_text(page)
                if not vision_model:
                    mark_done(page_number)
                    continue
                wanted = VISION_SCOPE_KINDS[vision_scope]
                matched = page_has_figures(page, wanted)
                others = tuple(kind for kind in FIGURE_KINDS if kind not in wanted)
                if matched or (others and page_has_figures(page, others)):
                    report.figure_pages.append(page_number)
                if (
                    not matched and vision_scope != VISION_SCOPE_ALL
                    and plain_texts[page_number]
                ):
                    report.text_layer_pages.append(page_number)
                    mark_done(page_number)
                    continue
                if stop.is_set():
                    mark_fallback(page_number, "Vision 解析已中止")
                    continue
                try:
                    image = render_page_to_base64(page)
                except Exception as exc:
                    mark_fallback(page_number, f"頁面渲染失敗：{exc}")
                    continue
                pending[pool.submit(analyze, page_number, image)] = page_number
                # Bound the number of rendered images held in memory.
                if len(pending) >= max_workers * 2:
                    wait_for_pages(pending)
            while pending:
                wait_for_pages(pending)
        except RunCancelled:
            # Do not wait for in-flight requests; their results are discarded.
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        except Exception as exc:
            pool.shutdown(wait=False, cancel_futures=True)
            raise ValueError("PDF 無法解析，請確認檔案內容有效") from exc
        pool.shutdown()

    for page_number in range(first_page, last_page + 1):
        text = vision_texts.get(page_number) or plain_texts[page_number]
        if vision_texts.get(page_number):
            report.vision_pages.append(page_number)
        report.pages.append(PageText(page_number, text))
        if not text:
            report.empty_pages.append(page_number)
    if abort_reasons:
        report.vision_abort_reason = abort_reasons[0]
    return report


def extract_pdf(
    path: str | Path,
    start_page: int = 1,
    end_page: int | None = None,
    **options: Any,
) -> tuple[list[PageText], list[int]]:
    report = extract_pdf_with_report(path, start_page, end_page, **options)
    return report.pages, report.empty_pages


@dataclass(frozen=True)
class PageFigure:
    page: int
    kind: str  # "table", "image" or "drawing"
    bbox: tuple[float, float, float, float]


FIGURE_KIND_LABELS = {"table": "表格", "image": "圖片", "drawing": "向量圖"}
# Vector paths closer than this many points belong to the same illustration.
FIGURE_CLUSTER_TOLERANCE = 12.0
FIGURE_MIN_SIZE = 36.0
FIGURE_RENDER_SCALE = 2.0
FIGURE_PADDING = 6.0
# A table must have ruled top and bottom edges covering this share of its width;
# callout leader lines around an illustration are often mistaken for a grid.
TABLE_BORDER_COVERAGE = 0.9
TABLE_BORDER_TOLERANCE = 2.0


def _is_rule_drawing(drawing: dict[str, Any], page_width: float) -> bool:
    """Long horizontal/vertical lines and rectangles: frames, rules and table borders."""
    rect = drawing["rect"]
    if max(rect.width, rect.height) < page_width * 0.2:
        return False
    for item in drawing["items"]:
        if item[0] == "l":
            start, end = item[1], item[2]
            if abs(start.x - end.x) > 1 and abs(start.y - end.y) > 1:
                return False
        elif item[0] == "qu":
            if not item[1].is_rectangular:
                return False
        elif item[0] != "re":
            return False
    return True


def _horizontal_segments(drawings: list[dict[str, Any]]) -> list[tuple[float, float, float]]:
    """Horizontal lines and rectangle edges as ``(x0, x1, y)``."""
    segments: list[tuple[float, float, float]] = []
    for drawing in drawings:
        for item in drawing["items"]:
            if item[0] == "l":
                start, end = item[1], item[2]
                if abs(start.y - end.y) <= 1:
                    segments.append((min(start.x, end.x), max(start.x, end.x), (start.y + end.y) / 2))
            elif item[0] == "re" or (item[0] == "qu" and item[1].is_rectangular):
                rect = item[1] if item[0] == "re" else item[1].rect
                segments.extend(((rect.x0, rect.x1, rect.y0), (rect.x0, rect.x1, rect.y1)))
    return segments


def _edge_coverage(segments: list[tuple[float, float, float]], x0: float, x1: float, y: float) -> float:
    spans = sorted(
        (max(start, x0), min(end, x1)) for start, end, line_y in segments
        if abs(line_y - y) <= TABLE_BORDER_TOLERANCE and end > x0 and start < x1
    )
    covered, reached = 0.0, x0
    for start, end in spans:
        if end > reached:
            covered += end - max(start, reached)
            reached = end
    return covered / (x1 - x0) if x1 > x0 else 0.0


def _has_table_border(rect: pymupdf.Rect, segments: list[tuple[float, float, float]]) -> bool:
    return all(
        _edge_coverage(segments, rect.x0, rect.x1, y) >= TABLE_BORDER_COVERAGE
        for y in (rect.y0, rect.y1)
    )


def _merge_nearby(rects: list[pymupdf.Rect], tolerance: float) -> list[pymupdf.Rect]:
    merged = [pymupdf.Rect(rect) for rect in rects]
    changed = True
    while changed:
        changed = False
        result: list[pymupdf.Rect] = []
        for rect in merged:
            for index, existing in enumerate(result):
                if (existing + (-tolerance, -tolerance, tolerance, tolerance)).intersects(rect):
                    result[index] = existing | rect
                    changed = True
                    break
            else:
                result.append(rect)
        merged = result
    return merged


def _figure_candidates(
    page: pymupdf.Page, kinds: tuple[str, ...] = FIGURE_KINDS,
) -> list[tuple[str, pymupdf.Rect]]:
    page_rect = page.rect
    body = _content_clip(page)
    found: dict[str, list[pymupdf.Rect]] = {}
    drawings = page.get_drawings() if {"table", "drawing"} & set(kinds) else []
    if "table" in kinds:
        segments = _horizontal_segments(drawings)
        found["table"] = [
            rect for rect in (pymupdf.Rect(table.bbox) for table in page.find_tables().tables)
            if _has_table_border(rect, segments)
        ]
    if "image" in kinds:
        found["image"] = [pymupdf.Rect(info["bbox"]) & page_rect for info in page.get_image_info()]
    if "drawing" in kinds:
        found["drawing"] = _merge_nearby(
            [
                drawing["rect"] for drawing in drawings
                if not _is_rule_drawing(drawing, page_rect.width)
                and (drawing["rect"].width > 0 or drawing["rect"].height > 0)
            ],
            FIGURE_CLUSTER_TOLERANCE,
        )
    return [
        (kind, rect)
        for kind, rects in found.items()
        for rect in rects
        if not rect.is_empty and body.contains((rect.tl + rect.br) / 2)
        and (kind == "table" or min(rect.width, rect.height) >= FIGURE_MIN_SIZE)
    ]


def page_has_figures(page: pymupdf.Page, kinds: tuple[str, ...] = FIGURE_KINDS) -> bool:
    """Whether ``detect_page_figures`` would find an element of ``kinds``, checking tables last.

    ``find_tables()`` is by far the slowest check, so it only runs on pages
    without images or vector illustrations. Detection errors count as a
    figure so the page still goes to the vision model.
    """
    diagram_kinds = tuple(kind for kind in DIAGRAM_KINDS if kind in kinds)
    try:
        return bool(
            (diagram_kinds and _figure_candidates(page, diagram_kinds))
            or ("table" in kinds and _figure_candidates(page, ("table",)))
        )
    except Exception:
        logger.warning("PDF 第 %s 頁圖表偵測失敗，視為含圖表", page.number + 1, exc_info=True)
        return True


def detect_page_figures(page: pymupdf.Page) -> list[PageFigure]:
    """Locate tables, embedded images and vector illustrations on a page.

    Elements centred in the header/footer margin, images and drawings smaller
    than ``FIGURE_MIN_SIZE``, and elements mostly inside a larger one are
    skipped. Vector
    paths are grouped by proximity after dropping frames and rules, so a
    bordered note box is not mistaken for an illustration.
    """
    candidates = _figure_candidates(page)
    # Largest first, so an element mostly inside another one (an icon in a
    # table cell, a callout inside an illustration) is dropped as a duplicate.
    kept: list[tuple[str, pymupdf.Rect]] = []
    for kind, rect in sorted(candidates, key=lambda item: -item[1].get_area()):
        if not any((rect & other).get_area() > rect.get_area() / 2 for _, other in kept):
            kept.append((kind, rect))
    return [
        PageFigure(page.number + 1, kind, tuple(rect))
        for kind, rect in sorted(kept, key=lambda item: (item[1].y0, item[1].x0))
    ]


def render_pdf_figures(
    path: str | Path,
    page_numbers: list[int],
    *,
    limit: int = 12,
    scale: float = FIGURE_RENDER_SCALE,
) -> list[tuple[PageFigure, bytes]]:
    """Crop the figures on the given 1-based pages into PNG images, in page order."""
    results: list[tuple[PageFigure, bytes]] = []
    with pymupdf.open(Path(path)) as document:
        for page_number in dict.fromkeys(page_numbers):
            if not 1 <= page_number <= document.page_count:
                continue
            page = document.load_page(page_number - 1)
            for figure in detect_page_figures(page):
                if len(results) >= limit:
                    return results
                clip = (pymupdf.Rect(figure.bbox) + (
                    -FIGURE_PADDING, -FIGURE_PADDING, FIGURE_PADDING, FIGURE_PADDING,
                )) & page.rect
                pixmap = page.get_pixmap(
                    matrix=pymupdf.Matrix(scale, scale), clip=clip, alpha=False,
                )
                results.append((figure, pixmap.tobytes("png")))
    return results


def _open_pdf_page(document: pymupdf.Document, page_number: int) -> pymupdf.Page:
    if not 1 <= page_number <= document.page_count:
        raise ValueError(f"頁碼超出範圍；PDF 共 {document.page_count} 頁")
    return document.load_page(page_number - 1)


def _document_language(document: Any) -> str | None:
    """Detect the language from the whole document's text layer (about 1 s per 150 pages)."""
    return detect_document_language(
        _page_text(document.load_page(index)) for index in range(document.page_count)
    )


def render_pdf_page(path: str | Path, page_number: int, scale: float = 1.5) -> bytes:
    """Render a whole 1-based PDF page as PNG, for side-by-side review."""
    with pymupdf.open(Path(path)) as document:
        page = _open_pdf_page(document, page_number)
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
        return pixmap.tobytes("png")


def extract_page_text_layer(path: str | Path, page_number: int) -> str:
    """The PDF text layer of one 1-based page, as used when vision is skipped."""
    with pymupdf.open(Path(path)) as document:
        return _page_text(_open_pdf_page(document, page_number))


def analyze_pdf_page(
    path: str | Path, page_number: int, vision_model: str,
    reasoning_effort: str | None = DEFAULT_VISION_REASONING_EFFORT,
) -> str:
    """Run the vision model on one 1-based page and return the serialized text."""
    with pymupdf.open(Path(path)) as document:
        image = render_page_to_base64(_open_pdf_page(document, page_number))
        language = _document_language(document)
    env = load_env()
    result = _analyze_page_image(
        image, page_number, base_url=env["MODEL_OPENAI_API_BASE"],
        api_key=env["MODEL_OPENAI_API_KEY"], model=vision_model, language=language,
        reasoning_effort=reasoning_effort,
    )
    return _serialize_vision_result(result)


def _short_reason(reason: str, limit: int = 160) -> str:
    flattened = " ".join(reason.split())
    return flattened if len(flattened) <= limit else f"{flattened[:limit]}…"


def format_vision_report(report: PdfExtraction) -> str:
    """Summarize vision usage for the PDF parsing status message."""
    if not report.vision_model:
        return "未使用 Vision，僅擷取文字層。"
    language = (
        f"圖表描述語言：{LANGUAGE_LABELS.get(report.language, report.language)}。"
        if report.language else ""
    )
    suffix = language + (
        f"{len(report.text_layer_pages)} 頁"
        f"{VISION_SCOPE_SKIP_LABELS.get(report.vision_scope, '無圖表')}，直接使用文字層。"
        if report.text_layer_pages else ""
    )
    if not report.fallback_pages:
        return f"Vision（{report.vision_model}）解析 {len(report.vision_pages)} 頁。{suffix}"
    if report.vision_abort_reason:
        return (
            f"⚠️ Vision（{report.vision_model}）已中止：{_short_reason(report.vision_abort_reason)}；"
            f"{len(report.fallback_pages)} 頁改用文字層擷取。{suffix}"
        )
    pages = sorted(report.fallback_pages)
    shown = "、".join(map(str, pages[:20])) + ("…" if len(pages) > 20 else "")
    first_reason = _short_reason(report.fallback_pages[pages[0]])
    return (
        f"⚠️ Vision（{report.vision_model}）解析 {len(report.vision_pages)} 頁；"
        f"{len(pages)} 頁改用文字層擷取（第 {shown} 頁，例：{first_reason}）。{suffix}"
    )
