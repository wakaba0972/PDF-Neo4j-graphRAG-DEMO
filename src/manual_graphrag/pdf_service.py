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
from .graph_service import RunCancelled, RunControl, _api_url, _chat_response_content, _post_json


HEADER_FOOTER_MARGIN_RATIO = 0.08
DEFAULT_RENDER_SCALE = 2.0
DEFAULT_VISION_MODEL = "gpt-6-luna"
DEFAULT_VISION_CONCURRENCY = 8
# HTTP errors that will fail identically on every page (bad key, no access,
# unknown model); stop sending requests instead of retrying each page.
VISION_ABORT_HTTP_CODES = {401, 403, 404}
VISION_SYSTEM_PROMPT = """You are an expert assistant for transcribing technical document pages. Convert the supplied page image into a JSON object with page_number and content fields.
Follow these rules strictly:
1. Transcribe all visible text completely, in its original language, without translating, summarizing, or omitting anything.
2. Convert every table, including specification tables, into Markdown table syntax.
3. For each diagram, figure, photo, exploded view, wiring diagram, flowchart, or screen illustration, write a concise description in the same language as the page text. State what the figure shows and list its labeled components, callouts, or steps. Do not invent labels that are not visible.
4. Add content items in visual top-to-bottom order. Each item must have a type of text, table, or diagram and a string data field.
5. Return exactly one valid JSON object and no Markdown fences or explanation."""

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


def _analyze_page_image(
    image_base64: str,
    page_number: int,
    *,
    base_url: str,
    api_key: str,
    model: str = DEFAULT_VISION_MODEL,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": VISION_SYSTEM_PROMPT},
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
        "temperature": 0.0,
        "reasoning_effort": "none",
    }
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
    max_workers: int = DEFAULT_VISION_CONCURRENCY,
    progress_callback: ProgressCallback | None = None,
    control: RunControl | None = None,
) -> PdfExtraction:
    """Extract page text, using a vision model when one is selected.

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

    try:
        document = pymupdf.open(pdf_path)
    except Exception as exc:
        raise ValueError("PDF 無法開啟，可能已損毀或加密") from exc

    vision_model = (vision_model or "").strip() or None
    report = PdfExtraction(pages=[], empty_pages=[], vision_model=vision_model)
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
                model=vision_model,
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


def _short_reason(reason: str, limit: int = 160) -> str:
    flattened = " ".join(reason.split())
    return flattened if len(flattened) <= limit else f"{flattened[:limit]}…"


def format_vision_report(report: PdfExtraction) -> str:
    """Summarize vision usage for the PDF parsing status message."""
    if not report.vision_model:
        return "未使用 Vision，僅擷取文字層。"
    if not report.fallback_pages:
        return f"Vision（{report.vision_model}）解析 {len(report.vision_pages)} 頁。"
    if report.vision_abort_reason:
        return (
            f"⚠️ Vision（{report.vision_model}）已中止：{_short_reason(report.vision_abort_reason)}；"
            f"{len(report.fallback_pages)} 頁改用文字層擷取。"
        )
    pages = sorted(report.fallback_pages)
    shown = "、".join(map(str, pages[:20])) + ("…" if len(pages) > 20 else "")
    first_reason = _short_reason(report.fallback_pages[pages[0]])
    return (
        f"⚠️ Vision（{report.vision_model}）解析 {len(report.vision_pages)} 頁；"
        f"{len(pages)} 頁改用文字層擷取（第 {shown} 頁，例：{first_reason}）。"
    )
