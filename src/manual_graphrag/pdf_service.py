from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import pymupdf

from .chunking import PageText
from .env_store import load_env
from .graph_service import _api_url, _chat_response_content, _post_json


HEADER_FOOTER_MARGIN_RATIO = 0.08
DEFAULT_RENDER_SCALE = 2.0
VISION_MODEL = "gpt-6-luna"
VISION_SYSTEM_PROMPT = """You are an expert assistant for analyzing automotive repair manuals. Convert the supplied manual page image into a JSON object with page_number and content fields.
Follow these rules strictly:
1. Transcribe all visible plain text completely without omission.
2. Convert every specification table or other table into Markdown table syntax.
3. For a vehicle component diagram, exploded view, or wiring diagram, provide a concise description using this exact format: "第{page_number}頁為{部件名稱}之結構圖，由{零件A}、{零件B}等零件構成。"
4. Add content items in visual top-to-bottom order. Each item must have a type of text, table, or diagram and a string data field.
5. Return exactly one valid JSON object and no Markdown fences or explanation."""


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
) -> dict[str, Any]:
    payload = {
        "model": VISION_MODEL,
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


def extract_pdf(
    path: str | Path, start_page: int = 1, end_page: int | None = None
) -> tuple[list[PageText], list[int]]:
    pdf_path = Path(path)
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("僅支援 PDF 檔案")

    try:
        document = pymupdf.open(pdf_path)
    except Exception as exc:
        raise ValueError("PDF 無法開啟，可能已損毀或加密") from exc

    pages: list[PageText] = []
    empty_pages: list[int] = []
    env = load_env()
    base_url = env["MODEL_OPENAI_API_BASE"]
    api_key = env["MODEL_OPENAI_API_KEY"]
    with document:
        if document.needs_pass:
            raise ValueError("目前不支援加密 PDF")
        first_page, last_page = _validate_page_range(
            document.page_count, start_page, end_page
        )
        try:
            for page_number in range(first_page, last_page + 1):
                page = document.load_page(page_number - 1)
                try:
                    encoded_image = render_page_to_base64(page)
                    result = _analyze_page_image(
                        encoded_image,
                        page_number,
                        base_url=base_url,
                        api_key=api_key,
                    )
                    text = _serialize_vision_result(result)
                    if not text:
                        text = page.get_text(
                            "text",
                            clip=_content_clip(page),
                            sort=True,
                        ).strip()
                except Exception:
                    # Keep PDF ingestion usable when rendering or the vision API fails.
                    try:
                        text = page.get_text(
                            "text",
                            clip=_content_clip(page),
                            sort=True,
                        ).strip()
                    except Exception:
                        text = ""
                pages.append(PageText(page_number, text))
                if not text:
                    empty_pages.append(page_number)
        except Exception as exc:
            raise ValueError("PDF 無法解析，請確認檔案內容有效") from exc

    return pages, empty_pages
