from __future__ import annotations

from pathlib import Path

import pymupdf

from .chunking import PageText


HEADER_FOOTER_MARGIN_RATIO = 0.08


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
    with document:
        if document.needs_pass:
            raise ValueError("目前不支援加密 PDF")
        first_page, last_page = _validate_page_range(
            document.page_count, start_page, end_page
        )
        try:
            for page_number in range(first_page, last_page + 1):
                page = document.load_page(page_number - 1)
                text = page.get_text(
                    "text",
                    clip=_content_clip(page),
                    sort=True,
                ).strip()
                pages.append(PageText(page_number, text))
                if not text:
                    empty_pages.append(page_number)
        except Exception as exc:
            raise ValueError("PDF 無法解析，請確認檔案內容有效") from exc

    return pages, empty_pages


def extract_text_file(path: str | Path) -> tuple[list[PageText], list[int]]:
    """Read a UTF-8 text document as one synthetic page for shared chunking."""
    text_path = Path(path)
    if text_path.suffix.lower() != ".txt":
        raise ValueError("僅支援 TXT 檔案")
    try:
        text = text_path.read_text(encoding="utf-8-sig").strip()
    except UnicodeDecodeError as exc:
        raise ValueError("TXT 必須使用 UTF-8 編碼") from exc
    except OSError as exc:
        raise ValueError(f"TXT 無法讀取：{exc}") from exc
    return [PageText(1, text)], ([1] if not text else [])
