from __future__ import annotations

import base64
import struct
from pathlib import Path

import pytest

from manual_graphrag import pdf_service


def test_render_page_to_base64_returns_png_of_content_area() -> None:
    document = pdf_service.pymupdf.open()
    page = document.new_page(width=600, height=800)

    encoded = pdf_service.render_page_to_base64(page)
    image_bytes = base64.b64decode(encoded, validate=True)
    assert struct.unpack(">II", image_bytes[16:24]) == (1200, 1344)
    with pdf_service.pymupdf.open(stream=image_bytes, filetype="png") as image:
        assert image.page_count == 1
        assert image[0].rect.width == 900
        assert image[0].rect.height == 1008
    document.close()


@pytest.mark.parametrize("scale", [0, -1])
def test_render_page_to_base64_rejects_non_positive_scale(scale: float) -> None:
    document = pdf_service.pymupdf.open()
    page = document.new_page()

    with pytest.raises(ValueError, match="影像渲染倍率必須大於 0"):
        pdf_service.render_page_to_base64(page, scale)
    document.close()


def test_analyze_page_image_sends_json_vision_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_post(url: str, payload: dict[str, object], api_key: str) -> dict[str, object]:
        captured.update(url=url, payload=payload, api_key=api_key)
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"page_number": 4, "content": '
                        '[{"type": "text", "data": "Fuse inspection"}]}'
                    }
                }
            ]
        }

    monkeypatch.setattr(pdf_service, "_post_json", fake_post)

    result = pdf_service._analyze_page_image(
        "encoded-image", 4, base_url="https://models.example/v1/", api_key="secret"
    )

    assert result == {
        "page_number": 4,
        "content": [{"type": "text", "data": "Fuse inspection"}],
    }
    assert captured["url"] == "https://models.example/v1/chat/completions"
    assert captured["api_key"] == "secret"
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["model"] == "gpt-6-luna"
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["temperature"] == 0.0
    assert payload["reasoning_effort"] == "none"
    assert "max_tokens" not in payload
    user_content = payload["messages"][1]["content"]
    assert user_content[1]["image_url"]["url"] == "data:image/png;base64,encoded-image"


def test_analyze_page_image_rejects_invalid_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pdf_service,
        "_post_json",
        lambda *args, **kwargs: {
            "choices": [{"message": {"content": "not json"}}]
        },
    )

    with pytest.raises(ValueError, match="模型回傳的不是有效 JSON"):
        pdf_service._analyze_page_image(
            "encoded-image", 1, base_url="https://models.example/v1", api_key=""
        )


def test_serialize_vision_result_separates_tables() -> None:
    assert pdf_service._serialize_vision_result(
        {
            "page_number": 1,
            "content": [
                {"type": "text", "data": "Inspection"},
                {"type": "table", "data": "| Pin | Signal |\n|---|---|"},
                {"type": "diagram", "data": "Page 1 shows the fuse diagram."},
            ],
        }
    ) == "Inspection\n\n| Pin | Signal |\n|---|---|\n\nPage 1 shows the fuse diagram."


def test_extract_pdf_uses_vision_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = FakeDocument(pages=[FakePage("native fallback")])
    monkeypatch.setattr(pdf_service.pymupdf, "open", lambda path: document)
    monkeypatch.setattr(pdf_service, "render_page_to_base64", lambda page: "image")
    monkeypatch.setattr(pdf_service, "load_env", lambda: {
        "MODEL_OPENAI_API_BASE": "https://models.example/v1",
        "MODEL_OPENAI_API_KEY": "secret",
    })
    calls: list[tuple[str, int, str, str]] = []

    def analyze(
        image: str, page: int, *, base_url: str, api_key: str
    ) -> dict[str, object]:
        calls.append((image, page, base_url, api_key))
        return {
            "page_number": page,
            "content": [{"type": "text", "data": "Vision result"}],
        }

    monkeypatch.setattr(pdf_service, "_analyze_page_image", analyze)

    pages, empty_pages = pdf_service.extract_pdf("manual.pdf")

    assert [(page.page, page.text) for page in pages] == [(1, "Vision result")]
    assert empty_pages == []
    assert calls == [("image", 1, "https://models.example/v1", "secret")]


class FakePage:
    def __init__(self, text: str, *, broken: bool = False) -> None:
        self.rect = pdf_service.pymupdf.Rect(0, 0, 600, 800)
        self.text = text
        self.broken = broken
        self.received_clip: pdf_service.pymupdf.Rect | None = None

    def get_text(self, output: str, **options: object) -> str:
        if self.broken:
            raise RuntimeError("text extraction failed")
        assert output == "text"
        assert options["sort"] is True
        self.received_clip = options["clip"]  # type: ignore[assignment]
        return self.text


class FakeDocument:
    def __init__(
        self,
        *,
        needs_pass: bool = False,
        pages: list[FakePage] | None = None,
        page_count: int | None = None,
    ) -> None:
        self.needs_pass = needs_pass
        self.pages = pages or [FakePage("first"), FakePage("second")]
        self.page_count = len(self.pages) if page_count is None else page_count
        self.loaded_pages: list[int] = []

    def __enter__(self) -> FakeDocument:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def load_page(self, page_index: int) -> FakePage:
        self.loaded_pages.append(page_index)
        return self.pages[page_index]


def test_extract_pdf_returns_plain_text_and_empty_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_page = FakePage("First page\n")
    second_page = FakePage("   ")
    document = FakeDocument(pages=[first_page, second_page])
    monkeypatch.setattr(pdf_service.pymupdf, "open", lambda path: document)

    pages, empty_pages = pdf_service.extract_pdf("manual.pdf")

    assert [(page.page, page.text) for page in pages] == [
        (1, "First page"),
        (2, ""),
    ]
    assert empty_pages == [2]
    assert document.loaded_pages == [0, 1]
    assert first_page.received_clip == pdf_service.pymupdf.Rect(0, 64, 600, 736)


def test_extract_pdf_rejects_non_pdf(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_open(path: object) -> None:
        raise AssertionError("non-PDF input must not be opened")

    monkeypatch.setattr(pdf_service.pymupdf, "open", unexpected_open)

    with pytest.raises(ValueError, match="僅支援 PDF 檔案"):
        pdf_service.extract_pdf("manual.txt")


def test_extract_pdf_rejects_encrypted_pdf(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        pdf_service.pymupdf,
        "open",
        lambda path: FakeDocument(needs_pass=True),
    )

    with pytest.raises(ValueError, match="目前不支援加密 PDF"):
        pdf_service.extract_pdf("encrypted.pdf")


def test_extract_pdf_wraps_open_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken_open(path: object) -> None:
        raise RuntimeError("broken document")

    monkeypatch.setattr(pdf_service.pymupdf, "open", broken_open)

    with pytest.raises(ValueError, match="PDF 無法開啟，可能已損毀或加密"):
        pdf_service.extract_pdf("broken.pdf")


def test_extract_pdf_reports_a_page_when_vision_and_text_extraction_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = FakeDocument(pages=[FakePage("", broken=True)])
    monkeypatch.setattr(pdf_service.pymupdf, "open", lambda path: document)

    pages, empty_pages = pdf_service.extract_pdf("broken.pdf")

    assert [(page.page, page.text) for page in pages] == [(1, "")]
    assert empty_pages == [1]


def test_extract_pdf_with_real_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf_path = tmp_path / "manual.pdf"
    document = pdf_service.pymupdf.open()
    first_page = document.new_page()
    first_page.insert_text((72, 200), "Hello PDF")
    second_page = document.new_page()
    second_page.insert_text((72, 200), "Second page")
    document.new_page()
    document.save(pdf_path)
    document.close()
    monkeypatch.setattr(
        pdf_service,
        "_analyze_page_image",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("offline")),
    )

    assert pdf_service.get_pdf_page_count(pdf_path) == 3
    pages, empty_pages = pdf_service.extract_pdf(pdf_path, 2, 3)

    assert [(page.page, page.text) for page in pages] == [(2, "Second page"), (3, "")]
    assert empty_pages == [3]


def test_extract_pdf_only_reads_requested_page_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = FakeDocument(
        pages=[FakePage("first"), FakePage("second"), FakePage(" "), FakePage("fourth")]
    )
    monkeypatch.setattr(pdf_service.pymupdf, "open", lambda path: document)

    pages, empty_pages = pdf_service.extract_pdf("manual.pdf", 2, 3)

    assert [(page.page, page.text) for page in pages] == [(2, "second"), (3, "")]
    assert empty_pages == [3]
    assert document.loaded_pages == [1, 2]


def test_extract_pdf_rejects_invalid_page_ranges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pdf_service.pymupdf,
        "open",
        lambda path: FakeDocument(page_count=3),
    )
    cases = [
        (0, None, "解析起始頁必須大於等於 1"),
        (3, 2, "解析結束頁不得小於起始頁"),
        (1, 4, "解析頁碼超出範圍；PDF 共 3 頁"),
    ]

    for start_page, end_page, message in cases:
        with pytest.raises(ValueError, match=message):
            pdf_service.extract_pdf("manual.pdf", start_page, end_page)


def test_extract_pdf_excludes_header_and_footer_regions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf_path = tmp_path / "with-margins.pdf"
    document = pdf_service.pymupdf.open()
    for page_number in range(1, 4):
        page = document.new_page(width=600, height=800)
        page.insert_text((72, 30), "Repeated header")
        page.insert_text((72, 200), f"Body content {page_number}")
        page.insert_text((72, 780), f"Page {page_number}")
    document.save(pdf_path)
    document.close()
    monkeypatch.setattr(
        pdf_service,
        "_analyze_page_image",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("offline")),
    )

    pages, empty_pages = pdf_service.extract_pdf(pdf_path)

    assert empty_pages == []
    assert [page.text for page in pages] == [
        "Body content 1",
        "Body content 2",
        "Body content 3",
    ]
