from __future__ import annotations

import base64
import struct
from pathlib import Path
from threading import Barrier, Thread
import time
import urllib.error

import pytest

from manual_graphrag import pdf_service


_real_page_has_figures = pdf_service.page_has_figures


@pytest.fixture(autouse=True)
def _fake_figure_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    # Fake pages have no drawing API; they declare whether they hold a figure,
    # either as a bool (any / no element) or as a set of element kinds.
    def has_figures(page: object, kinds: tuple[str, ...] = pdf_service.FIGURE_KINDS) -> bool:
        declared = getattr(page, "has_figures", True)
        if isinstance(declared, bool):
            return declared
        return bool(set(declared) & set(kinds))

    monkeypatch.setattr(pdf_service, "page_has_figures", has_figures)
    # Language detection reads every page; tests of it call it directly.
    monkeypatch.setattr(pdf_service, "_document_language", lambda document: None)


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
    calls: list[tuple[str, int, str, str, str, str | None]] = []

    def analyze(
        image: str, page: int, *, base_url: str, api_key: str, model: str,
        language: str | None, reasoning_effort: str | None,
    ) -> dict[str, object]:
        calls.append((image, page, base_url, api_key, model, reasoning_effort))
        return {
            "page_number": page,
            "content": [{"type": "text", "data": "Vision result"}],
        }

    monkeypatch.setattr(pdf_service, "_analyze_page_image", analyze)

    pages, empty_pages = pdf_service.extract_pdf("manual.pdf")

    assert [(page.page, page.text) for page in pages] == [(1, "Vision result")]
    assert empty_pages == []
    assert calls == [("image", 1, "https://models.example/v1", "secret", "gpt-6-luna", "none")]


def _vision_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pdf_service, "render_page_to_base64", lambda page: f"image-{page.text}")
    monkeypatch.setattr(pdf_service, "load_env", lambda: {
        "MODEL_OPENAI_API_BASE": "https://models.example/v1",
        "MODEL_OPENAI_API_KEY": "secret",
    })


def _vision_result(page: int, text: str) -> dict[str, object]:
    return {"page_number": page, "content": [{"type": "text", "data": text}]}


def test_analyze_page_image_uses_selected_model(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_post(url: str, payload: dict[str, object], api_key: str) -> dict[str, object]:
        captured.update(payload)
        return {"choices": [{"message": {"content": '{"page_number": 1, "content": []}'}}]}

    monkeypatch.setattr(pdf_service, "_post_json", fake_post)

    pdf_service._analyze_page_image(
        "image", 1, base_url="https://models.example/v1", api_key="", model="gpt-4.1-mini",
    )

    assert captured["model"] == "gpt-4.1-mini"
    # Only GPT-6 Luna takes reasoning_effort; other models keep temperature 0.
    assert captured["temperature"] == 0.0 and "reasoning_effort" not in captured


def test_analyze_page_image_sends_selected_reasoning_effort(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_post(url: str, payload: dict[str, object], api_key: str) -> dict[str, object]:
        captured.update(payload)
        return {"choices": [{"message": {"content": '{"page_number": 1, "content": []}'}}]}

    monkeypatch.setattr(pdf_service, "_post_json", fake_post)

    pdf_service._analyze_page_image(
        "image", 1, base_url="https://models.example/v1", api_key="", reasoning_effort="high",
    )

    # GPT-6 reasoning requests reject temperature unless effort is none.
    assert captured["reasoning_effort"] == "high" and "temperature" not in captured
    with pytest.raises(ValueError, match="不支援的 Vision 推理強度"):
        pdf_service._analyze_page_image(
            "image", 1, base_url="https://models.example/v1", api_key="", reasoning_effort="max",
        )


def test_vision_prompt_is_not_limited_to_automotive_manuals() -> None:
    prompt = pdf_service.VISION_SYSTEM_PROMPT
    assert "automotive" not in prompt
    assert "第{page_number}頁" not in prompt
    assert "same language as the page text" in prompt


def test_vision_prompt_asks_for_label_positions() -> None:
    prompt = pdf_service.VISION_SYSTEM_PROMPT
    assert "which view it is" in prompt
    assert "For every labeled component, state where it is on the object" in prompt
    assert "relative to nearby labeled components" in prompt


def test_extract_pdf_runs_vision_calls_concurrently_and_keeps_page_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = FakeDocument(pages=[FakePage(f"p{number}") for number in range(1, 4)])
    monkeypatch.setattr(pdf_service.pymupdf, "open", lambda path: document)
    _vision_env(monkeypatch)
    barrier = Barrier(3, timeout=5)

    def analyze(image: str, page: int, **kwargs: object) -> dict[str, object]:
        barrier.wait()  # Fails unless all three requests are in flight together.
        return _vision_result(page, f"vision {page}")

    monkeypatch.setattr(pdf_service, "_analyze_page_image", analyze)

    report = pdf_service.extract_pdf_with_report("manual.pdf", max_workers=3)

    assert [(page.page, page.text) for page in report.pages] == [
        (1, "vision 1"), (2, "vision 2"), (3, "vision 3"),
    ]
    assert report.vision_pages == [1, 2, 3]
    assert report.fallback_pages == {}


def test_extract_pdf_reports_progress_for_every_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = FakeDocument(pages=[FakePage("a"), FakePage("b")])
    monkeypatch.setattr(pdf_service.pymupdf, "open", lambda path: document)
    _vision_env(monkeypatch)
    monkeypatch.setattr(
        pdf_service, "_analyze_page_image",
        lambda image, page, **kwargs: _vision_result(page, "ok"),
    )
    progress: list[tuple[float, str]] = []

    pdf_service.extract_pdf_with_report(
        "manual.pdf", progress_callback=lambda fraction, message: progress.append((fraction, message)),
    )

    assert progress == [(0.5, "已解析 1 / 2 頁"), (1.0, "已解析 2 / 2 頁")]


def test_extract_pdf_records_fallback_pages_with_reasons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = FakeDocument(pages=[FakePage("first"), FakePage("second")])
    monkeypatch.setattr(pdf_service.pymupdf, "open", lambda path: document)
    _vision_env(monkeypatch)

    def analyze(image: str, page: int, **kwargs: object) -> dict[str, object]:
        if page == 2:
            raise ValueError("模型回傳的不是有效 JSON")
        return _vision_result(page, "vision first")

    monkeypatch.setattr(pdf_service, "_analyze_page_image", analyze)

    report = pdf_service.extract_pdf_with_report("manual.pdf")

    assert [page.text for page in report.pages] == ["vision first", "second"]
    assert report.vision_pages == [1]
    assert report.fallback_pages == {2: "模型回傳的不是有效 JSON"}
    assert report.vision_abort_reason is None
    assert pdf_service.format_vision_report(report) == (
        "⚠️ Vision（gpt-6-luna）解析 1 頁；1 頁改用文字層擷取（第 2 頁，例：模型回傳的不是有效 JSON）。"
    )


def test_extract_pdf_stops_calling_vision_after_authentication_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = FakeDocument(pages=[FakePage(f"text {number}") for number in range(1, 4)])
    monkeypatch.setattr(pdf_service.pymupdf, "open", lambda path: document)
    _vision_env(monkeypatch)
    calls: list[int] = []

    def analyze(image: str, page: int, **kwargs: object) -> dict[str, object]:
        calls.append(page)
        http_error = urllib.error.HTTPError("url", 401, "Unauthorized", None, None)
        raise ValueError("模型 API 回傳 HTTP 401：invalid key") from http_error

    monkeypatch.setattr(pdf_service, "_analyze_page_image", analyze)

    report = pdf_service.extract_pdf_with_report("manual.pdf", max_workers=1)

    assert calls == [1]
    assert [page.text for page in report.pages] == ["text 1", "text 2", "text 3"]
    assert sorted(report.fallback_pages) == [1, 2, 3]
    assert report.vision_abort_reason == "模型 API 回傳 HTTP 401：invalid key"
    assert "已中止" in pdf_service.format_vision_report(report)


def test_format_vision_report_flattens_long_api_errors() -> None:
    report = pdf_service.PdfExtraction(
        pages=[], empty_pages=[], vision_model="gpt-6-luna",
        fallback_pages={1: "Vision 解析已中止"},
        vision_abort_reason='模型 API 回傳 HTTP 404：{\n  "error": "' + "x" * 300 + '"}',
    )

    message = pdf_service.format_vision_report(report)

    assert "\n" not in message
    assert "…；1 頁改用文字層擷取。" in message


def test_extract_pdf_without_vision_model_skips_rendering_and_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = FakeDocument(pages=[FakePage("plain")])
    monkeypatch.setattr(pdf_service.pymupdf, "open", lambda path: document)
    fail = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not be called"))
    monkeypatch.setattr(pdf_service, "render_page_to_base64", fail)
    monkeypatch.setattr(pdf_service, "load_env", fail)
    monkeypatch.setattr(pdf_service, "_analyze_page_image", fail)

    report = pdf_service.extract_pdf_with_report("manual.pdf", vision_model=None)

    assert [page.text for page in report.pages] == ["plain"]
    assert report.vision_model is None and report.fallback_pages == {}
    assert pdf_service.format_vision_report(report) == "未使用 Vision，僅擷取文字層。"


class FakePage:
    def __init__(
        self, text: str, *, broken: bool = False, has_figures: bool | set[str] = True,
    ) -> None:
        self.rect = pdf_service.pymupdf.Rect(0, 0, 600, 800)
        self.text = text
        self.broken = broken
        self.has_figures = has_figures
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


def test_extract_pdf_stop_raises_and_skips_remaining_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = FakeDocument(pages=[FakePage(f"text {number}") for number in range(1, 6)])
    monkeypatch.setattr(pdf_service.pymupdf, "open", lambda path: document)
    _vision_env(monkeypatch)
    control = pdf_service.RunControl()
    calls: list[int] = []

    def analyze(image: str, page: int, **kwargs: object) -> dict[str, object]:
        calls.append(page)
        control.request_stop()
        return _vision_result(page, "ok")

    monkeypatch.setattr(pdf_service, "_analyze_page_image", analyze)

    with pytest.raises(pdf_service.RunCancelled):
        pdf_service.extract_pdf_with_report("manual.pdf", max_workers=1, control=control)

    assert calls == [1]
    assert len(document.loaded_pages) < 5


def test_extract_pdf_pause_holds_new_pages_until_resumed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = FakeDocument(pages=[FakePage("a"), FakePage("b")])
    monkeypatch.setattr(pdf_service.pymupdf, "open", lambda path: document)
    _vision_env(monkeypatch)
    monkeypatch.setattr(
        pdf_service, "_analyze_page_image",
        lambda image, page, **kwargs: _vision_result(page, f"vision {page}"),
    )
    control = pdf_service.RunControl()
    control.toggle_pause()
    result: list[pdf_service.PdfExtraction] = []
    worker = Thread(target=lambda: result.append(
        pdf_service.extract_pdf_with_report("manual.pdf", control=control)
    ))

    worker.start()
    time.sleep(0.5)
    assert document.loaded_pages == [] and worker.is_alive()
    control.toggle_pause()
    worker.join(timeout=5)

    assert [page.text for page in result[0].pages] == ["vision 1", "vision 2"]


def _figure_test_pdf(path: Path) -> Path:
    pymupdf = pdf_service.pymupdf
    document = pymupdf.open()
    page = document.new_page(width=600, height=800)
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 20, 20), False)
    pixmap.set_rect(pixmap.irect, (200, 30, 30))
    page.insert_image(pymupdf.Rect(100, 100, 250, 250), pixmap=pixmap)
    # An illustration made of separate nearby paths should become one figure.
    for x in (320, 360, 400):
        page.draw_circle((x, 320), 15)
    page.draw_line((320, 340), (400, 380))
    # A note box frame, a tiny icon and a header logo are not figures.
    page.draw_rect(pymupdf.Rect(50, 450, 550, 500))
    page.insert_text((60, 480), "Note text inside a frame")
    page.draw_circle((520, 300), 5)
    page.draw_circle((300, 30), 20)
    # A ruled 3 x 3 table.
    for y in (560, 590, 620, 650):
        page.draw_line((60, y), (540, y))
    for x in (60, 220, 380, 540):
        page.draw_line((x, 560), (x, 650))
    for row, y in enumerate((580, 610, 640)):
        for column, x in enumerate((70, 230, 390)):
            page.insert_text((x, y), f"r{row}c{column}")
    target = path / "figures.pdf"
    document.save(target)
    document.close()
    return target


def test_detect_page_figures_finds_images_illustrations_and_tables(tmp_path: Path) -> None:
    with pdf_service.pymupdf.open(_figure_test_pdf(tmp_path)) as document:
        figures = pdf_service.detect_page_figures(document[0])

    assert [figure.kind for figure in figures] == ["image", "drawing", "table"]
    assert all(figure.page == 1 for figure in figures)
    image, drawing, table = (pdf_service.pymupdf.Rect(f.bbox) for f in figures)
    assert image == pdf_service.pymupdf.Rect(100, 100, 250, 250)
    assert drawing.contains(pdf_service.pymupdf.Rect(305, 305, 415, 380))
    assert table.contains(pdf_service.pymupdf.Rect(70, 570, 530, 640))


def test_render_pdf_figures_crops_pngs_and_respects_limit(tmp_path: Path) -> None:
    path = _figure_test_pdf(tmp_path)

    results = pdf_service.render_pdf_figures(path, [1, 1, 5], scale=1.0)
    assert [figure.kind for figure, _ in results] == ["image", "drawing", "table"]
    figure, png = results[0]
    assert png.startswith(b"\x89PNG")
    # The crop is the figure plus padding on each side, not the whole page.
    assert struct.unpack(">II", png[16:24]) == (162, 162)

    assert len(pdf_service.render_pdf_figures(path, [1], limit=2)) == 2


def test_extract_pdf_figures_scope_sends_only_figure_and_textless_pages_to_vision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = FakeDocument(pages=[
        FakePage("diagram page"),
        FakePage("plain page", has_figures=False),
        FakePage("   ", has_figures=False),  # A scan without a text layer.
    ])
    monkeypatch.setattr(pdf_service.pymupdf, "open", lambda path: document)
    _vision_env(monkeypatch)
    sent: list[int] = []

    def analyze(image: str, page: int, **kwargs: object) -> dict[str, object]:
        sent.append(page)
        return _vision_result(page, f"vision {page}")

    monkeypatch.setattr(pdf_service, "_analyze_page_image", analyze)
    progress: list[float] = []

    report = pdf_service.extract_pdf_with_report(
        "manual.pdf", vision_scope=pdf_service.VISION_SCOPE_FIGURES, max_workers=1,
        progress_callback=lambda fraction, _message: progress.append(fraction),
    )

    assert sorted(sent) == [1, 3]
    assert [page.text for page in report.pages] == ["vision 1", "plain page", "vision 3"]
    assert report.figure_pages == [1]
    assert report.text_layer_pages == [2]
    assert [report.page_source(number) for number in (1, 2, 3)] == ["vision", "text", "vision"]
    assert progress[-1] == 1.0 and len(progress) == 3
    assert pdf_service.format_vision_report(report) == (
        "Vision（gpt-6-luna）解析 2 頁。1 頁無圖表，直接使用文字層。"
    )


@pytest.mark.parametrize(
    ("scope", "sent_pages", "label"),
    [
        (pdf_service.VISION_SCOPE_TABLES, [1, 4], "無表格"),
        (pdf_service.VISION_SCOPE_DIAGRAMS, [2, 4], "無圖片或向量圖"),
    ],
)
def test_extract_pdf_tables_and_diagrams_scopes_send_only_matching_pages(
    monkeypatch: pytest.MonkeyPatch, scope: str, sent_pages: list[int], label: str,
) -> None:
    document = FakeDocument(pages=[
        FakePage("table page", has_figures={"table"}),
        FakePage("diagram page", has_figures={"image"}),
        FakePage("plain page", has_figures=False),
        FakePage("   ", has_figures=False),  # A scan without a text layer.
    ])
    monkeypatch.setattr(pdf_service.pymupdf, "open", lambda path: document)
    _vision_env(monkeypatch)
    sent: list[int] = []

    def analyze(image: str, page: int, **kwargs: object) -> dict[str, object]:
        sent.append(page)
        return _vision_result(page, f"vision {page}")

    monkeypatch.setattr(pdf_service, "_analyze_page_image", analyze)

    report = pdf_service.extract_pdf_with_report("manual.pdf", vision_scope=scope, max_workers=1)

    assert sorted(sent) == sent_pages
    # figure_pages still lists every page with any element, for the review filter.
    assert report.figure_pages == [1, 2]
    assert report.text_layer_pages == sorted({1, 2, 3} - set(sent_pages))
    assert pdf_service.format_vision_report(report) == (
        f"Vision（gpt-6-luna）解析 2 頁。2 頁{label}，直接使用文字層。"
    )


def test_extract_pdf_all_scope_still_records_figure_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = FakeDocument(pages=[FakePage("a"), FakePage("b", has_figures=False)])
    monkeypatch.setattr(pdf_service.pymupdf, "open", lambda path: document)
    _vision_env(monkeypatch)
    monkeypatch.setattr(
        pdf_service, "_analyze_page_image",
        lambda image, page, **kwargs: _vision_result(page, f"vision {page}"),
    )

    report = pdf_service.extract_pdf_with_report("manual.pdf")

    assert report.vision_pages == [1, 2]
    assert report.figure_pages == [1]
    assert report.text_layer_pages == []


def test_extract_pdf_rejects_unknown_vision_scope() -> None:
    with pytest.raises(ValueError, match="Vision 解析範圍"):
        pdf_service.extract_pdf_with_report("manual.pdf", vision_scope="some")


def test_page_has_figures_matches_detect_page_figures(tmp_path: Path) -> None:
    with pdf_service.pymupdf.open(_figure_test_pdf(tmp_path)) as document:
        assert _real_page_has_figures(document[0]) is True
        assert _real_page_has_figures(document[0], ("table",)) is True
        assert _real_page_has_figures(document[0], pdf_service.DIAGRAM_KINDS) is True
    with pdf_service.pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((72, 200), "Only text on this page")
        page.draw_rect(pdf_service.pymupdf.Rect(50, 180, 550, 220))  # A note frame.
        assert _real_page_has_figures(page) is False
        assert pdf_service.detect_page_figures(page) == []
        # A ruled table without any illustration.
        for y in (560, 590, 620, 650):
            page.draw_line((60, y), (540, y))
        for x in (60, 220, 380, 540):
            page.draw_line((x, 560), (x, 650))
        for row, y in enumerate((580, 610, 640)):
            for column, x in enumerate((70, 230, 390)):
                page.insert_text((x, y), f"r{row}c{column}")
        assert _real_page_has_figures(page, ("table",)) is True
        assert _real_page_has_figures(page, pdf_service.DIAGRAM_KINDS) is False


def test_table_without_outer_side_borders_is_detected() -> None:
    with pdf_service.pymupdf.open() as document:
        page = document.new_page(width=600, height=800)
        # Horizontal rules and one inner column line, but no left or right border.
        for y in (200, 230, 290, 340):
            page.draw_line((60, y), (540, y))
        page.draw_line((200, 200), (200, 340))
        for y in (220, 260, 320):
            page.insert_text((70, y), "Button")
            page.insert_text((210, y), "Function")
        assert _real_page_has_figures(page, ("table",)) is True
        assert [figure.kind for figure in pdf_service.detect_page_figures(page)] == ["table"]


def test_table_needs_ruled_top_and_bottom_edges() -> None:
    pymupdf = pdf_service.pymupdf
    with pymupdf.open() as document:
        page = document.new_page(width=600, height=800)
        # A real table: its top edge is two cell borders, the bottom one long rule.
        page.draw_line((60, 100), (300, 100))
        page.draw_line((300, 100), (540, 100))
        page.draw_line((60, 160), (540, 160))
        # Callout leader lines: a label line on top only, partial at the bottom.
        page.draw_line((60, 300), (540, 300))
        page.draw_line((60, 360), (200, 360))
        segments = pdf_service._horizontal_segments(page.get_drawings())

    assert pdf_service._has_table_border(pymupdf.Rect(60, 100, 540, 160), segments) is True
    assert pdf_service._has_table_border(pymupdf.Rect(60, 300, 540, 360), segments) is False


def test_single_page_helpers_render_read_and_reanalyze(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "manual.pdf"
    with pdf_service.pymupdf.open() as document:
        for text in ("First page body", "Second page body"):
            document.new_page(width=600, height=800).insert_text((72, 300), text)
        document.save(path)
    _vision_env(monkeypatch)
    monkeypatch.setattr(pdf_service, "render_page_to_base64", lambda page: f"image-{page.number}")
    captured: dict[str, object] = {}

    def analyze(image: str, page: int, **kwargs: object) -> dict[str, object]:
        captured.update(image=image, page=page, **kwargs)
        return {"page_number": page, "content": [
            {"type": "text", "data": "Body"}, {"type": "table", "data": "| a |\n| --- |"},
        ]}

    monkeypatch.setattr(pdf_service, "_analyze_page_image", analyze)

    png = pdf_service.render_pdf_page(path, 2, scale=1.0)
    assert png.startswith(b"\x89PNG") and struct.unpack(">II", png[16:24]) == (600, 800)
    assert pdf_service.extract_page_text_layer(path, 2) == "Second page body"
    assert pdf_service.analyze_pdf_page(path, 2, "gpt-4.1-mini") == "Body\n\n| a |\n| --- |"
    assert captured["image"] == "image-1" and captured["page"] == 2
    assert captured["model"] == "gpt-4.1-mini" and captured["api_key"] == "secret"
    with pytest.raises(ValueError, match="頁碼超出範圍"):
        pdf_service.render_pdf_page(path, 3)


@pytest.mark.parametrize(("texts", "language"), [
    (["麥克風 閃光燈 後置相機 GPS 天線", "開始使用：請先設定裝置，點擊選單中的網路設定。"], "Traditional Chinese"),
    (["开始使用：请先设定装置，点击选单中的网络设定。这个设备的开关在右侧。"], "Simplified Chinese"),
    (["スマートフォンの背面にあるカメラを使用します。設定メニューを開いてください。"], "Japanese"),
    (["휴대전화 뒷면의 카메라를 사용합니다. 설정 메뉴를 여십시오."], "Korean"),
    (["Press the power button to turn on the device and open the settings menu. " * 3], "English"),
    (["Seite 12", "12"], None),
])
def test_detect_document_language(texts: list[str], language: str | None) -> None:
    assert pdf_service.detect_document_language(texts) == language


def test_vision_prompt_states_document_language_and_keeps_labels_verbatim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert pdf_service.vision_system_prompt(None) == pdf_service.VISION_SYSTEM_PROMPT
    assert "never translate a label" in pdf_service.VISION_SYSTEM_PROMPT
    prompt = pdf_service.vision_system_prompt("Traditional Chinese")
    assert "This document is written in Traditional Chinese" in prompt
    assert "only figure labels" in prompt
    captured: dict[str, object] = {}

    def fake_post(url: str, payload: dict[str, object], api_key: str) -> dict[str, object]:
        captured.update(payload)
        return {"choices": [{"message": {"content": '{"page_number": 1, "content": []}'}}]}

    monkeypatch.setattr(pdf_service, "_post_json", fake_post)
    pdf_service._analyze_page_image(
        "image", 1, base_url="https://models.example/v1", api_key="", language="Japanese",
    )
    assert "written in Japanese" in captured["messages"][0]["content"]  # type: ignore[index]


def test_extract_pdf_passes_reasoning_effort_to_every_vision_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = FakeDocument(pages=[FakePage("one"), FakePage("two")])
    monkeypatch.setattr(pdf_service.pymupdf, "open", lambda path: document)
    _vision_env(monkeypatch)
    efforts: list[object] = []

    def analyze(image: str, page: int, **kwargs: object) -> dict[str, object]:
        efforts.append(kwargs["reasoning_effort"])
        return _vision_result(page, f"vision {page}")

    monkeypatch.setattr(pdf_service, "_analyze_page_image", analyze)

    report = pdf_service.extract_pdf_with_report(
        "manual.pdf", vision_reasoning_effort="medium", max_workers=1,
    )
    assert efforts == ["medium", "medium"]
    assert report.vision_reasoning_effort == "medium"

    report = pdf_service.extract_pdf_with_report(
        "manual.pdf", vision_model="gpt-4.1-mini", vision_reasoning_effort="medium", max_workers=1,
    )
    assert efforts[2:] == [None, None]
    assert report.vision_reasoning_effort is None


def test_extract_pdf_passes_detected_language_to_every_vision_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = FakeDocument(pages=[FakePage("麥克風"), FakePage("閃光燈")])
    monkeypatch.setattr(pdf_service.pymupdf, "open", lambda path: document)
    monkeypatch.setattr(pdf_service, "_document_language", lambda document: "Traditional Chinese")
    _vision_env(monkeypatch)
    languages: list[object] = []

    def analyze(image: str, page: int, **kwargs: object) -> dict[str, object]:
        languages.append(kwargs["language"])
        return _vision_result(page, f"vision {page}")

    monkeypatch.setattr(pdf_service, "_analyze_page_image", analyze)

    report = pdf_service.extract_pdf_with_report("manual.pdf", max_workers=1)

    assert languages == ["Traditional Chinese"] * 2
    assert report.language == "Traditional Chinese"
    assert pdf_service.format_vision_report(report) == (
        "Vision（gpt-6-luna）解析 2 頁。圖表描述語言：繁體中文。"
    )


def test_extract_text_file_reads_utf8_and_uses_one_synthetic_page(tmp_path: Path) -> None:
    text_path = tmp_path / "article.txt"
    text_path.write_bytes(b"\xef\xbb\xbfFirst paragraph\n\nSecond paragraph\n")

    pages, empty_pages = pdf_service.extract_text_file(text_path)

    assert [(page.page, page.text) for page in pages] == [
        (1, "First paragraph\n\nSecond paragraph"),
    ]
    assert empty_pages == []


def test_extract_text_file_marks_empty_document(tmp_path: Path) -> None:
    text_path = tmp_path / "empty.txt"
    text_path.write_text(" \n", encoding="utf-8")

    pages, empty_pages = pdf_service.extract_text_file(text_path)

    assert [(page.page, page.text) for page in pages] == [(1, "")]
    assert empty_pages == [1]


def test_extract_text_file_rejects_invalid_encoding_and_extension(tmp_path: Path) -> None:
    invalid_text = tmp_path / "invalid.txt"
    invalid_text.write_bytes(b"\xff")
    with pytest.raises(ValueError, match="TXT 必須使用 UTF-8 編碼"):
        pdf_service.extract_text_file(invalid_text)

    with pytest.raises(ValueError, match="僅支援 TXT 檔案"):
        pdf_service.extract_text_file(tmp_path / "manual.pdf")
