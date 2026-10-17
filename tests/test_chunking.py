import pytest

from manual_graphrag.chunking import (
    PageText,
    TextChunk,
    chunk_pages,
    preview_rows_for_page,
)


def test_chunk_pages_preserves_page_provenance() -> None:
    chunks = chunk_pages([PageText(1, "abcd"), PageText(2, "efgh")], 6, 2)
    assert chunks[0].pages == (1, 2)
    assert chunks[1].pages == (2,)


def test_chunk_pages_returns_empty_for_blank_pages() -> None:
    assert chunk_pages([PageText(1, "  ")], 100, 10) == []


def test_chunk_pages_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError):
        chunk_pages([PageText(1, "text")], 10, 10)


def test_preview_rows_for_page_returns_all_related_chunks() -> None:
    chunks = [
        TextChunk(1, "first", (1,)),
        TextChunk(2, "shared", (1, 2)),
        TextChunk(3, "second", (2,)),
    ]

    assert [row[0] for row in preview_rows_for_page(chunks, 2)] == [2, 3]


def test_chunk_pages_tags_document_and_continues_numbering() -> None:
    chunks = chunk_pages(
        [PageText(1, "abcd")], 6, 2, document="manual.pdf", start_number=5
    )
    assert [chunk.number for chunk in chunks] == [5]
    assert chunks[0].document == "manual.pdf"


def _table(rows: int) -> str:
    lines = ["| 項目 | 數值 |", "| --- | --- |"]
    lines.extend(f"| 項目{index:02d} | {index:04d} |" for index in range(rows))
    return "\n".join(lines)


def test_chunk_pages_matches_sliding_window_for_plain_text() -> None:
    text = "".join(chr(ord("a") + index % 26) for index in range(50))
    chunks = chunk_pages([PageText(1, text)], 20, 5)
    assert [chunk.text for chunk in chunks] == [text[0:20], text[15:35], text[30:50]]


def test_chunk_pages_keeps_table_that_fits_in_one_chunk() -> None:
    table = _table(5)
    intro = "說明文字" * 20
    chunks = chunk_pages([PageText(3, f"{intro}\n\n{table}\n\n結尾")], 150, 20)

    table_chunks = [chunk for chunk in chunks if "| 項目 |" in chunk.text]
    assert len(table_chunks) == 1
    assert table in table_chunks[0].text
    assert all(
        "| 項目0" not in chunk.text for chunk in chunks if chunk not in table_chunks
    )


def test_chunk_pages_splits_long_table_by_rows_and_repeats_header() -> None:
    table = _table(30)
    chunks = chunk_pages([PageText(1, "規格表"), PageText(2, table)], 200, 50)

    header = "| 項目 | 數值 |\n| --- | --- |"
    rows = table.splitlines()[2:]
    seen: list[str] = []
    for chunk in chunks:
        lines = chunk.text.splitlines()
        if any(line in rows for line in lines):
            assert header in chunk.text
            assert len(chunk.text) <= 200
        seen.extend(line for line in lines if line in rows)
    assert len(chunks) > 2
    # Every row appears exactly once: rows are neither cut nor repeated by the overlap.
    assert seen == rows
    assert chunks[0].text.startswith("規格表")
    assert chunks[-1].pages == (2,)


def test_chunk_pages_keeps_oversized_single_row_whole() -> None:
    long_row = "| " + "x" * 120 + " |"
    table = "| a |\n| --- |\n" + long_row
    chunks = chunk_pages([PageText(1, table)], 60, 10)

    assert any(long_row in chunk.text for chunk in chunks)
    assert all(chunk.text.startswith("| a |") for chunk in chunks)


def test_chunk_pages_treats_table_without_delimiter_as_rows() -> None:
    table = "\n".join(f"| r{index} | v{index} |" for index in range(10))
    chunks = chunk_pages([PageText(1, table)], 40, 10)

    rows = table.splitlines()
    assert [line for chunk in chunks for line in chunk.text.splitlines()] == rows
