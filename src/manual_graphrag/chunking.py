from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class PageText:
    page: int
    text: str


@dataclass(frozen=True)
class TextChunk:
    number: int
    text: str
    pages: tuple[int, ...]
    document: str = ""
    document_id: str = ""


# A Markdown table delimiter row, e.g. "| --- | :---: |".
_TABLE_SEPARATOR = re.compile(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?$")


@dataclass(frozen=True)
class _Unit:
    """Smallest piece a chunk boundary may fall between: a character or a table row."""

    text: str
    pages: tuple[int, ...]
    table: int | None = None


@dataclass(frozen=True)
class _Table:
    start: int
    end: int
    header: str
    length: int


def _table_line_runs(text: str) -> list[list[tuple[int, int]]]:
    """Return runs of at least two consecutive Markdown table lines as (start, end) spans."""
    runs: list[list[tuple[int, int]]] = []
    current: list[tuple[int, int]] = []
    offset = 0
    for line in text.split("\n"):
        span = (offset, offset + len(line))
        offset = span[1] + 1
        if line.strip().startswith("|"):
            current.append(span)
            continue
        if len(current) >= 2:
            runs.append(current)
        current = []
    if len(current) >= 2:
        runs.append(current)
    return runs


def _units(pages: list[PageText]) -> tuple[list[_Unit], list[_Table]]:
    chars: list[str] = []
    owners: list[int] = []
    for page in pages:
        text = page.text.strip()
        if text:
            if chars:
                chars.extend("\n\n")
                owners.extend((page.page, page.page))
            chars.extend(text)
            owners.extend(page.page for _ in text)
    joined = "".join(chars)

    def unit(start: int, end: int, table: int | None = None) -> _Unit:
        return _Unit(joined[start:end], tuple(dict.fromkeys(owners[start:end])), table)

    units: list[_Unit] = []
    tables: list[_Table] = []
    position = 0
    for lines in _table_line_runs(joined):
        units.extend(unit(index, index + 1) for index in range(position, lines[0][0]))
        has_header = bool(_TABLE_SEPARATOR.match(joined[lines[1][0]:lines[1][1]].strip()))
        # Each row keeps its trailing newline; the header and delimiter rows form one unit.
        bounds = [start for start, _ in lines] + [lines[-1][1]]
        if has_header:
            bounds.pop(1)
        index = len(tables)
        first = len(units)
        units.extend(unit(start, end, index) for start, end in zip(bounds, bounds[1:]))
        header = joined[lines[0][0]:lines[1][1]] if has_header else ""
        tables.append(_Table(first, len(units), header, lines[-1][1] - lines[0][0]))
        position = lines[-1][1]
    units.extend(unit(index, index + 1) for index in range(position, len(joined)))
    return units, tables


def chunk_pages(
    pages: list[PageText],
    chunk_size: int,
    chunk_overlap: int,
    *,
    document: str = "",
    document_id: str = "",
    start_number: int = 1,
) -> list[TextChunk]:
    """Split page text into overlapping chunks without cutting Markdown tables.

    A table that fits in one chunk is never split; a longer table is split
    between rows and every continuation chunk repeats the table header. The
    overlap only repeats plain text, never table rows.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size 必須大於 0")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap 必須大於等於 0 且小於 chunk_size")

    units, tables = _units(pages)
    step = chunk_size - chunk_overlap
    chunks: list[TextChunk] = []
    start = 0
    while start < len(units):
        parts: list[str] = []
        page_numbers: list[int] = []
        length = 0
        first = units[start]
        if first.table is not None and start != tables[first.table].start and tables[first.table].header:
            prefix = tables[first.table].header + "\n"
            parts.append(prefix)
            length = len(prefix)
        has_content = False

        def add(selected: list[_Unit]) -> None:
            nonlocal length, has_content
            for item in selected:
                parts.append(item.text)
                page_numbers.extend(item.pages)
                length += len(item.text)
            has_content = True

        end = start
        while end < len(units):
            item = units[end]
            if item.table is None:
                if has_content and length + len(item.text) > step:
                    break
                add([item])
                end += 1
                continue
            table = tables[item.table]
            if end == table.start:
                if length + table.length <= chunk_size:
                    add(units[table.start:table.end])
                    end = table.end
                    continue
                if has_content and table.length <= chunk_size:
                    break
            if has_content and length + len(item.text) > chunk_size:
                break
            add([item])
            end += 1

        overlap_end = end
        while (
            overlap_end < len(units)
            and units[overlap_end].table is None
            and overlap_end - end < chunk_overlap
            and length < chunk_size
        ):
            add([units[overlap_end]])
            overlap_end += 1

        text = "".join(parts).strip()
        if text:
            chunks.append(TextChunk(
                start_number + len(chunks), text, tuple(dict.fromkeys(page_numbers)),
                document, document_id,
            ))
        if overlap_end == len(units):
            break
        start = end
    return chunks


def preview_rows(chunks: list[TextChunk], limit: int = 10) -> list[list[object]]:
    return [
        [chunk.number, chunk.document, ", ".join(map(str, chunk.pages)), len(chunk.text), chunk.text]
        for chunk in chunks[:limit]
    ]


def preview_rows_for_page(
    chunks: list[TextChunk], page_number: int
) -> list[list[object]]:
    return [
        [chunk.number, chunk.document, ", ".join(map(str, chunk.pages)), len(chunk.text), chunk.text]
        for chunk in chunks
        if page_number in chunk.pages
    ]
