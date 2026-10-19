#!/usr/bin/env python3
"""Create a compact XL-DocBench subset in this project's question-import format.

The source JSONL files must be downloaded from the official Microsoft dataset
repository into datasets/xl_docbench/original/.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import unquote, urlparse


ROOT = Path(__file__).resolve().parents[1] / "datasets" / "xl_docbench"
OUTPUT = ROOT / "selected"

SELECTED = {
    "single_document": [
        "adubench_single_000107",
        "adubench_single_000209",
        "adubench_single_000210",
        "adubench_single_000212",
        "adubench_single_000636",
        "adubench_single_000637",
        "adubench_single_000638",
        "adubench_single_000639",
        "adubench_single_000897",
        "adubench_single_001300",
    ],
    "cross_document": [
        "adubench_cross_000030",
        "adubench_cross_000031",
        "adubench_cross_000032",
        "adubench_cross_000033",
        "adubench_cross_000034",
        "adubench_cross_000122",
        "adubench_cross_000124",
    ],
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    original = ROOT / "original"
    document_rows = read_jsonl(original / "documents.jsonl")
    documents = {row["document_id"]: row for row in document_rows}
    file_name_by_id = {
        doc_id: Path(unquote(urlparse(row["url"]).path)).name
        for doc_id, row in documents.items()
    }

    all_questions: dict[str, dict] = {}
    all_questions.update({row["question_id"]: row for row in read_jsonl(original / "qa_single_doc.jsonl")})
    all_questions.update({row["question_id"]: row for row in read_jsonl(original / "qa_cross_doc.jsonl")})

    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}
    for category, question_ids in SELECTED.items():
        items = []
        page_totals: dict[str, set[int]] = {}
        for number, question_id in enumerate(question_ids, 1):
            row = all_questions[question_id]
            if row["metadata"].get("is_unanswerable"):
                raise ValueError(f"Unexpected unanswerable question: {question_id}")
            if row["metadata"].get("evidence_sources") != ["Text"]:
                raise ValueError(f"Non-text evidence in selection: {question_id}")

            source_documents = [row["document"]] if "document" in row else row["documents"]
            answer_sources = []
            for source in source_documents:
                refs = sorted(
                    {
                        page
                        for evidence in source.get("evidence_items", [])
                        for page in evidence.get("pages", [])
                        if isinstance(page, int) and not isinstance(page, bool) and page > 0
                    }
                )
                if not refs:
                    continue
                doc_id = source["document_id"]
                filename = file_name_by_id[doc_id]
                answer_sources.append({"document_name": filename, "pages": refs})
                page_totals.setdefault(filename, set()).update(refs)

            expected_docs = row["metadata"].get("n_evidence_documents", 0)
            if len(answer_sources) < 1 or len(answer_sources) > expected_docs:
                raise ValueError(f"Missing Gold evidence references: {question_id}")

            items.append(
                {
                    "number": number,
                    "question": row["question"],
                    "expected_answer": str(row["answer"]["value"]),
                    "question_sources": [],
                    "answer_sources": answer_sources,
                }
            )

        output_path = OUTPUT / f"{category}.json"
        output_path.write_text(
            json.dumps({"questions": items}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        summary[category] = {
            "count": len(items),
            "question_ids": question_ids,
            "documents": {name: sorted(pages) for name, pages in sorted(page_totals.items())},
            "unique_document_page_count": sum(map(len, page_totals.values())),
        }

    (OUTPUT / "selection_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
