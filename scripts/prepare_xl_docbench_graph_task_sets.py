#!/usr/bin/env python3
"""Build three compact XL-DocBench sets for GraphRAG-oriented experiments."""

from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import unquote, urlparse


ROOT = Path(__file__).resolve().parents[1] / "datasets" / "xl_docbench"
OUTPUT = ROOT / "graph_task_sets"

SELECTED = {
    "reference_chain": [
        "adubench_single_000172",
        "adubench_single_000173",
        "adubench_single_000097",
        "adubench_single_000036",
        "adubench_single_000145",
        "adubench_single_000213",
        "adubench_single_000100",
        "adubench_single_000706",
        "adubench_single_000625",
        "adubench_single_000858",
    ],
    "aggregation": [
        "adubench_cross_000070",
        "adubench_single_000898",
        "adubench_single_000174",
        "adubench_single_000746",
        "adubench_single_000055",
        "adubench_single_000850",
        "adubench_single_001023",
        "adubench_single_000901",
        "adubench_single_000031",
        "adubench_single_001278",
    ],
    "comparison": [
        "adubench_single_000707",
        "adubench_single_000710",
        "adubench_single_000340",
        "adubench_single_000341",
        "adubench_single_000344",
        "adubench_single_000101",
        "adubench_single_000649",
        "adubench_single_000146",
        "adubench_single_000787",
        "adubench_cross_000151",
    ],
}

EXPECTED_LABELS = {
    "reference_chain": "reference_chain",
    "aggregation": "aggregation",
    "comparison": "comparison",
}
VISUAL_TASK_TERMS = re.compile(r"\b(chart|table|figure|diagram|graph|slide|map)\b", re.IGNORECASE)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    original = ROOT / "original"
    doc_rows = read_jsonl(original / "documents.jsonl")
    doc_meta = {row["document_id"]: row for row in doc_rows}
    filename_by_id = {
        doc_id: Path(unquote(urlparse(row["url"]).path)).name
        for doc_id, row in doc_meta.items()
    }
    questions = {}
    for filename in ("qa_single_doc.jsonl", "qa_cross_doc.jsonl"):
        questions.update({row["question_id"]: row for row in read_jsonl(original / filename)})

    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for category, question_ids in SELECTED.items():
        imported = []
        page_map: dict[str, set[int]] = {}
        doc_ids: set[str] = set()
        for number, question_id in enumerate(question_ids, 1):
            row = questions[question_id]
            if row["metadata"].get("reasoning_type") != EXPECTED_LABELS[category]:
                raise ValueError(f"Wrong reasoning label for {question_id}")
            if row["metadata"].get("is_unanswerable"):
                raise ValueError(f"Unexpected unanswerable question: {question_id}")
            if row["metadata"].get("evidence_sources") != ["Text"]:
                raise ValueError(f"Question is not text-only: {question_id}")
            if VISUAL_TASK_TERMS.search(row["question"]):
                raise ValueError(f"Question explicitly refers to visual evidence: {question_id}")

            source_rows = [row["document"]] if "document" in row else row["documents"]
            refs = []
            for source in source_rows:
                doc_id = source["document_id"]
                # evidence_pages is the release-context PDF page index field.
                # evidence_items[].pages can refer to printed/annotator pagination.
                pages = sorted({
                    page for page in source.get("evidence_pages", [])
                    if isinstance(page, int) and not isinstance(page, bool) and page > 0
                })
                if not pages:
                    continue
                doc = doc_meta[doc_id]
                if any(
                    evidence.get("source_type") not in (None, "Text")
                    for evidence in source.get("evidence_items", [])
                ):
                    raise ValueError(f"Non-text evidence item in {question_id}")
                if max(pages) > int(doc.get("page_count") or 0):
                    raise ValueError(f"Gold page out of recorded PDF bounds in {question_id}: {doc_id}")
                refs.append({"document_name": filename_by_id[doc_id], "pages": pages})
                doc_ids.add(doc_id)
                page_map.setdefault(doc_id, set()).update(pages)

            expected_docs = int(row["metadata"].get("n_evidence_documents") or 0)
            if len(refs) != expected_docs:
                raise ValueError(
                    f"Evidence document count mismatch for {question_id}: {len(refs)} != {expected_docs}"
                )
            imported.append({
                "number": number,
                "question": row["question"],
                "expected_answer": str(row["answer"]["value"]),
                "question_sources": [],
                "answer_sources": refs,
            })

        (OUTPUT / f"{category}.json").write_text(
            json.dumps({"questions": imported}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest[category] = {
            "count": len(imported),
            "question_ids": question_ids,
            "source_documents": [
                {
                    "document_id": doc_id,
                    "filename": filename_by_id[doc_id],
                    "title": doc_meta[doc_id].get("title"),
                    "url": doc_meta[doc_id].get("url"),
                    "recorded_page_count": doc_meta[doc_id].get("page_count"),
                    "recorded_size_bytes": doc_meta[doc_id].get("file_size_bytes"),
                    "gold_pages": sorted(page_map[doc_id]),
                }
                for doc_id in sorted(doc_ids)
            ],
            "unique_gold_document_page_count": sum(map(len, page_map.values())),
        }

    manifest["totals"] = {
        "questions": sum(value["count"] for key, value in manifest.items() if key != "totals"),
        "unique_source_documents": len({
            doc["document_id"]
            for key, value in manifest.items() if key != "totals"
            for doc in value["source_documents"]
        }),
        "unique_gold_document_page_pairs": len({
            (doc["document_id"], page)
            for key, value in manifest.items() if key != "totals"
            for doc in value["source_documents"]
            for page in doc["gold_pages"]
        }),
    }
    (OUTPUT / "selection_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
