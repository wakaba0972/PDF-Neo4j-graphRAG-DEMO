"""Prepare a text-evidence, multi-document CiteVQA subset for page metrics.

The CiteVQA source PDFs are not downloaded by this script. They require
controlled access through the dataset authors' ModelScope PDF collection.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ANNOTATIONS = ROOT / "CiteVQA.json"
PDF_CATALOG = ROOT / "pdf_source.csv"
QUESTION_OUTPUT = ROOT / "questions_candidate_page_mapping_unverified.json"
MANIFEST_OUTPUT = ROOT / "source_manifest_candidate.json"
TEXT_EVIDENCE_TYPES = {
    "text",
    "title",
    "header",
    "footer",
    "list",
    "ref_text",
    "page_footnote",
    "aside_text",
    "code",
}
ALLOWED_QUESTION_TYPES = {"Factual Retrieval", "Complex Synthesis"}
MAX_PDF_COUNT = 40
MAX_PDF_BYTES = 15_000_000
MAX_PDF_PAGES = 80


def required_evidence(item: dict) -> list[dict]:
    return [
        evidence
        for evidence in item.get("Evidence", [])
        if evidence.get("necessity", "necessary") == "necessary"
    ]


def eligible(item: dict, catalog: dict[str, dict]) -> bool:
    evidence = required_evidence(item)
    if (
        item.get("dataset_type") != "Multi (N-Gold)"
        or item.get("Question_Type") not in ALLOWED_QUESTION_TYPES
        or not evidence
        or len({entry.get("source_doc_index") for entry in evidence}) < 2
    ):
        return False

    if any(
        entry.get("type") not in TEXT_EVIDENCE_TYPES
        or not entry.get("source_pdf_name")
        or not isinstance(entry.get("source_page_id"), int)
        or isinstance(entry.get("source_page_id"), bool)
        or entry["source_page_id"] <= 0
        for entry in evidence
    ):
        return False

    for relative_path in item.get("PDF_Source", []):
        filename = Path(relative_path).name
        metadata = catalog.get(filename)
        if metadata is None:
            return False
        if (
            int(metadata["content_length"]) > MAX_PDF_BYTES
            or int(metadata["page_count"]) > MAX_PDF_PAGES
        ):
            return False

    # Keep the exported source pages valid against the published PDF page count.
    for entry in evidence:
        metadata = catalog.get(Path(entry["source_pdf_name"]).name)
        if metadata is None or entry["source_page_id"] > int(metadata["page_count"]):
            return False
    return True


def pick_questions(items: list[dict], catalog: dict[str, dict]) -> list[dict]:
    candidates = sorted(
        (item for item in items if eligible(item, catalog)),
        key=lambda item: item["index"],
    )
    selected: list[dict] = []
    selected_files: set[str] = set()

    while True:
        possible = []
        for item in candidates:
            if item in selected:
                continue
            files = {Path(path).name for path in item.get("PDF_Source", [])}
            if len(selected_files | files) <= MAX_PDF_COUNT:
                possible.append(
                    (
                        len(files - selected_files),
                        item["Question_Type"] != "Factual Retrieval",
                        len(files),
                        item["index"],
                        item,
                    )
                )
        if not possible:
            break

        item = min(possible, key=lambda entry: entry[:4])[-1]
        selected.append(item)
        selected_files.update(Path(path).name for path in item.get("PDF_Source", []))

    return selected


def answer_sources(item: dict) -> list[dict]:
    pages_by_document: dict[str, set[int]] = {}
    for evidence in required_evidence(item):
        filename = Path(evidence["source_pdf_name"]).name
        pages_by_document.setdefault(filename, set()).add(evidence["source_page_id"])
    return [
        {"document_name": name, "pages": sorted(pages)}
        for name, pages in sorted(pages_by_document.items())
    ]


def main() -> None:
    annotation_bytes = ANNOTATIONS.read_bytes()
    items = json.loads(annotation_bytes)
    with PDF_CATALOG.open(newline="", encoding="utf-8") as handle:
        catalog = {f"{row['track_id']}.pdf": row for row in csv.DictReader(handle)}

    selected = pick_questions(items, catalog)
    all_files = sorted(
        {
            Path(path).name
            for item in selected
            for path in item.get("PDF_Source", [])
        }
    )

    questions = []
    for number, item in enumerate(selected, start=1):
        questions.append(
            {
                "number": number,
                "question": item["Question"],
                "expected_answer": item["Standard_Answer"],
                "question_sources": [],
                "answer_sources": answer_sources(item),
                "dataset_id": item["index"],
            }
        )

    QUESTION_OUTPUT.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset": "CiteVQA Multi (N-Gold), text-evidence candidate subset",
                "questions": questions,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    source_rows = []
    for filename in all_files:
        row = catalog[filename]
        source_rows.append(
            {
                "document_name": filename,
                "source_url": row["url"],
                "size_bytes": int(row["content_length"]),
                "page_count": int(row["page_count"]),
                "source_language": row["language_label"],
                "domain": row["domain_label_en"],
                "sub_domain": row["sub_domain_label_en"],
                "access_status": "Requires ModelScope PDF dataset approval",
            }
        )

    per_type = Counter(item["Question_Type"] for item in selected)
    per_language = Counter(item["language"] for item in selected)
    manifest = {
        "dataset": "CiteVQA",
        "subset_type": "Multi (N-Gold)",
        "subset_status": "candidate; verify PDF page mapping before import",
        "source_annotation_sha256": hashlib.sha256(annotation_bytes).hexdigest(),
        "selection": {
            "questions": len(questions),
            "unique_source_pdfs": len(source_rows),
            "question_types": dict(sorted(per_type.items())),
            "source_language_labels": dict(sorted(per_language.items())),
            "max_pdf_count": MAX_PDF_COUNT,
            "max_individual_pdf_bytes": MAX_PDF_BYTES,
            "max_individual_pdf_pages": MAX_PDF_PAGES,
            "evidence_policy": "All necessary evidence elements must be text-like; table, image, and equation evidence is excluded.",
            "question_type_policy": "Keep Factual Retrieval and Complex Synthesis; exclude Multimodal Parsing and Quantitative Reasoning.",
        },
        "page_mapping": {
            "status": "unverified",
            "source_field": "Evidence[].source_page_id",
            "provisional_policy": "Copied without offset after checking 1 <= source_page_id <= published PDF page_count.",
            "warning": "Verify these page IDs against the downloaded PDFs before using evidence-retrieval metrics. The dataset card describes source_page_id as 0-based or dataset-defined.",
        },
        "license_and_access": {
            "use": "Academic research and non-commercial use only, subject to original document rights.",
            "pdf_access": "The official repository requires an approved ModelScope application for the PDF collection. PDFs are intentionally not downloaded by this preparation script.",
            "pdf_dataset_url": "https://www.modelscope.cn/datasets/risemds/CiteVQA_PDF",
        },
        "question_manifest": [
            {
                "number": question["number"],
                "dataset_id": item["index"],
                "question_type": item["Question_Type"],
                "source_language_label": item["language"],
                "gold_document_pages": question["answer_sources"],
            }
            for question, item in zip(questions, selected, strict=True)
        ],
        "source_documents": source_rows,
    }
    MANIFEST_OUTPUT.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Prepared {len(questions)} candidate questions and {len(source_rows)} PDFs; "
        f"estimated source bytes: {sum(row['size_bytes'] for row in source_rows)}."
    )
    print(f"Wrote {QUESTION_OUTPUT.name} and {MANIFEST_OUTPUT.name}.")


if __name__ == "__main__":
    main()
