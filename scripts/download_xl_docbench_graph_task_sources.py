#!/usr/bin/env python3
"""Download and page-count-check the source PDFs used by XL-DocBench task sets."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1] / "datasets" / "xl_docbench"
MANIFEST = ROOT / "graph_task_sets" / "selection_manifest.json"
OUTPUT = ROOT / "graph_task_sets" / "sources"


def download(doc: dict) -> tuple[str, dict]:
    name = doc["filename"]
    path = OUTPUT / name
    if not path.is_file() or not path.read_bytes().startswith(b"%PDF-"):
        request = Request(doc["url"], headers={"User-Agent": "XL-DocBench-local-research/1.0"})
        with urlopen(request, timeout=90) as response:
            payload = response.read()
        if not payload.startswith(b"%PDF-"):
            raise ValueError(f"Downloaded content is not a PDF: {name}")
        path.write_bytes(payload)
    payload = path.read_bytes()
    actual_pages = len(PdfReader(path).pages)
    max_gold = max(doc["gold_pages"])
    result = {
        "path": str(path.relative_to(ROOT.parent.parent)),
        "downloaded_size_bytes": len(payload),
        "downloaded_page_count": actual_pages,
        "page_count_matches_release": actual_pages == int(doc["recorded_page_count"]),
        "gold_pages_in_bounds": max_gold <= actual_pages,
    }
    return name, result


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    docs = {
        doc["document_id"]: doc
        for category, data in manifest.items() if category != "totals"
        for doc in data["source_documents"]
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = {}
    failures = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(download, doc): doc["filename"] for doc in docs.values()}
        for future in as_completed(futures):
            expected_name = futures[future]
            try:
                name, info = future.result()
            except Exception as exc:
                failures[expected_name] = str(exc)
                print(f"[FAILED] {expected_name}: {exc}")
                continue
            results[name] = info
            status = "OK" if info["page_count_matches_release"] and info["gold_pages_in_bounds"] else "CHECK"
            print(f"[{status}] {name}: {info['downloaded_page_count']} pages, {info['downloaded_size_bytes']:,} bytes")

    for data in manifest.values():
        if not isinstance(data, dict) or "source_documents" not in data:
            continue
        for doc in data["source_documents"]:
            if doc["filename"] in results:
                doc["download_verification"] = results[doc["filename"]]
            if doc["filename"] in failures:
                doc["download_error"] = failures[doc["filename"]]
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    mismatches = [name for name, info in results.items() if not info["page_count_matches_release"]]
    out_of_bounds = [name for name, info in results.items() if not info["gold_pages_in_bounds"]]
    print(f"Downloaded {len(results)} of {len(docs)} source PDFs to {OUTPUT}")
    if mismatches:
        print("Page-count mismatches:", ", ".join(sorted(mismatches)))
    if out_of_bounds:
        raise SystemExit("Gold pages exceed downloaded PDF bounds: " + ", ".join(sorted(out_of_bounds)))
    if failures:
        raise SystemExit("Some source PDFs could not be downloaded: " + ", ".join(sorted(failures)))


if __name__ == "__main__":
    main()
