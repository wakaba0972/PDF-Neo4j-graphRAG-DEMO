# XL-DocBench compact subset

The local subset is prepared from the official Microsoft XL-DocBench release. It contains 10 single-document and 7 cross-document questions (17 total). Only answerable questions whose benchmark metadata lists `Text` as the evidence source are included; chart/table/image evidence questions are excluded.

Generated files are in `datasets/xl_docbench/selected/`:

- `single_document.json`
- `cross_document.json`
- `selection_manifest.json`

They use the project's question-set import schema. The manifest lists the original XL-DocBench question IDs, source filenames, and the union of Gold pages needed by the selected questions. The subset uses five source PDFs in total; single-document questions use three of them, and cross-document questions use at most three per question. Across both sets, the selected Gold annotations reference 36 unique document-page pairs. The five downloaded source PDFs contain 398 pages in total; they remain whole to preserve context for retrieval. One near-duplicate pair of date-comparison questions was omitted to keep the subset distinct.

To regenerate after downloading the official JSONL files into `datasets/xl_docbench/original/`, run:

```sh
python scripts/prepare_xl_docbench_subset.py
```

## Page/version caveat

The source annotations distinguish `evidence_pages` (PDF-index references for the release context) from `evidence_items[].pages` (annotator-supplied page references). The import files use the latter because those are the pages attached to each human-annotated supporting excerpt; they do not treat the two fields as interchangeable. The publisher-hosted PDFs may have changed since the benchmark release, so compare the source edition/page numbering before interpreting page-based retrieval metrics. The downloaded PDF files are kept locally and are not committed.

## Sources and terms

- Dataset and annotations: [Microsoft XL-DocBench on Hugging Face](https://huggingface.co/datasets/microsoft/XL-DocBench)
- Project page: [XL-DocBench](https://officeintelligence.github.io/xl-docbench/)

The dataset is marked with license `other`; source PDFs are not bundled by the dataset publisher and remain subject to their original licenses and terms. This repository stores the original metadata/question files and the locally downloaded source PDFs under the ignored `datasets/` directory; check the applicable terms before redistributing them.
