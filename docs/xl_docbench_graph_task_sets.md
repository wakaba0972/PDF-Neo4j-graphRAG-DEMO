# XL-DocBench GraphRAG task sets

Three 10-question subsets are generated locally from the official XL-DocBench annotations:

- `reference_chain.json`: `reasoning_type=reference_chain` (single-document cross-section references).
- `aggregation.json`: `reasoning_type=aggregation`; includes one multi-document aggregation question and single-document aggregation questions.
- `comparison.json`: `reasoning_type=comparison`; includes multi-page within-document comparisons and one cross-document comparison.

All selected questions are answerable, have `Text` evidence annotations, and do not ask the reader to interpret a chart, table, diagram, or image. An annotation label alone is not sufficient to establish this: XL-DocBench includes some visually formatted PDFs whose text annotations occur on pages with tables or graphics. The selected answers must be supported by the quoted prose, headings, or section references; the source PDFs themselves are not guaranteed to be free of visuals in unrelated content. Original benchmark IDs, source document IDs, source URLs, PDF page indices, and recorded source sizes are listed in `datasets/xl_docbench/graph_task_sets/selection_manifest.json`.

The three sets contain 30 questions total and use 17 unique PDFs across all sets (reference chain: 8 PDFs; aggregation: 10; comparison: 7). The downloaded PDFs' page counts were checked against the benchmark records; all selected source files match. Their Gold evidence spans 149 unique document-page pairs across the three sets. The aggregation set includes the only suitable answerable, text-only cross-document aggregation example found in this release; its other questions are within-document aggregation tasks.

## Prepare and download

Run the generator after the official JSONL files are available under `datasets/xl_docbench/original/`:

```sh
python scripts/prepare_xl_docbench_graph_task_sets.py
```

The generated category JSON files use the project's question-set import format. `answer_sources[].pages` is populated from XL-DocBench `evidence_pages`, which references page indices in the released PDF context. Do not substitute `evidence_items[].pages`: that field preserves annotator-supplied pagination and may differ from the PDF page index.

Download the selected PDFs and compare their page counts to the benchmark release:

```sh
python scripts/download_xl_docbench_graph_task_sources.py
```

The PDFs are kept under the ignored `datasets/xl_docbench/graph_task_sets/sources/` directory. The manifest records actual page counts and flags version mismatches. Confirm that the downloaded PDF edition/page count matches the benchmark release before interpreting page-based retrieval metrics. The dataset is labeled `other`, and source PDFs retain their original terms; do not redistribute them without checking those terms.

## Dataset source

- [Microsoft XL-DocBench dataset card and annotations](https://huggingface.co/datasets/microsoft/XL-DocBench)
- [XL-DocBench project page](https://officeintelligence.github.io/xl-docbench/)
