# CiteVQA target-2 evidence-retrieval candidate

This folder contains the public CiteVQA annotations and PDF catalog, plus a reproducible candidate subset for the project's page-based evidence metrics.

## Candidate selection

`prepare_text_multidoc_subset.py` selects multi-document (`Multi (N-Gold)`) questions that:

- have at least two distinct necessary evidence documents;
- use only text-like necessary evidence (no table, image, or equation evidence);
- are labeled `Factual Retrieval` or `Complex Synthesis`;
- use no more than 40 unique source PDFs, with each PDF no larger than 15 MB or 80 pages.

The selected subset currently contains 37 questions across 40 PDFs. Its estimated total PDF size is approximately 74 MB. The candidate has factual-retrieval and complex-synthesis questions and retains the original question IDs.

Run `python3 datasets/citevqa/prepare_text_multidoc_subset.py` from the repository root to rebuild:

- `questions_candidate_page_mapping_unverified.json`: project-shaped question set with Gold `answer_sources` grouped by PDF and page;
- `source_manifest_candidate.json`: source URLs, file sizes, page counts, domains, selected question IDs, and validation caveats.

## Important access and page-number caveat

The question JSON is structurally compatible with the project's import format, but it is not yet cleared for retrieval evaluation. CiteVQA's README states that its PDF collection requires an approved access application on ModelScope and restricts CiteVQA to academic, non-commercial use. Do not bypass that access process or redistribute the PDFs.

The source annotations provide `source_page_id`, but describe it as 0-based or dataset-defined. The candidate JSON provisionally copies positive page IDs without an offset and checks only that they are within each PDF's reported page count. After obtaining approved PDFs, verify each Gold page against the PDF's physical page index; only then should the file be renamed for normal import and used to calculate page-based metrics.

Every selected question has at least two Gold evidence documents. `question_sources` is empty because the annotation provides answer evidence, not a separate provenance for the wording of the question. The source-language field is preserved in the manifest; questions may be bilingual.

## Source and citation

- Dataset: [OpenDataLab CiteVQA](https://huggingface.co/datasets/opendatalab/CiteVQA)
- PDF access collection: [CiteVQA_PDF on ModelScope](https://www.modelscope.cn/datasets/risemds/CiteVQA_PDF)
- Paper: Ma et al., *CiteVQA: Benchmarking Evidence Attribution for Trustworthy Document Intelligence*, arXiv:2605.12882.

The JSON annotations are public, but the original source PDFs remain subject to their respective rights and the dataset's access terms.
