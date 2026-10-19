# MuSiQue-Ans accuracy-only sample

This directory holds only the MuSiQue `dev` split needed to build a small test set. The complete MuSiQue repository publishes 2–4-hop English QA; this project uses 100 answerable questions across all three hop counts: 40 two-hop, 35 three-hop, and 25 four-hop. Each question requires connecting supporting passages from distinct Wikipedia article titles. The deterministic sampler favors reusing supporting article titles to limit the number of documents. The sample is for answer-accuracy evaluation only and intentionally contains no Gold page citations.

## Files

- `original/musique_ans_v1.0_dev.jsonl`: source dev file, 30,439,728 bytes (2,417 records).
- `sample/questions_accuracy_100.json`: import this JSON as a question set.
- `sample/questions_accuracy_2hop.json`, `sample/questions_accuracy_3hop.json`, and `sample/questions_accuracy_4hop.json`: the same 100-question sample split by hop count (40, 35, and 25 questions).
- `sample/sources/*.txt`: plain-text article excerpts to upload as separate documents in 1-1.
- `sample/source_manifest.json`: original question IDs, source article titles/URLs, selection method, source checksum, and metric caveat.
- `sample_40docs/questions_accuracy_40docs.json`: the separate sample constrained to 40 source documents (66 answerable questions).
- `sample_40docs/questions_accuracy_2hop.json`, `sample_40docs/questions_accuracy_3hop.json`, and `sample_40docs/questions_accuracy_4hop.json`: its hop-count splits (20, 23, and 23 questions).
- `sample_40docs/sources/*.txt`: source corpus for the 40-document sample; upload these as a separate project from `sample/sources/`.
- `sample_40docs/source_manifest.json`: question IDs, source article titles/URLs, selection method, source checksum, and metric caveat for the 40-document sample.
- `prepare_accuracy_sample.py`: deterministic sample/corpus builder (`python3 datasets/musique/prepare_accuracy_sample.py` from repository root).
- `prepare_40doc_accuracy_sample.py`: deterministic 40-document sample/corpus builder.
- `prepare_hop_question_sets.py`: regenerates the six per-hop JSON files from the two original sample JSON files.

Each sampled question contributes all of its Gold supporting paragraphs (two, three, or four, depending on hop count) plus one distractor paragraph. Excerpts sharing an article title are combined into one TXT file. The TXT importer treats each file as one synthetic page to reuse the existing chunk pipeline; page-based Evidence Hit Rate/Recall/Precision/F1 must not be reported for this dataset. TXT files are UTF-8 plain text and contain no images or tables.

The MuSiQue project distributes the dataset under CC BY 4.0. The paragraph text originates from Wikipedia; this prepared subset retains article titles and source URLs in `source_manifest.json`. Attribute the source when redistributing and follow the applicable Wikipedia content license terms.

## Hop-specific question sets

The `sample` and `sample_40docs` corpora are intentionally kept separate because they contain different source-document collections. Each has a JSON question set for 2-hop, 3-hop, and 4-hop questions. The split files preserve each question's `dataset_id` and `hops` value, renumber `number` from 1 within each set, and retain empty `question_sources` / `answer_sources`; they are suitable for answer-accuracy comparisons by hop count, not evidence-recall evaluation.

To regenerate all six split files, run `python3 datasets/musique/prepare_hop_question_sets.py` from the repository root.

## Rebuild

The preparation script uses seed `20261018`, selects answerable 2-, 3-, and 4-hop items with one supporting paragraph per hop from distinct titles, favors questions that reuse already-selected support titles, and keeps one distractor per question. It removes existing `sample/sources/*.txt` before regenerating those derived excerpts.
