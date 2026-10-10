# 固定評測集格式

固定評測集應以版本控制的 JSON 保存，每題至少包含以下欄位：

```json
{
  "version": 1,
  "name": "手冊基準集 v1",
  "questions": [
    {
      "id": "manual-001",
      "question": "問題內容",
      "expected_answer": "標準答案或答案要點",
      "source_pages": [12],
      "source_chunk_numbers": [7],
      "question_type": "exact_value",
      "answerable": true
    }
  ]
}
```

`question_type` 可使用 `single_fact`、`exact_value`、`cross_page`、`relationship`、`unanswerable` 或 `multi_turn`。可回答題目必須有標準答案及頁碼或 chunk 來源；無答案題的 `expected_answer` 可留空。

使用 `python -m manual_graphrag.evaluation_baseline BASELINE.json RESULTS.json` 產生基準報告。結果檔可以直接是題目結果陣列，也可以是包含 `results` 陣列的 JSON 物件。
