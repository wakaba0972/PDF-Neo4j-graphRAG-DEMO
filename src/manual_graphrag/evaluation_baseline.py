from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


BASELINE_VERSION = 1
QUESTION_TYPES = {
    "single_fact",
    "exact_value",
    "cross_page",
    "relationship",
    "unanswerable",
    "multi_turn",
}


def load_baseline(path: str | Path) -> dict[str, Any]:
    """Load and validate a versioned, human-maintained evaluation set."""
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"無法讀取評測集：{source}") from exc
    if not isinstance(payload, dict) or payload.get("version") != BASELINE_VERSION:
        raise ValueError(f"評測集 version 必須是 {BASELINE_VERSION}")
    questions = payload.get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError("評測集 questions 必須是非空陣列")

    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for item in questions:
        if not isinstance(item, dict):
            raise ValueError("評測題目必須是 JSON 物件")
        question_id = str(item.get("id", "")).strip()
        question = str(item.get("question", "")).strip()
        question_type = str(item.get("question_type", "")).strip()
        if not question_id or question_id in seen_ids:
            raise ValueError("評測題目 id 必須存在且不可重複")
        if not question:
            raise ValueError(f"評測題目 {question_id} 缺少 question")
        if question_type not in QUESTION_TYPES:
            raise ValueError(f"評測題目 {question_id} 的 question_type 不受支援")
        answerable = item.get("answerable")
        if not isinstance(answerable, bool):
            raise ValueError(f"評測題目 {question_id} 的 answerable 必須是 boolean")
        expected_answer = str(item.get("expected_answer", "")).strip()
        if answerable and not expected_answer:
            raise ValueError(f"可回答題目 {question_id} 必須有 expected_answer")
        pages = _positive_ints(item.get("source_pages", []), "source_pages", question_id)
        chunks = _positive_ints(
            item.get("source_chunk_numbers", []), "source_chunk_numbers", question_id
        )
        if answerable and not pages and not chunks:
            raise ValueError(f"可回答題目 {question_id} 必須有來源頁碼或 chunk")
        seen_ids.add(question_id)
        normalized.append({
            "id": question_id,
            "question": question,
            "expected_answer": expected_answer,
            "source_pages": pages,
            "source_chunk_numbers": chunks,
            "question_type": question_type,
            "answerable": answerable,
        })
    return {
        "version": BASELINE_VERSION,
        "name": str(payload.get("name", "")).strip(),
        "questions": normalized,
    }


def _positive_ints(value: Any, field: str, question_id: str) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"評測題目 {question_id} 的 {field} 必須是陣列")
    result: list[int] = []
    for raw in value:
        try:
            number = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"評測題目 {question_id} 的 {field} 必須是正整數") from exc
        if number < 1:
            raise ValueError(f"評測題目 {question_id} 的 {field} 必須是正整數")
        if number not in result:
            result.append(number)
    return result


def summarize_results(
    results: list[dict[str, Any]],
    baseline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Calculate comparable retrieval, answer, latency and context metrics."""
    question_meta = {
        item["id"]: item for item in (baseline or {}).get("questions", [])
    }
    enriched = []
    for result in results:
        item = dict(result)
        question_id = str(item.get("id", "")).strip()
        if question_id in question_meta:
            item = {**question_meta[question_id], **item}
        enriched.append(item)
    answerable = [item for item in enriched if item.get("answerable", True)]
    unanswerable = [item for item in enriched if not item.get("answerable", True)]

    def rate(items: list[dict[str, Any]], field: str) -> float | None:
        if not items:
            return None
        return sum(bool(item.get(field, False)) for item in items) / len(items)

    def recall_rate(items: list[dict[str, Any]], limit: int) -> float | None:
        if not items:
            return None
        hits = []
        for item in items:
            rank = item.get("retrieval_rank")
            hits.append(
                bool(item.get(f"recall_at_{limit}", False))
                or isinstance(rank, (int, float)) and 1 <= rank <= limit
            )
        return sum(hits) / len(hits)

    ranks = [
        float(item["reciprocal_rank"])
        for item in answerable
        if isinstance(item.get("reciprocal_rank"), (int, float))
    ]
    latencies = _numbers(enriched, "latency_ms")
    context_tokens = _numbers(enriched, "context_tokens")
    return {
        "total": len(enriched),
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "recall_at_5": recall_rate(answerable, 5),
        "recall_at_10": recall_rate(answerable, 10),
        "answer_pass_rate": rate(answerable, "passed"),
        "routing_accuracy": rate(answerable, "routing_correct"),
        "abstention_accuracy": rate(unanswerable, "abstention_correct"),
        "mrr": mean(ranks) if ranks else None,
        "latency_ms_avg": mean(latencies) if latencies else None,
        "latency_ms_p95": _percentile(latencies, 0.95),
        "context_tokens_avg": mean(context_tokens) if context_tokens else None,
    }


def _numbers(items: list[dict[str, Any]], field: str) -> list[float]:
    return [
        float(item[field])
        for item in items
        if isinstance(item.get(field), (int, float))
    ]


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * percentile + 0.9999)))
    return ordered[index]


def _load_results(path: str | Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"無法讀取評測結果：{path}") from exc
    results = payload.get("results") if isinstance(payload, dict) else payload
    if not isinstance(results, list) or not all(isinstance(item, dict) for item in results):
        raise ValueError("評測結果必須是物件陣列，或包含 results 陣列的物件")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="計算 GraphRAG 固定評測集基準指標")
    parser.add_argument("baseline", type=Path, help="版本化評測集 JSON")
    parser.add_argument("results", type=Path, help="評測結果 JSON")
    parser.add_argument("--output", type=Path, help="輸出報告 JSON")
    args = parser.parse_args()
    report = summarize_results(_load_results(args.results), load_baseline(args.baseline))
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
