import json

import pytest

from manual_graphrag.evaluation_baseline import load_baseline, summarize_results


def _write_baseline(tmp_path, questions):
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps({"version": 1, "name": "test", "questions": questions}), encoding="utf-8")
    return path


def test_load_baseline_normalizes_and_validates_questions(tmp_path) -> None:
    path = _write_baseline(tmp_path, [{
        "id": "q1", "question": "問題", "expected_answer": "答案",
        "source_pages": [2, 2], "source_chunk_numbers": [3],
        "question_type": "exact_value", "answerable": True,
    }])

    result = load_baseline(path)

    assert result["questions"] == [{
        "id": "q1", "question": "問題", "expected_answer": "答案",
        "source_pages": [2], "source_chunk_numbers": [3],
        "question_type": "exact_value", "answerable": True,
    }]


def test_load_baseline_requires_source_for_answerable_question(tmp_path) -> None:
    path = _write_baseline(tmp_path, [{
        "id": "q1", "question": "問題", "expected_answer": "答案",
        "question_type": "single_fact", "answerable": True,
    }])

    with pytest.raises(ValueError, match="來源頁碼或 chunk"):
        load_baseline(path)


def test_load_baseline_accepts_unanswerable_question_without_answer(tmp_path) -> None:
    path = _write_baseline(tmp_path, [{
        "id": "q1", "question": "文件有提到嗎？", "expected_answer": "",
        "question_type": "unanswerable", "answerable": False,
    }])

    assert load_baseline(path)["questions"][0]["answerable"] is False


def test_summarize_results_calculates_quality_and_latency_metrics() -> None:
    report = summarize_results([
        {
            "id": "q1", "recall_at_5": True, "recall_at_10": True,
            "passed": True, "routing_correct": True,
            "reciprocal_rank": 1.0, "latency_ms": 100, "context_tokens": 200,
        },
        {
            "id": "q2", "recall_at_5": False, "recall_at_10": True,
            "passed": False, "routing_correct": False,
            "retrieval_rank": 7, "reciprocal_rank": 0.25,
            "latency_ms": 300, "context_tokens": 400,
        },
        {"id": "q3", "abstention_correct": True, "latency_ms": 500},
    ], {
        "questions": [
            {"id": "q1", "answerable": True},
            {"id": "q2", "answerable": True},
            {"id": "q3", "answerable": False},
        ]
    })

    assert report["total"] == 3
    assert report["recall_at_5"] == 0.5
    assert report["recall_at_10"] == 1.0
    assert report["answer_pass_rate"] == 0.5
    assert report["abstention_accuracy"] == 1.0
    assert report["mrr"] == 0.625
    assert report["latency_ms_avg"] == 300
    assert report["latency_ms_p95"] == 500
    assert report["context_tokens_avg"] == 300
