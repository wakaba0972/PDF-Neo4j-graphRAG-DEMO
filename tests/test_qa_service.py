from manual_graphrag import qa_service


def test_answer_graph_question_omits_max_tokens(monkeypatch) -> None:
    captured = {}

    def fake_post(url, payload, api_key, **kwargs):
        captured.update(url=url, payload=payload, api_key=api_key)
        return {
            "choices": [{
                "message": {"content": "答案"},
                "finish_reason": "stop",
            }],
        }

    monkeypatch.setattr(qa_service, "_post_json", fake_post)

    result = qa_service.answer_graph_question(
        "http://models/v1",
        "secret",
        "model-a",
        "問題",
        "基本檢索",
        [{"text": "證據"}],
        ["manual.pdf"],
    )

    assert result["answer"] == "答案"
    assert captured["url"] == "http://models/v1/chat/completions"
    assert captured["api_key"] == "secret"
    assert "max_tokens" not in captured["payload"]
    assert "只允許使用以下文件" in captured["payload"]["messages"][0]["content"]
    assert "manual.pdf" in captured["payload"]["messages"][0]["content"]


def test_answer_graph_question_bounds_evidence_context(monkeypatch) -> None:
    captured = {}

    def fake_post(url, payload, api_key, **kwargs):
        captured["payload"] = payload
        return {"choices": [{"message": {"content": "答案"}}]}

    monkeypatch.setattr(qa_service, "_post_json", fake_post)
    evidence = [
        {"evidence_id": "first", "kind": "原文", "text": "甲" * 100},
        {"evidence_id": "second", "kind": "原文", "text": "乙" * 100},
    ]

    result = qa_service.answer_graph_question(
        "http://models/v1", "secret", "model-a", "問題", "基本檢索",
        evidence, max_context_tokens=30,
    )

    assert [item["evidence_id"] for item in result["evidence"]] == ["first"]
    assert result["context_truncated"] is True
    assert result["context_tokens"] <= 30
    assert "second" not in captured["payload"]["messages"][1]["content"]


def test_fit_evidence_to_context_rejects_non_positive_budget() -> None:
    try:
        qa_service.fit_evidence_to_context([], 0)
    except ValueError as exc:
        assert "大於 0" in str(exc)
    else:
        raise AssertionError("expected a validation error")


def test_rerank_evidence_prioritizes_question_term_matches() -> None:
    evidence = [
        {
            "evidence_id": "unrelated",
            "kind": "關係",
            "text": "一般保養與清潔方式",
            "fusion_score": 0.2,
            "matched_by": ["official-hybrid"],
        },
        {
            "evidence_id": "relevant",
            "kind": "原文",
            "text": "設備出現 E01 時請重新啟動",
            "fusion_score": 0.1,
            "matched_by": ["official-hybrid"],
        },
    ]

    ranked = qa_service.rerank_evidence("E01 怎麼處理？", evidence, 1)

    assert [item["evidence_id"] for item in ranked] == ["relevant"]
    assert ranked[0]["matched_by"] == ["official-hybrid", "local-reranker"]
    assert ranked[0]["rerank_score"] > 0


def test_rerank_evidence_preserves_hybrid_order_for_equal_scores() -> None:
    evidence = [
        {"evidence_id": "first", "kind": "實體", "text": "無關"},
        {"evidence_id": "second", "kind": "實體", "text": "其他"},
    ]

    ranked = qa_service.rerank_evidence("E01", evidence, 2)

    assert [item["evidence_id"] for item in ranked] == ["first", "second"]
