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
    )

    assert result["answer"] == "答案"
    assert captured["url"] == "http://models/v1/chat/completions"
    assert captured["api_key"] == "secret"
    assert "max_tokens" not in captured["payload"]
    assert "目前專案" in captured["payload"]["messages"][0]["content"]
    assert "只允許使用以下文件" not in captured["payload"]["messages"][0]["content"]
    assert "每個編號項目前都要空一行" in captured["payload"]["messages"][0]["content"]


def test_luna_answer_uses_low_reasoning_without_temperature(monkeypatch) -> None:
    captured = {}

    def fake_post(url, payload, api_key, **kwargs):
        captured.update(payload=payload)
        return {"choices": [{"message": {"content": "答案"}, "finish_reason": "stop"}]}

    monkeypatch.setattr(qa_service, "_post_json", fake_post)
    qa_service.answer_graph_question(
        "http://models/v1", "", "gpt-6-luna", "問題", "混合檢索", [{"text": "證據"}],
    )

    assert captured["payload"]["reasoning_effort"] == "low"
    assert "temperature" not in captured["payload"]


def test_luna_none_reasoning_keeps_compatible_temperature(monkeypatch) -> None:
    captured = {}

    def fake_post(url, payload, api_key, **kwargs):
        captured.update(payload=payload)
        return {"choices": [{"message": {"content": "答案"}, "finish_reason": "stop"}]}

    monkeypatch.setattr(qa_service, "_post_json", fake_post)
    qa_service.answer_graph_question(
        "http://models/v1", "", "gpt-6-luna", "問題", "混合檢索", [{"text": "證據"}],
        "none",
    )

    assert captured["payload"]["reasoning_effort"] == "none"
    assert captured["payload"]["temperature"] == 0


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


def test_interleave_expanded_evidence_keeps_graph_hits_within_top_k() -> None:
    evidence = [
        {"evidence_id": f"direct-{index}", "matched_by": ["official-hybrid"]}
        for index in range(1, 7)
    ] + [
        {"evidence_id": f"graph-{index}", "matched_by": ["graph"]}
        for index in range(1, 4)
    ]

    selected = qa_service.interleave_expanded_evidence(evidence, 4)

    assert [item["evidence_id"] for item in selected] == [
        "direct-1", "graph-1", "direct-2", "graph-2",
    ]
    assert len(selected) == 4


def test_interleave_expanded_evidence_preserves_order_without_graph_hits() -> None:
    evidence = [{"evidence_id": str(index)} for index in range(4)]

    selected = qa_service.interleave_expanded_evidence(evidence, 2)

    assert [item["evidence_id"] for item in selected] == ["0", "1"]
