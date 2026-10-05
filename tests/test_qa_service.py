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


def test_embedding_vectors_reports_progress_per_batch(monkeypatch) -> None:
    def fake_post(url, payload, api_key):
        return {"data": [{"index": i, "embedding": [0.1]} for i in range(len(payload["input"]))]}

    monkeypatch.setattr(qa_service, "_post_json", fake_post)
    calls = []
    vectors = qa_service.embedding_vectors(
        "http://x", "key", "embed", ["t"] * 130,
        progress_callback=lambda done, total: calls.append((done, total)),
    )
    assert len(vectors) == 130
    assert calls == [(64, 130), (128, 130), (130, 130)]
