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
        "vector",
        [{"text": "證據"}],
    )

    assert result["answer"] == "答案"
    assert captured["url"] == "http://models/v1/chat/completions"
    assert captured["api_key"] == "secret"
    assert "max_tokens" not in captured["payload"]
    assert "目前專案" in captured["payload"]["messages"][0]["content"]
    assert "只允許使用以下文件" not in captured["payload"]["messages"][0]["content"]
    assert "每個編號項目前都要空一行" in captured["payload"]["messages"][0]["content"]
    assert "一般純文字，不要使用 Markdown 格式" in captured["payload"]["messages"][0]["content"]
    assert "粗體或斜體標記、項目符號、表格或程式碼區塊" in captured["payload"]["messages"][0]["content"]


def test_luna_answer_uses_low_reasoning_without_temperature(monkeypatch) -> None:
    captured = {}

    def fake_post(url, payload, api_key, **kwargs):
        captured.update(payload=payload)
        return {"choices": [{"message": {"content": "答案"}, "finish_reason": "stop"}]}

    monkeypatch.setattr(qa_service, "_post_json", fake_post)
    qa_service.answer_graph_question(
        "http://models/v1", "", "gpt-6-luna", "問題", "hybrid", [{"text": "證據"}],
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
        "http://models/v1", "", "gpt-6-luna", "問題", "hybrid", [{"text": "證據"}],
        "none",
    )

    assert captured["payload"]["reasoning_effort"] == "none"
    assert captured["payload"]["temperature"] == 0


def test_rerank_evidence_uses_model_ranking_and_marks_results(monkeypatch) -> None:
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
        {
            "evidence_id": "expanded",
            "kind": "原文",
            "text": "設備出現 E01 時請重新啟動",
            "fusion_score": 0,
            "matched_by": ["graph"],
        },
    ]
    captured = {}

    def fake_post(url, payload, api_key):
        captured.update(url=url, payload=payload, api_key=api_key)
        return {"choices": [{"message": {"content": '{"ranking":["2","1","0"]}'}}]}

    monkeypatch.setattr(qa_service, "_post_json", fake_post)

    ranked = qa_service.rerank_evidence(
        "http://models/v1", "secret", "gpt-4.1-mini", "E01 怎麼處理？", evidence, 1,
    )

    assert [item["evidence_id"] for item in ranked] == ["expanded"]
    assert ranked[0]["matched_by"] == ["graph", "llm-reranker"]
    assert captured["url"] == "http://models/v1/chat/completions"
    assert captured["api_key"] == "secret"
    prompt = captured["payload"]["messages"][1]["content"]
    assert "E01 怎麼處理？" in prompt
    assert '"expanded_from_graph": true' in prompt


def test_rerank_evidence_falls_back_to_retrieval_order_for_invalid_json(monkeypatch) -> None:
    evidence = [
        {"evidence_id": "first", "kind": "實體", "text": "一"},
        {"evidence_id": "second", "kind": "實體", "text": "二"},
        {"evidence_id": "third", "kind": "實體", "text": "三"},
    ]
    monkeypatch.setattr(qa_service, "_post_json", lambda *args, **kwargs: {
        "choices": [{"message": {"content": '{"ranking":["1"]}'}}],
    })

    ranked = qa_service.rerank_evidence(
        "http://models/v1", "", "model-a", "問題", evidence, 2,
    )

    assert [item["evidence_id"] for item in ranked] == ["first", "second"]


def test_legacy_reranker_remains_available_for_comparison() -> None:
    evidence = [
        {"evidence_id": "weak", "text": "保養方式", "matched_by": ["graph"]},
        {"evidence_id": "strong", "text": "E01 錯誤時重新啟動", "matched_by": ["graph"]},
    ]

    ranked = qa_service.legacy_rerank_evidence("E01 怎麼辦？", evidence, 1)

    assert [item["evidence_id"] for item in ranked] == ["strong"]
    assert ranked[0]["matched_by"] == ["graph", "legacy-reranker"]


def test_luna_reranker_uses_reasoning_effort_without_sampling(monkeypatch) -> None:
    captured = {}
    evidence = [{"evidence_id": str(index), "text": str(index)} for index in range(3)]

    def fake_post(_url, payload, _api_key):
        captured.update(payload=payload)
        return {"choices": [{"message": {"content": '{"ranking":["0","1","2"]}'}}]}

    monkeypatch.setattr(qa_service, "_post_json", fake_post)
    qa_service.rerank_evidence(
        "http://models/v1", "", "gpt-6-luna", "問題", evidence, 1, "high",
    )

    assert captured["payload"]["reasoning_effort"] == "high"
    assert "temperature" not in captured["payload"]


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
