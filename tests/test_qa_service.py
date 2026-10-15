import pytest

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
        "agentic",
        [{"text": "證據", "graph_path": [{"from": "A", "type": "影響", "to": "B"}]}],
    )

    assert result["answer"] == "答案"
    assert result["evidence"][0]["graph_path"][0]["to"] == "B"
    assert captured["url"] == "http://models/v1/chat/completions"
    assert captured["api_key"] == "secret"
    assert "max_tokens" not in captured["payload"]
    system_prompt = captured["payload"]["messages"][0]["content"]
    user_prompt = captured["payload"]["messages"][1]["content"]
    assert system_prompt.startswith("You are a vehicle-document knowledge-graph QA assistant.")
    assert "answer primarily in the language of the retrieved evidence" in system_prompt
    assert "You may reason across multiple pieces of evidence" in system_prompt
    assert "every conclusion must be supported by the supplied evidence" in system_prompt
    assert "cite the source document name and the page number" in system_prompt
    assert "(Source: <document name>, p. <page number>)" in system_prompt
    assert "never guess, infer, or invent citation details" in system_prompt
    assert "state that the page number is unavailable" in system_prompt
    assert "Use plain text, not Markdown" in system_prompt
    assert "leave a blank line before each numbered item" in system_prompt
    assert "Question: 問題" in user_prompt
    assert "Evidence:" in user_prompt


def test_answer_graph_question_rejects_empty_model_completion(monkeypatch) -> None:
    monkeypatch.setattr(
        qa_service, "_post_json",
        lambda *_args, **_kwargs: {
            "choices": [{"message": {"content": ""}, "finish_reason": "length"}],
        },
    )

    with pytest.raises(ValueError, match="回答模型回傳空白內容.*finish_reason=length"):
        qa_service.answer_graph_question(
            "http://models/v1", "secret", "model-a", "問題", "agentic",
            [{"evidence_id": "ev-1", "text": "來源證據"}],
        )


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
    assert prompt.startswith("Question:")
    assert "Candidate evidence:" in prompt
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
