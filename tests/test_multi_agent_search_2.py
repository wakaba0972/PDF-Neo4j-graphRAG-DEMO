from __future__ import annotations

from types import SimpleNamespace

from manual_graphrag.retrieval import (
    RetrievalConfig, RetrievalContext, RETRIEVAL_STRATEGIES,
)
from strategies.implementations.multi_agent_search_2 import strategy as multi_agent


class _Session:
    def __init__(self, rows):
        self.rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def run(self, _query):
        return SimpleNamespace(data=lambda: self.rows)


class _Driver:
    def __init__(self, rows):
        self.rows = rows

    def session(self, **_kwargs):
        return _Session(self.rows)


def _context(rows):
    return RetrievalContext(
        driver=_Driver(rows), database="neo4j", question="問題",
        embedding=[0.1], vector_index_name="vectors", fulltext_index_name="fulltext",
        retrieval_query="RETURN node", result_formatter=None,
        retry=lambda operation: operation(), model_base_url="https://api.example/v1",
        model_api_key="key", model="model-a", reasoning_effort="low",
    )


def _config():
    return RetrievalConfig(
        strategy_id="multi_agent_search_2", top_k=4,
        params={
            "candidate_top_k": 4, "seed_top_k": 2, "max_tool_calls": 3,
            "result_limit": 4, "max_runtime_seconds": 20.0,
            "effective_search_ratio": 3, "max_parallel_agents": 2,
            "discussion_rounds": 1, "verification_rounds": 1,
        },
    ).validated()


def test_multi_agent_search_2_is_registered_for_partitioned_graphs():
    assert "multi_agent_search_2" in RETRIEVAL_STRATEGIES.ids()
    spec = RETRIEVAL_STRATEGIES.get_spec("multi_agent_search_2")
    assert spec.supported_graph_modes == ("per_document",)
    assert spec.parameters["discussion_rounds"].default == 4
    assert spec.parameters["verification_rounds"].default == 3
    assert isinstance(
        RETRIEVAL_STRATEGIES.get("multi_agent_search_2"),
        multi_agent.MultiAgentSearch2Strategy,
    )


def test_search_2_keeps_agent_sessions_and_verifies_in_each_document(monkeypatch):
    context = _context([
        {"document_id": "doc-a", "file_name": "a.pdf"},
        {"document_id": "doc-b", "file_name": "b.pdf"},
    ])
    retrieval_calls = []

    def fake_retrieve(_self, scoped_context, worker_config):
        retrieval_calls.append((
            scoped_context.document_id, scoped_context.question,
            scoped_context.retrieval_query, worker_config.strategy_id,
            worker_config.params["max_tool_calls"],
        ))
        is_verification = "請在你負責的文件中查證" in scoped_context.question
        evidence_id = (
            f"verify-{scoped_context.document_id}" if is_verification
            else f"seed-{scoped_context.document_id}"
        )
        return [{
            "evidence_id": evidence_id,
            "document_id": scoped_context.document_id,
            "text": evidence_id,
            "score": 0.8 if not is_verification else 0.2,
            "fusion_score": 0.8 if not is_verification else 0.2,
            "source_pages": [1],
            "matched_by": ["seed"],
        }]

    chat_payloads = []

    def fake_post(_url, payload, _api_key):
        chat_payloads.append(payload)
        return {"choices": [{"message": {"content": f"Agent 訊息 {len(chat_payloads)}"}}]}

    monkeypatch.setattr(multi_agent.AgenticSearch3RetrievalStrategy, "retrieve", fake_retrieve)
    monkeypatch.setattr(
        multi_agent.MultiAgentSearch1Strategy, "_draft_answer",
        staticmethod(lambda _ctx, document, _evidence: f"{document['file_name']} 初步答案"),
    )
    monkeypatch.setattr(multi_agent, "_post_json", fake_post)

    results = multi_agent.MultiAgentSearch2Strategy().retrieve(context, _config())

    assert len(chat_payloads) == 4  # one discussion and one verification turn per document
    assert all(call[3] == "agentic_search_3" for call in retrieval_calls)
    assert {call[0] for call in retrieval_calls} == {"doc-a", "doc-b"}
    assert all("WHERE node.document_id = $document_id" in call[2]
               for call in retrieval_calls)
    assert all(call[4] <= 2 for call in retrieval_calls if "請在你負責的文件中查證" in call[1])
    assert {item["evidence_id"] for item in results} == {
        "seed-doc-a", "seed-doc-b", "verify-doc-a", "verify-doc-b",
    }
    assert all(any("verification" in marker for marker in item["matched_by"])
               for item in results if item["evidence_id"].startswith("verify"))
    assert all(set(item["multi_agent_answers"]) == {"a.pdf", "b.pdf"}
               for item in results)
    discussion = next(item["multi_agent_discussion"] for item in results
                      if item.get("multi_agent_discussion"))
    assert any(item["phase"] == "討論第 1 輪" for item in discussion)
    assert any(item["phase"] == "查證第 1 輪" for item in discussion)

