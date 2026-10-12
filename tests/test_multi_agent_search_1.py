from __future__ import annotations

from types import SimpleNamespace

import pytest

from manual_graphrag.retrieval import (
    RetrievalConfig, RetrievalContext, RETRIEVAL_STRATEGIES,
)
from strategies.implementations.multi_agent_search_1 import strategy as multi_agent


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
        strategy_id="multi_agent_search_1", top_k=4,
        params={
            "candidate_top_k": 4, "seed_top_k": 2, "max_tool_calls": 3,
            "result_limit": 4, "max_runtime_seconds": 20.0,
            "effective_search_ratio": 3, "max_parallel_agents": 2,
        },
    ).validated()


def test_multi_agent_search_1_is_registered_from_yaml():
    assert "multi_agent_search_1" in RETRIEVAL_STRATEGIES.ids()
    assert isinstance(
        RETRIEVAL_STRATEGIES.get("multi_agent_search_1"),
        multi_agent.MultiAgentSearch1Strategy,
    )


def test_multi_agent_search_1_scopes_one_worker_per_document(monkeypatch):
    context = _context([
        {"document_id": "doc-a", "file_name": "a.pdf"},
        {"document_id": "doc-b", "file_name": "b.pdf"},
    ])
    observed = []

    def fake_retrieve(_self, scoped_context, worker_config):
        observed.append((scoped_context.document_id, scoped_context.retrieval_query, worker_config.strategy_id))
        return [{
            "evidence_id": f"evidence-{scoped_context.document_id}",
            "document_id": scoped_context.document_id,
            "text": f"{scoped_context.document_id} 原文",
            "score": 0.8,
            "fusion_score": 0.8,
            "source_pages": [1],
        }]

    monkeypatch.setattr(
        multi_agent.AgenticSearch3RetrievalStrategy,
        "retrieve", fake_retrieve,
    )
    monkeypatch.setattr(
        multi_agent.MultiAgentSearch1Strategy,
        "_draft_answer", staticmethod(lambda _ctx, doc, _ev: f"{doc['file_name']} 答案"),
    )

    results = multi_agent.MultiAgentSearch1Strategy().retrieve(context, _config())

    assert {row[0] for row in observed} == {"doc-a", "doc-b"}
    assert all("WHERE node.document_id = $document_id" in row[1] for row in observed)
    assert all(row[2] == "agentic_search_3" for row in observed)
    assert {item["document_agent_name"] for item in results} == {"a.pdf", "b.pdf"}
    assert {item["document_agent_answer"] for item in results} == {"a.pdf 答案", "b.pdf 答案"}
    assert all(item["multi_agent_documents"] == ["a.pdf", "b.pdf"] for item in results)
    assert all(item["multi_agent_answers"] == {
        "a.pdf": "a.pdf 答案", "b.pdf": "b.pdf 答案",
    } for item in results)


def test_multi_agent_search_1_rejects_merged_graph():
    context = _context([{"document_id": "", "file_name": "legacy.pdf"}])
    with pytest.raises(ValueError, match="每份文件各自分區"):
        multi_agent.MultiAgentSearch1Strategy().retrieve(context, _config())
