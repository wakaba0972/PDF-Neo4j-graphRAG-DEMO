from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from manual_graphrag.retrieval import (
    RETRIEVAL_STRATEGIES,
    RetrievalConfig,
    RetrievalContext,
)
from strategies.implementations.multi_agent_search_3 import strategy as multi_agent


class _Transaction:
    def __init__(self, rows):
        self.rows = rows
        self.options = None
        self.query = ""
        self.parameters = {}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def run(self, query, parameters=None):
        self.query = query
        self.parameters = parameters or {}
        return SimpleNamespace(data=lambda: self.rows)


class _Session:
    def __init__(self, rows):
        self.transaction = _Transaction(rows)
        self.options = None

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def begin_transaction(self, **options):
        self.transaction.options = options
        return self.transaction

    def run(self, query, parameters=None):
        return self.transaction.run(query, parameters)


class _Driver:
    def __init__(self, rows):
        self.rows = rows
        self.sessions = []

    def session(self, **options):
        session = _Session(self.rows)
        session.options = options
        self.sessions.append(session)
        return session


def _context(driver):
    return RetrievalContext(
        driver=driver, database="neo4j", question="比較兩份手冊的保養週期",
        embedding=[0.1], vector_index_name="vectors", fulltext_index_name="fulltext",
        retrieval_query="RETURN evidence", result_formatter=None,
        retry=lambda operation: operation(), model_base_url="https://api.example/v1",
        model_api_key="secret", model="gpt-4.1-mini", reasoning_effort="low",
    )


def _config():
    return RetrievalConfig(
        strategy_id="multi_agent_search_3", top_k=5,
        params={
            "candidate_top_k": 5, "result_limit": 4,
            "max_orchestration_steps": 4, "max_parallel_agents": 2,
            "worker_max_tool_calls": 2, "max_runtime_seconds": 30.0,
        },
    ).validated()


def test_strategy_is_registered_for_document_partitioned_graphs():
    assert "multi_agent_search_3" in RETRIEVAL_STRATEGIES.ids()
    spec = RETRIEVAL_STRATEGIES.get_spec("multi_agent_search_3")
    assert spec.supported_graph_modes == ("per_document",)
    assert spec.parameters["max_orchestration_steps"].default == 8
    assert isinstance(
        RETRIEVAL_STRATEGIES.get("multi_agent_search_3"),
        multi_agent.MultiAgentSearch3Strategy,
    )


@pytest.mark.parametrize("query", [
    "MATCH (n:GraphEvidence) SET n.text = 'changed' RETURN n AS evidence",
    "MATCH (n:GraphEvidence) DETACH DELETE n RETURN n AS evidence",
    "MATCH (n:GraphEvidence) CALL db.labels() YIELD label RETURN n AS evidence",
    "MATCH (n:GraphEvidence) RETURN n AS evidence LIMIT 5",
    "MATCH p=(a:ExtractedEntity)-[:EXTRACTED_RELATION*..99]->(b) RETURN a AS evidence",
    "MATCH (n:UnknownLabel) RETURN n AS evidence",
    "MATCH (n:GraphEvidence) RETURN n AS evidence; CREATE (x)",
])
def test_cypher_validator_rejects_writes_admin_unbounded_and_unscoped_features(query):
    with pytest.raises(ValueError):
        multi_agent._validate_read_only_cypher(query)


def test_cypher_validator_accepts_bounded_read_only_graph_query():
    query = (
        "MATCH (seed:GraphEvidence)-[:MENTIONS_ENTITY|REPRESENTS]-"
        "(entity:ExtractedEntity)-[:EXTRACTED_RELATION*1..3]-(other:ExtractedEntity) "
        "MATCH (evidence:GraphEvidence)-[:MENTIONS_ENTITY]-(other) "
        "WHERE seed.kind = '原文' RETURN evidence { .evidence_id, .text, .document_id } AS evidence"
    )
    assert multi_agent._validate_read_only_cypher(query) == query


def test_read_only_query_is_wrapped_scoped_and_sanitized():
    evidence = {
        "evidence_id": "doc-a:chunk-1", "document_id": "doc-a",
        "kind": "原文", "text": "證據", "embedding": [0.5],
        "source_pages": [1], "source_documents": ["a.pdf"],
    }
    driver = _Driver([{"evidence": evidence}])
    context = _context(driver)

    results = multi_agent._run_read_only_cypher(
        context,
        "MATCH (e:GraphEvidence) RETURN e { .evidence_id, .kind, .text, .document_id, .source_pages, .source_documents } AS evidence",
        {}, ["doc-a"], {"doc-a", "doc-b"}, 6,
    )

    assert len(results) == 1
    assert "embedding" not in results[0]
    assert results[0]["evidence_id"] == "doc-a:chunk-1"
    session = driver.sessions[0]
    assert session.options["default_access_mode"] == multi_agent.READ_ACCESS
    assert session.transaction.options["timeout"] == 5.0
    assert "WITH evidence WHERE evidence.document_id IN $__scope_document_ids" in session.transaction.query
    assert session.transaction.parameters["__scope_document_ids"] == ["doc-a"]
    assert session.transaction.parameters["__result_limit"] == 6


def test_orchestrator_can_choose_only_one_document_agent(monkeypatch):
    docs = [
        {"document_id": "doc-a", "file_name": "a.pdf"},
        {"document_id": "doc-b", "file_name": "b.pdf"},
    ]
    driver = _Driver(docs)
    context = _context(driver)
    tool_call = {
        "id": "delegate-1", "type": "function",
        "function": {
            "name": "delegate_document_agents",
            "arguments": json.dumps({"tasks": [{
                "document_id": "doc-b", "task": "找出保養週期",
            }]}),
        },
    }
    turns = iter([
        {"role": "assistant", "content": None, "tool_calls": [tool_call]},
        {"role": "assistant", "content": "b.pdf 證據足以回答。"},
    ])
    monkeypatch.setattr(multi_agent, "_call_model", lambda *_args, **_kwargs: next(turns))
    worker_calls = []

    def fake_worker(_context, document, task, _config, _deadline):
        worker_calls.append((document["document_id"], task))
        return {
            "document": document,
            "answer": "每 12 個月保養。",
            "evidence": [{
                "evidence_id": "doc-b:ev-1", "document_id": "doc-b",
                "kind": "原文", "text": "每 12 個月保養。", "source_pages": [4],
                "score": 0.8, "fusion_score": 0.8,
            }],
        }

    monkeypatch.setattr(multi_agent.MultiAgentSearch3Strategy, "_worker", staticmethod(fake_worker))
    result = multi_agent.MultiAgentSearch3Strategy().retrieve(context, _config())

    assert worker_calls == [("doc-b", "找出保養週期")]
    assert [item["evidence_id"] for item in result] == ["doc-b:ev-1"]
    assert result[0]["multi_agent_answers"] == {"b.pdf": "每 12 個月保養。"}
    assert result[0]["multi_agent_documents"] == ["a.pdf", "b.pdf"]


def test_orchestrator_can_search_graph_itself_with_generated_cypher(monkeypatch):
    evidence = {
        "evidence_id": "doc-a:ev-1", "document_id": "doc-a",
        "kind": "原文", "text": "每 12 個月保養。", "source_pages": [3],
    }

    class _QueuedDriver:
        def __init__(self):
            self.rows = [
                [
                    {"document_id": "doc-a", "file_name": "a.pdf"},
                    {"document_id": "doc-b", "file_name": "b.pdf"},
                ],
                [{"evidence": evidence}],
            ]
            self.queries = []

        def session(self, **_kwargs):
            rows = self.rows.pop(0)
            driver = self

            class _ReadSession:
                def __enter__(self):
                    return self

                def __exit__(self, *_args):
                    return None

                def begin_transaction(self, **_options):
                    return self

                def run(self, query, parameters=None):
                    driver.queries.append((query, parameters or {}))
                    return SimpleNamespace(data=lambda: rows)

            return _ReadSession()

    driver = _QueuedDriver()
    context = _context(driver)
    tool_call = {
        "id": "query-1", "type": "function",
        "function": {
            "name": "run_read_only_cypher",
            "arguments": json.dumps({
                "cypher": (
                    "MATCH (e:GraphEvidence) "
                    "RETURN e { .evidence_id, .kind, .text, .document_id, .source_pages } AS evidence"
                ),
                "document_ids": ["doc-a"], "limit": 10,
            }),
        },
    }
    turns = iter([
        {"role": "assistant", "content": None, "tool_calls": [tool_call]},
        {"role": "assistant", "content": "已找到 a.pdf 第 3 頁證據。"},
    ])
    monkeypatch.setattr(multi_agent, "_call_model", lambda *_args, **_kwargs: next(turns))

    result = multi_agent.MultiAgentSearch3Strategy().retrieve(context, _config())

    assert [item["evidence_id"] for item in result] == ["doc-a:ev-1"]
    assert "CALL {" in driver.queries[1][0]
    assert driver.queries[1][1]["__scope_document_ids"] == ["doc-a"]
    assert result[0]["multi_agent_discussion"][0]["phase"] == "主 Agent 查詢第 1 步"


def test_luna_uses_responses_api_function_tools(monkeypatch):
    context = SimpleNamespace(
        model_base_url="https://api.example/v1", model_api_key="secret",
        model="gpt-6-luna", reasoning_effort="low",
    )
    captured = {}

    def fake_post(url, payload, _api_key, **_kwargs):
        captured["url"] = url
        captured["payload"] = payload
        return {"output": [{
            "type": "function_call", "call_id": "call-1",
            "name": "run_read_only_cypher", "arguments": '{"document_ids":["doc-a"]}',
        }]}

    monkeypatch.setattr(multi_agent, "_post_json", fake_post)
    message = multi_agent._call_model(
        context,
        [{"role": "system", "content": "系統"}, {"role": "user", "content": "問題"}],
        multi_agent._tool_schema(allow_delegation=False), 30, require_tool=True,
    )

    assert captured["url"].endswith("/responses")
    assert captured["payload"]["instructions"] == "系統"
    assert captured["payload"]["reasoning"] == {"effort": "low"}
    assert captured["payload"]["tool_choice"] == "required"
    assert captured["payload"]["store"] is False
    assert message["tool_calls"][0]["id"] == "call-1"
    assert message["__responses_output"][0]["type"] == "function_call"


def test_chat_completions_can_require_an_initial_tool_call(monkeypatch):
    context = SimpleNamespace(
        model_base_url="https://api.example/v1", model_api_key="secret",
        model="gpt-4.1-mini", reasoning_effort="low",
    )
    captured = {}

    def fake_post(url, payload, _api_key, **_kwargs):
        captured["url"] = url
        captured["payload"] = payload
        return {"choices": [{"message": {"tool_calls": []}}]}

    monkeypatch.setattr(multi_agent, "_post_json", fake_post)
    multi_agent._call_model(
        context,
        [{"role": "system", "content": "系統"}, {"role": "user", "content": "問題"}],
        multi_agent._tool_schema(allow_delegation=True), 30, require_tool=True,
    )

    assert captured["url"].endswith("/chat/completions")
    assert captured["payload"]["tool_choice"] == "required"
