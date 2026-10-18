from __future__ import annotations

from types import SimpleNamespace

import pytest

from manual_graphrag.retrieval import RetrievalConfig, RETRIEVAL_STRATEGIES
from strategies.implementations.agentic_search_1 import strategy as agentic1


class FakeTransaction:
    def __init__(self, rows):
        self.rows = rows
        self.options = None
        self.queries = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def run(self, query, parameters=None, **params):
        params = {**(parameters or {}), **params}
        self.queries.append((query, params))
        return SimpleNamespace(data=lambda: self.rows)


class FakeSession:
    def __init__(self, rows):
        self.transaction = FakeTransaction(rows)
        self.options = None

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def begin_transaction(self, **options):
        self.transaction.options = options
        return self.transaction


class FakeDriver:
    def __init__(self, rows):
        self.session_instance = FakeSession(rows)

    def session(self, **options):
        self.session_instance.options = options
        return self.session_instance


class FakeVectorRetriever:
    def __init__(self, **_kwargs):
        pass

    def search(self, **_kwargs):
        return SimpleNamespace(items=[SimpleNamespace(content={
            "evidence_id": "seed", "kind": "原文", "text": "起始證據",
            "source_documents": ["manual.pdf"], "source_pages": [1],
            "source_chunk_numbers": [10], "score": 0.9, "fusion_score": 0.9,
        })])


class EmptyVectorRetriever:
    def __init__(self, **_kwargs):
        pass

    def search(self, **_kwargs):
        return SimpleNamespace(items=[])


def _config() -> RetrievalConfig:
    return RetrievalConfig(
        strategy_id="agentic_search_1",
        top_k=5,
        params={
            "candidate_top_k": 5,
            "seed_top_k": 2,
            "max_tool_calls": 4,
            "result_limit": 3,
            "max_runtime_seconds": 45.0,
            "effective_search_ratio": 3,
        },
    ).validated()


def _context(driver):
    return SimpleNamespace(
        driver=driver, database="neo4j", question="問題？",
        embedding=[0.1], vector_index_name="vectors", fulltext_index_name="fulltext",
        retrieval_query="RETURN evidence", result_formatter=agentic1._format_record,
        retry=lambda operation: operation(), model_base_url="https://api.example/v1",
        model_api_key="secret", model="gpt-4.1-mini", reasoning_effort="low",
    )


def _call(call_id, name, arguments):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def test_agentic_search_1_is_loaded_from_yaml_registry():
    assert "agentic_search_1" in RETRIEVAL_STRATEGIES.ids()
    assert isinstance(RETRIEVAL_STRATEGIES.get("agentic_search_1"), agentic1.AgenticSearch1RetrievalStrategy)


def test_agentic_search_1_executes_llm_selected_readonly_tool_calls(monkeypatch):
    evidence = {
        "evidence_id": "text-hit", "kind": "原文", "text": "全文搜尋找到的內容",
        "source_documents": ["manual.pdf"], "source_pages": [2],
        "source_chunk_numbers": [20], "source_references_json": "[]",
    }
    driver = FakeDriver([{"evidence": evidence, "score": 0.7}])
    monkeypatch.setattr(agentic1, "VectorCypherRetriever", FakeVectorRetriever)
    turns = iter([
        {"tool_calls": [_call(
            "call-1", "search_document_text",
            '{"query":"型號設定","limit":100}',
        )]},
        {"content": "搜尋完成"},
    ])
    observed_messages = []

    def model_turn(_context, messages):
        observed_messages.append(messages)
        return next(turns)

    monkeypatch.setattr(
        agentic1.AgenticSearch1RetrievalStrategy,
        "_model_turn", staticmethod(model_turn),
    )

    results = agentic1.AgenticSearch1RetrievalStrategy().retrieve(_context(driver), _config())

    assert "You are a knowledge-graph retrieval agent" in observed_messages[0][0]["content"]
    assert "Use the retrieval tools" in observed_messages[0][1]["content"]
    assert "Search the current project’s source text" in agentic1._tool_definitions()[0]["function"]["description"]
    assert '"error"' not in observed_messages[1][-1]["content"]
    assert [item["evidence_id"] for item in results] == ["seed", "text-hit"]
    assert results[1]["agentic_search_tool"] == "search_document_text"
    assert driver.session_instance.options["default_access_mode"] == agentic1.READ_ACCESS
    assert driver.session_instance.transaction.options == {"timeout": 5.0}
    query, params = driver.session_instance.transaction.queries[0]
    assert "db.index.fulltext.queryNodes" in query
    assert "run_id" not in params
    assert params["limit"] == 3
    assert params["search_limit"] == 100


def test_agentic_search_1_only_expands_previously_observed_evidence_ids():
    context = _context(FakeDriver([]))
    with pytest.raises(ValueError, match="本次已找到的證據 ID"):
        agentic1.AgenticSearch1RetrievalStrategy._tool_result(
            "follow_graph_connections", {"evidence_ids": ["invented"]},
            context, {"known"}, 5,
        )


def test_agentic_search_1_uses_responses_api_for_luna_tool_calls(monkeypatch):
    driver = FakeDriver([{
        "evidence": {
            "evidence_id": "text-hit", "kind": "原文", "text": "全文搜尋找到的內容",
            "source_documents": ["manual.pdf"], "source_pages": [2],
            "source_chunk_numbers": [20], "source_references_json": "[]",
        },
        "score": 0.7,
    }])
    context = _context(driver)
    context.model = "gpt-6-luna"
    monkeypatch.setattr(agentic1, "VectorCypherRetriever", FakeVectorRetriever)
    requests = []
    responses = iter([
        {
            "output": [
                {"type": "reasoning", "id": "reason-1", "summary": []},
                {
                    "type": "function_call", "id": "fc-1", "call_id": "call-1",
                    "name": "search_document_text",
                    "arguments": '{"query":"型號設定","limit":3}',
                },
            ],
        },
        {"output": [{"type": "message", "role": "assistant"}], "output_text": "完成"},
    ])

    def post_json(url, payload, _key):
        requests.append((url, payload))
        return next(responses)

    monkeypatch.setattr(agentic1, "_post_json", post_json)

    results = agentic1.AgenticSearch1RetrievalStrategy().retrieve(context, _config())

    assert [item["evidence_id"] for item in results] == ["seed", "text-hit"]
    assert requests[0][0].endswith("/responses")
    assert requests[0][1]["reasoning"] == {"effort": "low"}
    assert requests[0][1]["tools"][0]["name"] == "search_document_text"
    next_input = requests[1][1]["input"]
    assert next_input[1]["type"] == "reasoning"
    assert next_input[2]["type"] == "function_call"
    assert next_input[3]["type"] == "function_call_output"
    assert next_input[3]["call_id"] == "call-1"


def test_agentic_search_1_can_start_with_fulltext_when_vector_has_no_seed(monkeypatch):
    driver = FakeDriver([{
        "evidence": {
            "evidence_id": "text-hit", "kind": "原文", "text": "全文證據",
            "source_documents": ["manual.pdf"], "source_pages": [2],
            "source_chunk_numbers": [20], "source_references_json": "[]",
        },
        "score": 0.7,
    }])
    monkeypatch.setattr(agentic1, "VectorCypherRetriever", EmptyVectorRetriever)
    turns = iter([
        {"tool_calls": [_call(
            "call-1", "search_document_text", '{"query":"重要步驟"}',
        )]},
        {"content": "已找到原文"},
    ])
    monkeypatch.setattr(
        agentic1.AgenticSearch1RetrievalStrategy,
        "_model_turn", staticmethod(lambda _context, _messages: next(turns)),
    )

    results = agentic1.AgenticSearch1RetrievalStrategy().retrieve(_context(driver), _config())

    assert [item["evidence_id"] for item in results] == ["text-hit"]


def test_document_scoped_agentic_graph_tool_filters_every_path_node():
    driver = FakeDriver([])
    context = _context(driver)
    context.document_id = "doc-a"

    agentic1.AgenticSearch1RetrievalStrategy._follow_connections(
        context, ["seed-a"], 5,
    )

    query, params = driver.session_instance.transaction.queries[-1]
    assert "all(entity IN nodes(path) WHERE entity.document_id = $document_id)" in query
    assert params["document_id"] == "doc-a"

