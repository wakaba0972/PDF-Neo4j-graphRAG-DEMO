from __future__ import annotations

from types import SimpleNamespace

import pytest

from manual_graphrag.retrieval import RetrievalConfig
from strategies.implementations.agentic_search_2 import strategy as agentic2


SAFE_QUERY = """
MATCH (evidence:GraphEvidence {run_id: $run_id, evidence_id: $evidence_id})
RETURN evidence {
  .evidence_id, .kind, .name, .text, .source_documents, .source_pages,
  .source_chunk_numbers, .source_references_json
} AS evidence
LIMIT 100
"""


def test_cypher_validator_scopes_parameters_and_clamps_limit():
    query, params = agentic2.validate_readonly_cypher(
        SAFE_QUERY, {"evidence_id": "seed-1"}, "run-1", result_limit=6,
    )

    assert query.endswith("LIMIT 6")
    assert params == {"run_id": "run-1", "evidence_id": "seed-1"}


@pytest.mark.parametrize("query", [
    SAFE_QUERY.replace("RETURN evidence", "CREATE (n) RETURN evidence"),
    SAFE_QUERY.replace("{run_id: $run_id, evidence_id: $evidence_id}", "{evidence_id: $evidence_id}"),
    SAFE_QUERY.replace("GraphEvidence", "SecretLabel"),
    SAFE_QUERY.replace("LIMIT 100", "LIMIT $max_rows"),
    SAFE_QUERY + "; MATCH (x) RETURN x",
])
def test_cypher_validator_rejects_unsafe_or_unscoped_queries(query):
    with pytest.raises(ValueError):
        agentic2.validate_readonly_cypher(
            query, {"evidence_id": "seed-1"}, "run-1", result_limit=6,
        )


class FakeTransaction:
    def __init__(self):
        self.queries = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def run(self, query, **params):
        self.queries.append((query, params))
        return SimpleNamespace(data=lambda: [{
            "evidence": {
                "evidence_id": "graph-result", "kind": "原文", "text": "第二輪找到的支持證據",
                "source_documents": ["manual.pdf"], "source_pages": [3],
                "source_chunk_numbers": [8], "source_references_json": "[]",
            },
        }])


class FakeSession:
    def __init__(self):
        self.transaction = FakeTransaction()
        self.options = None
        self.transaction_options = None

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def begin_transaction(self, **options):
        self.transaction_options = options
        return self.transaction


class FakeDriver:
    def __init__(self):
        self.current_session = FakeSession()

    def session(self, **options):
        self.current_session.options = options
        return self.current_session


class FakeVectorRetriever:
    def __init__(self, **_kwargs):
        pass

    def search(self, **_kwargs):
        return SimpleNamespace(items=[SimpleNamespace(content={
            "evidence_id": "seed", "kind": "原文", "text": "初始向量證據",
            "source_documents": ["manual.pdf"], "source_pages": [1],
            "score": 0.9, "fusion_score": 0.9,
        })])


def test_agentic_search_2_uses_multiple_llm_rounds_and_read_transaction(monkeypatch):
    driver = FakeDriver()
    monkeypatch.setattr(agentic2, "VectorCypherRetriever", FakeVectorRetriever)
    plans = []

    def plan(_context, accumulated, history, *_args):
        plans.append((len(accumulated), list(history)))
        if len(plans) == 1:
            return {
                "action": "query", "cypher": SAFE_QUERY,
                "params": {"evidence_id": "seed"}, "reason": "取得關聯原文",
            }
        return {"action": "stop", "reason": "已有足夠證據"}

    monkeypatch.setattr(agentic2.AgenticSearch2RetrievalStrategy, "_plan", staticmethod(plan))
    context = SimpleNamespace(
        driver=driver, database="neo4j", run_id="current-run", question="問題？",
        embedding=[0.1], vector_index_name="index", fulltext_index_name="fulltext",
        retrieval_query="RETURN evidence", result_formatter=agentic2._format_record,
        retry=lambda operation: operation(), model_base_url="https://api.example/v1",
        model_api_key="secret", model="gpt-4.1-mini", reasoning_effort="low",
    )
    config = RetrievalConfig(
        strategy_id="agentic_search_2", top_k=5,
        params={
            "candidate_top_k": 5, "seed_top_k": 2, "max_rounds": 3,
            "result_limit": 6, "max_runtime_seconds": 45.0,
            "effective_search_ratio": 3,
        },
    ).validated()

    results = agentic2.AgenticSearch2RetrievalStrategy().retrieve(context, config)

    assert [item["evidence_id"] for item in results] == ["seed", "graph-result"]
    assert results[1]["matched_by"] == ["agentic-search-2-cypher"]
    assert plans[1][0] == 2
    assert plans[1][1][0]["new_evidence_count"] == 1
    assert driver.current_session.options["default_access_mode"] == agentic2.READ_ACCESS
    assert driver.current_session.transaction_options == {"timeout": 5.0}
    assert driver.current_session.transaction.queries[0][1]["run_id"] == "current-run"
    assert driver.current_session.transaction.queries[0][0].endswith("LIMIT 6")

