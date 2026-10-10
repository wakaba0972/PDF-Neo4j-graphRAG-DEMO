from __future__ import annotations

from types import SimpleNamespace

from manual_graphrag.retrieval import RetrievalConfig
from strategies.implementations.agentic import strategy as agentic


class FakeSession:
    def __init__(self):
        self.queries = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def run(self, query, **params):
        self.queries.append((query, params))
        if "WHERE evidence.evidence_id IN $evidence_ids" in query:
            return SimpleNamespace(data=lambda: [{
                "entity_id": "entity-a", "name": "A",
                "type": "元件", "description": "起始元件",
                "seed_evidence_id": "seed",
            }])
        if "MATCH (current:ExtractedEntity" in query:
            current_ids = params["entity_ids"]
            name = "B" if "entity-a" in current_ids else "C"
            current_id = "entity-a" if name == "B" else "entity-b"
            entity_id = "entity-b" if name == "B" else "entity-c"
            return SimpleNamespace(data=lambda: [{
                "current_id": current_id, "current_name": current_id[-1].upper(),
                "neighbor_id": entity_id, "name": name, "type": "元件",
                "description": f"{name} 元件", "relation_type": "影響",
                "relation_description": f"關係到 {name}",
                "source_name": current_id[-1].upper(), "target_name": name,
                "source_pages": [1], "source_documents": ["manual.pdf"],
                "degree": 2,
            }])
        if "MENTIONS_ENTITY" in query and "elementId(entity) IN $entity_ids" in query:
            return SimpleNamespace(data=lambda: [{
                "evidence": {
                    "evidence_id": "chunk-b", "kind": "原文", "name": "",
                    "source": "", "target": "", "text": "B 影響的原文",
                    "source_pages": [2], "source_chunk_numbers": [2],
                    "source_documents": ["manual.pdf"], "source_references_json": "[]",
                },
                "entity_ids": ["entity-b"],
            }, {
                "evidence": {
                    "evidence_id": "chunk-c", "kind": "原文", "name": "",
                    "source": "", "target": "", "text": "C 影響的原文",
                    "source_pages": [3], "source_chunk_numbers": [3],
                    "source_documents": ["manual.pdf"], "source_references_json": "[]",
                },
                "entity_ids": ["entity-c"],
            }])
        raise AssertionError(f"unexpected Neo4j query: {query}")


class FakeDriver:
    def __init__(self):
        self.session_instance = FakeSession()

    def session(self, **_kwargs):
        return self.session_instance


class FakeVectorRetriever:
    def __init__(self, **kwargs):
        pass

    def search(self, **kwargs):
        evidence = {
            "evidence_id": "seed", "kind": "原文", "name": "",
            "source": "", "target": "", "text": "A 的起始原文",
            "source_pages": [1], "source_chunk_numbers": [1],
            "source_documents": ["manual.pdf"], "source_references_json": "[]",
            "score": 0.9, "fusion_score": 0.9, "matched_by": ["agentic-seed"],
        }
        return SimpleNamespace(items=[SimpleNamespace(content=evidence)])


def test_agentic_strategy_uses_llm_choices_and_real_graph_hops(monkeypatch):
    driver = FakeDriver()
    monkeypatch.setattr(agentic, "VectorCypherRetriever", FakeVectorRetriever)
    calls = []

    def choose_path(url, payload, api_key):
        calls.append((url, payload, api_key))
        return {"choices": [{"message": {"content": '{"expand":["0"]}'}}]}

    monkeypatch.setattr(agentic, "_post_json", choose_path)
    context = SimpleNamespace(
        driver=driver, database="neo4j", run_id="run-1", question="A 會影響什麼？",
        embedding=[0.1], vector_index_name="index", fulltext_index_name="fulltext",
        retrieval_query="RETURN evidence", result_formatter=agentic._format_record,
        retry=lambda operation: operation(), model_base_url="https://api.example/v1",
        model_api_key="key", model="gpt-4.1-mini", reasoning_effort="low",
    )
    config = RetrievalConfig(
        strategy_id="agentic", top_k=3,
        params={
            "candidate_top_k": 3, "seed_top_k": 1, "max_hops": 2,
            "max_branches": 2, "hop_decay": 0.8, "effective_search_ratio": 3,
        },
    ).validated()

    results = agentic.AgenticRetrievalStrategy().retrieve(context, config)

    assert [item["evidence_id"] for item in results] == ["seed", "chunk-b", "chunk-c"]
    assert results[1]["matched_by"] == ["graph", "agentic-graph"]
    assert results[1]["graph_hops"] == 1
    assert results[2]["graph_hops"] == 2
    assert len(calls) == 2
    assert all("EXTRACTED_RELATION" in query for query, _params in driver.session_instance.queries
               if "MATCH (current:ExtractedEntity" in query)
    assert all("Cypher" in call[1]["messages"][0]["content"] for call in calls)


def test_agentic_strategy_invalid_llm_selection_stops_expansion(monkeypatch):
    monkeypatch.setattr(agentic, "_post_json", lambda *_args: {
        "choices": [{"message": {"content": '{"expand":["999"]}'}}],
    })
    candidate = {"entity_id": "e1", "name": "候選", "path": [], "relationship": {}}
    context = SimpleNamespace(
        model_base_url="https://api.example/v1", model_api_key="key",
        model="gpt-4.1-mini", reasoning_effort="low",
    )

    assert agentic.AgenticRetrievalStrategy._choose_next(context, [candidate], "問題", 2) == []
