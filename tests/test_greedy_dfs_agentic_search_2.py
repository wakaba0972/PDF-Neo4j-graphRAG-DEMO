from __future__ import annotations

from types import SimpleNamespace

from manual_graphrag.retrieval import RetrievalConfig, RETRIEVAL_STRATEGIES
from strategies.implementations.greedy_dfs_agentic_search_2 import strategy as greedy_dfs_2


class FakeVectorRetriever:
    def __init__(self, **_kwargs):
        pass

    def search(self, **_kwargs):
        return SimpleNamespace(items=[SimpleNamespace(content={
            "evidence_id": "seed", "kind": "原文", "text": "起始證據",
            "source_documents": ["manual.pdf"], "source_pages": [1],
            "source_chunk_numbers": [1], "score": 0.9, "fusion_score": 0.9,
        })])


def _context():
    return SimpleNamespace(
        driver=object(), database="neo4j", question="What is the final procedure?",
        embedding=[0.1], vector_index_name="vectors", fulltext_index_name="fulltext",
        retrieval_query="RETURN evidence", result_formatter=greedy_dfs_2._format_record,
        retry=lambda operation: operation(), model_base_url="https://api.example/v1",
        model_api_key="secret", model="gpt-4.1-mini", reasoning_effort="high",
    )


def _config():
    return RetrievalConfig(
        strategy_id="greedy_dfs_agentic_search_2",
        top_k=3,
        params={
            "seed_top_k": 4, "candidate_pool_ratio": 4, "max_depth": 5,
            "branch_width": 3, "max_expansions": 20, "result_limit": 8,
            "max_runtime_seconds": 60.0, "effective_search_ratio": 3,
        },
    ).validated()


def _evidence(evidence_id):
    return {
        "evidence_id": evidence_id, "kind": "原文", "text": f"來源內容 {evidence_id}",
        "source_documents": ["manual.pdf"], "source_pages": [1],
        "source_chunk_numbers": [1], "source_references_json": "[]",
    }


def test_greedy_dfs_2_is_registered_separately_from_original():
    assert "greedy_dfs_agentic_search" in RETRIEVAL_STRATEGIES.ids()
    assert "greedy_dfs_agentic_search_2" in RETRIEVAL_STRATEGIES.ids()
    assert RETRIEVAL_STRATEGIES.get("greedy_dfs_agentic_search_2").strategy_id == (
        "greedy_dfs_agentic_search_2"
    )
    assert RETRIEVAL_STRATEGIES.get_spec("greedy_dfs_agentic_search_2").postprocessors == ()


def test_greedy_dfs_2_keeps_candidates_path_local_and_drops_pruned_sibling(monkeypatch):
    monkeypatch.setattr(greedy_dfs_2, "VectorCypherRetriever", FakeVectorRetriever)
    visited = []

    def follow(_context, evidence_ids, _limit):
        current = evidence_ids[0]
        visited.append(current)
        adjacent = {
            "seed": ["a", "b"], "a": ["c"], "c": [], "b": ["d"], "d": [],
        }.get(current, [])
        return [{"evidence": _evidence(item), "score": 0.0} for item in adjacent]

    monkeypatch.setattr(
        greedy_dfs_2.search2.AgenticSearch2RetrievalStrategy,
        "_follow_one_hop", staticmethod(follow),
    )
    decisions = iter([
        {"answer_supported": False, "keep_path_evidence_ids": ["seed"],
         "next_evidence_ids": ["a", "b"]},
        {"answer_supported": False, "keep_path_evidence_ids": ["a"],
         "next_evidence_ids": ["c"]},
        {"answer_supported": False, "keep_path_evidence_ids": ["c"],
         "next_evidence_ids": []},
        {"answer_supported": False, "keep_path_evidence_ids": ["seed", "b"],
         "next_evidence_ids": ["d"]},
        {"answer_supported": True, "keep_path_evidence_ids": ["d"],
         "next_evidence_ids": []},
    ])
    path_inputs = []
    final_pool = []

    def decide(_context, path, _adjacent):
        path_inputs.append([item["evidence_id"] for item in path])
        return next(decisions)

    def select(_context, pool, _top_k):
        final_pool.extend(item["evidence_id"] for item in pool)
        return ["d"]

    monkeypatch.setattr(greedy_dfs_2.GreedyDfsAgenticSearch2RetrievalStrategy,
                        "_decide", staticmethod(decide))
    monkeypatch.setattr(
        greedy_dfs_2.search2.AgenticSearch2RetrievalStrategy,
        "_selection_arguments", staticmethod(select),
    )

    results = greedy_dfs_2.GreedyDfsAgenticSearch2RetrievalStrategy().retrieve(
        _context(), _config(),
    )

    assert visited == ["seed", "a", "c", "b", "d"]
    assert path_inputs == [
        ["seed"], ["seed", "a"], ["a", "c"], ["seed", "b"],
        ["seed", "b", "d"],
    ]
    assert final_pool == ["d"]
    assert [item["evidence_id"] for item in results] == ["d"]
    assert results[0]["greedy_dfs_agentic_search_2_answer_path_found"] is True


def test_greedy_dfs_2_does_not_leak_abandoned_path_when_answer_is_not_found(monkeypatch):
    monkeypatch.setattr(greedy_dfs_2, "VectorCypherRetriever", FakeVectorRetriever)

    def follow(_context, evidence_ids, _limit):
        adjacent = {"seed": ["a", "b"], "a": [], "b": []}.get(evidence_ids[0], [])
        return [{"evidence": _evidence(item), "score": 0.0} for item in adjacent]

    monkeypatch.setattr(
        greedy_dfs_2.search2.AgenticSearch2RetrievalStrategy,
        "_follow_one_hop", staticmethod(follow),
    )
    decisions = iter([
        {"answer_supported": False, "keep_path_evidence_ids": ["seed"],
         "next_evidence_ids": ["a", "b"]},
        {"answer_supported": False, "keep_path_evidence_ids": ["a"],
         "next_evidence_ids": []},
        {"answer_supported": False, "keep_path_evidence_ids": ["b"],
         "next_evidence_ids": []},
    ])
    final_pool = []
    monkeypatch.setattr(
        greedy_dfs_2.GreedyDfsAgenticSearch2RetrievalStrategy,
        "_decide", staticmethod(lambda *_args: next(decisions)),
    )
    monkeypatch.setattr(
        greedy_dfs_2.search2.AgenticSearch2RetrievalStrategy,
        "_selection_arguments",
        staticmethod(lambda _ctx, pool, _k: final_pool.extend(
            item["evidence_id"] for item in pool
        ) or []),
    )

    results = greedy_dfs_2.GreedyDfsAgenticSearch2RetrievalStrategy().retrieve(
        _context(), _config(),
    )

    # At equal depth the later visited branch is the selected fallback path.
    assert final_pool == ["b"]
    assert [item["evidence_id"] for item in results] == ["b"]
    assert results[0]["greedy_dfs_agentic_search_2_answer_path_found"] is False


def test_greedy_dfs_2_filters_model_ids_and_uses_luna_responses_api(monkeypatch):
    context = _context()
    context.model = "gpt-6-luna"
    request = {}

    def post_json(url, payload, _key):
        request.update({"url": url, **payload})
        return {"output": [{
            "type": "function_call", "name": "choose_dfs_next_step",
            "arguments": '{"answer_supported":false,"keep_path_evidence_ids":["seed",'
                          '"unknown"],"next_evidence_ids":["unknown","next"]}',
        }]}

    monkeypatch.setattr(greedy_dfs_2, "_post_json", post_json)
    decision = greedy_dfs_2.GreedyDfsAgenticSearch2RetrievalStrategy._decide(
        context, [_evidence("seed")], [_evidence("next")],
    )

    assert decision == {
        "answer_supported": False,
        "keep_path_evidence_ids": ["seed"],
        "next_evidence_ids": ["next"],
    }
    assert request["url"].endswith("/responses")
    assert request["reasoning"] == {"effort": "high"}
    assert "omit irrelevant evidence" in request["instructions"]
