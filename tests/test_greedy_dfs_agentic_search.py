from __future__ import annotations

from types import SimpleNamespace

from manual_graphrag.retrieval import RetrievalConfig, RETRIEVAL_STRATEGIES
from strategies.implementations.greedy_dfs_agentic_search import strategy as greedy_dfs


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
        retrieval_query="RETURN evidence", result_formatter=greedy_dfs._format_record,
        retry=lambda operation: operation(), model_base_url="https://api.example/v1",
        model_api_key="secret", model="gpt-4.1-mini", reasoning_effort="high",
    )


def _config():
    return RetrievalConfig(
        strategy_id="greedy_dfs_agentic_search",
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


def test_greedy_dfs_is_registered_and_has_dfs_parameters():
    strategy_id = "greedy_dfs_agentic_search"
    assert strategy_id in RETRIEVAL_STRATEGIES.ids()
    assert RETRIEVAL_STRATEGIES.get(strategy_id).strategy_id == strategy_id
    spec = RETRIEVAL_STRATEGIES.get_spec(strategy_id)
    assert spec.postprocessors == ()
    assert spec.parameters["max_depth"].default == 5
    assert spec.parameters["branch_width"].default == 3


def test_greedy_dfs_goes_deep_first_then_backtracks_to_ranked_sibling(monkeypatch):
    monkeypatch.setattr(greedy_dfs, "VectorCypherRetriever", FakeVectorRetriever)
    visited = []

    def follow(_context, evidence_ids, _limit):
        current = evidence_ids[0]
        visited.append(current)
        adjacent = {
            "seed": ["a", "b"],
            "a": ["c"],
            "c": [],
            "b": ["d"],
            "d": [],
        }.get(current, [])
        return [{"evidence": _evidence(item), "score": 0.0} for item in adjacent]

    monkeypatch.setattr(
        greedy_dfs.search2.AgenticSearch2RetrievalStrategy,
        "_follow_one_hop", staticmethod(follow),
    )
    decisions = iter([
        {"answer_supported": False, "next_evidence_ids": ["a", "b"]},
        {"answer_supported": False, "next_evidence_ids": ["c"]},
        {"answer_supported": False, "next_evidence_ids": []},
        {"answer_supported": False, "next_evidence_ids": ["d"]},
        {"answer_supported": True, "next_evidence_ids": []},
    ])
    paths_seen = []

    def decide(_context, path, _candidates):
        paths_seen.append([item["evidence_id"] for item in path])
        return next(decisions)

    monkeypatch.setattr(greedy_dfs.GreedyDfsAgenticSearchRetrievalStrategy,
                        "_decide", staticmethod(decide))
    monkeypatch.setattr(
        greedy_dfs.search2.AgenticSearch2RetrievalStrategy,
        "_selection_arguments", staticmethod(lambda _ctx, _pool, _k: ["d", "seed"]),
    )

    results = greedy_dfs.GreedyDfsAgenticSearchRetrievalStrategy().retrieve(_context(), _config())

    assert visited == ["seed", "a", "c", "b", "d"]
    assert paths_seen == [
        ["seed"], ["seed", "a"], ["seed", "a", "c"],
        ["seed", "b"], ["seed", "b", "d"],
    ]
    assert [item["evidence_id"] for item in results] == ["d", "seed"]
    assert all(item["greedy_dfs_answer_path_found"] for item in results)


def test_greedy_dfs_rejects_llm_ids_not_in_current_neighbors(monkeypatch):
    context = _context()
    context.model = "gpt-4.1-mini"
    request = {}

    def post_json(_url, payload, _key):
        request.update(payload)
        return {"choices": [{"message": {"tool_calls": [{"function": {
            "arguments": '{"answer_supported":false,"next_evidence_ids":["unknown","known"]}',
        }}]}}]}

    monkeypatch.setattr(greedy_dfs, "_post_json", post_json)
    decision = greedy_dfs.GreedyDfsAgenticSearchRetrievalStrategy._decide(
        context, [_evidence("seed")], [_evidence("known")],
    )

    assert decision == {"answer_supported": False, "next_evidence_ids": ["known"]}
    assert request["tool_choice"]["function"]["name"] == "choose_dfs_next_step"
    assert "depth-first" in request["messages"][0]["content"]


def test_greedy_dfs_uses_responses_api_for_luna(monkeypatch):
    context = _context()
    context.model = "gpt-6-luna"
    request = {}

    def post_json(url, payload, _key):
        request.update({"url": url, **payload})
        return {"output": [{
            "type": "function_call", "name": "choose_dfs_next_step",
            "arguments": '{"answer_supported":false,"next_evidence_ids":["next"]}',
        }]}

    monkeypatch.setattr(greedy_dfs, "_post_json", post_json)
    decision = greedy_dfs.GreedyDfsAgenticSearchRetrievalStrategy._decide(
        context, [_evidence("seed")], [_evidence("next")],
    )

    assert decision["next_evidence_ids"] == ["next"]
    assert request["url"].endswith("/responses")
    assert request["reasoning"] == {"effort": "high"}


def test_greedy_dfs_keeps_seed_if_all_dfs_decisions_fail(monkeypatch):
    monkeypatch.setattr(greedy_dfs, "VectorCypherRetriever", FakeVectorRetriever)
    monkeypatch.setattr(
        greedy_dfs.search2.AgenticSearch2RetrievalStrategy,
        "_follow_one_hop", staticmethod(lambda *_args: [{"evidence": _evidence("next")}]),
    )
    monkeypatch.setattr(
        greedy_dfs.GreedyDfsAgenticSearchRetrievalStrategy,
        "_decide", staticmethod(lambda *_args: (_ for _ in ()).throw(TimeoutError("timeout"))),
    )
    monkeypatch.setattr(
        greedy_dfs.search2.AgenticSearch2RetrievalStrategy,
        "_selection_arguments", staticmethod(lambda *_args: []),
    )

    results = greedy_dfs.GreedyDfsAgenticSearchRetrievalStrategy().retrieve(_context(), _config())

    assert results
    assert results[0]["evidence_id"] == "seed"
    assert "greedy-dfs-agentic-search-selection-fallback" in results[0]["matched_by"]
