import pytest

from manual_graphrag.retrieval import (
    RetrievalConfig,
    RetrievalStrategyRegistry,
    strategy_id_from_label,
    strategy_label,
)


def test_retrieval_config_keeps_stable_id_and_strategy_parameters() -> None:
    config = RetrievalConfig.from_ui(
        "混合檢索", 8, "LLM Reranker", "證據擴展 V2", rerank_candidates=True,
    )

    assert config.to_dict() == {
        "strategy_id": "hybrid",
        "top_k": 8,
        "params": {
            "candidate_top_k": 24,
            "effective_search_ratio": 3,
            "ranker": "naive",
        },
        "reranker": {"id": "llm", "params": {}},
        "expansion": {"id": "graph_v2", "params": {"hops": 4}},
    }
    assert config.ui_values() == ("混合檢索", 8, "LLM Reranker", "證據擴展 V2")


def test_retrieval_config_rejects_old_shape() -> None:
    with pytest.raises(ValueError, match="新的 retrieval_config 格式"):
        RetrievalConfig.from_dict({"retrieval_mode": "混合檢索", "top_k": 8})


def test_retrieval_config_rejects_incomplete_new_shape() -> None:
    with pytest.raises(ValueError, match="reranker 必須包含"):
        RetrievalConfig.from_dict({
            "strategy_id": "hybrid", "top_k": 8, "params": {},
            "reranker": {}, "expansion": {},
        })


def test_retrieval_config_rejects_unsupported_strategy_parameters() -> None:
    config = RetrievalConfig(strategy_id="vector", params={"ranker": "naive"})

    with pytest.raises(ValueError, match="不支援的 params"):
        config.validated()


def test_strategy_labels_are_separate_from_persisted_ids() -> None:
    assert strategy_id_from_label("基本向量檢索") == "vector"
    assert strategy_label("hybrid") == "混合檢索"
    with pytest.raises(ValueError, match="策略 ID"):
        strategy_label("混合檢索")


def test_strategy_registry_rejects_duplicate_ids() -> None:
    class Strategy:
        strategy_id = "vector"

    registry = RetrievalStrategyRegistry()
    registry.register(Strategy())
    with pytest.raises(ValueError, match="已註冊"):
        registry.register(Strategy())
