import pytest

from manual_graphrag.retrieval import (
    RetrievalConfig,
    RetrievalParameterSpec,
    RetrievalStrategySpec,
    RetrievalStrategyRegistry,
    LABEL_STRATEGY_IDS,
    STRATEGY_LABELS,
    STRATEGY_SPECS,
    register_retrieval_strategy_spec,
    strategy_choices,
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


def test_strategy_spec_drives_parameter_validation_and_ui_choices(request) -> None:
    spec = RetrievalStrategySpec(
        "test_beam", "測試 Beam 搜尋",
        {"beam_width": RetrievalParameterSpec(int, default=4, minimum=1, maximum=12)},
    )
    register_retrieval_strategy_spec(spec)
    request.addfinalizer(lambda: (
        STRATEGY_SPECS.pop(spec.strategy_id, None),
        STRATEGY_LABELS.pop(spec.strategy_id, None),
        LABEL_STRATEGY_IDS.pop(spec.label, None),
    ))
    config = RetrievalConfig(strategy_id="test_beam", top_k=5, params={"beam_width": 8})
    assert config.validated() is config
    assert "測試 Beam 搜尋" in strategy_choices()
    assert strategy_id_from_label("測試 Beam 搜尋") == "test_beam"
    with pytest.raises(ValueError, match="不可大於 12"):
        RetrievalConfig(strategy_id="test_beam", params={"beam_width": 13}).validated()
