"""The only built-in strategy catalog: identity, parameters and implementation."""

from ..retrieval import (
    RETRIEVAL_STRATEGIES,
    RetrievalParameterSpec,
    RetrievalStrategySpec,
)


_COMMON_PARAMETERS = {
    "candidate_top_k": RetrievalParameterSpec(
        int, minimum=1, maximum=50, minimum_is_top_k=True,
        visible_in_ui=False,
    ),
    "effective_search_ratio": RetrievalParameterSpec(
        int, default=3, minimum=1, maximum=10,
        label="有效搜尋比例", control="slider",
    ),
}

STRATEGY_CATALOG = (
    (
        RetrievalStrategySpec("vector", "基本向量檢索", dict(_COMMON_PARAMETERS)),
        "manual_graphrag.retrieval_strategies.vector",
        "VectorRetrievalStrategy",
    ),
    (
        RetrievalStrategySpec(
            "hybrid", "混合檢索", {
                **_COMMON_PARAMETERS,
                "ranker": RetrievalParameterSpec(
                    str, default="naive", choices=("naive",),
                    visible_in_ui=False,
                ),
            },
        ),
        "manual_graphrag.retrieval_strategies.hybrid",
        "HybridRetrievalStrategy",
    ),
)


def register_builtin_strategies() -> None:
    if RETRIEVAL_STRATEGIES.ids():
        return
    for spec, module_path, class_name in STRATEGY_CATALOG:
        RETRIEVAL_STRATEGIES.register(spec, module_path, class_name)
