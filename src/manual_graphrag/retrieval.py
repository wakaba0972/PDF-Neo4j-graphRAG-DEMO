from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class RetrievalParameterSpec:
    """Validation metadata for a strategy-specific configuration parameter."""

    value_type: type
    default: Any = None
    minimum: int | float | None = None
    maximum: int | float | None = None
    choices: tuple[Any, ...] = ()
    minimum_is_top_k: bool = False

    def validate(self, name: str, value: Any, top_k: int) -> None:
        if self.value_type is int:
            valid_type = isinstance(value, int) and not isinstance(value, bool)
        else:
            valid_type = isinstance(value, self.value_type)
        if not valid_type:
            raise ValueError(f"檢索參數 {name} 型別錯誤，應為 {self.value_type.__name__}")
        minimum = top_k if self.minimum_is_top_k else self.minimum
        if minimum is not None and value < minimum:
            raise ValueError(f"檢索參數 {name} 不可小於 {minimum}")
        if self.maximum is not None and value > self.maximum:
            raise ValueError(f"檢索參數 {name} 不可大於 {self.maximum}")
        if self.choices and value not in self.choices:
            raise ValueError(f"檢索參數 {name} 必須是以下選項之一：{', '.join(map(str, self.choices))}")


@dataclass(frozen=True)
class RetrievalStrategySpec:
    strategy_id: str
    label: str
    parameters: dict[str, RetrievalParameterSpec] = field(default_factory=dict)

    def validate_params(self, params: dict[str, Any], top_k: int) -> None:
        unknown = set(params) - set(self.parameters)
        if unknown:
            raise ValueError(
                f"{self.strategy_id} 不支援的 params：{', '.join(sorted(unknown))}"
            )
        for name, value in params.items():
            self.parameters[name].validate(name, value, top_k)


STRATEGY_SPECS: dict[str, RetrievalStrategySpec] = {}
STRATEGY_LABELS: dict[str, str] = {}
LABEL_STRATEGY_IDS: dict[str, str] = {}


def register_retrieval_strategy_spec(spec: RetrievalStrategySpec) -> None:
    if not spec.strategy_id or not spec.label:
        raise ValueError("檢索策略必須設定穩定 ID 與顯示名稱")
    if spec.strategy_id in STRATEGY_SPECS or spec.label in LABEL_STRATEGY_IDS:
        raise ValueError(f"檢索策略 ID 或顯示名稱重複：{spec.strategy_id}")
    STRATEGY_SPECS[spec.strategy_id] = spec
    STRATEGY_LABELS[spec.strategy_id] = spec.label
    LABEL_STRATEGY_IDS[spec.label] = spec.strategy_id


_COMMON_PARAMETERS = {
    "candidate_top_k": RetrievalParameterSpec(
        int, minimum=1, maximum=50, minimum_is_top_k=True,
    ),
    "effective_search_ratio": RetrievalParameterSpec(int, default=3, minimum=1, maximum=10),
}
register_retrieval_strategy_spec(RetrievalStrategySpec(
    "vector", "基本向量檢索", dict(_COMMON_PARAMETERS),
))
register_retrieval_strategy_spec(RetrievalStrategySpec(
    "hybrid", "混合檢索", {
        **_COMMON_PARAMETERS,
        "ranker": RetrievalParameterSpec(str, default="naive", choices=("naive",)),
    },
))
RERANKER_IDS = {"disabled", "lexical", "llm"}
EXPANSION_IDS = {"disabled", "legacy_name_chunk", "graph_v2"}
RERANKER_LABEL_IDS = {
    "停用": "disabled", "Reranker": "lexical", "LLM Reranker": "llm",
}
EXPANSION_LABEL_IDS = {
    "停用": "disabled", "證據擴展": "legacy_name_chunk",
    "證據擴展 V2": "graph_v2",
}
RERANKER_ID_LABELS = {value: key for key, value in RERANKER_LABEL_IDS.items()}
EXPANSION_ID_LABELS = {value: key for key, value in EXPANSION_LABEL_IDS.items()}


def strategy_id_from_label(label: str) -> str:
    try:
        return LABEL_STRATEGY_IDS[label]
    except KeyError as exc:
        raise ValueError(f"不支援的檢索策略顯示名稱：{label}") from exc


def strategy_label(strategy_id: str) -> str:
    try:
        return STRATEGY_LABELS[strategy_id]
    except KeyError as exc:
        raise ValueError(f"不支援的檢索策略 ID：{strategy_id}") from exc


def strategy_choices() -> list[str]:
    """UI labels in registration order; views should not maintain duplicates."""
    return [spec.label for spec in STRATEGY_SPECS.values()]


@dataclass(frozen=True)
class RetrievalConfig:
    """Versioned-by-shape retrieval settings shared by UI, persistence and runtime."""

    strategy_id: str = "hybrid"
    top_k: int = 8
    params: dict[str, Any] = field(default_factory=dict)
    reranker: dict[str, Any] = field(
        default_factory=lambda: {"id": "disabled", "params": {}}
    )
    expansion: dict[str, Any] = field(
        default_factory=lambda: {"id": "disabled", "params": {}}
    )

    def validated(self) -> RetrievalConfig:
        strategy = STRATEGY_SPECS.get(self.strategy_id)
        if strategy is None:
            raise ValueError(f"不支援的檢索策略 ID：{self.strategy_id}")
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int):
            raise ValueError("Top K 必須是整數")
        if not 1 <= self.top_k <= 50:
            raise ValueError("Top K 必須介於 1 到 50")
        if not isinstance(self.params, dict):
            raise ValueError("檢索策略 params 必須是物件")
        strategy.validate_params(self.params, self.top_k)
        for component_name, component, allowed_ids in (
            ("reranker", self.reranker, RERANKER_IDS),
            ("expansion", self.expansion, EXPANSION_IDS),
        ):
            if not isinstance(component, dict) or set(component) != {"id", "params"}:
                raise ValueError(f"{component_name} 必須包含 id 與 params")
            if component.get("id") not in allowed_ids:
                raise ValueError(f"不支援的 {component_name} ID：{component.get('id')}")
            if not isinstance(component.get("params", {}), dict):
                raise ValueError(f"{component_name}.params 必須是物件")
            if component_name == "reranker" and component.get("params"):
                raise ValueError(f"{component['id']} 不接受 Reranker 參數")
            if component_name == "expansion":
                params = component.get("params", {})
                if component["id"] == "graph_v2":
                    if set(params) - {"hops"}:
                        raise ValueError("graph_v2 僅支援 hops 參數")
                    hops = params.get("hops", 4)
                    if isinstance(hops, bool) or not isinstance(hops, int) or not 1 <= hops <= 6:
                        raise ValueError("graph_v2.hops 必須介於 1 到 6")
                elif params:
                    raise ValueError(f"{component['id']} 不接受擴展參數")
        return self

    def to_dict(self) -> dict[str, Any]:
        self.validated()
        return {
            "strategy_id": self.strategy_id,
            "top_k": self.top_k,
            "params": dict(self.params),
            "reranker": {"id": self.reranker["id"], "params": dict(self.reranker.get("params", {}))},
            "expansion": {"id": self.expansion["id"], "params": dict(self.expansion.get("params", {}))},
        }

    @classmethod
    def from_ui(
        cls,
        strategy_label: str,
        top_k: int,
        reranker_label: str,
        expansion_label: str,
        *,
        rerank_candidates: bool = False,
    ) -> RetrievalConfig:
        try:
            strategy_id = strategy_id_from_label(strategy_label)
            reranker_id = RERANKER_LABEL_IDS[reranker_label]
            expansion_id = EXPANSION_LABEL_IDS[expansion_label]
        except KeyError as exc:
            raise ValueError(f"檢索設定選項無效：{exc.args[0]}") from exc
        selected_top_k = int(top_k)
        strategy = STRATEGY_SPECS[strategy_id]
        params: dict[str, Any] = {}
        if "candidate_top_k" in strategy.parameters:
            params["candidate_top_k"] = (
                min(selected_top_k * 3, 50) if rerank_candidates else selected_top_k
            )
        if "effective_search_ratio" in strategy.parameters:
            ratio_default = strategy.parameters["effective_search_ratio"].default
            params["effective_search_ratio"] = ratio_default if ratio_default is not None else 3
        params.update({
            name: parameter.default
            for name, parameter in strategy.parameters.items()
            if name not in params and parameter.default is not None
        })
        expansion_params = {"hops": 4} if expansion_id == "graph_v2" else {}
        return cls(
            strategy_id=strategy_id,
            top_k=selected_top_k,
            params=params,
            reranker={"id": reranker_id, "params": {}},
            expansion={"id": expansion_id, "params": expansion_params},
        ).validated()

    def ui_values(self) -> tuple[str, int, str, str]:
        self.validated()
        return (
            strategy_label(self.strategy_id), self.top_k,
            RERANKER_ID_LABELS[self.reranker["id"]],
            EXPANSION_ID_LABELS[self.expansion["id"]],
        )

    @classmethod
    def from_dict(cls, value: Any) -> RetrievalConfig:
        if not isinstance(value, dict) or set(value) != {
            "strategy_id", "top_k", "params", "reranker", "expansion",
        }:
            raise ValueError("專案未使用新的 retrieval_config 格式；請在新專案中重新設定檢索參數")
        config = cls(
            strategy_id=value["strategy_id"],
            top_k=value["top_k"],
            params=value["params"],
            reranker=value["reranker"],
            expansion=value["expansion"],
        )
        return config.validated()


@dataclass(frozen=True)
class RetrievalContext:
    driver: Any
    database: str
    run_id: str
    question: str
    embedding: list[float]
    vector_index_name: str
    fulltext_index_name: str
    retrieval_query: str
    result_formatter: Any
    retry: Any


class RetrievalStrategy(Protocol):
    strategy_id: str
    result_formatter: Any

    def retrieve(
        self, context: RetrievalContext, config: RetrievalConfig,
    ) -> list[dict[str, Any]]: ...


class RetrievalStrategyRegistry:
    def __init__(self) -> None:
        self._strategies: dict[str, RetrievalStrategy] = {}

    def register(self, strategy: RetrievalStrategy) -> None:
        strategy_label(strategy.strategy_id)
        if strategy.strategy_id in self._strategies:
            raise ValueError(f"檢索策略已註冊：{strategy.strategy_id}")
        self._strategies[strategy.strategy_id] = strategy

    def get(self, strategy_id: str) -> RetrievalStrategy:
        try:
            return self._strategies[strategy_id]
        except KeyError as exc:
            raise ValueError(f"沒有註冊檢索策略：{strategy_id}") from exc

    def ids(self) -> tuple[str, ...]:
        return tuple(self._strategies)
