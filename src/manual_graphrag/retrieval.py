from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


STRATEGY_LABELS = {
    "vector": "基本向量檢索",
    "hybrid": "混合檢索",
}
LABEL_STRATEGY_IDS = {label: strategy_id for strategy_id, label in STRATEGY_LABELS.items()}
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
        strategy_label(self.strategy_id)
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int):
            raise ValueError("Top K 必須是整數")
        if not 1 <= self.top_k <= 50:
            raise ValueError("Top K 必須介於 1 到 50")
        if not isinstance(self.params, dict):
            raise ValueError("檢索策略 params 必須是物件")
        if self.strategy_id == "vector":
            allowed_params = {"candidate_top_k", "effective_search_ratio"}
        else:
            allowed_params = {"candidate_top_k", "effective_search_ratio", "ranker"}
        unknown = set(self.params) - allowed_params
        if unknown:
            raise ValueError(f"{self.strategy_id} 不支援的 params：{', '.join(sorted(unknown))}")
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
        candidate_top_k = self.params.get("candidate_top_k", self.top_k)
        if (
            isinstance(candidate_top_k, bool)
            or not isinstance(candidate_top_k, int)
            or not self.top_k <= candidate_top_k <= 50
        ):
            raise ValueError("candidate_top_k 必須介於 Top K 與 50 之間")
        search_ratio = self.params.get("effective_search_ratio", 3)
        if (
            isinstance(search_ratio, bool)
            or not isinstance(search_ratio, int)
            or not 1 <= search_ratio <= 10
        ):
            raise ValueError("effective_search_ratio 必須介於 1 到 10")
        if self.strategy_id == "hybrid" and self.params.get("ranker", "naive") != "naive":
            raise ValueError("hybrid.ranker 目前僅支援 naive")
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
        params: dict[str, Any] = {
            "candidate_top_k": min(selected_top_k * 3, 50)
            if rerank_candidates else selected_top_k,
            "effective_search_ratio": 3,
        }
        if strategy_id == "hybrid":
            params["ranker"] = "naive"
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
