from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module, util
from pathlib import Path
import sys
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
    label: str | None = None
    control: str | None = None
    visible_in_ui: bool = True

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


class RetrievalStrategyRegistry:
    """Single registry for strategy metadata and its lazily-loaded implementation."""

    def __init__(self) -> None:
        self.specs: dict[str, RetrievalStrategySpec] = {}
        self.labels: dict[str, str] = {}
        self.ids_by_label: dict[str, str] = {}
        self._implementations: dict[str, tuple[str, str]] = {}
        self._implementation_files: dict[str, Path] = {}
        self._loaded: dict[str, Any] = {}

    def register(
        self, spec: RetrievalStrategySpec, module_path: str, class_name: str,
    ) -> None:
        if not spec.strategy_id or not spec.label:
            raise ValueError("檢索策略必須設定穩定 ID 與顯示名稱")
        if spec.strategy_id in self.specs or spec.label in self.ids_by_label:
            raise ValueError(f"檢索策略 ID 或顯示名稱重複：{spec.strategy_id}")
        if not module_path or not class_name:
            raise ValueError("檢索策略必須指定 implementation module 與 class")
        self.specs[spec.strategy_id] = spec
        self.labels[spec.strategy_id] = spec.label
        self.ids_by_label[spec.label] = spec.strategy_id
        self._implementations[spec.strategy_id] = (module_path, class_name)

    def register_file(
        self, spec: RetrievalStrategySpec, implementation_file: Path, class_name: str,
    ) -> None:
        self.register(
            spec,
            str(implementation_file),
            class_name,
        )
        self._implementation_files[spec.strategy_id] = implementation_file

    def clear(self) -> None:
        """Clear registrations while preserving exported mapping references."""
        self.specs.clear()
        self.labels.clear()
        self.ids_by_label.clear()
        self._implementations.clear()
        self._implementation_files.clear()
        self._loaded.clear()

    def get_spec(self, strategy_id: str) -> RetrievalStrategySpec:
        try:
            return self.specs[strategy_id]
        except KeyError as exc:
            raise ValueError(f"不支援的檢索策略 ID：{strategy_id}") from exc

    def get(self, strategy_id: str) -> Any:
        if strategy_id in self._loaded:
            return self._loaded[strategy_id]
        try:
            module_path, class_name = self._implementations[strategy_id]
        except KeyError as exc:
            raise ValueError(f"沒有註冊檢索策略：{strategy_id}") from exc
        if strategy_id in self._implementation_files:
            implementation_file = self._implementation_files[strategy_id]
            module_name = f"strategies.implementations.{strategy_id}.strategy"
            module = sys.modules.get(module_name)
            if module is None:
                module_spec = util.spec_from_file_location(module_name, implementation_file)
                if module_spec is None or module_spec.loader is None:
                    raise ValueError(f"無法載入檢索策略實作：{implementation_file}")
                module = util.module_from_spec(module_spec)
                sys.modules[module_name] = module
                module_spec.loader.exec_module(module)
        else:
            module = import_module(module_path)
        implementation_class = getattr(module, class_name)
        strategy = implementation_class()
        if getattr(strategy, "strategy_id", None) != strategy_id:
            raise ValueError(f"檢索策略實作 ID 與註冊 ID 不符：{strategy_id}")
        self._loaded[strategy_id] = strategy
        return strategy

    def ids(self) -> tuple[str, ...]:
        return tuple(self.specs)


RETRIEVAL_STRATEGIES = RetrievalStrategyRegistry()
STRATEGY_SPECS = RETRIEVAL_STRATEGIES.specs
STRATEGY_LABELS = RETRIEVAL_STRATEGIES.labels
LABEL_STRATEGY_IDS = RETRIEVAL_STRATEGIES.ids_by_label


def register_retrieval_strategy(
    spec: RetrievalStrategySpec, module_path: str, class_name: str,
) -> None:
    RETRIEVAL_STRATEGIES.register(spec, module_path, class_name)


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
        return RETRIEVAL_STRATEGIES.ids_by_label[label]
    except KeyError as exc:
        raise ValueError(f"不支援的檢索策略顯示名稱：{label}") from exc


def strategy_label(strategy_id: str) -> str:
    try:
        return RETRIEVAL_STRATEGIES.labels[strategy_id]
    except KeyError as exc:
        raise ValueError(f"不支援的檢索策略 ID：{strategy_id}") from exc


def strategy_choices() -> list[str]:
    """UI labels in registration order; views should not maintain duplicates."""
    return [spec.label for spec in RETRIEVAL_STRATEGIES.specs.values()]


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
        strategy = RETRIEVAL_STRATEGIES.get_spec(self.strategy_id)
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
        strategy = RETRIEVAL_STRATEGIES.get_spec(strategy_id)
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


from .retrieval_strategies.registry import load_strategy_specs

load_strategy_specs()
