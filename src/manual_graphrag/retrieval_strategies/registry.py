"""Discover retrieval strategy YAML specifications and implementation modules."""

from pathlib import Path
import sys
from typing import Any

import yaml

from ..retrieval import (
    RETRIEVAL_STRATEGIES,
    RetrievalParameterSpec,
    RetrievalStrategySpec,
)


_VALUE_TYPES = {"string": str, "integer": int, "number": float, "boolean": bool}


def _parameter_from_document(name: str, document: dict[str, Any]) -> RetrievalParameterSpec:
    if not isinstance(document, dict):
        raise ValueError(f"參數 {name} 規格必須是 YAML mapping")
    type_name = document.get("type")
    if type_name not in _VALUE_TYPES:
        raise ValueError(f"參數 {name} 的 type 必須是 string/integer/number/boolean")
    choices = document.get("choices", [])
    if not isinstance(choices, list):
        raise ValueError(f"參數 {name} 的 choices 必須是清單")
    return RetrievalParameterSpec(
        value_type=_VALUE_TYPES[type_name],
        default=document.get("default"),
        minimum=document.get("minimum"),
        maximum=document.get("maximum"),
        choices=tuple(choices),
        minimum_is_top_k=bool(document.get("minimum_is_top_k", False)),
        label=document.get("label"),
        control=document.get("control"),
        visible_in_ui=bool(document.get("visible", True)),
    )


def load_strategy_specs(strategies_dir: Path | None = None) -> tuple[str, ...]:
    """Reload all strategies from ``strategies/specifications``.

    Each YAML document declares metadata and an implementation file relative to
    ``strategies/implementations``. Invalid specifications fail startup with a
    path-specific error instead of silently hiding a strategy.
    """
    root = (strategies_dir or Path(__file__).resolve().parents[3] / "strategies").resolve()
    project_root = str(root.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    specs_dir = root / "specifications"
    implementations_dir = root / "implementations"
    spec_files = sorted(specs_dir.glob("*.yaml"))
    if not spec_files:
        raise ValueError(f"找不到檢索策略 YAML 規格：{specs_dir}")

    loaded: list[tuple[int, RetrievalStrategySpec, Path, str]] = []
    seen_ids: set[str] = set()
    seen_names: set[str] = set()
    for spec_file in spec_files:
        try:
            document = yaml.safe_load(spec_file.read_text(encoding="utf-8"))
            if not isinstance(document, dict):
                raise ValueError("規格根節點必須是 YAML mapping")
            strategy_id = document.get("id")
            name = document.get("name")
            order = document.get("order", 1000)
            implementation = document.get("implementation")
            parameters = document.get("parameters", {})
            if not isinstance(strategy_id, str) or not strategy_id.strip():
                raise ValueError("必須設定非空 id")
            if not isinstance(name, str) or not name.strip():
                raise ValueError("必須設定非空 name")
            if isinstance(order, bool) or not isinstance(order, int):
                raise ValueError("order 必須是整數")
            if not isinstance(implementation, dict):
                raise ValueError("implementation 必須包含 file 與 class")
            implementation_file = (implementations_dir / str(implementation.get("file", ""))).resolve()
            if not implementation_file.is_relative_to(implementations_dir.resolve()):
                raise ValueError("implementation.file 必須位於 implementations/ 內")
            if not implementation_file.is_file():
                raise ValueError(f"找不到實作檔案：{implementation_file}")
            class_name = implementation.get("class")
            if not isinstance(class_name, str) or not class_name.isidentifier():
                raise ValueError("implementation.class 必須是有效的 Python class 名稱")
            if not isinstance(parameters, dict):
                raise ValueError("parameters 必須是 YAML mapping")
            if strategy_id in seen_ids or name in seen_names:
                raise ValueError(f"策略 id 或 name 重複：{strategy_id} / {name}")
            seen_ids.add(strategy_id)
            seen_names.add(name)
            strategy_spec = RetrievalStrategySpec(
                strategy_id=strategy_id,
                label=name,
                parameters={key: _parameter_from_document(key, value)
                            for key, value in parameters.items()},
            )
            loaded.append((order, strategy_spec, implementation_file, class_name))
        except (OSError, yaml.YAMLError, TypeError, ValueError) as exc:
            raise ValueError(f"讀取策略規格 {spec_file} 失敗：{exc}") from exc

    loaded.sort(key=lambda item: (item[0], item[1].strategy_id))
    RETRIEVAL_STRATEGIES.clear()
    for _, strategy_spec, implementation_file, class_name in loaded:
        RETRIEVAL_STRATEGIES.register_file(strategy_spec, implementation_file, class_name)
    return tuple(spec.strategy_id for _, spec, _, _ in loaded)


def register_builtin_strategies() -> None:
    """Compatibility alias; strategy definitions now live in YAML files."""
    load_strategy_specs()
