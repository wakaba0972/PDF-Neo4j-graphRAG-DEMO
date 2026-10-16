"""YAML-discovered registry for graph extraction methods."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import util
from pathlib import Path
import sys
from threading import RLock
from typing import Any, Protocol

import yaml


class ExtractionMethod(Protocol):
    method_id: str
    requires_entity_type: bool

    def extraction_instructions(self) -> str: ...

    def output_shape(self) -> str: ...

    def entity_key(self, name: str, entity_type: str) -> tuple[str, ...]: ...

    def make_entity(self, item: dict[str, Any], name: str) -> dict[str, Any]: ...


@dataclass(frozen=True)
class ExtractionMethodSpec:
    method_id: str
    label: str
    description: str
    order: int


class ExtractionMethodRegistry:
    def __init__(self) -> None:
        self.specs: dict[str, ExtractionMethodSpec] = {}
        self.labels: dict[str, str] = {}
        self.ids_by_label: dict[str, str] = {}
        self._implementations: dict[str, tuple[Path, str]] = {}
        self._loaded: dict[str, ExtractionMethod] = {}
        self._lock = RLock()

    def clear(self) -> None:
        self.specs.clear()
        self.labels.clear()
        self.ids_by_label.clear()
        self._implementations.clear()
        self._loaded.clear()

    def register(
        self, spec: ExtractionMethodSpec, implementation_file: Path, class_name: str,
    ) -> None:
        if spec.method_id in self.specs or spec.label in self.ids_by_label:
            raise ValueError(f"抽取方法 ID 或顯示名稱重複：{spec.method_id} / {spec.label}")
        self.specs[spec.method_id] = spec
        self.labels[spec.method_id] = spec.label
        self.ids_by_label[spec.label] = spec.method_id
        self._implementations[spec.method_id] = (implementation_file, class_name)

    def get_spec(self, method_id: str) -> ExtractionMethodSpec:
        try:
            return self.specs[str(method_id)]
        except KeyError as exc:
            raise ValueError(f"不支援的圖譜抽取方法：{method_id}") from exc

    def get(self, method_id: str) -> ExtractionMethod:
        with self._lock:
            if method_id in self._loaded:
                return self._loaded[method_id]
            try:
                implementation_file, class_name = self._implementations[method_id]
            except KeyError as exc:
                raise ValueError(f"沒有註冊圖譜抽取方法：{method_id}") from exc
            module_name = f"extraction_methods.implementations.{method_id}.method"
            module = sys.modules.get(module_name)
            if module is None:
                module_spec = util.spec_from_file_location(module_name, implementation_file)
                if module_spec is None or module_spec.loader is None:
                    raise ValueError(f"無法載入抽取方法實作：{implementation_file}")
                module = util.module_from_spec(module_spec)
                sys.modules[module_name] = module
                try:
                    module_spec.loader.exec_module(module)
                except Exception:
                    sys.modules.pop(module_name, None)
                    raise
            implementation = getattr(module, class_name)()
            if implementation.method_id != method_id:
                raise ValueError(f"抽取方法實作 ID 與 YAML 註冊 ID 不符：{method_id}")
            for member in ("extraction_instructions", "output_shape", "entity_key", "make_entity"):
                if not callable(getattr(implementation, member, None)):
                    raise ValueError(f"抽取方法 {method_id} 缺少 {member}()")
            self._loaded[method_id] = implementation
            return implementation

    def choices(self) -> list[str]:
        return [spec.label for spec in self.specs.values()]

    def id_from_label(self, label: str) -> str:
        try:
            return self.ids_by_label[label]
        except KeyError as exc:
            raise ValueError(f"不支援的圖譜抽取方法：{label}") from exc


EXTRACTION_METHODS = ExtractionMethodRegistry()


def load_extraction_method_specs(root: Path | None = None) -> tuple[str, ...]:
    directory = (root or Path(__file__).resolve().parents[2] / "extraction_methods").resolve()
    implementations = (directory / "implementations").resolve()
    specs_directory = directory / "specifications"
    files = sorted(specs_directory.glob("*.yaml"))
    if not files:
        raise ValueError(f"找不到圖譜抽取方法 YAML 規格：{specs_directory}")

    loaded: list[tuple[ExtractionMethodSpec, Path, str]] = []
    seen_ids: set[str] = set()
    seen_labels: set[str] = set()
    for spec_file in files:
        try:
            document = yaml.safe_load(spec_file.read_text(encoding="utf-8"))
            if not isinstance(document, dict):
                raise ValueError("規格根節點必須是 YAML mapping")
            method_id, label = document.get("id"), document.get("name")
            description = document.get("description", "")
            order = document.get("order", 1000)
            implementation = document.get("implementation")
            if not isinstance(method_id, str) or not method_id.strip():
                raise ValueError("必須設定非空 id")
            if not isinstance(label, str) or not label.strip():
                raise ValueError("必須設定非空 name")
            if not isinstance(description, str):
                raise ValueError("description 必須是字串")
            if isinstance(order, bool) or not isinstance(order, int):
                raise ValueError("order 必須是整數")
            if not isinstance(implementation, dict):
                raise ValueError("implementation 必須包含 file 與 class")
            implementation_file = (implementations / str(implementation.get("file", ""))).resolve()
            if not implementation_file.is_relative_to(implementations) or not implementation_file.is_file():
                raise ValueError("implementation.file 必須指向 implementations/ 內的存在檔案")
            class_name = implementation.get("class")
            if not isinstance(class_name, str) or not class_name.isidentifier():
                raise ValueError("implementation.class 必須是有效的 Python class 名稱")
            if method_id in seen_ids or label in seen_labels:
                raise ValueError(f"抽取方法 ID 或顯示名稱重複：{method_id} / {label}")
            seen_ids.add(method_id)
            seen_labels.add(label)
            loaded.append((
                ExtractionMethodSpec(method_id, label, description, order),
                implementation_file,
                class_name,
            ))
        except (OSError, yaml.YAMLError, TypeError, ValueError) as exc:
            raise ValueError(f"讀取抽取方法規格 {spec_file} 失敗：{exc}") from exc

    loaded.sort(key=lambda item: (item[0].order, item[0].method_id))
    EXTRACTION_METHODS.clear()
    for spec, implementation_file, class_name in loaded:
        EXTRACTION_METHODS.register(spec, implementation_file, class_name)
    project_root = str(directory.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    return tuple(item[0].method_id for item in loaded)


load_extraction_method_specs()
