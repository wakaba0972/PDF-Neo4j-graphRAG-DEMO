from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

import pytest

from manual_graphrag.retrieval import (
    RetrievalConfig,
    RetrievalParameterSpec,
    RetrievalStrategySpec,
    RetrievalStrategyRegistry,
    LABEL_STRATEGY_IDS,
    STRATEGY_LABELS,
    STRATEGY_SPECS,
    RETRIEVAL_STRATEGIES,
    register_retrieval_strategy,
    strategy_choices,
    strategy_id_from_label,
    strategy_label,
)
from manual_graphrag.retrieval_strategies.registry import load_strategy_specs


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


def test_retrieval_config_repairs_stale_candidate_budget_when_loading() -> None:
    stored = RetrievalConfig.from_ui("混合檢索", 8, "停用", "停用").to_dict()
    stored["top_k"] = 10

    loaded = RetrievalConfig.from_dict(stored)

    assert loaded.top_k == 10
    assert loaded.params["candidate_top_k"] == 10


def test_retrieval_config_rejects_unsupported_strategy_parameters() -> None:
    config = RetrievalConfig(strategy_id="vector", params={"ranker": "naive"})

    with pytest.raises(ValueError, match="不支援的 params"):
        config.validated()


def test_strategy_labels_are_separate_from_persisted_ids() -> None:
    assert strategy_id_from_label("基本向量檢索") == "vector"
    assert strategy_label("hybrid") == "混合檢索"
    with pytest.raises(ValueError, match="策略 ID"):
        strategy_label("混合檢索")


def test_builtin_strategy_registry_loads_implementations_from_yaml() -> None:
    vector = RETRIEVAL_STRATEGIES.get("vector")
    hybrid = RETRIEVAL_STRATEGIES.get("hybrid")

    assert vector.strategy_id == "vector"
    assert hybrid.strategy_id == "hybrid"
    assert vector.__class__.__module__ == "strategies.implementations.vector.strategy"
    assert hybrid.__class__.__module__ == "strategies.implementations.hybrid.strategy"


def test_strategy_specs_load_yaml_and_implementation_from_strategy_directory(tmp_path: Path, request) -> None:
    root = tmp_path / "strategies"
    (root / "specifications").mkdir(parents=True)
    implementation = root / "implementations" / "sample" / "strategy.py"
    implementation.parent.mkdir(parents=True)
    implementation.write_text(
        "class SampleStrategy:\n"
        "    strategy_id = 'sample'\n"
        "    def retrieve(self, context, config): return []\n",
        encoding="utf-8",
    )
    (root / "specifications" / "sample.yaml").write_text(
        "id: sample\n"
        "name: 測試策略\n"
        "implementation:\n"
        "  file: sample/strategy.py\n"
        "  class: SampleStrategy\n"
        "parameters:\n"
        "  beam_width:\n"
        "    type: integer\n"
        "    default: 4\n"
        "    minimum: 1\n"
        "    maximum: 12\n"
        "    label: 搜尋寬度\n"
        "    control: slider\n",
        encoding="utf-8",
    )
    request.addfinalizer(lambda: load_strategy_specs())

    assert load_strategy_specs(root) == ("sample",)
    spec = RETRIEVAL_STRATEGIES.get_spec("sample")
    assert spec.label == "測試策略"
    assert spec.parameters["beam_width"].default == 4
    assert RETRIEVAL_STRATEGIES.get("sample").strategy_id == "sample"


def test_strategy_yaml_cannot_load_implementation_outside_directory(tmp_path: Path) -> None:
    root = tmp_path / "strategies"
    (root / "specifications").mkdir(parents=True)
    (root / "implementations").mkdir()
    (root / "specifications" / "unsafe.yaml").write_text(
        "id: unsafe\nname: unsafe\nimplementation:\n  file: ../outside.py\n  class: Unsafe\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="必須位於 implementations"):
        load_strategy_specs(root)


def test_strategy_registry_rejects_duplicate_ids() -> None:
    class Strategy:
        strategy_id = "vector"

    registry = RetrievalStrategyRegistry()
    spec = RetrievalStrategySpec("vector", "向量", {})
    registry.register(spec, "test.module", "Strategy")
    with pytest.raises(ValueError, match="重複"):
        registry.register(spec, "test.module", "Strategy")


def test_strategy_registry_serializes_concurrent_file_loading(tmp_path: Path, request) -> None:
    strategy_id = "concurrent_probe"
    module_name = f"strategies.implementations.{strategy_id}.strategy"
    implementation = tmp_path / "strategy.py"
    implementation.write_text(
        "import time\n"
        "time.sleep(0.05)\n"
        "class ConcurrentProbeStrategy:\n"
        "    strategy_id = 'concurrent_probe'\n"
        "    def retrieve(self, context, config): return []\n",
        encoding="utf-8",
    )
    registry = RetrievalStrategyRegistry()
    registry.register_file(
        RetrievalStrategySpec(strategy_id, "並行載入測試"),
        implementation,
        "ConcurrentProbeStrategy",
    )
    request.addfinalizer(lambda: sys.modules.pop(module_name, None))

    with ThreadPoolExecutor(max_workers=8) as executor:
        strategies = list(executor.map(lambda _: registry.get(strategy_id), range(8)))

    assert all(strategy.strategy_id == strategy_id for strategy in strategies)
    assert all(strategy is strategies[0] for strategy in strategies)


def test_strategy_spec_drives_parameter_validation_and_ui_choices(request) -> None:
    spec = RetrievalStrategySpec(
        "test_beam", "測試 Beam 搜尋",
        {"beam_width": RetrievalParameterSpec(int, default=4, minimum=1, maximum=12)},
    )
    register_retrieval_strategy(spec, "test.module", "Strategy")
    request.addfinalizer(lambda: (
        STRATEGY_SPECS.pop(spec.strategy_id, None),
        STRATEGY_LABELS.pop(spec.strategy_id, None),
        LABEL_STRATEGY_IDS.pop(spec.label, None),
        RETRIEVAL_STRATEGIES._implementations.pop(spec.strategy_id, None),
    ))
    config = RetrievalConfig(strategy_id="test_beam", top_k=5, params={"beam_width": 8})
    assert config.validated() is config
    assert "測試 Beam 搜尋" in strategy_choices()
    assert strategy_id_from_label("測試 Beam 搜尋") == "test_beam"
    with pytest.raises(ValueError, match="不可大於 12"):
        RetrievalConfig(strategy_id="test_beam", params={"beam_width": 13}).validated()
