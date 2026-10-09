from __future__ import annotations

import builtins
import sys
from types import ModuleType, SimpleNamespace

import pytest

from manual_graphrag.ragas_service import score_ragas_metrics


def test_ragas_metrics_require_retrieved_contexts_before_loading_optional_dependency() -> None:
    with pytest.raises(ValueError, match="沒有保存檢索證據"):
        score_ragas_metrics("https://api.openai.com/v1", "key", "gpt-6-luna", "Q", "A", "A", [])


def test_ragas_reports_missing_package_separately_from_broken_dependency(monkeypatch) -> None:
    original_import = builtins.__import__

    def missing_ragas(name, *args, **kwargs):
        if name == "ragas.llms":
            raise ModuleNotFoundError(
                "No module named 'langchain_community.chat_models.vertexai'",
                name="langchain_community.chat_models.vertexai",
            )
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_ragas)
    with pytest.raises(ValueError, match="RAGAS 相依套件載入失敗.*langchain_community.chat_models.vertexai"):
        score_ragas_metrics("https://api.openai.com/v1", "key", "gpt-6-luna", "Q", "A", "A", ["context"])

    def missing_package(name, *args, **kwargs):
        if name == "ragas.llms":
            raise ModuleNotFoundError("No module named 'ragas'", name="ragas")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_package)
    with pytest.raises(ValueError, match="缺少必要套件 ragas"):
        score_ragas_metrics("https://api.openai.com/v1", "key", "gpt-6-luna", "Q", "A", "A", ["context"])


def test_ragas_metrics_use_openai_compatible_client_and_three_core_metrics(monkeypatch) -> None:
    captured = {"client": {}, "calls": []}

    class FakeClient:
        def __init__(self, **kwargs):
            captured["client"].update(kwargs)

        async def close(self):
            captured["closed"] = True

    class FakeMetric:
        def __init__(self, llm):
            self.llm = llm
            captured["llm"] = llm

        async def ascore(self, **kwargs):
            captured["calls"].append(kwargs)
            return SimpleNamespace(value=len(captured["calls"]) / 10)

    openai_module = ModuleType("openai")
    openai_module.AsyncOpenAI = FakeClient
    ragas_module = ModuleType("ragas")
    llms_module = ModuleType("ragas.llms")
    llms_module.llm_factory = lambda model, client, **kwargs: (model, client, kwargs)
    metrics_module = ModuleType("ragas.metrics.collections")
    metrics_module.ContextPrecision = FakeMetric
    metrics_module.ContextRecall = FakeMetric
    metrics_module.Faithfulness = FakeMetric
    monkeypatch.setitem(sys.modules, "openai", openai_module)
    monkeypatch.setitem(sys.modules, "ragas", ragas_module)
    monkeypatch.setitem(sys.modules, "ragas.llms", llms_module)
    monkeypatch.setitem(sys.modules, "ragas.metrics", ModuleType("ragas.metrics"))
    monkeypatch.setitem(sys.modules, "ragas.metrics.collections", metrics_module)

    scores = score_ragas_metrics(
        "https://api.openai.com/v1/", "secret", "gpt-6-luna", "問題", "標準答案",
        "回答", ["檢索證據"],
    )

    assert captured["client"]["base_url"] == "https://api.openai.com/v1"
    assert captured["client"]["api_key"] == "secret"
    assert captured["client"]["max_retries"] == 5
    assert captured["llm"][0] == "gpt-6-luna"
    assert captured["llm"][2] == {"reasoning_effort": "medium"}
    assert captured["closed"] is True
    assert len(captured["calls"]) == 3
    assert captured["calls"][0] == {"user_input": "問題", "response": "回答", "retrieved_contexts": ["檢索證據"]}
    assert scores == {"faithfulness": 0.1, "context_precision": 0.2, "context_recall": 0.3}
