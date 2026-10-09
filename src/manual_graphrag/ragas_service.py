"""Optional RAGAS scoring for saved RAG answers and their retrieved contexts."""

from __future__ import annotations

import asyncio
from typing import Any


async def _score_with_ragas_async(
    base_url: str,
    api_key: str,
    model: str,
    question: str,
    reference: str,
    response: str,
    retrieved_contexts: list[str],
    reasoning_effort: str | None,
) -> dict[str, float]:
    if not retrieved_contexts:
        raise ValueError("沒有保存檢索證據，無法計算 RAGAS 指標；請重新生成回答後再評測。")

    try:
        from openai import AsyncOpenAI
        from ragas.llms import llm_factory
        from ragas.metrics.collections import ContextPrecision, ContextRecall, Faithfulness
    except ModuleNotFoundError as exc:
        if exc.name in {"openai", "ragas"}:
            raise ValueError(
                f"缺少必要套件 {exc.name}；請重新啟動程式安裝 requirements.txt 中的套件。"
            ) from exc
        raise ValueError(
            f"RAGAS 相依套件載入失敗（缺少 {exc.name or '未知模組'}）；"
            "請重新啟動程式更新 requirements.txt 中的套件。"
        ) from exc
    except ImportError as exc:
        raise ValueError(
            f"RAGAS 相依套件載入失敗：{exc}；請重新啟動程式更新 requirements.txt 中的套件。"
        ) from exc

    client = AsyncOpenAI(
        api_key=api_key or "", base_url=base_url.rstrip("/"),
        timeout=120.0, max_retries=5,
    )
    factory_options = (
        {"reasoning_effort": reasoning_effort or "medium"}
        if model.strip().casefold() == "gpt-6-luna" else {}
    )
    llm = llm_factory(model, client=client, **factory_options)
    shared = {
        "user_input": question,
        "reference": reference,
        "response": response,
        "retrieved_contexts": retrieved_contexts,
    }
    try:
        metric_specs = (
            ("faithfulness", Faithfulness(llm=llm), {key: shared[key] for key in ("user_input", "response", "retrieved_contexts")}),
            ("context_precision", ContextPrecision(llm=llm), {key: shared[key] for key in ("user_input", "reference", "retrieved_contexts")}),
            ("context_recall", ContextRecall(llm=llm), {key: shared[key] for key in ("user_input", "reference", "retrieved_contexts")}),
        )
        scores: dict[str, float] = {}
        for name, metric, kwargs in metric_specs:
            result = await metric.ascore(**kwargs)
            value = getattr(result, "value", result)
            if value is None:
                raise ValueError(f"RAGAS 指標 {name} 沒有回傳分數。")
            scores[name] = float(value)
        return scores
    finally:
        await client.close()


def score_ragas_metrics(
    base_url: str,
    api_key: str,
    model: str,
    question: str,
    reference: str,
    response: str,
    retrieved_contexts: list[str],
    reasoning_effort: str | None = None,
) -> dict[str, float]:
    """Score one answer; kept separate from the app's existing 0/1/2 judge."""
    return asyncio.run(_score_with_ragas_async(
        base_url, api_key, model, question, reference, response, retrieved_contexts,
        reasoning_effort,
    ))
