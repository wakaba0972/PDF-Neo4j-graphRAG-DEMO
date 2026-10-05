from __future__ import annotations

import json
import re
from typing import Any, Callable

from .graph_service import (
    GPT_6_LUNA_MODEL, GPT_6_LUNA_REASONING_EFFORTS,
    _api_url, _chat_response_content, _post_json,
)

RERANK_CANDIDATE_MULTIPLIER = 3
RERANK_MAX_CANDIDATES = 50


def _search_terms(text: str) -> set[str]:
    normalized = " ".join(text.casefold().split())
    words = set(re.findall(r"[a-z0-9_\-]+", normalized))
    chinese = "".join(re.findall(r"[\u3400-\u9fff]", normalized))
    chinese_bigrams = {
        chinese[index:index + 2]
        for index in range(max(len(chinese) - 1, 0))
    }
    return words | chinese_bigrams


def rerank_evidence(
    question: str,
    evidence: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    """Locally rerank hybrid candidates without sending document text elsewhere."""
    limit = max(int(top_k), 1)
    question_terms = _search_terms(question)
    ranked = []
    for original_rank, item in enumerate(evidence):
        evidence_terms = _search_terms(str(item.get("text", "")))
        overlap = len(question_terms & evidence_terms) / max(len(question_terms), 1)
        original_score = float(item.get("fusion_score", item.get("score", 0.0)) or 0.0)
        exact_bonus = sum(
            1 for term in question_terms
            if len(term) >= 3 and term in str(item.get("text", "")).casefold()
        )
        kind_bonus = 0.05 if item.get("kind") == "原文" else 0.0
        rerank_score = overlap * 2 + exact_bonus * 0.25 + original_score + kind_bonus
        reranked = {
            **item,
            "rerank_score": rerank_score,
            "matched_by": list(dict.fromkeys([
                *(item.get("matched_by") or []), "local-reranker"
            ])),
        }
        ranked.append((rerank_score, original_rank, reranked))
    ranked.sort(key=lambda value: (-value[0], value[1]))
    return [item for _, _, item in ranked[:limit]]


def embedding_vectors(
    base_url: str, api_key: str, model: str, texts: list[str],
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[list[float]]:
    if not model.strip():
        raise ValueError("此建圖結果沒有 Embedding 模型")
    vectors: list[list[float]] = []
    for start in range(0, len(texts), 64):
        batch = texts[start : start + 64]
        response = _post_json(
            _api_url(base_url, "embeddings"),
            {"model": model.strip(), "input": batch},
            api_key,
        )
        data = response.get("data")
        if not isinstance(data, list) or len(data) != len(batch):
            raise ValueError("Embedding API 回傳格式或數量不正確")
        ordered = sorted(data, key=lambda item: int(item.get("index", 0)))
        try:
            for item in ordered:
                vectors.append([float(value) for value in item["embedding"]])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Embedding API 回傳的向量格式不正確") from exc
        if progress_callback:
            progress_callback(len(vectors), len(texts))
    return vectors


def check_embedding_connection(base_url: str, api_key: str, model: str) -> None:
    if not model.strip():
        raise ValueError("請先填寫 Embedding 模型名稱")
    embedding_vectors(base_url, api_key, model, ["ping"])


def answer_graph_question(
    base_url: str,
    api_key: str,
    answer_model: str,
    question: str,
    retrieval_mode: str,
    evidence: list[dict[str, Any]],
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    if not question.strip():
        raise ValueError("請輸入問題")
    if not answer_model.strip():
        raise ValueError("請選擇問答 LLM")
    if retrieval_mode not in {
        "基本向量檢索", "混合檢索", "基本檢索", "關聯擴展檢索",
        "GraphRAG", "向量 RAG",
    }:
        raise ValueError("不支援的檢索模式")
    if not evidence:
        raise ValueError("Neo4j 檢索找不到相關證據")
    is_gpt_6_luna = answer_model.strip().casefold() == GPT_6_LUNA_MODEL
    effective_effort = reasoning_effort or "low"
    if is_gpt_6_luna and effective_effort not in GPT_6_LUNA_REASONING_EFFORTS:
        raise ValueError("GPT-6 Luna 推理強度設定無效")
    payload = {
        "model": answer_model.strip(),
        "messages": [
            {
                "role": "system",
                "content": "你是車型文件知識圖譜問答助手。只能根據目前專案提供的證據回答；證據不足時必須明確說明。使用繁體中文，並在相關敘述後標示來源頁碼。若回答採用編號條列（例如 1.、2.），每個編號項目前都要空一行。",
            },
            {
                "role": "user",
                "content": "問題：{}\n\n證據：\n{}".format(
                    question.strip(), json.dumps(evidence, ensure_ascii=False)
                ),
            },
        ],
    }
    if is_gpt_6_luna:
        payload["reasoning_effort"] = effective_effort
        if effective_effort == "none":
            payload["temperature"] = 0
    else:
        payload["temperature"] = 0
    response = _post_json(
        _api_url(base_url, "chat/completions"),
        payload,
        api_key,
    )
    answer, _ = _chat_response_content(response)
    return {"answer": answer.strip(), "evidence": evidence}
