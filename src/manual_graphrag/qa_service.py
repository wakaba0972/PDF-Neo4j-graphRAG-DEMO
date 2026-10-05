from __future__ import annotations

import json
from typing import Any

from .graph_service import (
    GPT_6_LUNA_MODEL, GPT_6_LUNA_REASONING_EFFORTS,
    _api_url, _chat_response_content, _extract_json_text, _post_json,
)

RERANK_CANDIDATE_MULTIPLIER = 3
RERANK_MAX_CANDIDATES = 50


def rerank_evidence(
    base_url: str,
    api_key: str,
    model: str,
    question: str,
    evidence: list[dict[str, Any]],
    top_k: int,
    reasoning_effort: str | None = None,
) -> list[dict[str, Any]]:
    """Use the configured answer model to rank candidates by answer relevance."""
    limit = max(int(top_k), 1)
    candidates = evidence[:RERANK_MAX_CANDIDATES]
    if len(candidates) <= limit:
        return candidates
    if not model.strip():
        raise ValueError("請選擇可用的回答模型以執行 LLM Reranker")

    candidate_payload = [
        {
            "id": str(index),
            "kind": item.get("kind", ""),
            "text": str(item.get("text", ""))[:3000],
            "source_pages": item.get("source_pages", []),
            "expanded_from_graph": "graph" in (item.get("matched_by") or []),
        }
        for index, item in enumerate(candidates)
    ]
    payload = {
        "model": model.strip(),
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是檢索證據重排器。依據證據對回答使用者問題的直接幫助程度排序；"
                    "優先選擇包含可驗證答案細節的證據，不要因為證據是圖譜擴展而降低排名。"
                    "只輸出 JSON 物件，格式為 {\"ranking\":[候選 id，從最相關到最不相關]}。"
                    "不得改寫、補充或臆測證據內容。"
                ),
            },
            {
                "role": "user",
                "content": "問題：{}\n候選證據：{}".format(
                    question.strip(), json.dumps(candidate_payload, ensure_ascii=False)
                ),
            },
        ],
    }
    is_gpt_6_luna = model.strip().casefold() == GPT_6_LUNA_MODEL
    effective_effort = reasoning_effort or "low"
    if is_gpt_6_luna:
        if effective_effort not in GPT_6_LUNA_REASONING_EFFORTS:
            raise ValueError("GPT-6 Luna 推理強度設定無效")
        payload["reasoning_effort"] = effective_effort
        if effective_effort == "none":
            payload["temperature"] = 0
    else:
        payload["temperature"] = 0

    response = _post_json(
        _api_url(base_url, "chat/completions"), payload, api_key
    )
    content, _ = _chat_response_content(response)
    try:
        parsed = _extract_json_text(content)
        ranking = parsed.get("ranking")
        expected_ids = {str(index) for index in range(len(candidates))}
        if (
            not isinstance(ranking, list)
            or len(ranking) != len(candidates)
            or {str(value) for value in ranking} != expected_ids
        ):
            return candidates[:limit]
        ordered = [candidates[int(value)] for value in ranking]
    except (ValueError, TypeError, AttributeError):
        return candidates[:limit]
    return [
        {
            **item,
            "matched_by": list(dict.fromkeys([
                *(item.get("matched_by") or []), "llm-reranker"
            ])),
        }
        for item in ordered[:limit]
    ]


def interleave_expanded_evidence(
    evidence: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    """Keep graph-expanded evidence in a bounded, interleaved context budget."""
    limit = max(int(top_k), 0)
    if not limit:
        return []

    primary = [
        item for item in evidence
        if "graph" not in (item.get("matched_by") or [])
    ]
    expanded = [
        item for item in evidence
        if "graph" in (item.get("matched_by") or [])
    ]
    if not expanded:
        return evidence[:limit]

    # Reserve up to half of the context slots for graph neighbors, while
    # retaining as many top-ranked direct hits as the remaining budget allows.
    expanded_slots = min(len(expanded), limit // 2)
    primary_slots = limit - expanded_slots
    primary = primary[:primary_slots]
    expanded = expanded[:expanded_slots]

    interleaved: list[dict[str, Any]] = []
    for index in range(max(len(primary), len(expanded))):
        if index < len(primary):
            interleaved.append(primary[index])
        if index < len(expanded):
            interleaved.append(expanded[index])
    return interleaved[:limit]


def embedding_vectors(base_url: str, api_key: str, model: str, texts: list[str]) -> list[list[float]]:
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
