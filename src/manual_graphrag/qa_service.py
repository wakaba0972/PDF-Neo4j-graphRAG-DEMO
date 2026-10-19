from __future__ import annotations

import json
import re
from typing import Any

from .graph_service import (
    GPT_6_LUNA_MODEL,
    GPT_6_LUNA_REASONING_EFFORTS,
    _api_url,
    _chat_response_content,
    _extract_json_text,
    _post_json,
)
from .retrieval import RETRIEVAL_STRATEGIES

RERANK_CANDIDATE_MULTIPLIER = 3
RERANK_MAX_CANDIDATES = 50
RERANKER_MODES = ("停用", "Reranker", "LLM Reranker")
EVIDENCE_EXPANSION_MODES = ("停用", "證據擴展")
EMBEDDING_BATCH_SIZE = 64
EMBEDDING_MAX_CONCURRENCY = 10


def _search_terms(text: str) -> set[str]:
    normalized = " ".join(text.casefold().split())
    words = set(re.findall(r"[a-z0-9_\-]+", normalized))
    chinese = "".join(re.findall(r"[\u3400-\u9fff]", normalized))
    chinese_bigrams = {
        chinese[index:index + 2]
        for index in range(max(len(chinese) - 1, 0))
    }
    return words | chinese_bigrams


def legacy_rerank_evidence(
    question: str,
    evidence: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    """Keep the original lexical reranker available for controlled comparisons."""
    limit = max(int(top_k), 1)
    question_terms = _search_terms(question)
    ranked = []
    for original_rank, item in enumerate(evidence):
        text = str(item.get("text", ""))
        evidence_terms = _search_terms(text)
        overlap = len(question_terms & evidence_terms) / max(len(question_terms), 1)
        original_score = float(item.get("fusion_score", item.get("score", 0.0)) or 0.0)
        exact_bonus = sum(
            1 for term in question_terms
            if len(term) >= 3 and term in text.casefold()
        )
        kind_bonus = 0.05 if item.get("kind") == "原文" else 0.0
        score = overlap * 2 + exact_bonus * 0.25 + original_score + kind_bonus
        ranked.append((score, original_rank, {
            **item,
            "rerank_score": score,
            "matched_by": list(dict.fromkeys([
                *(item.get("matched_by") or []), "legacy-reranker"
            ])),
        }))
    ranked.sort(key=lambda value: (-value[0], value[1]))
    return [item for _, _, item in ranked[:limit]]


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
                    "You are a retrieval-evidence reranker. Rank evidence by how directly it helps "
                    "answer the user's question. Prefer evidence containing verifiable answer details; "
                    "do not penalize evidence because it was found through graph expansion. Return "
                    "JSON only in this format: {\"ranking\":[candidate IDs from most to least relevant]}. "
                    "Do not rewrite, add to, or infer beyond the evidence."
                ),
            },
            {
                "role": "user",
                "content": "Question: {}\nCandidate evidence: {}".format(
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
    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[start : start + EMBEDDING_BATCH_SIZE]
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
    strategy_id: str,
    evidence: list[dict[str, Any]],
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    if not question.strip():
        raise ValueError("請輸入問題")
    if not answer_model.strip():
        raise ValueError("請選擇問答 LLM")
    if strategy_id not in RETRIEVAL_STRATEGIES.ids():
        raise ValueError("不支援的檢索策略 ID")
    if not evidence:
        raise ValueError("Neo4j 檢索找不到相關證據")
    is_gpt_6_luna = answer_model.strip().casefold() == GPT_6_LUNA_MODEL
    effective_effort = reasoning_effort or "low"
    if is_gpt_6_luna and effective_effort not in GPT_6_LUNA_REASONING_EFFORTS:
        raise ValueError("GPT-6 Luna 推理強度設定無效")
    system_instruction = (
        "You are a vehicle-document knowledge-graph QA assistant. Answer only from the evidence supplied "
        "for the current project, and clearly state when the evidence is insufficient. You may reason across "
        "multiple pieces of evidence and synthesize their information, but every conclusion must be supported "
        "by the supplied evidence. If the evidence is ambiguous, conflicting, or only partially supports an "
        "answer, clearly state the uncertainty and explain what the evidence does and does not establish. Do not "
        "present uncertain conclusions as facts. Do not fabricate facts or citation details. For every substantive factual "
        "claim, cite the source document name and the page number or page numbers provided in the evidence, "
        "placing the citation immediately after the claim in this format: (Source: <document name>, p. <page number>). "
        "If a claim is supported by multiple documents, cite each supporting document and its page numbers. "
        "Only cite documents and pages that the supplied evidence supports. Use only source names and page numbers "
        "explicitly provided in the evidence; never guess, infer, or invent citation details. If a document name "
        "is provided without a page number, cite the document name and state that the page number is unavailable. "
        "If no source location is available, state that it is unavailable. If evidence is associated with multiple "
        "pages, cite all page numbers exactly as provided; do not claim that a statement comes from one specific "
        "page unless the evidence identifies that page specifically. Unless the question explicitly requests a "
        "language, answer primarily in the language of the retrieved evidence. If the evidence uses multiple "
        "languages, use the language of the source that most directly supports the answer. Use plain text, not "
        "Markdown: do not output Markdown headings, bold or italic markers, bullet points, tables, or code blocks. "
        "If listing multiple points, use numbered items (for example, 1. and 2.) and leave a blank line before "
        "each numbered item."
    )
    payload = {
        "model": answer_model.strip(),
        "messages": [
            {
                "role": "system",
                "content": (
                    system_instruction
                ),
            },
            {
                "role": "user",
                "content": "Question: {}\n\nEvidence:\n{}".format(
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
    answer, finish_reason = _chat_response_content(response)
    answer = answer.strip()
    if not answer:
        reason = finish_reason or "unknown"
        raise ValueError(f"回答模型回傳空白內容（finish_reason={reason}）")
    return {"answer": answer, "evidence": evidence}
