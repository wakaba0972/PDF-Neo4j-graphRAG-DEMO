from __future__ import annotations

import json
import re
import time
from typing import Any

from neo4j.api import READ_ACCESS
from neo4j_graphrag.retrievers import VectorCypherRetriever
from neo4j_graphrag.types import RetrieverResultItem

from manual_graphrag.graph_service import (
    GPT_6_LUNA_MODEL,
    GPT_6_LUNA_REASONING_EFFORTS,
    _api_url,
    _chat_response_content,
    _extract_json_text,
    _post_json,
)
from manual_graphrag.retrieval import RetrievalConfig, RetrievalContext


_ALLOWED_NODE_LABELS = {"GraphEvidence", "ExtractedEntity"}
_ALLOWED_RELATION_TYPES = {
    "MENTIONS_ENTITY", "REPRESENTS", "EVIDENCE_SOURCE", "EVIDENCE_TARGET",
    "EXTRACTED_RELATION",
}
_FORBIDDEN_TOKENS = {
    "CALL", "CREATE", "DELETE", "DETACH", "DROP", "LOAD", "CSV", "SET",
    "REMOVE", "MERGE", "FOREACH", "UNION", "USE", "SHOW", "TERMINATE",
    "ALTER", "GRANT", "DENY", "REVOKE", "APOC", "DBMS", "GDS", "SYSTEM",
}
_SAFE_NODE_PATTERN = re.compile(
    r"\(\s*(?P<variable>[A-Za-z_]\w*)\s*:\s*"
    r"(?P<label>[A-Za-z_]\w*)\s*\{(?P<properties>[^{}]*)\}\s*\)"
)
_SAFE_RELATION_PATTERN = re.compile(
    r"\[\s*(?:[A-Za-z_]\w*\s*)?:\s*(?P<type>[A-Za-z_]\w*)"
    r"(?P<detail>[^\]]*)\]"
)


def _format_record(record: Any) -> RetrieverResultItem:
    evidence = dict(record["evidence"])
    score = float(record.get("score", 0.0) or 0.0)
    evidence.update({
        "score": score,
        "fusion_score": score,
        "matched_by": ["agentic-search-2-seed"],
    })
    return RetrieverResultItem(content=evidence, metadata={"score": score})


def validate_readonly_cypher(
    cypher: str, params: dict[str, Any], result_limit: int,
) -> tuple[str, dict[str, Any]]:
    """Allow a narrow, parameterized Cypher subset over the project graph."""
    if not isinstance(cypher, str) or not cypher.strip():
        raise ValueError("Agentic Search 2 沒有提供 Cypher 查詢")
    query = cypher.strip()
    if len(query) > 4_000:
        raise ValueError("Agentic Search 2 的 Cypher 超過 4,000 字元")
    if any(token in query for token in (";", "'", '"', "`", "//", "/*", "*/", "--")):
        raise ValueError("Agentic Search 2 的 Cypher 不可包含多語句、字串或註解")
    words = {word.upper() for word in re.findall(r"[A-Za-z_]\w*", query)}
    forbidden = words & _FORBIDDEN_TOKENS
    if forbidden:
        raise ValueError(f"Agentic Search 2 拒絕非唯讀 Cypher：{', '.join(sorted(forbidden))}")
    if len(re.findall(r"\bRETURN\b", query, flags=re.IGNORECASE)) != 1:
        raise ValueError("Agentic Search 2 每次只能使用一個 RETURN")
    if not re.search(r"\bMATCH\b", query, flags=re.IGNORECASE):
        raise ValueError("Agentic Search 2 查詢必須從 MATCH 開始")
    evidence_projection = re.search(
        r"\bRETURN\s+evidence\s*\{[^{}]*\.evidence_id[^{}]*\}\s+AS\s+evidence\b",
        query, flags=re.IGNORECASE | re.DOTALL,
    )
    if not evidence_projection:
        raise ValueError("Cypher 必須以 evidence { .evidence_id, ... } AS evidence 回傳證據")
    returned_fields = set(re.findall(r"\.([A-Za-z_]\w*)", evidence_projection.group(0)))
    required_fields = {
        "evidence_id", "kind", "text", "source_documents", "source_pages",
        "source_chunk_numbers", "source_references_json",
    }
    if not required_fields.issubset(returned_fields):
        raise ValueError("Cypher 回傳證據時必須保留文件、頁碼與 chunk 來源欄位")

    node_matches = list(_SAFE_NODE_PATTERN.finditer(query))
    node_count = len(re.findall(r"\(\s*[A-Za-z_]\w*\s*:", query))
    if not node_matches or len(node_matches) != node_count:
        raise ValueError("Cypher 中每個節點都必須使用受允許的標籤")
    has_evidence_node = False
    for match in node_matches:
        variable = match.group("variable")
        label = match.group("label")
        properties = match.group("properties")
        if label not in _ALLOWED_NODE_LABELS:
            raise ValueError(f"Agentic Search 2 不允許查詢節點標籤：{label}")
        if variable == "evidence" and label == "GraphEvidence":
            has_evidence_node = True
    if not has_evidence_node:
        raise ValueError("回傳的 evidence 必須是 GraphEvidence 節點")
    # Reject unlabeled nodes and function/group expressions. The first version
    # intentionally permits only plain patterns, keeping query review simple.
    without_nodes = _SAFE_NODE_PATTERN.sub("", query)
    if re.search(r"\(\s*[A-Za-z_]\w*\s*\)", without_nodes):
        raise ValueError("Cypher 不允許未標籤節點或函式呼叫")

    relation_matches = list(_SAFE_RELATION_PATTERN.finditer(query))
    relation_count = len(re.findall(r"\[\s*(?:[A-Za-z_]\w*\s*)?:", query))
    if len(relation_matches) != relation_count:
        raise ValueError("Cypher 關係格式不受支援")
    for match in relation_matches:
        relation_type = match.group("type")
        if relation_type not in _ALLOWED_RELATION_TYPES:
            raise ValueError(f"Agentic Search 2 不允許查詢關係：{relation_type}")
    parameter_names = set(re.findall(r"\$([A-Za-z_]\w*)", query))
    supplied_params = params if isinstance(params, dict) else None
    if supplied_params is None or len(supplied_params) > 8:
        raise ValueError("Agentic Search 2 的 Cypher 參數格式錯誤")
    if parameter_names != set(supplied_params):
        raise ValueError("Cypher 參數與提供的參數清單不一致")
    safe_params: dict[str, Any] = {}
    for name, value in supplied_params.items():
        if isinstance(value, str):
            if len(value) > 200:
                raise ValueError(f"Cypher 參數 {name} 過長")
        elif isinstance(value, list):
            if len(value) > 50 or not all(
                isinstance(item, (str, int, float, bool)) for item in value
            ):
                raise ValueError(f"Cypher 參數 {name} 清單不安全或過長")
        elif not isinstance(value, (int, float, bool)):
            raise ValueError(f"Cypher 參數 {name} 型別不受支援")
        safe_params[name] = value

    limit_match = re.search(r"\bLIMIT\s+(\d+)\s*$", query, flags=re.IGNORECASE)
    if not limit_match:
        raise ValueError("Cypher 必須以固定整數 LIMIT 結尾")
    query = query[:limit_match.start(1)] + str(min(int(limit_match.group(1)), result_limit))
    return query, safe_params


class AgenticSearch2RetrievalStrategy:
    """Let an LLM plan bounded Cypher searches over the current graph."""

    strategy_id = "agentic_search_2"
    result_formatter = staticmethod(_format_record)

    @staticmethod
    def _plan(
        context: RetrievalContext,
        accumulated: list[dict[str, Any]],
        history: list[dict[str, Any]],
        result_limit: int,
        round_number: int,
        max_rounds: int,
    ) -> dict[str, Any]:
        if not context.model_base_url.strip() or not context.model_api_key.strip() or not context.model.strip():
            raise ValueError("Agentic Search 2 需要可用的 OpenAI 問答模型設定")
        evidence_brief = [{
            "evidence_id": item.get("evidence_id", ""),
            "kind": item.get("kind", ""),
            "name": item.get("name", ""),
            "text": str(item.get("text", ""))[:700],
            "source_documents": item.get("source_documents", []),
            "source_pages": item.get("source_pages", []),
        } for item in accumulated[-12:]]
        schema = {
            "GraphEvidence": [
                "evidence_id", "kind", "name", "text", "source_documents",
                "source_pages", "source_chunk_numbers", "source_references_json", "document_id",
            ],
            "ExtractedEntity": ["name", "type", "description"],
            "EXTRACTED_RELATION": ["type", "description"],
            "graph_edges": [
                "(GraphEvidence)-[:MENTIONS_ENTITY|REPRESENTS|EVIDENCE_SOURCE|EVIDENCE_TARGET]->(ExtractedEntity)",
                "(ExtractedEntity)-[:EXTRACTED_RELATION]-(ExtractedEntity)",
            ],
        }
        payload: dict[str, Any] = {
            "model": context.model.strip(),
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是知識圖譜檢索 Agent。每次根據問題、已找到證據和前幾輪結果，"
                        "選擇下一個最有價值的唯讀 Cypher 查詢；沒有新方向或證據已足夠時停止。"
                        "只能查 GraphEvidence、ExtractedEntity，以及 MENTIONS_ENTITY、REPRESENTS、"
                        "EVIDENCE_SOURCE、EVIDENCE_TARGET、EXTRACTED_RELATION。每一個節點都必須在"
                        "不可使用 CALL、子查詢、函式、字串常值、註解或寫入子句；必須把查詢值放入 params。"
                        f"查詢須回傳 `evidence {{ .evidence_id, .kind, .name, .text, .source_documents, "
                        f".source_pages, .source_chunk_numbers, .source_references_json }} AS evidence LIMIT {result_limit}`。"
                        "只輸出 JSON，不輸出思考過程。格式："
                        "停止：{\"action\":\"stop\",\"reason\":\"簡短原因\"}；"
                        "查詢：{\"action\":\"query\",\"cypher\":\"...\",\"params\":{}}。"
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps({
                        "question": context.question,
                        "schema": schema,
                        "round": round_number,
                        "max_rounds": max_rounds,
                        "prior_searches": history[-4:],
                        "evidence_found": evidence_brief,
                    }, ensure_ascii=False),
                },
            ],
        }
        model_name = context.model.strip().casefold()
        if model_name == GPT_6_LUNA_MODEL:
            effort = context.reasoning_effort or "low"
            if effort not in GPT_6_LUNA_REASONING_EFFORTS:
                raise ValueError("Agentic Search 2 的 GPT-6 Luna 推理強度設定無效")
            payload["reasoning_effort"] = effort
            if effort == "none":
                payload["temperature"] = 0
        else:
            payload["temperature"] = 0
        response = _post_json(
            _api_url(context.model_base_url, "chat/completions"),
            payload, context.model_api_key,
        )
        content, _ = _chat_response_content(response)
        result = _extract_json_text(content)
        if not isinstance(result, dict):
            raise ValueError("Agentic Search 2 的規劃結果不是 JSON 物件")
        return result

    @staticmethod
    def _execute(
        context: RetrievalContext, cypher: str, params: dict[str, Any],
        result_limit: int,
    ) -> list[dict[str, Any]]:
        query, query_params = validate_readonly_cypher(
            cypher, params, result_limit,
        )

        def operation() -> list[dict[str, Any]]:
            with context.driver.session(
                database=context.database, default_access_mode=READ_ACCESS,
            ) as session:
                with session.begin_transaction(timeout=5.0) as transaction:
                    return transaction.run(query, **query_params).data()[:result_limit]

        return context.retry(operation)

    def retrieve(
        self, context: RetrievalContext, config: RetrievalConfig,
    ) -> list[dict[str, Any]]:
        params = config.params
        candidate_limit = int(params.get("candidate_top_k", config.top_k))
        seed_limit = min(int(params.get("seed_top_k", 4)), candidate_limit)
        max_rounds = int(params.get("max_rounds", 3))
        result_limit = int(params.get("result_limit", 8))
        max_runtime = float(params.get("max_runtime_seconds", 45))

        retriever = VectorCypherRetriever(
            driver=context.driver,
            index_name=context.vector_index_name,
            retrieval_query=context.retrieval_query,
            result_formatter=context.result_formatter,
            neo4j_database=context.database,
        )
        result = context.retry(lambda: retriever.search(
            query_vector=context.embedding,
            top_k=seed_limit,
            effective_search_ratio=params.get("effective_search_ratio", 3),
            query_params={},
        ))
        accumulated = [
            dict(item.content) for item in result.items if isinstance(item.content, dict)
        ]
        by_id: dict[str, dict[str, Any]] = {}
        for evidence in accumulated:
            evidence_id = str(evidence.get("evidence_id") or "")
            if evidence_id:
                by_id[evidence_id] = evidence

        history: list[dict[str, Any]] = []
        deadline = time.monotonic() + max_runtime
        for round_number in range(1, max_rounds + 1):
            if time.monotonic() >= deadline:
                break
            plan = self._plan(
                context, accumulated, history, result_limit, round_number, max_rounds,
            )
            action = str(plan.get("action", "")).casefold()
            if action == "stop":
                break
            if action != "query":
                raise ValueError("Agentic Search 2 必須回傳 query 或 stop")
            rows = self._execute(
                context, plan.get("cypher"), plan.get("params", {}), result_limit,
            )
            found: list[dict[str, Any]] = []
            for row in rows:
                evidence = row.get("evidence") if isinstance(row, dict) else None
                if not isinstance(evidence, dict):
                    continue
                evidence_id = str(evidence.get("evidence_id") or "")
                if not evidence_id or evidence_id in by_id:
                    continue
                item = {
                    **evidence,
                    "score": 1.0 / (round_number + 1),
                    "fusion_score": 1.0 / (round_number + 1),
                    "matched_by": ["agentic-search-2-cypher"],
                    "agentic_search_round": round_number,
                }
                by_id[evidence_id] = item
                accumulated.append(item)
                found.append(item)
            history.append({
                "round": round_number,
                "reason": str(plan.get("reason", ""))[:240],
                "result_count": len(rows),
                "new_evidence_count": len(found),
                "evidence_ids": [item["evidence_id"] for item in found[:result_limit]],
            })
        return accumulated[:candidate_limit]
