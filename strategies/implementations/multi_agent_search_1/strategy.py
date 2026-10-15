from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
import json
from typing import Any

from manual_graphrag.graph_service import _api_url, _chat_response_content, _post_json
from manual_graphrag.retrieval import RetrievalConfig, RetrievalContext
from strategies.implementations.agentic_search_3.strategy import (
    AgenticSearch3RetrievalStrategy,
)


_SCOPED_RETRIEVAL_QUERY = """
WITH node, score
WHERE node.document_id = $document_id
RETURN node {
    .evidence_id, .kind, .name, .source, .target, .text, .source_pages,
    .source_chunk_numbers, .source_documents, .source_references_json, .document_id
} AS evidence, score
"""


class MultiAgentSearch1Strategy:
    """Run one document-scoped Agentic Search 3 worker per graph document."""

    strategy_id = "multi_agent_search_1"
    result_formatter = AgenticSearch3RetrievalStrategy.result_formatter

    @staticmethod
    def _documents(context: RetrievalContext) -> list[dict[str, str]]:
        def operation() -> list[dict[str, str]]:
            with context.driver.session(
                database=context.database,
            ) as session:
                rows = session.run(
                    """
                    MATCH (document:GraphDocument)
                    WHERE coalesce(document.import_status, 'complete') = 'complete'
                      AND coalesce(document.document_id, '') <> ''
                    RETURN document.document_id AS document_id,
                           document.file_name AS file_name
                    ORDER BY file_name, document_id
                    """
                ).data()
            return [
                {
                    "document_id": str(row["document_id"]),
                    "file_name": str(row.get("file_name") or row["document_id"]),
                }
                for row in rows
                if row.get("document_id")
            ]

        return context.retry(operation)

    @staticmethod
    def _draft_answer(
        context: RetrievalContext,
        document: dict[str, str],
        evidence: list[dict[str, Any]],
    ) -> str:
        if not evidence:
            return "此文件中未找到可支持答案的證據。"
        if not context.model_base_url.strip() or not context.model_api_key.strip() or not context.model.strip():
            raise ValueError("Multi-Agent Search 1 需要可用的 OpenAI 問答模型設定")
        source_evidence = [
            {
                "text": str(item.get("text", ""))[:1800],
                "pages": item.get("source_pages", []),
                "source_documents": item.get("source_documents", []),
            }
            for item in evidence[:12]
        ]
        payload: dict[str, Any] = {
            "model": context.model.strip(),
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是負責單一文件的問答 Agent。只能根據提供的證據回答問題；"
                        "證據不足時明確說明，不可借用其他文件或常識補足。"
                        "使用證據的主要語言，以簡潔純文字回答，並標明可用頁碼。"
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps({
                        "document": document["file_name"],
                        "question": context.question,
                        "evidence": source_evidence,
                    }, ensure_ascii=False),
                },
            ],
        }
        if context.model.strip().casefold() == "gpt-6-luna":
            payload["reasoning_effort"] = context.reasoning_effort or "low"
            if payload["reasoning_effort"] == "none":
                payload["temperature"] = 0
        else:
            payload["temperature"] = 0
        response = _post_json(
            _api_url(context.model_base_url, "chat/completions"),
            payload,
            context.model_api_key,
        )
        answer, _ = _chat_response_content(response)
        return answer.strip() or "此文件 Agent 未產生可用答案。"

    def _run_document(
        self,
        context: RetrievalContext,
        config: RetrievalConfig,
        document: dict[str, str],
    ) -> list[dict[str, Any]]:
        scoped_context = replace(
            context,
            document_id=document["document_id"],
            retrieval_query=_SCOPED_RETRIEVAL_QUERY,
        )
        worker_config = RetrievalConfig(
            strategy_id="agentic_search_3",
            top_k=config.top_k,
            params={
                name: value for name, value in config.params.items()
                if name != "max_parallel_agents"
            },
            reranker={"id": "disabled", "params": {}},
            expansion={"id": "disabled", "params": {}},
        ).validated()
        evidence = AgenticSearch3RetrievalStrategy().retrieve(
            scoped_context, worker_config,
        )
        answer = self._draft_answer(scoped_context, document, evidence)
        return [
            {
                **item,
                "document_agent_answer": answer,
                "document_agent_name": document["file_name"],
                "multi_agent_documents": [],
                "matched_by": list(dict.fromkeys([
                    *(item.get("matched_by") or []), "multi-agent-document-worker",
                ])),
            }
            for item in evidence
        ]

    def retrieve(
        self, context: RetrievalContext, config: RetrievalConfig,
    ) -> list[dict[str, Any]]:
        documents = self._documents(context)
        if not documents:
            raise ValueError(
                "Multi-Agent Search 1 需要使用「每份文件各自分區」模式建圖；"
                "目前圖譜沒有可派工的 document_id，請重新抽取並匯入。"
            )

        max_workers = min(
            int(config.params.get("max_parallel_agents", 4)), len(documents),
        )
        evidence_by_id: dict[str, dict[str, Any]] = {}
        agent_answers: dict[str, str] = {}
        failures: list[str] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._run_document, context, config, document): document
                for document in documents
            }
            for future in as_completed(futures):
                document = futures[future]
                try:
                    for item in future.result():
                        agent_name = str(item.get("document_agent_name") or "")
                        agent_answer = str(item.get("document_agent_answer") or "").strip()
                        if agent_name and agent_answer:
                            agent_answers.setdefault(agent_name, agent_answer)
                        evidence_id = str(item.get("evidence_id") or "")
                        if evidence_id:
                            evidence_by_id[evidence_id] = item
                except Exception as exc:
                    failures.append(f"{document['file_name']}: {exc}")

        if not evidence_by_id and failures:
            raise ValueError("所有文件 Agent 均失敗：" + "；".join(failures[:3]))
        if failures and evidence_by_id:
            for item in evidence_by_id.values():
                item["multi_agent_warnings"] = failures[:5]
        document_names = [item["file_name"] for item in documents]
        for item in evidence_by_id.values():
            item["multi_agent_documents"] = document_names
            # Preserve every worker draft even if common Top K/reranking later
            # removes that worker's individual evidence rows.
            item["multi_agent_answers"] = agent_answers

        ordered = sorted(
            evidence_by_id.values(),
            key=lambda item: float(item.get("fusion_score", item.get("score", 0)) or 0),
            reverse=True,
        )
        limit = int(config.params.get("candidate_top_k", config.top_k))
        return ordered[:limit]
