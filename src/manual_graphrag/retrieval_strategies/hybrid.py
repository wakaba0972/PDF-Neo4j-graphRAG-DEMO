from __future__ import annotations

import re
from typing import Any

from neo4j_graphrag.retrievers import HybridCypherRetriever
from neo4j_graphrag.types import RetrieverResultItem

from ..retrieval import RetrievalConfig, RetrievalContext


def _escape_fulltext_query(question: str) -> str:
    """Escape Lucene query syntax while preserving terms for the CJK analyzer."""
    return re.sub(
        r"""([+\-!(){}\[\]^"~*?:\\/]|&&|\|\|)""",
        r"""\\\1""",
        question.strip(),
    )


def _format_hybrid_record(record: Any) -> RetrieverResultItem:
    evidence = dict(record["evidence"])
    score = float(record["score"])
    evidence.update({
        "score": score,
        "fusion_score": score,
        "matched_by": ["official-hybrid"],
    })
    return RetrieverResultItem(content=evidence, metadata={"score": score})


class HybridRetrievalStrategy:
    strategy_id = "hybrid"
    result_formatter = staticmethod(_format_hybrid_record)

    def retrieve(
        self, context: RetrievalContext, config: RetrievalConfig,
    ) -> list[dict[str, Any]]:
        retriever = HybridCypherRetriever(
            driver=context.driver,
            vector_index_name=context.vector_index_name,
            fulltext_index_name=context.fulltext_index_name,
            retrieval_query=context.retrieval_query,
            result_formatter=context.result_formatter,
            neo4j_database=context.database,
        )
        result = context.retry(lambda: retriever.search(
            query_text=_escape_fulltext_query(context.question),
            query_vector=context.embedding,
            top_k=config.params.get("candidate_top_k", config.top_k),
            effective_search_ratio=config.params.get("effective_search_ratio", 3),
            query_params={"run_id": context.run_id},
            ranker=config.params.get("ranker", "naive"),
        ))
        return [
            dict(item.content) for item in result.items
            if isinstance(item.content, dict)
        ]
