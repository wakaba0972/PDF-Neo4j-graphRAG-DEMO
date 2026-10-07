from __future__ import annotations

from typing import Any

from neo4j_graphrag.retrievers import VectorCypherRetriever
from neo4j_graphrag.types import RetrieverResultItem

from manual_graphrag.retrieval import RetrievalConfig, RetrievalContext


def _format_vector_record(record: Any) -> RetrieverResultItem:
    evidence = dict(record["evidence"])
    score = float(record["score"])
    evidence.update({
        "score": score,
        "fusion_score": score,
        "matched_by": ["official-vector"],
    })
    return RetrieverResultItem(content=evidence, metadata={"score": score})


class VectorRetrievalStrategy:
    strategy_id = "vector"
    result_formatter = staticmethod(_format_vector_record)

    def retrieve(
        self, context: RetrievalContext, config: RetrievalConfig,
    ) -> list[dict[str, Any]]:
        retriever = VectorCypherRetriever(
            driver=context.driver,
            index_name=context.vector_index_name,
            retrieval_query=context.retrieval_query,
            result_formatter=context.result_formatter,
            neo4j_database=context.database,
        )
        result = context.retry(lambda: retriever.search(
            query_vector=context.embedding,
            top_k=config.params.get("candidate_top_k", config.top_k),
            effective_search_ratio=config.params.get("effective_search_ratio", 3),
            query_params={"run_id": context.run_id},
        ))
        return [
            dict(item.content) for item in result.items
            if isinstance(item.content, dict)
        ]
