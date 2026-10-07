from __future__ import annotations

import atexit
import json
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from threading import RLock
from typing import Any

from neo4j import GraphDatabase
from neo4j.exceptions import (
    DriverError, Neo4jError, ServiceUnavailable, SessionExpired,
)
from neo4j_graphrag.exceptions import Neo4jGraphRagError
from neo4j_graphrag.retrievers import HybridCypherRetriever, VectorCypherRetriever
from neo4j_graphrag.types import RetrieverResultItem

from .retrieval import (
    RetrievalConfig, RetrievalContext, RetrievalStrategyRegistry,
)


_DRIVER_LOCK = RLock()
_DRIVER_CACHE: dict[tuple[Any, ...], Any] = {}


def _driver_for(uri: str, username: str, password: str) -> Any:
    """Reuse thread-safe Neo4j drivers; sessions remain short-lived per query."""
    factory = GraphDatabase.driver
    factory_key = (getattr(factory, "__self__", None), getattr(factory, "__func__", factory))
    key = (*factory_key, uri.strip(), username.strip(), password)
    with _DRIVER_LOCK:
        driver = _DRIVER_CACHE.get(key)
        if driver is None:
            driver = factory(uri.strip(), auth=(username.strip(), password))
            _DRIVER_CACHE[key] = driver
        return driver


@contextmanager
def _shared_driver(uri: str, username: str, password: str):
    """Borrow a cached driver without closing it when this operation ends."""
    yield _driver_for(uri, username, password)


def _retry_read(operation, attempts: int = 3):
    """Retry reads briefly when a Neo4j cluster is refreshing its READ routes."""
    for attempt in range(attempts):
        try:
            return operation()
        except (ServiceUnavailable, SessionExpired):
            if attempt + 1 >= attempts:
                raise
            time.sleep(0.2 * (2 ** attempt))
    raise RuntimeError("Neo4j read retry ended unexpectedly")


def _close_cached_drivers() -> None:
    with _DRIVER_LOCK:
        drivers = list(_DRIVER_CACHE.values())
        _DRIVER_CACHE.clear()
    for driver in drivers:
        try:
            driver.close()
        except Exception:
            pass


atexit.register(_close_cached_drivers)


@dataclass(frozen=True)
class ImportSummary:
    entity_count: int
    relationship_count: int


def vector_index_name(dimensions: int) -> str:
    if int(dimensions) < 1:
        raise ValueError("Embedding 向量維度必須大於 0")
    return f"graph_evidence_embedding_{int(dimensions)}"


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


RETRIEVAL_STRATEGIES = RetrievalStrategyRegistry()
RETRIEVAL_STRATEGIES.register(VectorRetrievalStrategy())
RETRIEVAL_STRATEGIES.register(HybridRetrievalStrategy())


def _expanded_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        **evidence,
        "score": 0.0,
        "fusion_score": 0.0,
        "matched_by": ["graph"],
    }


def _restore_source_references(item: dict[str, Any]) -> dict[str, Any]:
    raw = item.pop("source_references_json", None)
    if raw and not item.get("source_references"):
        try:
            references = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            references = []
        if isinstance(references, list):
            item["source_references"] = references
    return item


def check_neo4j_connection(
    uri: str, database: str, username: str, password: str
) -> None:
    if not uri.strip():
        raise ValueError("請先填寫 Neo4j URI")
    if not database.strip():
        raise ValueError("請先填寫 Neo4j Database")
    if not username.strip():
        raise ValueError("請先填寫 Neo4j Username")
    if not password:
        raise ValueError("請先填寫 Neo4j Password")
    try:
        with _shared_driver(uri, username, password) as driver:
            driver.verify_connectivity()
            with driver.session(database=database.strip()) as session:
                session.run("RETURN 1 AS value").consume()
    except (DriverError, Neo4jError, OSError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("請先填寫"):
            raise
        raise ValueError(f"Neo4j 連線失敗：{exc}") from exc


def ensure_project_database(uri: str, database: str, username: str, password: str) -> None:
    """Create the project database when supported, then verify it is usable."""
    if not re.fullmatch(r"[a-z][a-z0-9.-]*", database.strip()):
        raise ValueError("專案 Neo4j Database 名稱格式無效")
    try:
        with _shared_driver(uri, username, password) as driver:
            driver.verify_connectivity()
            with driver.session(database="system") as session:
                session.run(
                    f"CREATE DATABASE `{database.strip()}` IF NOT EXISTS WAIT 30 SECONDS"
                ).consume()
    except (DriverError, Neo4jError, OSError, ValueError) as exc:
        raise ValueError(
            "無法建立專案專屬 Neo4j Database；此功能需要支援多資料庫的 Neo4j Enterprise "
            f"及管理權限（Neo4j Community／Aura 不支援）：{exc}"
        ) from exc
    check_neo4j_connection(uri, database, username, password)


def load_latest_graph(
    uri: str, database: str, username: str, password: str
) -> dict[str, Any]:
    if not uri.strip():
        raise ValueError("請先填寫 Neo4j URI")
    if not database.strip():
        raise ValueError("請先填寫 Neo4j Database")
    if not username.strip():
        raise ValueError("請先填寫 Neo4j Username")
    if not password:
        raise ValueError("請先填寫 Neo4j Password")
    query = """
    MATCH (document:GraphDocument)
    WITH document ORDER BY document.updated_at DESC LIMIT 1
    OPTIONAL MATCH (entity:ExtractedEntity)-[:IN_DOCUMENT]->(document)
    WITH document, collect(DISTINCT entity {
        .name, .type, .description, .source_chunk_numbers, .source_pages,
        .source_documents, .source_references_json
    }) AS entities
    OPTIONAL MATCH (source:ExtractedEntity)-[relation:EXTRACTED_RELATION]->(target:ExtractedEntity)
    WHERE relation.run_id = document.run_id
    RETURN document.run_id AS run_id, document.file_name AS document,
           document.embedding_model AS embedding_model,
           document.embedding_dimensions AS embedding_dimensions,
           document.vector_index_name AS vector_index_name, entities,
           collect(DISTINCT relation {
               source: source.name, target: target.name, .type, .description,
               .source_chunk_numbers, .source_pages, .source_documents,
               .source_references_json
           }) AS relationships
    """
    try:
        with _shared_driver(uri, username, password) as driver:
            def fetch_latest_graph_record():
                with driver.session(database=database.strip()) as session:
                    return session.run(query).single()

            record = _retry_read(fetch_latest_graph_record)
    except (DriverError, Neo4jError, OSError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("請先填寫"):
            raise
        raise ValueError(f"Neo4j 查詢失敗：{exc}") from exc
    if record is None:
        raise ValueError("Neo4j 中沒有可供問答的 GraphDocument")
    result = dict(record)
    result["neo4j_imported"] = True
    result["entities"] = [
        _restore_source_references(item)
        for item in result.get("entities", [])
        if item
    ]
    result["relationships"] = [
        _restore_source_references(item)
        for item in result.get("relationships", [])
        if item
    ]
    return result


def search_graph_evidence(
    uri: str,
    database: str,
    username: str,
    password: str,
    run_id: str,
    question: str,
    embedding: list[float],
    config: RetrievalConfig,
) -> list[dict[str, Any]]:
    config.validated()
    graph_hops = config.expansion.get("params", {}).get("hops", 4)
    target_vector_index = vector_index_name(len(embedding))
    retrieval_top_k = config.params.get("candidate_top_k", config.top_k)
    retrieval_query = """
    WITH node, score
    WHERE node.run_id = $run_id
    RETURN node {
        .evidence_id, .kind, .name, .source, .target, .text, .source_pages,
        .source_chunk_numbers, .source_documents, .source_references_json
    } AS evidence, score
    """
    try:
        with _shared_driver(uri, username, password) as driver:
            strategy = RETRIEVAL_STRATEGIES.get(config.strategy_id)
            context = RetrievalContext(
                driver=driver, database=database.strip(), run_id=run_id,
                question=question, embedding=embedding,
                vector_index_name=target_vector_index,
                fulltext_index_name="graph_evidence_fulltext",
                retrieval_query=retrieval_query,
                result_formatter=strategy.result_formatter,
                retry=_retry_read,
            )
            selected = strategy.retrieve(context, config)[:retrieval_top_k]

            expansion_id = config.expansion["id"]
            if expansion_id != "disabled" and selected:
                with driver.session(database=database.strip()) as session:
                    seed_ids = list(dict.fromkeys(
                        item.get("evidence_id", "") for item in selected
                        if item.get("evidence_id")
                    ))
                    def fetch_source_chunks(
                        target_chunk_numbers: list[int],
                    ) -> list[dict[str, Any]]:
                        if not target_chunk_numbers:
                            return []
                        records = session.run(
                            """
                            MATCH (chunk:GraphEvidence {run_id: $run_id, kind: "原文"})
                            WHERE any(
                                number IN coalesce(chunk.source_chunk_numbers, [])
                                WHERE number IN $chunk_numbers
                            )
                            RETURN chunk {
                                .evidence_id, .kind, .name, .source, .target, .text,
                                .source_pages, .source_chunk_numbers, .source_documents,
                                .source_references_json
                            } AS evidence
                            """,
                            run_id=run_id,
                            chunk_numbers=target_chunk_numbers,
                        ).data()
                        chunks = [
                            _expanded_evidence(record["evidence"])
                            for record in records
                        ]
                        priority = {
                            number: index
                            for index, number in enumerate(target_chunk_numbers)
                        }
                        chunks.sort(key=lambda item: min(
                            (
                                priority[number]
                                for number in item.get("source_chunk_numbers", [])
                                if number in priority
                            ),
                            default=len(priority),
                        ))
                        return chunks[:retrieval_top_k]

                    if expansion_id == "legacy_name_chunk":
                        # Legacy behavior: one lookup by entity name / shared
                        # chunk numbers, followed by source-chunk recovery.
                        chunk_numbers = sorted({
                            number for item in selected
                            for number in item.get("source_chunk_numbers", [])
                        })
                        names = list(dict.fromkeys(
                            item.get("name", "") for item in selected
                            if item.get("kind") == "實體" and item.get("name")
                        ))
                        graph_chunk_numbers = list(dict.fromkeys(
                            number for item in selected if item.get("kind") != "原文"
                            for number in item.get("source_chunk_numbers", [])
                        ))
                        source_chunks = fetch_source_chunks(graph_chunk_numbers)
                        related_records = session.run(
                            """
                            MATCH (candidate:GraphEvidence {run_id: $run_id})
                            WHERE candidate.kind IN ['實體', '關係'] AND (
                                candidate.name IN $names OR candidate.source IN $names
                                OR candidate.target IN $names OR any(
                                    number IN coalesce(candidate.source_chunk_numbers, [])
                                    WHERE number IN $chunk_numbers
                                )
                            )
                            RETURN candidate {
                                .evidence_id, .kind, .name, .source, .target, .text,
                                .source_pages, .source_chunk_numbers, .source_documents,
                                .source_references_json
                            } AS evidence
                            LIMIT $top_k
                            """,
                            run_id=run_id,
                            names=names,
                            chunk_numbers=chunk_numbers,
                            top_k=retrieval_top_k,
                        ).data()
                        selected_ids = {item.get("evidence_id", "") for item in selected}
                        legacy_related = [
                            _expanded_evidence(record["evidence"])
                            for record in related_records
                            if isinstance(record.get("evidence"), dict)
                        ]
                        selected.extend(
                            item for item in [*legacy_related, *source_chunks]
                            if item.get("evidence_id", "") not in selected_ids
                        )
                        selected_ids = {item.get("evidence_id", "") for item in selected}
                        related_chunk_numbers = list(dict.fromkeys(
                            number for item in legacy_related
                            for number in item.get("source_chunk_numbers", [])
                        ))
                        selected.extend(
                            item for item in fetch_source_chunks(related_chunk_numbers)
                            if item.get("evidence_id", "") not in selected_ids
                        )
                        expanded_records = []
                    elif seed_ids:
                        expanded_records = session.run(
                            f"""
                            MATCH (seed:GraphEvidence {{run_id: $run_id}})
                            WHERE seed.evidence_id IN $seed_ids
                            MATCH path = (seed)-[
                                :REPRESENTS|EVIDENCE_SOURCE|EVIDENCE_TARGET|
                                 MENTIONS_ENTITY|EXTRACTED_RELATION*1..{graph_hops}
                            ]-(candidate:GraphEvidence {{run_id: $run_id}})
                            WHERE NOT (candidate.evidence_id IN $seed_ids)
                            WITH candidate, min(length(path)) AS distance
                            RETURN candidate {{
                                .evidence_id, .kind, .name, .source, .target, .text,
                                .source_pages, .source_chunk_numbers, .source_documents,
                                .source_references_json
                            }} AS evidence, distance
                            ORDER BY distance, candidate.evidence_id
                            LIMIT $top_k
                            """,
                            run_id=run_id,
                            seed_ids=seed_ids,
                            top_k=retrieval_top_k,
                        ).data()
                    else:
                        expanded_records = []
                    selected_ids = {item.get("evidence_id", "") for item in selected}
                    related_evidence = [
                        _expanded_evidence(record["evidence"])
                        for record in expanded_records
                        if isinstance(record.get("evidence"), dict)
                    ]
                    selected.extend(
                        item for item in related_evidence
                        if item.get("evidence_id", "") not in selected_ids
                    )
                    selected_ids = {item.get("evidence_id", "") for item in selected}

                    graph_chunk_numbers = list(dict.fromkeys(
                        number
                        for item in selected
                        if item.get("kind") != "原文"
                        for number in item.get("source_chunk_numbers", [])
                    ))
                    source_chunks = fetch_source_chunks(graph_chunk_numbers)
                    selected.extend(
                        item for item in source_chunks
                        if item.get("evidence_id", "") not in selected_ids
                    )
    except (DriverError, Neo4jError, Neo4jGraphRagError, OSError, ValueError) as exc:
        detail = str(exc)
        guidance = ""
        if "dimensionality" in detail.casefold() or "dimension" in detail.casefold() or "index" in detail.casefold():
            guidance = " 請使用目前的 Embedding 模型重新執行「Embedding 並匯入 Neo4j」以建立對應維度索引。"
        raise ValueError(f"Neo4j 檢索失敗：{exc}{guidance}") from exc
    except Exception as exc:
        # neo4j-graphrag 1.19.0 tries to format a missing-index error with the
        # nonexistent `self.index_name` attribute. Replace that implementation
        # detail with the actual index requested by this application.
        if isinstance(exc, AttributeError) and "index_name" in str(exc):
            detail = f"找不到向量索引 `{target_vector_index}`。"
        else:
            detail = str(exc)
        raise ValueError(
            f"Neo4j 檢索失敗：{detail} "
            "請使用目前的 Embedding 模型重新執行「Embedding 並匯入 Neo4j」。"
        ) from exc
    return [_restore_source_references(item) for item in selected]


def import_extraction(
    uri: str,
    database: str,
    username: str,
    password: str,
    run_id: str,
    document_name: str,
    llm_model: str,
    embedding_model: str,
    schema: dict[str, Any],
    entities: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
) -> ImportSummary:
    if not uri.strip():
        raise ValueError("請先填寫 Neo4j URI")
    if not database.strip():
        raise ValueError("請先填寫 Neo4j Database")
    if not username.strip():
        raise ValueError("請先填寫 Neo4j Username")
    if not password:
        raise ValueError("請先填寫 Neo4j Password")
    if not evidence or not evidence[0].get("embedding"):
        raise ValueError("沒有可寫入 Neo4j Vector Index 的向量證據")
    try:
        with _shared_driver(uri, username, password) as driver:
            driver.verify_connectivity()
            with driver.session(database=database.strip()) as session:
                dimensions = len(evidence[0]["embedding"])
                index_name = vector_index_name(dimensions)
                # Neo4j permits only one vector index for the same label and
                # property schema. IF NOT EXISTS would silently keep an older
                # index with a different dimension, so remove all indexes
                # owned by this application before creating the current one.
                existing_indexes = session.run(
                    "SHOW VECTOR INDEXES YIELD name "
                    "WHERE name = 'graph_evidence_embedding' "
                    "OR name STARTS WITH 'graph_evidence_embedding_' "
                    "RETURN name"
                ).data()
                for record in existing_indexes:
                    old_index_name = str(record.get("name", ""))
                    if re.fullmatch(r"graph_evidence_embedding(?:_\d+)?", old_index_name):
                        session.run(f"DROP INDEX `{old_index_name}` IF EXISTS").consume()
                session.run(
                    f"CREATE VECTOR INDEX {index_name} IF NOT EXISTS "
                    "FOR (e:GraphEvidence) ON (e.embedding) OPTIONS {"
                    "indexConfig: {`vector.dimensions`: "
                    f"{dimensions}, "
                    "`vector.similarity_function`: 'cosine'}}"
                ).consume()
                session.run(
                    "CREATE FULLTEXT INDEX graph_evidence_fulltext IF NOT EXISTS "
                    "FOR (e:GraphEvidence) ON EACH [e.text, e.name, e.source, e.target] "
                    "OPTIONS {indexConfig: {`fulltext.analyzer`: 'cjk'}}"
                ).consume()
                # Waiting by a just-created index name can race Neo4j schema
                # propagation and incorrectly raise IndexNotFound. Await all
                # indexes only after both DDL statements have committed.
                session.run("CALL db.awaitIndexes(300)").consume()
                counts = session.execute_write(
                    _write_graph,
                    run_id,
                    document_name,
                    llm_model,
                    embedding_model,
                    schema,
                    entities,
                    relationships,
                    evidence,
                    dimensions,
                    index_name,
                )
    except (DriverError, Neo4jError, OSError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("請先填寫"):
            raise
        raise ValueError(f"Neo4j 寫入失敗：{exc}") from exc
    return ImportSummary(counts["entity_count"], counts["relationship_count"])


def _write_graph(
    transaction: Any,
    run_id: str,
    document_name: str,
    llm_model: str,
    embedding_model: str,
    schema: dict[str, Any],
    entities: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    embedding_dimensions: int,
    vector_index: str,
) -> dict[str, int]:
    def with_source_references(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                **item,
                "source_references_json": json.dumps(
                    item.get("source_references", []), ensure_ascii=False
                ),
            }
            for item in items
        ]

    entities = with_source_references(entities)
    relationships = with_source_references(relationships)
    evidence = with_source_references(evidence)
    transaction.run(
        """
        MATCH (node)
        WHERE node:GraphDocument OR node:ExtractedEntity OR node:GraphEvidence
        DETACH DELETE node
        """
    ).consume()

    transaction.run(
        """
        MERGE (document:GraphDocument {run_id: $run_id})
        SET document.file_name = $document_name,
            document.llm_model = $llm_model,
            document.embedding_model = $embedding_model,
            document.embedding_dimensions = $embedding_dimensions,
            document.vector_index_name = $vector_index_name,
            document.schema_json = $schema_json,
            document.updated_at = datetime()
        """,
        run_id=run_id,
        document_name=document_name,
        llm_model=llm_model,
        embedding_model=embedding_model,
        embedding_dimensions=embedding_dimensions,
        vector_index_name=vector_index,
        schema_json=json.dumps(schema, ensure_ascii=False),
    ).consume()
    entity_result = transaction.run(
        """
        MATCH (document:GraphDocument {run_id: $run_id})
        UNWIND $entities AS item
        MERGE (entity:ExtractedEntity {
            run_id: $run_id,
            type: item.type,
            name: item.name
        })
        SET entity.description = item.description,
            entity.source_chunk_numbers = item.source_chunk_numbers,
            entity.source_pages = item.source_pages,
            entity.source_documents = item.source_documents,
            entity.source_references_json = item.source_references_json
        MERGE (entity)-[:IN_DOCUMENT]->(document)
        RETURN count(entity) AS count
        """,
        run_id=run_id,
        entities=entities,
    ).single()
    relationship_result = transaction.run(
        """
        UNWIND $relationships AS item
        MATCH (source:ExtractedEntity {run_id: $run_id, name: item.source})
        MATCH (target:ExtractedEntity {run_id: $run_id, name: item.target})
        WITH item, head(collect(source)) AS source, head(collect(target)) AS target
        MERGE (source)-[relation:EXTRACTED_RELATION {
            run_id: $run_id,
            type: item.type
        }]->(target)
        SET relation.description = item.description,
            relation.source_chunk_numbers = item.source_chunk_numbers,
            relation.source_pages = item.source_pages,
            relation.source_documents = item.source_documents,
            relation.source_references_json = item.source_references_json
        RETURN count(relation) AS count
        """,
        run_id=run_id,
        relationships=relationships,
    ).single()
    transaction.run(
        """
        MATCH (document:GraphDocument {run_id: $run_id})
        UNWIND $evidence AS item
        MERGE (evidence:GraphEvidence {run_id: $run_id, evidence_id: item.evidence_id})
        SET evidence.kind = item.kind, evidence.name = item.name,
            evidence.source = item.source, evidence.target = item.target,
            evidence.text = item.text, evidence.source_pages = item.source_pages,
            evidence.source_chunk_numbers = item.source_chunk_numbers,
            evidence.source_documents = item.source_documents,
            evidence.source_references_json = item.source_references_json,
            evidence.embedding = item.embedding
        MERGE (evidence)-[:IN_DOCUMENT]->(document)
        """,
        run_id=run_id,
        evidence=evidence,
    ).consume()
    transaction.run(
        """
        MATCH (evidence:GraphEvidence {run_id: $run_id, kind: '實體'})
        MATCH (entity:ExtractedEntity {run_id: $run_id, name: evidence.name})
        MERGE (evidence)-[:REPRESENTS]->(entity)
        """,
        run_id=run_id,
    ).consume()
    transaction.run(
        """
        MATCH (evidence:GraphEvidence {run_id: $run_id, kind: '關係'})
        MATCH (source:ExtractedEntity {run_id: $run_id, name: evidence.source})
        MATCH (target:ExtractedEntity {run_id: $run_id, name: evidence.target})
        MERGE (evidence)-[:EVIDENCE_SOURCE]->(source)
        MERGE (evidence)-[:EVIDENCE_TARGET]->(target)
        """,
        run_id=run_id,
    ).consume()
    transaction.run(
        """
        MATCH (chunk:GraphEvidence {run_id: $run_id, kind: '原文'})
        UNWIND coalesce(chunk.source_chunk_numbers, []) AS chunk_number
        MATCH (entity:ExtractedEntity {run_id: $run_id})
        WHERE chunk_number IN coalesce(entity.source_chunk_numbers, [])
        MERGE (chunk)-[:MENTIONS_ENTITY]->(entity)
        """,
        run_id=run_id,
    ).consume()
    return {
        "entity_count": int(entity_result["count"]) if entity_result else 0,
        "relationship_count": int(relationship_result["count"])
        if relationship_result
        else 0,
    }
