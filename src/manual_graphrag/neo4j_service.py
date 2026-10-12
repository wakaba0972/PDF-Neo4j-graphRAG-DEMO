from __future__ import annotations

import atexit
import json
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from threading import RLock
from typing import Any, Iterable

from neo4j import GraphDatabase
from neo4j.exceptions import (
    DriverError, Neo4jError, ServiceUnavailable, SessionExpired,
)
from neo4j_graphrag.exceptions import Neo4jGraphRagError
from .retrieval import (
    RETRIEVAL_STRATEGIES, RetrievalConfig, RetrievalContext,
)


_DRIVER_LOCK = RLock()
_DRIVER_CACHE: dict[tuple[Any, ...], Any] = {}
GRAPH_WRITE_BATCH_SIZE = 200


def _driver_for(uri: str, username: str, password: str) -> Any:
    """Reuse thread-safe Neo4j drivers; sessions remain short-lived per query."""
    factory = GraphDatabase.driver
    factory_key = (getattr(factory, "__self__", None), getattr(factory, "__func__", factory))
    key = (*factory_key, uri.strip(), username.strip(), password)
    with _DRIVER_LOCK:
        driver = _DRIVER_CACHE.get(key)
        if driver is None:
            driver = factory(
                uri.strip(),
                auth=(username.strip(), password),
                connection_timeout=30,
                liveness_check_timeout=30,
                max_connection_lifetime=1800,
            )
            _DRIVER_CACHE[key] = driver
        return driver


def _execute_write_with_retry(
    driver: Any,
    database: str,
    callback,
    *args: Any,
    attempts: int = 3,
) -> Any:
    """Use a fresh session for each attempt; tolerate a stale idle pool socket."""
    for attempt in range(attempts):
        try:
            with driver.session(database=database) as session:
                return session.execute_write(callback, *args)
        except (ServiceUnavailable, SessionExpired, OSError):
            if attempt + 1 >= attempts:
                raise
            time.sleep(0.5 * (2 ** attempt))
    raise RuntimeError("Neo4j write retry ended unexpectedly")


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
    embedding_dimensions: int = 0


def vector_index_name(dimensions: int) -> str:
    if int(dimensions) < 1:
        raise ValueError("Embedding 向量維度必須大於 0")
    return f"graph_evidence_embedding_{int(dimensions)}"


def _create_vector_index_query(index_name: str, dimensions: int) -> str:
    """Build the Neo4j vector-index DDL with both OPTIONS maps closed."""
    return (
        f"CREATE VECTOR INDEX {index_name} IF NOT EXISTS "
        "FOR (e:GraphEvidence) ON (e.embedding) OPTIONS {"
        "indexConfig: {`vector.dimensions`: "
        f"{int(dimensions)}, "
        "`vector.similarity_function`: 'cosine'}}"
    )


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
    WHERE coalesce(document.import_status, 'complete') = 'complete'
    WITH document ORDER BY document.updated_at DESC LIMIT 1
    OPTIONAL MATCH (entity:ExtractedEntity)-[:IN_DOCUMENT]->(document)
    WITH document, collect(DISTINCT entity {
        .name, .type, .description, .source_chunk_numbers, .source_pages,
        .source_documents, .source_references_json
    }) AS entities
    OPTIONAL MATCH (source:ExtractedEntity)-[relation:EXTRACTED_RELATION]->(target:ExtractedEntity)
    WHERE relation.document_id = document.document_id
    RETURN document.file_name AS document,
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


def load_latest_graph_metadata(
    uri: str, database: str, username: str, password: str
) -> dict[str, Any]:
    """Load only the active graph's metadata; do not materialize its full graph."""
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
    WHERE coalesce(document.import_status, 'complete') = 'complete'
    WITH document ORDER BY document.updated_at DESC LIMIT 1
    RETURN document.file_name AS document,
           document.embedding_model AS embedding_model,
           document.embedding_dimensions AS embedding_dimensions,
           document.vector_index_name AS vector_index_name
    """
    try:
        with _shared_driver(uri, username, password) as driver:
            def fetch_latest_graph_metadata():
                with driver.session(database=database.strip()) as session:
                    return session.run(query).single()

            record = _retry_read(fetch_latest_graph_metadata)
    except (DriverError, Neo4jError, OSError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("請先填寫"):
            raise
        raise ValueError(f"Neo4j 查詢失敗：{exc}") from exc
    if record is None:
        raise ValueError("Neo4j 中沒有可供問答的 GraphDocument")
    return {**dict(record), "neo4j_imported": True}


def search_graph_evidence(
    uri: str,
    database: str,
    username: str,
    password: str,
    question: str,
    embedding: list[float],
    config: RetrievalConfig,
    *,
    model_base_url: str = "",
    model_api_key: str = "",
    model: str = "",
    reasoning_effort: str = "low",
) -> list[dict[str, Any]]:
    config.validated()
    target_vector_index = vector_index_name(len(embedding))
    retrieval_top_k = config.params.get("candidate_top_k", config.top_k)

    def document_ids_for(items: list[dict[str, Any]]) -> list[str]:
        document_ids: list[str] = []
        for item in items:
            if "document_id" in item:
                value = str(item.get("document_id") or "")
                if value:
                    document_ids.append(value)
                continue
            references = item.get("source_references")
            if not isinstance(references, list):
                try:
                    references = json.loads(item.get("source_references_json") or "[]")
                except (TypeError, json.JSONDecodeError):
                    references = []
            if isinstance(references, list):
                document_ids.extend(
                    str(reference.get("document_id", ""))
                    for reference in references
                    if isinstance(reference, dict) and reference.get("document_id")
                )
        return list(dict.fromkeys(document_ids))
    retrieval_query = """
    WITH node, score
    RETURN node {
        .evidence_id, .kind, .name, .source, .target, .text, .source_pages,
        .source_chunk_numbers, .source_documents, .source_references_json, .document_id
    } AS evidence, score
    """
    try:
        with _shared_driver(uri, username, password) as driver:
            strategy = RETRIEVAL_STRATEGIES.get(config.strategy_id)
            context = RetrievalContext(
                driver=driver, database=database.strip(),
                question=question, embedding=embedding,
                vector_index_name=target_vector_index,
                fulltext_index_name="graph_evidence_fulltext",
                retrieval_query=retrieval_query,
                result_formatter=strategy.result_formatter,
                retry=_retry_read,
                model_base_url=model_base_url,
                model_api_key=model_api_key,
                model=model,
                reasoning_effort=reasoning_effort,
            )
            selected = strategy.retrieve(context, config)[:retrieval_top_k]

            expansion_id = config.expansion["id"]
            if expansion_id != "disabled" and selected:
                with driver.session(database=database.strip()) as session:
                    def fetch_source_chunks(
                        target_chunk_numbers: list[int], document_ids: list[str],
                    ) -> list[dict[str, Any]]:
                        if not target_chunk_numbers:
                            return []
                        records = session.run(
                            """
                            MATCH (chunk:GraphEvidence {kind: "原文"})
                            WHERE (size($document_ids) = 0 OR chunk.document_id IN $document_ids)
                              AND any(
                                number IN coalesce(chunk.source_chunk_numbers, [])
                                WHERE number IN $chunk_numbers
                            )
                            RETURN chunk {
                                .evidence_id, .kind, .name, .source, .target, .text,
                                .source_pages, .source_chunk_numbers, .source_documents,
                                .source_references_json, .document_id
                            } AS evidence
                            """,
                            chunk_numbers=target_chunk_numbers,
                            document_ids=document_ids,
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
                        document_ids = document_ids_for(selected)
                        source_chunks = fetch_source_chunks(graph_chunk_numbers, document_ids)
                        related_records = session.run(
                            """
                            MATCH (candidate:GraphEvidence)
                            WHERE candidate.kind IN ['實體', '關係']
                              AND (size($document_ids) = 0 OR candidate.document_id IN $document_ids)
                              AND (
                                candidate.name IN $names OR candidate.source IN $names
                                OR candidate.target IN $names OR any(
                                    number IN coalesce(candidate.source_chunk_numbers, [])
                                    WHERE number IN $chunk_numbers
                                )
                            )
                            RETURN candidate {
                                .evidence_id, .kind, .name, .source, .target, .text,
                                .source_pages, .source_chunk_numbers, .source_documents,
                                .source_references_json, .document_id
                            } AS evidence
                            LIMIT $top_k
                            """,
                            names=names,
                            chunk_numbers=chunk_numbers,
                            document_ids=document_ids,
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
                            item for item in fetch_source_chunks(related_chunk_numbers, document_ids)
                            if item.get("evidence_id", "") not in selected_ids
                        )
                    selected_ids = {item.get("evidence_id", "") for item in selected}

                    graph_chunk_numbers = list(dict.fromkeys(
                        number
                        for item in selected
                        if item.get("kind") != "原文"
                        for number in item.get("source_chunk_numbers", [])
                    ))
                    document_ids = document_ids_for(selected)
                    source_chunks = fetch_source_chunks(graph_chunk_numbers, document_ids)
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
        guidance = ""
        if isinstance(exc, AttributeError) and "index_name" in str(exc):
            detail = f"找不到向量索引 `{target_vector_index}`。"
            guidance = " 請使用目前的 Embedding 模型重新執行「Embedding 並匯入 Neo4j」。"
        else:
            detail = str(exc)
            normalized = detail.casefold()
            if "dimensionality" in normalized or "dimension" in normalized:
                guidance = " 請使用目前的 Embedding 模型重新執行「Embedding 並匯入 Neo4j」。"
            elif isinstance(exc, AttributeError) and (
                "strategy" in normalized or "has no attribute" in normalized
            ):
                detail = f"檢索策略實作載入失敗：{detail}。請重新啟動應用程式後再試。"
        raise ValueError(f"Neo4j 檢索失敗：{detail}{guidance}") from exc
    return [_restore_source_references(item) for item in selected]


def import_extraction(
    uri: str,
    database: str,
    username: str,
    password: str,
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
                    _create_vector_index_query(index_name, dimensions)
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


def import_extraction_batched(
    uri: str,
    database: str,
    username: str,
    password: str,
    document_name: str,
    llm_model: str,
    embedding_model: str,
    schema: dict[str, Any],
    entities: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
    evidence_batches: Iterable[list[dict[str, Any]]],
    total_evidence: int,
    progress=None,
    graph_documents: list[dict[str, str]] | None = None,
) -> ImportSummary:
    """Write each already-embedded evidence batch in its own Neo4j transaction."""
    if not uri.strip():
        raise ValueError("請先填寫 Neo4j URI")
    if not database.strip():
        raise ValueError("請先填寫 Neo4j Database")
    if not username.strip():
        raise ValueError("請先填寫 Neo4j Username")
    if not password:
        raise ValueError("請先填寫 Neo4j Password")

    batches = iter(evidence_batches)
    try:
        first_batch = next(batches)
    except StopIteration as exc:
        raise ValueError("沒有可寫入 Neo4j Vector Index 的向量證據") from exc
    if not first_batch or not first_batch[0].get("embedding"):
        raise ValueError("沒有可寫入 Neo4j Vector Index 的向量證據")
    dimensions = len(first_batch[0]["embedding"])
    index_name = vector_index_name(dimensions)
    entity_count = 0
    relationship_count = 0
    written = 0
    initialized = False
    previous_dimensions: int | None = None

    try:
        with _shared_driver(uri, username, password) as driver:
            driver.verify_connectivity()
            with driver.session(database=database.strip()) as session:
                previous = session.run(
                    "MATCH (document:GraphDocument) "
                    "WHERE coalesce(document.import_status, 'complete') = 'complete' "
                    "RETURN document.embedding_dimensions AS dimensions "
                    "ORDER BY document.updated_at DESC LIMIT 1"
                ).single()
                if previous and previous.get("dimensions") is not None:
                    previous_dimensions = int(previous["dimensions"])
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
                fulltext_indexes = session.run(
                    "SHOW FULLTEXT INDEXES YIELD name "
                    "WHERE name = 'graph_evidence_fulltext' RETURN name"
                ).data()
                if fulltext_indexes:
                    session.run("DROP INDEX graph_evidence_fulltext IF EXISTS").consume()
                for obsolete_index in (
                    "graph_evidence_run_id_evidence_id",
                    "extracted_entity_run_id_type_name",
                    "extracted_entity_run_id_name",
                ):
                    session.run(f"DROP INDEX `{obsolete_index}` IF EXISTS").consume()
                session.run("CALL db.awaitIndexes(300)").consume()
                if progress:
                    progress(0.06, "Neo4j 寫入索引就緒，開始寫入實體與關係…")

                try:
                    while _execute_write_with_retry(
                        driver, database.strip(), _delete_existing_graph_batch,
                    ):
                        pass
                    _execute_write_with_retry(
                        driver,
                        database.strip(),
                        _create_graph_documents,
                        graph_documents or [{"document_id": "", "file_name": document_name}],
                        llm_model,
                        embedding_model,
                        schema,
                        dimensions,
                        index_name,
                    )
                    initialized = True
                    for entity_batch in _list_batches(entities, GRAPH_WRITE_BATCH_SIZE):
                        entity_count += _execute_write_with_retry(
                            driver,
                            database.strip(),
                            _write_entity_batch,
                            entity_batch,
                        )
                    for relationship_batch in _list_batches(
                        relationships, GRAPH_WRITE_BATCH_SIZE,
                    ):
                        relationship_count += _execute_write_with_retry(
                            driver,
                            database.strip(),
                            _write_relationship_batch,
                            relationship_batch,
                        )
                    entities_by_chunk: dict[tuple[str, int], list[dict[str, str]]] = {}
                    for entity in entities:
                        for chunk_number in entity.get("source_chunk_numbers", []):
                            try:
                                normalized_chunk_number = int(chunk_number)
                            except (TypeError, ValueError):
                                continue
                            document_id = str(entity.get("document_id", ""))
                            entities_by_chunk.setdefault((document_id, normalized_chunk_number), []).append({
                                "name": str(entity.get("name", "")),
                                "type": str(entity.get("type", "")),
                                "document_id": document_id,
                            })

                    _execute_write_with_retry(
                        driver,
                        database.strip(),
                        _write_evidence_batch,
                        first_batch,
                    )
                    _execute_write_with_retry(
                        driver,
                        database.strip(),
                        _link_evidence_batch,
                        first_batch,
                        entities_by_chunk,
                    )
                    written = len(first_batch)
                    if progress:
                        progress(
                            0.06 + 0.84 * min(written / max(total_evidence, 1), 1.0),
                            f"已寫入 Neo4j 第 1 批（{written}/{total_evidence} 筆證據）",
                        )
                    batch_number = 1
                    for batch in batches:
                        batch_number += 1
                        if not batch:
                            continue
                        if any(
                            len(item.get("embedding") or []) != dimensions
                            for item in batch
                        ):
                            raise ValueError("同一批匯入的 Embedding 向量維度不一致")
                        _execute_write_with_retry(
                            driver,
                            database.strip(),
                            _write_evidence_batch,
                            batch,
                        )
                        _execute_write_with_retry(
                            driver,
                            database.strip(),
                            _link_evidence_batch,
                            batch,
                            entities_by_chunk,
                        )
                        written += len(batch)
                        if progress:
                            progress(
                                0.06 + 0.84 * min(written / max(total_evidence, 1), 1.0),
                                f"已寫入 Neo4j 第 {batch_number} 批（{written}/{total_evidence} 筆證據）",
                            )
                    if written != total_evidence:
                        raise ValueError(
                            f"Embedding 證據數量不符：預期 {total_evidence} 筆，實際寫入 {written} 筆"
                        )
                    if progress:
                        progress(0.95, "建立向量與全文檢索索引…")
                    session.run(
                        _create_vector_index_query(index_name, dimensions)
                    ).consume()
                    session.run(
                        "CREATE FULLTEXT INDEX graph_evidence_fulltext IF NOT EXISTS "
                        "FOR (e:GraphEvidence) ON EACH [e.text, e.name, e.source, e.target] "
                        "OPTIONS {indexConfig: {`fulltext.analyzer`: 'cjk'}}"
                    ).consume()
                    session.run("CALL db.awaitIndexes(300)").consume()
                    _execute_write_with_retry(
                        driver, database.strip(), _complete_graph_import,
                    )
                except Exception:
                    if initialized:
                        try:
                            while _execute_write_with_retry(
                                driver,
                                database.strip(),
                                _delete_existing_graph_batch,
                            ):
                                pass
                        except Exception:
                            pass
                    restore_dimensions = (
                        dimensions if initialized else (previous_dimensions or dimensions)
                    )
                    try:
                        restore_index_name = vector_index_name(restore_dimensions)
                        session.run(
                            _create_vector_index_query(
                                restore_index_name, restore_dimensions,
                            )
                        ).consume()
                        session.run(
                            "CREATE FULLTEXT INDEX graph_evidence_fulltext IF NOT EXISTS "
                            "FOR (e:GraphEvidence) ON EACH [e.text, e.name, e.source, e.target] "
                            "OPTIONS {indexConfig: {`fulltext.analyzer`: 'cjk'}}"
                        ).consume()
                        session.run("CALL db.awaitIndexes(300)").consume()
                    except Exception:
                        pass
                    raise
    except (DriverError, Neo4jError, OSError) as exc:
        raise ValueError(f"Neo4j 寫入失敗：{exc}") from exc

    if progress:
        progress(1.0, "Embedding 與 Neo4j 匯入完成")
    return ImportSummary(entity_count, relationship_count, dimensions)


def _write_graph(
    transaction: Any,
    document_name: str,
    llm_model: str,
    embedding_model: str,
    schema: dict[str, Any],
    entities: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    embedding_dimensions: int,
    vector_index: str,
    import_status: str = "complete",
) -> dict[str, int]:
    def with_source_references(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                **item,
                "document_id": str(item.get("document_id", "")),
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
        CREATE (document:GraphDocument {document_id: ''})
        SET document.file_name = $document_name,
            document.llm_model = $llm_model,
            document.embedding_model = $embedding_model,
            document.embedding_dimensions = $embedding_dimensions,
            document.vector_index_name = $vector_index_name,
            document.schema_json = $schema_json,
            document.import_status = $import_status,
            document.updated_at = datetime()
        """,
        document_name=document_name,
        llm_model=llm_model,
        embedding_model=embedding_model,
        embedding_dimensions=embedding_dimensions,
        vector_index_name=vector_index,
        schema_json=json.dumps(schema, ensure_ascii=False),
        import_status=import_status,
    ).consume()
    entity_result = transaction.run(
        """
        MATCH (document:GraphDocument)
        UNWIND $entities AS item
        MERGE (entity:ExtractedEntity {
            document_id: item.document_id,
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
        entities=entities,
    ).single()
    relationship_result = transaction.run(
        """
        UNWIND $relationships AS item
        MATCH (source:ExtractedEntity {name: item.source})
        MATCH (target:ExtractedEntity {name: item.target})
        WITH item, head(collect(source)) AS source, head(collect(target)) AS target
        MERGE (source)-[relation:EXTRACTED_RELATION {
            type: item.type
        }]->(target)
        SET relation.description = item.description,
            relation.document_id = item.document_id,
            relation.source_chunk_numbers = item.source_chunk_numbers,
            relation.source_pages = item.source_pages,
            relation.source_documents = item.source_documents,
            relation.source_references_json = item.source_references_json
        RETURN count(relation) AS count
        """,
        relationships=relationships,
    ).single()
    transaction.run(
        """
        MATCH (document:GraphDocument)
        UNWIND $evidence AS item
        MERGE (evidence:GraphEvidence {evidence_id: item.evidence_id})
        SET evidence.kind = item.kind, evidence.name = item.name,
            evidence.source = item.source, evidence.target = item.target,
            evidence.text = item.text, evidence.source_pages = item.source_pages,
            evidence.source_chunk_numbers = item.source_chunk_numbers,
            evidence.source_documents = item.source_documents,
            evidence.source_references_json = item.source_references_json,
            evidence.document_id = item.document_id,
            evidence.embedding = item.embedding
        MERGE (evidence)-[:IN_DOCUMENT]->(document)
        """,
        evidence=evidence,
    ).consume()
    transaction.run(
        """
        MATCH (evidence:GraphEvidence {kind: '實體'})
        MATCH (entity:ExtractedEntity {name: evidence.name})
        MERGE (evidence)-[:REPRESENTS]->(entity)
        """,
    ).consume()
    transaction.run(
        """
        MATCH (evidence:GraphEvidence {kind: '關係'})
        MATCH (source:ExtractedEntity {name: evidence.source})
        MATCH (target:ExtractedEntity {name: evidence.target})
        MERGE (evidence)-[:EVIDENCE_SOURCE]->(source)
        MERGE (evidence)-[:EVIDENCE_TARGET]->(target)
        """,
    ).consume()
    transaction.run(
        """
        MATCH (chunk:GraphEvidence {kind: '原文'})
        UNWIND coalesce(chunk.source_chunk_numbers, []) AS chunk_number
        MATCH (entity:ExtractedEntity)
        WHERE chunk_number IN coalesce(entity.source_chunk_numbers, [])
        MERGE (chunk)-[:MENTIONS_ENTITY]->(entity)
        """,
    ).consume()
    return {
        "entity_count": int(entity_result["count"]) if entity_result else 0,
        "relationship_count": int(relationship_result["count"])
        if relationship_result
        else 0,
    }


def _list_batches(items: list[dict[str, Any]], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _delete_existing_graph_batch(transaction: Any) -> int:
    result = transaction.run(
        "MATCH (node) "
        "WHERE node:GraphDocument OR node:ExtractedEntity OR node:GraphEvidence "
        "WITH node LIMIT $batch_size "
        "DETACH DELETE node "
        "RETURN count(node) AS deleted",
        batch_size=GRAPH_WRITE_BATCH_SIZE,
    ).single()
    return int(result["deleted"]) if result else 0


def _create_graph_document(
    transaction: Any,
    document_name: str,
    llm_model: str,
    embedding_model: str,
    schema: dict[str, Any],
    embedding_dimensions: int,
    vector_index: str,
) -> None:
    transaction.run(
        """
        CREATE (document:GraphDocument {
            document_id: '',
            file_name: $document_name,
            llm_model: $llm_model,
            embedding_model: $embedding_model,
            embedding_dimensions: $embedding_dimensions,
            vector_index_name: $vector_index,
            schema_json: $schema_json,
            import_status: 'staging',
            updated_at: datetime()
        })
        """,
        document_name=document_name,
        llm_model=llm_model,
        embedding_model=embedding_model,
        embedding_dimensions=embedding_dimensions,
        vector_index=vector_index,
        schema_json=json.dumps(schema, ensure_ascii=False),
    ).consume()


def _create_graph_documents(
    transaction: Any,
    documents: list[dict[str, str]],
    llm_model: str,
    embedding_model: str,
    schema: dict[str, Any],
    embedding_dimensions: int,
    vector_index: str,
) -> None:
    transaction.run(
        """
        UNWIND $documents AS item
        CREATE (document:GraphDocument {
            document_id: item.document_id,
            file_name: item.file_name,
            llm_model: $llm_model,
            embedding_model: $embedding_model,
            embedding_dimensions: $embedding_dimensions,
            vector_index_name: $vector_index,
            schema_json: $schema_json,
            import_status: 'staging',
            updated_at: datetime()
        })
        """,
        documents=documents,
        llm_model=llm_model,
        embedding_model=embedding_model,
        embedding_dimensions=embedding_dimensions,
        vector_index=vector_index,
        schema_json=json.dumps(schema, ensure_ascii=False),
    ).consume()


def _with_source_references(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            **item,
            "document_id": str(item.get("document_id", "")),
            "source_references_json": json.dumps(
                item.get("source_references", []), ensure_ascii=False,
            ),
        }
        for item in items
    ]


def _write_entity_batch(
    transaction: Any,
    entities: list[dict[str, Any]],
) -> int:
    result = transaction.run(
        """
        UNWIND $entities AS item
        MATCH (document:GraphDocument {document_id: item.document_id})
        MERGE (entity:ExtractedEntity {
            document_id: item.document_id,
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
        entities=_with_source_references(entities),
    ).single()
    return int(result["count"]) if result else 0


def _write_relationship_batch(
    transaction: Any,
    relationships: list[dict[str, Any]],
) -> int:
    result = transaction.run(
        """
        UNWIND $relationships AS item
        MATCH (source:ExtractedEntity {document_id: item.document_id, name: item.source})
        MATCH (target:ExtractedEntity {document_id: item.document_id, name: item.target})
        WITH item, head(collect(source)) AS source, head(collect(target)) AS target
        MERGE (source)-[relation:EXTRACTED_RELATION {
            type: item.type
        }]->(target)
        SET relation.description = item.description,
            relation.document_id = item.document_id,
            relation.source_chunk_numbers = item.source_chunk_numbers,
            relation.source_pages = item.source_pages,
            relation.source_documents = item.source_documents,
            relation.source_references_json = item.source_references_json
        RETURN count(relation) AS count
        """,
        relationships=_with_source_references(relationships),
    ).single()
    return int(result["count"]) if result else 0


def _write_evidence_batch(
    transaction: Any,
    evidence: list[dict[str, Any]],
) -> None:
    evidence = _with_source_references(evidence)
    transaction.run(
        """
        UNWIND $evidence AS item
        MATCH (document:GraphDocument {document_id: item.document_id})
        MERGE (evidence:GraphEvidence {evidence_id: item.evidence_id})
        SET evidence.kind = item.kind, evidence.name = item.name,
            evidence.source = item.source, evidence.target = item.target,
            evidence.text = item.text, evidence.source_pages = item.source_pages,
            evidence.source_chunk_numbers = item.source_chunk_numbers,
            evidence.source_documents = item.source_documents,
            evidence.source_references_json = item.source_references_json,
            evidence.document_id = item.document_id,
            evidence.embedding = item.embedding
        MERGE (evidence)-[:IN_DOCUMENT]->(document)
        """,
        evidence=evidence,
    ).consume()


def _link_evidence_batch(
    transaction: Any,
    evidence: list[dict[str, Any]],
    entities_by_chunk: dict[tuple[str, int], list[dict[str, str]]],
) -> None:
    entity_links = [
        {"evidence_id": item["evidence_id"], "name": item.get("name", ""),
         "document_id": item.get("document_id", "")}
        for item in evidence
        if item.get("kind") == "實體"
    ]
    if entity_links:
        transaction.run(
            """
            UNWIND $items AS item
            MATCH (evidence:GraphEvidence {evidence_id: item.evidence_id})
            MATCH (entity:ExtractedEntity {document_id: item.document_id, name: item.name})
            WITH evidence, head(collect(entity)) AS entity
            MERGE (evidence)-[:REPRESENTS]->(entity)
            """,
            items=entity_links,
        ).consume()

    relationship_links = [
        {
            "evidence_id": item["evidence_id"],
            "source": item.get("source", ""),
            "target": item.get("target", ""),
            "document_id": item.get("document_id", ""),
        }
        for item in evidence
        if item.get("kind") == "關係"
    ]
    if relationship_links:
        transaction.run(
            """
            UNWIND $items AS item
            MATCH (evidence:GraphEvidence {evidence_id: item.evidence_id})
            MATCH (source:ExtractedEntity {document_id: item.document_id, name: item.source})
            WITH evidence, item, head(collect(source)) AS source
            MATCH (target:ExtractedEntity {document_id: item.document_id, name: item.target})
            WITH evidence, source, head(collect(target)) AS target
            MERGE (evidence)-[:EVIDENCE_SOURCE]->(source)
            MERGE (evidence)-[:EVIDENCE_TARGET]->(target)
            """,
            items=relationship_links,
        ).consume()

    mention_links = []
    for item in evidence:
        if item.get("kind") != "原文":
            continue
        for chunk_number in item.get("source_chunk_numbers", []):
            try:
                linked_entities = entities_by_chunk.get(
                    (str(item.get("document_id", "")), int(chunk_number)), []
                )
            except (TypeError, ValueError):
                continue
            for entity in linked_entities:
                mention_links.append({
                    "evidence_id": item["evidence_id"],
                    "name": entity["name"],
                    "type": entity["type"],
                    "document_id": entity["document_id"],
                })
    if mention_links:
        transaction.run(
            """
            UNWIND $items AS item
            MATCH (chunk:GraphEvidence {evidence_id: item.evidence_id})
            MATCH (entity:ExtractedEntity {
                document_id: item.document_id, name: item.name, type: item.type
            })
            MERGE (chunk)-[:MENTIONS_ENTITY]->(entity)
            """,
            items=mention_links,
        ).consume()


def _complete_graph_import(transaction: Any) -> None:
    transaction.run(
        "MATCH (document:GraphDocument) "
        "SET document.import_status = 'complete', document.updated_at = datetime()",
    ).consume()
