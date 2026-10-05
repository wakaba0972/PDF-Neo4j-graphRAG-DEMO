from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from neo4j import GraphDatabase
from neo4j.exceptions import DriverError, Neo4jError
from neo4j_graphrag.exceptions import Neo4jGraphRagError
from neo4j_graphrag.retrievers import HybridCypherRetriever, VectorCypherRetriever
from neo4j_graphrag.types import RetrieverResultItem


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
        with GraphDatabase.driver(uri.strip(), auth=(username.strip(), password)) as driver:
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
        with GraphDatabase.driver(uri.strip(), auth=(username.strip(), password)) as driver:
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
        with GraphDatabase.driver(uri.strip(), auth=(username.strip(), password)) as driver:
            driver.verify_connectivity()
            with driver.session(database=database.strip()) as session:
                record = session.run(query).single()
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
    retrieval_mode: str,
    top_k: int,
    candidate_top_k: int | None = None,
    expand_evidence: bool = True,
) -> list[dict[str, Any]]:
    target_vector_index = vector_index_name(len(embedding))
    retrieval_top_k = max(int(candidate_top_k or top_k), int(top_k))
    retrieval_query = """
    WITH node, score
    WHERE node.run_id = $run_id
    RETURN node {
        .evidence_id, .kind, .name, .source, .target, .text, .source_pages,
        .source_chunk_numbers, .source_documents, .source_references_json
    } AS evidence, score
    """
    try:
        with GraphDatabase.driver(uri.strip(), auth=(username.strip(), password)) as driver:
            query_params = {"run_id": run_id}
            if retrieval_mode in {"基本向量檢索", "基本檢索", "向量 RAG"}:
                retriever = VectorCypherRetriever(
                    driver=driver,
                    index_name=target_vector_index,
                    retrieval_query=retrieval_query,
                    result_formatter=_format_vector_record,
                    neo4j_database=database.strip(),
                )
                result = retriever.search(
                    query_vector=embedding,
                    top_k=retrieval_top_k,
                    effective_search_ratio=3,
                    query_params=query_params,
                )
            else:
                retriever = HybridCypherRetriever(
                    driver=driver,
                    vector_index_name=target_vector_index,
                    fulltext_index_name="graph_evidence_fulltext",
                    retrieval_query=retrieval_query,
                    result_formatter=_format_hybrid_record,
                    neo4j_database=database.strip(),
                )
                result = retriever.search(
                    query_text=_escape_fulltext_query(question),
                    query_vector=embedding,
                    top_k=retrieval_top_k,
                    effective_search_ratio=3,
                    query_params=query_params,
                    ranker="naive",
                )
            selected = [
                dict(item.content) for item in result.items
                if isinstance(item.content, dict)
            ][:retrieval_top_k]

            if expand_evidence and retrieval_mode in {"混合檢索", "關聯擴展檢索", "GraphRAG"} and selected:
                with driver.session(database=database.strip()) as session:
                    chunk_numbers = sorted({
                        number
                        for item in selected
                        for number in item.get("source_chunk_numbers", [])
                    })
                    names = [
                        item.get("name", "")
                        for item in selected
                        if item.get("kind") == "實體" and item.get("name")
                    ]
                    graph_chunk_numbers = list(dict.fromkeys(
                        number
                        for item in selected
                        if item.get("kind") != "原文"
                        for number in item.get("source_chunk_numbers", [])
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

                    source_chunks = fetch_source_chunks(graph_chunk_numbers)
                    selected_ids = {
                        item.get("evidence_id", "") for item in selected
                    }
                    selected.extend(
                        item for item in source_chunks
                        if item.get("evidence_id", "") not in selected_ids
                    )
                    selected_ids = {item.get("evidence_id", "") for item in selected}
                    related_entities = session.run(
                        """
                        MATCH (entity:GraphEvidence {run_id: $run_id, kind: '實體'})
                        WHERE (
                            entity.name IN $names OR any(
                                number IN coalesce(entity.source_chunk_numbers, [])
                                WHERE number IN $chunk_numbers
                            )
                        )
                        RETURN entity {
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
                    entity_evidence = [
                        _expanded_evidence(record["evidence"])
                        for record in related_entities
                    ]
                    selected.extend(
                        item for item in entity_evidence
                        if item.get("evidence_id", "") not in selected_ids
                    )
                    expanded_names = list(dict.fromkeys(
                        names + [item.get("name", "") for item in entity_evidence]
                    ))
                    selected_ids = {item.get("evidence_id", "") for item in selected}
                    related = session.run(
                        """
                        MATCH (relation:GraphEvidence {run_id: $run_id, kind: '關係'})
                        WHERE (
                            relation.source IN $names OR relation.target IN $names OR any(
                                number IN coalesce(relation.source_chunk_numbers, [])
                                WHERE number IN $chunk_numbers
                            )
                        )
                        RETURN relation {
                            .evidence_id, .kind, .name, .source, .target, .text,
                            .source_pages, .source_chunk_numbers, .source_documents,
                            .source_references_json
                        } AS evidence
                        LIMIT $top_k
                        """,
                        run_id=run_id,
                        names=expanded_names,
                        chunk_numbers=chunk_numbers,
                        top_k=retrieval_top_k,
                    ).data()
                    relation_evidence = [
                        _expanded_evidence(record["evidence"])
                        for record in related
                    ]
                    selected.extend(
                        item for item in relation_evidence
                        if item.get("evidence_id", "") not in selected_ids
                    )

                    existing_source_chunk_numbers = {
                        number
                        for item in selected
                        if item.get("kind") == "原文"
                        for number in item.get("source_chunk_numbers", [])
                    }
                    expanded_chunk_numbers = list(dict.fromkeys(
                        number
                        for item in [*entity_evidence, *relation_evidence]
                        for number in item.get("source_chunk_numbers", [])
                        if number not in existing_source_chunk_numbers
                    ))
                    second_pass_chunks = fetch_source_chunks(
                        expanded_chunk_numbers
                    )
                    selected_ids = {
                        item.get("evidence_id", "") for item in selected
                    }
                    selected.extend(
                        item for item in second_pass_chunks
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
        with GraphDatabase.driver(uri.strip(), auth=(username.strip(), password)) as driver:
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
    return {
        "entity_count": int(entity_result["count"]) if entity_result else 0,
        "relationship_count": int(relationship_result["count"])
        if relationship_result
        else 0,
    }
