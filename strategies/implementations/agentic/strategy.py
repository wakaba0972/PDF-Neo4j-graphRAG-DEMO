from __future__ import annotations

import json
from typing import Any

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


def _format_record(record: Any) -> RetrieverResultItem:
    evidence = dict(record["evidence"])
    score = float(record["score"])
    evidence.update({
        "score": score,
        "fusion_score": score,
        "matched_by": ["agentic-seed"],
    })
    return RetrieverResultItem(content=evidence, metadata={"score": score})


class AgenticRetrievalStrategy:
    """Use an LLM to choose bounded next hops over real Neo4j graph edges."""

    strategy_id = "agentic"
    result_formatter = staticmethod(_format_record)

    @staticmethod
    def _read(context: RetrievalContext, query: str, **params: Any) -> list[dict[str, Any]]:
        def operation() -> list[dict[str, Any]]:
            with context.driver.session(database=context.database) as session:
                return session.run(query, **params).data()

        return context.retry(operation)

    @staticmethod
    def _seed_entities(
        context: RetrievalContext, seeds: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        evidence_ids = list(dict.fromkeys(
            str(item.get("evidence_id", "")) for item in seeds
            if item.get("evidence_id")
        ))
        if not evidence_ids:
            return []
        records = AgenticRetrievalStrategy._read(
            context,
            """
            MATCH (evidence:GraphEvidence)
            WHERE evidence.evidence_id IN $evidence_ids
            OPTIONAL MATCH (evidence)-[:REPRESENTS|MENTIONS_ENTITY|EVIDENCE_SOURCE|EVIDENCE_TARGET]->
                (entity:ExtractedEntity)
            WITH evidence, entity WHERE entity IS NOT NULL
            RETURN DISTINCT elementId(entity) AS entity_id,
                   entity.name AS name, entity.type AS type,
                   entity.description AS description,
                   evidence.document_id AS document_id,
                   evidence.evidence_id AS seed_evidence_id
            LIMIT $limit
            """,
            evidence_ids=evidence_ids,
            limit=max(len(evidence_ids) * 20, 20),
        )
        return records

    @staticmethod
    def _choose_next(
        context: RetrievalContext,
        candidates: list[dict[str, Any]],
        question: str,
        max_branches: int,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []
        if not context.model_base_url.strip() or not context.model_api_key.strip() or not context.model.strip():
            raise ValueError("Agentic Search 需要可用的 OpenAI 問答模型設定")

        options = [{
            "id": str(index),
            "path_so_far": candidate["path"][:-1],
            "next_relationship": candidate["relationship"],
            "next_entity": {
                "name": candidate["name"],
                "type": candidate.get("type", ""),
                "description": candidate.get("description", ""),
                "relationship_degree": candidate.get("degree", 0),
            },
        } for index, candidate in enumerate(candidates)]
        payload: dict[str, Any] = {
            "model": context.model.strip(),
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a knowledge-graph retrieval agent. Given the question, select which "
                        "entities in the candidate paths are worth exploring next. Prefer direct "
                        "relationships that may supply information needed to answer the question; "
                        "avoid expanding through generic, highly connected entities. Candidate "
                        "relationships and descriptions are retrieval clues, not final answers. "
                        "Do not write Cypher or invent candidate options. If the current evidence "
                        "is sufficient or all candidates are irrelevant, return an empty expand list "
                        "and set stop=true. Return JSON only: {\"expand\":[candidate ids],\"stop\":true or false}."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\nSelect at most {max_branches} candidates in this round.\n"
                        f"Candidate paths: {json.dumps(options, ensure_ascii=False)}"
                    ),
                },
            ],
        }
        model = context.model.strip().casefold()
        if model == GPT_6_LUNA_MODEL:
            effort = context.reasoning_effort or "low"
            if effort not in GPT_6_LUNA_REASONING_EFFORTS:
                raise ValueError("Agentic Search 的 GPT-6 Luna 推理強度設定無效")
            payload["reasoning_effort"] = effort
            if effort == "none":
                payload["temperature"] = 0
        else:
            payload["temperature"] = 0

        response = _post_json(
            _api_url(context.model_base_url, "chat/completions"),
            payload,
            context.model_api_key,
        )
        content, _ = _chat_response_content(response)
        try:
            selection = _extract_json_text(content)
            if selection.get("stop") is True:
                return []
            selected_ids = selection.get("expand")
            if not isinstance(selected_ids, list):
                return []
            selected_indices = []
            for value in selected_ids:
                try:
                    index = int(value)
                except (TypeError, ValueError, OverflowError):
                    continue
                if 0 <= index < len(candidates) and index not in selected_indices:
                    selected_indices.append(index)
                if len(selected_indices) >= max_branches:
                    break
            return [candidates[index] for index in selected_indices]
        except (TypeError, ValueError, AttributeError):
            # Invalid tool output stops expansion safely; it never triggers an
            # unconstrained query or falls back to walking every neighbor.
            return []

    def retrieve(
        self, context: RetrievalContext, config: RetrievalConfig,
    ) -> list[dict[str, Any]]:
        params = config.params
        candidate_limit = int(params.get("candidate_top_k", config.top_k))
        seed_limit = min(int(params.get("seed_top_k", 8)), candidate_limit)
        max_hops = int(params.get("max_hops", 2))
        max_branches = int(params.get("max_branches", 4))
        hop_decay = float(params.get("hop_decay", 0.8))

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
        seeds = [
            dict(item.content) for item in result.items
            if isinstance(item.content, dict)
        ]
        if not seeds:
            return []

        entities = self._seed_entities(context, seeds)
        seed_scores = {
            str(seed.get("evidence_id")): float(seed.get("fusion_score", 0.0) or 0.0)
            for seed in seeds if seed.get("evidence_id")
        }
        frontier_by_id: dict[str, dict[str, Any]] = {}
        for seed_entity in entities:
            entity_id = seed_entity["entity_id"]
            entity_state = {
                **seed_entity,
                "hop": 0,
                "seed_score": seed_scores.get(seed_entity.get("seed_evidence_id", ""), 0.0),
                "path": [],
            }
            previous = frontier_by_id.get(entity_id)
            if previous is None or entity_state["seed_score"] > previous["seed_score"]:
                frontier_by_id[entity_id] = entity_state
        frontier = list(frontier_by_id.values())
        visited = {item["entity_id"] for item in frontier}
        expanded_entity_paths: dict[str, dict[str, Any]] = {}

        for _hop in range(max_hops):
            if not frontier:
                break
            frontier_by_id = {item["entity_id"]: item for item in frontier}
            neighbor_records = self._read(
                context,
                """
                MATCH (current:ExtractedEntity)
                WHERE elementId(current) IN $entity_ids
                MATCH (current)-[relation:EXTRACTED_RELATION]-
                      (neighbor:ExtractedEntity)
                WHERE NOT (elementId(neighbor) IN $visited_ids)
                OPTIONAL MATCH (neighbor)-[adjacent:EXTRACTED_RELATION]-()
                WITH current, relation, neighbor, count(DISTINCT adjacent) AS degree
                RETURN elementId(current) AS current_id, current.name AS current_name,
                       elementId(neighbor) AS neighbor_id, neighbor.name AS name,
                       neighbor.type AS type, neighbor.description AS description,
                       relation.type AS relation_type, relation.description AS relation_description,
                       startNode(relation).name AS source_name,
                       endNode(relation).name AS target_name,
                       coalesce(relation.source_pages, []) AS source_pages,
                       coalesce(relation.source_documents, []) AS source_documents,
                       relation.document_id AS document_id,
                       degree
                ORDER BY degree ASC, name ASC
                LIMIT $limit
                """,
                entity_ids=list(frontier_by_id),
                visited_ids=list(visited),
                limit=min(max_branches * 8, 80),
            )
            candidates_by_id: dict[str, dict[str, Any]] = {}
            for record in neighbor_records:
                current = frontier_by_id.get(record.get("current_id"))
                neighbor_id = record.get("neighbor_id")
                if not current or not neighbor_id:
                    continue
                relationship = {
                    "from": record.get("source_name", ""),
                    "type": record.get("relation_type", ""),
                    "to": record.get("target_name", ""),
                    "traversed_from": current.get("name", ""),
                    "traversed_to": record.get("name", ""),
                    "description": record.get("relation_description", ""),
                    "source_pages": record.get("source_pages", []),
                    "source_documents": record.get("source_documents", []),
                }
                path = [*current["path"], relationship]
                candidate = {
                    "entity_id": neighbor_id,
                    "name": record.get("name", ""),
                    "type": record.get("type", ""),
                    "description": record.get("description", ""),
                    "degree": int(record.get("degree", 0) or 0),
                    "relationship": relationship,
                    "path": path,
                    "hop": current["hop"] + 1,
                    "seed_score": current["seed_score"],
                }
                existing = candidates_by_id.get(neighbor_id)
                if existing is None or (
                    len(path), -candidate["seed_score"]
                ) < (
                    len(existing["path"]), -existing["seed_score"]
                ):
                    candidates_by_id[neighbor_id] = candidate
            candidates = list(candidates_by_id.values())
            if not candidates:
                break
            selected = self._choose_next(
                context, candidates, context.question, max_branches,
            )
            frontier = []
            for candidate in selected:
                entity_id = candidate["entity_id"]
                if entity_id in visited:
                    continue
                visited.add(entity_id)
                expanded_entity_paths[entity_id] = candidate
                frontier.append(candidate)

        expanded = []
        if expanded_entity_paths:
            records = self._read(
                context,
                """
                MATCH (chunk:GraphEvidence {kind: '原文'})
                      -[:MENTIONS_ENTITY]->(entity:ExtractedEntity)
                WHERE elementId(entity) IN $entity_ids
                WITH DISTINCT chunk, collect(DISTINCT elementId(entity)) AS entity_ids
                RETURN chunk {
                    .evidence_id, .kind, .name, .source, .target, .text,
                    .source_pages, .source_chunk_numbers, .source_documents,
                    .source_references_json, .document_id
                } AS evidence, entity_ids
                LIMIT $limit
                """,
                entity_ids=list(expanded_entity_paths),
                limit=min(candidate_limit * max_branches * max_hops, 100),
            )
            seed_ids = {item.get("evidence_id") for item in seeds}
            best_path_by_entity = expanded_entity_paths
            for record in records:
                evidence = record.get("evidence")
                if not isinstance(evidence, dict) or evidence.get("evidence_id") in seed_ids:
                    continue
                paths = [best_path_by_entity[entity_id] for entity_id in record.get("entity_ids", [])
                         if entity_id in best_path_by_entity]
                if not paths:
                    continue
                path = min(paths, key=lambda item: (item["hop"], -item["seed_score"]))
                score = path["seed_score"] * (hop_decay ** path["hop"])
                expanded.append({
                    **evidence,
                    "score": score,
                    "fusion_score": score,
                    "matched_by": ["graph", "agentic-graph"],
                    "graph_path": path["path"],
                    "graph_hops": path["hop"],
                })

        # Keep seed evidence and agent-selected graph evidence interleaved so
        # discoveries are not sliced away when reranking is disabled.
        seeds.sort(key=lambda item: float(item.get("fusion_score", 0.0) or 0.0), reverse=True)
        expanded.sort(key=lambda item: float(item.get("fusion_score", 0.0) or 0.0), reverse=True)
        output: list[dict[str, Any]] = []
        seed_index = expanded_index = 0
        while len(output) < candidate_limit and (seed_index < len(seeds) or expanded_index < len(expanded)):
            if seed_index < len(seeds):
                output.append(seeds[seed_index])
                seed_index += 1
            if expanded_index < len(expanded) and len(output) < candidate_limit:
                output.append(expanded[expanded_index])
                expanded_index += 1
        return output
