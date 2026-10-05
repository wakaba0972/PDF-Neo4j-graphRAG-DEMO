import json

import pytest

from manual_graphrag import neo4j_service


class FakeResult:
    def __init__(self, count=None, rows=None):
        self.count = count
        self.rows = rows or []
        self.consumed = False

    def consume(self):
        self.consumed = True
        return self

    def single(self):
        return None if self.count is None else {"count": self.count}

    def data(self):
        return self.rows


class FakeTransaction:
    def __init__(self):
        self.calls = []

    def run(self, query, **parameters):
        self.calls.append((query, parameters))
        if "RETURN count(entity)" in query:
            return FakeResult(2)
        if "RETURN count(relation)" in query:
            return FakeResult(1)
        return FakeResult()


class FakeSession:
    def __init__(self, transaction):
        self.transaction = transaction
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def run(self, query, **parameters):
        self.calls.append((query, parameters))
        if "SHOW VECTOR INDEXES" in query:
            return FakeResult(rows=[
                {"name": "graph_evidence_embedding"},
                {"name": "graph_evidence_embedding_3072"},
                {"name": "unrelated_vector_index"},
            ])
        return FakeResult()

    def execute_write(self, callback, *args):
        return callback(self.transaction, *args)


class FakeDriver:
    def __init__(self, transaction):
        self.transaction = transaction
        self.verified = False
        self.database = None
        self.session_instance = FakeSession(transaction)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def verify_connectivity(self):
        self.verified = True

    def session(self, *, database):
        self.database = database
        return self.session_instance


def test_import_extraction_writes_document_entities_and_relationships(monkeypatch) -> None:
    transaction = FakeTransaction()
    driver = FakeDriver(transaction)
    captured = {}

    def fake_driver(uri, auth):
        captured.update({"uri": uri, "auth": auth})
        return driver

    monkeypatch.setattr(neo4j_service.GraphDatabase, "driver", fake_driver)
    entities = [
        {
            "name": "設備 A",
            "type": "DEVICE",
            "description": "設備",
            "source_chunk_numbers": [1],
            "source_pages": [2],
        },
        {
            "name": "設備 B",
            "type": "DEVICE",
            "description": "配件",
            "source_chunk_numbers": [1],
            "source_pages": [2],
        },
    ]
    relationships = [
        {
            "source": "設備 A",
            "type": "USES",
            "target": "設備 B",
            "description": "使用",
            "source_chunk_numbers": [1],
            "source_pages": [2],
        }
    ]

    summary = neo4j_service.import_extraction(
        "bolt://db",
        "neo4j",
        "user",
        "password",
        "run-1",
        "manual.pdf",
        "llm",
        "embed",
        {"entity_types": [{"name": "DEVICE"}]},
        entities,
        relationships,
        [{
            "evidence_id": "chunk-1", "kind": "原文", "text": "設備說明",
            "embedding": [0.1], "source_pages": [2],
            "source_chunk_numbers": [1],
        }],
    )

    assert summary == neo4j_service.ImportSummary(2, 1)
    assert captured == {"uri": "bolt://db", "auth": ("user", "password")}
    assert driver.verified is True
    assert driver.database == "neo4j"
    index_calls = driver.session(database="neo4j").calls
    assert len(index_calls) == 6
    assert "SHOW VECTOR INDEXES" in index_calls[0][0]
    assert "DROP INDEX `graph_evidence_embedding`" in index_calls[1][0]
    assert "DROP INDEX `graph_evidence_embedding_3072`" in index_calls[2][0]
    assert all("unrelated_vector_index" not in query for query, _ in index_calls)
    assert "CREATE VECTOR INDEX graph_evidence_embedding_1" in index_calls[3][0]
    assert "CREATE FULLTEXT INDEX" in index_calls[4][0]
    assert "fulltext.analyzer" in index_calls[4][0]
    assert "db.awaitIndexes(300)" in index_calls[5][0]
    assert index_calls[5][1] == {}
    assert len(transaction.calls) == 8
    assert "DETACH DELETE node" in transaction.calls[0][0]
    assert "GraphDocument" in transaction.calls[1][0]
    assert transaction.calls[1][1]["embedding_dimensions"] == 1
    assert transaction.calls[1][1]["vector_index_name"] == "graph_evidence_embedding_1"
    assert transaction.calls[2][1]["entities"][0]["name"] == entities[0]["name"]
    assert transaction.calls[3][1]["relationships"][0]["type"] == relationships[0]["type"]
    assert "EXTRACTED_RELATION" in transaction.calls[3][0]
    assert "GraphEvidence" in transaction.calls[4][0]
    assert "entity.source_documents" in transaction.calls[2][0]
    assert "relation.source_documents" in transaction.calls[3][0]
    assert "evidence.source_documents" in transaction.calls[4][0]
    assert "source_references_json" in transaction.calls[2][0]
    assert transaction.calls[2][1]["entities"][0]["source_references_json"] == "[]"
    assert "REPRESENTS" in transaction.calls[5][0]
    assert "EVIDENCE_SOURCE" in transaction.calls[6][0]
    assert "EVIDENCE_TARGET" in transaction.calls[6][0]
    assert "MENTIONS_ENTITY" in transaction.calls[7][0]


def test_restore_source_references_from_neo4j_json() -> None:
    item = {
        "source_documents": ["a.pdf", "b.pdf"],
        "source_references_json": json.dumps([
            {"document": "a.pdf", "chunk_numbers": [1], "pages": [3]},
            {"document": "b.pdf", "chunk_numbers": [8], "pages": [2]},
        ]),
    }

    restored = neo4j_service._restore_source_references(item)

    assert restored["source_references"] == [
        {"document": "a.pdf", "chunk_numbers": [1], "pages": [3]},
        {"document": "b.pdf", "chunk_numbers": [8], "pages": [2]},
    ]
    assert "source_references_json" not in restored


@pytest.mark.parametrize(
    "values,message",
    [
        (("", "neo4j", "user", "password"), "Neo4j URI"),
        (("bolt://db", "", "user", "password"), "Neo4j Database"),
        (("bolt://db", "neo4j", "", "password"), "Neo4j Username"),
        (("bolt://db", "neo4j", "user", ""), "Neo4j Password"),
    ],
)
def test_import_extraction_validates_connection_fields(values, message) -> None:
    with pytest.raises(ValueError, match=message):
        neo4j_service.import_extraction(
            *values,
            "run-1",
            "manual.pdf",
            "llm",
            "embed",
            {},
            [],
            [],
            [],
        )


def test_import_extraction_wraps_driver_errors(monkeypatch) -> None:
    def broken_driver(uri, auth):
        raise OSError("database unavailable")

    monkeypatch.setattr(neo4j_service.GraphDatabase, "driver", broken_driver)

    with pytest.raises(ValueError, match="Neo4j 寫入失敗"):
        neo4j_service.import_extraction(
            "bolt://db",
            "neo4j",
            "user",
            "password",
            "run-1",
            "manual.pdf",
            "llm",
            "embed",
            {},
            [],
            [],
            [{"embedding": [0.1]}],
        )


def _evidence(evidence_id: str, score: float, kind: str = "原文") -> dict:
    return {
        "evidence_id": evidence_id,
        "kind": kind,
        "name": evidence_id if kind == "實體" else "",
        "source": "",
        "target": "",
        "text": evidence_id,
        "source_pages": [1],
        "source_chunk_numbers": [1],
        "score": score,
    }


def test_vector_index_name_uses_embedding_dimensions() -> None:
    assert neo4j_service.vector_index_name(1536) == "graph_evidence_embedding_1536"
    assert neo4j_service.vector_index_name(3072) == "graph_evidence_embedding_3072"
    with pytest.raises(ValueError, match="維度"):
        neo4j_service.vector_index_name(0)


def test_escape_fulltext_query_escapes_lucene_syntax() -> None:
    assert neo4j_service._escape_fulltext_query("E01 +(重試):A/B") == (
        r"E01 \+\(重試\)\:A\/B"
    )


def test_format_hybrid_record_preserves_evidence_and_official_score() -> None:
    item = neo4j_service._format_hybrid_record({
        "evidence": _evidence("both", 0.0),
        "score": 0.85,
    })

    assert item.content["evidence_id"] == "both"
    assert item.content["matched_by"] == ["official-hybrid"]
    assert item.content["fusion_score"] == 0.85
    assert item.metadata == {"score": 0.85}


class SearchSession:
    def __init__(self):
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def run(self, query, **parameters):
        self.calls.append((query, parameters))
        if "count(node)" in query:
            return FakeResult(3)
        raise AssertionError(f"unexpected query: {query}")


class SearchDriver:
    def __init__(self, session):
        self.session_instance = session

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def session(self, *, database):
        return self.session_instance


class FakeOfficialRetriever:
    initialization = None
    search_arguments = None

    def __init__(self, **kwargs):
        type(self).initialization = kwargs

    def search(self, **kwargs):
        type(self).search_arguments = kwargs
        evidence = [
            _evidence("both", 0.9),
            _evidence("vector-only", 0.8),
            _evidence("keyword-only", 0.7),
        ]
        for item in evidence:
            item.update({
                "matched_by": ["official-hybrid"],
                "fusion_score": item["score"],
            })
        return type("Result", (), {
            "items": [type("Item", (), {"content": item}) for item in evidence]
        })()


class FakeOfficialVectorRetriever:
    initialization = None
    search_arguments = None

    def __init__(self, **kwargs):
        type(self).initialization = kwargs

    def search(self, **kwargs):
        type(self).search_arguments = kwargs
        evidence = [_evidence("vector-hit", 0.9)]
        formatted = type(self).initialization["result_formatter"]({
            "evidence": evidence[0], "score": evidence[0]["score"],
        })
        return type("Result", (), {
            "items": [type("Item", (), {"content": formatted.content})()]
        })()


def test_missing_dimension_index_replaces_upstream_attribute_error(monkeypatch) -> None:
    session = SearchSession()
    monkeypatch.setattr(
        neo4j_service.GraphDatabase,
        "driver",
        lambda *args, **kwargs: SearchDriver(session),
    )

    class MissingIndexRetriever:
        def __init__(self, **kwargs):
            raise AttributeError("'HybridCypherRetriever' object has no attribute 'index_name'")

    monkeypatch.setattr(
        neo4j_service, "HybridCypherRetriever", MissingIndexRetriever
    )

    with pytest.raises(ValueError) as error:
        neo4j_service.search_graph_evidence(
            "bolt://db", "neo4j", "user", "password", "run-1",
            "question", [0.1] * 1536, "混合檢索", 3,
        )

    message = str(error.value)
    assert "graph_evidence_embedding_1536" in message
    assert "重新執行「Embedding 並匯入 Neo4j」" in message
    assert "object has no attribute" not in message


def test_search_dimension_error_instructs_user_to_reimport(monkeypatch) -> None:
    session = SearchSession()
    monkeypatch.setattr(
        neo4j_service.GraphDatabase,
        "driver",
        lambda *args, **kwargs: SearchDriver(session),
    )

    class BrokenRetriever:
        def __init__(self, **kwargs):
            pass

        def search(self, **kwargs):
            raise ValueError(
                "Vector index has configured dimensionality 3072, "
                "but the provided vector has dimension 1536"
            )

    monkeypatch.setattr(neo4j_service, "HybridCypherRetriever", BrokenRetriever)

    with pytest.raises(ValueError, match="重新執行.*Embedding 並匯入 Neo4j"):
        neo4j_service.search_graph_evidence(
            "bolt://db", "neo4j", "user", "password", "run-1",
            "question", [0.1] * 1536, "混合檢索", 3,
        )


def test_search_graph_evidence_uses_official_hybrid_retriever(monkeypatch) -> None:
    session = SearchSession()
    monkeypatch.setattr(
        neo4j_service.GraphDatabase,
        "driver",
        lambda *args, **kwargs: SearchDriver(session),
    )
    monkeypatch.setattr(
        neo4j_service, "HybridCypherRetriever", FakeOfficialRetriever
    )

    results = neo4j_service.search_graph_evidence(
        "bolt://db", "neo4j", "user", "password", "run-1",
        "E01 +(重試)", [0.1], "混合檢索", 3,
        candidate_top_k=9,
        expand_evidence=False,
    )

    assert [item["evidence_id"] for item in results] == [
        "both", "vector-only", "keyword-only",
    ]
    initialization = FakeOfficialRetriever.initialization
    assert initialization["vector_index_name"] == "graph_evidence_embedding_1"
    assert initialization["fulltext_index_name"] == "graph_evidence_fulltext"
    assert initialization["neo4j_database"] == "neo4j"
    assert "$run_id" in initialization["retrieval_query"]
    assert "$document_names" not in initialization["retrieval_query"]
    arguments = FakeOfficialRetriever.search_arguments
    assert "source_documents" in initialization["retrieval_query"]
    assert arguments["query_text"] == r"E01 \+\(重試\)"
    assert arguments["query_vector"] == [0.1]
    assert arguments["top_k"] == 9
    assert arguments["effective_search_ratio"] == 3
    assert arguments["query_params"] == {"run_id": "run-1"}
    assert arguments["ranker"] == "naive"
    assert not any("count(node)" in query for query, _parameters in session.calls)


def test_basic_vector_retrieval_uses_vector_cypher_retriever(monkeypatch) -> None:
    session = SearchSession()
    monkeypatch.setattr(
        neo4j_service.GraphDatabase,
        "driver",
        lambda *args, **kwargs: SearchDriver(session),
    )
    monkeypatch.setattr(
        neo4j_service, "VectorCypherRetriever", FakeOfficialVectorRetriever
    )
    monkeypatch.setattr(
        neo4j_service, "HybridCypherRetriever",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("混合檢索不應執行")),
    )

    results = neo4j_service.search_graph_evidence(
        "bolt://db", "neo4j", "user", "password", "run-1",
        "E01 如何處理", [0.1], "基本向量檢索", 3,
    )

    assert [item["evidence_id"] for item in results] == ["vector-hit"]
    assert results[0]["matched_by"] == ["official-vector"]
    initialization = FakeOfficialVectorRetriever.initialization
    assert initialization["index_name"] == "graph_evidence_embedding_1"
    assert initialization["neo4j_database"] == "neo4j"
    assert "$run_id" in initialization["retrieval_query"]
    assert "$document_names" not in initialization["retrieval_query"]
    assert FakeOfficialVectorRetriever.search_arguments == {
        "query_vector": [0.1],
        "top_k": 3,
        "effective_search_ratio": 3,
        "query_params": {"run_id": "run-1"},
    }
    assert not any("count(node)" in query for query, _parameters in session.calls)


class ExpansionSearchSession:
    def __init__(self):
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def run(self, query, **parameters):
        self.calls.append((query, parameters))
        if "MATCH path = (seed)-[" in query:
            return FakeResult(rows=[
                {"evidence": {
                    "evidence_id": "relation-misfire-spark-plug", "kind": "關係",
                    "name": "", "source": "第 1 缸失火", "target": "火星塞",
                    "text": "第 1 缸失火可能由火星塞造成", "source_pages": [3],
                    "source_chunk_numbers": [3], "source_documents": ["car.pdf"],
                    "source_references_json": "[]",
                }, "distance": 2},
                {"evidence": {
                    "evidence_id": "entity-spark-plug", "kind": "實體",
                    "name": "火星塞", "source": "", "target": "",
                    "text": "實體：火星塞", "source_pages": [2],
                    "source_chunk_numbers": [2], "source_documents": ["car.pdf"],
                    "source_references_json": "[]",
                }, "distance": 3},
            ])
        if 'kind: "原文"' in query:
            rows = [
                {
                    "evidence": {
                        "evidence_id": f"chunk-{number}",
                        "kind": "原文",
                        "name": "",
                        "source": "",
                        "target": "",
                        "text": f"原文 {number}",
                        "source_pages": [number],
                        "source_chunk_numbers": [number],
                        "source_documents": ["car.pdf"],
                        "source_references_json": "[]",
                    }
                }
                for number in parameters["chunk_numbers"]
            ]
            return FakeResult(rows=rows)
        raise AssertionError(f"unexpected query: {query}")


class ExpansionRetriever:
    def __init__(self, **kwargs):
        pass

    def search(self, **kwargs):
        evidence = {
            "evidence_id": "entity-p0301",
            "kind": "實體",
            "name": "P0301",
            "source": "",
            "target": "",
            "text": "實體：P0301",
            "source_pages": [1],
            "source_chunk_numbers": [1],
            "source_documents": ["car.pdf"],
            "source_references_json": "[]",
            "score": 0.9,
            "fusion_score": 0.9,
            "matched_by": ["official-hybrid"],
        }
        return type("Result", (), {
            "items": [type("Item", (), {"content": evidence})()]
        })()


def test_graph_expansion_fetches_new_source_chunks_once(monkeypatch) -> None:
    session = ExpansionSearchSession()
    monkeypatch.setattr(
        neo4j_service.GraphDatabase,
        "driver",
        lambda *args, **kwargs: SearchDriver(session),
    )
    monkeypatch.setattr(
        neo4j_service, "HybridCypherRetriever", ExpansionRetriever
    )

    results = neo4j_service.search_graph_evidence(
        "bolt://db",
        "neo4j",
        "user",
        "password",
        "run-1",
        "P0301 怎麼處理",
        [0.1],
        "GraphRAG",
        5,
    )

    assert [item["evidence_id"] for item in results] == [
        "entity-p0301",
        "relation-misfire-spark-plug",
        "entity-spark-plug",
        "chunk-1",
        "chunk-3",
        "chunk-2",
    ]
    source_queries = [
        parameters["chunk_numbers"]
        for query, parameters in session.calls
        if 'kind: "原文"' in query
    ]
    assert source_queries == [[1, 3, 2]]
    expansion_query = next(query for query, _parameters in session.calls if "MATCH path = (seed)-[" in query)
    assert "EXTRACTED_RELATION" in expansion_query
    assert "*1..4" in expansion_query
    assert "name IN $names" not in expansion_query


def test_graph_expansion_can_be_disabled(monkeypatch) -> None:
    session = ExpansionSearchSession()
    monkeypatch.setattr(
        neo4j_service.GraphDatabase,
        "driver",
        lambda *args, **kwargs: SearchDriver(session),
    )
    monkeypatch.setattr(
        neo4j_service, "HybridCypherRetriever", ExpansionRetriever
    )

    results = neo4j_service.search_graph_evidence(
        "bolt://db", "neo4j", "user", "password", "run-1",
        "P0301 怎麼處理", [0.1], "GraphRAG", 5,
        expand_evidence=False,
    )

    assert [item["evidence_id"] for item in results] == ["entity-p0301"]
    assert not any(
        'kind: "原文"' in query or "kind: '實體'" in query or "kind: '關係'" in query
        for query, _parameters in session.calls
    )


def test_ensure_project_database_creates_then_checks_database(monkeypatch) -> None:
    class Session:
        def __init__(self, database):
            self.database = database
            self.queries = []

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def run(self, query):
            self.queries.append(query)
            return FakeResult()

    class Driver:
        def __init__(self):
            self.sessions = []

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def verify_connectivity(self):
            pass

        def session(self, *, database):
            session = Session(database)
            self.sessions.append(session)
            return session

    driver = Driver()
    monkeypatch.setattr(neo4j_service.GraphDatabase, "driver", lambda *args, **kwargs: driver)

    neo4j_service.ensure_project_database("bolt://db", "vehicle-123abc", "user", "pass")

    assert driver.sessions[0].database == "system"
    assert "CREATE DATABASE `vehicle-123abc` IF NOT EXISTS WAIT 30 SECONDS" in driver.sessions[0].queries[0]
    assert driver.sessions[1].database == "vehicle-123abc"
    assert driver.sessions[1].queries[0] == "RETURN 1 AS value"


def test_ensure_project_database_rejects_unsafe_name():
    with pytest.raises(ValueError, match="名稱格式無效"):
        neo4j_service.ensure_project_database("bolt://db", "x` DROP DATABASE neo4j", "u", "p")
