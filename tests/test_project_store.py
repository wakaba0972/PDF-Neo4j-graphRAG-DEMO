from pathlib import Path
import hashlib
import gzip
import json
from datetime import datetime

import pytest

from manual_graphrag.audit import actor_context
from manual_graphrag.project_store import (
    append_question,
    create_project,
    delete_project,
    export_project_archive,
    import_project_archive,
    list_projects,
    load_project,
    load_project_metadata,
    load_project_summary,
    remove_document,
    save_project,
)
from manual_graphrag.question_set_store import create_question_set, set_project_question_set_bindings


def test_project_round_trip_and_listing(tmp_path) -> None:
    project = create_project("設備手冊", tmp_path)
    saved = save_project(
        project["project_id"],
        {"settings": {"api_key": "secret", "model_endpoint": "https://old.example/v1", "chunk_size": 1200}},
        root=tmp_path,
    )

    assert "api_key" not in saved["settings"]
    assert "model_endpoint" not in saved["settings"]
    assert load_project(project["project_id"], tmp_path)["settings"]["chunk_size"] == 1200
    assert list_projects(tmp_path) == [("設備手冊", project["project_id"])]
    assert saved["neo4j_database"].startswith("vehicle-")


def test_project_listing_uses_summary_without_reading_full_project_json(tmp_path, monkeypatch) -> None:
    import manual_graphrag.project_store as project_store

    project = create_project("快速清單", tmp_path)
    original_read_json = project_store.read_json
    reads = []

    def track_summary_reads(path):
        reads.append(Path(path).name)
        return original_read_json(path)

    monkeypatch.setattr(project_store, "read_json", track_summary_reads)

    assert list_projects(tmp_path) == [("快速清單", project["project_id"])]
    assert reads == ["summary.json"]


def test_load_project_summary_reads_existing_summary_without_loading_project(tmp_path, monkeypatch) -> None:
    import manual_graphrag.project_store as project_store

    project = create_project("摘要讀取", tmp_path)

    def fail_project_load(*args, **kwargs):
        raise AssertionError("existing summaries should not load project.json")

    monkeypatch.setattr(project_store, "load_project", fail_project_load)

    assert load_project_summary(project["project_id"], tmp_path)["name"] == "摘要讀取"


def test_project_name_can_be_updated_without_changing_base_name(tmp_path) -> None:
    project = create_project("61 | s25", tmp_path)

    saved = save_project(
        project["project_id"], {"name": "61 | s25 | 1500-200-None"}, root=tmp_path
    )

    assert saved["name"] == "61 | s25 | 1500-200-None"
    assert saved["base_name"] == "61 | s25"
    assert list_projects(tmp_path) == [("61 | s25 | 1500-200-None", project["project_id"])]


def test_project_summary_file_tracks_creator_documents_and_build_settings(tmp_path) -> None:
    project = create_project("61 | s25", tmp_path, actor="61")
    save_project(project["project_id"], {
        "settings": {
            "chunk_size": 1500, "chunk_overlap": 200,
        },
        "evaluation": {"questions": [{"number": 1}, {"number": 2}]},
        "documents_meta": [{"file_name": "guide.pdf"}, {"file_name": "parts.pdf"}],
        "chunks": [{"number": 1}, {"number": 2}, {"number": 3}],
        "graph_state": {
            "document": "guide.pdf",
            "build_config": {
                "chunk_size": 1500,
                "chunk_overlap": 200,
            },
        },
    }, root=tmp_path)

    summary_path = tmp_path / project["project_id"] / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary == load_project_summary(project["project_id"], tmp_path)
    assert summary["created_by"] == "61"
    assert summary["document_count"] == 2
    assert summary["chunk_count"] == 3
    assert summary["question_count"] == 2
    assert summary["chunk_size"] == 1500
    assert summary["chunk_overlap"] == 200
    assert "schema_granularity" not in summary
    assert "graph_partition_mode" not in summary
    assert summary["graph_built"] is True


def test_legacy_schema_fields_are_removed_from_saved_project(tmp_path) -> None:
    project = create_project("schema migration", tmp_path)
    save_project(project["project_id"], {
        "settings": {"schema_text": "{}", "schema_granularity": "詳細"},
        "graph_state": {
            "schema": {"entity_types": []}, "schema_json": "{}",
            "schema_granularity": "詳細", "document": "manual.pdf",
        },
    }, root=tmp_path)

    loaded = load_project(project["project_id"], tmp_path)

    assert "schema_text" not in loaded["settings"]
    assert "schema_granularity" not in loaded["settings"]
    assert not ({"schema", "schema_json", "schema_granularity"} & loaded["graph_state"].keys())


def test_existing_project_summary_drops_obsolete_graph_partition_mode(tmp_path) -> None:
    project = create_project("舊摘要", tmp_path)
    save_project(project["project_id"], {
        "settings": {"graph_partition_mode": "per_document"},
        "graph_state": {"graph_partition_mode": "per_document", "document": "guide.pdf"},
    }, root=tmp_path)
    summary_path = tmp_path / project["project_id"] / "summary.json"
    old_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    old_summary["graph_partition_mode"] = "per_document"
    summary_path.write_text(json.dumps(old_summary), encoding="utf-8")

    upgraded = load_project_summary(project["project_id"], tmp_path)

    assert "graph_partition_mode" not in upgraded
    assert "graph_partition_mode" not in json.loads(summary_path.read_text(encoding="utf-8"))


def test_loading_legacy_project_removes_graph_run_id(tmp_path) -> None:
    project = create_project("舊圖譜狀態", tmp_path)
    save_project(project["project_id"], {
        "settings": {"graph_partition_mode": "per_document"},
        "graph_state": {
            "run_id": "legacy-graph-run",
            "graph_partition_mode": "per_document",
            "document": "manual.pdf",
            "entity_count": 1,
        },
    }, root=tmp_path)

    loaded = load_project(project["project_id"], root=tmp_path)
    persisted = json.loads(
        (tmp_path / project["project_id"] / "project.json").read_text(encoding="utf-8")
    )

    assert "run_id" not in loaded["graph_state"]
    assert "run_id" not in persisted["graph_state"]
    assert "graph_partition_mode" not in loaded["graph_state"]
    assert "graph_partition_mode" not in persisted["graph_state"]
    assert "graph_partition_mode" not in loaded["settings"]
    assert "graph_partition_mode" not in persisted["settings"]


def test_graph_entities_and_relationships_are_stored_in_compressed_sidecar(tmp_path) -> None:
    project = create_project("外置圖譜", tmp_path)
    graph = {
        "run_id": "graph-run",
        "entities": [{"name": "A", "type": "PART", "description": "entity"}],
        "relationships": [{"source": "A", "target": "B", "type": "CONNECTS"}],
        "chunks": [{"number": 1, "text": "chunk text"}],
    }

    save_project(project["project_id"], {"graph_state": graph}, root=tmp_path)

    project_file = tmp_path / project["project_id"] / "project.json"
    persisted = json.loads(project_file.read_text(encoding="utf-8"))
    graph_file = tmp_path / project["project_id"] / "graph.json.gz"
    with gzip.open(graph_file, "rt", encoding="utf-8") as handle:
        sidecar = json.load(handle)
    metadata = load_project_metadata(project["project_id"], tmp_path)["graph_state"]
    full = load_project(project["project_id"], tmp_path)["graph_state"]

    assert "entities" not in persisted["graph_state"]
    assert "relationships" not in persisted["graph_state"]
    assert metadata["entity_count"] == 1
    assert metadata["relationship_count"] == 1
    assert sidecar == {"entities": graph["entities"], "relationships": graph["relationships"]}
    assert full["entities"] == graph["entities"]
    assert full["relationships"] == graph["relationships"]
    assert full["chunks"] == graph["chunks"]

    save_project(project["project_id"], {
        "graph_state": {"neo4j_imported": True},
    }, root=tmp_path)
    assert load_project(project["project_id"], tmp_path)["graph_state"]["entities"] == graph["entities"]


def test_chunks_are_stored_in_separate_compressed_sidecar_and_loaded_on_demand(tmp_path, monkeypatch) -> None:
    import manual_graphrag.project_store as project_store

    project = create_project("外置 chunks", tmp_path)
    chunks = [{"number": 1, "text": "原文內容", "pages": [1], "document": "manual.pdf"}]
    graph = {"run_id": "graph-run", "chunks": chunks, "entities": [{"name": "設備"}]}
    save_project(project["project_id"], {"chunks": chunks, "graph_state": graph}, root=tmp_path)

    project_dir = tmp_path / project["project_id"]
    stored = json.loads((project_dir / "project.json").read_text(encoding="utf-8"))
    with gzip.open(project_dir / "chunks.json.gz", "rt", encoding="utf-8") as handle:
        sidecar = json.load(handle)

    assert "chunks" not in stored
    assert "chunks" not in stored["graph_state"]
    assert stored["chunk_count"] == 1
    assert sidecar == {"chunks": chunks}

    def fail_chunk_read(*args, **kwargs):
        raise AssertionError("metadata loads must not decompress chunks")

    monkeypatch.setattr(project_store, "_read_chunk_data", fail_chunk_read)
    metadata = load_project_metadata(project["project_id"], tmp_path)
    assert "chunks" not in metadata
    monkeypatch.undo()

    loaded = load_project(project["project_id"], tmp_path)
    assert loaded["chunks"] == chunks
    assert loaded["graph_state"]["chunks"] == chunks

    save_project(project["project_id"], {"experiment": {"groups": []}}, root=tmp_path)
    assert load_project(project["project_id"], tmp_path)["chunks"] == chunks


def test_loading_legacy_inline_chunks_migrates_to_compressed_sidecar(tmp_path) -> None:
    from manual_graphrag.storage import write_json

    project = create_project("舊 chunks 格式", tmp_path)
    project_file = tmp_path / project["project_id"] / "project.json"
    chunks = [{"number": 1, "text": "舊格式原文"}]
    payload = json.loads(project_file.read_text(encoding="utf-8"))
    payload["chunks"] = chunks
    payload["graph_state"] = {"run_id": "legacy-run", "chunks": chunks}
    write_json(project_file, payload)

    metadata = load_project_metadata(project["project_id"], tmp_path)
    migrated = json.loads(project_file.read_text(encoding="utf-8"))

    assert "chunks" not in migrated
    assert "chunks" not in migrated["graph_state"]
    assert metadata["chunk_count"] == 1
    assert load_project(project["project_id"], tmp_path)["chunks"] == chunks
    assert load_project(project["project_id"], tmp_path)["graph_state"]["chunks"] == chunks


def test_loading_legacy_inline_graph_migrates_it_to_sidecar(tmp_path) -> None:
    project = create_project("舊圖譜格式", tmp_path)
    project_file = tmp_path / project["project_id"] / "project.json"
    legacy_graph = {
        "run_id": "legacy-run",
        "entities": [{"name": "legacy entity"}],
        "relationships": [{"source": "legacy entity", "target": "B"}],
    }
    payload = json.loads(project_file.read_text(encoding="utf-8"))
    payload["graph_state"] = legacy_graph
    project_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    metadata = load_project_metadata(project["project_id"], tmp_path)
    migrated_file = json.loads(project_file.read_text(encoding="utf-8"))

    assert "entities" not in migrated_file["graph_state"]
    assert "relationships" not in migrated_file["graph_state"]
    assert (tmp_path / project["project_id"] / "graph.json.gz").is_file()
    assert metadata["graph_state"]["entity_count"] == 1
    assert load_project(project["project_id"], tmp_path)["graph_state"]["entities"] == legacy_graph["entities"]


def test_each_project_gets_a_stable_unique_database(tmp_path) -> None:
    first = create_project("Model A", tmp_path)
    second = create_project("Model B", tmp_path)

    assert first["neo4j_database"] == load_project(first["project_id"], tmp_path)["neo4j_database"]
    assert first["neo4j_database"] != second["neo4j_database"]


def test_project_activity_log_records_actor_local_time_and_operation(tmp_path) -> None:
    project = create_project("61 | Manual", tmp_path, actor="61")
    with actor_context("58", "save_project_for_ui"):
        save_project(project["project_id"], {"settings": {"top_k": 8}}, root=tmp_path)

    events = [
        json.loads(line)
        for line in (tmp_path / project["project_id"] / "activity.log").read_text().splitlines()
    ]

    assert [event["user"] for event in events] == ["61", "58"]
    assert events[1]["action"] == "save_project_for_ui"
    assert events[1]["details"]["fields"] == ["settings"]
    assert all(datetime.fromisoformat(event["timestamp"]).utcoffset() is not None for event in events)


def test_save_project_copies_documents(tmp_path) -> None:
    source = tmp_path / "manual.pdf"
    source.write_bytes(b"pdf")
    root = tmp_path / "projects"
    project = create_project("Manual", root)

    saved = save_project(project["project_id"], {}, [str(source)], root)

    stored = Path(saved["documents"][0]["path"])
    assert stored.read_bytes() == b"pdf"
    assert stored.parent.name == "documents"


def test_save_project_accumulates_multiple_documents(tmp_path) -> None:
    first = tmp_path / "a.pdf"
    first.write_bytes(b"a")
    second = tmp_path / "b.pdf"
    second.write_bytes(b"b")
    root = tmp_path / "projects"
    project = create_project("Multi", root)

    save_project(project["project_id"], {}, [str(first)], root)
    saved = save_project(project["project_id"], {}, [str(second)], root)

    assert [doc["name"] for doc in saved["documents"]] == ["a.pdf", "b.pdf"]


def test_remove_document_deletes_file_and_entry(tmp_path) -> None:
    first = tmp_path / "a.pdf"
    first.write_bytes(b"a")
    second = tmp_path / "b.pdf"
    second.write_bytes(b"b")
    root = tmp_path / "projects"
    project = create_project("Multi", root)
    save_project(project["project_id"], {}, [str(first), str(second)], root)

    updated = remove_document(project["project_id"], "a.pdf", root)

    assert [doc["name"] for doc in updated["documents"]] == ["b.pdf"]
    assert not (root / project["project_id"] / "documents" / "a.pdf").exists()


def test_document_pages_are_saved_per_document_and_removed_with_it(tmp_path) -> None:
    from manual_graphrag.project_store import load_document_pages, save_document_pages

    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"a")
    root = tmp_path / "projects"
    project_id = create_project("Pages", root)["project_id"]
    save_project(project_id, {}, [str(pdf)], root)
    assert load_document_pages(project_id, root) == {}

    record = {"pages": [{"page": 1, "text": "中文", "source": "vision"}], "chunked": False}
    save_document_pages(project_id, "a.pdf", record, root)
    save_document_pages(project_id, "b.pdf", {"pages": [], "chunked": True}, root)
    assert load_document_pages(project_id, root)["a.pdf"] == record
    assert (root / project_id / "pages.json.gz").is_file()

    remove_document(project_id, "a.pdf", root)
    assert list(load_document_pages(project_id, root)) == ["b.pdf"]
    with pytest.raises(ValueError, match="找不到專案"):
        save_document_pages("missing", "a.pdf", record, root)


def test_load_project_migrates_legacy_single_document(tmp_path) -> None:
    root = tmp_path / "projects"
    project = create_project("Legacy", root)
    from manual_graphrag.storage import write_json

    project_path = root / project["project_id"] / "project.json"
    legacy = {
        **project,
        "document": {"name": "old.pdf", "path": "old.pdf"},
        "preview_state": {"file_name": "old.pdf"},
    }
    legacy.pop("documents", None)
    legacy.pop("documents_meta", None)
    write_json(project_path, legacy)

    loaded = load_project(project["project_id"], root)

    assert loaded["documents"] == [{"name": "old.pdf", "path": "old.pdf"}]
    assert loaded["documents_meta"] == [{"file_name": "old.pdf"}]


def test_append_question_preserves_history(tmp_path) -> None:
    project = create_project("QA", tmp_path)
    append_question(project["project_id"], {"question": "Q1", "answer": "A1"}, tmp_path)
    updated = append_question(
        project["project_id"], {"question": "Q2", "answer": "A2"}, tmp_path
    )

    assert [item["question"] for item in updated["questions"]] == ["Q1", "Q2"]
    assert all(item["asked_at"] for item in updated["questions"])


def test_create_project_validates_name_and_collision(tmp_path) -> None:
    with pytest.raises(ValueError, match="專案名稱"):
        create_project("  ", tmp_path)
    create_project("same", tmp_path)
    with pytest.raises(ValueError, match="相同識別碼"):
        create_project("same", tmp_path)


def test_delete_project_removes_only_selected_project(tmp_path) -> None:
    first = create_project("first", tmp_path)
    second = create_project("second", tmp_path)

    deleted_name = delete_project(first["project_id"], tmp_path)

    assert deleted_name == "first"
    assert not (tmp_path / first["project_id"]).exists()
    assert load_project(second["project_id"], tmp_path)["name"] == "second"
    with pytest.raises(ValueError, match="找不到"):
        delete_project(first["project_id"], tmp_path)


def test_legacy_shared_graph_requires_reimport_to_project_database(tmp_path) -> None:
    from manual_graphrag.storage import write_json

    project = create_project("Legacy imported", tmp_path)
    path = tmp_path / project["project_id"] / "project.json"
    payload = load_project(project["project_id"], tmp_path)
    old_hash = hashlib.sha256(project["project_id"].encode("utf-8")).hexdigest()[:16]
    payload["neo4j_database"] = f"vehicle_{old_hash}"
    payload["graph_state"] = {"run_id": "old", "neo4j_imported": True}
    write_json(path, payload)

    loaded = load_project(project["project_id"], tmp_path)

    assert loaded["neo4j_database"].startswith("vehicle-")
    assert loaded["graph_state"]["neo4j_imported"] is False
    assert "請重新匯入" in loaded["graph_state"]["neo4j_error"]


def test_project_archive_round_trip_preserves_local_data_and_rebases_paths(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "projects"
    project = create_project("車型 A", root)
    pdf = tmp_path / "manual.pdf"
    pdf.write_bytes(b"pdf bytes")
    saved = save_project(
        project["project_id"],
        {
            "settings": {"neo4j_password": "secret", "answer_model": "gpt-4.1-mini"},
            "documents_meta": [{"file_name": "manual.pdf", "file_path": str(pdf)}],
            "chunks": [{"number": 1, "text": "chunk"}],
            "graph_state": {"run_id": "graph-run", "neo4j_imported": True},
            "questions": [{"question": "Q", "answer": "A"}],
            "evaluation": {"results": [{"correct": True}]},
            "experiment": {"groups": [{"name": "control"}]},
        },
        [str(pdf)],
        root,
    )
    question_set = create_question_set("手冊題", [{"number": 1, "question": "Q", "expected_answer": "A"}])
    set_project_question_set_bindings(
        project["project_id"], [question_set["question_set_id"]], projects_root=root,
    )
    exports_dir = root / project["project_id"] / "exports"
    exports_dir.mkdir()
    (exports_dir / "result.json").write_text('{"result": true}', encoding="utf-8")

    archive = export_project_archive(project["project_id"], root, tmp_path)
    imported = import_project_archive(archive, root)

    assert imported["project_id"] != project["project_id"]
    assert imported["neo4j_database"] != saved["neo4j_database"]
    assert imported["name"] == "車型 A（匯入 2）"
    assert imported["settings"]["neo4j_password"] == "secret"
    assert imported["questions"] == [{"question": "Q", "answer": "A"}]
    assert imported["evaluation"] == {"results": [{"correct": True}]}
    assert imported["experiment"] == {"groups": [{"name": "control"}]}
    assert imported["question_set_ids"] == [question_set["question_set_id"]]
    assert imported["graph_state"]["neo4j_imported"] is False
    assert "不包含外部" in imported["graph_state"]["neo4j_error"]
    restored_pdf = Path(imported["documents"][0]["path"])
    assert restored_pdf.read_bytes() == b"pdf bytes"
    assert Path(imported["documents_meta"][0]["file_path"]) == restored_pdf
    assert (root / imported["project_id"] / "exports" / "result.json").read_text(encoding="utf-8") == '{"result": true}'


def test_import_project_archive_rejects_path_traversal(tmp_path) -> None:
    import json
    import zipfile

    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("manifest.json", json.dumps({"format": "manual-graphrag-project", "archive_version": 1}))
        archive.writestr("project/../outside.txt", "unsafe")

    with pytest.raises(ValueError, match="不安全"):
        import_project_archive(archive_path, tmp_path / "projects")
