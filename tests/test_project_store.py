from pathlib import Path
import hashlib
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
    remove_document,
    save_project,
)


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


def test_project_name_can_be_updated_without_changing_base_name(tmp_path) -> None:
    project = create_project("Zhao | s25", tmp_path)

    saved = save_project(
        project["project_id"], {"name": "Zhao | s25 | 1500-200-None"}, root=tmp_path
    )

    assert saved["name"] == "Zhao | s25 | 1500-200-None"
    assert saved["base_name"] == "Zhao | s25"
    assert list_projects(tmp_path) == [("Zhao | s25 | 1500-200-None", project["project_id"])]


def test_each_project_gets_a_stable_unique_database(tmp_path) -> None:
    first = create_project("Model A", tmp_path)
    second = create_project("Model B", tmp_path)

    assert first["neo4j_database"] == load_project(first["project_id"], tmp_path)["neo4j_database"]
    assert first["neo4j_database"] != second["neo4j_database"]


def test_project_activity_log_records_actor_local_time_and_operation(tmp_path) -> None:
    project = create_project("Zhao | Manual", tmp_path, actor="Zhao")
    with actor_context("Christine", "save_project_for_ui"):
        save_project(project["project_id"], {"settings": {"top_k": 8}}, root=tmp_path)

    events = [
        json.loads(line)
        for line in (tmp_path / project["project_id"] / "activity.log").read_text().splitlines()
    ]

    assert [event["user"] for event in events] == ["Zhao", "Christine"]
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


def test_project_archive_round_trip_preserves_local_data_and_rebases_paths(tmp_path) -> None:
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
