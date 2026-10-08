from __future__ import annotations

import pytest

from manual_graphrag import question_set_store as store
from manual_graphrag.project_store import create_project, load_project, save_project


def test_question_sets_are_central_and_project_bindings_are_ids(tmp_path):
    project = create_project("binding-project", tmp_path / "projects")
    item = store.create_question_set(
        "手冊題", [{"number": 1, "question": "Q", "expected_answer": "A"}],
        root=tmp_path / "sets",
    )

    saved = store.set_project_question_set_bindings(
        project["project_id"], [item["question_set_id"]],
        projects_root=tmp_path / "projects", question_sets_root=tmp_path / "sets",
    )

    assert saved["question_set_ids"] == [item["question_set_id"]]
    assert store.bound_question_sets(saved, tmp_path / "sets")[0]["question_set_id"] == item["question_set_id"]
    with pytest.raises(ValueError, match="找不到題目集"):
        store.set_project_question_set_bindings(
            project["project_id"], ["missing"],
            projects_root=tmp_path / "projects", question_sets_root=tmp_path / "sets",
        )


def test_migration_moves_legacy_question_sets_idempotently_and_keeps_backup(tmp_path):
    projects = tmp_path / "projects"
    sets = tmp_path / "sets"
    project = create_project("legacy", projects)
    legacy = [{"number": 1, "question": "Q", "expected_answer": "A"}]
    save_project(project["project_id"], {"question_sets": [
        {"question_set_id": "old-id", "name": "舊題集", "questions": legacy},
    ]}, root=projects)

    first = store.migrate_project_question_sets(
        project["project_id"], projects_root=projects, question_sets_root=sets,
    )
    second = store.migrate_project_question_sets(
        project["project_id"], projects_root=projects, question_sets_root=sets,
    )

    persisted = load_project(project["project_id"], projects)
    assert first == second
    assert len(store.list_question_sets(sets)) == 1
    assert persisted["question_sets"][0]["questions"] == legacy
    assert persisted["question_sets_migrated"] is True
    assert persisted["question_set_ids"] == [first[0]["question_set_id"]]


def test_migration_imports_legacy_manual_questions_but_not_generated_questions(tmp_path):
    project = create_project("legacy-evaluation", tmp_path / "projects")
    questions = [{"number": 1, "question": "Imported", "expected_answer": "A"}]
    save_project(project["project_id"], {"evaluation": {"questions": questions}}, root=tmp_path / "projects")
    imported = store.migrate_project_question_sets(
        project["project_id"], projects_root=tmp_path / "projects", question_sets_root=tmp_path / "sets",
    )
    assert imported[0]["questions"] == questions

    generated = create_project("generated", tmp_path / "projects")
    save_project(generated["project_id"], {"evaluation": {
        "questions": questions, "preferences": {"generation_model": "gpt-test"},
    }}, root=tmp_path / "projects")
    assert store.migrate_project_question_sets(
        generated["project_id"], projects_root=tmp_path / "projects", question_sets_root=tmp_path / "sets",
    ) == []


def test_bound_question_set_deletion_can_be_checked_across_projects(tmp_path):
    project = create_project("bound", tmp_path / "projects")
    item = store.create_question_set("Q", [{"question": "Q"}], root=tmp_path / "sets")
    store.set_project_question_set_bindings(
        project["project_id"], [item["question_set_id"]],
        projects_root=tmp_path / "projects", question_sets_root=tmp_path / "sets",
    )
    assert store.question_set_binding_projects(item["question_set_id"], tmp_path / "projects") == [
        (project["name"], project["project_id"]),
    ]
