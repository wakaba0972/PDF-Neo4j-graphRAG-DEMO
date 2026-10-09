import json
from datetime import datetime

from manual_graphrag.audit import actor_context
from manual_graphrag.experiment_project_store import (
    create_experiment_project,
    delete_experiment_project,
    list_experiment_projects,
    load_experiment_project,
    save_experiment_project,
)


def test_experiment_project_round_trip_and_member_question_sets(tmp_path):
    project = create_experiment_project("比較測試", tmp_path)
    saved = save_experiment_project(project["experiment_project_id"], {
        "members": ["vehicle-a", "vehicle-b"],
        "questions_by_project": {"vehicle-a": [{"question": "Q", "expected_answer": "A"}]},
    }, tmp_path)

    assert saved["members"] == ["vehicle-a", "vehicle-b"]
    assert load_experiment_project(project["experiment_project_id"], tmp_path)["questions_by_project"]["vehicle-a"][0]["question"] == "Q"
    assert list_experiment_projects(tmp_path) == [("比較測試", "比較測試")]


def test_experiment_project_activity_log_tracks_actor_and_local_time(tmp_path):
    project = create_experiment_project("61 | Compare", tmp_path, actor="61")
    with actor_context("57", "save_experiment_project_members_for_ui"):
        save_experiment_project(
            project["experiment_project_id"], {"members": ["vehicle-a"]}, tmp_path,
        )

    events = [
        json.loads(line)
        for line in (
            tmp_path / project["experiment_project_id"] / "activity.log"
        ).read_text().splitlines()
    ]

    assert [event["user"] for event in events] == ["61", "57"]
    assert events[1]["action"] == "save_experiment_project_members_for_ui"
    assert events[1]["details"]["fields"] == ["members"]
    assert all(datetime.fromisoformat(event["timestamp"]).utcoffset() is not None for event in events)


def test_delete_experiment_project_removes_only_selected_project(tmp_path):
    first = create_experiment_project("第一個", tmp_path)
    second = create_experiment_project("第二個", tmp_path)

    assert delete_experiment_project(first["experiment_project_id"], tmp_path) == "第一個"
    assert list_experiment_projects(tmp_path) == [("第二個", "第二個")]
    assert load_experiment_project(second["experiment_project_id"], tmp_path)["members"] == []
