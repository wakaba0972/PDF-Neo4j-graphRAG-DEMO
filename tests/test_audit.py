from functools import partial
from typing import Any, get_type_hints

from gradio.utils import get_function_params

from manual_graphrag.audit import (
    bind_gradio_callbacks_to_actor,
    current_actor,
    _record_callback_event,
)


class _ActorState:
    _id = 42


class _BlockFunction:
    def __init__(self):
        self.name = "sample_mutation"
        self.inputs = [object()]

        def callback(value):
            return current_actor(), value

        self.fn = callback


class _App:
    def __init__(self):
        self.config = {"dependencies": [{"id": 0, "inputs": [7]}]}
        self.fns = {0: _BlockFunction()}


def test_gradio_callbacks_receive_actor_in_isolated_context():
    app = _App()
    actor_state = _ActorState()

    bind_gradio_callbacks_to_actor(app, actor_state)

    assert app.fns[0].inputs[-1] is actor_state
    assert app.config["dependencies"][0]["inputs"] == [7, 42]
    assert app.fns[0].fn("payload", "61") == ("61", "payload")
    assert current_actor() is None


def test_shared_gradio_callback_only_receives_one_actor_input():
    app = _App()
    original_function = app.fns[0]
    app.fns[1] = app.fns[0]
    app.config["dependencies"].append({"id": 1, "inputs": [8]})
    actor_state = _ActorState()

    bind_gradio_callbacks_to_actor(app, actor_state)

    assert len(app.fns[0].inputs) == 2
    assert app.fns[0].inputs[-1] is actor_state
    assert [dependency["inputs"] for dependency in app.config["dependencies"]] == [
        [7, 42], [8, 42],
    ]
    assert original_function.fn("payload", "61") == ("61", "payload")


def test_distinct_callbacks_sharing_input_list_only_append_actor_once():
    app = _App()
    first_function = app.fns[0]
    second_function = _BlockFunction()
    second_function.inputs = first_function.inputs
    app.fns[1] = second_function
    app.config["dependencies"].append({"id": 1, "inputs": [8]})
    actor_state = _ActorState()

    bind_gradio_callbacks_to_actor(app, actor_state)

    assert len(first_function.inputs) == 2
    assert first_function.inputs[-1] is actor_state
    assert first_function.fn("first", "61") == ("61", "first")
    assert second_function.fn("second", "61") == ("61", "second")


def test_partial_gradio_callback_keeps_annotation_module_for_api_schema():
    def callback(_action: str, value: Any) -> Any:
        return value

    app = _App()
    app.fns[0].fn = partial(callback, "remove")

    bind_gradio_callbacks_to_actor(app, _ActorState())

    wrapped = app.fns[0].fn
    assert wrapped.__module__ == callback.__module__
    assert get_type_hints(wrapped) == {
        "_action": str, "value": Any, "return": Any,
    }
    assert get_function_params(wrapped)


def test_audit_ignores_long_document_text_when_finding_project_ids(tmp_path, monkeypatch):
    from manual_graphrag import experiment_project_store, project_store

    projects_dir = tmp_path / "projects"
    experiments_dir = tmp_path / "experiments"
    project_dir = projects_dir / "project-a"
    project_dir.mkdir(parents=True)
    (project_dir / "project.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(project_store, "PROJECTS_DIR", projects_dir)
    monkeypatch.setattr(experiment_project_store, "EXPERIMENT_PROJECTS_DIR", experiments_dir)

    long_document_text = "個人履歷內容\n" * 2_000
    _record_callback_event(
        "neo4j_import", (long_document_text, {"project_id": "project-a"}), {},
    )

    audit_log = project_dir / "activity.log"
    assert audit_log.is_file()
    assert '"action": "neo4j_import"' in audit_log.read_text(encoding="utf-8")


def test_audit_does_not_traverse_large_project_payload_after_finding_id(tmp_path, monkeypatch):
    from pathlib import Path
    from manual_graphrag import experiment_project_store, project_store

    projects_dir = tmp_path / "projects"
    experiments_dir = tmp_path / "experiments"
    project_dir = projects_dir / "project-a"
    project_dir.mkdir(parents=True)
    (project_dir / "project.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(project_store, "PROJECTS_DIR", projects_dir)
    monkeypatch.setattr(experiment_project_store, "EXPERIMENT_PROJECTS_DIR", experiments_dir)

    path_probes = []
    original_is_file = Path.is_file

    def count_path_probe(path):
        path_probes.append(path)
        return original_is_file(path)

    monkeypatch.setattr(Path, "is_file", count_path_probe)
    _record_callback_event("project_loaded", ({
        "project_id": "project-a",
        "chunks": [{"text": f"short chunk text {index}"} for index in range(500)],
        "graph_state": {"entities": [{"description": f"entity {index}"} for index in range(500)]},
    },), {})

    assert path_probes == []
    assert (project_dir / "activity.log").is_file()
