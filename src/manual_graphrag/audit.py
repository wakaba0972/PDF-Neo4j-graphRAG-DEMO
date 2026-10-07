from __future__ import annotations

import json
import inspect
from functools import partial, wraps
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, get_type_hints


_CURRENT_ACTOR: ContextVar[str | None] = ContextVar("manual_graphrag_actor", default=None)
_CURRENT_ACTION: ContextVar[str | None] = ContextVar("manual_graphrag_action", default=None)
_AUDIT_EVENTS_IN_CALLBACK: ContextVar[int] = ContextVar("manual_graphrag_audit_count", default=0)
_AUDIT_LOCK = Lock()


def current_actor() -> str | None:
    return _CURRENT_ACTOR.get()


def current_action() -> str | None:
    return _CURRENT_ACTION.get()


class actor_context:
    """Set the user identity for one Gradio callback/request."""

    def __init__(self, actor: str | None, action: str | None = None) -> None:
        self.actor = actor.strip() if isinstance(actor, str) and actor.strip() else None
        self.action = action
        self._actor_token = None
        self._action_token = None
        self._count_token = None

    def __enter__(self):
        self._actor_token = _CURRENT_ACTOR.set(self.actor)
        self._action_token = _CURRENT_ACTION.set(self.action)
        self._count_token = _AUDIT_EVENTS_IN_CALLBACK.set(0)
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._action_token is not None:
            _CURRENT_ACTION.reset(self._action_token)
        if self._count_token is not None:
            _AUDIT_EVENTS_IN_CALLBACK.reset(self._count_token)
        if self._actor_token is not None:
            _CURRENT_ACTOR.reset(self._actor_token)


def bind_gradio_callbacks_to_actor(app: Any, actor_state: Any) -> None:
    """Append per-session user state to callbacks and scope it as audit context."""
    dependencies = app.config.get("dependencies", [])
    bound_callbacks: set[int] = set()
    bound_block_functions: set[int] = set()
    for dependency in dependencies:
        function_id = dependency.get("id")
        block_function = app.fns.get(function_id)
        if block_function is None or block_function.fn is None:
            continue
        dependency["inputs"].append(actor_state._id)
        original = block_function.fn
        callback_identity = id(original)
        if callback_identity in bound_callbacks:
            continue
        bound_callbacks.add(callback_identity)
        block_function_identity = id(block_function)
        if block_function_identity not in bound_block_functions:
            block_function.inputs.append(actor_state)
            bound_block_functions.add(block_function_identity)
        annotation_source = original
        while isinstance(annotation_source, partial):
            annotation_source = annotation_source.func
        original_signature = inspect.signature(original)
        parameters = list(original_signature.parameters.values())
        actor_parameter = inspect.Parameter(
            "_audit_actor", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None,
        )
        varargs_index = next(
            (index for index, parameter in enumerate(parameters)
             if parameter.kind is inspect.Parameter.VAR_POSITIONAL),
            None,
        )
        if varargs_index is None:
            parameters.append(actor_parameter)
        else:
            parameters.insert(varargs_index, actor_parameter)
        try:
            wrapped_signature = original_signature.replace(parameters=parameters)
        except ValueError:
            wrapped_signature = original_signature

        @wraps(original)
        def wrapped(
            *args: Any, __original=original,
            __action=block_function.name, **kwargs: Any,
        ) -> Any:
            actor = args[-1] if args else None
            callback_inputs = args[:-1]
            with actor_context(actor, __action):
                try:
                    result = __original(*callback_inputs, **kwargs)
                except Exception as exc:
                    if _AUDIT_EVENTS_IN_CALLBACK.get() == 0:
                        _record_callback_event(
                            __action, callback_inputs,
                            {"outcome": "failed", "error_type": type(exc).__name__},
                        )
                    raise
                if _AUDIT_EVENTS_IN_CALLBACK.get() == 0:
                    _record_callback_event(
                        __action, callback_inputs, {"outcome": "completed"},
                    )
                return result

        wrapped.__signature__ = wrapped_signature
        wrapped.__module__ = getattr(annotation_source, "__module__", wrapped.__module__)
        try:
            wrapped.__annotations__ = get_type_hints(annotation_source)
        except (NameError, TypeError):
            pass
        block_function.fn = wrapped


def _record_callback_event(
    action: str, values: tuple[Any, ...], details: dict[str, Any],
) -> None:
    from .project_store import PROJECTS_DIR
    from .experiment_project_store import EXPERIMENT_PROJECTS_DIR

    project_ids: set[str] = set()
    experiment_ids: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            project_id = value.get("project_id")
            experiment_id = value.get("experiment_project_id")
            if isinstance(project_id, str):
                project_ids.add(project_id)
            if isinstance(experiment_id, str):
                experiment_ids.add(experiment_id)
            for nested in value.values():
                visit(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                visit(nested)
        elif isinstance(value, str):
            candidate = Path(value)
            if candidate.name == value and (Path(PROJECTS_DIR) / value / "project.json").is_file():
                project_ids.add(value)
            if candidate.name == value and (
                Path(EXPERIMENT_PROJECTS_DIR) / value / "experiment.json"
            ).is_file():
                experiment_ids.add(value)

    for value in values:
        visit(value)
    for project_id in project_ids:
        append_audit_event(
            Path(PROJECTS_DIR) / project_id / "activity.log", action, details=details,
        )
    for project_id in experiment_ids:
        append_audit_event(
            Path(EXPERIMENT_PROJECTS_DIR) / project_id / "activity.log", action,
            details=details,
        )


def append_audit_event(
    path: str | Path,
    action: str,
    *,
    actor: str | None = None,
    details: dict[str, Any] | None = None,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "user": actor or current_actor() or "未選擇使用者",
        "action": action,
    }
    if details:
        event["details"] = details
    with _AUDIT_LOCK, target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        handle.flush()
    _AUDIT_EVENTS_IN_CALLBACK.set(_AUDIT_EVENTS_IN_CALLBACK.get() + 1)
    return target
