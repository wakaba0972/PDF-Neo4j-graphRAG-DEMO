from __future__ import annotations

import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .audit import append_audit_event, current_action
from .storage import read_json, write_json

EXPERIMENT_PROJECTS_DIR = Path("data/experiment_projects")


def _safe_id(name: str) -> str:
    value = re.sub(r"[^\w\-]+", "-", name.strip(), flags=re.UNICODE).strip("-_ ")
    return value[:60] or f"experiment-{uuid4().hex[:8]}"


def list_experiment_projects(root: str | Path = EXPERIMENT_PROJECTS_DIR) -> list[tuple[str, str]]:
    base = Path(root)
    result = []
    if not base.exists():
        return result
    for path in base.glob("*/experiment.json"):
        try:
            data = read_json(path)
            result.append((str(data.get("name") or path.parent.name), path.parent.name))
        except (OSError, ValueError, TypeError):
            continue
    return sorted(result, key=lambda item: item[0].casefold())


def create_experiment_project(
    name: str, root: str | Path = EXPERIMENT_PROJECTS_DIR, *, actor: str | None = None,
) -> dict[str, Any]:
    clean_name = name.strip()
    if not clean_name:
        raise ValueError("請輸入實驗專案名稱")
    project_id = _safe_id(clean_name)
    directory = Path(root) / project_id
    if directory.exists():
        raise ValueError("已有相同識別碼的實驗專案，請改用其他名稱")
    now = datetime.now(timezone.utc).isoformat()
    data = {
        "experiment_project_id": project_id,
        "name": clean_name,
        "created_at": now,
        "updated_at": now,
        "members": [],
        "groups": [],
        "results": [],
        "summary_rows": [],
        "detail_rows": [],
    }
    write_json(directory / "experiment.json", data)
    append_audit_event(
        directory / "activity.log", current_action() or "experiment_project_created",
        actor=actor,
    )
    return data


def load_experiment_project(
    project_id: str, root: str | Path = EXPERIMENT_PROJECTS_DIR,
) -> dict[str, Any]:
    if not project_id or Path(project_id).name != project_id:
        raise ValueError("請選擇有效的實驗專案")
    path = Path(root) / project_id / "experiment.json"
    if not path.is_file():
        raise ValueError("找不到指定實驗專案")
    data = read_json(path)
    data.setdefault("members", [])
    data.setdefault("groups", [])
    data.setdefault("results", [])
    return data


def save_experiment_project(
    project_id: str, payload: dict[str, Any],
    root: str | Path = EXPERIMENT_PROJECTS_DIR,
) -> dict[str, Any]:
    current = load_experiment_project(project_id, root)
    updated = {
        **current, **payload,
        "experiment_project_id": project_id,
        "name": current["name"],
        "created_at": current["created_at"],
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(Path(root) / project_id / "experiment.json", updated)
    append_audit_event(
        Path(root) / project_id / "activity.log",
        current_action() or "experiment_project_updated",
        details={"fields": sorted(payload)},
    )
    return updated


def delete_experiment_project(
    project_id: str, root: str | Path = EXPERIMENT_PROJECTS_DIR,
) -> str:
    data = load_experiment_project(project_id, root)
    directory = Path(root) / project_id
    append_audit_event(
        directory / "activity.log", current_action() or "experiment_project_deleted",
    )
    shutil.rmtree(directory)
    return str(data["name"])
