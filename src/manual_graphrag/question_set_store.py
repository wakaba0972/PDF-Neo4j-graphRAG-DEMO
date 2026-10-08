from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from .project_store import PROJECTS_DIR, load_project, list_projects, save_project
from .storage import read_json, write_json

QUESTION_SETS_DIR = Path("data/question_sets")


def _question_set_path(question_set_id: str, root: str | Path) -> Path:
    if not question_set_id or Path(question_set_id).name != question_set_id:
        raise ValueError("請選擇有效的題目集")
    return Path(root) / f"{question_set_id}.json"


def list_question_sets(root: str | Path = QUESTION_SETS_DIR) -> list[dict[str, Any]]:
    sets: list[dict[str, Any]] = []
    base = Path(root)
    if not base.exists():
        return sets
    for path in base.glob("*.json"):
        try:
            item = read_json(path)
            if item.get("question_set_id") and isinstance(item.get("questions"), list):
                sets.append(item)
        except (OSError, ValueError, TypeError):
            continue
    return sorted(sets, key=lambda item: str(item.get("name", "")).casefold())


def get_question_set(question_set_id: str, root: str | Path = QUESTION_SETS_DIR) -> dict[str, Any]:
    path = _question_set_path(question_set_id, root)
    if not path.is_file():
        raise ValueError("找不到指定題目集")
    return read_json(path)


def create_question_set(
    name: str, questions: list[dict[str, Any]], *, source_file: str = "",
    root: str | Path = QUESTION_SETS_DIR, question_set_id: str | None = None,
) -> dict[str, Any]:
    clean_name = str(name).strip()
    if not clean_name:
        raise ValueError("題目集名稱不可空白")
    if not questions:
        raise ValueError("題目集至少需要一題")
    identifier = question_set_id or uuid4().hex
    item = {
        "question_set_id": identifier,
        "name": clean_name,
        "source_file": str(source_file or ""),
        "questions": questions,
    }
    write_json(_question_set_path(identifier, root), item)
    return item


def delete_question_set(question_set_id: str, root: str | Path = QUESTION_SETS_DIR) -> None:
    _question_set_path(question_set_id, root).unlink(missing_ok=True)


def bound_question_sets(
    project: dict[str, Any], root: str | Path = QUESTION_SETS_DIR,
) -> list[dict[str, Any]]:
    result = []
    for identifier in project.get("question_set_ids") or []:
        try:
            result.append(get_question_set(str(identifier), root))
        except ValueError:
            continue
    return result


def set_project_question_set_bindings(
    project_id: str, question_set_ids: list[str], *,
    projects_root: str | Path = PROJECTS_DIR,
    question_sets_root: str | Path = QUESTION_SETS_DIR,
) -> dict[str, Any]:
    # Validate all IDs before changing project state.
    available = {item["question_set_id"] for item in list_question_sets(question_sets_root)}
    selected = list(dict.fromkeys(str(item) for item in question_set_ids))
    missing = [item for item in selected if item not in available]
    if missing:
        raise ValueError(f"找不到題目集：{', '.join(missing)}")
    return save_project(project_id, {"question_set_ids": selected}, root=projects_root)


def question_set_binding_projects(
    question_set_id: str, projects_root: str | Path = PROJECTS_DIR,
) -> list[tuple[str, str]]:
    bound = []
    for name, project_id in list_projects(projects_root):
        try:
            project = load_project(project_id, projects_root)
        except (OSError, ValueError):
            continue
        if question_set_id in (project.get("question_set_ids") or []):
            bound.append((name, project_id))
    return bound


def _fingerprint(questions: list[dict[str, Any]]) -> str:
    serialized = json.dumps(questions, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def migrate_project_question_sets(
    project_id: str, *, projects_root: str | Path = PROJECTS_DIR,
    question_sets_root: str | Path = QUESTION_SETS_DIR,
) -> list[dict[str, Any]]:
    """Move legacy per-project question sets into the central library, idempotently.

    Original project data is retained as a migration backup. New reads use the
    canonical question_set_ids binding.
    """
    project = load_project(project_id, projects_root)
    if project.get("question_sets_migrated"):
        return bound_question_sets(project, question_sets_root)

    legacy_sets = list(project.get("question_sets") or [])
    evaluation = project.get("evaluation") or {}
    preferences = evaluation.get("preferences") or {}
    legacy_questions = evaluation.get("questions") or []
    if legacy_questions and not preferences.get("generation_model"):
        legacy_sets.append({
            "question_set_id": "legacy-imported",
            "name": "舊版匯入題目集",
            "source_file": "",
            "questions": legacy_questions,
        })

    central = list_question_sets(question_sets_root)
    fingerprints = {_fingerprint(item.get("questions") or []): item for item in central}
    identifiers: list[str] = []
    for index, legacy in enumerate(legacy_sets):
        if not isinstance(legacy, dict) or not legacy.get("questions"):
            continue
        questions = list(legacy["questions"])
        fingerprint = _fingerprint(questions)
        existing = fingerprints.get(fingerprint)
        if existing is None:
            old_id = str(legacy.get("question_set_id") or f"legacy-{index}")
            candidate_id = f"migrated-{project_id}-{old_id}"
            # IDs become filenames; hash the potentially user-controlled legacy ID.
            safe_id = "migrated-" + hashlib.sha256(candidate_id.encode()).hexdigest()[:24]
            existing = create_question_set(
                str(legacy.get("name") or f"{project.get('name', project_id)} 題目集 {index + 1}"),
                questions, source_file=str(legacy.get("source_file") or ""),
                root=question_sets_root, question_set_id=safe_id,
            )
            fingerprints[fingerprint] = existing
        if existing["question_set_id"] not in identifiers:
            identifiers.append(existing["question_set_id"])
    project["question_set_ids"] = identifiers
    project["question_sets_migrated"] = True
    save_project(project_id, {
        "question_set_ids": identifiers,
        "question_sets_migrated": True,
    }, root=projects_root)
    return bound_question_sets(project, question_sets_root)


def migrate_all_project_question_sets(
    projects_root: str | Path = PROJECTS_DIR,
    question_sets_root: str | Path = QUESTION_SETS_DIR,
) -> int:
    migrated = 0
    for _name, project_id in list_projects(projects_root):
        project = load_project(project_id, projects_root)
        if not project.get("question_sets_migrated"):
            migrate_project_question_sets(
                project_id, projects_root=projects_root,
                question_sets_root=question_sets_root,
            )
            migrated += 1
    return migrated
