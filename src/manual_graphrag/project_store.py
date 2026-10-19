from __future__ import annotations

import re
import hashlib
import gzip
import json
import os
import shutil
import stat
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from pathlib import PurePosixPath
from threading import RLock
from typing import Any
from uuid import uuid4

from .audit import append_audit_event, current_action, current_actor
from .storage import read_json, write_json

PROJECTS_DIR = Path("data/projects")
PROJECT_CONNECTION_SETTINGS = {"model_endpoint", "api_key"}
_PROJECT_WRITE_LOCK = RLock()
PROJECT_ARCHIVE_FORMAT = "manual-graphrag-project"
PROJECT_ARCHIVE_VERSION = 1
GRAPH_DATA_FILENAME = "graph.json.gz"
CHUNK_DATA_FILENAME = "chunks.json.gz"
# Per-page parse results (vision or text layer) and their human review state.
PAGE_DATA_FILENAME = "pages.json.gz"


def project_database_name(project_id: str) -> str:
    """Return a stable, Neo4j-safe database name isolated to this project."""
    digest = hashlib.sha256(project_id.encode("utf-8")).hexdigest()[:16]
    return f"vehicle-{digest}"


def build_project_summary(project: dict[str, Any]) -> dict[str, Any]:
    """Return a compact, human-readable snapshot of the project's build metadata."""
    settings = project.get("settings") or {}
    graph_state = project.get("graph_state") or {}
    build_config = graph_state.get("build_config") or {}
    documents = project.get("documents_meta") or project.get("documents") or []
    document_names = {
        str(item.get("file_name") or item.get("name") or "")
        for item in documents if isinstance(item, dict)
    }
    document_count = len({name for name in document_names if name})
    if not document_count:
        document_count = len(project.get("documents") or [])
    bound_question_count = None
    if "question_set_ids" in project:
        bound_question_count = 0
        for question_set_id in project.get("question_set_ids") or []:
            if not isinstance(question_set_id, str) or Path(question_set_id).name != question_set_id:
                continue
            try:
                question_set = read_json(Path("data/question_sets") / f"{question_set_id}.json")
                bound_question_count += len(question_set.get("questions") or [])
            except (OSError, ValueError, TypeError):
                continue
    return {
        "project_id": project.get("project_id", ""),
        "name": project.get("name", ""),
        "created_by": project.get("created_by") or "未知（舊專案）",
        "document_count": document_count,
        "chunk_count": int(project.get("chunk_count", len(project.get("chunks") or [])) or 0),
        "question_count": bound_question_count if bound_question_count is not None else (
            sum(len(item.get("questions") or []) for item in project.get("question_sets") or [])
            if project.get("question_sets") else
            (
                0 if ((project.get("evaluation") or {}).get("preferences") or {}).get("generation_model")
                else len((project.get("evaluation") or {}).get("questions") or [])
            )
        ),
        "chunk_size": build_config.get("chunk_size", settings.get("chunk_size")),
        "chunk_overlap": build_config.get("chunk_overlap", settings.get("chunk_overlap")),
        "graph_built": bool(graph_state.get("document") or graph_state.get("graph_data_file")),
        "updated_at": project.get("updated_at", ""),
    }


def _write_project_summary(project_dir: Path, project: dict[str, Any]) -> dict[str, Any]:
    summary = build_project_summary(project)
    write_json(project_dir / "summary.json", summary)
    return summary


def load_project_summary(
    project_id: str, root: str | Path = PROJECTS_DIR,
) -> dict[str, Any]:
    if not project_id or Path(project_id).name != project_id:
        raise ValueError("請選擇有效的專案")
    project_dir = Path(root) / project_id
    summary_path = project_dir / "summary.json"
    if summary_path.is_file():
        summary = read_json(summary_path)
        if "graph_partition_mode" not in summary:
            return summary
        # Remove the obsolete per-document graph mode from old summaries.
        project = _read_project_metadata(project_id, root)
        summary = {**summary, **build_project_summary(project)}
        summary.pop("graph_partition_mode", None)
        write_json(summary_path, summary)
        return summary
    # One-time compatibility path for projects created before summary.json
    # existed. Normal summary reads stay small and do not parse all chunks.
    project = _read_project_metadata(project_id, root)
    _write_project_summary(project_dir, project)
    return read_json(summary_path)


def _without_model_credentials(settings: Any) -> Any:
    if not isinstance(settings, dict):
        return settings
    return {key: value for key, value in settings.items() if key not in PROJECT_CONNECTION_SETTINGS}



def _project_id(name: str) -> str:
    normalized = re.sub(r"[^\w\-]+", "-", name.strip(), flags=re.UNICODE).strip("-_")
    return normalized[:60] or f"project-{uuid4().hex[:8]}"


def list_projects(root: str | Path = PROJECTS_DIR) -> list[tuple[str, str]]:
    projects: list[tuple[str, str]] = []
    base = Path(root)
    if not base.exists():
        return projects
    for config_path in base.glob("*/project.json"):
        try:
            summary_path = config_path.parent / "summary.json"
            if summary_path.is_file():
                summary = read_json(summary_path)
                name = summary.get("name")
            else:
                # Migrate older projects once; subsequent listings read only
                # the compact summary file.
                data = read_json(config_path)
                summary = _write_project_summary(config_path.parent, data)
                name = summary.get("name")
            projects.append((str(name or config_path.parent.name), config_path.parent.name))
        except (OSError, ValueError, TypeError):
            continue
    return sorted(projects, key=lambda item: item[0].casefold())


def create_project(
    name: str, root: str | Path = PROJECTS_DIR, *, actor: str | None = None,
) -> dict[str, Any]:
    clean_name = name.strip()
    if not clean_name:
        raise ValueError("請輸入專案名稱")
    base = Path(root)
    project_id = _project_id(clean_name)
    target = base / project_id
    if target.exists():
        raise ValueError("已有相同識別碼的專案，請改用其他名稱")
    now = datetime.now(timezone.utc).isoformat()
    project = {
        "project_id": project_id,
        "neo4j_database": project_database_name(project_id),
        "name": clean_name,
        "base_name": clean_name,
        "created_by": actor or current_actor() or "未知",
        "created_at": now,
        "updated_at": now,
        "settings": {},
        "documents": [],
        "documents_meta": [],
        "graph_state": {},
        "questions": [],
    }
    write_json(target / "project.json", project)
    _write_project_summary(target, project)
    append_audit_event(
        target / "activity.log", current_action() or "project_created", actor=actor,
    )
    return project


def _read_project_metadata(
    project_id: str, root: str | Path = PROJECTS_DIR,
) -> dict[str, Any]:
    if not project_id or Path(project_id).name != project_id:
        raise ValueError("請選擇有效的專案")
    target = Path(root) / project_id / "project.json"
    if not target.is_file():
        raise ValueError("找不到指定專案")
    project = read_json(target)
    stored_database = str(project.get("neo4j_database") or "")
    legacy_generated_database = stored_database.startswith("vehicle_")
    had_project_database = bool(stored_database) and not legacy_generated_database
    if not had_project_database:
        project["neo4j_database"] = project_database_name(project_id)
    if not had_project_database:
        graph_state = project.get("graph_state")
        if isinstance(graph_state, dict) and graph_state.get("neo4j_imported"):
            graph_state["neo4j_imported"] = False
            graph_state["neo4j_error"] = "已切換為專案專屬 Neo4j Database，請重新匯入此專案的圖譜。"
    if "documents" not in project:
        legacy_document = project.pop("document", None)
        project["documents"] = [legacy_document] if legacy_document else []
    if "documents_meta" not in project:
        legacy_preview = project.pop("preview_state", None)
        project["documents_meta"] = [legacy_preview] if legacy_preview else []
    project["settings"] = _without_model_credentials(project.get("settings", {}))
    return project


def _write_gzip_json(project_dir: Path, filename: str, payload: dict[str, Any]) -> None:
    target = project_dir / filename
    descriptor, temporary_name = tempfile.mkstemp(
        dir=project_dir, prefix=f".{filename}.", suffix=".tmp",
    )
    os.close(descriptor)
    try:
        with gzip.open(temporary_name, "wt", encoding="utf-8", compresslevel=6) as handle:
            json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
        with open(temporary_name, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _write_graph_data(project_dir: Path, graph_payload: dict[str, Any]) -> None:
    _write_gzip_json(project_dir, GRAPH_DATA_FILENAME, graph_payload)


def _write_chunk_data(project_dir: Path, chunk_payload: dict[str, Any]) -> None:
    _write_gzip_json(project_dir, CHUNK_DATA_FILENAME, chunk_payload)


def _read_graph_data(project_dir: Path, graph_state: dict[str, Any]) -> dict[str, Any]:
    filename = str(graph_state.get("graph_data_file") or "")
    if not filename:
        return {}
    if Path(filename).name != filename:
        raise ValueError("專案圖譜檔案路徑無效")
    target = project_dir / filename
    if not target.is_file():
        raise ValueError(f"找不到專案圖譜檔：{filename}")
    with gzip.open(target, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("專案圖譜檔案格式錯誤")
    return payload


def _read_chunk_data(project_dir: Path, project: dict[str, Any]) -> dict[str, Any]:
    filename = str(project.get("chunks_data_file") or "")
    if not filename:
        return {}
    if Path(filename).name != filename:
        raise ValueError("專案 chunks 檔案路徑無效")
    target = project_dir / filename
    if not target.is_file():
        raise ValueError(f"找不到專案 chunks 檔：{filename}")
    with gzip.open(target, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("chunks", []), list):
        raise ValueError("專案 chunks 檔案格式錯誤")
    return payload


def load_project_metadata(
    project_id: str, root: str | Path = PROJECTS_DIR,
) -> dict[str, Any]:
    """Load project metadata without materializing chunks or graph entities/edges."""
    with _PROJECT_WRITE_LOCK:
        project = _read_project_metadata(project_id, root)
        graph_state = project.get("graph_state")
        if not isinstance(graph_state, dict):
            graph_state = {}
            project["graph_state"] = graph_state
        project_dir = Path(root) / project_id
        changed = False
        settings = project.get("settings")
        if isinstance(settings, dict) and settings.pop("graph_partition_mode", None) is not None:
            changed = True
        if isinstance(settings, dict):
            for obsolete_key in ("schema_granularity", "schema_text"):
                if obsolete_key in settings:
                    settings.pop(obsolete_key, None)
                    changed = True
        if graph_state.pop("graph_partition_mode", None) is not None:
            changed = True
        for obsolete_key in ("schema", "schema_json", "schema_granularity"):
            if obsolete_key in graph_state:
                graph_state.pop(obsolete_key, None)
                changed = True
        if "run_id" in graph_state:
            graph_state.pop("run_id", None)
            changed = True
        has_inline_graph = "entities" in graph_state or "relationships" in graph_state
        if has_inline_graph and not graph_state.get("graph_data_file"):
            graph_payload = {
                "entities": graph_state.pop("entities", []),
                "relationships": graph_state.pop("relationships", []),
            }
            _write_graph_data(project_dir, graph_payload)
            graph_state["graph_data_file"] = GRAPH_DATA_FILENAME
            graph_state["entity_count"] = len(graph_payload["entities"])
            graph_state["relationship_count"] = len(graph_payload["relationships"])
            changed = True
        inline_chunks = project.pop("chunks", None)
        if (inline_chunks is not None or "chunks" in graph_state) and not project.get("chunks_data_file"):
            inline_chunks = inline_chunks or []
            graph_chunks = graph_state.pop("chunks", None)
            payload = {"chunks": inline_chunks}
            if graph_chunks is not None and graph_chunks != inline_chunks:
                payload["graph_chunks"] = graph_chunks
                graph_state["graph_chunks_in_sidecar"] = True
            elif graph_chunks is not None:
                graph_state["graph_chunks_from_project"] = True
            _write_chunk_data(project_dir, payload)
            project["chunks_data_file"] = CHUNK_DATA_FILENAME
            project["chunk_count"] = len(inline_chunks)
            changed = True
        if changed:
            write_json(project_dir / "project.json", project)
            _write_project_summary(project_dir, project)
        return project


def load_project(project_id: str, root: str | Path = PROJECTS_DIR) -> dict[str, Any]:
    """Load full project content, including compressed graph and chunk sidecars."""
    project = load_project_metadata(project_id, root)
    chunk_payload = _read_chunk_data(Path(root) / project_id, project)
    if chunk_payload:
        project["chunks"] = chunk_payload.get("chunks", [])
    else:
        project["chunks"] = []
    graph_state = project.get("graph_state")
    if isinstance(graph_state, dict):
        graph_state.update(_read_graph_data(Path(root) / project_id, graph_state))
        if graph_state.pop("graph_chunks_from_project", False):
            graph_state["chunks"] = project.get("chunks", [])
        elif graph_state.pop("graph_chunks_in_sidecar", False):
            graph_state["chunks"] = chunk_payload.get("graph_chunks", [])
    return project


def load_project_chunks(
    project_id: str, root: str | Path = PROJECTS_DIR,
) -> list[dict[str, Any]]:
    """Load only chunk text, without decompressing potentially large graph data."""
    project = load_project_metadata(project_id, root)
    return _read_chunk_data(Path(root) / project_id, project).get("chunks", [])


def load_document_pages(
    project_id: str, root: str | Path = PROJECTS_DIR,
) -> dict[str, dict[str, Any]]:
    """Per-document page records: ``{file_name: {"pages": [...], "chunked": bool}}``.

    Documents parsed before page records existed are simply absent.
    """
    target = Path(root) / project_id / PAGE_DATA_FILENAME
    if not target.is_file():
        return {}
    with gzip.open(target, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    documents = payload.get("documents") if isinstance(payload, dict) else None
    if not isinstance(documents, dict):
        raise ValueError("專案頁面解析檔案格式錯誤")
    return documents


def save_document_pages(
    project_id: str,
    file_name: str,
    record: dict[str, Any] | None,
    root: str | Path = PROJECTS_DIR,
) -> None:
    """Replace one document's page records; ``None`` removes the document."""
    with _PROJECT_WRITE_LOCK:
        project_dir = Path(root) / project_id
        if not (project_dir / "project.json").is_file():
            raise ValueError("找不到專案")
        documents = load_document_pages(project_id, root)
        if record is None:
            if documents.pop(file_name, None) is None:
                return
        else:
            documents[file_name] = record
        _write_gzip_json(project_dir, PAGE_DATA_FILENAME, {"documents": documents})


def delete_project(project_id: str, root: str | Path = PROJECTS_DIR) -> str:
    project = _read_project_metadata(project_id, root)
    target = Path(root) / project_id
    append_audit_event(target / "activity.log", current_action() or "project_deleted")
    shutil.rmtree(target)
    return str(project["name"])


def save_project(
    project_id: str,
    payload: dict[str, Any],
    document_paths: list[str] | None = None,
    root: str | Path = PROJECTS_DIR,
) -> dict[str, Any]:
    with _PROJECT_WRITE_LOCK:
        saved = _save_project_unlocked(project_id, payload, document_paths, root)
        append_audit_event(
            Path(root) / project_id / "activity.log",
            current_action() or "project_updated",
            details={"fields": sorted(payload)},
        )
        return saved


def export_project_archive(
    project_id: str,
    root: str | Path = PROJECTS_DIR,
    output_dir: str | Path | None = None,
) -> Path:
    """Package all files stored locally for one project into a portable ZIP."""
    if not project_id or Path(project_id).name != project_id:
        raise ValueError("請選擇有效的專案")
    project_dir = Path(root) / project_id
    project = load_project_metadata(project_id, root)
    destination_dir = Path(output_dir) if output_dir else Path(tempfile.gettempdir())
    destination_dir.mkdir(parents=True, exist_ok=True)
    fd, archive_name = tempfile.mkstemp(
        prefix=f"{project_id}-project-", suffix=".zip", dir=destination_dir
    )
    os.close(fd)
    try:
        with zipfile.ZipFile(archive_name, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            manifest = {
                "format": PROJECT_ARCHIVE_FORMAT,
                "archive_version": PROJECT_ARCHIVE_VERSION,
                "project_name": project.get("name", project_id),
                "neo4j_database_included": False,
                "question_set_ids": list(project.get("question_set_ids") or []),
            }
            archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
            for path in sorted(project_dir.rglob("*")):
                if path.is_symlink():
                    raise ValueError("專案資料夾包含不支援匯出的符號連結")
                if path.is_file():
                    archive.write(path, Path("project") / path.relative_to(project_dir))
            for question_set_id in project.get("question_set_ids") or []:
                if not isinstance(question_set_id, str) or Path(question_set_id).name != question_set_id:
                    continue
                question_set_path = Path("data/question_sets") / f"{question_set_id}.json"
                if question_set_path.is_file():
                    archive.write(question_set_path, Path("project/central_question_sets") / question_set_path.name)
        append_audit_event(
            project_dir / "activity.log", current_action() or "project_exported",
        )
        return Path(archive_name)
    except Exception:
        Path(archive_name).unlink(missing_ok=True)
        raise


def import_project_archive(
    archive_path: str | Path,
    root: str | Path = PROJECTS_DIR,
) -> dict[str, Any]:
    """Safely restore a project archive under a new, unique project identity."""
    archive_path = Path(archive_path)
    if not archive_path.is_file():
        raise ValueError("找不到匯入的專案封裝檔")
    base = Path(root)
    base.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        if len(members) > 20_000 or sum(item.file_size for item in members) > 5 * 1024**3:
            raise ValueError("專案封裝檔超出允許的大小")
        try:
            manifest = json.loads(archive.read("manifest.json"))
        except (KeyError, json.JSONDecodeError) as exc:
            raise ValueError("不是有效的專案封裝檔（缺少 manifest.json）") from exc
        if not isinstance(manifest, dict):
            raise ValueError("專案封裝檔的 manifest 格式錯誤")
        if manifest.get("format") != PROJECT_ARCHIVE_FORMAT or manifest.get("archive_version") != PROJECT_ARCHIVE_VERSION:
            raise ValueError("不支援此專案封裝格式或版本")
        safe_members: list[tuple[zipfile.ZipInfo, PurePosixPath]] = []
        for member in members:
            if member.filename == "manifest.json":
                continue
            relative = PurePosixPath(member.filename)
            mode = member.external_attr >> 16
            if (relative.is_absolute() or ".." in relative.parts or "\\" in member.filename
                    or not relative.parts or relative.parts[0] != "project"
                    or stat.S_ISLNK(mode)):
                raise ValueError("專案封裝檔包含不安全的檔案路徑")
            safe_members.append((member, relative))
        if not any(str(path) == "project/project.json" for _, path in safe_members):
            raise ValueError("專案封裝檔缺少 project.json")

        project_payload = json.loads(archive.read("project/project.json"))
        if not isinstance(project_payload, dict):
            raise ValueError("專案封裝檔中的 project.json 格式錯誤")
        original_name = str(project_payload.get("name") or manifest.get("project_name") or "匯入專案").strip()
        base_name = original_name or "匯入專案"
        actor = current_actor()
        if actor and not base_name.startswith(f"{actor} | "):
            base_name = f"{actor} | {base_name}"
        clean_name = base_name
        suffix = 2
        while _project_id(clean_name) in {project_id for _, project_id in list_projects(base)}:
            clean_name = f"{base_name}（匯入 {suffix}）"
            suffix += 1
        project_id = _project_id(clean_name)
        target = base / project_id
        if target.exists():
            raise ValueError("無法為匯入專案配置唯一識別碼")
        target.mkdir(parents=True)
        try:
            question_set_ids = list(project_payload.get("question_set_ids") or [])
            restored_question_set_ids: dict[str, str] = {}
            for member, relative in safe_members:
                if member.is_dir():
                    continue
                if len(relative.parts) == 3 and relative.parts[1] == "central_question_sets":
                    identifier = relative.parts[2][:-5] if relative.parts[2].endswith(".json") else ""
                    if not identifier or Path(identifier).name != identifier:
                        raise ValueError("專案封裝中的題目集 ID 無效")
                    if identifier not in question_set_ids:
                        raise ValueError("專案封裝包含未綁定的題目集")
                    item = json.loads(archive.read(member))
                    if not isinstance(item, dict) or not isinstance(item.get("questions"), list):
                        raise ValueError("專案封裝中的題目集格式錯誤")
                    set_root = Path("data/question_sets")
                    set_root.mkdir(parents=True, exist_ok=True)
                    candidate_id = identifier
                    existing_path = set_root / f"{candidate_id}.json"
                    if existing_path.is_file():
                        existing = read_json(existing_path)
                        if existing.get("questions") != item.get("questions"):
                            candidate_id = uuid4().hex
                        else:
                            restored_question_set_ids[identifier] = candidate_id
                            continue
                    item["question_set_id"] = candidate_id
                    write_json(set_root / f"{candidate_id}.json", item)
                    restored_question_set_ids[identifier] = candidate_id
                    continue
                output_path = target.joinpath(*relative.parts[1:])
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, output_path.open("wb") as destination:
                    shutil.copyfileobj(source, destination)

            now = datetime.now(timezone.utc).isoformat()
            project_payload.update({
                "project_id": project_id,
                "neo4j_database": project_database_name(project_id),
                "name": clean_name,
                "base_name": clean_name,
                "created_by": actor or current_actor() or "未知",
                "created_at": now,
                "updated_at": now,
            })
            if question_set_ids:
                project_payload["question_set_ids"] = [
                    restored_question_set_ids.get(identifier, identifier)
                    for identifier in question_set_ids
                ]
            project_payload["settings"] = _without_model_credentials(project_payload.get("settings", {}))
            if isinstance(project_payload["settings"], dict):
                project_payload["settings"]["neo4j_database"] = project_database_name(project_id)
            documents_dir = target / "documents"
            for document in project_payload.get("documents") or []:
                filename = Path(str(document.get("name") or "")).name
                document_path = documents_dir / filename
                document["path"] = str(document_path) if filename and document_path.is_file() else ""
            for metadata in project_payload.get("documents_meta") or []:
                filename = Path(str(metadata.get("file_name") or "")).name
                document_path = documents_dir / filename
                if filename and document_path.is_file():
                    metadata["file_path"] = str(document_path)
            graph_state = project_payload.get("graph_state")
            if isinstance(graph_state, dict) and graph_state:
                graph_state["neo4j_imported"] = False
                graph_state["neo4j_error"] = "匯入封裝不包含外部 Neo4j Database；請重新執行 Embedding 並匯入 Neo4j。"
            write_json(target / "project.json", project_payload)
            _write_project_summary(target, project_payload)
            append_audit_event(
                target / "activity.log", current_action() or "project_imported",
                actor=actor,
            )
        except Exception:
            shutil.rmtree(target, ignore_errors=True)
            raise
    return load_project_metadata(project_id, base)


def _save_project_unlocked(
    project_id: str,
    payload: dict[str, Any],
    document_paths: list[str] | None = None,
    root: str | Path = PROJECTS_DIR,
) -> dict[str, Any]:
    current = load_project_metadata(project_id, root)
    project_dir = Path(root) / project_id
    documents = list(payload.get("documents", current.get("documents") or []))
    if document_paths:
        documents_dir = project_dir / "documents"
        documents_dir.mkdir(parents=True, exist_ok=True)
        by_name = {doc.get("name"): index for index, doc in enumerate(documents)}
        for document_path in document_paths:
            source = Path(document_path)
            if not source.is_file():
                raise ValueError("找不到要保存的 PDF 文件")
            destination = documents_dir / source.name
            if source.resolve() != destination.resolve():
                shutil.copy2(source, destination)
            record = {"name": source.name, "path": str(destination)}
            if source.name in by_name:
                documents[by_name[source.name]] = record
            else:
                by_name[source.name] = len(documents)
                documents.append(record)
    payload_for_storage = dict(payload)
    incoming_settings = payload_for_storage.get("settings")
    if isinstance(incoming_settings, dict):
        incoming_settings = dict(incoming_settings)
        incoming_settings.pop("schema_granularity", None)
        incoming_settings.pop("schema_text", None)
        payload_for_storage["settings"] = incoming_settings
    incoming_graph_state = payload_for_storage.get("graph_state")
    if isinstance(incoming_graph_state, dict):
        incoming_graph_state = dict(incoming_graph_state)
        incoming_graph_state.pop("run_id", None)
        for obsolete_key in ("schema", "schema_json", "schema_granularity"):
            incoming_graph_state.pop(obsolete_key, None)
        payload_for_storage["graph_state"] = incoming_graph_state
    graph_state_for_chunks = payload_for_storage.get("graph_state")
    has_graph_chunks = isinstance(graph_state_for_chunks, dict) and "chunks" in graph_state_for_chunks
    if "chunks" in payload or has_graph_chunks:
        has_project_chunks = "chunks" in payload
        if has_project_chunks:
            chunks = payload_for_storage.pop("chunks") or []
        elif current.get("chunks_data_file"):
            chunks = _read_chunk_data(project_dir, current).get("chunks", [])
        else:
            chunks = []
        chunk_payload = {"chunks": chunks}
        if isinstance(graph_state_for_chunks, dict) and has_graph_chunks:
            graph_chunks = graph_state_for_chunks.get("chunks") or []
            graph_state_for_chunks = dict(graph_state_for_chunks)
            graph_state_for_chunks.pop("chunks", None)
            if graph_chunks == chunks:
                graph_state_for_chunks["graph_chunks_from_project"] = True
                graph_state_for_chunks.pop("graph_chunks_in_sidecar", None)
            else:
                chunk_payload["graph_chunks"] = graph_chunks
                graph_state_for_chunks["graph_chunks_in_sidecar"] = True
                graph_state_for_chunks.pop("graph_chunks_from_project", None)
            payload_for_storage["graph_state"] = graph_state_for_chunks
        else:
            existing_graph_state = current.get("graph_state") or {}
            old_chunk_payload = (
                _read_chunk_data(project_dir, current) if current.get("chunks_data_file") else {}
            )
            if existing_graph_state.get("graph_chunks_in_sidecar"):
                if "graph_chunks" in old_chunk_payload:
                    chunk_payload["graph_chunks"] = old_chunk_payload["graph_chunks"]
                if isinstance(graph_state_for_chunks, dict):
                    graph_state_for_chunks["graph_chunks_in_sidecar"] = True
        _write_chunk_data(project_dir, chunk_payload)
        payload_for_storage["chunks_data_file"] = CHUNK_DATA_FILENAME
        payload_for_storage["chunk_count"] = len(chunks) if has_project_chunks else int(
            current.get("chunk_count", len(chunks)) or 0
        )
    if isinstance(payload.get("graph_state"), dict):
        current_graph_state = current.get("graph_state") or {}
        incoming_graph_state = dict(payload_for_storage.get("graph_state") or {})
        if "entities" in incoming_graph_state or "relationships" in incoming_graph_state:
            entities = incoming_graph_state.pop("entities", [])
            relationships = incoming_graph_state.pop("relationships", [])
            _write_graph_data(project_dir, {
                "entities": entities,
                "relationships": relationships,
            })
            incoming_graph_state["graph_data_file"] = GRAPH_DATA_FILENAME
            incoming_graph_state["entity_count"] = len(entities)
            incoming_graph_state["relationship_count"] = len(relationships)
            payload_for_storage["graph_state"] = incoming_graph_state
        else:
            # Status/settings-only updates must keep the previously persisted
            # graph sidecar reference and its counts.
            payload_for_storage["graph_state"] = {
                **current_graph_state,
                **incoming_graph_state,
            }
    updated = {
        **current,
        **payload_for_storage,
        "project_id": project_id,
        "name": str(payload.get("name") or current["name"]),
        "base_name": str(current.get("base_name") or current["name"]),
        "created_at": current["created_at"],
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "documents": documents,
    }
    updated["settings"] = _without_model_credentials(updated.get("settings", {}))
    write_json(project_dir / "project.json", updated)
    _write_project_summary(project_dir, updated)
    return updated


def remove_document(
    project_id: str, file_name: str, root: str | Path = PROJECTS_DIR
) -> dict[str, Any]:
    current = load_project_metadata(project_id, root)
    documents = current.get("documents") or []
    remaining = [doc for doc in documents if doc.get("name") != file_name]
    for doc in documents:
        if doc.get("name") == file_name:
            path = Path(doc.get("path", ""))
            if path.is_file():
                path.unlink()
    save_document_pages(project_id, file_name, None, root)
    return save_project(project_id, {"documents": remaining}, root=root)


def append_question(
    project_id: str, record: dict[str, Any], root: str | Path = PROJECTS_DIR
) -> dict[str, Any]:
    project = load_project_metadata(project_id, root)
    questions = list(project.get("questions") or [])
    questions.append({"asked_at": datetime.now(timezone.utc).isoformat(), **record})
    return save_project(project_id, {"questions": questions}, root=root)
