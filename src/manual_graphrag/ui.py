from __future__ import annotations

import csv
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

import gradio as gr

from .chunking import TextChunk, chunk_pages, preview_rows
from .config import (
    public_settings,
)
from .env_store import load_env, save_env
from .evaluation_service import (
    generate_document_summary,
    generate_evaluation_questions,
    judge_evaluation_answer,
    questions_are_similar,
    select_relevant_documents,
)
from .graph_service import (
    RunCancelled,
    RunControl,
    check_model_connection,
    extract_graph,
    list_models,
    plan_graph_schema,
    validate_schema,
)
from .neo4j_service import (
    check_neo4j_connection,
    import_extraction,
    load_latest_graph,
    search_graph_evidence,
    vector_index_name,
)
from .pdf_service import extract_pdf
from .project_store import (
    append_question,
    create_project,
    delete_project,
    list_projects,
    load_project,
    remove_document,
    save_project,
)
from .qa_service import (
    RERANK_CANDIDATE_MULTIPLIER,
    RERANK_MAX_CANDIDATES,
    answer_graph_question,
    check_embedding_connection,
    embedding_vectors,
    rerank_evidence,
)
from .service_settings import (
    capture_service_settings,
    configured_models,
    load_service_settings,
    preferred_service_model,
    provider_models,
    resolve_model_service,
    restore_service_settings,
    save_service_settings,
    service_choice_items,
    service_choices,
)
from .storage import write_json

DEFAULT_LLM_MODEL = "gpt-4.1-mini"


def connection_summary(
    neo4j_uri: str,
    neo4j_database: str,
    neo4j_username: str,
    neo4j_password: str,
    model_endpoint: str,
    api_key: str,
) -> tuple[str, dict[str, object]]:
    missing = [
        label
        for label, value in {
            "Neo4j URI": neo4j_uri,
            "Database": neo4j_database,
            "Username": neo4j_username,
            "Password": neo4j_password,
            "模型端點": model_endpoint,
        }.items()
        if not value.strip()
    ]
    if missing:
        return f"⚠️ 尚未填寫：{'、'.join(missing)}", {}
    settings = public_settings(
        {
            "neo4j_uri": neo4j_uri,
            "neo4j_database": neo4j_database,
            "neo4j_username": neo4j_username,
            "neo4j_password": neo4j_password,
            "model_endpoint": model_endpoint,
            "api_key": api_key,
        }
    )
    return "✅ 欄位格式已通過初步檢查；實際連線將於下一版接入。", settings


def check_neo4j_for_ui(
    uri: str, database: str, username: str, password: str
) -> tuple[str, bool]:
    try:
        check_neo4j_connection(uri, database, username, password)
    except ValueError as exc:
        return f"❌ {exc}", False
    return "✅ Neo4j 連線成功，且可存取指定 Database。", True


def check_model_service_for_ui(base_url: str, api_key: str) -> str:
    try:
        check_model_connection(base_url, api_key)
    except ValueError as exc:
        return f"❌ {exc}"
    return "✅ 模型服務連線成功。"


def check_embedding_service_for_ui(base_url: str, api_key: str, model: str) -> str:
    try:
        check_embedding_connection(base_url, api_key, model)
    except ValueError as exc:
        return f"❌ {exc}"
    return "✅ Embedding 服務連線成功。"


def selected_models_for_ui(rows: list[list[Any]] | None) -> list[str]:
    return list(dict.fromkeys(
        str(row[1]) for row in (rows or []) if len(row) == 2 and row[0] is True
    ))


def resolve_model_credentials_for_ui(
    state: dict[str, Any], model: str | None,
) -> tuple[str, str]:
    try:
        base_url, api_key, _ = resolve_model_service(state, model)
    except ValueError:
        return "", ""
    return base_url, api_key


def render_service_for_ui(state: dict[str, Any]) -> tuple[Any, ...]:
    profile = state["profiles"][state["active"]]
    ollama = state["active"] == "Ollama"
    allowed = service_choices(state)
    choice_items = service_choice_items(state)
    rows = profile["rows"] if ollama else [[True, model] for model in provider_models(state, state["active"])]
    fallback = preferred_service_model(state)
    return (
        state, profile["base_url"], profile["api_key"],
        gr.update(value=rows, visible=ollama), gr.update(visible=not ollama),
        gr.update(visible=ollama), profile["status"],
        *(gr.update(choices=choice_items, value=model if model in allowed else fallback) for model in profile["models"]),
    )


def service_action_for_ui(
    action: str, provider: str, state: dict[str, Any], base_url: str, api_key: str,
    rows: list[list[Any]], *models: str | None,
) -> tuple[Any, ...]:
    state = capture_service_settings(state, base_url, api_key, rows, list(models))
    if action == "switch":
        if provider not in state["profiles"]:
            raise gr.Error("不支援的服務")
        state["active"] = provider
    profile = state["profiles"][state["active"]]
    try:
        if action == "test" and state["active"] in {"OpenAI", "Voyage"}:
            provider_name = state["active"]
            allowed = configured_models(state["kind"], provider_name)
            profile["connected"] = False
            if provider_name == "Voyage":
                check_embedding_connection(profile["base_url"], profile["api_key"], allowed[0])
            else:
                check_model_connection(profile["base_url"], profile["api_key"])
            profile["connected"] = True
            profile["models"] = [model if model in allowed else allowed[0] for model in profile["models"]]
            profile["status"] = "✅ 連線成功；可用模型：" + "、".join(allowed)
        elif action == "fetch" and state["active"] == "Ollama":
            profile["connected"] = False
            available = list_models(profile["base_url"], profile["api_key"])
            selected = selected_models_for_ui(profile["rows"])
            profile["rows"] = [[model in selected, model] for model in available]
            profile["connected"] = True
            profile["status"] = f"✅ 已取得 {len(available)} 個模型；請勾選此服務要使用的模型。"
        result = render_service_for_ui(state)
        save_service_settings(state)
        return result
    except (OSError, ValueError) as exc:
        profile["status"] = f"❌ {exc}"
        profile["connected"] = False
        try:
            save_service_settings(state)
        except OSError:
            profile["status"] += "；服務設定保存失敗。"
        return render_service_for_ui(state)


def load_project_with_services_for_ui(
    project_id: str, llm_state: dict[str, Any], embedding_state: dict[str, Any],
) -> tuple[Any, ...]:
    values = list(load_project_for_ui(project_id))
    llm_state = restore_service_settings(
        llm_state, values[6], values[7], {0: values[8], 1: values[18], 4: values[10]},
    )
    embedding_profile = embedding_state["profiles"][embedding_state["active"]]
    embedding_state = restore_service_settings(
        embedding_state, embedding_profile["base_url"], embedding_profile["api_key"], {0: values[9]},
    )
    llm = render_service_for_ui(llm_state)
    embedding = render_service_for_ui(embedding_state)
    values[6:8] = llm[1:3]
    values[8], values[18], values[10], values[9] = llm[7], llm[8], llm[11], embedding[7]
    save_service_settings(llm_state)
    save_service_settings(embedding_state)
    return (*values, llm_state["active"], llm[0], *llm[3:7], llm[9], llm[10],
            embedding_state["active"], embedding[0], *embedding[3:7])


def load_evaluation_with_services_for_ui(
    project_id: str, llm_state: dict[str, Any],
) -> tuple[Any, ...]:
    values = list(load_evaluation_for_ui(project_id))
    allowed = service_choices(llm_state)
    choice_items = service_choice_items(llm_state)
    for index in (3, 4):
        if not isinstance(values[index], dict):
            values[index] = gr.update(choices=choice_items, value=values[index] if values[index] in allowed else None)
    return tuple(values)


def reload_env_with_services_for_ui() -> tuple[Any, ...]:
    env = load_env()
    llm_state = load_service_settings("llm", env)
    embedding_state = load_service_settings("embedding", env)
    return (
        env["NEO4J_URI"], env["NEO4J_DATABASE"], env["NEO4J_USERNAME"], env["NEO4J_PASSWORD"],
        llm_state["active"], *render_service_for_ui(llm_state),
        embedding_state["active"], *render_service_for_ui(embedding_state),
        "✅ 已重新讀取 .env；OpenAI 請重新測試連線。",
    )


def persist_env_settings(
    neo4j_uri: str,
    neo4j_database: str,
    neo4j_username: str,
    neo4j_password: str,
    model_endpoint: str,
    api_key: str,
    embedding_api_base: str,
    embedding_api_key: str,
    build_model: str,
    embedding_model: str,
    answer_model: str,
) -> str:
    save_env({
        "NEO4J_URI": neo4j_uri,
        "NEO4J_DATABASE": neo4j_database,
        "NEO4J_USERNAME": neo4j_username,
        "NEO4J_PASSWORD": neo4j_password,
    })
    return "✅ 連線設定已自動儲存；模型選擇保存在 config/model_settings.yaml"


def reload_env_settings() -> tuple[str, ...]:
    env = load_env()
    llm = load_service_settings("llm", env)
    embedding = load_service_settings("embedding", env)
    llm_profile = llm["profiles"][llm["active"]]
    embedding_profile = embedding["profiles"][embedding["active"]]
    return (
        env["NEO4J_URI"], env["NEO4J_DATABASE"], env["NEO4J_USERNAME"], env["NEO4J_PASSWORD"],
        llm_profile["base_url"], llm_profile["api_key"],
        embedding_profile["base_url"], embedding_profile["api_key"],
        llm_profile["models"][0], embedding_profile["models"][0], llm_profile["models"][4],
        "✅ 已重新讀取 .env 與模型 YAML",
    )


def workflow_tabs_for_ui(
    project_id: str,
    neo4j_connected: bool,
    llm_state: dict[str, Any],
    embedding_state: dict[str, Any],
) -> tuple[dict[str, Any], ...]:
    def has_available_service(state: dict[str, Any]) -> bool:
        try:
            return any(provider_models(state, provider) for provider in state["profiles"])
        except ValueError:
            return False

    enabled = bool(
        project_id and neo4j_connected
        and has_available_service(llm_state)
        and has_available_service(embedding_state)
    )
    return tuple(gr.update(interactive=enabled) for _ in range(6))


def lock_project_tabs_for_ui(project_id: str) -> tuple[dict[str, Any], ...]:
    """Keep delete-project output compatibility; successful setup uses workflow_tabs_for_ui."""
    return tuple(gr.update(interactive=False) for _ in range(6))


def delete_project_for_ui(
    project_id: str,
) -> tuple[Any, ...]:
    try:
        name = delete_project(project_id)
    except (OSError, ValueError) as exc:
        return (
            gr.update(), gr.update(), f"❌ {exc}",
            *lock_project_tabs_for_ui(project_id), False,
        )
    return (
        gr.update(choices=_project_choices(), value=None), {},
        f"✅ 已刪除專案「{name}」。", *lock_project_tabs_for_ui(""), True,
    )


def refresh_projects_after_delete_for_ui(deleted: bool) -> dict[str, Any]:
    if not deleted:
        return gr.update()
    return gr.update(choices=_project_choices(), value=None)


def _project_choices() -> list[tuple[str, str]]:
    return list_projects()


def create_project_for_ui(name: str) -> tuple[dict[str, Any], dict[str, Any], str]:
    try:
        project = create_project(name)
    except ValueError as exc:
        return gr.update(), {}, f"❌ {exc}"
    return gr.update(choices=_project_choices(), value=project["project_id"]), project, f"✅ 已建立專案「{project['name']}」。"


def refresh_projects_for_ui(project: dict[str, Any] | None = None) -> dict[str, Any]:
    choices = _project_choices()
    available_ids = {project_id for _, project_id in choices}
    selected_id = str((project or {}).get("project_id") or "")
    return gr.update(
        choices=choices, value=selected_id if selected_id in available_ids else None
    )


def reset_new_project_pdf_status_for_ui() -> str:
    return "尚未解析 PDF。可重複上傳多份 PDF，逐一加入同一個專案。"


def _chunk_dicts(chunks: list[TextChunk]) -> list[dict[str, Any]]:
    return [
        {"number": c.number, "text": c.text, "pages": list(c.pages), "document": c.document}
        for c in chunks
    ]


def _stored_chunks(items: list[dict[str, Any]]) -> list[TextChunk]:
    return [
        TextChunk(
            int(i["number"]), str(i["text"]), tuple(i.get("pages") or []),
            str(i.get("document", "")),
        )
        for i in items
    ]


def _document_rows(documents: list[dict[str, Any]]) -> list[list[object]]:
    return [
        [
            False,
            doc.get("file_name", ""),
            f"{doc.get('page_start', '')}–{doc.get('page_end', '')}",
            doc.get("chunk_count", 0),
        ]
        for doc in documents
    ]


def _document_choices(documents: list[dict[str, Any]]) -> dict[str, Any]:
    return gr.update(
        choices=[doc.get("file_name", "") for doc in documents], value=[]
    )


def _chunk_rows(chunks: list[TextChunk]) -> list[list[object]]:
    return preview_rows(chunks, limit=len(chunks))


def _document_status(index: int, total: int, doc: dict[str, Any], chunk_count: int) -> str:
    if not total:
        return "尚未解析任何 PDF。"
    name = doc.get("file_name", "")
    page_range = f"{doc.get('page_start', '')}–{doc.get('page_end', '')}"
    return f"文件 {index + 1} / {total}：{name}（第 {page_range} 頁），共 {chunk_count} 個 chunk。"


def _history_rows(questions: list[dict[str, Any]]) -> list[list[object]]:
    return [[i.get("asked_at", ""), i.get("question", ""), i.get("answer", ""),
             i.get("retrieval_mode", ""), i.get("document", "")] for i in reversed(questions)]


def _display_retrieval_mode(value: str | None) -> str:
    return {
        "GraphRAG": "關聯擴展檢索",
        "向量 RAG": "基本檢索",
    }.get(value or "", value or "關聯擴展檢索")


def _source_display(item: dict[str, Any]) -> tuple[str, str, str]:
    references = item.get("source_references") or []
    if not references:
        return (
            ", ".join(map(str, item.get("source_chunk_numbers", []))),
            ", ".join(map(str, item.get("source_pages", []))),
            "、".join(item.get("source_documents", [])),
        )
    chunk_lines = []
    page_lines = []
    documents = []
    for reference in references:
        document = str(reference.get("document") or "未標示文件")
        documents.append(document)
        chunk_lines.append(
            f"{document}：{', '.join(map(str, reference.get('chunk_numbers', [])))}"
        )
        page_lines.append(
            f"{document}：{', '.join(map(str, reference.get('pages', [])))}"
        )
    return "\n".join(chunk_lines), "\n".join(page_lines), "\n".join(documents)


def _graph_rows(graph: dict[str, Any]) -> tuple[list[list[object]], list[list[object]]]:
    entities = [[
        item.get("name", ""), item.get("type", ""), item.get("description", ""),
        *_source_display(item),
    ] for item in graph.get("entities", [])]
    relationships = [[
        item.get("source", ""), item.get("type", ""), item.get("target", ""),
        item.get("description", ""),
        *_source_display(item),
    ] for item in graph.get("relationships", [])]
    return entities, relationships


def save_project_for_ui(
    project_id: str, documents: list[dict[str, Any]],
    chunks: list[TextChunk], graph_state: dict[str, Any],
    neo4j_uri: str, neo4j_database: str, neo4j_username: str, neo4j_password: str,
    _model_endpoint: str, _api_key: str, graph_llm_model: str,
    graph_embedding_model: str, answer_model: str,
    chunk_size: int, chunk_overlap: int,
    graph_temperature: float,
    schema_granularity: str,
    max_concurrent_requests: int, schema_sampling_mode: str,
    schema_sample_page_count: int, extraction_llm_model: str,
    extraction_max_concurrent_requests: int,
    retrieval_mode: str, top_k: int, schema_text: str,
) -> tuple[dict[str, Any], str]:
    if not project_id:
        return {}, "❌ 請先建立或載入專案。"
    settings = {
        "neo4j_uri": neo4j_uri, "neo4j_database": neo4j_database,
        "neo4j_username": neo4j_username, "neo4j_password": neo4j_password,
        "graph_llm_model": graph_llm_model, "graph_embedding_model": graph_embedding_model,
        "answer_model": answer_model,
        "chunk_size": int(chunk_size), "chunk_overlap": int(chunk_overlap),
        "graph_temperature": float(graph_temperature),
        "schema_granularity": schema_granularity,
        "max_concurrent_requests": int(max_concurrent_requests),
        "schema_sampling_mode": schema_sampling_mode,
        "schema_sample_page_count": int(schema_sample_page_count),
        "extraction_llm_model": extraction_llm_model,
        "extraction_max_concurrent_requests": int(extraction_max_concurrent_requests),
        "retrieval_mode": retrieval_mode,
        "top_k": int(top_k), "schema_text": schema_text or "",
    }
    documents = documents or []
    try:
        project = save_project(project_id, {
            "settings": settings, "documents_meta": documents,
            "chunks": _chunk_dicts(chunks or []), "graph_state": graph_state or {},
        }, [doc["file_path"] for doc in documents if doc.get("file_path")])
    except (OSError, ValueError) as exc:
        return {}, f"❌ {exc}"
    return project, f"✅ 專案「{project['name']}」已保存。"


def load_project_for_ui(project_id: str) -> tuple[Any, ...]:
    try:
        project = load_project(project_id)
    except (OSError, ValueError) as exc:
        raise gr.Error(str(exc))
    settings = project.get("settings") or {}
    env, get = load_env(), settings.get
    llm_settings = load_service_settings("llm", env)
    embedding_settings = load_service_settings("embedding", env)
    llm_profile = llm_settings["profiles"][llm_settings["active"]]
    embedding_profile = embedding_settings["profiles"][embedding_settings["active"]]
    documents = project.get("documents_meta") or []
    chunks = _stored_chunks(project.get("chunks") or [])
    graph = project.get("graph_state") or {}
    active_preview = documents[-1] if documents else {}
    active_chunks = [
        chunk for chunk in chunks if chunk.document == active_preview.get("file_name")
    ] if active_preview else []
    document_status = (
        _document_status(len(documents) - 1, len(documents), active_preview, len(active_chunks))
        if documents else "請先解析 PDF。"
    )
    entity_rows, relationship_rows = _graph_rows(graph)
    if graph.get("run_id"):
        graph_status = (
            f"✅ 已載入專案保存的圖譜：{len(entity_rows)} 個實體、"
            f"{len(relationship_rows)} 筆關係。"
        )
        import_status = (
            "✅ 此圖譜已匯入 Neo4j。" if graph.get("neo4j_imported")
            else "此圖譜尚未匯入 Neo4j。"
        )
    else:
        graph_status, import_status = "尚未執行抽取。", "尚未執行 Embedding 與匯入。"
    return (
        project, f"✅ 已載入專案「{project['name']}」。",
        get("neo4j_uri", env["NEO4J_URI"]), get("neo4j_database", env["NEO4J_DATABASE"]),
        get("neo4j_username", env["NEO4J_USERNAME"]), get("neo4j_password", env["NEO4J_PASSWORD"]),
        llm_profile["base_url"], llm_profile["api_key"],
        get("graph_llm_model", DEFAULT_LLM_MODEL), get("graph_embedding_model", embedding_profile["models"][0]),
        get("answer_model", DEFAULT_LLM_MODEL),
        get("chunk_size", 1500), get("chunk_overlap", 200), get("graph_temperature", 0),
        get("schema_granularity", "平衡"),
        get("max_concurrent_requests", 3),
        gr.update(
            value=get("schema_sampling_mode", "全部頁面"),
            visible=get("schema_sampling_mode", "全部頁面") == "隨機抽取 N 頁",
        ),
        get("schema_sample_page_count", 10), get("extraction_llm_model", DEFAULT_LLM_MODEL),
        get("extraction_max_concurrent_requests", 3),
        _display_retrieval_mode(get("retrieval_mode")), get("top_k", 8), get("schema_text", ""),
        documents, chunks, graph, active_preview, active_chunks,
        _document_rows(documents), _document_choices(documents),
        _chunk_rows(active_chunks), document_status,
        _history_rows(project.get("questions") or []),
        entity_rows, relationship_rows, graph_status, import_status,
    )


def answer_question_for_project_ui(project_id: str, *args: Any) -> tuple[Any, ...]:
    status, answer, sources = answer_question_for_ui(*args)
    if not status.startswith("✅") or not project_id:
        note = "" if project_id else "⚠️ 未選擇專案，問答未加入專案紀錄。"
        return status, answer, sources, gr.update(), note
    try:
        current = load_project(project_id)
        record = {
            "question": str(args[9]).strip(), "answer": answer,
            "answer_model": str(args[8]), "retrieval_mode": str(args[10]),
            "top_k": int(args[11]), "sources": sources,
            "document": (current.get("graph_state") or {}).get("document", ""),
        }
        project = append_question(project_id, record)
    except (OSError, ValueError) as exc:
        return status, answer, sources, gr.update(), f"⚠️ 回答成功，但專案紀錄保存失敗：{exc}"
    return status, answer, sources, _history_rows(project.get("questions") or []), "✅ 問答紀錄已加入目前專案。"


def _evaluation_question_rows(questions: list[dict[str, Any]]) -> list[list[object]]:
    return [[item["number"], item["question"], item["expected_answer"],
             ", ".join(map(str, item.get("source_pages", []))),
             item.get("document", "")] for item in questions]


def _questions_from_rows(rows: Any) -> list[dict[str, Any]]:
    if hasattr(rows, "values"):
        rows = rows.values.tolist()
    if not isinstance(rows, list) or not rows:
        raise ValueError("題目不可為空")
    questions = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, (list, tuple)) or len(row) < 3:
            raise ValueError(f"第 {index} 列格式不正確")
        question = str(row[1] or "").strip()
        answer = str(row[2] or "").strip()
        if not question or not answer:
            raise ValueError(f"第 {index} 題的問題與標準答案不可為空")
        raw_pages = row[3] if len(row) > 3 else ""
        if isinstance(raw_pages, (list, tuple)):
            page_values = raw_pages
        else:
            page_values = str(raw_pages or "").replace("，", ",").split(",")
        try:
            pages = [int(value) for value in page_values if str(value).strip()]
        except ValueError as exc:
            raise ValueError(f"第 {index} 題的來源頁碼必須是逗號分隔的整數") from exc
        document = str(row[4]).strip() if len(row) > 4 and row[4] is not None else ""
        questions.append({
            "number": index, "question": question,
            "expected_answer": answer, "source_pages": pages,
            "document": document,
        })
    return questions


def save_evaluation_questions_for_ui(
    project_id: str, rows: Any, evaluation: dict[str, Any]
) -> tuple[str, dict[str, Any], list[list[object]]]:
    if not project_id:
        return "❌ 請先建立或載入專案。", evaluation or {}, []
    try:
        questions = _questions_from_rows(rows)
        updated = dict(evaluation or {})
        updated.update({"questions": questions, "results": [], "dirty": False})
        save_project(project_id, {"evaluation": updated})
    except (OSError, ValueError) as exc:
        failed = dict(evaluation or {})
        if "questions" in locals():
            failed.update({"questions": questions, "results": [], "dirty": True})
        return f"❌ {exc}；自動儲存失敗。", failed, []
    return f"✅ 已自動儲存 {len(questions)} 道題目。", updated, []


def import_evaluation_questions_for_ui(
    project_id: str, file_path: str | None, evaluation: dict[str, Any]
) -> tuple[str, list[list[object]], dict[str, Any], list[list[object]]]:
    if not project_id:
        return "❌ 請先建立或載入專案。", [], evaluation or {}, []
    if not file_path:
        return "❌ 請選擇 JSON 或 CSV 題目檔。", [], evaluation or {}, []
    try:
        path = Path(file_path)
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            items = payload.get("questions") if isinstance(payload, dict) else payload
            if not isinstance(items, list):
                raise ValueError("JSON 必須是題目陣列或包含 questions 陣列")
            rows = [[item.get("number", index), item.get("question", ""),
                     item.get("expected_answer", ""), item.get("source_pages", []),
                     item.get("document", "")]
                    for index, item in enumerate(items, start=1) if isinstance(item, dict)]
        elif path.suffix.lower() == ".csv":
            with path.open(encoding="utf-8-sig", newline="") as handle:
                items = list(csv.DictReader(handle))
            rows = [[item.get("number", index), item.get("question", ""),
                     item.get("expected_answer", ""), item.get("source_pages", ""),
                     item.get("document", "")]
                    for index, item in enumerate(items, start=1)]
        else:
            raise ValueError("只支援 .json 或 .csv 題目檔")
        questions = _questions_from_rows(rows)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return f"❌ 匯入失敗：{exc}", [], evaluation or {}, []
    updated = dict(evaluation or {})
    updated.update({"questions": questions, "results": [], "dirty": False})
    try:
        save_project(project_id, {"evaluation": updated})
    except (OSError, ValueError) as exc:
        updated["dirty"] = True
        return f"❌ 匯入成功但自動儲存失敗：{exc}", _evaluation_question_rows(questions), updated, []
    return (f"✅ 已匯入並自動儲存 {len(questions)} 道題目。",
            _evaluation_question_rows(questions), updated, [])


def export_evaluation_questions_for_ui(
    project_id: str, rows: Any
) -> tuple[str, str | None]:
    if not project_id:
        return "❌ 請先建立或載入專案。", None
    try:
        questions = _questions_from_rows(rows)
        output = write_json(
            Path("data/projects") / project_id / "exports" / "questions.json",
            {"questions": questions},
        )
    except (OSError, ValueError) as exc:
        return f"❌ 匯出失敗：{exc}", None
    return f"✅ 已匯出 {len(questions)} 道題目。", str(output)


def _evaluation_result_rows(results: list[dict[str, Any]]) -> list[list[object]]:
    return [[
        item["number"],
        item["question"],
        item["expected_answer"],
        item.get("document", ""),
        "、".join(item.get("selected_documents") or []),
        "✅ 正確" if item.get("routing_correct") else "❌ 錯誤",
        item.get("actual_answer", ""),
        "✅ 通過" if item.get("passed") else "❌ 未通過",
        item.get("reason", ""),
    ] for item in results]


def _evaluation_summary(results: list[dict[str, Any]], *, loaded: bool = False) -> str:
    passed = sum(bool(item.get("passed")) for item in results)
    routing_passed = sum(bool(item.get("routing_correct")) for item in results)
    complete_passed = sum(
        bool(item.get("routing_correct") and item.get("passed")) for item in results
    )
    total = len(results)
    recall_at_5 = sum(bool(item.get("recall_at_5")) for item in results) / total
    mrr = sum(float(item.get("reciprocal_rank", 0)) for item in results) / total

    document_totals: dict[str, list[int]] = {}
    for item in results:
        document = str(item.get("document") or "未標示文件")
        stats = document_totals.setdefault(document, [0, 0, 0])
        stats[0] += int(bool(item.get("routing_correct")))
        stats[1] += int(bool(item.get("passed")))
        stats[2] += 1
    document_summary = "\n".join(
        f"- {document}：路由正確 {stats[0]} / {stats[2]}；"
        f"答案正確 {stats[1]} / {stats[2]}"
        for document, stats in document_totals.items()
    )

    heading = "已載入測試結果" if loaded else "測試完成"
    failed = total - passed
    accuracy = passed / total * 100
    return (
        f"## {heading}｜總共答對 {passed} 題 / {total} 題  "
        f"\n文件路由正確：{routing_passed} / {total}｜"
        f"路由且答案正確：{complete_passed} / {total}  "
        f"\n答錯：{failed} 題｜答案正確率：{accuracy:.1f}%"
        f"\n\n### 各 PDF 結果\n{document_summary}"
        f"  \nRecall@5：{recall_at_5:.1%}｜MRR：{mrr:.3f}"
    )


def _source_values_for_document(display: str, documents: str, document: str) -> set[int]:
    for line in str(display).splitlines():
        prefix, separator, values = line.partition("：")
        if separator and prefix == document:
            return {int(value) for value in re.findall(r"\d+", values)}
    listed_documents = {
        value for value in re.split(r"[、\n]", str(documents)) if value
    }
    if listed_documents == {document}:
        return {int(value) for value in re.findall(r"\d+", str(display))}
    return set()


def _retrieval_rank(question: dict[str, Any], rows: list[list[object]]) -> int | None:
    document = str(question.get("document") or "")
    expected_chunks = {
        int(value) for value in question.get("source_chunk_numbers", [])
    }
    expected_pages = {int(value) for value in question.get("source_pages", [])}
    for rank, row in enumerate(rows, start=1):
        chunks = _source_values_for_document(row[5], row[6], document)
        pages = _source_values_for_document(row[4], row[6], document)
        if expected_chunks and chunks & expected_chunks:
            return rank
        if not expected_chunks and expected_pages and pages & expected_pages:
            return rank
    return None


def load_evaluation_for_ui(project_id: str) -> tuple[Any, ...]:
    if not project_id:
        return {}, [], [], gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), "請先選擇專案。"
    try:
        project = load_project(project_id)
    except (OSError, ValueError) as exc:
        return {}, [], [], gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), f"❌ {exc}"
    evaluation = dict(project.get("evaluation") or {})
    evaluation.setdefault("dirty", False)
    preferences = evaluation.get("preferences") or {}
    questions = evaluation.get("questions") or []
    results = evaluation.get("results") or []
    legacy_model = preferences.get("model", DEFAULT_LLM_MODEL)
    return (
        evaluation, _evaluation_question_rows(questions), _evaluation_result_rows(results),
        preferences.get("generation_model", legacy_model),
        preferences.get("test_model", legacy_model), preferences.get("question_count", 10),
        _display_retrieval_mode(preferences.get("retrieval_mode")), preferences.get("top_k", 8),
        preferences.get("allow_parallel_generation", False),
        preferences.get("use_reranker", True),
        preferences.get("test_max_concurrent_requests", 3),
        (_evaluation_summary(results, loaded=True) if results else
         f"已載入 {len(questions)} 道題目與 0 筆測試結果。"),
    )


def save_evaluation_preferences_for_ui(
    project_id: str, generation_model: str, test_model: str, question_count: int,
    retrieval_mode: str, top_k: int,
    allow_parallel_generation: bool = False,
    test_max_concurrent_requests: int = 3,
    use_reranker: bool = True,
) -> str:
    if not project_id:
        return "⚠️ 請先選擇專案。"
    try:
        test_concurrency = int(test_max_concurrent_requests)
        if test_concurrency < 1:
            raise ValueError("最大並行請求數必須大於 0")
        project = load_project(project_id)
        evaluation = dict(project.get("evaluation") or {})
        evaluation["preferences"] = {
            "generation_model": generation_model, "test_model": test_model,
            "question_count": int(question_count),
            "retrieval_mode": retrieval_mode, "top_k": int(top_k),
            "allow_parallel_generation": bool(allow_parallel_generation),
            "use_reranker": bool(use_reranker),
            "test_max_concurrent_requests": int(test_max_concurrent_requests),
        }
        save_project(project_id, {"evaluation": evaluation})
    except (OSError, TypeError, ValueError) as exc:
        return f"❌ 自動保存失敗：{exc}"
    return "✅ 自動測試設定已保存。"


def _document_summary_rows(summaries: list[dict[str, Any]]) -> list[list[str]]:
    return [[
        str(item.get("document", "")), str(item.get("summary", "")),
        "、".join(item.get("identifiers", item.get("product_names", [])) or []),
        "、".join(item.get("topics") or []),
        "、".join(item.get("keywords") or []),
    ] for item in summaries]


def load_document_summaries_for_ui(project_id: str) -> tuple[list[list[str]], str]:
    if not project_id:
        return [], "請先選擇專案。"
    try:
        evaluation = dict(load_project(project_id).get("evaluation") or {})
    except (OSError, ValueError) as exc:
        return [], f"❌ {exc}"
    summaries = evaluation.get("document_summaries") or []
    if not summaries:
        return [], "尚未建立 PDF 摘要。"
    return _document_summary_rows(summaries), f"✅ 已載入 {len(summaries)} 份 PDF 摘要。"


def generate_document_summaries_for_ui(
    project_id: str, model_endpoint: str, api_key: str, model: str,
    chunks: list[TextChunk], max_concurrent_requests: int,
    run_control: RunControl,
    progress=gr.Progress(),
) -> tuple[str, list[list[str]], dict[str, Any]]:
    if not project_id:
        return "❌ 請先建立或載入專案。", [], {}
    if not model:
        return "❌ 請先選擇摘要模型。", [], {}
    run_control.reset()
    try:
        concurrency = int(max_concurrent_requests)
        if concurrency < 1:
            raise ValueError("最大並行請求數必須大於 0")
        chunks_by_document: dict[str, list[TextChunk]] = {}
        for chunk in chunks:
            chunks_by_document.setdefault(chunk.document, []).append(chunk)
        if not chunks_by_document:
            raise ValueError("請先解析 PDF 並產生 chunks")
        documents = list(chunks_by_document.values())
        total = len(documents)
        progress(0.0, desc=f"準備建立 {total} 份 PDF 摘要")

        def build_summary(selected: list[TextChunk]) -> dict[str, Any]:
            run_control.check()
            run_control.wait_if_paused()
            return generate_document_summary(
                model_endpoint, api_key, model, selected, control=run_control,
            )

        results: list[dict[str, Any] | None] = [None] * total
        completed = 0
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(build_summary, selected): index
                for index, selected in enumerate(documents)
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    results[index] = future.result()
                except RunCancelled:
                    raise
                except Exception as exc:
                    raise RuntimeError(f"PDF 摘要建立失敗：{exc}") from exc
                completed += 1
                progress(completed / total, desc=f"已完成 {completed} / {total} 份 PDF 摘要")
        summaries = results
        project = load_project(project_id)
        evaluation = dict(project.get("evaluation") or {})
        preferences = dict(evaluation.get("preferences") or {})
        preferences.update({
            "summary_model": model,
            "summary_max_concurrent_requests": concurrency,
        })
        evaluation.update({"preferences": preferences, "document_summaries": summaries})
        save_project(project_id, {"evaluation": evaluation})
    except RunCancelled:
        return "⏹ 已停止（使用者中止 PDF 摘要建立）。", [], {}
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        return f"❌ {exc}", [], {}
    return (f"✅ 已建立並保存 {len(summaries)} 份 PDF 摘要。",
            _document_summary_rows(summaries), evaluation)


def generate_evaluation_for_ui(
    project_id: str, model_endpoint: str, api_key: str, generation_model: str,
    test_model: str, question_count: int, retrieval_mode: str, top_k: int,
    chunks: list[TextChunk], allow_parallel_generation: bool = False,
    test_max_concurrent_requests: int = 3,
    use_reranker: bool = True,
) -> tuple[str, list[list[object]], dict[str, Any], list[list[object]]]:
    if not project_id:
        return "❌ 請先建立或載入專案。", [], {}, []
    if not generation_model:
        return "❌ 請先勾選並選擇生題模型。", [], {}, []
    try:
        count = int(question_count)
        if not 1 <= count <= 100:
            raise ValueError("題目數量必須介於 1 到 100")
        chunks_by_document: dict[str, list[TextChunk]] = {}
        for chunk in chunks:
            chunks_by_document.setdefault(chunk.document, []).append(chunk)
        document_chunks = list(chunks_by_document.values())
        if not document_chunks:
            raise ValueError("請先解析 PDF 並產生 chunks")
        existing_evaluation = dict(load_project(project_id).get("evaluation") or {})
        summaries_by_document = {
            str(item.get("document", "")): item
            for item in existing_evaluation.get("document_summaries", [])
        }
        accepted_questions: list[str] = []
        accepted_lock = Lock()

        def generate_for_document(
            document: str,
            selected_chunks: list[TextChunk],
        ) -> list[dict[str, Any]]:
            accepted: list[dict[str, Any]] = []
            last_error = ""
            attempts = 0
            max_attempts = count * 5
            while len(accepted) < count and attempts < max_attempts:
                attempts += 1
                with accepted_lock:
                    excluded = list(accepted_questions)
                try:
                    batch = generate_evaluation_questions(
                        model_endpoint,
                        api_key,
                        generation_model,
                        selected_chunks,
                        1,
                        excluded,
                        summaries_by_document.get(document),
                    )
                except ValueError as exc:
                    last_error = str(exc)
                    continue
                for question in batch:
                    with accepted_lock:
                        if any(
                            questions_are_similar(question["question"], existing)
                            for existing in accepted_questions
                        ):
                            continue
                        accepted_questions.append(question["question"])
                        accepted.append(question)
                        break
            if len(accepted) < count:
                detail = f"；最後錯誤：{last_error}" if last_error else ""
                raise ValueError(
                    f"無法在去重後補足 {document} 的題目數"
                    f"（{len(accepted)}/{count}）{detail}"
                )
            return accepted

        document_items = list(chunks_by_document.items())
        if allow_parallel_generation:
            with ThreadPoolExecutor(max_workers=len(document_items)) as executor:
                document_results = list(executor.map(
                    lambda item: generate_for_document(item[0], item[1]),
                    document_items,
                ))
        else:
            document_results = [
                generate_for_document(document, selected_chunks)
                for document, selected_chunks in document_items
            ]

        questions = []
        for document_questions in document_results:
            for question in document_questions:
                questions.append({**question, "number": len(questions) + 1})
        existing_preferences = dict(existing_evaluation.get("preferences") or {})
        evaluation = {
            **existing_evaluation,
            "preferences": {**existing_preferences, "generation_model": generation_model, "test_model": test_model,
                            "question_count": int(question_count),
                            "retrieval_mode": retrieval_mode, "top_k": int(top_k),
                            "allow_parallel_generation": bool(allow_parallel_generation),
                            "use_reranker": bool(use_reranker),
                            "test_max_concurrent_requests": int(test_max_concurrent_requests)},
            "questions": questions, "results": [], "dirty": False,
        }
        save_project(project_id, {"evaluation": evaluation})
    except (OSError, TypeError, ValueError) as exc:
        return f"❌ {exc}", [], {}, []
    return (
        f"✅ 已從 {len(document_chunks)} 份 PDF 各建立 {count} 道題目，共 {len(questions)} 道。",
        _evaluation_question_rows(questions), evaluation, [],
    )


def run_evaluation_for_ui(
    project_id: str, model_endpoint: str, api_key: str,
    embedding_api_base: str, embedding_api_key: str,
    neo4j_uri: str, neo4j_database: str, neo4j_username: str, neo4j_password: str,
    model: str, retrieval_mode: str, top_k: int, evaluation: dict[str, Any],
    max_concurrent_requests: int = 3,
    use_reranker: bool = True,
    progress=gr.Progress(),
) -> tuple[str, list[list[object]], dict[str, Any]]:
    questions = evaluation.get("questions") if evaluation else None
    if not project_id:
        return "❌ 請先建立或載入專案。", [], evaluation or {}
    if not questions:
        return "❌ 請先建立測試題目。", [], evaluation or {}
    if evaluation.get("dirty"):
        return "❌ 題目或答案尚未完成自動儲存，請稍後再試。", [], evaluation
    if not model:
        return "❌ 請先勾選並選擇回答與評判模型。", [], evaluation
    concurrency = int(max_concurrent_requests)
    if concurrency < 1:
        return "❌ 測試最大並行請求數必須大於 0", [], evaluation
    document_summaries = evaluation.get("document_summaries") or []
    if not document_summaries:
        return "❌ 尚未建立 PDF 路由摘要，請先到「3. PDF 摘要」建立摘要。", [], evaluation

    def evaluate(item: dict[str, Any]) -> dict[str, Any]:
        try:
            routing = select_relevant_documents(
                model_endpoint,
                api_key,
                model,
                item["question"],
                document_summaries,
            )
        except ValueError as exc:
            return {
                **item,
                "selected_documents": [],
                "routing_reason": f"文件路由失敗：{exc}",
                "routing_confidence": 0.0,
                "routing_correct": False,
                "actual_answer": "",
                "passed": False,
                "reason": f"文件路由失敗：{exc}",
            }
        selected_documents = routing["documents"]
        expected_document = str(item.get("document") or "")
        routing_correct = expected_document in selected_documents
        status, actual, evidence_rows = answer_question_for_ui(
            model_endpoint, api_key, embedding_api_base, embedding_api_key,
            neo4j_uri, neo4j_database, neo4j_username, neo4j_password,
            model, item["question"], retrieval_mode, max(int(top_k), 5),
            selected_documents, use_reranker,
        )
        retrieval_rank = _retrieval_rank(item, evidence_rows)
        if status.startswith("✅"):
            try:
                judgment = judge_evaluation_answer(
                    model_endpoint, api_key, model, item["question"],
                    item["expected_answer"], actual,
                )
            except ValueError as exc:
                judgment = {"passed": False, "reason": f"評判失敗：{exc}"}
        else:
            judgment = {"passed": False, "reason": status}
        return {
            **item,
            "selected_documents": selected_documents,
            "routing_reason": routing["reason"],
            "routing_confidence": routing["confidence"],
            "routing_correct": routing_correct,
            "actual_answer": actual,
            "retrieval_rank": retrieval_rank,
            "recall_at_5": retrieval_rank is not None and retrieval_rank <= 5,
            "reciprocal_rank": 1 / retrieval_rank if retrieval_rank else 0.0,
            **judgment,
        }

    results: list[dict[str, Any] | None] = [None] * len(questions)
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(evaluate, item): index
            for index, item in enumerate(questions)
        }
        completed = 0
        for future in as_completed(futures):
            results[futures[future]] = future.result()
            completed += 1
            progress(completed / len(questions), desc=f"已完成 {completed} / {len(questions)} 題")
    results = [result for result in results if result is not None]
    updated = dict(evaluation)
    updated["results"] = results
    try:
        save_project(project_id, {"evaluation": updated})
    except (OSError, ValueError) as exc:
        return f"❌ 測試已完成，但保存失敗：{exc}", _evaluation_result_rows(results), updated
    return _evaluation_summary(results), _evaluation_result_rows(results), updated


def _add_single_document(
    file_path: str,
    parsed_chunk_size: int,
    parsed_chunk_overlap: int,
    documents: list[dict[str, Any]],
    chunks: list[TextChunk],
) -> tuple[str, dict[str, Any] | None, list[TextChunk]]:
    file_name = Path(file_path).name
    if any(doc.get("file_name") == file_name for doc in documents):
        return f"❌「{file_name}」：專案中已有同名文件，請先移除或重新命名後再上傳。", None, []
    try:
        pages, empty_pages = extract_pdf(file_path)
        next_number = max((chunk.number for chunk in chunks), default=0) + 1
        new_chunks = chunk_pages(
            pages, parsed_chunk_size, parsed_chunk_overlap,
            document=file_name, start_number=next_number,
        )
        if not new_chunks:
            raise ValueError("PDF 沒有可解析文字；掃描文件需在後續版本加入 OCR。")
    except (ValueError, TypeError) as exc:
        return f"❌「{file_name}」：{exc}", None, []
    parsed_start, parsed_end = pages[0].page, pages[-1].page
    doc_state = {
        "file_path": file_path,
        "file_name": file_name,
        "page_count": len(pages),
        "page_start": parsed_start,
        "page_end": parsed_end,
        "empty_pages": empty_pages,
        "chunk_count": len(new_chunks),
        "config": {"chunk_size": parsed_chunk_size, "chunk_overlap": parsed_chunk_overlap},
    }
    note = f"✅「{file_name}」第 {parsed_start}–{parsed_end} 頁，產生 {len(new_chunks)} 個 chunk。"
    if empty_pages:
        note += f" 無文字頁面：{', '.join(map(str, empty_pages))}。"
    return note, doc_state, new_chunks


def add_document_for_ui(
    file_paths: list[str] | str | None,
    chunk_size: int,
    chunk_overlap: int,
    documents: list[dict[str, Any]],
    chunks: list[TextChunk],
) -> tuple[
    str, list[list[object]], list[dict[str, Any]], list[TextChunk],
    dict[str, Any], list[TextChunk], str,
    list[list[object]], dict[str, Any], dict[str, Any],
]:
    documents = list(documents or [])
    chunks = list(chunks or [])
    if not file_paths:
        return (
            "請先上傳 PDF。", [], documents, chunks, {}, [],
            "尚未解析任何 PDF。", _document_rows(documents), gr.update(),
            _document_choices(documents),
        )
    try:
        parsed_chunk_size = int(chunk_size)
        parsed_chunk_overlap = int(chunk_overlap)
        if not 100 <= parsed_chunk_size <= 10000:
            raise ValueError("chunk_size 必須介於 100 到 10,000")
        if not 0 <= parsed_chunk_overlap < parsed_chunk_size:
            raise ValueError("chunk_overlap 必須大於等於 0 且小於 chunk_size")
    except (ValueError, TypeError) as exc:
        return (
            f"❌ {exc}", [], documents, chunks, {}, [],
            "無法解析頁面。", _document_rows(documents), gr.update(),
            _document_choices(documents),
        )

    paths = file_paths if isinstance(file_paths, list) else [file_paths]
    notes: list[str] = []
    last_doc_state: dict[str, Any] = {}
    last_new_chunks: list[TextChunk] = []
    for file_path in paths:
        note, doc_state, new_chunks = _add_single_document(
            file_path, parsed_chunk_size, parsed_chunk_overlap, documents, chunks,
        )
        notes.append(note)
        if doc_state is not None:
            documents = documents + [doc_state]
            chunks = chunks + new_chunks
            last_doc_state, last_new_chunks = doc_state, new_chunks

    summary = f"（專案累計 {len(chunks)} 個 chunk，{len(documents)} 份文件）"
    if len(notes) > 1:
        status = "\n".join(f"- {note}" for note in notes) + f"\n\n{summary}"
    else:
        status = notes[0] + summary
    if not last_doc_state:
        return (
            status, [], documents, chunks, {}, [],
            "無法解析頁面。", _document_rows(documents), gr.update(),
            _document_choices(documents),
        )
    return (
        status,
        _chunk_rows(last_new_chunks),
        documents,
        chunks,
        last_doc_state,
        last_new_chunks,
        _document_status(len(documents) - 1, len(documents), last_doc_state, len(last_new_chunks)),
        _document_rows(documents),
        gr.update(value=None),
        _document_choices(documents),
    )


def remove_document_for_ui(
    project_id: str,
    file_names: Any,
    documents: list[dict[str, Any]],
    chunks: list[TextChunk],
) -> tuple[
    str, list[dict[str, Any]], list[TextChunk], dict[str, Any], list[TextChunk],
    str, list[list[object]], dict[str, Any], list[list[object]],
]:
    documents = list(documents or [])
    chunks = list(chunks or [])
    table_rows: Any = file_names
    if isinstance(table_rows, dict) and "data" in table_rows:
        table_rows = table_rows["data"]
    elif hasattr(table_rows, "values") and hasattr(table_rows.values, "tolist"):
        table_rows = table_rows.values.tolist()
    if isinstance(table_rows, list) and table_rows and isinstance(table_rows[0], (list, tuple)):
        names = [
            str(row[1]) for row in table_rows
            if len(row) >= 2 and bool(row[0]) and row[1]
        ]
    else:
        values = table_rows if isinstance(table_rows, list) else [table_rows]
        names = [str(name) for name in values if isinstance(name, str) and name]
    if not names:
        return (
            "請先選擇要移除的 PDF。", documents, chunks, {}, [],
            "請先解析 PDF。", _document_rows(documents),
            _document_choices(documents), [],
        )
    if project_id:
        for file_name in names:
            try:
                remove_document(project_id, file_name)
            except (OSError, ValueError) as exc:
                return (
                    f"❌ {exc}", documents, chunks, {}, [],
                    "請先解析 PDF。", _document_rows(documents),
                    _document_choices(documents), [],
                )
    name_set = set(names)
    documents = [doc for doc in documents if doc.get("file_name") not in name_set]
    chunks = [chunk for chunk in chunks if chunk.document not in name_set]
    active_preview = documents[-1] if documents else {}
    active_chunks = [
        chunk for chunk in chunks if chunk.document == active_preview.get("file_name")
    ] if active_preview else []
    status = (
        f"✅ 已移除 {len(names)} 份文件「{'、'.join(names)}」"
        f"（專案剩餘 {len(chunks)} 個 chunk、{len(documents)} 份文件）。"
    )
    document_status = (
        _document_status(len(documents) - 1, len(documents), active_preview, len(active_chunks))
        if documents else "請先解析 PDF。"
    )
    return (
        status,
        documents,
        chunks,
        active_preview,
        active_chunks,
        document_status,
        _document_rows(documents),
        _document_choices(documents),
        _chunk_rows(active_chunks),
    )


def switch_document(
    offset: int,
    documents: list[dict[str, Any]],
    chunks: list[TextChunk],
    active: dict[str, Any],
) -> tuple[dict[str, Any], list[TextChunk], list[list[object]], str]:
    documents = documents or []
    if not documents:
        return {}, [], [], "尚未解析任何 PDF。"
    names = [doc.get("file_name", "") for doc in documents]
    current_name = (active or {}).get("file_name", "")
    current_index = names.index(current_name) if current_name in names else len(documents) - 1
    new_index = (current_index + offset) % len(documents)
    doc = documents[new_index]
    doc_chunks = [chunk for chunk in (chunks or []) if chunk.document == doc.get("file_name")]
    return (
        doc, doc_chunks, _chunk_rows(doc_chunks),
        _document_status(new_index, len(documents), doc, len(doc_chunks)),
    )


def previous_document(
    documents: list[dict[str, Any]], chunks: list[TextChunk], active: dict[str, Any]
) -> tuple[dict[str, Any], list[TextChunk], list[list[object]], str]:
    return switch_document(-1, documents, chunks, active)


def next_document(
    documents: list[dict[str, Any]], chunks: list[TextChunk], active: dict[str, Any]
) -> tuple[dict[str, Any], list[TextChunk], list[list[object]], str]:
    return switch_document(1, documents, chunks, active)


def save_config(state: dict[str, Any]) -> tuple[str, str | None]:
    if not state:
        return "請先成功預覽 PDF。", None
    output = write_json(Path("data/exports/latest-config.json"), state)
    return f"設定已儲存：{output}", str(output)


def _select_schema_planning_chunks(
    chunks: list[TextChunk], sampling_mode: str, sample_page_count: int
) -> tuple[list[TextChunk], list[int]]:
    if not chunks:
        raise ValueError("請先在 PDF 頁面解析並產生 chunks")
    available_pages = sorted({page for chunk in chunks for page in chunk.pages})
    if sampling_mode == "全部頁面":
        return chunks, available_pages
    if sampling_mode != "隨機抽取 N 頁":
        raise ValueError("不支援的 Schema 規劃範圍")
    sample_page_count = int(sample_page_count)
    if sample_page_count < 1:
        raise ValueError("隨機抽取頁數必須至少為 1")
    if sample_page_count > len(available_pages):
        raise ValueError(
            f"隨機抽取頁數不可超過可用頁數 {len(available_pages)}"
        )
    sampled_pages = sorted(random.sample(available_pages, sample_page_count))
    sampled_page_set = set(sampled_pages)
    selected_chunks = [
        chunk for chunk in chunks if sampled_page_set.intersection(chunk.pages)
    ]
    return selected_chunks, sampled_pages


def plan_schema_for_ui(
    model_endpoint: str,
    api_key: str,
    llm_model: str,
    temperature: float,
    schema_granularity: str,
    max_concurrent_requests: int,
    sampling_mode: str,
    sample_page_count: int,
    chunks: list[TextChunk],
    run_control: RunControl,
    progress=gr.Progress(),
) -> tuple[str, str]:
    if not llm_model:
        return "❌ 請先勾選並選擇 Schema 規劃 LLM。", ""
    run_control.reset()
    try:
        planning_chunks, selected_pages = _select_schema_planning_chunks(
            chunks, sampling_mode, sample_page_count
        )
        plan = plan_graph_schema(
            model_endpoint,
            api_key,
            llm_model,
            planning_chunks,
            float(temperature),
            lambda value, description: progress(value, desc=description),
            schema_granularity,
            int(max_concurrent_requests),
            control=run_control,
        )
    except RunCancelled:
        return "⏹ 已停止（使用者中止 Schema 規劃）。", ""
    except ValueError as exc:
        return f"❌ {exc}", ""
    if sampling_mode == "全部頁面":
        scope_note = f"全部 {len(selected_pages)} 頁"
    else:
        sampled_page_text = ", ".join(map(str, selected_pages))
        scope_note = (
            f"隨機抽取 {len(selected_pages)} 頁（頁碼：{sampled_page_text}）"
        )
    note = (
        f"✅ 已使用 {llm_model} 規劃 schema；參考 {scope_note}、"
        f"{plan.analyzed_chunks} 個 chunk，"
        f"共 {plan.batch_count} 批、{plan.merge_rounds} 輪整合。"
        f"粒度：{schema_granularity}。"
        f"最大並行請求數：{int(max_concurrent_requests)}。"
        "請確認或編輯後再進行抽取。"
    )
    return note, json.dumps(plan.schema, ensure_ascii=False, indent=2)


def extract_graph_for_ui(
    model_endpoint: str,
    api_key: str,
    llm_model: str,
    temperature: float,
    max_concurrent_requests: int,
    chunks: list[TextChunk],
    schema_text: str,
    documents: list[dict[str, Any]],
    run_control: RunControl,
    progress=gr.Progress(),
) -> tuple[str, list[list[object]], list[list[object]], dict[str, Any]]:
    if not llm_model:
        return "❌ 請先勾選並選擇知識圖譜抽取 LLM。", [], [], {}
    run_control.reset()
    try:
        raw_schema = json.loads(schema_text)
        if not isinstance(raw_schema, dict):
            raise ValueError("schema 必須是 JSON 物件")
        schema = validate_schema(raw_schema)
        extraction = extract_graph(
            model_endpoint,
            api_key,
            llm_model,
            chunks,
            schema,
            float(temperature),
            int(max_concurrent_requests),
            lambda value, description: progress(value, desc=description),
            control=run_control,
        )
    except RunCancelled:
        return "⏹ 已停止（使用者中止抽取）。", [], [], {}
    except json.JSONDecodeError:
        return "❌ schema 不是有效 JSON。", [], [], {}
    except (ValueError, RuntimeError) as exc:
        return f"❌ {exc}", [], [], {}

    entity_rows = [
        [
            item["name"],
            item["type"],
            item["description"],
            *_source_display(item),
        ]
        for item in extraction.entities
    ]
    relationship_rows = [
        [
            item["source"],
            item["type"],
            item["target"],
            item["description"],
            *_source_display(item),
        ]
        for item in extraction.relationships
    ]
    run_id = str(uuid4())
    document_name = "、".join(doc.get("file_name", "") for doc in (documents or []))
    graph_state = {
        "run_id": run_id,
        "document": document_name,
        "llm_model": llm_model,
        "temperature": float(temperature),
        "max_concurrent_requests": int(max_concurrent_requests),
        "schema": schema,
        "entities": extraction.entities,
        "relationships": extraction.relationships,
        "chunks": [
            {
                "number": chunk.number, "text": chunk.text,
                "pages": list(chunk.pages), "document": chunk.document,
            }
            for chunk in chunks
        ],
    }
    graph_state["neo4j_imported"] = False
    status = (
        f"✅ 已處理 {extraction.processed_chunks} 個 chunk，抽取 "
        f"{len(extraction.entities)} 個實體與 "
        f"{len(extraction.relationships)} 筆關係（最大並行請求數：{int(max_concurrent_requests)}）。請確認結果後進行 Embedding 並匯入 Neo4j。"
    )
    return status, entity_rows, relationship_rows, graph_state


def request_stop_for_ui(run_control: RunControl) -> str:
    run_control.request_stop()
    return "⏹ 已送出停止要求，正在等待目前批次結束…"


def toggle_pause_for_ui(run_control: RunControl) -> tuple[str, dict[str, Any]]:
    paused = run_control.toggle_pause()
    if paused:
        return (
            "⏸ 已暫停：目前批次會跑完，但不會再送出新的批次。",
            gr.update(value="▶ 繼續"),
        )
    return "▶ 已繼續，將開始送出新的批次。", gr.update(value="⏸ 暫停")


def _build_graph_evidence(
    entities: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    def source_variants(item: dict[str, Any]) -> list[dict[str, Any]]:
        references = item.get("source_references") or []
        if not references:
            return [{
                "source_pages": item.get("source_pages", []),
                "source_chunk_numbers": item.get("source_chunk_numbers", []),
                "source_documents": item.get("source_documents", []),
                "source_references": [],
            }]
        return [{
            "source_pages": reference.get("pages", []),
            "source_chunk_numbers": reference.get("chunk_numbers", []),
            "source_documents": [reference.get("document", "")],
            "source_references": [reference],
        } for reference in references]

    evidence = []
    for item in chunks:
        number = int(item.get("number", 0))
        document = str(item.get("document", ""))
        evidence.append({
            "evidence_id": f"chunk-{number}", "kind": "原文",
            "name": "", "source": "", "target": "",
            "text": str(item.get("text", "")),
            "source_pages": item.get("pages", []),
            "source_chunk_numbers": [number],
            "source_documents": [document] if document else [],
            "source_references": [{
                "document": document,
                "chunk_numbers": [number],
                "pages": item.get("pages", []),
            }],
        })
    for index, item in enumerate(entities):
        for source_index, sources in enumerate(source_variants(item)):
            evidence.append({
                "evidence_id": f"entity-{index}-{source_index}", "kind": "實體",
                "name": item.get("name", ""), "source": "", "target": "",
                "text": "實體：{}；類型：{}；說明：{}".format(
                    item.get("name", ""), item.get("type", ""), item.get("description", "")
                ),
                **sources,
            })
    for index, item in enumerate(relationships):
        for source_index, sources in enumerate(source_variants(item)):
            evidence.append({
                "evidence_id": f"relationship-{index}-{source_index}",
                "kind": "關係", "name": "",
                "source": item.get("source", ""), "target": item.get("target", ""),
                "text": "關係：{} -[{}]-> {}；說明：{}".format(
                    item.get("source", ""), item.get("type", ""),
                    item.get("target", ""), item.get("description", "")
                ),
                **sources,
            })
    return evidence


def import_graph_for_ui(
    embedding_api_base: str,
    embedding_api_key: str,
    neo4j_uri: str,
    neo4j_database: str,
    neo4j_username: str,
    neo4j_password: str,
    embedding_model: str,
    graph_state: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    if not graph_state or not graph_state.get("run_id"):
        return "❌ 請先完成知識圖譜抽取。", graph_state or {}
    if not (embedding_model or "").strip():
        return "❌ 請選擇 Embedding 模型。", graph_state
    updated_state = dict(graph_state)
    updated_state["embedding_model"] = embedding_model.strip()
    try:
        evidence = _build_graph_evidence(
            updated_state["entities"],
            updated_state["relationships"],
            updated_state.get("chunks", []),
        )
        if not evidence:
            raise ValueError("沒有可建立向量索引的原文、實體或關係")
        vectors = embedding_vectors(
            embedding_api_base, embedding_api_key, embedding_model,
            [item["text"] for item in evidence],
        )
        for item, vector in zip(evidence, vectors):
            item["embedding"] = vector
        updated_state["embedding_dimensions"] = len(vectors[0])
        updated_state["vector_index_name"] = vector_index_name(len(vectors[0]))
        imported = import_extraction(
            neo4j_uri,
            neo4j_database,
            neo4j_username,
            neo4j_password,
            updated_state["run_id"],
            updated_state.get("document", ""),
            updated_state.get("llm_model", ""),
            updated_state.get("embedding_model", ""),
            updated_state["schema"],
            updated_state["entities"],
            updated_state["relationships"],
            evidence,
        )
    except (KeyError, ValueError) as exc:
        updated_state["neo4j_imported"] = False
        updated_state["neo4j_error"] = str(exc)
        return f"❌ {exc}", updated_state

    updated_state["neo4j_imported"] = True
    updated_state.pop("neo4j_error", None)
    return (
        f"✅ 已清空本工具既有圖譜；建立 {len(evidence)} 筆向量證據、Vector Index 與 Full-text Index；已匯入 Neo4j {imported.entity_count} 個實體與 "
        f"{imported.relationship_count} 筆關係。",
        updated_state,
    )


def answer_question_for_ui(
    model_endpoint: str,
    api_key: str,
    embedding_api_base: str,
    embedding_api_key: str,
    neo4j_uri: str,
    neo4j_database: str,
    neo4j_username: str,
    neo4j_password: str,
    answer_model: str,
    question: str,
    retrieval_mode: str,
    top_k: int,
    document_names: list[str] | None = None,
    use_reranker: bool = True,
) -> tuple[str, str, list[list[object]]]:
    if not answer_model:
        return "❌ 請先勾選並選擇問答 LLM。", "", []
    if not question.strip():
        return "請輸入問題。", "", []
    try:
        graph_state = load_latest_graph(
            neo4j_uri, neo4j_database, neo4j_username, neo4j_password
        )
        question_vector = embedding_vectors(
            embedding_api_base, embedding_api_key, graph_state.get("embedding_model", ""),
            [question.strip()],
        )[0]
        evidence = search_graph_evidence(
            neo4j_uri, neo4j_database, neo4j_username, neo4j_password,
            graph_state["run_id"], question.strip(), question_vector,
            retrieval_mode, int(top_k), document_names,
            candidate_top_k=(
                min(int(top_k) * RERANK_CANDIDATE_MULTIPLIER, RERANK_MAX_CANDIDATES)
                if use_reranker else int(top_k)
            ),
        )
        if use_reranker:
            evidence = rerank_evidence(question, evidence, int(top_k))
        else:
            evidence = evidence[:int(top_k)]
        result = answer_graph_question(
            model_endpoint, api_key, answer_model, question, retrieval_mode, evidence, document_names
        )
    except ValueError as exc:
        return f"❌ {exc}", "", []
    rows = [
        [
            item["kind"],
            item["text"],
            ", ".join(item.get("matched_by", [])),
            f"{item.get('fusion_score', 0.0):.4f}",
            _source_display(item)[1],
            _source_display(item)[0],
            _source_display(item)[2],
        ]
        for item in result["evidence"]
    ]
    record = {
        "run_id": graph_state.get("run_id"),
        "document": graph_state.get("document", ""),
        "answer_model": answer_model,
        "embedding_model": graph_state.get("embedding_model", ""),
        "question": question.strip(),
        "retrieval_mode": retrieval_mode,
        "top_k": int(top_k),
        "answer": result["answer"],
        "evidence": result["evidence"],
    }
    output = write_json(Path("data/qa") / f"{uuid4()}.json", record)
    status = (
        f"✅ {retrieval_mode} 已使用 {len(rows)} 筆證據完成回答；"
        f"文件：{graph_state.get('document', '未知')}；紀錄：{output}。"
    )
    return status, result["answer"], rows


def _current_project_banner(project: dict[str, Any]) -> str:
    name = (project or {}).get("name")
    return f"### 📁 目前專案：{name}" if name else "### 📁 目前專案：尚未選擇"


def build_app() -> gr.Blocks:
    env = load_env()
    llm_settings = load_service_settings("llm", env)
    embedding_settings = load_service_settings("embedding", env)
    llm_choices = service_choice_items(llm_settings)
    embedding_choices = service_choice_items(embedding_settings)
    embedding_allowed = service_choices(embedding_settings)
    llm_profile = llm_settings["profiles"][llm_settings["active"]]
    embedding_profile = embedding_settings["profiles"][embedding_settings["active"]]
    preferred_llm = preferred_service_model(llm_settings)
    initial_llm_credentials = [
        resolve_model_credentials_for_ui(llm_settings, preferred_llm)
        for _ in llm_profile["models"]
    ]
    initial_embedding_credentials = resolve_model_credentials_for_ui(
        embedding_settings, embedding_profile["models"][0]
    )
    with gr.Blocks(title="PDF GraphRAG 測試工具", fill_width=True) as app:
        gr.Markdown(
            "# PDF GraphRAG 測試工具\n"
            "上傳使用手冊、調整建圖參數，並測試 Neo4j GraphRAG。"
        )
        current_project_banner = gr.Markdown(_current_project_banner({}))
        llm_service_state = gr.State(llm_settings)
        embedding_service_state = gr.State(embedding_settings)
        project_state = gr.State({})
        documents_state = gr.State([])
        active_preview_state = gr.State({})
        active_chunks_state = gr.State([])
        chunk_state = gr.State([])
        graph_state = gr.State({})
        evaluation_state = gr.State({})
        run_control_state = gr.State(RunControl())
        summary_run_control_state = gr.State(RunControl())
        neo4j_connected_state = gr.State(False)

        with gr.Tab("0. 專案設定") as project_tab:
            gr.Markdown("### 專案工作區\n建立或載入專案後，可保存本頁面所有連線、模型、參數、Chunk、文件、建圖狀態與問答紀錄。")
            with gr.Row():
                project_selector = gr.Dropdown(
                    choices=_project_choices(), label="現有專案", interactive=True
                )
                load_project_button = gr.Button("載入專案", variant="primary")
            with gr.Row():
                new_project_name = gr.Textbox(label="新專案名稱", placeholder="例如：ALCX17 使用手冊")
                create_project_button = gr.Button("建立新專案", variant="primary")
                delete_project_button = gr.Button("刪除專案", variant="stop")
                delete_project_completed = gr.State(False)
            project_status = gr.Markdown("尚未選擇專案；載入後，設定與處理結果都會自動保存。")
            gr.Markdown("⚠️ 專案設定保存在本機 `data/projects/`，其中 Neo4j Password 為明文；模型 API Key 僅保存在 `.env`。")

        with gr.Tab("1. 連線設定"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Neo4j")
                    neo4j_uri = gr.Textbox(label="URI", value=env["NEO4J_URI"])
                    neo4j_database = gr.Textbox(label="Database", value=env["NEO4J_DATABASE"])
                    neo4j_username = gr.Textbox(label="Username", value=env["NEO4J_USERNAME"])
                    neo4j_password = gr.Textbox(label="Password", value=env["NEO4J_PASSWORD"], type="password")
                    neo4j_test_button = gr.Button("測試 Neo4j 連線", variant="primary")
                    neo4j_connection_status = gr.Markdown()
                with gr.Column():
                    gr.Markdown("### 模型服務（對話／建圖用）")
                    llm_provider = gr.Radio(["OpenAI", "Ollama"], value=llm_settings["active"], label="模型服務來源")
                    model_endpoint = gr.Textbox(label="API Base URL", value=llm_profile["base_url"])
                    api_key = gr.Textbox(
                        label="API Key（Ollama 免填）", value=llm_profile["api_key"], type="password"
                    )
                    model_test_button = gr.Button("測試模型服務連線", variant="primary", visible=llm_settings["active"] == "OpenAI")
                    model_list_button = gr.Button("獲得模型清單", variant="primary", visible=llm_settings["active"] == "Ollama")
                    llm_models_table = gr.Dataframe(
                        value=llm_profile["rows"], visible=llm_settings["active"] == "Ollama",
                        headers=["使用", "模型名稱"], datatype=["bool", "str"],
                        type="array", interactive=True, static_columns=[1],
                        row_count=(0, "fixed"), col_count=(2, "fixed"),
                        label="LLM 模型清單", column_widths=[80, "80%"],
                    )
                    model_connection_status = gr.Markdown(llm_profile["status"])
                    gr.Markdown("### Embedding 服務")
                    embedding_provider = gr.Radio(["OpenAI", "Ollama", "Voyage"], value=embedding_settings["active"], label="Embedding 服務來源")
                    embedding_api_base = gr.Textbox(
                        label="Embedding API Base URL", value=embedding_profile["base_url"]
                    )
                    embedding_api_key = gr.Textbox(
                        label="Embedding API Key（Ollama 免填）",
                        value=embedding_profile["api_key"], type="password",
                    )
                    embedding_test_button = gr.Button("測試 Embedding 服務連線", variant="primary", visible=embedding_settings["active"] == "OpenAI")
                    embedding_list_button = gr.Button("獲得 Embedding 模型清單", variant="primary", visible=embedding_settings["active"] == "Ollama")
                    embedding_models_table = gr.Dataframe(
                        value=embedding_profile["rows"], visible=embedding_settings["active"] == "Ollama",
                        headers=["使用", "模型名稱"], datatype=["bool", "str"],
                        type="array", interactive=True, static_columns=[1],
                        row_count=(0, "fixed"), col_count=(2, "fixed"),
                        label="Embedding 模型清單", column_widths=[80, "80%"],
                    )
                    embedding_connection_status = gr.Markdown(embedding_profile["status"])
                    reload_button = gr.Button("重新讀取 .env")
            gr.Markdown("⚠️ Password、API Base URL 與 API Key 會寫入本機 `.env`；模型清單與選擇保存在 `config/model_settings.yaml`。")
            env_status = gr.Markdown("啟動時已讀取 `.env` 與模型 YAML；欄位修改後會自動儲存。")

        with gr.Tab("2. PDF 與參數", interactive=False) as pdf_tab:
            with gr.Row():
                with gr.Column(scale=1):
                    pdf_file = gr.File(
                        label="PDF 使用手冊（可一次選取多個檔案）",
                        file_types=[".pdf"], file_count="multiple", type="filepath",
                    )
                    chunk_size = gr.Slider(100, 10000, value=1500, step=100, label="Chunk size（字元）")
                    chunk_overlap = gr.Slider(0, 2000, value=200, step=50, label="Chunk overlap（字元）")
                    preview_button = gr.Button("解析並加入專案", variant="primary")
                    export_button = gr.Button("匯出目前設定")
                    export_file = gr.File(label="設定 JSON", interactive=False)
                    gr.Markdown("##### 已加入本專案的 PDF")
                    documents_table = gr.Dataframe(
                        headers=["選取", "文件", "頁碼範圍", "Chunk 數"],
                        datatype=["bool", "str", "str", "number"],
                        interactive=True, static_columns=[1, 2, 3],
                        row_count=(0, "fixed"), col_count=(4, "fixed"),
                        wrap=True,
                    )
                    remove_document_selector = gr.Dropdown(
                        label="選擇要移除的 PDF（可多選）", choices=[],
                        multiselect=True, interactive=True, visible=False,
                    )
                    remove_document_button = gr.Button("移除選定的 PDF", variant="stop")
                with gr.Column(scale=3):
                    preview_status = gr.Markdown("尚未解析 PDF。可重複上傳多份 PDF，逐一加入同一個專案。")
                    with gr.Row():
                        previous_button = gr.Button("上一份文件", scale=1)
                        next_button = gr.Button("下一份文件", scale=1)
                    page_status = gr.Markdown("請先解析 PDF。")
                    chunk_table = gr.Dataframe(
                        headers=["編號", "文件", "頁碼", "字元數", "內容"],
                        datatype=["number", "str", "str", "number", "str"],
                        interactive=False,
                        wrap=True,
                        max_height=750,
                        column_widths=[80, 160, 120, 100, 900],
                    )

        with gr.Tab("3. PDF 摘要", interactive=False) as summary_tab:
            gr.Markdown(
                "### 建立 PDF 路由摘要\n"
                "為每份 PDF 建立可區分來源之識別資訊與主題的摘要，供自動測試時由 LLM 選擇檢索文件。"
            )
            with gr.Row():
                summary_model = gr.Dropdown(
                    choices=llm_choices, value=preferred_llm, allow_custom_value=False,
                    label="摘要模型",
                )
                summary_max_concurrent_requests = gr.Number(
                    value=3, minimum=1, precision=0, label="摘要最大並行請求數"
                )
            generate_summaries_button = gr.Button("建立／重新建立全部 PDF 摘要", variant="primary")
            with gr.Row():
                summary_pause_button = gr.Button("⏸ 暫停")
                summary_stop_button = gr.Button("⏹ 停止", variant="stop")
            summary_run_control_status = gr.Markdown(
                "「暫停」「停止」在建立摘要執行中可使用：暫停只會停止送出新請求"
                "（已送出的請求仍會跑完）；停止會盡快中止整個流程。"
            )
            summary_status = gr.Markdown("尚未建立 PDF 摘要。")
            document_summaries_table = gr.Dataframe(
                headers=["文件", "摘要", "文件識別資訊", "主題", "關鍵詞"],
                datatype=["str", "str", "str", "str", "str"],
                interactive=False, wrap=True,
            )

        with gr.Tab("4. 建圖", interactive=False) as graph_tab:
            gr.Markdown("### 規劃並抽取知識圖譜")
            with gr.Row():
                pause_button = gr.Button("⏸ 暫停")
                stop_button = gr.Button("⏹ 停止", variant="stop")
            run_control_status = gr.Markdown(
                "「暫停」「停止」在下方「規劃 Schema」或「確認 Schema 並抽取」"
                "執行中都可使用：暫停只會停止送出新批次（已送出的批次仍會跑完）；"
                "停止會盡快中止整個流程。"
            )
            with gr.Group():
                gr.Markdown("#### ① 規劃 Schema")
                gr.Markdown(
                    "選擇 LLM，從上一頁產生的 chunks 規劃實體與關係類型。"
                )
                with gr.Row():
                    graph_llm_model = gr.Dropdown(
                        choices=llm_choices,
                        value=preferred_llm,
                        allow_custom_value=False,
                        label="Schema 規劃 LLM",
                    )
                    graph_temperature = gr.Slider(
                        0, 2, value=0, step=0.1, label="Temperature"
                    )
                with gr.Row():
                    schema_granularity = gr.Radio(
                        ["粗略", "平衡", "詳細"],
                        value="平衡",
                        label="Schema 粒度",
                    )
                    max_concurrent_requests = gr.Number(
                        value=3, minimum=1, precision=0, label="最大並行請求數"
                    )
                with gr.Row():
                    schema_sampling_mode = gr.Radio(
                        ["全部頁面", "隨機抽取 N 頁"],
                        value="全部頁面",
                        label="Schema 規劃範圍",
                    )
                    schema_sample_page_count = gr.Number(
                        value=10, minimum=1, precision=0, label="隨機抽取頁數 N", visible=False
                    )
                plan_schema_button = gr.Button(
                    "分析文件並規劃 Schema", variant="primary"
                )
                plan_status = gr.Markdown("請先在 PDF 頁面解析並產生 chunks。")
                gr.HTML(
                    """
                    <style>
                    .schema-scroll-editor .cm-content {
                        font-size: 17px;
                        line-height: 1.6;
                    }
                    </style>
                    """,
                    padding=False,
                )
                schema_editor = gr.Code(
                    label="實體與關係 Schema（可編輯 JSON）",
                    language="json",
                    interactive=True,
                    lines=18,
                    max_lines=18,
                    wrap_lines=False,
                    elem_classes="schema-scroll-editor",
                )

            with gr.Group():
                gr.Markdown("#### ② 確認 Schema 並抽取知識圖譜")
                gr.Markdown(
                    "確認上方 JSON 後執行全部 chunks；檢查抽取結果後，再手動匯入 Neo4j。"
                )
                extraction_llm_model = gr.Dropdown(
                    choices=llm_choices,
                    value=preferred_llm,
                    allow_custom_value=False,
                    label="知識圖譜抽取 LLM",
                )
                extraction_max_concurrent_requests = gr.Number(
                    value=3,
                    minimum=1,
                    precision=0,
                    label="最大並行請求數",
                )
                generate_graph_button = gr.Button(
                    "確認 Schema 並抽取", variant="primary"
                )
                build_status = gr.Markdown("尚未執行抽取。")
                gr.Markdown("##### 抽取結果")
                entity_table = gr.Dataframe(
                    headers=["實體", "類型", "說明", "來源 Chunks", "來源頁碼", "來源文件"],
                    interactive=False,
                    wrap=True,
                )
                relationship_table = gr.Dataframe(
                    headers=["來源實體", "關係", "目標實體", "說明", "來源 Chunks", "來源頁碼", "來源文件"],
                    interactive=False,
                    wrap=True,
                )

            with gr.Group():
                gr.Markdown("#### ③ Embedding 並匯入 Neo4j")
                gr.Markdown(
                    "確認上方抽取結果後，選擇 Embedding 模型並匯入 Neo4j。"
                )
                graph_embedding_model = gr.Dropdown(
                    choices=embedding_choices,
                    value=embedding_profile["models"][0] if embedding_profile["models"][0] in embedding_allowed else None,
                    allow_custom_value=False,
                    label="Embedding 模型",
                )
                gr.Markdown(
                    "⚠️ 每次匯入都會先清空本工具在目前 Neo4j Database 中建立的圖譜，再寫入本次結果。"
                )
                import_graph_button = gr.Button(
                    "Embedding 並匯入 Neo4j", variant="primary"
                )
                import_status = gr.Markdown("尚未執行 Embedding 與匯入。")

        with gr.Tab("5. 自動問答測試", interactive=False) as evaluation_tab:
            gr.Markdown(
                "### 從 PDF 自動建立問答測試集\n"
                "每份 PDF 建立指定數量的題目與標準答案，再一鍵執行目前的 RAG 並由模型判斷答案是否正確。"
                "勾選允許並行時，不同 PDF 可同時生題，但同一份 PDF 同時只會送出一個請求；未勾選時全部依序處理。"
            )
            with gr.Group():
                gr.Markdown("#### 生題設定")
                with gr.Row():
                    evaluation_generation_model = gr.Dropdown(
                        choices=llm_choices,
                        value=preferred_llm,
                        allow_custom_value=False,
                        label="生題模型",
                    )
                    evaluation_question_count = gr.Number(value=10, minimum=1, maximum=100, precision=0, label="每份 PDF 題目數 N")
                    evaluation_allow_parallel_generation = gr.Checkbox(
                        value=False, label="允許並行"
                    )
                generate_evaluation_button = gr.Button("從 PDF 建立題目與答案", variant="primary")
            with gr.Group():
                gr.Markdown("#### 測試模型設定")
                with gr.Row():
                    evaluation_test_model = gr.Dropdown(
                        choices=llm_choices,
                        value=preferred_llm,
                        allow_custom_value=False,
                        label="回答與評判模型",
                    )
                    evaluation_retrieval_mode = gr.Radio(["基本檢索", "關聯擴展檢索"], value="關聯擴展檢索", label="檢索模式")
                    evaluation_top_k = gr.Slider(1, 50, value=8, step=1, label="Top K")
                    evaluation_use_reranker = gr.Checkbox(
                        value=True, label="使用 Reranker"
                    )
                    evaluation_test_max_concurrent_requests = gr.Number(value=3, minimum=1, precision=0, label="測試最大並行請求數")
                run_evaluation_button = gr.Button("一鍵測試", variant="primary")
            with gr.Row():
                evaluation_import_file = gr.File(
                    label="匯入題目（JSON／CSV）", file_types=[".json", ".csv"], type="filepath"
                )
                import_evaluation_button = gr.Button("匯入題目")
                export_evaluation_button = gr.Button("匯出題目")
                evaluation_export_file = gr.File(label="題目 JSON", interactive=False)
            gr.HTML(
                """<style>
                .evaluation-metrics-box {
                    border: 2px solid var(--border-color-primary) !important;
                    border-radius: 12px !important;
                    padding: 16px 22px !important;
                    margin: 18px 0 12px !important;
                    background: var(--background-fill-secondary) !important;
                }
                .evaluation-metrics {font-size: 24px !important; line-height: 1.7 !important;}
                .evaluation-table table {font-size: 18px !important;}
                .evaluation-table td, .evaluation-table th {padding: 10px !important;}
                .evaluation-results-table table,
                .evaluation-results-table td,
                .evaluation-results-table th {font-size: 14px !important;}
                </style>""",
                padding=False,
            )
            gr.Markdown("#### 測試題目")
            evaluation_questions_table = gr.Dataframe(
                headers=["編號", "問題", "標準答案", "來源頁碼", "來源文件"],
                datatype=["number", "str", "str", "str", "str"],
                type="array", interactive=True, wrap=True,
                elem_classes="evaluation-table",
            )
            with gr.Group(elem_classes="evaluation-metrics-box"):
                evaluation_status = gr.Markdown(
                    "請先載入專案並解析 PDF。", elem_classes="evaluation-metrics"
                )
            gr.Markdown("#### 測試結果")
            evaluation_results_table = gr.Dataframe(
                headers=["編號", "問題", "標準答案", "預期 PDF", "選定 PDF", "路由", "實際答案", "答案結果", "評判理由"],
                interactive=False, wrap=True,
                elem_classes=["evaluation-table", "evaluation-results-table"],
            )

        with gr.Tab("6. 問答測試", interactive=False) as qa_tab:
            gr.Markdown(
                "直接使用連線設定中的 Neo4j；預設查詢最近更新的建圖結果。"
                "可選擇是否由本機 Reranker 重排 Hybrid Search 候選。"
            )
            answer_model = gr.Dropdown(
                choices=llm_choices,
                value=preferred_llm,
                allow_custom_value=False,
                label="問答 LLM",
            )
            question = gr.Textbox(label="問題", placeholder="例如：設備出現 E01 時該如何處理？")
            with gr.Row():
                retrieval_mode = gr.Radio(["基本檢索", "關聯擴展檢索"], value="關聯擴展檢索", label="檢索模式")
                top_k = gr.Slider(1, 50, value=8, step=1, label="Top K")
                use_reranker = gr.Checkbox(value=True, label="使用 Reranker")
            ask_button = gr.Button("送出問題", variant="primary")
            answer_status = gr.Markdown()
            gr.HTML(
                """
                <style>
                .answer-panel {
                    border: 2px solid var(--border-color-primary);
                    border-radius: 12px;
                    padding: 18px 22px;
                    background: var(--background-fill-secondary);
                }
                .answer-content,
                .answer-content p,
                .answer-content li {
                    font-size: 20px !important;
                    line-height: 1.75 !important;
                }
                </style>
                """,
                padding=False,
            )
            with gr.Group(elem_classes="answer-panel"):
                gr.Markdown("### 回答")
                answer = gr.Markdown(elem_classes="answer-content")
            gr.Markdown("### 檢索來源")
            answer_sources = gr.Dataframe(
                headers=[
                    "類型", "證據", "Retriever", "官方混合分數",
                    "來源頁碼", "來源 Chunks", "來源文件",
                ],
                interactive=False,
                wrap=True,
            )

        with gr.Tab("7. 歷史紀錄", interactive=False) as history_tab:
            gr.Markdown("目前專案的問答紀錄；成功問答後會自動追加並保存。")
            project_history_status = gr.Markdown()
            history_table = gr.Dataframe(
                headers=["時間", "問題", "回答", "模式", "文件"],
                interactive=False, wrap=True,
            )

        schema_model_endpoint = gr.State(initial_llm_credentials[0][0])
        schema_model_key = gr.State(initial_llm_credentials[0][1])
        extraction_model_endpoint = gr.State(initial_llm_credentials[1][0])
        extraction_model_key = gr.State(initial_llm_credentials[1][1])
        summary_model_endpoint = gr.State(initial_llm_credentials[5][0])
        summary_model_key = gr.State(initial_llm_credentials[5][1])
        generation_model_endpoint = gr.State(initial_llm_credentials[2][0])
        generation_model_key = gr.State(initial_llm_credentials[2][1])
        evaluation_model_endpoint = gr.State(initial_llm_credentials[3][0])
        evaluation_model_key = gr.State(initial_llm_credentials[3][1])
        answer_model_endpoint = gr.State(initial_llm_credentials[4][0])
        answer_model_key = gr.State(initial_llm_credentials[4][1])
        selected_embedding_endpoint = gr.State(initial_embedding_credentials[0])
        selected_embedding_key = gr.State(initial_embedding_credentials[1])
        qa_document_filter = gr.State(None)

        summary_tab.select(
            load_document_summaries_for_ui, inputs=project_selector,
            outputs=[document_summaries_table, summary_status],
        )
        generate_summaries_button.click(
            generate_document_summaries_for_ui,
            inputs=[project_selector, summary_model_endpoint, summary_model_key,
                    summary_model, chunk_state, summary_max_concurrent_requests,
                    summary_run_control_state],
            outputs=[summary_status, document_summaries_table, evaluation_state],
            show_progress="minimal",
        )
        summary_pause_button.click(
            toggle_pause_for_ui,
            inputs=[summary_run_control_state],
            outputs=[summary_run_control_status, summary_pause_button],
            queue=False,
        )
        summary_stop_button.click(
            request_stop_for_ui,
            inputs=[summary_run_control_state],
            outputs=[summary_run_control_status],
            queue=False,
        )

        evaluation_tab.select(
            load_evaluation_with_services_for_ui, inputs=[project_selector, llm_service_state],
            outputs=[evaluation_state, evaluation_questions_table, evaluation_results_table,
                     evaluation_generation_model, evaluation_test_model, evaluation_question_count,
                     evaluation_retrieval_mode, evaluation_top_k, evaluation_allow_parallel_generation,
                     evaluation_use_reranker,
                     evaluation_test_max_concurrent_requests, evaluation_status],
        )
        evaluation_preference_inputs = [
            project_selector, evaluation_generation_model, evaluation_test_model, evaluation_question_count,
            evaluation_retrieval_mode, evaluation_top_k,
            evaluation_allow_parallel_generation, evaluation_test_max_concurrent_requests,
            evaluation_use_reranker,
        ]
        for component in [evaluation_generation_model, evaluation_test_model, evaluation_question_count,
                          evaluation_retrieval_mode, evaluation_top_k,
                          evaluation_allow_parallel_generation, evaluation_use_reranker,
                          evaluation_test_max_concurrent_requests]:
            component.input(
                save_evaluation_preferences_for_ui,
                inputs=evaluation_preference_inputs, outputs=evaluation_status,
                show_progress="hidden",
            )
        evaluation_questions_table.input(
            save_evaluation_questions_for_ui,
            inputs=[project_selector, evaluation_questions_table, evaluation_state],
            outputs=[evaluation_status, evaluation_state, evaluation_results_table],
            show_progress="hidden",
        )
        import_evaluation_button.click(
            import_evaluation_questions_for_ui,
            inputs=[project_selector, evaluation_import_file, evaluation_state],
            outputs=[evaluation_status, evaluation_questions_table,
                     evaluation_state, evaluation_results_table],
        )
        export_evaluation_button.click(
            export_evaluation_questions_for_ui,
            inputs=[project_selector, evaluation_questions_table],
            outputs=[evaluation_status, evaluation_export_file],
        )
        generate_evaluation_button.click(
            generate_evaluation_for_ui,
            inputs=[project_selector, generation_model_endpoint, generation_model_key, evaluation_generation_model,
                    evaluation_test_model, evaluation_question_count, evaluation_retrieval_mode,
                    evaluation_top_k, chunk_state, evaluation_allow_parallel_generation,
                    evaluation_test_max_concurrent_requests,
                    evaluation_use_reranker],
            outputs=[evaluation_status, evaluation_questions_table,
                     evaluation_state, evaluation_results_table],
        )
        run_evaluation_button.click(
            run_evaluation_for_ui,
            inputs=[project_selector, evaluation_model_endpoint, evaluation_model_key,
                    selected_embedding_endpoint, selected_embedding_key,
                    neo4j_uri, neo4j_database, neo4j_username, neo4j_password,
                    evaluation_test_model, evaluation_retrieval_mode,
                    evaluation_top_k, evaluation_state, evaluation_test_max_concurrent_requests,
                    evaluation_use_reranker],
            outputs=[evaluation_status, evaluation_results_table, evaluation_state],
        )

        project_setting_inputs = [
            project_selector, documents_state, chunk_state, graph_state,
            neo4j_uri, neo4j_database, neo4j_username, neo4j_password,
            model_endpoint, api_key, graph_llm_model, graph_embedding_model,
            answer_model, chunk_size, chunk_overlap,
            graph_temperature, schema_granularity,
            max_concurrent_requests,
            schema_sampling_mode, schema_sample_page_count, extraction_llm_model,
            extraction_max_concurrent_requests, retrieval_mode,
            top_k, schema_editor,
        ]
        project_load_outputs = [
            project_state, project_status,
            neo4j_uri, neo4j_database, neo4j_username, neo4j_password,
            model_endpoint, api_key, graph_llm_model, graph_embedding_model,
            answer_model, chunk_size, chunk_overlap,
            graph_temperature, schema_granularity,
            max_concurrent_requests,
            schema_sampling_mode, schema_sample_page_count, extraction_llm_model,
            extraction_max_concurrent_requests, retrieval_mode,
            top_k, schema_editor,
            documents_state, chunk_state, graph_state,
            active_preview_state, active_chunks_state,
            documents_table, remove_document_selector,
            chunk_table, page_status, history_table,
            entity_table, relationship_table, build_status, import_status,
        ]
        project_tab.select(refresh_projects_for_ui, inputs=project_state, outputs=project_selector)
        app.load(refresh_projects_for_ui, inputs=project_state, outputs=project_selector)
        create_project_event = create_project_button.click(
            create_project_for_ui, inputs=new_project_name,
            outputs=[project_selector, project_state, project_status],
        )
        create_project_event.success(
            reset_new_project_pdf_status_for_ui,
            outputs=preview_status, show_progress="hidden",
        )
        load_project_event = load_project_button.click(
            load_project_with_services_for_ui,
            inputs=[project_selector, llm_service_state, embedding_service_state],
            outputs=[*project_load_outputs,
                     llm_provider, llm_service_state, llm_models_table, model_test_button, model_list_button,
                     model_connection_status, evaluation_generation_model, evaluation_test_model,
                     embedding_provider, embedding_service_state, embedding_models_table,
                     embedding_test_button, embedding_list_button, embedding_connection_status],
        )
        initialize_project_event = create_project_event.success(
            load_project_with_services_for_ui,
            inputs=[project_selector, llm_service_state, embedding_service_state],
            outputs=[*project_load_outputs,
                     llm_provider, llm_service_state, llm_models_table, model_test_button, model_list_button,
                     model_connection_status, evaluation_generation_model, evaluation_test_model,
                     embedding_provider, embedding_service_state, embedding_models_table,
                     embedding_test_button, embedding_list_button, embedding_connection_status],
            show_progress="hidden",
        )
        protected_tabs = [pdf_tab, summary_tab, graph_tab, evaluation_tab, qa_tab, history_tab]
        delete_project_event = delete_project_button.click(
            delete_project_for_ui,
            inputs=project_selector,
            outputs=[project_selector, project_state, project_status,
                     pdf_tab, summary_tab, graph_tab, evaluation_tab, qa_tab, history_tab,
                     delete_project_completed],
            js="""(projectId) => {
                if (!window.confirm('確定要刪除此專案嗎？專案設定、PDF、圖譜、題庫與紀錄都會永久刪除。')) {
                    throw new Error('使用者取消刪除');
                }
                return projectId;
            }""",
        )
        delete_project_event.then(
            refresh_projects_after_delete_for_ui,
            inputs=delete_project_completed,
            outputs=project_selector,
            show_progress="hidden",
        )
        access_inputs = [project_selector, neo4j_connected_state, llm_service_state, embedding_service_state]
        initialize_project_event.success(
            workflow_tabs_for_ui, inputs=access_inputs, outputs=protected_tabs,
        )
        load_project_event.success(
            workflow_tabs_for_ui, inputs=access_inputs, outputs=protected_tabs,
        )
        project_state.change(
            _current_project_banner, inputs=project_state, outputs=current_project_banner,
            show_progress="hidden",
        )
        auto_save_components = [
            neo4j_uri, neo4j_database, neo4j_username, neo4j_password,
            model_endpoint, api_key, graph_llm_model, graph_embedding_model,
            answer_model, chunk_size, chunk_overlap,
            graph_temperature, schema_granularity,
            max_concurrent_requests,
            schema_sampling_mode, schema_sample_page_count, extraction_llm_model,
            extraction_max_concurrent_requests, retrieval_mode,
            top_k, schema_editor,
        ]
        for component in auto_save_components:
            component.input(
                save_project_for_ui, inputs=project_setting_inputs,
                outputs=[project_state, project_status], show_progress="hidden",
            )

        neo4j_test_event = neo4j_test_button.click(
            check_neo4j_for_ui,
            inputs=[neo4j_uri, neo4j_database, neo4j_username, neo4j_password],
            outputs=[neo4j_connection_status, neo4j_connected_state],
        )
        neo4j_test_event.then(
            workflow_tabs_for_ui, inputs=access_inputs, outputs=protected_tabs,
            show_progress="hidden",
        )
        for neo4j_field in [neo4j_uri, neo4j_database, neo4j_username, neo4j_password]:
            invalidation = neo4j_field.input(
                lambda: (False, "設定已變更，請重新測試 Neo4j 連線。"),
                outputs=[neo4j_connected_state, neo4j_connection_status],
                show_progress="hidden",
            )
            invalidation.then(
                workflow_tabs_for_ui, inputs=access_inputs, outputs=protected_tabs,
                show_progress="hidden",
            )
        llm_model_fields = [
            graph_llm_model, extraction_llm_model, evaluation_generation_model,
            evaluation_test_model, answer_model, summary_model,
        ]
        llm_service_outputs = [
            llm_service_state, model_endpoint, api_key, llm_models_table,
            model_test_button, model_list_button, model_connection_status, *llm_model_fields,
        ]
        embedding_service_outputs = [
            embedding_service_state, embedding_api_base, embedding_api_key, embedding_models_table,
            embedding_test_button, embedding_list_button, embedding_connection_status, graph_embedding_model,
        ]
        for provider, state, endpoint, key, table, test_button, fetch_button, fields, outputs in [
            (llm_provider, llm_service_state, model_endpoint, api_key, llm_models_table,
             model_test_button, model_list_button, llm_model_fields, llm_service_outputs),
            (embedding_provider, embedding_service_state, embedding_api_base, embedding_api_key,
             embedding_models_table, embedding_test_button, embedding_list_button,
             [graph_embedding_model], embedding_service_outputs),
        ]:
            inputs = [provider, state, endpoint, key, table, *fields]
            service_events = [
                provider.input(partial(service_action_for_ui, "switch"), inputs=inputs, outputs=outputs, concurrency_id="service-settings"),
                test_button.click(partial(service_action_for_ui, "test"), inputs=inputs, outputs=outputs, concurrency_id="service-settings"),
                fetch_button.click(partial(service_action_for_ui, "fetch"), inputs=inputs, outputs=outputs, concurrency_id="service-settings"),
            ]
            for component in [endpoint, key, table, *fields]:
                service_events.append(component.input(partial(service_action_for_ui, "edit"), inputs=inputs, outputs=outputs, show_progress="hidden", concurrency_id="service-settings"))
            for service_event in service_events:
                service_event.then(
                    workflow_tabs_for_ui, inputs=access_inputs, outputs=protected_tabs,
                    show_progress="hidden",
                )
        for field, endpoint_state, key_state in [
            (graph_llm_model, schema_model_endpoint, schema_model_key),
            (summary_model, summary_model_endpoint, summary_model_key),
            (extraction_llm_model, extraction_model_endpoint, extraction_model_key),
            (evaluation_generation_model, generation_model_endpoint, generation_model_key),
            (evaluation_test_model, evaluation_model_endpoint, evaluation_model_key),
            (answer_model, answer_model_endpoint, answer_model_key),
        ]:
            field.change(
                resolve_model_credentials_for_ui,
                inputs=[llm_service_state, field], outputs=[endpoint_state, key_state],
                show_progress="hidden",
            )
        graph_embedding_model.change(
            resolve_model_credentials_for_ui,
            inputs=[embedding_service_state, graph_embedding_model],
            outputs=[selected_embedding_endpoint, selected_embedding_key],
            show_progress="hidden",
        )
        env_inputs = [
            neo4j_uri,
            neo4j_database,
            neo4j_username,
            neo4j_password,
            model_endpoint,
            api_key,
            embedding_api_base,
            embedding_api_key,
            graph_llm_model,
            graph_embedding_model,
            answer_model,
        ]
        for component in env_inputs:
            component.change(persist_env_settings, inputs=env_inputs, outputs=env_status)
        reload_event = reload_button.click(
            reload_env_with_services_for_ui,
            outputs=[neo4j_uri, neo4j_database, neo4j_username, neo4j_password,
                     llm_provider, *llm_service_outputs,
                     embedding_provider, *embedding_service_outputs, env_status],
        )
        reload_reset_event = reload_event.then(
            lambda: False, outputs=neo4j_connected_state, show_progress="hidden",
        )
        reload_reset_event.then(
            workflow_tabs_for_ui, inputs=access_inputs, outputs=protected_tabs,
            show_progress="hidden",
        )
        preview_event = preview_button.click(
            add_document_for_ui,
            inputs=[
                pdf_file,
                chunk_size,
                chunk_overlap,
                documents_state,
                chunk_state,
            ],
            outputs=[
                preview_status,
                chunk_table,
                documents_state,
                chunk_state,
                active_preview_state,
                active_chunks_state,
                page_status,
                documents_table,
                pdf_file,
                remove_document_selector,
            ],
        )
        preview_event.then(
            save_project_for_ui, inputs=project_setting_inputs,
            outputs=[project_state, project_status], show_progress="hidden",
        )
        remove_document_event = remove_document_button.click(
            remove_document_for_ui,
            inputs=[project_selector, documents_table, documents_state, chunk_state],
            outputs=[
                preview_status,
                documents_state,
                chunk_state,
                active_preview_state,
                active_chunks_state,
                page_status,
                documents_table,
                remove_document_selector,
                chunk_table,
            ],
        )
        remove_document_event.then(
            save_project_for_ui, inputs=project_setting_inputs,
            outputs=[project_state, project_status], show_progress="hidden",
        )
        previous_button.click(
            previous_document,
            inputs=[documents_state, chunk_state, active_preview_state],
            outputs=[active_preview_state, active_chunks_state, chunk_table, page_status],
        )
        next_button.click(
            next_document,
            inputs=[documents_state, chunk_state, active_preview_state],
            outputs=[active_preview_state, active_chunks_state, chunk_table, page_status],
        )
        export_button.click(
            save_config, inputs=[active_preview_state], outputs=[preview_status, export_file]
        )
        schema_sampling_mode.change(
            lambda mode: gr.update(visible=mode == "隨機抽取 N 頁"),
            inputs=schema_sampling_mode,
            outputs=schema_sample_page_count,
        )
        plan_schema_button.click(
            plan_schema_for_ui,
            inputs=[
                schema_model_endpoint,
                schema_model_key,
                graph_llm_model,
                graph_temperature,
                schema_granularity,
                max_concurrent_requests,
                schema_sampling_mode,
                schema_sample_page_count,
                chunk_state,
                run_control_state,
            ],
            outputs=[plan_status, schema_editor],
        )
        extraction_event = generate_graph_button.click(
            extract_graph_for_ui,
            inputs=[
                extraction_model_endpoint,
                extraction_model_key,
                extraction_llm_model,
                graph_temperature,
                extraction_max_concurrent_requests,
                chunk_state,
                schema_editor,
                documents_state,
                run_control_state,
            ],
            outputs=[build_status, entity_table, relationship_table, graph_state],
            show_progress="minimal",
        )
        pause_button.click(
            toggle_pause_for_ui,
            inputs=[run_control_state],
            outputs=[run_control_status, pause_button],
            queue=False,
        )
        stop_button.click(
            request_stop_for_ui,
            inputs=[run_control_state],
            outputs=[run_control_status],
            queue=False,
        )
        extraction_event.then(
            save_project_for_ui, inputs=project_setting_inputs,
            outputs=[project_state, project_status], show_progress="hidden",
        )
        import_event = import_graph_button.click(
            import_graph_for_ui,
            inputs=[
                selected_embedding_endpoint,
                selected_embedding_key,
                neo4j_uri,
                neo4j_database,
                neo4j_username,
                neo4j_password,
                graph_embedding_model,
                graph_state,
            ],
            outputs=[import_status, graph_state],
        )
        import_event.then(
            save_project_for_ui, inputs=project_setting_inputs,
            outputs=[project_state, project_status], show_progress="hidden",
        )
        ask_button.click(
            answer_question_for_project_ui,
            inputs=[
                project_selector,
                answer_model_endpoint,
                answer_model_key,
                selected_embedding_endpoint,
                selected_embedding_key,
                neo4j_uri,
                neo4j_database,
                neo4j_username,
                neo4j_password,
                answer_model,
                question,
                retrieval_mode,
                top_k,
                qa_document_filter,
                use_reranker,
            ],
            outputs=[answer_status, answer, answer_sources, history_table, project_history_status],
        )
    return app
