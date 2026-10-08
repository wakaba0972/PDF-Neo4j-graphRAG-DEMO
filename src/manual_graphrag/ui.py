from __future__ import annotations

import csv
from copy import deepcopy
import hashlib
import json
import re
import zipfile
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

import gradio as gr

from .audit import bind_gradio_callbacks_to_actor, current_actor
from .chunking import TextChunk, chunk_pages, preview_rows
from .config import (
    public_settings,
)
from .env_store import load_env, save_env
from .experiment_project_store import (
    create_experiment_project,
    delete_experiment_project,
    list_experiment_projects,
    load_experiment_project,
    save_experiment_project,
)
from .evaluation_service import (
    evaluation_questions_are_similar,
    expand_page_context_until_complete,
    generate_evaluation_questions,
    judge_evaluation_answer,
    sample_random_page_context,
    verify_evaluation_judgment,
)
from .graph_service import (
    GPT_6_LUNA_MODEL,
    GPT_6_LUNA_REASONING_EFFORTS,
    RunCancelled,
    RunControl,
    check_model_connection,
    extract_graph,
    plan_graph_schema,
    validate_schema,
)
from .neo4j_service import (
    ensure_project_database,
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
    export_project_archive,
    import_project_archive,
    list_projects,
    load_project,
    load_project_summary,
    project_database_name,
    remove_document,
    save_project,
)
from .question_set_store import (
    bound_question_sets,
    create_question_set,
    list_question_sets,
    migrate_all_project_question_sets,
    migrate_project_question_sets,
    set_project_question_set_bindings,
)
from .qa_service import (
    RERANK_CANDIDATE_MULTIPLIER,
    RERANK_MAX_CANDIDATES,
    EVIDENCE_EXPANSION_MODES,
    RERANKER_MODES,
    answer_graph_question,
    check_embedding_connection,
    embedding_vectors,
    interleave_expanded_evidence,
    legacy_rerank_evidence,
    rerank_evidence,
)
from .retrieval import (
    RETRIEVAL_STRATEGIES,
    RetrievalConfig,
    strategy_choices,
    strategy_id_from_label,
    strategy_label,
)
from .service_settings import (
    capture_service_settings,
    configured_models,
    load_service_settings,
    preferred_service_model,
    resolve_model_service,
    restore_service_settings,
    save_service_settings,
    service_choice_items,
    service_choices,
)
from .storage import write_json

DEFAULT_LLM_MODEL = GPT_6_LUNA_MODEL
DEFAULT_EVALUATION_MODEL = GPT_6_LUNA_MODEL
DEFAULT_REASONING_EFFORT = "low"
DEFAULT_JUDGE_REASONING_EFFORT = "medium"
DEFAULT_MAX_CONCURRENT_REQUESTS = 10
KNOWN_USERS = ("Jay", "Christine", "Swallow", "Tai", "Zhao")


def current_user_banner_for_ui(user: str | None) -> str:
    return f"### 👤 目前使用者：{user}" if user in KNOWN_USERS else "### 👤 目前使用者：尚未選擇"


def user_access_tabs_for_ui(user: str | None) -> tuple[dict[str, Any], ...]:
    enabled = user in KNOWN_USERS
    return tuple(gr.update(interactive=enabled) for _ in range(5))


def select_user_for_ui(user: str | None) -> tuple[Any, ...]:
    return current_user_banner_for_ui(user), *user_access_tabs_for_ui(user)


def _user_prefixed_name(name: str, user: str | None) -> str:
    clean_name = str(name or "").strip()
    if user not in KNOWN_USERS:
        raise ValueError("請先在 0-0 選擇使用者")
    if not clean_name:
        raise ValueError("請輸入專案名稱")
    prefix = f"{user} | "
    return clean_name if clean_name.startswith(prefix) else f"{prefix}{clean_name}"


def _model_choices_with_fallback(
    choices: list[tuple[str, str]], model: str | None,
) -> list[tuple[str, str]]:
    result = list(choices)
    if model and model not in {value for _, value in result}:
        display = (
            f"OpenAI｜{model}（目前不可用）" if model.casefold() == GPT_6_LUNA_MODEL
            else f"{model}（目前不可用）"
        )
        result.append((display, model))
    return result


def reasoning_effort_visibility(model: str | None) -> dict[str, Any]:
    return gr.update(
        visible=str(model or "").strip().casefold() == GPT_6_LUNA_MODEL,
        value=DEFAULT_REASONING_EFFORT,
    )


def experiment_group_reasoning_effort_visibility(
    model: str | None, group_name: str | None,
) -> dict[str, Any]:
    return gr.update(
        visible=(
            bool(str(group_name or "").strip())
            and str(model or "").strip().casefold() == GPT_6_LUNA_MODEL
        ),
        value=DEFAULT_REASONING_EFFORT,
    )


def _reasoning_effort_kwargs(model: str | None, effort: str | None) -> dict[str, str]:
    if str(model or "").strip().casefold() != GPT_6_LUNA_MODEL:
        return {}
    return {"reasoning_effort": effort or DEFAULT_REASONING_EFFORT}


def _reasoning_effort_record(model: str | None, effort: str | None, field: str) -> dict[str, str]:
    return (
        {field: effort or DEFAULT_REASONING_EFFORT}
        if str(model or "").strip().casefold() == GPT_6_LUNA_MODEL else {}
    )


def _result_score(item: dict[str, Any]) -> int:
    """Read the 0-2 score while accepting legacy boolean result records."""
    score = item.get("score")
    if not isinstance(score, bool) and score in (0, 1, 2):
        return int(score)
    return 2 if bool(item.get("passed")) else 0


def _display_score(item: dict[str, Any]) -> int | None:
    if "score" in item or "passed" in item:
        return _result_score(item)
    return None


def _normalize_judgment(judgment: dict[str, Any]) -> dict[str, Any]:
    score = judgment.get("score")
    if isinstance(score, bool) or score not in (0, 1, 2):
        score = 2 if bool(judgment.get("passed")) else 0
    else:
        score = int(score)
    return {**judgment, "score": score, "passed": score == 2}


def _verify_judgment_if_enabled(
    judgment: dict[str, Any], verification_enabled: bool,
    endpoint: str, api_key: str, model: str,
    question: str, expected_answer: str, actual_answer: str,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    result = {**judgment, "verification_enabled": bool(verification_enabled)}
    if not verification_enabled:
        return {**result, "verification_changed": None}
    judgment = _normalize_judgment(judgment)
    first_score = _result_score(judgment)
    first_reason = str(judgment.get("reason", ""))
    try:
        verified = verify_evaluation_judgment(
            endpoint, api_key, model, question, expected_answer, actual_answer,
            first_score, first_reason,
            **_reasoning_effort_kwargs(model, reasoning_effort),
        )
    except ValueError as exc:
        return {
            **result,
            "first_score": first_score,
            "first_passed": first_score == 2,
            "first_reason": first_reason,
            "verification_changed": None,
            "verification_error": str(exc),
        }
    verified = _normalize_judgment(verified)
    return {
        **result,
        "first_score": first_score,
        "first_passed": first_score == 2,
        "verification_score": verified.get("score", 2 if verified.get("passed") else 0),
        "first_reason": first_reason,
        "verification_passed": verified["score"] == 2,
        "verification_reason": verified["reason"],
        "verification_changed": verified.get("score", 2 if verified.get("passed") else 0) != first_score,
        "score": verified.get("score", 2 if verified.get("passed") else 0),
        "passed": verified.get("score", 2 if verified.get("passed") else 0) == 2,
        "reason": verified["reason"],
    }


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
    uri: str, database: str, username: str, password: str,
    project_id: str | None = None,
) -> tuple[str, bool]:
    try:
        ensure_project_database(uri, database, username, password)
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


def resolve_model_credentials_for_ui(
    state: dict[str, Any], model: str | None,
) -> tuple[str, str]:
    try:
        base_url, api_key, _ = resolve_model_service(state, model)
    except ValueError:
        return "", ""
    return base_url, api_key


def refresh_model_credentials_for_ui(
    llm_state: dict[str, Any], embedding_state: dict[str, Any],
    graph_model: str | None, extraction_model: str | None,
    evaluation_generation_model: str | None, evaluation_answer_model: str | None,
    evaluation_judge_model: str | None, answer_model: str | None,
    embedding_model: str | None,
) -> tuple[str, ...]:
    """Refresh every cached model endpoint after service or project settings change."""
    credentials = [
        resolve_model_credentials_for_ui(llm_state, model)
        for model in (
            graph_model, extraction_model, evaluation_generation_model,
            evaluation_answer_model, evaluation_judge_model, answer_model,
        )
    ]
    credentials.append(resolve_model_credentials_for_ui(embedding_state, embedding_model))
    return tuple(value for pair in credentials for value in pair)


def render_service_for_ui(state: dict[str, Any]) -> tuple[Any, ...]:
    profile = state["profiles"]["OpenAI"]
    allowed = service_choices(state)
    choice_items = service_choice_items(state)
    fallback = preferred_service_model(state)
    return (
        state, profile["base_url"], profile["api_key"],
        [], gr.update(visible=True), False, profile["status"],
        *(gr.update(choices=choice_items, value=model if model in allowed else fallback) for model in profile["models"]),
    )


def service_action_for_ui(
    action: str, provider: str, state: dict[str, Any], base_url: str, api_key: str,
    rows: list[list[Any]], *models: str | None,
) -> tuple[Any, ...]:
    if provider != "OpenAI":
        raise gr.Error("目前僅支援 OpenAI 模型服務")
    state = capture_service_settings(state, base_url, api_key, rows, list(models))
    profile = state["profiles"]["OpenAI"]
    try:
        if action == "test":
            settings_kind = "llm" if state["kind"] == "experiment_llm" else state["kind"]
            configured_models(settings_kind)
            profile["connected"] = False
            if settings_kind == "embedding":
                model = profile["models"][0] or preferred_service_model(state)
                check_embedding_connection(profile["base_url"], profile["api_key"], model)
            else:
                check_model_connection(profile["base_url"], profile["api_key"])
            profile["connected"] = True
            profile["status"] = "✅ OpenAI API 連線成功。"
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


def experiment_service_action_for_ui(
    action: str, provider: str, state: dict[str, Any], base_url: str, api_key: str,
    rows: list[list[Any]],
) -> tuple[Any, ...]:
    """Update the isolated 1-series LLM service without touching 0-series settings."""
    models = list(state["profiles"]["OpenAI"]["models"])
    rendered = service_action_for_ui(action, provider, state, base_url, api_key, rows, *models)
    return (*rendered[:7], gr.update(visible=True))


def load_project_with_services_for_ui(
    project_id: str, llm_state: dict[str, Any], embedding_state: dict[str, Any],
) -> tuple[Any, ...]:
    values = list(load_project_for_ui(project_id))
    llm_state = restore_service_settings(
        llm_state, values[6], values[7], {0: values[8], 1: values[16], 4: values[10]},
    )
    embedding_profile = embedding_state["profiles"][embedding_state["active"]]
    embedding_state = restore_service_settings(
        embedding_state, embedding_profile["base_url"], embedding_profile["api_key"], {0: values[9]},
    )
    llm = render_service_for_ui(llm_state)
    embedding = render_service_for_ui(embedding_state)
    values[6:8] = llm[1:3]
    available_llm_choices = service_choice_items(llm_state)
    for index, selection in ((8, llm[7]), (16, llm[8]), (10, llm[11])):
        selected = selection.get("value")
        if selected is None and values[index] == DEFAULT_LLM_MODEL:
            selected = DEFAULT_LLM_MODEL
        choices = (
            _model_choices_with_fallback(available_llm_choices, DEFAULT_LLM_MODEL)
            if selected == DEFAULT_LLM_MODEL else available_llm_choices
        )
        values[index] = gr.update(value=selected, choices=choices)
    values[9] = embedding[7]
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
    for index in (3, 4, 12):
        if not isinstance(values[index], dict):
            selected = values[index]
            model_choices = (
                _model_choices_with_fallback(choice_items, selected)
                if index == 12 or selected == DEFAULT_LLM_MODEL else list(choice_items)
            )
            values[index] = gr.update(
                choices=model_choices,
                value=(selected if index == 12 or selected in allowed
                       or selected == DEFAULT_LLM_MODEL else None),
            )
    answer_status, evaluate_update = _evaluation_answer_availability(values[0])
    return (*values, answer_status, evaluate_update)


def reload_env_with_services_for_ui() -> tuple[Any, ...]:
    env = load_env()
    llm_state = load_service_settings("llm", env)
    embedding_state = load_service_settings("embedding", env)
    llm_profile = llm_state["profiles"]["OpenAI"]
    embedding_profile = embedding_state["profiles"]["OpenAI"]
    return (
        env["NEO4J_URI"], env["NEO4J_USERNAME"], env["NEO4J_PASSWORD"],
        llm_state["active"], *render_service_for_ui(llm_state),
        embedding_state["active"], *render_service_for_ui(embedding_state),
        llm_profile["base_url"], llm_profile["api_key"],
        "✅ 已重新讀取共用 OpenAI 設定。",
        "✅ 已重新讀取 .env；可用連線測試診斷服務，無須先測試即可操作。",
    )


def persist_env_settings(
    neo4j_uri: str,
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
        env["NEO4J_URI"], env["NEO4J_USERNAME"], env["NEO4J_PASSWORD"],
        llm_profile["base_url"], llm_profile["api_key"],
        embedding_profile["base_url"], embedding_profile["api_key"],
        llm_profile["models"][0], embedding_profile["models"][0], llm_profile["models"][4],
        "✅ 已重新讀取 .env 與模型 YAML",
    )


def persist_global_api_settings_for_ui(
    action: str,
    llm_state: dict[str, Any], embedding_state: dict[str, Any],
    experiment_llm_state: dict[str, Any],
    api_endpoint: str, api_key: str,
    *models: str | None,
) -> tuple[Any, ...]:
    """Persist one shared OpenAI endpoint/key and refresh every service state."""
    llm = service_action_for_ui(
        action, "OpenAI", llm_state, api_endpoint, api_key, [], *models[:6],
    )
    embedding = service_action_for_ui(
        action, "OpenAI", embedding_state, api_endpoint, api_key, [], *models[6:7],
    )
    experiment = capture_service_settings(
        experiment_llm_state, api_endpoint, api_key, [],
        list(experiment_llm_state["profiles"]["OpenAI"]["models"]),
    )
    save_service_settings(experiment)
    status = (
        f"✅ 共用 OpenAI 設定已保存。{llm[6]} {embedding[6]}"
        if action == "test" else "✅ 共用 OpenAI 設定已自動保存。"
    )
    return (*llm, *embedding, experiment, status)


def workflow_tabs_for_ui(
    project_id: str,
    neo4j_connected: bool,
    llm_state: dict[str, Any],
    embedding_state: dict[str, Any],
    workspace_mode: str = "single",
    project: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], ...]:
    # Connection checks are diagnostic only. Only one project workspace is
    # active at a time; loading an experiment workspace locks the single-project tabs.
    enabled = bool(project_id) and workspace_mode == "single"
    loaded_project_id = (project or {}).get("project_id")
    connection_enabled = enabled and (
        loaded_project_id == project_id if project is not None else True
    )
    return (
        gr.update(interactive=connection_enabled),
        *(gr.update(interactive=enabled) for _ in range(5)),
    )


def experiment_workflow_tabs_for_ui(
    workspace_mode: str,
    project: dict[str, Any] | None,
    connection_statuses: dict[str, bool] | None,
) -> tuple[dict[str, Any], ...]:
    members = (project or {}).get("members", [])
    statuses = connection_statuses or {}
    enabled = (
        workspace_mode == "experiment" and bool(members)
        and all(statuses.get(member_id) is True for member_id in members)
    )
    connection_enabled = (
        workspace_mode == "experiment"
        and bool((project or {}).get("experiment_project_id"))
    )
    return gr.update(interactive=connection_enabled), gr.update(interactive=enabled)


def activate_workspace_for_ui(
    workspace_mode: str, project: dict[str, Any] | None,
    current_workspace_mode: str = "",
    current_connection_statuses: dict[str, bool] | None = None,
) -> tuple[str, dict[str, bool], dict[str, Any], dict[str, Any]]:
    """Switch workspace modes and retain checks only when the experiment is unchanged."""
    active_mode = workspace_mode if project else ""
    members = (project or {}).get("members", [])
    statuses = current_connection_statuses or {}
    retain_statuses = (
        active_mode == current_workspace_mode == "experiment"
        and bool(members)
        and all(member_id in statuses for member_id in members)
    )
    if not retain_statuses:
        statuses = {}
    tabs = experiment_workflow_tabs_for_ui(active_mode, project, statuses)
    return active_mode, statuses, *tabs


def reset_experiment_project_connection_for_ui(
    project: dict[str, Any] | None,
) -> tuple[dict[str, bool], dict[str, Any], dict[str, Any]]:
    disabled = gr.update(interactive=False)
    connection_enabled = gr.update(
        interactive=bool((project or {}).get("experiment_project_id")),
    )
    return {}, connection_enabled, disabled


def lock_project_tabs_for_ui(project_id: str) -> tuple[dict[str, Any], ...]:
    return tuple(gr.update(interactive=False) for _ in range(6))


def delete_project_for_ui(
    project_id: str,
) -> tuple[Any, ...]:
    try:
        name = delete_project(project_id)
    except (OSError, ValueError) as exc:
        return (
            gr.update(interactive=True), gr.update(), f"❌ {exc}",
            *lock_project_tabs_for_ui(project_id), False,
        )
    choices = _project_choices()
    return (
        gr.update(choices=choices, value=choices[0][1] if choices else None, interactive=True), {},
        f"✅ 已刪除專案「{name}」。", *lock_project_tabs_for_ui(""), True,
    )


def request_project_deletion_for_ui(
    project_id: str | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not isinstance(project_id, str) or not project_id:
        return (
            gr.update(value="請先選擇要刪除的專案。", visible=True),
            gr.update(visible=False), gr.update(visible=False),
            gr.update(interactive=True),
        )
    return (
        gr.update(
            value=f"確定永久刪除專案「{project_id}」及其本機資料？",
            visible=True,
        ),
        gr.update(visible=True), gr.update(visible=True),
        gr.update(interactive=False),
    )


def cancel_project_deletion_for_ui() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    return (
        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
        gr.update(interactive=True),
    )


def delete_confirmed_project_for_ui(project_id: str) -> tuple[Any, ...]:
    result = delete_project_for_ui(project_id)
    return (*result, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False))


def refresh_projects_after_delete_for_ui(deleted: bool) -> dict[str, Any]:
    if not deleted:
        return gr.update(interactive=True)
    choices = _project_choices()
    return gr.update(choices=choices, value=choices[0][1] if choices else None, interactive=True)


def _project_choices() -> list[tuple[str, str]]:
    return list_projects()


def create_project_for_ui(name: str) -> tuple[dict[str, Any], dict[str, Any], str]:
    try:
        user = current_actor()
        project = create_project(_user_prefixed_name(name, user), actor=user)
    except (OSError, ValueError) as exc:
        return gr.update(), {}, f"❌ {exc}"
    return gr.update(choices=_project_choices(), value=project["project_id"]), project, f"✅ 已建立專案「{project['name']}」。"


def export_project_for_ui(project_id: str | None) -> tuple[str | None, str]:
    if not project_id:
        return None, "❌ 請先選擇專案。"
    try:
        archive_path = export_project_archive(project_id)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        return None, f"❌ 專案匯出失敗：{exc}"
    return str(archive_path), (
        "✅ 專案封裝已建立，請下載保存。封裝含專案設定（包括 Neo4j 密碼）、PDF、題庫與本機結果；"
        "不包含外部 Neo4j Database 本體。請勿將未加密封裝分享給他人。"
    )


def import_project_for_ui(archive_path: str | None) -> tuple[dict[str, Any], dict[str, Any], str]:
    if not archive_path:
        raise gr.Error("請先選擇專案封裝檔。")
    try:
        project = import_project_archive(archive_path)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise gr.Error(f"專案匯入失敗：{exc}") from exc
    return (
        gr.update(choices=_project_choices(), value=project["project_id"]),
        project,
        f"✅ 已匯入專案「{project['name']}」。外部 Neo4j Database 未包含；圖譜資料需重新匯入 Neo4j。",
    )


def refresh_projects_for_ui(project: dict[str, Any] | None = None) -> dict[str, Any]:
    choices = _project_choices()
    available_ids = {project_id for _, project_id in choices}
    selected_id = str((project or {}).get("project_id") or "")
    return gr.update(
        choices=choices,
        value=selected_id if selected_id in available_ids else choices[0][1] if choices else None,
    )


def ensure_project_selection_for_ui(project_id: str | None) -> dict[str, Any]:
    choices = _project_choices()
    available_ids = {available_id for _, available_id in choices}
    selected_id = str(project_id or "")
    return gr.update(
        choices=choices,
        value=selected_id if selected_id in available_ids else choices[0][1] if choices else None,
    )


def reset_new_project_pdf_status_for_ui() -> str:
    return "尚未解析 PDF。可重複上傳多份 PDF，逐一加入同一個專案。"


def _chunk_dicts(chunks: list[TextChunk]) -> list[dict[str, Any]]:
    return [
        {"number": c.number, "text": c.text, "pages": list(c.pages),
         "document": c.document, "document_id": c.document_id}
        for c in chunks
    ]


def _stored_chunks(items: list[dict[str, Any]]) -> list[TextChunk]:
    return [
        TextChunk(
            int(i["number"]), str(i["text"]), tuple(i.get("pages") or []),
            str(i.get("document", "")), str(i.get("document_id", "")),
        )
        for i in items
    ]


def _document_ids_from_project(project: dict[str, Any]) -> dict[str, str]:
    paths = {item.get("name"): item.get("path") for item in project.get("documents", [])}
    result = {}
    for document in project.get("documents_meta", []):
        name = str(document.get("file_name") or "")
        document_id = str(document.get("document_id") or "")
        path = paths.get(name) or document.get("file_path")
        if not document_id and path:
            try:
                document_id = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            except OSError:
                pass
        if name and document_id:
            result[name] = document_id
    return result


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


def _retrieval_config_from_controls(
    strategy: str, top_k: int, reranker: str, expansion: str,
    strategy_params: Any = None,
) -> RetrievalConfig:
    config = RetrievalConfig.from_ui(
        strategy, int(top_k), reranker, expansion,
        rerank_candidates=reranker != "停用",
    )
    if strategy_params in (None, ""):
        params = {}
    elif isinstance(strategy_params, dict):
        params = strategy_params
    elif isinstance(strategy_params, (int, float)) and not isinstance(strategy_params, bool):
        # Single numeric native controls (for example a Gradio Slider) are
        # accepted directly and mapped to the matching declared parameter.
        parameter_names = {
            name
            for strategy_id in RETRIEVAL_STRATEGIES.ids()
            for name, spec in RETRIEVAL_STRATEGIES.get_spec(strategy_id).parameters.items()
            if spec.visible_in_ui and spec.value_type in (int, float)
        }
        if len(parameter_names) != 1:
            raise ValueError("策略參數控制值無法對應到唯一參數")
        params = {next(iter(parameter_names)): strategy_params}
    else:
        try:
            params = json.loads(str(strategy_params))
        except json.JSONDecodeError as exc:
            raise ValueError("策略參數必須是有效的 JSON 物件") from exc
    if not isinstance(params, dict):
        raise ValueError("策略參數必須是 JSON 物件")
    return replace(config, params={**config.params, **params}).validated()


def _visible_strategy_parameters() -> list[tuple[str, Any]]:
    """Return unique, UI-visible parameters from the declarative strategy catalog."""
    parameters: dict[str, Any] = {}
    for strategy_id in RETRIEVAL_STRATEGIES.ids():
        for name, spec in RETRIEVAL_STRATEGIES.get_spec(strategy_id).parameters.items():
            if spec.visible_in_ui:
                parameters.setdefault(name, spec)
    return list(parameters.items())


def _strategy_parameter_value(config: RetrievalConfig, name: str) -> Any:
    spec = dict(_visible_strategy_parameters())[name]
    return config.params.get(name, spec.default)


def _create_strategy_parameter_controls(*, visible: bool = True, compact: bool = False) -> list[Any]:
    """Render native Gradio controls directly from the registered parameter specs."""
    controls = []
    for name, spec in _visible_strategy_parameters():
        label = spec.label or name.replace("_", " ").title()
        common = {"label": label, "visible": visible}
        if compact:
            common.update({"show_label": False, "scale": 2})
        if spec.choices:
            controls.append(gr.Dropdown(choices=list(spec.choices), value=spec.default, **common))
        elif spec.value_type is bool:
            controls.append(gr.Checkbox(value=bool(spec.default), **common))
        elif spec.control == "slider" and spec.minimum is not None and spec.maximum is not None:
            controls.append(gr.Slider(
                minimum=spec.minimum, maximum=spec.maximum,
                step=1 if spec.value_type is int else 0.1,
                value=spec.default, **common,
            ))
        elif spec.value_type in (int, float):
            controls.append(gr.Number(
                value=spec.default, minimum=spec.minimum, maximum=spec.maximum,
                precision=0 if spec.value_type is int else 3, **common,
            ))
        else:
            controls.append(gr.Textbox(value=spec.default, **common))
    return controls


def _strategy_parameter_entries() -> list[tuple[str, str, Any]]:
    return [
        (strategy_id, name, spec)
        for strategy_id in RETRIEVAL_STRATEGIES.ids()
        for name, spec in RETRIEVAL_STRATEGIES.get_spec(strategy_id).parameters.items()
        if spec.visible_in_ui
    ]


def _create_experiment_strategy_parameter_controls(
    initial_strategy_id: str | None = None,
) -> list[Any]:
    """Create a separate native control for every strategy-declared UI parameter."""
    controls = []
    for strategy_id, name, spec in _strategy_parameter_entries():
        label = spec.label or name.replace("_", " ").title()
        common = {
            "label": label, "show_label": True,
            "visible": strategy_id == initial_strategy_id, "scale": 2,
        }
        if spec.choices:
            control = gr.Dropdown(choices=list(spec.choices), value=spec.default, **common)
        elif spec.value_type is bool:
            control = gr.Checkbox(value=bool(spec.default), **common)
        elif spec.control == "slider" and spec.minimum is not None and spec.maximum is not None:
            control = gr.Slider(
                minimum=spec.minimum, maximum=spec.maximum,
                step=1 if spec.value_type is int else 0.1,
                value=spec.default, **common,
            )
        elif spec.value_type in (int, float):
            control = gr.Number(
                value=spec.default, minimum=spec.minimum, maximum=spec.maximum,
                precision=0 if spec.value_type is int else 3, **common,
            )
        else:
            control = gr.Textbox(value="" if spec.default is None else str(spec.default), **common)
        controls.append(control)
    return controls


def _experiment_strategy_parameter_updates(
    config: RetrievalConfig, row_visible: bool,
) -> list[Any]:
    return [
        gr.update(
            value=config.params.get(name, spec.default),
            visible=row_visible and strategy_id == config.strategy_id,
        )
        for strategy_id, name, spec in _strategy_parameter_entries()
    ]


def _visible_params_for_strategy(config: RetrievalConfig) -> dict[str, Any]:
    return {
        name: config.params.get(name, spec.default)
        for strategy_id, name, spec in _strategy_parameter_entries()
        if strategy_id == config.strategy_id
    }


def _evaluation_strategy_parameter_values(config: RetrievalConfig) -> tuple[Any, ...]:
    return (
        _visible_params_for_strategy(config),
        *_experiment_strategy_parameter_updates(config, True),
    )


def _sync_experiment_strategy_parameter_controls(
    strategy: str, group_name: str, *values: Any,
) -> tuple[Any, ...]:
    strategy_id = strategy_id_from_label(strategy)
    entries = _strategy_parameter_entries()
    params = {
        name: value
        for (entry_strategy_id, name, _spec), value in zip(entries, values)
        if entry_strategy_id == strategy_id
    }
    visible = bool(str(group_name or "").strip())
    updates = [
        gr.update(
            value=value,
            visible=visible and entry_strategy_id == strategy_id,
        )
        for (entry_strategy_id, _name, _spec), value in zip(entries, values)
    ]
    return (params, *updates)


def _sync_single_strategy_parameter_controls(
    strategy: str, *values: Any,
) -> tuple[Any, ...]:
    """Sync the single-project evaluation controls using the strategy catalog."""
    strategy_id = strategy_id_from_label(strategy)
    entries = _strategy_parameter_entries()
    params = {
        name: value
        for (entry_strategy_id, name, _spec), value in zip(entries, values)
        if entry_strategy_id == strategy_id
    }
    updates = [
        gr.update(
            value=value,
            visible=entry_strategy_id == strategy_id,
        )
        for (entry_strategy_id, _name, _spec), value in zip(entries, values)
    ]
    return (params, *updates)


def _group_retrieval_config(group: dict[str, Any]) -> RetrievalConfig:
    return RetrievalConfig.from_dict(group.get("retrieval_config"))


def _group_retrieval_values(group: dict[str, Any]) -> tuple[str, int, str, str]:
    return _group_retrieval_config(group).ui_values()


def _default_retrieval_config() -> dict[str, Any]:
    return _retrieval_config_from_controls(
        "混合檢索", 8, "停用", "停用",
    ).to_dict()


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


def _project_name_for_build(
    project: dict[str, Any], graph_state: dict[str, Any]
) -> str | None:
    build_config = graph_state.get("build_config")
    if not isinstance(build_config, dict):
        return None
    try:
        base_name = str(project.get("base_name") or project.get("name") or "")
        granularity = build_config.get("schema_granularity") or "None"
        return (
            f"{base_name} | {int(build_config['chunk_size'])}-"
            f"{int(build_config['chunk_overlap'])}-{granularity}"
        )
    except (KeyError, TypeError, ValueError):
        return None


def save_project_for_ui(
    project_id: str, documents: list[dict[str, Any]],
    chunks: list[TextChunk], graph_state: dict[str, Any],
    neo4j_uri: str, neo4j_database: str, neo4j_username: str, neo4j_password: str,
    _model_endpoint: str, _api_key: str, graph_llm_model: str,
    graph_embedding_model: str, answer_model: str,
    chunk_size: int, chunk_overlap: int,
    graph_temperature: float,
    schema_granularity: str,
    max_concurrent_requests: int, extraction_llm_model: str,
    extraction_max_concurrent_requests: int,
    retrieval_mode: str, top_k: int, schema_text: str,
    graph_reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    extraction_reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    answer_reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    retrieval_strategy_params_json: str = "{}",
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
        "extraction_llm_model": extraction_llm_model,
        "extraction_max_concurrent_requests": int(extraction_max_concurrent_requests),
        "retrieval_config": _retrieval_config_from_controls(
            retrieval_mode, int(top_k), "停用", "停用",
            retrieval_strategy_params_json,
        ).to_dict(),
        "schema_text": schema_text or "",
        "graph_reasoning_effort": graph_reasoning_effort or DEFAULT_REASONING_EFFORT,
        "extraction_reasoning_effort": extraction_reasoning_effort or DEFAULT_REASONING_EFFORT,
        "answer_reasoning_effort": answer_reasoning_effort or DEFAULT_REASONING_EFFORT,
    }
    documents = documents or []
    graph_state = graph_state or {}
    try:
        name = _project_name_for_build(load_project(project_id), graph_state)
    except (OSError, ValueError):
        name = None
    try:
        project = save_project(project_id, {
            "settings": settings, "documents_meta": documents,
            "chunks": _chunk_dicts(chunks or []), "graph_state": graph_state,
            **({"name": name} if name else {}),
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
    project_retrieval = RetrievalConfig.from_dict(
        settings.get("retrieval_config") or _default_retrieval_config()
    )
    env, get = load_env(), settings.get
    llm_settings = load_service_settings("llm", env)
    embedding_settings = load_service_settings("embedding", env)
    llm_profile = llm_settings["profiles"][llm_settings["active"]]
    embedding_profile = embedding_settings["profiles"][embedding_settings["active"]]
    documents = project.get("documents_meta") or []
    chunks = _stored_chunks(project.get("chunks") or [])
    document_ids = _document_ids_from_project(project)
    chunks = [
        chunk if chunk.document_id or chunk.document not in document_ids else TextChunk(
            chunk.number, chunk.text, chunk.pages, chunk.document,
            document_ids[chunk.document],
        )
        for chunk in chunks
    ]
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
            else f"⚠️ {graph['neo4j_error']}" if graph.get("neo4j_error")
            else "此圖譜尚未匯入 Neo4j。"
        )
    else:
        graph_status, import_status = "尚未執行抽取。", "尚未執行 Embedding 與匯入。"
    return (
        project, f"✅ 已載入專案「{project['name']}」。",
        get("neo4j_uri", env["NEO4J_URI"]), project.get("neo4j_database") or project_database_name(project_id),
        get("neo4j_username", env["NEO4J_USERNAME"]), get("neo4j_password", env["NEO4J_PASSWORD"]),
        llm_profile["base_url"], llm_profile["api_key"],
        get("graph_llm_model", DEFAULT_LLM_MODEL), get("graph_embedding_model", embedding_profile["models"][0]),
        get("answer_model", DEFAULT_LLM_MODEL),
        get("chunk_size", 1500), get("chunk_overlap", 200), get("graph_temperature", 0),
        get("schema_granularity", "平衡"),
        get("max_concurrent_requests", DEFAULT_MAX_CONCURRENT_REQUESTS),
        get("extraction_llm_model", DEFAULT_LLM_MODEL),
        get("extraction_max_concurrent_requests", DEFAULT_MAX_CONCURRENT_REQUESTS),
        *project_retrieval.ui_values()[:2], get("schema_text", ""),
        documents, chunks, graph, active_preview, active_chunks,
        _document_rows(documents), _document_choices(documents),
        _chunk_rows(active_chunks), document_status,
        entity_rows, relationship_rows, graph_status, import_status,
        gr.update(
            value=get("graph_reasoning_effort", DEFAULT_REASONING_EFFORT),
            visible=str(get("graph_llm_model", DEFAULT_LLM_MODEL) or "").casefold() == GPT_6_LUNA_MODEL,
            choices=list(GPT_6_LUNA_REASONING_EFFORTS),
        ),
        gr.update(
            value=get("extraction_reasoning_effort", DEFAULT_REASONING_EFFORT),
            visible=str(get("extraction_llm_model", DEFAULT_LLM_MODEL) or "").casefold() == GPT_6_LUNA_MODEL,
            choices=list(GPT_6_LUNA_REASONING_EFFORTS),
        ),
        gr.update(
            value=get("answer_reasoning_effort", DEFAULT_REASONING_EFFORT),
            visible=str(get("answer_model", DEFAULT_LLM_MODEL) or "").casefold() == GPT_6_LUNA_MODEL,
            choices=list(GPT_6_LUNA_REASONING_EFFORTS),
        ),
        _strategy_parameter_value(project_retrieval, "effective_search_ratio"),
    )


def answer_question_for_project_ui(project_id: str, *args: Any) -> tuple[Any, ...]:
    if len(args) > 15:
        status, answer, sources = answer_question_for_ui(
            *args[:15], strategy_params=args[15],
        )
    else:
        status, answer, sources = answer_question_for_ui(*args)
    if not status.startswith("✅") or not project_id:
        return status, answer, sources
    try:
        current = load_project(project_id)
        config = _retrieval_config_from_controls(
            str(args[10]), int(args[11]),
            str(args[12]) if len(args) > 12 else "停用",
            str(args[13]) if len(args) > 13 else "停用",
            args[15] if len(args) > 15 else None,
        )
        record = {
            "question": str(args[9]).strip(), "answer": answer,
            "answer_model": str(args[8]), "retrieval_config": config.to_dict(),
            "sources": sources,
            "document": (current.get("graph_state") or {}).get("document", ""),
        }
        if len(args) > 14:
            record.update(_reasoning_effort_record(args[8], args[14], "reasoning_effort"))
        append_question(project_id, record)
    except (OSError, ValueError) as exc:
        return f"{status}｜⚠️ 專案紀錄保存失敗：{exc}", answer, sources
    return f"{status}｜✅ 問答紀錄已加入目前專案。", answer, sources


def _evaluation_question_rows(questions: list[dict[str, Any]]) -> list[list[object]]:
    return [[
        item["number"], item["question"], item["expected_answer"],
        _source_references_cell(item.get("question_sources") or _legacy_question_sources(item)),
        _source_references_cell(item.get("answer_sources") or _legacy_answer_sources(item)),
    ] for item in questions]


def _parse_source_pages(value: Any, question_number: int, label: str) -> list[int]:
    page_values = value if isinstance(value, (list, tuple)) else (
        str(value or "").replace("，", ",").split(",")
    )
    try:
        return list(dict.fromkeys(
            int(page) for page in page_values if str(page).strip()
        ))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"第 {question_number} 題的{label}必須是逗號分隔的整數"
        ) from exc


def _source_references_cell(references: list[dict[str, Any]]) -> str:
    lines = []
    for reference in references:
        name = str(reference.get("document_name") or reference.get("document") or "未標示文件")
        pages = ", ".join(map(str, reference.get("pages", [])))
        lines.append(f"{name}：{pages}")
    return "\n".join(lines)


def _legacy_question_sources(item: dict[str, Any]) -> list[dict[str, Any]]:
    return _legacy_sources(item.get("question_source_pages", item.get("source_pages", [])), item.get("document", ""))


def _legacy_answer_sources(item: dict[str, Any]) -> list[dict[str, Any]]:
    return _legacy_sources(item.get("answer_source_pages", item.get("source_pages", [])), item.get("document", ""))


def _legacy_sources(pages: Any, document: Any) -> list[dict[str, Any]]:
    names = [name.strip() for name in re.split(r"[、\n]", str(document or "")) if name.strip()]
    values = list(pages or []) if isinstance(pages, (list, tuple)) else [
        value.strip() for value in str(pages or "").replace("，", ",").split(",") if value.strip()
    ]
    if values and not names:
        names = [""]
    return [{"document_id": "", "document_name": name, "pages": values} for name in names]


def _parse_source_references(
    value: Any, question_number: int, label: str, legacy_document: str = "",
) -> list[dict[str, Any]]:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            try:
                value = json.loads(text)
            except json.JSONDecodeError:
                pass
        if isinstance(value, str):
            parsed = []
            for line in value.replace("\r", "").splitlines():
                if not line.strip():
                    continue
                prefix, separator, page_text = line.rpartition("：")
                if not separator:
                    return _legacy_sources(
                        _parse_source_pages(value, question_number, label), legacy_document
                    )
                match = re.fullmatch(r"(.*?)\s+\[([^\]]+)\]", prefix)
                name, document_id = (match.group(1), match.group(2)) if match else (prefix, "")
                parsed.append({
                    "document_id": document_id, "document_name": name,
                    "pages": _parse_source_pages(page_text, question_number, label),
                })
            value = parsed
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"第 {question_number} 題的{label}必須是文件與頁碼清單")
    references = []
    for raw in value:
        if not isinstance(raw, dict):
            raise ValueError(f"第 {question_number} 題的{label}格式不正確")
        name = str(raw.get("document_name", raw.get("document", "")) or "").strip()
        document_id = str(raw.get("document_id", "") or "").strip()
        pages = _parse_source_pages(raw.get("pages", []), question_number, label)
        if not name and not document_id:
            raise ValueError(f"第 {question_number} 題的{label}缺少來源文件")
        reference = {"document_id": document_id, "document_name": name, "pages": pages}
        existing = next((item for item in references if item["document_id"] == document_id and item["document_name"] == name), None)
        if existing:
            existing["pages"] = list(dict.fromkeys([*existing["pages"], *pages]))
        else:
            references.append(reference)
    return references


def _questions_from_rows(rows: Any) -> list[dict[str, Any]]:
    if hasattr(rows, "values"):
        rows = rows.values.tolist()
    if not isinstance(rows, list) or not rows:
        raise ValueError("題目不可為空")
    questions = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, (list, tuple)) or len(row) < 3:
            raise ValueError(f"第 {index} 列格式不正確")
        raw_number = row[0] if row else index
        try:
            number = int(raw_number) if str(raw_number or "").strip() else index
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"第 {index} 題的題號必須是正整數") from exc
        if number < 1:
            raise ValueError(f"第 {index} 題的題號必須是正整數")
        question = str(row[1] or "").strip()
        answer = str(row[2] or "").strip()
        if not question or not answer:
            raise ValueError(f"第 {index} 題的問題與標準答案不可為空")
        if len(row) == 5:
            question_sources = _parse_source_references(row[3], index, "題目來源")
            answer_sources = _parse_source_references(row[4], index, "答案來源")
        elif len(row) >= 6:
            document = str(row[5] or "")
            question_pages = _parse_source_pages(row[3], index, "題目來源頁碼")
            answer_pages = _parse_source_pages(row[4], index, "答案來源頁碼")
            question_sources = _legacy_sources(question_pages, document)
            answer_sources = _legacy_sources(answer_pages, document)
        else:
            legacy_pages = _parse_source_pages(
                row[3] if len(row) > 3 else "", index, "來源頁碼"
            )
            document = str(row[4] or "") if len(row) > 4 else ""
            question_sources = answer_sources = _legacy_sources(legacy_pages, document)
        question_pages = list(dict.fromkeys(page for ref in question_sources for page in ref["pages"]))
        answer_pages = list(dict.fromkeys(page for ref in answer_sources for page in ref["pages"]))
        documents = list(dict.fromkeys(ref["document_name"] for ref in [*question_sources, *answer_sources] if ref["document_name"]))
        questions.append({
            "number": number, "question": question,
            "expected_answer": answer,
            "question_source_pages": question_pages,
            "answer_source_pages": answer_pages,
            "source_pages": answer_pages,
            "question_sources": question_sources,
            "answer_sources": answer_sources,
            "document": "、".join(documents),
        })
    return questions


def _questions_from_file(file_path: str) -> list[dict[str, Any]]:
    path = Path(file_path)
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        items = payload.get("questions") if isinstance(payload, dict) else payload
        if not isinstance(items, list):
            raise ValueError("JSON 必須是題目陣列或包含 questions 陣列")
        rows = [(
            [item.get("number", index), item.get("question", ""), item.get("expected_answer", ""),
             item.get("question_sources", []), item.get("answer_sources", [])]
            if "question_sources" in item or "answer_sources" in item else
            [item.get("number", index), item.get("question", ""), item.get("expected_answer", ""),
             item.get("question_source_pages", item.get("source_pages", [])),
             item.get("answer_source_pages", item.get("source_pages", [])), item.get("document", "")]
        ) for index, item in enumerate(items, start=1) if isinstance(item, dict)]
    elif path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            items = list(csv.DictReader(handle))
        rows = [(
            [item.get("number", index), item.get("question", ""), item.get("expected_answer", ""),
             item.get("question_sources", ""), item.get("answer_sources", "")]
            if "question_sources" in item or "answer_sources" in item else
            [item.get("number", index), item.get("question", ""), item.get("expected_answer", ""),
             item.get("question_source_pages", item.get("source_pages", "")),
             item.get("answer_source_pages", item.get("source_pages", "")), item.get("document", "")]
        ) for index, item in enumerate(items, start=1)]
    else:
        raise ValueError("只支援 .json 或 .csv 題目檔")
    return _questions_from_rows(rows)


def load_project_question_set_for_ui(project_id: str | None) -> tuple[str, list[list[object]]]:
    if not project_id:
        return "請先載入專案。", []
    try:
        project = load_project(project_id)
    except (OSError, ValueError) as exc:
        return f"❌ 題目集載入失敗：{exc}", []
    question_sets = _project_question_sets(project)
    if not question_sets:
        return "目前尚未匯入題目集。", []
    selected = question_sets[0]
    return (
        f"目前題目集「{selected['name']}」：{len(selected['questions'])} 題。",
        _evaluation_question_rows(selected["questions"]),
    )


def _project_question_sets(
    project: dict[str, Any], project_id: str | None = None,
) -> list[dict[str, Any]]:
    if project_id:
        question_sets = deepcopy(migrate_project_question_sets(project_id))
        return [
            {**item, "questions": _attach_project_document_ids(item.get("questions") or [], project_id)}
            for item in question_sets
        ]
    question_sets = project.get("question_sets") or []
    normalized = [
        {
            **item,
            "question_set_id": str(item.get("question_set_id") or f"legacy-{index}"),
            "name": str(item.get("name") or f"題目集 {index + 1}"),
            "questions": list(item.get("questions") or []),
        }
        for index, item in enumerate(question_sets)
        if isinstance(item, dict) and item.get("questions")
    ]
    if normalized:
        return normalized

    # Older 1-6 imports lived in evaluation.questions. Generated 1-5 sets
    # include generation preferences, so they are deliberately not migrated.
    evaluation = project.get("evaluation") or {}
    legacy_questions = evaluation.get("questions") or []
    preferences = evaluation.get("preferences") or {}
    if legacy_questions and not preferences.get("generation_model"):
        return [{
            "question_set_id": "legacy-imported",
            "name": "舊版匯入題目集",
            "questions": list(legacy_questions),
        }]
    return []


def central_question_sets_for_ui(selected_id: str | None = None) -> tuple[str, Any, list[list[object]]]:
    question_sets = list_question_sets()
    selected = next((item for item in question_sets if item["question_set_id"] == selected_id), None)
    choices = [(f"{item['name']}（{len(item['questions'])} 題）", item["question_set_id"])
               for item in question_sets]
    if selected is None and question_sets:
        selected = question_sets[0]
    status = f"中央題庫共 {len(question_sets)} 份題目集。"
    rows = _evaluation_question_rows(selected.get("questions", [])) if selected else []
    return status, gr.update(choices=choices, value=selected.get("question_set_id") if selected else None), rows


def load_central_question_sets_for_ui(selected_id: str | None = None) -> tuple[str, Any, list[list[object]]]:
    migrate_all_project_question_sets()
    return central_question_sets_for_ui(selected_id)


def import_central_question_set_for_ui(file_path: str | None, name: str | None = None) -> tuple[str, Any, list[list[object]], Any]:
    if not file_path:
        status, selector, rows = central_question_sets_for_ui()
        return "❌ 請選擇 JSON 或 CSV 題目集。", selector, rows, gr.update()
    try:
        questions = _questions_from_file(file_path)
        clean_name = (name or "").strip() or Path(file_path).stem.strip() or "匯入題目集"
        existing_names = {item["name"] for item in list_question_sets()}
        base_name = clean_name
        suffix = 2
        while clean_name in existing_names:
            clean_name = f"{base_name} ({suffix})"
            suffix += 1
        item = create_question_set(clean_name, questions, source_file=Path(file_path).name)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        status, selector, rows = central_question_sets_for_ui()
        return f"❌ 題目集匯入失敗：{exc}", selector, rows, gr.update()
    status, selector, _rows = central_question_sets_for_ui(item["question_set_id"])
    return f"✅ 已匯入中央題目集「{clean_name}」，共 {len(questions)} 題。", selector, _evaluation_question_rows(questions), gr.update(value=None)


def project_question_set_bindings_for_ui(
    project_id: str | None, selected_ids: list[str] | None = None,
) -> tuple[str, Any]:
    if not project_id:
        return "請先選擇專案。", gr.update(choices=[], value=[])
    try:
        migrate_project_question_sets(project_id)
        project = load_project(project_id)
        items = list_question_sets()
        choices = [(f"{item['name']}（{len(item['questions'])} 題）", item["question_set_id"])
                   for item in items]
        selected = list(selected_ids) if selected_ids is not None else list(project.get("question_set_ids") or [])
        selected = [item for item in selected if item in {value for _, value in choices}]
        selected_names = [label.split("（", 1)[0] for label, value in choices if value in selected]
        summary = "、".join(selected_names) if selected_names else "尚未綁定題目集"
        return (
            f"專案「{project['name']}」目前綁定 {len(selected)} 份題目集：{summary}。",
            gr.update(choices=choices, value=selected),
        )
    except (OSError, ValueError) as exc:
        return f"❌ 載入綁定失敗：{exc}", gr.update(choices=[] , value=[])


def save_project_question_set_bindings_for_ui(
    project_id: str | None, selected_ids: list[str] | None,
) -> tuple[str, Any]:
    if not project_id:
        return "❌ 請先選擇專案。", gr.update(choices=[], value=[])
    try:
        requested = list(dict.fromkeys(str(item) for item in (selected_ids or [])))
        project = set_project_question_set_bindings(project_id, requested)
        persisted = load_project(project_id)
        actual = list(persisted.get("question_set_ids") or [])
        if actual != requested:
            raise ValueError("重新讀取後的綁定與選取內容不一致，請重試")
        _status, selector = project_question_set_bindings_for_ui(project_id)
        return f"✅ 已保存並驗證「{project['name']}」的 {len(actual)} 份題目集綁定。", selector
    except (OSError, ValueError) as exc:
        _status, selector = project_question_set_bindings_for_ui(project_id)
        return f"❌ 儲存綁定失敗：{exc}", selector


def bound_question_set_choices_for_ui(project_id: str | None) -> Any:
    if not project_id:
        return gr.update(choices=[], value=[])
    question_sets = _project_question_sets(load_project(project_id), project_id)
    choices = [(f"{item['name']}（{len(item['questions'])} 題）", item["question_set_id"])
               for item in question_sets]
    return gr.update(choices=choices, value=[value for _, value in choices])


def refresh_binding_project_choices_for_ui(project_id: str | None) -> tuple[Any, str, Any]:
    choices = _project_choices()
    available = {value for _, value in choices}
    selected = project_id if project_id in available else (choices[0][1] if choices else None)
    status, question_set_update = project_question_set_bindings_for_ui(selected)
    return gr.update(choices=choices, value=selected), status, question_set_update


def select_experiment_question_sets_for_ui(
    project_id: str | None, selected_ids: list[str] | None,
) -> tuple[list[dict[str, Any]], str]:
    if project_id:
        available = _project_question_sets(load_project(project_id), project_id)
        selected = set(selected_ids or [])
        sets = [item for item in available if item["question_set_id"] in selected]
    else:
        sets = []
    questions = _flatten_question_sets(sets)
    if not sets:
        return [], "尚未選擇可供測試的題目集。"
    names = "、".join(item["name"] for item in sets)
    return questions, f"目前使用 {len(sets)} 份題目集「{names}」，共 {len(questions)} 題。"


def _flatten_question_sets(question_sets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flattened = []
    for question_set in question_sets:
        for question in question_set.get("questions") or []:
            flattened.append({
                **question,
                "question_set_id": question_set.get("question_set_id"),
                "question_set_name": question_set.get("name", "題目集"),
            })
    return flattened


def project_question_sets_for_ui(
    project_id: str | None, selected_id: str | None = None,
) -> tuple[str, Any, Any, list[list[object]]]:
    if not project_id:
        return "請先載入專案。", gr.update(choices=[], value=None), None, []
    try:
        question_sets = _project_question_sets(load_project(project_id))
    except (OSError, ValueError) as exc:
        return f"❌ 題目集載入失敗：{exc}", gr.update(choices=[], value=None), None, []
    choices = [(item["name"], item["question_set_id"]) for item in question_sets]
    selected = next(
        (item for item in question_sets if item["question_set_id"] == selected_id),
        question_sets[0] if question_sets else None,
    )
    selected_value = selected.get("question_set_id") if selected else None
    rows = _evaluation_question_rows(selected.get("questions", [])) if selected else []
    status = (
        f"已載入 {len(question_sets)} 份獨立題目集；目前顯示「{selected['name']}」共 {len(rows)} 題。"
        if selected else "目前尚未匯入題目集。"
    )
    return status, gr.update(choices=choices, value=selected_value), selected_value, rows


def import_project_question_set_for_ui(
    project_id: str | None, file_path: str | None,
) -> tuple[str, Any, Any, list[list[object]], Any]:
    if not project_id:
        return "❌ 請先載入專案。", gr.update(), gr.update(), [], gr.update()
    if not file_path:
        status, choices_update, selected, rows = project_question_sets_for_ui(project_id)
        return "❌ 請選擇 JSON 或 CSV 題目集。", choices_update, selected, rows, gr.update()
    try:
        project = load_project(project_id)
        questions = _questions_from_file(file_path)
        questions = _attach_project_document_ids(questions, project_id)
        question_sets = _project_question_sets(project)
        base_name = Path(file_path).stem.strip() or "匯入題目集"
        existing_names = {item["name"] for item in question_sets}
        name = base_name
        suffix = 2
        while name in existing_names:
            name = f"{base_name} ({suffix})"
            suffix += 1
        question_set = {
            "question_set_id": uuid4().hex,
            "name": name,
            "source_file": Path(file_path).name,
            "questions": questions,
        }
        question_sets.append(question_set)
        save_project(project_id, {"question_sets": question_sets})
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        status, choices_update, selected, rows = project_question_sets_for_ui(project_id)
        return f"❌ 題目集匯入失敗：{exc}", choices_update, selected, rows, gr.update()
    choices = [(item["name"], item["question_set_id"]) for item in question_sets]
    return (
        f"✅ 已匯入並保存獨立題目集「{name}」，共 {len(questions)} 道題目。",
        gr.update(choices=choices, value=question_set["question_set_id"]),
        question_set["question_set_id"], _evaluation_question_rows(questions),
        gr.update(value=None),
    )


def delete_project_question_set_for_ui(
    project_id: str | None, question_set_id: str | None,
) -> tuple[str, Any, Any, list[list[object]]]:
    """Delete only the selected imported question set and refresh the preview."""
    if not project_id:
        return "❌ 請先載入專案。", gr.update(), gr.update(), []
    if not question_set_id:
        status, choices, selected, rows = project_question_sets_for_ui(project_id)
        return "❌ 請先選擇要刪除的題目集。", choices, selected, rows
    try:
        project = load_project(project_id)
        question_sets = _project_question_sets(project)
        target = next(
            (item for item in question_sets if item["question_set_id"] == question_set_id),
            None,
        )
        if target is None:
            raise ValueError("找不到選取的題目集，請重新整理清單後再試")
        if question_set_id == "legacy-imported":
            # This compatibility set is backed by the old imported-question
            # fields rather than project.question_sets.
            evaluation = dict(project.get("evaluation") or {})
            evaluation.pop("questions", None)
            save_project(project_id, {"evaluation": evaluation})
        else:
            question_sets = [
                item for item in question_sets
                if item["question_set_id"] != question_set_id
            ]
            save_project(project_id, {"question_sets": question_sets})
        status, choices, selected, rows = project_question_sets_for_ui(project_id)
        return f"✅ 已刪除題目集「{target['name']}」。{status}", choices, selected, rows
    except (OSError, TypeError, ValueError) as exc:
        status, choices, selected, rows = project_question_sets_for_ui(project_id)
        return f"❌ 題目集刪除失敗：{exc}", choices, selected, rows


def _built_project_choices() -> list[tuple[str, str]]:
    choices = []
    for name, project_id in list_projects():
        try:
            project = load_project(project_id)
        except (OSError, ValueError):
            continue
        graph_state = project.get("graph_state") or {}
        if graph_state.get("neo4j_imported"):
            choices.append((name, project_id))
    return choices


def experiment_project_choices_for_ui() -> list[tuple[str, str]]:
    return [(name, project_id) for name, project_id in list_experiment_projects()]


def create_experiment_project_for_ui(name: str) -> tuple[Any, dict[str, Any], str]:
    try:
        user = current_actor()
        project = create_experiment_project(_user_prefixed_name(name, user), actor=user)
    except (OSError, ValueError) as exc:
        return gr.update(), {}, f"❌ {exc}"
    choices = experiment_project_choices_for_ui()
    return gr.update(choices=choices, value=project["experiment_project_id"]), project, f"✅ 已建立並載入實驗專案「{project['name']}」。"


def delete_experiment_project_for_ui(project_id: str | None) -> tuple[Any, ...]:
    """Delete the selected experiment workspace and clear its visible page state."""
    def unchanged(message: str, selector_update: Any = None) -> tuple[Any, ...]:
        prefix = [gr.update() for _ in range(8)]
        if selector_update is not None:
            prefix[0] = selector_update
        prefix[4] = message
        suffix = [gr.update() for _ in range(9)]
        suffix[0] = message
        return (*prefix, *[gr.update() for _ in range(EXPERIMENT_GROUP_LIMIT * EXPERIMENT_GROUP_COMPONENTS)], *suffix)

    if not isinstance(project_id, str) or not project_id:
        return unchanged("操作已取消，或尚未選擇實驗專案；沒有刪除任何資料。")
    try:
        deleted_name = delete_experiment_project(project_id)
    except (OSError, ValueError) as exc:
        return unchanged(
            f"❌ {exc}",
            gr.update(choices=experiment_project_choices_for_ui(), value=project_id, interactive=True),
        )
    status = f"✅ 已刪除實驗專案「{deleted_name}」；其成員車型專案及 Neo4j 資料庫不受影響。"
    return (
        gr.update(choices=experiment_project_choices_for_ui(), value=None, interactive=True), {}, [],
        gr.update(choices=_built_project_choices(), value=[]), status,
        [], {}, "請先在 2-0 載入實驗專案。",
        *_inline_group_updates([], _configured_service_choice_items(load_service_settings("experiment_llm"))),
        "實驗組設定會自動儲存。", DEFAULT_MAX_CONCURRENT_REQUESTS,
        [], [], [], [], "實驗專案已刪除。",
        "實驗專案已刪除，請先載入或建立實驗專案。",
        gr.update(value="開始評測", interactive=False),
    )


def request_experiment_project_deletion_for_ui(
    project_id: str | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not isinstance(project_id, str) or not project_id:
        return (
            gr.update(value="請先選擇要刪除的實驗專案。", visible=True),
            gr.update(visible=False), gr.update(visible=False),
            gr.update(interactive=True),
        )
    return (
        gr.update(
            value=f"確定刪除實驗專案「{project_id}」？只會刪除實驗設定與結果，不會刪除成員專案或 Neo4j 資料庫。",
            visible=True,
        ),
        gr.update(visible=True), gr.update(visible=True),
        gr.update(interactive=False),
    )


def cancel_experiment_project_deletion_for_ui() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    return (
        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
        gr.update(interactive=True),
    )


def delete_confirmed_experiment_project_for_ui(project_id: str | None) -> tuple[Any, ...]:
    result = delete_experiment_project_for_ui(project_id)
    return (*result, "", gr.update(visible=False), gr.update(visible=False))


def load_experiment_project_for_ui(
    project_id: str | None,
) -> tuple[dict[str, Any], list[list[Any]], Any, str]:
    if not project_id:
        return {}, [], gr.update(choices=_built_project_choices(), value=[]), "請建立或選擇實驗專案。"
    try:
        project = load_experiment_project(project_id)
        project = _migrate_legacy_experiment_project_questions(project)
        members = []
        for member_id in project.get("members", []):
            try:
                members.append(_experiment_member_summary_row(member_id))
            except (OSError, ValueError):
                members.append(["（原專案不存在）", member_id, "未知", 0, 0, 0, "", "", "None", ""])
        return (
            project, members,
            gr.update(choices=_built_project_choices(), value=project.get("members", [])),
            f"✅ 已載入實驗專案「{project['name']}」，包含 {len(members)} 個專案。",
        )
    except (OSError, ValueError) as exc:
        return {}, [], gr.update(choices=_built_project_choices(), value=[]), f"❌ {exc}"


def experiment_project_banner_for_ui(project: dict[str, Any] | None) -> str:
    """Show the active experiment workspace in the global project banner."""
    name = (project or {}).get("name")
    return f"### 📁 目前專案：{name}" if name else _current_project_banner({})


def save_experiment_project_members_for_ui(
    project_id: str | None, member_ids: list[str] | None,
) -> tuple[dict[str, Any], list[list[Any]], str]:
    if not project_id:
        return {}, [], "❌ 請先建立或載入實驗專案。"
    selected = list(dict.fromkeys(member_ids or []))
    available = dict((project_id, name) for name, project_id in _built_project_choices())
    missing = [member_id for member_id in selected if member_id not in available]
    if missing:
        return load_experiment_project(project_id), [], "❌ 成員只能是已完成建圖且仍存在的專案。"
    try:
        current = load_experiment_project(project_id)
        known = dict((member_id, name) for name, member_id in list_projects())
        rows = []
        for member_id in selected:
            row = _experiment_member_summary_row(member_id)
            row[0] = known.get(member_id, row[0])
            rows.append(row)
        current = save_experiment_project(project_id, {
            "members": selected,
        })
        return current, rows, f"✅ 已保存 {len(selected)} 個已建圖專案。"
    except (OSError, ValueError) as exc:
        return load_experiment_project(project_id), [], f"❌ 成員設定保存失敗：{exc}"


def _experiment_member_summary_row(project_id: str) -> list[Any]:
    member = load_project(project_id)
    summary = load_project_summary(project_id)
    question_count = sum(
        len(item.get("questions") or [])
        for item in migrate_project_question_sets(project_id)
    )
    return [
        member.get("name", project_id),
        project_id,
        summary.get("created_by", "未知"),
        summary.get("document_count", 0),
        summary.get("chunk_count", 0),
        question_count,
        summary.get("chunk_size") if summary.get("chunk_size") is not None else "",
        summary.get("chunk_overlap") if summary.get("chunk_overlap") is not None else "",
        summary.get("schema_granularity") or "None",
        member.get("neo4j_database", ""),
    ]


def test_experiment_project_connections_for_ui(
    project_id: str | None, neo4j_uri: str, neo4j_username: str, neo4j_password: str,
) -> tuple[list[list[str]], dict[str, bool], str]:
    if not project_id:
        return [], {}, "❌ 請先載入實驗專案。"
    try:
        project = load_experiment_project(project_id)
    except (OSError, ValueError) as exc:
        return [], {}, f"❌ {exc}"
    rows, statuses = [], {}
    for member_id in project.get("members", []):
        try:
            member = load_project(member_id)
            status, connected = check_neo4j_for_ui(
                neo4j_uri, member.get("neo4j_database", project_database_name(member_id)),
                neo4j_username, neo4j_password,
            )
            rows.append([member.get("name", member_id), member_id,
                         member.get("neo4j_database", ""), status])
            statuses[member_id] = connected
        except (OSError, ValueError) as exc:
            rows.append([member_id, member_id, "", f"❌ {exc}"])
            statuses[member_id] = False
    passed = sum(statuses.values())
    status = f"✅ {passed} / {len(rows)} 個專案連線成功。" if rows and passed == len(rows) else f"⚠️ {passed} / {len(rows)} 個專案連線成功。"
    return rows, statuses, status


def _migrate_legacy_experiment_project_questions(
    project: dict[str, Any],
) -> dict[str, Any]:
    """Move question sets formerly stored on an experiment project onto each member project."""
    legacy_map = project.get("questions_by_project") or {}
    if not legacy_map:
        return project
    for member_id in project.get("members", []):
        legacy_questions = legacy_map.get(member_id) or []
        if not legacy_questions:
            continue
        try:
            member = load_project(member_id)
        except (OSError, ValueError):
            continue
        bound_sets = migrate_project_question_sets(member_id)
        if bound_sets:
            continue
        item = create_question_set(
            "實驗專案舊版題目集",
            legacy_questions,
            question_set_id=f"legacy-experiment-{hashlib.sha256(member_id.encode()).hexdigest()[:16]}",
        )
        set_project_question_set_bindings(member_id, [item["question_set_id"]])
    return project


def _load_experiment_member_question_sets(
    project: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    project = _migrate_legacy_experiment_project_questions(project)
    member_projects: dict[str, dict[str, Any]] = {}
    questions_by_member: dict[str, list[dict[str, Any]]] = {}
    for member_id in project.get("members", []):
        member = load_project(member_id)
        member_projects[member_id] = member
        questions_by_member[member_id] = _flatten_question_sets(
            _project_question_sets(member, member_id)
        )
    return member_projects, questions_by_member


def load_experiment_project_setup_for_ui(
    project: dict[str, Any] | None, experiment_llm_state: dict[str, Any] | None = None,
) -> tuple[Any, ...]:
    groups = (project or {}).get("groups", [])
    group_slots = _inline_group_updates(
        groups, _configured_service_choice_items(
            experiment_llm_state or load_service_settings("experiment_llm")
        ),
    )
    return (
        *group_slots,
        (project or {}).get("max_concurrent_requests", DEFAULT_MAX_CONCURRENT_REQUESTS),
        (project or {}).get("summary_rows", []), (project or {}).get("detail_rows", []),
        (project or {}).get("status", "請設定實驗組並執行。"),
        "實驗組設定會自動儲存。",
        gr.update(value=bool((project or {}).get("verification_enabled", False))),
    )


def load_experiment_project_runtime_state_for_ui(
    project: dict[str, Any] | None, experiment_llm_state: dict[str, Any] | None = None,
) -> tuple[Any, Any, int, list[dict[str, Any]], list[dict[str, Any]], Any, str]:
    project = project or {}
    llm_settings = experiment_llm_state or load_service_settings("experiment_llm")
    choices = _model_choices_with_fallback(
        _configured_service_choice_items(llm_settings), DEFAULT_LLM_MODEL,
    )
    judge_model = (
        project.get("judge_model") or DEFAULT_LLM_MODEL
    )
    judge_effort = project.get("judge_reasoning_effort", DEFAULT_JUDGE_REASONING_EFFORT)
    return (
        gr.update(value=judge_model, choices=choices),
        gr.update(value=judge_effort, visible=str(judge_model).casefold() == GPT_6_LUNA_MODEL,
                  choices=list(GPT_6_LUNA_REASONING_EFFORTS)),
        project.get("judge_max_concurrent_requests", DEFAULT_MAX_CONCURRENT_REQUESTS),
        project.get("pending_answers", []), project.get("results", []),
        gr.update(interactive=bool(project.get("pending_answers"))),
        (f"✅ 已保存 {len(project.get('pending_answers', []))} 個待評測回答。請按「開始評測」。"
         if project.get("pending_answers") else "尚未生成實驗回答；請先按「檢索並生成回答」。"),
    )


def save_experiment_project_groups_from_rows_for_ui(
    project: dict[str, Any] | None, rows: Any,
) -> tuple[dict[str, Any], str]:
    if not project:
        return {}, "❌ 請先載入實驗專案。"
    try:
        submitted = rows.tolist() if hasattr(rows, "tolist") else list(rows or [])
        groups = []
        for row in submitted:
            if len(row) < 6:
                raise ValueError("實驗組欄位不完整")
            name, model, mode, top_k, reranker, expansion = row[:6]
            name, model = str(name or "").strip(), str(model or "").strip()
            if not name and not model:
                continue
            if not name or not model:
                raise ValueError("每個實驗組都要填寫名稱與回答模型")
            top_k = int(top_k)
            retrieval_config = _retrieval_config_from_controls(
                mode, top_k, reranker, expansion,
            )
            groups.append({
                "name": name, "answer_model": model,
                "retrieval_config": retrieval_config.to_dict(),
            })
        if not groups and project.get("groups"):
            return project, "✅ 保留已保存的實驗組設定；如需刪除請使用該列的移除按鈕。"
        if len({item["name"] for item in groups}) != len(groups):
            raise ValueError("實驗組名稱不可重複")
        updated = save_experiment_project(project["experiment_project_id"], {"groups": groups})
    except (OSError, TypeError, ValueError, OverflowError) as exc:
        return project, f"❌ 實驗組設定未儲存：{exc}"
    return updated, f"✅ 已儲存 {len(groups)} 個實驗組設定。"


def save_experiment_project_inline_groups_for_ui(
    project: dict[str, Any] | None, *values: Any, allow_empty: bool = False,
) -> tuple[dict[str, Any], str]:
    if not project:
        return {}, "❌ 請先載入實驗專案。"
    try:
        groups = _groups_from_inline_values(values)
        current = load_experiment_project(project["experiment_project_id"])
        saved_groups = current.get("groups") or []
        if saved_groups and not groups and not allow_empty:
            return current, "✅ 已保留已保存的實驗組；如需刪除，請使用該列的「移除」按鈕。"
        changed = saved_groups != groups
        status = (
            "⚠️ 實驗組設定已變更；畫面保留最近一次測試結果。"
            if current.get("results") else "✅ 實驗組設定已自動儲存。"
        )
        updated = save_experiment_project(current["experiment_project_id"], {
            "groups": groups,
            "pending_answers": [] if changed else current.get("pending_answers", []),
            "status": status,
        })
    except (OSError, TypeError, ValueError, OverflowError) as exc:
        return project, f"❌ 實驗組設定未儲存：{exc}"
    return updated, status


def _experiment_project_inline_group_response(
    project: dict[str, Any], llm_state: dict[str, Any], status: str,
) -> tuple[Any, ...]:
    return (
        *_inline_group_updates(
            list(project.get("groups") or []),
            _configured_service_choice_items(llm_state),
        ),
        project, status,
    )


def add_experiment_project_inline_group_for_ui(
    project: dict[str, Any] | None, llm_state: dict[str, Any], *values: Any,
) -> tuple[Any, ...]:
    if not project:
        return (*[gr.update() for _ in range(EXPERIMENT_GROUP_LIMIT * EXPERIMENT_GROUP_COMPONENTS)], {},
                "❌ 請先載入實驗專案。")
    try:
        groups = _groups_from_inline_values(values)
        if len(groups) >= EXPERIMENT_GROUP_LIMIT:
            raise ValueError(f"最多可設定 {EXPERIMENT_GROUP_LIMIT} 個實驗組")
        choices = _configured_service_choice_items(llm_state)
        available = [value for _label, value in choices]
        model = DEFAULT_LLM_MODEL if DEFAULT_LLM_MODEL in available else next(iter(available), None)
        if not model:
            raise ValueError("請先設定可用的回答模型")
        groups.append({
            "name": _next_experiment_project_group_name(groups),
            "answer_model": model,
            "answer_reasoning_effort": DEFAULT_REASONING_EFFORT,
            "retrieval_config": _default_retrieval_config(),
        })
        updated, status = save_experiment_project_inline_groups_for_ui(
            project, *_inline_group_values(groups), allow_empty=True,
        )
        if status.startswith("❌"):
            raise ValueError(status.removeprefix("❌ "))
        return _experiment_project_inline_group_response(updated, llm_state, status)
    except (OSError, TypeError, ValueError, OverflowError) as exc:
        return _experiment_project_inline_group_response(
            project, llm_state, f"❌ {exc}",
        )


def remove_experiment_project_inline_group_for_ui(
    row_index: int, project: dict[str, Any] | None,
    llm_state: dict[str, Any], *values: Any,
) -> tuple[Any, ...]:
    if not project:
        return (*[gr.update() for _ in range(EXPERIMENT_GROUP_LIMIT * EXPERIMENT_GROUP_COMPONENTS)], {},
                "❌ 請先載入實驗專案。")
    try:
        groups = _groups_from_inline_values(values)
        if not 0 <= row_index < len(groups):
            raise ValueError("找不到要移除的實驗組")
        removed = groups.pop(row_index)
        updated, status = save_experiment_project_inline_groups_for_ui(
            project, *_inline_group_values(groups), allow_empty=True,
        )
        if status.startswith("❌"):
            raise ValueError(status.removeprefix("❌ "))
        status = f"✅ 已移除實驗組「{removed['name']}」。"
        updated = save_experiment_project(updated["experiment_project_id"], {"status": status})
        return _experiment_project_inline_group_response(updated, llm_state, status)
    except (OSError, TypeError, ValueError, OverflowError) as exc:
        return _experiment_project_inline_group_response(
            project, llm_state, f"❌ {exc}",
        )


def save_experiment_project_judge_settings_for_ui(
    project: dict[str, Any] | None, judge_model: str | None, reasoning_effort: str | None,
    max_concurrent_requests: int | float = DEFAULT_MAX_CONCURRENT_REQUESTS,
) -> tuple[dict[str, Any], str]:
    if not project:
        return {}, "⚠️ 請先載入實驗專案。"
    try:
        updated = save_experiment_project(project["experiment_project_id"], {
            "judge_model": judge_model or "",
            "judge_reasoning_effort": reasoning_effort or DEFAULT_JUDGE_REASONING_EFFORT,
            "judge_max_concurrent_requests": max(1, int(max_concurrent_requests)),
        })
        return updated, "✅ 全域評測模型與最大並行請求數已自動儲存。"
    except (OSError, TypeError, ValueError, OverflowError) as exc:
        return project, f"❌ 評測設定保存失敗：{exc}"


def _group_reranker_mode(group: dict[str, Any]) -> str:
    return _group_retrieval_config(group).ui_values()[2]


def _group_expansion_mode(group: dict[str, Any]) -> str:
    return _group_retrieval_config(group).ui_values()[3]


def _selected_retrieval_mode(value: str) -> str:
    if value not in RERANKER_MODES and value not in EVIDENCE_EXPANSION_MODES:
        raise ValueError("檢索後處理選項無效")
    return value


def _experiment_group_rows(groups: list[dict[str, Any]]) -> list[list[object]]:
    return [[item["name"], item["answer_model"], *_group_retrieval_values(item)] for item in groups]


EXPERIMENT_GROUP_LIMIT = 12
EXPERIMENT_GROUP_FIELDS = 8
EXPERIMENT_GROUP_COMPONENTS = EXPERIMENT_GROUP_FIELDS + len(_strategy_parameter_entries()) + 1


def _inline_group_values(groups: list[dict[str, Any]]) -> list[Any]:
    values: list[Any] = []
    for index in range(EXPERIMENT_GROUP_LIMIT):
        item = groups[index] if index < len(groups) else {}
        config = (
            _group_retrieval_config(item) if item else
            RetrievalConfig.from_dict(_default_retrieval_config())
        )
        strategy, top_k, reranker, expansion = config.ui_values()
        values.extend([
            item.get("name", ""), item.get("answer_model"),
            item.get("answer_reasoning_effort", DEFAULT_REASONING_EFFORT),
            strategy, top_k, reranker, expansion,
            _visible_params_for_strategy(config),
        ])
    return values


def _inline_group_updates(
    groups: list[dict[str, Any]], model_choices: list[tuple[str, str]] | None = None,
) -> list[Any]:
    choices = list(model_choices or [])
    available_models = {value for _, value in choices}
    for item in groups:
        model = item.get("answer_model")
        if model and model not in available_models:
            choices.append((f"{model}（目前不可用）", model))
            available_models.add(model)
    values = []
    for index in range(EXPERIMENT_GROUP_LIMIT):
        item = groups[index] if index < len(groups) else {}
        visible = bool(item)
        config = (
            _group_retrieval_config(item) if item else
            RetrievalConfig.from_dict(_default_retrieval_config())
        )
        strategy, top_k, reranker, expansion = config.ui_values()
        model_update = gr.update(value=item.get("answer_model"), visible=visible)
        answer_effort_model = item.get("answer_model")
        if model_choices is not None or choices:
            model_update["choices"] = choices
        values.extend([
            gr.update(value=item.get("name", ""), visible=visible),
            model_update,
            gr.update(
                value=item.get("answer_reasoning_effort", DEFAULT_REASONING_EFFORT),
                visible=visible and str(answer_effort_model or "").casefold() == GPT_6_LUNA_MODEL,
                choices=list(GPT_6_LUNA_REASONING_EFFORTS),
            ),
            gr.update(value=strategy, visible=visible),
            gr.update(value=top_k, visible=visible),
            gr.update(value=reranker, visible=visible),
            gr.update(value=expansion, visible=visible),
            gr.update(value=_visible_params_for_strategy(config)),
            *_experiment_strategy_parameter_updates(config, visible),
            gr.update(visible=visible),
        ])
    return values


def _groups_from_inline_values(values: tuple[Any, ...]) -> list[dict[str, Any]]:
    groups = []
    seen_empty_name = False
    for index in range(EXPERIMENT_GROUP_LIMIT):
        offset = index * EXPERIMENT_GROUP_FIELDS
        (name, answer_model, answer_effort, mode, top_k, reranker,
         expansion, strategy_params) = values[offset:offset + EXPERIMENT_GROUP_FIELDS]
        name = str(name or "").strip()
        if not name:
            if any(str(values[later * EXPERIMENT_GROUP_FIELDS] or "").strip()
                   for later in range(index + 1, EXPERIMENT_GROUP_LIMIT)):
                raise ValueError("請使用該列的「移除」按鈕，不要清空中間列的名稱")
            seen_empty_name = True
            continue
        if seen_empty_name:
            raise ValueError("實驗組列不可留空缺；請使用「移除」按鈕")
        if not answer_model:
            raise ValueError(f"「{name}」尚未選擇回答模型")
        try:
            top_k = int(top_k)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"「{name}」的 Top K 必須是整數") from exc
        if not 1 <= top_k <= 50:
            raise ValueError(f"「{name}」的 Top K 必須介於 1 到 50")
        retrieval_config = _retrieval_config_from_controls(
            mode, top_k, reranker, expansion, strategy_params,
        ).to_dict()
        groups.append({
            "name": name, "answer_model": str(answer_model),
            **_reasoning_effort_record(answer_model, answer_effort, "answer_reasoning_effort"),
            "retrieval_config": retrieval_config,
        })
    names = [item["name"] for item in groups]
    if len(names) != len(set(names)):
        raise ValueError("實驗組名稱不可重複")
    return groups


def _save_experiment_data(project_id: str, data: dict[str, Any]) -> None:
    if not project_id:
        raise ValueError("請先建立或載入專案")
    payload: dict[str, Any] = {"experiment": data}
    save_project(project_id, payload)


def save_inline_experiment_groups_for_ui(
    project_id: str, questions: list[dict[str, Any]], max_concurrent_requests: int | float,
    current_groups: list[dict[str, Any]] | None, *values: Any,
    allow_empty: bool = False,
) -> tuple[str, list[dict[str, Any]], str]:
    try:
        groups = _groups_from_inline_values(values)
        concurrency = int(max_concurrent_requests)
        if concurrency < 1:
            raise ValueError("測試最大並行請求數必須大於 0")
        project = load_project(project_id)
        previous = project.get("experiment") or {}
        saved_groups = previous.get("groups") or []
        if (current_groups or saved_groups) and not groups and not allow_empty:
            # Gradio may send a transient empty component snapshot while the tab
            # is restoring persisted values. Keep the saved groups and avoid
            # turning that UI initialization event into a destructive write.
            return "✅ 已保留已保存的實驗組設定。", list(saved_groups), gr.update()
        result_status = (
            "⚠️ 實驗設定已變更；畫面保留的是最近一次執行結果。"
            if previous.get("results") else "實驗組設定已自動儲存。"
        )
        _save_experiment_data(project_id, {
            **previous, "questions": questions or [], "groups": groups,
            "max_concurrent_requests": concurrency,
            "pending_answers": (
                [] if previous.get("groups") != groups else previous.get("pending_answers", [])
            ),
            "status": result_status,
        })
    except (OSError, TypeError, ValueError) as exc:
        return f"❌ 實驗設定儲存失敗：{exc}", list(current_groups or []), gr.update()
    return "✅ 實驗組與並行設定已自動儲存。", groups, result_status


def add_inline_experiment_group_for_ui(
    project_id: str, questions: list[dict[str, Any]], max_concurrent_requests: int | float,
    llm_state: dict[str, Any], current_groups: list[dict[str, Any]] | None,
    *values: Any,
) -> tuple[Any, ...]:
    try:
        groups = _groups_from_inline_values(values)
        if len(groups) >= EXPERIMENT_GROUP_LIMIT:
            raise ValueError(f"最多可設定 {EXPERIMENT_GROUP_LIMIT} 個實驗組")
        model = (
            DEFAULT_LLM_MODEL if DEFAULT_LLM_MODEL in service_choices(llm_state)
            else preferred_service_model(llm_state)
        )
        if not model:
            raise ValueError("請先設定可用的 LLM 模型")
        next_index = 1
        existing_names = {item["name"] for item in groups}
        while f"實驗組 {next_index}" in existing_names:
            next_index += 1
        groups.append({
            "name": f"實驗組 {next_index}", "answer_model": model,
            "retrieval_config": _default_retrieval_config(),
        })
        status, saved_groups, result_status = save_inline_experiment_groups_for_ui(
            project_id, questions, max_concurrent_requests, groups,
            *_inline_group_values(groups)
        )
        if status.startswith("❌"):
            raise ValueError(status.removeprefix("❌ "))
        return (*_inline_group_updates(saved_groups, service_choice_items(llm_state)), saved_groups, status, result_status)
    except (OSError, TypeError, ValueError) as exc:
        return (
            *_inline_group_updates(list(current_groups or []), service_choice_items(llm_state)),
            list(current_groups or []), f"❌ {exc}", gr.update(),
        )


def remove_inline_experiment_group_for_ui(
    row_index: int, project_id: str, questions: list[dict[str, Any]],
    max_concurrent_requests: int | float, llm_state: dict[str, Any],
    current_groups: list[dict[str, Any]] | None,
    *values: Any,
) -> tuple[Any, ...]:
    groups = _groups_from_inline_values(values)
    if 0 <= row_index < len(groups):
        groups.pop(row_index)
    status, saved_groups, result_status = save_inline_experiment_groups_for_ui(
        project_id, questions, max_concurrent_requests, current_groups,
        *_inline_group_values(groups), allow_empty=True,
    )
    return (
        *_inline_group_updates(saved_groups, service_choice_items(llm_state)),
        saved_groups, status, result_status,
    )


def _experiment_answer_availability(
    pending_answers: list[dict[str, Any]] | None,
) -> tuple[str, dict[str, Any]]:
    answers = pending_answers or []
    if answers:
        return (
            f"✅ 目前有 {len(answers)} 個實驗題次的回答，可以進行評測。",
            gr.update(interactive=True),
        )
    return "尚未生成實驗回答；請先按「檢索並生成回答」。", gr.update(interactive=False)


def experiment_answer_progress_for_ui(run_control: RunControl) -> dict[str, Any]:
    progress = getattr(run_control, "answer_generation_progress", None)
    if not progress:
        return gr.update(value="", visible=False)
    completed, total = progress
    return gr.update(
        value=(
            '<style>.experiment-answer-progress{display:flex;align-items:center;gap:12px;'
            'font-size:14px}.experiment-answer-progress progress{flex:1;height:12px;'
            'accent-color:#2563eb}</style><div class="experiment-answer-progress">'
            f'<progress value="{completed}" max="{total}"></progress>'
            f'<span>回答生成進度：{completed} / {total}</span></div>'
        ),
        visible=True,
    )


def load_experiment_for_ui(
    project_id: str, llm_state: dict[str, Any],
    selected_question_set_ids: list[str] | None = None,
) -> tuple[Any, ...]:
    project: dict[str, Any] = {}
    try:
        project = load_project(project_id) if project_id else {}
        data = project.get("experiment") or {}
        evaluation = project.get("evaluation") or {}
    except (OSError, ValueError) as exc:
        data = {}
        evaluation = {}
        status = f"❌ 實驗資料載入失敗：{exc}"
    else:
        status = data.get("status", "請選擇題目集並設定實驗組。")
    if project:
        project_sets = _project_question_sets(
            project, project_id if project.get("project_id") else None,
        )
        selected = set(selected_question_set_ids) if selected_question_set_ids is not None else {
            item["question_set_id"] for item in project_sets
        }
        question_sets = [item for item in project_sets
                         if item["question_set_id"] in selected]
    else:
        question_sets = []
    questions = _flatten_question_sets(question_sets)
    question_status = (
        f"目前使用 {len(question_sets)} 份題目集，共 {len(questions)} 題。"
        if question_sets else "尚未選擇可供測試的題目集。"
    )
    status = f"{question_status} {status}"
    groups = data.get("groups") or []
    global_judge_model = data.get("judge_model") or DEFAULT_EVALUATION_MODEL
    global_judge_effort = data.get("judge_reasoning_effort") or DEFAULT_JUDGE_REASONING_EFFORT
    results = data.get("results", [])
    pending_answers = data.get("pending_answers") or results
    summary_rows = data.get("summary_rows", [])
    detail_rows = data.get("detail_rows", [])
    if not results:
        for row in detail_rows:
            if len(row) > 6 and isinstance(row[6], str):
                row[6] = row[6].strip() in {"✅ 通過", "正確", "通過", "True", "true", "1"}
    judge_choices = _model_choices_with_fallback(
        service_choice_items(llm_state), global_judge_model,
    )
    if results and all(
        "group_name" in item and "answer_model" in item
        and "question" in item and "expected_answer" in item
        for item in results
    ) and all("answer_model" in group for group in groups):
        summary_rows = _single_experiment_summary_rows(groups, results, global_judge_model)
        detail_rows = _single_experiment_detail_rows(results)
    return (
        questions, f"目前專案題庫共 {len(questions)} 題。", groups, results,
        data.get("max_concurrent_requests", DEFAULT_MAX_CONCURRENT_REQUESTS), status,
        summary_rows, detail_rows,
        *_inline_group_updates(groups, service_choice_items(llm_state)),
        gr.update(
            value=global_judge_model,
            choices=judge_choices,
        ),
        gr.update(
            value=global_judge_effort,
            visible=str(global_judge_model or "").casefold() == GPT_6_LUNA_MODEL,
            choices=list(GPT_6_LUNA_REASONING_EFFORTS),
        ),
        data.get("judge_max_concurrent_requests", DEFAULT_MAX_CONCURRENT_REQUESTS),
        pending_answers,
        *_experiment_answer_availability(pending_answers),
        gr.update(value=bool(data.get("verification_enabled", False))),
    )


def save_experiment_judge_settings_for_ui(
    project_id: str, judge_model: str | None, reasoning_effort: str | None,
    max_concurrent_requests: int | float = DEFAULT_MAX_CONCURRENT_REQUESTS,
) -> str:
    if not project_id:
        return "⚠️ 請先選擇專案。"
    try:
        project = load_project(project_id)
        data = dict(project.get("experiment") or {})
        concurrency = int(max_concurrent_requests)
        if concurrency < 1:
            raise ValueError("評測最大並行請求數必須大於 0")
        data["judge_model"] = judge_model or ""
        data["judge_reasoning_effort"] = reasoning_effort or DEFAULT_JUDGE_REASONING_EFFORT
        data["judge_max_concurrent_requests"] = concurrency
        data["status"] = (
            "⚠️ 實驗設定已變更；畫面保留的是最近一次執行結果。"
            if data.get("results") else "實驗組設定已自動儲存。"
        )
        _save_experiment_data(project_id, data)
    except (OSError, TypeError, ValueError) as exc:
        return f"❌ 評測模型設定儲存失敗：{exc}"
    return "✅ 評測模型設定已自動儲存。"


def refresh_experiment_model_choices_for_ui(
    llm_state: dict[str, Any], *models: str | None,
) -> list[Any]:
    choices = _configured_service_choice_items(llm_state)
    allowed = service_choices(llm_state)
    choice_values = {value for _label, value in choices}
    updates = []
    for model in models:
        model_choices = list(choices)
        if model and model not in choice_values and model not in allowed:
            model_choices.append((f"{model}（目前不可用）", model))
        default_model = (
            DEFAULT_LLM_MODEL if DEFAULT_LLM_MODEL in choice_values
            else (choices[0][1] if choices else None)
        )
        updates.append(gr.update(choices=model_choices, value=model or default_model))
    return updates


def _configured_service_choice_items(state: dict[str, Any]) -> list[tuple[str, str]]:
    """Return the OpenAI allowlist for the selected service purpose."""
    kind = "llm" if state.get("kind") == "experiment_llm" else state.get("kind", "llm")
    try:
        return [(f"OpenAI｜{model}", model) for model in configured_models(kind)]
    except ValueError:
        return []


def add_experiment_group_for_ui(
    name: str | None,
    answer_model: str | None,
    retrieval_mode: str,
    top_k: int | float,
    use_reranker: str,
    expand_evidence: str,
    groups: list[dict[str, Any]] | None,
) -> tuple[str, list[list[object]], list[dict[str, Any]]]:
    current = list(groups or [])
    model = str(answer_model or "").strip()
    if not model:
        return "❌ 請選擇回答模型。", _experiment_group_rows(current), current
    group_name = str(name or "").strip() or f"實驗組 {len(current) + 1}"
    if any(item["name"] == group_name for item in current):
        return f"❌ 實驗組名稱「{group_name}」已存在。", _experiment_group_rows(current), current
    try:
        selected_top_k = int(top_k)
        retrieval_config = _retrieval_config_from_controls(
            retrieval_mode, selected_top_k, use_reranker, expand_evidence,
        ).to_dict()
    except (TypeError, ValueError, OverflowError) as exc:
        return f"❌ {exc}", _experiment_group_rows(current), current
    current.append({
        "name": group_name,
        "answer_model": model,
        "retrieval_config": retrieval_config,
    })
    return f"✅ 已加入「{group_name}」。", _experiment_group_rows(current), current


def save_evaluation_questions_for_ui(
    project_id: str, rows: Any, evaluation: dict[str, Any]
) -> tuple[str, dict[str, Any], list[list[object]]]:
    if not project_id:
        return "❌ 請先建立或載入專案。", evaluation or {}, []
    try:
        questions = _questions_from_rows(rows)
        questions = _preserve_source_document_ids(questions, (evaluation or {}).get("questions"))
        questions = _attach_project_document_ids(questions, project_id)
        updated = dict(evaluation or {})
        updated.update({"questions": questions, "results": [], "pending_answers": [], "dirty": False})
        save_project(project_id, {"evaluation": updated})
    except (OSError, ValueError) as exc:
        failed = dict(evaluation or {})
        if "questions" in locals():
            failed.update({"questions": questions, "results": [], "pending_answers": [], "dirty": True})
        return f"❌ {exc}；自動儲存失敗。", failed, []
    return f"✅ 已自動儲存 {len(questions)} 道題目。", updated, []


def import_evaluation_questions_for_ui(
    project_id: str, file_path: str | None, evaluation: dict[str, Any]
) -> tuple[str, list[list[object]], dict[str, Any], list[list[object]]]:
    current = dict(evaluation or {})

    def unchanged(
        status: str,
    ) -> tuple[str, list[list[object]], dict[str, Any], list[list[object]]]:
        return (
            status,
            _evaluation_question_rows(current.get("questions") or []),
            current,
            _evaluation_result_rows(current.get("results") or []),
        )

    if not project_id:
        return unchanged("❌ 請先建立或載入專案。")
    if not file_path:
        return unchanged("❌ 請選擇 JSON 或 CSV 題目檔。")
    try:
        questions = _questions_from_file(file_path)
        questions = _attach_project_document_ids(questions, project_id)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return unchanged(f"❌ 匯入失敗：{exc}")
    updated = dict(evaluation or {})
    updated.update({"questions": questions, "results": [], "pending_answers": [], "dirty": False})
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
        saved_questions = (load_project(project_id).get("evaluation") or {}).get("questions") or []
        questions = _preserve_source_document_ids(questions, saved_questions)
        questions = _attach_project_document_ids(questions, project_id)
        export_questions = [{
            key: question[key]
            for key in ("number", "question", "expected_answer", "question_sources", "answer_sources")
        } for question in questions]
        output = write_json(
            Path("data/projects") / project_id / "exports" / "questions.json",
            {"questions": export_questions},
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
        item.get("actual_answer", ""),
        _result_score(item),
        item.get("verification_changed"),
        item.get("reason", ""),
    ] for item in results]


def _evaluation_answer_availability(evaluation: dict[str, Any] | None) -> tuple[str, dict[str, Any]]:
    current = evaluation or {}
    answers = current.get("pending_answers") or current.get("results") or []
    if answers:
        evaluated = bool(current.get("results"))
        message = (
            f"✅ 目前有 {len(answers)} 題回答，已完成評測。"
            if evaluated else f"✅ 目前有 {len(answers)} 題回答，可以進行評測。"
        )
        return message, gr.update(interactive=True)
    return "尚未生成測試回答；請先按「檢索並生成回答」。", gr.update(interactive=False)


def update_manual_evaluation_for_ui(
    project_id: str, rows: Any, evaluation: dict[str, Any],
) -> tuple[str, list[list[object]], dict[str, Any]]:
    """Persist 0-2 manual scores and refresh aggregate metrics."""
    current = dict(evaluation or {})
    results = [dict(item) for item in current.get("results") or []]
    if not project_id or not results:
        return "❌ 尚無可修改的測試結果。", _evaluation_result_rows(results), current
    try:
        submitted = rows.tolist() if hasattr(rows, "tolist") else list(rows or [])
        if len(submitted) != len(results):
            raise ValueError("結果列數與已評測題目不符")
        changed = 0
        for item, row in zip(results, submitted):
            if len(row) < 7:
                raise ValueError("測試結果欄位不完整")
            try:
                score = int(row[5])
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("答案判定只能選 0、1 或 2") from exc
            if score not in (0, 1, 2) or float(row[5]) != score:
                raise ValueError("答案判定只能選 0、1 或 2")
            if score != _result_score(item):
                item["score"] = score
                item["passed"] = score == 2
                item["reason"] = "人工評判"
                item["manual_judgment"] = True
                changed += 1
        current["results"] = results
        save_project(project_id, {"evaluation": current})
    except (OSError, TypeError, ValueError) as exc:
        return f"❌ 人工評判保存失敗：{exc}", _evaluation_result_rows(results), current
    summary = _evaluation_summary(results)
    return (f"{summary}\n\n✅ 已保存人工評判變更 {changed} 筆。",
            _evaluation_result_rows(results), current)


def update_manual_evaluation_score_for_ui(
    project_id: str, index: int, score: int, evaluation: dict[str, Any],
) -> tuple[str, list[list[object]], dict[str, Any]]:
    results = (evaluation or {}).get("results") or []
    rows = _evaluation_result_rows(results)
    if index < 0 or index >= len(rows):
        return "❌ 題目編號無效。", rows, evaluation or {}
    try:
        normalized_score = int(score)
    except (TypeError, ValueError, OverflowError):
        return "❌ 答案判定只能選 0、1 或 2。", rows, evaluation or {}
    if normalized_score not in (0, 1, 2):
        return "❌ 答案判定只能選 0、1 或 2。", rows, evaluation or {}
    rows[index][5] = normalized_score
    return update_manual_evaluation_for_ui(project_id, rows, evaluation)


def manual_result_editability_for_ui(enabled: bool) -> dict[str, Any]:
    """Unlock only the answer-score column when requested."""
    locked_columns = [0, 1, 2, 3, 4, 6, 7]
    return gr.update(static_columns=locked_columns if enabled else list(range(8)))


def experiment_table_editability_for_ui(enabled: bool) -> dict[str, Any]:
    """Unlock only the answer-score cell; other columns remain static."""
    return gr.update(interactive=bool(enabled))


def reset_manual_result_editability_for_ui() -> tuple[dict[str, Any], ...]:
    return (
        gr.update(value=False), gr.update(static_columns=list(range(8))),
        gr.update(value=False), gr.update(interactive=False),
    )


def _evaluation_summary(results: list[dict[str, Any]], *, loaded: bool = False) -> str:
    passed = sum(_result_score(item) == 2 for item in results)
    partial = sum(_result_score(item) == 1 for item in results)
    total = len(results)
    recall_at_5 = sum(bool(item.get("recall_at_5")) for item in results) / total
    recall_at_10 = sum(
        bool(item.get(
            "recall_at_10",
            item.get("retrieval_rank") is not None
            and int(item["retrieval_rank"]) <= 10,
        ))
        for item in results
    ) / total
    mrr = sum(float(item.get("reciprocal_rank", 0)) for item in results) / total

    document_totals: dict[str, list[int]] = {}
    for item in results:
        document = str(item.get("document") or "未標示文件")
        stats = document_totals.setdefault(document, [0, 0, 0])
        stats[0] += int(_result_score(item) == 2)
        stats[1] += 1
        stats[2] += int(_result_score(item) == 1)
    document_summary = "\n".join(
        f"- {document}：完全正確 {stats[0]} / {stats[1]}，部分正確 {stats[2]}"
        for document, stats in document_totals.items()
    )

    heading = "已載入測試結果" if loaded else "測試完成"
    failed = sum(_result_score(item) == 0 for item in results)
    accuracy = sum(_result_score(item) for item in results) / (2 * total) * 100
    return (
        f"## {heading}｜完全正確 {passed} 題 / {total} 題  "
        f"\n部分正確：{partial} 題｜錯誤：{failed} 題｜得分正確率：{accuracy:.1f}%"
        f"\n\n### 各 PDF 結果\n{document_summary}"
        f"  \nRecall@5：{recall_at_5:.1%}｜Recall@10：{recall_at_10:.1%}｜MRR：{mrr:.3f}"
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


def _project_document_ids(project_id: str) -> dict[str, str]:
    try:
        project = load_project(project_id)
    except (OSError, ValueError):
        return {}
    return _document_ids_from_project(project)


def _attach_project_document_ids(
    questions: list[dict[str, Any]], project_id: str,
) -> list[dict[str, Any]]:
    ids_by_name = _project_document_ids(project_id)
    if not ids_by_name:
        return questions
    for question in questions:
        for field in ("question_sources", "answer_sources"):
            for reference in question.get(field, []):
                name = str(reference.get("document_name") or "")
                if not reference.get("document_id") and name in ids_by_name:
                    reference["document_id"] = ids_by_name[name]
    return questions


def _preserve_source_document_ids(
    questions: list[dict[str, Any]], previous_questions: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Keep opaque document identities while source cells show only readable names/pages."""
    previous_by_question = {
        (item.get("number"), item.get("question")): item
        for item in (previous_questions or [])
    }
    for question in questions:
        previous = previous_by_question.get((question.get("number"), question.get("question")))
        if not previous:
            continue
        for field in ("question_sources", "answer_sources"):
            old_references = previous.get(field) or []
            for reference in question.get(field, []):
                if reference.get("document_id"):
                    continue
                same_name = [
                    old for old in old_references
                    if old.get("document_name") == reference.get("document_name")
                    and old.get("document_id")
                ]
                if len(same_name) == 1:
                    reference["document_id"] = same_name[0]["document_id"]
                elif same_name:
                    new_pages = set(reference.get("pages", []))
                    overlapping = [old for old in same_name if new_pages.intersection(old.get("pages", []))]
                    if len(overlapping) == 1:
                        reference["document_id"] = overlapping[0]["document_id"]
    return questions


def _retrieval_rank(
    question: dict[str, Any], rows: list[list[object]],
    document_ids_by_name: dict[str, str] | None = None,
) -> int | None:
    document_ids_by_name = document_ids_by_name or {}
    references = question.get("answer_sources") or _legacy_answer_sources(question)
    expected_sources: set[tuple[tuple[str, str], int]] = set()
    for reference in references:
        name = str(reference.get("document_name") or "")
        document_id = str(reference.get("document_id") or "")
        for page in reference.get("pages", []):
            identity = (
                ("id", document_id or document_ids_by_name[name])
                if document_id or name in document_ids_by_name
                else ("name", name)
            )
            if document_id or name:
                expected_sources.add((identity, int(page)))
                if document_id and document_id not in document_ids_by_name.values() and name:
                    expected_sources.add((("name", name), int(page)))
    if not expected_sources:
        document = str(question.get("document") or "")
        expected_pages = {
            int(value) for value in question.get(
                "answer_source_pages", question.get("source_pages", [])
            )
        }
        if document and expected_pages:
            identity = (
                ("id", document_ids_by_name[document])
                if document in document_ids_by_name else ("name", document)
            )
            expected_sources = {(identity, page) for page in expected_pages}
    if not expected_sources:
        return None
    ranked_pages: list[tuple[tuple[str, str], int]] = []
    seen_pages: set[tuple[tuple[str, str], int]] = set()
    for row in rows:
        documents = [value for value in re.split(r"[、\n]", str(row[6])) if value]
        for source_document in documents:
            pages = _source_values_for_document(row[4], row[6], source_document)
            for page in sorted(pages):
                identity = (
                    ("id", document_ids_by_name[source_document])
                    if source_document in document_ids_by_name
                    else ("name", source_document)
                )
                source = (identity, page)
                if source not in seen_pages:
                    seen_pages.add(source)
                    ranked_pages.append(source)
    for rank, source in enumerate(ranked_pages, start=1):
        if source in expected_sources:
            return rank
    return None


def load_evaluation_for_ui(project_id: str) -> tuple[Any, ...]:
    default_retrieval = RetrievalConfig.from_dict(_default_retrieval_config())
    if not project_id:
        return ({}, [], [], *([gr.update()] * 11),
                *([gr.update(value=DEFAULT_REASONING_EFFORT, visible=False)] * 3),
                *_evaluation_strategy_parameter_values(default_retrieval),
                "請先選擇專案。", gr.update(value=False))
    try:
        project = load_project(project_id)
    except (OSError, ValueError) as exc:
        return ({}, [], [], *([gr.update()] * 11),
                *([gr.update(value=DEFAULT_REASONING_EFFORT, visible=False)] * 3),
                *_evaluation_strategy_parameter_values(default_retrieval),
                f"❌ {exc}", gr.update(value=False))
    evaluation = dict(project.get("evaluation") or {})
    evaluation.setdefault("dirty", False)
    preferences = evaluation.get("preferences") or {}
    retrieval_config = RetrievalConfig.from_dict(
        preferences.get("retrieval_config") or _default_retrieval_config()
    )
    retrieval_strategy_label, retrieval_top_k, reranker_label, expansion_label = (
        retrieval_config.ui_values()
    )
    questions = evaluation.get("questions") or []
    results = evaluation.get("results") or []
    generation_model = preferences.get("generation_model", DEFAULT_LLM_MODEL)
    test_model = preferences.get("test_model", DEFAULT_LLM_MODEL)
    judge_model = preferences.get("judge_model") or DEFAULT_EVALUATION_MODEL
    return (
        evaluation, _evaluation_question_rows(questions), _evaluation_result_rows(results),
        generation_model, test_model, preferences.get("question_count", 10),
        retrieval_strategy_label, retrieval_top_k,
        preferences.get("allow_parallel_generation", False),
        reranker_label, expansion_label,
        preferences.get("test_max_concurrent_requests", DEFAULT_MAX_CONCURRENT_REQUESTS),
        judge_model,
        preferences.get("judge_max_concurrent_requests", DEFAULT_MAX_CONCURRENT_REQUESTS),
        gr.update(value=preferences.get("generation_reasoning_effort", DEFAULT_REASONING_EFFORT),
                  visible=str(generation_model or "").casefold() == GPT_6_LUNA_MODEL,
                  choices=list(GPT_6_LUNA_REASONING_EFFORTS)),
        gr.update(value=preferences.get("test_reasoning_effort", DEFAULT_REASONING_EFFORT),
                  visible=str(test_model or "").casefold() == GPT_6_LUNA_MODEL,
                  choices=list(GPT_6_LUNA_REASONING_EFFORTS)),
        gr.update(value=preferences.get("judge_reasoning_effort", DEFAULT_JUDGE_REASONING_EFFORT),
                  visible=str(judge_model or "").casefold() == GPT_6_LUNA_MODEL,
                  choices=list(GPT_6_LUNA_REASONING_EFFORTS)),
        *_evaluation_strategy_parameter_values(retrieval_config),
        (_evaluation_summary(results, loaded=True) if results else
         f"已載入 {len(questions)} 道題目與 0 筆測試結果。"),
        gr.update(value=bool(evaluation.get("verification_enabled", False))),
    )


def save_evaluation_preferences_for_ui(
    project_id: str, generation_model: str, test_model: str, question_count: int,
    retrieval_mode: str, top_k: int,
    allow_parallel_generation: bool = False,
    test_max_concurrent_requests: int = DEFAULT_MAX_CONCURRENT_REQUESTS,
    use_reranker: str = "停用",
    expand_evidence: str = "停用",
    judge_model: str | None = None,
    generation_reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    test_reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    judge_reasoning_effort: str = DEFAULT_JUDGE_REASONING_EFFORT,
    judge_max_concurrent_requests: int = DEFAULT_MAX_CONCURRENT_REQUESTS,
    strategy_params_json: Any = None,
) -> str:
    if not project_id:
        return "⚠️ 請先選擇專案。"
    try:
        test_concurrency = int(test_max_concurrent_requests)
        if test_concurrency < 1:
            raise ValueError("最大並行請求數必須大於 0")
        judge_concurrency = int(judge_max_concurrent_requests)
        if judge_concurrency < 1:
            raise ValueError("評測最大並行請求數必須大於 0")
        project = load_project(project_id)
        evaluation = dict(project.get("evaluation") or {})
        retrieval_config = _retrieval_config_from_controls(
            retrieval_mode, int(top_k), use_reranker, expand_evidence,
            strategy_params_json,
        )
        evaluation["preferences"] = {
            "generation_model": generation_model, "test_model": test_model,
            "judge_model": judge_model or test_model,
            "question_count": int(question_count),
            "retrieval_config": retrieval_config.to_dict(),
            "allow_parallel_generation": bool(allow_parallel_generation),
            "test_max_concurrent_requests": int(test_max_concurrent_requests),
            "judge_max_concurrent_requests": judge_concurrency,
            "generation_reasoning_effort": generation_reasoning_effort or DEFAULT_REASONING_EFFORT,
            "test_reasoning_effort": test_reasoning_effort or DEFAULT_REASONING_EFFORT,
            "judge_reasoning_effort": judge_reasoning_effort or DEFAULT_JUDGE_REASONING_EFFORT,
        }
        save_project(project_id, {"evaluation": evaluation})
    except (OSError, TypeError, ValueError) as exc:
        return f"❌ 自動保存失敗：{exc}"
    return "✅ 自動測試設定已保存。"


def generate_evaluation_for_ui(
    project_id: str, model_endpoint: str, api_key: str, generation_model: str,
    test_model: str, question_count: int, retrieval_mode: str, top_k: int,
    chunks: list[TextChunk], allow_parallel_generation: bool = False,
    test_max_concurrent_requests: int = DEFAULT_MAX_CONCURRENT_REQUESTS,
    use_reranker: str = "停用",
    expand_evidence: str = "停用",
    reasoning_effort: str | None = None,
    strategy_params_json: Any = None,
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
        accepted_questions: list[dict[str, Any]] = []
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
                    focus_page, _ = sample_random_page_context(selected_chunks)
                    page_context = expand_page_context_until_complete(
                        model_endpoint,
                        api_key,
                        generation_model,
                        selected_chunks,
                        focus_page,
                        **_reasoning_effort_kwargs(generation_model, reasoning_effort),
                    )
                    batch = generate_evaluation_questions(
                        model_endpoint,
                        api_key,
                        generation_model,
                        page_context,
                        1,
                        excluded,
                        focus_page,
                        **_reasoning_effort_kwargs(generation_model, reasoning_effort),
                    )
                except ValueError as exc:
                    last_error = str(exc)
                    continue
                for question in batch:
                    with accepted_lock:
                        if any(
                            evaluation_questions_are_similar(question, existing)
                            for existing in accepted_questions
                        ):
                            continue
                        accepted_questions.append(question)
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
        retrieval_config = _retrieval_config_from_controls(
            retrieval_mode, int(top_k), use_reranker, expand_evidence,
            strategy_params_json,
        )
        existing_preferences = dict(existing_evaluation.get("preferences") or {})
        evaluation = {
            **existing_evaluation,
            "preferences": {**existing_preferences, "generation_model": generation_model, "test_model": test_model,
                            "question_count": int(question_count),
                            "retrieval_config": retrieval_config.to_dict(),
                            "allow_parallel_generation": bool(allow_parallel_generation),
                            "test_max_concurrent_requests": int(test_max_concurrent_requests),
                            "generation_reasoning_effort": reasoning_effort or DEFAULT_REASONING_EFFORT},
            "questions": questions, "results": [], "pending_answers": [], "dirty": False,
        }
        save_project(project_id, {"evaluation": evaluation})
    except (OSError, TypeError, ValueError) as exc:
        return f"❌ {exc}", [], {}, []
    return (
        f"✅ 已從 {len(document_chunks)} 份 PDF 各建立 {count} 道題目，共 {len(questions)} 道。",
        _evaluation_question_rows(questions), evaluation, [],
    )


def generate_evaluation_answers_for_ui(
    project_id: str, model_endpoint: str, api_key: str,
    embedding_api_base: str, embedding_api_key: str,
    neo4j_uri: str, neo4j_database: str, neo4j_username: str, neo4j_password: str,
    model: str, retrieval_mode: str, top_k: int, evaluation: dict[str, Any],
    max_concurrent_requests: int = DEFAULT_MAX_CONCURRENT_REQUESTS,
    use_reranker: str = "停用",
    expand_evidence: str = "停用",
    answer_reasoning_effort: str | None = None,
    strategy_params: Any = None,
    progress=gr.Progress(),
) -> tuple[str, dict[str, Any], list[list[object]]]:
    """Generate and persist answers without exposing them in the results table."""
    questions = (evaluation or {}).get("questions") or []
    if not project_id:
        return "❌ 請先建立或載入專案。", evaluation or {}, []
    if not questions:
        return "❌ 請先建立測試題目。", evaluation or {}, []
    if (evaluation or {}).get("dirty"):
        return "❌ 題目尚未完成自動儲存，請稍後再試。", evaluation, []
    if not model:
        return "❌ 請選擇回答模型。", evaluation, []
    try:
        retrieval_config = _retrieval_config_from_controls(
            retrieval_mode, int(top_k), use_reranker, expand_evidence,
            strategy_params,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        return f"❌ {exc}", evaluation, []
    _, _, reranker_mode, expansion_mode = retrieval_config.ui_values()
    # Recall@10 requires fetching at least ten candidates, even when the
    # selected answer Top K is lower. Keep the internal candidate budget in
    # sync with that effective value before passing explicit strategy params;
    # otherwise the downstream config validator rejects candidate_top_k < 10.
    answer_top_k = max(int(top_k), 10)
    answer_strategy_params = dict(retrieval_config.params)
    if "candidate_top_k" in answer_strategy_params:
        answer_strategy_params["candidate_top_k"] = max(
            answer_strategy_params["candidate_top_k"], answer_top_k,
        )
    try:
        concurrency = int(max_concurrent_requests)
        if concurrency < 1:
            raise ValueError("測試最大並行請求數必須大於 0")
    except (TypeError, ValueError) as exc:
        return f"❌ {exc}", evaluation, []

    document_ids_by_name = _project_document_ids(project_id)

    def answer(item: dict[str, Any]) -> dict[str, Any]:
        status, actual, evidence_rows = answer_question_for_ui(
            model_endpoint, api_key, embedding_api_base, embedding_api_key,
            neo4j_uri, neo4j_database, neo4j_username, neo4j_password,
            model, item["question"], retrieval_mode, answer_top_k,
            reranker_mode, expansion_mode,
            **_reasoning_effort_kwargs(model, answer_reasoning_effort),
            reranker_mode=reranker_mode,
            evidence_expansion_mode=expansion_mode,
            strategy_params=answer_strategy_params,
        )
        rank = _retrieval_rank(item, evidence_rows, document_ids_by_name)
        return {
            **item,
            "answer_model": model,
            "retrieval_config": retrieval_config.to_dict(),
            **_reasoning_effort_record(model, answer_reasoning_effort, "answer_reasoning_effort"),
            "actual_answer": actual,
            "answer_status": status,
            "retrieval_rank": rank,
            "recall_at_5": rank is not None and rank <= 5,
            "recall_at_10": rank is not None and rank <= 10,
            "reciprocal_rank": 1 / rank if rank else 0.0,
        }

    answers: list[dict[str, Any] | None] = [None] * len(questions)
    try:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {executor.submit(answer, item): index for index, item in enumerate(questions)}
            for completed, future in enumerate(as_completed(futures), start=1):
                answers[futures[future]] = future.result()
                progress(completed / len(questions), desc=f"已生成 {completed} / {len(questions)} 題回答")
        updated = dict(evaluation)
        updated["pending_answers"] = [item for item in answers if item is not None]
        updated["results"] = []
        save_project(project_id, {"evaluation": updated})
    except (OSError, TypeError, ValueError) as exc:
        return f"❌ 回答生成或保存失敗：{exc}", evaluation, []
    return f"✅ 已生成並保存 {len(answers)} 題回答；尚未顯示，請按「進行評測」。", updated, []


def evaluate_generated_answers_for_ui(
    project_id: str, judge_model_endpoint: str, judge_api_key: str,
    judge_model: str, evaluation: dict[str, Any],
    max_concurrent_requests: int = DEFAULT_MAX_CONCURRENT_REQUESTS,
    judge_reasoning_effort: str | None = None,
    verification_enabled: bool = False,
    progress=gr.Progress(),
) -> tuple[str, list[list[object]], dict[str, Any]]:
    """Judge saved answers and only then display the complete per-question results."""
    current = dict(evaluation or {})
    pending = current.get("pending_answers") or []
    if not project_id:
        return "❌ 請先建立或載入專案。", [], current
    if not pending:
        return "❌ 請先按「檢索並生成回答」完成回答生成。", [], current
    if not judge_model or not judge_model_endpoint:
        return "❌ 請選擇可用的評測模型。", [], current
    judge_reasoning_effort = judge_reasoning_effort or DEFAULT_JUDGE_REASONING_EFFORT
    try:
        concurrency = int(max_concurrent_requests)
        if concurrency < 1:
            raise ValueError("評測最大並行請求數必須大於 0")
    except (TypeError, ValueError) as exc:
        return f"❌ {exc}", [], current

    def evaluate(item: dict[str, Any]) -> dict[str, Any]:
        status = str(item.get("answer_status") or "")
        if status.startswith("✅"):
            try:
                judgment = judge_evaluation_answer(
                    judge_model_endpoint, judge_api_key, judge_model,
                    item["question"], item["expected_answer"], item.get("actual_answer", ""),
                    **_reasoning_effort_kwargs(judge_model, judge_reasoning_effort),
                )
                judgment = _verify_judgment_if_enabled(
                    judgment, verification_enabled, judge_model_endpoint, judge_api_key,
                    judge_model, item["question"], item["expected_answer"],
                    item.get("actual_answer", ""), judge_reasoning_effort,
                )
            except ValueError as exc:
                judgment = {"passed": False, "reason": f"評判失敗：{exc}"}
        else:
            judgment = {"passed": False, "reason": status or "回答生成失敗"}
        judgment.setdefault("verification_enabled", bool(verification_enabled))
        judgment.setdefault("verification_changed", None)
        return {
            **item,
            "judge_model": judge_model,
            **_reasoning_effort_record(judge_model, judge_reasoning_effort, "judge_reasoning_effort"),
            **_normalize_judgment(judgment),
        }

    results: list[dict[str, Any] | None] = [None] * len(pending)
    try:
        with ThreadPoolExecutor(max_workers=min(concurrency, len(pending))) as executor:
            futures = {executor.submit(evaluate, item): index for index, item in enumerate(pending)}
            for completed, future in enumerate(as_completed(futures), start=1):
                results[futures[future]] = future.result()
                progress(completed / len(pending), desc=f"已評測 {completed} / {len(pending)} 題")
        final_results = [item for item in results if item is not None]
        current["results"] = final_results
        current["judge_model"] = judge_model
        current["verification_enabled"] = bool(verification_enabled)
        save_project(project_id, {"evaluation": current})
    except (OSError, TypeError, ValueError) as exc:
        return f"❌ 評測或保存失敗：{exc}", [], current
    return _evaluation_summary(final_results), _evaluation_result_rows(final_results), current


def run_evaluation_for_ui(
    project_id: str, model_endpoint: str, api_key: str,
    embedding_api_base: str, embedding_api_key: str,
    neo4j_uri: str, neo4j_database: str, neo4j_username: str, neo4j_password: str,
    model: str, retrieval_mode: str, top_k: int, evaluation: dict[str, Any],
    max_concurrent_requests: int = DEFAULT_MAX_CONCURRENT_REQUESTS,
    use_reranker: str = "停用",
    expand_evidence: str = "停用",
    judge_model_endpoint: str | None = None,
    judge_api_key: str | None = None,
    judge_model: str | None = None,
    answer_reasoning_effort: str | None = None,
    judge_reasoning_effort: str | None = None,
    verification_enabled: bool = False,
    progress=gr.Progress(),
) -> tuple[str, list[list[object]], dict[str, Any]]:
    questions = evaluation.get("questions") if evaluation else None
    if not project_id:
        return "❌ 請先建立或載入專案。", [], evaluation or {}
    if not questions:
        return "❌ 請先建立測試題目。", [], evaluation or {}
    if evaluation.get("dirty"):
        return "❌ 題目或答案尚未完成自動儲存，請稍後再試。", [], evaluation
    effective_judge_model = judge_model or (model if judge_model_endpoint is None else "")
    if not model or not effective_judge_model:
        return "❌ 請選擇回答模型與評測模型。", [], evaluation
    if judge_model_endpoint is not None and not judge_model_endpoint:
        return "❌ 無法解析評測模型的服務設定。", [], evaluation
    judge_reasoning_effort = judge_reasoning_effort or DEFAULT_JUDGE_REASONING_EFFORT
    try:
        retrieval_config = _retrieval_config_from_controls(
            retrieval_mode, int(top_k), use_reranker, expand_evidence,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        return f"❌ {exc}", [], evaluation
    _, _, reranker_mode, expansion_mode = retrieval_config.ui_values()
    concurrency = int(max_concurrent_requests)
    if concurrency < 1:
        return "❌ 測試最大並行請求數必須大於 0", [], evaluation
    document_ids_by_name = _project_document_ids(project_id)
    def evaluate(item: dict[str, Any]) -> dict[str, Any]:
        status, actual, evidence_rows = answer_question_for_ui(
            model_endpoint, api_key, embedding_api_base, embedding_api_key,
            neo4j_uri, neo4j_database, neo4j_username, neo4j_password,
            model, item["question"], retrieval_mode, max(int(top_k), 10),
            reranker_mode, expansion_mode,
            **_reasoning_effort_kwargs(model, answer_reasoning_effort),
            reranker_mode=reranker_mode,
            evidence_expansion_mode=expansion_mode,
        )
        retrieval_rank = _retrieval_rank(item, evidence_rows, document_ids_by_name)
        if status.startswith("✅"):
            try:
                judgment = judge_evaluation_answer(
                    judge_model_endpoint or model_endpoint,
                    api_key if judge_api_key is None else judge_api_key,
                    effective_judge_model, item["question"],
                    item["expected_answer"], actual,
                    **_reasoning_effort_kwargs(effective_judge_model, judge_reasoning_effort),
                )
                judgment = _verify_judgment_if_enabled(
                    judgment, verification_enabled,
                    judge_model_endpoint or model_endpoint,
                    api_key if judge_api_key is None else judge_api_key,
                    effective_judge_model, item["question"], item["expected_answer"],
                    actual, judge_reasoning_effort,
                )
            except ValueError as exc:
                judgment = {"passed": False, "reason": f"評判失敗：{exc}"}
        else:
            judgment = {"passed": False, "reason": status}
        judgment.setdefault("verification_enabled", bool(verification_enabled))
        judgment.setdefault("verification_changed", None)
        return {
            **item,
            "answer_model": model,
            "judge_model": effective_judge_model,
            "retrieval_config": retrieval_config.to_dict(),
            **_reasoning_effort_record(model, answer_reasoning_effort, "answer_reasoning_effort"),
            **_reasoning_effort_record(effective_judge_model, judge_reasoning_effort, "judge_reasoning_effort"),
            "actual_answer": actual,
            "retrieval_rank": retrieval_rank,
            "recall_at_5": retrieval_rank is not None and retrieval_rank <= 5,
            "recall_at_10": retrieval_rank is not None and retrieval_rank <= 10,
            "reciprocal_rank": 1 / retrieval_rank if retrieval_rank else 0.0,
            **_normalize_judgment(judgment),
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
    updated["verification_enabled"] = bool(verification_enabled)
    try:
        save_project(project_id, {"evaluation": updated})
    except (OSError, ValueError) as exc:
        return f"❌ 測試已完成，但保存失敗：{exc}", _evaluation_result_rows(results), updated
    return _evaluation_summary(results), _evaluation_result_rows(results), updated


def run_experiment_groups_for_ui(
    project_id: str,
    questions: list[dict[str, Any]],
    groups: list[dict[str, Any]],
    max_concurrent_requests: int | float,
    llm_state: dict[str, Any],
    embedding_api_base: str,
    embedding_api_key: str,
    neo4j_uri: str,
    neo4j_database: str,
    neo4j_username: str,
    neo4j_password: str,
    judge_model: str | None = None,
    judge_reasoning_effort: str | None = None,
    persist: bool = True,
    verification_enabled: bool = False,
    progress=gr.Progress(),
) -> tuple[str, list[list[object]], list[list[object]], list[dict[str, Any]]]:
    if not project_id:
        return "❌ 請先建立或載入專案。", [], [], []
    if not questions:
        return "❌ 請先在 0-3 為此專案綁定題目集，再回到 1-7 選擇。", [], [], []
    if not groups:
        return "❌ 請至少加入一個實驗組。", [], [], []
    try:
        concurrency = int(max_concurrent_requests)
    except (TypeError, ValueError, OverflowError):
        return "❌ 測試最大並行請求數必須是整數。", [], [], []
    if concurrency < 1:
        return "❌ 測試最大並行請求數必須大於 0。", [], [], []

    project = load_project(project_id)
    previous = project.get("experiment") or {}
    effective_judge_model = str(
        judge_model or previous.get("judge_model")
        or next((group.get("judge_model") for group in groups if group.get("judge_model")), "")
        or groups[0].get("answer_model", "")
    ).strip()
    if not effective_judge_model:
        return "❌ 請選擇全域評測模型。", [], [], []
    effective_judge_effort = (
        judge_reasoning_effort or previous.get("judge_reasoning_effort")
        or next((group.get("judge_reasoning_effort") for group in groups if group.get("judge_reasoning_effort")), None)
        or DEFAULT_JUDGE_REASONING_EFFORT
    )
    persisted_groups = [{
        key: value for key, value in group.items()
        if key not in {"judge_model", "judge_reasoning_effort"}
    } for group in groups]

    judge_endpoint, judge_key = resolve_model_credentials_for_ui(llm_state, effective_judge_model)
    if not judge_endpoint:
        return "❌ 無法解析全域評測模型的服務設定。", [], [], []
    credentials: list[tuple[str, str]] = []
    document_ids_by_name = _project_document_ids(project_id)
    for group in groups:
        endpoint, key = resolve_model_credentials_for_ui(llm_state, group["answer_model"])
        if not endpoint:
            return f"❌ 無法解析實驗組「{group['name']}」的回答模型服務。", [], [], []
        credentials.append((endpoint, key))

    tasks = [
        (group_index, question_index)
        for group_index in range(len(groups))
        for question_index in range(len(questions))
    ]
    results: list[dict[str, Any]] = [{} for _ in tasks]

    def evaluate(task_index: int) -> dict[str, Any]:
        group_index, question_index = tasks[task_index]
        group = groups[group_index]
        item = questions[question_index]
        endpoint, key = credentials[group_index]
        strategy, top_k, reranker_mode, expansion_mode = _group_retrieval_values(group)
        status, actual, evidence_rows = answer_question_for_ui(
            endpoint, key, embedding_api_base, embedding_api_key,
            neo4j_uri, neo4j_database, neo4j_username, neo4j_password,
            group["answer_model"], item["question"], strategy,
            top_k, reranker_mode, expansion_mode,
            **_reasoning_effort_kwargs(
                group["answer_model"],
                group.get("answer_reasoning_effort"),
            ),
        )
        rank = _retrieval_rank(item, evidence_rows, document_ids_by_name)
        if status.startswith("✅"):
            try:
                judgment = judge_evaluation_answer(
                    judge_endpoint, judge_key, effective_judge_model, item["question"],
                    item["expected_answer"], actual,
                    **_reasoning_effort_kwargs(
                        effective_judge_model, effective_judge_effort,
                    ),
                )
            except ValueError as exc:
                judgment = {"passed": False, "reason": f"評判失敗：{exc}"}
        else:
            judgment = {"passed": False, "reason": status}
        if status.startswith("✅") and "verification_error" not in judgment:
            judgment = _verify_judgment_if_enabled(
                judgment, verification_enabled, judge_endpoint, judge_key,
                effective_judge_model, item["question"], item["expected_answer"],
                actual, effective_judge_effort,
            )
        judgment.setdefault("verification_enabled", bool(verification_enabled))
        judgment.setdefault("verification_changed", None)
        return {
            "group_index": group_index,
            "group_name": group["name"],
            "answer_model": group["answer_model"],
            "retrieval_config": _group_retrieval_config(group).to_dict(),
            "judge_model": effective_judge_model,
            **_reasoning_effort_record(
                group["answer_model"], group.get("answer_reasoning_effort"), "answer_reasoning_effort",
            ),
            **_reasoning_effort_record(
                effective_judge_model, effective_judge_effort, "judge_reasoning_effort",
            ),
            "number": item.get("number", question_index + 1),
            "question": item["question"],
            "question_set_id": item.get("question_set_id", ""),
            "question_set_name": item.get("question_set_name", "未分類題目集"),
            "expected_answer": item["expected_answer"],
            "document": item.get("document", ""),
            "actual_answer": actual,
            "retrieval_rank": rank,
            "recall_at_5": rank is not None and rank <= 5,
            "recall_at_10": rank is not None and rank <= 10,
            "reciprocal_rank": 1 / rank if rank else 0.0,
            **_normalize_judgment(judgment),
        }

    try:
        with ThreadPoolExecutor(max_workers=min(concurrency, len(tasks))) as executor:
            futures = {
                executor.submit(evaluate, index): index
                for index in range(len(tasks))
            }
            completed = 0
            for future in as_completed(futures):
                if future.cancelled():
                    continue
                result = future.result()
                results[futures[future]] = result
                completed += 1
                progress(
                    completed / len(tasks),
                    desc=f"已完成 {completed} / {len(tasks)} 個實驗題次",
                )
    except (OSError, TypeError, ValueError) as exc:
        return f"❌ 實驗執行失敗：{exc}", [], [], []

    completed_results = results
    summary_rows = _single_experiment_summary_rows(
        groups, completed_results, effective_judge_model,
    )
    detail_rows = [[
        item["group_name"], item["answer_model"], item["judge_model"],
        item["number"], item["question"], item["document"],
        item["expected_answer"], item["actual_answer"],
        _result_score(item),
        item.get("verification_changed"), item["reason"],
        item["retrieval_rank"],
    ] for item in completed_results]
    status = f"✅ 已完成 {len(groups)} 個實驗組，共 {len(tasks)} 個題次；結果已自動儲存。"
    if persist:
        try:
            project = load_project(project_id)
            previous = project.get("experiment") or {}
            _save_experiment_data(project_id, {
                **previous, "questions": questions, "groups": persisted_groups,
                "max_concurrent_requests": concurrency,
                "judge_model": effective_judge_model,
                "judge_reasoning_effort": effective_judge_effort,
                "verification_enabled": bool(verification_enabled),
                "results": completed_results, "summary_rows": summary_rows,
                "detail_rows": detail_rows, "status": status,
            })
        except (OSError, ValueError) as exc:
            status = f"⚠️ 實驗已完成，但結果保存失敗：{exc}"
    return status, summary_rows, detail_rows, completed_results


def _single_experiment_summary_rows(
    groups: list[dict[str, Any]], results: list[dict[str, Any]], judge_model: str,
) -> list[list[object]]:
    summaries = []
    for group_index, group in enumerate(groups):
        group_results = [
            item for item in results
            if item.get("group_index") == group_index or item.get("group_name") == group["name"]
        ]
        buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for item in group_results:
            set_id = str(item.get("question_set_id") or "unclassified")
            set_name = str(item.get("question_set_name") or "未分類題目集")
            source_project = str(item.get("source_project_id") or "")
            buckets.setdefault((source_project, set_id, set_name), []).append(item)
        for (source_project, _set_id, set_name), selected in buckets.items():
            total = len(selected)
            correct = sum(_result_score(item) == 2 for item in selected)
            weighted_accuracy = sum(_result_score(item) for item in selected) / (2 * total) if total else 0
            source_project_name = next(
                (item.get("source_project_name", "") for item in selected), "",
            )
            summaries.append([
                group["name"], group["answer_model"], _group_reranker_mode(group),
                _group_expansion_mode(group), judge_model, total,
                f"{correct} / {total}", sum(_result_score(item) == 1 for item in selected),
                f"{weighted_accuracy:.1%}" if total else "—",
                f"{sum(bool(item.get('recall_at_5')) for item in selected) / total:.1%}" if total else "—",
                f"{sum(bool(item.get('recall_at_10')) for item in selected) / total:.1%}" if total else "—",
                f"{sum(float(item.get('reciprocal_rank', 0)) for item in selected) / total:.3f}" if total else "—",
                source_project_name, set_name,
            ])
    return summaries


def _single_experiment_detail_rows(results: list[dict[str, Any]]) -> list[list[object]]:
    return [[
        item["group_name"], item["number"], item.get("document", ""),
        item.get("question_set_name", ""), item["question"],
        item["expected_answer"], item.get("actual_answer", ""),
        _display_score(item), item.get("verification_changed"), item.get("reason", ""),
    ] for item in results]


def _experiment_project_detail_rows(results: list[dict[str, Any]]) -> list[list[object]]:
    return [[
        item["group_name"], item["source_project_name"], item["number"],
        item.get("document", ""), item.get("question_set_name", ""),
        item["question"], item["expected_answer"],
        item.get("actual_answer", ""), _display_score(item),
        item.get("verification_changed"), item.get("reason", ""),
    ] for item in results]


def generate_experiment_answers_for_ui(
    project_id: str, questions: list[dict[str, Any]], groups: list[dict[str, Any]],
    max_concurrent_requests: int | float, llm_state: dict[str, Any],
    embedding_api_base: str, embedding_api_key: str,
    neo4j_uri: str, neo4j_database: str, neo4j_username: str, neo4j_password: str,
    run_control: RunControl | None = None,
    progress=gr.Progress(),
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], list[list[Any]], list[list[Any]]]:
    """Generate answers for every group/question pair without judging or displaying them."""
    if not project_id:
        return "❌ 請先建立或載入專案。", [], [], [], []
    if not questions:
        return "❌ 請先在 0-3 為此專案綁定題目集，再回到 1-7 選擇。", [], [], [], []
    if not groups:
        return "❌ 請至少加入一個實驗組。", [], [], [], []
    try:
        concurrency = int(max_concurrent_requests)
        if concurrency < 1:
            raise ValueError("最大並行請求數必須大於 0")
    except (TypeError, ValueError, OverflowError) as exc:
        return f"❌ {exc}", [], [], [], []
    control = run_control or RunControl()
    control.reset()
    control.answer_generation_progress = None
    credentials = []
    for group in groups:
        endpoint, key = resolve_model_credentials_for_ui(llm_state, group["answer_model"])
        if not endpoint:
            return f"❌ 無法解析實驗組「{group['name']}」的回答模型服務。", [], [], [], []
        credentials.append((endpoint, key))
    document_ids_by_name = _project_document_ids(project_id)
    tasks = [
        (group_index, question_index)
        for group_index in range(len(groups))
        for question_index in range(len(questions))
    ]
    answers: list[dict[str, Any] | None] = [None] * len(tasks)
    control.answer_generation_progress = (0, len(tasks))

    def answer(task_index: int) -> dict[str, Any]:
        group_index, question_index = tasks[task_index]
        group, question = groups[group_index], questions[question_index]
        endpoint, key = credentials[group_index]
        strategy, top_k, reranker_mode, expansion_mode = _group_retrieval_values(group)
        status, actual, evidence_rows = answer_question_for_ui(
            endpoint, key, embedding_api_base, embedding_api_key,
            neo4j_uri, neo4j_database, neo4j_username, neo4j_password,
            group["answer_model"], question["question"], strategy,
            top_k, reranker_mode, expansion_mode,
            **_reasoning_effort_kwargs(group["answer_model"], group.get("answer_reasoning_effort")),
            strategy_params=_group_retrieval_config(group).params,
        )
        rank = _retrieval_rank(question, evidence_rows, document_ids_by_name)
        return {
            "group_index": group_index, "group_name": group["name"],
            "answer_model": group["answer_model"],
            "retrieval_config": _group_retrieval_config(group).to_dict(),
            **_reasoning_effort_record(
                group["answer_model"], group.get("answer_reasoning_effort"), "answer_reasoning_effort",
            ),
            "number": question.get("number", question_index + 1),
            "question_set_id": question.get("question_set_id", ""),
            "question_set_name": question.get("question_set_name", "未分類題目集"),
            "question": question["question"], "expected_answer": question["expected_answer"],
            "document": question.get("document", ""), "actual_answer": actual,
            "answer_status": status, "retrieval_rank": rank,
            "recall_at_5": rank is not None and rank <= 5,
            "recall_at_10": rank is not None and rank <= 10,
            "reciprocal_rank": 1 / rank if rank else 0.0,
        }

    try:
        with ThreadPoolExecutor(max_workers=min(concurrency, len(tasks))) as executor:
            futures = {executor.submit(answer, index): index for index in range(len(tasks))}
            completed = 0
            for future in as_completed(futures):
                if future.cancelled():
                    continue
                answers[futures[future]] = future.result()
                if answers[futures[future]] is not None:
                    completed += 1
                control.answer_generation_progress = (completed, len(tasks))
                progress(completed / len(tasks), desc=f"已生成 {completed} / {len(tasks)} 個實驗題次回答")
        generated = [item for item in answers if item is not None]
        project = load_project(project_id)
        previous = project.get("experiment") or {}
        saved_groups = [{key: value for key, value in group.items()
                         if key not in {"judge_model", "judge_reasoning_effort"}}
                        for group in groups]
        status = f"✅ 已生成 {len(generated)} 個實驗題次回答並填入逐題表格；尚未評測。"
        details = _single_experiment_detail_rows(generated)
        _save_experiment_data(project_id, {
            **previous, "questions": questions, "groups": saved_groups,
            "max_concurrent_requests": concurrency, "pending_answers": generated,
            "results": [], "summary_rows": [], "detail_rows": details, "status": status,
        })
    except (OSError, TypeError, ValueError) as exc:
        return f"❌ 回答生成或保存失敗：{exc}", [], [], [], []
    return status, generated, [], [], details


def evaluate_experiment_answers_for_ui(
    project_id: str, pending_answers: list[dict[str, Any]], groups: list[dict[str, Any]],
    judge_model: str, judge_reasoning_effort: str, max_concurrent_requests: int | float,
    judge_endpoint: str, judge_key: str,
    verification_enabled: bool = False,
    progress=gr.Progress(),
) -> tuple[str, list[dict[str, Any]], list[list[Any]], list[list[Any]]]:
    if not project_id:
        return "❌ 請先選擇專案。", [], [], []
    if not pending_answers:
        return "❌ 請先按「檢索並生成回答」完成回答生成。", [], [], []
    if not judge_model or not judge_endpoint:
        return "❌ 請選擇可用的評測模型。", [], [], []
    judge_reasoning_effort = judge_reasoning_effort or DEFAULT_JUDGE_REASONING_EFFORT
    try:
        concurrency = int(max_concurrent_requests)
        if concurrency < 1:
            raise ValueError("評測最大並行請求數必須大於 0")
    except (TypeError, ValueError, OverflowError) as exc:
        return f"❌ {exc}", [], [], []
    def judge(item: dict[str, Any]) -> dict[str, Any]:
        if str(item.get("answer_status", "")).startswith("✅"):
            try:
                verdict = judge_evaluation_answer(
                    judge_endpoint, judge_key, judge_model, item["question"],
                    item["expected_answer"], item.get("actual_answer", ""),
                    **_reasoning_effort_kwargs(judge_model, judge_reasoning_effort),
                )
                verdict = _verify_judgment_if_enabled(
                    verdict, verification_enabled, judge_endpoint, judge_key,
                    judge_model, item["question"], item["expected_answer"],
                    item.get("actual_answer", ""), judge_reasoning_effort,
                )
            except ValueError as exc:
                verdict = {"passed": False, "reason": f"評判失敗：{exc}"}
        else:
            verdict = {"passed": False, "reason": item.get("answer_status") or "回答生成失敗"}
        verdict.setdefault("verification_enabled", bool(verification_enabled))
        verdict.setdefault("verification_changed", None)
        return {
            **item, "judge_model": judge_model,
            **_reasoning_effort_record(judge_model, judge_reasoning_effort, "judge_reasoning_effort"),
            **_normalize_judgment(verdict),
        }

    evaluated: list[dict[str, Any] | None] = [None] * len(pending_answers)
    try:
        with ThreadPoolExecutor(max_workers=min(concurrency, len(pending_answers))) as executor:
            futures = {executor.submit(judge, item): index for index, item in enumerate(pending_answers)}
            completed = 0
            for future in as_completed(futures):
                if future.cancelled():
                    continue
                evaluated[futures[future]] = future.result()
                completed += 1
                progress(completed / len(pending_answers), desc=f"已評測 {completed} / {len(pending_answers)} 個實驗題次")
        results = [item for item in evaluated if item is not None]
        project = load_project(project_id)
        previous = project.get("experiment") or {}
        summary = _single_experiment_summary_rows(groups, results, judge_model)
        details = _single_experiment_detail_rows(results)
        full = sum(_result_score(item) == 2 for item in results)
        partial = sum(_result_score(item) == 1 for item in results)
        rate = sum(_result_score(item) for item in results) / (2 * len(results))
        status = f"✅ 評測完成｜完全正確 {full}、部分正確 {partial} / {len(results)} 個實驗題次；得分正確率 {rate:.1%}。"
        _save_experiment_data(project_id, {
            **previous, "pending_answers": pending_answers, "results": results,
            "summary_rows": summary, "detail_rows": details,
            "judge_model": judge_model, "judge_reasoning_effort": judge_reasoning_effort,
            "judge_max_concurrent_requests": concurrency, "status": status,
        })
    except (OSError, TypeError, ValueError) as exc:
        return f"❌ 評測或保存失敗：{exc}", [], [], []
    return status, results, summary, details


def evaluate_experiment_answers_with_services_for_ui(
    project_id: str, pending_answers: list[dict[str, Any]], groups: list[dict[str, Any]],
    judge_model: str, judge_reasoning_effort: str, max_concurrent_requests: int | float,
    llm_state: dict[str, Any],
    verification_enabled: bool = False,
    progress=gr.Progress(),
) -> tuple[str, list[dict[str, Any]], list[list[Any]], list[list[Any]]]:
    judge_endpoint, judge_key = resolve_model_credentials_for_ui(llm_state, judge_model)
    if not judge_endpoint:
        return "❌ 無法取得評測模型設定；請確認模型已列入 OpenAI 允許清單。", [], [], []
    return evaluate_experiment_answers_for_ui(
        project_id, pending_answers, groups, judge_model, judge_reasoning_effort,
        max_concurrent_requests, judge_endpoint, judge_key,
        verification_enabled=verification_enabled, progress=progress,
    )


def update_manual_experiment_result_for_ui(
    project_id: str, rows: Any, results: list[dict[str, Any]],
) -> tuple[str, list[list[Any]], list[list[Any]], list[dict[str, Any]]]:
    current = [dict(item) for item in results or []]
    if not project_id or not current:
        return "❌ 尚無可人工修改的實驗結果。", [], [], current
    project: dict[str, Any] = {}
    try:
        submitted = rows.tolist() if hasattr(rows, "tolist") else list(rows or [])
        if len(submitted) != len(current):
            raise ValueError("結果列數與實驗題次不符")
        scores = []
        for row in submitted:
            if len(row) < 8:
                raise ValueError("逐題結果欄位不完整")
            try:
                score = int(row[7])
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("答案判定只能選 0、1 或 2") from exc
            if score not in (0, 1, 2) or float(row[7]) != score:
                raise ValueError("答案判定只能選 0、1 或 2")
            scores.append(score)
        changed = 0
        for item, score in zip(current, scores):
            if score != _result_score(item):
                item["score"] = score
                item["passed"] = score == 2
                item["reason"] = "人工評判"
                item["manual_judgment"] = True
                changed += 1
        project = load_project(project_id)
        data = dict(project.get("experiment") or {})
        groups = data.get("groups") or []
        judge_model = data.get("judge_model") or (current[0].get("judge_model") if current else "")
        summary = _single_experiment_summary_rows(groups, current, judge_model)
        details = _single_experiment_detail_rows(current)
        _save_experiment_data(project_id, {
            **data, "results": current, "summary_rows": summary,
            "detail_rows": details,
        })
    except (OSError, TypeError, ValueError) as exc:
        return (
            f"❌ 人工評判保存失敗：{exc}",
            _single_experiment_summary_rows(
                (project.get("experiment") or {}).get("groups", []), current,
                (project.get("experiment") or {}).get("judge_model", ""),
            ) if project_id else [],
            _single_experiment_detail_rows(current), current,
        )
    correct = sum(_result_score(item) == 2 for item in current)
    partial = sum(_result_score(item) == 1 for item in current)
    rate = sum(_result_score(item) for item in current) / (2 * len(current))
    return (
        f"✅ 評測完成｜已保存人工評判變更 {changed} 筆；完全正確 {correct}、部分正確 {partial} / {len(current)} 題；得分正確率 {rate:.1%}。",
        summary, details, current,
    )


def export_experiment_results_for_ui(
    project_id: str, detail_rows: Any = None,
) -> tuple[str, str | None, str | None]:
    if not project_id:
        return "❌ 請先建立或載入專案。", None, None
    try:
        project = load_project(project_id)
        experiment = project.get("experiment") or {}
        saved_results = experiment.get("results") or []
        if detail_rows is not None and saved_results:
            sync_status, _, _, _ = update_manual_experiment_result_for_ui(
                project_id, detail_rows, saved_results,
            )
            if sync_status.startswith("❌"):
                return f"❌ 匯出前同步人工判定失敗：{sync_status.removeprefix('❌ ').strip()}", None, None
            project = load_project(project_id)
            experiment = project.get("experiment") or {}
        groups = experiment.get("groups") or []
        results = experiment.get("results") or []
        summary_rows = experiment.get("summary_rows") or []
        if not summary_rows and not results:
            return "❌ 尚無實驗結果可匯出。", None, None

        summaries = {}
        for row in summary_rows:
            if not row:
                continue
            if len(row) >= 12:
                summaries[str(row[0])] = {
                    "answer_model": row[1], "judge_model": row[4],
                    "question_count": row[5], "correct_total": row[6],
                    "partial_count": row[7], "accuracy": row[8],
                    "recall_at_5": row[9], "recall_at_10": row[10], "mrr": row[11],
                }
            elif len(row) >= 11:
                summaries[str(row[0])] = {
                    "answer_model": row[1], "judge_model": row[4],
                    "question_count": row[5], "correct_total": row[6],
                    "accuracy": row[7], "recall_at_5": row[8],
                    "recall_at_10": row[9], "mrr": row[10],
                }
            elif len(row) >= 9:
                summaries[str(row[0])] = {
                    "answer_model": row[1], "judge_model": row[2],
                    "question_count": row[3], "correct_total": row[4],
                    "accuracy": row[5], "recall_at_5": row[6],
                    "recall_at_10": row[7], "mrr": row[8],
                }
            else:
                summaries[str(row[0])] = {
                    "answer_model": row[1] if len(row) >= 8 else None,
                    "judge_model": row[2] if len(row) >= 8 else None,
                    "question_count": row[3] if len(row) >= 8 else row[1] if len(row) > 1 else 0,
                    "accuracy": row[4] if len(row) >= 8 else row[2] if len(row) > 2 else None,
                    "recall_at_5": row[5] if len(row) >= 8 else row[3] if len(row) > 3 else None,
                    "recall_at_10": row[6] if len(row) >= 8 else row[4] if len(row) > 4 else None,
                    "mrr": row[7] if len(row) >= 8 else row[5] if len(row) > 5 else None,
                }
        result_fields = (
            "number", "question", "document", "expected_answer", "actual_answer",
            "answer_model", "judge_model", "score", "passed", "reason", "retrieval_rank", "recall_at_5", "recall_at_10",
            "reciprocal_rank", "answer_reasoning_effort", "judge_reasoning_effort",
            "manual_judgment", "verification_enabled", "first_passed", "first_reason",
            "verification_passed", "verification_reason", "verification_changed",
            "verification_error",
        )
        exported_groups = []
        for group_index, group in enumerate(groups):
            name = str(group.get("name", ""))
            group_results = [
                {
                    **{key: result.get(key) for key in result_fields if key in result},
                    "score": _result_score(result),
                    "passed": _result_score(result) == 2,
                    "manual_judgment": bool(result.get("manual_judgment", False)),
                }
                for result in results
                if result.get("group_index") == group_index
                or ("group_index" not in result and result.get("group_name") == name)
            ]
            saved_summary = summaries.get(name, {})
            if group_results:
                question_count = len(group_results)
                correct_count = sum(_result_score(result) == 2 for result in group_results)
                partial_count = sum(_result_score(result) == 1 for result in group_results)
                accuracy = sum(_result_score(result) for result in group_results) / (2 * question_count)
                recall_at_5 = (
                    sum(bool(result.get("recall_at_5")) for result in group_results)
                    / question_count
                )
                recall_at_10 = (
                    sum(bool(result.get("recall_at_10")) for result in group_results)
                    / question_count
                )
                mrr = (
                    sum(float(result.get("reciprocal_rank", 0) or 0) for result in group_results)
                    / question_count
                )
            else:
                question_count = int(saved_summary.get("question_count", 0) or 0)
                raw_accuracy = saved_summary.get("accuracy", 0) or 0
                try:
                    accuracy = float(str(raw_accuracy).rstrip("%"))
                    if isinstance(raw_accuracy, str) and raw_accuracy.endswith("%"):
                        accuracy /= 100
                except (TypeError, ValueError):
                    accuracy = 0.0
                correct_count = int(
                    saved_summary.get("correct_count", round(accuracy * question_count)) or 0
                )
                partial_count = int(saved_summary.get("partial_count", 0) or 0)

                def rate(value: Any) -> float | None:
                    if value is None:
                        return None
                    try:
                        parsed = float(str(value).rstrip("%"))
                        return parsed / 100 if isinstance(value, str) and value.endswith("%") else parsed
                    except (TypeError, ValueError):
                        return None

                recall_at_5 = rate(saved_summary.get("recall_at_5"))
                recall_at_10 = rate(saved_summary.get("recall_at_10"))
                try:
                    mrr = float(saved_summary.get("mrr")) if saved_summary.get("mrr") is not None else None
                except (TypeError, ValueError):
                    mrr = None
            summary = {
                "answer_model": saved_summary.get("answer_model") or group.get("answer_model"),
                "judge_model": (
                    saved_summary.get("judge_model") or experiment.get("judge_model")
                    or group.get("judge_model", group.get("answer_model"))
                ),
                "question_count": question_count,
                "correct_count": correct_count,
                "partial_count": partial_count,
                "correct_total": f"{correct_count} / {question_count}",
                "accuracy": accuracy,
                "recall_at_5": recall_at_5,
                "recall_at_10": recall_at_10,
                "mrr": mrr,
            }
            parameters = {
                "answer_model": group.get("answer_model"),
                "retrieval_config": _group_retrieval_config(group).to_dict(),
            }
            if "answer_reasoning_effort" in group or str(group.get("answer_model") or "").casefold() == GPT_6_LUNA_MODEL:
                parameters["answer_reasoning_effort"] = group.get(
                    "answer_reasoning_effort", DEFAULT_REASONING_EFFORT,
                )
            exported_groups.append({
                "name": name,
                "parameters": parameters,
                "summary": summary,
                "results": group_results,
            })

        total_questions = sum(item["summary"]["question_count"] for item in exported_groups)
        total_correct = sum(item["summary"]["correct_count"] for item in exported_groups)

        def weighted_metric(key: str) -> float | None:
            values = [
                (item["summary"].get(key), item["summary"]["question_count"])
                for item in exported_groups
                if item["summary"].get(key) is not None and item["summary"]["question_count"]
            ]
            denominator = sum(count for _, count in values)
            return sum(float(value) * count for value, count in values) / denominator if denominator else None

        overall_summary = {
            "question_count": total_questions,
            "correct_count": total_correct,
            "partial_count": sum(item["summary"]["partial_count"] for item in exported_groups),
            "correct_total": f"{total_correct} / {total_questions}",
            "accuracy": (weighted_metric("accuracy") or 0.0) if total_questions else 0.0,
            "recall_at_5": weighted_metric("recall_at_5"),
            "recall_at_10": weighted_metric("recall_at_10"),
            "mrr": weighted_metric("mrr"),
        }
        payload = {
            "schema_version": 4,
            "project": {"project_id": project_id, "name": project.get("name", "")},
            "max_concurrent_requests": experiment.get("max_concurrent_requests", DEFAULT_MAX_CONCURRENT_REQUESTS),
            "evaluation": {
                "judge_model": experiment.get("judge_model") or next(
                    (group.get("judge_model") for group in groups if group.get("judge_model")),
                    groups[0].get("answer_model") if groups else None,
                ),
                "judge_reasoning_effort": experiment.get(
                    "judge_reasoning_effort", DEFAULT_JUDGE_REASONING_EFFORT,
                ),
            },
            "question_count": len(experiment.get("questions") or []),
            "summary": overall_summary,
            "groups": exported_groups,
        }
        output_id = uuid4().hex
        output = write_json(
            Path("data/projects") / project_id / "exports" / f"experiment-results-{output_id}.json",
            payload,
        )
        compact_payload = {
            "schema_version": 2,
            "format": "manual-graphrag-experiment-summary",
            "project": {"project_id": project_id, "name": project.get("name", "")},
            "max_concurrent_requests": experiment.get("max_concurrent_requests", DEFAULT_MAX_CONCURRENT_REQUESTS),
            "evaluation": payload["evaluation"],
            "summary": overall_summary,
            "groups": [
                {"name": item["name"], "parameters": item["parameters"], "summary": item["summary"]}
                for item in exported_groups
            ],
        }
        compact_output = write_json(
            Path("data/projects") / project_id / "exports" / f"experiment-summary-{output_id}.json",
            compact_payload,
        )
    except (OSError, TypeError, ValueError) as exc:
        return f"❌ 匯出失敗：{exc}", None, None
    return (
        f"✅ 已匯出 {len(exported_groups)} 個實驗組；總答對 {total_correct} / {total_questions} 個實驗題次。",
        str(output), str(compact_output),
    )


def add_experiment_project_group_for_ui(
    project: dict[str, Any] | None, name: str | None, answer_model: str | None,
    answer_effort: str | None, retrieval_mode: str, top_k: int | float,
    use_reranker: str, expand_evidence: str,
) -> tuple[dict[str, Any], list[list[Any]], Any, str, Any]:
    def response(state: dict[str, Any], status: str):
        groups = state.get("groups", [])
        rows = [[
            group.get("name"), group.get("answer_model"), *_group_retrieval_values(group),
        ] for group in state.get("groups", [])]
        choices = [(group.get("name", ""), group.get("name", "")) for group in groups]
        return (state, rows, gr.update(choices=choices, value=choices[0][1] if choices else None),
                status, gr.update(value=_next_experiment_project_group_name(groups)))

    if not project:
        return {}, [], gr.update(choices=[], value=None), "❌ 請先載入實驗專案。", gr.update(value="實驗組 1")
    groups = list(project.get("groups") or [])
    clean_name = str(name or "").strip() or _next_experiment_project_group_name(groups)
    if any(group.get("name") == clean_name for group in groups):
        return response(project, f"❌ 實驗組名稱「{clean_name}」已存在。")
    if not answer_model:
        return response(project, "❌ 請選擇回答模型。")
    try:
        selected_top_k = int(top_k)
        retrieval_config = _retrieval_config_from_controls(
            retrieval_mode, selected_top_k, use_reranker, expand_evidence,
        ).to_dict()
    except (TypeError, ValueError, OverflowError) as exc:
        return response(project, f"❌ {exc}")
    group = {
        "name": clean_name, "answer_model": answer_model,
        "retrieval_config": retrieval_config,
    }
    group.update(_reasoning_effort_record(answer_model, answer_effort, "answer_reasoning_effort"))
    groups.append(group)
    try:
        updated = save_experiment_project(project["experiment_project_id"], {
            "groups": groups, "results": [], "summary_rows": [], "detail_rows": [],
        })
    except (OSError, ValueError) as exc:
        return response(project, f"❌ 實驗組保存失敗：{exc}")
    return response(updated, f"✅ 已加入實驗組「{clean_name}」。")


def _next_experiment_project_group_name(groups: list[dict[str, Any]]) -> str:
    names = {str(group.get("name", "")) for group in groups}
    index = 1
    while f"實驗組 {index}" in names:
        index += 1
    return f"實驗組 {index}"


def remove_experiment_project_group_for_ui(
    project: dict[str, Any] | None, name: str | None,
) -> tuple[dict[str, Any], list[list[Any]], Any, str]:
    def response(state: dict[str, Any], status: str):
        groups = state.get("groups", [])
        rows = [[
            group.get("name"), group.get("answer_model"), *_group_retrieval_values(group),
        ] for group in groups]
        choices = [(group.get("name", ""), group.get("name", "")) for group in groups]
        return state, rows, gr.update(choices=choices, value=choices[0][1] if choices else None), status

    if not project or not name:
        return response(project or {}, "❌ 請選擇要移除的實驗組。")
    groups = [group for group in project.get("groups", []) if group.get("name") != name]
    try:
        updated = save_experiment_project(project["experiment_project_id"], {
            "groups": groups, "results": [], "summary_rows": [], "detail_rows": [],
        })
    except (OSError, ValueError) as exc:
        return response(project, f"❌ 實驗組更新失敗：{exc}")
    return response(updated, f"✅ 已移除實驗組「{name}」。")


def generate_experiment_project_answers_for_ui(
    project: dict[str, Any] | None, max_concurrent_requests: int | float,
    llm_state: dict[str, Any], embedding_api_base: str, embedding_api_key: str,
    neo4j_uri: str, neo4j_username: str, neo4j_password: str,
    progress=gr.Progress(),
) -> tuple[str, list[dict[str, Any]], list[list[Any]], list[list[Any]], dict[str, Any]]:
    """Generate all member-project answers, but defer judging and showing rows."""
    if not project:
        return "❌ 請先載入實驗專案。", [], [], [], {}
    try:
        current = load_experiment_project(project["experiment_project_id"])
        concurrency = int(max_concurrent_requests)
        if concurrency < 1:
            raise ValueError("最大並行請求數必須大於 0")
    except (OSError, TypeError, ValueError, OverflowError) as exc:
        return f"❌ {exc}", [], [], [], project
    members = current.get("members") or []
    groups = current.get("groups") or []
    if not members or not groups:
        return "❌ 請先加入成員專案並設定實驗組。", [], [], [], current
    try:
        members_data, question_map = _load_experiment_member_question_sets(current)
    except (OSError, ValueError) as exc:
        return f"❌ 成員專案載入失敗：{exc}", [], [], [], current
    missing = [member for member in members if not question_map.get(member)]
    if missing:
        return f"❌ 尚有 {len(missing)} 個成員專案沒有題目集，請至 0-3 綁定題目集。", [], [], [], current
    for member_id in members:
        member = members_data[member_id]
        if not (member.get("graph_state") or {}).get("neo4j_imported"):
            return f"❌ 專案「{member.get('name', member_id)}」尚未完成建圖匯入。", [], [], [], current
    credentials = {}
    for group in groups:
        endpoint, key = resolve_model_credentials_for_ui(llm_state, group.get("answer_model", ""))
        if not endpoint:
            return f"❌ 無法解析實驗組「{group.get('name', '')}」的回答模型服務。", [], [], [], current
        credentials[group["name"]] = (endpoint, key)
    tasks = [(gi, mid, qi) for gi in range(len(groups)) for mid in members for qi in range(len(question_map[mid]))]
    answers: list[dict[str, Any] | None] = [None] * len(tasks)

    def answer(task_index: int) -> dict[str, Any]:
        gi, member_id, qi = tasks[task_index]
        group, question = groups[gi], question_map[member_id][qi]
        member = members_data[member_id]
        endpoint, key = credentials[group["name"]]
        strategy, top_k, reranker_mode, expansion_mode = _group_retrieval_values(group)
        status, actual, evidence_rows = answer_question_for_ui(
            endpoint, key, embedding_api_base, embedding_api_key, neo4j_uri,
            member.get("neo4j_database") or project_database_name(member_id),
            neo4j_username, neo4j_password, group["answer_model"], question["question"],
            strategy, top_k, reranker_mode, expansion_mode,
            **_reasoning_effort_kwargs(group["answer_model"], group.get("answer_reasoning_effort")),
            strategy_params=_group_retrieval_config(group).params,
        )
        rank = _retrieval_rank(question, evidence_rows, _project_document_ids(member_id))
        return {
            "group_index": gi, "group_name": group["name"],
            "retrieval_config": _group_retrieval_config(group).to_dict(),
            "source_project_id": member_id, "source_project_name": member.get("name", member_id),
            "answer_model": group["answer_model"], "number": question.get("number", qi + 1),
            "question_set_id": question.get("question_set_id", ""),
            "question_set_name": question.get("question_set_name", "未分類題目集"),
            "question": question["question"], "expected_answer": question["expected_answer"],
            "document": question.get("document", ""), "actual_answer": actual,
            "answer_status": status, "retrieval_rank": rank,
            "recall_at_5": rank is not None and rank <= 5,
            "recall_at_10": rank is not None and rank <= 10,
            "reciprocal_rank": 1 / rank if rank else 0.0,
        }

    try:
        with ThreadPoolExecutor(max_workers=min(concurrency, len(tasks))) as executor:
            futures = {executor.submit(answer, index): index for index in range(len(tasks))}
            completed = 0
            for future in as_completed(futures):
                if future.cancelled():
                    continue
                answers[futures[future]] = future.result()
                completed += 1
                progress(completed / len(tasks), desc=f"已生成 {completed} / {len(tasks)} 個跨專案實驗題次回答")
        generated = [item for item in answers if item is not None]
        details = _experiment_project_detail_rows(generated)
        status = f"✅ 已生成 {len(generated)} 個實驗題次回答並填入逐題表格；尚未評測。"
        updated = save_experiment_project(current["experiment_project_id"], {
            "pending_answers": generated, "results": [], "summary_rows": [], "detail_rows": details,
            "max_concurrent_requests": concurrency, "status": status,
        })
    except (OSError, TypeError, ValueError) as exc:
        return f"❌ 回答生成或保存失敗：{exc}", [], [], [], current
    return status, generated, [], details, updated


def evaluate_experiment_project_answers_for_ui(
    project: dict[str, Any] | None, pending_answers: list[dict[str, Any]] | None,
    judge_model: str | None, judge_reasoning_effort: str | None,
    max_concurrent_requests: int | float, llm_state: dict[str, Any],
    verification_enabled: bool = False,
    progress=gr.Progress(),
) -> tuple[str, list[dict[str, Any]], list[list[Any]], list[list[Any]], dict[str, Any]]:
    if not project or not pending_answers:
        return "❌ 請先完成「檢索並生成回答」。", [], [], [], project or {}
    endpoint, key = resolve_model_credentials_for_ui(llm_state, judge_model or "")
    if not endpoint:
        return "❌ 請選擇可用的評測模型。", [], [], [], project
    judge_reasoning_effort = judge_reasoning_effort or DEFAULT_JUDGE_REASONING_EFFORT
    try:
        concurrency = int(max_concurrent_requests)
        if concurrency < 1:
            raise ValueError("評測最大並行請求數必須大於 0")
    except (TypeError, ValueError, OverflowError) as exc:
        return f"❌ {exc}", [], [], [], project
    evaluated: list[dict[str, Any] | None] = [None] * len(pending_answers)

    def judge(index: int) -> dict[str, Any]:
        item = pending_answers[index]
        if str(item.get("answer_status", "")).startswith("✅"):
            try:
                verdict = judge_evaluation_answer(
                    endpoint, key, judge_model, item["question"], item["expected_answer"],
                    item.get("actual_answer", ""),
                    **_reasoning_effort_kwargs(judge_model, judge_reasoning_effort),
                )
                verdict = _verify_judgment_if_enabled(
                    verdict, verification_enabled, endpoint, key, judge_model,
                    item["question"], item["expected_answer"],
                    item.get("actual_answer", ""), judge_reasoning_effort,
                )
            except ValueError as exc:
                verdict = {"passed": False, "reason": f"評判失敗：{exc}"}
        else:
            verdict = {"passed": False, "reason": item.get("answer_status") or "回答生成失敗"}
        verdict.setdefault("verification_enabled", bool(verification_enabled))
        verdict.setdefault("verification_changed", None)
        return {**item, "judge_model": judge_model,
            **_reasoning_effort_record(judge_model, judge_reasoning_effort, "judge_reasoning_effort"), **_normalize_judgment(verdict)}

    try:
        with ThreadPoolExecutor(max_workers=min(concurrency, len(pending_answers))) as executor:
            futures = {executor.submit(judge, index): index for index in range(len(pending_answers))}
            completed = 0
            for future in as_completed(futures):
                if future.cancelled():
                    continue
                evaluated[futures[future]] = future.result()
                completed += 1
                progress(completed / len(pending_answers), desc=f"已評測 {completed} / {len(pending_answers)} 個跨專案實驗題次")
        results = [item for item in evaluated if item is not None]
        groups = (project or {}).get("groups") or []
        summary = _single_experiment_summary_rows(groups, results, judge_model)
        details = _experiment_project_detail_rows(results)
        correct = sum(_result_score(item) == 2 for item in results)
        partial = sum(_result_score(item) == 1 for item in results)
        rate = sum(_result_score(item) for item in results) / (2 * len(results))
        status = f"✅ 評測完成｜完全正確 {correct}、部分正確 {partial} / {len(results)} 個跨專案實驗題次；得分正確率 {rate:.1%}。"
        updated = save_experiment_project(project["experiment_project_id"], {
            "pending_answers": pending_answers, "results": results, "summary_rows": summary,
            "detail_rows": details, "judge_model": judge_model,
            "verification_enabled": bool(verification_enabled),
            "judge_reasoning_effort": judge_reasoning_effort or DEFAULT_JUDGE_REASONING_EFFORT,
            "judge_max_concurrent_requests": concurrency, "status": status,
        })
    except (OSError, TypeError, ValueError) as exc:
        return f"❌ 評測或保存失敗：{exc}", [], [], [], project
    return status, results, summary, details, updated


def update_manual_experiment_project_result_for_ui(
    project: dict[str, Any] | None, rows: Any, results: list[dict[str, Any]] | None,
) -> tuple[str, list[list[Any]], list[list[Any]], dict[str, Any]]:
    current = [dict(item) for item in results or []]
    if not project or not current:
        return "❌ 尚無可人工修改的實驗結果。", [], [], project or {}
    try:
        submitted = rows.tolist() if hasattr(rows, "tolist") else list(rows or [])
        if len(submitted) != len(current):
            raise ValueError("結果列數與實驗題次不符")
        scores = []
        for row in submitted:
            if len(row) < 9:
                raise ValueError("逐題結果欄位不完整")
            try:
                score = int(row[8])
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("答案判定只能選 0、1 或 2") from exc
            if score not in (0, 1, 2) or float(row[8]) != score:
                raise ValueError("答案判定只能選 0、1 或 2")
            scores.append(score)
        changed = 0
        for item, score in zip(current, scores):
            if score != _result_score(item):
                item["score"] = score
                item["passed"] = score == 2
                item["reason"] = "人工評判"
                item["manual_judgment"] = True
                changed += 1
        groups = project.get("groups") or []
        judge_model = project.get("judge_model") or str(current[0].get("judge_model", ""))
        summary = _single_experiment_summary_rows(groups, current, judge_model)
        details = _experiment_project_detail_rows(current)
        updated = save_experiment_project(project["experiment_project_id"], {
            "results": current, "summary_rows": summary, "detail_rows": details,
            "status": f"✅ 評測完成｜完全正確 {sum(_result_score(item) == 2 for item in current)}、部分正確 {sum(_result_score(item) == 1 for item in current)} / {len(current)} 個跨專案實驗題次；得分正確率 {sum(_result_score(item) for item in current) / (2 * len(current)):.1%}。",
        })
    except (OSError, TypeError, ValueError) as exc:
        groups = (project or {}).get("groups") or []
        judge_model = (project or {}).get("judge_model") or str(current[0].get("judge_model", ""))
        return (
            f"❌ 人工評判保存失敗：{exc}",
            _single_experiment_summary_rows(groups, current, judge_model),
            _experiment_project_detail_rows(current), project,
        )
    return f"✅ 已保存人工評判變更 {changed} 筆；完全正確 {sum(_result_score(item) == 2 for item in current)}、部分正確 {sum(_result_score(item) == 1 for item in current)} / {len(current)} 個跨專案實驗題次；得分正確率 {sum(_result_score(item) for item in current) / (2 * len(current)):.1%}。", summary, details, updated


def run_experiment_project_for_ui(
    project: dict[str, Any] | None, max_concurrent_requests: int | float,
    llm_state: dict[str, Any], embedding_api_base: str, embedding_api_key: str,
    neo4j_uri: str, neo4j_username: str, neo4j_password: str,
    judge_model: str | None, judge_reasoning_effort: str | None,
    progress=gr.Progress(),
) -> tuple[str, list[list[Any]], list[list[Any]], dict[str, Any]]:
    if not project:
        return "❌ 請先載入實驗專案。", [], [], {}
    project_id = project.get("experiment_project_id")
    try:
        current = load_experiment_project(project_id)
    except (OSError, ValueError) as exc:
        return f"❌ {exc}", [], [], project
    members = current.get("members") or []
    groups = current.get("groups") or []
    if not members:
        return "❌ 請先在 2-0 加入已建圖專案。", [], [], current
    if not groups:
        return "❌ 請至少新增一個實驗組。", [], [], current
    try:
        members_data, question_map = _load_experiment_member_question_sets(current)
    except (OSError, ValueError) as exc:
        return f"❌ 成員專案載入失敗：{exc}", [], [], current
    missing = [member_id for member_id in members if not question_map.get(member_id)]
    if missing:
        return f"❌ 尚有 {len(missing)} 個成員專案沒有題目集，請至 0-3 綁定題目集。", [], [], current
    if not judge_model:
        return "❌ 請選擇全域評測模型。", [], [], current
    all_results: list[dict[str, Any]] = []
    member_names: dict[str, str] = {}
    for member_id in members:
        member = members_data[member_id]
        if not (member.get("graph_state") or {}).get("neo4j_imported"):
            return f"❌ 專案「{member['name']}」尚未完成建圖匯入。", [], [], current
        member_names[member_id] = member.get("name", member_id)
        status, _summary, _details, results = run_experiment_groups_for_ui(
            member_id, question_map[member_id], groups, max_concurrent_requests,
            llm_state, embedding_api_base, embedding_api_key,
            neo4j_uri, member.get("neo4j_database") or project_database_name(member_id),
            neo4j_username, neo4j_password, judge_model,
            judge_reasoning_effort, False, progress,
        )
        if status.startswith("❌"):
            return status, [], [], current
        for result in results:
            all_results.append({
                **result, "source_project_id": member_id,
                "source_project_name": member_names[member_id],
            })

    summary_rows = _single_experiment_summary_rows(groups, all_results, judge_model)
    detail_rows = [[
        item["group_name"], item["source_project_name"], item["answer_model"],
        item["judge_model"], item["number"], item["question"], item.get("document", ""),
        item["expected_answer"], item["actual_answer"],
        _result_score(item), item.get("reason", ""),
        item.get("retrieval_rank"), item.get("question_set_name", ""),
    ] for item in all_results]
    status = f"✅ 已完成 {len(groups)} 個實驗組，涵蓋 {len(members)} 個專案、{len(all_results)} 個題次。"
    try:
        updated = save_experiment_project(project_id, {
            "results": all_results, "summary_rows": summary_rows,
            "detail_rows": detail_rows, "judge_model": judge_model,
            "judge_reasoning_effort": judge_reasoning_effort or DEFAULT_JUDGE_REASONING_EFFORT,
            "max_concurrent_requests": int(max_concurrent_requests), "status": status,
        })
    except (OSError, TypeError, ValueError) as exc:
        return f"⚠️ 實驗已完成，但結果保存失敗：{exc}", summary_rows, detail_rows, current
    return status, summary_rows, detail_rows, updated


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
        document_id = hashlib.sha256(Path(file_path).read_bytes()).hexdigest()
        pages, empty_pages = extract_pdf(file_path)
        next_number = max((chunk.number for chunk in chunks), default=0) + 1
        new_chunks = chunk_pages(
            pages, parsed_chunk_size, parsed_chunk_overlap,
            document=file_name, document_id=document_id, start_number=next_number,
        )
        if not new_chunks:
            raise ValueError("PDF 沒有可解析文字；掃描文件需在後續版本加入 OCR。")
    except (ValueError, TypeError) as exc:
        return f"❌「{file_name}」：{exc}", None, []
    parsed_start, parsed_end = pages[0].page, pages[-1].page
    doc_state = {
        "file_path": file_path,
        "file_name": file_name,
        "document_id": document_id,
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


def schema_documents_for_ui(
    documents: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    names = [
        str(document.get("file_name") or "")
        for document in (documents or [])
        if document.get("file_name")
    ]
    return gr.update(value=[[True, name] for name in names])


def _selected_schema_documents(rows: Any) -> list[str]:
    if isinstance(rows, dict) and "data" in rows:
        rows = rows["data"]
    elif hasattr(rows, "values") and hasattr(rows.values, "tolist"):
        rows = rows.values.tolist()
    if not isinstance(rows, list):
        return []
    return list(dict.fromkeys(
        str(row[1])
        for row in rows
        if isinstance(row, (list, tuple))
        and len(row) >= 2
        and row[0] is True
        and row[1]
    ))


def _select_schema_planning_chunks(
    chunks: list[TextChunk], document_rows: Any
) -> tuple[list[TextChunk], list[str], int]:
    if not chunks:
        raise ValueError("請先在 PDF 頁面解析並產生 chunks")
    selected = _selected_schema_documents(document_rows)
    if not selected:
        raise ValueError("請至少勾選一份用於規劃 Schema 的 PDF")
    available_documents = {chunk.document for chunk in chunks}
    unknown = [document for document in selected if document not in available_documents]
    if unknown:
        unknown_text = "、".join(unknown)
        raise ValueError(f"找不到選取的 PDF：{unknown_text}")
    selected_set = set(selected)
    selected_chunks = [
        chunk for chunk in chunks if chunk.document in selected_set
    ]
    page_count = len({
        (chunk.document, page)
        for chunk in selected_chunks
        for page in chunk.pages
    })
    return selected_chunks, selected, page_count


def plan_schema_for_ui(
    model_endpoint: str,
    api_key: str,
    llm_model: str,
    temperature: float,
    schema_granularity: str,
    max_concurrent_requests: int,
    document_rows: Any,
    chunks: list[TextChunk],
    run_control: RunControl,
    reasoning_effort: str | None = None,
    progress=gr.Progress(),
) -> tuple[str, str]:
    if not llm_model:
        return "❌ 請先勾選並選擇 Schema 規劃 LLM。", ""
    run_control.reset()
    try:
        planning_chunks, planned_documents, page_count = _select_schema_planning_chunks(
            chunks, document_rows
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
            **_reasoning_effort_kwargs(llm_model, reasoning_effort),
        )
    except RunCancelled:
        return "⏹ 已停止（使用者中止 Schema 規劃）。", ""
    except ValueError as exc:
        return f"❌ {exc}", ""
    document_text = "、".join(planned_documents)
    scope_note = (
        f"{len(planned_documents)} 份 PDF（{document_text}）的全部 {page_count} 頁"
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
    chunk_size: int = 0,
    chunk_overlap: int = 0,
    schema_granularity: str = "平衡",
    reasoning_effort: str | None = None,
    progress=gr.Progress(),
) -> tuple[str, list[list[object]], list[list[object]], dict[str, Any]]:
    if not llm_model:
        return "❌ 請先勾選並選擇知識圖譜抽取 LLM。", [], [], {}
    run_control.reset()
    try:
        schema = None
        if str(schema_text or "").strip():
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
            **_reasoning_effort_kwargs(llm_model, reasoning_effort),
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
        "build_config": {
            "chunk_size": int(chunk_size),
            "chunk_overlap": int(chunk_overlap),
            "schema_granularity": schema_granularity if schema is not None else None,
        },
        "entities": extraction.entities,
        "relationships": extraction.relationships,
        "chunks": [
            {
                "number": chunk.number, "text": chunk.text,
                "pages": list(chunk.pages), "document": chunk.document,
                "document_id": chunk.document_id,
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
                "document_id": str(item.get("document_id", "")),
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
        # Provision the project database as part of the real import operation;
        # the connection-test page is diagnostic and must not be a prerequisite.
        ensure_project_database(
            neo4j_uri, neo4j_database, neo4j_username, neo4j_password,
        )
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
    use_reranker: str = "停用",
    expand_evidence: str = "停用",
    reasoning_effort: str | None = None,
    reranker_mode: str | None = None,
    evidence_expansion_mode: str | None = None,
    strategy_params: Any = None,
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
        effective_reranker_mode = reranker_mode or _selected_retrieval_mode(use_reranker)
        effective_expansion_mode = evidence_expansion_mode or _selected_retrieval_mode(expand_evidence)
        retrieval_config = _retrieval_config_from_controls(
            retrieval_mode, int(top_k), effective_reranker_mode,
            effective_expansion_mode, strategy_params,
        )
        should_expand = retrieval_config.expansion["id"] != "disabled"
        should_rerank = retrieval_config.reranker["id"] != "disabled"
        evidence = search_graph_evidence(
            neo4j_uri, neo4j_database, neo4j_username, neo4j_password,
            graph_state["run_id"], question.strip(), question_vector,
            retrieval_config,
        )
        if retrieval_config.reranker["id"] == "llm":
            evidence = rerank_evidence(
                model_endpoint, api_key, answer_model, question, evidence,
                int(top_k), reasoning_effort,
            )
        elif retrieval_config.reranker["id"] == "lexical":
            evidence = legacy_rerank_evidence(question, evidence, int(top_k))
        elif should_expand:
            evidence = interleave_expanded_evidence(evidence, int(top_k))
        else:
            evidence = evidence[:int(top_k)]
        result = answer_graph_question(
            model_endpoint, api_key, answer_model, question,
            retrieval_config.strategy_id, evidence,
            **_reasoning_effort_kwargs(answer_model, reasoning_effort),
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
    status = (
        f"✅ {strategy_label(retrieval_config.strategy_id)} 已使用 {len(rows)} 筆證據完成回答；"
        f"文件：{graph_state.get('document', '未知')}。"
    )
    return status, result["answer"], rows


def _current_project_banner(project: dict[str, Any]) -> str:
    name = (project or {}).get("name")
    return f"### 📁 目前專案：{name}" if name else "### 📁 目前專案：尚未選擇"


def build_app() -> gr.Blocks:
    env = load_env()
    llm_settings = load_service_settings("llm", env)
    experiment_llm_settings = load_service_settings("experiment_llm", env)
    embedding_settings = load_service_settings("embedding", env)
    llm_choices = _model_choices_with_fallback(
        service_choice_items(llm_settings), DEFAULT_LLM_MODEL,
    )
    evaluation_judge_choices = _model_choices_with_fallback(
        llm_choices, DEFAULT_EVALUATION_MODEL,
    )
    embedding_choices = service_choice_items(embedding_settings)
    embedding_allowed = service_choices(embedding_settings)
    llm_profile = llm_settings["profiles"][llm_settings["active"]]
    embedding_profile = embedding_settings["profiles"][embedding_settings["active"]]
    preferred_llm = DEFAULT_LLM_MODEL
    experiment_llm_choices = _model_choices_with_fallback(
        _configured_service_choice_items(experiment_llm_settings), DEFAULT_LLM_MODEL,
    )
    experiment_default_llm = DEFAULT_LLM_MODEL
    project_choices = _project_choices()
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
        gr.HTML("""<style>
        .project-workspace-action {align-self:flex-end !important;}
        .project-workspace-action button {height:38px !important;min-height:38px !important;}
        </style>""")
        current_project_banner = gr.Markdown(_current_project_banner({}))
        current_user_banner = gr.Markdown(current_user_banner_for_ui(None))
        llm_service_state = gr.State(llm_settings)
        experiment_llm_service_state = gr.State(experiment_llm_settings)
        embedding_service_state = gr.State(embedding_settings)
        project_state = gr.State({})
        documents_state = gr.State([])
        active_preview_state = gr.State({})
        active_chunks_state = gr.State([])
        chunk_state = gr.State([])
        graph_state = gr.State({})
        evaluation_state = gr.State({})
        run_control_state = gr.State(RunControl())
        neo4j_connected_state = gr.State(False)
        workspace_mode_state = gr.State("")
        single_workspace_mode = gr.State("single")
        experiment_workspace_mode = gr.State("experiment")
        experiment_project_state = gr.State({})
        experiment_project_connection_state = gr.State({})
        experiment_project_pending_answers_state = gr.State([])
        experiment_project_results_state = gr.State([])
        # Internal service states are shared by workflows; credentials are editable only on 0-4.
        llm_provider = gr.State("OpenAI")
        model_endpoint = gr.State(llm_profile["base_url"])
        api_key = gr.State(llm_profile["api_key"])
        model_test_button = gr.State(False)
        model_list_button = gr.State(False)
        llm_models_table = gr.State([])
        model_connection_status = gr.State(llm_profile["status"])
        embedding_provider = gr.State("OpenAI")
        embedding_api_base = gr.State(embedding_profile["base_url"])
        embedding_api_key = gr.State(embedding_profile["api_key"])
        embedding_test_button = gr.State(False)
        embedding_list_button = gr.State(False)
        embedding_models_table = gr.State([])
        embedding_connection_status = gr.State(embedding_profile["status"])

        with gr.Tab("0-0 使用者"):
            gr.Markdown("請先選擇目前操作使用者，選擇後才能開啟專案與其他設定頁面。")
            current_user_dropdown = gr.Dropdown(
                choices=[("請選擇使用者", ""), *((name, name) for name in KNOWN_USERS)],
                value="", label="目前使用者",
                interactive=True,
            )

        with gr.Tab("0-2 管理題目集", interactive=False) as question_set_manager_tab:
            gr.Markdown("此頁只負責將題目集匯入中央題庫；專案使用哪些題目集，請至 0-3 設定。")
            with gr.Row():
                central_question_file = gr.File(
                    label="題目集（JSON／CSV）", file_types=[".json", ".csv"], type="filepath",
                )
                central_question_name = gr.Textbox(label="題目集名稱（可留空使用檔名）")
                import_central_question_button = gr.Button("匯入中央題庫", variant="primary")
            central_question_status = gr.Markdown("尚未載入中央題庫。")
            central_question_selector = gr.Dropdown(choices=[], value=None, label="已匯入中央題目集")
            central_question_table = gr.Dataframe(
                headers=["題號", "題目", "正確答案", "題目來源（文件與頁碼）", "答案來源（文件與頁碼）"],
                datatype=["number", "str", "str", "str", "str"], interactive=False, wrap=True,
            )

        with gr.Tab("0-3 綁定題目集", interactive=False) as question_set_binding_tab:
            gr.Markdown("集中設定各專案可使用的中央題目集。1-7 與跨專案測試都會使用這裡保存的綁定。")
            binding_project_selector = gr.Dropdown(
                choices=project_choices, value=project_choices[0][1] if project_choices else None,
                label="專案",
            )
            binding_question_set_selector = gr.CheckboxGroup(choices=[], value=[], label="已綁定題目集")
            binding_question_set_status = gr.Markdown("請選擇專案。")
            save_question_set_bindings_button = gr.Button("保存綁定", variant="primary")

        with gr.Tab("0-4 API Key 設定", interactive=False) as api_settings_tab:
            gr.Markdown("所有模型與 Embedding 呼叫共用同一組 OpenAI API 設定，並寫入本機 `.env`。")
            with gr.Row():
                global_api_endpoint = gr.Textbox(label="OpenAI API Base URL", value=llm_profile["base_url"])
                global_api_key = gr.Textbox(label="OpenAI API Key", value=llm_profile["api_key"], type="password")
            with gr.Row():
                global_api_test_button = gr.Button("測試 OpenAI 連線", variant="primary")
                reload_button = gr.Button("重新讀取 .env")
            global_api_status = gr.Markdown("API 設定修改後會自動儲存。連線測試僅供診斷，不是後續操作的前置條件。")

        with gr.Tab("1-0 專案設定", interactive=False) as project_tab:
            gr.Markdown("### 專案工作區\n建立或載入專案後，可保存本頁面所有連線、模型、參數、Chunk、文件、建圖狀態與問答紀錄。")
            with gr.Row():
                project_selector = gr.Dropdown(
                    choices=project_choices,
                    value=project_choices[0][1] if project_choices else None,
                    label="現有專案", interactive=True,
                )
                load_project_button = gr.Button(
                    "載入專案", variant="primary", elem_classes="project-workspace-action",
                )
                delete_project_button = gr.Button(
                    "刪除專案", variant="stop", elem_classes="project-workspace-action",
                )
            with gr.Row():
                project_delete_confirmation_status = gr.Markdown(visible=False)
                confirm_delete_project_button = gr.Button(
                    "確認刪除", variant="stop", visible=False,
                )
                cancel_delete_project_button = gr.Button("取消", visible=False)
            with gr.Row():
                new_project_name = gr.Textbox(label="新專案名稱", placeholder="例如：ALCX17 使用手冊")
                create_project_button = gr.Button(
                    "建立新專案", variant="primary", elem_classes="project-workspace-action",
                )
                delete_project_completed = gr.State(False)
            with gr.Row():
                export_project_button = gr.Button("匯出專案封裝")
                export_project_file = gr.File(label="下載專案封裝", interactive=False)
                import_project_file = gr.File(label="選擇專案封裝 ZIP", type="filepath", file_types=[".zip"])
                import_project_button = gr.Button("匯入專案封裝", variant="secondary")
            project_status = gr.Markdown("尚未選擇專案；載入後，設定與處理結果都會自動保存。")
            gr.Markdown("⚠️ 專案設定保存在本機 `data/projects/`，其中 Neo4j Password 為明文；模型 API Key 僅保存在 `.env`。")

        with gr.Tab("1-1 連線設定", interactive=False) as connection_tab:
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Neo4j")
                    neo4j_uri = gr.Textbox(label="URI", value=env["NEO4J_URI"])
                    neo4j_database = gr.Textbox(label="專案專屬 Neo4j Database（匯入圖譜時自動建立）", value="", interactive=False)
                    neo4j_username = gr.Textbox(label="Username", value=env["NEO4J_USERNAME"])
                    neo4j_password = gr.Textbox(label="Password", value=env["NEO4J_PASSWORD"], type="password")
                    neo4j_test_button = gr.Button("測試 Neo4j 連線", variant="primary")
                    neo4j_connection_status = gr.Markdown()
            gr.Markdown("Neo4j 連線設定保存在此頁。模型與 Embedding API 請至 0-4 API Key 設定管理。")
            env_status = gr.Markdown("Neo4j 設定欄位修改後會自動儲存。")

        with gr.Tab("1-2 PDF 與參數", interactive=False) as pdf_tab:
            with gr.Row():
                with gr.Column(scale=1):
                    pdf_file = gr.File(
                        label="PDF 使用手冊（可一次選取多個檔案）",
                        file_types=[".pdf"], file_count="multiple", type="filepath",
                    )
                    chunk_size = gr.Slider(100, 10000, value=1500, step=100, label="Chunk size（字元）")
                    chunk_overlap = gr.Slider(0, 2000, value=200, step=50, label="Chunk overlap（字元）")
                    preview_button = gr.Button("解析並加入專案", variant="primary")
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

        with gr.Tab("1-3 建圖", interactive=False) as graph_tab:
            gr.Markdown("### 規劃並抽取知識圖譜")
            with gr.Row():
                pause_button = gr.Button("⏸ 暫停")
                stop_button = gr.Button("⏹ 停止", variant="stop")
            run_control_status = gr.Markdown(
                "「暫停」「停止」在下方「規劃 Schema」或「抽取實體與關係」"
                "執行中都可使用：暫停只會停止送出新批次（已送出的批次仍會跑完）；"
                "停止會盡快中止整個流程。"
            )
            with gr.Group():
                gr.Markdown("#### ①（可選）規劃 Schema")
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
                    graph_reasoning_effort = gr.Dropdown(
                        choices=list(GPT_6_LUNA_REASONING_EFFORTS),
                        value=DEFAULT_REASONING_EFFORT, label="推理強度",
                        visible=str(preferred_llm or "").casefold() == GPT_6_LUNA_MODEL,
                    )
                    graph_temperature = gr.Slider(
                        0, 2, value=0, step=0.1, label="Temperature",
                        info="選用 GPT-6 Luna 且推理強度非 none 時，API 請求會略過此參數。",
                    )
                with gr.Row():
                    schema_granularity = gr.Radio(
                        ["粗略", "平衡", "詳細"],
                        value="平衡",
                        label="Schema 粒度",
                    )
                    max_concurrent_requests = gr.Number(
                        value=DEFAULT_MAX_CONCURRENT_REQUESTS, minimum=1, precision=0,
                        label="最大並行請求數",
                    )
                schema_documents = gr.Dataframe(
                    headers=["使用", "PDF"],
                    datatype=["bool", "str"],
                    type="array",
                    value=[],
                    interactive=True,
                    static_columns=[1],
                    row_count=(0, "fixed"),
                    col_count=(2, "fixed"),
                    max_height=300,
                    wrap=True,
                    label="用於規劃 Schema 的 PDF",
                )
                gr.Markdown("預設全選；每份勾選的 PDF 都會讀取整份文件。")
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
                gr.Markdown("#### ② 抽取實體與關係")
                gr.Markdown(
                    "若已規劃 Schema，可確認上方 JSON 後抽取；若跳過規劃並留白，"
                    "系統會直接從文件抽取實體與關係。檢查結果後，再手動匯入 Neo4j。"
                )
                extraction_llm_model = gr.Dropdown(
                    choices=llm_choices,
                    value=preferred_llm,
                    allow_custom_value=False,
                    label="知識圖譜抽取 LLM",
                )
                extraction_reasoning_effort = gr.Dropdown(
                    choices=list(GPT_6_LUNA_REASONING_EFFORTS),
                    value=DEFAULT_REASONING_EFFORT, label="推理強度",
                    visible=str(preferred_llm or "").casefold() == GPT_6_LUNA_MODEL,
                )
                extraction_max_concurrent_requests = gr.Number(
                    value=DEFAULT_MAX_CONCURRENT_REQUESTS,
                    minimum=1,
                    precision=0,
                    label="最大並行請求數",
                )
                generate_graph_button = gr.Button(
                    "抽取實體與關係", variant="primary"
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

        with gr.Tab("1-4 問答測試", interactive=False) as qa_tab:
            gr.Markdown(
                "直接使用連線設定中的 Neo4j；預設查詢最近更新的建圖結果。"
                "可選擇使用 LLM Reranker 重排候選，或啟用證據擴展 V2。"
            )
            answer_model = gr.Dropdown(
                choices=llm_choices,
                value=preferred_llm,
                allow_custom_value=False,
                label="問答 LLM",
            )
            answer_reasoning_effort = gr.Dropdown(
                choices=list(GPT_6_LUNA_REASONING_EFFORTS),
                value=DEFAULT_REASONING_EFFORT, label="推理強度",
                visible=str(preferred_llm or "").casefold() == GPT_6_LUNA_MODEL,
            )
            question = gr.Textbox(label="問題", placeholder="例如：設備出現 E01 時該如何處理？")
            with gr.Row():
                retrieval_mode = gr.Radio(strategy_choices(), value="混合檢索", label="檢索模式")
                top_k = gr.Slider(1, 50, value=8, step=1, label="Top K")
                use_reranker = gr.Dropdown(
                    choices=list(RERANKER_MODES), value="停用", label="Reranker 模式",
                    allow_custom_value=False,
                )
                expand_evidence = gr.Dropdown(
                    choices=list(EVIDENCE_EXPANSION_MODES), value="停用", label="證據擴展模式",
                    info="在策略檢索後，透過圖譜關係擴展候選證據。", allow_custom_value=False,
                )
            retrieval_strategy_params_json = _create_strategy_parameter_controls()[0]
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
                .answer-content textarea {
                    font-size: 20px !important;
                    line-height: 1.75 !important;
                }
                </style>
                """,
                padding=False,
            )
            with gr.Group(elem_classes="answer-panel"):
                gr.Markdown("### 回答")
                answer = gr.Textbox(
                    show_label=False, lines=8, max_lines=24, interactive=False,
                    elem_classes="answer-content",
                )
            gr.Markdown("### 檢索來源")
            answer_sources = gr.Dataframe(
                headers=[
                    "類型", "證據", "Retriever", "檢索分數",
                    "來源頁碼", "來源 Chunks", "來源文件",
                ],
                interactive=False,
                wrap=True,
            )

        with gr.Tab("1-5 自動問答測試", interactive=False) as evaluation_tab:
            gr.Markdown(
                "### 從 PDF 自動建立問答測試集\n"
                "每份 PDF 建立指定數量的題目與標準答案；先生成測試回答，再獨立進行模型評測。"
                "勾選允許並行時，不同 PDF 可同時生題，但同一份 PDF 同時只會送出一個請求；未勾選時全部依序處理。"
            )
            with gr.Row():
                evaluation_import_file = gr.File(
                    label="匯入題目（JSON／CSV）", file_types=[".json", ".csv"], type="filepath"
                )
                import_evaluation_button = gr.Button("匯入題目")
                export_evaluation_button = gr.Button("匯出題目")
                evaluation_export_file = gr.File(label="題目 JSON", interactive=False)
            with gr.Group():
                gr.Markdown("#### 生題設定")
                with gr.Row():
                    evaluation_generation_model = gr.Dropdown(
                        choices=llm_choices,
                        value=preferred_llm,
                        allow_custom_value=False,
                        label="生題模型",
                    )
                    evaluation_generation_effort = gr.Dropdown(
                        choices=list(GPT_6_LUNA_REASONING_EFFORTS),
                        value=DEFAULT_REASONING_EFFORT, label="推理強度",
                        visible=str(preferred_llm or "").casefold() == GPT_6_LUNA_MODEL,
                    )
                    evaluation_question_count = gr.Number(value=10, minimum=1, maximum=100, precision=0, label="每份 PDF 題目數 N")
                    evaluation_allow_parallel_generation = gr.Checkbox(
                        value=False, label="允許並行"
                    )
                generate_evaluation_button = gr.Button("從 PDF 建立題目與答案", variant="primary")
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
                .evaluation-judge-button button {
                    background: #f59e0b !important;
                    border-color: #f59e0b !important;
                    color: #1f2937 !important;
                }
                .evaluation-judge-button button:hover {
                    background: #d97706 !important;
                    border-color: #d97706 !important;
                }
                </style>""",
                padding=False,
            )
            gr.Markdown("#### 測試題目")
            evaluation_questions_table = gr.Dataframe(
                headers=["題號", "題目", "正確答案", "題目來源（文件與頁碼）", "答案來源（文件與頁碼）"],
                datatype=["number", "str", "str", "str", "str"],
                type="array", interactive=True, wrap=True,
                elem_classes="evaluation-table",
            )
            with gr.Group():
                gr.Markdown("#### 回答模型設定")
                with gr.Row():
                    evaluation_test_model = gr.Dropdown(
                        choices=llm_choices, value=preferred_llm,
                        allow_custom_value=False, label="回答模型",
                    )
                    evaluation_test_effort = gr.Dropdown(
                        choices=list(GPT_6_LUNA_REASONING_EFFORTS),
                        value=DEFAULT_REASONING_EFFORT, label="推理強度",
                        visible=str(preferred_llm or "").casefold() == GPT_6_LUNA_MODEL,
                    )
                    evaluation_retrieval_mode = gr.Dropdown(
                        strategy_choices(), value="混合檢索", label="檢索策略",
                        allow_custom_value=False,
                    )
                    evaluation_top_k = gr.Number(
                        value=8, minimum=1, maximum=50, precision=0, label="Top K",
                    )
                    evaluation_use_reranker = gr.Dropdown(
                        choices=list(RERANKER_MODES), value="停用", label="Reranker",
                        allow_custom_value=False,
                    )
                    evaluation_expand_evidence = gr.Dropdown(
                        choices=list(EVIDENCE_EXPANSION_MODES), value="停用", label="證據擴展",
                        info="在策略檢索後，透過圖譜關係擴展候選證據。",
                        allow_custom_value=False,
                    )
                evaluation_strategy_params_state = gr.State(
                    _visible_params_for_strategy(
                        RetrievalConfig.from_dict(_default_retrieval_config())
                    )
                )
                evaluation_strategy_parameter_controls = _create_experiment_strategy_parameter_controls(
                    initial_strategy_id="hybrid",
                )
                with gr.Row():
                    evaluation_test_max_concurrent_requests = gr.Number(
                        value=DEFAULT_MAX_CONCURRENT_REQUESTS, minimum=1, precision=0, label="最大並行請求數",
                    )
                generate_evaluation_answers_button = gr.Button("檢索並生成回答", variant="primary")
            with gr.Group(elem_classes="evaluation-metrics-box"):
                evaluation_answers_status = gr.Markdown(
                    "尚未生成測試回答；請先按「檢索並生成回答」。",
                    elem_classes="evaluation-metrics",
                )
            with gr.Group():
                gr.Markdown("#### 評測模型設定")
                with gr.Row():
                    evaluation_judge_model = gr.Dropdown(
                        choices=evaluation_judge_choices, value=DEFAULT_EVALUATION_MODEL,
                        allow_custom_value=False, label="評測模型",
                    )
                    evaluation_judge_effort = gr.Dropdown(
                        choices=list(GPT_6_LUNA_REASONING_EFFORTS),
                        value=DEFAULT_JUDGE_REASONING_EFFORT, label="推理強度",
                        visible=True,
                    )
                    evaluation_judge_max_concurrent_requests = gr.Number(
                        value=DEFAULT_MAX_CONCURRENT_REQUESTS, minimum=1, precision=0, label="最大並行請求數",
                    )
                evaluation_verify_judgment = gr.Checkbox(
                    value=False, label="二階段驗證：複核第一輪判定與理由",
                    info="再呼叫一次評測模型；依第二輪複核結果作為最終判定。",
                )
                run_evaluation_button = gr.Button(
                    "進行評測", variant="primary", interactive=False,
                    elem_classes="evaluation-judge-button",
                )
            gr.Markdown("#### 測試結果")
            evaluation_manual_edit_enabled = gr.Checkbox(
                value=False, label="啟用答案結果人工修改",
                info="勾選後可直接在表格的答案判定欄輸入 0、1 或 2。",
            )
            evaluation_results_table = gr.Dataframe(
                headers=["編號", "問題", "標準答案", "來源 PDF", "實際答案", "答案判定（0錯誤／1部分正確／2全對）", "複核後判定有變更", "評判理由"],
                datatype=["number", "str", "str", "str", "str", "number", "bool", "str"],
                type="array", interactive=True, static_columns=list(range(8)), wrap=True,
                elem_classes=["evaluation-table", "evaluation-results-table"],
            )
            with gr.Group(elem_classes="evaluation-metrics-box"):
                evaluation_status = gr.Markdown(
                    "請先載入專案並解析 PDF。", elem_classes="evaluation-metrics"
                )
        with gr.Tab("1-7 單一專案實驗", interactive=False) as experiment_tab:
            gr.Markdown(
                "使用 0-3 綁定至目前專案的題目集；建立多個不同回答／檢索設定的實驗組，"
                "先生成各組回答，再獨立評測並比較答案正確率、Recall@5、Recall@10 與 MRR。"
            )
            experiment_questions_state = gr.State([])
            with gr.Group():
                gr.Markdown("#### 使用已綁定題目集")
                experiment_bound_question_sets = gr.CheckboxGroup(
                    choices=[], value=[], label="選擇此專案已綁定的題目集",
                )
            experiment_groups_state = gr.State([])
            experiment_results_state = gr.State([])
            experiment_pending_answers_state = gr.State([])
            experiment_run_control_state = gr.State(RunControl())
            experiment_answer_model = gr.Dropdown(
                choices=llm_choices, value=preferred_llm, allow_custom_value=False,
                visible=False, label="實驗預設模型",
            )
            experiment_question_status = gr.Markdown("尚未載入專案題目集。")
            gr.Markdown("#### 回答模型設定｜實驗組（直接編輯欄位；每次變更會自動儲存）")
            experiment_group_rows: list[list[Any]] = []
            for row_index in range(EXPERIMENT_GROUP_LIMIT):
                with gr.Row():
                    group_name = gr.Textbox(label="實驗組名稱", placeholder=f"實驗組 {row_index + 1}", visible=False, scale=2)
                    group_model = gr.Dropdown(
                        choices=llm_choices, value=None, allow_custom_value=False,
                        label="回答模型", visible=False, scale=2,
                    )
                    group_answer_effort = gr.Dropdown(
                        choices=list(GPT_6_LUNA_REASONING_EFFORTS),
                        value=DEFAULT_REASONING_EFFORT, label="回答推理強度",
                        visible=False, scale=1,
                    )
                    group_retrieval = gr.Dropdown(
                        choices=strategy_choices(), value="混合檢索",
                        label="檢索策略", visible=False, scale=2,
                    )
                    group_top_k = gr.Number(value=8, minimum=1, maximum=50, precision=0, label="Top K", visible=False, scale=1)
                    group_reranker = gr.Dropdown(
                        choices=list(RERANKER_MODES), value="停用", label="Reranker",
                        visible=False, scale=2,
                    )
                    group_expansion = gr.Dropdown(
                        choices=list(EVIDENCE_EXPANSION_MODES), value="停用",
                        label="證據擴展", visible=False, scale=2,
                    )
                    group_strategy_params = gr.State({})
                    group_strategy_parameter_controls = _create_experiment_strategy_parameter_controls()
                    delete_group_button = gr.Button("移除", size="sm", visible=False, scale=1)
                experiment_group_rows.append([
                    group_name, group_model, group_answer_effort,
                    group_retrieval, group_top_k,
                    group_reranker, group_expansion, group_strategy_params,
                    *group_strategy_parameter_controls,
                    delete_group_button,
                ])
            experiment_group_fields = [component for row in experiment_group_rows for component in row[:EXPERIMENT_GROUP_FIELDS]]
            experiment_group_all_components = [
                component for row in experiment_group_rows
                for component in [*row[:EXPERIMENT_GROUP_FIELDS], *row[EXPERIMENT_GROUP_FIELDS:-1], row[-1]]
            ]
            add_experiment_group_button = gr.Button("新增實驗組")
            experiment_group_status = gr.Markdown()
            with gr.Row():
                experiment_max_concurrent_requests = gr.Number(
                    value=DEFAULT_MAX_CONCURRENT_REQUESTS, minimum=1, precision=0,
                    label="最大並行請求數",
                )
            generate_experiments_answers_button = gr.Button("檢索並生成回答", variant="primary")
            with gr.Group(elem_classes="evaluation-metrics-box"):
                experiment_answer_progress = gr.HTML(value="", visible=False, padding=False)
                experiment_answer_progress_timer = gr.Timer(value=0.4, active=False)
                experiment_answers_status = gr.Markdown(
                    "尚未生成實驗回答；請先按「檢索並生成回答」。",
                    elem_classes="evaluation-metrics",
                )
            with gr.Group():
                gr.Markdown("#### 評測模型設定")
                with gr.Row():
                    experiment_judge_model = gr.Dropdown(
                        choices=evaluation_judge_choices, value=DEFAULT_EVALUATION_MODEL, allow_custom_value=False,
                        label="全域評測模型", scale=2,
                    )
                    experiment_judge_effort = gr.Dropdown(
                        choices=list(GPT_6_LUNA_REASONING_EFFORTS),
                        value=DEFAULT_JUDGE_REASONING_EFFORT, label="評測推理強度",
                        visible=True, scale=1,
                    )
                    experiment_judge_max_concurrent_requests = gr.Number(
                        value=DEFAULT_MAX_CONCURRENT_REQUESTS, minimum=1, precision=0,
                        label="最大並行請求數",
                    )
                experiment_verify_judgment = gr.Checkbox(
                    value=False, label="二階段驗證：複核第一輪判定與理由",
                    info="再呼叫一次評測模型；依第二輪複核結果作為最終判定。",
                )
                evaluate_experiments_button = gr.Button(
                    "進行評測", variant="primary", interactive=False,
                    elem_classes="evaluation-judge-button",
                )
            gr.HTML(
                """<style>
                .evaluation-judge-button button {background:#f59e0b !important;border-color:#f59e0b !important;color:#1f2937 !important;}
                .evaluation-judge-button button:hover {background:#d97706 !important;border-color:#d97706 !important;}
                </style>""",
                padding=False,
            )
            experiment_status = gr.Markdown()
            gr.Markdown("#### 實驗組摘要")
            experiment_summary_table = gr.Dataframe(
                headers=["實驗組", "回答模型", "Reranker", "證據擴展", "評測模型", "題數", "全對 / 總題數", "部分正確數", "得分正確率", "Recall@5", "Recall@10", "MRR", "成員專案", "題目集"],
                interactive=False, wrap=True,
            )
            gr.Markdown("#### 逐題結果")
            experiment_manual_edit_enabled = gr.Checkbox(
                value=False, label="啟用答案結果人工修改",
                info="勾選後可直接在逐題結果表格的答案判定欄輸入 0、1 或 2。",
            )
            experiment_details_table = gr.Dataframe(
                headers=[
                    "實驗組", "題號", "來源文件", "題目集", "題目", "正確答案",
                    "實際答案", "答案判定（0錯誤／1部分正確／2全對）",
                    "複核後判定有變更", "評判理由",
                ],
                datatype=["str", "number", "str", "str", "str", "str", "str", "number", "bool", "str"],
                type="array", interactive=False, static_columns=[0, 1, 2, 3, 4, 5, 6, 8, 9],
                column_widths=[90, 60, 140, 160, 420, 380, 300, 150, 220, 120],
                wrap=True, elem_classes=["evaluation-table", "evaluation-results-table"],
            )
            with gr.Row():
                export_experiment_results_button = gr.Button("匯出實驗結果 JSON")
                experiment_export_file = gr.File(label="完整逐題結果 JSON", interactive=False)
                experiment_compact_export_file = gr.File(label="簡潔指標結果 JSON", interactive=False)
            experiment_export_status = gr.Markdown()

        with gr.Tab("2-0 實驗專案", interactive=False) as experiment_project_tab:
            gr.Markdown("建立實驗專案，並加入多個已完成建圖的 1 系列專案。各成員專案的 Neo4j Database 仍彼此獨立。")
            with gr.Row():
                available_experiment_projects = experiment_project_choices_for_ui()
                experiment_project_selector = gr.Dropdown(
                    choices=available_experiment_projects,
                    value=(available_experiment_projects[0][1]
                           if available_experiment_projects else None),
                    label="現有實驗專案",
                )
                load_experiment_project_button = gr.Button(
                    "載入實驗專案", elem_classes="project-workspace-action",
                )
                delete_experiment_project_button = gr.Button(
                    "刪除實驗專案", variant="stop", elem_classes="project-workspace-action",
                )
            with gr.Row():
                experiment_project_delete_confirmation_status = gr.Markdown(visible=False)
                confirm_delete_experiment_project_button = gr.Button(
                    "確認刪除", variant="stop", visible=False,
                )
                cancel_delete_experiment_project_button = gr.Button("取消", visible=False)
            with gr.Row():
                new_experiment_project_name = gr.Textbox(label="新實驗專案名稱")
                create_experiment_project_button = gr.Button(
                    "建立實驗專案", variant="primary", elem_classes="project-workspace-action",
                )
            experiment_project_status = gr.Markdown("建立或載入實驗專案。")
            experiment_project_members = gr.Dropdown(
                choices=_built_project_choices(), value=[], multiselect=True,
                label="加入已建圖的 1 系列專案",
                info="只列出已完成 Neo4j 圖譜匯入的專案。",
            )
            save_experiment_members_button = gr.Button("保存成員專案")
            experiment_project_members_table = gr.Dataframe(
                headers=[
                    "專案", "專案 ID", "建立者", "文件數", "Chunk 數", "問題集題數",
                    "Chunk Size", "Overlap", "Schema 粒度", "Neo4j Database",
                ],
                datatype=["str", "str", "str", "number", "number", "number", "number", "number", "str", "str"],
                interactive=False, wrap=True,
            )

        with gr.Tab("2-1 成員專案連線測試", interactive=False) as experiment_connection_tab:
            gr.Markdown("使用 1-1 的 Neo4j URI／帳密，逐一測試實驗專案內各成員專案自己的 Database；回答與 Embedding 模型共用 0-4 的 OpenAI 設定。")
            test_experiment_connections_button = gr.Button("測試所有成員專案連線", variant="primary")
            experiment_connection_status = gr.Markdown("請先在 2-0 載入實驗專案。")
            experiment_connection_table = gr.Dataframe(
                headers=["專案", "專案 ID", "Neo4j Database", "連線結果"],
                datatype=["str", "str", "str", "str"], interactive=False, wrap=True,
            )
            gr.Markdown("回答模型服務共用 0-4 的 OpenAI 設定。")

        with gr.Tab("2-2 自動實驗測試", interactive=False) as experiment_project_test_tab:
            gr.Markdown("每個實驗組會套用至實驗專案內所有成員專案，使用各專案在 0-3 綁定的題目集及自己的 Neo4j Database 執行。")
            gr.Markdown("#### 回答模型設定｜實驗組（直接編輯欄位；每次變更會自動儲存）")
            experiment_project_group_rows: list[list[Any]] = []
            for row_index in range(EXPERIMENT_GROUP_LIMIT):
                with gr.Row():
                    group_name = gr.Textbox(
                        label="實驗組名稱",
                        placeholder=f"實驗組 {row_index + 1}", visible=False, scale=2,
                    )
                    group_model = gr.Dropdown(
                        choices=experiment_llm_choices, value=None, allow_custom_value=False,
                        label="回答模型", visible=False, scale=2,
                    )
                    group_answer_effort = gr.Dropdown(
                        choices=list(GPT_6_LUNA_REASONING_EFFORTS),
                        value=DEFAULT_REASONING_EFFORT, label="回答推理強度",
                        visible=False, scale=1,
                    )
                    group_retrieval = gr.Dropdown(
                        choices=strategy_choices(), value="混合檢索",
                        label="檢索策略", visible=False, scale=2,
                    )
                    group_top_k = gr.Number(
                        value=8, minimum=1, maximum=50, precision=0, label="Top K",
                        visible=False, scale=1,
                    )
                    group_reranker = gr.Dropdown(
                        choices=list(RERANKER_MODES), value="停用", label="Reranker",
                        visible=False, scale=2,
                    )
                    group_expansion = gr.Dropdown(
                        choices=list(EVIDENCE_EXPANSION_MODES), value="停用",
                        label="證據擴展", visible=False, scale=2,
                    )
                    group_strategy_params = gr.State({})
                    group_strategy_parameter_controls = _create_experiment_strategy_parameter_controls()
                    delete_group_button = gr.Button("移除", size="sm", visible=False, scale=1)
                experiment_project_group_rows.append([
                    group_name, group_model, group_answer_effort,
                    group_retrieval, group_top_k, group_reranker,
                    group_expansion, group_strategy_params,
                    *group_strategy_parameter_controls, delete_group_button,
                ])
            experiment_project_group_fields = [
                component for row in experiment_project_group_rows
                for component in row[:EXPERIMENT_GROUP_FIELDS]
            ]
            experiment_project_group_all_components = [
                component for row in experiment_project_group_rows
                for component in [*row[:EXPERIMENT_GROUP_FIELDS], *row[EXPERIMENT_GROUP_FIELDS:-1], row[-1]]
            ]
            add_experiment_project_group_button = gr.Button("新增實驗組")
            experiment_project_groups_status = gr.Markdown("實驗組設定會自動儲存。")
            with gr.Row():
                experiment_project_max_concurrency = gr.Number(
                    value=DEFAULT_MAX_CONCURRENT_REQUESTS, minimum=1, precision=0,
                    label="最大並行請求數",
                )
            gr.Markdown("若遇到 OpenAI TPM 429，可先將並行數調低至 1–3；系統會退避並錯開重試。")
            run_experiment_project_button = gr.Button("檢索並生成回答", variant="primary")
            experiment_project_answer_progress = gr.HTML(value="", visible=False, padding=False)
            experiment_project_answers_status = gr.Markdown("尚未生成實驗回答；請先按「檢索並生成回答」。")
            gr.Markdown("#### 評測模型設定")
            with gr.Row():
                experiment_project_global_judge_model = gr.Dropdown(
                    choices=experiment_llm_choices, value=experiment_default_llm,
                    label="全域評測模型", scale=2,
                )
                experiment_project_judge_effort = gr.Dropdown(
                    choices=list(GPT_6_LUNA_REASONING_EFFORTS), value=DEFAULT_JUDGE_REASONING_EFFORT,
                    label="評測推理強度", visible=preferred_llm == GPT_6_LUNA_MODEL,
                )
                experiment_project_judge_concurrency = gr.Number(
                    value=DEFAULT_MAX_CONCURRENT_REQUESTS, minimum=1, precision=0, label="評測最大並行請求數",
                )
            experiment_project_verify_judgment = gr.Checkbox(
                value=False, label="二階段驗證：複核第一輪判定與理由",
                info="再呼叫一次評測模型；依第二輪複核結果作為最終判定。",
            )
            evaluate_experiment_project_button = gr.Button(
                "開始評測", variant="primary", interactive=False,
                elem_classes="evaluation-judge-button",
            )
            gr.HTML(
                "<style>.evaluation-judge-button button {background:#f59e0b !important;border-color:#f59e0b !important;color:#1f2937 !important;}</style>",
                padding=False,
            )
            experiment_project_test_status = gr.Markdown("請在 2-0 加入專案，並至 0-3 為成員專案綁定題目集。")
            experiment_project_summary_table = gr.Dataframe(
                headers=["實驗組", "回答模型", "Reranker", "證據擴展", "評測模型", "題數", "全對 / 總題數", "部分正確數", "得分正確率", "Recall@5", "Recall@10", "MRR", "成員專案", "題目集"],
                interactive=False, wrap=True,
            )
            experiment_project_details_table = gr.Dataframe(
                headers=["實驗組", "成員專案", "題號", "來源文件", "題目集", "題目", "正確答案", "實際答案", "答案判定（0錯誤／1部分正確／2全對）", "複核後判定有變更", "評判理由"],
                datatype=["str", "str", "number", "str", "str", "str", "str", "str", "number", "bool", "str"],
                type="array", interactive=False, static_columns=[0, 1, 2, 3, 4, 5, 6, 7, 9, 10],
                column_widths=[90, 120, 60, 140, 160, 380, 360, 300, 150, 220, 120],
                wrap=True,
            )
            experiment_project_manual_edit = gr.Checkbox(
                value=False, label="啟用答案結果人工修改",
                info="勾選後可直接在逐題結果表格的答案判定欄輸入 0、1 或 2。",
            )

        schema_model_endpoint = gr.State(initial_llm_credentials[0][0])
        schema_model_key = gr.State(initial_llm_credentials[0][1])
        extraction_model_endpoint = gr.State(initial_llm_credentials[1][0])
        extraction_model_key = gr.State(initial_llm_credentials[1][1])
        generation_model_endpoint = gr.State(initial_llm_credentials[2][0])
        generation_model_key = gr.State(initial_llm_credentials[2][1])
        evaluation_model_endpoint = gr.State(initial_llm_credentials[3][0])
        evaluation_model_key = gr.State(initial_llm_credentials[3][1])
        evaluation_judge_endpoint = gr.State(initial_llm_credentials[3][0])
        evaluation_judge_key = gr.State(initial_llm_credentials[3][1])
        answer_model_endpoint = gr.State(initial_llm_credentials[4][0])
        answer_model_key = gr.State(initial_llm_credentials[4][1])
        selected_embedding_endpoint = gr.State(initial_embedding_credentials[0])
        selected_embedding_key = gr.State(initial_embedding_credentials[1])
        protected_tabs = [
            connection_tab, pdf_tab, graph_tab, qa_tab, evaluation_tab, experiment_tab,
        ]
        experiment_workflow_tabs = [experiment_connection_tab, experiment_project_test_tab]
        access_inputs = [project_selector, neo4j_connected_state, llm_service_state,
                         embedding_service_state, workspace_mode_state, project_state]

        evaluation_tab.select(
            load_evaluation_with_services_for_ui, inputs=[project_selector, llm_service_state],
            outputs=[evaluation_state, evaluation_questions_table, evaluation_results_table,
                     evaluation_generation_model, evaluation_test_model, evaluation_question_count,
                     evaluation_retrieval_mode, evaluation_top_k, evaluation_allow_parallel_generation,
                     evaluation_use_reranker,
                     evaluation_expand_evidence,
                     evaluation_test_max_concurrent_requests, evaluation_judge_model,
                     evaluation_judge_max_concurrent_requests,
                     evaluation_generation_effort, evaluation_test_effort,
                     evaluation_judge_effort,
                     evaluation_strategy_params_state,
                     *evaluation_strategy_parameter_controls,
                     evaluation_status,
                     evaluation_verify_judgment,
                     evaluation_answers_status, run_evaluation_button],
        )
        evaluation_state.change(
            _evaluation_answer_availability,
            inputs=evaluation_state,
            outputs=[evaluation_answers_status, run_evaluation_button],
            show_progress="hidden",
        )
        question_set_manager_tab.select(
            load_central_question_sets_for_ui, inputs=central_question_selector,
            outputs=[central_question_status, central_question_selector, central_question_table],
            show_progress="hidden",
        )
        central_question_selector.change(
            central_question_sets_for_ui, inputs=central_question_selector,
            outputs=[central_question_status, central_question_selector, central_question_table],
            show_progress="hidden",
        )
        import_central_question_button.click(
            import_central_question_set_for_ui,
            inputs=[central_question_file, central_question_name],
            outputs=[central_question_status, central_question_selector, central_question_table, central_question_file],
        )
        refresh_binding_projects_event = question_set_binding_tab.select(
            refresh_binding_project_choices_for_ui, inputs=binding_project_selector,
            outputs=[binding_project_selector, binding_question_set_status, binding_question_set_selector],
            show_progress="hidden",
        )
        binding_project_selector.change(
            project_question_set_bindings_for_ui,
            inputs=[binding_project_selector],
            outputs=[binding_question_set_status, binding_question_set_selector],
            show_progress="hidden",
        )
        save_question_set_bindings_button.click(
            save_project_question_set_bindings_for_ui,
            inputs=[binding_project_selector, binding_question_set_selector],
            outputs=[binding_question_set_status, binding_question_set_selector],
        )
        experiment_bound_question_sets.change(
            select_experiment_question_sets_for_ui,
            inputs=[project_selector, experiment_bound_question_sets],
            outputs=[experiment_questions_state, experiment_question_status],
            show_progress="hidden",
        )
        experiment_question_choices_event = experiment_tab.select(
            bound_question_set_choices_for_ui,
            inputs=[project_selector], outputs=[experiment_bound_question_sets],
            show_progress="hidden",
        )
        experiment_question_choices_event.then(
            load_experiment_for_ui,
            inputs=[project_selector, llm_service_state, experiment_bound_question_sets],
            outputs=[
                experiment_questions_state, experiment_question_status,
                experiment_groups_state, experiment_results_state,
                experiment_max_concurrent_requests, experiment_status,
                experiment_summary_table, experiment_details_table,
                *experiment_group_all_components,
                experiment_judge_model, experiment_judge_effort,
                experiment_judge_max_concurrent_requests,
                experiment_pending_answers_state, experiment_answers_status,
                evaluate_experiments_button, experiment_verify_judgment,
            ],
        )
        experiment_pending_answers_state.change(
            _experiment_answer_availability,
            inputs=experiment_pending_answers_state,
            outputs=[experiment_answers_status, evaluate_experiments_button],
            show_progress="hidden",
        )
        evaluation_preference_inputs = [
            project_selector, evaluation_generation_model, evaluation_test_model, evaluation_question_count,
            evaluation_retrieval_mode, evaluation_top_k,
            evaluation_allow_parallel_generation, evaluation_test_max_concurrent_requests,
            evaluation_use_reranker,
            evaluation_expand_evidence, evaluation_judge_model,
            evaluation_generation_effort, evaluation_test_effort, evaluation_judge_effort,
            evaluation_judge_max_concurrent_requests, evaluation_strategy_params_state,
        ]
        for component in [evaluation_generation_model, evaluation_test_model, evaluation_question_count,
                          evaluation_top_k,
                          evaluation_allow_parallel_generation, evaluation_use_reranker,
                          evaluation_expand_evidence,
                          evaluation_test_max_concurrent_requests, evaluation_judge_model,
                          evaluation_judge_max_concurrent_requests,
                          evaluation_generation_effort, evaluation_test_effort,
                          evaluation_judge_effort]:
            component.input(
                save_evaluation_preferences_for_ui,
                inputs=evaluation_preference_inputs, outputs=evaluation_status,
                show_progress="hidden",
            )
        evaluation_strategy_controls = [
            evaluation_retrieval_mode, *evaluation_strategy_parameter_controls,
        ]
        for component in evaluation_strategy_controls:
            component.input(
                _sync_single_strategy_parameter_controls,
                inputs=evaluation_strategy_controls,
                outputs=[evaluation_strategy_params_state,
                         *evaluation_strategy_parameter_controls],
                show_progress="hidden",
            ).then(
                save_evaluation_preferences_for_ui,
                inputs=evaluation_preference_inputs,
                outputs=evaluation_status,
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
                    evaluation_use_reranker, evaluation_expand_evidence,
                    evaluation_generation_effort, evaluation_strategy_params_state],
            outputs=[evaluation_status, evaluation_questions_table,
                     evaluation_state, evaluation_results_table],
        )
        generate_evaluation_answers_button.click(
            generate_evaluation_answers_for_ui,
            inputs=[project_selector, evaluation_model_endpoint, evaluation_model_key,
                    selected_embedding_endpoint, selected_embedding_key,
                    neo4j_uri, neo4j_database, neo4j_username, neo4j_password,
                    evaluation_test_model, evaluation_retrieval_mode,
                    evaluation_top_k, evaluation_state, evaluation_test_max_concurrent_requests,
                    evaluation_use_reranker, evaluation_expand_evidence,
                    evaluation_test_effort, evaluation_strategy_params_state],
            outputs=[evaluation_status, evaluation_state, evaluation_results_table],
        )
        run_evaluation_button.click(
            evaluate_generated_answers_for_ui,
            inputs=[project_selector, evaluation_judge_endpoint, evaluation_judge_key,
                    evaluation_judge_model, evaluation_state,
                    evaluation_judge_max_concurrent_requests, evaluation_judge_effort,
                    evaluation_verify_judgment],
            outputs=[evaluation_status, evaluation_results_table, evaluation_state],
        )
        evaluation_manual_edit_enabled.change(
            manual_result_editability_for_ui,
            inputs=evaluation_manual_edit_enabled,
            outputs=evaluation_results_table,
            show_progress="hidden",
        )
        evaluation_results_table.input(
            update_manual_evaluation_for_ui,
            inputs=[project_selector, evaluation_results_table, evaluation_state],
            outputs=[evaluation_status, evaluation_results_table, evaluation_state],
            show_progress="hidden",
        )
        add_experiment_group_button.click(
            add_inline_experiment_group_for_ui,
            inputs=[project_selector, experiment_questions_state,
                    experiment_max_concurrent_requests, llm_service_state,
                    experiment_groups_state,
                    *experiment_group_fields],
            outputs=[*experiment_group_all_components, experiment_groups_state,
                     experiment_group_status, experiment_status],
        ).then(lambda: [], outputs=experiment_pending_answers_state, show_progress="hidden")
        inline_group_save_inputs = [
            project_selector, experiment_questions_state, experiment_max_concurrent_requests,
            experiment_groups_state, *experiment_group_fields,
        ]
        strategy_selectors = {row[3] for row in experiment_group_rows}
        parameter_states = {row[7] for row in experiment_group_rows}
        for group_component in [*experiment_group_fields, experiment_max_concurrent_requests]:
            if group_component not in strategy_selectors and group_component not in parameter_states:
                group_component.input(
                    save_inline_experiment_groups_for_ui,
                    inputs=inline_group_save_inputs,
                    outputs=[experiment_group_status, experiment_groups_state, experiment_status],
                    show_progress="hidden",
                )
            if group_component in experiment_group_fields and group_component not in parameter_states:
                group_component.input(lambda: [], outputs=experiment_pending_answers_state, show_progress="hidden")
        for component in [experiment_judge_model, experiment_judge_effort,
                          experiment_judge_max_concurrent_requests]:
            component.input(
                save_experiment_judge_settings_for_ui,
                inputs=[project_selector, experiment_judge_model, experiment_judge_effort,
                        experiment_judge_max_concurrent_requests],
                outputs=experiment_group_status,
                show_progress="hidden",
            )
        experiment_judge_model.change(
            reasoning_effort_visibility,
            inputs=experiment_judge_model, outputs=experiment_judge_effort,
            show_progress="hidden",
        )
        for row_index, row in enumerate(experiment_group_rows):
            parameter_state = row[7]
            parameter_controls = row[EXPERIMENT_GROUP_FIELDS:-1]
            parameter_sync_inputs = [row[3], row[0], *parameter_controls]
            parameter_sync_outputs = [parameter_state, *parameter_controls]
            for trigger in [row[0], row[3], *parameter_controls]:
                trigger.change(
                    _sync_experiment_strategy_parameter_controls,
                    inputs=parameter_sync_inputs,
                    outputs=parameter_sync_outputs,
                    show_progress="hidden",
                )
            for parameter_control in parameter_controls:
                parameter_control.input(
                    lambda: [], outputs=experiment_pending_answers_state,
                    show_progress="hidden",
                )
            parameter_state.change(
                save_inline_experiment_groups_for_ui,
                inputs=inline_group_save_inputs,
                outputs=[experiment_group_status, experiment_groups_state, experiment_status],
                show_progress="hidden",
            )
            row[-1].click(
                partial(remove_inline_experiment_group_for_ui, row_index),
                inputs=[project_selector, experiment_questions_state,
                        experiment_max_concurrent_requests, llm_service_state,
                        experiment_groups_state,
                        *experiment_group_fields],
                outputs=[*experiment_group_all_components, experiment_groups_state,
                         experiment_group_status, experiment_status],
                show_progress="hidden",
            ).then(lambda: [], outputs=experiment_pending_answers_state, show_progress="hidden")
        experiment_answer_progress_timer.tick(
            experiment_answer_progress_for_ui,
            inputs=experiment_run_control_state,
            outputs=experiment_answer_progress,
            show_progress="hidden", queue=False,
        )
        generate_experiments_answers_button.click(
            lambda: gr.update(active=True),
            outputs=experiment_answer_progress_timer,
            queue=False,
            show_progress="hidden",
        ).then(
            generate_experiment_answers_for_ui,
            inputs=[
                project_selector, experiment_questions_state, experiment_groups_state,
                experiment_max_concurrent_requests, llm_service_state,
                selected_embedding_endpoint, selected_embedding_key,
                neo4j_uri, neo4j_database, neo4j_username, neo4j_password,
                experiment_run_control_state,
            ],
            outputs=[
                experiment_status, experiment_pending_answers_state, experiment_results_state,
                experiment_summary_table, experiment_details_table,
            ],
            show_progress="minimal",
        ).then(
            lambda: (gr.update(active=False), gr.update(value="", visible=False)),
            outputs=[experiment_answer_progress_timer, experiment_answer_progress],
            show_progress="hidden",
        )
        evaluate_experiments_button.click(
            evaluate_experiment_answers_with_services_for_ui,
            inputs=[
                project_selector, experiment_pending_answers_state, experiment_groups_state,
                experiment_judge_model, experiment_judge_effort,
                experiment_judge_max_concurrent_requests, llm_service_state,
                experiment_verify_judgment,
            ],
            outputs=[
                experiment_status, experiment_results_state,
                experiment_summary_table, experiment_details_table,
            ],
            show_progress="minimal",
        )
        experiment_details_table.input(
            update_manual_experiment_result_for_ui,
            inputs=[project_selector, experiment_details_table, experiment_results_state],
            outputs=[experiment_status, experiment_summary_table,
                     experiment_details_table, experiment_results_state],
            show_progress="hidden",
        )
        experiment_manual_edit_enabled.change(
            experiment_table_editability_for_ui,
            inputs=experiment_manual_edit_enabled,
            outputs=experiment_details_table,
            show_progress="hidden",
        )
        export_experiment_results_button.click(
            export_experiment_results_for_ui,
            inputs=[project_selector, experiment_details_table],
            outputs=[experiment_export_status, experiment_export_file, experiment_compact_export_file],
            show_progress="hidden",
        )

        experiment_project_outputs = [
            experiment_project_state, experiment_project_members_table,
            experiment_project_members, experiment_project_status,
        ]
        experiment_project_delete_outputs = [
            experiment_project_selector, experiment_project_state,
            experiment_project_members_table, experiment_project_members,
            experiment_project_status, experiment_connection_table,
            experiment_project_connection_state, experiment_connection_status,
            *experiment_project_group_all_components,
            experiment_project_groups_status, experiment_project_max_concurrency,
            experiment_project_pending_answers_state, experiment_project_results_state,
            experiment_project_summary_table,
            experiment_project_details_table, experiment_project_test_status,
            experiment_project_answers_status, evaluate_experiment_project_button,
        ]
        delete_experiment_project_button.click(
            request_experiment_project_deletion_for_ui,
            inputs=[experiment_project_selector],
            outputs=[
                experiment_project_delete_confirmation_status,
                confirm_delete_experiment_project_button,
                cancel_delete_experiment_project_button,
                experiment_project_selector,
            ],
            show_progress="hidden",
        )
        confirm_delete_experiment_project_button.click(
            delete_confirmed_experiment_project_for_ui,
            inputs=[experiment_project_selector],
            outputs=[
                *experiment_project_delete_outputs,
                experiment_project_delete_confirmation_status,
                confirm_delete_experiment_project_button,
                cancel_delete_experiment_project_button,
            ],
        ).then(
            activate_workspace_for_ui,
            inputs=[experiment_workspace_mode, experiment_project_state,
                    workspace_mode_state, experiment_project_connection_state],
            outputs=[workspace_mode_state, experiment_project_connection_state,
                     *experiment_workflow_tabs],
        ).then(
            experiment_project_banner_for_ui,
            inputs=experiment_project_state, outputs=current_project_banner,
        )
        cancel_delete_experiment_project_button.click(
            cancel_experiment_project_deletion_for_ui,
            outputs=[
                experiment_project_delete_confirmation_status,
                confirm_delete_experiment_project_button,
                cancel_delete_experiment_project_button,
                experiment_project_selector,
            ],
            show_progress="hidden",
        )
        load_experiment_project_button.click(
            load_experiment_project_for_ui,
            inputs=[experiment_project_selector], outputs=experiment_project_outputs,
        ).then(
            activate_workspace_for_ui,
            inputs=[experiment_workspace_mode, experiment_project_state,
                    workspace_mode_state, experiment_project_connection_state],
            outputs=[workspace_mode_state, experiment_project_connection_state,
                     *experiment_workflow_tabs],
        ).then(
            workflow_tabs_for_ui, inputs=[project_selector, neo4j_connected_state,
                                          llm_service_state, embedding_service_state,
                                          workspace_mode_state, project_state], outputs=protected_tabs,
        ).then(experiment_project_banner_for_ui, inputs=experiment_project_state, outputs=current_project_banner)
        experiment_project_selector.change(
            load_experiment_project_for_ui,
            inputs=[experiment_project_selector], outputs=experiment_project_outputs,
            show_progress="hidden",
        ).then(
            activate_workspace_for_ui,
            inputs=[experiment_workspace_mode, experiment_project_state,
                    workspace_mode_state, experiment_project_connection_state],
            outputs=[workspace_mode_state, experiment_project_connection_state,
                     *experiment_workflow_tabs], show_progress="hidden",
        ).then(
            workflow_tabs_for_ui, inputs=[project_selector, neo4j_connected_state,
                                          llm_service_state, embedding_service_state,
                                          workspace_mode_state, project_state], outputs=protected_tabs,
            show_progress="hidden",
        ).then(experiment_project_banner_for_ui, inputs=experiment_project_state, outputs=current_project_banner, show_progress="hidden")
        experiment_project_tab.select(
            load_experiment_project_for_ui,
            inputs=[experiment_project_selector], outputs=experiment_project_outputs,
        ).then(
            activate_workspace_for_ui,
            inputs=[experiment_workspace_mode, experiment_project_state,
                    workspace_mode_state, experiment_project_connection_state],
            outputs=[workspace_mode_state, experiment_project_connection_state,
                     *experiment_workflow_tabs],
        ).then(
            workflow_tabs_for_ui, inputs=[project_selector, neo4j_connected_state,
                                          llm_service_state, embedding_service_state,
                                          workspace_mode_state, project_state], outputs=protected_tabs,
        ).then(experiment_project_banner_for_ui, inputs=experiment_project_state, outputs=current_project_banner)
        create_experiment_project_button.click(
            create_experiment_project_for_ui,
            inputs=[new_experiment_project_name],
            outputs=[experiment_project_selector, experiment_project_state, experiment_project_status],
        ).then(
            load_experiment_project_for_ui,
            inputs=[experiment_project_selector], outputs=experiment_project_outputs,
        ).then(
            activate_workspace_for_ui,
            inputs=[experiment_workspace_mode, experiment_project_state,
                    workspace_mode_state, experiment_project_connection_state],
            outputs=[workspace_mode_state, experiment_project_connection_state,
                     *experiment_workflow_tabs],
        ).then(
            workflow_tabs_for_ui, inputs=[project_selector, neo4j_connected_state,
                                          llm_service_state, embedding_service_state,
                                          workspace_mode_state, project_state], outputs=protected_tabs,
        ).then(
            experiment_project_banner_for_ui, inputs=experiment_project_state, outputs=current_project_banner,
        )
        save_experiment_members_button.click(
            save_experiment_project_members_for_ui,
            inputs=[experiment_project_selector, experiment_project_members],
            outputs=[experiment_project_state, experiment_project_members_table, experiment_project_status],
        ).then(
            reset_experiment_project_connection_for_ui,
            inputs=[experiment_project_state],
            outputs=[experiment_project_connection_state,
                     *experiment_workflow_tabs],
        )
        test_experiment_connections_button.click(
            test_experiment_project_connections_for_ui,
            inputs=[experiment_project_selector, neo4j_uri, neo4j_username, neo4j_password],
            outputs=[experiment_connection_table, experiment_project_connection_state, experiment_connection_status],
        ).then(
            experiment_workflow_tabs_for_ui,
            inputs=[workspace_mode_state, experiment_project_state, experiment_project_connection_state],
            outputs=experiment_workflow_tabs,
        )
        experiment_project_test_tab.select(
            load_experiment_project_setup_for_ui,
            inputs=[experiment_project_state, experiment_llm_service_state],
            outputs=[
                *experiment_project_group_all_components,
                experiment_project_max_concurrency, experiment_project_summary_table,
                experiment_project_details_table, experiment_project_test_status,
                experiment_project_groups_status, experiment_project_verify_judgment,
            ],
        ).then(
            load_experiment_project_runtime_state_for_ui,
            inputs=[experiment_project_state, experiment_llm_service_state],
            outputs=[experiment_project_global_judge_model, experiment_project_judge_effort,
                     experiment_project_judge_concurrency, experiment_project_pending_answers_state,
                     experiment_project_results_state, evaluate_experiment_project_button,
                     experiment_project_answers_status],
        ).then(
            refresh_experiment_model_choices_for_ui,
            inputs=[experiment_llm_service_state,
                    *[row[1] for row in experiment_project_group_rows],
                    experiment_project_global_judge_model],
            outputs=[*[row[1] for row in experiment_project_group_rows],
                     experiment_project_global_judge_model],
            show_progress="hidden",
        )
        add_experiment_project_group_button.click(
            add_experiment_project_inline_group_for_ui,
            inputs=[experiment_project_state, experiment_llm_service_state,
                    *experiment_project_group_fields],
            outputs=[
                *experiment_project_group_all_components,
                experiment_project_state, experiment_project_groups_status,
            ],
        )
        experiment_project_group_save_inputs = [experiment_project_state, *experiment_project_group_fields]
        strategy_selectors = {row[3] for row in experiment_project_group_rows}
        parameter_states = {row[7] for row in experiment_project_group_rows}
        for component in experiment_project_group_fields:
            if component not in strategy_selectors and component not in parameter_states:
                component.input(
                    save_experiment_project_inline_groups_for_ui,
                    inputs=experiment_project_group_save_inputs,
                    outputs=[experiment_project_state, experiment_project_groups_status],
                    show_progress="hidden",
                )
        for row_index, row in enumerate(experiment_project_group_rows):
            parameter_state = row[7]
            parameter_controls = row[EXPERIMENT_GROUP_FIELDS:-1]
            parameter_sync_inputs = [row[3], row[0], *parameter_controls]
            parameter_sync_outputs = [parameter_state, *parameter_controls]
            for trigger in [row[0], row[3], *parameter_controls]:
                trigger.change(
                    _sync_experiment_strategy_parameter_controls,
                    inputs=parameter_sync_inputs,
                    outputs=parameter_sync_outputs,
                    show_progress="hidden",
                )
            parameter_state.change(
                save_experiment_project_inline_groups_for_ui,
                inputs=experiment_project_group_save_inputs,
                outputs=[experiment_project_state, experiment_project_groups_status],
                show_progress="hidden",
            )
            row[-1].click(
                partial(remove_experiment_project_inline_group_for_ui, row_index),
                inputs=[experiment_project_state, experiment_llm_service_state,
                        *experiment_project_group_fields],
                outputs=[*experiment_project_group_all_components,
                         experiment_project_state, experiment_project_groups_status],
                show_progress="hidden",
            )
            row[1].change(
                experiment_group_reasoning_effort_visibility,
                inputs=[row[1], row[0]], outputs=row[2],
                show_progress="hidden",
            )
        experiment_project_global_judge_model.input(
            save_experiment_project_judge_settings_for_ui,
            inputs=[experiment_project_state, experiment_project_global_judge_model,
                    experiment_project_judge_effort, experiment_project_judge_concurrency],
            outputs=[experiment_project_state, experiment_project_groups_status],
            show_progress="hidden",
        )
        for component in [experiment_project_judge_effort, experiment_project_judge_concurrency]:
            component.input(
                save_experiment_project_judge_settings_for_ui,
                inputs=[experiment_project_state, experiment_project_global_judge_model,
                        experiment_project_judge_effort, experiment_project_judge_concurrency],
                outputs=[experiment_project_state, experiment_project_groups_status],
                show_progress="hidden",
            )
        experiment_project_global_judge_model.change(
            reasoning_effort_visibility,
            inputs=experiment_project_global_judge_model, outputs=experiment_project_judge_effort,
            show_progress="hidden",
        )
        experiment_project_generation_event = run_experiment_project_button.click(
            generate_experiment_project_answers_for_ui,
            inputs=[
                experiment_project_state, experiment_project_max_concurrency,
                experiment_llm_service_state, selected_embedding_endpoint, selected_embedding_key,
                neo4j_uri, neo4j_username, neo4j_password,
            ],
            outputs=[
                experiment_project_answers_status, experiment_project_pending_answers_state,
                experiment_project_summary_table, experiment_project_details_table,
                experiment_project_state,
            ],
            show_progress="minimal",
        )
        experiment_project_generation_event.then(
            lambda pending: gr.update(interactive=bool(pending)),
            inputs=experiment_project_pending_answers_state,
            outputs=evaluate_experiment_project_button, show_progress="hidden",
        )
        evaluate_experiment_project_button.click(
            evaluate_experiment_project_answers_for_ui,
            inputs=[experiment_project_state, experiment_project_pending_answers_state,
                    experiment_project_global_judge_model, experiment_project_judge_effort,
                    experiment_project_judge_concurrency, experiment_llm_service_state,
                    experiment_project_verify_judgment],
            outputs=[experiment_project_test_status, experiment_project_results_state,
                     experiment_project_summary_table, experiment_project_details_table,
                     experiment_project_state],
            show_progress="minimal",
        )
        experiment_project_manual_edit.input(
            experiment_table_editability_for_ui,
            inputs=experiment_project_manual_edit, outputs=experiment_project_details_table,
            show_progress="hidden",
        )
        experiment_project_details_table.input(
            update_manual_experiment_project_result_for_ui,
            inputs=[experiment_project_state, experiment_project_details_table,
                    experiment_project_results_state],
            outputs=[experiment_project_test_status, experiment_project_summary_table,
                     experiment_project_details_table, experiment_project_state],
            show_progress="hidden",
        )
        project_setting_inputs = [
            project_selector, documents_state, chunk_state, graph_state,
            neo4j_uri, neo4j_database, neo4j_username, neo4j_password,
            model_endpoint, api_key, graph_llm_model, graph_embedding_model,
            answer_model, chunk_size, chunk_overlap,
            graph_temperature, schema_granularity,
            max_concurrent_requests,
            extraction_llm_model,
            extraction_max_concurrent_requests, retrieval_mode,
            top_k, schema_editor, graph_reasoning_effort,
            extraction_reasoning_effort, answer_reasoning_effort,
            retrieval_strategy_params_json,
        ]
        project_load_outputs = [
            project_state, project_status,
            neo4j_uri, neo4j_database, neo4j_username, neo4j_password,
            model_endpoint, api_key, graph_llm_model, graph_embedding_model,
            answer_model, chunk_size, chunk_overlap,
            graph_temperature, schema_granularity,
            max_concurrent_requests,
            extraction_llm_model,
            extraction_max_concurrent_requests, retrieval_mode,
            top_k, schema_editor,
            documents_state, chunk_state, graph_state,
            active_preview_state, active_chunks_state,
            documents_table, remove_document_selector,
            chunk_table, page_status,
            entity_table, relationship_table, build_status, import_status,
            graph_reasoning_effort, extraction_reasoning_effort,
            answer_reasoning_effort, retrieval_strategy_params_json,
        ]
        for effort_control in [
            graph_reasoning_effort, extraction_reasoning_effort, answer_reasoning_effort,
        ]:
            effort_control.input(
                save_project_for_ui, inputs=project_setting_inputs,
                outputs=[project_state, project_status], show_progress="hidden",
            )
        for retrieval_control in [retrieval_mode, top_k, retrieval_strategy_params_json]:
            retrieval_control.input(
                save_project_for_ui, inputs=project_setting_inputs,
                outputs=[project_state, project_status], show_progress="hidden",
            )
        current_user_dropdown.change(
            select_user_for_ui,
            inputs=current_user_dropdown,
            outputs=[current_user_banner, question_set_manager_tab, question_set_binding_tab,
                     api_settings_tab, project_tab, experiment_project_tab],
            show_progress="hidden",
        )
        project_tab.select(refresh_projects_for_ui, inputs=project_state, outputs=project_selector)
        app.load(refresh_projects_for_ui, inputs=project_state, outputs=project_selector)
        project_selector.input(
            ensure_project_selection_for_ui,
            inputs=project_selector, outputs=project_selector,
            show_progress="hidden",
        ).then(
            reset_manual_result_editability_for_ui,
            outputs=[evaluation_manual_edit_enabled, evaluation_results_table,
                     experiment_manual_edit_enabled, experiment_details_table],
            show_progress="hidden",
        )
        create_project_event = create_project_button.click(
            create_project_for_ui, inputs=new_project_name,
            outputs=[project_selector, project_state, project_status],
        )
        export_project_button.click(
            export_project_for_ui, inputs=project_selector,
            outputs=[export_project_file, project_status],
        )
        import_project_event = import_project_button.click(
            import_project_for_ui, inputs=import_project_file,
            outputs=[project_selector, project_state, project_status],
        )
        import_load_event = import_project_event.success(
            load_project_with_services_for_ui,
            inputs=[project_selector, llm_service_state, embedding_service_state],
            outputs=[*project_load_outputs,
                     llm_provider, llm_service_state, llm_models_table, model_test_button, model_list_button,
                     model_connection_status, evaluation_generation_model, evaluation_test_model,
                     embedding_provider, embedding_service_state, embedding_models_table,
                     embedding_test_button, embedding_list_button, embedding_connection_status],
            show_progress="hidden",
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
        load_project_event.success(
            schema_documents_for_ui,
            inputs=documents_state,
            outputs=schema_documents,
            show_progress="hidden",
        )
        initialize_project_event.success(
            schema_documents_for_ui,
            inputs=documents_state,
            outputs=schema_documents,
            show_progress="hidden",
        )
        delete_project_button.click(
            request_project_deletion_for_ui,
            inputs=project_selector,
            outputs=[project_delete_confirmation_status,
                     confirm_delete_project_button, cancel_delete_project_button,
                     project_selector],
            show_progress="hidden",
        )
        delete_project_event = confirm_delete_project_button.click(
            delete_confirmed_project_for_ui,
            inputs=project_selector,
            outputs=[project_selector, project_state, project_status,
                     connection_tab, pdf_tab, graph_tab, qa_tab, evaluation_tab, experiment_tab,
                     delete_project_completed, project_delete_confirmation_status,
                     confirm_delete_project_button, cancel_delete_project_button],
        )
        delete_project_event.then(
            refresh_projects_after_delete_for_ui,
            inputs=delete_project_completed,
            outputs=project_selector,
            show_progress="hidden",
        )
        cancel_delete_project_button.click(
            cancel_project_deletion_for_ui,
            outputs=[project_delete_confirmation_status,
                     confirm_delete_project_button, cancel_delete_project_button,
                     project_selector],
            show_progress="hidden",
        )
        for project_load_event in [import_load_event, load_project_event, initialize_project_event]:
            project_load_event.then(
                lambda: False, outputs=neo4j_connected_state, show_progress="hidden",
            ).then(
                activate_workspace_for_ui,
                inputs=[single_workspace_mode, project_state,
                        workspace_mode_state, experiment_project_connection_state],
                outputs=[workspace_mode_state, experiment_project_connection_state,
                         *experiment_workflow_tabs], show_progress="hidden",
            ).then(
                workflow_tabs_for_ui, inputs=access_inputs, outputs=protected_tabs,
                show_progress="hidden",
            )
        project_state.change(
            _current_project_banner, inputs=project_state, outputs=current_project_banner,
            show_progress="hidden",
        )
        auto_save_components = [
            neo4j_uri, neo4j_database, neo4j_username, neo4j_password,
            graph_llm_model, graph_embedding_model,
            answer_model, chunk_size, chunk_overlap,
            graph_temperature, schema_granularity,
            max_concurrent_requests,
            extraction_llm_model,
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
            inputs=[neo4j_uri, neo4j_database, neo4j_username, neo4j_password, project_selector],
            outputs=[neo4j_connection_status, neo4j_connected_state],
        )
        neo4j_test_event.then(
            workflow_tabs_for_ui, inputs=access_inputs, outputs=protected_tabs,
            show_progress="hidden",
        )
        for neo4j_field in [neo4j_uri, neo4j_database, neo4j_username, neo4j_password]:
            invalidation = neo4j_field.input(
                lambda: (False, "設定已變更，請重新測試 Neo4j 連線。", {}),
                outputs=[neo4j_connected_state, neo4j_connection_status,
                         experiment_project_connection_state],
                show_progress="hidden",
            )
            invalidation.then(
                workflow_tabs_for_ui, inputs=access_inputs, outputs=protected_tabs,
                show_progress="hidden",
            ).then(
                experiment_workflow_tabs_for_ui,
                inputs=[workspace_mode_state, experiment_project_state,
                        experiment_project_connection_state],
                outputs=experiment_workflow_tabs, show_progress="hidden",
            )
        llm_model_fields = [
            graph_llm_model, extraction_llm_model, evaluation_generation_model,
            evaluation_test_model, answer_model, experiment_answer_model,
        ]
        credential_refresh_inputs = [
            llm_service_state, embedding_service_state,
            graph_llm_model, extraction_llm_model, evaluation_generation_model,
            evaluation_test_model, evaluation_judge_model, answer_model,
            graph_embedding_model,
        ]
        credential_refresh_outputs = [
            schema_model_endpoint, schema_model_key,
            extraction_model_endpoint, extraction_model_key,
            generation_model_endpoint, generation_model_key,
            evaluation_model_endpoint, evaluation_model_key,
            evaluation_judge_endpoint, evaluation_judge_key,
            answer_model_endpoint, answer_model_key,
            selected_embedding_endpoint, selected_embedding_key,
        ]
        for project_load_event in [import_load_event, load_project_event, initialize_project_event]:
            project_load_event.then(
                refresh_model_credentials_for_ui,
                inputs=credential_refresh_inputs,
                outputs=credential_refresh_outputs,
                show_progress="hidden",
            )
        llm_service_outputs = [
            llm_service_state, model_endpoint, api_key, llm_models_table,
            model_test_button, model_list_button, model_connection_status, *llm_model_fields,
        ]
        embedding_service_outputs = [
            embedding_service_state, embedding_api_base, embedding_api_key, embedding_models_table,
            embedding_test_button, embedding_list_button, embedding_connection_status, graph_embedding_model,
        ]
        global_service_outputs = [
            *llm_service_outputs, *embedding_service_outputs,
            experiment_llm_service_state, global_api_status,
        ]
        global_service_inputs = [
            llm_service_state, embedding_service_state, experiment_llm_service_state,
            global_api_endpoint, global_api_key,
            *llm_model_fields, graph_embedding_model,
        ]
        global_service_events = [
            global_api_test_button.click(
                partial(persist_global_api_settings_for_ui, "test"),
                inputs=global_service_inputs, outputs=global_service_outputs,
                concurrency_id="service-settings",
            ),
        ]
        for component in [global_api_endpoint, global_api_key]:
            global_service_events.append(component.input(
                partial(persist_global_api_settings_for_ui, "edit"),
                inputs=global_service_inputs, outputs=global_service_outputs,
                show_progress="hidden", concurrency_id="service-settings",
            ))
        for service_event in global_service_events:
            service_event.then(
                refresh_experiment_model_choices_for_ui,
                inputs=[llm_service_state, evaluation_judge_model,
                        experiment_judge_model,
                        *[row[1] for row in experiment_group_rows]],
                outputs=[evaluation_judge_model, experiment_judge_model,
                         *[row[1] for row in experiment_group_rows]],
                show_progress="hidden",
            ).then(
                refresh_model_credentials_for_ui,
                inputs=credential_refresh_inputs, outputs=credential_refresh_outputs,
                show_progress="hidden",
            )
        for field, endpoint_state, key_state in [
            (graph_llm_model, schema_model_endpoint, schema_model_key),
            (extraction_llm_model, extraction_model_endpoint, extraction_model_key),
            (evaluation_generation_model, generation_model_endpoint, generation_model_key),
            (evaluation_test_model, evaluation_model_endpoint, evaluation_model_key),
            (evaluation_judge_model, evaluation_judge_endpoint, evaluation_judge_key),
            (answer_model, answer_model_endpoint, answer_model_key),
        ]:
            field.change(
                resolve_model_credentials_for_ui,
                inputs=[llm_service_state, field], outputs=[endpoint_state, key_state],
                show_progress="hidden",
            )
        for model_field, effort_field in [
            (graph_llm_model, graph_reasoning_effort),
            (extraction_llm_model, extraction_reasoning_effort),
            (answer_model, answer_reasoning_effort),
            (evaluation_generation_model, evaluation_generation_effort),
            (evaluation_test_model, evaluation_test_effort),
            (evaluation_judge_model, evaluation_judge_effort),
            (experiment_project_global_judge_model, experiment_project_judge_effort),
        ]:
            model_field.change(
                reasoning_effort_visibility, inputs=model_field, outputs=effort_field,
                show_progress="hidden",
            )
        for row in experiment_project_group_rows:
            row[1].change(
                experiment_group_reasoning_effort_visibility,
                inputs=[row[1], row[0]], outputs=row[2],
                show_progress="hidden",
            )
        for row in experiment_group_rows:
            row[1].change(
                experiment_group_reasoning_effort_visibility,
                inputs=[row[1], row[0]], outputs=row[2],
                show_progress="hidden",
            )
        graph_embedding_model.change(
            resolve_model_credentials_for_ui,
            inputs=[embedding_service_state, graph_embedding_model],
            outputs=[selected_embedding_endpoint, selected_embedding_key],
            show_progress="hidden",
        )
        env_inputs = [neo4j_uri, neo4j_username, neo4j_password,
                      model_endpoint, api_key, embedding_api_base,
                      embedding_api_key, graph_llm_model, graph_embedding_model,
                      answer_model]
        for component in [neo4j_uri, neo4j_username, neo4j_password]:
            component.change(persist_env_settings, inputs=env_inputs, outputs=env_status)
        reload_event = reload_button.click(
            reload_env_with_services_for_ui,
            outputs=[neo4j_uri, neo4j_username, neo4j_password,
                     llm_provider, *llm_service_outputs,
                     embedding_provider, *embedding_service_outputs,
                     global_api_endpoint, global_api_key,
                     global_api_status, env_status],
        )
        reload_reset_event = reload_event.then(
            lambda: False, outputs=neo4j_connected_state, show_progress="hidden",
        )
        reload_reset_event.then(
            refresh_model_credentials_for_ui,
            inputs=credential_refresh_inputs,
            outputs=credential_refresh_outputs,
            show_progress="hidden",
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
        ).then(
            schema_documents_for_ui,
            inputs=documents_state,
            outputs=schema_documents,
            show_progress="hidden",
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
        ).then(
            schema_documents_for_ui,
            inputs=documents_state,
            outputs=schema_documents,
            show_progress="hidden",
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
        plan_schema_button.click(
            plan_schema_for_ui,
            inputs=[
                schema_model_endpoint,
                schema_model_key,
                graph_llm_model,
                graph_temperature,
                schema_granularity,
                max_concurrent_requests,
                schema_documents,
                chunk_state,
                run_control_state,
                graph_reasoning_effort,
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
                chunk_size,
                chunk_overlap,
                schema_granularity,
                extraction_reasoning_effort,
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
                use_reranker,
                expand_evidence,
                answer_reasoning_effort,
                retrieval_strategy_params_json,
            ],
            outputs=[answer_status, answer, answer_sources],
        )
    bind_gradio_callbacks_to_actor(app, current_user_dropdown)
    return app
