import csv
import json
from pathlib import Path
from threading import Barrier, Event, Lock, Thread

import gradio as gr
import pytest

from manual_graphrag import service_settings as settings
from manual_graphrag import ui
from manual_graphrag.chunking import PageText, TextChunk
from manual_graphrag.ui import build_app, connection_summary, persist_env_settings


def test_build_app_returns_blocks() -> None:
    assert isinstance(build_app(), gr.Blocks)


def test_current_project_banner_reflects_project_name() -> None:
    assert ui._current_project_banner({}) == "### 📁 目前專案：尚未選擇"
    assert ui._current_project_banner({"name": "手冊專案"}) == "### 📁 目前專案：手冊專案"


def test_build_app_wires_project_state_change_to_banner() -> None:
    app = build_app()
    assert any(
        component.get("props", {}).get("value") == "### 📁 目前專案：尚未選擇"
        for component in app.config["components"]
    )
    dependency = next(
        item for item in app.config["dependencies"]
        if str(item.get("api_name", "")).startswith("_current_project_banner")
    )
    assert any(
        (target[1] if isinstance(target, (list, tuple)) else None) == "change"
        for target in dependency.get("targets", [])
    )


def test_model_fields_only_offer_initially_checked_models() -> None:
    app = build_app()
    labels = {
        "Schema 規劃 LLM",
        "知識圖譜抽取 LLM",
        "Embedding 模型",
        "生題模型",
        "回答模型", "評測模型",
        "問答 LLM",
    }
    fields = {
        component.get("props", {}).get("label"): component
        for component in app.config["components"]
        if component.get("props", {}).get("label") in labels
    }

    assert set(fields) == labels
    assert all(field["type"] == "dropdown" for field in fields.values())
    assert all(not field["props"]["allow_custom_value"] for field in fields.values())
    for field in fields.values():
        for display, value in field["props"]["choices"]:
            assert display in {
                f"OpenAI｜{value}", f"Ollama｜{value}",
                f"OpenAI｜{value}（目前不可用）" if value == "gpt-6-luna" else "",
            }


def test_graph_controls_are_above_schema_and_type_limit_fields_are_removed() -> None:
    app = build_app()
    components = app.config["components"]
    ids_by_value = {
        value: component["id"]
        for component in components
        if isinstance((value := component.get("props", {}).get("value")), str)
    }
    schema_heading = ids_by_value["#### ① 規劃 Schema"]
    assert ids_by_value["⏸ 暫停"] < schema_heading
    assert ids_by_value["⏹ 停止"] < schema_heading
    labels = {component.get("props", {}).get("label") for component in components}
    assert "最大實體類型數" not in labels
    assert "最大關係類型數" not in labels


def test_pages_three_through_five_default_judge_fields_to_luna(monkeypatch) -> None:
    monkeypatch.setattr(ui, "service_choices", lambda state: ["gpt-4.1-mini", "gpt-4o-mini"])
    monkeypatch.setattr(
        ui, "service_choice_items",
        lambda state: [("OpenAI｜gpt-4.1-mini", "gpt-4.1-mini"), ("OpenAI｜gpt-4o-mini", "gpt-4o-mini")],
    )
    monkeypatch.setattr(ui, "preferred_service_model", lambda state: "gpt-4.1-mini")
    app = build_app()
    labels = {
        "Schema 規劃 LLM", "知識圖譜抽取 LLM", "生題模型", "回答模型", "評測模型", "問答 LLM",
    }
    fields = [
        component for component in app.config["components"]
        if component.get("props", {}).get("label") in labels
        and component.get("props", {}).get("visible", True)
    ]
    assert len(fields) == 6
    assert all(
        field["props"]["value"] == (
            "gpt-6-luna" if field["props"].get("label") == "評測模型" else "gpt-4.1-mini"
        )
        for field in fields
    )


def test_pause_and_stop_buttons_bypass_the_queue() -> None:
    app = build_app()
    pause_button = next(
        component for component in app.config["components"]
        if component.get("props", {}).get("value") == "⏸ 暫停"
    )
    stop_button = next(
        component for component in app.config["components"]
        if component.get("props", {}).get("value") == "⏹ 停止"
    )
    pause_dependency = next(
        item for item in app.config["dependencies"]
        if str(item.get("api_name", "")).startswith("toggle_pause_for_ui")
    )
    stop_dependencies = [
        item for item in app.config["dependencies"]
        if str(item.get("api_name", "")).startswith("request_stop_for_ui")
    ]
    assert stop_button["props"]["variant"] == "stop"
    assert pause_dependency["queue"] is False
    assert all(dependency["queue"] is False for dependency in stop_dependencies)
    assert any(target[0] == pause_button["id"] for target in pause_dependency["targets"])
    assert any(
        target[0] == stop_button["id"]
        for dependency in stop_dependencies for target in dependency["targets"]
    )
    experiment_stop_button = next(
        component for component in app.config["components"]
        if component.get("props", {}).get("value") == "停止實驗"
    )
    assert any(
        target[0] == experiment_stop_button["id"]
        for dependency in stop_dependencies for target in dependency["targets"]
    )


def test_plan_schema_button_uses_primary_variant() -> None:
    app = build_app()
    button = next(
        component
        for component in app.config["components"]
        if component.get("props", {}).get("value") == "分析文件並規劃 Schema"
    )

    assert button["props"]["variant"] == "primary"


def test_project_page_uses_automatic_refresh_and_save() -> None:
    app = build_app()
    button_values = {
        component.get("props", {}).get("value")
        for component in app.config["components"]
        if component.get("type") == "button"
    }

    assert "重新整理專案清單" not in button_values
    assert "保存目前專案設定" not in button_values
    assert any(
        dependency.get("api_name") == "refresh_projects_for_ui"
        and any(trigger[1] == "select" for trigger in dependency.get("targets", []))
        for dependency in app.config["dependencies"]
    )
    assert any(
        str(dependency.get("api_name", "")).startswith("save_project_for_ui")
        and any(trigger[1] == "input" for trigger in dependency.get("targets", []))
        for dependency in app.config["dependencies"]
    )
    one_click_button = next(
        component for component in app.config["components"]
        if component.get("props", {}).get("value") == "一鍵測試"
    )
    one_click_dependency = next(
        dependency for dependency in app.config["dependencies"]
        if str(dependency.get("api_name", "")).startswith("test_all_connections_for_ui")
    )
    assert any(target[0] == one_click_button["id"] for target in one_click_dependency["targets"])
    assert len(one_click_dependency["outputs"]) == 24


def test_project_selector_defaults_to_first_available_project(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    project = ui.create_project("預設選取")
    app = build_app()
    selector = next(
        component for component in app.config["components"]
        if component.get("props", {}).get("label") == "現有專案"
    )

    assert selector["props"]["value"] == project["project_id"]
    assert ui.refresh_projects_for_ui(None)["value"] == project["project_id"]
    assert ui.ensure_project_selection_for_ui(None)["value"] == project["project_id"]
    assert ui.ensure_project_selection_for_ui(project["project_id"])["value"] == project["project_id"]


def test_pdf_upload_starts_with_valid_multi_document_controls() -> None:
    app = build_app()
    fields = {
        component.get("props", {}).get("label"): component
        for component in app.config["components"]
    }

    upload = fields["PDF 使用手冊（可一次選取多個檔案）"]["props"]
    assert upload["file_count"] == "multiple"
    assert fields["選擇要移除的 PDF（可多選）"]["props"]["choices"] == []
    assert any(
        component.get("props", {}).get("headers")
        == ["選取", "文件", "頁碼範圍", "Chunk 數"]
        for component in app.config["components"]
    )


def test_evaluation_generation_uses_parallel_checkbox() -> None:
    app = build_app()
    fields = {
        component.get("props", {}).get("label"): component
        for component in app.config["components"]
    }

    assert fields["允許並行"]["type"] == "checkbox"
    assert fields["允許並行"]["props"]["value"] is False
    assert "生題最大並行請求數（跨 PDF）" not in fields


def test_all_concurrency_inputs_show_ollama_recommendation() -> None:
    app = build_app()
    concurrency_inputs = [
        component for component in app.config["components"]
        if "最大並行請求數" in str(component.get("props", {}).get("label", ""))
    ]

    assert len(concurrency_inputs) == 8
    assert all(
        component["props"].get("info") == ui.OLLAMA_CONCURRENCY_HINT
        for component in concurrency_inputs
    )


def test_reranker_and_graph_evidence_expansion_default_off() -> None:
    app = build_app()
    defaults = {
        component.get("props", {}).get("label"): component.get("props", {}).get("value")
        for component in app.config["components"]
        if component.get("props", {}).get("label") in {"使用 LLM Reranker", "證據擴展 V2"}
    }

    assert defaults == {"使用 LLM Reranker": False, "證據擴展 V2": False}



def test_evaluation_results_table_uses_smaller_font_class() -> None:
    app = build_app()
    results_table = next(
        component
        for component in app.config["components"]
        if component.get("props", {}).get("headers")
        == ["編號", "問題", "標準答案", "來源 PDF", "實際答案", "答案判定（勾選=正確）", "評判理由"]
    )
    html_styles = "\n".join(
        str(component.get("props", {}).get("value", ""))
        for component in app.config["components"]
        if component.get("type") == "html"
    )

    assert "evaluation-results-table" in results_table["props"]["elem_classes"]
    assert results_table["props"]["interactive"] is False
    assert any(
        component.get("props", {}).get("label") == "啟用答案結果人工修改"
        and component.get("props", {}).get("value") is False
        for component in app.config["components"]
    )
    assert ".evaluation-results-table table" in html_styles
    assert "font-size: 14px !important" in html_styles


def test_single_project_pages_stay_locked_but_experiment_pages_are_enabled() -> None:
    app = build_app()
    protected_labels = {
        "0-2 PDF 與參數", "0-3 建圖",
        "0-4 問答測試", "0-5 自動問答測試", "0-6 單一專案實驗",
    }
    tabs = [
        component for component in app.config["components"]
        if component.get("props", {}).get("label") in protected_labels
    ]
    experiment_tabs = [
        component for component in app.config["components"]
        if component.get("props", {}).get("label") in {
            "1-0 實驗專案", "1-1 成員專案連線測試",
            "1-2 問題集準備", "1-3 自動實驗測試",
        }
    ]

    assert len(tabs) == 5
    assert len(experiment_tabs) == 4
    assert all(tab["props"]["interactive"] is True for tab in experiment_tabs)
    assert not any(
        "歷史紀錄" in str(component.get("props", {}).get("label", ""))
        for component in app.config["components"]
    )
    connection_tab = next(
        component for component in app.config["components"]
        if component.get("props", {}).get("label") == "0-1 連線設定"
    )
    assert connection_tab["props"].get("interactive", True) is True
    assert all(tab["props"]["interactive"] is False for tab in tabs)
    llm = settings.load_service_settings("llm")
    embedding = settings.load_service_settings("embedding")
    assert all(update["interactive"] is False for update in ui.workflow_tabs_for_ui("", False, llm, embedding))
    llm["profiles"]["OpenAI"]["connected"] = True
    embedding["profiles"]["OpenAI"]["connected"] = True
    assert all(update["interactive"] is False for update in ui.workflow_tabs_for_ui("project", False, llm, embedding))
    assert all(update["interactive"] is True for update in ui.workflow_tabs_for_ui("project", True, llm, embedding))
    page_labels = {
        component.get("props", {}).get("label") for component in app.config["components"]
    }
    assert {
        "0-0 專案設定", "0-1 連線設定", "0-2 PDF 與參數", "0-3 建圖",
        "0-4 問答測試", "0-5 自動問答測試", "0-6 單一專案實驗",
        "1-0 實驗專案", "1-1 成員專案連線測試", "1-2 問題集準備",
        "1-3 自動實驗測試",
    } <= page_labels
    gate_dependencies = [
        dependency for dependency in app.config["dependencies"]
        if str(dependency.get("api_name", "")).startswith("workflow_tabs_for_ui")
    ]
    assert len(gate_dependencies) >= 5
    assert all(len(dependency["outputs"]) == 5 for dependency in gate_dependencies)



def test_workflow_gate_requires_project_neo4j_llm_and_embedding() -> None:
    llm = settings.load_service_settings("llm")
    embedding = settings.load_service_settings("embedding")
    assert all(update["interactive"] is False for update in ui.workflow_tabs_for_ui("", False, llm, embedding))
    assert all(update["interactive"] is False for update in ui.workflow_tabs_for_ui("project", False, llm, embedding))
    llm["profiles"]["OpenAI"]["connected"] = True
    assert all(update["interactive"] is False for update in ui.workflow_tabs_for_ui("project", True, llm, embedding))
    embedding["profiles"]["OpenAI"]["connected"] = True
    assert all(update["interactive"] is True for update in ui.workflow_tabs_for_ui("project", True, llm, embedding))


def test_ollama_models_require_current_successful_fetch(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    state = settings.load_service_settings("llm")
    profile = state["profiles"]["Ollama"]
    state["active"] = "Ollama"
    profile["rows"] = [[True, "local-model"]]
    assert settings.service_choices(state) == []
    monkeypatch.setattr(ui, "list_models", lambda *args: ["local-model"])
    fetched = ui.service_action_for_ui(
        "fetch", "Ollama", state, profile["base_url"], profile["api_key"],
        profile["rows"], *profile["models"],
    )
    assert len(fetched) == 13
    assert len(fetched[7:]) == 6
    assert settings.service_choices(fetched[0]) == ["local-model"]
    monkeypatch.setattr(ui, "list_models", lambda *args: (_ for _ in ()).throw(ValueError("offline")))
    failed = ui.service_action_for_ui(
        "fetch", "Ollama", fetched[0], fetched[1], fetched[2], fetched[3]["value"],
        *[field["value"] for field in fetched[7:]],
    )
    assert settings.service_choices(failed[0]) == []

def test_delete_project_refreshes_list_after_server_delete(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    project = ui.create_project("刪除測試")

    deleted = ui.delete_project_for_ui(project["project_id"])

    assert "已刪除" in deleted[2]
    assert all(update["interactive"] is False for update in deleted[3:-1])
    assert deleted[-1] is True
    refreshed = ui.refresh_projects_after_delete_for_ui(deleted[-1])
    assert refreshed["choices"] == []
    assert refreshed["value"] is None
    assert ui.list_projects() == []


def test_failed_delete_does_not_clear_selection(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    result = ui.delete_project_for_ui("missing")

    assert result[2].startswith("❌")
    assert result[-1] is False
    assert "choices" not in ui.refresh_projects_after_delete_for_ui(result[-1])


def test_deleting_selected_project_selects_another_available_project(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    deleted_project = ui.create_project("A 專案")
    remaining_project = ui.create_project("B 專案")

    result = ui.delete_project_for_ui(deleted_project["project_id"])
    refreshed = ui.refresh_projects_after_delete_for_ui(result[-1])

    assert result[0]["value"] == remaining_project["project_id"]
    assert refreshed["choices"] == [("B 專案", remaining_project["project_id"])]
    assert refreshed["value"] == remaining_project["project_id"]


def test_project_archive_ui_callbacks_export_and_import(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    project = ui.create_project("封裝測試")
    pdf = tmp_path / "manual.pdf"
    pdf.write_bytes(b"pdf")
    ui.save_project(project["project_id"], {}, [str(pdf)])

    archive_path, export_status = ui.export_project_for_ui(project["project_id"])
    selected, imported, import_status = ui.import_project_for_ui(archive_path)

    assert Path(archive_path).is_file()
    assert "不包含外部 Neo4j" in export_status
    assert imported["project_id"] == selected["value"]
    assert imported["project_id"] != project["project_id"]
    assert Path(imported["documents"][0]["path"]).read_bytes() == b"pdf"
    assert "已匯入專案" in import_status


def test_delete_button_uses_browser_confirmation() -> None:
    app = build_app()
    button = next(
        component for component in app.config["components"]
        if component.get("props", {}).get("value") == "刪除專案"
    )
    dependency = next(
        item for item in app.config["dependencies"]
        if any(target[0] == button["id"] for target in item.get("targets", []))
    )
    assert "window.confirm" in dependency["js"]
    assert "throw new Error" in dependency["js"]
    assert len(dependency["inputs"]) == 1
    assert any(
        str(item.get("api_name", "")).startswith("refresh_projects_after_delete_for_ui")
        and item.get("trigger_after") == dependency["id"]
        for item in app.config["dependencies"]
    )


def test_build_app_has_automatic_evaluation_page() -> None:
    app = build_app()
    values = [
        component.get("props", {}).get("value")
        for component in app.config["components"]
    ]

    assert "從 PDF 建立題目與答案" in values
    assert values.count("檢索並生成回答") == 3
    assert "進行評測" in values
    assert "匯入題目" in values
    assert "儲存題目" not in values
    assert "匯出題目" in values
    assert "#### 生題設定" in values
    assert "#### 回答模型設定" in values
    assert "#### 評測模型設定" in values
    components = app.config["components"]
    evaluation_judge_model = next(
        component for component in components
        if component.get("props", {}).get("label") == "評測模型"
    )
    experiment_judge_model = next(
        component for component in components
        if component.get("props", {}).get("label") == "全域評測模型"
    )
    assert evaluation_judge_model["props"]["value"] == "gpt-6-luna"
    assert experiment_judge_model["props"]["value"] == "gpt-6-luna"
    labels = [component.get("props", {}).get("label") for component in components]
    values_by_component = [component.get("props", {}).get("value") for component in components]
    import_button_index = values_by_component.index("匯入題目")
    generation_heading_index = values_by_component.index("#### 生題設定")
    questions_heading_index = values_by_component.index("#### 測試題目")
    answer_heading_index = values_by_component.index("#### 回答模型設定")
    judge_heading_index = values_by_component.index("#### 評測模型設定")
    result_heading_index = values_by_component.index("#### 測試結果")
    assert import_button_index < generation_heading_index
    question_table = next(
        component for component in components
        if component.get("props", {}).get("headers")
        == ["題號", "題目", "正確答案", "題目來源（文件與頁碼）", "答案來源（文件與頁碼）"]
    )
    question_table_index = components.index(question_table)
    assert questions_heading_index < question_table_index < answer_heading_index < judge_heading_index < result_heading_index
    assert "最大並行請求數" in labels
    assert any(
        component.get("props", {}).get("label") == "最大並行請求數"
        for component in components[answer_heading_index:judge_heading_index]
    )
    assert any(
        component.get("props", {}).get("label") == "最大並行請求數"
        for component in components[judge_heading_index:result_heading_index]
    )
    assert not any(
        component.get("props", {}).get("label") == "測試最大並行請求數"
        for component in components[answer_heading_index:result_heading_index]
    )
    assert any(
        str(dependency.get("api_name", "")).startswith("save_evaluation_questions_for_ui")
        and any(
            tuple(target) == (question_table["id"], "input")
            for target in dependency.get("targets", [])
        )
        for dependency in app.config["dependencies"]
    )
    assert not any("摘要" in str(component.get("props", {}).get("label", "")) for component in app.config["components"])
    assert not any("選定 PDF" in str(component.get("props", {}).get("headers", [])) for component in app.config["components"])
    labels = [
        component.get("props", {}).get("label") for component in app.config["components"]
    ]
    assert labels.index("0-0 專案設定") < labels.index("0-1 連線設定")
    assert labels.index("0-4 問答測試") < labels.index("0-5 自動問答測試")
    assert labels.index("0-5 自動問答測試") < labels.index("0-6 單一專案實驗")
    assert labels.index("0-6 單一專案實驗") < labels.index("1-0 實驗專案")
    assert labels.index("1-0 實驗專案") < labels.index("1-1 成員專案連線測試")
    assert labels.index("1-1 成員專案連線測試") < labels.index("1-2 問題集準備")
    assert labels.index("1-2 問題集準備") < labels.index("1-3 自動實驗測試")
    components = app.config["components"]
    question_table_index = next(
        index for index, component in enumerate(components)
        if component.get("props", {}).get("headers")
        == ["題號", "題目", "正確答案", "題目來源（文件與頁碼）", "答案來源（文件與頁碼）"]
    )
    answer_availability_index = next(
        index for index, component in enumerate(components)
        if component.get("props", {}).get("value") == "尚未生成測試回答；請先按「檢索並生成回答」。"
    )
    answer_heading_index = next(
        index for index, component in enumerate(components)
        if component.get("props", {}).get("value") == "#### 回答模型設定"
    )
    judge_heading_index = next(
        index for index, component in enumerate(components)
        if component.get("props", {}).get("value") == "#### 評測模型設定"
    )
    result_title_index = next(
        index for index, component in enumerate(components)
        if component.get("props", {}).get("value") == "#### 測試結果"
    )
    result_table_index = next(
        index for index, component in enumerate(components)
        if component.get("props", {}).get("headers")
        == ["編號", "問題", "標準答案", "來源 PDF", "實際答案", "答案判定（勾選=正確）", "評判理由"]
    )
    assert question_table_index < answer_heading_index < answer_availability_index < judge_heading_index < result_title_index < result_table_index
    results_table = components[result_table_index]
    assert results_table["props"]["interactive"] is False
    assert results_table["props"]["datatype"][5] == "bool"
    assert 5 not in results_table["props"]["static_columns"]
    judge_button = next(
        component for component in components
        if component.get("props", {}).get("value") == "進行評測"
    )
    assert judge_button["props"]["interactive"] is False
    assert "evaluation-judge-button" in judge_button["props"]["elem_classes"]
    assert any(
        str(dependency.get("api_name", "")).startswith("update_manual_evaluation_for_ui")
        and any(tuple(target) == (results_table["id"], "input") for target in dependency.get("targets", []))
        for dependency in app.config["dependencies"]
    )


def test_evaluation_button_availability_tracks_saved_answers() -> None:
    no_answers, disabled = ui._evaluation_answer_availability({"questions": [{"number": 1}]})
    has_answers, enabled = ui._evaluation_answer_availability({
        "pending_answers": [{"number": 1, "actual_answer": "A"}],
    })

    assert "尚未生成" in no_answers
    assert disabled["interactive"] is False
    assert "有 1 題回答" in has_answers
    assert enabled["interactive"] is True


def test_build_app_has_manual_neo4j_import_button() -> None:
    app = build_app()

    assert any(
        component.get("props", {}).get("value") == "Embedding 並匯入 Neo4j"
        for component in app.config["components"]
    )



def test_connection_summary_does_not_expose_secrets() -> None:
    status, settings = connection_summary(
        "bolt://localhost:7687", "neo4j", "neo4j", "password", "http://localhost:11434/v1", "key"
    )
    assert status.startswith("✅")
    assert "password" not in settings
    assert "api_key" not in settings


def test_one_click_connection_test_runs_neo4j_llm_and_embedding(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    project = ui.create_project("connection-tests")
    neo4j_calls = []
    service_calls = []
    monkeypatch.setattr(
        ui, "check_neo4j_for_ui",
        lambda *args: (neo4j_calls.append(args) or "✅ Neo4j", True),
    )

    def fake_service_action(action, provider, state, endpoint, api_key, rows, *models):
        service_calls.append((action, provider, endpoint, api_key, rows, models, state["active"]))
        return [f"{action}:{provider}"] * (13 if state["kind"] == "llm" else 8)

    monkeypatch.setattr(ui, "service_action_for_ui", fake_service_action)
    llm_state = {"kind": "llm", "active": "OpenAI", "profiles": {"OpenAI": {}, "Ollama": {}}}
    embedding_state = {"kind": "embedding", "active": "OpenAI", "profiles": {"OpenAI": {}, "Voyage": {}}}

    result = ui.test_all_connections_for_ui(
        project["project_id"], "bolt://neo4j", "", "user", "password",
        "Ollama", llm_state, "http://llm", "llm-key", [[True, "llama"]],
        "Voyage", embedding_state, "http://embedding", "embedding-key", [],
        "model-1", "model-2", "model-3", "model-4", "model-5", "model-6", "voyage-model",
    )

    assert result[0]["value"] == project["neo4j_database"]
    assert result[1:3] == ("✅ Neo4j", True)
    assert len(neo4j_calls) == 1
    assert neo4j_calls[0] == ("bolt://neo4j", project["neo4j_database"], "user", "password")
    assert [(call[0], call[1], call[-1]) for call in service_calls] == [
        ("fetch", "Ollama", "Ollama"), ("test", "Voyage", "Voyage"),
    ]
    assert result[3:16] == ("fetch:Ollama",) * 13
    assert result[16:] == ("test:Voyage",) * 8


def test_persist_env_settings_writes_all_fields(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    status = persist_env_settings(
        "bolt://db", "user", "pass", "http://models", "key",
        "http://embeddings", "embed-key", "build", "embed", "answer",
    )
    content = (tmp_path / ".env").read_text(encoding="utf-8")
    assert status.startswith("✅")
    assert 'NEO4J_PASSWORD="pass"' in content
    assert 'ANSWER_MODEL' not in content
    assert 'BUILD_MODEL' not in content
    assert 'MODEL_SERVICE_PROFILES' not in content




def test_pdf_table_has_checkbox_and_new_project_resets_visible_status() -> None:
    app = build_app()
    components = app.config["components"]
    table = next(
        component for component in components
        if component.get("props", {}).get("headers") == ["選取", "文件", "頁碼範圍", "Chunk 數"]
    )
    assert table["props"]["datatype"] == ["bool", "str", "str", "number"]
    assert table["props"]["interactive"] is True
    create_button = next(
        component for component in components
        if component.get("props", {}).get("value") == "建立新專案"
    )
    create_event = next(
        dependency for dependency in app.config["dependencies"]
        if (create_button["id"], "click") in dependency["targets"]
    )
    assert any(
        dependency.get("trigger_after") == create_event["id"]
        and str(dependency.get("api_name", "")).startswith("reset_new_project_pdf_status_for_ui")
        for dependency in app.config["dependencies"]
    )
    schema_selector = next(
        component for component in components
        if component.get("props", {}).get("label") == "用於規劃 Schema 的 PDF"
    )
    assert schema_selector["type"] == "dataframe"
    assert schema_selector["props"]["headers"] == ["使用", "PDF"]
    assert schema_selector["props"]["datatype"] == ["bool", "str"]
    assert schema_selector["props"]["static_columns"] == [1]
    assert schema_selector["props"]["max_height"] == 300
    assert not any(
        component.get("props", {}).get("label") in {"Schema 規劃範圍", "隨機抽取頁數 N"}
        for component in components
    )

def test_build_app_has_independent_provider_switches_and_ollama_tables() -> None:
    app = build_app()
    components = app.config["components"]
    buttons = {c["props"]["value"]: c for c in components if c["type"] == "button"}
    assert "⚡ 套用 Ollama 本機預設（省 token）" not in buttons
    assert not {"重新整理模型清單", "重新整理 Embedding 模型清單"} & buttons.keys()
    for label, provider_label, fetch_label, test_label, count in [
        ("LLM 模型清單", "模型服務來源", "獲得模型清單", "測試模型服務連線", 6),
        ("Embedding 模型清單", "Embedding 服務來源", "獲得 Embedding 模型清單", "測試 Embedding 服務連線", 1),
    ]:
        table = next(c for c in components if c.get("props", {}).get("label") == label)
        provider = next(c for c in components if c.get("props", {}).get("label") == provider_label)
        expected_providers = [("OpenAI", "OpenAI"), ("Ollama", "Ollama")]
        if provider_label == "Embedding 服務來源":
            expected_providers.append(("Voyage", "Voyage"))
        assert provider["props"]["choices"] == expected_providers
        ollama = provider["props"]["value"] == "Ollama"
        assert buttons[fetch_label]["props"]["visible"] == ollama
        assert buttons[test_label]["props"]["visible"] != ollama
        assert table["props"]["visible"] == ollama
        assert table["props"]["datatype"] == ["bool", "str"]
        assert table["props"]["static_columns"] == [1]
        fetch = next(d for d in app.config["dependencies"] if (buttons[fetch_label]["id"], "click") in d["targets"])
        selection = next(d for d in app.config["dependencies"] if (table["id"], "input") in d["targets"])
        switch = next(d for d in app.config["dependencies"] if (provider["id"], "input") in d["targets"])
        assert fetch["outputs"] == selection["outputs"] == switch["outputs"]
        assert len(fetch["outputs"]) == count + 7


def test_project_ui_create_save_and_load(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _, created, status = ui.create_project_for_ui("手冊專案")
    assert status.startswith("✅")
    document = tmp_path / "manual.pdf"
    document.write_bytes(b"pdf")
    values = [
        created["project_id"],
        [{"file_name": "manual.pdf", "file_path": str(document)}],
        [TextChunk(1, "內容", (1,), "manual.pdf")], {
            "run_id": "run-1", "document": "manual.pdf", "neo4j_imported": True,
            "entities": [{"name": "設備", "type": "DEVICE", "description": "說明",
                          "source_chunk_numbers": [1], "source_pages": [1]}],
            "relationships": [{"source": "設備", "type": "USES", "target": "零件",
                               "description": "使用", "source_chunk_numbers": [1],
                               "source_pages": [1]}],
        },
        "bolt://db", "neo4j", "user", "pass", "http://models", "key",
        "build", "embed", "answer", 1200, 100, 0.2,
        "詳細", 2, "extract", 2,
        "關聯擴展檢索", 6, '{"entity_types": []}',
    ]
    saved, save_status = ui.save_project_for_ui(*values)
    loaded = ui.load_project_for_ui(created["project_id"])
    assert save_status.startswith("✅")
    assert Path(saved["documents"][0]["path"]).read_bytes() == b"pdf"
    assert loaded[0]["project_id"] == created["project_id"]
    assert loaded[11:13] == (1200, 100)
    assert loaded[22][0].text == "內容"
    stored_project = ui.load_project(created["project_id"])
    assert "model_endpoint" not in stored_project["settings"]
    assert "api_key" not in stored_project["settings"]
    assert stored_project["graph_state"]["entities"][0]["name"] == "設備"
    assert stored_project["graph_state"]["relationships"][0]["type"] == "USES"
    assert loaded[30][0][:2] == ["設備", "DEVICE"]
    assert loaded[31][0][:3] == ["設備", "USES", "零件"]
    assert "1 個實體、1 筆關係" in loaded[32]
    assert "已匯入 Neo4j" in loaded[33]


def test_load_project_ignores_legacy_model_credentials(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ui, "load_env", lambda: {
        "MODEL_OPENAI_API_BASE": "https://current.example/v1",
        "MODEL_OPENAI_API_KEY": "current-key",
        "MODEL_OLLAMA_API_BASE": "http://localhost:11434/v1",
        "MODEL_OLLAMA_API_KEY": "",
        "EMBEDDING_OPENAI_API_BASE": "https://current.example/v1",
        "EMBEDDING_OPENAI_API_KEY": "current-key",
        "EMBEDDING_OLLAMA_API_BASE": "http://localhost:11434/v1",
        "EMBEDDING_OLLAMA_API_KEY": "",
        "EMBEDDING_VOYAGE_API_BASE": "https://api.voyageai.com/v1",
        "EMBEDDING_VOYAGE_API_KEY": "",
        "NEO4J_URI": "bolt://db",
        "NEO4J_USERNAME": "user",
        "NEO4J_PASSWORD": "password",
    })
    _, created, _ = ui.create_project_for_ui("Legacy")
    project_path = tmp_path / "data" / "projects" / created["project_id"] / "project.json"
    ui.write_json(project_path, {
        **created,
        "documents": [],
        "documents_meta": [],
        "settings": {
            "model_endpoint": "https://legacy.example/v1",
            "api_key": "legacy-key",
        },
    })

    loaded = ui.load_project_for_ui(created["project_id"])

    migrated = ui.load_project(created["project_id"])
    assert loaded[6:8] == ("https://current.example/v1", "current-key")
    assert "model_endpoint" not in migrated["settings"]
    assert "api_key" not in migrated["settings"]


def test_project_answer_appends_history(monkeypatch) -> None:
    monkeypatch.setattr(ui, "answer_question_for_ui", lambda *args: ("✅ 完成", "答案", [["來源"]]))
    monkeypatch.setattr(ui, "load_project", lambda project_id: {"graph_state": {"document": "manual.pdf"}})
    captured = {}
    def fake_append(project_id, record):
        captured.update(record)
        return {"questions": [record]}
    monkeypatch.setattr(ui, "append_question", fake_append)
    result = ui.answer_question_for_project_ui(
        "project",
        "endpoint", "key", "embedding-endpoint", "embedding-key",
        "bolt", "neo4j", "user", "pass",
        "answer-model", "問題", "關聯擴展檢索", 8, None, True,
    )
    assert captured["document"] == "manual.pdf"
    assert captured["sources"] == [["來源"]]
    assert captured["question"] == "問題"
    assert captured["answer_model"] == "answer-model"
    assert captured["retrieval_mode"] == "關聯擴展檢索"
    assert captured["top_k"] == 8
    assert result[0].endswith("✅ 問答紀錄已加入目前專案。")


def test_generate_evaluation_for_ui_saves_questions(monkeypatch) -> None:
    questions = [{"number": 1, "question": "Q", "expected_answer": "A", "source_pages": [1]}]
    generated = {}

    def fake_generate(_endpoint, _key, _model, chunks, _count, _excluded=None, focus_page=None):
        generated["chunks"] = chunks
        generated["focus_page"] = focus_page
        return questions

    sampled_chunks = [TextChunk(7, "sampled page with context", (7,), "manual.pdf")]
    monkeypatch.setattr(ui, "sample_random_page_context", lambda _chunks: (7, sampled_chunks))
    monkeypatch.setattr(ui, "expand_page_context_until_complete", lambda *args: sampled_chunks)
    monkeypatch.setattr(ui, "generate_evaluation_questions", fake_generate)
    monkeypatch.setattr(ui, "load_project", lambda *_: {"evaluation": {}})
    captured = {}
    real_executor = ui.ThreadPoolExecutor

    def recording_executor(max_workers):
        captured["generation_workers"] = max_workers
        return real_executor(max_workers=max_workers)

    monkeypatch.setattr(ui, "ThreadPoolExecutor", recording_executor)
    monkeypatch.setattr(ui, "save_project", lambda project_id, payload: captured.update(payload) or {})

    status, rows, state, results = ui.generate_evaluation_for_ui(
        "project", "endpoint", "key", "generation-model", "test-model", 1, "關聯擴展檢索", 8,
        [TextChunk(1, "text", (1,))], True, 5, True, False,
    )

    assert status.startswith("✅")
    assert rows[0][1:3] == ["Q", "A"]
    assert state["questions"] == questions
    assert state["preferences"]["generation_model"] == "generation-model"
    assert captured["evaluation"]["questions"] == questions
    assert state["preferences"]["allow_parallel_generation"] is True
    assert captured["generation_workers"] == 1
    assert state["preferences"]["test_max_concurrent_requests"] == 5
    assert state["preferences"]["expand_evidence"] is False
    assert generated["chunks"] == sampled_chunks
    assert generated["focus_page"] == 7
    assert results == []


def test_generate_evaluation_distributes_questions_across_documents(monkeypatch) -> None:
    requested_documents = []
    exclusions_by_document = {"a.pdf": [], "b.pdf": []}
    captured = {}
    real_executor = ui.ThreadPoolExecutor

    def recording_executor(max_workers):
        captured["generation_workers"] = max_workers
        return real_executor(max_workers=max_workers)

    def fake_generate(_endpoint, _key, _model, chunks, count, _excluded=None, _focus_page=None):
        document = chunks[0].document
        exclusions_by_document[document].append(list(_excluded or []))
        requested_documents.append(document)
        sequence = requested_documents.count(document)
        topics = {
            ("a.pdf", 1): "network address lookup",
            ("a.pdf", 2): "maintenance interval",
            ("b.pdf", 1): "control panel reset",
            ("b.pdf", 2): "warning light meaning",
        }
        return [{
            "number": 1,
            "question": f"{document} {topics[(document, sequence)]}?",
            "expected_answer": f"answer: {topics[(document, sequence)]}",
            "source_pages": [1],
            "source_chunk_numbers": [sequence if document == "a.pdf" else sequence + 2],
            "document": chunks[0].document,
        }]

    monkeypatch.setattr(ui, "expand_page_context_until_complete", lambda *args: args[3])
    monkeypatch.setattr(ui, "generate_evaluation_questions", fake_generate)
    monkeypatch.setattr(ui, "load_project", lambda *_: {"evaluation": {}})
    monkeypatch.setattr(ui, "ThreadPoolExecutor", recording_executor)
    monkeypatch.setattr(ui, "save_project", lambda *args: {})

    status, _rows, state, _results = ui.generate_evaluation_for_ui(
        "project", "endpoint", "key", "generation-model", "test-model",
        2, "基本檢索", 8,
        [TextChunk(1, "a", (1,), "a.pdf"), TextChunk(2, "b", (1,), "b.pdf")],
        True, 3,
    )

    assert status.startswith("✅")
    assert "2 份 PDF 各建立 2 道題目，共 4 道" in status
    assert requested_documents.count("a.pdf") == 2
    assert requested_documents.count("b.pdf") == 2
    assert [item["document"] for item in state["questions"]] == [
        "a.pdf", "a.pdf", "b.pdf", "b.pdf",
    ]
    assert captured["generation_workers"] == 2
    assert "a.pdf network address lookup?" in [
        entry["question"] for entry in exclusions_by_document["a.pdf"][1]
    ]
    assert "b.pdf control panel reset?" in [
        entry["question"] for entry in exclusions_by_document["b.pdf"][1]
    ]


def test_generate_evaluation_refills_duplicate_questions(monkeypatch) -> None:
    calls = {"count": 0}

    def fake_generate(_endpoint, _key, _model, chunks, _count, excluded=None, _focus_page=None):
        calls["count"] += 1
        question = (
            "馬達異音時如何檢查？" if calls["count"] == 1
            else "馬達出現異常聲音要怎麼檢查？" if calls["count"] == 2
            else "煞車油多久需要更換一次？"
        )
        if excluded:
            assert any(item["question"] == "馬達異音時如何檢查？" for item in excluded)
        return [{
            "number": 1,
            "question": question,
            "expected_answer": "檢查馬達固定螺絲" if calls["count"] < 3 else "每兩年更換",
            "source_pages": [1],
            "source_chunk_numbers": [1],
            "document": chunks[0].document,
        }]

    monkeypatch.setattr(ui, "expand_page_context_until_complete", lambda *args: args[3])
    monkeypatch.setattr(ui, "generate_evaluation_questions", fake_generate)
    monkeypatch.setattr(ui, "load_project", lambda *_: {"evaluation": {}})
    monkeypatch.setattr(ui, "save_project", lambda *args: {})

    status, _rows, state, _results = ui.generate_evaluation_for_ui(
        "project",
        "endpoint",
        "key",
        "generation-model",
        "test-model",
        2,
        "基本檢索",
        8,
        [TextChunk(1, "內容", (1,), "manual.pdf")],
        False,
        3,
    )

    assert status.startswith("✅")
    assert [item["question"] for item in state["questions"]] == [
        "馬達異音時如何檢查？", "煞車油多久需要更換一次？",
    ]
    assert calls["count"] == 3


def test_parallel_generation_refills_duplicates_across_documents(monkeypatch) -> None:
    first_requests = Barrier(2)
    call_counts = {"a.pdf": 0, "b.pdf": 0}
    call_lock = Lock()

    def fake_generate(_endpoint, _key, _model, chunks, _count, excluded=None, _focus_page=None):
        document = chunks[0].document
        with call_lock:
            call_counts[document] += 1
            attempt = call_counts[document]
        if attempt == 1:
            first_requests.wait()
            question = "跨文件重複問題？"
        else:
            assert any(item["question"] == "跨文件重複問題？" for item in (excluded or []))
            question = f"{document} {'電源檢查流程' if document == 'a.pdf' else '安全警示燈含義'}？"
        return [{
            "number": 1,
            "question": question,
            "expected_answer": "答案",
            "source_pages": [1],
            "source_chunk_numbers": [1 if document == "a.pdf" else 2],
            "document": document,
        }]

    monkeypatch.setattr(ui, "expand_page_context_until_complete", lambda *args: args[3])
    monkeypatch.setattr(ui, "generate_evaluation_questions", fake_generate)
    monkeypatch.setattr(ui, "load_project", lambda *_: {"evaluation": {}})
    monkeypatch.setattr(ui, "save_project", lambda *_: {})

    status, _rows, state, _results = ui.generate_evaluation_for_ui(
        "project", "endpoint", "key", "generation-model", "test-model",
        1, "基本檢索", 8,
        [TextChunk(1, "a", (1,), "a.pdf"), TextChunk(2, "b", (1,), "b.pdf")],
        True, 3,
    )

    assert status.startswith("✅")
    questions = [item["question"] for item in state["questions"]]
    assert questions.count("跨文件重複問題？") == 1
    assert len(questions) == len(set(questions)) == 2
    assert sum(call_counts.values()) == 3


def test_generate_evaluation_processes_all_documents_sequentially_when_parallel_disabled(
    monkeypatch,
) -> None:
    requested_documents = []

    def fake_generate(_endpoint, _key, _model, chunks, _count, *_args):
        document = chunks[0].document
        requested_documents.append(document)
        return [{
            "number": 1,
            "question": f"{document} {'網路位址查詢方式' if document == 'a.pdf' else '安全鎖解除步驟'}？",
            "expected_answer": f"answer for {document}",
            "source_pages": [1],
            "source_chunk_numbers": [1 if document == "a.pdf" else 2],
            "document": document,
        }]

    monkeypatch.setattr(ui, "expand_page_context_until_complete", lambda *args: args[3])
    monkeypatch.setattr(ui, "generate_evaluation_questions", fake_generate)
    monkeypatch.setattr(
        ui, "ThreadPoolExecutor",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("未允許並行時不應建立執行緒池")
        ),
    )
    monkeypatch.setattr(ui, "load_project", lambda *_: {"evaluation": {}})
    monkeypatch.setattr(ui, "save_project", lambda *_: {})

    status, _rows, state, _results = ui.generate_evaluation_for_ui(
        "project", "endpoint", "key", "generation-model", "test-model",
        1, "基本檢索", 8,
        [TextChunk(1, "a", (1,), "a.pdf"), TextChunk(2, "b", (1,), "b.pdf")],
        False, 3,
    )

    assert status.startswith("✅")
    assert requested_documents == ["a.pdf", "b.pdf"]
    assert state["preferences"]["allow_parallel_generation"] is False


def test_retrieval_rank_uses_document_page_even_when_chunk_differs() -> None:
    question = {
        "document": "manual.pdf",
        "source_pages": [1],
        "source_chunk_numbers": [42],
    }
    rows = [
        ["原文", "錯誤文件的同頁證據", "", "0.9", "other.pdf：1", "manual.pdf：42", "other.pdf、manual.pdf"],
        ["原文", "同文件同頁但不同 chunk", "", "0.8", "manual.pdf：1", "manual.pdf：9", "manual.pdf"],
    ]

    rank = ui._retrieval_rank(question, rows)

    assert rank == 2
    assert 1 / rank == 0.5


def test_retrieval_rank_does_not_fall_back_to_chunk_without_expected_pages() -> None:
    question = {"document": "manual.pdf", "source_chunk_numbers": [42]}
    rows = [["原文", "相同 chunk", "", "0.9", "manual.pdf：1", "manual.pdf：42", "manual.pdf"]]

    assert ui._retrieval_rank(question, rows) is None


def test_retrieval_rank_counts_unique_document_pages() -> None:
    question = {"document": "manual.pdf", "source_pages": [1]}
    rows = [
        ["原文", "第 2 頁證據 A", "", "0.9", "manual.pdf：2", "manual.pdf：4", "manual.pdf"],
        ["原文", "第 2 頁證據 B", "", "0.8", "manual.pdf：2", "manual.pdf：5", "manual.pdf"],
        ["原文", "第 1 頁證據", "", "0.7", "manual.pdf：1", "manual.pdf：6", "manual.pdf"],
    ]

    assert ui._retrieval_rank(question, rows) == 2


def test_retrieval_rank_maps_legacy_source_name_to_document_id() -> None:
    question = {"document": "manual.pdf", "answer_source_pages": [1]}
    rows = [["原文", "同文件同頁", "", "0.9", "manual.pdf：1", "manual.pdf：7", "manual.pdf"]]

    assert ui._retrieval_rank(question, rows, {"manual.pdf": "stable-document-id"}) == 1


def test_run_evaluation_for_ui_judges_and_saves(monkeypatch) -> None:
    monkeypatch.setattr(
        ui, "answer_question_for_ui",
        lambda *args: ("✅ 完成", "實際答案", [[
            "原文", "證據", "official-hybrid", "0.9",
            "manual.pdf：1", "manual.pdf：9", "manual.pdf",
        ]]),
    )
    real_executor = ui.ThreadPoolExecutor

    def recording_executor(max_workers):
        captured["test_workers"] = max_workers
        return real_executor(max_workers=max_workers)

    monkeypatch.setattr(ui, "judge_evaluation_answer", lambda *args: {"passed": True, "reason": "正確"})
    captured = {}
    monkeypatch.setattr(ui, "ThreadPoolExecutor", recording_executor)
    monkeypatch.setattr(ui, "save_project", lambda project_id, payload: captured.update(payload) or {})
    evaluation = {
        "questions": [{
            "number": 1, "question": "Q", "expected_answer": "A",
            "source_pages": [1], "document": "manual.pdf",
        }],
    }

    status, rows, updated = ui.run_evaluation_for_ui(
        "project", "endpoint", "key", "embed-endpoint", "embed-key",
        "bolt", "neo4j", "user", "pass",
        "model", "關聯擴展檢索", 8, evaluation, 2,
    )

    assert "總共答對 1 題 / 1 題" in status
    assert "manual.pdf：答案正確 1 / 1" in status
    assert "答錯：0 題" in status
    assert "答案正確率：100.0%" in status
    assert "Recall@5：100.0%" in status
    assert "Recall@10：100.0%" in status
    assert "MRR：1.000" in status
    assert rows[0][3:] == ["manual.pdf", "實際答案", True, "正確"]
    assert captured["test_workers"] == 2
    assert updated["results"][0]["passed"] is True
    assert captured["evaluation"] == updated


def test_generate_answers_defers_display_and_evaluation(monkeypatch) -> None:
    captured = {}
    answer_calls = []
    monkeypatch.setattr(
        ui, "answer_question_for_ui",
        lambda *args: answer_calls.append(args) or ("✅ 完成", "隱藏的回答", []),
    )
    monkeypatch.setattr(ui, "save_project", lambda project_id, payload: captured.update(payload) or {})
    evaluation = {"questions": [{"number": 1, "question": "Q", "expected_answer": "A"}]}

    status, pending, rows = ui.generate_evaluation_answers_for_ui(
        "project", "endpoint", "key", "embed", "embed-key", "bolt", "database",
        "user", "pass", "answer-model", "基本向量檢索", 5, evaluation,
    )

    assert status.startswith("✅ 已生成並保存 1 題回答")
    assert rows == []
    assert pending["results"] == []
    assert pending["pending_answers"][0]["actual_answer"] == "隱藏的回答"
    assert len(answer_calls) == 1
    assert captured["evaluation"] == pending


def test_evaluation_judges_saved_answers_without_generating_again(monkeypatch) -> None:
    captured = {}
    real_executor = ui.ThreadPoolExecutor

    def recording_executor(max_workers):
        captured["judge_workers"] = max_workers
        return real_executor(max_workers=max_workers)

    monkeypatch.setattr(ui, "answer_question_for_ui", lambda *_args: (_ for _ in ()).throw(AssertionError()))
    monkeypatch.setattr(ui, "judge_evaluation_answer", lambda *args: {"passed": True, "reason": "正確"})
    monkeypatch.setattr(ui, "save_project", lambda project_id, payload: captured.update(payload) or {})
    monkeypatch.setattr(ui, "ThreadPoolExecutor", recording_executor)
    first_answer = {
            "number": 1, "question": "Q", "expected_answer": "A",
            "actual_answer": "回答內容", "answer_status": "✅ 完成",
            "retrieval_rank": 1, "recall_at_5": True, "recall_at_10": True,
            "reciprocal_rank": 1.0,
    }
    evaluation = {
        "questions": [{"number": 1, "question": "Q", "expected_answer": "A"}],
        "pending_answers": [first_answer, {**first_answer, "number": 2, "question": "Q2"}],
    }

    status, rows, updated = ui.evaluate_generated_answers_for_ui(
        "project", "judge-endpoint", "judge-key", "judge-model", evaluation,
        max_concurrent_requests=2,
    )

    assert "總共答對 2 題 / 2 題" in status
    assert rows[0][4] == "回答內容"
    assert rows[0][5] is True
    assert updated["results"][0]["reason"] == "正確"
    assert captured["evaluation"] == updated
    assert captured["judge_workers"] == 2


def test_manual_evaluation_edit_updates_reason_and_accuracy(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(ui, "save_project", lambda project_id, payload: captured.update(payload) or {})
    evaluation = {"results": [
        {"number": 1, "question": "Q1", "expected_answer": "A1", "passed": True, "reason": "正確"},
        {"number": 2, "question": "Q2", "expected_answer": "A2", "passed": True, "reason": "正確"},
    ]}
    rows = ui._evaluation_result_rows(evaluation["results"])
    rows[1][5] = False

    status, updated_rows, updated = ui.update_manual_evaluation_for_ui("project", rows, evaluation)

    assert "總共答對 1 題 / 2 題" in status
    assert "答案正確率：50.0%" in status
    assert "人工評判變更 1 筆" in status
    assert updated["results"][0]["reason"] == "正確"
    assert updated["results"][1]["passed"] is False
    assert updated["results"][1]["reason"] == "人工評判"
    assert updated_rows[1][5] is False
    assert captured["evaluation"] == updated


def test_run_evaluation_for_ui_forwards_credentials_to_answer_question_for_ui(monkeypatch) -> None:
    captured = {}

    def fake_answer_question_for_ui(*args):
        captured["args"] = args
        return "✅ 完成", "實際答案", []

    monkeypatch.setattr(ui, "answer_question_for_ui", fake_answer_question_for_ui)
    monkeypatch.setattr(ui, "judge_evaluation_answer", lambda *args: {"passed": True, "reason": "正確"})
    monkeypatch.setattr(ui, "save_project", lambda project_id, payload: {})
    evaluation = {
        "questions": [{
            "number": 1, "question": "Q", "expected_answer": "A",
            "source_pages": [1], "document": "manual.pdf",
        }],
    }

    ui.run_evaluation_for_ui(
        "project", "endpoint", "key", "embed-endpoint", "embed-key",
        "bolt", "neo4j", "user", "pass",
        "model", "關聯擴展檢索", 8, evaluation,
    )

    assert captured["args"] == (
        "endpoint", "key", "embed-endpoint", "embed-key",
        "bolt", "neo4j", "user", "pass",
        "model", "Q", "關聯擴展檢索", 10, False, False,
    )


def test_run_evaluation_uses_separate_judge_model_and_records_both(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(
        ui, "answer_question_for_ui",
        lambda *args: ("✅ 完成", "回答內容", []),
    )
    monkeypatch.setattr(
        ui, "judge_evaluation_answer",
        lambda *args: captured.update(judge_args=args) or {"passed": True, "reason": "正確"},
    )
    monkeypatch.setattr(ui, "save_project", lambda _project_id, payload: captured.update(payload) or {})
    evaluation = {"questions": [{"number": 1, "question": "Q", "expected_answer": "A"}]}

    _status, _rows, updated = ui.run_evaluation_for_ui(
        "project", "answer-endpoint", "answer-key", "embed", "embed-key",
        "bolt", "database", "user", "pass", "answer-model", "混合檢索", 5,
        evaluation, 1, False, False,
        "judge-endpoint", "judge-key", "judge-model",
    )

    assert captured["judge_args"][:3] == ("judge-endpoint", "judge-key", "judge-model")
    assert updated["results"][0]["answer_model"] == "answer-model"
    assert updated["results"][0]["judge_model"] == "judge-model"


def test_run_evaluation_requires_selected_judge_model(monkeypatch) -> None:
    result = ui.run_evaluation_for_ui(
        "project", "answer-endpoint", "answer-key", "embed", "embed-key",
        "bolt", "database", "user", "pass", "answer-model", "混合檢索", 5,
        {"questions": [{"number": 1, "question": "Q", "expected_answer": "A"}]},
        1, False, False, "judge-endpoint", "judge-key", None,
    )

    assert result[0] == "❌ 請選擇回答模型與評測模型。"


def test_edit_questions_auto_save_and_clear_results(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(ui, "save_project", lambda project_id, payload: captured.update(payload) or {})

    status, state, results = ui.save_evaluation_questions_for_ui(
        "project", [[9, "修改後問題", "修改後答案", "2, 3"]],
        {"results": [{"passed": True}]},
    )

    assert status.startswith("✅ 已自動儲存")
    assert state["dirty"] is False
    assert state["results"] == []
    assert state["questions"][0]["number"] == 9
    assert state["questions"][0]["source_pages"] == [2, 3]
    assert captured["evaluation"] == state
    assert results == []


def test_save_and_export_edited_questions(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    project = ui.create_project("project")
    rows = [[1, "問題", "答案", "1", "2, 4", "manual.pdf"]]

    status, state, _ = ui.save_evaluation_questions_for_ui(
        project["project_id"], rows, {"dirty": True}
    )
    export_status, export_path = ui.export_evaluation_questions_for_ui(
        project["project_id"], rows
    )

    assert status.startswith("✅")
    assert state["dirty"] is False
    assert state["questions"][0]["question_source_pages"] == [1]
    assert state["questions"][0]["answer_source_pages"] == [2, 4]
    assert export_status.startswith("✅")
    payload = json.loads(Path(export_path).read_text(encoding="utf-8"))
    assert payload["questions"][0] == {
        "number": 1,
        "question": "問題",
        "expected_answer": "答案",
        "question_sources": [{"document_id": "", "document_name": "manual.pdf", "pages": [1]}],
        "answer_sources": [{"document_id": "", "document_name": "manual.pdf", "pages": [2, 4]}],
    }


def test_editing_source_cells_preserves_document_ids(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    project = ui.create_project("preserve-source-ids")
    previous = [{
        "number": 1, "question": "Q", "expected_answer": "A",
        "question_sources": [{"document_id": "doc-1", "document_name": "manual.pdf", "pages": [1]}],
        "answer_sources": [{"document_id": "doc-2", "document_name": "parts.pdf", "pages": [2]}],
    }]
    visible_rows = [[1, "Q", "A", "manual.pdf：1", "parts.pdf：2"]]

    status, updated, _ = ui.save_evaluation_questions_for_ui(
        project["project_id"], visible_rows, {"questions": previous},
    )

    assert status.startswith("✅")
    assert updated["questions"][0]["question_sources"][0]["document_id"] == "doc-1"
    assert updated["questions"][0]["answer_sources"][0]["document_id"] == "doc-2"
    export_status, export_path = ui.export_evaluation_questions_for_ui(
        project["project_id"], visible_rows,
    )
    exported = json.loads(Path(export_path).read_text(encoding="utf-8"))
    assert export_status.startswith("✅")
    assert exported["questions"][0]["question_sources"][0]["document_id"] == "doc-1"
    assert exported["questions"][0]["answer_sources"][0]["document_id"] == "doc-2"


def test_import_questions_supports_json_and_csv(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    project = ui.create_project("import-test")
    json_file = tmp_path / "questions.json"
    json_file.write_text(json.dumps({"questions": [
        {
            "question": "JSON Q", "expected_answer": "JSON A",
            "question_source_pages": [1, 2], "answer_source_pages": [2, 3],
            "document": "json.pdf",
        }
    ]}), encoding="utf-8")
    csv_file = tmp_path / "questions.csv"
    csv_file.write_text(
        "number,question,expected_answer,question_source_pages,answer_source_pages,document\n"
        '1,CSV Q,CSV A,2,"3,4",csv.pdf\n',
        encoding="utf-8",
    )

    json_result = ui.import_evaluation_questions_for_ui(project["project_id"], str(json_file), {})
    csv_result = ui.import_evaluation_questions_for_ui(project["project_id"], str(csv_file), {})

    assert json_result[2]["questions"][0]["question"] == "JSON Q"
    assert json_result[2]["questions"][0]["question_source_pages"] == [1, 2]
    assert json_result[2]["questions"][0]["answer_source_pages"] == [2, 3]
    assert csv_result[2]["questions"][0]["expected_answer"] == "CSV A"
    assert csv_result[2]["questions"][0]["answer_source_pages"] == [3, 4]
    assert json_result[2]["dirty"] is False
    assert csv_result[2]["dirty"] is False
    assert "已匯入並自動儲存" in csv_result[0]


def test_import_without_file_preserves_existing_questions_and_results(monkeypatch) -> None:
    evaluation = {
        "questions": [{
            "number": 1, "question": "現有題目", "expected_answer": "現有答案",
            "source_pages": [4], "document": "manual.pdf",
        }],
        "results": [{
            "number": 1, "question": "現有題目", "expected_answer": "現有答案",
            "document": "manual.pdf", "actual_answer": "回答", "passed": True,
            "reason": "正確",
        }],
    }
    monkeypatch.setattr(ui, "save_project", lambda *_: pytest.fail("空匯入不應保存"))

    status, question_rows, updated, result_rows = ui.import_evaluation_questions_for_ui(
        "project", None, evaluation
    )

    assert status == "❌ 請選擇 JSON 或 CSV 題目檔。"
    assert question_rows == [[1, "現有題目", "現有答案", "manual.pdf：4", "manual.pdf：4"]]
    assert updated == evaluation
    assert result_rows[0][1:4] == ["現有題目", "現有答案", "manual.pdf"]


def test_import_empty_file_preserves_existing_questions(tmp_path, monkeypatch) -> None:
    empty_file = tmp_path / "empty.json"
    empty_file.write_text("{\"questions\": []}", encoding="utf-8")
    evaluation = {
        "questions": [{
            "number": 1, "question": "現有題目", "expected_answer": "現有答案",
            "source_pages": [], "document": "",
        }],
        "results": [],
    }
    monkeypatch.setattr(ui, "save_project", lambda *_: pytest.fail("空匯入不應保存"))

    status, question_rows, updated, _result_rows = ui.import_evaluation_questions_for_ui(
        "project", str(empty_file), evaluation
    )

    assert status.startswith("❌ 匯入失敗：題目不可為空")
    assert question_rows == [[1, "現有題目", "現有答案", "", ""]]
    assert updated == evaluation


def test_import_experiment_questions_supports_question_set_fields(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    project = ui.create_project("experiment-import")
    question_file = tmp_path / "experiment.json"
    question_file.write_text(json.dumps({"questions": [{
        "number": 3,
        "question": "問題？",
        "expected_answer": "答案",
        "question_source_pages": [1, 2],
        "answer_source_pages": [3, 4],
        "document": "manual.pdf",
    }]}), encoding="utf-8")

    ui.save_project(project["project_id"], {"experiment": {
        "groups": [{"name": "existing"}], "results": [{"passed": True}],
        "summary_rows": [["existing", 1]], "detail_rows": [["existing", 1]],
    }})
    status, rows, questions, results, summaries, details, experiment_status, pending = ui.import_experiment_questions_for_ui(
        str(question_file), project_id=project["project_id"]
    )

    assert status.startswith("✅ 已匯入 1 道")
    assert rows == [[3, "問題？", "答案", "manual.pdf：1, 2", "manual.pdf：3, 4"]]
    assert questions[0]["answer_source_pages"] == [3, 4]
    assert results == summaries == details == []
    assert experiment_status == status
    persisted = ui.load_project(project["project_id"])["experiment"]
    assert persisted["questions"] == questions
    assert persisted["groups"] == [{"name": "existing"}]
    assert persisted["results"] == []
    assert persisted["pending_answers"] == []
    assert pending == []


def test_import_and_roundtrip_cross_document_provenance(tmp_path) -> None:
    question_file = tmp_path / "cross-document.json"
    sources = [
        {"document_id": "a" * 36, "document_name": "part-a.pdf", "pages": [2]},
        {"document_id": "b" * 36, "document_name": "part-b.pdf", "pages": [7, 8]},
    ]
    question_file.write_text(json.dumps({"questions": [{
        "number": 1, "question": "跨文件問題？", "expected_answer": "跨文件答案",
        "question_sources": sources, "answer_sources": [sources[1]],
    }]}), encoding="utf-8")

    questions = ui._questions_from_file(str(question_file))
    rows = ui._evaluation_question_rows(questions)
    assert rows[0][3] == "part-a.pdf：2\npart-b.pdf：7, 8"
    assert rows[0][4] == "part-b.pdf：7, 8"
    reparsed = ui._preserve_source_document_ids(ui._questions_from_rows(rows), questions)

    assert reparsed[0]["question_sources"] == sources
    assert reparsed[0]["answer_sources"] == [sources[1]]
    ranked = ui._retrieval_rank(reparsed[0], [
        ["原文", "錯誤文件同頁", "", "0.9", "other.pdf：7", "other.pdf：2", "other.pdf"],
        ["原文", "命中答案來源", "", "0.8", "part-b.pdf：7, 8", "part-b.pdf：9", "part-b.pdf"],
    ])
    assert ranked == 2
    renamed_rank = ui._retrieval_rank(
        reparsed[0], [["原文", "相同內容的新檔名", "", "0.9", "renamed.pdf：7, 8", "renamed.pdf：9", "renamed.pdf"]],
        {"renamed.pdf": sources[1]["document_id"]},
    )
    assert renamed_rank == 1
    assert ui._retrieval_rank(
        reparsed[0], [["原文", "不同文件但同名", "", "0.9", "part-b.pdf：7, 8", "part-b.pdf：9", "part-b.pdf"]],
        {"part-b.pdf": "different-document-id"},
    ) is None

    csv_file = tmp_path / "cross-document.csv"
    with csv_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "number", "question", "expected_answer", "question_sources", "answer_sources",
        ])
        writer.writeheader()
        writer.writerow({
            "number": 1, "question": "跨文件問題？", "expected_answer": "跨文件答案",
            "question_sources": json.dumps(sources, ensure_ascii=False),
            "answer_sources": json.dumps([sources[1]], ensure_ascii=False),
        })
    assert ui._questions_from_file(str(csv_file))[0]["answer_sources"] == [sources[1]]


def test_experiment_import_without_file_preserves_current_question_set() -> None:
    current = [{
        "number": 1, "question": "保留題目", "expected_answer": "答案",
        "question_source_pages": [1], "answer_source_pages": [2],
        "document": "manual.pdf",
    }]

    status, rows, questions, results, summaries, details, _experiment_status, pending = ui.import_experiment_questions_for_ui(None, current)

    assert status == "❌ 請選擇 JSON 或 CSV 題目集。"
    assert rows == [[1, "保留題目", "答案", "manual.pdf：1", "manual.pdf：2"]]
    assert questions == current
    assert results == summaries == details == []
    assert pending == []


def test_add_experiment_group_for_ui_stores_selected_settings() -> None:
    status, rows, groups = ui.add_experiment_group_for_ui(
        "混合擴展", "model-a", "混合檢索", 12, True, True, [],
    )

    assert status == "✅ 已加入「混合擴展」。"
    assert rows == [["混合擴展", "model-a", "混合檢索", 12, "Reranker", "證據擴展"]]
    assert groups[0] == {
        "name": "混合擴展", "answer_model": "model-a", "retrieval_mode": "混合檢索",
        "top_k": 12, "use_reranker": True, "expand_evidence": True,
    }


def test_inline_groups_round_trip_reranker_and_expansion_versions() -> None:
    groups = [{
        "name": "比較組", "answer_model": "model-a",
        "retrieval_mode": "混合檢索", "top_k": 8,
        "reranker_mode": "LLM Reranker",
        "evidence_expansion_mode": "證據擴展 V2",
        "use_reranker": True, "expand_evidence": True,
    }]

    restored = ui._groups_from_inline_values(tuple(ui._inline_group_values(groups)))

    assert restored == [groups[0]]


def test_inline_experiment_groups_autosave_and_reload(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    project = ui.create_project("experiment-autosave")
    groups = [{
        "name": "向量組", "answer_model": "gpt-4.1-mini",
        "retrieval_mode": "基本向量檢索", "top_k": 5,
        "use_reranker": False, "expand_evidence": True,
    }]
    questions = [{"number": 1, "question": "Q", "expected_answer": "A"}]

    status, saved_groups, result_status = ui.save_inline_experiment_groups_for_ui(
        project["project_id"], questions, 3, [], *ui._inline_group_values(groups),
    )
    ui.save_project(project["project_id"], {"experiment": {
        **ui.load_project(project["project_id"])["experiment"],
        "results": [{"passed": True}], "summary_rows": [["向量組", 1]],
        "detail_rows": [["向量組", 1]], "status": "上次實驗已完成",
    }})
    llm_state = settings.load_service_settings("llm")
    restored = ui.load_experiment_for_ui(project["project_id"], llm_state)
    stored = ui.load_project(project["project_id"])["experiment"]

    assert status.startswith("✅")
    expected_groups = [{
        **groups[0], "reranker_mode": "停用", "evidence_expansion_mode": "證據擴展",
    }]
    assert saved_groups == expected_groups
    assert stored["groups"] == expected_groups
    assert ui.load_project(project["project_id"])["experiment_group_settings"] == expected_groups
    assert stored["questions"] == questions
    assert restored[0] == questions
    assert restored[2] == [{
        key: value for key, value in expected_groups[0].items()
        if key not in {"judge_model", "judge_reasoning_effort"}
    }]
    assert restored[4] == 3
    assert result_status == "實驗組設定已自動儲存。"
    assert restored[3] == [{"passed": True}]
    assert restored[6] == [["向量組", 1]]
    assert restored[7] == [["向量組", 1]]
    assert restored[8]["visible"] is True
    assert restored[8]["value"] == "向量組"
    assert restored[9]["value"] == "gpt-4.1-mini"
    assert restored[10]["value"] == "low"
    assert restored[11]["value"] == "基本向量檢索"
    assert restored[12]["value"] == 5
    assert restored[13]["value"] == "停用"
    assert restored[14]["value"] == "證據擴展"
    assert restored[15]["visible"] is True
    assert restored[8 + ui.EXPERIMENT_GROUP_LIMIT * 8]["value"] == "gpt-6-luna"


def test_luna_reasoning_controls_are_visible_only_for_luna_and_default_low() -> None:
    assert ui.reasoning_effort_visibility("gpt-6-luna") == {
        "visible": True, "value": "low", "__type__": "update",
    }
    assert ui.reasoning_effort_visibility("gpt-4o-mini")["visible"] is False


def test_model_selection_dynamically_toggles_reasoning_effort_control() -> None:
    app = build_app()
    components = app.config["components"]
    model = next(c for c in components if c.get("props", {}).get("label") == "問答 LLM")
    effort_ids = {
        c["id"] for c in components if c.get("props", {}).get("label") == "推理強度"
    }
    dependency = next(
        item for item in app.config["dependencies"]
        if model["id"] in item["inputs"]
        and effort_ids.intersection(item["outputs"])
    )
    assert effort_ids.intersection(dependency["outputs"])
    assert (model["id"], "change") in dependency["targets"]


def test_experiment_group_reasoning_controls_follow_each_selected_model() -> None:
    luna_group = {
        "name": "Luna 組", "answer_model": "gpt-6-luna", "judge_model": "gpt-4o-mini",
    }
    updates = ui._inline_group_updates([luna_group])
    assert updates[2]["visible"] is True
    assert updates[2]["value"] == "low"
    saved = ui._groups_from_inline_values(tuple(ui._inline_group_values([luna_group])))[0]
    assert saved["answer_reasoning_effort"] == "low"
    assert "judge_model" not in saved
    assert "judge_reasoning_effort" not in saved


def test_experiment_global_judge_settings_save_and_reload(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    project = ui.create_project("experiment-global-judge")
    groups = [{
        "name": "Luna 回答組", "answer_model": "gpt-6-luna",
        "retrieval_mode": "混合檢索", "top_k": 8,
        "use_reranker": False, "expand_evidence": False,
    }]
    ui.save_project(project["project_id"], {"experiment": {"groups": groups}})

    assert ui.save_experiment_judge_settings_for_ui(
        project["project_id"], "gpt-6-luna", "high", 7,
    ).startswith("✅")
    loaded = ui.load_experiment_for_ui(
        project["project_id"], settings.load_service_settings("llm"),
    )

    assert loaded[8 + ui.EXPERIMENT_GROUP_LIMIT * 8]["value"] == "gpt-6-luna"
    assert loaded[9 + ui.EXPERIMENT_GROUP_LIMIT * 8]["value"] == "high"
    assert loaded[9 + ui.EXPERIMENT_GROUP_LIMIT * 8]["visible"] is True
    assert loaded[10 + ui.EXPERIMENT_GROUP_LIMIT * 8] == 7
    assert ui.load_project(project["project_id"])["experiment"]["groups"] == groups
    assert ui.load_project(project["project_id"])["experiment"]["judge_model"] == "gpt-6-luna"


def test_empty_inline_autosave_preserves_saved_experiment_groups(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    project = ui.create_project("protect-experiment-groups")
    groups = [{
        "name": "保留組", "answer_model": "model-a",
        "judge_model": "judge-a",
        "retrieval_mode": "混合檢索", "top_k": 7,
        "use_reranker": True, "expand_evidence": False,
    }]
    ui.save_project(project["project_id"], {"experiment": {"groups": groups}})

    status, _state_groups, _result_status = ui.save_inline_experiment_groups_for_ui(
        project["project_id"], [], 5, [], *ui._inline_group_values([]),
    )

    assert status == "✅ 已保留已保存的實驗組設定。"
    assert _state_groups == groups
    assert ui.load_project(project["project_id"])["experiment"]["groups"] == groups


def test_experiment_reload_recovers_saved_group_settings_backup(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    project = ui.create_project("experiment-group-backup")
    groups = [{
        "name": "備援組", "answer_model": "disconnected-model",
        "judge_model": "judge-model",
        "retrieval_mode": "基本向量檢索", "top_k": 4,
        "use_reranker": False, "expand_evidence": True,
    }]
    ui.save_project(project["project_id"], {
        "experiment": {"groups": [], "summary_rows": [["備援組", 1]]},
        "experiment_group_settings": groups,
    })

    restored = ui.load_experiment_for_ui(
        project["project_id"], settings.load_service_settings("llm"),
    )

    assert restored[2] == [{
        key: value for key, value in groups[0].items()
        if key not in {"judge_model", "judge_reasoning_effort"}
    }]
    assert restored[8]["value"] == "備援組"
    assert restored[9]["value"] == "disconnected-model"
    assert ("disconnected-model（目前不可用）", "disconnected-model") in restored[9]["choices"]
    assert restored[8 + ui.EXPERIMENT_GROUP_LIMIT * 8]["value"] == "judge-model"
    migrated = ui.load_project(project["project_id"])
    assert "judge_model" not in migrated["experiment"]["groups"][0]
    assert migrated["experiment"]["judge_model"] == "judge-model"


def test_experiment_default_concurrency_is_five(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    project = ui.create_project("experiment-default-concurrency")

    restored = ui.load_experiment_for_ui(project["project_id"], settings.load_service_settings("llm"))

    assert restored[4] == 5
    judge_index = 8 + ui.EXPERIMENT_GROUP_LIMIT * 8
    assert restored[judge_index]["value"] == "gpt-6-luna"
    assert ("OpenAI｜gpt-6-luna（目前不可用）", "gpt-6-luna") in restored[judge_index]["choices"]
    app = build_app()
    components = app.config["components"]
    values = [component.get("props", {}).get("value") for component in components]
    answer_heading = values.index("#### 回答模型設定｜實驗組（直接編輯欄位；每次變更會自動儲存）")
    summary_heading = values.index("#### 實驗組摘要")
    concurrency_defaults = [
        component["props"]["value"] for component in components[answer_heading:summary_heading]
        if component.get("props", {}).get("label") == "最大並行請求數"
    ]
    assert concurrency_defaults == [5, 5]


def test_experiment_answer_availability_tracks_pending_answers() -> None:
    no_answers, disabled = ui._experiment_answer_availability([])
    has_answers, enabled = ui._experiment_answer_availability([{"actual_answer": "A"}])

    assert "尚未生成" in no_answers
    assert disabled["interactive"] is False
    assert "有 1 個實驗題次" in has_answers
    assert enabled["interactive"] is True


def test_experiment_answer_progress_updates_the_inline_status_panel() -> None:
    control = ui.RunControl()
    assert ui.experiment_answer_progress_for_ui(control)["visible"] is False

    control.answer_generation_progress = (3, 8)
    update = ui.experiment_answer_progress_for_ui(control)

    assert update["visible"] is True
    assert '<progress value="3" max="8"></progress>' in update["value"]
    assert "回答生成進度：3 / 8" in update["value"]


def test_generate_experiment_answers_does_not_judge_or_display(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(ui, "resolve_model_credentials_for_ui", lambda *_: ("answer-endpoint", "key"))
    monkeypatch.setattr(ui, "answer_question_for_ui", lambda *_args, **_kwargs: ("✅ 完成", "隱藏答案", []))
    monkeypatch.setattr(ui, "judge_evaluation_answer", lambda *_args: pytest.fail("回答生成階段不應評測"))
    monkeypatch.setattr(ui, "load_project", lambda _project_id: {"experiment": {}})
    monkeypatch.setattr(ui, "save_project", lambda _project_id, payload: captured.update(payload) or {})
    questions = [{"number": 1, "question": "Q", "expected_answer": "A"}]
    groups = [{
        "name": "G", "answer_model": "answer-model", "retrieval_mode": "混合檢索",
        "top_k": 8, "use_reranker": False, "expand_evidence": False,
    }]

    status, pending, results, summaries, details = ui.generate_experiment_answers_for_ui(
        "project", questions, groups, 2, {}, "embed", "key", "bolt", "database", "user", "pass",
    )

    assert status.startswith("✅ 已生成 1 個實驗題次回答")
    assert pending[0]["actual_answer"] == "隱藏答案"
    assert results == summaries == details == []
    assert captured["experiment"]["pending_answers"] == pending
    assert captured["experiment"]["results"] == []


def test_evaluate_experiment_answers_and_manual_edit_recompute_summary(monkeypatch) -> None:
    group = {
        "name": "G", "answer_model": "answer-model", "retrieval_mode": "混合檢索",
        "top_k": 8, "use_reranker": False, "expand_evidence": False,
    }
    saved = {}
    monkeypatch.setattr(ui, "load_project", lambda _project_id: {"experiment": {"groups": [group]}})
    monkeypatch.setattr(ui, "save_project", lambda _project_id, payload: saved.update(payload) or {})
    monkeypatch.setattr(ui, "judge_evaluation_answer", lambda *_args: {"passed": True, "reason": "正確"})
    pending = [{
        "group_index": 0, "group_name": "G", "answer_model": "answer-model",
        "number": 1, "question": "Q", "expected_answer": "A", "actual_answer": "答案",
        "answer_status": "✅ 完成", "retrieval_rank": 1, "recall_at_5": True,
        "recall_at_10": True, "reciprocal_rank": 1.0,
    }]

    status, results, summaries, details = ui.evaluate_experiment_answers_for_ui(
        "project", pending, [group], "judge-model", "low", 3, "judge-endpoint", "judge-key",
    )

    assert status.startswith("✅ 評測完成")
    assert summaries[0][6:8] == ["1 / 1", "100.0%"]
    assert details[0][6] is True
    edited = [list(details[0])]
    edited[0][6] = False
    manual_status, manual_summary, manual_details, updated = ui.update_manual_experiment_result_for_ui(
        "project", edited, results,
    )

    assert "人工評判變更 1 筆" in manual_status
    assert manual_status.startswith("✅ 評測完成｜")
    assert "答對 0 / 1 個實驗題次" in manual_status
    assert manual_summary[0][6:8] == ["0 / 1", "0.0%"]
    assert manual_details[0][6] is False
    assert updated[0]["reason"] == "人工評判"
    assert saved["experiment"]["results"] == updated


def test_experiment_evaluation_resolves_current_model_credentials_at_click_time(monkeypatch) -> None:
    resolved = {}
    monkeypatch.setattr(
        ui, "resolve_model_credentials_for_ui",
        lambda state, model: (resolved.update(state=state, model=model) or ("endpoint", "secret")),
    )
    monkeypatch.setattr(
        ui, "evaluate_experiment_answers_for_ui",
        lambda *args: (resolved.update(core_args=args) or ("✅ 評測完成", [], [], [])),
    )
    llm_state = {"kind": "llm", "active": "OpenAI"}

    result = ui.evaluate_experiment_answers_with_services_for_ui(
        "project", [{"question": "Q"}], [], "gpt-6-luna", "low", 5, llm_state,
    )

    assert result[0] == "✅ 評測完成"
    assert resolved["state"] is llm_state
    assert resolved["model"] == "gpt-6-luna"
    assert resolved["core_args"][6:8] == ("endpoint", "secret")


def test_experiment_evaluation_reports_unavailable_current_model(monkeypatch) -> None:
    monkeypatch.setattr(ui, "resolve_model_credentials_for_ui", lambda *_: ("", ""))

    result = ui.evaluate_experiment_answers_with_services_for_ui(
        "project", [{"question": "Q"}], [], "gpt-6-luna", "low", 5, {},
    )

    assert "請先測試模型服務連線" in result[0]


def test_inline_experiment_group_add_and_remove(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    project = ui.create_project("experiment-rows")
    llm_state = settings.load_service_settings("llm")
    llm_state["profiles"]["OpenAI"]["connected"] = True
    empty_values = ui._inline_group_values([])

    added = ui.add_inline_experiment_group_for_ui(
        project["project_id"], [], 1, llm_state, [], *empty_values,
    )
    groups = added[-3]
    assert groups[0]["name"] == "實驗組 1"
    assert added[0]["visible"] is True

    groups.append({
        "name": "保留組", "answer_model": "gpt-4o-mini",
        "retrieval_mode": "混合檢索", "top_k": 8,
        "use_reranker": False, "expand_evidence": False,
    })
    monkeypatch.setattr(
        ui, "service_choice_items",
        lambda state: (
            [("OpenAI｜gpt-4o-mini", "gpt-4o-mini")] if state is llm_state else []
        ),
    )

    removed = ui.remove_inline_experiment_group_for_ui(
        0, project["project_id"], [], 1, llm_state, groups,
        *ui._inline_group_values(groups),
    )
    assert removed[-3][0]["name"] == "保留組"
    assert removed[1]["value"] == "gpt-4o-mini"
    assert removed[1]["choices"] == [("OpenAI｜gpt-4o-mini", "gpt-4o-mini")]
    assert removed[1]["visible"] is True
    assert removed[8]["visible"] is False
    assert ui.load_project(project["project_id"])["experiment"]["groups"] == removed[-3]
    assert ui.load_project(project["project_id"])["experiment_group_settings"] == removed[-3]


def test_experiment_ui_uses_inline_dropdowns_and_no_group_dataframe() -> None:
    app = build_app()
    components = app.config["components"]
    assert any(
        "答對數 / 總題數" in component.get("props", {}).get("headers", [])
        for component in components
    )
    assert not any(component.get("props", {}).get("headers") == [
        "實驗組", "回答模型", "評測模型", "檢索模式", "Top K", "Reranker", "擴展圖譜證據",
    ] for component in components)
    assert sum(
        component.get("props", {}).get("choices") == [
            ("基本向量檢索", "基本向量檢索"), ("混合檢索", "混合檢索"),
        ]
        and component.get("type") == "dropdown"
        for component in components
    ) == ui.EXPERIMENT_GROUP_LIMIT + 1
    assert sum(
        component.get("props", {}).get("label") == "全域評測模型"
        and component.get("type") == "dropdown"
        for component in components
    ) == 2
    assert any(component.get("props", {}).get("value") == "新增實驗組" for component in components)
    answer_heading = next(
        item for item in components
        if item.get("props", {}).get("value") == "#### 回答模型設定｜實驗組（直接編輯欄位；每次變更會自動儲存）"
    )
    import_button = next(item for item in components if item.get("props", {}).get("value") == "匯入實驗題目集")
    question_table = next(
        item for item in components
        if item.get("props", {}).get("headers")
        == ["題號", "題目", "正確答案", "題目來源（文件與頁碼）", "答案來源（文件與頁碼）"]
        and item["id"] > import_button["id"]
    )
    judge_heading = next(
        item for item in components
        if item.get("props", {}).get("value") == "#### 評測模型設定"
        and item["id"] > answer_heading["id"]
    )
    summary_heading = next(
        item for item in components if item.get("props", {}).get("value") == "#### 實驗組摘要"
    )
    assert question_table["id"] < answer_heading["id"] < judge_heading["id"] < summary_heading["id"]
    evaluate_button = next(item for item in components if item.get("props", {}).get("value") == "進行評測")
    assert evaluate_button["props"]["interactive"] is False
    assert "evaluation-judge-button" in evaluate_button["props"]["elem_classes"]
    result_table = next(
        item for item in components
        if "答案結果（勾選=正確）" in item.get("props", {}).get("headers", [])
    )
    assert result_table["props"]["interactive"] is False
    assert result_table["props"]["datatype"][6] == "bool"
    assert "回答模型" not in result_table["props"]["headers"]
    assert "評測模型" not in result_table["props"]["headers"]
    assert result_table["props"]["headers"][1:4] == ["題號", "來源文件", "題目"]
    assert "答案來源排名" not in result_table["props"]["headers"]
    assert len(result_table["props"]["headers"]) == 8
    column_widths = [int(str(width).removesuffix("px")) for width in result_table["props"]["column_widths"]]
    assert column_widths[0] < column_widths[4] and column_widths[0] < column_widths[5]
    assert column_widths[1] < column_widths[4] and column_widths[1] < column_widths[5]
    assert column_widths[2] < column_widths[4] and column_widths[2] < column_widths[5]
    assert column_widths[6] < column_widths[4] and column_widths[6] < column_widths[5]
    assert 6 not in result_table["props"]["static_columns"]
    assert any(
        component.get("props", {}).get("label") == "啟用答案結果人工修改"
        and component.get("props", {}).get("value") is False
        for component in components
    )
    edit_toggle = next(
        component for component in components
        if component.get("props", {}).get("label") == "啟用答案結果人工修改"
    )
    assert any(
        str(dependency.get("api_name", "")).startswith("manual_result_editability_for_ui")
        and any(tuple(target) == (edit_toggle["id"], "change") for target in dependency.get("targets", []))
        for dependency in app.config["dependencies"]
    )
    assert any(
        str(dependency.get("api_name", "")).startswith("update_manual_experiment_result_for_ui")
        and any(tuple(target) == (result_table["id"], "input") for target in dependency.get("targets", []))
        for dependency in app.config["dependencies"]
    )
    evaluation_dependency = next(
        dependency for dependency in app.config["dependencies"]
        if str(dependency.get("api_name", "")).startswith("evaluate_experiment_answers_with_services_for_ui")
    )
    assert len(evaluation_dependency["inputs"]) == 8
    assert any(
        str(dependency.get("api_name", "")).startswith("load_experiment_for_ui")
        for dependency in app.config["dependencies"]
    )
    assert any(
        str(dependency.get("api_name", "")).startswith("experiment_answer_progress_for_ui")
        and dependency.get("queue") is False
        for dependency in app.config["dependencies"]
    )
    export_button = next(
        component for component in components
        if component.get("props", {}).get("value") == "匯出實驗結果 JSON"
    )
    export_dependency = next(
        dependency for dependency in app.config["dependencies"]
        if str(dependency.get("api_name", "")).startswith("export_experiment_results_for_ui")
        and any(target[0] == export_button["id"] for target in dependency["targets"])
    )
    assert result_table["id"] in export_dependency["inputs"]
    compact_export = next(
        component for component in components
        if component.get("props", {}).get("label") == "簡潔指標結果 JSON"
    )
    assert compact_export["id"] in export_dependency["outputs"]


def test_export_experiment_results_includes_group_parameters_summary_and_details(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    project = ui.create_project("experiment-export")
    groups = [{
        "name": "向量組", "answer_model": "model-a", "judge_model": "judge-a",
        "retrieval_mode": "基本向量檢索", "top_k": 6,
        "use_reranker": False, "expand_evidence": True,
    }]
    result = {
        "group_index": 0, "group_name": "向量組", "number": 2,
        "answer_model": "model-a", "judge_model": "judge-a",
        "question": "問題二", "document": "manual.pdf", "expected_answer": "標準答案",
        "actual_answer": "模型答案", "passed": True, "reason": "正確",
        "retrieval_rank": 1, "recall_at_5": True, "recall_at_10": True,
        "reciprocal_rank": 1.0,
    }
    ui.save_project(project["project_id"], {"experiment": {
        "questions": [{"question": "問題二"}], "groups": groups,
        "max_concurrent_requests": 5, "status": "實驗已完成",
        "results": [result],
        "summary_rows": [["向量組", "model-a", "judge-a", 1, "1 / 1", "100.0%", "100.0%", "100.0%", "1.000"]],
        "detail_rows": [["向量組", 2, "問題二"]],
    }})

    status, file_path, compact_file_path = ui.export_experiment_results_for_ui(project["project_id"])
    payload = json.loads(Path(file_path).read_text(encoding="utf-8"))
    compact = json.loads(Path(compact_file_path).read_text(encoding="utf-8"))

    assert status.startswith("✅ 已匯出 1 個實驗組；總答對 1 / 1")
    assert payload["schema_version"] == 3
    assert "status" not in payload
    assert payload["project"] == {
        "project_id": project["project_id"], "name": "experiment-export",
    }
    assert payload["max_concurrent_requests"] == 5
    assert payload["evaluation"] == {
        "judge_model": "judge-a", "judge_reasoning_effort": "low",
    }
    assert payload["summary"] == {
        "question_count": 1, "correct_count": 1, "correct_total": "1 / 1",
        "accuracy": 1.0, "recall_at_5": 1.0, "recall_at_10": 1.0, "mrr": 1.0,
    }
    assert payload["groups"][0] == {
        "name": "向量組",
            "parameters": {
                "answer_model": "model-a", "retrieval_mode": "基本向量檢索",
                "top_k": 6, "use_reranker": False, "expand_evidence": True,
                "reranker_mode": "停用", "evidence_expansion_mode": "證據擴展",
        },
        "summary": {
            "answer_model": "model-a", "judge_model": "judge-a",
            "question_count": 1, "correct_count": 1, "correct_total": "1 / 1",
            "accuracy": 1.0, "recall_at_5": 1.0,
            "recall_at_10": 1.0, "mrr": 1.0,
        },
        "results": [{
            key: value for key, value in result.items()
            if key not in {"group_index", "group_name"}
        } | {"manual_judgment": False}],
    }
    assert compact["format"] == "manual-graphrag-experiment-summary"
    assert compact["summary"] == payload["summary"]
    assert compact["groups"] == [{
        "name": "向量組",
        "parameters": payload["groups"][0]["parameters"],
        "summary": payload["groups"][0]["summary"],
    }]
    assert all("results" not in group for group in compact["groups"])
    assert all(not {"question", "expected_answer", "actual_answer"}.intersection(group) for group in compact["groups"])
    compact_text = json.dumps(compact, ensure_ascii=False)
    assert all(text not in compact_text for text in ("問題二", "標準答案", "模型答案"))


def test_export_experiment_results_syncs_latest_manual_judgment(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    project = ui.create_project("experiment-export-manual-edit")
    group = {
        "name": "向量組", "answer_model": "model-a", "retrieval_mode": "混合檢索",
        "top_k": 8, "use_reranker": False, "expand_evidence": False,
    }
    result = {
        "group_index": 0, "group_name": "向量組", "number": 1,
        "answer_model": "model-a", "judge_model": "judge-a",
        "question": "問題", "document": "manual.pdf", "expected_answer": "標準答案",
        "actual_answer": "模型答案", "passed": True, "reason": "正確",
        "retrieval_rank": 1, "recall_at_5": True, "recall_at_10": True,
        "reciprocal_rank": 1.0,
    }
    ui.save_project(project["project_id"], {"experiment": {
        "groups": [group], "judge_model": "judge-a", "results": [result],
        "summary_rows": [["向量組", "model-a", "judge-a", 1, "1 / 1", "100.0%", "100.0%", "100.0%", "1.000"]],
    }})
    edited_rows = ui._single_experiment_detail_rows([result])
    edited_rows[0][6] = False

    status, file_path, compact_file_path = ui.export_experiment_results_for_ui(project["project_id"], edited_rows)
    exported = json.loads(Path(file_path).read_text(encoding="utf-8"))
    compact = json.loads(Path(compact_file_path).read_text(encoding="utf-8"))

    assert status.startswith("✅ 已匯出")
    assert exported["groups"][0]["results"][0]["passed"] is False
    assert exported["groups"][0]["results"][0]["reason"] == "人工評判"
    assert exported["groups"][0]["results"][0]["manual_judgment"] is True
    assert exported["groups"][0]["summary"] == {
        "answer_model": "model-a", "judge_model": "judge-a",
        "question_count": 1, "correct_count": 0, "correct_total": "0 / 1",
        "accuracy": 0.0, "recall_at_5": 1.0, "recall_at_10": 1.0, "mrr": 1.0,
    }
    assert exported["summary"]["correct_total"] == "0 / 1"
    assert compact["summary"]["correct_total"] == "0 / 1"
    assert compact["groups"][0]["summary"]["correct_total"] == "0 / 1"
    _second_status, second_file_path, second_compact_file_path = ui.export_experiment_results_for_ui(
        project["project_id"], edited_rows,
    )
    assert second_file_path != file_path
    assert second_compact_file_path != compact_file_path


def test_run_experiment_groups_outputs_each_group_summary_and_details(monkeypatch) -> None:
    captured = {"calls": [], "workers": 0}
    real_executor = ui.ThreadPoolExecutor

    def recording_executor(max_workers):
        captured["workers"] = max_workers
        return real_executor(max_workers=max_workers)

    def fake_answer(*args, **kwargs):
        captured["calls"].append(args)
        page = 2 if args[9] == "題目一" else 3
        return "✅ 完成", "實際答案", [[
            "原文", "證據", "retriever", "0.9", f"manual.pdf：{page}",
            "manual.pdf：99", "manual.pdf",
        ]]

    monkeypatch.setattr(ui, "resolve_model_credentials_for_ui", lambda _state, model: (model, "key"))
    monkeypatch.setattr(ui, "ThreadPoolExecutor", recording_executor)
    monkeypatch.setattr(ui, "answer_question_for_ui", fake_answer)
    monkeypatch.setattr(ui, "judge_evaluation_answer", lambda *_: {"passed": True, "reason": "正確"})
    saved = {}
    monkeypatch.setattr(ui, "load_project", lambda _project_id: {"experiment": {}})
    monkeypatch.setattr(ui, "save_project", lambda _project_id, payload: saved.update(payload) or {})
    questions = [
        {"number": 1, "question": "題目一", "expected_answer": "答案一",
         "answer_source_pages": [2], "document": "manual.pdf"},
        {"number": 2, "question": "題目二", "expected_answer": "答案二",
         "answer_source_pages": [3], "document": "manual.pdf"},
    ]
    groups = [
        {"name": "向量", "answer_model": "model-a", "judge_model": "judge-x", "retrieval_mode": "基本向量檢索",
         "top_k": 3, "use_reranker": False, "expand_evidence": False},
        {"name": "混合擴展", "answer_model": "model-b", "judge_model": "judge-y", "retrieval_mode": "混合檢索",
         "top_k": 12, "use_reranker": True, "expand_evidence": True},
    ]

    status, summaries, details, results = ui.run_experiment_groups_for_ui(
        "project", questions, groups, 2, {}, "embed", "embed-key",
        "bolt", "database", "user", "pass",
    )

    assert status.startswith("✅ 已完成 2 個實驗組")
    assert summaries == [
        ["向量", "model-a", "停用", "停用", "judge-x", 2, "2 / 2", "100.0%", "100.0%", "100.0%", "1.000"],
        ["混合擴展", "model-b", "Reranker", "證據擴展", "judge-x", 2, "2 / 2", "100.0%", "100.0%", "100.0%", "1.000"],
    ]
    assert len(details) == len(results) == 4
    assert captured["workers"] == 2
    assert {call[8] for call in captured["calls"]} == {"model-a", "model-b"}
    assert {call[11] for call in captured["calls"]} == {3, 12}
    assert {item["judge_model"] for item in results} == {"judge-x"}
    assert saved["experiment"]["summary_rows"][0][1:5] == [
        "model-a", "停用", "停用", "judge-x",
    ]
    assert saved["experiment"]["judge_model"] == "judge-x"
    assert all("judge_model" not in group for group in saved["experiment"]["groups"])
    assert saved["experiment"]["results"] == results
    assert saved["experiment"]["summary_rows"] == summaries


def test_experiment_reload_migrates_legacy_summary_rows_to_show_models(monkeypatch) -> None:
    groups = [{
        "name": "舊組", "answer_model": "answer-model", "retrieval_mode": "混合檢索",
        "top_k": 8, "use_reranker": False, "expand_evidence": False,
    }]
    monkeypatch.setattr(ui, "load_project", lambda _project_id: {"experiment": {
        "groups": groups,
        "summary_rows": [["舊組", 2, "50.0%", "100.0%", "100.0%", "1.000"]],
        "detail_rows": [[
            "舊組", "answer-model", "answer-model", 1, "Q", "manual.pdf",
            "A", "A", "✅ 通過", "正確", 1,
        ]],
    }})
    monkeypatch.setattr(ui, "service_choice_items", lambda _state: [("answer-model", "answer-model")])

    loaded = ui.load_experiment_for_ui("project", {})

    assert loaded[6] == [[
        "舊組", "answer-model", "停用", "停用", "answer-model", 2,
        "1 / 2", "50.0%", "100.0%", "100.0%", "1.000",
    ]]
    assert loaded[7][0][:4] == ["舊組", 1, "manual.pdf", "Q"]
    assert loaded[7][0][6] is True


def test_run_experiment_groups_stops_after_current_tasks_and_saves_partial_results(monkeypatch) -> None:
    control = ui.RunControl()
    started = Event()
    release = Event()
    calls = []
    monkeypatch.setattr(ui, "resolve_model_credentials_for_ui", lambda *_: ("endpoint", "key"))

    def blocking_answer(*args, **kwargs):
        calls.append(args[9])
        started.set()
        assert release.wait(timeout=5)
        return "✅ 完成", "答案", []

    monkeypatch.setattr(ui, "answer_question_for_ui", blocking_answer)
    monkeypatch.setattr(ui, "judge_evaluation_answer", lambda *_: {"passed": True, "reason": "正確"})
    saved = {}
    monkeypatch.setattr(ui, "load_project", lambda _project_id: {"experiment": {}})
    monkeypatch.setattr(ui, "save_project", lambda _project_id, payload: saved.update(payload) or {})
    questions = [
        {"number": i, "question": f"Q{i}", "expected_answer": "A"}
        for i in range(1, 4)
    ]
    groups = [{
        "name": "測試組", "answer_model": "model-a", "retrieval_mode": "混合檢索",
        "top_k": 5, "use_reranker": False, "expand_evidence": False,
    }]
    returned = []

    def run():
        returned.extend(ui.run_experiment_groups_for_ui(
            "project", questions, groups, 1, {}, "embed", "key",
            "bolt", "database", "user", "pass", control,
        ))

    runner = Thread(target=run)
    runner.start()
    assert started.wait(timeout=5)
    assert ui.request_stop_for_ui(control).startswith("⏹")
    release.set()
    runner.join(timeout=5)

    assert not runner.is_alive()
    assert returned[0].startswith("⏹ 實驗已停止")
    assert "1 / 3" in returned[0]
    assert len(returned[3]) == 1
    assert len(calls) == 1
    assert saved["experiment"]["results"] == returned[3]
    assert saved["experiment"]["summary_rows"][0][5] == 1


def test_switch_document_cycles_through_documents() -> None:
    documents = [
        {"file_name": "a.pdf", "page_start": 1, "page_end": 2},
        {"file_name": "b.pdf", "page_start": 1, "page_end": 2},
    ]
    chunks = [
        TextChunk(1, "a-text", (1,), "a.pdf"),
        TextChunk(2, "b-text", (1,), "b.pdf"),
    ]

    doc, doc_chunks, rows, status = ui.next_document(documents, chunks, documents[0])

    assert doc["file_name"] == "b.pdf"
    assert [chunk.document for chunk in doc_chunks] == ["b.pdf"]
    assert [row[0] for row in rows] == [2]
    assert status == "文件 2 / 2：b.pdf（第 1–2 頁），共 1 個 chunk。"

    back, back_chunks, back_rows, back_status = ui.previous_document(documents, chunks, doc)
    assert back["file_name"] == "a.pdf"
    assert [chunk.document for chunk in back_chunks] == ["a.pdf"]
    assert back_status == "文件 1 / 2：a.pdf（第 1–2 頁），共 1 個 chunk。"


def test_switch_document_wraps_around_and_handles_empty_list() -> None:
    documents = [{"file_name": "only.pdf", "page_start": 1, "page_end": 1}]
    chunks = [TextChunk(1, "text", (1,), "only.pdf")]

    doc, *_rest = ui.next_document(documents, chunks, documents[0])
    assert doc["file_name"] == "only.pdf"

    doc, doc_chunks, rows, status = ui.previous_document([], [], {})
    assert doc == {}
    assert doc_chunks == []
    assert rows == []
    assert status == "尚未解析任何 PDF。"


def test_add_document_for_ui_initializes_active_document(tmp_path, monkeypatch) -> None:
    pdf_path = tmp_path / "manual.pdf"
    pdf_path.write_bytes(b"pdf bytes")
    pages = [PageText(2, "two"), PageText(3, "three")]
    chunks = [
        TextChunk(1, "two", (2,), "manual.pdf"),
        TextChunk(2, "three", (3,), "manual.pdf"),
    ]
    monkeypatch.setattr(ui, "extract_pdf", lambda path: (pages, []))
    monkeypatch.setattr(
        ui, "chunk_pages",
        lambda *args, **kwargs: [
            TextChunk(chunk.number, chunk.text, chunk.pages, kwargs["document"], kwargs["document_id"])
            for chunk in chunks
        ],
    )

    result = ui.add_document_for_ui(str(pdf_path), 100, 0, [], [])
    (
        status, rows, documents, stored_chunks, active_preview, active_chunks,
        document_status, documents_rows, pdf_reset, remove_choices,
    ) = result

    assert status.startswith("✅「manual.pdf」第 2–3 頁，產生 2 個 chunk")
    assert [row[0] for row in rows] == [1, 2]
    assert documents[0]["file_name"] == "manual.pdf"
    assert documents[0]["page_count"] == 2
    assert documents[0]["page_start"] == 2
    assert documents[0]["page_end"] == 3
    assert len(documents[0]["document_id"]) == 64
    assert all(chunk.document_id == documents[0]["document_id"] for chunk in stored_chunks)
    assert [chunk.text for chunk in stored_chunks] == [chunk.text for chunk in chunks]
    assert active_chunks == stored_chunks
    assert active_preview["file_name"] == "manual.pdf"
    assert document_status == "文件 1 / 1：manual.pdf（第 2–3 頁），共 2 個 chunk。"
    assert documents_rows == [[False, "manual.pdf", "2–3", 2]]
    assert remove_choices["choices"] == ["manual.pdf"]


def test_add_document_for_ui_rejects_duplicate_document_name(monkeypatch) -> None:
    monkeypatch.setattr(
        ui, "extract_pdf", lambda path: ((_ for _ in ()).throw(AssertionError()))
    )
    existing = [{"file_name": "manual.pdf"}]

    status, *_ = ui.add_document_for_ui("manual.pdf", 100, 0, existing, [])

    assert status.startswith("❌")
    assert "同名文件" in status



def test_document_table_selection_removes_checked_pdfs(monkeypatch) -> None:
    removed = []
    monkeypatch.setattr(ui, "remove_document", lambda project_id, name: removed.append(name))
    documents = [
        {"file_name": "a.pdf", "page_start": 1, "page_end": 1},
        {"file_name": "b.pdf", "page_start": 1, "page_end": 2},
    ]
    table = [[True, "a.pdf", "1–1", 1], [False, "b.pdf", "1–2", 2]]
    result = ui.remove_document_for_ui("project", table, documents, [])
    assert removed == ["a.pdf"]
    assert [doc["file_name"] for doc in result[1]] == ["b.pdf"]
    assert result[6] == [[False, "b.pdf", "1–2", 0]]



def test_document_table_dataframe_selection_removes_checked_pdf(monkeypatch) -> None:
    import pandas as pd

    removed = []
    monkeypatch.setattr(ui, "remove_document", lambda project_id, name: removed.append(name))
    documents = [
        {"file_name": "a.pdf", "page_start": 1, "page_end": 1},
        {"file_name": "b.pdf", "page_start": 1, "page_end": 2},
    ]
    table = pd.DataFrame([
        [True, "a.pdf", "1–1", 1],
        [False, "b.pdf", "1–2", 2],
    ], columns=["選取", "文件", "頁碼範圍", "Chunk 數"])
    result = ui.remove_document_for_ui("project", table, documents, [])
    assert removed == ["a.pdf"]
    assert [doc["file_name"] for doc in result[1]] == ["b.pdf"]

def test_new_project_resets_pdf_status_message() -> None:
    assert ui.reset_new_project_pdf_status_for_ui() == (
        "尚未解析 PDF。可重複上傳多份 PDF，逐一加入同一個專案。"
    )

def test_remove_document_for_ui_prunes_documents_and_chunks_without_project(monkeypatch) -> None:
    monkeypatch.setattr(
        ui, "remove_document", lambda *args: (_ for _ in ()).throw(AssertionError("不應呼叫"))
    )
    documents = [
        {"file_name": "a.pdf", "page_start": 1, "page_end": 2},
        {"file_name": "b.pdf", "page_start": 1, "page_end": 2},
    ]
    chunks = [
        TextChunk(1, "a-text", (1,), "a.pdf"),
        TextChunk(2, "b-text", (1,), "b.pdf"),
    ]

    result = ui.remove_document_for_ui("", "a.pdf", documents, chunks)
    (
        status, remaining_documents, remaining_chunks, active_preview, active_chunks,
        document_status, documents_rows, remove_choices, rows,
    ) = result

    assert status.startswith("✅ 已移除 1 份文件「a.pdf」")
    assert [doc["file_name"] for doc in remaining_documents] == ["b.pdf"]
    assert [chunk.document for chunk in remaining_chunks] == ["b.pdf"]
    assert active_preview["file_name"] == "b.pdf"
    assert active_chunks == remaining_chunks
    assert documents_rows == [[False, "b.pdf", "1–2", 0]]
    assert remove_choices["choices"] == ["b.pdf"]


def test_remove_document_for_ui_deletes_from_project_when_loaded(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(
        ui, "remove_document",
        lambda project_id, file_name: captured.update(project_id=project_id, file_name=file_name),
    )
    documents = [{"file_name": "a.pdf", "page_start": 1, "page_end": 1}]

    status, remaining_documents, *_ = ui.remove_document_for_ui(
        "project-1", "a.pdf", documents, []
    )

    assert captured == {"project_id": "project-1", "file_name": "a.pdf"}
    assert remaining_documents == []
    assert status.startswith("✅")


def test_remove_document_for_ui_supports_multi_select(monkeypatch) -> None:
    removed = []
    monkeypatch.setattr(
        ui, "remove_document",
        lambda project_id, file_name: removed.append(file_name),
    )
    documents = [
        {"file_name": "a.pdf", "page_start": 1, "page_end": 1},
        {"file_name": "b.pdf", "page_start": 1, "page_end": 1},
        {"file_name": "c.pdf", "page_start": 1, "page_end": 1},
    ]
    chunks = [
        TextChunk(1, "a", (1,), "a.pdf"),
        TextChunk(2, "b", (1,), "b.pdf"),
        TextChunk(3, "c", (1,), "c.pdf"),
    ]

    status, remaining_documents, remaining_chunks, *_ = ui.remove_document_for_ui(
        "project-1", ["a.pdf", "b.pdf"], documents, chunks
    )

    assert sorted(removed) == ["a.pdf", "b.pdf"]
    assert [doc["file_name"] for doc in remaining_documents] == ["c.pdf"]
    assert [chunk.document for chunk in remaining_chunks] == ["c.pdf"]
    assert "2 份文件" in status


def test_add_document_for_ui_accepts_multiple_files_at_once(tmp_path, monkeypatch) -> None:
    paths = [tmp_path / "a.pdf", tmp_path / "b.pdf"]
    for path in paths:
        path.write_bytes(path.name.encode())
    def fake_extract_pdf(path):
        name = Path(path).stem
        return [PageText(1, name)], []

    monkeypatch.setattr(ui, "extract_pdf", fake_extract_pdf)

    result = ui.add_document_for_ui(
        [str(path) for path in paths], 1500, 200, [], []
    )
    (
        status, rows, documents, chunks, active_preview, active_chunks,
        document_status, documents_rows, pdf_reset, remove_choices,
    ) = result

    assert [doc["file_name"] for doc in documents] == ["a.pdf", "b.pdf"]
    assert [chunk.number for chunk in chunks] == [1, 2]
    assert [chunk.document for chunk in chunks] == ["a.pdf", "b.pdf"]
    assert len({chunk.document_id for chunk in chunks}) == 2
    assert active_preview["file_name"] == "b.pdf"
    assert "a.pdf" in status and "b.pdf" in status
    assert "2 份文件" in status
    assert status.count("- ✅") == 2
    assert status.splitlines()[0].startswith("- ✅「a.pdf」")
    assert status.splitlines()[1].startswith("- ✅「b.pdf」")


def test_request_stop_for_ui_sets_flag_on_shared_control() -> None:
    control = ui.RunControl()

    status = ui.request_stop_for_ui(control)

    assert status.startswith("⏹")
    with pytest.raises(ui.RunCancelled):
        control.check()


def test_toggle_pause_for_ui_flips_state_and_button_label() -> None:
    control = ui.RunControl()

    status, button_update = ui.toggle_pause_for_ui(control)
    assert status.startswith("⏸")
    assert button_update["value"] == "▶ 繼續"
    assert control.is_paused is True

    status, button_update = ui.toggle_pause_for_ui(control)
    assert status.startswith("▶")
    assert button_update["value"] == "⏸ 暫停"
    assert control.is_paused is False


def test_plan_schema_for_ui_reports_stopped_status(monkeypatch) -> None:
    monkeypatch.setattr(
        ui, "plan_graph_schema",
        lambda *args, **kwargs: (_ for _ in ()).throw(ui.RunCancelled("stopped")),
    )

    status, schema_text = ui.plan_schema_for_ui(
        "http://models/v1", "key", "llm", 0.3, "平衡", 3,
        [[True, "manual.pdf"]], [TextChunk(1, "text", (1,), "manual.pdf")], ui.RunControl(),
    )

    assert status.startswith("⏹")
    assert schema_text == ""


def test_schema_documents_default_to_all_and_require_a_selection() -> None:
    update = ui.schema_documents_for_ui([
        {"file_name": "a.pdf"},
        {"file_name": "b.pdf"},
    ])
    assert update["value"] == [[True, "a.pdf"], [True, "b.pdf"]]

    status, schema_text = ui.plan_schema_for_ui(
        "http://models/v1",
        "key",
        "llm",
        0.3,
        "平衡",
        3,
        [],
        [TextChunk(1, "text", (1,), "a.pdf")],
        ui.RunControl(),
    )
    assert status == "❌ 請至少勾選一份用於規劃 Schema 的 PDF"
    assert schema_text == ""


def test_extract_graph_for_ui_reports_stopped_status(monkeypatch) -> None:
    monkeypatch.setattr(
        ui, "extract_graph",
        lambda *args, **kwargs: (_ for _ in ()).throw(ui.RunCancelled("stopped")),
    )
    schema = {"entity_types": [{"name": "DEVICE"}], "relationship_types": [{"name": "USES"}]}

    result = ui.extract_graph_for_ui(
        "http://models/v1", "key", "llm", 0.2, 3,
        [TextChunk(1, "text", (1,))], json.dumps(schema), {}, ui.RunControl(),
    )

    assert result == ("⏹ 已停止（使用者中止抽取）。", [], [], {})


def test_plan_schema_for_ui_returns_editable_json(monkeypatch) -> None:
    from manual_graphrag.graph_service import SchemaPlan

    captured = {}

    def fake_plan(*args, **kwargs):
        captured["chunks"] = args[3]
        return SchemaPlan(
            {"entity_types": [{"name": "DEVICE"}], "relationship_types": [{"name": "USES"}]},
            5,
            5,
            2,
            1,
        )

    monkeypatch.setattr(
        ui,
        "plan_graph_schema",
        fake_plan,
    )

    status, schema_text = ui.plan_schema_for_ui(
        "http://models/v1",
        "key",
        "llm",
        0.3,
        "平衡",
        3,
        [[True, "a.pdf"], [False, "b.pdf"]],
        [
            TextChunk(1, "A", (1,), "a.pdf"),
            TextChunk(2, "B", (1, 2), "b.pdf"),
        ],
        ui.RunControl(),
    )

    assert status.startswith("✅")
    assert "1 份 PDF（a.pdf）的全部 1 頁、5 個 chunk" in status
    assert [chunk.document for chunk in captured["chunks"]] == ["a.pdf"]
    assert "共 2 批、1 輪整合" in status
    assert "粒度：平衡。" in status
    assert "類型上限" not in status
    assert json.loads(schema_text)["entity_types"][0]["name"] == "DEVICE"


def test_extract_graph_for_ui_formats_tables_and_state(monkeypatch) -> None:
    from manual_graphrag.graph_service import GraphExtraction

    extraction = GraphExtraction(
        entities=[
            {
                "name": "設備 A",
                "type": "DEVICE",
                "description": "設備",
                "source_chunk_numbers": [1],
                "source_pages": [3],
                "source_documents": ["a.pdf", "b.pdf"],
                "source_references": [
                    {"document": "a.pdf", "chunk_numbers": [1], "pages": [3]},
                    {"document": "b.pdf", "chunk_numbers": [8], "pages": [2]},
                ],
            }
        ],
        relationships=[
            {
                "source": "設備 A",
                "type": "USES",
                "target": "設備 B",
                "description": "使用",
                "source_chunk_numbers": [1],
                "source_pages": [3],
            }
        ],
        processed_chunks=1,
    )
    monkeypatch.setattr(ui, "extract_graph", lambda *args, **kwargs: extraction)
    monkeypatch.setattr(
        ui,
        "import_extraction",
        lambda *args: (_ for _ in ()).throw(AssertionError("抽取時不應匯入 Neo4j")),
    )
    schema = {
        "entity_types": [{"name": "DEVICE"}],
        "relationship_types": [{"name": "USES"}],
    }

    status, entities, relationships, state = ui.extract_graph_for_ui(
        "http://models/v1",
        "key",
        "llm",
        0.2,
        3,
        [TextChunk(1, "text", (3,))],
        json.dumps(schema),
        [{"file_name": "manual.pdf"}],
        ui.RunControl(),
    )

    assert status.startswith("✅ 已處理 1 個 chunk")
    assert entities[0][:2] == ["設備 A", "DEVICE"]
    assert entities[0][3:] == [
        "a.pdf：1\nb.pdf：8", "a.pdf：3\nb.pdf：2", "a.pdf\nb.pdf"
    ]
    assert relationships[0][:3] == ["設備 A", "USES", "設備 B"]
    assert state["document"] == "manual.pdf"
    assert state["temperature"] == 0.2
    assert "max_output_tokens" not in state
    assert state["max_concurrent_requests"] == 3
    assert state["chunks"][0]["text"] == "text"
    assert state["neo4j_imported"] is False
    assert "進行 Embedding 並匯入 Neo4j" in status


def test_graph_evidence_separates_merged_sources_by_pdf() -> None:
    references = [
        {"document": "a.pdf", "chunk_numbers": [1], "pages": [3]},
        {"document": "b.pdf", "chunk_numbers": [8], "pages": [2]},
    ]
    evidence = ui._build_graph_evidence(
        [{
            "name": "共用設備", "type": "DEVICE", "description": "設備",
            "source_references": references,
        }],
        [{
            "source": "共用設備", "type": "USES", "target": "配件",
            "description": "使用", "source_references": references,
        }],
        [],
    )

    assert len(evidence) == 4
    assert [item["source_documents"] for item in evidence] == [
        ["a.pdf"], ["b.pdf"], ["a.pdf"], ["b.pdf"]
    ]
    assert [item["source_chunk_numbers"] for item in evidence] == [
        [1], [8], [1], [8]
    ]
    assert all(len(item["source_references"]) == 1 for item in evidence)


def test_extract_graph_for_ui_rejects_invalid_schema() -> None:
    result = ui.extract_graph_for_ui(
        "http://models/v1", "", "llm", 0.2, 3, [], "not-json", {}, ui.RunControl()
    )

    assert result == ("❌ schema 不是有效 JSON。", [], [], {})


def _importable_graph_state() -> dict:
    return {
        "run_id": "run-1",
        "document": "manual.pdf",
        "llm_model": "llm",
        "schema": {
            "entity_types": [{"name": "DEVICE"}],
            "relationship_types": [{"name": "USES"}],
        },
        "entities": [{
            "name": "設備 A", "type": "DEVICE", "description": "設備",
            "source_chunk_numbers": [1], "source_pages": [1],
        }],
        "relationships": [],
        "chunks": [{"number": 1, "text": "設備說明", "pages": [1]}],
        "neo4j_imported": False,
    }


def test_import_graph_for_ui_imports_saved_extraction(monkeypatch) -> None:
    from manual_graphrag.neo4j_service import ImportSummary

    captured = {}

    def fake_import(*args):
        captured["args"] = args
        return ImportSummary(2, 1)

    monkeypatch.setattr(ui, "embedding_vectors", lambda *args: [[0.1]] * 2)
    monkeypatch.setattr(ui, "import_extraction", fake_import)

    status, state = ui.import_graph_for_ui(
        "http://models/v1", "key", "bolt://db", "neo4j", "user", "password",
        "embed", _importable_graph_state(),
    )

    assert status.startswith("✅ 已清空本工具既有圖譜")
    assert state["neo4j_imported"] is True
    assert state["embedding_model"] == "embed"
    assert state["embedding_dimensions"] == 1
    assert state["vector_index_name"] == "graph_evidence_embedding_1"
    assert captured["args"][4] == "run-1"


def test_import_graph_for_ui_keeps_state_when_import_fails(monkeypatch) -> None:
    monkeypatch.setattr(ui, "embedding_vectors", lambda *args: [[0.1]] * 2)
    monkeypatch.setattr(
        ui,
        "import_extraction",
        lambda *args: (_ for _ in ()).throw(ValueError("Neo4j 寫入失敗：offline")),
    )

    status, state = ui.import_graph_for_ui(
        "http://models/v1", "key", "bolt://db", "neo4j", "user", "password",
        "embed", _importable_graph_state(),
    )

    assert status == "❌ Neo4j 寫入失敗：offline"
    assert state["neo4j_imported"] is False
    assert state["neo4j_error"] == "Neo4j 寫入失敗：offline"


def test_import_graph_for_ui_requires_extraction() -> None:
    assert ui.import_graph_for_ui(
        "http://models/v1", "key", "bolt://db", "neo4j", "user", "password",
        "embed", {},
    ) == ("❌ 請先完成知識圖譜抽取。", {})


def test_answer_question_for_ui_can_disable_reranker(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        ui,
        "load_latest_graph",
        lambda *args: {
            "run_id": "run-1",
            "document": "manual.pdf",
            "embedding_model": "embed",
        },
    )
    monkeypatch.setattr(ui, "embedding_vectors", lambda *args: [[0.1]])
    captured = {}

    def fake_search(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return [{
            "evidence_id": "chunk-1",
            "kind": "原文",
            "text": "E01 排除方式",
            "source_pages": [3],
            "source_chunk_numbers": [2],
            "score": 0.03,
            "fusion_score": 0.03,
            "matched_by": ["official-hybrid"],
        }]

    monkeypatch.setattr(ui, "search_graph_evidence", fake_search)
    monkeypatch.setattr(
        ui,
        "answer_graph_question",
        lambda *args: {"answer": "請重新啟動。", "evidence": args[-1]},
    )

    status, answer, rows = ui.answer_question_for_ui(
        "http://models/v1", "key", "http://embed/v1", "embed-key",
        "bolt://db", "neo4j", "user", "password",
        "answer", " E01 怎麼處理？ ", "基本檢索", 8, False, False,
    )

    assert status.startswith("✅ 基本檢索")
    assert answer == "請重新啟動。"
    assert captured["args"][5] == "E01 怎麼處理？"
    assert captured["kwargs"]["candidate_top_k"] == 8
    assert captured["kwargs"]["expand_evidence"] is False
    assert rows == [[
        "原文", "E01 排除方式", "official-hybrid",
        "0.0300", "3", "2", "",
    ]]


def test_answer_question_for_ui_keeps_expanded_evidence_without_reranker(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ui, "load_latest_graph", lambda *args: {
        "run_id": "run-1", "document": "manual.pdf", "embedding_model": "embed",
    })
    monkeypatch.setattr(ui, "embedding_vectors", lambda *args: [[0.1]])
    monkeypatch.setattr(ui, "search_graph_evidence", lambda *args, **kwargs: [
        {"evidence_id": "direct-1", "kind": "原文", "text": "直接命中", "matched_by": ["official-hybrid"]},
        {"evidence_id": "direct-2", "kind": "原文", "text": "另一直接命中", "matched_by": ["official-hybrid"]},
        {"evidence_id": "graph-1", "kind": "關係", "text": "圖譜關聯證據", "matched_by": ["graph"]},
        {"evidence_id": "graph-2", "kind": "原文", "text": "關聯原文", "matched_by": ["graph"]},
    ])
    captured = {}

    def fake_answer(*args, **kwargs):
        captured["evidence"] = args[5]
        return {"answer": "答案", "evidence": args[5]}

    monkeypatch.setattr(ui, "answer_graph_question", fake_answer)

    status, answer, _rows = ui.answer_question_for_ui(
        "http://models/v1", "key", "http://embed/v1", "embed-key",
        "bolt://db", "neo4j", "user", "password",
        "answer", "問題", "混合檢索", 4, False, True,
    )

    assert status.startswith("✅ 混合檢索")
    assert answer == "答案"
    assert [item["evidence_id"] for item in captured["evidence"]] == [
        "direct-1", "graph-1", "direct-2", "graph-2",
    ]


def test_answer_question_for_ui_runs_reranker_with_answer_model(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ui, "load_latest_graph", lambda *args: {
        "run_id": "run-1", "document": "manual.pdf", "embedding_model": "embed",
    })
    monkeypatch.setattr(ui, "embedding_vectors", lambda *args: [[0.1]])
    monkeypatch.setattr(ui, "search_graph_evidence", lambda *args, **kwargs: [
        {"evidence_id": "one", "kind": "原文", "text": "候選一", "matched_by": ["official-hybrid"]},
        {"evidence_id": "two", "kind": "原文", "text": "候選二", "matched_by": ["official-hybrid"]},
    ])
    captured = {}

    def fake_rerank(*args):
        captured["args"] = args
        return [args[4][1]]

    monkeypatch.setattr(ui, "rerank_evidence", fake_rerank)
    monkeypatch.setattr(
        ui, "answer_graph_question",
        lambda *args, **kwargs: {"answer": "答案", "evidence": args[5]},
    )

    status, answer, _rows = ui.answer_question_for_ui(
        "http://models/v1", "secret", "http://embed/v1", "embed-key",
        "bolt://db", "neo4j", "user", "password",
        "gpt-4.1-mini", "問題", "混合檢索", 1, True, False,
    )

    assert status.startswith("✅ 混合檢索")
    assert answer == "答案"
    assert captured["args"][:4] == (
        "http://models/v1", "secret", "gpt-4.1-mini", "問題",
    )
    assert captured["args"][5] == 1


def test_evaluation_preferences_keep_generation_and_test_models_separate(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(ui, "load_project", lambda project_id: {"evaluation": {}})
    monkeypatch.setattr(ui, "save_project", lambda project_id, payload: captured.update(payload) or {})

    status = ui.save_evaluation_preferences_for_ui(
        "project", "generation-model", "test-model", 12, "基本檢索", 6,
        True, 5, False, False, "judge-model",
        judge_max_concurrent_requests=7,
    )

    assert status.startswith("✅")
    assert captured["evaluation"]["preferences"] == {
        "generation_model": "generation-model",
        "test_model": "test-model",
        "judge_model": "judge-model",
        "question_count": 12,
        "retrieval_mode": "基本檢索",
        "top_k": 6,
        "allow_parallel_generation": True,
        "use_reranker": False,
        "expand_evidence": False,
        "test_max_concurrent_requests": 5,
        "judge_max_concurrent_requests": 7,
        "generation_reasoning_effort": "low",
        "test_reasoning_effort": "low",
        "judge_reasoning_effort": "low",
    }


def test_load_evaluation_restores_saved_summary(monkeypatch) -> None:
    monkeypatch.setattr(ui, "load_project", lambda project_id: {
        "evaluation": {
            "questions": [{"number": 1, "question": "Q", "expected_answer": "A"}],
            "results": [
                {
                    "number": 1, "question": "Q1", "expected_answer": "A1",
                    "document": "manual.pdf", "actual_answer": "A1", "passed": True,
                    "reason": "正確", "recall_at_5": True, "reciprocal_rank": 0.5,
                    "retrieval_rank": 2,
                },
                {
                    "number": 2, "question": "Q2", "expected_answer": "A2",
                    "document": "manual.pdf", "actual_answer": "", "passed": False,
                    "reason": "錯誤", "recall_at_5": False, "reciprocal_rank": 0.0,
                    "retrieval_rank": None,
                },
            ],
        }
    })

    loaded = ui.load_evaluation_for_ui("project")

    assert "已載入測試結果" in loaded[-1]
    assert "答案正確率：50.0%" in loaded[-1]
    assert "Recall@5：50.0%" in loaded[-1]
    assert "Recall@10：50.0%" in loaded[-1]
    assert "MRR：0.250" in loaded[-1]
    assert len(loaded[2]) == 2


def test_load_evaluation_restores_separate_judge_model(monkeypatch) -> None:
    monkeypatch.setattr(ui, "load_project", lambda _project_id: {
        "evaluation": {"preferences": {
            "test_model": "answer-model", "judge_model": "judge-model",
        }},
    })

    loaded = ui.load_evaluation_for_ui("project")

    assert loaded[4] == "answer-model"
    assert loaded[12] == "judge-model"
    assert loaded[13] == 3
    assert "已載入" in loaded[-1]



def test_load_evaluation_supports_legacy_shared_model(monkeypatch) -> None:
    monkeypatch.setattr(ui, "load_project", lambda project_id: {
        "evaluation": {"preferences": {"model": "legacy-model", "retrieval_mode": "GraphRAG"}}
    })

    loaded = ui.load_evaluation_for_ui("project")

    assert loaded[8] is False
    assert loaded[9] is False
    assert loaded[10] is False
    assert loaded[11] == 3
    assert loaded[3:5] == ("legacy-model", "legacy-model")
    assert loaded[12] == "gpt-6-luna"
    assert loaded[6] == "混合檢索"


def test_evaluation_judge_defaults_to_luna_without_saved_preference(monkeypatch) -> None:
    monkeypatch.setattr(ui, "load_project", lambda _project_id: {"evaluation": {}})

    loaded = ui.load_evaluation_for_ui("project")

    assert loaded[12] == "gpt-6-luna"
    assert loaded[16]["visible"] is True


def test_project_list_refreshes_on_page_load_and_tab_select_without_focus_rerender(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    app = build_app()
    selector = next(
        component for component in app.config["components"]
        if component.get("props", {}).get("label") == "現有專案"
    )
    assert selector["props"]["choices"] == []
    dependencies = [
        dependency for dependency in app.config["dependencies"]
        if str(dependency.get("api_name", "")).startswith("refresh_projects_for_ui")
    ]
    triggers = {
        target[1] for dependency in dependencies for target in dependency["targets"]
    }
    assert all(len(dependency["inputs"]) == 1 for dependency in dependencies)
    assert triggers == {"load", "select"}
    assert "focus" not in triggers
    assert all(dependency["outputs"] == [selector["id"]] for dependency in dependencies)

    project = ui.create_project("新增專案")
    for dependency in dependencies:
        update = app.fns[dependency["id"]].fn()
        assert update["choices"] == [("新增專案", project["project_id"])]
        assert update["value"] == project["project_id"]

    selected = ui.refresh_projects_for_ui(project)
    assert selected["choices"] == [("新增專案", project["project_id"])]
    assert selected["value"] == project["project_id"]

    ui.delete_project(project["project_id"])
    deleted = ui.refresh_projects_for_ui(project)
    assert deleted["choices"] == []
    assert deleted["value"] is None


def test_unselected_models_report_actionable_errors_without_network() -> None:
    plan = ui.plan_schema_for_ui(
        "", "", None, 0, "平衡", 1, [], [], ui.RunControl(),
    )
    extraction = ui.extract_graph_for_ui("", "", None, 0, 1, [], "{}", [], ui.RunControl())
    generation = ui.generate_evaluation_for_ui("project", "", "", None, None, 1, "基本檢索", 1, [])
    evaluation = ui.run_evaluation_for_ui(
        "project", "", "", "", "", "", "", "", "", None, "基本檢索", 1,
        {"questions": [{"question": "Q"}]},
    )
    answer = ui.answer_question_for_ui("", "", "", "", "", "", "", "", None, "Q", "基本檢索", 1)
    embedding = ui.import_graph_for_ui("", "", "", "", "", "", None, {"run_id": "run"})
    for result in [plan, extraction, generation, evaluation, answer, embedding]:
        assert result[0].startswith("❌")
        assert "選擇" in result[0]


def test_model_selections_are_not_saved_in_env(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    ui.persist_env_settings("", "", "", "", "", "", "", "build", "embed", "answer")
    content = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "BUILD_MODEL" not in content
    assert "EMBEDDING_MODEL" not in content
    assert "ANSWER_MODEL" not in content


def test_experiment_project_members_require_existing_built_graphs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    built = ui.create_project("已建圖車型")
    unbuilt = ui.create_project("未建圖車型")
    ui.save_project(built["project_id"], {"graph_state": {"neo4j_imported": True}})
    experiment = ui.create_experiment_project("跨車型比較")

    assert ui._built_project_choices() == [("已建圖車型", built["project_id"])]
    state, rows, status = ui.save_experiment_project_members_for_ui(
        experiment["experiment_project_id"], [built["project_id"], unbuilt["project_id"]],
    )
    assert status.startswith("❌")
    assert rows == []
    assert state["members"] == []

    state, rows, status = ui.save_experiment_project_members_for_ui(
        experiment["experiment_project_id"], [built["project_id"]],
    )
    assert status.startswith("✅")
    assert state["members"] == [built["project_id"]]
    assert rows == [["已建圖車型", built["project_id"], built["neo4j_database"]]]


def test_import_experiment_project_questions_saves_per_member(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    member = ui.create_project("車型 A")
    ui.save_project(member["project_id"], {"graph_state": {"neo4j_imported": True}})
    experiment = ui.create_experiment_project("多車型實驗")
    experiment = ui.save_experiment_project(experiment["experiment_project_id"], {
        "members": [member["project_id"]],
    })
    question_file = tmp_path / "questions.json"
    question_file.write_text(json.dumps([{
        "number": 1, "question": "Q", "expected_answer": "A",
        "question_sources": [{"document": "guide.pdf", "pages": [1, 2]}],
        "answer_sources": [{"document": "guide.pdf", "pages": [3]}],
    }]), encoding="utf-8")

    updated, rows, status, file_update = ui.import_experiment_project_questions_for_ui(
        str(question_file), experiment, member["project_id"],
    )

    assert status.startswith("✅")
    assert len(rows) == 1
    assert len(updated["questions_by_project"][member["project_id"]]) == 1
    assert file_update == {"value": None, "__type__": "update"}
    assert not ui.load_project(member["project_id"]).get("experiment")


def test_experiment_project_banner_uses_experiment_name():
    assert ui.experiment_project_banner_for_ui({"name": "跨車型實驗"}) == "### 📁 目前專案：跨車型實驗"
    assert ui.experiment_project_banner_for_ui({}) == "### 📁 目前專案：尚未選擇"


def test_experiment_project_model_dropdowns_have_distinct_configured_choices():
    app = ui.build_app()
    components = app.config["components"]
    target_labels = {
        "跨專案回答模型", "跨專案評測模型", "全域評測模型",
    }
    selectors = [
        item for item in components
        if item.get("props", {}).get("label") in target_labels
    ]
    assert {item["props"]["label"] for item in selectors} == target_labels
    assert len({item["id"] for item in selectors}) == len(selectors)
    assert all(item["props"].get("choices") for item in selectors)


def test_add_experiment_project_group_uses_sequential_default_name(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    experiment = ui.create_experiment_project("自動命名")
    experiment, _rows, _remove, _status, next_name = ui.add_experiment_project_group_for_ui(
        experiment, "", "gpt-4.1-mini", "low", "混合檢索", 8, False, False,
    )
    assert experiment["groups"][0]["name"] == "實驗組 1"
    assert next_name == {"value": "實驗組 2", "__type__": "update"}


def test_multi_project_experiment_runs_each_projects_own_database(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    members = [ui.create_project("Model A"), ui.create_project("Model B")]
    for member in members:
        ui.save_project(member["project_id"], {"graph_state": {"neo4j_imported": True}})
    experiment = ui.create_experiment_project("cross-model")
    experiment = ui.save_experiment_project(experiment["experiment_project_id"], {
        "members": [member["project_id"] for member in members],
        "questions_by_project": {
            member["project_id"]: [{
                "number": 1, "question": f"Q {index}", "expected_answer": "A",
            }]
            for index, member in enumerate(members, start=1)
        },
        "groups": [{
            "name": "向量組", "answer_model": "model-a", "retrieval_mode": "基本向量檢索",
            "top_k": 5, "use_reranker": False, "expand_evidence": False,
        }],
    })
    calls = []

    def run_single(project_id, questions, groups, _concurrency, _llm_state,
                   _embedding_base, _embedding_key, _uri, database, _username,
                   _password, _control, _judge, _effort, persist, _progress):
        calls.append((project_id, questions, database, persist))
        return "✅ 已完成", [], [], [{
            "group_index": 0, "group_name": "向量組", "answer_model": "model-a",
            "judge_model": "judge", "number": 1,
            "question": questions[0]["question"], "expected_answer": "A",
            "actual_answer": "A", "passed": project_id == members[0]["project_id"],
            "reason": "判定", "retrieval_rank": 1, "recall_at_5": True,
            "recall_at_10": True, "reciprocal_rank": 1.0,
        }]

    monkeypatch.setattr(ui, "run_experiment_groups_for_ui", run_single)
    status, summaries, details, saved = ui.run_experiment_project_for_ui(
        experiment, 2, {}, "embed", "key", "bolt", "user", "password",
        "judge", "low",
    )

    assert status.startswith("✅ 已完成 1 個實驗組")
    assert [call[0] for call in calls] == [member["project_id"] for member in members]
    assert [call[2] for call in calls] == [member["neo4j_database"] for member in members]
    assert all(call[3] is False for call in calls)
    assert summaries[0][5:8] == [2, "1 / 2", "50.0%"]
    assert {row[1] for row in details} == {member["name"] for member in members}
    assert len(saved["results"]) == 2


def test_experiment_project_answers_are_generated_then_evaluated_separately(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    member = ui.create_project("跨專案成員")
    ui.save_project(member["project_id"], {"graph_state": {"neo4j_imported": True}})
    experiment = ui.create_experiment_project("分階段測試")
    question = {"number": 1, "question": "Q", "expected_answer": "A", "document": "manual.pdf"}
    experiment = ui.save_experiment_project(experiment["experiment_project_id"], {
        "members": [member["project_id"]],
        "questions_by_project": {member["project_id"]: [question]},
        "groups": [{"name": "G", "answer_model": "answer", "retrieval_mode": "混合檢索",
                    "top_k": 5, "use_reranker": False, "expand_evidence": False}],
    })
    monkeypatch.setattr(ui, "resolve_model_credentials_for_ui", lambda *_: ("endpoint", "key"))
    monkeypatch.setattr(ui, "answer_question_for_ui", lambda *args, **kwargs: ("✅ 完成", "A", []))
    generated_status, pending, summaries, details, saved = ui.generate_experiment_project_answers_for_ui(
        experiment, 2, {}, "embed", "key", "bolt", "user", "password",
    )
    assert generated_status.startswith("✅ 已生成")
    assert len(pending) == 1 and pending[0]["actual_answer"] == "A"
    assert summaries == details == []
    assert saved["pending_answers"] == pending

    monkeypatch.setattr(ui, "judge_evaluation_answer", lambda *args, **kwargs: {"passed": True, "reason": "符合"})
    evaluated_status, results, summaries, details, saved = ui.evaluate_experiment_project_answers_for_ui(
        saved, pending, "judge", "low", 2, {},
    )
    assert evaluated_status == "✅ 評測完成｜答對 1 / 1 個跨專案實驗題次。"
    assert results[0]["passed"] is True
    assert summaries[0][6] == "1 / 1"
    assert details[0][1:3] == ["跨專案成員", 1]

    details[0][7] = False
    edited_status, summaries, _details, saved = ui.update_manual_experiment_project_result_for_ui(
        saved, details, results,
    )
    assert "答對 0 / 1" in edited_status
    assert summaries[0][6] == "0 / 1"
    assert saved["results"][0]["reason"] == "人工評判"


def test_experiment_project_inline_groups_preserve_saved_groups_on_empty_snapshot(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    experiment = ui.create_experiment_project("保留實驗組")
    experiment = ui.save_experiment_project(experiment["experiment_project_id"], {
        "groups": [{"name": "G", "answer_model": "answer", "retrieval_mode": "混合檢索",
                    "top_k": 8, "use_reranker": False, "expand_evidence": False}],
        "pending_answers": [{"actual_answer": "A"}],
    })

    restored, message = ui.save_experiment_project_groups_from_rows_for_ui(experiment, [])
    assert message.startswith("✅ 保留已保存")
    assert restored["groups"] == experiment["groups"]
    assert restored["pending_answers"] == experiment["pending_answers"]
