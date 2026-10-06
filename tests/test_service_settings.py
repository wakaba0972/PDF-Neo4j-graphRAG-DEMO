import yaml

import pytest

from manual_graphrag import service_settings as settings, ui
from manual_graphrag.env_store import DEFAULTS


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def choices(provider, models):
    return [(f"{provider}｜{model}", model) for model in models]


def service(kind="llm"):
    return settings.load_service_settings(kind, dict(DEFAULTS))


def act(state, action, *, provider=None, base=None, key=None, rows=None, models=None):
    p = state["profiles"][state["active"]]
    displayed = ui.render_service_for_ui(state)
    return ui.service_action_for_ui(
        action, provider or state["active"], state,
        p["base_url"] if base is None else base,
        p["api_key"] if key is None else key,
        p["rows"] if rows is None else rows,
        *([u["value"] for u in displayed[7:]] if models is None else models),
    )


def test_legacy_builtin_openai_model_list_is_upgraded_without_customizing_other_lists(tmp_path, monkeypatch):
    path = tmp_path / "model_settings.yaml"
    monkeypatch.setattr(settings, "MODEL_SETTINGS_PATH", path)
    path.write_text(
        "openai_models:\n"
        "  llm: [gpt-4.1-mini, gpt-4o-mini]\n"
        "  embedding: [text-embedding-3-small, text-embedding-3-large]\n",
        encoding="utf-8",
    )

    assert settings.openai_models("llm") == ["gpt-4.1-mini", "gpt-4o-mini", "gpt-6-luna"]
    assert settings.openai_models("embedding") == [
        "text-embedding-3-small", "text-embedding-3-large",
    ]


def test_custom_openai_model_allowlist_is_not_overwritten(tmp_path, monkeypatch):
    path = tmp_path / "model_settings.yaml"
    monkeypatch.setattr(settings, "MODEL_SETTINGS_PATH", path)
    path.write_text(
        "openai_models:\n  llm: [custom-chat]\n  embedding: [custom-embed]\n",
        encoding="utf-8",
    )

    assert settings.openai_models("llm") == ["custom-chat"]
    assert settings.openai_models("embedding") == ["custom-embed"]


@pytest.mark.parametrize("kind,expected", [
    ("llm", ["gpt-4.1-mini", "gpt-4o-mini", "gpt-6-luna"]),
    ("embedding", ["text-embedding-3-small", "text-embedding-3-large"]),
])
def test_openai_models_are_selectable_before_connection_test(monkeypatch, kind, expected):
    calls = []
    monkeypatch.setattr(ui, "check_model_connection", lambda base, key: calls.append((base, key)))
    monkeypatch.setattr(ui, "check_embedding_connection", lambda *args: calls.append(args))
    state = service(kind)
    initial = ui.render_service_for_ui(state)
    expected_choices = choices("OpenAI", expected)
    assert all(u["choices"] == expected_choices for u in initial[7:])
    tested = act(state, "test", key="test-key")
    assert calls == [
        ("https://api.openai.com/v1", "test-key")
        if kind == "llm" else
        ("https://api.openai.com/v1", "test-key", "text-embedding-3-small")
    ]
    assert tested[4]["visible"] is True
    assert tested[5] is False
    assert tested[3] == []
    assert all(u["choices"] == expected_choices for u in tested[7:])
    assert tested[6].startswith("✅")
    # Neither a forged table nor the hidden fetch action may add extra OpenAI models.
    edited = act(tested[0], "edit", rows=[[True, "expensive-model"]], models=["expensive-model"] * settings.MODEL_COUNTS[kind])
    assert all(u["choices"] == expected_choices and u["value"] == expected[0] for u in edited[7:])
    with pytest.raises(ui.gr.Error, match="僅支援 OpenAI"):
        act(tested[0], "fetch", provider="Ollama")


def test_default_model_credentials_refresh_before_service_connection(monkeypatch):
    monkeypatch.setattr(ui, "check_model_connection", lambda *_: None)
    monkeypatch.setattr(ui, "check_embedding_connection", lambda *_: None)
    llm = act(service("llm"), "test", key="llm-key")[0]
    embedding = act(service("embedding"), "test", key="llm-key")[0]

    credentials = ui.refresh_model_credentials_for_ui(
        llm, embedding, "gpt-6-luna", "gpt-6-luna", "gpt-6-luna",
        "gpt-6-luna", "gpt-6-luna", "gpt-6-luna", "text-embedding-3-small",
    )

    assert credentials == tuple(
        value
        for _ in range(6)
        for value in ("https://api.openai.com/v1", "llm-key")
    ) + ("https://api.openai.com/v1", "llm-key")


def test_experiment_llm_credentials_share_global_openai_settings(monkeypatch):
    monkeypatch.setattr(ui, "check_model_connection", lambda *_: None)
    experiment = service("experiment_llm")
    tested = act(
        experiment, "test", base="https://experiment.example/v1", key="experiment-secret",
    )[0]

    assert tested["kind"] == "experiment_llm"
    assert tested["profiles"]["OpenAI"]["api_key"] == "experiment-secret"
    assert tested["profiles"]["OpenAI"]["base_url"] == "https://experiment.example/v1"
    assert settings.load_service_settings("llm")["profiles"]["OpenAI"]["api_key"] == "experiment-secret"
    assert settings.load_service_settings("llm")["profiles"]["OpenAI"]["base_url"] == "https://experiment.example/v1"
    assert settings.load_service_settings("experiment_llm")["profiles"]["OpenAI"]["api_key"] == "experiment-secret"
    experiment_profile = settings.load_service_settings("experiment_llm")["profiles"]["OpenAI"]
    assert experiment_profile["base_url"] == "https://experiment.example/v1"


def test_global_api_settings_persist_once_and_are_shared(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    llm = service("llm")
    embedding = service("embedding")
    experiment = service("experiment_llm")

    outputs = ui.persist_global_api_settings_for_ui(
        "edit", llm, embedding, experiment,
        "https://llm.example/v1", "llm-key",
        *(["gpt-6-luna"] * 6), "text-embedding-3-small",
    )

    assert len(outputs) == 23
    assert settings.load_service_settings("llm")["profiles"]["OpenAI"]["api_key"] == "llm-key"
    assert settings.load_service_settings("experiment_llm")["profiles"]["OpenAI"]["api_key"] == "llm-key"
    assert settings.load_service_settings("embedding")["profiles"]["OpenAI"]["api_key"] == "llm-key"
    env_text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert 'MODEL_OPENAI_API_KEY="llm-key"' in env_text
    assert "EMBEDDING_OPENAI_API_" not in env_text
    assert "EXPERIMENT_MODEL_OPENAI_API_KEY" not in env_text


def test_only_openai_profiles_are_loaded_and_saved(tmp_path, monkeypatch):
    path = tmp_path / "model_settings.yaml"
    monkeypatch.setattr(settings, "MODEL_SETTINGS_PATH", path)
    document = settings._default_document()
    document["services"]["llm"]["profiles"]["Ollama"] = {"models": ["legacy"] * 6}
    document["services"]["embedding"]["profiles"]["Voyage"] = {"models": ["voyage-4"]}
    document["voyage_models"] = ["voyage-4"]
    path.write_text(yaml.safe_dump(document), encoding="utf-8")

    loaded = service("llm")
    assert loaded["active"] == "OpenAI"
    assert list(loaded["profiles"]) == ["OpenAI"]
    act(loaded, "edit")

    saved = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert list(saved["services"]["llm"]["profiles"]) == ["OpenAI"]
    assert list(saved["services"]["embedding"]["profiles"]) == ["OpenAI"]
    assert "voyage_models" not in saved


def test_connection_failure_and_credential_edit_do_not_revoke_openai_models(monkeypatch):
    monkeypatch.setattr(ui, "check_model_connection", lambda *args: None)
    connected = act(service(), "test", key="key")
    edited = act(connected[0], "edit", key="different")
    assert all(u["choices"] for u in edited[7:])
    def fail(*args):
        raise ValueError("連線失敗")
    monkeypatch.setattr(ui, "check_model_connection", fail)
    failed = act(connected[0], "test")
    assert failed[6] == "❌ 連線失敗"
    assert all(u["choices"] for u in failed[7:])


def test_custom_yaml_controls_allowlist_and_invalid_yaml_fails_closed(tmp_path, monkeypatch):
    path = tmp_path / "model_settings.yaml"
    monkeypatch.setattr(settings, "MODEL_SETTINGS_PATH", path)
    monkeypatch.setattr(ui, "check_model_connection", lambda *args: None)
    document = settings._default_document()
    document["openai_models"] = {"llm": ["custom-mini"], "embedding": ["custom-embed"]}
    path.write_text(yaml.safe_dump(document))
    connected = act(service(), "test")
    assert connected[7]["choices"] == choices("OpenAI", ["custom-mini"])
    path.write_text("openai_models: {}\n")
    failed = ui.service_action_for_ui(
        "test", "OpenAI", connected[0], connected[1], connected[2], [],
        *( ["custom-mini"] * settings.MODEL_COUNTS["llm"] ),
    )
    assert "設定檔無效" in failed[6]
    assert failed[7]["choices"] == []



def test_non_openai_provider_is_rejected():
    with pytest.raises(ui.gr.Error, match="僅支援 OpenAI"):
        act(service("embedding"), "test", provider="Voyage")

def test_saved_project_cannot_bypass_openai_allowlist_or_connection(monkeypatch):
    monkeypatch.setattr(ui, "check_model_connection", lambda *args: None)
    project = ui.create_project("saved")
    ui.save_project(project["project_id"], {"settings": {
        "model_endpoint": "https://api.openai.com/v1", "api_key": "",
        "graph_llm_model": "gpt-expensive", "extraction_llm_model": "gpt-4o-mini",
        "answer_model": "gpt-expensive", "graph_embedding_model": "other-embed",
    }})
    before = ui.load_project_with_services_for_ui(project["project_id"], service(), service("embedding"))
    assert before[8]["choices"]
    llm = act(service(), "test")[0]
    embed = act(service("embedding"), "test")[0]
    loaded = ui.load_project_with_services_for_ui(project["project_id"], llm, embed)
    assert loaded[8]["value"] == "gpt-4.1-mini"
    assert loaded[16]["value"] == "gpt-4o-mini"
    assert loaded[9]["value"] == "text-embedding-3-small"
    assert loaded[8]["choices"] == choices("OpenAI", settings.openai_models("llm"))
    assert loaded[9]["choices"] == choices("OpenAI", settings.openai_models("embedding"))


def test_new_project_load_defaults_llms_to_luna_and_parallelism_to_ten(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    project = ui.create_project("default-luna-project")
    loaded = ui.load_project_with_services_for_ui(
        project["project_id"], service(), service("embedding"),
    )

    assert loaded[8]["value"] == "gpt-6-luna"
    assert loaded[10]["value"] == "gpt-6-luna"
    assert loaded[16]["value"] == "gpt-6-luna"
    assert loaded[15] == 10
    assert loaded[17] == 10


def test_evaluation_cannot_restore_unapproved_models(monkeypatch):
    monkeypatch.setattr(ui, "check_model_connection", lambda *args: None)
    project = ui.create_project("saved")
    ui.save_project(project["project_id"], {"evaluation": {"preferences": {
        "generation_model": "unapproved", "test_model": "gpt-4o-mini",
    }}})
    state = act(service(), "test")[0]
    loaded = ui.load_evaluation_with_services_for_ui(project["project_id"], state)
    assert loaded[3]["value"] is None
    assert loaded[4]["value"] == "gpt-4o-mini"


def test_load_migrates_profiles_for_experiment_model_role(tmp_path, monkeypatch):
    document = settings._default_document()
    for profile in document["services"]["llm"]["profiles"].values():
        profile["models"] = ["legacy-model"] * 5
    path = tmp_path / "model_settings.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    monkeypatch.setattr(settings, "MODEL_SETTINGS_PATH", path)

    loaded = settings.load_service_settings("llm", dict(DEFAULTS))

    assert settings.MODEL_COUNTS["llm"] == 6
    assert loaded["profiles"]["OpenAI"]["models"] == ["legacy-model"] * 6
    assert list(loaded["profiles"]) == ["OpenAI"]


def test_profile_load_rejects_invalid_yaml_without_exposing_keys(tmp_path, monkeypatch):
    path = tmp_path / "model_settings.yaml"
    path.write_text("services:\n  llm:\n    secret: sensitive\n", encoding="utf-8")
    monkeypatch.setattr(settings, "MODEL_SETTINGS_PATH", path)
    with pytest.raises(ValueError, match="services.llm") as error:
        settings.load_service_settings("llm", dict(DEFAULTS))
    assert "sensitive" not in str(error.value)


def test_reload_restores_shared_profile_and_revokes_openai_connection(monkeypatch):
    monkeypatch.setattr(ui, "check_model_connection", lambda *args: None)
    act(service(), "test", key="llm-key")
    act(service("embedding"), "test", key="embedding-key")
    loaded = ui.reload_env_with_services_for_ui()
    assert loaded[3] == "OpenAI"
    assert loaded[6] == "embedding-key"
    assert loaded[17] == "OpenAI"
    assert loaded[20] == "embedding-key"
    assert loaded[11]["choices"]
    assert loaded[25]["choices"]


def test_invalid_yaml_after_successful_connection_revokes_models(tmp_path, monkeypatch):
    path = tmp_path / "model_settings.yaml"
    monkeypatch.setattr(settings, "MODEL_SETTINGS_PATH", path)
    monkeypatch.setattr(ui, "check_model_connection", lambda *args: None)
    document = settings._default_document()
    document["openai_models"] = {"llm": ["mini"], "embedding": ["embed"]}
    path.write_text(yaml.safe_dump(document))
    connected = act(service(), "test")
    path.write_text("openai_models: {}\n")
    failed = ui.service_action_for_ui("test", "OpenAI", connected[0], connected[1], connected[2], [], *(["mini"] * 5))
    assert "設定檔無效" in failed[6]
    assert failed[7]["choices"] == []



def test_preferred_llm_model_uses_openai_allowlist(monkeypatch):
    state = service("llm")
    assert settings.preferred_service_model(state) == settings.openai_models("llm")[0]
    state = service()
    assert settings.resolve_model_service(state, "gpt-6-luna") == (
        "https://api.openai.com/v1", "", "gpt-6-luna",
    )


def test_unavailable_model_cannot_be_routed():
    with pytest.raises(ValueError, match="未列入 OpenAI"):
        settings.resolve_model_service(service(), "unknown")
