from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import tempfile
from threading import RLock
from typing import Any

import yaml

from .env_store import load_env, save_env


MODEL_SETTINGS_PATH = Path("config/model_settings.yaml")
MODEL_COUNTS = {"llm": 6, "experiment_llm": 6, "embedding": 1}
PROVIDERS_BY_KIND = {
    "llm": ("OpenAI",),
    "experiment_llm": ("OpenAI",),
    "embedding": ("OpenAI",),
}
_SETTINGS_LOCK = RLock()
DEFAULT_OPENAI_MODELS = {
    "llm": ["gpt-4.1-mini", "gpt-4o-mini", "gpt-6-luna"],
    "embedding": ["text-embedding-3-small", "text-embedding-3-large"],
}
LEGACY_DEFAULT_OPENAI_MODELS = {
    "llm": ["gpt-4.1-mini", "gpt-4o-mini"],
}
DEFAULT_SELECTIONS = {
    "llm": ["gpt-6-luna"] * 6,
    "experiment_llm": ["gpt-6-luna"] * 6,
    "embedding": ["text-embedding-3-small"],
}


def _default_document() -> dict[str, Any]:
    return {
        "openai_models": deepcopy(DEFAULT_OPENAI_MODELS),
        "services": {
            kind: {
                "active": "OpenAI",
                "profiles": {
                    "OpenAI": {
                        "rows": [],
                        "models": deepcopy(DEFAULT_SELECTIONS[kind]),
                    }
                },
            } for kind in MODEL_COUNTS
        },
    }


def _load_document(path: Path | None = None) -> dict[str, Any]:
    path = MODEL_SETTINGS_PATH if path is None else path
    if not path.exists():
        return _default_document()
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError()
        return data
    except (OSError, yaml.YAMLError, ValueError) as exc:
        raise ValueError(f"模型設定檔無效：{path}") from exc


def _save_document(data: dict[str, Any], path: Path | None = None) -> Path:
    path = MODEL_SETTINGS_PATH if path is None else path
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            yaml.safe_dump(data, handle, allow_unicode=True, sort_keys=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return path


def openai_models(kind: str) -> list[str]:
    try:
        models = _load_document()["openai_models"][kind]
        if not isinstance(models, list) or not models or any(not isinstance(model, str) or not model.strip() for model in models):
            raise ValueError()
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"模型設定檔無效：{MODEL_SETTINGS_PATH}（openai_models.{kind}）") from exc
    normalized = list(dict.fromkeys(model.strip() for model in models))
    # Upgrade the built-in legacy list without overriding intentional custom
    # allowlists saved by the user.
    if normalized == LEGACY_DEFAULT_OPENAI_MODELS.get(kind):
        return list(DEFAULT_OPENAI_MODELS[kind])
    return normalized


def configured_models(kind: str) -> list[str]:
    return openai_models(kind)


def preferred_service_model(state: dict[str, Any]) -> str | None:
    models = service_choices(state)
    return models[0] if models else None


def _credentials(env: dict[str, str], kind: str) -> tuple[str, str]:
    prefix = "EMBEDDING" if kind == "embedding" else "MODEL"
    return env[f"{prefix}_OPENAI_API_BASE"], env[f"{prefix}_OPENAI_API_KEY"]


def load_service_settings(kind: str, env: dict[str, str] | None = None) -> dict[str, Any]:
    if kind not in MODEL_COUNTS:
        raise ValueError(f"不支援的模型服務種類：{kind}")
    env = load_env() if env is None else env
    document = _load_document()
    try:
        services = document.get("services", {})
        if not isinstance(services, dict):
            raise ValueError()
        saved = services.get(kind)
        if saved is None:
            if kind in services:
                raise ValueError()
            saved = _default_document()["services"][kind]
        if not isinstance(saved, dict):
            raise ValueError()
        saved_profiles = saved.get("profiles", {})
        if not isinstance(saved_profiles, dict):
            raise ValueError()
        if "profiles" not in saved or "OpenAI" not in saved_profiles:
            raise ValueError()
        saved_profile = saved_profiles.get("OpenAI", {})
        if saved_profile is None:
            saved_profile = {}
        if not isinstance(saved_profile, dict):
            raise ValueError()
        models = saved_profile.get("models", DEFAULT_SELECTIONS[kind])
        if kind == "llm" and isinstance(models, list) and len(models) == 5:
            # Older profiles predate the experiment answer-model selector.
            models = [*models, models[4]]
        if not isinstance(models, list) or len(models) != MODEL_COUNTS[kind] or any(model is not None and not isinstance(model, str) for model in models):
            raise ValueError()
        base_url, api_key = _credentials(env, kind)
        profiles = {"OpenAI": {
            "base_url": base_url, "api_key": api_key, "rows": [],
            "models": deepcopy(models), "connected": False,
            "status": "尚未測試連線；此測試僅供診斷，不影響後續操作。",
        }}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"模型設定檔無效：{MODEL_SETTINGS_PATH}（services.{kind}）") from exc
    return {"kind": kind, "active": "OpenAI", "profiles": profiles}


def service_choices(state: dict[str, Any]) -> list[str]:
    kind = "llm" if state["kind"] == "experiment_llm" else state["kind"]
    try:
        return configured_models(kind)
    except ValueError:
        return []


def service_choice_items(state: dict[str, Any]) -> list[tuple[str, str]]:
    return [(f"OpenAI｜{model}", model) for model in service_choices(state)]


def resolve_model_service(state: dict[str, Any], model: str | None) -> tuple[str, str, str]:
    if not model:
        raise ValueError("請先選擇模型")
    if model not in service_choices(state):
        raise ValueError(f"模型「{model}」未列入 OpenAI 模型允許清單")
    profile = state["profiles"]["OpenAI"]
    return profile["base_url"], profile["api_key"], model


def save_service_settings(state: dict[str, Any]) -> None:
    kind = state["kind"]
    with _SETTINGS_LOCK:
        document = _load_document()
        document.setdefault("openai_models", deepcopy(DEFAULT_OPENAI_MODELS))
        document.pop("voyage_models", None)
        for saved_kind, saved_service in (document.get("services") or {}).items():
            if isinstance(saved_service, dict):
                openai_profile = (saved_service.get("profiles") or {}).get("OpenAI")
                saved_service["active"] = "OpenAI"
                saved_service["profiles"] = {"OpenAI": openai_profile or {
                    "rows": [], "models": deepcopy(DEFAULT_SELECTIONS.get(saved_kind, [])),
                }}
        document.setdefault("services", {})[kind] = {
            "active": "OpenAI",
            "profiles": {"OpenAI": {
                "rows": [], "models": deepcopy(state["profiles"]["OpenAI"]["models"]),
            }},
        }
        _save_document(document)
        prefix = "EMBEDDING" if kind == "embedding" else "MODEL"
        profile = state["profiles"]["OpenAI"]
        save_env({f"{prefix}_OPENAI_API_BASE": profile["base_url"],
                  f"{prefix}_OPENAI_API_KEY": profile["api_key"]})


def capture_service_settings(state: dict[str, Any], base_url: str, api_key: str, rows: list[list[Any]], models: list[str | None]) -> dict[str, Any]:
    state = deepcopy(state)
    state["active"] = "OpenAI"
    profile = state["profiles"]["OpenAI"]
    changed = (profile["base_url"], profile["api_key"]) != (base_url, api_key)
    profile.update(base_url=base_url, api_key=api_key)
    if changed:
        profile.update(connected=False, status="設定已變更；連線測試可用於診斷，執行請求時會驗證連線。")
    if models:
        allowed = service_choices(state)
        profile["models"] = [model if model in allowed else None for model in models]
    return state


def restore_service_settings(state: dict[str, Any], base_url: str, api_key: str, models: dict[int, str | None]) -> dict[str, Any]:
    state = deepcopy(state)
    state["active"] = "OpenAI"
    profile = state["profiles"]["OpenAI"]
    if (profile["base_url"], profile["api_key"]) != (base_url, api_key):
        profile.update(base_url=base_url, api_key=api_key, connected=False, status="設定已變更；連線測試可用於診斷，執行請求時會驗證連線。")
    for index, model in models.items():
        profile["models"][index] = model
    return state
