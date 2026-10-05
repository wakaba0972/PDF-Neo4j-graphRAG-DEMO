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
    "llm": ("OpenAI", "Ollama"),
    "experiment_llm": ("OpenAI", "Ollama"),
    "embedding": ("OpenAI", "Ollama", "Voyage"),
}
_SETTINGS_LOCK = RLock()
DEFAULT_OPENAI_MODELS = {
    "llm": ["gpt-4.1-mini", "gpt-4o-mini", "gpt-6-luna"],
    "embedding": ["text-embedding-3-small", "text-embedding-3-large"],
}
LEGACY_DEFAULT_OPENAI_MODELS = {
    "llm": ["gpt-4.1-mini", "gpt-4o-mini"],
}
DEFAULT_VOYAGE_MODELS = [
    "voyage-4-large", "voyage-4", "voyage-4-lite", "voyage-code-3",
    "voyage-finance-2", "voyage-law-2",
]
DEFAULT_SELECTIONS = {
    "llm": ["gpt-6-luna"] * 6,
    "experiment_llm": ["gpt-6-luna"] * 6,
    "embedding": ["text-embedding-3-small"],
}


def _default_document() -> dict[str, Any]:
    return {
        "openai_models": deepcopy(DEFAULT_OPENAI_MODELS),
        "voyage_models": deepcopy(DEFAULT_VOYAGE_MODELS),
        "services": {
            kind: {
                "active": "OpenAI",
                "profiles": {
                    provider: {
                        "rows": [],
                        "models": (
                            deepcopy(DEFAULT_SELECTIONS[kind]) if provider == "OpenAI"
                            else (["voyage-4"] if provider == "Voyage" else [None] * MODEL_COUNTS[kind])
                        ),
                    }
                    for provider in PROVIDERS_BY_KIND[kind]
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


def configured_models(kind: str, provider: str) -> list[str]:
    if provider == "OpenAI":
        return openai_models(kind)
    if provider == "Voyage" and kind == "embedding":
        try:
            models = _load_document()["voyage_models"]
            if not isinstance(models, list) or not models or any(not isinstance(model, str) or not model.strip() for model in models):
                raise ValueError()
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"模型設定檔無效：{MODEL_SETTINGS_PATH}（voyage_models）") from exc
        return list(dict.fromkeys(model.strip() for model in models))
    raise ValueError(f"{provider} 沒有固定模型清單")


def preferred_service_model(state: dict[str, Any]) -> str | None:
    providers = ("OpenAI", "Voyage", "Ollama") if state["kind"] == "embedding" else ("OpenAI", "Ollama")
    for provider in providers:
        if provider in state["profiles"]:
            models = provider_models(state, provider)
            if models:
                return models[0]
    return None


def _credentials(env: dict[str, str], kind: str, provider: str) -> tuple[str, str]:
    prefix = (
        "EXPERIMENT_MODEL" if kind == "experiment_llm"
        else ("MODEL" if kind == "llm" else "EMBEDDING")
    )
    provider_key = provider.upper()
    return env[f"{prefix}_{provider_key}_API_BASE"], env[f"{prefix}_{provider_key}_API_KEY"]


def load_service_settings(kind: str, env: dict[str, str] | None = None) -> dict[str, Any]:
    if kind not in MODEL_COUNTS:
        raise ValueError(f"不支援的模型服務種類：{kind}")
    env = load_env() if env is None else env
    document = _load_document()
    try:
        saved = (document.get("services") or {}).get(kind)
        if saved is None:
            saved = _default_document()["services"][kind]
        active = saved["active"]
        if active not in PROVIDERS_BY_KIND[kind]:
            raise ValueError()
        profiles = {}
        for provider in PROVIDERS_BY_KIND[kind]:
            profile = saved["profiles"][provider]
            rows, models = profile["rows"], profile["models"]
            if kind == "llm" and isinstance(models, list) and len(models) == 5:
                # Older profiles predate the experiment answer-model selector.
                models = [*models, models[4]]
            if not isinstance(models, list) or len(models) != MODEL_COUNTS[kind] or any(model is not None and not isinstance(model, str) for model in models):
                raise ValueError()
            if not isinstance(rows, list) or any(not isinstance(row, list) or len(row) != 2 or not isinstance(row[0], bool) or not isinstance(row[1], str) for row in rows):
                raise ValueError()
            base_url, api_key = _credentials(env, kind, provider)
            profiles[provider] = {"base_url": base_url, "api_key": api_key, "rows": deepcopy(rows), "models": deepcopy(models), "connected": False, "status": "請先測試連線。"}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"模型設定檔無效：{MODEL_SETTINGS_PATH}（services.{kind}）") from exc
    profiles["OpenAI"]["rows"] = []
    if "Voyage" in profiles:
        profiles["Voyage"]["rows"] = []
    return {"kind": kind, "active": active, "profiles": profiles}


def provider_models(state: dict[str, Any], provider: str) -> list[str]:
    profile = state["profiles"][provider]
    if not profile["connected"]:
        return []
    if provider in {"OpenAI", "Voyage"}:
        kind = "llm" if state["kind"] == "experiment_llm" else state["kind"]
        return configured_models(kind, provider)
    return list(dict.fromkeys(row[1] for row in profile["rows"] if row[0]))


def service_choices(state: dict[str, Any]) -> list[str]:
    return list(dict.fromkeys([model for provider in PROVIDERS_BY_KIND[state["kind"]] for model in provider_models(state, provider)]))


def service_choice_items(state: dict[str, Any]) -> list[tuple[str, str]]:
    return [(f"{provider}｜{model}", model) for provider in PROVIDERS_BY_KIND[state["kind"]] for model in provider_models(state, provider)]


def resolve_model_service(state: dict[str, Any], model: str | None) -> tuple[str, str, str]:
    if not model:
        raise ValueError("請先選擇模型")
    providers = [provider for provider in PROVIDERS_BY_KIND[state["kind"]] if model in provider_models(state, provider)]
    if not providers:
        raise ValueError(f"模型「{model}」目前不可用，請先完成服務連線或勾選模型")
    provider = state["active"] if state["active"] in providers else providers[0]
    profile = state["profiles"][provider]
    return profile["base_url"], profile["api_key"], model


def save_service_settings(state: dict[str, Any]) -> None:
    kind = state["kind"]
    with _SETTINGS_LOCK:
        document = _load_document()
        document.setdefault("openai_models", deepcopy(DEFAULT_OPENAI_MODELS))
        document.setdefault("voyage_models", deepcopy(DEFAULT_VOYAGE_MODELS))
        document.setdefault("services", {})[kind] = {
            "active": state["active"],
            "profiles": {
                provider: {key: deepcopy(profile[key]) for key in ("rows", "models")}
                for provider, profile in state["profiles"].items()
            },
        }
        _save_document(document)
        prefix = (
            "EXPERIMENT_MODEL" if kind == "experiment_llm"
            else ("MODEL" if kind == "llm" else "EMBEDDING")
        )
        connection_values = {}
        for provider, profile in state["profiles"].items():
            provider_key = provider.upper()
            connection_values[f"{prefix}_{provider_key}_API_BASE"] = profile["base_url"]
            connection_values[f"{prefix}_{provider_key}_API_KEY"] = profile["api_key"]
        save_env(connection_values)


def capture_service_settings(state: dict[str, Any], base_url: str, api_key: str, rows: list[list[Any]], models: list[str | None]) -> dict[str, Any]:
    state = deepcopy(state)
    profile = state["profiles"][state["active"]]
    changed = (profile["base_url"], profile["api_key"]) != (base_url, api_key)
    profile.update(base_url=base_url, api_key=api_key)
    if changed:
        profile.update(connected=False, rows=[], models=[None] * MODEL_COUNTS[state["kind"]], status="設定已變更，請重新測試連線或取得模型清單。")
    else:
        if state["active"] == "Ollama":
            checked = {row[1] for row in rows if len(row) == 2 and row[0] is True}
            profile["rows"] = [[model in checked, model] for _, model in profile["rows"]]
        if state["active"] == "Ollama" or profile["connected"]:
            allowed = service_choices(state) if state["active"] == "Ollama" else models
            profile["models"] = [model if model in allowed else None for model in models]
    return state


def restore_service_settings(state: dict[str, Any], base_url: str, api_key: str, models: dict[int, str | None]) -> dict[str, Any]:
    state = deepcopy(state)
    provider = next((name for name, profile in state["profiles"].items() if profile["base_url"] == base_url), state["active"])
    state["active"] = provider
    profile = state["profiles"][provider]
    if (profile["base_url"], profile["api_key"]) != (base_url, api_key):
        profile.update(base_url=base_url, api_key=api_key, connected=False, rows=[], status="請先測試連線或取得模型清單。")
    for index, model in models.items():
        profile["models"][index] = model
    return state
