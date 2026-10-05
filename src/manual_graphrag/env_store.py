from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from threading import RLock


_ENV_LOCK = RLock()

ENV_KEYS = (
    "NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD",
    "MODEL_OPENAI_API_BASE", "MODEL_OPENAI_API_KEY",
    "MODEL_OLLAMA_API_BASE", "MODEL_OLLAMA_API_KEY",
    "EXPERIMENT_MODEL_OPENAI_API_BASE", "EXPERIMENT_MODEL_OPENAI_API_KEY",
    "EXPERIMENT_MODEL_OLLAMA_API_BASE", "EXPERIMENT_MODEL_OLLAMA_API_KEY",
    "EMBEDDING_OPENAI_API_BASE", "EMBEDDING_OPENAI_API_KEY",
    "EMBEDDING_OLLAMA_API_BASE", "EMBEDDING_OLLAMA_API_KEY",
    "EMBEDDING_VOYAGE_API_BASE", "EMBEDDING_VOYAGE_API_KEY",
)

# These keys belonged to the former mixed connection/model format. save_env removes
# them so an existing .env is migrated the next time settings are saved.
LEGACY_ENV_KEYS = {
    "MODEL_API_BASE", "MODEL_API_KEY", "BUILD_MODEL",
    "EMBEDDING_API_BASE", "EMBEDDING_API_KEY", "EMBEDDING_MODEL", "ANSWER_MODEL",
    "MODEL_SERVICE_PROFILES", "EMBEDDING_SERVICE_PROFILES",
}

DEFAULTS = {
    "NEO4J_URI": "bolt://localhost:7687",
    "NEO4J_USERNAME": "neo4j", "NEO4J_PASSWORD": "",
    "MODEL_OPENAI_API_BASE": "https://api.openai.com/v1", "MODEL_OPENAI_API_KEY": "",
    "MODEL_OLLAMA_API_BASE": "http://localhost:11434/v1", "MODEL_OLLAMA_API_KEY": "",
    "EXPERIMENT_MODEL_OPENAI_API_BASE": "https://api.openai.com/v1",
    "EXPERIMENT_MODEL_OPENAI_API_KEY": "",
    "EXPERIMENT_MODEL_OLLAMA_API_BASE": "http://localhost:11434/v1",
    "EXPERIMENT_MODEL_OLLAMA_API_KEY": "",
    "EMBEDDING_OPENAI_API_BASE": "https://api.openai.com/v1", "EMBEDDING_OPENAI_API_KEY": "",
    "EMBEDDING_OLLAMA_API_BASE": "http://localhost:11434/v1", "EMBEDDING_OLLAMA_API_KEY": "",
    "EMBEDDING_VOYAGE_API_BASE": "https://api.voyageai.com/v1", "EMBEDDING_VOYAGE_API_KEY": "",
}


def _decode_value(raw: str) -> str:
    value = raw.strip()
    if value.startswith(("\"", "'")):
        try:
            return str(json.loads(value))
        except (json.JSONDecodeError, TypeError):
            if len(value) >= 2 and value[-1] == value[0]:
                return value[1:-1]
    return value


def load_env(path: str | Path = ".env") -> dict[str, str]:
    values = dict(DEFAULTS)
    target = Path(path)
    if not target.exists():
        return values
    for raw_line in target.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if key in ENV_KEYS:
            values[key] = _decode_value(raw_value)
    return values


def save_env(values: dict[str, object], path: str | Path = ".env") -> Path:
    with _ENV_LOCK:
        return _save_env(values, path)


def _save_env(values: dict[str, object], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    current_lines = target.read_text(encoding="utf-8").splitlines() if target.exists() else []
    current = load_env(target)
    current.update({key: str(value) for key, value in values.items() if key in ENV_KEYS})
    written: set[str] = set()
    output: list[str] = []
    for line in current_lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in LEGACY_ENV_KEYS:
                continue
            if key in ENV_KEYS:
                output.append(f"{key}={json.dumps(current[key], ensure_ascii=False)}")
                written.add(key)
                continue
        output.append(line)
    if output and output[-1] != "":
        output.append("")
    for key in ENV_KEYS:
        if key not in written:
            output.append(f"{key}={json.dumps(current[key], ensure_ascii=False)}")
    descriptor, temporary_name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write("\n".join(output).rstrip() + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return target
