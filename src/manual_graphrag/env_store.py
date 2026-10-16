from __future__ import annotations

import errno
import json
import os
import tempfile
from pathlib import Path
from threading import RLock


_ENV_LOCK = RLock()

ENV_KEYS = (
    "NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD",
    "MODEL_OPENAI_API_BASE", "MODEL_OPENAI_API_KEY",
)

# These keys belonged to the former mixed connection/model format. save_env removes
# them so an existing .env is migrated the next time settings are saved.
LEGACY_ENV_KEYS = {
    "MODEL_API_BASE", "MODEL_API_KEY", "BUILD_MODEL",
    "EMBEDDING_API_BASE", "EMBEDDING_API_KEY", "EMBEDDING_MODEL", "ANSWER_MODEL",
    "MODEL_SERVICE_PROFILES", "EMBEDDING_SERVICE_PROFILES",
    "MODEL_OLLAMA_API_BASE", "MODEL_OLLAMA_API_KEY",
    "EXPERIMENT_MODEL_OLLAMA_API_BASE", "EXPERIMENT_MODEL_OLLAMA_API_KEY",
    "EXPERIMENT_MODEL_OPENAI_API_BASE", "EXPERIMENT_MODEL_OPENAI_API_KEY",
    "EMBEDDING_OPENAI_API_BASE", "EMBEDDING_OPENAI_API_KEY",
    "EMBEDDING_OLLAMA_API_BASE", "EMBEDDING_OLLAMA_API_KEY",
    "EMBEDDING_VOYAGE_API_BASE", "EMBEDDING_VOYAGE_API_KEY",
}

DEFAULTS = {
    "NEO4J_URI": "bolt://localhost:7687",
    "NEO4J_USERNAME": "neo4j", "NEO4J_PASSWORD": "",
    "MODEL_OPENAI_API_BASE": "https://api.openai.com/v1", "MODEL_OPENAI_API_KEY": "",
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
    # Unquoted values may carry a shell-style trailing comment: KEY=value  # note
    for marker in (" #", "\t#"):
        if marker in value:
            value = value.split(marker, 1)[0].rstrip()
    return value


def _read_env_text(target: Path) -> str:
    """Decode .env files saved by Windows editors as well as plain UTF-8.

    Notepad may add a UTF-8 BOM (which would hide the first key) and
    PowerShell 5 ``>``/``Out-File`` writes UTF-16 with a BOM.
    """
    if target.is_dir():
        raise IsADirectoryError(
            f"{target} 是資料夾而不是檔案；Docker 掛載前請先在主機建立 .env（cp .env.example .env）"
        )
    data = target.read_bytes()
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    return data.decode("utf-8-sig")


def _env_key(raw_key: str) -> str:
    key = raw_key.strip()
    if key.startswith("export "):
        key = key[len("export "):].strip()
    return key


def load_env(path: str | Path = ".env") -> dict[str, str]:
    values = dict(DEFAULTS)
    target = Path(path)
    if not target.exists():
        return values
    legacy_embedding_values: dict[str, str] = {}
    present_keys: set[str] = set()
    for raw_line in _read_env_text(target).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        key = _env_key(key)
        present_keys.add(key)
        if key in ENV_KEYS:
            values[key] = _decode_value(raw_value)
        elif key in {"EMBEDDING_OPENAI_API_BASE", "EMBEDDING_OPENAI_API_KEY"}:
            legacy_embedding_values[key] = _decode_value(raw_value)
    if not values["MODEL_OPENAI_API_KEY"] and legacy_embedding_values.get("EMBEDDING_OPENAI_API_KEY"):
        values["MODEL_OPENAI_API_KEY"] = legacy_embedding_values["EMBEDDING_OPENAI_API_KEY"]
        if "MODEL_OPENAI_API_BASE" not in present_keys:
            values["MODEL_OPENAI_API_BASE"] = legacy_embedding_values.get(
                "EMBEDDING_OPENAI_API_BASE", values["MODEL_OPENAI_API_BASE"],
            )
    return values


def save_env(values: dict[str, object], path: str | Path = ".env") -> Path:
    with _ENV_LOCK:
        return _save_env(values, path)


def _save_env(values: dict[str, object], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    current_lines = _read_env_text(target).splitlines() if target.exists() else []
    current = load_env(target)
    current.update({key: str(value) for key, value in values.items() if key in ENV_KEYS})
    written: set[str] = set()
    output: list[str] = []
    for line in current_lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = _env_key(stripped.split("=", 1)[0])
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
    text = "\n".join(output).rstrip() + "\n"
    descriptor, temporary_name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.replace(temporary_name, target)
        except OSError as exc:
            # Docker single-file bind mounts (-v ./.env:/app/.env) cannot be
            # replaced by rename (EBUSY/EXDEV); fall back to rewriting in place.
            if exc.errno not in {errno.EBUSY, errno.EXDEV, errno.EPERM}:
                raise
            Path(temporary_name).unlink(missing_ok=True)
            with target.open("w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return target
