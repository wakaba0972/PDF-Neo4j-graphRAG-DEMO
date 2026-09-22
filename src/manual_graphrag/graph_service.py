from __future__ import annotations

import hashlib
import http.client
import json
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable

from .chunking import TextChunk
from .env_store import load_env

SCHEMA_CONTEXT_LIMIT = 30_000
SCHEMA_MERGE_LIMIT = 12_000
SCHEMA_DESCRIPTION_LIMIT = 120
EXTRACTION_BATCH_LIMIT = 12_000
RATE_LIMIT_MAX_RETRIES = 6
RATE_LIMIT_BASE_DELAY_SECONDS = 2.0
RATE_LIMIT_MAX_DELAY_SECONDS = 30.0
NETWORK_MAX_RETRIES = 2
NETWORK_RETRY_BASE_DELAY_SECONDS = 0.5
CHAT_COMPLETION_TIMEOUT_SECONDS = 600


class RunCancelled(Exception):
    """Raised when a user stops a running schema-planning or extraction job."""


class RunControl:
    """Cooperative stop/pause signal shared between a running job and its UI buttons.

    Checked at safe points (before a batch starts, during rate-limit backoff);
    an already in-flight HTTP request cannot be interrupted mid-call.
    """

    def __init__(self) -> None:
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()

    def __deepcopy__(self, memo: dict[int, Any]) -> "RunControl":
        # Gradio deepcopies a gr.State's initial value per browser session;
        # threading.Event isn't picklable, and each session needs its own
        # independent, freshly-reset control anyway.
        return RunControl()

    def reset(self) -> None:
        self._stop_event.clear()
        self._pause_event.clear()

    def request_stop(self) -> None:
        self._stop_event.set()

    def toggle_pause(self) -> bool:
        if self._pause_event.is_set():
            self._pause_event.clear()
        else:
            self._pause_event.set()
        return self._pause_event.is_set()

    @property
    def is_paused(self) -> bool:
        return self._pause_event.is_set()

    def check(self) -> None:
        if self._stop_event.is_set():
            raise RunCancelled("使用者已停止")

    def wait_if_paused(self) -> None:
        while self._pause_event.is_set():
            self.check()
            time.sleep(0.2)

    def sleep(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while True:
            self.check()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(0.2, remaining))


@dataclass(frozen=True)
class SchemaPlan:
    schema: dict[str, Any]
    analyzed_chunks: int
    total_chunks: int
    batch_count: int
    merge_rounds: int


def schema_planning_signature(
    chunks: list[TextChunk],
    schema_granularity: str,
    llm_model: str,
    temperature: float,
) -> str:
    """Fingerprint of the inputs that determine schema-planning batches/groups.

    Used to tell whether a saved partial-progress checkpoint still applies, or
    the chunk selection/model/settings changed since it was recorded.
    """
    fingerprint = {
        "granularity": schema_granularity,
        "model": llm_model,
        "temperature": temperature,
        "chunks": [
            [chunk.number, chunk.document, list(chunk.pages), len(chunk.text)]
            for chunk in chunks
        ],
    }
    payload = json.dumps(fingerprint, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _attach_schema_planning_resume(
    exc: BaseException,
    *,
    signature: str,
    stage: str,
    candidates: list[dict[str, Any]] | None,
    succeeded: dict[int, dict[str, Any]],
    total: int | None = None,
) -> None:
    exc.schema_planning_resume = {  # type: ignore[attr-defined]
        "signature": signature,
        "stage": stage,
        "candidates": candidates,
        "succeeded": {str(index): schema for index, schema in succeeded.items()},
        "total": total,
    }


@dataclass(frozen=True)
class GraphExtraction:
    entities: list[dict[str, Any]]
    relationships: list[dict[str, Any]]
    processed_chunks: int


def _api_url(base_url: str, resource: str) -> str:
    base = base_url.strip().rstrip("/")
    if not base:
        raise ValueError("請先填寫模型 API Base URL")
    return f"{base}/{resource.lstrip('/')}"


def _rate_limit_retry_delay(exc: urllib.error.HTTPError, attempt: int) -> float:
    retry_after = exc.headers.get("Retry-After") if exc.headers else None
    if retry_after:
        try:
            return max(float(retry_after), 0.5)
        except ValueError:
            pass
    return min(RATE_LIMIT_BASE_DELAY_SECONDS * (2**attempt), RATE_LIMIT_MAX_DELAY_SECONDS)


def _is_ollama_chat_url(url: str) -> bool:
    ollama_base_url = load_env().get("MODEL_OLLAMA_API_BASE", "").strip()
    return bool(
        ollama_base_url
        and url.rstrip("/") == _api_url(ollama_base_url, "chat/completions").rstrip("/")
    )


def _read_streamed_chat_response(response: Any) -> dict[str, Any]:
    content_parts: list[str] = []
    finish_reason = ""
    received_event = False
    completed = False
    for raw_line in response:
        try:
            line = raw_line.decode("utf-8").strip()
        except UnicodeDecodeError as exc:
            raise ValueError("模型串流回傳包含無效文字編碼") from exc
        if not line or line.startswith(":") or not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            completed = True
            break
        try:
            event = json.loads(data)
            choice = event["choices"][0]
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise ValueError("模型串流回傳格式不正確") from exc
        received_event = True
        delta = choice.get("delta") or {}
        part = delta.get("content")
        if isinstance(part, str):
            content_parts.append(part)
        if choice.get("finish_reason"):
            finish_reason = str(choice["finish_reason"])
            completed = True
    if not received_event:
        raise ValueError("模型串流沒有回傳任何內容")
    if not completed:
        raise ConnectionError("模型串流在完成前中斷")
    return {
        "choices": [{
            "message": {"content": "".join(content_parts)},
            "finish_reason": finish_reason,
        }]
    }


def _post_json(
    url: str,
    payload: dict[str, Any],
    api_key: str,
    timeout: int = CHAT_COMPLETION_TIMEOUT_SECONDS,
    on_retry: Callable[[int, float], None] | None = None,
    control: RunControl | None = None,
) -> dict[str, Any]:
    stream = _is_ollama_chat_url(url)
    if stream:
        payload = {**payload, "stream": True}
    headers = {"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}
    if stream:
        headers["Accept"] = "text/event-stream"
    if api_key.strip():
        headers["Authorization"] = f"Bearer {api_key.strip()}"
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    rate_limit_retries = 0
    network_retries = 0
    while True:
        if control:
            control.check()
            control.wait_if_paused()
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                result = (
                    _read_streamed_chat_response(response)
                    if stream else json.loads(response.read().decode("utf-8"))
                )
            break
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and rate_limit_retries < RATE_LIMIT_MAX_RETRIES:
                delay = _rate_limit_retry_delay(exc, rate_limit_retries)
                rate_limit_retries += 1
                exc.close()
                if on_retry:
                    on_retry(rate_limit_retries, delay)
                if control:
                    control.sleep(delay)
                else:
                    time.sleep(delay)
                continue
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise ValueError(f"模型 API 回傳 HTTP {exc.code}：{detail}") from exc
        except (
            urllib.error.URLError,
            TimeoutError,
            http.client.IncompleteRead,
            ConnectionError,
        ) as exc:
            if network_retries < NETWORK_MAX_RETRIES:
                delay = NETWORK_RETRY_BASE_DELAY_SECONDS * (2**network_retries)
                network_retries += 1
                if on_retry:
                    on_retry(network_retries, delay)
                if control:
                    control.sleep(delay)
                else:
                    time.sleep(delay)
                continue
            raise ValueError(
                f"模型 API 連線或回應中斷，重試 {NETWORK_MAX_RETRIES} 次後仍失敗：{exc}"
            ) from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("模型 API 回傳的不是有效 JSON") from exc
    if not isinstance(result, dict):
        raise ValueError("模型 API 回傳格式不正確")
    return result


def _get_json(url: str, api_key: str, timeout: int = 30) -> Any:
    headers = {"User-Agent": "Mozilla/5.0"}
    if api_key.strip():
        headers["Authorization"] = f"Bearer {api_key.strip()}"
    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise ValueError(f"模型服務回傳 HTTP {exc.code}：{detail}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ValueError(f"模型服務連線失敗：{exc}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("模型服務回傳的不是有效 JSON") from exc


def check_model_connection(base_url: str, api_key: str) -> None:
    _get_json(_api_url(base_url, "models"), api_key)


def list_models(base_url: str, api_key: str) -> list[str]:
    """List model ids advertised by an OpenAI-compatible /models endpoint.

    Works against OpenAI, Ollama (`http://localhost:11434/v1`), and any other
    provider implementing the same schema, letting the UI offer local Ollama
    model names instead of requiring users to type them from memory.
    """
    payload = _get_json(_api_url(base_url, "models"), api_key)
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        raise ValueError("模型服務回傳格式不正確，缺少 data 陣列")
    models = sorted(
        {
            str(item["id"]).strip()
            for item in data
            if isinstance(item, dict) and str(item.get("id", "")).strip()
        }
    )
    if not models:
        raise ValueError("模型服務目前沒有回傳任何可用模型")
    return models


def _extract_json_text(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()

    decoder = json.JSONDecoder()
    for index, character in enumerate(candidate):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(candidate[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("模型未回傳有效的 JSON 結果")


def _chat_response_content(response: dict[str, Any]) -> tuple[str, str]:
    try:
        choice = response["choices"][0]
        content = choice["message"]["content"]
        finish_reason = str(choice.get("finish_reason") or "")
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("模型 API 回傳缺少 choices/message/content") from exc
    if not isinstance(content, str):
        raise ValueError("模型回傳內容格式不正確")
    return content, finish_reason


def _chat_json(
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0,
    validator: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    on_retry: Callable[[int, float], None] | None = None,
    control: RunControl | None = None,
) -> dict[str, Any]:
    if not 0 <= temperature <= 2:
        raise ValueError("temperature 必須介於 0 到 2")
    if not model.strip():
        raise ValueError("請選擇 LLM 模型")

    url = _api_url(base_url, "chat/completions")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    def request_json(request_messages: list[dict[str, str]], request_temperature: float):
        return _post_json(
            url,
            {
                "model": model.strip(),
                "temperature": request_temperature,
                "messages": request_messages,
            },
            api_key,
            on_retry=on_retry,
            control=control,
        )

    def parse_and_validate(raw_content: str) -> dict[str, Any]:
        parsed = _extract_json_text(raw_content)
        return validator(parsed) if validator else parsed

    response = request_json(messages, temperature)
    content, finish_reason = _chat_response_content(response)
    try:
        return parse_and_validate(content)
    except ValueError as first_error:
        repair_messages = [
            *messages,
            {"role": "assistant", "content": content},
            {
                "role": "user",
                "content": (
                    "上一個回覆無法通過 JSON 解析或必要結構驗證。"
                    f"驗證錯誤：{first_error}。請修正並只輸出一個完整 JSON 物件，"
                    "不得加入 Markdown code fence 或說明文字，也不得省略必要欄位或回傳空陣列。"
                ),
            },
        ]
        repaired_response = request_json(repair_messages, 0)
        repaired_content, repaired_finish_reason = _chat_response_content(repaired_response)
        try:
            return parse_and_validate(repaired_content)
        except ValueError as exc:
            if finish_reason in {"length", "max_tokens"} or repaired_finish_reason in {
                "length",
                "max_tokens",
            }:
                raise ValueError(
                    "模型 JSON 連續兩次無法通過解析或結構驗證，且模型因長度限制截斷；"
                    "請提高 Ollama context length 或減少 Schema 類型數量"
                ) from exc
            raise ValueError(
                f"模型 JSON 連續兩次無法通過解析或結構驗證：{exc}"
            ) from first_error


def _chunk_label(chunk: TextChunk) -> str:
    pages = ",".join(map(str, chunk.pages))
    return f"[CHUNK {chunk.number}; PAGES {pages}]\n{chunk.text}"


def _chunk_batches(
    chunks: list[TextChunk], limit: int
) -> list[list[TextChunk]]:
    batches: list[list[TextChunk]] = []
    current: list[TextChunk] = []
    size = 0
    for chunk in chunks:
        chunk_size = len(_chunk_label(chunk)) + 2
        if current and size + chunk_size > limit:
            batches.append(current)
            current = []
            size = 0
        current.append(chunk)
        size += chunk_size
    if current:
        batches.append(current)
    return batches


def _schema_groups(
    schemas: list[dict[str, Any]], limit: int
) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    size = 0
    for schema in schemas:
        schema_size = len(json.dumps(schema, ensure_ascii=False)) + 2
        if current and size + schema_size > limit:
            groups.append(current)
            current = []
            size = 0
        current.append(schema)
        size += schema_size
    if current:
        groups.append(current)
    if len(groups) == len(schemas) and len(schemas) > 1:
        return [schemas[index : index + 2] for index in range(0, len(schemas), 2)]
    return groups



def _compact_schema(schema: dict[str, Any]) -> dict[str, Any]:
    schema = validate_schema(schema)

    def compact_description(value: Any) -> str:
        return " ".join(str(value or "").split())[:SCHEMA_DESCRIPTION_LIMIT]

    entity_types = [
        {
            "name": str(item["name"]).strip(),
            "description": compact_description(item.get("description")),
        }
        for item in schema["entity_types"]
    ]
    relationship_types = []
    for item in schema["relationship_types"]:
        relationship = {
            "name": str(item["name"]).strip(),
            "description": compact_description(item.get("description")),
        }
        for field in ("source_types", "target_types"):
            values = item.get(field, [])
            relationship[field] = (
                list(
                    dict.fromkeys(
                        str(value).strip() for value in values if str(value).strip()
                    )
                )
                if isinstance(values, list)
                else []
            )
        relationship_types.append(relationship)
    return {
        "entity_types": entity_types,
        "relationship_types": relationship_types,
    }



def validate_schema(schema: dict[str, Any]) -> dict[str, Any]:
    entity_types = schema.get("entity_types")
    relationship_types = schema.get("relationship_types")
    if not isinstance(entity_types, list) or not entity_types:
        raise ValueError("schema 必須包含非空的 entity_types 陣列")
    if not isinstance(relationship_types, list) or not relationship_types:
        raise ValueError("schema 必須包含非空的 relationship_types 陣列")
    for group_name, values in (
        ("entity_types", entity_types),
        ("relationship_types", relationship_types),
    ):
        for value in values:
            if not isinstance(value, dict) or not str(value.get("name", "")).strip():
                raise ValueError(f"{group_name} 每一項都必須包含 name")
    return schema


def plan_graph_schema(
    base_url: str,
    api_key: str,
    llm_model: str,
    chunks: list[TextChunk],
    temperature: float = 0,
    progress_callback: Callable[[float | None, str], None] | None = None,
    schema_granularity: str = "平衡",
    max_concurrent_requests: int = 3,
    control: RunControl | None = None,
    resume_state: dict[str, Any] | None = None,
) -> SchemaPlan:
    if not chunks:
        raise ValueError("請先在 PDF 頁面解析並產生 chunks")
    granularity_guidance = {
        "粗略": "只保留最核心的跨章節概念，積極合併上下位與近義類型。",
        "平衡": "保留支援主要查詢所需的通用類型，合併過細的上下位與近義類型。",
        "詳細": "可保留有明確查詢價值的專業子類型，但仍不得把具體實例當成類型。",
    }
    if schema_granularity not in granularity_guidance:
        raise ValueError("Schema 粒度必須是粗略、平衡或詳細")
    if int(max_concurrent_requests) < 1:
        raise ValueError("最大並行請求數必須大於 0")
    max_concurrent_requests = int(max_concurrent_requests)

    signature = schema_planning_signature(
        chunks, schema_granularity, llm_model, temperature
    )
    resume = (
        resume_state
        if resume_state and resume_state.get("signature") == signature
        else None
    )

    planning_rules = (
        f"Schema 粒度：{schema_granularity}。{granularity_guidance[schema_granularity]}"
        "只建立可重複使用、可泛化的類型；具體名稱、型號、編號、人物、組織或章節"
        "應在抽取階段成為實體，不得直接成為類型。只有重複出現且具有獨立查詢或關係"
        "價值的概念才建立類型；相近概念應合併到較高階類型。"
    )
    batches = _chunk_batches(chunks, SCHEMA_CONTEXT_LIMIT)
    candidates: list[dict[str, Any] | None] = [None] * len(batches)
    resume_merge_candidates: list[dict[str, Any]] | None = None
    resume_merge_succeeded: dict[int, dict[str, Any]] = {}

    if resume and resume.get("stage") == "batches":
        for key, schema in (resume.get("succeeded") or {}).items():
            index = int(key)
            if 0 <= index < len(candidates):
                candidates[index] = schema
    elif resume and resume.get("stage") == "merge":
        resume_merge_candidates = list(resume.get("candidates") or [])
        resume_merge_succeeded = {
            int(key): schema
            for key, schema in (resume.get("succeeded") or {}).items()
        }

    def plan_batch(index: int, batch: list[TextChunk]) -> dict[str, Any]:
        if control:
            control.check()
            control.wait_if_paused()
        context = "\n\n".join(_chunk_label(chunk) for chunk in batch)

        def report_retry(attempt: int, delay: float) -> None:
            if progress_callback:
                progress_callback(
                    None,
                    f"第 {index + 1} / {len(batches)} 批遇到速率限制（HTTP 429），"
                    f"{delay:.1f} 秒後自動重試（第 {attempt} / {RATE_LIMIT_MAX_RETRIES} 次）…",
                )

        candidate = _chat_json(
            base_url,
            api_key,
            llm_model,
            "你是知識圖譜 schema 設計專家。只輸出 JSON，不要 Markdown 或說明文字。",
            "請根據這一批文件內容提出候選實體與關係類型。"
            f"{planning_rules}"
            "名稱使用英文大寫 snake case，說明使用繁體中文。"
            "輸出格式：{\"entity_types\":[{\"name\":\"...\",\"description\":\"...\"}],"
            "\"relationship_types\":[{\"name\":\"...\",\"description\":\"...\","
            "\"source_types\":[\"...\"],\"target_types\":[\"...\"]}]}。\n\n"
            f"文件 chunks：\n{context}",
            temperature,
            validate_schema,
            on_retry=report_retry,
            control=control,
        )
        return _compact_schema(candidate)

    if resume_merge_candidates is None:
        analyzed = sum(
            len(batch)
            for batch, candidate in zip(batches, candidates)
            if candidate is not None
        )
        executor = ThreadPoolExecutor(max_workers=max_concurrent_requests)
        futures = {}
        pending_indices = [
            index for index, candidate in enumerate(candidates) if candidate is None
        ]
        next_pending = 0

        def submit_available_batches() -> None:
            nonlocal next_pending
            while (
                next_pending < len(pending_indices)
                and len(futures) < max_concurrent_requests
            ):
                index = pending_indices[next_pending]
                batch = batches[index]
                future = executor.submit(plan_batch, index, batch)
                futures[future] = (index, batch)
                next_pending += 1

        submit_available_batches()
        try:
            while futures:
                future = next(as_completed(tuple(futures)))
                index, batch = futures.pop(future)
                try:
                    candidates[index] = future.result()
                except ValueError as exc:
                    wrapped = ValueError(
                        f"Schema 規劃第 {index + 1} / {len(batches)} 批失敗：{exc}"
                    )
                    _attach_schema_planning_resume(
                        wrapped,
                        signature=signature,
                        stage="batches",
                        candidates=None,
                        succeeded={
                            i: c for i, c in enumerate(candidates) if c is not None
                        },
                        total=len(batches),
                    )
                    raise wrapped from exc
                analyzed += len(batch)
                if progress_callback:
                    progress_callback(
                        0.75 * analyzed / len(chunks),
                        f"已完成 {sum(item is not None for item in candidates)} / "
                        f"{len(batches)} 批（已分析 {analyzed} / {len(chunks)} chunks）",
                    )
                submit_available_batches()
        except BaseException as exc:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            if isinstance(exc, RunCancelled) and not hasattr(
                exc, "schema_planning_resume"
            ):
                _attach_schema_planning_resume(
                    exc,
                    signature=signature,
                    stage="batches",
                    candidates=None,
                    succeeded={
                        i: c for i, c in enumerate(candidates) if c is not None
                    },
                    total=len(batches),
                )
            raise
        else:
            executor.shutdown(wait=True)
        candidates = [candidate for candidate in candidates if candidate is not None]
    else:
        candidates = resume_merge_candidates
        if progress_callback:
            progress_callback(
                0.75, f"已延續先前進度，略過已完成的 {len(batches)} 批規劃"
            )

    merge_rounds = 0
    pending_merge_prefill = resume_merge_succeeded
    while len(candidates) > 1:
        if control:
            try:
                control.check()
            except RunCancelled as exc:
                _attach_schema_planning_resume(
                    exc,
                    signature=signature,
                    stage="merge",
                    candidates=candidates,
                    succeeded={},
                )
                raise
        merge_rounds += 1
        round_input = candidates
        groups = _schema_groups(candidates, SCHEMA_MERGE_LIMIT)
        merged: list[dict[str, Any] | None] = [None] * len(groups)
        prefill = pending_merge_prefill if merge_rounds == 1 else {}
        pending_merge_prefill = {}
        for index, schema in prefill.items():
            if 0 <= index < len(merged):
                merged[index] = schema

        def merge_group(index: int, group: list[dict[str, Any]]) -> dict[str, Any]:
            if control:
                control.check()
                control.wait_if_paused()
            if len(group) == 1:
                return group[0]

            def report_retry(attempt: int, delay: float) -> None:
                if progress_callback:
                    progress_callback(
                        None,
                        f"第 {merge_rounds} 輪整合第 {index + 1} / {len(groups)} 組"
                        f"遇到速率限制（HTTP 429），{delay:.1f} 秒後自動重試"
                        f"（第 {attempt} / {RATE_LIMIT_MAX_RETRIES} 次）…",
                    )

            result = _chat_json(
                base_url,
                api_key,
                llm_model,
                "你是知識圖譜 schema 整合專家。只輸出 JSON，不要 Markdown 或說明文字。",
                "合併以下候選 Schema；這不是候選類型的聯集。"
                f"{planning_rules}"
                "請積極去除重複、統一同義名稱並合併上下位與近義類型。"
                "只保留類型名稱、簡短說明，以及關係的 source_types "
                "與 target_types；不要輸出範例、屬性或其他欄位。"
                "輸出格式必須維持 entity_types 與 relationship_types。\n\n"
                f"候選 Schema：\n{json.dumps(group, ensure_ascii=False)}",
                temperature,
                validate_schema,
                on_retry=report_retry,
                control=control,
            )
            return _compact_schema(result)

        completed_groups = sum(1 for item in merged if item is not None)
        executor = ThreadPoolExecutor(max_workers=max_concurrent_requests)
        futures = {
            executor.submit(merge_group, index, group): index
            for index, group in enumerate(groups)
            if merged[index] is None
        }
        try:
            for future in as_completed(futures):
                index = futures[future]
                try:
                    merged[index] = future.result()
                except ValueError as exc:
                    wrapped = ValueError(
                        f"Schema 第 {merge_rounds} 輪整合第 "
                        f"{index + 1} / {len(groups)} 組失敗：{exc}"
                    )
                    _attach_schema_planning_resume(
                        wrapped,
                        signature=signature,
                        stage="merge",
                        candidates=round_input,
                        succeeded={
                            i: c for i, c in enumerate(merged) if c is not None
                        },
                        total=len(groups),
                    )
                    raise wrapped from exc
                completed_groups += 1
                if progress_callback:
                    progress_callback(
                        min(
                            0.99,
                            0.75
                            + 0.24 * (1 - 0.5 ** (merge_rounds - 1))
                            + 0.24
                            * (0.5**merge_rounds)
                            * completed_groups
                            / len(groups),
                        ),
                        f"第 {merge_rounds} 輪 Schema 整合："
                        f"已完成 {completed_groups} / {len(groups)} 組",
                    )
        except BaseException as exc:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            if isinstance(exc, RunCancelled) and not hasattr(
                exc, "schema_planning_resume"
            ):
                _attach_schema_planning_resume(
                    exc,
                    signature=signature,
                    stage="merge",
                    candidates=round_input,
                    succeeded={i: c for i, c in enumerate(merged) if c is not None},
                    total=len(groups),
                )
            raise
        else:
            executor.shutdown(wait=True)
        candidates = [item for item in merged if item is not None]
    if progress_callback:
        progress_callback(1.0, f"已分析全部 {len(chunks)} / {len(chunks)} chunks")
    return SchemaPlan(
        candidates[0], len(chunks), len(chunks), len(batches), merge_rounds
    )


def _batches(chunks: list[TextChunk]) -> list[list[TextChunk]]:
    return _chunk_batches(chunks, EXTRACTION_BATCH_LIMIT)


def extract_graph(
    base_url: str,
    api_key: str,
    llm_model: str,
    chunks: list[TextChunk],
    schema: dict[str, Any],
    temperature: float = 0,
    max_concurrent_requests: int = 3,
    progress_callback: Callable[[float | None, str], None] | None = None,
    control: RunControl | None = None,
) -> GraphExtraction:
    if not chunks:
        raise ValueError("請先在 PDF 頁面解析並產生 chunks")
    if int(max_concurrent_requests) < 1:
        raise ValueError("最大並行請求數必須至少為 1")
    max_concurrent_requests = int(max_concurrent_requests)
    schema = validate_schema(schema)
    allowed_entity_types = {
        str(item["name"]).casefold() for item in schema["entity_types"]
    }
    allowed_relationship_types = {
        str(item["name"]).casefold() for item in schema["relationship_types"]
    }
    entities: dict[tuple[str, str], dict[str, Any]] = {}
    relationships: dict[tuple[str, str, str], dict[str, Any]] = {}
    chunk_lookup = {chunk.number: chunk for chunk in chunks}

    batches = _batches(chunks)
    processed_chunks = 0
    if progress_callback:
        progress_callback(0.0, f"準備抽取 {len(chunks)} 個 chunks")

    def extract_batch(index: int, batch: list[TextChunk]) -> dict[str, Any]:
        if control:
            control.check()
            control.wait_if_paused()
        context = "\n\n".join(_chunk_label(chunk) for chunk in batch)

        def report_retry(attempt: int, delay: float) -> None:
            if progress_callback:
                progress_callback(
                    None,
                    f"第 {index + 1} / {len(batches)} 批遇到速率限制（HTTP 429），"
                    f"{delay:.1f} 秒後自動重試（第 {attempt} / {RATE_LIMIT_MAX_RETRIES} 次）…",
                )

        return _chat_json(
            base_url,
            api_key,
            llm_model,
            "你是知識圖譜資訊抽取器。只能依據提供的文件內容抽取，禁止臆測。只輸出 JSON。",
            "依照 schema 抽取實體與關係。entity.type 與 relationship.type 必須來自 schema；"
            "source_chunk_numbers 必須引用提供的 CHUNK 編號。關係的 source 與 target 使用實體 name。"
            "輸出格式：{\"entities\":[{\"name\":\"...\",\"type\":\"...\","
            "\"description\":\"...\",\"source_chunk_numbers\":[1]}],"
            "\"relationships\":[{\"source\":\"...\",\"target\":\"...\","
            "\"type\":\"...\",\"description\":\"...\",\"source_chunk_numbers\":[1]}]}。\n\n"
            f"schema：\n{json.dumps(schema, ensure_ascii=False)}\n\n文件：\n{context}",
            temperature,
            on_retry=report_retry,
            control=control,
        )

    results: list[dict[str, Any] | None] = [None] * len(batches)
    with ThreadPoolExecutor(max_workers=max_concurrent_requests) as executor:
        futures = {
            executor.submit(extract_batch, index, batch): (index, batch)
            for index, batch in enumerate(batches)
        }
        for future in as_completed(futures):
            batch_index, batch = futures[future]
            try:
                results[batch_index] = future.result()
            except RunCancelled:
                raise
            except Exception as exc:
                raise RuntimeError(
                    f"知識圖譜抽取第 {batch_index + 1} / {len(batches)} 批失敗：{exc}"
                ) from exc
            processed_chunks += len(batch)
            if progress_callback:
                completed_batches = sum(result is not None for result in results)
                progress_callback(
                    processed_chunks / len(chunks),
                    f"已收到 {completed_batches} / {len(batches)} 批回應"
                    f"（{processed_chunks} / {len(chunks)} chunks）",
                )

    if progress_callback:
        progress_callback(1.0, "全部批次已完成，正在統一去重與整合")
    for result in results:
        if result is None:
            raise RuntimeError("知識圖譜抽取結果不完整")
        for item in result.get("entities", []):
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            entity_type = str(item.get("type", "")).strip()
            if (
                not name
                or not entity_type
                or entity_type.casefold() not in allowed_entity_types
            ):
                continue
            numbers = _source_numbers(item, chunk_lookup)
            key = (entity_type.casefold(), name.casefold())
            current = entities.setdefault(
                key,
                {
                    "name": name,
                    "type": entity_type,
                    "description": str(item.get("description", "")).strip(),
                    "source_chunk_numbers": [],
                    "source_pages": [],
                    "source_documents": [],
                    "source_references": [],
                },
            )
            _merge_sources(current, numbers, chunk_lookup)
        for item in result.get("relationships", []):
            if not isinstance(item, dict):
                continue
            source = str(item.get("source", "")).strip()
            target = str(item.get("target", "")).strip()
            relation_type = str(item.get("type", "")).strip()
            if (
                not source
                or not target
                or not relation_type
                or relation_type.casefold() not in allowed_relationship_types
            ):
                continue
            numbers = _source_numbers(item, chunk_lookup)
            key = (source.casefold(), relation_type.casefold(), target.casefold())
            current = relationships.setdefault(
                key,
                {
                    "source": source,
                    "type": relation_type,
                    "target": target,
                    "description": str(item.get("description", "")).strip(),
                    "source_chunk_numbers": [],
                    "source_pages": [],
                    "source_documents": [],
                    "source_references": [],
                },
            )
            _merge_sources(current, numbers, chunk_lookup)
    return GraphExtraction(
        list(entities.values()), list(relationships.values()), len(chunks)
    )


def _source_numbers(
    item: dict[str, Any], chunk_lookup: dict[int, TextChunk]
) -> list[int]:
    raw = item.get("source_chunk_numbers", [])
    if not isinstance(raw, list):
        return []
    numbers: list[int] = []
    for value in raw:
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue
        if number in chunk_lookup and number not in numbers:
            numbers.append(number)
    return numbers


def _merge_sources(
    item: dict[str, Any], numbers: list[int], chunk_lookup: dict[int, TextChunk]
) -> None:
    for number in numbers:
        if number not in item["source_chunk_numbers"]:
            item["source_chunk_numbers"].append(number)
        for page in chunk_lookup[number].pages:
            if page not in item["source_pages"]:
                item["source_pages"].append(page)
        document = chunk_lookup[number].document
        if document and document not in item["source_documents"]:
            item["source_documents"].append(document)
        reference = next(
            (
                value
                for value in item["source_references"]
                if value["document"] == document
            ),
            None,
        )
        if reference is None:
            reference = {"document": document, "chunk_numbers": [], "pages": []}
            item["source_references"].append(reference)
        if number not in reference["chunk_numbers"]:
            reference["chunk_numbers"].append(number)
        for page in chunk_lookup[number].pages:
            if page not in reference["pages"]:
                reference["pages"].append(page)
