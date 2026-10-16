from __future__ import annotations

import http.client
import json
import random
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable
from uuid import uuid4

from .chunking import TextChunk
from .extraction_methods import EXTRACTION_METHODS

SCHEMA_CONTEXT_LIMIT = 30_000
SCHEMA_MERGE_LIMIT = 12_000
SCHEMA_DESCRIPTION_LIMIT = 120
RATE_LIMIT_MAX_RETRIES = 6
GPT_6_LUNA_MODEL = "gpt-6-luna"
GPT_6_LUNA_REASONING_EFFORTS = ("none", "low", "medium", "high", "xhigh")
RATE_LIMIT_BASE_DELAY_SECONDS = 2.0
RATE_LIMIT_MAX_DELAY_SECONDS = 30.0
NETWORK_MAX_RETRIES = 2
NETWORK_RETRY_BASE_DELAY_SECONDS = 0.5
CHAT_COMPLETION_TIMEOUT_SECONDS = 600


class RunCancelled(Exception):
    """Raised when a user stops a running extraction job."""


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


@dataclass(frozen=True)
class GraphExtraction:
    entities: list[dict[str, Any]]
    relationships: list[dict[str, Any]]
    processed_chunks: int
    extraction_method_id: str = "typed_name"


def _api_url(base_url: str, resource: str) -> str:
    base = base_url.strip().rstrip("/")
    if not base:
        raise ValueError("請先填寫模型 API Base URL")
    return f"{base}/{resource.lstrip('/')}"


def _rate_limit_retry_delay(exc: urllib.error.HTTPError, attempt: int) -> float:
    retry_after = exc.headers.get("Retry-After") if exc.headers else None
    if retry_after:
        try:
            delay = max(float(retry_after), 0.5)
        except ValueError:
            delay = min(
                RATE_LIMIT_BASE_DELAY_SECONDS * (2**attempt),
                RATE_LIMIT_MAX_DELAY_SECONDS,
            )
    else:
        delay = min(
            RATE_LIMIT_BASE_DELAY_SECONDS * (2**attempt),
            RATE_LIMIT_MAX_DELAY_SECONDS,
        )
    # Spread parallel workers' retries so they do not all hit the TPM window
    # again at the same instant (a thundering herd after Retry-After).
    jitter = random.uniform(0.0, min(1.0, max(0.1, delay * 0.25)))
    return min(delay + jitter, RATE_LIMIT_MAX_DELAY_SECONDS)


def _post_json(
    url: str,
    payload: dict[str, Any],
    api_key: str,
    timeout: int = CHAT_COMPLETION_TIMEOUT_SECONDS,
    on_retry: Callable[[int, float], None] | None = None,
    control: RunControl | None = None,
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}
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
                result = json.loads(response.read().decode("utf-8"))
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
    reasoning_effort: str | None = None,
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

    is_gpt_6_luna = model.strip().casefold() == GPT_6_LUNA_MODEL
    effective_effort = reasoning_effort or "low"
    if is_gpt_6_luna and effective_effort not in GPT_6_LUNA_REASONING_EFFORTS:
        raise ValueError(
            "GPT-6 Luna 推理強度必須是 none、low、medium、high、xhigh 或 max"
        )

    def request_json(request_messages: list[dict[str, str]], request_temperature: float):
        payload = {
            "model": model.strip(),
            "messages": request_messages,
        }
        if is_gpt_6_luna:
            payload["reasoning_effort"] = effective_effort
            # GPT-6 reasoning requests reject sampling parameters unless effort is none.
            if effective_effort == "none":
                payload["temperature"] = request_temperature
        else:
            payload["temperature"] = request_temperature
        return _post_json(
            url,
            payload,
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
                    "The previous response failed JSON parsing or required-structure validation. "
                    f"Validation error: {first_error}. Fix it and return exactly one complete JSON object. "
                    "Do not include Markdown code fences or explanations, omit required fields, "
                    "or return empty arrays."
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
                    "請提高模型可用輸出長度設定或減少 Schema 類型數量"
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
    max_concurrent_requests: int = 10,
    control: RunControl | None = None,
    reasoning_effort: str | None = None,
) -> SchemaPlan:
    if not chunks:
        raise ValueError("請先在 PDF 頁面解析並產生 chunks")
    granularity_guidance = {
        "粗略": "Keep only the most important cross-section concepts; aggressively merge synonyms and broader/narrower categories.",
        "平衡": "Keep reusable types needed for common queries; merge overly specific or synonymous categories.",
        "詳細": "Keep specialized subtypes when they clearly help retrieval, but never use concrete instances as types.",
    }
    granularity_labels = {"粗略": "coarse", "平衡": "balanced", "詳細": "detailed"}
    if schema_granularity not in granularity_guidance:
        raise ValueError("Schema 粒度必須是粗略、平衡或詳細")
    if int(max_concurrent_requests) < 1:
        raise ValueError("最大並行請求數必須大於 0")
    max_concurrent_requests = int(max_concurrent_requests)

    planning_rules = (
        f"Schema granularity: {granularity_labels[schema_granularity]}. "
        f"{granularity_guidance[schema_granularity]} "
        "Create reusable, generalizable types only. Specific names, models, identifiers, people, "
        "organizations, and section names belong in extracted entities, not types. Create a type "
        "only when a concept recurs and has independent retrieval or relational value; merge "
        "similar concepts into a broader type. Keep entity names, relationship names, and "
        "descriptions in the source language; for English source text, use English and preserve "
        "the source's terminology rather than translating it."
    )
    batches = _chunk_batches(chunks, SCHEMA_CONTEXT_LIMIT)
    candidates: list[dict[str, Any] | None] = [None] * len(batches)

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
            "You are a knowledge-graph schema designer. Return JSON only; do not include Markdown or explanations.",
            "Propose candidate entity and relationship types based on this batch of document text. "
            f"{planning_rules}"
            "Use concise UPPER_SNAKE_CASE type names and descriptions in the source language. "
            "Return this JSON shape: {\"entity_types\":[{\"name\":\"...\",\"description\":\"...\"}],"
            "\"relationship_types\":[{\"name\":\"...\",\"description\":\"...\","
            "\"source_types\":[\"...\"],\"target_types\":[\"...\"]}]} .\n\n"
            f"Document chunks:\n{context}",
            temperature,
            validate_schema,
            on_retry=report_retry,
            control=control,
            reasoning_effort=reasoning_effort,
        )
        return _compact_schema(candidate)

    analyzed = 0
    executor = ThreadPoolExecutor(max_workers=max_concurrent_requests)
    futures = {}
    next_batch_index = 0

    def submit_available_batches() -> None:
        nonlocal next_batch_index
        while (
            next_batch_index < len(batches)
            and len(futures) < max_concurrent_requests
        ):
            batch = batches[next_batch_index]
            future = executor.submit(plan_batch, next_batch_index, batch)
            futures[future] = (next_batch_index, batch)
            next_batch_index += 1

    submit_available_batches()
    try:
        while futures:
            future = next(as_completed(tuple(futures)))
            index, batch = futures.pop(future)
            try:
                candidates[index] = future.result()
            except ValueError as exc:
                raise ValueError(
                    f"Schema 規劃第 {index + 1} / {len(batches)} 批失敗：{exc}"
                ) from exc
            analyzed += len(batch)
            if progress_callback:
                progress_callback(
                    0.75 * analyzed / len(chunks),
                    f"已完成 {sum(item is not None for item in candidates)} / "
                    f"{len(batches)} 批（已分析 {analyzed} / {len(chunks)} chunks）",
                )
            submit_available_batches()
    except BaseException:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    candidates = [candidate for candidate in candidates if candidate is not None]

    merge_rounds = 0
    while len(candidates) > 1:
        if control:
            control.check()
        merge_rounds += 1
        groups = _schema_groups(candidates, SCHEMA_MERGE_LIMIT)
        merged: list[dict[str, Any] | None] = [None] * len(groups)

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
                "You are a knowledge-graph schema consolidation specialist. Return JSON only; "
                "do not include Markdown or explanations.",
                "Consolidate the candidate schemas below; do not simply take the union of all types. "
                f"{planning_rules}"
                "Remove duplicates, standardize synonyms, and merge broader/narrower or similar types. "
                "Keep only type names, concise descriptions, and relationship source_types and "
                "target_types. Do not add examples, properties, or other fields. Preserve the "
                "entity_types and relationship_types JSON keys.\n\n"
                f"Candidate schemas:\n{json.dumps(group, ensure_ascii=False)}",
                temperature,
                validate_schema,
                on_retry=report_retry,
                control=control,
                reasoning_effort=reasoning_effort,
            )
            return _compact_schema(result)

        completed_groups = 0
        with ThreadPoolExecutor(max_workers=max_concurrent_requests) as executor:
            futures = {
                executor.submit(merge_group, index, group): index
                for index, group in enumerate(groups)
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    merged[index] = future.result()
                except ValueError as exc:
                    raise ValueError(
                        f"Schema 第 {merge_rounds} 輪整合第 "
                        f"{index + 1} / {len(groups)} 組失敗：{exc}"
                    ) from exc
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
        candidates = [item for item in merged if item is not None]
    if progress_callback:
        progress_callback(1.0, f"已分析全部 {len(chunks)} / {len(chunks)} chunks")
    return SchemaPlan(
        candidates[0], len(chunks), len(chunks), len(batches), merge_rounds
    )


def _batches(chunks: list[TextChunk]) -> list[list[TextChunk]]:
    # Keep provenance deterministic: every extraction response belongs to
    # exactly one input chunk, so the program can attach its source directly.
    return [[chunk] for chunk in chunks]


def extract_graph(
    base_url: str,
    api_key: str,
    llm_model: str,
    chunks: list[TextChunk],
    extraction_method_id: str = "typed_name",
    temperature: float = 0,
    max_concurrent_requests: int = 10,
    progress_callback: Callable[[float | None, str], None] | None = None,
    control: RunControl | None = None,
    reasoning_effort: str | None = None,
) -> GraphExtraction:
    if not chunks:
        raise ValueError("請先在 PDF 頁面解析並產生 chunks")
    if int(max_concurrent_requests) < 1:
        raise ValueError("最大並行請求數必須至少為 1")
    max_concurrent_requests = int(max_concurrent_requests)
    method_id = extraction_method_id or "typed_name"
    method = EXTRACTION_METHODS.get(method_id)
    entities: dict[tuple[str, ...], dict[str, Any]] = {}
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

        extraction_instructions = (
            method.extraction_instructions()
            + " Use exact entity names in relationship source and target fields. "
            "Source page, chunk, document, and entity identifiers are assigned by the program; "
            "never generate or return identifiers or source-location fields.\n\n"
        )
        return _chat_json(
            base_url,
            api_key,
            llm_model,
            "You are a knowledge-graph information extraction system. Extract only facts supported "
            "by the provided document and do not hallucinate. Return JSON only.",
            extraction_instructions
            + "Use entity names for relationship source and target. Source chunk provenance and "
            "all entity/relationship identifiers are assigned by the program; do not output them. "
            "Return this JSON shape exactly: "
            + method.output_shape()
            + ".\n\n"
            f"Document:\n{context}",
            temperature,
            on_retry=report_retry,
            control=control,
            reasoning_effort=reasoning_effort,
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
    for result, batch in zip(results, batches):
        if result is None:
            raise RuntimeError("知識圖譜抽取結果不完整")
        source_chunk_number = batch[0].number
        for item in result.get("entities", []):
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            entity_type = str(item.get("type", "")).strip()
            if not name or (method.requires_entity_type and not entity_type):
                continue
            numbers = [source_chunk_number]
            key = method.entity_key(name, entity_type)
            current = entities.setdefault(
                key,
                {
                    **method.make_entity(item, name),
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
            if not source or not target or not relation_type:
                continue
            numbers = [source_chunk_number]
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
    entity_values = list(entities.values())
    entity_ids_by_name: dict[str, str] = {}
    for entity in entity_values:
        entity["entity_id"] = uuid4().hex
        normalized_name = " ".join(str(entity["name"]).casefold().split())
        entity_ids_by_name.setdefault(normalized_name, entity["entity_id"])
    relationship_values = list(relationships.values())
    for relationship in relationship_values:
        relationship["relationship_id"] = uuid4().hex
        relationship["source_entity_id"] = entity_ids_by_name.get(
            " ".join(str(relationship["source"]).casefold().split()), ""
        )
        relationship["target_entity_id"] = entity_ids_by_name.get(
            " ".join(str(relationship["target"]).casefold().split()), ""
        )
    return GraphExtraction(
        entity_values, relationship_values, len(chunks), method_id
    )


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
        document_id = chunk_lookup[number].document_id
        if document and document not in item["source_documents"]:
            item["source_documents"].append(document)
        reference = next(
            (
                value
                for value in item["source_references"]
                if value.get("document_id", "") == document_id
                and value.get("document", "") == document
            ),
            None,
        )
        if reference is None:
            reference = {
                "document_id": document_id, "document": document,
                "chunk_numbers": [], "pages": [],
            }
            item["source_references"].append(reference)
        if number not in reference["chunk_numbers"]:
            reference["chunk_numbers"].append(number)
        for page in chunk_lookup[number].pages:
            if page not in reference["pages"]:
                reference["pages"].append(page)
