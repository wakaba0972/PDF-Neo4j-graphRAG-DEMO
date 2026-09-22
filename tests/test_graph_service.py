import http.client
import io
import json
import urllib.error

import pytest

from manual_graphrag import graph_service
from manual_graphrag.chunking import TextChunk

SCHEMA = {
    "entity_types": [{"name": "DEVICE", "description": "設備"}],
    "relationship_types": [
        {
            "name": "USES",
            "description": "使用",
            "source_types": ["DEVICE"],
            "target_types": ["DEVICE"],
        }
    ],
}


def chat_response(payload: dict[str, object]) -> dict[str, object]:
    return {
        "choices": [
            {
                "message": {
                    "content": f"```json\n{json.dumps(payload, ensure_ascii=False)}\n```"
                }
            }
        ]
    }


def test_plan_graph_schema_parses_fenced_json_and_uses_chunks(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_post(url, payload, api_key, timeout=120, **kwargs):
        captured.update({"url": url, "payload": payload, "api_key": api_key})
        return chat_response(SCHEMA)

    monkeypatch.setattr(graph_service, "_post_json", fake_post)
    chunks = [TextChunk(1, "設備 A 使用設備 B", (2,))]

    plan = graph_service.plan_graph_schema(
        "http://localhost:11434/v1", "secret", "model-a", chunks, 0.4
    )

    assert plan.schema == SCHEMA
    assert plan.analyzed_chunks == 1
    assert plan.batch_count == 1
    assert plan.merge_rounds == 0
    assert plan.total_chunks == 1
    assert captured["url"] == "http://localhost:11434/v1/chat/completions"
    assert captured["api_key"] == "secret"
    assert captured["payload"]["temperature"] == 0.4
    assert "max_tokens" not in captured["payload"]
    assert "[CHUNK 1; PAGES 2]" in captured["payload"]["messages"][1]["content"]


def test_schema_planning_analyzes_all_chunks_and_merges_hierarchically(monkeypatch) -> None:
    monkeypatch.setattr(graph_service, "SCHEMA_CONTEXT_LIMIT", 100)
    chunks = [TextChunk(number, "x" * 20, (number,)) for number in range(1, 11)]
    prompts: list[str] = []
    progress_updates: list[tuple[float, str]] = []

    def fake_chat(*args, **kwargs):
        prompts.append(args[4])
        return SCHEMA

    monkeypatch.setattr(graph_service, "_chat_json", fake_chat)

    plan = graph_service.plan_graph_schema(
        "http://models/v1",
        "",
        "llm",
        chunks,
        progress_callback=lambda value, description: progress_updates.append(
            (value, description)
        ),
    )

    chunk_prompts = [prompt for prompt in prompts if "文件 chunks" in prompt]
    seen = {
        number
        for number in range(1, 11)
        if any(f"CHUNK {number};" in prompt for prompt in chunk_prompts)
    }
    assert seen == set(range(1, 11))
    assert plan.analyzed_chunks == plan.total_chunks == 10
    assert plan.batch_count == len(chunk_prompts)
    assert plan.batch_count > 1
    assert plan.merge_rounds > 0
    assert [value for value, _ in progress_updates] == sorted(
        value for value, _ in progress_updates
    )
    assert progress_updates[-1] == (1.0, "已分析全部 10 / 10 chunks")



def test_schema_groups_use_smaller_merge_limit(monkeypatch) -> None:
    monkeypatch.setattr(graph_service, "SCHEMA_CONTEXT_LIMIT", 100)
    monkeypatch.setattr(graph_service, "SCHEMA_MERGE_LIMIT", 150)
    chunks = [TextChunk(number, "x" * 20, (number,)) for number in range(1, 11)]
    merge_group_sizes = []

    def fake_chat(*args, **kwargs):
        prompt = args[4]
        if "候選 Schema" in prompt:
            merge_group_sizes.append(prompt.count('"entity_types"'))
        return SCHEMA

    monkeypatch.setattr(graph_service, "_chat_json", fake_chat)

    plan = graph_service.plan_graph_schema("http://models/v1", "", "llm", chunks)

    assert plan.merge_rounds > 1
    assert merge_group_sizes
    assert all(size <= 2 for size in merge_group_sizes)


def test_schema_merge_failure_cancels_queued_groups(monkeypatch) -> None:
    monkeypatch.setattr(graph_service, "SCHEMA_CONTEXT_LIMIT", 100)
    monkeypatch.setattr(graph_service, "SCHEMA_MERGE_LIMIT", 150)
    chunks = [TextChunk(number, "x" * 20, (number,)) for number in range(1, 11)]
    merge_calls: list[int] = []

    def fake_chat(*args, **kwargs):
        prompt = args[4]
        if "候選 Schema" in prompt:
            merge_calls.append(1)
            raise ValueError("模型逾時")
        return SCHEMA

    monkeypatch.setattr(graph_service, "_chat_json", fake_chat)

    with pytest.raises(
        ValueError,
        match=r"Schema 第 1 輪整合第 1 / \d+ 組失敗：模型逾時",
    ):
        graph_service.plan_graph_schema(
            "http://models/v1",
            "",
            "llm",
            chunks,
            max_concurrent_requests=1,
        )

    assert merge_calls == [1]


def test_schema_merge_failure_resume_state_skips_completed_groups(monkeypatch) -> None:
    monkeypatch.setattr(graph_service, "SCHEMA_CONTEXT_LIMIT", 100)
    monkeypatch.setattr(graph_service, "SCHEMA_MERGE_LIMIT", 150)
    chunks = [TextChunk(number, "x" * 20, (number,)) for number in range(1, 8)]

    merge_call_count = 0

    def fake_chat_first(*args, **kwargs):
        nonlocal merge_call_count
        prompt = args[4]
        if "候選 Schema" not in prompt:
            return SCHEMA
        merge_call_count += 1
        if merge_call_count == 1:
            return SCHEMA
        raise ValueError("模型逾時")

    monkeypatch.setattr(graph_service, "_chat_json", fake_chat_first)

    with pytest.raises(
        ValueError,
        match=r"Schema 第 1 輪整合第 2 / 2 組失敗：模型逾時",
    ) as excinfo:
        graph_service.plan_graph_schema(
            "http://models/v1", "", "llm", chunks, max_concurrent_requests=1,
        )

    resume_state = getattr(excinfo.value, "schema_planning_resume", None)
    assert resume_state is not None
    assert resume_state["stage"] == "merge"
    assert resume_state["candidates"] is not None
    assert len(resume_state["candidates"]) == 4
    assert resume_state["succeeded"] == {"0": graph_service._compact_schema(SCHEMA)}

    resume_merge_calls = 0
    batch_phase_seen = False

    def fake_chat_second(*args, **kwargs):
        nonlocal resume_merge_calls, batch_phase_seen
        prompt = args[4]
        if "候選 Schema" not in prompt:
            batch_phase_seen = True
            return SCHEMA
        resume_merge_calls += 1
        return SCHEMA

    monkeypatch.setattr(graph_service, "_chat_json", fake_chat_second)

    plan = graph_service.plan_graph_schema(
        "http://models/v1", "", "llm", chunks,
        max_concurrent_requests=1,
        resume_state=resume_state,
    )

    assert plan.schema == SCHEMA
    assert not batch_phase_seen
    assert resume_merge_calls == 2


def test_schema_planning_resume_state_ignored_when_signature_changes(monkeypatch) -> None:
    monkeypatch.setattr(graph_service, "SCHEMA_CONTEXT_LIMIT", 100)
    monkeypatch.setattr(graph_service, "SCHEMA_MERGE_LIMIT", 150)
    chunks = [TextChunk(number, "x" * 20, (number,)) for number in range(1, 8)]

    calls: list[str] = []

    def fake_chat(*args, **kwargs):
        prompt = args[4]
        calls.append("merge" if "候選 Schema" in prompt else "batch")
        return SCHEMA

    monkeypatch.setattr(graph_service, "_chat_json", fake_chat)

    stale_resume = {
        "signature": "does-not-match",
        "stage": "merge",
        "candidates": [SCHEMA] * 4,
        "succeeded": {"0": SCHEMA, "1": SCHEMA},
    }

    plan = graph_service.plan_graph_schema(
        "http://models/v1", "", "llm", chunks,
        max_concurrent_requests=1,
        resume_state=stale_resume,
    )

    assert plan.schema == SCHEMA
    assert "batch" in calls


def test_compact_schema_keeps_only_required_merge_fields(monkeypatch) -> None:
    monkeypatch.setattr(graph_service, "SCHEMA_DESCRIPTION_LIMIT", 8)
    schema = {
        "entity_types": [
            {
                "name": " DEVICE ",
                "description": "很長   的設備類型說明",
                "properties": ["serial"],
            }
        ],
        "relationship_types": [
            {
                "name": " USES ",
                "description": "使用關係說明文字",
                "source_types": ["DEVICE", "DEVICE", ""],
                "target_types": [" DEVICE "],
                "example": "A uses B",
            }
        ],
    }

    compact = graph_service._compact_schema(schema)

    assert compact == {
        "entity_types": [{"name": "DEVICE", "description": "很長 的設備類型"}],
        "relationship_types": [
            {
                "name": "USES",
                "description": "使用關係說明文字",
                "source_types": ["DEVICE"],
                "target_types": ["DEVICE"],
            }
        ],
    }




def test_schema_planning_prompt_applies_granularity_without_type_limits(monkeypatch) -> None:
    prompts = []

    def fake_chat(*args, **kwargs):
        prompts.append(args[4])
        return SCHEMA

    monkeypatch.setattr(graph_service, "_chat_json", fake_chat)
    graph_service.plan_graph_schema(
        "http://models/v1", "", "llm",
        [TextChunk(1, "設備 A 使用設備 B", (1,))],
        schema_granularity="粗略",
    )
    assert "Schema 粒度：粗略" in prompts[0]
    assert "具體名稱、型號、編號、人物、組織或章節" in prompts[0]
    assert "類型最多" not in prompts[0]
    assert "上限" not in prompts[0]


def test_schema_planning_rejects_invalid_granularity() -> None:
    with pytest.raises(ValueError, match="Schema 粒度"):
        graph_service.plan_graph_schema(
            "http://models/v1", "", "llm",
            [TextChunk(1, "text", (1,))], schema_granularity="未知",
        )


def test_schema_planning_accepts_unlimited_type_counts(monkeypatch) -> None:
    large_schema = {
        "entity_types": [{"name": f"ENTITY_{index}"} for index in range(40)],
        "relationship_types": [{"name": f"REL_{index}"} for index in range(50)],
    }
    calls = []
    monkeypatch.setattr(
        graph_service, "_post_json",
        lambda *args, **kwargs: calls.append(args[1]) or chat_response(large_schema),
    )
    plan = graph_service.plan_graph_schema(
        "http://models/v1", "", "llm", [TextChunk(1, "text", (1,))],
    )
    assert len(plan.schema["entity_types"]) == 40
    assert len(plan.schema["relationship_types"]) == 50
    assert len(calls) == 1


def test_schema_planning_reports_rate_limit_wait_via_progress(monkeypatch) -> None:
    def fake_post(url, payload, api_key, timeout=120, on_retry=None, **kwargs):
        if on_retry:
            on_retry(1, 2.5)
        return chat_response(SCHEMA)

    monkeypatch.setattr(graph_service, "_post_json", fake_post)
    progress_updates: list[tuple[float | None, str]] = []

    graph_service.plan_graph_schema(
        "http://models/v1", "", "llm", [TextChunk(1, "text", (1,))],
        progress_callback=lambda value, description: progress_updates.append(
            (value, description)
        ),
    )

    wait_messages = [desc for value, desc in progress_updates if value is None]
    assert wait_messages
    assert "速率限制" in wait_messages[0]
    assert "2.5" in wait_messages[0]


def test_extract_graph_reports_rate_limit_wait_via_progress(monkeypatch) -> None:
    def fake_chat_json(*args, **kwargs):
        on_retry = kwargs.get("on_retry")
        if on_retry:
            on_retry(1, 3.0)
        return {"entities": [], "relationships": []}

    monkeypatch.setattr(graph_service, "_chat_json", fake_chat_json)
    progress_updates: list[tuple[float | None, str]] = []

    graph_service.extract_graph(
        "http://models/v1", "", "llm", [TextChunk(1, "text", (1,))], SCHEMA,
        progress_callback=lambda value, description: progress_updates.append(
            (value, description)
        ),
    )

    wait_messages = [desc for value, desc in progress_updates if value is None]
    assert wait_messages
    assert "速率限制" in wait_messages[0]
    assert "3.0" in wait_messages[0]


def test_run_control_toggle_pause_reports_state() -> None:
    control = graph_service.RunControl()
    assert control.is_paused is False
    assert control.toggle_pause() is True
    assert control.is_paused is True
    assert control.toggle_pause() is False
    assert control.is_paused is False


def test_run_control_check_raises_when_stopped() -> None:
    control = graph_service.RunControl()
    control.request_stop()
    with pytest.raises(graph_service.RunCancelled):
        control.check()
    control.reset()
    control.check()


def test_plan_graph_schema_stops_immediately_when_already_cancelled(monkeypatch) -> None:
    monkeypatch.setattr(
        graph_service, "_chat_json",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("不應呼叫模型")),
    )
    control = graph_service.RunControl()
    control.request_stop()

    with pytest.raises(graph_service.RunCancelled):
        graph_service.plan_graph_schema(
            "http://models/v1", "", "llm", [TextChunk(1, "text", (1,))],
            control=control,
        )


def test_extract_graph_stops_immediately_when_already_cancelled(monkeypatch) -> None:
    monkeypatch.setattr(
        graph_service, "_chat_json",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("不應呼叫模型")),
    )
    control = graph_service.RunControl()
    control.request_stop()

    with pytest.raises(graph_service.RunCancelled):
        graph_service.extract_graph(
            "http://models/v1", "", "llm", [TextChunk(1, "text", (1,))], SCHEMA,
            control=control,
        )


def test_plan_graph_schema_stops_mid_run_without_running_remaining_batches(monkeypatch) -> None:
    monkeypatch.setattr(graph_service, "SCHEMA_CONTEXT_LIMIT", 10)
    chunks = [TextChunk(number, "x" * 20, (number,)) for number in range(1, 6)]
    control = graph_service.RunControl()
    calls: list[int] = []

    def fake_chat(*args, **kwargs):
        calls.append(1)
        control.request_stop()
        return SCHEMA

    monkeypatch.setattr(graph_service, "_chat_json", fake_chat)

    with pytest.raises(graph_service.RunCancelled):
        graph_service.plan_graph_schema(
            "http://models/v1", "", "llm", chunks,
            max_concurrent_requests=1,
            control=control,
        )

    assert len(calls) < 5


def test_plan_graph_schema_failure_does_not_run_queued_batches(monkeypatch) -> None:
    monkeypatch.setattr(graph_service, "SCHEMA_CONTEXT_LIMIT", 10)
    chunks = [TextChunk(number, "x" * 20, (number,)) for number in range(1, 6)]
    calls: list[int] = []

    def fake_chat(*args, **kwargs):
        calls.append(1)
        raise ValueError("模型逾時")

    monkeypatch.setattr(graph_service, "_chat_json", fake_chat)

    with pytest.raises(
        ValueError,
        match=r"Schema 規劃第 1 / 5 批失敗：模型逾時",
    ):
        graph_service.plan_graph_schema(
            "http://models/v1",
            "",
            "llm",
            chunks,
            max_concurrent_requests=1,
        )

    assert calls == [1]


def test_plan_graph_schema_batch_failure_resume_skips_completed_batches(monkeypatch) -> None:
    monkeypatch.setattr(graph_service, "SCHEMA_CONTEXT_LIMIT", 10)
    chunks = [TextChunk(number, "x" * 20, (number,)) for number in range(1, 6)]

    batch_call_count = 0

    def fake_chat_first(*args, **kwargs):
        nonlocal batch_call_count
        batch_call_count += 1
        if batch_call_count == 1:
            return SCHEMA
        raise ValueError("模型逾時")

    monkeypatch.setattr(graph_service, "_chat_json", fake_chat_first)

    with pytest.raises(
        ValueError,
        match=r"Schema 規劃第 2 / 5 批失敗：模型逾時",
    ) as excinfo:
        graph_service.plan_graph_schema(
            "http://models/v1", "", "llm", chunks, max_concurrent_requests=1,
        )

    resume_state = getattr(excinfo.value, "schema_planning_resume", None)
    assert resume_state is not None
    assert resume_state["stage"] == "batches"
    assert resume_state["candidates"] is None
    assert resume_state["succeeded"] == {"0": graph_service._compact_schema(SCHEMA)}

    batch_calls_second: list[int] = []

    def fake_chat_second(*args, **kwargs):
        prompt = args[4]
        if "候選 Schema" not in prompt:
            batch_calls_second.append(1)
        return SCHEMA

    monkeypatch.setattr(graph_service, "_chat_json", fake_chat_second)

    plan = graph_service.plan_graph_schema(
        "http://models/v1", "", "llm", chunks,
        max_concurrent_requests=1,
        resume_state=resume_state,
    )

    assert plan.schema == SCHEMA
    assert len(batch_calls_second) == 4


def test_extract_graph_batches_deduplicates_and_keeps_sources(monkeypatch) -> None:
    monkeypatch.setattr(graph_service, "EXTRACTION_BATCH_LIMIT", 45)
    chunks = [
        TextChunk(1, "設備 A", (1,), document="manual-a.pdf"),
        TextChunk(2, "設備 A 使用設備 B", (2,), document="manual-b.pdf"),
    ]
    responses = iter(
        [
            {
                "entities": [
                    {
                        "name": "設備 A",
                        "type": "DEVICE",
                        "description": "主要設備",
                        "source_chunk_numbers": [1],
                    }
                ],
                "relationships": [],
            },
            {
                "entities": [
                    {
                        "name": "設備 A",
                        "type": "DEVICE",
                        "description": "主要設備",
                        "source_chunk_numbers": [2],
                    },
                    {
                        "name": "設備 B",
                        "type": "DEVICE",
                        "description": "配件",
                        "source_chunk_numbers": [2],
                    },
                    {
                        "name": "忽略",
                        "type": "UNKNOWN",
                        "source_chunk_numbers": [2],
                    },
                ],
                "relationships": [
                    {
                        "source": "設備 A",
                        "target": "設備 B",
                        "type": "USES",
                        "description": "搭配使用",
                        "source_chunk_numbers": [2, 999],
                    }
                ],
            },
        ]
    )
    monkeypatch.setattr(graph_service, "_chat_json", lambda *args, **kwargs: next(responses))

    extraction = graph_service.extract_graph(
        "http://models/v1", "", "llm", chunks, SCHEMA
    )

    assert extraction.processed_chunks == 2
    assert len(extraction.entities) == 2
    assert extraction.entities[0]["source_chunk_numbers"] == [1, 2]
    assert extraction.entities[0]["source_pages"] == [1, 2]
    assert extraction.entities[0]["source_documents"] == ["manual-a.pdf", "manual-b.pdf"]
    assert extraction.entities[0]["source_references"] == [
        {"document": "manual-a.pdf", "chunk_numbers": [1], "pages": [1]},
        {"document": "manual-b.pdf", "chunk_numbers": [2], "pages": [2]},
    ]
    assert extraction.relationships == [
        {
            "source": "設備 A",
            "type": "USES",
            "target": "設備 B",
            "description": "搭配使用",
            "source_chunk_numbers": [2],
            "source_pages": [2],
            "source_documents": ["manual-b.pdf"],
            "source_references": [
                {"document": "manual-b.pdf", "chunk_numbers": [2], "pages": [2]}
            ],
        }
    ]


def test_validate_schema_rejects_missing_types() -> None:
    with pytest.raises(ValueError, match="entity_types"):
        graph_service.validate_schema({"entity_types": [], "relationship_types": []})


def test_plan_graph_schema_requires_chunks() -> None:
    with pytest.raises(ValueError, match="先在 PDF 頁面解析"):
        graph_service.plan_graph_schema("http://models/v1", "", "llm", [])


def test_chat_json_rejects_invalid_generation_options() -> None:
    with pytest.raises(ValueError, match="temperature"):
        graph_service._chat_json("http://models/v1", "", "llm", "system", "user", 2.1)


def _rate_limit_error(retry_after: str | None = None) -> urllib.error.HTTPError:
    headers = {"Retry-After": retry_after} if retry_after else {}
    return urllib.error.HTTPError(
        "http://models/v1/chat/completions", 429, "Too Many Requests",
        headers, io.BytesIO(b'{"error": {"message": "rate_limit_exceeded"}}'),
    )


def test_post_json_retries_after_rate_limit_then_succeeds(monkeypatch) -> None:
    calls = {"count": 0}
    sleeps: list[float] = []
    first_error = _rate_limit_error("0")

    def fake_urlopen(request, timeout=None):
        calls["count"] += 1
        if calls["count"] == 1:
            raise first_error
        return io.BytesIO(json.dumps({"ok": True}).encode("utf-8"))

    monkeypatch.setattr(graph_service.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(graph_service.time, "sleep", lambda seconds: sleeps.append(seconds))

    result = graph_service._post_json("http://models/v1/chat/completions", {"a": 1}, "key")

    assert result == {"ok": True}
    assert calls["count"] == 2
    assert sleeps == [0.5]
    assert first_error.fp.closed, "retried HTTPError response must be closed, not leaked"


def test_post_json_uses_extended_chat_timeout(monkeypatch) -> None:
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["timeout"] = timeout
        return io.BytesIO(b'{"ok": true}')

    monkeypatch.setattr(graph_service.urllib.request, "urlopen", fake_urlopen)

    assert graph_service._post_json("http://models/v1/chat/completions", {}, "") == {"ok": True}
    assert captured["timeout"] == graph_service.CHAT_COMPLETION_TIMEOUT_SECONDS == 600


def test_identifies_only_configured_ollama_chat_url(monkeypatch) -> None:
    monkeypatch.setattr(
        graph_service,
        "load_env",
        lambda: {"MODEL_OLLAMA_API_BASE": "http://ollama:11434/v1"},
    )

    assert graph_service._is_ollama_chat_url(
        "http://ollama:11434/v1/chat/completions"
    )
    assert not graph_service._is_ollama_chat_url(
        "http://ollama:11434/v1/embeddings"
    )
    assert not graph_service._is_ollama_chat_url(
        "http://openai/v1/chat/completions"
    )


def test_post_json_collects_ollama_chat_stream(monkeypatch) -> None:
    captured = {}
    chunks = [
        {"choices": [{"delta": {"role": "assistant", "content": "{\"ok\":"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "true}"}, "finish_reason": "stop"}]},
    ]
    stream = "".join(
        f"data: {json.dumps(chunk)}\n\n" for chunk in chunks
    ) + "data: [DONE]\n\n"

    def fake_urlopen(request, timeout=None):
        captured["payload"] = json.loads(request.data)
        captured["accept"] = request.headers.get("Accept")
        return io.BytesIO(stream.encode("utf-8"))

    monkeypatch.setattr(graph_service, "_is_ollama_chat_url", lambda _url: True)
    monkeypatch.setattr(graph_service.urllib.request, "urlopen", fake_urlopen)

    response = graph_service._post_json(
        "http://ollama:11434/v1/chat/completions", {}, ""
    )

    assert captured == {
        "payload": {"stream": True},
        "accept": "text/event-stream",
    }
    assert graph_service._chat_response_content(response) == (
        "{\"ok\":true}", "stop"
    )


def test_post_json_keeps_non_ollama_requests_non_streaming(monkeypatch) -> None:
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["payload"] = json.loads(request.data)
        return io.BytesIO(b"{\"ok\": true}")

    monkeypatch.setattr(graph_service, "_is_ollama_chat_url", lambda _url: False)
    monkeypatch.setattr(graph_service.urllib.request, "urlopen", fake_urlopen)

    assert graph_service._post_json(
        "http://openai/v1/chat/completions", {}, ""
    ) == {"ok": True}
    assert captured["payload"] == {}


def test_post_json_retries_ollama_stream_without_completion(monkeypatch) -> None:
    calls = {"count": 0}
    sleeps = []

    def event(content, finish_reason=None):
        return (
            "data: "
            + json.dumps({
                "choices": [{
                    "delta": {"content": content},
                    "finish_reason": finish_reason,
                }]
            })
            + "\n\n"
        ).encode("utf-8")

    def fake_urlopen(request, timeout=None):
        calls["count"] += 1
        if calls["count"] == 1:
            return io.BytesIO(event("殘缺"))
        return io.BytesIO(event("完整", "stop"))

    monkeypatch.setattr(graph_service, "_is_ollama_chat_url", lambda _url: True)
    monkeypatch.setattr(graph_service.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(
        graph_service.time, "sleep", lambda seconds: sleeps.append(seconds)
    )

    response = graph_service._post_json(
        "http://ollama:11434/v1/chat/completions", {}, ""
    )

    assert graph_service._chat_response_content(response) == ("完整", "stop")
    assert calls["count"] == 2
    assert sleeps == [graph_service.NETWORK_RETRY_BASE_DELAY_SECONDS]


def test_post_json_raises_after_exhausting_rate_limit_retries(monkeypatch) -> None:
    calls = {"count": 0}

    def fake_urlopen(request, timeout=None):
        calls["count"] += 1
        raise _rate_limit_error()

    monkeypatch.setattr(graph_service.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(graph_service.time, "sleep", lambda seconds: None)

    with pytest.raises(ValueError, match="429"):
        graph_service._post_json("http://models/v1/chat/completions", {"a": 1}, "key")

    assert calls["count"] == graph_service.RATE_LIMIT_MAX_RETRIES + 1


def test_post_json_retries_incomplete_response_then_succeeds(monkeypatch) -> None:
    calls = {"count": 0}
    sleeps = []

    def fake_urlopen(request, timeout=None):
        calls["count"] += 1
        if calls["count"] == 1:
            raise http.client.IncompleteRead(b'{"partial":', 10)
        return io.BytesIO(json.dumps({"ok": True}).encode("utf-8"))

    monkeypatch.setattr(graph_service.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(graph_service.time, "sleep", lambda seconds: sleeps.append(seconds))

    result = graph_service._post_json(
        "http://models/v1/chat/completions", {"a": 1}, "key"
    )

    assert result == {"ok": True}
    assert calls["count"] == 2
    assert sleeps == [0.5]


def test_post_json_reports_incomplete_response_after_retries(monkeypatch) -> None:
    calls = {"count": 0}

    def fake_urlopen(request, timeout=None):
        calls["count"] += 1
        raise http.client.IncompleteRead(b"partial", 10)

    monkeypatch.setattr(graph_service.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(graph_service.time, "sleep", lambda _seconds: None)

    with pytest.raises(ValueError, match="回應中斷.*重試 2 次"):
        graph_service._post_json(
            "http://models/v1/chat/completions", {"a": 1}, "key"
        )

    assert calls["count"] == graph_service.NETWORK_MAX_RETRIES + 1


def test_list_models_returns_sorted_unique_ids(monkeypatch) -> None:
    payload = {
        "data": [
            {"id": "llama3.1:8b"},
            {"id": "nomic-embed-text"},
            {"id": "llama3.1:8b"},
            {"id": ""},
            "not-a-dict",
        ]
    }

    def fake_urlopen(request, timeout=None):
        assert request.full_url == "http://localhost:11434/v1/models"
        return io.BytesIO(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr(graph_service.urllib.request, "urlopen", fake_urlopen)

    models = graph_service.list_models("http://localhost:11434/v1", "")

    assert models == ["llama3.1:8b", "nomic-embed-text"]


def test_list_models_rejects_missing_data_field(monkeypatch) -> None:
    def fake_urlopen(request, timeout=None):
        return io.BytesIO(json.dumps({"object": "list"}).encode("utf-8"))

    monkeypatch.setattr(graph_service.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(ValueError, match="data 陣列"):
        graph_service.list_models("http://localhost:11434/v1", "")


def test_list_models_rejects_empty_model_list(monkeypatch) -> None:
    def fake_urlopen(request, timeout=None):
        return io.BytesIO(json.dumps({"data": []}).encode("utf-8"))

    monkeypatch.setattr(graph_service.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(ValueError, match="沒有回傳任何可用模型"):
        graph_service.list_models("http://localhost:11434/v1", "")


def test_check_model_connection_reports_http_error(monkeypatch) -> None:
    def fake_urlopen(request, timeout=None):
        raise urllib.error.HTTPError(
            "http://models/v1/models", 401, "Unauthorized", {}, io.BytesIO(b"bad key")
        )

    monkeypatch.setattr(graph_service.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(ValueError, match="401"):
        graph_service.check_model_connection("http://models/v1", "wrong-key")


def test_schema_planning_stops_when_any_batch_fails(monkeypatch) -> None:
    monkeypatch.setattr(graph_service, "SCHEMA_CONTEXT_LIMIT", 45)
    chunks = [TextChunk(number, "x" * 20, (number,)) for number in range(1, 4)]
    calls = 0

    def fake_chat(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("second batch failed")
        return SCHEMA

    monkeypatch.setattr(graph_service, "_chat_json", fake_chat)

    with pytest.raises(ValueError, match="second batch failed"):
        graph_service.plan_graph_schema("http://models/v1", "", "llm", chunks)
    # All batches are submitted to the thread pool up front, so another worker
    # may start before the failing future is observed and cancellation begins.
    assert 2 <= calls <= 3
    # Failure is contextualized with the exact batch and no partial plan is returned.


def test_extract_json_text_accepts_prose_around_json() -> None:
    result = graph_service._extract_json_text(
        '以下是結果：\n{"entity_types": [], "relationship_types": []}\n請確認。'
    )

    assert result == {"entity_types": [], "relationship_types": []}


def test_chat_json_retries_invalid_output_once(monkeypatch) -> None:
    responses = iter(
        [
            {
                "choices": [
                    {"message": {"content": "not json"}, "finish_reason": "stop"}
                ]
            },
            chat_response(SCHEMA),
        ]
    )
    payloads = []

    def fake_post(url, payload, api_key, timeout=120, **kwargs):
        payloads.append(payload)
        return next(responses)

    monkeypatch.setattr(graph_service, "_post_json", fake_post)

    result = graph_service._chat_json(
        "http://models/v1", "", "llm", "system", "user", 0.7
    )

    assert result == SCHEMA
    assert len(payloads) == 2
    assert payloads[0]["temperature"] == 0.7
    assert payloads[1]["temperature"] == 0
    assert len(payloads[1]["messages"]) == 4
    assert "請修正" in payloads[1]["messages"][-1]["content"]



def test_chat_json_retries_schema_that_fails_structure_validation(monkeypatch) -> None:
    invalid_schema = {"entity_types": [], "relationship_types": []}
    responses = iter([chat_response(invalid_schema), chat_response(SCHEMA)])
    payloads = []

    def fake_post(url, payload, api_key, timeout=120, **kwargs):
        payloads.append(payload)
        return next(responses)

    monkeypatch.setattr(graph_service, "_post_json", fake_post)

    result = graph_service._chat_json(
        "http://models/v1",
        "",
        "llm",
        "system",
        "user",
        0.7,
        graph_service.validate_schema,
    )

    assert result == SCHEMA
    assert len(payloads) == 2
    assert payloads[1]["temperature"] == 0
    repair_prompt = payloads[1]["messages"][-1]["content"]
    assert "entity_types" in repair_prompt
    assert "不得省略必要欄位或回傳空陣列" in repair_prompt


def test_chat_json_reports_schema_validation_error_after_retry(monkeypatch) -> None:
    invalid_schema = {"entity_types": [], "relationship_types": []}
    monkeypatch.setattr(
        graph_service, "_post_json", lambda *args, **kwargs: chat_response(invalid_schema)
    )

    with pytest.raises(ValueError, match="entity_types"):
        graph_service._chat_json(
            "http://models/v1",
            "",
            "llm",
            "system",
            "user",
            0,
            graph_service.validate_schema,
        )



def test_chat_json_reports_token_truncation_after_retry(monkeypatch) -> None:
    monkeypatch.setattr(
        graph_service,
        "_post_json",
        lambda *args, **kwargs: {
            "choices": [
                {"message": {"content": '{"entity_types": ['}, "finish_reason": "length"}
            ]
        },
    )

    with pytest.raises(ValueError, match="提高 Ollama context length"):
        graph_service._chat_json("http://models/v1", "", "llm", "system", "user", 0)
