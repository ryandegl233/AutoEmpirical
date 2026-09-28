from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from Benchmark.src import ase2022_camel_mas_baseline as mas


_ADVISORY = "Instruction: Check the original crash evidence.\nInput: USE_IMMUTABLE_EVIDENCE"
_RATIONALE = json.dumps(
    {
        "decision": "accepted_fault",
        "decision_rationale": {
            "reason": "The supplied report describes an observed crash.",
            "evidence_refs": [{"field": "body", "quote": "The process crashes.\n"}],
            "evidence_limitations": "The report does not identify the cause.",
        },
    },
    ensure_ascii=False,
)


class _ScriptedProvider:
    """Replace only the external SDK boundary; CAMEL and tracing stay real."""

    def __init__(self, outputs: list[str | Exception]) -> None:
        self._outputs = iter(outputs)
        self.calls: list[dict] = []
        self.endpoints: list[str] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        self.beta = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=self.parse)))

    def create(self, **kwargs):
        return self._complete("create", kwargs)

    def parse(self, **kwargs):
        return self._complete("parse", kwargs)

    def _complete(self, endpoint: str, kwargs: dict):
        from openai.types.chat import ChatCompletion

        self.calls.append(copy.deepcopy(kwargs))
        self.endpoints.append(endpoint)
        output = next(self._outputs)
        if isinstance(output, Exception):
            raise output
        return ChatCompletion.model_validate(
            {
                "id": f"provider-response-{len(self.calls)}",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": output,
                            "reasoning_content": "private-provider-reasoning",
                        },
                    }
                ],
                "usage": {"prompt_tokens": 40, "completion_tokens": 10, "total_tokens": 50},
                "api_key": "provider-secret",
            }
        )


@pytest.fixture
def provider_factory(monkeypatch):
    pytest.importorskip("camel")
    openai = pytest.importorskip("openai")
    providers: list[_ScriptedProvider] = []
    pending: list[_ScriptedProvider] = []

    def queue(*outputs: list[str | Exception]):
        queued = [_ScriptedProvider(items) for items in outputs]
        pending.extend(queued)
        return queued

    def create(**_kwargs):
        provider = pending.pop(0)
        providers.append(provider)
        return provider

    monkeypatch.setattr(openai, "OpenAI", create)
    return queue


def _record(record_id: str) -> dict[str, str]:
    return {
        "record_id": record_id,
        "title": "Observed crash",
        "body": "The process crashes.\nAn error is printed.",
        "comments": "[]",
        "state": "closed",
        "created_at": "2021-01-01T00:00:00Z",
        "issue_url": f"https://example.test/{record_id}",
    }


def _society_factory(*, capture_trace: bool = True, society_mode: str = "evidence_anchored"):
    return mas.make_camel_society_factory(
        "gpt-4o-mini",
        "client-secret",
        "https://example.test/v1",
        max_retries=0,
        capture_trace=capture_trace,
        society_mode=society_mode,
    )


def _run(record_id: str, society_factory, **kwargs):
    return mas.run_roleplaying_society_record(
        _record(record_id),
        stage="stage2",
        taxonomy={"symptom": ["Crash"], "root_cause": ["Cause"]},
        model="gpt-4o-mini",
        society_factory=society_factory,
        **kwargs,
    )


def test_real_society_preserves_provider_messages_and_binds_each_role_attempt(provider_factory) -> None:
    user, assistant = provider_factory(
        [_ADVISORY, _ADVISORY],
        ["Need another review.\n", _RATIONALE],
    )

    row = _run(
        "issue-a", _society_factory(), max_turns=2, explanation_mode="evidence_rationale"
    )

    assert row["invalid"] is False, row["error"]
    trace = row["explanation_audit"]["request_trace"]
    assert {entry["call_id"] for entry in trace} == {
        "issue-a:ai_user:1", "issue-a:ai_user:2",
        "issue-a:ai_assistant:1", "issue-a:ai_assistant:2",
    }
    for role, provider in (("ai_user", user), ("ai_assistant", assistant)):
        entries = [entry for entry in trace if entry["role"] == role]
        assert len(entries) == 2
        for index, entry in enumerate(entries):
            assert entry["record_id"] == "issue-a"
            assert entry["attempt"] == index + 1
            assert entry["request"]["messages"] == provider.calls[index]["messages"], {
                "provider_content_types": [type(message.get("content")).__name__ for message in provider.calls[index]["messages"]]
            }
            assert entry["request"]["model"] == "gpt-4o-mini"
            assert entry["request"]["temperature"] == 0.0
            assert entry["response"]["id"] == f"provider-response-{index + 1}"
            assert entry["response"]["usage"]["total_tokens"] == 50
            assert entry["error_type"] is None
    assistant_entries = [entry for entry in trace if entry["role"] == "ai_assistant"]
    assert assistant_entries[0]["response"]["choices"][0]["message"]["content"] == "Need another review.\n"
    assert assistant_entries[1]["response"]["choices"][0]["message"]["content"] == _RATIONALE
    assert any(
        message.get("role") == "assistant" and message.get("content") == "Need another review.\n"
        for message in assistant_entries[1]["request"]["messages"]
    )
    assert row["society"]["api_request_count"] == len(trace) == 4
    serialized = json.dumps(trace, ensure_ascii=False, allow_nan=False)
    assert "provider-secret" not in serialized
    assert "client-secret" not in serialized
    assert "private-provider-reasoning" not in serialized


def test_failed_sdk_call_is_retained_in_invalid_society_audit(provider_factory) -> None:
    provider_factory([_ADVISORY], [TimeoutError("test provider timeout")])

    row = _run(
        "issue-failed", _society_factory(), max_turns=1, explanation_mode="evidence_rationale"
    )

    assert row["invalid"] is True
    audit = row["explanation_audit"]
    assert audit["status"] == "invalid"
    assert audit["request_trace_available"] is True
    failed = [entry for entry in audit["request_trace"] if entry["role"] == "ai_assistant"]
    assert len(failed) == 1
    assert failed[0]["call_id"] == "issue-failed:ai_assistant:1"
    assert failed[0]["record_id"] == "issue-failed"
    assert failed[0]["error_type"] == "TimeoutError"
    assert failed[0]["response"] is None
    assert "The process crashes.\n" in failed[0]["request"]["messages"][-1]["content"]
    assert row["society"]["api_request_count"] == 2
    assert row["society"]["role_request_stats"]["ai_assistant"]["usage_observed_request_count"] == 0


def test_forced_finalizer_parse_calls_and_failed_retry_reach_the_record_audit(provider_factory) -> None:
    _, _, finalizer = provider_factory(
        [_ADVISORY], ["No final classification yet."], [TimeoutError("test timeout"), _RATIONALE]
    )
    finalizer_factory = mas.make_camel_agent_factory(
        "gpt-4o-mini", "client-secret", "https://example.test/v1", max_retries=0, capture_trace=True
    )

    row = _run(
        "issue-final", _society_factory(), max_turns=1,
        explanation_mode="evidence_rationale", finalizer_factory=finalizer_factory,
        finalizer_max_retries=1,
    )

    assert row["invalid"] is False, row["error"]
    assert row["output_source"] == "forced_finalizer"
    entries = [entry for entry in row["explanation_audit"]["request_trace"] if entry["role"] == "forced_finalizer"]
    assert [entry["call_id"] for entry in entries] == [
        "issue-final:forced_finalizer:1", "issue-final:forced_finalizer:2"
    ]
    assert finalizer.endpoints == ["parse", "parse"]
    assert entries[0]["error_type"] == "TimeoutError"
    assert entries[0]["response"] is None
    assert entries[1]["error_type"] is None
    assert entries[1]["request"]["response_format"] == {"class_name": "Stage2RationaleOutput"}
    assert entries[1]["response"]["choices"][0]["message"]["content"] == _RATIONALE
    for entry, call in zip(entries, finalizer.calls, strict=True):
        assert entry["request"]["messages"] == call["messages"]
        assert entry["record_id"] == "issue-final"
    assert row["society"]["api_request_count"] == 4
    assert row["society"]["forced_finalizer"]["api_request_count"] == 2


def test_default_label_only_run_has_no_full_request_capture(provider_factory) -> None:
    provider_factory([_ADVISORY], ['{"decision":"accepted_fault"}'])
    created = []
    real_factory = _society_factory(capture_trace=False)

    def factory(task):
        society = real_factory(task)
        created.append(society)
        return society

    row = _run("issue-default", factory, max_turns=1)

    assert row["invalid"] is False, row["error"]
    assert row["final_prediction"] == {"decision": "accepted_fault"}
    assert "explanation_audit" not in row
    assert "request_trace" not in json.dumps(row)
    assert all("request_trace" not in stats for stats in created[0]._mas_role_request_stats.values())


def test_native_task_specifier_failure_preserves_calls_before_society_exists(provider_factory) -> None:
    _, _, task_specifier = provider_factory([], [], [TimeoutError("task specification failed")])

    row = _run(
        "issue-init", _society_factory(society_mode="native"), max_turns=1,
        society_mode="native", explanation_mode="evidence_rationale",
    )

    assert row["invalid"] is True
    assert row["society"]["turn_count"] == 0
    assert row["society"]["stop_reason"] == "error"
    assert row["society"]["api_request_count"] == 1
    assert row["explanation_audit"]["request_trace_available"] is True
    trace = row["explanation_audit"]["request_trace"]
    assert len(trace) == 1
    assert trace[0]["role"] == "task_specifier"
    assert trace[0]["record_id"] == "issue-init"
    assert trace[0]["call_id"] == "issue-init:task_specifier:1"
    assert trace[0]["error_type"] == "TimeoutError"
    assert trace[0]["response"] is None
    assert trace[0]["request"]["messages"] == task_specifier.calls[0]["messages"]


def _http_completion(content: str) -> dict:
    return {
        "id": "http-completion-1", "object": "chat.completion", "created": 0,
        "model": "gpt-4o-mini",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 40, "completion_tokens": 10, "total_tokens": 50},
    }


def test_real_sdk_malformed_parse_content_survives_in_bound_finalizer_audit(monkeypatch) -> None:
    pytest.importorskip("camel")
    openai = pytest.importorskip("openai")
    httpx = pytest.importorskip("httpx")
    actual_openai = openai.OpenAI
    malformed = '{"decision":\n'
    plans = [[_ADVISORY], ["Still reviewing the evidence."], [malformed, _RATIONALE]]
    clients = []

    def create(**kwargs):
        outputs = iter(plans.pop(0))

        def handle(request):
            assert request.url.host == "example.test"
            return httpx.Response(200, json=_http_completion(next(outputs)))

        client = actual_openai(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handle)))
        clients.append(client)
        return client

    monkeypatch.setattr(openai, "OpenAI", create)
    finalizer_factory = mas.make_camel_agent_factory(
        "gpt-4o-mini", "transport-secret", "https://example.test/v1", max_retries=0, capture_trace=True
    )
    try:
        row = _run(
            "issue-http", _society_factory(), max_turns=1,
            explanation_mode="evidence_rationale", finalizer_factory=finalizer_factory,
            finalizer_max_retries=0,
        )
    finally:
        for client in clients:
            client.close()

    assert row["invalid"] is False, row["error"]
    entries = [entry for entry in row["explanation_audit"]["request_trace"] if entry["role"] == "forced_finalizer"]
    assert [entry["call_id"] for entry in entries] == [
        "issue-http:forced_finalizer:1", "issue-http:forced_finalizer:2"
    ]
    first = entries[0]
    assert first["record_id"] == "issue-http"
    assert first["error_type"] == "ValidationError"
    assert first["response"] is None
    assert len(first["http_exchanges"]) == 1
    exchange = first["http_exchanges"][0]
    assert exchange["sdk_attempt"] == 1
    assert exchange["status_code"] == 200
    assert exchange["read_error_type"] is None
    assert exchange["response"]["choices"][0]["message"]["content"] == malformed
    assert exchange["request"]["messages"] == first["request"]["messages"]
    assert entries[1]["http_exchanges"][0]["response"]["choices"][0]["message"]["content"] == _RATIONALE
    assert entries[1]["response"]["choices"][0]["message"]["content"] == _RATIONALE
    assert "transport-secret" not in json.dumps(entries, allow_nan=False)


def test_real_sdk_internal_http_retry_is_captured_under_one_sdk_call() -> None:
    openai = pytest.importorskip("openai")
    httpx = pytest.importorskip("httpx")
    statuses = iter([500, 200])

    def handle(_request):
        status = next(statuses)
        payload = {"error": {"message": "temporary failure", "type": "server_error"}} if status == 500 else _http_completion("visible retry result\n")
        return httpx.Response(status, json=payload)

    with openai.OpenAI(
        api_key="retry-secret", base_url="https://example.test/v1", max_retries=1,
        http_client=httpx.Client(transport=httpx.MockTransport(handle)),
    ) as sdk:
        counting = mas._CountingOpenAIClient(sdk, capture_trace=True)
        response = counting.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": "Question\nwith exact whitespace."}]
        )

    assert response.choices[0].message.content == "visible retry result\n"
    trace = counting.request_stats["request_trace"]
    assert len(trace) == 1
    assert counting.request_stats["api_request_count"] == 1
    assert [exchange["status_code"] for exchange in trace[0]["http_exchanges"]] == [500, 200]
    assert [exchange["sdk_attempt"] for exchange in trace[0]["http_exchanges"]] == [1, 1]
    assert trace[0]["http_exchanges"][1]["request"]["messages"] == [
        {"role": "user", "content": "Question\nwith exact whitespace."}
    ]
    assert trace[0]["http_exchanges"][1]["response"]["choices"][0]["message"]["content"] == "visible retry result\n"
    assert "retry-secret" not in json.dumps(trace, allow_nan=False)


def test_real_sdk_default_client_does_not_install_http_capture() -> None:
    openai = pytest.importorskip("openai")
    httpx = pytest.importorskip("httpx")
    with openai.OpenAI(
        api_key="default-secret", base_url="https://example.test/v1", max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(lambda _request: httpx.Response(200, json=_http_completion("answer")))),
    ) as sdk:
        initial_hooks = list(sdk._client.event_hooks["response"])
        counting = mas._CountingOpenAIClient(sdk)
        response = counting.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "question"}])

        assert response.choices[0].message.content == "answer"
        assert sdk._client.event_hooks["response"] == initial_hooks
        assert "request_trace" not in counting.request_stats
        assert "http_exchange_trace" not in counting.request_stats
