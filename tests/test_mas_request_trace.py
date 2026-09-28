from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from Benchmark.src.mas_request_trace import capture_request, capture_response


def test_request_preserves_the_actual_conversation_and_generation_settings() -> None:
    messages = [
        {"role": "system", "content": "Classify the supplied evidence.\nKeep uncertainty."},
        {"role": "user", "content": "证据 A\n\n  indented code\r\n"},
        {"role": "assistant", "content": '{"label":"unknown"}\n'},
    ]

    captured = capture_request(
        {
            "messages": messages,
            "model": "test-model",
            "temperature": 0.0,
            "max_tokens": 512,
            "max_completion_tokens": 1024,
            "top_p": 0.9,
            "seed": 7,
            "stop": ["END\n"],
            "stream": False,
        }
    )

    assert json.loads(json.dumps(captured, ensure_ascii=False, allow_nan=False)) == {
        "messages": messages,
        "model": "test-model",
        "temperature": 0.0,
        "max_tokens": 512,
        "max_completion_tokens": 1024,
        "top_p": 0.9,
        "seed": 7,
        "stop": ["END\n"],
        "stream": False,
    }
    messages[1]["content"] = "later mutation"
    assert captured["messages"][1]["content"] == "证据 A\n\n  indented code\r\n"


def test_request_omits_transport_secrets_even_inside_known_configuration_fields() -> None:
    secret = "credential-must-not-be-in-trace"
    captured = capture_request(
        {
            "messages": [{"role": "user", "content": "visible", "api_key": secret}],
            "model": "test-model",
            "api_key": secret,
            "base_url": f"https://user:{secret}@example.test",
            "extra_headers": {"Authorization": secret},
            "extra_body": {"api_key": secret},
            "metadata": {"Authorization": secret},
            "temperature": {"api_key": secret},
            "stop": ["END", {"Authorization": secret}],
            "response_format": {
                "type": "json_schema",
                "api_key": secret,
                "json_schema": {
                    "name": "Verdict",
                    "strict": True,
                    "schema": {"properties": {"credential": {"default": secret}}},
                    "headers": {"Authorization": secret},
                },
            },
        }
    )

    assert secret not in json.dumps(captured, allow_nan=False)
    assert captured["messages"] == [{"role": "user", "content": "visible"}]
    assert captured["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "Verdict", "strict": True},
    }
    assert "temperature" not in captured


def test_response_schema_class_is_named_without_dumping_its_fields() -> None:
    class Verdict:
        credential = "class-secret-must-not-be-in-trace"

    captured = capture_request({"response_format": Verdict})

    assert captured == {"response_format": {"class_name": "Verdict"}}


@pytest.mark.parametrize("as_object", [False, True])
def test_response_records_visible_answer_finish_reason_and_usage(as_object: bool) -> None:
    message = {
        "role": "assistant",
        "content": '{"decision":"accepted",\n"rationale":"Observed failure."}\n',
        "reasoning_content": "private reasoning excluded",
        "parsed": {"credential": "parsed-object-secret"},
    }
    choice = {"index": 0, "message": message, "finish_reason": "stop"}
    usage = {
        "prompt_tokens": 100,
        "completion_tokens": 25,
        "total_tokens": 125,
        "completion_tokens_details": {"reasoning_tokens": 3, "api_key": "usage-secret"},
    }
    response = {
        "id": "completion-123",
        "model": "test-model",
        "choices": [choice],
        "usage": usage,
        "api_key": "response-secret",
    }
    if as_object:
        choice["message"] = SimpleNamespace(**message)
        response["choices"] = [SimpleNamespace(**choice)]
        response["usage"] = SimpleNamespace(**usage)
        response = SimpleNamespace(**response)

    captured = capture_response(response)

    assert captured["choices"] == [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": '{"decision":"accepted",\n"rationale":"Observed failure."}\n',
            },
            "finish_reason": "stop",
        }
    ]
    assert captured["id"] == "completion-123"
    assert captured["model"] == "test-model"
    assert captured["usage"]["total_tokens"] == 125
    assert "secret" not in json.dumps(captured, allow_nan=False)
    assert "private reasoning" not in json.dumps(captured)


def test_model_dump_response_does_not_leak_extra_provider_fields() -> None:
    class DumpableResponse:
        def model_dump(self):
            return {
                "id": "dump-1",
                "choices": [{"message": {"role": "assistant", "content": "answer\n"}}],
                "connection": {"api_key": "dump-secret"},
            }

    assert capture_response(DumpableResponse()) == {
        "id": "dump-1",
        "choices": [{"message": {"role": "assistant", "content": "answer\n"}}],
    }


def test_structured_text_and_tool_messages_keep_visible_evidence() -> None:
    captured = capture_request(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "exact\ntext", "api_key": "secret"}],
                },
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "read_evidence", "arguments": '{"id":"a"}\n'},
                            "headers": {"Authorization": "secret"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call-1", "content": "Evidence result.\n"},
            ]
        }
    )

    assert captured["messages"][0]["content"] == [{"type": "text", "text": "exact\ntext"}]
    assert captured["messages"][1]["content"] is None
    assert captured["messages"][1]["tool_calls"][0]["function"] == {
        "name": "read_evidence",
        "arguments": '{"id":"a"}\n',
    }
    assert captured["messages"][2] == {
        "role": "tool", "tool_call_id": "call-1", "content": "Evidence result.\n"
    }
    assert "secret" not in json.dumps(captured)


def test_unknown_objects_and_nonfinite_numbers_never_use_repr_or_break_json() -> None:
    class CredentialObject:
        def __repr__(self):
            raise AssertionError("Opaque provider objects must never be represented")

        def __str__(self):
            raise AssertionError("Opaque provider objects must never be stringified")

    opaque = CredentialObject()
    captured = capture_request(
        {
            "model": opaque,
            "temperature": float("nan"),
            "max_tokens": opaque,
            "messages": [{"role": "user", "content": opaque}],
            "response_format": opaque,
        }
    )

    json.dumps(captured, allow_nan=False)
    assert "model" not in captured
    assert "temperature" not in captured
    assert "max_tokens" not in captured
    assert "content" not in captured["messages"][0]
    assert capture_response(opaque) == {}
