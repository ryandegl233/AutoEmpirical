"""Capture observable chat traffic without copying provider configuration.

Message text is intentionally preserved verbatim. This is a field allowlist,
not a redactor for credentials a caller has already placed in visible text.
Unknown objects are omitted rather than converted with ``str`` or ``repr``.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any


_MISSING = object()
_NUMBER = (int, float)
_REQUEST_FIELDS = {
    "model": (str,),
    "temperature": _NUMBER,
    "top_p": _NUMBER,
    "max_tokens": (int,),
    "max_completion_tokens": (int,),
    "n": (int,),
    "seed": (int,),
    "frequency_penalty": _NUMBER,
    "presence_penalty": _NUMBER,
    "stream": (bool,),
    "logprobs": (bool,),
    "top_logprobs": (int,),
    "parallel_tool_calls": (bool,),
    "reasoning_effort": (str,),
    "verbosity": (str,),
}


def _record(value: Any) -> Any:
    """Support SDK models and small test doubles without a provider import."""
    if isinstance(value, dict) or isinstance(value, type):
        return value
    try:
        dump = getattr(value, "model_dump", None)
        if callable(dump):
            result = dump()
            if isinstance(result, dict):
                return result
    except Exception:
        pass
    return value


def _get(record: Any, name: str) -> Any:
    if isinstance(record, dict):
        return record.get(name, _MISSING)
    if isinstance(record, type):
        return _MISSING
    try:
        return getattr(record, name, _MISSING)
    except Exception:
        return _MISSING


def _select(record: Any, fields: dict[str, tuple[type, ...]]) -> dict:
    result = {}
    for name, allowed_types in fields.items():
        value = _get(record, name)
        if isinstance(value, Enum):
            value = value.value
        if str in allowed_types and isinstance(value, str):
            value = str.__str__(value)
        if type(value) not in allowed_types:
            continue
        if type(value) is float and not math.isfinite(value):
            continue
        result[name] = value
    return result


def _function(value: Any) -> dict:
    return _select(_record(value), {"name": (str,), "arguments": (str,)})


def _message(value: Any) -> dict:
    record = _record(value)
    result = _select(
        record,
        {"role": (str,), "name": (str,), "tool_call_id": (str,), "refusal": (str,)},
    )
    content = _get(record, "content")
    if content is None or isinstance(content, str):
        result["content"] = None if content is None else str.__str__(content)
    elif isinstance(content, (list, tuple)):
        # Keep visible text blocks; do not copy arbitrary URLs, binary payloads,
        # provider metadata, or fields containing internal reasoning.
        blocks = []
        for block in content:
            item = _record(block)
            if _get(item, "type") in ("text", "input_text", "output_text", "refusal"):
                blocks.append(
                    _select(item, {"type": (str,), "text": (str,), "refusal": (str,)})
                )
        result["content"] = blocks
    tool_calls = _get(record, "tool_calls")
    if isinstance(tool_calls, (list, tuple)):
        calls = []
        for tool_call in tool_calls:
            item = _record(tool_call)
            call = _select(item, {"id": (str,), "type": (str,)})
            function = _function(_get(item, "function"))
            if function:
                call["function"] = function
            if call:
                calls.append(call)
        result["tool_calls"] = calls
    function_call = _function(_get(record, "function_call"))
    if function_call:
        result["function_call"] = function_call
    return result


def _response_format(value: Any) -> dict:
    if isinstance(value, type):
        return {"class_name": value.__name__}
    record = _record(value)
    result = _select(record, {"type": (str,)})
    schema = _select(
        _record(_get(record, "json_schema")), {"name": (str,), "strict": (bool,)}
    )
    if schema:
        result["json_schema"] = schema
    return result


def capture_request(kwargs: dict) -> dict:
    """Snapshot visible messages and recognized generation settings as JSON data."""
    result = _select(kwargs, _REQUEST_FIELDS)
    messages = _get(kwargs, "messages")
    if isinstance(messages, (list, tuple)):
        result["messages"] = [_message(message) for message in messages]
    stop = _get(kwargs, "stop")
    if type(stop) is str:
        result["stop"] = stop
    elif isinstance(stop, (list, tuple)):
        result["stop"] = [item for item in stop if type(item) is str]
    response_format = _response_format(_get(kwargs, "response_format"))
    if response_format:
        result["response_format"] = response_format
    return result


def capture_response(response: Any) -> dict:
    """Snapshot visible answer choices and token counts, excluding hidden content."""
    record = _record(response)
    result = _select(record, {"id": (str,), "model": (str,)})
    choices = _get(record, "choices")
    if isinstance(choices, (list, tuple)):
        captured_choices = []
        for choice in choices:
            item = _record(choice)
            captured = _select(item, {"index": (int,), "finish_reason": (str,)})
            if _get(item, "finish_reason") is None:
                captured["finish_reason"] = None
            message = _get(item, "message")
            if message is not _MISSING and message is not None:
                captured["message"] = _message(message)
            captured_choices.append(captured)
        result["choices"] = captured_choices
    usage = _record(_get(record, "usage"))
    captured_usage = _select(
        usage,
        {"prompt_tokens": (int,), "completion_tokens": (int,), "total_tokens": (int,)},
    )
    detail_fields = {
        "prompt_tokens_details": ("cached_tokens", "audio_tokens"),
        "completion_tokens_details": (
            "reasoning_tokens", "audio_tokens", "accepted_prediction_tokens", "rejected_prediction_tokens"
        ),
    }
    for detail_name, names in detail_fields.items():
        detail = _select(_record(_get(usage, detail_name)), {name: (int,) for name in names})
        if detail:
            captured_usage[detail_name] = detail
    if captured_usage:
        result["usage"] = captured_usage
    return result
