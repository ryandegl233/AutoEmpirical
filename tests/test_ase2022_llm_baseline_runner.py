import http.client
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from Benchmark.src import ase2022_llm_baseline as baseline


TAXONOMY = {
    "symptom": ["Crash"],
    "root_cause": ["Incorrect Code Logic"],
}


class _ControlledCompletionEvent:
    def __init__(self) -> None:
        self.wait_entered = threading.Event()
        self._condition = threading.Condition()
        self._is_set = False
        self._timeout_triggered = False

    def set(self) -> None:
        with self._condition:
            self._is_set = True
            self._condition.notify_all()

    def trigger_timeout(self) -> None:
        with self._condition:
            self._timeout_triggered = True
            self._condition.notify_all()

    def wait(self, timeout: float | None = None) -> bool:
        del timeout
        self.wait_entered.set()
        with self._condition:
            assert self._condition.wait_for(
                lambda: self._is_set or self._timeout_triggered,
                timeout=5.0,
            )
            return self._is_set

    def is_set(self) -> bool:
        with self._condition:
            return self._is_set


def _install_controlled_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> _ControlledCompletionEvent:
    completed = _ControlledCompletionEvent()
    events = iter((completed, threading.Event()))
    monkeypatch.setattr(
        baseline,
        "threading",
        SimpleNamespace(
            Event=lambda: next(events),
            Lock=threading.Lock,
            Thread=threading.Thread,
        ),
    )
    return completed


class _ReusableHTTPConnection:
    def __init__(
        self,
        host: str,
        *,
        timeout: float,
        entered: threading.Event | None = None,
        release: threading.Event | None = None,
    ) -> None:
        self.host = host
        self.timeout = timeout
        self.entered = entered
        self.release = release
        self.requests: list[tuple[str, str, bytes, dict[str, str]]] = []
        self.closed = False

    def connect(self) -> None:
        return None

    def request(
        self,
        method: str,
        path: str,
        *,
        body: bytes,
        headers: dict[str, str],
    ) -> None:
        self.requests.append((method, path, body, headers))
        if self.entered is not None:
            self.entered.set()
        if self.release is not None:
            assert self.release.wait(timeout=2)

    def getresponse(self) -> "_FakeHTTPResponse":
        return _FakeHTTPResponse(
            body=json.dumps(
                {
                    "choices": [{"message": {"content": '{"answer":"ok"}'}}],
                    "usage": {"prompt_tokens": 11, "completion_tokens": 7},
                }
            ).encode("utf-8")
        )

    def close(self) -> None:
        self.closed = True


class _FakeHTTPResponse:
    def __init__(
        self,
        *,
        body: bytes = b"",
        lines: tuple[bytes, ...] = (),
    ) -> None:
        self.body = body
        self.lines = lines
        self.status = 200
        self.headers: dict[str, str] = {}

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def __iter__(self):
        return (
            physical_line
            for chunk in self.lines
            for physical_line in chunk.splitlines(keepends=True)
        )

    def read(self) -> bytes:
        return self.body


class _CloseRequiredStreamResponse(_FakeHTTPResponse):
    def read(self) -> bytes:
        raise AssertionError("completed SSE response must not be drained")


class _CloseRequiredStreamConnection(_ReusableHTTPConnection):
    def __init__(self, host: str, *, timeout: float) -> None:
        super().__init__(host, timeout=timeout)
        self.active_response: _CloseRequiredStreamResponse | None = None
        self.close_count = 0

    def request(
        self,
        method: str,
        path: str,
        *,
        body: bytes,
        headers: dict[str, str],
    ) -> None:
        if self.active_response is not None and self.close_count < len(self.requests):
            raise http.client.ResponseNotReady("prior stream connection was not closed")
        super().request(method, path, body=body, headers=headers)

    def getresponse(self) -> _CloseRequiredStreamResponse:
        response = _CloseRequiredStreamResponse(lines=_complete_stream())
        self.active_response = response
        return response

    def close(self) -> None:
        super().close()
        self.close_count += 1


class _SlowTrickleResponse(_FakeHTTPResponse):
    def __init__(self, closed: threading.Event) -> None:
        super().__init__(
            body=json.dumps(
                {
                    "choices": [{"message": {"content": '{"answer":"ok"}'}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                }
            ).encode("utf-8")
        )
        self._closed = closed

    def read(self) -> bytes:
        collected = bytearray()
        for byte in self.body:
            if self._closed.wait(timeout=0.005):
                raise OSError("response closed at absolute deadline")
            collected.append(byte)
        return bytes(collected)

    def close(self) -> None:
        self._closed.set()


class _SlowTrickleConnection(_ReusableHTTPConnection):
    def __init__(self, host: str, *, timeout: float) -> None:
        super().__init__(host, timeout=timeout)
        self.closed_event = threading.Event()
        self.response = _SlowTrickleResponse(self.closed_event)

    def getresponse(self) -> _SlowTrickleResponse:
        return self.response

    def close(self) -> None:
        super().close()
        self.closed_event.set()


class _BlockedHeaderConnection(_ReusableHTTPConnection):
    def __init__(self, host: str, *, timeout: float) -> None:
        super().__init__(host, timeout=timeout)
        self.closed_event = threading.Event()
        self.finished = threading.Event()

    def getresponse(self) -> _FakeHTTPResponse:
        try:
            assert self.closed_event.wait(timeout=2.0)
            raise OSError("connection closed while waiting for headers")
        finally:
            self.finished.set()

    def close(self) -> None:
        super().close()
        self.closed_event.set()


class _UninterruptibleHeaderConnection(_ReusableHTTPConnection):
    """Simulate an OS/DNS phase which cannot be joined inside the deadline."""

    def getresponse(self) -> _FakeHTTPResponse:
        time.sleep(0.5)
        raise OSError("late header failure")


class _DelayedConnectConnection(_ReusableHTTPConnection):
    def __init__(self, host: str, *, timeout: float) -> None:
        super().__init__(host, timeout=timeout)
        self.connect_entered = threading.Event()
        self.release_connect = threading.Event()
        self.worker_closed = threading.Event()
        self.connected = False
        self.request_count = 0
        self._close_count = 0
        self._close_lock = threading.Lock()

    def connect(self) -> None:
        self.connect_entered.set()
        assert self.release_connect.wait(timeout=2.0)
        self.connected = True

    def request(
        self,
        method: str,
        path: str,
        *,
        body: bytes,
        headers: dict[str, str],
    ) -> None:
        if not self.connected:
            self.connect()
        self.request_count += 1
        super().request(method, path, body=body, headers=headers)

    def close(self) -> None:
        super().close()
        with self._close_lock:
            self._close_count += 1
            if self._close_count >= 2:
                self.worker_closed.set()


class _BlockedRequestConnection(_ReusableHTTPConnection):
    def __init__(self, host: str, *, timeout: float) -> None:
        super().__init__(host, timeout=timeout)
        self.request_started = threading.Event()
        self.release_request = threading.Event()

    def request(
        self,
        method: str,
        path: str,
        *,
        body: bytes,
        headers: dict[str, str],
    ) -> None:
        super().request(method, path, body=body, headers=headers)
        self.request_started.set()
        assert self.release_request.wait(timeout=2.0)


class _CleanupFailureResponse(_FakeHTTPResponse):
    def __init__(self) -> None:
        super().__init__(
            body=json.dumps(
                {
                    "choices": [{"message": {"content": '{"answer":"ok"}'}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                }
            ).encode("utf-8")
        )
        self.read_completed = threading.Event()
        self.close_count = 0

    def read(self) -> bytes:
        self.read_completed.set()
        return super().read()

    def close(self) -> None:
        self.close_count += 1
        raise RuntimeError("response cleanup failed")


class _CleanupFailureConnection(_ReusableHTTPConnection):
    def __init__(self, host: str, *, timeout: float) -> None:
        super().__init__(host, timeout=timeout)
        self.response = _CleanupFailureResponse()
        self.close_count = 0

    def getresponse(self) -> _CleanupFailureResponse:
        return self.response

    def close(self) -> None:
        self.closed = True
        self.close_count += 1
        raise RuntimeError("connection cleanup failed")


class _BlockedRequestCleanupFailureConnection(_BlockedRequestConnection):
    def __init__(self, host: str, *, timeout: float) -> None:
        super().__init__(host, timeout=timeout)
        self.close_count = 0
        self.second_close = threading.Event()

    def close(self) -> None:
        self.closed = True
        self.close_count += 1
        if self.close_count >= 2:
            self.second_close.set()
        raise RuntimeError("connection cleanup failed")


def test_pooled_chat_client_reuses_one_connection_without_changing_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connections: list[_ReusableHTTPConnection] = []

    def connection_factory(host: str, *, timeout: float) -> _ReusableHTTPConnection:
        connection = _ReusableHTTPConnection(host, timeout=timeout)
        connections.append(connection)
        return connection

    monkeypatch.setattr(baseline.http.client, "HTTPSConnection", connection_factory)
    client = baseline.PooledChatCompletionClient(
        base_url="https://api-slb.micuapi.ai/v1",
        max_connections=1,
    )

    first = client.call_model(
        system_prompt="system",
        user_prompt="user",
        model="deepseek-v4-flash",
        api_key="secret",
        wire_api="chat_completions",
        max_tokens=321,
        json_mode=True,
        thinking_enabled=False,
        user_agent="test-agent",
    )
    second = client.call_model(
        system_prompt="system",
        user_prompt="user",
        model="deepseek-v4-flash",
        api_key="secret",
        wire_api="chat_completions",
        max_tokens=321,
        json_mode=True,
        thinking_enabled=False,
        user_agent="test-agent",
    )
    client.close()

    assert first == second == '{"answer":"ok"}'
    assert len(connections) == 1
    assert len(connections[0].requests) == 2
    method, path, body, headers = connections[0].requests[0]
    assert (method, path) == ("POST", "/v1/chat/completions")
    assert json.loads(body) == {
        "model": "deepseek-v4-flash",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ],
        "temperature": 0.0,
        "max_tokens": 321,
        "response_format": {"type": "json_object"},
        "thinking": {"type": "disabled"},
    }
    assert headers == {
        "Authorization": "Bearer secret",
        "Content-Type": "application/json",
        "User-Agent": "test-agent",
    }
    assert connections[0].closed is True


def test_fresh_chat_completion_uses_per_call_timeout_and_closes_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connections: list[_ReusableHTTPConnection] = []

    def connection_factory(host: str, *, timeout: float) -> _ReusableHTTPConnection:
        connection = _ReusableHTTPConnection(host, timeout=timeout)
        connections.append(connection)
        return connection

    monkeypatch.setattr(baseline.http.client, "HTTPSConnection", connection_factory)

    first = baseline.call_fresh_chat_completion(
        system_prompt="system",
        user_prompt="user",
        model="deepseek-v4-flash",
        api_key="secret",
        base_url="https://api-slb.micuapi.ai/v1",
        timeout_seconds=12.5,
        max_tokens=321,
        json_mode=True,
        user_agent="test-agent",
    )
    second = baseline.call_fresh_chat_completion(
        system_prompt="system",
        user_prompt="user",
        model="deepseek-v4-flash",
        api_key="secret",
        base_url="https://api-slb.micuapi.ai/v1",
        timeout_seconds=7.0,
    )

    assert first == second == '{"answer":"ok"}'
    assert [connection.timeout for connection in connections] == [12.5, 7.0]
    assert all(connection.closed for connection in connections)
    assert len(connections) == 2
    assert json.loads(connections[0].requests[0][2])["thinking"] == {"type": "disabled"}
    assert "stream" not in json.loads(connections[0].requests[0][2])


def test_fresh_chat_completion_streams_when_explicitly_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connections: list[_CloseRequiredStreamConnection] = []

    def connection_factory(
        host: str, *, timeout: float
    ) -> _CloseRequiredStreamConnection:
        connection = _CloseRequiredStreamConnection(host, timeout=timeout)
        connections.append(connection)
        return connection

    monkeypatch.setattr(baseline.http.client, "HTTPSConnection", connection_factory)

    result = baseline.call_fresh_chat_completion(
        system_prompt="system",
        user_prompt="user",
        model="deepseek-v4-flash-0731",
        api_key="secret",
        base_url="https://www.micuapi.ai/v1",
        timeout_seconds=75.0,
        max_tokens=2400,
        json_mode=True,
        user_agent="test-agent",
        stream=True,
    )

    payload = json.loads(connections[0].requests[0][2])
    assert payload["thinking"] == {"type": "disabled"}
    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}
    assert result == '{"answer":"ok"}'
    assert result.prompt_tokens == 101
    assert result.completion_tokens == 23
    assert result.reasoning_characters == 12
    assert connections[0].close_count == 1


def test_fresh_chat_completion_omits_provider_specific_thinking_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connections: list[_CloseRequiredStreamConnection] = []

    def connection_factory(
        host: str, *, timeout: float
    ) -> _CloseRequiredStreamConnection:
        connection = _CloseRequiredStreamConnection(host, timeout=timeout)
        connections.append(connection)
        return connection

    monkeypatch.setattr(baseline.http.client, "HTTPSConnection", connection_factory)

    baseline.call_fresh_chat_completion(
        system_prompt="system",
        user_prompt="user",
        model="gemini-3.7-flash",
        api_key="secret",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        timeout_seconds=75.0,
        max_tokens=2400,
        json_mode=True,
        thinking_enabled=None,
        stream=True,
    )

    payload = json.loads(connections[0].requests[0][2])
    assert "thinking" not in payload
    assert payload["stream"] is True


def test_fresh_chat_completion_stops_a_slow_trickle_at_absolute_request_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connections: list[_SlowTrickleConnection] = []

    def connection_factory(host: str, *, timeout: float) -> _SlowTrickleConnection:
        connection = _SlowTrickleConnection(host, timeout=timeout)
        connections.append(connection)
        return connection

    monkeypatch.setattr(baseline.http.client, "HTTPSConnection", connection_factory)
    started = time.monotonic()

    with pytest.raises(TimeoutError, match="absolute request deadline"):
        baseline.call_fresh_chat_completion(
            system_prompt="system",
            user_prompt="user",
            model="deepseek-v4-flash",
            api_key="secret",
            base_url="https://api-slb.micuapi.ai/v1",
            timeout_seconds=0.05,
            json_mode=True,
        )

    assert time.monotonic() - started < 0.5
    assert connections[0].closed is True
    assert connections[0].closed_event.is_set()


def test_fresh_chat_completion_actively_cancels_headers_at_earlier_global_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connections: list[_BlockedHeaderConnection] = []

    def connection_factory(host: str, *, timeout: float) -> _BlockedHeaderConnection:
        connection = _BlockedHeaderConnection(host, timeout=timeout)
        connections.append(connection)
        return connection

    monkeypatch.setattr(baseline.http.client, "HTTPSConnection", connection_factory)
    started = time.monotonic()

    with pytest.raises(TimeoutError, match="absolute request deadline"):
        baseline.call_fresh_chat_completion(
            system_prompt="system",
            user_prompt="user",
            model="deepseek-v4-flash",
            api_key="secret",
            base_url="https://api-slb.micuapi.ai/v1",
            timeout_seconds=5.0,
            deadline_monotonic=started + 0.05,
            json_mode=True,
        )

    assert time.monotonic() - started < 0.5
    assert connections[0].closed is True
    assert connections[0].finished.wait(timeout=0.2)


def test_fresh_chat_completion_never_joins_a_stuck_worker_past_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connections: list[_UninterruptibleHeaderConnection] = []

    def connection_factory(
        host: str, *, timeout: float
    ) -> _UninterruptibleHeaderConnection:
        connection = _UninterruptibleHeaderConnection(host, timeout=timeout)
        connections.append(connection)
        return connection

    monkeypatch.setattr(baseline.http.client, "HTTPSConnection", connection_factory)
    started = time.monotonic()

    with pytest.raises(TimeoutError, match="absolute request deadline"):
        baseline.call_fresh_chat_completion(
            system_prompt="system",
            user_prompt="user",
            model="deepseek-v4-flash",
            api_key="secret",
            base_url="https://api-slb.micuapi.ai/v1",
            timeout_seconds=0.03,
            json_mode=True,
        )

    assert time.monotonic() - started < 0.08
    assert connections[0].closed is True


def test_fresh_deadline_cancels_after_delayed_connect_before_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed = _install_controlled_completion(monkeypatch)
    connection = _DelayedConnectConnection("api-slb.micuapi.ai", timeout=10.0)
    monkeypatch.setattr(
        baseline.http.client,
        "HTTPSConnection",
        lambda _host, *, timeout: connection,
    )
    executor = ThreadPoolExecutor(max_workers=1)

    try:
        future = executor.submit(
            baseline.call_fresh_chat_completion,
            system_prompt="system",
            user_prompt="user",
            model="deepseek-v4-flash",
            api_key="secret",
            base_url="https://api-slb.micuapi.ai/v1",
            timeout_seconds=10.0,
            json_mode=True,
        )
        assert connection.connect_entered.wait(timeout=2.0)
        assert completed.wait_entered.wait(timeout=2.0)
        completed.trigger_timeout()
        with pytest.raises(TimeoutError, match="absolute request deadline"):
            future.result(timeout=2.0)
    finally:
        connection.release_connect.set()
        executor.shutdown(wait=True)

    assert connection.worker_closed.wait(timeout=2.0)
    assert completed.is_set()
    assert connection.request_count == 0


def test_fresh_worker_does_not_send_when_connect_crosses_deadline_before_watchdog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ControlledClock:
        def __init__(self) -> None:
            self.now = 100.0

        def monotonic(self) -> float:
            return self.now

    clock = ControlledClock()

    class DeadlineCrossingConnection(_ReusableHTTPConnection):
        def __init__(self, host: str, *, timeout: float) -> None:
            super().__init__(host, timeout=timeout)
            self.request_count = 0

        def connect(self) -> None:
            clock.now = 102.0

        def request(
            self,
            method: str,
            path: str,
            *,
            body: bytes,
            headers: dict[str, str],
        ) -> None:
            self.request_count += 1
            super().request(method, path, body=body, headers=headers)

    class InlineThread:
        def __init__(self, *, target: object, **_kwargs: object) -> None:
            assert callable(target)
            self._target = target

        def start(self) -> None:
            self._target()

    connection = DeadlineCrossingConnection("api-slb.micuapi.ai", timeout=1.0)
    monkeypatch.setattr(baseline, "time", SimpleNamespace(monotonic=clock.monotonic))
    monkeypatch.setattr(
        baseline,
        "threading",
        SimpleNamespace(
            Event=threading.Event,
            Lock=threading.Lock,
            Thread=InlineThread,
        ),
    )
    monkeypatch.setattr(
        baseline.http.client,
        "HTTPSConnection",
        lambda _host, *, timeout: connection,
    )
    release_count = 0

    def release() -> None:
        nonlocal release_count
        release_count += 1

    completion_lease = baseline.RequestCompletionLease(release)

    with pytest.raises(TimeoutError, match="absolute request deadline"):
        baseline.call_fresh_chat_completion(
            system_prompt="system",
            user_prompt="user",
            model="deepseek-v4-flash",
            api_key="secret",
            base_url="https://api-slb.micuapi.ai/v1",
            timeout_seconds=1.0,
            json_mode=True,
            completion_lease=completion_lease,
        )

    completion_lease.release_if_caller_owned()
    completion_lease.release_from_worker()
    assert connection.request_count == 0
    assert release_count == 1


def test_fresh_deadline_retains_completion_lease_until_started_request_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed = _install_controlled_completion(monkeypatch)
    connection = _BlockedRequestConnection("api-slb.micuapi.ai", timeout=10.0)
    monkeypatch.setattr(
        baseline.http.client,
        "HTTPSConnection",
        lambda _host, *, timeout: connection,
    )
    admission = threading.BoundedSemaphore(1)
    assert admission.acquire(blocking=False)
    release_count = 0
    lease_released = threading.Event()

    def release_admission() -> None:
        nonlocal release_count
        release_count += 1
        admission.release()
        lease_released.set()

    completion_lease = baseline.RequestCompletionLease(release_admission)
    executor = ThreadPoolExecutor(max_workers=1)

    try:
        future = executor.submit(
            baseline.call_fresh_chat_completion,
            system_prompt="system",
            user_prompt="user",
            model="deepseek-v4-flash",
            api_key="secret",
            base_url="https://api-slb.micuapi.ai/v1",
            timeout_seconds=10.0,
            json_mode=True,
            completion_lease=completion_lease,
        )
        assert connection.request_started.wait(timeout=2.0)
        assert completed.wait_entered.wait(timeout=2.0)
        completed.trigger_timeout()
        with pytest.raises(TimeoutError, match="absolute request deadline"):
            future.result(timeout=2.0)
        assert not admission.acquire(blocking=False)
    finally:
        connection.release_request.set()
        executor.shutdown(wait=True)

    assert lease_released.wait(timeout=2.0)
    assert completed.is_set()
    assert admission.acquire(timeout=2.0)
    admission.release()
    completion_lease.release_if_caller_owned()
    completion_lease.release_from_worker()
    assert release_count == 1


def test_fresh_completion_lease_releases_when_worker_start_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_controlled_completion(monkeypatch)
    connection = _CleanupFailureConnection("api-slb.micuapi.ai", timeout=10.0)
    monkeypatch.setattr(
        baseline.http.client,
        "HTTPSConnection",
        lambda _host, *, timeout: connection,
    )

    class FailingStartThread:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def start(self) -> None:
            raise RuntimeError("thread unavailable")

    monkeypatch.setattr(baseline.threading, "Thread", FailingStartThread)
    release_count = 0

    def release() -> None:
        nonlocal release_count
        release_count += 1

    completion_lease = baseline.RequestCompletionLease(release)

    with pytest.raises(RuntimeError, match="thread unavailable"):
        baseline.call_fresh_chat_completion(
            system_prompt="system",
            user_prompt="user",
            model="deepseek-v4-flash",
            api_key="secret",
            base_url="https://api-slb.micuapi.ai/v1",
            timeout_seconds=10.0,
            completion_lease=completion_lease,
        )

    completion_lease.release_if_caller_owned()
    completion_lease.release_from_worker()
    assert release_count == 1
    assert connection.closed is True
    assert connection.close_count == 1


def test_fresh_cleanup_exceptions_preserve_result_and_complete_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed = _install_controlled_completion(monkeypatch)
    worker_exited = threading.Event()
    real_thread = threading.Thread

    class TrackingThread:
        def __init__(
            self,
            *,
            target: object,
            name: str,
            daemon: bool,
        ) -> None:
            assert callable(target)

            def tracked_target() -> None:
                try:
                    target()
                finally:
                    worker_exited.set()

            self._thread = real_thread(
                target=tracked_target,
                name=name,
                daemon=daemon,
            )

        def start(self) -> None:
            self._thread.start()

    monkeypatch.setattr(baseline.threading, "Thread", TrackingThread)
    connection = _CleanupFailureConnection("api-slb.micuapi.ai", timeout=10.0)
    monkeypatch.setattr(
        baseline.http.client,
        "HTTPSConnection",
        lambda _host, *, timeout: connection,
    )
    release_count = 0

    def release() -> None:
        nonlocal release_count
        release_count += 1

    completion_lease = baseline.RequestCompletionLease(release)
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(
            baseline.call_fresh_chat_completion,
            system_prompt="system",
            user_prompt="user",
            model="deepseek-v4-flash",
            api_key="secret",
            base_url="https://api-slb.micuapi.ai/v1",
            timeout_seconds=10.0,
            json_mode=True,
            completion_lease=completion_lease,
        )
        assert connection.response.read_completed.wait(timeout=5.0)
        assert worker_exited.wait(timeout=5.0)
        completed.trigger_timeout()
        assert future.result(timeout=5.0) == '{"answer":"ok"}'
    finally:
        executor.shutdown(wait=True)

    assert completed.is_set()
    assert release_count == 1
    assert connection.response.close_count >= 1
    assert connection.close_count >= 1


def test_fresh_timeout_cleanup_exception_preserves_timeout_and_releases_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed = _install_controlled_completion(monkeypatch)
    connection = _BlockedRequestCleanupFailureConnection(
        "api-slb.micuapi.ai", timeout=10.0
    )
    monkeypatch.setattr(
        baseline.http.client,
        "HTTPSConnection",
        lambda _host, *, timeout: connection,
    )
    release_count = 0
    lease_released = threading.Event()

    def release() -> None:
        nonlocal release_count
        release_count += 1
        lease_released.set()

    completion_lease = baseline.RequestCompletionLease(release)
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(
            baseline.call_fresh_chat_completion,
            system_prompt="system",
            user_prompt="user",
            model="deepseek-v4-flash",
            api_key="secret",
            base_url="https://api-slb.micuapi.ai/v1",
            timeout_seconds=10.0,
            completion_lease=completion_lease,
        )
        assert connection.request_started.wait(timeout=2.0)
        assert completed.wait_entered.wait(timeout=2.0)
        completed.trigger_timeout()
        with pytest.raises(TimeoutError, match="absolute request deadline"):
            future.result(timeout=2.0)
    finally:
        connection.release_request.set()
        assert connection.second_close.wait(timeout=2.0)
        executor.shutdown(wait=True)

    assert completed.is_set()
    assert lease_released.wait(timeout=2.0)
    assert release_count == 1


def test_pooled_chat_client_bounds_concurrent_provider_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    connections: list[_ReusableHTTPConnection] = []

    def connection_factory(host: str, *, timeout: float) -> _ReusableHTTPConnection:
        connection = _ReusableHTTPConnection(
            host,
            timeout=timeout,
            entered=entered,
            release=release,
        )
        connections.append(connection)
        return connection

    monkeypatch.setattr(baseline.http.client, "HTTPSConnection", connection_factory)
    client = baseline.PooledChatCompletionClient(
        base_url="https://api-slb.micuapi.ai/v1",
        max_connections=1,
    )
    kwargs = {
        "system_prompt": "system",
        "user_prompt": "user",
        "model": "deepseek-v4-flash",
        "api_key": "secret",
        "wire_api": "chat_completions",
    }

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(client.call_model, **kwargs)
        assert entered.wait(timeout=2)
        second = executor.submit(client.call_model, **kwargs)
        assert len(connections[0].requests) == 1
        release.set()
        assert first.result(timeout=2) == '{"answer":"ok"}'
        assert second.result(timeout=2) == '{"answer":"ok"}'

    assert len(connections) == 1
    assert len(connections[0].requests) == 2
    client.close()


def _stream_event(
    *,
    content: str | None = None,
    reasoning_content: str | None = None,
    finish_reason: str | None = None,
    usage: dict[str, int] | None = None,
) -> bytes:
    choices = []
    if (
        content is not None
        or reasoning_content is not None
        or finish_reason is not None
    ):
        choices = [
            {
                "index": 0,
                "delta": {
                    "content": content,
                    "reasoning_content": reasoning_content,
                    "role": "assistant",
                },
                "finish_reason": finish_reason,
                "logprobs": None,
            }
        ]
    payload = {
        "id": "response-1",
        "choices": choices,
        "created": 1,
        "model": "deepseek-v4-flash",
        "object": "chat.completion.chunk",
        "system_fingerprint": "test-fingerprint",
        "usage": usage,
    }
    return f"data: {json.dumps(payload)}\n\n".encode()


def _complete_stream() -> tuple[bytes, ...]:
    return (
        _stream_event(reasoning_content="hidden"),
        _stream_event(reasoning_content=" chain"),
        _stream_event(content='{"answer":'),
        _stream_event(content='"ok"}', finish_reason="stop"),
        _stream_event(
            usage={
                "prompt_tokens": 101,
                "completion_tokens": 23,
                "total_tokens": 124,
                "prompt_cache_hit_tokens": 64,
                "prompt_cache_miss_tokens": 37,
            }
        ),
        b"data: [DONE]\n\n",
    )


def test_pooled_stream_closes_connection_without_draining_before_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connections: list[_CloseRequiredStreamConnection] = []

    def connection_factory(
        host: str, *, timeout: float
    ) -> _CloseRequiredStreamConnection:
        connection = _CloseRequiredStreamConnection(host, timeout=timeout)
        connections.append(connection)
        return connection

    monkeypatch.setattr(baseline.http.client, "HTTPSConnection", connection_factory)
    client = baseline.PooledChatCompletionClient(
        base_url="https://api-slb.micuapi.ai/v1",
        max_connections=1,
    )
    kwargs = {
        "system_prompt": "system",
        "user_prompt": "user",
        "model": "deepseek-v4-flash",
        "api_key": "secret",
        "wire_api": "chat_completions",
        "thinking_enabled": True,
    }

    try:
        first = client.call_model(**kwargs)
        second = client.call_model(**kwargs)

        assert first == second == '{"answer":"ok"}'
        assert len(connections) == 1
        assert len(connections[0].requests) == 2
        assert connections[0].close_count == 2
    finally:
        client.close()


def test_pooled_stream_attempt_telemetry_separates_parse_and_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        baseline.http.client,
        "HTTPSConnection",
        _CloseRequiredStreamConnection,
    )
    client = baseline.PooledChatCompletionClient(
        base_url="https://api-slb.micuapi.ai/v1",
        max_connections=1,
    )
    events: list[dict[str, object]] = []

    try:
        baseline.call_model_with_retries(
            max_retries=0,
            retry_delay_seconds=0,
            attempt_observer=events.append,
            model_call=client.call_model,
            system_prompt="system",
            user_prompt="user",
            model="deepseek-v4-flash",
            api_key="secret",
            wire_api="chat_completions",
            thinking_enabled=True,
        )
    finally:
        client.close()

    assert len(events) == 1
    assert events[0]["status"] == "success"
    assert events[0]["stream_parse_seconds"] >= 0
    assert events[0]["post_stream_cleanup_seconds"] >= 0
    assert (
        events[0]["stream_parse_seconds"] + events[0]["post_stream_cleanup_seconds"]
        <= events[0]["latency_seconds"]
    )


def test_thinking_chat_completion_streams_content_and_usage(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return _FakeHTTPResponse(lines=_complete_stream())

    monkeypatch.setattr(baseline.urllib.request, "urlopen", fake_urlopen)

    result = baseline.call_model(
        system_prompt="system",
        user_prompt="user",
        model="deepseek-v4-flash",
        api_key="key",
        base_url="https://example.test",
        wire_api="chat_completions",
        max_tokens=32768,
        json_mode=True,
        thinking_enabled=True,
    )

    assert captured["payload"]["stream"] is True
    assert captured["payload"]["stream_options"] == {"include_usage": True}
    assert result == '{"answer":"ok"}'
    assert result.prompt_tokens == 101
    assert result.completion_tokens == 23
    assert result.reasoning_characters == 12
    assert "hidden" not in result


def test_chat_completion_sends_provider_user_agent_to_v1_endpoint(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["authorization"] = request.get_header("Authorization")
        captured["user_agent"] = request.get_header("User-agent")
        captured["timeout"] = timeout
        return _FakeHTTPResponse(
            body=json.dumps(
                {
                    "choices": [{"message": {"content": '{"answer":"ok"}'}}],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 2},
                }
            ).encode()
        )

    monkeypatch.setattr(baseline.urllib.request, "urlopen", fake_urlopen)

    result = baseline.call_model(
        system_prompt="system",
        user_prompt="user",
        model="deepseek-v4-flash",
        api_key="test-only-key",
        base_url="https://www.micuapi.ai/v1",
        wire_api="chat_completions",
        json_mode=True,
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:149.0) "
            "Gecko/20100101 Firefox/149.0"
        ),
    )

    assert result == '{"answer":"ok"}'
    assert captured == {
        "url": "https://www.micuapi.ai/v1/chat/completions",
        "authorization": "Bearer test-only-key",
        "user_agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:149.0) "
            "Gecko/20100101 Firefox/149.0"
        ),
        "timeout": 120,
    }


def test_thinking_stream_joins_data_lines_at_sse_event_boundary(
    monkeypatch,
) -> None:
    split_content_event = (
        b'data: {"choices": [\n'
        b'data: {"index": 0, "delta": {"content": "joined"}, '
        b'"finish_reason": "stop"}], "usage": null}\n\n'
    )
    lines = (
        split_content_event,
        _stream_event(
            usage={
                "prompt_tokens": 7,
                "completion_tokens": 2,
                "total_tokens": 9,
            }
        ),
        b"data: [DONE]\n\n",
    )
    monkeypatch.setattr(
        baseline.urllib.request,
        "urlopen",
        lambda request, timeout: _FakeHTTPResponse(lines=lines),
    )

    result = baseline.call_model(
        system_prompt="system",
        user_prompt="user",
        model="deepseek-v4-flash",
        api_key="key",
        base_url="https://example.test",
        wire_api="chat_completions",
        json_mode=True,
        thinking_enabled=True,
    )

    assert result == "joined"


@pytest.mark.parametrize("missing", ("done", "finish_reason"))
def test_thinking_stream_requires_finish_reason_and_done(monkeypatch, missing) -> None:
    lines = list(_complete_stream())
    if missing == "done":
        lines.pop()
    else:
        lines[3] = _stream_event(content='"ok"}')
    monkeypatch.setattr(
        baseline.urllib.request,
        "urlopen",
        lambda request, timeout: _FakeHTTPResponse(lines=tuple(lines)),
    )

    with pytest.raises(ConnectionError) as captured:
        baseline.call_model(
            system_prompt="system",
            user_prompt="user",
            model="deepseek-v4-flash",
            api_key="key",
            base_url="https://example.test",
            wire_api="chat_completions",
            json_mode=True,
            thinking_enabled=True,
        )

    assert baseline._is_retryable_network_error(captured.value)


def test_thinking_stream_rejects_nonstop_finish_reason(monkeypatch) -> None:
    lines = list(_complete_stream())
    lines[3] = _stream_event(content='"ok"}', finish_reason="length")
    monkeypatch.setattr(
        baseline.urllib.request,
        "urlopen",
        lambda request, timeout: _FakeHTTPResponse(lines=tuple(lines)),
    )

    with pytest.raises(ConnectionError, match="finish_reason") as captured:
        baseline.call_model(
            system_prompt="system",
            user_prompt="user",
            model="deepseek-v4-flash",
            api_key="key",
            base_url="https://example.test",
            wire_api="chat_completions",
            json_mode=True,
            thinking_enabled=True,
        )

    assert baseline._is_retryable_network_error(captured.value)


@pytest.mark.parametrize(
    "choices",
    (
        [
            {
                "index": 0,
                "delta": {"content": "zero"},
                "finish_reason": None,
            },
            {
                "index": 1,
                "delta": {"content": "one"},
                "finish_reason": "stop",
            },
        ],
        [
            {
                "index": 1,
                "delta": {"content": "wrong-choice"},
                "finish_reason": "stop",
            }
        ],
    ),
    ids=("multiple-choices", "nonzero-choice"),
)
def test_thinking_stream_accepts_only_one_choice_at_index_zero(
    monkeypatch,
    choices,
) -> None:
    content_event = (
        "data: "
        + json.dumps(
            {
                "id": "response-1",
                "choices": choices,
                "usage": None,
            }
        )
        + "\n\n"
    ).encode()
    lines = (
        content_event,
        _stream_event(
            usage={
                "prompt_tokens": 7,
                "completion_tokens": 2,
                "total_tokens": 9,
            }
        ),
        b"data: [DONE]\n\n",
    )
    monkeypatch.setattr(
        baseline.urllib.request,
        "urlopen",
        lambda request, timeout: _FakeHTTPResponse(lines=lines),
    )

    with pytest.raises(ConnectionError, match="choice") as captured:
        baseline.call_model(
            system_prompt="system",
            user_prompt="user",
            model="deepseek-v4-flash",
            api_key="key",
            base_url="https://example.test",
            wire_api="chat_completions",
            json_mode=True,
            thinking_enabled=True,
        )

    assert baseline._is_retryable_network_error(captured.value)


def test_malformed_thinking_stream_is_retried_then_succeeds(monkeypatch) -> None:
    responses = iter(
        (
            _FakeHTTPResponse(lines=(b"data: {malformed}\n\n",)),
            _FakeHTTPResponse(lines=_complete_stream()),
        )
    )
    calls = 0

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        return next(responses)

    monkeypatch.setattr(baseline.urllib.request, "urlopen", fake_urlopen)

    result, attempts = baseline.call_model_with_retries(
        max_retries=1,
        retry_delay_seconds=0,
        system_prompt="system",
        user_prompt="user",
        model="deepseek-v4-flash",
        api_key="key",
        base_url="https://example.test",
        wire_api="chat_completions",
        json_mode=True,
        thinking_enabled=True,
    )

    assert result == '{"answer":"ok"}'
    assert attempts == 2
    assert calls == 2


def test_retry_attempt_observer_reports_recovered_failures_without_error_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses: list[object] = [
        http.client.RemoteDisconnected("SECRET_PROVIDER_BODY"),
        baseline.ModelResponseText(
            '{"answer":"ok"}',
            prompt_tokens=11,
            completion_tokens=7,
        ),
    ]
    events: list[dict[str, object]] = []

    def fake_call_model(**_kwargs: object) -> str:
        response = responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        assert isinstance(response, str)
        return response

    monkeypatch.setattr(baseline, "call_model", fake_call_model)

    result, attempts = baseline.call_model_with_retries(
        max_retries=1,
        retry_delay_seconds=0,
        attempt_observer=events.append,
    )

    assert result == '{"answer":"ok"}'
    assert attempts == 2
    assert [event["status"] for event in events] == [
        "retryable_error",
        "success",
    ]
    assert events[0]["cause_type"] == "remote_disconnected"
    assert events[0]["retryable"] is True
    assert events[0]["attempt"] == 1
    assert events[1]["attempt"] == 2
    assert all(float(event["latency_seconds"]) >= 0 for event in events)
    assert "SECRET_PROVIDER_BODY" not in json.dumps(events)


@pytest.mark.parametrize(
    "connection_state_error",
    (
        http.client.CannotSendRequest("stale pooled connection"),
        http.client.ResponseNotReady("prior response still active"),
    ),
    ids=("cannot-send-request", "response-not-ready"),
)
def test_retry_recovers_from_stale_pooled_connection_state(
    connection_state_error: BaseException,
) -> None:
    responses: list[object] = [
        connection_state_error,
        baseline.ModelResponseText(
            '{"answer":"ok"}',
            prompt_tokens=11,
            completion_tokens=7,
        ),
    ]
    events: list[dict[str, object]] = []

    def fake_call_model(**_kwargs: object) -> str:
        response = responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        assert isinstance(response, str)
        return response

    result, attempts = baseline.call_model_with_retries(
        max_retries=1,
        retry_delay_seconds=0,
        attempt_observer=events.append,
        model_call=fake_call_model,
    )

    assert result == '{"answer":"ok"}'
    assert attempts == 2
    assert [event["status"] for event in events] == [
        "retryable_error",
        "success",
    ]
    assert events[0]["cause_type"] == "connection_error"


@pytest.mark.parametrize(
    "malformed_event",
    (
        b'data: {"reasoning_content":"SECRET_REASONING"\n\n',
        b'data: {"reasoning_content":"SECRET_REASONING\xff"}\n\n',
    ),
    ids=("invalid-json", "invalid-utf8"),
)
def test_thinking_stream_parse_errors_do_not_chain_raw_payload(
    monkeypatch,
    malformed_event,
) -> None:
    monkeypatch.setattr(
        baseline.urllib.request,
        "urlopen",
        lambda request, timeout: _FakeHTTPResponse(lines=(malformed_event,)),
    )

    with pytest.raises(ConnectionError) as captured:
        baseline.call_model(
            system_prompt="system",
            user_prompt="user",
            model="deepseek-v4-flash",
            api_key="key",
            base_url="https://example.test",
            wire_api="chat_completions",
            json_mode=True,
            thinking_enabled=True,
        )

    assert captured.value.__cause__ is None
    assert "SECRET_REASONING" not in str(captured.value)


def test_nonthinking_chat_completion_remains_nonstreaming(monkeypatch) -> None:
    captured: dict[str, object] = {}
    body = json.dumps(
        {
            "id": "response-1",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": '{"ok":true}'},
                }
            ],
            "created": 1,
            "model": "deepseek-v4-flash",
            "object": "chat.completion",
            "system_fingerprint": "test-fingerprint",
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 4,
                "total_tokens": 15,
                "prompt_cache_hit_tokens": 0,
                "prompt_cache_miss_tokens": 11,
            },
        }
    ).encode()

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data)
        return _FakeHTTPResponse(body=body)

    monkeypatch.setattr(baseline.urllib.request, "urlopen", fake_urlopen)

    result = baseline.call_model(
        system_prompt="system",
        user_prompt="user",
        model="deepseek-v4-flash",
        api_key="key",
        base_url="https://example.test",
        wire_api="chat_completions",
        json_mode=True,
        thinking_enabled=False,
    )

    assert "stream" not in captured["payload"]
    assert result == '{"ok":true}'
    assert result.prompt_tokens == 11
    assert result.completion_tokens == 4


def _prompt(record_id: str) -> dict[str, str]:
    return {
        "record_id": record_id,
        "paper_id": "paper",
        "issue_url": f"https://example.test/{record_id}",
        "system_prompt": "system",
        "user_prompt": "user",
    }


def _example(record_id: str) -> dict[str, str]:
    return {
        "record_id": record_id,
        "symptom": "Crash",
        "root_cause": "Incorrect Code Logic",
    }


def test_call_model_forwards_role_token_budget(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_chat(*args, **kwargs):
        captured.update(kwargs)
        return "{}"

    monkeypatch.setattr(baseline, "call_chat_completion", fake_chat)

    baseline.call_model(
        system_prompt="system",
        user_prompt="user",
        model="model",
        api_key="key",
        base_url="https://example.test",
        wire_api="chat_completions",
        max_tokens=700,
    )

    assert captured["max_tokens"] == 700


def test_call_model_forwards_structured_json_mode(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_chat(*args, **kwargs):
        captured.update(kwargs)
        return "{}"

    monkeypatch.setattr(baseline, "call_chat_completion", fake_chat)

    baseline.call_model(
        system_prompt="system",
        user_prompt="user",
        model="model",
        api_key="key",
        base_url="https://example.test",
        wire_api="chat_completions",
        json_mode=True,
        thinking_enabled=False,
    )

    assert captured["json_mode"] is True
    assert captured["thinking_enabled"] is False


def _run(tmp_path, monkeypatch, prompts, examples, fake_call, **kwargs):
    prompts_path = tmp_path / "prompts.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    metrics_path = tmp_path / "metrics.json"
    prompts_path.write_text(
        "\n".join(json.dumps(prompt) for prompt in prompts) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(baseline, "call_model", fake_call)
    metrics = baseline.run_llm_prompts(
        prompts_path=prompts_path,
        examples=examples,
        taxonomy=TAXONOMY,
        predictions_path=predictions_path,
        metrics_path=metrics_path,
        model="test-model",
        api_key="key",
        base_url="https://example.test/v1",
        retry_delay_seconds=0,
        **kwargs,
    )
    rows = [json.loads(line) for line in predictions_path.read_text().splitlines()]
    return metrics, rows, predictions_path


def test_retries_remote_disconnect_then_succeeds(tmp_path, monkeypatch) -> None:
    attempts = 0

    def fake_call(**_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise http.client.RemoteDisconnected("connection closed")
        return '{"symptom":"Crash","root_cause":"Incorrect Code Logic"}'

    metrics, rows, _ = _run(
        tmp_path,
        monkeypatch,
        [_prompt("r1")],
        [_example("r1")],
        fake_call,
        max_retries=2,
    )

    assert attempts == 2
    assert metrics["valid_count"] == 1
    assert rows[0]["attempts"] == 2


def test_resume_skips_existing_prediction(tmp_path, monkeypatch) -> None:
    prompts = [_prompt("r1"), _prompt("r2")]
    calls = []

    def fake_call(**kwargs):
        calls.append(kwargs)
        return '{"symptom":"Crash","root_cause":"Incorrect Code Logic"}'

    prompts_path = tmp_path / "prompts.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    metrics_path = tmp_path / "metrics.json"
    prompts_path.write_text(
        "\n".join(json.dumps(prompt) for prompt in prompts) + "\n",
        encoding="utf-8",
    )
    predictions_path.write_text(
        json.dumps(
            {
                "record_id": "r1",
                "paper_id": "paper",
                "issue_url": "https://example.test/r1",
                "model": "test-model",
                "raw_output": "{}",
                "latency_seconds": 1.0,
                "symptom": "Crash",
                "root_cause": "Incorrect Code Logic",
                "invalid": False,
                "error": "",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(baseline, "call_model", fake_call)

    metrics = baseline.run_llm_prompts(
        prompts_path=prompts_path,
        examples=[_example("r1"), _example("r2")],
        taxonomy=TAXONOMY,
        predictions_path=predictions_path,
        metrics_path=metrics_path,
        model="test-model",
        api_key="key",
        base_url="https://example.test/v1",
        resume=True,
        max_retries=0,
        retry_delay_seconds=0,
    )
    rows = [json.loads(line) for line in predictions_path.read_text().splitlines()]

    assert len(calls) == 1
    assert [row["record_id"] for row in rows] == ["r1", "r2"]
    assert metrics["n"] == 2
    assert metrics["valid_count"] == 2


def test_resume_retries_previous_network_failure(tmp_path, monkeypatch) -> None:
    prompts = [_prompt("r1")]

    def disconnected(**_kwargs):
        raise http.client.RemoteDisconnected("connection closed")

    first_metrics, first_rows, predictions_path = _run(
        tmp_path,
        monkeypatch,
        prompts,
        [_example("r1")],
        disconnected,
        max_retries=0,
    )

    assert first_metrics["invalid_count"] == 1
    assert first_rows[0]["request_failed"] is True

    calls = 0

    def succeeds(**_kwargs):
        nonlocal calls
        calls += 1
        return '{"symptom":"Crash","root_cause":"Incorrect Code Logic"}'

    monkeypatch.setattr(baseline, "call_model", succeeds)
    metrics = baseline.run_llm_prompts(
        prompts_path=tmp_path / "prompts.jsonl",
        examples=[_example("r1")],
        taxonomy=TAXONOMY,
        predictions_path=predictions_path,
        metrics_path=tmp_path / "metrics.json",
        model="test-model",
        api_key="key",
        base_url="https://example.test/v1",
        resume=True,
        max_retries=0,
        retry_delay_seconds=0,
    )

    assert calls == 1
    assert metrics["valid_count"] == 1
