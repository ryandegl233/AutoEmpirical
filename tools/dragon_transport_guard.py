"""Bounded provider diagnostics, circuit breaking and conservative JSON unwrapping."""
import http.client
import io
import json
import re
import threading
import urllib.error
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def extract_json_object(raw):
    """Accept one complete object only; do not repair, choose among, or fill JSON."""
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON field')
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError('Non-finite JSON value')

    text = str(raw).strip()
    start = re.search(r'[\[{]', text)
    if start is None:
        return None
    try:
        value, end = json.JSONDecoder(object_pairs_hook=unique_pairs,
            parse_constant=reject_constant).raw_decode(text, start.start())
    except ValueError:
        return None
    if not isinstance(value, dict):
        return None
    suffix = text[end:].strip()
    if suffix:
        prefix = text[:start.start()].rstrip().lower()
        if suffix != '```' or not (prefix.endswith('```json') or prefix.endswith('```')):
            return None
    return text[start.start():end]


class ErrorBodyConnection:
    """Capture error bodies before the frozen pooled client discards them."""
    def __init__(self, connection):
        self.connection = connection

    def __getattr__(self, name):
        return getattr(self.connection, name)

    def getresponse(self):
        response = self.connection.getresponse()
        if not 200 <= int(response.status) < 300:
            body = response.read(65536)
            raise urllib.error.HTTPError('https://newapi.dragon3api.com/v1/chat/completions',
                int(response.status), response.reason, response.headers, io.BytesIO(body))
        return response


class TransportGuard:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.blocked = False
        self.block_reason = None
        self.failures = 0

    def save(self, event):
        event['created_at'] = datetime.now(timezone.utc).isoformat()
        (self.directory / (uuid.uuid4().hex + '.json')).write_text(
            json.dumps(event, ensure_ascii=False, indent=2), encoding='utf-8')

    def fail(self, reason, *, immediate=False):
        with self.lock:
            self.failures += 1
            if immediate or self.failures >= 3:
                self.blocked = True
                self.block_reason = reason

    def call(self, transport, **kwargs):
        with self.lock:
            if self.blocked:
                # Frozen workflow may attempt schema/network fallback calls;
                # reject them locally without contacting the provider.
                raise urllib.error.HTTPError('https://newapi.dragon3api.com/v1/chat/completions',
                    503, 'Circuit open: ' + str(self.block_reason), {}, io.BytesIO(b''))
        try:
            response = transport(**kwargs)
        except urllib.error.HTTPError as error:
            body = error.read(65536).decode('utf-8', 'replace') if error.fp is not None else ''
            if kwargs.get('api_key'):
                body = body.replace(kwargs['api_key'], '[REDACTED]')
            body = re.sub(r'AIza[0-9A-Za-z_-]{20,}', '[REDACTED]', body)
            error.fp = io.BytesIO(body.encode('utf-8'))
            missing_channel = 'get_channel_failed' in body
            immediate = missing_channel or error.code in {400, 401, 402, 403, 404, 413, 429}
            if immediate or 500 <= error.code <= 599:
                self.fail('get_channel_failed' if missing_channel else 'HTTP ' + str(error.code), immediate=immediate)
            self.save(dict(kind='http_error', http_status=error.code, error_body=body,
                retry_after=error.headers.get('Retry-After') if error.headers else None,
                request_id=error.headers.get('X-Request-Id') if error.headers else None,
                circuit_open=self.blocked, circuit_reason=self.block_reason))
            raise
        except (OSError, http.client.HTTPException) as error:
            self.fail(type(error).__name__)
            self.save(dict(kind='network_error', error_type=type(error).__name__,
                           circuit_open=self.blocked, circuit_reason=self.block_reason))
            raise
        with self.lock:
            self.failures = 0
        normalized = extract_json_object(response)
        if normalized is not None and normalized != str(response):
            self.save(dict(kind='json_wrapper_removed', raw_response=str(response),
                           normalized_response=normalized))
            replacement = type(response)(normalized)
            if hasattr(response, '__dict__'):
                replacement.__dict__.update(response.__dict__)
            return replacement
        return response


@contextmanager
def guarded_client(directory):
    from Benchmark.src.ase2022_llm_baseline import PooledChatCompletionClient
    guard = TransportGuard(directory)
    original_init = PooledChatCompletionClient.__init__
    original_call = PooledChatCompletionClient.call_model

    def guarded_init(self, **kwargs):
        original_init(self, **kwargs)
        connections = [self._connections.get_nowait() for _ in range(self._connections.qsize())]
        for connection in connections:
            self._connections.put(ErrorBodyConnection(connection))

    def guarded_call(self, **kwargs):
        return guard.call(lambda **options: original_call(self, **options), **kwargs)

    PooledChatCompletionClient.__init__ = guarded_init
    PooledChatCompletionClient.call_model = guarded_call
    try:
        yield guard
    finally:
        PooledChatCompletionClient.__init__ = original_init
        PooledChatCompletionClient.call_model = original_call
