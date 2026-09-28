import importlib
import io
import json
import sys
import urllib.error
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))


def module():
    return importlib.import_module('dragon_transport_guard')


def test_missing_channel_blocks_further_network_calls_and_redacts_key(tmp_path):
    m = module()
    guard = m.TransportGuard(tmp_path)
    calls = []

    def failing(**kwargs):
        calls.append(1)
        body = json.dumps({'error': {'code': 'get_channel_failed', 'message': 'no channel secret-test'}}).encode()
        raise urllib.error.HTTPError('https://newapi.dragon3api.com/v1/chat/completions', 500, 'error', {}, io.BytesIO(body))

    for _ in range(4):
        with pytest.raises(urllib.error.HTTPError):
            guard.call(failing, api_key='secret-test')
    assert len(calls) == 1
    assert guard.blocked
    logs = ''.join(p.read_text(encoding='utf-8') for p in tmp_path.glob('*.json'))
    assert 'get_channel_failed' in logs
    assert 'secret-test' not in logs


def test_transient_failures_trip_after_three_and_success_resets_count(tmp_path):
    m = module()
    guard = m.TransportGuard(tmp_path)

    def unavailable(**kwargs):
        raise urllib.error.HTTPError('url', 503, 'unavailable', {}, io.BytesIO(b'upstream busy'))

    for _ in range(2):
        with pytest.raises(urllib.error.HTTPError):
            guard.call(unavailable)
    assert not guard.blocked
    guard.call(lambda **kwargs: '{}')
    for _ in range(2):
        with pytest.raises(urllib.error.HTTPError):
            guard.call(unavailable)
    assert not guard.blocked
    with pytest.raises(urllib.error.HTTPError):
        guard.call(unavailable)
    assert guard.blocked


@pytest.mark.parametrize('raw,expected', [
    ('Explanation.\n```json\n{"a":1}\n```', '{"a":1}'),
    ('Explanation.\n{"a":{"b":2}}', '{"a":{"b":2}}'),
    ('{"a":1}', '{"a":1}'),
    ('{"a":1}\n{"a":2}', None),
    ('Explanation.\n{"a":1', None),
    ('{"a":1,"a":2}', None),
    ('{"a":NaN}', None),
    ('{"a":1} trailing explanation', None),
])
def test_only_unambiguous_complete_json_is_unwrapped(raw, expected):
    assert module().extract_json_object(raw) == expected


def test_normalization_retains_raw_output_and_token_metadata(tmp_path):
    from Benchmark.src.ase2022_llm_baseline import ModelResponseText
    guard = module().TransportGuard(tmp_path)
    response = ModelResponseText('Explanation.\n{"a":1}', prompt_tokens=30, completion_tokens=8,
                                 response_model='gemini-3.7-flash', finish_reason='stop')
    result = guard.call(lambda **kwargs: response)
    assert str(result) == '{"a":1}'
    assert result.prompt_tokens == 30 and result.completion_tokens == 8
    assert result.response_model == 'gemini-3.7-flash'
    saved = [json.loads(p.read_text()) for p in tmp_path.glob('*.json')]
    assert any(e.get('raw_response') == str(response) and e.get('normalized_response') == '{"a":1}' for e in saved)


def test_http_connection_error_keeps_body_for_diagnostics():
    m = module()

    class Response:
        status = 500
        reason = 'Internal Server Error'
        headers = {}

        def read(self, limit=None):
            return b'{"error":{"code":"get_channel_failed"}}'

    class Connection:
        def getresponse(self):
            return Response()

    with pytest.raises(urllib.error.HTTPError) as caught:
        m.ErrorBodyConnection(Connection()).getresponse()
    assert b'get_channel_failed' in caught.value.read()


def test_safe_worker_returns_nonzero_on_missing_channel_without_more_requests(tmp_path, monkeypatch):
    import http.client
    import runpy
    import experiment_two_dragon_safe_worker as safe
    from run_experiment_two_dragon_safe import safe_transport_manifest
    from Benchmark.src.ase2022_llm_baseline import PooledChatCompletionClient
    original_call = PooledChatCompletionClient.call_model
    network_calls = []

    class Response:
        status = 500
        reason = 'error'
        headers = {}

        def read(self, limit=None):
            return b'{"error":{"code":"get_channel_failed","message":"no channel"}}'

    class Connection:
        def __init__(self, *args, **kwargs):
            pass

        def request(self, method, path, body, headers):
            network_calls.append(json.loads(body))

        def getresponse(self):
            return Response()

        def close(self):
            pass

    monkeypatch.setattr(http.client, 'HTTPSConnection', Connection)
    (tmp_path / 'protocol.json').write_text(json.dumps(dict(file_hashes={}, model='gemini-3.7-flash',
                                                         transport=safe_transport_manifest())))
    monkeypatch.setenv('AE_EXPERIMENT_TWO_RESUME_PROTOCOL', str(tmp_path / 'protocol.json'))
    monkeypatch.setattr(sys, 'argv', ['worker', '--output-dir', str(tmp_path.parent), '--experiment-id', tmp_path.name,
        '--arm-id', 'E00', '--run-id', 'case01', '--model', 'gemini-3.7-flash', '--base-url', 'https://newapi.dragon3api.com/v1'])

    def fake_workflow(*args, **kwargs):
        client = PooledChatCompletionClient(base_url='https://newapi.dragon3api.com/v1', max_connections=1)
        try:
            for _ in range(4):
                with pytest.raises(urllib.error.HTTPError):
                    client.call_model(system_prompt='JSON', user_prompt='test', model='gemini-3.7-flash',
                        api_key='test-key', base_url='https://newapi.dragon3api.com/v1', wire_api='chat_completions')
        finally:
            client.close()
        raise SystemExit(0)

    monkeypatch.setattr(runpy, 'run_path', fake_workflow)
    assert safe.main() == 75
    assert len(network_calls) == 1
    assert network_calls[0]['response_format'] == {'type': 'json_object'}
    assert PooledChatCompletionClient.call_model is original_call
