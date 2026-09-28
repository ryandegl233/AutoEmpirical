import importlib
import io
import json
import sys
import urllib.error
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))


def module():
    return importlib.import_module('run_experiment_two_fluxion')


def test_fluxion_credentials_are_isolated():
    m = module()
    with pytest.raises(ValueError, match='FLUXION_API_KEY'):
        m.resolve_config({'DRAGONAPI_KEY': 'dragon', 'GOOGLE_API_KEY': 'google'})
    assert m.resolve_config({'FLUXION_API_KEY': 'fluxion', 'DRAGONAPI_KEY': 'dragon'}) == {
        'base_url': 'https://fluxionai.space/v1', 'api_key': 'fluxion'}
    with pytest.raises(ValueError):
        m.resolve_config({'FLUXION_API_KEY': 'fluxion', 'FLUXION_BASE_URL': 'https://elsewhere.test/v1'})


def test_runtime_records_fluxion_and_restores_official_guard():
    m = module()
    import run_experiment_two_dragon as legacy
    from Benchmark.src import llm_provider_config as provider
    from Benchmark.src.adaptive_empirical_workflow import experiment_manifest as em
    before = legacy.transport_manifest()
    with m.fluxion_configuration():
        with legacy.dragon_runtime({'transport': m.transport_manifest()}):
            assert provider.resolve_gemini_config({'FLUXION_API_KEY': 'fluxion'})['api_key'] == 'fluxion'
            assert provider.normalize_gemini_base_url(m.BASE_URL) == m.BASE_URL
            cfg = {'architecture': 'adaptive_empirical_expert_workflow', 'backend': m.BASE_URL}
            em.config_hash(cfg)
            assert cfg['transport']['provider'] == 'fluxion'
            assert cfg['transport']['credential_variable'] == 'FLUXION_API_KEY'
    assert legacy.transport_manifest() == before
    with pytest.raises(ValueError):
        provider.normalize_gemini_base_url(m.BASE_URL)


def test_guard_preserves_fluxion_error_origin_and_stops_network(tmp_path, monkeypatch):
    m = module()
    from Benchmark.src.ase2022_llm_baseline import PooledChatCompletionClient
    calls = []
    def failing(self, **kwargs):
        calls.append(1)
        raise urllib.error.HTTPError(m.BASE_URL + '/chat/completions', 500, 'failed', {},
                                     io.BytesIO(b'{"error":{"code":"get_channel_failed"}}'))
    monkeypatch.setattr(PooledChatCompletionClient, 'call_model', failing)
    with m.fluxion_guarded_client(tmp_path) as guard:
        for _ in range(3):
            with pytest.raises(urllib.error.HTTPError) as error:
                PooledChatCompletionClient.call_model(None)
            assert error.value.url == 'https://fluxionai.space/v1/chat/completions'
        assert guard.blocked
    assert len(calls) == 1


def test_preflight_rejects_missing_model_without_substitution(tmp_path, monkeypatch):
    m = module()
    class Reply(io.BytesIO):
        status = 200
    class Opener:
        def open(self, request, timeout):
            assert request.full_url == 'https://fluxionai.space/v1/models'
            return Reply(b'{"data":[{"id":"another-model"}]}')
    monkeypatch.setattr(m.urllib.request, 'build_opener', lambda *args: Opener())
    with pytest.raises(ValueError, match='gemini-3.7-flash'):
        m.preflight({'FLUXION_API_KEY': 'fake'}, 'gemini-3.7-flash', tmp_path)
    rows=[json.loads(p.read_text()) for p in tmp_path.glob('*.json')]
    assert rows[0]['model_available'] is False
