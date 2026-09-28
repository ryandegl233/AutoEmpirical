import importlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))


def module():
    return importlib.import_module('run_experiment_two_dragon')


def test_dragon_credentials_never_fall_back_to_google_or_another_proxy():
    m = module()
    with pytest.raises(ValueError, match='DRAGONAPI_KEY'):
        m.resolve_config({'GOOGLE_API_KEY': 'google-secret', 'SELF_API': 'other-secret'})
    cfg = m.resolve_config({'DRAGONAPI_KEY': 'dragon-secret', 'GOOGLE_API_KEY': 'google-secret'})
    assert cfg['api_key'] == 'dragon-secret'
    assert cfg['base_url'] == 'https://newapi.dragon3api.com/v1'
    for address in ['http://newapi.dragon3api.com/v1', 'https://evil.test/v1',
                    'https://newapi.dragon3api.com/v1?key=x']:
        with pytest.raises(ValueError):
            m.resolve_config({'DRAGONAPI_KEY': 'dragon-secret', 'DRAGONAPI_BASE_URL': address})


def source_batch(tmp_path):
    m = module()
    source = tmp_path / 'source'
    source.mkdir()
    code = tmp_path / 'frozen.py'
    code.write_text('frozen code')
    protocol = dict(cases=[1, 2], arms=['E00'], record_ids=['r1', 'r2'], model='gemini-3.7-flash',
                    configuration={'case_workers': 2}, file_hashes={str(code): m.runner.digest(code)})
    (source / 'protocol.json').write_text(json.dumps(protocol))
    for case, valid in [(1, True), (2, False)]:
        out = source / 'shards' / f'case{case:02}' / 'evaluation48/E00' / f'case{case:02}'
        out.mkdir(parents=True)
        (out / 'predictions_model.jsonl').write_text(json.dumps({'record_id': f'r{case}', 'stage3_valid': valid}))
        (out / 'run_manifest.json').write_text(json.dumps({'resolved_config': {'backend': 'google'}}))
    return source


def test_continuation_preserves_source_and_only_inherits_valid_outputs(tmp_path):
    m = module()
    source = source_batch(tmp_path)
    before = {str(p.relative_to(source)): p.read_bytes() for p in source.rglob('*') if p.is_file()}
    target = tmp_path / 'continued'
    m.prepare_continuation(source, target)
    after = {str(p.relative_to(source)): p.read_bytes() for p in source.rglob('*') if p.is_file() and p.name != 'runner.lock'}
    assert before == after
    copied = list(target.rglob('predictions_*.jsonl'))
    assert len(copied) == 1
    assert json.loads(copied[0].read_text())['record_id'] == 'r1'
    protocol = m.runner.read(target / 'protocol.json')
    assert protocol['transport']['provider'] == 'dragonapi'
    assert protocol['configuration']['case_workers'] == 1
    inherited = m.runner.read(target / 'inherited_results.json')['results']
    assert [(r['arm'], r['case']) for r in inherited] == [('E00', 1)]
    assert inherited[0]['prediction_sha256'] == m.runner.digest(copied[0])
    with pytest.raises(ValueError, match='exists'):
        m.prepare_continuation(source, target)


def test_relay_changes_manifest_backend_without_relaxing_official_path_globally(monkeypatch):
    m = module()
    from Benchmark.src import llm_provider_config as provider
    from Benchmark.src.adaptive_empirical_workflow import experiment_manifest as em
    monkeypatch.setenv('DRAGONAPI_KEY', 'dragon-only')
    original = provider.resolve_gemini_config
    protocol = {'transport': m.transport_manifest()}
    with m.dragon_runtime(protocol):
        cfg = provider.resolve_gemini_config({'DRAGONAPI_KEY': 'dragon-only'})
        assert cfg['api_key'] == 'dragon-only'
        assert provider.normalize_gemini_base_url(m.BASE_URL) == m.BASE_URL
        with pytest.raises(ValueError):
            provider.normalize_gemini_base_url('https://generativelanguage.googleapis.com/v1beta/openai')
        payload = {'architecture': 'adaptive_empirical_expert_workflow', 'backend': m.BASE_URL}
        em.config_hash(payload)
        assert payload['transport']['provider'] == 'dragonapi'
        assert payload['transport']['model_identity_verified'] is False
    assert provider.resolve_gemini_config is original
    with pytest.raises(ValueError):
        provider.normalize_gemini_base_url(m.BASE_URL)


def test_stopping_status_prevents_draining_remaining_cases():
    m = module()
    assert m.should_stop([429])
    assert m.should_stop([402])
    assert m.should_stop([401, 200])
    assert not m.should_stop([200])
    assert not m.should_stop([])
