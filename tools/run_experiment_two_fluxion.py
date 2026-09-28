"""Fluxion continuation using frozen workflow code and isolated provider credentials."""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import run_experiment_two_dragon as legacy
from dragon_transport_guard import guarded_client

ROOT = legacy.ROOT
BASE_URL = 'https://fluxionai.space/v1'


def normalize_url(value):
    if value.strip().rstrip('/') != BASE_URL:
        raise ValueError('Fluxion requires https://fluxionai.space/v1')
    return BASE_URL


def resolve_config(env, *, base_url_override=None):
    key = (env.get('FLUXION_API_KEY') or '').strip()
    if not key:
        raise ValueError('Missing FLUXION_API_KEY in worktree .env')
    return dict(base_url=normalize_url(base_url_override or env.get('FLUXION_BASE_URL') or BASE_URL), api_key=key)


def transport_manifest():
    return dict(provider='fluxion', base_url=BASE_URL, wire_api='chat_completions',
        response_format={'type': 'json_object'}, credential_variable='FLUXION_API_KEY',
        model_identity_verified=False, compatibility_revision='fluxion-safe-v1',
        json_handling='unwrap one complete unambiguous JSON object; unchanged schema validation',
        circuit_breaker='immediate channel/auth/billing/parameter errors; three consecutive transient failures',
        note='Requested model name is preserved; relay upstream equivalence is not established.')


@contextmanager
def fluxion_configuration():
    overrides = dict(BASE_URL=BASE_URL, resolve_config=resolve_config,
                     normalize_url=normalize_url, transport_manifest=transport_manifest)
    originals = {k: getattr(legacy, k) for k in overrides}
    for key, value in overrides.items():
        setattr(legacy, key, value)
    try:
        yield
    finally:
        for key, value in originals.items():
            setattr(legacy, key, value)


@contextmanager
def fluxion_guarded_client(directory):
    from Benchmark.src.ase2022_llm_baseline import PooledChatCompletionClient
    with guarded_client(directory) as guard:
        guarded_call = PooledChatCompletionClient.call_model

        def routed_call(self, **kwargs):
            try:
                return guarded_call(self, **kwargs)
            except urllib.error.HTTPError as error:
                # The frozen guard uses a Dragon URL for synthetic errors.
                # Actual connections use this client's Fluxion base URL.
                error.url = BASE_URL + '/chat/completions'
                raise

        PooledChatCompletionClient.call_model = routed_call
        try:
            yield guard
        finally:
            PooledChatCompletionClient.call_model = guarded_call


def preflight(env, model, directory):
    config = resolve_config(env)
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None
    report = dict(created_at=datetime.now(timezone.utc).isoformat(), provider='fluxion',
                  base_url=BASE_URL, requested_model=model)
    try:
        request = urllib.request.Request(BASE_URL + '/models', headers={'Authorization': 'Bearer ' + config['api_key']})
        with urllib.request.build_opener(NoRedirect).open(request, timeout=30) as response:
            report['http_status'] = response.status
            data = json.load(response)
        report['available_models'] = [r['id'] for r in data.get('data', [])]
        report['model_available'] = model in report['available_models']
    except urllib.error.HTTPError as error:
        report.update(http_status=error.code, error_body=error.read(6000).decode('utf-8', 'replace').replace(config['api_key'], '[REDACTED]'))
    except (OSError, ValueError) as error:
        report['error_type'] = type(error).__name__
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (uuid.uuid4().hex + '-fluxion-preflight.json')
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    if not report.get('model_available'):
        raise ValueError(f'Fluxion preflight failed for {model}; HTTP {report.get("http_status")}. See {path}')
    print(f'Fluxion preflight OK: {model}.', flush=True)
    return report


def main(argv=None):
    from Benchmark.scripts.run_ase2022_llm_baseline import _load_env_file
    _load_env_file(ROOT / '.env')
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--batch-id', default='exp2-night-v3-fluxion')
    parser.add_argument('--source-batch', default='exp2-night-v3-dragon-safe')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--score-only', action='store_true')
    args, _ = parser.parse_known_args(argv)
    supplied = list(sys.argv[1:] if argv is None else argv)
    import re
    if any(not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', name) for name in [args.batch_id, args.source_batch]):
        raise ValueError('Invalid batch name')
    if not any(x == '--batch-id' or x.startswith('--batch-id=') for x in supplied):
        supplied += ['--batch-id', args.batch_id]
    if not any(x == '--source-batch' or x.startswith('--source-batch=') for x in supplied):
        supplied += ['--source-batch', args.source_batch]
    if args.run and not args.score_only:
        base = ROOT / 'reports/experiment_two' / args.batch_id
        source = ROOT / 'reports/experiment_two' / args.source_batch
        protocol = legacy.runner.read((base if base.exists() else source) / 'protocol.json')
        legacy.runner.verify_frozen(protocol)
        preflight(os.environ, protocol['model'], ROOT / 'reports/gemini_diagnostics')
    original_prepare = legacy.prepare_continuation
    original_worker = legacy.runner.run_worker

    def prepare(source, target):
        original_prepare(source, target)
        path = Path(target) / 'protocol.json'
        protocol = legacy.runner.read(path)
        for name in ['run_experiment_two_fluxion.py', 'experiment_two_fluxion_worker.py', 'dragon_transport_guard.py']:
            code = Path(__file__).with_name(name).resolve()
            protocol['file_hashes'][str(code)] = legacy.runner.digest(code)
        path.write_text(json.dumps(protocol, indent=2), encoding='utf-8')

    def worker(cmd, env, log, cancel):
        cmd = list(cmd)
        cmd[2] = str(Path(__file__).with_name('experiment_two_fluxion_worker.py'))
        return original_worker(cmd, env, log, cancel)

    with fluxion_configuration():
        legacy.prepare_continuation = prepare
        legacy.runner.run_worker = worker
        try:
            return legacy.main(supplied)
        finally:
            legacy.prepare_continuation = original_prepare
            legacy.runner.run_worker = original_worker


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(str(error))
