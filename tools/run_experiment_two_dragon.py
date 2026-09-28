"""Continue experiment two through DragonAPI, retaining provenance of inherited results."""
import argparse
import json
import os
import re
import shutil
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Benchmark.scripts import run_experiment_two as runner

BASE_URL = 'https://newapi.dragon3api.com/v1'


def normalize_url(value):
    if value.strip().rstrip('/') != BASE_URL:
        raise ValueError('DragonAPI requires https://newapi.dragon3api.com/v1')
    return BASE_URL


def resolve_config(env, *, base_url_override=None):
    key = (env.get('DRAGONAPI_KEY') or '').strip()
    if not key:
        raise ValueError('Missing DRAGONAPI_KEY in worktree .env')
    base = normalize_url(base_url_override or env.get('DRAGONAPI_BASE_URL') or BASE_URL)
    return dict(base_url=base, api_key=key)


def transport_manifest():
    return dict(provider='dragonapi', base_url=BASE_URL, wire_api='chat_completions',
                response_format={'type': 'json_object'},
                credential_variable='DRAGONAPI_KEY', model_identity_verified=False,
                note='Requested model name is preserved; relay upstream equivalence is not established.')


def should_stop(statuses):
    return any(s in {401, 402, 403, 429} for s in statuses)


@contextmanager
def dragon_runtime(protocol):
    """Explicit, process-local route; never reuse Google credentials at a relay."""
    from Benchmark.src import llm_provider_config as provider
    from Benchmark.scripts import run_ase2022_llm_baseline as baseline
    from Benchmark.src.adaptive_empirical_workflow import experiment_manifest as em
    expected = transport_manifest()
    if protocol.get('transport') != expected:
        raise ValueError('DragonAPI transport differs from the continuation protocol')
    original_resolve = provider.resolve_gemini_config
    original_normalize = provider.normalize_gemini_base_url
    original_baseline = baseline.resolve_gemini_config
    original_hash = em.config_hash

    def annotated_hash(payload):
        if payload.get('architecture') == 'adaptive_empirical_expert_workflow':
            payload['transport'] = dict(expected)
        return original_hash(payload)

    provider.resolve_gemini_config = resolve_config
    provider.normalize_gemini_base_url = normalize_url
    baseline.resolve_gemini_config = resolve_config
    em.config_hash = annotated_hash
    try:
        yield
    finally:
        provider.resolve_gemini_config = original_resolve
        provider.normalize_gemini_base_url = original_normalize
        baseline.resolve_gemini_config = original_baseline
        em.config_hash = original_hash


def prepare_continuation(source, target):
    """Copy valid checkpoints only; never move or rewrite the source batch."""
    from run_experiment_two_sharded import completed_row
    source, target = Path(source).resolve(), Path(target).resolve()
    if target.exists():
        raise ValueError('Continuation directory already exists; use --resume')
    with runner.batch_lock(source):
        protocol = runner.read(source / 'protocol.json')
        runner.verify_frozen(protocol)
        inherited = []
        copies = []
        for arm in protocol['arms']:
            for case in protocol['cases']:
                row = completed_row(source, arm, case, protocol['record_ids'][case - 1])
                if row is None or row.get('stage3_valid') is not True:
                    continue
                paths = [*source.joinpath('evaluation48', arm, f'case{case:02}').glob('predictions_*.jsonl'),
                         *source.joinpath('shards', f'case{case:02}', 'evaluation48', arm, f'case{case:02}').glob('predictions_*.jsonl')]
                prediction = paths[0]
                manifest = prediction.parent / 'run_manifest.json'
                if not manifest.is_file():
                    raise ValueError('Valid source result lacks manifest: ' + str(prediction))
                relative = Path('shards') / f'case{case:02}' / 'evaluation48' / arm / f'case{case:02}'
                inherited.append(dict(arm=arm, case=case, source_prediction=str(prediction),
                    prediction_sha256=runner.digest(prediction), source_manifest=str(manifest),
                    source_manifest_sha256=runner.digest(manifest),
                    source_backend=runner.read(manifest).get('resolved_config', {}).get('backend'),
                    target_prediction=str(relative / prediction.name)))
                copies.append((prediction, manifest, relative))
        target.mkdir(parents=True)
        for prediction, manifest, relative in copies:
            out = target / relative
            out.mkdir(parents=True)
            shutil.copy2(prediction, out / prediction.name)
            shutil.copy2(manifest, out / manifest.name)
        protocol['configuration']['case_workers'] = 1
        protocol['transport'] = transport_manifest()
        protocol['continuation'] = dict(source_batch=str(source),
            source_protocol_sha256=runner.digest(source / 'protocol.json'),
            created_at=datetime.now(timezone.utc).isoformat(), inherited_valid=len(inherited),
            note='Mixed-provider exploratory continuation; inherited results retain their original manifests.')
        for path in [Path(__file__), Path(__file__).with_name('experiment_two_dragon_worker.py')]:
            protocol['file_hashes'][str(path.resolve())] = runner.digest(path)
        (target / 'inherited_results.json').write_text(json.dumps(dict(results=inherited), indent=2), encoding='utf-8')
        protocol['file_hashes'][str(target / 'inherited_results.json')] = runner.digest(target / 'inherited_results.json')
        for record in inherited:
            protocol['file_hashes'][str(target / record['target_prediction'])] = record['prediction_sha256']
        (target / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
        print(f'Continuation prepared: {len(inherited)} valid results inherited; source batch preserved.', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-batch', default='exp2-night-v3')
    parser.add_argument('--batch-id', default='exp2-night-v3-dragon')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--score-only', action='store_true')
    args = parser.parse_args(argv)
    for name in (args.source_batch, args.batch_id):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', name):
            parser.error('Invalid batch name')
    if args.batch_id == args.source_batch:
        parser.error('Use a distinct continuation batch to preserve the official-API run')
    base = ROOT / 'reports/experiment_two' / args.batch_id
    if args.run:
        from Benchmark.scripts.run_ase2022_llm_baseline import _load_env_file
        _load_env_file(ROOT / '.env')
        resolve_config(os.environ)
    if not base.exists():
        if args.resume or args.score_only:
            parser.error('Continuation not prepared; omit --resume on first use')
        prepare_continuation(ROOT / 'reports/experiment_two' / args.source_batch, base)
    elif not args.resume and not args.score_only:
        parser.error('Continuation exists; use --resume')
    protocol = runner.read(base / 'protocol.json')
    runner.verify_frozen(protocol)
    if protocol.get('transport') != transport_manifest():
        parser.error('Not a DragonAPI continuation batch')
    if not args.run and not args.score_only:
        print('Prepared only. Add --run --resume to enable model calls.')
        return 0
    import run_experiment_two_utf8 as entry
    original_worker = runner.run_worker
    original_score = runner.score

    def dragon_worker(cmd, env, log, cancel):
        cmd = list(cmd)
        cmd[2] = str(Path(__file__).with_name('experiment_two_dragon_worker.py'))
        cmd += ['--base-url', BASE_URL]
        return original_worker(cmd, env, log, cancel)

    def annotated_score(current, directory):
        original_score(current, directory)
        path = directory / 'analysis/summary.json'
        summary = runner.read(path)
        summary['continuation'] = protocol['continuation']
        summary['transport'] = protocol['transport']
        summary['interpretation'] = 'Mixed official/relay results. Differences cannot be attributed solely to rule/graph interventions. Cost totals here exclude inherited calls; see source batch.'
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    old_argv = sys.argv
    sys.argv = [__file__, '--batch-id', args.batch_id]
    sys.argv += ['--score-only'] if args.score_only else ['--run', '--resume', '--serial']
    runner.run_worker = dragon_worker
    runner.score = annotated_score
    try:
        return entry.main()
    finally:
        runner.run_worker = original_worker
        runner.score = original_score
        sys.argv = old_argv


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(str(error))
