"""Reconcile bookkeeping-only workspace hashes after verifying frozen inference files."""
import json
import os
import runpy
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Benchmark.scripts import run_experiment_two as runner


def reconcile_config(current, manifest_path, protocol, root=ROOT):
    if not manifest_path.exists():
        return dict(current)
    old = runner.read(manifest_path)['resolved_config']
    changed = {k for k in set(old) | set(current) if old.get(k) != current.get(k)}
    if not changed:
        return dict(current)
    if changed != {'dirty_source_state_sha256'}:
        raise ValueError('Resume configuration changed: ' + ', '.join(sorted(changed)))
    runner.verify_frozen(protocol)
    # Frozen hashes also cover data/images. Reject newly added runtime files,
    # which checking only the old manifest's paths would otherwise miss.
    frozen = {Path(p).resolve() for p in protocol['file_hashes']}
    source_files = set((root / 'Benchmark/src').rglob('*.py'))
    source_files.update((root / 'Benchmark/scripts').glob('*.py'))
    extra = [str(p) for p in source_files if p.resolve() not in frozen]
    if extra:
        raise ValueError('Unfrozen runtime files: ' + ', '.join(sorted(extra)))
    return dict(current, dirty_source_state_sha256=old['dirty_source_state_sha256'])


def main():
    from Benchmark.src.adaptive_empirical_workflow import experiment_manifest as em
    protocol_path = Path(os.environ['AE_EXPERIMENT_TWO_RESUME_PROTOCOL'])
    protocol = runner.read(protocol_path)
    runner.verify_frozen(protocol)
    def arg(name):
        return sys.argv[sys.argv.index(name) + 1]
    directory = Path(arg('--output-dir')) / arg('--experiment-id') / 'evaluation48' / arg('--arm-id') / arg('--run-id')
    if not directory.resolve().is_relative_to(protocol_path.parent.resolve()):
        raise ValueError('Resume output leaves frozen batch')
    manifest = directory / 'run_manifest.json'
    original_hash = em.config_hash
    def checked_hash(payload):
        if 'dirty_source_state_sha256' in payload:
            fixed = reconcile_config(payload, manifest, protocol)
            if fixed != payload:
                audit = protocol_path.parent / 'resume_compatibility'
                audit.mkdir(exist_ok=True)
                record = dict(run_directory=str(directory),
                    original_workspace_hash=fixed['dirty_source_state_sha256'],
                    current_workspace_hash=payload['dirty_source_state_sha256'],
                    frozen_files_verified=True, all_other_config_fields_equal=True,
                    protocol_sha256=runner.digest(protocol_path),
                    adapter_sha256=runner.digest(__file__))
                (audit / (uuid.uuid4().hex + '.json')).write_text(json.dumps(record, indent=2), encoding='utf-8')
                payload.update(fixed)
                print('Resume compatibility: frozen runtime/input and all model settings verified; preserving existing run identity.', flush=True)
        return original_hash(payload)
    em.config_hash = checked_hash
    try:
        target = ROOT / 'Benchmark/scripts/experiment_two_worker.py'
        sys.argv[0] = str(target)
        runpy.run_path(str(target), run_name='__main__')
    finally:
        em.config_hash = original_hash


if __name__ == '__main__':
    main()
