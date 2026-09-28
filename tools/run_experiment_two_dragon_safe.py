"""Explicit compatibility revision; old frozen batches and adapters stay unchanged."""
import sys
from pathlib import Path

import run_experiment_two_dragon as legacy


def safe_transport_manifest():
    return dict(legacy_transport_manifest(), compatibility_revision='dragon-safe-v1',
        json_handling='unwrap one complete unambiguous JSON object; unchanged schema validation',
        circuit_breaker='immediate channel/auth/billing/parameter errors; three consecutive transient failures')


legacy_transport_manifest = legacy.transport_manifest


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if not any(v == '--source-batch' or v.startswith('--source-batch=') for v in args):
        args += ['--source-batch', 'exp2-night-v3-dragon']
    if not any(v == '--batch-id' or v.startswith('--batch-id=') for v in args):
        args += ['--batch-id', 'exp2-night-v3-dragon-safe']
    original_prepare = legacy.prepare_continuation
    original_worker = legacy.runner.run_worker

    def prepare(source, target):
        original_prepare(source, target)
        path = Path(target) / 'protocol.json'
        protocol = legacy.runner.read(path)
        for name in ['run_experiment_two_dragon_safe.py', 'experiment_two_dragon_safe_worker.py', 'dragon_transport_guard.py']:
            source_file = Path(__file__).with_name(name).resolve()
            protocol['file_hashes'][str(source_file)] = legacy.runner.digest(source_file)
        path.write_text(__import__('json').dumps(protocol, indent=2), encoding='utf-8')

    def worker(cmd, env, log, cancel):
        cmd = list(cmd)
        cmd[2] = str(Path(__file__).with_name('experiment_two_dragon_safe_worker.py'))
        return original_worker(cmd, env, log, cancel)

    legacy.transport_manifest = safe_transport_manifest
    legacy.prepare_continuation = prepare
    legacy.runner.run_worker = worker
    try:
        return legacy.main(args)
    finally:
        legacy.transport_manifest = legacy_transport_manifest
        legacy.prepare_continuation = original_prepare
        legacy.runner.run_worker = original_worker


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(str(error))
