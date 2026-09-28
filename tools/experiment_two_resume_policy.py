"""Resume failed case/arm runs once, preserving frozen inference and attempt history."""
import argparse
import json
import shutil
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def archive_invalid(runner, base, arm, case, rid):
    row = runner.completed_row(base, arm, case, rid)
    if row is None or row.get('stage3_valid') is True:
        return
    if row.get('stage3_valid') is not False:
        raise ValueError(f'Missing boolean validity: {arm}/case{case:02}')
    # Both layouts are supported. completed_row has already rejected duplicates.
    candidates = [base / 'evaluation48' / arm / f'case{case:02}',
                  base / 'shards' / f'case{case:02}' / 'evaluation48' / arm / f'case{case:02}']
    source = next(p for p in candidates if list(p.glob('predictions_*.jsonl')))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '-' + uuid.uuid4().hex[:8]
    target = base / 'resume_history' / arm / f'case{case:02}' / stamp
    for path in (source, target):
        if not path.resolve().is_relative_to(base.resolve()):
            raise ValueError('Resume archive path leaves the batch directory')
    target.mkdir(parents=True)
    prediction = next(source.glob('predictions_*.jsonl'))
    # Save metrics/manifests before moving the single prediction file. A failed
    # copy leaves the original checkpoint intact. No directories are removed.
    for path in source.iterdir():
        if path.is_file() and path != prediction:
            shutil.copy2(path, target / path.name)
    (target / 'resume_attempt.json').write_text(json.dumps(dict(
        policy='resume_invalid_and_missing_once_per_invocation', arm=arm, case=case,
        record_id=rid, previous_valid=False, original_directory=str(source),
        policy_code_sha256=runner.digest(__file__),
        protocol_sha256=runner.digest(base / 'protocol.json'),
        traces='Original request_traces retained; worker log remains append-only.',
    ), indent=2), encoding='utf-8')
    prediction.rename(target / prediction.name)
    print(f'Retry invalid {case:02}/{arm}; previous result: {target}', flush=True)


@contextmanager
def resume_policy(runner, argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--batch-id')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--score-only', action='store_true')
    parser.add_argument('--serial', action='store_true')
    args, _ = parser.parse_known_args(argv)
    if args.serial and (not args.run or not args.resume or args.score_only):
        raise ValueError('--serial requires --run --resume')
    if args.serial and any(v == '--case-workers' or v.startswith('--case-workers=') for v in argv):
        raise ValueError('Use --serial alone to resume with one case worker')
    original = runner.run_case_jobs
    original_worker = runner.run_worker

    def resume_worker(cmd, env, log, cancel):
        cmd = list(cmd)
        cmd[2] = str(Path(__file__).with_name('experiment_two_resume_worker.py'))
        env = dict(env, AE_EXPERIMENT_TWO_RESUME_PROTOCOL=str(
            runner.ROOT / 'reports/experiment_two' / args.batch_id / 'protocol.json'))
        return original_worker(cmd, env, log, cancel)

    def resumed_jobs(protocol, workers, job, cancel):
        # Native main validates the arguments and frozen hashes, and holds the
        # batch lock before reaching this scheduler.
        base = runner.ROOT / 'reports/experiment_two' / args.batch_id
        if args.serial:
            audit = base / 'execution_overrides'
            audit.mkdir(exist_ok=True)
            record = dict(created_at=datetime.now(timezone.utc).isoformat(),
                original_case_workers=workers, effective_case_workers=1,
                within_case_execution_unchanged=True,
                protocol_sha256=runner.digest(base / 'protocol.json'),
                policy_code_sha256=runner.digest(__file__),
                reason='User requested original serial case scheduling for resume')
            (audit / (uuid.uuid4().hex + '.json')).write_text(
                json.dumps(record, indent=2), encoding='utf-8')
            workers = 1
            print('Serial resume: one case at a time; original within-case workflow retained.', flush=True)

        def resumed_job(arm, case):
            runner.verify_frozen(protocol)
            archive_invalid(runner, base, arm, case, protocol['record_ids'][case - 1])
            return job(arm, case)

        return original(protocol, workers, resumed_job, cancel)

    if args.run and args.resume and not args.score_only:
        runner.run_case_jobs = resumed_jobs
        runner.run_worker = resume_worker
    try:
        yield
    finally:
        runner.run_case_jobs = original
        runner.run_worker = original_worker
