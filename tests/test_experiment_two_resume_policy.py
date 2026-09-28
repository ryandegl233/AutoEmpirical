import importlib.util
import json
import sys
import threading
import time
from pathlib import Path

import pytest


def load_entry():
    path = Path(__file__).resolve().parents[1] / 'tools/run_experiment_two_utf8.py'
    spec = importlib.util.spec_from_file_location('utf8_resume_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.path.insert(0, str(path.parent))
    import run_experiment_two_sharded as sharded
    return module, sharded.runner


@pytest.mark.parametrize('legacy', [False, True])
@pytest.mark.parametrize('serial', [False, True])
def test_resume_skips_valid_retries_invalid_once_and_preserves_history(tmp_path, monkeypatch, legacy, serial):
    entry, runner = load_entry()
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    base = tmp_path / 'Benchmark/runs/experiment_two/test'
    base.mkdir(parents=True)
    gold = tmp_path / 'gold.jsonl'
    gold.write_text('\n'.join(json.dumps(dict(record_id=f'r{i}', symptom='s', root_cause='r')) for i in range(1, 4)))
    protocol = dict(cases=[1, 2, 3], record_ids=['r1', 'r2', 'r3'], arms=['E00'],
                    file_hashes={}, configuration={'case_workers': 2}, graph_serialization='graph',
                    inputs={k: k for k in ['cohort', 'split', 'examples', 'example_source', 'supplemental']},
                    model='fake', gold_path=str(gold), gold_sha256=runner.digest(gold))
    (base / 'protocol.json').write_text(json.dumps(protocol))
    protocol_bytes = (base / 'protocol.json').read_bytes()

    def directory(case):
        root = base if legacy else base / 'shards' / f'case{case:02}'
        return root / 'evaluation48/E00' / f'case{case:02}'

    def result(case, valid):
        return dict(record_id=f'r{case}', stage3_valid=valid, symptom_prediction='s', root_cause_prediction='r')

    for case, valid in [(1, True), (2, False)]:
        out = directory(case)
        out.mkdir(parents=True)
        (out / 'run_manifest.json').write_text('{}')
        (out / 'predictions_fake.jsonl').write_text(json.dumps(result(case, valid)))
        (out / 'metrics_fake.json').write_text('{"old":true}')
    valid_bytes = (directory(1) / 'predictions_fake.jsonl').read_bytes()
    old_bytes = (directory(2) / 'predictions_fake.jsonl').read_bytes()
    calls = []
    active = peak = 0
    lock = threading.Lock()

    def local_worker(cmd, env, log, cancel):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.03)
        case = int(cmd[cmd.index('--run-id') + 1][4:])
        calls.append(case)
        out = Path(cmd[cmd.index('--output-dir') + 1]) / cmd[cmd.index('--experiment-id') + 1] / 'evaluation48/E00' / f'case{case:02}'
        out.mkdir(parents=True, exist_ok=True)
        # Remains invalid: this invocation must not loop until success.
        (out / 'predictions_fake.jsonl').write_text(json.dumps(result(case, case != 2)))
        with lock:
            active -= 1
        return 0

    monkeypatch.setattr(runner, 'run_worker', local_worker)
    monkeypatch.setattr(sys, 'argv', ['entry', '--batch-id', 'test', '--run', '--resume', '--arms', 'E00'] + (['--serial'] if serial else []))
    assert entry.main() == 0
    assert (base / 'protocol.json').read_bytes() == protocol_bytes
    if serial:
        assert peak == 1
        audits = list((base / 'execution_overrides').glob('*.json'))
        assert len(audits) == 1
        audit = json.loads(audits[0].read_text())
        assert audit['original_case_workers'] == 2
        assert audit['effective_case_workers'] == 1
        assert audit['within_case_execution_unchanged'] is True
    assert sorted(calls) == [2, 3]
    assert (directory(1) / 'predictions_fake.jsonl').read_bytes() == valid_bytes
    archived = list((base / 'resume_history').rglob('predictions_fake.jsonl'))
    assert len(archived) == 1
    assert archived[0].read_bytes() == old_bytes
    assert (archived[0].parent / 'metrics_fake.json').read_text() == '{"old":true}'
    assert json.loads((base / 'analysis/summary.json').read_text())['arms']['E00']['valid'] == 2
    # Scoring alone must not archive or schedule anything.
    calls.clear()
    monkeypatch.setattr(sys, 'argv', ['entry', '--batch-id', 'test', '--score-only'])
    assert entry.main() == 0
    assert calls == []
    assert len(list((base / 'resume_history').rglob('predictions_fake.jsonl'))) == 1
