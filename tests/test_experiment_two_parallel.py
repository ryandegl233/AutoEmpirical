import threading
import time
import json
import os
import sys
from collections import defaultdict

import pytest

from Benchmark.scripts import run_experiment_two as runner


def test_parallel_cases_overlap_but_each_case_preserves_rotated_arm_order():
    assert hasattr(runner, 'run_case_jobs'), 'case scheduler not implemented'
    protocol=dict(cases=[1,2,3,4],arms=list(runner.ARMS))
    barrier=threading.Barrier(2);lock=threading.Lock()
    active=0;peak=0;seen=defaultdict(list)
    def job(arm,case):
        nonlocal active,peak
        with lock:
            active+=1;peak=max(peak,active);seen[case].append(arm)
        if case<=2 and len(seen[case])==1: barrier.wait(timeout=3)
        time.sleep(.005)
        with lock: active-=1
        return 0
    assert runner.run_case_jobs(protocol,2,job,threading.Event())==0
    assert peak==2
    for i,case in enumerate(protocol['cases']):
        assert seen[case]==list(runner.ARMS[i:]+runner.ARMS[:i])


def test_failed_case_stops_scheduling_and_serial_mode_matches_old_order():
    assert hasattr(runner, 'run_case_jobs'), 'case scheduler not implemented'
    seen=[]
    def job(arm,case):
        seen.append((arm,case))
        return 7 if (arm,case)==('E10',1) else 0
    result=runner.run_case_jobs(dict(cases=[1,2,3],arms=list(runner.ARMS)),1,job,threading.Event())
    assert result==7 and seen==[('E00',1),('E10',1)]


def test_cancellation_prevents_any_case_from_starting():
    assert hasattr(runner, 'run_case_jobs'), 'case scheduler not implemented'
    cancel=threading.Event();cancel.set()
    assert runner.run_case_jobs(dict(cases=[1,2],arms=list(runner.ARMS)),2,
        lambda *_: pytest.fail('cancelled job started'),cancel)==130


def test_same_batch_cannot_have_two_writers(tmp_path):
    assert hasattr(runner,'batch_lock'), 'batch lock not implemented'
    with runner.batch_lock(tmp_path):
        with pytest.raises(ValueError,match='already running'):
            with runner.batch_lock(tmp_path): pass
    with runner.batch_lock(tmp_path): pass


def test_cancel_reaps_running_child_and_retains_its_log(tmp_path):
    cancel=threading.Event();log=tmp_path/'child.log'
    timer=threading.Timer(.4,cancel.set);timer.start()
    try:
        code=runner.run_worker([sys.executable,'-u','-c',
            'import time; print("child started",flush=True); time.sleep(30)'],os.environ.copy(),log,cancel)
    finally: timer.join()
    assert code==130
    assert 'child started' in log.read_text()


def test_parallel_cli_writes_isolated_results_and_resume_skips_them(tmp_path,monkeypatch):
    # Replace only frozen external inputs and model process with synthetic local workers.
    # Actual CLI scheduling, subprocess handling, checkpoints and scoring all run.
    gold=tmp_path/'gold.jsonl'
    gold.write_text(''.join(json.dumps(dict(record_id=f'r{i}',symptom='Crash',root_cause='Unknown'))+'\n' for i in [1,2]))
    protocol=dict(configuration={'case_workers':2},file_hashes={},cases=[1,2],record_ids=['r1','r2'],
        arms=list(runner.ARMS),graph_serialization='graph',gold_path=str(gold),gold_sha256=runner.digest(gold))
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    monkeypatch.setattr(runner,'freeze',lambda args:protocol)
    source='''import json,sys,time
from pathlib import Path
base,arm,case=sys.argv[1:]; case=int(case)
out=Path(base)/'evaluation48'/arm/f'case{case:02}'
out.mkdir(parents=True,exist_ok=True)
time.sleep(.05)
(out/'predictions_fake.jsonl').write_text(json.dumps(dict(record_id=f'r{case}',stage3_valid=True,symptom_prediction='Crash',root_cause_prediction='Unknown'))+'\\n')
print('finished',arm,case)
'''
    monkeypatch.setattr(runner,'command',lambda p,b,a,c,dry=False:[sys.executable,'-c',source,str(b),a,str(c)])
    assert runner.main(['--batch-id','synthetic','--run','--case-workers','2'])==0
    base=tmp_path/'reports/experiment_two/synthetic'
    paths=list(base.rglob('predictions_fake.jsonl'))
    assert len(paths)==8
    before={p:p.read_bytes() for p in paths}
    monkeypatch.setattr(runner,'run_worker',lambda *args:pytest.fail('completed case rerun'))
    assert runner.main(['--batch-id','synthetic','--run','--resume'])==0
    assert {p:p.read_bytes() for p in paths}==before
    with pytest.raises(SystemExit):
        runner.main(['--batch-id','synthetic','--run','--resume','--case-workers','3'])
    summary=runner.read(base/'analysis/summary.json')
    assert all(a['joint_correct']==2 for a in summary['arms'].values())
