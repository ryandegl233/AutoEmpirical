import importlib.util
import json
from pathlib import Path


def load_monitor():
    path=Path(__file__).resolve().parents[1]/'tools/watch_experiment_two.py'
    assert path.exists(), 'read-only progress monitor is missing'
    spec=importlib.util.spec_from_file_location('progress_monitor',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_counts_completed_invalid_and_partial_outputs_without_modification(tmp_path):
    m=load_monitor()
    (tmp_path/'protocol.json').write_text(json.dumps(dict(cases=[1,2],arms=['E00','E10'],record_ids=['r1','r2'])))
    for arm,case,text in [('E00',1,json.dumps(dict(record_id='r1',stage3_valid=True))),
                          ('E10',1,json.dumps(dict(record_id='r1',stage3_valid=False))),
                          ('E00',2,'{"record_id":')]:
        d=tmp_path/'evaluation48'/arm/f'case{case:02}';d.mkdir(parents=True)
        (d/'predictions_fake.jsonl').write_text(text)
    before={p:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    s=m.snapshot(tmp_path)
    assert (s['total'],s['done'],s['valid'],s['invalid'])==(4,2,1,1)
    assert s['unreadable']==['E00/case02']
    assert {p:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}==before


def test_trace_without_prediction_is_not_counted_as_completed(tmp_path):
    m=load_monitor()
    (tmp_path/'protocol.json').write_text(json.dumps(dict(cases=[1],arms=['E00'],record_ids=['r1'])))
    trace=tmp_path/'request_traces/E00/case01';trace.mkdir(parents=True)
    (trace/'a.request.json').write_text('{}')
    s=m.snapshot(tmp_path)
    assert s['done']==0
    assert s['started'][0]['requests']==1 and s['started'][0]['responses']==0
    (trace/'a.response.json').write_text('{}')
    assert m.snapshot(tmp_path)['done']==0


def test_empty_open_output_is_pending_not_corrupt(tmp_path):
    m=load_monitor()
    (tmp_path/'protocol.json').write_text(json.dumps(dict(cases=[1],arms=['E00'],record_ids=['r1'])))
    d=tmp_path/'evaluation48/E00/case01';d.mkdir(parents=True)
    (d/'predictions_fake.jsonl').write_text('')
    s=m.snapshot(tmp_path)
    assert s['done']==0 and not s['unreadable']
    assert s['started'][0]['case']=='E00/case01'


def test_monitor_counts_case_shards_after_layout_repair(tmp_path):
    m=load_monitor()
    (tmp_path/'protocol.json').write_text(json.dumps(dict(cases=[1],arms=['E00'],record_ids=['r1'])))
    d=tmp_path/'shards/case01/evaluation48/E00/case01';d.mkdir(parents=True)
    (d/'predictions_fake.jsonl').write_text(json.dumps(dict(record_id='r1',stage3_valid=True)))
    assert m.snapshot(tmp_path)['done']==1
