import importlib.util
import json
from pathlib import Path

import pytest
from Benchmark.scripts import run_experiment_two as original
from Benchmark.src.adaptive_empirical_workflow.experiment_manifest import build_run_identity, prepare_run_directory


def recovery():
    path=Path(__file__).resolve().parents[1]/'tools/run_experiment_two_sharded.py'
    assert path.exists(), 'case-isolated recovery entry is missing'
    spec=importlib.util.spec_from_file_location('sharded',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def protocol():
    return dict(inputs={k:k for k in ['cohort','split','examples','example_source','supplemental']},
        record_ids=['r1','r2'],model='fake',graph_serialization='graph')


def identity(cmd):
    def arg(flag): return cmd[cmd.index(flag)+1]
    return build_run_identity(output_root=arg('--output-dir'),experiment_id=arg('--experiment-id'),
        split_id='evaluation48',arm_id=arg('--arm-id'),run_id=arg('--run-id'),
        resolved_config={'record_ids':[arg('--record-ids')],'policy':'unchanged'})


def test_original_collision_reproduces_and_shards_keep_strict_hash_check(tmp_path):
    legacy=tmp_path/'legacy'
    prepare_run_directory(identity(original.command(protocol(),legacy,'E10',2,dry=True)))
    with pytest.raises(ValueError,match='arm configuration mismatch'):
        prepare_run_directory(identity(original.command(protocol(),legacy,'E10',1,dry=True)))
    fix=recovery();base=tmp_path/'fixed'
    a=identity(fix.command(protocol(),base,'E10',2,dry=True))
    b=identity(fix.command(protocol(),base,'E10',1,dry=True))
    assert a.config_hash!=b.config_hash
    prepare_run_directory(a);prepare_run_directory(b)
    assert a.run_directory.parent!=b.run_directory.parent
    assert original.command(protocol(),base,'E10',1,dry=True)[-1]=='--dry-run'


def test_reads_old_and_new_results_without_rerunning_invalid_or_changing_files(tmp_path):
    fix=recovery()
    for base,case,valid in [(tmp_path,1,False),(tmp_path/'shards/case02',2,True)]:
        out=base/'evaluation48/E10'/f'case{case:02}';out.mkdir(parents=True)
        (out/'predictions_fake.jsonl').write_text(json.dumps(dict(record_id=f'r{case}',stage3_valid=valid)))
    assert fix.completed_row(tmp_path,'E10',1,'r1')['stage3_valid'] is False
    assert fix.completed_row(tmp_path,'E10',2,'r2')['stage3_valid'] is True
    duplicate=tmp_path/'shards/case01/evaluation48/E10/case01';duplicate.mkdir(parents=True)
    (duplicate/'predictions_fake.jsonl').write_text(json.dumps(dict(record_id='r1',stage3_valid=True)))
    with pytest.raises(ValueError,match='Duplicate'):
        fix.completed_row(tmp_path,'E10',1,'r1')


def test_resume_unfinished_existing_native_run_uses_its_original_directory(tmp_path):
    fix=recovery()
    out=tmp_path/'evaluation48/E01/case02';out.mkdir(parents=True)
    (out/'run_manifest.json').write_text('{}')
    assert fix.command(protocol(),tmp_path,'E01',2,dry=True)==original.command(protocol(),tmp_path,'E01',2,dry=True)


def test_recovery_audit_rejects_changed_layout_version(tmp_path):
    fix=recovery()
    with fix.batch_lock(tmp_path): pass
    path=tmp_path/'execution_layout.json'
    old=path.read_bytes()
    with fix.batch_lock(tmp_path): pass
    assert path.read_bytes()==old
    path.write_text('{}')
    with pytest.raises(ValueError,match='layout'):
        with fix.batch_lock(tmp_path): pass
