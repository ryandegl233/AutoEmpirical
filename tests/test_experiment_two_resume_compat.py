import importlib.util
import json
from pathlib import Path

import pytest


def module():
    path = Path(__file__).resolve().parents[1] / 'tools/experiment_two_resume_worker.py'
    assert path.exists(), 'native resume compatibility adapter missing'
    spec = importlib.util.spec_from_file_location('resume_compat_test', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_native_config_accepts_only_workspace_digest_change_with_frozen_runtime(tmp_path):
    fix = module()
    from Benchmark.src.adaptive_empirical_workflow.experiment_manifest import (
        build_run_identity, prepare_run_directory, write_run_manifest)
    root = tmp_path / 'repo'
    source = root / 'Benchmark/src/worker.py'
    source.parent.mkdir(parents=True)
    source.write_text('# frozen runtime')
    protocol = {'file_hashes': {str(source): fix.runner.digest(source)}}
    old = dict(model='fake', code_revision='unchanged', dirty_source_state_sha256='old', record_ids=['r1'])
    identity = build_run_identity(output_root=tmp_path, experiment_id='batch', split_id='evaluation48',
                                  arm_id='E00', run_id='case01', resolved_config=old)
    write_run_manifest(identity, {'config_hash': identity.config_hash})
    current = dict(old, dirty_source_state_sha256='new')
    fixed = fix.reconcile_config(current, identity.manifest_path, protocol, root)
    assert fixed == old
    assert current['dirty_source_state_sha256'] == 'new'
    resumed = build_run_identity(output_root=tmp_path, experiment_id='batch', split_id='evaluation48',
                                 arm_id='E00', run_id='case01', resolved_config=fixed)
    assert prepare_run_directory(resumed) == identity.run_directory
    with pytest.raises(ValueError, match='model'):
        fix.reconcile_config(dict(current, model='changed'), identity.manifest_path, protocol, root)
    source.write_text('# changed runtime')
    with pytest.raises(ValueError, match='Frozen'):
        fix.reconcile_config(current, identity.manifest_path, protocol, root)
    source.write_text('# frozen runtime')
    (source.parent / 'extra.py').write_text('# unregistered code')
    with pytest.raises(ValueError, match='Unfrozen'):
        fix.reconcile_config(current, identity.manifest_path, protocol, root)


def test_no_existing_run_needs_no_compatibility_override(tmp_path):
    fix = module()
    current = {'model': 'fake', 'dirty_source_state_sha256': 'current'}
    assert fix.reconcile_config(current, tmp_path / 'absent.json', {}, tmp_path) == current
