from __future__ import annotations

import json
from pathlib import Path

import pytest

from Benchmark.src.adaptive_empirical_workflow.experiment_manifest import (
    build_run_identity,
    config_hash,
    prepare_run_directory,
    write_run_manifest,
)


def _config(*, thinking: str = "off", concurrency: int = 1) -> dict[str, object]:
    return {
        "domain": "ase2022",
        "model": "deepseek-v4-flash",
        "agent_policy": {"thinking": thinking},
        "taxonomy_sha256": "a" * 64,
        "cohort_sha256": "b" * 64,
        "split_manifest_sha256": "c" * 64,
        "prompt_hashes": {"team_a": "d" * 64, "team_b": "e" * 64},
        "code_revision": "abc1234",
        "concurrency": concurrency,
    }


def test_config_hash_changes_for_algorithm_policy_but_not_execution_concurrency() -> (
    None
):
    assert config_hash(_config(thinking="off", concurrency=1)) == config_hash(
        _config(thinking="off", concurrency=8)
    )
    assert config_hash(_config(thinking="off")) != config_hash(
        _config(thinking="all-stage3")
    )


def test_run_identity_builds_collision_safe_experiment_split_arm_run_path(
    tmp_path: Path,
) -> None:
    identity = build_run_identity(
        output_root=tmp_path,
        experiment_id="dual-team-v1",
        split_id="validation",
        arm_id="a3-all-thinking",
        run_id="run-01",
        resolved_config=_config(thinking="all-stage3"),
    )
    assert identity.run_directory == (
        tmp_path / "dual-team-v1" / "validation" / "a3-all-thinking" / "run-01"
    )
    assert identity.config_hash == config_hash(_config(thinking="all-stage3"))


def test_prepare_run_directory_rejects_existing_different_configuration(
    tmp_path: Path,
) -> None:
    first = build_run_identity(
        output_root=tmp_path,
        experiment_id="exp",
        split_id="dev",
        arm_id="a0",
        run_id="run-1",
        resolved_config=_config(thinking="off"),
    )
    prepare_run_directory(first)
    write_run_manifest(first, {"config_hash": first.config_hash})

    changed = build_run_identity(
        output_root=tmp_path,
        experiment_id="exp",
        split_id="dev",
        arm_id="a0",
        run_id="run-1",
        resolved_config=_config(thinking="all-stage3"),
    )
    with pytest.raises(ValueError, match="existing run configuration mismatch"):
        prepare_run_directory(changed)

    stored = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    assert stored["config_hash"] == first.config_hash


def test_arm_rejects_different_configuration_in_another_run(tmp_path: Path) -> None:
    first = build_run_identity(
        output_root=tmp_path,
        experiment_id="exp",
        split_id="dev",
        arm_id="a0",
        run_id="run-1",
        resolved_config=_config(thinking="off"),
    )
    write_run_manifest(first, {"config_hash": first.config_hash})
    second = build_run_identity(
        output_root=tmp_path,
        experiment_id="exp",
        split_id="dev",
        arm_id="a0",
        run_id="run-2",
        resolved_config=_config(thinking="all-stage3"),
    )

    with pytest.raises(ValueError, match="arm configuration mismatch"):
        prepare_run_directory(second)
