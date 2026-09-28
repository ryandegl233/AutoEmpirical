from __future__ import annotations

from Benchmark.src.adaptive_empirical_workflow.concurrency_profile import (
    choose_concurrency,
)
from Benchmark.scripts.profile_adaptive_concurrency import (
    candidate_arguments,
    profile_request_fingerprint,
)
from Benchmark.scripts import profile_adaptive_concurrency as profile_script
import json
import pytest


def _candidate(
    concurrency: int,
    *,
    elapsed: float,
    stage2: float,
    stage3: float,
    end_to_end: float,
    invalid: int = 0,
) -> dict[str, object]:
    return {
        "concurrency": concurrency,
        "elapsed_seconds": elapsed,
        "metrics": {
            "stage2": {"accuracy": stage2, "invalid_count": invalid},
            "stage3": {"joint_accuracy": stage3, "invalid_count": invalid},
            "end_to_end": {"exact_match_accuracy": end_to_end},
        },
    }


def test_choose_concurrency_rejects_faster_candidate_with_lower_accuracy() -> None:
    winner = choose_concurrency(
        [
            _candidate(3, elapsed=100, stage2=1, stage3=1, end_to_end=1),
            _candidate(5, elapsed=70, stage2=1, stage3=0.9, end_to_end=0.9),
        ]
    )

    assert winner["concurrency"] == 3


def test_choose_concurrency_uses_latency_after_accuracy_and_validity_tie() -> None:
    winner = choose_concurrency(
        [
            _candidate(3, elapsed=100, stage2=1, stage3=1, end_to_end=1),
            _candidate(4, elapsed=80, stage2=1, stage3=1, end_to_end=1),
            _candidate(
                5,
                elapsed=60,
                stage2=1,
                stage3=1,
                end_to_end=1,
                invalid=1,
            ),
        ]
    )

    assert winner["concurrency"] == 4


def test_choose_concurrency_penalizes_stage3_invalid_outputs() -> None:
    valid = _candidate(3, elapsed=90, stage2=1, stage3=1, end_to_end=1)
    invalid = _candidate(5, elapsed=60, stage2=1, stage3=1, end_to_end=1)
    invalid["metrics"]["stage2"]["invalid_count"] = 0
    invalid["metrics"]["stage3"]["invalid_count"] = 1

    assert choose_concurrency([valid, invalid])["concurrency"] == 3


def test_profile_candidate_runs_are_fresh_inside_an_audited_profile(
    tmp_path,
) -> None:
    normal = candidate_arguments(
        ["--domain", "ase2022"],
        concurrency=4,
        output_root=tmp_path,
        force_rerun=False,
    )
    forced = candidate_arguments(
        ["--domain", "ase2022"],
        concurrency=4,
        output_root=tmp_path,
        force_rerun=True,
    )

    assert "--no-resume" in normal
    assert "--no-resume" in forced


def test_completed_profile_is_reused_without_paid_calls(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    profile = {
        "status": "completed",
        "request_fingerprint": profile_request_fingerprint(
            [3, 4, 5],
            [],
        ),
        "winner": {"concurrency": 4},
        "candidates": [],
    }
    profile_path = tmp_path / "concurrency_profile.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    monkeypatch.setattr(
        profile_script,
        "run_cli",
        lambda args: (_ for _ in ()).throw(
            AssertionError("cached profile must not call the model")
        ),
    )

    profile_script.main(["--output-root", str(tmp_path)])

    assert json.loads(capsys.readouterr().out) == profile


def test_interrupted_forced_profile_is_not_reused(tmp_path) -> None:
    profile_path = tmp_path / "concurrency_profile.json"
    profile_path.write_text(
        json.dumps({"status": "in_progress"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="incomplete concurrency profile"):
        profile_script.main(["--output-root", str(tmp_path)])


def test_completed_profile_with_different_request_is_not_reused(
    tmp_path,
) -> None:
    profile_path = tmp_path / "concurrency_profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "request_fingerprint": profile_request_fingerprint(
                    [3, 4, 5],
                    ["--domain", "issta2024"],
                ),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="different request"):
        profile_script.main(
            [
                "--output-root",
                str(tmp_path),
                "--",
                "--domain",
                "ase2022",
            ]
        )


def test_incomplete_profile_requires_explicit_force_before_paid_rerun(
    tmp_path,
) -> None:
    (tmp_path / "concurrency-3").mkdir()

    with pytest.raises(ValueError, match="incomplete concurrency profile"):
        profile_script.main(["--output-root", str(tmp_path)])


def test_profile_rejects_hidden_no_resume_override(tmp_path) -> None:
    with pytest.raises(ValueError, match="must not override"):
        profile_script.main(
            [
                "--output-root",
                str(tmp_path),
                "--",
                "--domain",
                "ase2022",
                "--no-resume",
            ]
        )
