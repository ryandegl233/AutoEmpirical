from __future__ import annotations

import csv
from pathlib import Path

from Benchmark.src.issta2024_bugs_in_pods_baseline import (
    SAFE_MODEL_FIELDS,
    _diff_eligibility_audit,
    _read_stage,
    _has_complete_sampling_evidence,
    build_stage2_user_prompt,
    build_stage3_user_prompt,
)


def complete_record() -> dict[str, str]:
    return {
        "record_id": "record-1",
        "paper_id": "issta2024_bugs_in_pods_understanding_bugs",
        "source_project": "runc",
        "issue_url": "https://github.com/opencontainers/runc/commit/" + "a" * 40,
        "title": "Fix state transition",
        "body": "Prevent the process from entering an invalid state.",
        "comments": "not_available_in_source",
        "state": "committed",
        "created_at": "2024-01-02T03:04:05Z",
        "changed_files": '["state.go"]',
        "code_diff": (
            "diff --git a/state.go b/state.go\n"
            "--- a/state.go\n"
            "+++ b/state.go\n"
            "@@ -1 +1 @@\n"
            "-state = invalid\n"
            "+state = ready\n"
        ),
        "decision": "accepted_fault",
        "symptom": "Wrong Container Behavior",
        "root_cause": "Wrong Code Logic",
    }


def test_code_diff_is_a_model_input_field() -> None:
    assert "code_diff" in SAFE_MODEL_FIELDS


def test_sampling_rejects_record_without_code_diff() -> None:
    record = complete_record()
    record["code_diff"] = ""

    assert _has_complete_sampling_evidence(record) is False


def test_sampling_rejects_diff_that_cannot_fit_the_input_budget() -> None:
    record = complete_record()
    record["code_diff"] = "x" * 200_001

    assert _has_complete_sampling_evidence(record) is False


def test_stage2_and_stage3_prompts_include_the_actual_code_diff() -> None:
    record = complete_record()

    stage2_prompt = build_stage2_user_prompt(record)
    stage3_prompt = build_stage3_user_prompt(record)

    for prompt in (stage2_prompt, stage3_prompt):
        assert "Code Diff (unified diff from GitHub API):" in prompt
        assert "-state = invalid" in prompt
        assert "+state = ready" in prompt
        assert "Wrong Code Logic" not in prompt


def test_diff_eligibility_audit_separates_missing_and_oversized_records() -> None:
    eligible = complete_record()
    missing = {**complete_record(), "record_id": "record-2", "code_diff": ""}
    oversized = {
        **complete_record(),
        "record_id": "record-3",
        "decision": "rejected_candidate",
        "code_diff": "x" * 200_001,
    }

    audit = _diff_eligibility_audit([eligible, missing, oversized])

    assert audit["nonempty_count"] == 2
    assert audit["eligible_count"] == 1
    assert audit["missing_count"] == 1
    assert audit["oversized_count"] == 1
    assert audit["oversized_by_decision"] == {"rejected_candidate": 1}


def test_stage_reader_accepts_a_code_diff_larger_than_csv_default_limit(
    tmp_path: Path,
) -> None:
    path = tmp_path / "stage1.csv"
    row = {
        "record_id": "record-1",
        "paper_id": "issta2024_bugs_in_pods_understanding_bugs",
        "code_diff": "x" * 150_000,
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    loaded = _read_stage(path, "Stage 1")

    assert loaded["record-1"]["code_diff"] == "x" * 150_000
