from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from Benchmark.src import ase2022_stage2_filter_baseline as baseline


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "record_id",
        "paper_id",
        "issue_url",
        "title",
        "body",
        "comments",
        "state",
        "created_at",
        "symptom",
        "root_cause",
        "source_file",
        "original_label_json",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _row(record_id: str, title: str) -> dict[str, str]:
    return {
        "record_id": record_id,
        "paper_id": baseline.ASE2022_PAPER_ID,
        "issue_url": f"https://example.test/{record_id}",
        "title": title,
        "body": f"body {record_id}",
        "comments": f"comments {record_id}",
        "state": "closed",
        "created_at": "2021-01-01T00:00:00Z",
        "symptom": "LEAKED_SYMPTOM",
        "root_cause": "LEAKED_CAUSE",
        "source_file": "LEAKED_SOURCE",
        "original_label_json": '{"stage":"stage2"}',
    }


def test_loader_derives_membership_labels_without_leaking_fields(tmp_path: Path) -> None:
    stage1 = tmp_path / "stage1.csv"
    stage2 = tmp_path / "stage2.csv"
    _write_csv(stage1, [_row("a", "accepted"), _row("b", "rejected")])
    _write_csv(stage2, [_row("a", "accepted")])

    examples = baseline.load_ase2022_stage2_filter_examples(stage1, stage2)

    assert [row["decision"] for row in examples] == [
        baseline.ACCEPTED_FAULT,
        baseline.REJECTED_CANDIDATE,
    ]
    assert set(examples[0]) == {
        "record_id",
        "paper_id",
        "issue_url",
        "title",
        "body",
        "comments",
        "state",
        "created_at",
        "decision",
    }
    assert "LEAKED" not in repr(examples)


@pytest.mark.parametrize("dataset", ["stage1", "stage2"])
def test_loader_rejects_duplicate_record_ids(tmp_path: Path, dataset: str) -> None:
    stage1_rows = [_row("a", "one"), _row("b", "two")]
    stage2_rows = [_row("a", "one")]
    if dataset == "stage1":
        stage1_rows.append(_row("a", "duplicate"))
    else:
        stage2_rows.append(_row("a", "duplicate"))
    stage1 = tmp_path / "stage1.csv"
    stage2 = tmp_path / "stage2.csv"
    _write_csv(stage1, stage1_rows)
    _write_csv(stage2, stage2_rows)

    with pytest.raises(ValueError, match="duplicate record_id"):
        baseline.load_ase2022_stage2_filter_examples(stage1, stage2)


def test_loader_rejects_stage2_records_missing_from_stage1(tmp_path: Path) -> None:
    stage1 = tmp_path / "stage1.csv"
    stage2 = tmp_path / "stage2.csv"
    _write_csv(stage1, [_row("a", "one")])
    _write_csv(stage2, [_row("missing", "missing")])

    with pytest.raises(ValueError, match="absent from Stage 1"):
        baseline.load_ase2022_stage2_filter_examples(stage1, stage2)


def test_balanced_sample_is_deterministic_and_balanced() -> None:
    examples = []
    for index in range(10):
        examples.append({**_row(f"p{index}", "positive"), "decision": baseline.ACCEPTED_FAULT})
        examples.append({**_row(f"n{index}", "negative"), "decision": baseline.REJECTED_CANDIDATE})

    first = baseline.select_balanced_sample(examples, per_class=4, seed=17)
    second = baseline.select_balanced_sample(examples, per_class=4, seed=17)

    assert [row["record_id"] for row in first] == [row["record_id"] for row in second]
    assert sum(row["decision"] == baseline.ACCEPTED_FAULT for row in first) == 4
    assert sum(row["decision"] == baseline.REJECTED_CANDIDATE for row in first) == 4


def test_balanced_sample_rejects_insufficient_class_pool() -> None:
    examples = [
        {**_row("p", "positive"), "decision": baseline.ACCEPTED_FAULT},
        {**_row("n", "negative"), "decision": baseline.REJECTED_CANDIDATE},
    ]
    with pytest.raises(ValueError, match="requested 2"):
        baseline.select_balanced_sample(examples, per_class=2)


def test_prompt_contains_evidence_but_not_ground_truth_or_leaked_values() -> None:
    example = {
        **_row("p1", "A real crash"),
        "decision": baseline.ACCEPTED_FAULT,
    }
    system_prompt = baseline.build_system_prompt()
    user_prompt = baseline.build_user_prompt(example)
    combined = system_prompt + user_prompt

    assert "A real crash" in user_prompt
    assert "body p1" in user_prompt
    assert baseline.ACCEPTED_FAULT not in user_prompt
    assert baseline.REJECTED_CANDIDATE not in user_prompt
    for leaked in ("LEAKED_SYMPTOM", "LEAKED_CAUSE", "LEAKED_SOURCE", "root_cause"):
        assert leaked not in combined


@pytest.mark.parametrize("decision", [baseline.ACCEPTED_FAULT, baseline.REJECTED_CANDIDATE])
def test_parse_filter_response_accepts_exact_json(decision: str) -> None:
    parsed = baseline.parse_filter_response(json.dumps({"decision": decision}))
    assert parsed == {
        "decision": decision,
        "invalid": False,
        "error": "",
        "normalized_output": json.dumps({"decision": decision}),
        "format_normalized": False,
        "output_format": "strict_json",
    }


@pytest.mark.parametrize(
    ("raw", "decision"),
    [
        ('```json\n{"decision":"accepted_fault"}\n```', baseline.ACCEPTED_FAULT),
        ('```\n{"decision":"rejected_candidate"}\n```', baseline.REJECTED_CANDIDATE),
    ],
)
def test_parse_filter_response_accepts_single_markdown_json_fence(
    raw: str, decision: str
) -> None:
    parsed = baseline.parse_filter_response(raw)
    assert parsed["decision"] == decision
    assert parsed["invalid"] is False
    assert parsed["normalized_output"] == f'{{"decision":"{decision}"}}'
    assert parsed["format_normalized"] is True
    assert parsed["output_format"] == "markdown_fenced_json"


@pytest.mark.parametrize(
    "raw",
    [
        '{"decision":"accepted_fault","reason":"extra"}',
        '{"decision":"maybe"}',
        'answer: {"decision":"accepted_fault"}',
        'Explanation\n```json\n{"decision":"accepted_fault"}\n```',
        '```json\n{"decision":"accepted_fault"}\n```\n```json\n{"decision":"rejected_candidate"}\n```',
        "not json",
        "[]",
    ],
)
def test_parse_filter_response_rejects_non_contract_outputs(raw: str) -> None:
    parsed = baseline.parse_filter_response(raw)
    assert parsed["invalid"] is True
    assert parsed["decision"] == ""
    assert parsed["error"]


def test_evaluate_filter_predictions_reports_valid_denominator_and_metrics() -> None:
    examples = [
        {"record_id": "tp", "decision": baseline.ACCEPTED_FAULT},
        {"record_id": "fn", "decision": baseline.ACCEPTED_FAULT},
        {"record_id": "fp", "decision": baseline.REJECTED_CANDIDATE},
        {"record_id": "tn", "decision": baseline.REJECTED_CANDIDATE},
        {"record_id": "invalid", "decision": baseline.ACCEPTED_FAULT},
    ]
    predictions = [
        {"record_id": "tp", "decision": baseline.ACCEPTED_FAULT, "invalid": False},
        {"record_id": "fn", "decision": baseline.REJECTED_CANDIDATE, "invalid": False},
        {"record_id": "fp", "decision": baseline.ACCEPTED_FAULT, "invalid": False},
        {"record_id": "tn", "decision": baseline.REJECTED_CANDIDATE, "invalid": False},
        {"record_id": "invalid", "decision": "", "invalid": True},
    ]

    metrics = baseline.evaluate_filter_predictions(examples, predictions)

    assert metrics["n"] == 5
    assert metrics["valid_count"] == 4
    assert metrics["invalid_count"] == 1
    assert metrics["true_positive"] == 1
    assert metrics["false_negative"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["true_negative"] == 1
    assert metrics["accuracy"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5
    assert metrics["specificity"] == 0.5
    assert metrics["balanced_accuracy"] == 0.5
    assert metrics["invalid_rate"] == 0.2


def test_writers_create_leakage_safe_prompts_and_manifest(tmp_path: Path) -> None:
    all_examples = [
        {**_row("p", "positive"), "decision": baseline.ACCEPTED_FAULT},
        {**_row("n", "negative"), "decision": baseline.REJECTED_CANDIDATE},
    ]
    sample_path = baseline.write_sample_csv(all_examples, tmp_path / "sample.csv")
    prompts_path = baseline.write_prompts_jsonl(all_examples, tmp_path / "prompts.jsonl")
    manifest_path = baseline.write_manifest(
        all_examples,
        all_examples,
        tmp_path / "manifest.json",
        seed=11,
        per_class=1,
        stage1_path="one.csv",
        stage2_path="two.csv",
        mode="balanced_pilot",
    )

    with sample_path.open(encoding="utf-8") as handle:
        sample_rows = list(csv.DictReader(handle))
    prompt_rows = [json.loads(line) for line in prompts_path.read_text(encoding="utf-8").splitlines()]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert len(sample_rows) == 2
    assert {row["decision"] for row in sample_rows} == baseline.ALLOWED_DECISIONS
    assert len(prompt_rows) == 2
    assert {row["ground_truth"] for row in prompt_rows} == baseline.ALLOWED_DECISIONS
    for row in prompt_rows:
        messages = row["system_prompt"] + row["user_prompt"]
        assert row["ground_truth"] not in row["user_prompt"]
        assert "LEAKED" not in messages
    assert manifest["mode"] == "balanced_pilot"
    assert manifest["seed"] == 11
    assert manifest["all_counts"] == {baseline.ACCEPTED_FAULT: 1, baseline.REJECTED_CANDIDATE: 1}
    assert manifest["selected_counts"] == manifest["all_counts"]
    assert manifest["label_semantics"][baseline.ACCEPTED_FAULT].startswith("record_id is present")
