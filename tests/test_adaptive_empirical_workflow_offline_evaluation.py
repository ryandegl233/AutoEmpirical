from __future__ import annotations

import ast
import csv
import inspect
import json
from pathlib import Path

import pytest

from Benchmark.src.adaptive_empirical_workflow.experiment import (
    evaluate_evidence_only_execution,
    evaluate_experiment,
)
from Benchmark.src.adaptive_empirical_workflow import offline_evaluation
from Benchmark.src.adaptive_empirical_workflow.offline_evaluation import (
    evaluate_registered_baseline_preservation_experiment,
)


def test_production_offline_evaluator_exposes_no_registry_injection() -> None:
    assert not hasattr(
        offline_evaluation, "_evaluate_baseline_preservation_with_registry"
    )
    assert not hasattr(offline_evaluation, "EvaluationTrustedRegistry")
    tree = ast.parse(inspect.getsource(offline_evaluation))
    assert all(
        argument.arg not in {"registry", "trust_root"}
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        for argument in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
    )


def test_evidence_only_execution_never_reports_accuracy_as_zero() -> None:
    records = [{"record_id": "r1", "title": "x", "body": "y"}]
    rows = [
        {
            "record_id": "r1",
            "stage3_valid": True,
            "symptom_prediction": "Crash",
            "root_cause_prediction": "Incorrect Code Logic",
            "audit": None,
        }
    ]

    metrics = evaluate_evidence_only_execution(records, rows, stage="stage3")
    assert metrics["evaluation"]["status"] == "not_evaluated"

    def keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(map(str, value)).union(*(keys(item) for item in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(item) for item in value))
        return set()

    assert not any(
        token in key.lower()
        for key in keys(metrics)
        for token in ("accuracy", "correct", "help", "harm")
    )
    with pytest.raises(ValueError, match="label accuracy evaluation requires"):
        evaluate_experiment(records, rows, stage="stage3")


def test_evidence_only_execution_reports_revision_funnel_without_outcome_metrics() -> (
    None
):
    records = [{"record_id": "r1", "title": "x", "body": "y"}]
    rows = [
        {
            "record_id": "r1",
            "stage3_valid": True,
            "audit": {
                "stage3": {
                    "baseline_anchor": {"valid": True},
                    "revision_certificates": [],
                    "revision_audit": {
                        "proposal_digests": ["a" * 64],
                        "routed_card_ids": ["root/api-vs-logic"],
                        "proposal_entries": [
                            {
                                "proposal_digest": "a" * 64,
                                "dimension": "root_cause",
                                "boundary_card_id": "root/api-vs-logic",
                                "baseline_label": "API Misuse",
                                "proposed_label": "Incorrect Code Logic",
                            }
                        ],
                        "assessment_attempts": [
                            {
                                "proposal_digest": "a" * 64,
                                "dimension": "root_cause",
                                "boundary_card_id": "root/api-vs-logic",
                                "assessor_team_id": "A",
                                "returned_assessment_digest": None,
                                "verdict": None,
                                "operational_failure": True,
                            }
                        ],
                        "dual_revise_proposal_digests": [],
                        "consistency_attempts": [],
                        "consistency_pass_proposal_digests": [],
                        "issued_certificate_digests": [],
                        "issued_certificate_entries": [],
                        "applied_certificate_digests": [],
                        "ambiguous_certified_dimensions": [],
                        "failed_proposal_digests": ["a" * 64],
                    },
                }
            },
        }
    ]

    metrics = evaluate_evidence_only_execution(records, rows, stage="stage3")

    assert metrics["stage3"]["baseline_preservation"]["revision_funnel"] == {
        "instrumented_record_count": 1,
        "proposal_count": 1,
        "routed_card_count": 1,
        "assessment_attempt_count": 1,
        "dual_revise_count": 0,
        "cross_attempt_count": 0,
        "cross_pass_count": 0,
        "issued_count": 0,
        "applied_count": 0,
        "ambiguous_dimension_count": 0,
        "operational_failure_count": 1,
    }
    assert "accuracy" not in json.dumps(metrics).lower()
    assert "help" not in json.dumps(metrics).lower()
    assert "harm" not in json.dumps(metrics).lower()


def test_offline_trusted_labels_use_exact_fixed_denominator(tmp_path: Path) -> None:
    manifest_path = Path(
        "Benchmark/configs/evaluation_trust/"
        "ase2022_contaminated_dev50_labels_v1.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split = json.loads(
        Path(manifest["selection"]["split_manifest_relative_path"]).read_text(
            encoding="utf-8"
        )
    )
    selected_ids = split["splits"][manifest["selection"]["split_name"]]["record_ids"]
    with Path(manifest["label_source"]["relative_path"]).open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        labels = {row["record_id"]: row for row in csv.DictReader(handle)}
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        "".join(
            json.dumps(
                {
                    "record_id": record_id,
                    "stage3_valid": True,
                    "symptom_prediction": labels[record_id]["symptom"],
                    "root_cause_prediction": labels[record_id]["root_cause"],
                    "audit": None,
                }
            )
            + "\n"
            for record_id in selected_ids
        ),
        encoding="utf-8",
    )
    metrics = evaluate_registered_baseline_preservation_experiment(
        predictions,
        evaluation_manifest_path=manifest_path,
    )
    assert metrics["evaluation"]["fixed_denominator"] == 50
    assert metrics["stage3"]["joint_accuracy"] == 1.0

    predictions.write_text(
        predictions.read_text(encoding="utf-8")
        + json.dumps({"record_id": "extra"})
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exactly match"):
        evaluate_registered_baseline_preservation_experiment(
            predictions,
            evaluation_manifest_path=manifest_path,
        )
