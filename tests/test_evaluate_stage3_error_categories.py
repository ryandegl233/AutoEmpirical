from __future__ import annotations

import csv
import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest


def _evaluator():
    return importlib.import_module("Benchmark.scripts.evaluate_stage3_error_categories")


def _record_ids_sha256(*record_ids: str) -> str:
    payload = "".join(f"{record_id}\n" for record_id in sorted(record_ids))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_fixture(tmp_path: Path) -> dict[str, Path]:
    cohort = tmp_path / "cohort.csv"
    with cohort.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("record_id", "symptom", "root_cause"),
        )
        writer.writeheader()
        writer.writerows(
            (
                {
                    "record_id": "r1",
                    "symptom": "Crash",
                    "root_cause": "Incorrect Code Logic",
                },
                {
                    "record_id": "r2",
                    "symptom": "Crash",
                    "root_cause": "Incorrect Code Logic",
                },
                {
                    "record_id": "r3-not-in-audit-split",
                    "symptom": "Crash",
                    "root_cause": "Incorrect Code Logic",
                },
            )
        )
    baseline = tmp_path / "baseline.jsonl"
    baseline.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "record_id": "r1",
                        "final_prediction": {
                            "symptom": "Incorrect Functionality",
                            "root_cause": "Incorrect Code Logic",
                        },
                        "invalid": False,
                    }
                ),
                json.dumps(
                    {
                        "record_id": "r2",
                        "final_prediction": {
                            "symptom": "Crash",
                            "root_cause": "API Misuse",
                        },
                        "invalid": False,
                    }
                ),
                json.dumps(
                    {
                        "record_id": "r3-not-in-audit-split",
                        "final_prediction": {
                            "symptom": "Crash",
                            "root_cause": "Incorrect Code Logic",
                        },
                        "invalid": False,
                    }
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    candidate = tmp_path / "candidate.jsonl"
    candidate.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "record_id": "r1",
                        "stage3_valid": True,
                        "symptom_prediction": "Crash",
                        "root_cause_prediction": "Incorrect Code Logic",
                        "audit": {
                            "stage3": {
                                "evidence_deltas": [],
                                "boundary_card_ids": [],
                            }
                        },
                    }
                ),
                json.dumps(
                    {
                        "record_id": "r2",
                        "stage3_valid": True,
                        "symptom_prediction": "Crash",
                        "root_cause_prediction": "API Misuse",
                        "audit": {
                            "stage3": {
                                "evidence_deltas": [],
                                "boundary_card_ids": [],
                            }
                        },
                    }
                ),
                json.dumps(
                    {
                        "record_id": "r3-not-in-audit-split",
                        "stage3_valid": True,
                        "symptom_prediction": "Crash",
                        "root_cause_prediction": "Incorrect Code Logic",
                        "audit": {
                            "stage3": {
                                "evidence_deltas": [],
                                "boundary_card_ids": [],
                            }
                        },
                    }
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    categories = tmp_path / "categories.json"
    categories.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_id": "fixture",
                "split_kind": "contaminated_development",
                "record_ids": ["r1", "r2"],
                "record_ids_sha256": _record_ids_sha256("r1", "r2"),
                "categories": [
                    {"record_id": "r1", "dimension": "symptom", "category": "S1"},
                    {
                        "record_id": "r2",
                        "dimension": "root_cause",
                        "category": "R2",
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return {
        "cohort": cohort,
        "baseline": baseline,
        "candidate": candidate,
        "categories": categories,
        "output": tmp_path / "diagnostics.json",
    }


def test_offline_error_category_script_runs_from_outside_repository(
    tmp_path: Path,
) -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "Benchmark"
        / "scripts"
        / "evaluate_stage3_error_categories.py"
    )

    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--category-manifest" in completed.stdout


def test_offline_error_category_cli_normalizes_baseline_and_writes_provenance(
    tmp_path: Path,
) -> None:
    evaluator = _evaluator()
    paths = _write_fixture(tmp_path)

    result = evaluator.run_cli(
        [
            "--cohort-path",
            str(paths["cohort"]),
            "--baseline-predictions",
            str(paths["baseline"]),
            "--candidate-predictions",
            str(paths["candidate"]),
            "--category-manifest",
            str(paths["categories"]),
            "--output-path",
            str(paths["output"]),
            "--bootstrap-samples",
            "20",
        ]
    )

    saved = json.loads(paths["output"].read_text(encoding="utf-8"))
    assert result == saved
    assert saved["schema_version"] == 1
    assert saved["dataset_id"] == "fixture"
    assert saved["split_kind"] == "contaminated_development"
    assert saved["claim_scope"] == "development_only"
    assert saved["inputs"] == {
        "cohort_sha256": evaluator.file_sha256(paths["cohort"]),
        "baseline_predictions_sha256": evaluator.file_sha256(paths["baseline"]),
        "candidate_predictions_sha256": evaluator.file_sha256(paths["candidate"]),
        "category_manifest_sha256": evaluator.file_sha256(paths["categories"]),
    }
    assert saved["paired"]["baseline_joint_accuracy"] == 0.0
    assert saved["paired"]["candidate_joint_accuracy"] == 0.5
    assert saved["error_categories"]["categories"]["S1"]["recovery_count"] == 1
    assert saved["error_categories"]["categories"]["R2"]["recovery_count"] == 0


@pytest.mark.parametrize(
    ("update", "message"),
    (
        ({"split_kind": "final"}, "contaminated_development"),
        ({"record_ids_sha256": "0" * 64}, "record_ids_sha256 mismatch"),
        ({"schema_version": 2}, "unsupported category manifest schema"),
    ),
)
def test_offline_error_category_cli_rejects_unbound_or_non_development_manifest(
    tmp_path: Path, update: dict[str, object], message: str
) -> None:
    evaluator = _evaluator()
    paths = _write_fixture(tmp_path)
    manifest = json.loads(paths["categories"].read_text(encoding="utf-8"))
    manifest.update(update)
    paths["categories"].write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        evaluator.run_cli(
            [
                "--cohort-path",
                str(paths["cohort"]),
                "--baseline-predictions",
                str(paths["baseline"]),
                "--candidate-predictions",
                str(paths["candidate"]),
                "--category-manifest",
                str(paths["categories"]),
                "--output-path",
                str(paths["output"]),
            ]
        )
