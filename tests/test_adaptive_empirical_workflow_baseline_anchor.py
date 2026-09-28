from __future__ import annotations

import hashlib
import json
import inspect
import ast
from pathlib import Path

import pytest

from Benchmark.src.adaptive_empirical_workflow.baseline_anchor import (
    load_baseline_anchor_bundle,
    load_baseline_anchors,
    load_baseline_anchors_from_bytes,
)
from Benchmark.src.adaptive_empirical_workflow import baseline_anchor as anchor_module


TAXONOMY = {
    "symptom": ["Crash", "Incorrect Functionality"],
    "root_cause": ["API Misuse", "Incorrect Code Logic"],
}


def _valid_row(record_id: str) -> dict[str, object]:
    return {
        "record_id": record_id,
        "config_hash": "a" * 64,
        "final_prediction": {
            "symptom": "Crash",
            "root_cause": "Incorrect Code Logic",
        },
        "invalid": False,
    }


def _write(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_loads_explicit_subset_in_manifest_order_and_preserves_invalid_status(
    tmp_path: Path,
) -> None:
    invalid = {
        "record_id": "r2",
        "config_hash": "a" * 64,
        "final_prediction": {},
        "invalid": True,
    }
    source = _write(
        tmp_path / "baseline.jsonl",
        [_valid_row("extra"), invalid, _valid_row("r1")],
    )
    expected_sha = hashlib.sha256(source.read_bytes()).hexdigest()

    anchors = load_baseline_anchors(
        source,
        expected_record_ids=("r1", "r2"),
        taxonomy=TAXONOMY,
    )

    assert list(anchors) == ["r1", "r2"]
    assert anchors["r1"].valid is True
    assert anchors["r1"].symptom_label == "Crash"
    assert anchors["r1"].root_cause_label == "Incorrect Code Logic"
    assert anchors["r2"].valid is False
    assert anchors["r2"].symptom_label is None
    assert anchors["r2"].root_cause_label is None
    assert anchors["r1"].source_config_hash == "a" * 64
    assert anchors["r1"].source_predictions_sha256 == expected_sha
    assert anchors["r2"].source_predictions_sha256 == expected_sha


def test_path_and_from_bytes_baseline_loaders_are_equivalent(tmp_path: Path) -> None:
    source = _write(
        tmp_path / "baseline.jsonl",
        [_valid_row("extra"), _valid_row("r2"), _valid_row("r1")],
    )
    expected_ids = ("r1", "r2")

    anchors_from_path = load_baseline_anchors(
        source,
        expected_record_ids=expected_ids,
        taxonomy=TAXONOMY,
    )
    anchors_from_bytes = load_baseline_anchors_from_bytes(
        source.read_bytes(),
        expected_record_ids=expected_ids,
        taxonomy=TAXONOMY,
    )

    assert anchors_from_bytes == anchors_from_path


@pytest.mark.parametrize(
    ("rows", "expected_ids", "message"),
    (
        ([_valid_row("r1")], ("r1", "r2"), "missing expected record_ids"),
        ([_valid_row("r1"), _valid_row("r1")], ("r1",), "duplicate record_id"),
        (
            [
                {
                    **_valid_row("r1"),
                    "final_prediction": {
                        "symptom": "Unknown Symptom",
                        "root_cause": "Incorrect Code Logic",
                    },
                }
            ],
            ("r1",),
            "outside symptom taxonomy",
        ),
        (
            [
                _valid_row("r1"),
                {**_valid_row("r2"), "config_hash": "b" * 64},
            ],
            ("r1", "r2"),
            "single config_hash",
        ),
        (
            [{**_valid_row("r1"), "invalid": "false"}],
            ("r1",),
            "malformed invalid flag",
        ),
    ),
)
def test_loader_fails_closed_on_unbound_or_malformed_artifacts(
    tmp_path: Path,
    rows: list[dict[str, object]],
    expected_ids: tuple[str, ...],
    message: str,
) -> None:
    source = _write(tmp_path / "baseline.jsonl", rows)

    with pytest.raises(ValueError, match=message):
        load_baseline_anchors(
            source,
            expected_record_ids=expected_ids,
            taxonomy=TAXONOMY,
        )


def test_loader_rejects_empty_or_duplicate_expected_record_ids(tmp_path: Path) -> None:
    source = _write(tmp_path / "baseline.jsonl", [_valid_row("r1")])

    with pytest.raises(ValueError, match="expected_record_ids"):
        load_baseline_anchors(source, expected_record_ids=(), taxonomy=TAXONOMY)
    with pytest.raises(ValueError, match="expected_record_ids"):
        load_baseline_anchors(
            source,
            expected_record_ids=("r1", "r1"),
            taxonomy=TAXONOMY,
        )


def test_loader_reports_missing_artifact_as_a_validation_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot read Baseline predictions artifact"):
        load_baseline_anchors(
            tmp_path / "missing.jsonl",
            expected_record_ids=("r1",),
            taxonomy=TAXONOMY,
        )


def test_invalid_row_cannot_smuggle_a_candidate_prediction(tmp_path: Path) -> None:
    row = {**_valid_row("r1"), "invalid": True}
    source = _write(tmp_path / "baseline.jsonl", [row])

    with pytest.raises(ValueError, match="cannot contain final_prediction"):
        load_baseline_anchors(
            source,
            expected_record_ids=("r1",),
            taxonomy=TAXONOMY,
        )


@pytest.mark.parametrize(
    "contamination",
    (
        {"ground_truth": {"symptom": "Crash"}},
        {"audit": {"gold_label": "Crash"}},
        {"metadata": [{"label_answer": "Crash"}]},
        {"gt": {"symptom": "Crash"}},
        {"metadata": {"GT": "Crash"}},
    ),
)
def test_loader_rejects_ground_truth_fields_recursively(
    tmp_path: Path, contamination: dict[str, object]
) -> None:
    row = {**_valid_row("r1"), **contamination}
    source = _write(tmp_path / "baseline.jsonl", [row])

    with pytest.raises(ValueError, match="ground-truth field"):
        load_baseline_anchors(
            source,
            expected_record_ids=("r1",),
            taxonomy=TAXONOMY,
        )


def test_loaded_anchor_cannot_be_mutated_after_hash_binding(tmp_path: Path) -> None:
    source = _write(tmp_path / "baseline.jsonl", [_valid_row("r1")])
    anchor = load_baseline_anchors(
        source,
        expected_record_ids=("r1",),
        taxonomy=TAXONOMY,
    )["r1"]

    with pytest.raises(Exception, match="frozen"):
        anchor.symptom_label = "Build Failure"


def test_self_signed_bundle_is_rejected_even_when_internally_well_formed(
    tmp_path: Path,
) -> None:
    source = _write(
        tmp_path / "baseline.jsonl",
        [_valid_row("extra"), _valid_row("r2"), _valid_row("r1")],
    )

    self_signed = tmp_path / "trust.json"
    self_signed.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="trust manifest"):
        load_baseline_anchor_bundle(
            source,
            trust_manifest_path=self_signed,
            expected_domain="ase2022",
            expected_record_ids=("r1", "r2"),
            taxonomy=TAXONOMY,
        )


def test_production_loader_exposes_no_mutable_registry_or_registry_override() -> None:
    assert not hasattr(anchor_module, "_REGISTERED_BASELINE_TRUST_ROOTS")
    assert "registry" not in inspect.signature(load_baseline_anchor_bundle).parameters
    assert not hasattr(anchor_module, "_load_baseline_anchor_bundle_with_registry")
    assert not hasattr(anchor_module, "TrustedRegistry")
    tree = ast.parse(inspect.getsource(anchor_module))
    assert all(
        argument.arg not in {"registry", "trust_root"}
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        for argument in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
    )


def test_candidate_trust_root_contains_provenance_but_no_labels(tmp_path, monkeypatch) -> None:
    from Benchmark.scripts import prepare_baseline_trust_root as prepare
    monkeypatch.setattr(prepare, "ROOT", tmp_path)
    artifact = _write(tmp_path / "synthetic.jsonl", [_valid_row("r1")])
    output = tmp_path / "trust.json"
    prepare.register_baseline_trust_root(artifact, output=output, artifact_id="synthetic-test",
        domain="ase2022", baseline_code_sha256="b" * 64)
    manifest = json.loads(output.read_text(encoding="utf-8"))
    serialized = json.dumps(manifest).lower()
    assert manifest["status"] == "candidate"
    assert manifest["record_set"]["count"] == 1
    assert len(manifest["artifact"]["sha256"]) == 64
    assert len(manifest["baseline_run"]["config_sha256"]) == 64
    assert len(manifest["baseline_run"]["code_sha256"]) == 64
    assert all(label.lower() not in serialized for labels in TAXONOMY.values() for label in labels)
