from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from Benchmark.scripts.evaluate_targeted_sla_experiment import run_cli
from Benchmark.src.adaptive_empirical_workflow import offline_evaluation
from Benchmark.src.adaptive_empirical_workflow.baseline_anchor import (
    baseline_anchor_hash,
    load_baseline_anchors,
)
from Benchmark.src.adaptive_empirical_workflow.experiment_manifest import config_hash
from Benchmark.src.adaptive_empirical_workflow.offline_evaluation import (
    TARGETED_SLA_RECORD_IDS,
    evaluate_bound_targeted_sla_experiment,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _write_bound_inputs(
    tmp_path: Path,
    *,
    provider: str = "micu",
) -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    evaluation_manifest = (
        root / "Benchmark/configs/evaluation_trust/"
        "ase2022_contaminated_dev50_labels_v1.json"
    )
    evaluation = json.loads(evaluation_manifest.read_text(encoding="utf-8"))
    with (root / evaluation["label_source"]["relative_path"]).open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        gold_by_id = {row["record_id"]: row for row in csv.DictReader(handle)}
    taxonomy_path = (
        root / "Benchmark/inputs/ase2022_issue_only_holdout_seed20260806/"
        "ase2022_issue_only_holdout_taxonomy.json"
    )
    taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    baseline_config_hash = "a" * 64
    baseline_path = tmp_path / "baseline.jsonl"
    baseline_rows: list[dict[str, object]] = []
    for index, record_id in enumerate(TARGETED_SLA_RECORD_IDS):
        gold = gold_by_id[record_id]
        symptom = gold["symptom"]
        root_cause = gold["root_cause"]
        if index < 13:
            symptom = next(label for label in taxonomy["symptom"] if label != symptom)
            root_cause = next(
                label for label in taxonomy["root_cause"] if label != root_cause
            )
        baseline_rows.append(
            {
                "record_id": record_id,
                "invalid": False,
                "config_hash": baseline_config_hash,
                "final_prediction": {
                    "symptom": symptom,
                    "root_cause": root_cause,
                },
            }
        )
    baseline_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in baseline_rows),
        encoding="utf-8",
    )
    anchors = load_baseline_anchors(
        baseline_path,
        expected_record_ids=TARGETED_SLA_RECORD_IDS,
        taxonomy=taxonomy,
    )
    anchor_digests = {
        record_id: baseline_anchor_hash(anchors[record_id])
        for record_id in TARGETED_SLA_RECORD_IDS
    }
    baseline_binding = {
        "source_config_hash": baseline_config_hash,
        "source_predictions_sha256": _sha256(baseline_path),
        "selected_anchor_set_sha256": _canonical_sha256(
            [
                anchors[record_id].model_dump(mode="json")
                for record_id in TARGETED_SLA_RECORD_IDS
            ]
        ),
        "anchor_digest_map_sha256": _canonical_sha256(anchor_digests),
        "record_anchor_digests": anchor_digests,
    }
    resolved_config = {
        "architecture": "adaptive_empirical_expert_workflow",
        "domain": "ase2022",
        "stage": "stage3",
        "provider": provider,
        "execution_profile": "targeted-sla30",
        "development_only": True,
        "baseline_preservation_protocol": False,
        "record_ids": list(TARGETED_SLA_RECORD_IDS),
        "taxonomy_sha256": _sha256(taxonomy_path),
        "targeted_sla_baseline": baseline_binding,
        "code_revision": "c" * 40,
        "dirty_source_state_sha256": "d" * 64,
    }
    run_config_hash = config_hash(resolved_config)
    predictions_path = tmp_path / "predictions.jsonl"
    prediction_rows = [
        {
            "record_id": record_id,
            "config_hash": run_config_hash,
            "stage3_valid": True,
            "symptom_prediction": gold_by_id[record_id]["symptom"],
            "root_cause_prediction": gold_by_id[record_id]["root_cause"],
            "call_count": 2,
            "fallback": False,
            "audit": {
                "targeted_sla": {
                    "decision_status": {
                        "symptom": "model_verified",
                        "root_cause": "model_verified",
                    },
                    "call_audit": [],
                    "arbitrated": False,
                }
            },
        }
        for record_id in TARGETED_SLA_RECORD_IDS
    ]
    predictions_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in prediction_rows),
        encoding="utf-8",
    )
    run_manifest_path = tmp_path / "run_manifest.json"
    run_manifest_path.write_text(
        json.dumps(
            {
                **resolved_config,
                "config_hash": run_config_hash,
                "record_count": 14,
                "taxonomy_artifact_id": taxonomy_path.relative_to(root).as_posix(),
                "targeted_sla_baseline": baseline_binding,
                "resolved_config": resolved_config,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "evaluation_manifest": evaluation_manifest,
        "baseline_path": baseline_path,
        "baseline_sha256": _sha256(baseline_path),
        "baseline_config_hash": baseline_config_hash,
        "predictions_path": predictions_path,
        "predictions_sha256": _sha256(predictions_path),
        "run_manifest_path": run_manifest_path,
        "run_manifest_sha256": _sha256(run_manifest_path),
        "run_config_hash": run_config_hash,
        "taxonomy_path": taxonomy_path,
    }


def _evaluate(inputs: dict[str, object]) -> dict[str, object]:
    return evaluate_bound_targeted_sla_experiment(
        inputs["predictions_path"],
        expected_predictions_sha256=str(inputs["predictions_sha256"]),
        run_manifest_path=inputs["run_manifest_path"],
        expected_run_manifest_sha256=str(inputs["run_manifest_sha256"]),
        baseline_anchor_path=inputs["baseline_path"],
        expected_baseline_sha256=str(inputs["baseline_sha256"]),
        expected_baseline_config_hash=str(inputs["baseline_config_hash"]),
        evaluation_manifest_path=inputs["evaluation_manifest"],
    )


def test_bound_targeted_sla_evaluation_accepts_gemini_provider(
    tmp_path: Path,
) -> None:
    inputs = _write_bound_inputs(tmp_path, provider="gemini")

    result = _evaluate(inputs)

    assert result["provider"] == "gemini"


def _physical_bytes(path: Path) -> bytes:
    with path.open("rb") as handle:
        return handle.read()


def _simulate_replacement_after_first_read(
    monkeypatch: pytest.MonkeyPatch,
    *,
    target: Path,
    replacement: bytes,
) -> None:
    original_read_bytes = Path.read_bytes
    original_read_text = Path.read_text
    original = original_read_bytes(target)
    target_reads = 0
    resolved_target = target.resolve()

    def next_target_bytes() -> bytes:
        nonlocal target_reads
        raw = original if target_reads == 0 else replacement
        target_reads += 1
        return raw

    def replacing_read_bytes(path: Path) -> bytes:
        if path.resolve() == resolved_target:
            return next_target_bytes()
        return original_read_bytes(path)

    def replacing_read_text(
        path: Path,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> str:
        if path.resolve() == resolved_target:
            return next_target_bytes().decode(encoding or "utf-8", errors or "strict")
        return original_read_text(
            path,
            encoding=encoding,
            errors=errors,
            newline=newline,
        )

    monkeypatch.setattr(Path, "read_bytes", replacing_read_bytes)
    monkeypatch.setattr(Path, "read_text", replacing_read_text)


def test_prediction_replacement_cannot_change_metrics_behind_original_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _write_bound_inputs(tmp_path)
    predictions_path = inputs["predictions_path"]
    taxonomy_path = inputs["taxonomy_path"]
    assert isinstance(predictions_path, Path)
    assert isinstance(taxonomy_path, Path)
    rows = [
        json.loads(line)
        for line in predictions_path.read_text(encoding="utf-8").splitlines()
    ]
    taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    rows[0]["symptom_prediction"] = next(
        label for label in taxonomy["symptom"] if label != rows[0]["symptom_prediction"]
    )
    replacement = "".join(
        json.dumps(row, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")
    original_sha = str(inputs["predictions_sha256"])
    original_physical = _physical_bytes(predictions_path)
    _simulate_replacement_after_first_read(
        monkeypatch,
        target=predictions_path,
        replacement=replacement,
    )

    artifact = _evaluate(inputs)

    assert artifact["bindings"]["predictions"]["sha256"] == original_sha  # type: ignore[index]
    assert artifact["targeted_sla"]["recovery"] == 13  # type: ignore[index]
    assert _physical_bytes(predictions_path) == original_physical


def test_taxonomy_replacement_probe_never_writes_tracked_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _write_bound_inputs(tmp_path)
    taxonomy_path = inputs["taxonomy_path"]
    assert isinstance(taxonomy_path, Path)
    original_physical = _physical_bytes(taxonomy_path)
    original_write_bytes = Path.write_bytes

    def reject_taxonomy_write(path: Path, data: bytes) -> int:
        if path.resolve() == taxonomy_path.resolve():
            raise AssertionError("replacement probe wrote the tracked taxonomy")
        return original_write_bytes(path, data)

    monkeypatch.setattr(Path, "write_bytes", reject_taxonomy_write)
    _simulate_replacement_after_first_read(
        monkeypatch,
        target=taxonomy_path,
        replacement=b"{}\n",
    )

    artifact = _evaluate(inputs)

    assert artifact["evaluation_status"] == "evaluated"
    assert _physical_bytes(taxonomy_path) == original_physical


@pytest.mark.parametrize(
    ("input_key", "replacement"),
    (
        ("run_manifest_path", b"{}\n"),
        ("baseline_path", b"\n"),
        ("taxonomy_path", b"{}\n"),
    ),
)
def test_bound_input_replacement_uses_original_immutable_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    input_key: str,
    replacement: bytes,
) -> None:
    inputs = _write_bound_inputs(tmp_path)
    target = inputs[input_key]
    assert isinstance(target, Path)
    original_physical = _physical_bytes(target)
    _simulate_replacement_after_first_read(
        monkeypatch, target=target, replacement=replacement
    )

    artifact = _evaluate(inputs)

    assert artifact["evaluation_status"] == "evaluated"
    assert artifact["targeted_sla"]["recovery"] == 13  # type: ignore[index]
    assert _physical_bytes(target) == original_physical


def test_targeted_evaluator_reads_every_bound_input_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _write_bound_inputs(tmp_path)
    root = Path(offline_evaluation.__file__).resolve().parents[3]
    evaluation_manifest = inputs["evaluation_manifest"]
    assert isinstance(evaluation_manifest, Path)
    manifest = json.loads(evaluation_manifest.read_text(encoding="utf-8"))
    bound_paths = {
        Path(inputs["predictions_path"]).resolve(),
        Path(inputs["run_manifest_path"]).resolve(),
        Path(inputs["baseline_path"]).resolve(),
        Path(inputs["taxonomy_path"]).resolve(),
        evaluation_manifest.resolve(),
        (root / str(manifest["label_source"]["relative_path"])).resolve(),
        (root / str(manifest["selection"]["split_manifest_relative_path"])).resolve(),
    }
    original_read_bytes = Path.read_bytes
    read_counts = {path: 0 for path in bound_paths}

    def counting_read_bytes(path: Path) -> bytes:
        resolved = path.resolve()
        if resolved in read_counts:
            read_counts[resolved] += 1
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", counting_read_bytes)

    artifact = _evaluate(inputs)

    assert artifact["evaluation_status"] == "evaluated"
    assert read_counts == {path: 1 for path in bound_paths}


def test_targeted_offline_cli_writes_aggregate_bound_evaluation_artifact(
    tmp_path: Path,
) -> None:
    inputs = _write_bound_inputs(tmp_path)
    output = tmp_path / "targeted-evaluation.json"

    artifact = run_cli(
        [
            "--predictions-path",
            str(inputs["predictions_path"]),
            "--predictions-sha256",
            str(inputs["predictions_sha256"]),
            "--run-manifest-path",
            str(inputs["run_manifest_path"]),
            "--run-manifest-sha256",
            _sha256(inputs["run_manifest_path"]),  # type: ignore[arg-type]
            "--baseline-anchor-path",
            str(inputs["baseline_path"]),
            "--baseline-sha256",
            _sha256(inputs["baseline_path"]),  # type: ignore[arg-type]
            "--baseline-config-hash",
            str(inputs["baseline_config_hash"]),
            "--evaluation-manifest",
            str(inputs["evaluation_manifest"]),
            "--output-path",
            str(output),
        ]
    )

    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert artifact["evaluation_status"] == "evaluated"
    assert artifact["record_count"] == 14
    assert artifact["targeted_sla"]["recovery"] == 13
    assert artifact["targeted_sla"]["harm"] == 0
    assert artifact["bindings"]["predictions"]["sha256"] == inputs["predictions_sha256"]
    assert (
        artifact["bindings"]["run_manifest"]["config_hash"] == inputs["run_config_hash"]
    )
    assert (
        artifact["bindings"]["baseline"]["source_config_hash"]
        == inputs["baseline_config_hash"]
    )
    assert artifact["bindings"]["gold"]["label_source_sha256"] == (
        "c15807393140f4c9b46a1e877c13d459bac44f07eb3c120f2d5604b752c928da"
    )
    assert "Crash" not in json.dumps(artifact)


def test_targeted_offline_evaluation_rejects_content_and_config_drift(
    tmp_path: Path,
) -> None:
    inputs = _write_bound_inputs(tmp_path)

    with pytest.raises(ValueError, match="prediction artifact hash mismatch"):
        evaluate_bound_targeted_sla_experiment(
            inputs["predictions_path"],
            expected_predictions_sha256="0" * 64,
            run_manifest_path=inputs["run_manifest_path"],
            expected_run_manifest_sha256=_sha256(inputs["run_manifest_path"]),  # type: ignore[arg-type]
            baseline_anchor_path=inputs["baseline_path"],
            expected_baseline_sha256=_sha256(inputs["baseline_path"]),  # type: ignore[arg-type]
            expected_baseline_config_hash=str(inputs["baseline_config_hash"]),
            evaluation_manifest_path=inputs["evaluation_manifest"],
        )

    with pytest.raises(ValueError, match="run manifest hash mismatch"):
        evaluate_bound_targeted_sla_experiment(
            inputs["predictions_path"],
            expected_predictions_sha256=str(inputs["predictions_sha256"]),
            run_manifest_path=inputs["run_manifest_path"],
            expected_run_manifest_sha256="0" * 64,
            baseline_anchor_path=inputs["baseline_path"],
            expected_baseline_sha256=_sha256(inputs["baseline_path"]),  # type: ignore[arg-type]
            expected_baseline_config_hash=str(inputs["baseline_config_hash"]),
            evaluation_manifest_path=inputs["evaluation_manifest"],
        )

    predictions_path = inputs["predictions_path"]
    assert isinstance(predictions_path, Path)
    rows = [
        json.loads(line)
        for line in predictions_path.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["config_hash"] = "f" * 64
    predictions_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="prediction config_hash mismatch"):
        evaluate_bound_targeted_sla_experiment(
            predictions_path,
            expected_predictions_sha256=_sha256(predictions_path),
            run_manifest_path=inputs["run_manifest_path"],
            expected_run_manifest_sha256=_sha256(inputs["run_manifest_path"]),  # type: ignore[arg-type]
            baseline_anchor_path=inputs["baseline_path"],
            expected_baseline_sha256=_sha256(inputs["baseline_path"]),  # type: ignore[arg-type]
            expected_baseline_config_hash=str(inputs["baseline_config_hash"]),
            evaluation_manifest_path=inputs["evaluation_manifest"],
        )
