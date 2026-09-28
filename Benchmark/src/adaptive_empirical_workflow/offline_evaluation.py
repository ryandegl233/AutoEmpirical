"""Offline-only evaluation for evidence-only Baseline-preservation runs."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .baseline_anchor import baseline_anchor_hash, load_baseline_anchors_from_bytes
from .evaluation import stage3_team_diagnostics, targeted_sla_diagnostics
from .experiment import evaluate_experiment
from .experiment_manifest import config_hash
from .splits import load_restricted_gold_for_evaluation


TARGETED_SLA_RECORD_IDS = (
    "ase2022_towards_understanding_the_faults_of:2a5208a3df4eec74",
    "ase2022_towards_understanding_the_faults_of:32ae23d9727d5093",
    "ase2022_towards_understanding_the_faults_of:40bb6dce12037e19",
    "ase2022_towards_understanding_the_faults_of:5386d681244ca925",
    "ase2022_towards_understanding_the_faults_of:53bd1ca154a158aa",
    "ase2022_towards_understanding_the_faults_of:5a54b197ecd4d068",
    "ase2022_towards_understanding_the_faults_of:5b1cffb974ddbbb6",
    "ase2022_towards_understanding_the_faults_of:622611082d25a6f1",
    "ase2022_towards_understanding_the_faults_of:73210f3efed8b6c4",
    "ase2022_towards_understanding_the_faults_of:81e596760174916a",
    "ase2022_towards_understanding_the_faults_of:8360c81d83d42fc1",
    "ase2022_towards_understanding_the_faults_of:9c9f1bf59a72d6fd",
    "ase2022_towards_understanding_the_faults_of:bfc31c239d338946",
    "ase2022_towards_understanding_the_faults_of:cb2aaf9e7123b878",
)

_REGISTERED_DEVELOPMENT_EVALUATION_SHA256 = (
    "e8bd5d3566d2c2991497c71147f125a78b8963a93a2bbd945bb400ae82815086"
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class _ArtifactSnapshot:
    path: Path
    raw: bytes
    sha256: str


def _read_snapshot(path: Path, label: str) -> _ArtifactSnapshot:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read {label}") from error
    return _ArtifactSnapshot(
        path=path,
        raw=raw,
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def _jsonl_from_bytes(raw: bytes) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("cannot read prediction artifact") from error
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            raise ValueError(f"invalid prediction JSONL row {line_number}") from None
        if not isinstance(row, dict):
            raise ValueError(f"prediction row {line_number} must be an object")
        rows.append(row)
    return rows


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    return _jsonl_from_bytes(_read_snapshot(path, "prediction artifact").raw)


def _repository_relative(path: Path, *, root: Path, label: str) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError(f"{label} path escapes repository") from error


def _load_registered_targeted_gold(
    evaluation_manifest_path: Path,
) -> tuple[list[dict[str, str]], dict[str, object]]:
    root = Path(__file__).resolve().parents[3]
    registered_path = (
        root / "Benchmark/configs/evaluation_trust/"
        "ase2022_contaminated_dev50_labels_v1.json"
    ).resolve()
    if evaluation_manifest_path.resolve() != registered_path:
        raise ValueError("evaluation manifest is not a pre-registered trust root")
    manifest_snapshot = _read_snapshot(
        evaluation_manifest_path, "evaluation trust manifest"
    )
    manifest = _json_object_from_bytes(
        manifest_snapshot.raw, label="evaluation trust manifest"
    )
    if _canonical_sha256(manifest) != _REGISTERED_DEVELOPMENT_EVALUATION_SHA256:
        raise ValueError("evaluation trust manifest hash mismatch")
    if not isinstance(manifest, dict):
        raise ValueError("evaluation trust manifest schema mismatch")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("status") != "active"
        or manifest.get("domain") != "ase2022"
        or manifest.get("split_kind") != "contaminated_development"
        or manifest.get("access_policy")
        != {
            "runner_may_load": False,
            "offline_evaluator_only": True,
            "emit_aggregate_only": True,
        }
    ):
        raise ValueError("evaluation trust manifest is not an active offline policy")
    label = manifest.get("label_source")
    selection = manifest.get("selection")
    if not isinstance(label, dict) or not isinstance(selection, dict):
        raise ValueError("evaluation trust bindings are malformed")
    label_path = (root / str(label.get("relative_path"))).resolve()
    split_path = (root / str(selection.get("split_manifest_relative_path"))).resolve()
    _repository_relative(label_path, root=root, label="label source")
    _repository_relative(split_path, root=root, label="split manifest")
    label_snapshot = _read_snapshot(label_path, "trusted label source")
    split_snapshot = _read_snapshot(split_path, "trusted split manifest")
    split_manifest = _json_object_from_bytes(
        split_snapshot.raw, label="trusted split manifest"
    )
    if (
        label.get("format") != "csv"
        or label_snapshot.sha256 != label.get("sha256")
        or _canonical_sha256(split_manifest) != selection.get("split_manifest_sha256")
    ):
        raise ValueError("offline evaluation input digest mismatch")
    try:
        label_rows = list(
            csv.DictReader(label_snapshot.raw.decode("utf-8-sig").splitlines())
        )
    except UnicodeDecodeError as error:
        raise ValueError("trusted label source is not UTF-8 CSV") from error
    label_ids = [str(row.get("record_id") or "") for row in label_rows]
    split = split_manifest.get("splits", {}).get(selection.get("split_name"))
    development_ids = (
        tuple(split.get("record_ids", ())) if isinstance(split, dict) else ()
    )
    targeted_set = frozenset(TARGETED_SLA_RECORD_IDS)
    if (
        len(label_rows) != label.get("record_count")
        or len(label_ids) != len(set(label_ids))
        or _canonical_sha256(sorted(label_ids)) != label.get("record_ids_sha256")
        or len(development_ids) != selection.get("record_count")
        or _canonical_sha256(list(development_ids))
        != selection.get("record_ids_sha256")
        or not targeted_set.issubset(development_ids)
    ):
        raise ValueError("trusted targeted evaluation record set mismatch")
    indexed = {str(row["record_id"]): row for row in label_rows}
    gold = [
        {
            "record_id": record_id,
            "decision": str(indexed[record_id]["decision"]),
            "symptom": str(indexed[record_id]["symptom"]),
            "root_cause": str(indexed[record_id]["root_cause"]),
        }
        for record_id in TARGETED_SLA_RECORD_IDS
    ]
    bindings: dict[str, object] = {
        "evaluation_manifest_relative_path": _repository_relative(
            evaluation_manifest_path, root=root, label="evaluation manifest"
        ),
        "evaluation_manifest_sha256": _REGISTERED_DEVELOPMENT_EVALUATION_SHA256,
        "artifact_id": str(manifest.get("artifact_id")),
        "label_source_relative_path": str(label.get("relative_path")),
        "label_source_sha256": str(label.get("sha256")),
        "selection_split_manifest_relative_path": str(
            selection.get("split_manifest_relative_path")
        ),
        "selection_split_manifest_sha256": str(selection.get("split_manifest_sha256")),
    }
    return gold, bindings


def _json_object_from_bytes(raw: bytes, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _require_sha256(value: str, *, label: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


def evaluate_bound_targeted_sla_experiment(
    predictions_path: str | Path,
    *,
    expected_predictions_sha256: str,
    run_manifest_path: str | Path,
    expected_run_manifest_sha256: str,
    baseline_anchor_path: str | Path,
    expected_baseline_sha256: str,
    expected_baseline_config_hash: str,
    evaluation_manifest_path: str | Path,
) -> dict[str, Any]:
    """Evaluate the fixed 14-record SLA run with all inputs content-bound."""

    _require_sha256(expected_predictions_sha256, label="prediction artifact hash")
    _require_sha256(expected_run_manifest_sha256, label="run manifest hash")
    _require_sha256(expected_baseline_sha256, label="Baseline artifact hash")
    _require_sha256(expected_baseline_config_hash, label="Baseline config hash")
    root = Path(__file__).resolve().parents[3].resolve()
    prediction_source = Path(predictions_path)
    baseline_source = Path(baseline_anchor_path)
    run_manifest_source = Path(run_manifest_path)
    prediction_snapshot = _read_snapshot(prediction_source, "prediction artifact")
    run_manifest_snapshot = _read_snapshot(run_manifest_source, "run manifest")
    baseline_snapshot = _read_snapshot(baseline_source, "Baseline artifact")
    if prediction_snapshot.sha256 != expected_predictions_sha256:
        raise ValueError("prediction artifact hash mismatch")
    actual_run_manifest_sha256 = run_manifest_snapshot.sha256
    if actual_run_manifest_sha256 != expected_run_manifest_sha256:
        raise ValueError("run manifest hash mismatch")
    if baseline_snapshot.sha256 != expected_baseline_sha256:
        raise ValueError("Baseline artifact hash mismatch")

    run_manifest = _json_object_from_bytes(
        run_manifest_snapshot.raw, label="run manifest"
    )
    resolved_config = run_manifest.get("resolved_config")
    if not isinstance(resolved_config, dict):
        raise ValueError("run manifest resolved_config is malformed")
    run_config_hash = run_manifest.get("config_hash")
    if (
        not isinstance(run_config_hash, str)
        or config_hash(resolved_config) != run_config_hash
    ):
        raise ValueError("run manifest config_hash mismatch")
    record_ids = tuple(str(value) for value in resolved_config.get("record_ids", ()))
    if (
        len(record_ids) != len(TARGETED_SLA_RECORD_IDS)
        or len(record_ids) != len(set(record_ids))
        or frozenset(record_ids) != frozenset(TARGETED_SLA_RECORD_IDS)
        or run_manifest.get("record_count") != len(record_ids)
    ):
        raise ValueError("run manifest must bind the exact targeted SLA record set")
    provider = resolved_config.get("provider")
    if provider not in {"micu", "gemini"}:
        raise ValueError("run manifest provider is not supported for targeted SLA")
    expected_profile = {
        "architecture": "adaptive_empirical_expert_workflow",
        "domain": "ase2022",
        "stage": "stage3",
        "execution_profile": "targeted-sla30",
        "development_only": True,
        "baseline_preservation_protocol": False,
    }
    if any(
        resolved_config.get(key) != value for key, value in expected_profile.items()
    ):
        raise ValueError("run manifest is not a targeted-sla30 development run")
    for key in (
        *expected_profile,
        "provider",
        "record_ids",
        "taxonomy_sha256",
        "targeted_sla_baseline",
    ):
        if run_manifest.get(key) != resolved_config.get(key):
            raise ValueError(f"run manifest top-level {key} binding mismatch")

    taxonomy_artifact_id = run_manifest.get("taxonomy_artifact_id")
    if not isinstance(taxonomy_artifact_id, str) or not taxonomy_artifact_id:
        raise ValueError("run manifest taxonomy artifact binding is missing")
    taxonomy_path = (root / taxonomy_artifact_id).resolve()
    taxonomy_relative_path = _repository_relative(
        taxonomy_path, root=root, label="taxonomy artifact"
    )
    taxonomy_snapshot = _read_snapshot(taxonomy_path, "taxonomy artifact")
    taxonomy_sha256 = taxonomy_snapshot.sha256
    if taxonomy_sha256 != resolved_config.get("taxonomy_sha256"):
        raise ValueError("taxonomy artifact hash mismatch")
    taxonomy = _json_object_from_bytes(taxonomy_snapshot.raw, label="taxonomy artifact")

    predictions = _jsonl_from_bytes(prediction_snapshot.raw)
    prediction_ids = tuple(str(row.get("record_id") or "") for row in predictions)
    if prediction_ids != record_ids:
        raise ValueError("prediction IDs must exactly match run manifest order")
    for row in predictions:
        if row.get("config_hash") != run_config_hash:
            raise ValueError("prediction config_hash mismatch")

    anchors = load_baseline_anchors_from_bytes(
        baseline_snapshot.raw,
        expected_record_ids=record_ids,
        taxonomy=taxonomy,
    )
    first_anchor = anchors[record_ids[0]]
    if first_anchor.source_config_hash != expected_baseline_config_hash:
        raise ValueError("Baseline config hash mismatch")
    anchor_digests = {
        record_id: baseline_anchor_hash(anchors[record_id]) for record_id in record_ids
    }
    expected_baseline_binding = {
        "source_config_hash": expected_baseline_config_hash,
        "source_predictions_sha256": expected_baseline_sha256,
        "selected_anchor_set_sha256": _canonical_sha256(
            [anchors[record_id].model_dump(mode="json") for record_id in record_ids]
        ),
        "anchor_digest_map_sha256": _canonical_sha256(anchor_digests),
        "record_anchor_digests": anchor_digests,
    }
    baseline_binding = resolved_config.get("targeted_sla_baseline")
    if not isinstance(baseline_binding, dict) or any(
        baseline_binding.get(key) != value
        for key, value in expected_baseline_binding.items()
    ):
        raise ValueError("run manifest Baseline binding mismatch")

    registered_gold, gold_binding = _load_registered_targeted_gold(
        Path(evaluation_manifest_path)
    )
    gold_by_id = {row["record_id"]: row for row in registered_gold}
    ordered_gold = [gold_by_id[record_id] for record_id in record_ids]
    diagnostics = targeted_sla_diagnostics(
        ordered_gold,
        predictions,
        dict(anchors),
    )
    if diagnostics.get("evaluation_status") != "evaluated":
        raise ValueError("targeted SLA diagnostics did not evaluate")

    code_revision = resolved_config.get("code_revision")
    dirty_source_state = resolved_config.get("dirty_source_state_sha256")
    if not isinstance(code_revision, str) or not code_revision:
        raise ValueError("run manifest source revision binding is missing")
    if not isinstance(dirty_source_state, str):
        raise ValueError("run manifest source-state binding is missing")
    _require_sha256(dirty_source_state, label="run source-state hash")
    return {
        "schema_version": 1,
        "evaluation_status": "evaluated",
        "provider": provider,
        "record_count": len(record_ids),
        "record_ids_sha256": _canonical_sha256(sorted(record_ids)),
        "record_order_sha256": _canonical_sha256(list(record_ids)),
        "bindings": {
            "predictions": {
                "sha256": expected_predictions_sha256,
                "config_hash": run_config_hash,
            },
            "run_manifest": {
                "sha256": actual_run_manifest_sha256,
                "config_hash": run_config_hash,
                "code_revision": code_revision,
                "dirty_source_state_sha256": dirty_source_state,
            },
            "baseline": expected_baseline_binding,
            "gold": gold_binding,
            "taxonomy": {
                "relative_path": taxonomy_relative_path,
                "sha256": taxonomy_sha256,
            },
        },
        "targeted_sla": diagnostics,
    }


def evaluate_registered_baseline_preservation_experiment(
    predictions_path: str | Path,
    *,
    evaluation_manifest_path: str | Path,
) -> dict[str, Any]:
    """Evaluate after execution using a separately trusted label source."""

    manifest_path = Path(evaluation_manifest_path)
    root = Path(__file__).resolve().parents[3]
    registered_path = (
        root / "Benchmark/configs/evaluation_trust/"
        "ase2022_contaminated_dev50_labels_v1.json"
    ).resolve()
    expected_manifest_sha = (
        "e8bd5d3566d2c2991497c71147f125a78b8963a93a2bbd945bb400ae82815086"
    )
    if manifest_path.resolve() != registered_path:
        raise ValueError("evaluation manifest is not a pre-registered trust root")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("cannot read evaluation trust manifest") from error
    if _canonical_sha256(manifest) != expected_manifest_sha:
        raise ValueError("evaluation trust manifest hash mismatch")
    required = {
        "schema_version",
        "status",
        "artifact_id",
        "domain",
        "split_kind",
        "label_source",
        "selection",
        "access_policy",
    }
    if not isinstance(manifest, dict) or set(manifest) != required:
        raise ValueError("evaluation trust manifest schema mismatch")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("status") != "active"
        or manifest.get("domain") != "ase2022"
        or manifest.get("split_kind") != "contaminated_development"
        or manifest.get("access_policy")
        != {
            "runner_may_load": False,
            "offline_evaluator_only": True,
            "emit_aggregate_only": True,
        }
    ):
        raise ValueError("evaluation trust manifest is not an active offline policy")
    label = manifest.get("label_source")
    selection = manifest.get("selection")
    if not isinstance(label, dict) or not isinstance(selection, dict):
        raise ValueError("evaluation trust bindings are malformed")
    root = root.resolve()
    label_path = (root / str(label.get("relative_path"))).resolve()
    split_path = (root / str(selection.get("split_manifest_relative_path"))).resolve()
    for path in (label_path, split_path):
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError("evaluation trust path escapes repository") from error
    try:
        label_raw = label_path.read_bytes()
        split_manifest = json.loads(split_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("cannot read trusted offline evaluation inputs") from error
    if (
        label.get("format") != "csv"
        or hashlib.sha256(label_raw).hexdigest() != label.get("sha256")
        or _canonical_sha256(split_manifest) != selection.get("split_manifest_sha256")
    ):
        raise ValueError("offline evaluation input digest mismatch")
    try:
        label_rows = list(csv.DictReader(label_raw.decode("utf-8-sig").splitlines()))
    except UnicodeDecodeError as error:
        raise ValueError("trusted label source is not UTF-8 CSV") from error
    label_ids = [str(row.get("record_id") or "") for row in label_rows]
    if (
        len(label_rows) != label.get("record_count")
        or len(label_ids) != len(set(label_ids))
        or _canonical_sha256(sorted(label_ids)) != label.get("record_ids_sha256")
    ):
        raise ValueError("trusted label record set mismatch")
    split_name = selection.get("split_name")
    split = split_manifest.get("splits", {}).get(split_name)
    selected_ids = tuple(split.get("record_ids", ())) if isinstance(split, dict) else ()
    if (
        split_manifest.get("split_kind") != "contaminated_development"
        or split_manifest.get("label_access_policy", {}).get("runner_may_load_labels")
        is not False
        or len(selected_ids) != selection.get("record_count")
        or _canonical_sha256(list(selected_ids)) != selection.get("record_ids_sha256")
        or set(selected_ids) - set(label_ids)
    ):
        raise ValueError("trusted evaluation selection mismatch")
    predictions = _load_jsonl(Path(predictions_path))
    prediction_ids = [str(row.get("record_id") or "") for row in predictions]
    if tuple(prediction_ids) != selected_ids:
        raise ValueError("prediction IDs must exactly match trusted evaluation order")
    indexed = {str(row["record_id"]): row for row in label_rows}
    gold = [
        {
            "record_id": record_id,
            "decision": str(indexed[record_id]["decision"]),
            "symptom": str(indexed[record_id]["symptom"]),
            "root_cause": str(indexed[record_id]["root_cause"]),
        }
        for record_id in selected_ids
    ]
    metrics = evaluate_experiment(gold, predictions, stage="stage3")
    metrics["evaluation"] = {
        "status": "evaluated_offline",
        "artifact_id": str(manifest["artifact_id"]),
        "fixed_denominator": len(selected_ids),
        "evaluation_manifest_sha256": expected_manifest_sha,
    }
    metrics["stage3"]["dual_team"] = stage3_team_diagnostics(gold, predictions)
    return metrics


def evaluate_registered_restricted_holdout(
    predictions_path: str | Path,
    *,
    split_manifest_path: str | Path,
    restricted_gold_path: str | Path,
    split_name: str,
) -> dict[str, Any]:
    """Evaluate validation/final predictions without returning record labels."""

    if split_name not in {"validation", "final"}:
        raise ValueError("restricted evaluation split must be validation or final")
    root = Path(__file__).resolve().parents[3]
    registered = (
        root / "Benchmark/configs/splits/"
        "ase2022_stage3_uncontaminated_seed20260816_revision4/split_manifest.json"
    ).resolve()
    supplied = Path(split_manifest_path).resolve()
    if supplied != registered:
        raise ValueError("split manifest is not the registered revision4 holdout")
    manifest = json.loads(supplied.read_text(encoding="utf-8"))
    if _canonical_sha256(manifest) != (
        "7bdbc734d724e2f5c6c8ecba7468e7e80a755752cf68d354aada05fb286fd835"
    ):
        raise ValueError("registered holdout split manifest hash mismatch")
    policy = manifest.get("label_access_policy", {})
    if (
        manifest.get("status") != "active"
        or policy.get("runner_may_load_gold") is not False
        or policy.get("evaluation_requires_separate_explicit_gold_artifact") is not True
    ):
        raise ValueError("registered holdout evaluation policy mismatch")
    gold_rows = load_restricted_gold_for_evaluation(
        Path(restricted_gold_path),
        expected_sha256=str(policy.get("restricted_gold_sha256") or ""),
    )
    gold_by_id = {
        row["record_id"]: row for row in gold_rows if row["split"] == split_name
    }
    selected = manifest.get("splits", {}).get(split_name, {})
    selected_ids = tuple(selected.get("record_ids", ()))
    if set(gold_by_id) != set(selected_ids):
        raise ValueError("restricted gold IDs do not match registered split")
    predictions = _load_jsonl(Path(predictions_path))
    if tuple(str(row.get("record_id") or "") for row in predictions) != selected_ids:
        raise ValueError("prediction IDs must exactly match registered split order")
    records = [
        {
            "record_id": record_id,
            "decision": "accepted_fault",
            "symptom": gold_by_id[record_id]["symptom"],
            "root_cause": gold_by_id[record_id]["root_cause"],
        }
        for record_id in selected_ids
    ]
    metrics = evaluate_experiment(records, predictions, stage="stage3")
    metrics["stage3"]["dual_team"] = stage3_team_diagnostics(records, predictions)
    metrics["evaluation"] = {
        "status": "evaluated_offline_restricted",
        "split": split_name,
        "fixed_denominator": len(selected_ids),
        "split_manifest_sha256": _canonical_sha256(manifest),
        "gold_security_policy": "owner_only_acl_v2",
    }
    return metrics
