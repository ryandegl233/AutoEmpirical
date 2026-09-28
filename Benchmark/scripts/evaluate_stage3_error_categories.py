from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Benchmark.src.adaptive_empirical_workflow.evaluation import (
    paired_stage3_comparison,
    stage3_error_category_diagnostics,
)


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _record_ids_sha256(record_ids: Sequence[str]) -> str:
    payload = "".join(f"{record_id}\n" for record_id in sorted(record_ids))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_cohort(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        records = [dict(row) for row in csv.DictReader(handle)]
    required = {"record_id", "symptom", "root_cause"}
    if any(not required.issubset(record) for record in records):
        raise ValueError("cohort must contain record_id, symptom, and root_cause")
    return records


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSONL row {line_number}: {path}") from error
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row {line_number} must be an object: {path}")
        rows.append(row)
    return rows


def _normalize_prediction(row: dict[str, object]) -> dict[str, object]:
    record_id = row.get("record_id")
    if not isinstance(record_id, str) or not record_id:
        raise ValueError("prediction row is missing record_id")
    if "final_prediction" in row or "invalid" in row:
        final = row.get("final_prediction")
        invalid = row.get("invalid")
        if type(invalid) is not bool:
            raise ValueError(f"baseline row {record_id} has malformed invalid flag")
        if final is not None and not isinstance(final, dict):
            raise ValueError(f"baseline row {record_id} has malformed final_prediction")
        symptom = final.get("symptom") if isinstance(final, dict) else None
        root_cause = final.get("root_cause") if isinstance(final, dict) else None
        valid = not invalid and isinstance(symptom, str) and isinstance(root_cause, str)
        return {
            "record_id": record_id,
            "stage3_valid": valid,
            "symptom_prediction": symptom if valid else None,
            "root_cause_prediction": root_cause if valid else None,
            "audit": row.get("audit"),
        }
    valid = row.get("stage3_valid")
    if type(valid) is not bool:
        raise ValueError(f"prediction row {record_id} has malformed stage3_valid")
    symptom = row.get("symptom_prediction")
    root_cause = row.get("root_cause_prediction")
    if valid and (not isinstance(symptom, str) or not isinstance(root_cause, str)):
        raise ValueError(f"valid prediction row {record_id} has empty labels")
    return {
        "record_id": record_id,
        "stage3_valid": valid,
        "symptom_prediction": symptom,
        "root_cause_prediction": root_cause,
        "audit": row.get("audit"),
    }


def _select_prediction_rows(
    rows: Sequence[dict[str, object]],
    record_ids: Sequence[str],
    *,
    label: str,
) -> list[dict[str, object]]:
    by_id: dict[str, dict[str, object]] = {}
    for row in rows:
        record_id = row["record_id"]
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"{label} prediction row is missing record_id")
        if record_id in by_id:
            raise ValueError(f"duplicate {label} prediction record_id: {record_id}")
        by_id[record_id] = row
    missing_ids = set(record_ids) - set(by_id)
    if missing_ids:
        raise ValueError(
            f"{label} predictions missing manifest record_ids: "
            + ", ".join(sorted(missing_ids))
        )
    return [by_id[record_id] for record_id in record_ids]


def _load_category_manifest(
    path: Path, records: Sequence[dict[str, str]]
) -> tuple[dict[str, object], list[dict[str, str]]]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid category manifest: {path}") from error
    if not isinstance(manifest, dict):
        raise ValueError("category manifest must be an object")
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported category manifest schema")
    if manifest.get("split_kind") != "contaminated_development":
        raise ValueError(
            "error-category tuning manifest must use contaminated_development"
        )
    dataset_id = manifest.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id:
        raise ValueError("category manifest dataset_id must be non-empty")
    record_ids = manifest.get("record_ids")
    if (
        not isinstance(record_ids, list)
        or not record_ids
        or not all(isinstance(record_id, str) and record_id for record_id in record_ids)
    ):
        raise ValueError("category manifest record_ids must be a non-empty list")
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("category manifest record_ids must be unique")
    expected_hash = _record_ids_sha256(record_ids)
    if manifest.get("record_ids_sha256") != expected_hash:
        raise ValueError("category manifest record_ids_sha256 mismatch")
    cohort_by_id: dict[str, dict[str, str]] = {}
    for record in records:
        record_id = str(record.get("record_id") or "")
        if not record_id:
            raise ValueError("cohort row is missing record_id")
        if record_id in cohort_by_id:
            raise ValueError(f"duplicate record_id in cohort: {record_id}")
        cohort_by_id[record_id] = record
    missing_ids = set(record_ids) - set(cohort_by_id)
    if missing_ids:
        raise ValueError(
            "category manifest record_ids absent from cohort: "
            + ", ".join(sorted(missing_ids))
        )
    categories = manifest.get("categories")
    if not isinstance(categories, list) or not all(
        isinstance(row, dict) for row in categories
    ):
        raise ValueError("category manifest categories must be a list of objects")
    return manifest, [cohort_by_id[record_id] for record_id in record_ids]


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}-", suffix=".json.tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        Path(temporary_name).replace(path)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="离线评估 Stage 3 错误类别恢复、Baseline 保护与可观察性。"
    )
    parser.add_argument("--cohort-path", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--candidate-predictions", type=Path, required=True)
    parser.add_argument("--category-manifest", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260810)
    return parser


def run_cli(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    cohort_records = _load_cohort(args.cohort_path)
    baseline = [
        _normalize_prediction(row) for row in _load_jsonl(args.baseline_predictions)
    ]
    candidate = [
        _normalize_prediction(row) for row in _load_jsonl(args.candidate_predictions)
    ]
    manifest, records = _load_category_manifest(args.category_manifest, cohort_records)
    record_ids = [record["record_id"] for record in records]
    baseline = _select_prediction_rows(baseline, record_ids, label="baseline")
    candidate = _select_prediction_rows(candidate, record_ids, label="candidate")
    payload = {
        "schema_version": 1,
        "dataset_id": manifest["dataset_id"],
        "split_kind": manifest["split_kind"],
        "claim_scope": "development_only",
        "record_ids_sha256": manifest["record_ids_sha256"],
        "inputs": {
            "cohort_sha256": file_sha256(args.cohort_path),
            "baseline_predictions_sha256": file_sha256(args.baseline_predictions),
            "candidate_predictions_sha256": file_sha256(args.candidate_predictions),
            "category_manifest_sha256": file_sha256(args.category_manifest),
        },
        "paired": paired_stage3_comparison(
            candidate,
            baseline,
            records,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        ),
        "error_categories": stage3_error_category_diagnostics(
            records,
            baseline,
            candidate,
            manifest["categories"],
        ),
    }
    _write_json_atomic(args.output_path, payload)
    return payload


def main() -> None:
    run_cli()


if __name__ == "__main__":
    main()
