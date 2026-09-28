"""Experiment selection, resumable JSONL execution, and metrics."""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any

from .audit import workflow_audit_record
from .contracts import (
    ArbitrationSource,
    BaselineAnchor,
    EvidenceDimension,
    EvidenceItem,
    Stage3RevisionAudit,
    TaxonomyStructure,
)
from .global_supervisor import pending_consistency
from .controller import (
    Stage2Controller,
    Stage2WorkflowConfig,
    Stage3Controller,
    Stage3WorkflowConfig,
    Stage3WorkflowResult,
)
from .domains import build_record_runtime
from Benchmark.src.annotation_contracts import annotation_metrics, annotation_mode, labels_equal
from .evaluation import baseline_preservation_diagnostics
from .frozen_evidence_runtime import FrozenEvidenceProjection
from .sla_budget import RecordSlaBudget, SlaBudgetExhausted
from .sla_controller import TargetedSlaController
from .workflow import AdaptiveEmpiricalWorkflow, AdaptiveWorkflowResult

RecordRunner = Callable[[dict[str, str]], dict[str, object]]


@dataclass(frozen=True)
class ProgressSnapshot:
    completed: int
    total: int
    record_id: str
    status: str
    elapsed_seconds: float
    eta_seconds: float | None
    resumed_count: int


def select_records(
    records: Sequence[dict[str, str]],
    *,
    record_ids: Sequence[str] | None,
    limit: int | None,
) -> list[dict[str, str]]:
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    selected = list(records)
    if record_ids is not None:
        requested = set(record_ids)
        known = {record["record_id"] for record in records}
        unknown = requested - known
        if unknown:
            raise ValueError(f"unknown record_ids: {sorted(unknown)}")
        selected = [record for record in selected if record["record_id"] in requested]
    return selected if limit is None else selected[:limit]


def _load_existing(
    path: Path,
    *,
    config_hash: str,
) -> dict[str, dict[str, object]]:
    existing: dict[str, dict[str, object]] = {}
    if not path.exists():
        return existing
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid JSONL at {path}:{line_number}: {error}"
            ) from error
        if not isinstance(row, dict):
            raise ValueError(f"invalid prediction row at {path}:{line_number}")
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(
                "prediction record_id must be a non-empty string at "
                f"{path}:{line_number}"
            )
        if row.get("config_hash") != config_hash:
            raise ValueError(f"config_hash mismatch at {path}:{line_number}")
        if record_id in existing:
            raise ValueError(f"duplicate record_id in predictions: {record_id}")
        existing[record_id] = row
    return existing


def _write_ordered_jsonl(
    path: Path,
    rows: Sequence[dict[str, object]],
) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".aew-",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run_records(
    records: Sequence[dict[str, str]],
    *,
    predictions_path: str | Path,
    config_hash: str,
    record_runner: RecordRunner,
    resume: bool,
    concurrency: int = 1,
    progress_callback: Callable[[ProgressSnapshot], None] | None = None,
    clock: Callable[[], float] = monotonic,
    failure_row_factory: (
        Callable[[dict[str, str], SlaBudgetExhausted], dict[str, object]] | None
    ) = None,
) -> list[dict[str, object]]:
    """Persist completions incrementally and finalize rows in input record order."""

    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    selected_ids = [record.get("record_id") for record in records]
    if any(
        not isinstance(record_id, str) or not record_id for record_id in selected_ids
    ):
        raise ValueError("records require non-empty record_id values")
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("duplicate record_id in records")
    selected_id_set = set(selected_ids)
    output = Path(predictions_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_existing(output, config_hash=config_hash) if resume else {}
    unknown_existing = set(existing) - selected_id_set
    if unknown_existing:
        raise ValueError(
            "unknown record_id in existing predictions: " f"{sorted(unknown_existing)}"
        )
    mode = "a" if resume else "w"
    rows_by_id = {
        record_id: existing[record_id]
        for record_id in selected_ids
        if record_id in existing
    }
    pending = [record for record in records if record["record_id"] not in existing]
    resumed_count = len(rows_by_id)
    started_at = clock()
    live_completed = 0

    def notify(record_id: str, status: str) -> None:
        if progress_callback is None:
            return
        elapsed = max(0.0, clock() - started_at)
        completed = resumed_count + live_completed
        if completed >= len(records):
            eta = 0.0
        elif live_completed and elapsed > 0:
            eta = elapsed / live_completed * (len(records) - completed)
        else:
            eta = None
        progress_callback(
            ProgressSnapshot(
                completed=completed,
                total=len(records),
                record_id=record_id,
                status=status,
                elapsed_seconds=elapsed,
                eta_seconds=eta,
                resumed_count=resumed_count,
            )
        )

    notify("", "completed" if not pending else "starting")

    def normalize_row(
        record: dict[str, str], row: dict[str, object]
    ) -> dict[str, object]:
        result_id = row.get("record_id")
        expected_id = record["record_id"]
        if not isinstance(result_id, str) or not result_id:
            raise ValueError(f"record result for {expected_id} is missing record_id")
        if result_id not in selected_id_set:
            raise ValueError(f"unknown record_id in record result: {result_id}")
        if result_id != expected_id:
            raise ValueError(
                "duplicate or mismatched record_id in record result: "
                f"expected {expected_id}, got {result_id}"
            )
        row["config_hash"] = config_hash
        return row

    def execute(record: dict[str, str]) -> dict[str, object]:
        return normalize_row(record, dict(record_runner(record)))

    with output.open(mode, encoding="utf-8", newline="\n") as handle:
        with ThreadPoolExecutor(
            max_workers=concurrency,
            thread_name_prefix="adaptive-record",
        ) as executor:
            future_to_record = {
                executor.submit(execute, record): record for record in pending
            }
            for future in as_completed(future_to_record):
                record = future_to_record[future]
                try:
                    row = future.result()
                except SlaBudgetExhausted as error:
                    if failure_row_factory is None:
                        raise
                    row = normalize_row(
                        record, dict(failure_row_factory(record, error))
                    )
                record_id = record["record_id"]
                rows_by_id[record_id] = row
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                live_completed += 1
                status = (
                    "completed"
                    if resumed_count + live_completed >= len(records)
                    else "running"
                )
                notify(record_id, status)
    actual_ids = set(rows_by_id)
    if actual_ids != selected_id_set or len(rows_by_id) != len(selected_ids):
        missing = sorted(selected_id_set - actual_ids)
        unknown = sorted(actual_ids - selected_id_set)
        raise ValueError(
            f"prediction result record set mismatch: missing={missing}, unknown={unknown}"
        )
    ordered_rows = [rows_by_id[record_id] for record_id in selected_ids]
    _write_ordered_jsonl(output, ordered_rows)
    return ordered_rows


def run_targeted_sla_record(
    record: dict[str, str],
    *,
    domain: str,
    taxonomy: dict[str, list[str]],
    controller: TargetedSlaController,
    baseline_anchor: BaselineAnchor,
    record_budget: RecordSlaBudget,
    retrieved_at: str | None = None,
    frozen_evidence_projection: FrozenEvidenceProjection | None = None,
) -> dict[str, object]:
    """Run one targeted SLA record and persist only its bounded audit surface."""

    runtime = build_record_runtime(
        record,
        taxonomy=taxonomy,
        domain=domain,
        retrieved_at=retrieved_at,
        frozen_evidence_projection=frozen_evidence_projection,
    )
    started = monotonic()
    result = controller.run(runtime.ledger.view(), baseline_anchor, record_budget)
    elapsed = max(0.0, monotonic() - started)
    call_audit = [
        _serialize_targeted_sla_call_audit(call)
        for call in getattr(result, "call_audit", ())
    ]
    role_timings = _targeted_sla_role_timings(call_audit, fallback_seconds=elapsed)
    decision_status = {
        (
            dimension.value
            if isinstance(dimension, EvidenceDimension)
            else str(dimension)
        ): (status.value if hasattr(status, "value") else str(status))
        for dimension, status in result.decision_status.items()
    }
    diagnosis = getattr(result, "diagnosis", None)
    baseline_comparison = (
        {
            "symptom_matches_baseline": bool(
                getattr(diagnosis, "symptom_matches_baseline")
            ),
            "root_cause_matches_baseline": bool(
                getattr(diagnosis, "root_cause_matches_baseline")
            ),
        }
        if diagnosis is not None
        else None
    )
    verification = getattr(result, "verification", None)
    verifier_evidence: dict[str, dict[str, object]] = {}
    if verification is not None:
        for dimension in ("symptom", "root_cause"):
            part = getattr(verification, dimension, None)
            if part is None:
                continue
            verdict = getattr(part, "verdict", None)
            verifier_evidence[dimension] = {
                "verdict": getattr(verdict, "value", verdict),
                "supporting_evidence_ids": list(
                    getattr(part, "supporting_evidence_ids", ())
                ),
                "counter_evidence_ids": list(getattr(part, "counter_evidence_ids", ())),
            }
    arbitration = getattr(result, "arbitration", None)
    arbitration_evidence: dict[str, dict[str, object]] = {}
    if arbitration is not None:
        for dimension in ("symptom", "root_cause"):
            choice = getattr(arbitration, dimension, None)
            if choice is None:
                continue
            arbitration_evidence[dimension] = {
                "selected_label": getattr(choice, "selected_label"),
                "supporting_evidence_ids": list(
                    getattr(choice, "supporting_evidence_ids", ())
                ),
            }
    audit = {
        "targeted_sla": {
            "decision_status": decision_status,
            "call_count": result.call_count,
            "role_timings": role_timings,
            "call_audit": call_audit,
            "fallback": result.fallback,
            "budget_remaining_seconds": result.budget_remaining_seconds,
            "arbitrated": arbitration is not None,
            "baseline_comparison": baseline_comparison,
            "verifier_evidence": verifier_evidence,
            "arbitration_evidence": arbitration_evidence,
        }
    }
    return {
        "record_id": record["record_id"],
        "stage2_prediction": None,
        "stage2_valid": False,
        "symptom_prediction": result.final_decision.symptom_label,
        "root_cause_prediction": result.final_decision.root_cause_label,
        "stage3_valid": True,
        "stop_reason": "targeted_sla30",
        "decision_status": decision_status,
        "call_count": result.call_count,
        "role_timings": audit["targeted_sla"]["role_timings"],
        "call_audit": call_audit,
        "fallback": result.fallback,
        "budget_remaining_seconds": result.budget_remaining_seconds,
        "audit": audit,
    }


def _targeted_sla_audit_value(call: object, name: str) -> object:
    return call.get(name) if isinstance(call, Mapping) else getattr(call, name, None)


def _serialize_targeted_sla_call_audit(call: object) -> dict[str, object]:
    """Persist only the bounded role, retry, and sanitized provider-error audit."""

    raw_errors = _targeted_sla_audit_value(call, "provider_errors")
    provider_errors = (
        [_serialize_targeted_sla_provider_error(error) for error in raw_errors]
        if isinstance(raw_errors, (list, tuple))
        else []
    )
    return {
        "role": str(_targeted_sla_audit_value(call, "role")),
        "latency_seconds": float(_targeted_sla_audit_value(call, "latency_seconds")),
        "schema_attempts": int(_targeted_sla_audit_value(call, "schema_attempts")),
        "network_attempts": int(_targeted_sla_audit_value(call, "network_attempts")),
        "provider_errors": provider_errors,
    }


def _serialize_targeted_sla_provider_error(error: object) -> dict[str, object]:
    return {
        "status": str(_targeted_sla_audit_value(error, "status")),
        "retryable": bool(_targeted_sla_audit_value(error, "retryable")),
        "cause_type": _targeted_sla_audit_value(error, "cause_type"),
        "http_status": _targeted_sla_audit_value(error, "http_status"),
        "retry_after_seconds": _targeted_sla_audit_value(error, "retry_after_seconds"),
    }


def _targeted_sla_role_timings(
    calls: Sequence[dict[str, object]], *, fallback_seconds: float
) -> dict[str, float]:
    timings: dict[str, float] = {}
    for call in calls:
        role = call.get("role")
        latency = call["latency_seconds"]
        if (
            not isinstance(role, str)
            or not role.startswith("sla_")
            or isinstance(latency, bool)
            or not isinstance(latency, (int, float))
            or latency < 0
        ):
            continue
        timings[role] = timings.get(role, 0.0) + float(latency)
    return timings or {"targeted_sla_controller": fallback_seconds}


def evaluate_experiment(
    records: Sequence[dict[str, str]],
    rows: Sequence[dict[str, object]],
    *,
    stage: str = "all",
    taxonomy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Report fixed-denominator and selective metrics for one stage mode."""

    if stage not in {"stage2", "stage3", "all"}:
        raise ValueError("stage must be stage2, stage3, or all")

    record_ids = [str(record.get("record_id") or "") for record in records]
    if any(not record_id for record_id in record_ids):
        raise ValueError("evaluation record is missing record_id")
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("duplicate record_id in evaluation records")

    predictions: dict[str, dict[str, object]] = {}
    for row in rows:
        record_id = str(row.get("record_id") or "")
        if not record_id:
            raise ValueError("prediction row is missing record_id")
        if record_id in predictions:
            raise ValueError(f"duplicate record_id in predictions: {record_id}")
        predictions[record_id] = row
    expected_ids = set(record_ids)
    actual_ids = set(predictions)
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        raise ValueError(f"record set mismatch: missing={missing}, extra={extra}")
    total = len(records)
    required_gold = {"decision", "symptom", "root_cause"}
    if stage in {"stage3", "all"} and any(
        not required_gold.issubset(record) for record in records
    ):
        raise ValueError(
            "label accuracy evaluation requires decision, symptom, and root_cause; "
            "use execution metrics for evidence-only cohorts"
        )
    accepted = [
        record for record in records if record.get("decision") == "accepted_fault"
    ]

    def stage_audit(
        row: dict[str, object], stage_name: str
    ) -> dict[str, object] | None:
        audit = row.get("audit")
        if not isinstance(audit, dict):
            return None
        value = audit.get(stage_name)
        return value if isinstance(value, dict) else None

    def telemetry(
        cohort: Sequence[dict[str, str]],
        *,
        stage_name: str,
        correctness: dict[str, bool],
    ) -> dict[str, object]:
        stop_reasons: Counter[str] = Counter()
        dimensions: Counter[str] = Counter()
        retrieval_statuses: Counter[str] = Counter(
            {"found": 0, "absent": 0, "unavailable": 0}
        )
        source_counts: Counter[str] = Counter(
            {
                "direct_consensus": 0,
                "targeted_arbitration": 0,
                **({"fallback_uncertain": 0} if stage_name == "stage3" else {}),
                **({"single_team_degraded": 0} if stage_name == "stage3" else {}),
                **({"policy_composition": 0} if stage_name == "stage2" else {}),
            }
        )
        known_resolution_sources = {source.value for source in ArbitrationSource}
        source_correct: Counter[str] = Counter()
        budget_exhausted_count = 0

        for record in cohort:
            row = predictions.get(record["record_id"], {})
            audit = stage_audit(row, stage_name)
            valid = row.get(f"{stage_name}_valid") is True
            if audit is not None:
                for delta in audit.get("evidence_deltas", []):
                    if isinstance(delta, dict):
                        status = str(delta.get("status", "")).lower()
                        if status:
                            retrieval_statuses[status] += 1
                budget_exhausted_count += int(audit.get("budget_exhausted") is True)
                final_decision = audit.get("final_decision")
                if valid and isinstance(final_decision, dict):
                    source = str(final_decision.get("source", ""))
                    if source in known_resolution_sources:
                        source_counts[source] += 1
                        source_correct[source] += int(
                            correctness.get(record["record_id"], False)
                        )
                unresolved = audit.get("unresolved")
                if not valid and isinstance(unresolved, dict):
                    reason = str(unresolved.get("stop_reason", ""))
                    if reason:
                        stop_reasons[reason] += 1
                    for dimension in unresolved.get("dimensions", []):
                        dimensions[str(dimension)] += 1
                    continue
                verification = audit.get("verification")
                if (
                    not valid
                    and isinstance(verification, dict)
                    and verification.get("valid") is False
                ):
                    stop_reasons[f"{stage_name}_verification_failed"] += 1
                    continue
            if not valid:
                reason = str(row.get("stop_reason") or "unknown_unresolved")
                stop_reasons[reason] += 1

        source_accuracies = {
            source: (source_correct[source] / count if count else None)
            for source, count in sorted(source_counts.items())
        }
        denominator = len(cohort)
        return {
            "stop_reason_counts": dict(sorted(stop_reasons.items())),
            "unresolved_dimension_counts": dict(sorted(dimensions.items())),
            "retrieval_request_count": sum(retrieval_statuses.values()),
            "retrieval_status_counts": dict(sorted(retrieval_statuses.items())),
            "budget_exhausted_count": budget_exhausted_count,
            "budget_exhausted_rate": (
                budget_exhausted_count / denominator if denominator else 0.0
            ),
            "resolution_source_counts": dict(sorted(source_counts.items())),
            "resolution_source_accuracies": source_accuracies,
        }

    result: dict[str, Any] = {"n": total}
    if stage in {"stage2", "all"}:
        stage2_correctness = {
            record["record_id"]: (
                predictions.get(record["record_id"], {}).get("stage2_valid") is True
                and predictions.get(record["record_id"], {}).get("stage2_prediction")
                == record.get("decision")
            )
            for record in records
        }
        stage2_resolved = sum(
            predictions.get(record["record_id"], {}).get("stage2_valid") is True
            for record in records
        )
        stage2_correct = sum(stage2_correctness.values())
        result["stage2"] = {
            "n": total,
            "resolved_count": stage2_resolved,
            "coverage": stage2_resolved / total if total else 0.0,
            "accuracy": stage2_correct / total if total else 0.0,
            "selective_accuracy": (
                stage2_correct / stage2_resolved if stage2_resolved else None
            ),
            "correct_count": stage2_correct,
            "unresolved_count": total - stage2_resolved,
            "invalid_count": total - stage2_resolved,
            **telemetry(
                records,
                stage_name="stage2",
                correctness=stage2_correctness,
            ),
        }

    if stage in {"stage3", "all"}:
        stage3_rows = {
            record["record_id"]: predictions.get(record["record_id"], {})
            for record in accepted
        }
        stage3_resolved = sum(
            row.get("stage3_valid") is True for row in stage3_rows.values()
        )
        symptom_correctness = {
            record["record_id"]: (
                stage3_rows[record["record_id"]].get("stage3_valid") is True
                and (labels_equal(taxonomy, "symptom", stage3_rows[record["record_id"]].get("symptom_prediction"), record.get("symptom"))
                     if taxonomy is not None else stage3_rows[record["record_id"]].get("symptom_prediction") == record.get("symptom"))
            )
            for record in accepted
        }
        root_correctness = {
            record["record_id"]: (
                stage3_rows[record["record_id"]].get("stage3_valid") is True
                and stage3_rows[record["record_id"]].get("root_cause_prediction")
                == record.get("root_cause")
            )
            for record in accepted
        }
        joint_correctness = {
            record["record_id"]: (
                symptom_correctness[record["record_id"]]
                and root_correctness[record["record_id"]]
            )
            for record in accepted
        }
        symptom_correct = sum(symptom_correctness.values())
        root_correct = sum(root_correctness.values())
        joint_correct = sum(joint_correctness.values())
        accepted_count = len(accepted)
        missed_by_stage2 = sum(
            not (
                row.get("stage2_valid") is True
                and row.get("stage2_prediction") == "accepted_fault"
            )
            for row in stage3_rows.values()
        )
        if stage == "stage3":
            stage3_invalid = accepted_count - stage3_resolved
            missed_by_stage2 = 0
        else:
            stage3_invalid = sum(
                row.get("stage2_valid") is True
                and row.get("stage2_prediction") == "accepted_fault"
                and row.get("stage3_valid") is not True
                for row in stage3_rows.values()
            )
        result["stage3"] = {
            "n": accepted_count,
            "resolved_count": stage3_resolved,
            "coverage": (stage3_resolved / accepted_count if accepted_count else 0.0),
            "symptom_accuracy": (
                symptom_correct / accepted_count if accepted_count else 0.0
            ),
            "root_cause_accuracy": (
                root_correct / accepted_count if accepted_count else 0.0
            ),
            "joint_accuracy": (
                joint_correct / accepted_count if accepted_count else 0.0
            ),
            "selective_symptom_accuracy": (
                symptom_correct / stage3_resolved if stage3_resolved else None
            ),
            "selective_root_cause_accuracy": (
                root_correct / stage3_resolved if stage3_resolved else None
            ),
            "selective_joint_accuracy": (
                joint_correct / stage3_resolved if stage3_resolved else None
            ),
            "unresolved_count": accepted_count - stage3_resolved,
            "invalid_count": stage3_invalid,
            "missed_by_stage2_count": missed_by_stage2,
            **telemetry(
                accepted,
                stage_name="stage3",
                correctness=joint_correctness,
            ),
        }
        preservation_rows = [predictions[record["record_id"]] for record in accepted]
        if any(
            isinstance(row.get("audit"), dict)
            and isinstance(row["audit"].get("stage3"), dict)
            and row["audit"]["stage3"].get("baseline_anchor") is not None
            for row in preservation_rows
        ):
            result["stage3"]["baseline_preservation"] = (
                baseline_preservation_diagnostics(accepted, preservation_rows)
            )

    if stage == "all":
        end_to_end_correct = 0
        for record in records:
            row = predictions.get(record["record_id"], {})
            if record.get("decision") == "rejected_candidate":
                end_to_end_correct += int(
                    row.get("stage2_valid") is True
                    and row.get("stage2_prediction") == "rejected_candidate"
                )
                continue
            end_to_end_correct += int(
                row.get("stage2_valid") is True
                and row.get("stage2_prediction") == "accepted_fault"
                and joint_correctness.get(record["record_id"], False)
            )
        result["end_to_end"] = {
            "exact_match_accuracy": (end_to_end_correct / total if total else 0.0),
            "correct_count": end_to_end_correct,
        }
    if taxonomy is not None and taxonomy.get("annotation_modes") and stage in {"stage3", "all"}:
        native_rows = [{"record_id": record["record_id"],
                        "symptom": predictions[record["record_id"]].get("symptom_prediction"),
                        "root_cause": predictions[record["record_id"]].get("root_cause_prediction"),
                        "invalid": predictions[record["record_id"]].get("stage3_valid") is not True}
                       for record in accepted]
        native = annotation_metrics(accepted, native_rows, taxonomy)
        result["stage3"].update(native)
        for dimension in ("symptom", "root_cause", "joint"):
            accuracy = native[f"{dimension}_accuracy"]
            result["stage3"][f"selective_{dimension}_accuracy"] = (accuracy * len(accepted) / stage3_resolved
                if accuracy is not None and stage3_resolved else None)
        if native["joint_accuracy"] is None:
            result["stage3"]["resolution_source_accuracies"] = {key: None for key in result["stage3"]["resolution_source_accuracies"]}
            result["stage3"]["resolution_source_accuracy_reason"] = "Joint annotation accuracy is not defined for free-text or constant symptoms."
            if stage == "all":
                result["end_to_end"] = {
                    "exact_match_accuracy": None, "correct_count": None,
                    "scoring_reason": "Symptom has no discriminative categorical exact-match score.",
                    "filter_and_root_cause_accuracy": sum(
                        predictions[record["record_id"]].get("stage2_valid") is True and
                        predictions[record["record_id"]].get("stage2_prediction") == record.get("decision") and
                        (record.get("decision") == "rejected_candidate" or root_correctness.get(record["record_id"], False))
                        for record in records) / total if total else 0.0,
                }
    return result


def evaluate_evidence_only_execution(
    records: Sequence[dict[str, str]],
    rows: Sequence[dict[str, object]],
    *,
    stage: str,
) -> dict[str, Any]:
    """Report execution coverage/provenance without consulting outcome labels."""

    if stage not in {"stage3", "all"}:
        raise ValueError("evidence-only execution metrics support stage3 or all")
    expected = tuple(str(record.get("record_id") or "") for record in records)
    actual = tuple(str(row.get("record_id") or "") for row in rows)
    if (
        any(not record_id for record_id in expected + actual)
        or len(expected) != len(set(expected))
        or len(actual) != len(set(actual))
        or set(expected) != set(actual)
    ):
        raise ValueError("evidence-only execution record set mismatch")
    indexed = {str(row["record_id"]): row for row in rows}
    resolved = sum(
        indexed[record_id].get("stage3_valid") is True for record_id in expected
    )
    stop_reasons: Counter[str] = Counter()
    actions = {
        dimension: Counter(
            {"preserved": 0, "revised": 0, "baseline_unavailable_fallback": 0}
        )
        for dimension in ("symptom", "root_cause")
    }
    certificates = 0
    baseline_available = 0
    revision_funnel: Counter[str] = Counter(
        {
            "instrumented_record_count": 0,
            "proposal_count": 0,
            "routed_card_count": 0,
            "assessment_attempt_count": 0,
            "dual_revise_count": 0,
            "cross_attempt_count": 0,
            "cross_pass_count": 0,
            "issued_count": 0,
            "applied_count": 0,
            "ambiguous_dimension_count": 0,
            "operational_failure_count": 0,
        }
    )
    for record_id in expected:
        row = indexed[record_id]
        audit = row.get("audit")
        stage3 = audit.get("stage3") if isinstance(audit, dict) else None
        stage3 = stage3 if isinstance(stage3, dict) else {}
        baseline = stage3.get("baseline_anchor")
        baseline_available += int(
            isinstance(baseline, dict) and baseline.get("valid") is True
        )
        unresolved = stage3.get("unresolved")
        if isinstance(unresolved, dict) and unresolved.get("stop_reason"):
            stop_reasons[str(unresolved["stop_reason"])] += 1
        elif row.get("stop_reason"):
            stop_reasons[str(row["stop_reason"])] += 1
        preservation = stage3.get("preservation_result")
        if isinstance(preservation, dict):
            for dimension in actions:
                action = preservation.get(f"{dimension}_action")
                if action in actions[dimension]:
                    actions[dimension][str(action)] += 1
        issued = stage3.get("revision_certificates")
        certificates += len(issued) if isinstance(issued, list) else 0
        raw_revision_audit = stage3.get("revision_audit")
        if raw_revision_audit is not None:
            if not isinstance(raw_revision_audit, dict):
                raise ValueError(f"record {record_id} has malformed revision audit")
            revision_audit = Stage3RevisionAudit.model_validate(raw_revision_audit)
            revision_funnel["instrumented_record_count"] += 1
            revision_funnel["proposal_count"] += len(revision_audit.proposal_digests)
            revision_funnel["routed_card_count"] += len(revision_audit.routed_card_ids)
            revision_funnel["assessment_attempt_count"] += len(
                revision_audit.assessment_attempts
            )
            revision_funnel["dual_revise_count"] += len(
                revision_audit.dual_revise_proposal_digests
            )
            revision_funnel["cross_attempt_count"] += len(
                revision_audit.consistency_attempts
            )
            revision_funnel["cross_pass_count"] += sum(
                attempt.status is not None and attempt.status.value == "consistent"
                for attempt in revision_audit.consistency_attempts
            )
            revision_funnel["issued_count"] += len(
                revision_audit.issued_certificate_digests
            )
            revision_funnel["applied_count"] += len(
                revision_audit.applied_certificate_digests
            )
            revision_funnel["ambiguous_dimension_count"] += len(
                revision_audit.ambiguous_certified_dimensions
            )
            revision_funnel["operational_failure_count"] += sum(
                attempt.operational_failure
                for attempt in (
                    *revision_audit.assessment_attempts,
                    *revision_audit.consistency_attempts,
                )
            )
    denominator = len(expected)
    return {
        "n": denominator,
        "evaluation": {
            "status": "not_evaluated",
            "reason": "evidence_only_runner_cohort",
        },
        "stage3": {
            "n": denominator,
            "resolved_count": resolved,
            "coverage": resolved / denominator if denominator else 0.0,
            "invalid_count": denominator - resolved,
            "stop_reason_counts": dict(sorted(stop_reasons.items())),
            "baseline_preservation": {
                "n": denominator,
                "evaluation_status": "not_evaluated",
                "baseline_available_count": baseline_available,
                "actions": {
                    dimension: dict(counts) for dimension, counts in actions.items()
                },
                "certificates": {"issued_count": certificates},
                "revision_funnel": dict(revision_funnel),
            },
        },
    }


def run_adaptive_record(
    record: dict[str, str],
    *,
    domain: str,
    taxonomy: dict[str, list[str]],
    agents: object,
    stage: str,
    retrieved_at: str | None = None,
    stage2_config: Stage2WorkflowConfig | None = None,
    stage3_config: Stage3WorkflowConfig | None = None,
    baseline_anchor: BaselineAnchor | None = None,
    taxonomy_structure: TaxonomyStructure | None = None,
    expected_baseline_config_hash: str | None = None,
    expected_baseline_predictions_sha256: str | None = None,
    expected_baseline_anchor_hash: str | None = None,
    expected_taxonomy_structure_hash: str | None = None,
    frozen_evidence_projection: FrozenEvidenceProjection | None = None,
    supplemental_items: Sequence[EvidenceItem] = (),
) -> dict[str, object]:
    """Execute one record through the configured gated workflow."""

    if stage not in {"stage2", "stage3", "all"}:
        raise ValueError("stage must be stage2, stage3, or all")
    runtime = build_record_runtime(
        record,
        taxonomy=taxonomy,
        domain=domain,
        retrieved_at=retrieved_at,
        frozen_evidence_projection=frozen_evidence_projection,
    )
    if supplemental_items and (stage != "stage3" or frozen_evidence_projection is not None):
        raise ValueError("supplemental items require Stage 3 without registered frozen evidence")
    for item in supplemental_items:
        runtime.ledger.append(item)
    stage2_role_names = (
        "fault_evidence_analyst",
        "scope_boundary_analyst",
        "repair_causality_analyst",
    )
    present_stage2_roles: dict[str, object] = {}
    legacy_stage2: object | None = None
    role_owned_stage2 = False
    if stage in {"stage2", "all"}:
        present_stage2_roles = {
            name: getattr(agents, name)
            for name in stage2_role_names
            if hasattr(agents, name)
        }
        required_stage2_roles = (
            stage2_role_names if domain == "issta2024" else stage2_role_names[:2]
        )
        if present_stage2_roles:
            if any(
                not callable(value) for value in present_stage2_roles.values()
            ) or any(
                name not in present_stage2_roles for name in required_stage2_roles
            ):
                raise ValueError("partial role-owned Stage 2 adapter surface")
            role_owned_stage2 = True
        else:
            legacy_stage2 = getattr(agents, "stage2_analyst", None)
            if not callable(legacy_stage2):
                raise ValueError(
                    "Stage 2 adapter must expose either a complete role-owned surface "
                    "or legacy stage2_analyst only"
                )
    stage2_controller = (
        Stage2Controller(
            analyst=(None if role_owned_stage2 else legacy_stage2),
            fault_evidence_analyst=(
                getattr(agents, "fault_evidence_analyst") if role_owned_stage2 else None
            ),
            scope_boundary_analyst=(
                getattr(agents, "scope_boundary_analyst") if role_owned_stage2 else None
            ),
            repair_causality_analyst=(
                present_stage2_roles["repair_causality_analyst"]
                if role_owned_stage2 and domain == "issta2024"
                else None
            ),
            readiness=getattr(agents, "evidence_readiness"),
            specialists=runtime.specialists,
            arbitrator=getattr(agents, "stage2_arbitrator", None),
            config=stage2_config,
        )
        if stage in {"stage2", "all"}
        else None
    )
    stage3_controller = (
        Stage3Controller(
            readiness=getattr(agents, "evidence_readiness"),
            joint_anchor=getattr(agents, "joint_anchor", None),
            symptom_verifier=getattr(agents, "symptom_verifier", None),
            root_cause_verifier=getattr(agents, "root_cause_verifier", None),
            symptom_analyst=getattr(agents, "symptom_analyst", None),
            root_cause_analyst=getattr(agents, "root_cause_analyst", None),
            consistency_checker=(
                pending_consistency
                if getattr(agents, "global_supervisor_enabled", False)
                else getattr(agents, "consistency_checker", None)
            ),
            global_supervisor=(getattr(agents, "supervise_framework")
                if getattr(agents, "global_supervisor_enabled", False) else None),
            boundary_challenger=getattr(agents, "boundary_challenger", None),
            specialists=runtime.specialists,
            arbitrator=getattr(agents, "stage3_arbitrator", None),
            baseline_anchor=baseline_anchor,
            taxonomy_structure=taxonomy_structure,
            baseline_revision_assessor=(
                getattr(agents, "baseline_revision_assessment", None)
                if any(
                    value is not None
                    for value in (
                        baseline_anchor,
                        taxonomy_structure,
                        expected_baseline_config_hash,
                        expected_baseline_predictions_sha256,
                        expected_baseline_anchor_hash,
                        expected_taxonomy_structure_hash,
                    )
                )
                else None
            ),
            baseline_revision_consistency=(
                getattr(agents, "baseline_revision_consistency", None)
                if any(
                    value is not None
                    for value in (
                        baseline_anchor,
                        taxonomy_structure,
                        expected_baseline_config_hash,
                        expected_baseline_predictions_sha256,
                        expected_baseline_anchor_hash,
                        expected_taxonomy_structure_hash,
                    )
                )
                else None
            ),
            expected_baseline_config_hash=expected_baseline_config_hash,
            expected_baseline_predictions_sha256=(expected_baseline_predictions_sha256),
            expected_baseline_anchor_hash=expected_baseline_anchor_hash,
            expected_taxonomy_structure_hash=expected_taxonomy_structure_hash,
            config=stage3_config,
        )
        if stage in {"stage3", "all"}
        else None
    )
    if stage == "all":
        assert stage2_controller is not None
        assert stage3_controller is not None
        result = AdaptiveEmpiricalWorkflow(
            stage2=stage2_controller,
            stage3=stage3_controller,
        ).run(runtime.ledger)
    elif stage == "stage2":
        assert stage2_controller is not None
        stage2_result = stage2_controller.run(runtime.ledger)
        result = AdaptiveWorkflowResult(
            stage2=stage2_result,
            stage3=None,
            stop_reason="stage2_only",
        )
    else:
        assert stage3_controller is not None
        result = AdaptiveWorkflowResult(
            stage2=None,
            stage3=stage3_controller.run(runtime.ledger),
            stop_reason="stage3_only",
        )

    stage2_valid = (
        result.stage2 is not None
        and result.stage2.verification is not None
        and result.stage2.verification.valid
        and result.stage2.final_decision is not None
    )
    stage2_prediction = (
        result.stage2.final_decision.decision.value
        if (
            stage2_valid
            and result.stage2 is not None
            and result.stage2.final_decision is not None
        )
        else None
    )
    stage3_result = (
        result.stage3 if isinstance(result.stage3, Stage3WorkflowResult) else None
    )
    stage3_valid = (
        stage3_result is not None
        and stage3_result.verification is not None
        and stage3_result.verification.valid
        and stage3_result.final_decision is not None
    )
    external_stop_reason = result.stop_reason
    if stage3_result is not None and not stage3_valid:
        if stage3_result.unresolved is not None:
            external_stop_reason = stage3_result.unresolved.stop_reason
        elif (
            stage3_result.verification is not None
            and not stage3_result.verification.valid
        ):
            external_stop_reason = "stage3_verification_failed"
    return {
        "record_id": record["record_id"],
        "stage2_prediction": stage2_prediction,
        "stage2_valid": stage2_valid,
        "symptom_prediction": (
            stage3_result.final_decision.symptom_label
            if stage3_valid
            and stage3_result is not None
            and stage3_result.final_decision is not None
            else None
        ),
        "root_cause_prediction": (
            stage3_result.final_decision.root_cause_label
            if stage3_valid
            and stage3_result is not None
            and stage3_result.final_decision is not None
            else None
        ),
        "stage3_valid": stage3_valid,
        "stop_reason": external_stop_reason,
        "audit": workflow_audit_record(runtime.ledger, result),
    }
