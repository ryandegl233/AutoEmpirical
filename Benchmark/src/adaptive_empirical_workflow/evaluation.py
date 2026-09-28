"""Strict paired and team-level evaluation for Stage 3 experiments."""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Sequence
from typing import Any

from .contracts import LabelRevisionCertificate, Stage3RevisionAudit


_SLA_FALLBACK_STATUSES = frozenset(
    {
        "transport_fallback",
        "schema_fallback",
        "budget_fallback",
        "unresolved_fallback",
    }
)


def _targeted_baseline_labels(value: object) -> tuple[str | None, str | None]:
    if isinstance(value, dict):
        symptom = value.get("symptom_label", value.get("symptom_prediction"))
        root_cause = value.get("root_cause_label", value.get("root_cause_prediction"))
    else:
        symptom = getattr(value, "symptom_label", None)
        root_cause = getattr(value, "root_cause_label", None)
    return (
        symptom if isinstance(symptom, str) and symptom else None,
        root_cause if isinstance(root_cause, str) and root_cause else None,
    )


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def targeted_sla_diagnostics(
    records: Sequence[dict[str, str]],
    rows: Sequence[dict[str, object]],
    baseline_by_id: dict[str, object],
) -> dict[str, object]:
    """Summarize bounded SLA execution without sending any evaluation data to agents."""

    expected = [str(record.get("record_id") or "") for record in records]
    indexed_rows = _index_unique(rows, label="targeted SLA prediction")
    if (
        any(not record_id for record_id in expected)
        or len(expected) != len(set(expected))
        or set(indexed_rows) != set(expected)
        or set(baseline_by_id) != set(expected)
    ):
        raise ValueError("targeted SLA record set mismatch")

    fallback_counts: Counter[str] = Counter()
    provider_errors: Counter[str] = Counter()
    latencies: list[float] = []
    role_call_count = 0
    provider_request_count = 0
    completed = 0
    arbitrated = 0
    for record_id in expected:
        row = indexed_rows[record_id]
        raw_calls = row.get("call_count", 0)
        if (
            isinstance(raw_calls, int)
            and not isinstance(raw_calls, bool)
            and raw_calls >= 0
        ):
            role_call_count += raw_calls
        audit = row.get("audit")
        targeted = audit.get("targeted_sla") if isinstance(audit, dict) else None
        targeted = targeted if isinstance(targeted, dict) else {}
        statuses = targeted.get("decision_status", row.get("decision_status", {}))
        statuses = statuses if isinstance(statuses, dict) else {}
        fallback = bool(row.get("fallback", targeted.get("fallback", False)))
        for status in statuses.values():
            if isinstance(status, str) and status in _SLA_FALLBACK_STATUSES:
                fallback_counts[status] += 1
                fallback = True
        completed += int(row.get("stage3_valid") is True and not fallback)
        arbitrated += int(targeted.get("arbitrated") is True)
        call_audit = targeted.get("call_audit", row.get("call_audit", []))
        if not isinstance(call_audit, list):
            continue
        for call in call_audit:
            if not isinstance(call, dict):
                continue
            network_attempts = call.get("network_attempts")
            if (
                isinstance(network_attempts, int)
                and not isinstance(network_attempts, bool)
                and network_attempts >= 0
            ):
                provider_request_count += network_attempts
            latency = call.get("latency_seconds")
            if (
                isinstance(latency, (int, float))
                and not isinstance(latency, bool)
                and math.isfinite(float(latency))
                and latency >= 0
            ):
                latencies.append(float(latency))
            for error in call.get("provider_errors", []):
                if not isinstance(error, dict):
                    continue
                cause = error.get("cause_type")
                provider_errors[
                    cause if isinstance(cause, str) else "unknown_error"
                ] += 1

    labeled = all(
        isinstance(record.get("symptom"), str)
        and isinstance(record.get("root_cause"), str)
        for record in records
    )
    result: dict[str, object] = {
        "n": len(expected),
        "coverage": (
            sum(
                indexed_rows[record_id].get("stage3_valid") is True
                for record_id in expected
            )
            / len(expected)
            if expected
            else 0.0
        ),
        "completed_model_decisions": completed,
        "fallback_counts": dict(sorted(fallback_counts.items())),
        "role_call_count": role_call_count,
        "model_call_count": provider_request_count,
        "provider_request_count": provider_request_count,
        "total_call_count": provider_request_count,
        "arbitrated_count": arbitrated,
        "call_latency_seconds": {
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
        },
        "provider_error_distribution": dict(sorted(provider_errors.items())),
    }
    if not labeled:
        result["evaluation_status"] = "not_evaluated"
        return result

    recovery = 0
    harm = 0
    for record in records:
        record_id = record["record_id"]
        baseline_symptom, baseline_root = _targeted_baseline_labels(
            baseline_by_id[record_id]
        )
        baseline_correct = (
            baseline_symptom == record["symptom"]
            and baseline_root == record["root_cause"]
        )
        candidate = indexed_rows[record_id]
        candidate_correct = bool(
            candidate.get("stage3_valid") is True
            and candidate.get("symptom_prediction") == record["symptom"]
            and candidate.get("root_cause_prediction") == record["root_cause"]
        )
        recovery += int(not baseline_correct and candidate_correct)
        harm += int(baseline_correct and not candidate_correct)
    result.update(
        {
            "evaluation_status": "evaluated",
            "recovery": recovery,
            "harm": harm,
            "net_recovery": recovery - harm,
        }
    )
    return result


def _index_unique(
    rows: Sequence[dict[str, object]], *, label: str
) -> dict[str, dict[str, object]]:
    indexed: dict[str, dict[str, object]] = {}
    for row in rows:
        record_id = str(row.get("record_id") or "")
        if not record_id:
            raise ValueError(f"{label} row is missing record_id")
        if record_id in indexed:
            raise ValueError(f"duplicate record_id in {label}: {record_id}")
        indexed[record_id] = row
    return indexed


def _gold_index(
    records: Sequence[dict[str, str]],
) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    for record in records:
        record_id = str(record.get("record_id") or "")
        if not record_id:
            raise ValueError("gold row is missing record_id")
        if record_id in indexed:
            raise ValueError(f"duplicate record_id in gold: {record_id}")
        indexed[record_id] = record
    return indexed


def _joint_correct(row: dict[str, object], record: dict[str, str]) -> bool:
    return bool(
        row.get("stage3_valid") is True
        and row.get("symptom_prediction") == record.get("symptom")
        and row.get("root_cause_prediction") == record.get("root_cause")
    )


def _dimension_correct(
    row: dict[str, object], record: dict[str, str], dimension: str
) -> bool:
    if dimension == "joint":
        return _joint_correct(row, record)
    if dimension not in {"symptom", "root_cause"}:
        raise ValueError(f"unsupported Stage 3 dimension: {dimension}")
    return bool(
        row.get("stage3_valid") is True
        and row.get(f"{dimension}_prediction") == record.get(dimension)
    )


def _paired_dimension_summary(
    *,
    ordered_ids: Sequence[str],
    records: dict[str, dict[str, str]],
    baseline: dict[str, dict[str, object]],
    candidate: dict[str, dict[str, object]],
    dimension: str,
) -> tuple[dict[str, object], dict[str, object]]:
    baseline_correct_ids = [
        record_id
        for record_id in ordered_ids
        if _dimension_correct(baseline[record_id], records[record_id], dimension)
    ]
    baseline_error_ids = [
        record_id for record_id in ordered_ids if record_id not in baseline_correct_ids
    ]
    harmed_ids = [
        record_id
        for record_id in baseline_correct_ids
        if not _dimension_correct(candidate[record_id], records[record_id], dimension)
    ]
    recovered_ids = [
        record_id
        for record_id in baseline_error_ids
        if _dimension_correct(candidate[record_id], records[record_id], dimension)
    ]
    still_wrong_ids = [
        record_id for record_id in baseline_error_ids if record_id not in recovered_ids
    ]
    preserved = len(baseline_correct_ids) - len(harmed_ids)
    return (
        {
            "baseline_correct_count": len(baseline_correct_ids),
            "preserved_correct_count": preserved,
            "harm_count": len(harmed_ids),
            "preservation_rate": (
                preserved / len(baseline_correct_ids) if baseline_correct_ids else None
            ),
            "harmed_record_ids": harmed_ids,
        },
        {
            "baseline_error_count": len(baseline_error_ids),
            "recovery_count": len(recovered_ids),
            "recovery_rate": (
                len(recovered_ids) / len(baseline_error_ids)
                if baseline_error_ids
                else None
            ),
            "recovered_record_ids": recovered_ids,
            "still_wrong_record_ids": still_wrong_ids,
        },
    )


def _instrumentation_summary(
    ordered_ids: Sequence[str], candidate: dict[str, dict[str, object]]
) -> tuple[dict[str, object], dict[str, object]]:
    evidence_instrumented = 0
    evidence_requests = 0
    evidence_found = 0
    records_with_found = 0
    boundary_instrumented = 0
    boundary_usage = 0
    records_with_boundary = 0
    used_cards: set[str] = set()
    for record_id in ordered_ids:
        stage3 = _stage3_audit(candidate[record_id])
        if "evidence_deltas" in stage3:
            deltas = stage3["evidence_deltas"]
            if not isinstance(deltas, list) or not all(
                isinstance(delta, dict) and isinstance(delta.get("status"), str)
                for delta in deltas
            ):
                raise ValueError(
                    f"record {record_id} has malformed evidence expansion audit"
                )
            evidence_instrumented += 1
            evidence_requests += len(deltas)
            found_for_record = sum(delta["status"] == "found" for delta in deltas)
            evidence_found += found_for_record
            records_with_found += int(found_for_record > 0)
        revision_audit = stage3.get("revision_audit")
        if revision_audit is not None:
            if not isinstance(revision_audit, dict):
                raise ValueError(f"record {record_id} has malformed revision audit")
            card_ids = list(
                Stage3RevisionAudit.model_validate(revision_audit).routed_card_ids
            )
        elif "boundary_card_ids" in stage3:
            card_ids = stage3["boundary_card_ids"]
        else:
            card_ids = None
        if card_ids is not None:
            if not isinstance(card_ids, list) or not all(
                isinstance(card_id, str) and card_id for card_id in card_ids
            ):
                raise ValueError(
                    f"record {record_id} has malformed boundary card audit"
                )
            boundary_instrumented += 1
            boundary_usage += len(card_ids)
            records_with_boundary += int(bool(card_ids))
            used_cards.update(card_ids)
    denominator = len(ordered_ids)
    return (
        {
            "instrumented": evidence_instrumented == denominator,
            "instrumented_record_count": evidence_instrumented,
            "missing_record_count": denominator - evidence_instrumented,
            "request_count": evidence_requests,
            "found_count": evidence_found,
            "hit_rate": (
                evidence_found / evidence_requests if evidence_requests else None
            ),
            "records_with_found_evidence_count": records_with_found,
        },
        {
            "instrumented": boundary_instrumented == denominator,
            "instrumented_record_count": boundary_instrumented,
            "missing_record_count": denominator - boundary_instrumented,
            "usage_count": boundary_usage,
            "record_usage_count": records_with_boundary,
            "used_card_ids": sorted(used_cards),
        },
    )


def stage3_error_category_diagnostics(
    records: Sequence[dict[str, str]],
    baseline_rows: Sequence[dict[str, object]],
    candidate_rows: Sequence[dict[str, object]],
    category_rows: Sequence[dict[str, str]],
) -> dict[str, Any]:
    """Explain candidate recovery and harm against a complete baseline-error audit."""

    gold = _gold_index(records)
    baseline = _index_unique(baseline_rows, label="baseline")
    candidate = _index_unique(candidate_rows, label="candidate")
    if set(gold) != set(baseline) or set(gold) != set(candidate):
        raise ValueError("record sets differ between candidate, baseline, and gold")
    ordered_ids = tuple(gold)

    indexed_categories: dict[tuple[str, str], str] = {}
    category_dimensions: dict[str, str] = {}
    for row in category_rows:
        record_id = str(row.get("record_id") or "")
        if record_id not in gold:
            raise ValueError(f"unknown category record_id: {record_id}")
        dimension = str(row.get("dimension") or "")
        if dimension not in {"symptom", "root_cause"}:
            raise ValueError(f"invalid error category dimension: {dimension}")
        category = str(row.get("category") or "")
        if not category:
            raise ValueError("error category must be non-empty")
        key = (record_id, dimension)
        if key in indexed_categories:
            raise ValueError(f"duplicate error category for {record_id} {dimension}")
        if _dimension_correct(baseline[record_id], gold[record_id], dimension):
            raise ValueError(
                f"category marks a baseline-correct prediction: {record_id} {dimension}"
            )
        previous_dimension = category_dimensions.setdefault(category, dimension)
        if previous_dimension != dimension:
            raise ValueError(f"error category spans multiple dimensions: {category}")
        indexed_categories[key] = category

    expected_error_keys = {
        (record_id, dimension)
        for record_id in ordered_ids
        for dimension in ("symptom", "root_cause")
        if not _dimension_correct(baseline[record_id], gold[record_id], dimension)
    }
    missing = expected_error_keys - set(indexed_categories)
    if missing:
        rendered = ", ".join(
            f"{record_id}:{dimension}" for record_id, dimension in sorted(missing)
        )
        raise ValueError(f"uncategorized baseline errors: {rendered}")

    categories: dict[str, dict[str, object]] = {}
    for category in sorted(category_dimensions):
        dimension = category_dimensions[category]
        category_ids = [
            record_id
            for record_id in ordered_ids
            if indexed_categories.get((record_id, dimension)) == category
        ]
        recovered_ids = [
            record_id
            for record_id in category_ids
            if _dimension_correct(candidate[record_id], gold[record_id], dimension)
        ]
        still_wrong_ids = [
            record_id for record_id in category_ids if record_id not in recovered_ids
        ]
        categories[category] = {
            "dimension": dimension,
            "n": len(category_ids),
            "recovery_count": len(recovered_ids),
            "recovery_rate": len(recovered_ids) / len(category_ids),
            "recovered_record_ids": recovered_ids,
            "still_wrong_record_ids": still_wrong_ids,
        }

    preservation: dict[str, dict[str, object]] = {}
    baseline_errors: dict[str, dict[str, object]] = {}
    for dimension in ("symptom", "root_cause", "joint"):
        preservation[dimension], baseline_errors[dimension] = _paired_dimension_summary(
            ordered_ids=ordered_ids,
            records=gold,
            baseline=baseline,
            candidate=candidate,
            dimension=dimension,
        )
    evidence_expansion, boundary_cards = _instrumentation_summary(
        ordered_ids, candidate
    )
    return {
        "n": len(ordered_ids),
        "categories": categories,
        "baseline_preservation": preservation,
        "baseline_errors": baseline_errors,
        "evidence_expansion": evidence_expansion,
        "boundary_cards": boundary_cards,
    }


def _exact_mcnemar_two_sided(wins: int, losses: int) -> float:
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index) for index in range(min(wins, losses) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def _quantile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        return 0.0
    position = (len(sorted_values) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def paired_stage3_comparison(
    candidate_rows: Sequence[dict[str, object]],
    baseline_rows: Sequence[dict[str, object]],
    records: Sequence[dict[str, str]],
    *,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 20260810,
) -> dict[str, Any]:
    """Compare two Stage 3 systems on the exact same fixed denominator."""

    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    candidate = _index_unique(candidate_rows, label="candidate")
    baseline = _index_unique(baseline_rows, label="baseline")
    gold = _gold_index(records)
    if set(candidate) != set(baseline) or set(candidate) != set(gold):
        raise ValueError("record sets differ between candidate, baseline, and gold")

    ordered_ids = tuple(gold)
    candidate_correct = tuple(
        _joint_correct(candidate[record_id], gold[record_id])
        for record_id in ordered_ids
    )
    baseline_correct = tuple(
        _joint_correct(baseline[record_id], gold[record_id])
        for record_id in ordered_ids
    )
    wins = sum(
        left and not right for left, right in zip(candidate_correct, baseline_correct)
    )
    losses = sum(
        right and not left for left, right in zip(candidate_correct, baseline_correct)
    )
    ties = len(ordered_ids) - wins - losses
    deltas = tuple(
        int(left) - int(right)
        for left, right in zip(candidate_correct, baseline_correct)
    )
    delta = sum(deltas) / len(deltas) if deltas else 0.0

    rng = random.Random(bootstrap_seed)
    bootstrap = []
    if deltas:
        for _ in range(bootstrap_samples):
            bootstrap.append(
                sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas)
            )
    else:
        bootstrap = [0.0] * bootstrap_samples
    bootstrap.sort()

    return {
        "n": len(ordered_ids),
        "candidate_joint_accuracy": (
            sum(candidate_correct) / len(ordered_ids) if ordered_ids else 0.0
        ),
        "baseline_joint_accuracy": (
            sum(baseline_correct) / len(ordered_ids) if ordered_ids else 0.0
        ),
        "joint_accuracy_delta": delta,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "mcnemar_exact_two_sided_p": _exact_mcnemar_two_sided(wins, losses),
        "bootstrap": {
            "method": "paired_record_resampling",
            "seed": bootstrap_seed,
            "samples": bootstrap_samples,
            "confidence_level": 0.95,
            "lower": _quantile(bootstrap, 0.025),
            "upper": _quantile(bootstrap, 0.975),
        },
    }


def _stage3_audit(row: dict[str, object]) -> dict[str, object]:
    audit = row.get("audit")
    if not isinstance(audit, dict):
        return {}
    stage3 = audit.get("stage3")
    return stage3 if isinstance(stage3, dict) else {}


def _canonical_dict_digest(value: dict[str, object]) -> str:
    import hashlib
    import json

    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _revision_funnel_summary(
    ordered_ids: Sequence[str],
    gold: dict[str, dict[str, str]],
    predictions: dict[str, dict[str, object]],
) -> dict[str, object]:
    instrumented = proposal_count = routed_card_count = 0
    assessment_attempts = assessment_returns = dual_assessed = dual_revise = 0
    cross_attempts = cross_returns = cross_passes = 0
    issued = applied = ambiguity = operational_failures = failed_proposals = 0
    applied_help = applied_harm = 0
    per_card: dict[str, Counter[str]] = {}

    def card_counts(card_id: str) -> Counter[str]:
        return per_card.setdefault(
            card_id,
            Counter(
                {
                    "proposal_count": 0,
                    "assessment_attempt_count": 0,
                    "dual_revise_count": 0,
                    "cross_attempt_count": 0,
                    "issued_count": 0,
                    "applied_count": 0,
                    "help_count": 0,
                    "harm_count": 0,
                }
            ),
        )

    for record_id in ordered_ids:
        row = predictions[record_id]
        stage3 = _stage3_audit(row)
        raw_audit = stage3.get("revision_audit")
        if raw_audit is None:
            continue
        if not isinstance(raw_audit, dict):
            raise ValueError(f"record {record_id} has malformed revision audit")
        audit = Stage3RevisionAudit.model_validate(raw_audit)
        instrumented += 1
        proposal_count += len(audit.proposal_digests)
        routed_card_count += len(audit.routed_card_ids)
        proposal_cards = dict(
            zip(audit.proposal_digests, audit.routed_card_ids, strict=True)
        )
        for card_id in audit.routed_card_ids:
            card_counts(card_id)["proposal_count"] += 1

        assessment_attempts += len(audit.assessment_attempts)
        assessment_returns += sum(
            not attempt.operational_failure for attempt in audit.assessment_attempts
        )
        operational_failures += sum(
            attempt.operational_failure for attempt in audit.assessment_attempts
        )
        assessment_teams: dict[str, set[str]] = {}
        for attempt in audit.assessment_attempts:
            card_counts(attempt.boundary_card_id)["assessment_attempt_count"] += 1
            if not attempt.operational_failure:
                assessment_teams.setdefault(attempt.proposal_digest, set()).add(
                    attempt.assessor_team_id
                )
        dual_assessed += sum(teams == {"A", "B"} for teams in assessment_teams.values())
        dual_revise += len(audit.dual_revise_proposal_digests)
        for digest in audit.dual_revise_proposal_digests:
            card_id = proposal_cards.get(digest)
            if card_id is not None:
                card_counts(card_id)["dual_revise_count"] += 1

        cross_attempts += len(audit.consistency_attempts)
        cross_returns += sum(
            not attempt.operational_failure for attempt in audit.consistency_attempts
        )
        cross_passes += sum(
            attempt.status is not None and attempt.status.value == "consistent"
            for attempt in audit.consistency_attempts
        )
        operational_failures += sum(
            attempt.operational_failure for attempt in audit.consistency_attempts
        )
        for attempt in audit.consistency_attempts:
            card_counts(attempt.boundary_card_id)["cross_attempt_count"] += 1

        certificate_rows = stage3.get("revision_certificates")
        certificate_rows = (
            certificate_rows if isinstance(certificate_rows, list) else []
        )
        preservation = stage3.get("preservation_result")
        preservation = preservation if isinstance(preservation, dict) else {}
        applied_rows = preservation.get("applied_certificates")
        applied_rows = applied_rows if isinstance(applied_rows, list) else []
        if not all(isinstance(certificate, dict) for certificate in certificate_rows):
            raise ValueError(f"record {record_id} has malformed issued certificate")
        if not all(isinstance(certificate, dict) for certificate in applied_rows):
            raise ValueError(f"record {record_id} has malformed applied certificate")
        issued_models = [
            LabelRevisionCertificate.model_validate(certificate)
            for certificate in certificate_rows
        ]
        applied_models = [
            LabelRevisionCertificate.model_validate(certificate)
            for certificate in applied_rows
        ]
        issued_digests = tuple(
            _canonical_dict_digest(certificate.model_dump(mode="json"))
            for certificate in issued_models
        )
        applied_digests = tuple(
            _canonical_dict_digest(certificate.model_dump(mode="json"))
            for certificate in applied_models
        )
        if len(issued_digests) != len(set(issued_digests)):
            raise ValueError(f"record {record_id} has duplicate issued certificate")
        if len(applied_digests) != len(set(applied_digests)):
            raise ValueError(f"record {record_id} has duplicate applied certificate")
        if issued_digests != audit.issued_certificate_digests:
            raise ValueError(f"record {record_id} issued certificate audit mismatch")
        if applied_digests != audit.applied_certificate_digests:
            raise ValueError(f"record {record_id} applied certificate audit mismatch")
        entry_by_digest = {
            entry.certificate_digest: entry
            for entry in audit.issued_certificate_entries
        }
        for digest, certificate in zip(issued_digests, issued_models, strict=True):
            entry = entry_by_digest.get(digest)
            if entry is None or (
                certificate.proposal_digest != entry.proposal_digest
                or certificate.dimension != entry.dimension
                or certificate.boundary_card_id != entry.boundary_card_id
                or certificate.baseline_label != entry.baseline_label
                or certificate.proposed_label != entry.proposed_label
            ):
                raise ValueError(
                    f"record {record_id} certificate does not match canonical proposal chain"
                )
            card_counts(certificate.boundary_card_id)["issued_count"] += 1
        issued += len(issued_models)
        applied += len(applied_models)
        for certificate in applied_models:
            card_id = certificate.boundary_card_id
            dimension = certificate.dimension.value
            counts = card_counts(card_id)
            counts["applied_count"] += 1
            anchor = stage3.get("baseline_anchor")
            anchor = anchor if isinstance(anchor, dict) else {}
            baseline_correct = anchor.get("valid") is True and anchor.get(
                f"{dimension}_label"
            ) == gold[record_id].get(dimension)
            post_correct = row.get("stage3_valid") is True and row.get(
                f"{dimension}_prediction"
            ) == gold[record_id].get(dimension)
            help_count = int(not baseline_correct and post_correct)
            harm_count = int(baseline_correct and not post_correct)
            applied_help += help_count
            applied_harm += harm_count
            counts["help_count"] += help_count
            counts["harm_count"] += harm_count

        ambiguity += len(audit.ambiguous_certified_dimensions)
        failed_proposals += len(audit.failed_proposal_digests)

    denominator = len(ordered_ids)
    return {
        "instrumented": instrumented == denominator,
        "instrumented_record_count": instrumented,
        "missing_record_count": denominator - instrumented,
        "proposal_count": proposal_count,
        "routed_card_count": routed_card_count,
        "missing_route_count": proposal_count - routed_card_count,
        "assessment_attempt_count": assessment_attempts,
        "assessment_return_count": assessment_returns,
        "dual_assessed_count": dual_assessed,
        "dual_revise_count": dual_revise,
        "cross_attempt_count": cross_attempts,
        "cross_return_count": cross_returns,
        "cross_pass_count": cross_passes,
        "issued_count": issued,
        "applied_count": applied,
        "ambiguous_dimension_count": ambiguity,
        "operational_failure_count": operational_failures,
        "failed_proposal_count": failed_proposals,
        "applied_help_count": applied_help,
        "applied_harm_count": applied_harm,
        "per_card": {
            card_id: dict(counts) for card_id, counts in sorted(per_card.items())
        },
    }


def baseline_preservation_diagnostics(
    records: Sequence[dict[str, str]],
    rows: Sequence[dict[str, object]],
) -> dict[str, Any]:
    """Summarize frozen Baseline, pre-gate, and post-gate outcomes by record."""

    gold = _gold_index(records)
    predictions = _index_unique(rows, label="candidate")
    if set(gold) != set(predictions):
        raise ValueError("record sets differ between preservation rows and gold")
    ordered_ids = tuple(gold)
    dimensions = ("symptom", "root_cause", "joint")
    actions = {
        dimension: Counter(
            {
                "preserved": 0,
                "revised": 0,
                "baseline_unavailable_fallback": 0,
            }
        )
        for dimension in ("symptom", "root_cause")
    }
    correctness: dict[str, dict[str, dict[str, bool]]] = {
        source: {dimension: {} for dimension in dimensions}
        for source in ("baseline", "pre_gate", "post_gate")
    }
    baseline_valid: dict[str, bool] = {}
    attempted = issued = applied = applied_correct = 0
    assessment_verdicts: Counter[str] = Counter()
    revision_agreement = consistency_passes = operational_failures = 0
    unknown_citations = incapable_citations = 0
    trust_or_context_failures = ordinary_preserved = hardstop_preserved = 0
    hardstop_count = digest_mismatches = 0
    source_validity: dict[str, dict[str, bool]] = {
        source: {} for source in ("baseline", "pre_gate", "post_gate")
    }

    def diagnostic_strings(value: object) -> list[str]:
        if isinstance(value, dict):
            return [
                text for nested in value.values() for text in diagnostic_strings(nested)
            ]
        if isinstance(value, (list, tuple)):
            return [text for nested in value for text in diagnostic_strings(nested)]
        return [str(value).lower()] if isinstance(value, str) else []

    def labels_correct(
        symptom: object,
        root_cause: object,
        record: dict[str, str],
    ) -> dict[str, bool]:
        symptom_correct = symptom == record.get("symptom")
        root_correct = root_cause == record.get("root_cause")
        return {
            "symptom": symptom_correct,
            "root_cause": root_correct,
            "joint": symptom_correct and root_correct,
        }

    for record_id in ordered_ids:
        record = gold[record_id]
        row = predictions[record_id]
        stage3 = _stage3_audit(row)
        anchor = stage3.get("baseline_anchor")
        anchor = anchor if isinstance(anchor, dict) else {}
        valid_anchor = anchor.get("valid") is True
        baseline_valid[record_id] = valid_anchor
        source_validity["baseline"][record_id] = valid_anchor
        baseline_values = labels_correct(
            anchor.get("symptom_label") if valid_anchor else None,
            anchor.get("root_cause_label") if valid_anchor else None,
            record,
        )
        pre = stage3.get("pre_gate_candidate")
        pre = pre if isinstance(pre, dict) else {}
        pre_values = labels_correct(
            pre.get("symptom_label"), pre.get("root_cause_label"), record
        )
        source_validity["pre_gate"][record_id] = bool(
            pre.get("symptom_label") and pre.get("root_cause_label")
        )
        post_values = labels_correct(
            row.get("symptom_prediction") if row.get("stage3_valid") is True else None,
            (
                row.get("root_cause_prediction")
                if row.get("stage3_valid") is True
                else None
            ),
            record,
        )
        source_validity["post_gate"][record_id] = row.get("stage3_valid") is True
        for dimension in dimensions:
            correctness["baseline"][dimension][record_id] = baseline_values[dimension]
            correctness["pre_gate"][dimension][record_id] = pre_values[dimension]
            correctness["post_gate"][dimension][record_id] = post_values[dimension]

        preservation = stage3.get("preservation_result")
        preservation = preservation if isinstance(preservation, dict) else {}
        for dimension in ("symptom", "root_cause"):
            action = preservation.get(f"{dimension}_action")
            if action in actions[dimension]:
                actions[dimension][str(action)] += 1
        certificates = stage3.get("revision_certificates")
        certificate_rows = certificates if isinstance(certificates, list) else []
        issued += len(certificate_rows)
        applied_rows = preservation.get("applied_certificates")
        applied_rows = applied_rows if isinstance(applied_rows, list) else []
        applied += len(applied_rows)
        for certificate in applied_rows:
            if not isinstance(certificate, dict):
                continue
            dimension = str(certificate.get("dimension") or "")
            if dimension in {"symptom", "root_cause"}:
                applied_correct += int(post_values[dimension])

        reports = stage3.get("reports")
        report_rows = reports if isinstance(reports, list) else []
        assessments_by_dimension: dict[str, list[tuple[str, object]]] = {}
        for report in report_rows:
            if not isinstance(report, dict):
                continue
            for assessment in report.get("baseline_revision_assessments", []):
                if not isinstance(assessment, dict):
                    continue
                verdict = str(assessment.get("verdict") or "")
                if verdict:
                    assessment_verdicts[verdict] += 1
                attempted += int(verdict == "revise")
                dimension = str(assessment.get("dimension") or "")
                assessments_by_dimension.setdefault(dimension, []).append(
                    (verdict, assessment.get("proposed_label"))
                )
            for consistency in report.get("revision_consistency", []):
                if isinstance(consistency, dict):
                    consistency_passes += int(consistency.get("status") == "consistent")
        revision_agreement += sum(
            len(values) == 2 and values[0] == values[1] and values[0][0] == "revise"
            for values in assessments_by_dimension.values()
        )
        failures = stage3.get("component_failures")
        operational_failures += len(failures) if isinstance(failures, list) else 0
        unresolved = stage3.get("unresolved")
        stop_reason = (
            str(unresolved.get("stop_reason") or "")
            if isinstance(unresolved, dict)
            else ""
        )
        diagnostics = " ".join(
            diagnostic_strings(
                {
                    "stop_reason": stop_reason,
                    "unresolved": unresolved,
                    "verification": stage3.get("verification"),
                    "pre_gate_verification": stage3.get("pre_gate_verification"),
                    "component_failures": stage3.get("component_failures"),
                    "validity_report": stage3.get("validity_report"),
                }
            )
        )
        explicit_unknown = "unknown" in diagnostics and any(
            token in diagnostics for token in ("citation", "evidence", "id")
        )
        incapable = any(
            token in diagnostics
            for token in (
                "incapable citation",
                "dimension-incapable",
                "not capable",
                "wrong dimension",
                "capability mismatch",
            )
        )
        unknown = explicit_unknown or (
            stop_reason.endswith("invalid_citation") and not incapable
        )
        digest_mismatch = "digest" in diagnostics and any(
            token in diagnostics for token in ("mismatch", "does not match", "invalid")
        )
        trust_or_context = any(
            token in diagnostics
            for token in (
                "trust",
                "context_invalid",
                "context invalid",
                "provenance mismatch",
                "baseline_preservation_gate_failed",
            )
        )
        unknown_citations += int(unknown)
        incapable_citations += int(incapable)
        digest_mismatches += int(digest_mismatch)
        trust_or_context_failures += int(trust_or_context)
        ordinary_reasons = {
            "no_consistent_stage3_candidate",
            "bounded_component_failure",
            "stage3_component_failure",
        }
        hardstop = bool(stop_reason and stop_reason not in ordinary_reasons)
        hardstop_count += int(hardstop)
        preserved = any(
            actions_for_record == "preserved"
            for actions_for_record in (
                preservation.get("symptom_action"),
                preservation.get("root_cause_action"),
            )
        )
        ordinary_preserved += int(preserved and stop_reason in ordinary_reasons)
        hardstop_preserved += int(preserved and hardstop)

    denominator = len(ordered_ids)

    def metric_block(source: str) -> dict[str, object]:
        block = {
            dimension: {
                "correct_count": sum(correctness[source][dimension].values()),
                "accuracy": (
                    sum(correctness[source][dimension].values()) / denominator
                    if denominator
                    else 0.0
                ),
            }
            for dimension in dimensions
        }
        valid_count = sum(source_validity[source].values())
        block.update(
            {
                "coverage": valid_count / denominator if denominator else 0.0,
                "invalid_count": denominator - valid_count,
            }
        )
        return block

    paired_by_dimension: dict[str, dict[str, int]] = {}
    for dimension in dimensions:
        harm_prevented = harm = help_count = 0
        for record_id in ordered_ids:
            if not baseline_valid[record_id]:
                continue
            baseline_is_correct = correctness["baseline"][dimension][record_id]
            pre_is_correct = correctness["pre_gate"][dimension][record_id]
            post_is_correct = correctness["post_gate"][dimension][record_id]
            harm_prevented += int(
                baseline_is_correct and not pre_is_correct and post_is_correct
            )
            harm += int(baseline_is_correct and not post_is_correct)
            help_count += int(not baseline_is_correct and post_is_correct)
        paired_by_dimension[dimension] = {
            "baseline_harm_prevented_count": harm_prevented,
            "baseline_correct_to_final_wrong_count": harm,
            "baseline_wrong_to_final_correct_count": help_count,
            "net_corrections": help_count - harm,
        }

    return {
        "n": denominator,
        "baseline": metric_block("baseline"),
        "pre_gate": metric_block("pre_gate"),
        "post_gate": metric_block("post_gate"),
        "actions": {dimension: dict(counter) for dimension, counter in actions.items()},
        "certificates": {
            "attempted_count": attempted,
            "issued_count": issued,
            "applied_count": applied,
            "revision_precision": applied_correct / applied if applied else None,
        },
        "paired": paired_by_dimension["joint"],
        "paired_by_dimension": paired_by_dimension,
        "roles": {
            "assessment_verdict_counts": dict(sorted(assessment_verdicts.items())),
            "revision_agreement_count": revision_agreement,
            "consistency_pass_count": consistency_passes,
            "operational_failure_count": operational_failures,
        },
        "revision_funnel": _revision_funnel_summary(ordered_ids, gold, predictions),
        "safety": {
            "unknown_citation_count": unknown_citations,
            "incapable_citation_count": incapable_citations,
            "trust_or_context_failure_count": trust_or_context_failures,
            "hardstop_count": hardstop_count,
            "digest_mismatch_count": digest_mismatches,
            "ordinary_uncertainty_preserved_count": ordinary_preserved,
            "hardstop_preserved_count": hardstop_preserved,
        },
    }


def _team_joint(
    report: dict[str, object], record: dict[str, str]
) -> tuple[bool, tuple[object, object]]:
    symptom = report.get("symptom")
    root = report.get("root_cause")
    symptom_label = symptom.get("label") if isinstance(symptom, dict) else None
    root_label = root.get("label") if isinstance(root, dict) else None
    return (
        symptom_label == record.get("symptom")
        and root_label == record.get("root_cause"),
        (symptom_label, root_label),
    )


def _anchor_joint(
    report: dict[str, object], record: dict[str, str]
) -> tuple[bool, tuple[object, object]] | None:
    anchor = report.get("anchor")
    if anchor is None:
        return None
    if not isinstance(anchor, dict):
        raise ValueError("malformed Stage 3 team report anchor")
    symptom = anchor.get("symptom")
    root = anchor.get("root_cause")
    if not isinstance(symptom, dict) or not isinstance(root, dict):
        raise ValueError("malformed Stage 3 team report anchor")
    symptom_label = symptom.get("label")
    root_label = root.get("label")
    if not isinstance(symptom_label, str) or not isinstance(root_label, str):
        raise ValueError("malformed Stage 3 team report anchor")
    return (
        symptom_label == record.get("symptom")
        and root_label == record.get("root_cause"),
        (symptom_label, root_label),
    )


def _accepted_corrections(report: dict[str, object]) -> int:
    correction_audit = report.get("correction_audit")
    if correction_audit in (None, []):
        return 0
    if not isinstance(correction_audit, list) or len(correction_audit) != 2:
        raise ValueError("malformed Stage 3 team report correction audit")
    expected_dimensions = ("symptom", "root_cause")
    accepted = 0
    for entry, expected_dimension in zip(correction_audit, expected_dimensions):
        if (
            not isinstance(entry, dict)
            or entry.get("dimension") != expected_dimension
            or type(entry.get("accepted")) is not bool
        ):
            raise ValueError("malformed Stage 3 team report correction audit")
        accepted += int(entry["accepted"])
    return accepted


def stage3_team_diagnostics(
    records: Sequence[dict[str, str]], rows: Sequence[dict[str, object]]
) -> dict[str, Any]:
    """Measure independent team behavior from serialized Stage 3 audits."""

    predictions = _index_unique(rows, label="predictions")
    gold = _gold_index(records)
    if set(predictions) != set(gold):
        raise ValueError("record sets differ between predictions and gold")

    team_correct = {"A": 0, "B": 0}
    anchor_reach = {"A": 0, "B": 0}
    anchor_correct = {"A": 0, "B": 0}
    agreement = 0
    both_wrong = 0
    composed_oracle_union = 0
    anchor_oracle_union = 0
    anchor_preservation = 0
    reached_anchors = 0
    accepted_corrections = 0
    correction_help = 0
    correction_harm = 0
    correction_neutral_changed = 0
    both_team_reach = 0
    single_team_degraded_reach = 0
    sources: Counter[str] = Counter()
    source_correct: Counter[str] = Counter()
    challenger_actions: Counter[str] = Counter()
    arbitration_flip_help = 0
    arbitration_flip_harm = 0
    for record_id, record in gold.items():
        stage3 = _stage3_audit(predictions[record_id])
        reports = stage3.get("reports")
        if reports in (None, []):
            both_wrong += 1
            continue
        if not isinstance(reports, list):
            raise ValueError(f"record {record_id} has malformed Stage 3 team reports")
        if not all(isinstance(report, dict) for report in reports):
            raise ValueError(f"record {record_id} has malformed Stage 3 team reports")
        team_ids = [str(report.get("team_id")) for report in reports]
        if len(reports) == 1 and team_ids[0] in {"A", "B"}:
            single_team_degraded_reach += 1
            indexed_reports = {team_ids[0]: reports[0]}
            a_correct, a_labels = (
                _team_joint(indexed_reports["A"], record)
                if "A" in indexed_reports
                else (False, (None, None))
            )
            b_correct, b_labels = (
                _team_joint(indexed_reports["B"], record)
                if "B" in indexed_reports
                else (False, (None, None))
            )
        elif (
            len(reports) == 2
            and len(set(team_ids)) == 2
            and set(team_ids) == {"A", "B"}
        ):
            indexed_reports = {str(report.get("team_id")): report for report in reports}
            both_team_reach += 1
            a_correct, a_labels = _team_joint(indexed_reports["A"], record)
            b_correct, b_labels = _team_joint(indexed_reports["B"], record)
        else:
            raise ValueError(f"record {record_id} must contain exactly teams A and B")
        team_correct["A"] += int(a_correct)
        team_correct["B"] += int(b_correct)
        agreement += int(len(reports) == 2 and a_labels == b_labels)
        both_wrong += int(not a_correct and not b_correct)
        composed_oracle_union += int(a_correct or b_correct)
        record_anchor_correct = False
        for team_id, report in indexed_reports.items():
            composed_correct, composed_labels = _team_joint(report, record)
            accepted_corrections += _accepted_corrections(report)
            anchor_result = _anchor_joint(report, record)
            if anchor_result is None:
                continue
            is_anchor_correct, anchor_labels = anchor_result
            anchor_reach[team_id] += 1
            anchor_correct[team_id] += int(is_anchor_correct)
            record_anchor_correct = record_anchor_correct or is_anchor_correct
            reached_anchors += 1
            labels_changed = composed_labels != anchor_labels
            anchor_preservation += int(not labels_changed)
            correction_help += int(not is_anchor_correct and composed_correct)
            correction_harm += int(is_anchor_correct and not composed_correct)
            correction_neutral_changed += int(
                labels_changed and is_anchor_correct == composed_correct
            )
        anchor_oracle_union += int(record_anchor_correct)
        final = stage3.get("final_decision")
        if isinstance(final, dict) and final.get("source"):
            source = str(final["source"])
            final_correct = _joint_correct(predictions[record_id], record)
            sources[source] += 1
            source_correct[source] += int(final_correct)
            if source == "targeted_arbitration":
                arbitration_flip_help += int(
                    final_correct and not (a_correct or b_correct)
                )
                arbitration_flip_harm += int(
                    not final_correct and (a_correct or b_correct)
                )
        challenge = stage3.get("boundary_challenge")
        if isinstance(challenge, dict) and challenge.get("action"):
            challenger_actions[str(challenge["action"])] += 1

    denominator = len(gold)
    return {
        "n": denominator,
        "both_team_reach_count": both_team_reach,
        "both_team_reach_rate": both_team_reach / denominator if denominator else 0.0,
        "single_team_degraded_reach_count": single_team_degraded_reach,
        "single_team_degraded_reach_rate": (
            single_team_degraded_reach / denominator if denominator else 0.0
        ),
        "team_a": {
            "joint_correct_count": team_correct["A"],
            "joint_accuracy": team_correct["A"] / denominator if denominator else 0.0,
            "anchor_reach_count": anchor_reach["A"],
            "anchor_reach_rate": (
                anchor_reach["A"] / denominator if denominator else 0.0
            ),
            "anchor_joint_correct_count": anchor_correct["A"],
            "anchor_joint_accuracy": (
                anchor_correct["A"] / denominator if denominator else 0.0
            ),
            "composed_joint_correct_count": team_correct["A"],
            "composed_joint_accuracy": (
                team_correct["A"] / denominator if denominator else 0.0
            ),
        },
        "team_b": {
            "joint_correct_count": team_correct["B"],
            "joint_accuracy": team_correct["B"] / denominator if denominator else 0.0,
            "anchor_reach_count": anchor_reach["B"],
            "anchor_reach_rate": (
                anchor_reach["B"] / denominator if denominator else 0.0
            ),
            "anchor_joint_correct_count": anchor_correct["B"],
            "anchor_joint_accuracy": (
                anchor_correct["B"] / denominator if denominator else 0.0
            ),
            "composed_joint_correct_count": team_correct["B"],
            "composed_joint_accuracy": (
                team_correct["B"] / denominator if denominator else 0.0
            ),
        },
        "label_agreement_count": agreement,
        "label_agreement_rate": agreement / denominator if denominator else 0.0,
        "both_wrong_count": both_wrong,
        "both_wrong_rate": both_wrong / denominator if denominator else 0.0,
        "anchor_oracle_union_joint_accuracy": (
            anchor_oracle_union / denominator if denominator else 0.0
        ),
        "composed_oracle_union_joint_accuracy": (
            composed_oracle_union / denominator if denominator else 0.0
        ),
        "oracle_union_joint_accuracy": (
            composed_oracle_union / denominator if denominator else 0.0
        ),
        "anchor_preservation_count": anchor_preservation,
        "anchor_preservation_rate": (
            anchor_preservation / reached_anchors if reached_anchors else None
        ),
        "corrections": {
            "accepted_count": accepted_corrections,
            "help_count": correction_help,
            "harm_count": correction_harm,
            "neutral_changed_count": correction_neutral_changed,
            "net_help_count": correction_help - correction_harm,
            "directional_precision": (
                correction_help / (correction_help + correction_harm)
                if correction_help + correction_harm
                else None
            ),
        },
        "resolution_source_counts": dict(sorted(sources.items())),
        "direct_certificate_accuracy": (
            source_correct["direct_consensus"] / sources["direct_consensus"]
            if sources["direct_consensus"]
            else None
        ),
        "challenger_action_counts": dict(sorted(challenger_actions.items())),
        "challenger_yield_count": sum(
            count for action, count in challenger_actions.items() if action != "pass"
        ),
        "challenger_yield_rate": (
            sum(
                count
                for action, count in challenger_actions.items()
                if action != "pass"
            )
            / denominator
            if denominator
            else 0.0
        ),
        "arbitration_flip_help_count": arbitration_flip_help,
        "arbitration_flip_harm_count": arbitration_flip_harm,
    }
