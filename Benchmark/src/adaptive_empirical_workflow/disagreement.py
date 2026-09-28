"""Deterministic disagreement localization and Stage 2 resolution."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from .contracts import (
    ArbitrationSource,
    BoundaryChallenge,
    BoundaryChallengeAction,
    ConsistencyStatus,
    DisagreementMap,
    EvidenceExplicitness,
    EvidenceItem,
    EvidenceReadinessReport,
    EvidenceSufficiency,
    Stage2AnalysisReport,
    Stage2ArbitrationDecision,
    Stage2FinalDecision,
    Stage3ArbitrationDecision,
    Stage3FinalDecision,
    Stage3TeamReport,
)
from .stage2_policy import stage2_acceptance_errors
from .evidence_capabilities import (
    PATCH_SOURCE_TYPES,
    mechanism_capable,
    normalized_source_type,
    observation_capable,
)
from .verification import stage3_report_evidence_ids


def _serialized(value: object) -> object:
    return value.value if hasattr(value, "value") else value


def _normalized_claim(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def _direct_dimension_evidence(
    items: tuple[EvidenceItem, ...],
    *,
    dimension: str,
) -> bool:
    capability = observation_capable if dimension == "symptom" else mechanism_capable
    return any(
        item.explicitness is EvidenceExplicitness.DIRECT and capability(item)
        for item in items
    )


def _claim_is_shared_but_inferred(
    claim_a: str,
    claim_b: str,
    items_a: tuple[EvidenceItem, ...],
    items_b: tuple[EvidenceItem, ...],
    *,
    dimension: str,
) -> bool:
    if _normalized_claim(claim_a) != _normalized_claim(claim_b):
        return False
    capability = observation_capable if dimension == "symptom" else mechanism_capable
    return all(
        any(item.explicitness is EvidenceExplicitness.INFERRED for item in items)
        and not any(
            item.explicitness is EvidenceExplicitness.DIRECT and capability(item)
            for item in items
        )
        for items in (items_a, items_b)
    )


def _stage2_compared_tests(
    report: Stage2AnalysisReport,
    domain: str | None,
) -> dict[str, object]:
    if domain == "ase2022" and report.decision.value == "accepted_fault":
        return {
            name: report.evidence_tests[name]
            for name in ("fault_existence", "scope_exclusion")
        }
    return report.evidence_tests


def build_stage2_disagreement_map(
    report_a: Stage2AnalysisReport,
    report_b: Stage2AnalysisReport,
    *,
    confidence_threshold: float = 0.8,
    domain: str | None = None,
    readiness: EvidenceReadinessReport | None = None,
    evidence_items: tuple[EvidenceItem, ...] | None = None,
) -> DisagreementMap:
    """Compare only decision-relevant fields of two frozen reports."""

    compared = {
        "decision": (report_a.decision, report_b.decision),
        "evidence_tests": (
            _stage2_compared_tests(report_a, domain),
            _stage2_compared_tests(report_b, domain),
        ),
        "evidence_sufficiency": (
            report_a.evidence_sufficiency,
            report_b.evidence_sufficiency,
        ),
        "unresolved_evidence_gaps": (
            report_a.unresolved_evidence_gaps,
            report_b.unresolved_evidence_gaps,
        ),
    }
    details = {
        dimension: {
            "team_a": _serialized(values[0]),
            "team_b": _serialized(values[1]),
        }
        for dimension, values in compared.items()
        if values[0] != values[1]
    }
    if readiness is not None:
        insufficient = [
            dimension.dimension.value
            for dimension in readiness.dimensions
            if not dimension.sufficient
        ]
        if insufficient:
            details["insufficient_required_readiness"] = {"dimensions": insufficient}
    if evidence_items is not None:
        cited_ids = _stage2_cited_ids(report_a, report_b)
        evidence_by_id = {item.evidence_id: item for item in evidence_items}
        unknown_ids = [
            evidence_id
            for evidence_id in cited_ids
            if evidence_id not in evidence_by_id
        ]
        if unknown_ids:
            details["unknown_cited_evidence"] = {"evidence_ids": unknown_ids}
        support_a = _known_items(report_a.supporting_evidence_ids, evidence_by_id)
        support_b = _known_items(report_b.supporting_evidence_ids, evidence_by_id)
        shared_inferred_dimensions = [
            dimension
            for dimension, claim_a, claim_b in (
                (
                    "fault_existence",
                    report_a.fault_claim,
                    report_b.fault_claim,
                ),
                (
                    "repair_causality",
                    report_a.repair_claim,
                    report_b.repair_claim,
                ),
            )
            if report_a.decision is report_b.decision
            and _claim_is_shared_but_inferred(
                claim_a,
                claim_b,
                support_a,
                support_b,
                dimension=(
                    "symptom" if dimension == "fault_existence" else "root_cause"
                ),
            )
        ]
        if shared_inferred_dimensions:
            details["shared_unsupported_inference"] = {
                "dimensions": shared_inferred_dimensions,
                "evidence_ids": [item.evidence_id for item in (*support_a, *support_b)],
            }
    if not details:
        gate_failures = _stage2_consensus_failures(
            report_a,
            report_b,
            confidence_threshold,
            domain=domain,
        )
        if gate_failures:
            details["consensus_gate"] = {
                "team_a": gate_failures,
                "team_b": gate_failures,
            }
    return DisagreementMap(
        dimensions=list(details),
        details=details,
        requires_arbitration=bool(details),
    )


def _known_items(
    evidence_ids: list[str],
    evidence_by_id: dict[str, EvidenceItem],
) -> tuple[EvidenceItem, ...]:
    return tuple(
        evidence_by_id[evidence_id]
        for evidence_id in evidence_ids
        if evidence_id in evidence_by_id
    )


def _stage3_team_cited_ids(team: Stage3TeamReport) -> list[str]:
    return list(stage3_report_evidence_ids(team))


def _stage3_all_cited_ids(
    team_a: Stage3TeamReport,
    team_b: Stage3TeamReport,
) -> list[str]:
    return list(
        dict.fromkeys(_stage3_team_cited_ids(team_a) + _stage3_team_cited_ids(team_b))
    )


def _stage2_cited_ids(
    report_a: Stage2AnalysisReport,
    report_b: Stage2AnalysisReport,
) -> list[str]:
    return list(
        dict.fromkeys(
            report_a.supporting_evidence_ids
            + report_a.counter_evidence_ids
            + report_b.supporting_evidence_ids
            + report_b.counter_evidence_ids
        )
    )


def _unique_evidence(
    report_a: Stage2AnalysisReport,
    report_b: Stage2AnalysisReport,
) -> list[str]:
    return list(
        dict.fromkeys(
            report_a.supporting_evidence_ids + report_b.supporting_evidence_ids
        )
    )


def _stage2_consensus_failures(
    report_a: Stage2AnalysisReport,
    report_b: Stage2AnalysisReport,
    confidence_threshold: float,
    *,
    domain: str | None = None,
) -> list[str]:
    failures: list[str] = []
    if report_a.evidence_sufficiency is not EvidenceSufficiency.SUFFICIENT:
        failures.append("team_a_evidence_insufficient")
    if report_b.evidence_sufficiency is not EvidenceSufficiency.SUFFICIENT:
        failures.append("team_b_evidence_insufficient")
    ase_accepted_policy_met = (
        domain == "ase2022"
        and report_a.decision.value == "accepted_fault"
        and report_b.decision.value == "accepted_fault"
        and not stage2_acceptance_errors(domain, report_a.evidence_tests)
        and not stage2_acceptance_errors(domain, report_b.evidence_tests)
    )
    if not ase_accepted_policy_met:
        if report_a.unresolved_evidence_gaps:
            failures.append("team_a_unresolved_evidence_gaps")
        if report_b.unresolved_evidence_gaps:
            failures.append("team_b_unresolved_evidence_gaps")
        if report_a.confidence < confidence_threshold:
            failures.append("team_a_below_confidence_threshold")
        if report_b.confidence < confidence_threshold:
            failures.append("team_b_below_confidence_threshold")
    return failures


def resolve_stage2_reports(
    report_a: Stage2AnalysisReport,
    report_b: Stage2AnalysisReport,
    *,
    arbitration: Stage2ArbitrationDecision | None = None,
    confidence_threshold: float = 0.8,
    domain: str | None = None,
) -> Stage2FinalDecision | None:
    """Resolve only strong consensus or an explicit targeted arbitration."""

    if arbitration is not None:
        if arbitration.resolution_status.value == "unresolved":
            return None
        assert arbitration.decision is not None
        assert arbitration.confidence is not None
        return Stage2FinalDecision(
            decision=arbitration.decision,
            confidence=arbitration.confidence,
            rationale=arbitration.rationale,
            supporting_evidence_ids=arbitration.supporting_evidence_ids,
            source=ArbitrationSource.TARGETED_ARBITRATION,
        )

    consensus_is_eligible = (
        report_a.decision is report_b.decision
        and _stage2_compared_tests(report_a, domain)
        == _stage2_compared_tests(report_b, domain)
        and not _stage2_consensus_failures(
            report_a,
            report_b,
            confidence_threshold,
            domain=domain,
        )
    )
    if not consensus_is_eligible:
        return None

    return Stage2FinalDecision(
        decision=report_a.decision,
        confidence=min(report_a.confidence, report_b.confidence),
        rationale=(
            "Two independent teams reached the same evidence-sufficient "
            "decision and passed the deterministic consensus gate."
        ),
        supporting_evidence_ids=_unique_evidence(report_a, report_b),
        source=ArbitrationSource.DIRECT_CONSENSUS,
    )


def build_stage3_disagreement_map(
    team_a: Stage3TeamReport,
    team_b: Stage3TeamReport,
    *,
    confidence_threshold: float = 0.8,
    readiness: EvidenceReadinessReport | None = None,
    evidence_items: tuple[EvidenceItem, ...] | None = None,
    taxonomy: Mapping[str, Sequence[str]] | None = None,
) -> DisagreementMap:
    compared = {
        "symptom_label": (team_a.symptom.label, team_b.symptom.label),
        "root_cause_label": (
            team_a.root_cause.label,
            team_b.root_cause.label,
        ),
        "causal_consistency": (
            team_a.consistency.status,
            team_b.consistency.status,
        ),
    }
    details = {
        dimension: {
            "team_a": _serialized(values[0]),
            "team_b": _serialized(values[1]),
        }
        for dimension, values in compared.items()
        if values[0] != values[1]
    }
    evidence_by_id = {item.evidence_id: item for item in (evidence_items or ())}
    all_cited_ids = _stage3_all_cited_ids(team_a, team_b)
    if evidence_items is not None:
        unknown_ids = [
            evidence_id
            for evidence_id in all_cited_ids
            if evidence_id not in evidence_by_id
        ]
        if unknown_ids:
            details["unknown_cited_evidence"] = {"evidence_ids": unknown_ids}

        unsupported_teams: list[str] = []
        for team_name, team in (("team_a", team_a), ("team_b", team_b)):
            root_items = _known_items(
                team.root_cause.supporting_evidence_ids,
                evidence_by_id,
            )
            if not root_items:
                details.setdefault("missing_root_cause_evidence", {})[team_name] = list(
                    team.root_cause.supporting_evidence_ids
                )
            normalized_label = "_".join(team.root_cause.label.lower().split())
            label_is_specific = normalized_label not in {
                "unknown",
                "other",
                "unclear",
                "insufficient_evidence",
            }
            mechanism_supported = any(mechanism_capable(item) for item in root_items)
            if label_is_specific and not mechanism_supported:
                unsupported_teams.append(team_name)

            symptom_items = _known_items(
                team.symptom.supporting_evidence_ids,
                evidence_by_id,
            )
            if not symptom_items:
                details.setdefault("missing_symptom_evidence", {})[team_name] = list(
                    team.symptom.supporting_evidence_ids
                )
            patch_sourced = bool(symptom_items) and all(
                normalized_source_type(item.source_type) in PATCH_SOURCE_TYPES
                for item in symptom_items
            )
            has_observable_impact = any(
                observation_capable(item) for item in symptom_items
            )
            if symptom_items and not has_observable_impact:
                details.setdefault("missing_symptom_observation", {})[team_name] = [
                    item.evidence_id for item in symptom_items
                ]
            if patch_sourced and not has_observable_impact:
                details.setdefault("patch_only_symptom_evidence", {})[team_name] = [
                    item.evidence_id for item in symptom_items
                ]

            if taxonomy is not None:
                for role_name, report in (
                    ("symptom", team.symptom),
                    ("root_cause", team.root_cause),
                ):
                    candidate_ids = {
                        *report.supporting_evidence_ids,
                        *report.counter_evidence_ids,
                    }
                    boundary_ids = report.boundary_evidence_ids
                    boundary_items = _known_items(boundary_ids, evidence_by_id)
                    taxonomy_labels = tuple(taxonomy.get(role_name, ()))
                    has_near_neighbor = len(taxonomy_labels) > 1
                    alternative_is_valid = report.label in taxonomy_labels and (
                        (
                            has_near_neighbor
                            and report.alternative_label != report.label
                            and report.alternative_label in taxonomy_labels
                        )
                        or (
                            not has_near_neighbor
                            and report.alternative_label == report.label
                        )
                    )
                    citations_are_owned = bool(boundary_ids) and set(
                        boundary_ids
                    ).issubset(candidate_ids)
                    boundary_is_supported = alternative_is_valid and (
                        not has_near_neighbor
                        or (
                            citations_are_owned
                            and len(boundary_items) == len(boundary_ids)
                            and _direct_dimension_evidence(
                                boundary_items,
                                dimension=role_name,
                            )
                        )
                    )
                    if not boundary_is_supported:
                        details.setdefault(f"unsupported_{role_name}_boundary", {})[
                            team_name
                        ] = {
                            "alternative_label": report.alternative_label,
                            "boundary_evidence_ids": list(boundary_ids),
                        }

        if unsupported_teams:
            details["unsupported_root_cause_specificity"] = {
                "teams": unsupported_teams,
                "required_evidence": (
                    "code, test, repair, commit, or developer confirmation"
                ),
            }

        shared_inferred_dimensions: list[str] = []
        symptom_a = _known_items(team_a.symptom.supporting_evidence_ids, evidence_by_id)
        symptom_b = _known_items(team_b.symptom.supporting_evidence_ids, evidence_by_id)
        if (
            team_a.symptom.label == team_b.symptom.label
            and _claim_is_shared_but_inferred(
                team_a.symptom.behavior_claim,
                team_b.symptom.behavior_claim,
                symptom_a,
                symptom_b,
                dimension="symptom",
            )
        ):
            shared_inferred_dimensions.append("symptom")
        root_a = _known_items(team_a.root_cause.supporting_evidence_ids, evidence_by_id)
        root_b = _known_items(team_b.root_cause.supporting_evidence_ids, evidence_by_id)
        if (
            team_a.root_cause.label == team_b.root_cause.label
            and _claim_is_shared_but_inferred(
                team_a.root_cause.defect_mechanism,
                team_b.root_cause.defect_mechanism,
                root_a,
                root_b,
                dimension="root_cause",
            )
        ):
            shared_inferred_dimensions.append("root_cause")
        if shared_inferred_dimensions:
            shared_inferred_items: list[EvidenceItem] = []
            if "symptom" in shared_inferred_dimensions:
                shared_inferred_items.extend((*symptom_a, *symptom_b))
            if "root_cause" in shared_inferred_dimensions:
                shared_inferred_items.extend((*root_a, *root_b))
            details["shared_unsupported_inference"] = {
                "dimensions": shared_inferred_dimensions,
                "evidence_ids": list(
                    dict.fromkeys(item.evidence_id for item in shared_inferred_items)
                ),
            }

    failed_consistency = {
        team_name: team.consistency.status.value
        for team_name, team in (("team_a", team_a), ("team_b", team_b))
        if team.consistency.status.value != "consistent"
    }
    if failed_consistency:
        details["failed_causal_consistency"] = failed_consistency
    return DisagreementMap(
        dimensions=list(details),
        details=details,
        requires_arbitration=bool(details),
    )


def add_boundary_challenge_dimension(
    disagreement: DisagreementMap,
    challenge: BoundaryChallenge,
) -> DisagreementMap:
    """Add only review authority owned by the non-classifying challenger."""

    if challenge.action not in (
        BoundaryChallengeAction.SYMPTOM_REVIEW,
        BoundaryChallengeAction.CAUSE_REVIEW,
    ):
        return disagreement
    dimension = challenge.action.value
    details = dict(disagreement.details)
    details[dimension] = {
        "rationale": challenge.rationale,
        "cited_evidence_ids": list(challenge.cited_evidence_ids),
    }
    dimensions = list(disagreement.dimensions)
    if dimension not in dimensions:
        dimensions.append(dimension)
    return DisagreementMap(
        dimensions=dimensions,
        details=details,
        requires_arbitration=True,
    )


def _stage3_consensus_failures(
    team_a: Stage3TeamReport,
    team_b: Stage3TeamReport,
    confidence_threshold: float,
) -> list[str]:
    failures: list[str] = []
    for team_name, team in (("team_a", team_a), ("team_b", team_b)):
        if team.consistency.status.value != "consistent":
            failures.append(f"{team_name}_causal_consistency_failed")
        for role_name, report in (
            ("symptom", team.symptom),
            ("root_cause", team.root_cause),
        ):
            if report.evidence_sufficiency is not EvidenceSufficiency.SUFFICIENT:
                failures.append(f"{team_name}_{role_name}_evidence_insufficient")
            if report.unresolved_evidence_gaps:
                failures.append(f"{team_name}_{role_name}_unresolved_evidence_gaps")
            if report.confidence < confidence_threshold:
                failures.append(f"{team_name}_{role_name}_below_confidence_threshold")
    return failures


def resolve_stage3_reports(
    team_a: Stage3TeamReport,
    team_b: Stage3TeamReport,
    *,
    arbitration: Stage3ArbitrationDecision | None = None,
    confidence_threshold: float = 0.8,
    evidence_items: tuple[EvidenceItem, ...] | None = None,
    force_fallback: bool = False,
) -> Stage3FinalDecision | None:
    if arbitration is not None:
        if arbitration.resolution_status.value == "resolved":
            assert arbitration.symptom_label is not None
            assert arbitration.root_cause_label is not None
            assert arbitration.confidence is not None
            return Stage3FinalDecision(
                symptom_label=arbitration.symptom_label,
                root_cause_label=arbitration.root_cause_label,
                confidence=arbitration.confidence,
                rationale=arbitration.rationale,
                supporting_evidence_ids=arbitration.supporting_evidence_ids,
                source=ArbitrationSource.TARGETED_ARBITRATION,
            )

    reports = (team_a, team_b)
    eligible = (
        not force_fallback
        and not (
            arbitration is not None
            and arbitration.resolution_status.value == "unresolved"
        )
        and team_a.symptom.label == team_b.symptom.label
        and team_a.root_cause.label == team_b.root_cause.label
        and not _stage3_consensus_failures(team_a, team_b, confidence_threshold)
    )
    if eligible:
        evidence_ids = list(
            dict.fromkeys(
                evidence_id
                for team in reports
                for evidence_id in (
                    *team.symptom.supporting_evidence_ids,
                    *team.root_cause.supporting_evidence_ids,
                    *team.consistency.supporting_evidence_ids,
                )
            )
        )
        return Stage3FinalDecision(
            symptom_label=team_a.symptom.label,
            root_cause_label=team_a.root_cause.label,
            confidence=min(
                team_a.symptom.confidence,
                team_a.root_cause.confidence,
                team_b.symptom.confidence,
                team_b.root_cause.confidence,
            ),
            rationale=(
                "Two independent teams agree on both labels with sufficient "
                "evidence and a consistent causal chain."
            ),
            supporting_evidence_ids=evidence_ids,
            source=ArbitrationSource.DIRECT_CONSENSUS,
        )

    evidence_by_id = (
        {item.evidence_id: item for item in evidence_items}
        if evidence_items is not None
        else None
    )

    def fallback_score(
        indexed_team: tuple[int, Stage3TeamReport],
    ) -> tuple[int, int, int, int]:
        index, team = indexed_team
        component_ids = (
            team.symptom.supporting_evidence_ids,
            team.root_cause.supporting_evidence_ids,
            team.consistency.supporting_evidence_ids,
        )
        if evidence_by_id is None:
            coverage = sum(bool(ids) for ids in component_ids)
            explicitness_tier = 0
        else:
            coverage = sum(
                any(evidence_id in evidence_by_id for evidence_id in ids)
                for ids in component_ids
            )
            cited_items = [
                evidence_by_id[evidence_id]
                for ids in component_ids
                for evidence_id in ids
                if evidence_id in evidence_by_id
            ]
            explicitness_tier = int(
                bool(cited_items)
                and all(
                    item.explicitness is EvidenceExplicitness.DIRECT
                    for item in cited_items
                )
            )
        consistency_passed = int(team.consistency.status.value == "consistent")
        return coverage, consistency_passed, explicitness_tier, -index

    eligible_fallback_reports = tuple(
        (index, team)
        for index, team in enumerate(reports)
        if team.consistency.status.value == "consistent"
    )
    if not eligible_fallback_reports:
        return None

    _, selected = max(eligible_fallback_reports, key=fallback_score)
    selected_evidence_ids = tuple(
        dict.fromkeys(
            (
                *selected.symptom.supporting_evidence_ids,
                *selected.root_cause.supporting_evidence_ids,
                *selected.consistency.supporting_evidence_ids,
            )
        )
    )
    return Stage3FinalDecision(
        symptom_label=selected.symptom.label,
        root_cause_label=selected.root_cause.label,
        confidence=min(selected.symptom.confidence, selected.root_cause.confidence),
        rationale=(
            "Valid candidates remained uncertain; the deterministic fallback used "
            "valid citation coverage, causal consistency, evidence explicitness, "
            "and stable team order without increasing confidence."
        ),
        supporting_evidence_ids=selected_evidence_ids,
        source=ArbitrationSource.FALLBACK_UNCERTAIN,
    )


def resolve_single_stage3_report(
    report: Stage3TeamReport,
) -> Stage3FinalDecision | None:
    """Deterministically preserve one complete, consistency-approved team."""

    if report.consistency.status is not ConsistencyStatus.CONSISTENT:
        return None
    evidence_ids = list(
        dict.fromkeys(
            (
                *report.symptom.supporting_evidence_ids,
                *report.root_cause.supporting_evidence_ids,
                *report.consistency.supporting_evidence_ids,
            )
        )
    )
    return Stage3FinalDecision(
        symptom_label=report.symptom.label,
        root_cause_label=report.root_cause.label,
        confidence=min(report.symptom.confidence, report.root_cause.confidence),
        rationale=(
            "One complete independent team passed causal consistency after the "
            "other team's component failure."
        ),
        supporting_evidence_ids=evidence_ids,
        source=ArbitrationSource.SINGLE_TEAM_DEGRADED,
    )
