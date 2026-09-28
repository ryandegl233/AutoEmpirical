"""Deterministic controllers for the adaptive empirical workflow."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import hashlib
import json
import re

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from Benchmark.src.annotation_contracts import label_valid

from .agents import ModelTransportError, StructuredOutputError
from .capabilities import required_readiness_dimensions
from .contracts import (
    ArbitrationSource,
    BaselineAnchor,
    BaselinePreservationResult,
    BaselineRevisionAssessment,
    AnonymousStage3TeamReport,
    BoundaryChallenge,
    BoundaryCard,
    BoundaryChallengeAction,
    CausalConsistencyReport,
    ConsistencyStatus,
    AnonymousStage2Report,
    DisagreementMap,
    DimensionReadiness,
    DimensionVerificationReport,
    EvidenceDelta,
    EvidenceItem,
    EvidenceRequest,
    EvidenceReadinessReport,
    EvidenceValidityIssue,
    EvidenceValidityReport,
    EvidenceView,
    EvidenceDimension,
    EvidenceSufficiency,
    FaultEvidenceAssessment,
    JointAnchorReport,
    RepairCausalityAssessment,
    ResolutionStatus,
    RevisionAssessmentVerdict,
    RevisionAssessorTask,
    RevisionAssessmentAuditAttempt,
    RevisionCertificateAuditEntry,
    RevisionConsistencyReport,
    RevisionConsistencyAuditAttempt,
    RevisionProposalEnvelope,
    RevisionProposalAuditEntry,
    baseline_revision_assessment_digest,
    revision_consistency_report_digest,
    revision_proposal_digest,
    HETEROGENEOUS_REVISION_ASSESSMENT_POLICY,
    TASK_SPECIFIC_REVISION_CROSS_POLICY,
    RootCauseReport,
    ScopeBoundaryAssessment,
    Stage2AnalysisReport,
    Stage2ArbitrationDecision,
    Stage2Decision,
    Stage2ArbitrationPacket,
    Stage2FinalDecision,
    Stage2RoleAssessments,
    Stage2ScopeExclusion,
    Stage2Verification,
    Stage3ArbitrationDecision,
    Stage3ArbitrationPacket,
    Stage3ComponentFailure,
    Stage3FinalDecision,
    Stage3RevisionAudit,
    Stage3EvidencePhaseResult,
    Stage3TeamReport,
    Stage3Verification,
    SymptomReport,
    TeamComposedCandidate,
    TaxonomyStructure,
    LabelRevisionCertificate,
    TestOutcome,
    UnresolvedDecision,
    sanitize_schema_name,
    sanitize_validation_summary,
)
from .baseline_preservation import (
    PreservationGateError,
    apply_baseline_preservation_gate,
    compose_baseline_preservation_decision,
    validate_baseline_preservation_context,
)
from .disagreement import (
    add_boundary_challenge_dimension,
    build_stage2_disagreement_map,
    build_stage3_disagreement_map,
    resolve_stage2_reports,
    resolve_single_stage3_report,
    resolve_stage3_reports,
)
from .ledger import EvidenceLedger
from .frozen_evidence_runtime import derive_frozen_evidence_view
from .evidence_capabilities import (
    assess_stage3_evidence_validity,
    mechanism_capable,
    supports_readiness_dimension,
)
from .specialists import (
    SpecialistRegistry,
    apply_evidence_delta,
    evidence_request_key,
    merge_duplicate_requests,
)
from .verification import (
    readiness_provenance_errors,
    stage3_report_evidence_ids,
    stage3_report_provenance_errors,
    verify_stage2_decision,
    verify_stage3_decision,
)
from .stage2_policy import artifact_scope_exclusion, compose_stage2_decision
from .stage3_composition import (
    boundary_challenge_semantic_errors,
    build_revision_proposals,
    CompositionError,
    compose_baseline_revision_certificates,
    compose_stage3_team_candidate,
)
from .taxonomy_structure import route_boundary_card

Stage2Analyst = Callable[[str, EvidenceView], Stage2AnalysisReport]
FaultEvidenceAnalyst = Callable[[EvidenceView], FaultEvidenceAssessment]
ScopeBoundaryAnalyst = Callable[[EvidenceView], ScopeBoundaryAssessment]
RepairCausalityAnalyst = Callable[[EvidenceView], RepairCausalityAssessment]
EvidenceReadiness = Callable[[str, EvidenceView], EvidenceReadinessReport]
Stage2Arbitrator = Callable[[Stage2ArbitrationPacket], Stage2ArbitrationDecision]

_COMPOSITION_EVIDENCE_ID = re.compile(r"^feg-node-([a-z0-9_-]{1,64})$")


@dataclass(frozen=True)
class ReadinessPhaseResult:
    """Audit state produced by the shared bounded readiness phase."""

    readiness_report: EvidenceReadinessReport
    evidence_deltas: tuple[EvidenceDelta, ...]
    retrieval_source_request_ids: tuple[tuple[str, ...], ...]
    budget_exhausted: bool
    final_ledger_version: int


def _readiness_requests(
    report: EvidenceReadinessReport,
) -> tuple[EvidenceRequest, ...]:
    return tuple(
        request
        for dimension in report.dimensions
        for request in dimension.evidence_requests
    )


def _evidence_is_ready(
    report: EvidenceReadinessReport,
    evidence_items: tuple[EvidenceItem, ...],
    *,
    domain: str,
) -> bool:
    return not readiness_provenance_errors(
        report,
        evidence_items,
        domain=domain,
    )


def _validate_readiness_report(
    report: EvidenceReadinessReport,
    *,
    task: str,
    domain: str,
) -> None:
    if report.task != task:
        raise ValueError("readiness report task must match requested task")
    required = required_readiness_dimensions(domain, task)
    reported = tuple(dimension.dimension for dimension in report.dimensions)
    if len(reported) != len(required) or set(reported) != set(required):
        raise ValueError("readiness dimensions must exactly match required dimensions")


def run_readiness_phase(
    *,
    task: str,
    ledger: EvidenceLedger,
    readiness: EvidenceReadiness,
    specialists: SpecialistRegistry,
    max_retrieval_rounds: int,
    max_specialist_calls: int,
) -> ReadinessPhaseResult:
    """Retrieve only readiness-requested evidence until ready or bounded stop."""

    initial_view = ledger.view()
    report = readiness(task, initial_view)
    _validate_readiness_report(
        report,
        task=task,
        domain=initial_view.domain_profile,
    )

    deltas: list[EvidenceDelta] = []
    retrieval_sources: list[tuple[str, ...]] = []
    processed_requests: set[tuple[object, str, str, str, int]] = set()
    specialist_calls = 0
    budget_exhausted = False
    rounds_used = 0

    while not _evidence_is_ready(
        report,
        ledger.view().items,
        domain=initial_view.domain_profile,
    ):
        merged = tuple(
            request_group
            for request_group in merge_duplicate_requests(_readiness_requests(report))
            if evidence_request_key(request_group.request) not in processed_requests
        )
        if not merged:
            break
        if rounds_used >= max_retrieval_rounds:
            budget_exhausted = True
            break

        rounds_used += 1
        appended_any = False
        for request_group in merged:
            if specialist_calls >= max_specialist_calls:
                budget_exhausted = True
                break
            processed_requests.add(evidence_request_key(request_group.request))
            delta = specialists.run(request_group.request, ledger)
            specialist_calls += 1
            deltas.append(delta)
            retrieval_sources.append(request_group.source_request_ids)
            if apply_evidence_delta(ledger, delta):
                appended_any = True

        if appended_any:
            report = readiness(task, ledger.view())
            _validate_readiness_report(
                report,
                task=task,
                domain=initial_view.domain_profile,
            )
        if budget_exhausted or not appended_any:
            break

    return ReadinessPhaseResult(
        readiness_report=report,
        evidence_deltas=tuple(deltas),
        retrieval_source_request_ids=tuple(retrieval_sources),
        budget_exhausted=budget_exhausted,
        final_ledger_version=ledger.version,
    )


class _FrozenStage3EvidenceLedger(EvidenceLedger):
    """Minimal ledger facade that exposes only one valid immutable snapshot."""

    def __init__(self, view: EvidenceView) -> None:
        super().__init__(
            record_id=view.record_id,
            task=view.task,
            taxonomy=view.model_dump(mode="python")["taxonomy"],
            domain_profile=view.domain_profile,
            initial_items=view.items,
        )
        self._view = view

    def view(self, evidence_ids: Sequence[str] | None = None) -> EvidenceView:
        return derive_frozen_evidence_view(self._view, evidence_ids)


def _stage3_valid_evidence(
    ledger: EvidenceLedger,
    retrieved_quarantine: tuple[EvidenceValidityIssue, ...] = (),
) -> tuple[EvidenceValidityReport, EvidenceView]:
    validity_report = assess_stage3_evidence_validity(ledger.view())
    if retrieved_quarantine:
        quarantined_by_id = {
            issue.evidence_id: issue
            for issue in (*validity_report.quarantined, *retrieved_quarantine)
            if issue.evidence_id not in validity_report.valid_evidence_ids
        }
        validity_report = EvidenceValidityReport(
            valid_evidence_ids=validity_report.valid_evidence_ids,
            quarantined=tuple(quarantined_by_id.values()),
            capabilities_by_evidence_id=(validity_report.capabilities_by_evidence_id),
        )
    return validity_report, ledger.view(validity_report.valid_evidence_ids)


def _screen_stage3_retrieved_items(
    delta: EvidenceDelta,
    *,
    ledger: EvidenceLedger,
    reference_view: EvidenceView,
) -> tuple[tuple[str, ...], tuple[EvidenceValidityIssue, ...]]:
    """Append only deterministically admitted same-record retrieval items."""

    appended_ids: list[str] = []
    quarantined: list[EvidenceValidityIssue] = []
    for item in delta.items:
        if item.evidence_id in ledger.evidence_ids:
            existing = ledger.get(item.evidence_id)
            canonical_identity = (
                "record_id",
                "source_type",
                "source_uri",
                "content",
                "content_sha256",
                "explicitness",
            )
            if any(
                getattr(existing, field) != getattr(item, field)
                for field in canonical_identity
            ):
                raise ValueError(
                    f"evidence_id {item.evidence_id!r} conflicts with existing evidence"
                )
            # Different readiness requests may independently retrieve the same
            # frozen passage. Keep both deltas for audit, but ingest the passage
            # only once so the append-only ledger remains strictly versioned.
            continue
        candidate_view = EvidenceView(
            record_id=reference_view.record_id,
            task=reference_view.task,
            taxonomy=reference_view.taxonomy,
            domain_profile=reference_view.domain_profile,
            ledger_version=reference_view.ledger_version,
            items=(item,),
        )
        item_validity = assess_stage3_evidence_validity(candidate_view)
        if item_validity.valid_evidence_ids:
            ledger.append(item)
            appended_ids.append(item.evidence_id)
        else:
            quarantined.extend(item_validity.quarantined)
    return tuple(appended_ids), tuple(quarantined)


def run_stage3_evidence_phase(
    *,
    ledger: EvidenceLedger,
    readiness: EvidenceReadiness,
    specialists: SpecialistRegistry,
    max_retrieval_rounds: int,
    max_specialist_calls: int,
) -> Stage3EvidencePhaseResult:
    """Run bounded retrieval while treating Stage 3 gaps as diagnostics only."""

    validity_report, valid_view = _stage3_valid_evidence(ledger)
    try:
        report = readiness("stage3", _isolated_classification_view(valid_view))
    except (StructuredOutputError, ModelTransportError) as error:
        report = _stage3_readiness_failure_gap(valid_view, error)
    _validate_readiness_report(
        report,
        task="stage3",
        domain=valid_view.domain_profile,
    )

    deltas: list[EvidenceDelta] = []
    retrieval_sources: list[tuple[str, ...]] = []
    retrieved_quarantine: list[EvidenceValidityIssue] = []
    processed_requests: set[tuple[object, str, str, str, int]] = set()
    specialist_calls = 0
    budget_exhausted = False
    rounds_used = 0

    while not _evidence_is_ready(
        report,
        valid_view.items,
        domain=valid_view.domain_profile,
    ):
        merged = tuple(
            request_group
            for request_group in merge_duplicate_requests(_readiness_requests(report))
            if evidence_request_key(request_group.request) not in processed_requests
        )
        if not merged:
            break
        if rounds_used >= max_retrieval_rounds:
            budget_exhausted = True
            break

        rounds_used += 1
        appended_any = False
        for request_group in merged:
            if specialist_calls >= max_specialist_calls:
                budget_exhausted = True
                break
            processed_requests.add(evidence_request_key(request_group.request))
            delta = specialists.run(
                request_group.request,
                _FrozenStage3EvidenceLedger(valid_view),
            )
            specialist_calls += 1
            deltas.append(delta)
            retrieval_sources.append(request_group.source_request_ids)
            appended_ids, quarantined = _screen_stage3_retrieved_items(
                delta,
                ledger=ledger,
                reference_view=valid_view,
            )
            retrieved_quarantine.extend(quarantined)
            if appended_ids:
                appended_any = True

        validity_report, valid_view = _stage3_valid_evidence(
            ledger,
            tuple(retrieved_quarantine),
        )
        if appended_any:
            try:
                report = readiness(
                    "stage3",
                    _isolated_classification_view(valid_view),
                )
            except (StructuredOutputError, ModelTransportError) as error:
                report = _stage3_readiness_failure_gap(valid_view, error)
            _validate_readiness_report(
                report,
                task="stage3",
                domain=valid_view.domain_profile,
            )
        if budget_exhausted or not appended_any:
            break

    return Stage3EvidencePhaseResult(
        validity_report=validity_report,
        gap_report=report,
        evidence_deltas=tuple(deltas),
        retrieval_source_request_ids=tuple(retrieval_sources),
        budget_exhausted=budget_exhausted,
        final_ledger_version=ledger.version,
    )


def _stage3_readiness_failure_gap(
    view: EvidenceView,
    error: StructuredOutputError | ModelTransportError,
) -> EvidenceReadinessReport:
    """Create an auditable diagnostic gap for declared model failures only."""

    failure_kind = (
        "schema validation"
        if isinstance(error, StructuredOutputError)
        else "model transport"
    )
    valid_ids = tuple(item.evidence_id for item in view.items)
    evidence_summary = (
        f" Valid frozen evidence available for classification: {valid_ids!r}."
        if valid_ids
        else " No valid frozen evidence is available for classification."
    )
    missing_fact = (
        f"Stage 3 readiness assessment unavailable after recoverable "
        f"{failure_kind} failure.{evidence_summary}"
    )
    return EvidenceReadinessReport(
        task="stage3",
        dimensions=tuple(
            DimensionReadiness(
                dimension=dimension,
                sufficient=False,
                confirmed_evidence_ids=(),
                missing_facts=(missing_fact,),
                evidence_requests=(),
            )
            for dimension in required_readiness_dimensions(
                view.domain_profile,
                "stage3",
            )
        ),
    )


def _unresolved(
    phase: ReadinessPhaseResult,
    evidence_items: tuple[EvidenceItem, ...],
    *,
    domain: str,
) -> UnresolvedDecision | None:
    known_ids = {item.evidence_id for item in evidence_items}
    unresolved_dimensions = tuple(
        dimension
        for dimension in phase.readiness_report.dimensions
        if not dimension.sufficient
        or not dimension.confirmed_evidence_ids
        or any(
            evidence_id not in known_ids
            for evidence_id in dimension.confirmed_evidence_ids
        )
        or (
            bool(dimension.confirmed_evidence_ids)
            and not all(
                supports_readiness_dimension(
                    item,
                    dimension.dimension,
                    domain=domain,
                )
                for item in evidence_items
                if item.evidence_id in dimension.confirmed_evidence_ids
            )
        )
    )
    if not unresolved_dimensions:
        return None
    provenance_invalid = any(
        dimension.sufficient
        and (
            not dimension.confirmed_evidence_ids
            or any(
                evidence_id not in known_ids
                for evidence_id in dimension.confirmed_evidence_ids
            )
            or not all(
                supports_readiness_dimension(
                    item,
                    dimension.dimension,
                    domain=domain,
                )
                for item in evidence_items
                if item.evidence_id in dimension.confirmed_evidence_ids
            )
        )
        for dimension in unresolved_dimensions
    )
    missing_facts: list[str] = []
    for dimension in unresolved_dimensions:
        if dimension.missing_facts:
            missing_facts.extend(dimension.missing_facts)
            continue
        if not dimension.confirmed_evidence_ids:
            missing_facts.append(
                f"Required readiness dimension {dimension.dimension.value} "
                "has no confirmed evidence."
            )
            continue
        unknown = [
            evidence_id
            for evidence_id in dimension.confirmed_evidence_ids
            if evidence_id not in known_ids
        ]
        missing_facts.extend(
            f"Required readiness dimension {dimension.dimension.value} cites "
            f"unknown evidence_id {evidence_id!r}."
            for evidence_id in unknown
        )
        known_dimension_items = tuple(
            item
            for item in evidence_items
            if item.evidence_id in dimension.confirmed_evidence_ids
        )
        incompatible_ids = tuple(
            item.evidence_id
            for item in known_dimension_items
            if not supports_readiness_dimension(
                item,
                dimension.dimension,
                domain=domain,
            )
        )
        if incompatible_ids:
            missing_facts.append(
                f"Required readiness dimension {dimension.dimension.value} cites "
                "capability-incompatible confirmed evidence IDs: "
                f"{', '.join(incompatible_ids)}."
            )
    return UnresolvedDecision(
        status=ResolutionStatus.UNRESOLVED,
        dimensions=tuple(dimension.dimension for dimension in unresolved_dimensions),
        missing_facts=tuple(missing_facts),
        stop_reason=(
            "evidence_readiness_invalid_provenance"
            if provenance_invalid
            else (
                "evidence_readiness_budget_exhausted"
                if phase.budget_exhausted
                else "evidence_readiness_no_progress"
            )
        ),
        attempted_retrieval_statuses=tuple(
            delta.status for delta in phase.evidence_deltas
        ),
    )


def _arbitration_unresolved(
    disagreement: DisagreementMap,
    phase: ReadinessPhaseResult | Stage3EvidencePhaseResult,
    *,
    arbitration: Stage2ArbitrationDecision | Stage3ArbitrationDecision | None = None,
    missing_facts: tuple[str, ...] = (),
    stop_reason: str = "arbitration_evidence_insufficient",
) -> UnresolvedDecision:
    active_dimensions = tuple(disagreement.dimensions)
    supplied_unresolved = tuple(
        arbitration.unresolved_dimensions if arbitration is not None else ()
    )
    supplied_is_authorized = (
        arbitration is not None
        and arbitration.resolution_status is ResolutionStatus.UNRESOLVED
        and bool(supplied_unresolved)
        and set(supplied_unresolved).issubset(active_dimensions)
    )
    dimensions = supplied_unresolved if supplied_is_authorized else active_dimensions
    facts = tuple(
        arbitration.missing_facts
        if supplied_is_authorized and arbitration is not None
        else missing_facts
    )
    if not facts:
        facts = tuple(
            f"Arbitration could not resolve required dimension {dimension}."
            for dimension in dimensions
        )
    return UnresolvedDecision(
        status=ResolutionStatus.UNRESOLVED,
        dimensions=dimensions,
        missing_facts=facts,
        stop_reason=stop_reason,
        attempted_retrieval_statuses=tuple(
            delta.status for delta in phase.evidence_deltas
        ),
    )


def _arbitration_scope_errors(
    disagreement: DisagreementMap,
    arbitration: Stage2ArbitrationDecision | Stage3ArbitrationDecision,
    reports: (
        tuple[Stage2AnalysisReport, Stage2AnalysisReport]
        | tuple[Stage3TeamReport, Stage3TeamReport]
    ),
) -> tuple[str, ...]:
    active = set(disagreement.dimensions)
    if arbitration.resolution_status is ResolutionStatus.UNRESOLVED:
        unresolved = set(arbitration.unresolved_dimensions)
        if not unresolved or not unresolved.issubset(active):
            return (
                "Arbitration returned unresolved dimensions outside packet authority.",
            )
        return ()

    errors: list[str] = []
    resolved = set(arbitration.resolved_dimensions)
    if resolved != active:
        errors.append(
            "Resolved arbitration must account for every active packet dimension."
        )
    if isinstance(arbitration, Stage2ArbitrationDecision):
        if "decision" not in active and any(
            report.decision is not arbitration.decision for report in reports
        ):
            errors.append(
                "Arbitration changed Stage 2 decision without decision authority."
            )
        return tuple(errors)

    stage3_reports = reports
    if "symptom_label" not in active and any(
        report.symptom.label != arbitration.symptom_label for report in stage3_reports
    ):
        errors.append(
            "Arbitration changed symptom label without symptom_label authority."
        )
    if "root_cause_label" not in active and any(
        report.root_cause.label != arbitration.root_cause_label
        for report in stage3_reports
    ):
        errors.append(
            "Arbitration changed root-cause label without root_cause_label authority."
        )
    return tuple(errors)


def _stage2_relevant_ids(
    reports: tuple[Stage2AnalysisReport, Stage2AnalysisReport],
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            evidence_id
            for report in reports
            for evidence_id in (
                report.supporting_evidence_ids + report.counter_evidence_ids
            )
        )
    )


def _stage3_relevant_ids(
    reports: tuple[Stage3TeamReport, Stage3TeamReport],
    disagreement: DisagreementMap,
    challenge: BoundaryChallenge | None = None,
) -> tuple[str, ...]:
    dimensions = list(disagreement.dimensions)
    shared_detail = disagreement.details.get("shared_unsupported_inference", {})
    if isinstance(shared_detail, Mapping):
        dimensions.extend(
            str(dimension) for dimension in shared_detail.get("dimensions", ())
        )
    cited: list[str] = []
    for report in reports:
        cited.extend(stage3_report_evidence_ids(report, dimensions=dimensions))
    if challenge is not None and challenge.action in (
        BoundaryChallengeAction.SYMPTOM_REVIEW,
        BoundaryChallengeAction.CAUSE_REVIEW,
    ):
        cited.extend(challenge.cited_evidence_ids)
    return tuple(dict.fromkeys(cited))


def _stage3_candidate_ids(
    reports: tuple[Stage3TeamReport, ...],
) -> tuple[str, ...]:
    """Return the complete candidate citation union before packet narrowing."""

    return tuple(
        dict.fromkeys(
            evidence_id
            for report in reports
            for evidence_id in stage3_report_evidence_ids(report)
        )
    )


def _stage3_candidate_taxonomy_errors(
    reports: tuple[Stage3TeamReport, ...],
    taxonomy: Mapping[str, Sequence[str]],
) -> tuple[str, ...]:
    errors: list[str] = []
    for report in reports:
        for dimension, candidate in (
            ("symptom", report.symptom),
            ("root_cause", report.root_cause),
        ):
            allowed = taxonomy.get(dimension, ())
            if not label_valid(taxonomy, dimension, candidate.label):
                errors.append(
                    f"team {report.team_id} returned unknown {dimension} "
                    f"label {candidate.label!r}"
                )
            if not label_valid(taxonomy, dimension, candidate.alternative_label):
                errors.append(
                    f"team {report.team_id} returned unknown {dimension} "
                    f"alternative label {candidate.alternative_label!r}"
                )
    return tuple(errors)


def _boundary_challenge_errors(
    challenge: BoundaryChallenge,
    reports: tuple[Stage3TeamReport, Stage3TeamReport],
    view: EvidenceView,
) -> tuple[str, ...]:
    return boundary_challenge_semantic_errors(challenge, reports, view)


def _boundary_challenge_unresolved(
    phase: ReadinessPhaseResult | Stage3EvidencePhaseResult,
    challenge: BoundaryChallenge,
    *,
    stop_reason: str,
    missing_facts: tuple[str, ...],
) -> UnresolvedDecision:
    return UnresolvedDecision(
        status=ResolutionStatus.UNRESOLVED,
        dimensions=(challenge.action.value,),
        missing_facts=missing_facts,
        stop_reason=stop_reason,
        attempted_retrieval_statuses=tuple(
            delta.status for delta in phase.evidence_deltas
        ),
    )


def _resolve_relevant_evidence(
    view: EvidenceView,
    evidence_ids: tuple[str, ...],
) -> tuple[tuple[EvidenceItem, ...], tuple[str, ...]]:
    evidence_by_id = {item.evidence_id: item for item in view.items}
    missing = tuple(
        evidence_id for evidence_id in evidence_ids if evidence_id not in evidence_by_id
    )
    relevant = tuple(
        EvidenceItem.model_validate(
            evidence_by_id[evidence_id].model_dump(mode="python")
        )
        for evidence_id in evidence_ids
        if evidence_id in evidence_by_id
    )
    return relevant, missing


def _assessment_evidence_ids(assessment: object) -> tuple[str, ...]:
    supporting = getattr(assessment, "supporting_evidence_ids", ())
    counter = getattr(assessment, "counter_evidence_ids", ())
    return tuple(dict.fromkeys((*supporting, *counter)))


def _stage2_role_assessment_errors(
    domain: str,
    assessments: Stage2RoleAssessments,
    view: EvidenceView,
) -> tuple[str, ...]:
    evidence_by_id = {item.evidence_id: item for item in view.items}
    owned = (
        (
            "fault_existence",
            EvidenceDimension.FAULT_EXISTENCE,
            assessments.fault_evidence,
        ),
        (
            "study_scope",
            EvidenceDimension.STUDY_SCOPE,
            assessments.scope_boundary,
        ),
        (
            "repair_causality",
            EvidenceDimension.REPAIR_CAUSALITY,
            assessments.repair_causality,
        ),
    )
    errors: list[str] = []
    if domain == "issta2024" and assessments.repair_causality is None:
        errors.append("issta2024 requires a repair_causality assessment")
    if domain == "ase2022" and assessments.repair_causality is not None:
        errors.append("ase2022 minimal composition must not include repair analysis")
    for name, dimension, assessment in owned:
        if assessment is None:
            continue
        cited_ids = _assessment_evidence_ids(assessment)
        if not cited_ids:
            errors.append(f"{name} assessment cites no evidence")
            continue
        for evidence_id in cited_ids:
            item = evidence_by_id.get(evidence_id)
            if item is None:
                errors.append(
                    f"{name} assessment cites unknown evidence_id {evidence_id!r}"
                )
            elif not supports_readiness_dimension(
                item,
                dimension,
                domain=domain,
            ):
                errors.append(
                    f"{name} assessment cites capability-incompatible "
                    f"evidence_id {evidence_id!r}"
                )
    required_assessments = (
        assessments.fault_evidence,
        assessments.scope_boundary,
        *((assessments.repair_causality,) if domain == "issta2024" else ()),
    )
    errors.extend(
        "required role assessment returned unknown outcome"
        for assessment in required_assessments
        if assessment is not None and assessment.outcome is TestOutcome.UNKNOWN
    )
    return tuple(dict.fromkeys(errors))


def _stage2_tests(assessments: Stage2RoleAssessments) -> dict[str, TestOutcome]:
    return {
        "fault_existence": assessments.fault_evidence.outcome,
        "scope_exclusion": assessments.scope_boundary.outcome,
        "repair_causality": (
            assessments.repair_causality.outcome
            if assessments.repair_causality is not None
            else TestOutcome.UNKNOWN
        ),
    }


def _compose_stage2_report(
    domain: str,
    assessments: Stage2RoleAssessments,
) -> Stage2AnalysisReport:
    tests = _stage2_tests(assessments)
    all_assessments = tuple(
        assessment
        for assessment in (
            assessments.fault_evidence,
            assessments.scope_boundary,
            assessments.repair_causality,
        )
        if assessment is not None
    )
    supporting = list(
        dict.fromkeys(
            evidence_id
            for assessment in all_assessments
            for evidence_id in assessment.supporting_evidence_ids
        )
    )
    counter = list(
        dict.fromkeys(
            evidence_id
            for assessment in all_assessments
            for evidence_id in assessment.counter_evidence_ids
        )
    )
    if not supporting:
        supporting = list(dict.fromkeys(counter))
    return Stage2AnalysisReport(
        team_id="policy_composition",
        decision=compose_stage2_decision(domain, tests),
        confidence=1.0,
        fault_claim=assessments.fault_evidence.claim,
        repair_claim=(
            assessments.repair_causality.claim
            if assessments.repair_causality is not None
            else ("Repair causality is optional and was not assessed for ASE2022."
                  if domain == "ase2022" else f"Repair causality is optional and was not assessed for {domain}.")
        ),
        supporting_evidence_ids=supporting,
        counter_evidence_ids=counter,
        evidence_tests=tests,
        alternative_hypothesis=(
            "Each named test preserves its counter-evidence without model decision authority."
        ),
        decision_boundary=(
            f"The final {domain} decision is composed only by deterministic named-test policy."
        ),
        evidence_sufficiency=EvidenceSufficiency.SUFFICIENT,
        unresolved_evidence_gaps=[],
        evidence_requests=[],
    )


class Stage2WorkflowConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_retrieval_rounds: int = Field(default=2, ge=0, le=10)
    max_specialist_calls: int = Field(default=6, ge=0, le=100)
    consensus_confidence: float = Field(default=0.8, ge=0.0, le=1.0)


@dataclass(frozen=True)
class Stage2WorkflowResult:
    reports: tuple[Stage2AnalysisReport, ...]
    final_decision: Stage2FinalDecision | None
    verification: Stage2Verification | None
    evidence_deltas: tuple[EvidenceDelta, ...]
    retrieval_source_request_ids: tuple[tuple[str, ...], ...]
    budget_exhausted: bool
    scope_exclusion: Stage2ScopeExclusion | None = None
    readiness_report: EvidenceReadinessReport | None = None
    classification_ledger_version: int | None = None
    unresolved: UnresolvedDecision | None = None
    role_assessments: Stage2RoleAssessments | None = None


def _anonymous(report: Stage2AnalysisReport) -> AnonymousStage2Report:
    return AnonymousStage2Report.model_validate(
        report.model_dump(exclude={"team_id", "evidence_requests"})
    )


def _isolated_classification_view(view: EvidenceView) -> EvidenceView:
    """Rebuild one role-owned view without exposing the canonical instance."""

    return derive_frozen_evidence_view(view)


class Stage2Controller:
    """Runs isolated teams, bounded retrieval, arbitration, and verification."""

    def __init__(
        self,
        *,
        analyst: Stage2Analyst | None = None,
        fault_evidence_analyst: FaultEvidenceAnalyst | None = None,
        scope_boundary_analyst: ScopeBoundaryAnalyst | None = None,
        repair_causality_analyst: RepairCausalityAnalyst | None = None,
        readiness: EvidenceReadiness,
        specialists: SpecialistRegistry | None = None,
        arbitrator: Stage2Arbitrator | None = None,
        config: Stage2WorkflowConfig | None = None,
    ) -> None:
        any_role_owned = any(
            candidate is not None
            for candidate in (
                fault_evidence_analyst,
                scope_boundary_analyst,
                repair_causality_analyst,
            )
        )
        if analyst is not None and any_role_owned:
            raise ValueError("legacy analyst cannot be mixed with role-owned analysts")
        if analyst is None and not (
            fault_evidence_analyst is not None and scope_boundary_analyst is not None
        ):
            raise ValueError(
                "role-owned analysts must provide fault evidence and scope boundary "
                "together, with repair causality only where domain policy requires it"
            )
        self._analyst = analyst
        self._fault_evidence_analyst = fault_evidence_analyst
        self._scope_boundary_analyst = scope_boundary_analyst
        self._repair_causality_analyst = repair_causality_analyst
        self._readiness = readiness
        self._specialists = specialists or SpecialistRegistry()
        self._arbitrator = arbitrator
        self._config = config or Stage2WorkflowConfig()

    def _analyze_teams(
        self, view: EvidenceView
    ) -> tuple[Stage2AnalysisReport, Stage2AnalysisReport]:
        if self._analyst is None:
            raise RuntimeError("legacy team analysis is not configured")
        view_a = _isolated_classification_view(view)
        view_b = _isolated_classification_view(view)
        with ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="stage2-team"
        ) as executor:
            future_a = executor.submit(self._analyst, "A", view_a)
            future_b = executor.submit(self._analyst, "B", view_b)
            report_a = future_a.result()
            report_b = future_b.result()
        if report_a.team_id != "A" or report_b.team_id != "B":
            raise ValueError("analyst report team_id does not match assigned team")
        return report_a, report_b

    def _analyze_roles(
        self,
        view: EvidenceView,
    ) -> Stage2RoleAssessments:
        assert self._fault_evidence_analyst is not None
        assert self._scope_boundary_analyst is not None
        include_repair = view.domain_profile == "issta2024"
        if include_repair and self._repair_causality_analyst is None:
            raise ValueError("issta2024 requires repair_causality_analyst")
        with ThreadPoolExecutor(
            max_workers=3 if include_repair else 2,
            thread_name_prefix="stage2-role",
        ) as executor:
            fault_future = executor.submit(
                self._fault_evidence_analyst,
                _isolated_classification_view(view),
            )
            scope_future = executor.submit(
                self._scope_boundary_analyst,
                _isolated_classification_view(view),
            )
            repair_future = (
                executor.submit(
                    self._repair_causality_analyst,
                    _isolated_classification_view(view),
                )
                if include_repair and self._repair_causality_analyst is not None
                else None
            )
            return Stage2RoleAssessments(
                fault_evidence=fault_future.result(),
                scope_boundary=scope_future.result(),
                repair_causality=(
                    repair_future.result() if repair_future is not None else None
                ),
            )

    @staticmethod
    def _scope_excluded_result(
        ledger: EvidenceLedger,
        boundary: str,
        evidence_id: str,
        source_uri: str,
    ) -> Stage2WorkflowResult:
        final_decision = Stage2FinalDecision(
            decision=Stage2Decision.REJECTED,
            confidence=1.0,
            rationale=boundary,
            supporting_evidence_ids=[evidence_id],
            source=ArbitrationSource.DETERMINISTIC_SCOPE_GATE,
        )
        return Stage2WorkflowResult(
            reports=(),
            final_decision=final_decision,
            verification=verify_stage2_decision(
                final_decision,
                ledger,
            ),
            evidence_deltas=(),
            retrieval_source_request_ids=[],
            budget_exhausted=False,
            readiness_report=None,
            classification_ledger_version=None,
            unresolved=None,
            scope_exclusion=Stage2ScopeExclusion(
                rule_id="ase2022_issue_artifact_only",
                boundary=boundary,
                evidence_id=evidence_id,
                source_uri=source_uri,
            ),
        )

    def run(self, ledger: EvidenceLedger) -> Stage2WorkflowResult:
        view = ledger.view()
        domain = view.domain_profile
        if self._analyst is None:
            if domain == "ase2022" and self._repair_causality_analyst is not None:
                raise ValueError(
                    "ase2022 role-owned mode must omit repair_causality_analyst"
                )
            if domain == "issta2024" and self._repair_causality_analyst is None:
                raise ValueError(
                    "issta2024 role-owned mode requires repair_causality_analyst"
                )
        candidate_items = tuple(
            item for item in view.items if item.source_type == "record_summary"
        )
        if candidate_items:
            candidate = candidate_items[0]
            scope_boundary = artifact_scope_exclusion(
                domain,
                candidate.source_uri,
            )
            if scope_boundary is not None:
                return self._scope_excluded_result(
                    ledger,
                    scope_boundary,
                    candidate.evidence_id,
                    candidate.source_uri,
                )

        readiness_phase = run_readiness_phase(
            task="stage2",
            ledger=ledger,
            readiness=self._readiness,
            specialists=self._specialists,
            max_retrieval_rounds=self._config.max_retrieval_rounds,
            max_specialist_calls=self._config.max_specialist_calls,
        )
        canonical_classification_view = ledger.view()
        unresolved = _unresolved(
            readiness_phase,
            canonical_classification_view.items,
            domain=canonical_classification_view.domain_profile,
        )
        if unresolved is not None:
            return Stage2WorkflowResult(
                reports=(),
                final_decision=None,
                verification=None,
                evidence_deltas=readiness_phase.evidence_deltas,
                retrieval_source_request_ids=list(
                    readiness_phase.retrieval_source_request_ids
                ),
                budget_exhausted=readiness_phase.budget_exhausted,
                readiness_report=readiness_phase.readiness_report,
                classification_ledger_version=None,
                unresolved=unresolved,
            )

        classification_ledger_version = canonical_classification_view.ledger_version
        if self._analyst is None:
            assessments = self._analyze_roles(canonical_classification_view)
            assessment_errors = _stage2_role_assessment_errors(
                domain,
                assessments,
                canonical_classification_view,
            )
            if assessment_errors:
                return Stage2WorkflowResult(
                    reports=(),
                    final_decision=None,
                    verification=None,
                    evidence_deltas=readiness_phase.evidence_deltas,
                    retrieval_source_request_ids=list(
                        readiness_phase.retrieval_source_request_ids
                    ),
                    budget_exhausted=readiness_phase.budget_exhausted,
                    readiness_report=readiness_phase.readiness_report,
                    classification_ledger_version=classification_ledger_version,
                    unresolved=UnresolvedDecision(
                        status=ResolutionStatus.UNRESOLVED,
                        dimensions=tuple(
                            dimension.value
                            for dimension in required_readiness_dimensions(
                                domain,
                                "stage2",
                            )
                        ),
                        missing_facts=assessment_errors,
                        stop_reason="stage2_role_assessment_invalid",
                        attempted_retrieval_statuses=tuple(
                            delta.status for delta in readiness_phase.evidence_deltas
                        ),
                    ),
                    role_assessments=assessments,
                )
            report = _compose_stage2_report(domain, assessments)
            final_decision = Stage2FinalDecision(
                decision=report.decision,
                confidence=1.0,
                rationale=(
                    "Deterministic paper policy composed the role-owned named-test outcomes."
                ),
                supporting_evidence_ids=report.supporting_evidence_ids,
                source=ArbitrationSource.POLICY_COMPOSITION,
            )
            verification = verify_stage2_decision(
                final_decision,
                ledger,
                (report,),
                readiness=readiness_phase.readiness_report,
            )
            return Stage2WorkflowResult(
                reports=(report,),
                final_decision=final_decision,
                verification=verification,
                evidence_deltas=readiness_phase.evidence_deltas,
                retrieval_source_request_ids=list(
                    readiness_phase.retrieval_source_request_ids
                ),
                budget_exhausted=readiness_phase.budget_exhausted,
                readiness_report=readiness_phase.readiness_report,
                classification_ledger_version=classification_ledger_version,
                unresolved=None,
                role_assessments=assessments,
            )
        reports = self._analyze_teams(canonical_classification_view)

        disagreement = build_stage2_disagreement_map(
            *reports,
            confidence_threshold=self._config.consensus_confidence,
            domain=domain,
            readiness=readiness_phase.readiness_report,
            evidence_items=canonical_classification_view.items,
        )
        arbitration: Stage2ArbitrationDecision | None = None
        arbitration_evidence_ids: tuple[str, ...] | None = None
        final_decision = (
            None
            if disagreement.requires_arbitration
            else resolve_stage2_reports(
                reports[0],
                reports[1],
                confidence_threshold=self._config.consensus_confidence,
                domain=domain,
            )
        )
        if disagreement.requires_arbitration and self._arbitrator is None:
            unresolved = _arbitration_unresolved(
                disagreement,
                readiness_phase,
            )
        elif disagreement.requires_arbitration:
            relevant_evidence, missing_ids = _resolve_relevant_evidence(
                canonical_classification_view,
                _stage2_relevant_ids(reports),
            )
            if missing_ids:
                unresolved = _arbitration_unresolved(
                    disagreement,
                    readiness_phase,
                    missing_facts=tuple(
                        f"Candidate report cites unknown evidence_id {evidence_id!r}."
                        for evidence_id in missing_ids
                    ),
                )
            else:
                arbitration_evidence_ids = tuple(
                    item.evidence_id for item in relevant_evidence
                )
                packet = Stage2ArbitrationPacket(
                    domain_profile=canonical_classification_view.domain_profile,
                    taxonomy=canonical_classification_view.taxonomy,
                    disagreement=disagreement,
                    report_a=_anonymous(reports[0]),
                    report_b=_anonymous(reports[1]),
                    classification_ledger_version=classification_ledger_version,
                    relevant_evidence=relevant_evidence,
                )
                arbitration = self._arbitrator(packet)
                arbitration_scope_errors = _arbitration_scope_errors(
                    disagreement,
                    arbitration,
                    reports,
                )
                if arbitration_scope_errors:
                    unresolved = _arbitration_unresolved(
                        disagreement,
                        readiness_phase,
                        arbitration=arbitration,
                        missing_facts=arbitration_scope_errors,
                    )
                elif arbitration.resolution_status is ResolutionStatus.UNRESOLVED:
                    unresolved = _arbitration_unresolved(
                        disagreement,
                        readiness_phase,
                        arbitration=arbitration,
                    )
                else:
                    final_decision = resolve_stage2_reports(
                        reports[0],
                        reports[1],
                        arbitration=arbitration,
                        domain=domain,
                    )

        verification = (
            verify_stage2_decision(
                final_decision,
                ledger,
                reports,
                readiness=readiness_phase.readiness_report,
                disagreement=disagreement,
                arbitration=arbitration,
                arbitration_evidence_ids=arbitration_evidence_ids,
            )
            if final_decision is not None
            else None
        )
        return Stage2WorkflowResult(
            reports=reports,
            final_decision=final_decision,
            verification=verification,
            evidence_deltas=readiness_phase.evidence_deltas,
            retrieval_source_request_ids=list(
                readiness_phase.retrieval_source_request_ids
            ),
            budget_exhausted=readiness_phase.budget_exhausted,
            readiness_report=readiness_phase.readiness_report,
            classification_ledger_version=classification_ledger_version,
            unresolved=unresolved,
        )


Stage3SymptomAnalyst = Callable[[str, EvidenceView], SymptomReport]
Stage3RootCauseAnalyst = Callable[[str, EvidenceView], RootCauseReport]
Stage3JointAnchor = Callable[[str, EvidenceView], JointAnchorReport]
Stage3DimensionVerifier = Callable[
    [str, JointAnchorReport, EvidenceView],
    DimensionVerificationReport,
]
Stage3ConsistencyChecker = Callable[
    [str, SymptomReport, RootCauseReport, EvidenceView],
    CausalConsistencyReport,
]
Stage3BoundaryChallenger = Callable[
    [tuple[Stage3TeamReport, Stage3TeamReport], EvidenceView],
    BoundaryChallenge,
]
Stage3Arbitrator = Callable[[Stage3ArbitrationPacket], Stage3ArbitrationDecision]
Stage3BaselineRevisionAssessor = Callable[
    [str, RevisionProposalEnvelope, BaselineAnchor, BoundaryCard, EvidenceView],
    BaselineRevisionAssessment,
]
Stage3BaselineRevisionConsistency = Callable[
    [
        str,
        str,
        RevisionProposalEnvelope,
        BaselineRevisionAssessment,
        BoundaryCard,
        EvidenceView,
    ],
    RevisionConsistencyReport,
]


class Stage3WorkflowConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_retrieval_rounds: int = Field(default=2, ge=0, le=10)
    max_specialist_calls: int = Field(default=6, ge=0, le=100)
    consensus_confidence: float = Field(default=0.8, ge=0.0, le=1.0)


@dataclass(frozen=True)
class Stage3WorkflowResult:
    reports: tuple[Stage3TeamReport, ...]
    final_decision: Stage3FinalDecision | None
    verification: Stage3Verification | None
    evidence_deltas: tuple[EvidenceDelta, ...]
    retrieval_source_request_ids: tuple[tuple[str, ...], ...]
    budget_exhausted: bool
    validity_report: EvidenceValidityReport
    readiness_report: EvidenceReadinessReport | None = None
    classification_ledger_version: int | None = None
    unresolved: UnresolvedDecision | None = None
    boundary_challenge: BoundaryChallenge | None = None
    supervision: dict | None = None
    component_failures: tuple[Stage3ComponentFailure, ...] = ()
    pre_gate_candidate: Stage3FinalDecision | None = None
    pre_gate_verification: Stage3Verification | None = None
    pre_gate_disagreement: DisagreementMap | None = None
    pre_gate_arbitration: Stage3ArbitrationDecision | None = None
    pre_gate_arbitration_evidence_ids: tuple[str, ...] | None = None
    pre_gate_consensus_confidence: float | None = None
    baseline_anchor: BaselineAnchor | None = None
    revision_certificates: tuple[LabelRevisionCertificate, ...] = ()
    preservation_result: BaselinePreservationResult | None = None
    expected_baseline_anchor_hash: str | None = None
    expected_taxonomy_structure_hash: str | None = None
    revision_audit: Stage3RevisionAudit | None = None
    preservation_audit_snapshot_json: str | None = None

    def __post_init__(self) -> None:
        def canonical_model(value: BaseModel) -> BaseModel:
            return type(value).model_validate(value.model_dump(mode="python"))

        object.__setattr__(
            self,
            "reports",
            tuple(canonical_model(report) for report in self.reports),
        )
        object.__setattr__(
            self,
            "evidence_deltas",
            tuple(canonical_model(delta) for delta in self.evidence_deltas),
        )
        object.__setattr__(
            self,
            "component_failures",
            tuple(canonical_model(failure) for failure in self.component_failures),
        )
        object.__setattr__(
            self,
            "revision_certificates",
            tuple(
                canonical_model(certificate)
                for certificate in self.revision_certificates
            ),
        )
        for field_name in (
            "final_decision",
            "verification",
            "validity_report",
            "unresolved",
            "boundary_challenge",
            "pre_gate_candidate",
            "pre_gate_verification",
            "pre_gate_disagreement",
            "pre_gate_arbitration",
            "baseline_anchor",
            "preservation_result",
            "revision_audit",
        ):
            value = getattr(self, field_name)
            if isinstance(value, BaseModel):
                object.__setattr__(self, field_name, canonical_model(value))
        if self.readiness_report is not None:
            try:
                canonical_readiness = canonical_model(self.readiness_report)
            except ValueError:
                canonical_readiness = None
            object.__setattr__(self, "readiness_report", canonical_readiness)
        object.__setattr__(
            self,
            "retrieval_source_request_ids",
            tuple(tuple(group) for group in self.retrieval_source_request_ids),
        )
        if self.baseline_anchor is None or self.preservation_audit_snapshot_json:
            return

        def snapshot_value(value: object) -> object:
            if isinstance(value, BaseModel):
                return value.model_dump(mode="json")
            if isinstance(value, Enum):
                return value.value
            if is_dataclass(value) and not isinstance(value, type):
                return {
                    field.name: snapshot_value(getattr(value, field.name))
                    for field in fields(value)
                    if field.name != "preservation_audit_snapshot_json"
                }
            if isinstance(value, Mapping):
                return {str(key): snapshot_value(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [snapshot_value(item) for item in value]
            return value

        payload = {
            field.name: snapshot_value(getattr(self, field.name))
            for field in fields(self)
            if field.name != "preservation_audit_snapshot_json"
        }
        object.__setattr__(
            self,
            "preservation_audit_snapshot_json",
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )


@dataclass(frozen=True)
class _Stage3TeamAnalysis:
    reports: tuple[Stage3TeamReport, ...]
    complete_team_ids: tuple[str, ...]
    component_failures: tuple[Stage3ComponentFailure, ...]


@dataclass(frozen=True)
class _Stage3PreservationOutcome:
    reports: tuple[Stage3TeamReport, ...]
    final_decision: Stage3FinalDecision | None
    verification: Stage3Verification | None
    certificates: tuple[LabelRevisionCertificate, ...]
    preservation_result: BaselinePreservationResult | None
    component_failures: tuple[Stage3ComponentFailure, ...]
    revision_audit: Stage3RevisionAudit
    unresolved: UnresolvedDecision | None = None


def _canonical_model_digest(value: BaseModel) -> str:
    return hashlib.sha256(
        json.dumps(
            value.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _component_failure(
    *, team_id: str, role: str, error: BaseException
) -> Stage3ComponentFailure:
    """Keep operational diagnostics useful without retaining model completions."""

    if isinstance(error, ValidationError):
        summary = "schema validation failed"
        validation_summary: Mapping[str, object] = {
            "fields": [
                ".".join(str(part) for part in detail.get("loc", ())) or "__root__"
                for detail in error.errors(include_input=False, include_url=False)
            ],
            "codes": [
                str(detail.get("type", ""))
                for detail in error.errors(include_input=False, include_url=False)
            ],
        }
    elif type(error).__name__ == "StructuredOutputError":
        summary = "structured output failed schema validation after bounded retries"
        candidate_summary = getattr(error, "validation_summary", {})
        validation_summary = (
            candidate_summary if isinstance(candidate_summary, Mapping) else {}
        )
    elif type(error).__name__ == "ModelTransportError":
        summary = "model transport failed during component execution"
        validation_summary = {}
    elif isinstance(error, CompositionError):
        summary = "team composition rejected verifier output"
        evidence_fields = [
            token
            for evidence_id in error.evidence_ids
            if (token := _composition_evidence_field(evidence_id)) is not None
        ]
        fields = (
            [
                error.dimension.value,
                *evidence_fields,
            ]
            if error.dimension is not None
            else evidence_fields
        )
        validation_summary = {
            "codes": [error.code],
            "fields": fields,
        }
    else:
        summary = f"{type(error).__name__} during component execution"
        validation_summary = {}
    return Stage3ComponentFailure(
        team_id=team_id,
        role=role,
        error_type=type(error).__name__,
        message=summary,
        schema_name=(
            sanitize_schema_name(getattr(error, "schema_name", None))
            if type(error).__name__ == "StructuredOutputError"
            else None
        ),
        validation_summary=sanitize_validation_summary(validation_summary),
    )


def _composition_evidence_field(evidence_id: str) -> str | None:
    """Encode a bounded CompositionError evidence ID for the generic sanitizer."""

    match = _COMPOSITION_EVIDENCE_ID.fullmatch(evidence_id)
    if match is None:
        return None
    suffix = match.group(1)
    safe_suffix = suffix if "-" not in suffix else suffix.encode("utf-8").hex()
    return f"evidence_id.feg_node_{safe_suffix}"


def _single_team_degraded_decision(
    report: Stage3TeamReport,
) -> Stage3FinalDecision | None:
    return resolve_single_stage3_report(report)


class Stage3Controller:
    """Runs split symptom/cause analysis followed by causal consistency."""

    def __init__(
        self,
        *,
        readiness: EvidenceReadiness,
        consistency_checker: Stage3ConsistencyChecker,
        joint_anchor: Stage3JointAnchor | None = None,
        symptom_verifier: Stage3DimensionVerifier | None = None,
        root_cause_verifier: Stage3DimensionVerifier | None = None,
        symptom_analyst: Stage3SymptomAnalyst | None = None,
        root_cause_analyst: Stage3RootCauseAnalyst | None = None,
        global_supervisor: Callable | None = None,
        boundary_challenger: Stage3BoundaryChallenger | None = None,
        specialists: SpecialistRegistry | None = None,
        arbitrator: Stage3Arbitrator | None = None,
        baseline_anchor: BaselineAnchor | None = None,
        taxonomy_structure: TaxonomyStructure | None = None,
        baseline_revision_assessor: Stage3BaselineRevisionAssessor | None = None,
        baseline_revision_consistency: Stage3BaselineRevisionConsistency | None = None,
        expected_baseline_config_hash: str | None = None,
        expected_baseline_predictions_sha256: str | None = None,
        expected_baseline_anchor_hash: str | None = None,
        expected_taxonomy_structure_hash: str | None = None,
        config: Stage3WorkflowConfig | None = None,
    ) -> None:
        anchored_values = (joint_anchor, symptom_verifier, root_cause_verifier)
        anchored_present = any(value is not None for value in anchored_values)
        anchored_complete = all(callable(value) for value in anchored_values)
        if anchored_present and not anchored_complete:
            raise ValueError("partial baseline-anchored Stage 3 adapter surface")

        legacy_values = (symptom_analyst, root_cause_analyst)
        legacy_present = any(value is not None for value in legacy_values)
        legacy_complete = all(callable(value) for value in legacy_values)
        if legacy_present and not legacy_complete:
            raise ValueError("partial legacy Stage 3 adapter surface")
        if not anchored_complete and not legacy_complete:
            raise ValueError(
                "Stage 3 adapter must expose either a complete baseline-anchored "
                "or legacy surface"
            )
        if not callable(consistency_checker):
            raise ValueError("Stage 3 consistency_checker must be callable")

        preservation_values = (
            baseline_anchor,
            taxonomy_structure,
            baseline_revision_assessor,
            baseline_revision_consistency,
            expected_baseline_config_hash,
            expected_baseline_predictions_sha256,
            expected_baseline_anchor_hash,
            expected_taxonomy_structure_hash,
        )
        preservation_present = any(value is not None for value in preservation_values)
        preservation_complete = (
            baseline_anchor is not None
            and taxonomy_structure is not None
            and callable(baseline_revision_assessor)
            and callable(baseline_revision_consistency)
            and expected_baseline_config_hash is not None
            and expected_baseline_predictions_sha256 is not None
            and expected_baseline_anchor_hash is not None
            and expected_taxonomy_structure_hash is not None
        )
        if preservation_present and not preservation_complete:
            raise ValueError("partial Baseline preservation Stage 3 adapter surface")

        self._readiness = readiness
        self._symptom_analyst = symptom_analyst
        self._root_cause_analyst = root_cause_analyst
        self._joint_anchor = joint_anchor
        self._dimension_verifiers = {
            "symptom_verifier": symptom_verifier,
            "root_cause_verifier": root_cause_verifier,
        }
        self._uses_anchored_surface = anchored_complete
        self._consistency_checker = consistency_checker
        self._global_supervisor = global_supervisor
        self._boundary_challenger = boundary_challenger
        self._specialists = specialists or SpecialistRegistry()
        self._arbitrator = arbitrator
        self._config = config or Stage3WorkflowConfig()
        self._baseline_anchor = baseline_anchor
        self._taxonomy_structure = taxonomy_structure
        self._baseline_revision_assessor = baseline_revision_assessor
        self._baseline_revision_consistency = baseline_revision_consistency
        self._expected_baseline_config_hash = expected_baseline_config_hash
        self._expected_baseline_predictions_sha256 = (
            expected_baseline_predictions_sha256
        )
        self._expected_baseline_anchor_hash = expected_baseline_anchor_hash
        self._expected_taxonomy_structure_hash = expected_taxonomy_structure_hash
        self._uses_baseline_preservation = preservation_complete

    def _analyze_legacy_teams(self, view: EvidenceView) -> _Stage3TeamAnalysis:
        assert self._symptom_analyst is not None
        assert self._root_cause_analyst is not None
        symptom_a_view = _isolated_classification_view(view)
        root_a_view = _isolated_classification_view(view)
        symptom_b_view = _isolated_classification_view(view)
        root_b_view = _isolated_classification_view(view)
        with ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="stage3-analysis"
        ) as executor:
            symptom_a_future = executor.submit(
                self._symptom_analyst, "A", symptom_a_view
            )
            root_a_future = executor.submit(self._root_cause_analyst, "A", root_a_view)
            symptom_b_future = executor.submit(
                self._symptom_analyst, "B", symptom_b_view
            )
            root_b_future = executor.submit(self._root_cause_analyst, "B", root_b_view)
            component_futures: tuple[tuple[str, str, Future[object]], ...] = (
                ("A", "symptom_analyst", symptom_a_future),
                ("A", "root_cause_analyst", root_a_future),
                ("B", "symptom_analyst", symptom_b_future),
                ("B", "root_cause_analyst", root_b_future),
            )
            results: dict[tuple[str, str], object] = {}
            failures: list[Stage3ComponentFailure] = []
            for team_id, role, future in component_futures:
                try:
                    results[(team_id, role)] = future.result()
                except Exception as error:
                    failures.append(
                        _component_failure(team_id=team_id, role=role, error=error)
                    )

        complete = tuple(
            team_id
            for team_id in ("A", "B")
            if (team_id, "symptom_analyst") in results
            and (team_id, "root_cause_analyst") in results
        )
        checker_results: dict[str, CausalConsistencyReport] = {}
        if complete:
            with ThreadPoolExecutor(
                max_workers=len(complete), thread_name_prefix="stage3-checker"
            ) as executor:
                checker_futures = {
                    team_id: executor.submit(
                        self._consistency_checker,
                        team_id,
                        results[(team_id, "symptom_analyst")],
                        results[(team_id, "root_cause_analyst")],
                        _isolated_classification_view(view),
                    )
                    for team_id in complete
                }
                for team_id in complete:
                    try:
                        checker_results[team_id] = checker_futures[team_id].result()
                    except Exception as error:
                        failures.append(
                            _component_failure(
                                team_id=team_id,
                                role="causal_consistency_checker",
                                error=error,
                            )
                        )

        reports = tuple(
            Stage3TeamReport(
                team_id=team_id,
                symptom=results[(team_id, "symptom_analyst")],
                root_cause=results[(team_id, "root_cause_analyst")],
                consistency=checker_results[team_id],
            )
            for team_id in complete
            if team_id in checker_results
        )
        return _Stage3TeamAnalysis(
            reports=reports,
            complete_team_ids=complete,
            component_failures=tuple(failures),
        )

    def _analyze_anchored_teams(self, view: EvidenceView) -> _Stage3TeamAnalysis:
        assert self._joint_anchor is not None
        assert all(
            verifier is not None for verifier in self._dimension_verifiers.values()
        )
        failures: list[Stage3ComponentFailure] = []
        anchors: dict[str, JointAnchorReport] = {}
        with ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="stage3-anchor"
        ) as executor:
            anchor_futures = {
                team_id: executor.submit(
                    self._joint_anchor,
                    team_id,
                    _isolated_classification_view(view),
                )
                for team_id in ("A", "B")
            }
            for team_id in ("A", "B"):
                try:
                    anchors[team_id] = anchor_futures[team_id].result()
                except Exception as error:
                    failures.append(
                        _component_failure(
                            team_id=team_id,
                            role="joint_anchor",
                            error=error,
                        )
                    )

        verifier_results: dict[tuple[str, str], DimensionVerificationReport] = {}
        if anchors:
            with ThreadPoolExecutor(
                max_workers=4, thread_name_prefix="stage3-verifier"
            ) as executor:
                verifier_futures = {
                    (team_id, role): executor.submit(
                        verifier,
                        team_id,
                        anchors[team_id],
                        _isolated_classification_view(view),
                    )
                    for team_id in anchors
                    for role, verifier in self._dimension_verifiers.items()
                    if verifier is not None
                }
                for team_id in ("A", "B"):
                    if team_id not in anchors:
                        continue
                    for role in ("symptom_verifier", "root_cause_verifier"):
                        try:
                            verifier_results[(team_id, role)] = verifier_futures[
                                (team_id, role)
                            ].result()
                        except Exception as error:
                            failures.append(
                                _component_failure(
                                    team_id=team_id,
                                    role=role,
                                    error=error,
                                )
                            )

        candidates: dict[str, TeamComposedCandidate] = {}
        for team_id in ("A", "B"):
            if team_id not in anchors or any(
                (team_id, role) not in verifier_results
                for role in ("symptom_verifier", "root_cause_verifier")
            ):
                continue
            try:
                candidates[team_id] = compose_stage3_team_candidate(
                    team_id,
                    anchors[team_id],
                    verifier_results[(team_id, "symptom_verifier")],
                    verifier_results[(team_id, "root_cause_verifier")],
                    _isolated_classification_view(view),
                )
            except CompositionError as error:
                failures.append(
                    _component_failure(
                        team_id=team_id,
                        role="team_composer",
                        error=error,
                    )
                )

        checker_results: dict[str, CausalConsistencyReport] = {}
        if candidates:
            with ThreadPoolExecutor(
                max_workers=len(candidates), thread_name_prefix="stage3-checker"
            ) as executor:
                checker_futures = {
                    team_id: executor.submit(
                        self._consistency_checker,
                        team_id,
                        candidate.symptom,
                        candidate.root_cause,
                        _isolated_classification_view(view),
                    )
                    for team_id, candidate in candidates.items()
                }
                for team_id in ("A", "B"):
                    if team_id not in candidates:
                        continue
                    try:
                        checker_results[team_id] = checker_futures[team_id].result()
                    except Exception as error:
                        failures.append(
                            _component_failure(
                                team_id=team_id,
                                role="causal_consistency_checker",
                                error=error,
                            )
                        )

        reports = tuple(
            Stage3TeamReport(
                team_id=candidate.team_id,
                symptom=candidate.symptom,
                root_cause=candidate.root_cause,
                consistency=checker_results[team_id],
                anchor=candidate.anchor,
                verifications=candidate.verifications,
                correction_audit=candidate.correction_audit,
            )
            for team_id in ("A", "B")
            if team_id in checker_results
            for candidate in (candidates[team_id],)
        )
        return _Stage3TeamAnalysis(
            reports=reports,
            complete_team_ids=tuple(candidates),
            component_failures=tuple(failures),
        )

    def _analyze_teams(self, view: EvidenceView) -> _Stage3TeamAnalysis:
        if self._uses_anchored_surface:
            return self._analyze_anchored_teams(view)
        return self._analyze_legacy_teams(view)

    def _apply_baseline_preservation(
        self,
        *,
        reports: tuple[Stage3TeamReport, ...],
        candidate: Stage3FinalDecision | None,
        candidate_verification: Stage3Verification | None,
        view: EvidenceView,
        component_failures: tuple[Stage3ComponentFailure, ...],
        readiness: EvidenceReadinessReport | None = None,
        disagreement: DisagreementMap | None = None,
        arbitration: Stage3ArbitrationDecision | None = None,
        arbitration_evidence_ids: tuple[str, ...] | None = None,
        boundary_challenge: BoundaryChallenge | None = None,
    ) -> _Stage3PreservationOutcome:
        assert self._baseline_anchor is not None
        assert self._taxonomy_structure is not None
        assert self._baseline_revision_assessor is not None
        assert self._baseline_revision_consistency is not None
        assert self._expected_baseline_config_hash is not None
        assert self._expected_baseline_predictions_sha256 is not None
        assert self._expected_baseline_anchor_hash is not None
        assert self._expected_taxonomy_structure_hash is not None

        anchor = self._baseline_anchor
        proposals = build_revision_proposals(
            anchor=anchor,
            reports=reports,
            structure=self._taxonomy_structure,
            view=view,
            boundary_challenge=boundary_challenge,
        )
        assessments: dict[tuple[str, str], BaselineRevisionAssessment] = {}
        failures = list(component_failures)
        calls = tuple(
            (team_id, proposal) for proposal in proposals for team_id in ("A", "B")
        )
        if calls:
            cards = {
                proposal.boundary_card_id: route_boundary_card(
                    self._taxonomy_structure,
                    proposal.dimension,
                    proposal.baseline_label,
                    proposal.proposed_label,
                )
                for proposal in proposals
            }
            if calls:
                with ThreadPoolExecutor(
                    max_workers=len(calls), thread_name_prefix="baseline-revision"
                ) as executor:
                    futures = {
                        (team_id, proposal.boundary_card_id): executor.submit(
                            self._baseline_revision_assessor,
                            team_id,
                            proposal,
                            anchor,
                            cards[proposal.boundary_card_id],
                            _isolated_classification_view(view),
                        )
                        for team_id, proposal in calls
                    }
                    for team_id, proposal in calls:
                        key = (team_id, proposal.boundary_card_id)
                        try:
                            returned = futures[key].result()
                            if not isinstance(returned, BaselineRevisionAssessment):
                                raise StructuredOutputError(
                                    "assessment did not return the typed contract",
                                    schema_name="BaselineRevisionAssessment",
                                    validation_summary={
                                        "fields": ["assessment"],
                                        "codes": ["type_mismatch"],
                                    },
                                )
                            expected_task = (
                                RevisionAssessorTask.ENTAILMENT
                                if team_id == "A"
                                else RevisionAssessorTask.FALSIFICATION
                            )
                            expected_raw = (
                                returned.raw_entailment_result
                                if team_id == "A"
                                else returned.raw_falsification_result
                            )
                            if (
                                returned.assessor_task is not expected_task
                                or returned.raw_result_digest is None
                                or expected_raw is None
                            ):
                                raise StructuredOutputError(
                                    "assessment lacks fixed heterogeneous task provenance",
                                    schema_name="BaselineRevisionAssessment",
                                    validation_summary={
                                        "fields": ["assessor_task", "raw_result"],
                                        "codes": ["policy_downgrade"],
                                    },
                                )
                            assessments[key] = returned
                        except (StructuredOutputError, ModelTransportError) as error:
                            failures.append(
                                _component_failure(
                                    team_id=team_id,
                                    role=(
                                        f"baseline_revision_{proposal.dimension.value}_"
                                        f"{revision_proposal_digest(proposal)[:12]}"
                                    ),
                                    error=error,
                                )
                            )

        consistency_records: dict[tuple[str, str], RevisionConsistencyReport] = {}
        revision_calls = tuple(
            (checker_team_id, owner_team_id, proposal, owner_assessment)
            for proposal in proposals
            if all(
                (assessment := assessments.get((team_id, proposal.boundary_card_id)))
                is not None
                and assessment.verdict is RevisionAssessmentVerdict.REVISE
                for team_id in ("A", "B")
            )
            for checker_team_id, owner_team_id in (("A", "B"), ("B", "A"))
            for owner_assessment in (
                assessments[(owner_team_id, proposal.boundary_card_id)],
            )
        )
        if revision_calls:
            with ThreadPoolExecutor(
                max_workers=len(revision_calls),
                thread_name_prefix="baseline-revision-consistency",
            ) as executor:
                futures = {
                    (checker_team_id, proposal.boundary_card_id): executor.submit(
                        self._baseline_revision_consistency,
                        checker_team_id,
                        owner_team_id,
                        proposal,
                        owner_assessment,
                        cards[proposal.boundary_card_id],
                        _isolated_classification_view(view),
                    )
                    for checker_team_id, owner_team_id, proposal, owner_assessment in revision_calls
                }
                for key, future in futures.items():
                    try:
                        consistency_records[key] = future.result()
                    except (StructuredOutputError, ModelTransportError) as error:
                        failures.append(
                            _component_failure(
                                team_id=key[0],
                                role=("baseline_revision_consistency_" f"{key[1]}"),
                                error=error,
                            )
                        )

        augmented_reports = tuple(
            report.model_copy(
                update={
                    "baseline_revision_assessments": tuple(
                        assessment
                        for proposal in proposals
                        for assessment in (
                            assessments.get(
                                (report.team_id, proposal.boundary_card_id)
                            ),
                        )
                        if assessment is not None
                    ),
                    "revision_consistency": tuple(
                        consistency
                        for proposal in proposals
                        for consistency in (
                            consistency_records.get(
                                (report.team_id, proposal.boundary_card_id)
                            ),
                        )
                        if consistency is not None
                    ),
                }
            )
            for report in reports
        )
        certificates = compose_baseline_revision_certificates(
            anchor=anchor,
            reports=augmented_reports,
            proposals=proposals,
            structure=self._taxonomy_structure,
            view=view,
            boundary_challenge=boundary_challenge,
        )

        proposal_digests = {
            proposal.boundary_card_id: revision_proposal_digest(proposal)
            for proposal in proposals
        }
        assessment_attempts = tuple(
            RevisionAssessmentAuditAttempt(
                proposal_digest=proposal_digests[proposal.boundary_card_id],
                dimension=proposal.dimension,
                boundary_card_id=proposal.boundary_card_id,
                assessor_team_id=team_id,
                returned_assessment_digest=(
                    baseline_revision_assessment_digest(returned)
                    if returned is not None
                    else None
                ),
                returned_assessment=returned,
                verdict=(returned.verdict if returned is not None else None),
                operational_failure=returned is None,
            )
            for team_id, proposal in calls
            for returned in (assessments.get((team_id, proposal.boundary_card_id)),)
        )
        dual_revise = tuple(
            revision_proposal_digest(proposal)
            for proposal in proposals
            if all(
                (returned := assessments.get((team_id, proposal.boundary_card_id)))
                is not None
                and returned.verdict is RevisionAssessmentVerdict.REVISE
                for team_id in ("A", "B")
            )
        )
        consistency_attempts = tuple(
            RevisionConsistencyAuditAttempt(
                proposal_digest=revision_proposal_digest(proposal),
                dimension=proposal.dimension,
                boundary_card_id=proposal.boundary_card_id,
                checker_team_id=checker_team_id,
                assessment_owner_team_id=owner_team_id,
                returned_consistency_digest=(
                    revision_consistency_report_digest(returned)
                    if returned is not None
                    else None
                ),
                returned_consistency=returned,
                status=(returned.status if returned is not None else None),
                operational_failure=returned is None,
            )
            for checker_team_id, owner_team_id, proposal, _ in revision_calls
            for returned in (
                consistency_records.get((checker_team_id, proposal.boundary_card_id)),
            )
        )
        consistency_pass = tuple(
            revision_proposal_digest(proposal)
            for proposal in proposals
            if all(
                (
                    returned := consistency_records.get(
                        (checker_team_id, proposal.boundary_card_id)
                    )
                )
                is not None
                and returned.status is ConsistencyStatus.CONSISTENT
                for checker_team_id in ("A", "B")
            )
        )
        fully_certified_by_dimension: dict[EvidenceDimension, list[str]] = {}
        for proposal in proposals:
            digest = revision_proposal_digest(proposal)
            if digest in consistency_pass:
                fully_certified_by_dimension.setdefault(proposal.dimension, []).append(
                    digest
                )
        ambiguous_dimensions = tuple(
            dimension
            for dimension in (EvidenceDimension.SYMPTOM, EvidenceDimension.ROOT_CAUSE)
            if len(fully_certified_by_dimension.get(dimension, ())) > 1
        )
        failed_proposals = tuple(
            digest
            for digest in (revision_proposal_digest(proposal) for proposal in proposals)
            if any(
                attempt.proposal_digest == digest and attempt.operational_failure
                for attempt in (*assessment_attempts, *consistency_attempts)
            )
        )

        def revision_audit(
            applied_certificates: tuple[LabelRevisionCertificate, ...] = (),
        ) -> Stage3RevisionAudit:
            return Stage3RevisionAudit(
                assessment_policy_version=HETEROGENEOUS_REVISION_ASSESSMENT_POLICY,
                cross_policy_version=TASK_SPECIFIC_REVISION_CROSS_POLICY,
                proposal_digests=tuple(
                    revision_proposal_digest(proposal) for proposal in proposals
                ),
                routed_card_ids=tuple(
                    proposal.boundary_card_id for proposal in proposals
                ),
                proposal_entries=tuple(
                    RevisionProposalAuditEntry(
                        proposal_digest=revision_proposal_digest(proposal),
                        dimension=proposal.dimension,
                        boundary_card_id=proposal.boundary_card_id,
                        baseline_label=proposal.baseline_label,
                        proposed_label=proposal.proposed_label,
                    )
                    for proposal in proposals
                ),
                assessment_attempts=assessment_attempts,
                dual_revise_proposal_digests=dual_revise,
                consistency_attempts=consistency_attempts,
                consistency_pass_proposal_digests=consistency_pass,
                issued_certificate_digests=tuple(
                    _canonical_model_digest(certificate) for certificate in certificates
                ),
                issued_certificate_entries=tuple(
                    RevisionCertificateAuditEntry(
                        certificate_digest=_canonical_model_digest(certificate),
                        proposal_digest=certificate.proposal_digest,
                        dimension=certificate.dimension,
                        boundary_card_id=certificate.boundary_card_id,
                        baseline_label=certificate.baseline_label,
                        proposed_label=certificate.proposed_label,
                    )
                    for certificate in certificates
                ),
                applied_certificate_digests=tuple(
                    _canonical_model_digest(certificate)
                    for certificate in applied_certificates
                ),
                ambiguous_certified_dimensions=ambiguous_dimensions,
                failed_proposal_digests=failed_proposals,
            )

        try:
            preservation = apply_baseline_preservation_gate(
                anchor=anchor,
                candidate=candidate,
                candidate_verification=candidate_verification,
                certificates=certificates,
                proposals=proposals,
                structure=self._taxonomy_structure,
                view=view,
                expected_source_config_hash=self._expected_baseline_config_hash,
                expected_source_predictions_sha256=(
                    self._expected_baseline_predictions_sha256
                ),
                expected_anchor_hash=self._expected_baseline_anchor_hash,
                expected_structure_hash=self._expected_taxonomy_structure_hash,
            )
        except PreservationGateError as error:
            return _Stage3PreservationOutcome(
                reports=augmented_reports,
                final_decision=None,
                verification=None,
                certificates=certificates,
                preservation_result=None,
                component_failures=tuple(failures),
                revision_audit=revision_audit(),
                unresolved=UnresolvedDecision(
                    status=ResolutionStatus.UNRESOLVED,
                    dimensions=(
                        EvidenceDimension.SYMPTOM,
                        EvidenceDimension.ROOT_CAUSE,
                    ),
                    missing_facts=(str(error),),
                    stop_reason="baseline_preservation_gate_failed",
                ),
            )

        final = compose_baseline_preservation_decision(
            preservation=preservation,
            candidate=candidate,
        )
        completed_revision_audit = revision_audit(preservation.applied_certificates)
        verification = verify_stage3_decision(
            final,
            _FrozenStage3EvidenceLedger(view),
            reports=augmented_reports,
            baseline_anchor=anchor,
            pre_gate_candidate=candidate,
            pre_gate_verification=candidate_verification,
            revision_certificates=certificates,
            preservation_result=preservation,
            revision_audit=completed_revision_audit,
            taxonomy_structure=self._taxonomy_structure,
            expected_baseline_config_hash=self._expected_baseline_config_hash,
            expected_baseline_predictions_sha256=(
                self._expected_baseline_predictions_sha256
            ),
            expected_baseline_anchor_hash=self._expected_baseline_anchor_hash,
            expected_taxonomy_structure_hash=self._expected_taxonomy_structure_hash,
            expected_revision_assessment_policy=(
                HETEROGENEOUS_REVISION_ASSESSMENT_POLICY
            ),
            expected_revision_cross_policy=TASK_SPECIFIC_REVISION_CROSS_POLICY,
            readiness=readiness,
            disagreement=disagreement,
            arbitration=arbitration,
            arbitration_evidence_ids=arbitration_evidence_ids,
            boundary_challenge=boundary_challenge,
            component_failures=tuple(failures),
            consensus_confidence=self._config.consensus_confidence,
        )
        return _Stage3PreservationOutcome(
            reports=augmented_reports,
            final_decision=final,
            verification=verification,
            certificates=certificates,
            preservation_result=preservation,
            component_failures=tuple(failures),
            revision_audit=completed_revision_audit,
        )

    def run(self, ledger: EvidenceLedger) -> Stage3WorkflowResult:
        if self._uses_baseline_preservation:
            assert self._baseline_anchor is not None
            assert self._taxonomy_structure is not None
            assert self._expected_baseline_config_hash is not None
            assert self._expected_baseline_predictions_sha256 is not None
            assert self._expected_baseline_anchor_hash is not None
            assert self._expected_taxonomy_structure_hash is not None
            initial_view = ledger.view()
            try:
                validate_baseline_preservation_context(
                    anchor=self._baseline_anchor,
                    structure=self._taxonomy_structure,
                    view=initial_view,
                    expected_source_config_hash=self._expected_baseline_config_hash,
                    expected_source_predictions_sha256=(
                        self._expected_baseline_predictions_sha256
                    ),
                    expected_anchor_hash=self._expected_baseline_anchor_hash,
                    expected_structure_hash=self._expected_taxonomy_structure_hash,
                )
            except PreservationGateError as error:
                return Stage3WorkflowResult(
                    reports=(),
                    final_decision=None,
                    verification=None,
                    evidence_deltas=(),
                    retrieval_source_request_ids=[],
                    budget_exhausted=False,
                    validity_report=assess_stage3_evidence_validity(initial_view),
                    unresolved=UnresolvedDecision(
                        status=ResolutionStatus.UNRESOLVED,
                        dimensions=(
                            EvidenceDimension.SYMPTOM,
                            EvidenceDimension.ROOT_CAUSE,
                        ),
                        missing_facts=(str(error),),
                        stop_reason="baseline_preservation_context_invalid",
                    ),
                    baseline_anchor=self._baseline_anchor,
                    expected_baseline_anchor_hash=self._expected_baseline_anchor_hash,
                    expected_taxonomy_structure_hash=(
                        self._expected_taxonomy_structure_hash
                    ),
                )
        evidence_phase = run_stage3_evidence_phase(
            ledger=ledger,
            readiness=self._readiness,
            specialists=self._specialists,
            max_retrieval_rounds=self._config.max_retrieval_rounds,
            max_specialist_calls=self._config.max_specialist_calls,
        )
        canonical_classification_view = ledger.view(
            evidence_phase.validity_report.valid_evidence_ids
        )
        if not evidence_phase.validity_report.valid_evidence_ids:
            return Stage3WorkflowResult(
                reports=(),
                final_decision=None,
                verification=None,
                evidence_deltas=evidence_phase.evidence_deltas,
                retrieval_source_request_ids=list(
                    evidence_phase.retrieval_source_request_ids
                ),
                budget_exhausted=evidence_phase.budget_exhausted,
                validity_report=evidence_phase.validity_report,
                readiness_report=evidence_phase.gap_report,
                classification_ledger_version=None,
                unresolved=UnresolvedDecision(
                    status=ResolutionStatus.UNRESOLVED,
                    dimensions=(
                        EvidenceDimension.SYMPTOM,
                        EvidenceDimension.ROOT_CAUSE,
                    ),
                    missing_facts=(
                        "No valid task-relevant evidence remains for Stage 3.",
                    ),
                    stop_reason="no_valid_stage3_evidence",
                    attempted_retrieval_statuses=tuple(
                        delta.status for delta in evidence_phase.evidence_deltas
                    ),
                ),
                baseline_anchor=(
                    self._baseline_anchor if self._uses_baseline_preservation else None
                ),
                expected_baseline_anchor_hash=(
                    self._expected_baseline_anchor_hash
                    if self._uses_baseline_preservation
                    else None
                ),
                expected_taxonomy_structure_hash=(
                    self._expected_taxonomy_structure_hash
                    if self._uses_baseline_preservation
                    else None
                ),
            )

        if self._uses_baseline_preservation:
            assert self._baseline_anchor is not None
            assert self._taxonomy_structure is not None
            assert self._expected_baseline_config_hash is not None
            assert self._expected_baseline_predictions_sha256 is not None
            assert self._expected_baseline_anchor_hash is not None
            assert self._expected_taxonomy_structure_hash is not None
            try:
                validate_baseline_preservation_context(
                    anchor=self._baseline_anchor,
                    structure=self._taxonomy_structure,
                    view=canonical_classification_view,
                    expected_source_config_hash=self._expected_baseline_config_hash,
                    expected_source_predictions_sha256=(
                        self._expected_baseline_predictions_sha256
                    ),
                    expected_anchor_hash=self._expected_baseline_anchor_hash,
                    expected_structure_hash=self._expected_taxonomy_structure_hash,
                )
            except PreservationGateError as error:
                return Stage3WorkflowResult(
                    reports=(),
                    final_decision=None,
                    verification=None,
                    evidence_deltas=evidence_phase.evidence_deltas,
                    retrieval_source_request_ids=list(
                        evidence_phase.retrieval_source_request_ids
                    ),
                    budget_exhausted=evidence_phase.budget_exhausted,
                    validity_report=evidence_phase.validity_report,
                    readiness_report=evidence_phase.gap_report,
                    unresolved=UnresolvedDecision(
                        status=ResolutionStatus.UNRESOLVED,
                        dimensions=(
                            EvidenceDimension.SYMPTOM,
                            EvidenceDimension.ROOT_CAUSE,
                        ),
                        missing_facts=(str(error),),
                        stop_reason="baseline_preservation_context_invalid",
                    ),
                    baseline_anchor=self._baseline_anchor,
                    expected_baseline_anchor_hash=self._expected_baseline_anchor_hash,
                    expected_taxonomy_structure_hash=(
                        self._expected_taxonomy_structure_hash
                    ),
                )

        classification_ledger_version = canonical_classification_view.ledger_version
        analysis = self._analyze_teams(canonical_classification_view)
        reports = analysis.reports
        component_failures = analysis.component_failures
        _, missing_ids = _resolve_relevant_evidence(
            canonical_classification_view,
            _stage3_candidate_ids(reports),
        )
        provenance_errors = tuple(
            error
            for report in reports
            for error in stage3_report_provenance_errors(
                report,
                canonical_classification_view,
            )
        )
        taxonomy_errors = _stage3_candidate_taxonomy_errors(
            reports,
            canonical_classification_view.taxonomy,
        )
        if missing_ids or provenance_errors or taxonomy_errors:
            missing_facts = tuple(
                f"Candidate report cites unknown evidence_id {evidence_id!r}."
                for evidence_id in missing_ids
            )
            if not missing_facts and provenance_errors:
                missing_facts = tuple(
                    f"Candidate report provenance invalid: {error}."
                    for error in provenance_errors
                )
            if not missing_facts:
                missing_facts = tuple(
                    f"Candidate report taxonomy invalid: {error}."
                    for error in taxonomy_errors
                )
            return Stage3WorkflowResult(
                reports=reports,
                final_decision=None,
                verification=None,
                evidence_deltas=evidence_phase.evidence_deltas,
                retrieval_source_request_ids=list(
                    evidence_phase.retrieval_source_request_ids
                ),
                budget_exhausted=evidence_phase.budget_exhausted,
                validity_report=evidence_phase.validity_report,
                readiness_report=evidence_phase.gap_report,
                classification_ledger_version=classification_ledger_version,
                unresolved=UnresolvedDecision(
                    status=ResolutionStatus.UNRESOLVED,
                    dimensions=("stage3_components",),
                    missing_facts=missing_facts,
                    stop_reason=(
                        "stage3_candidate_invalid_taxonomy"
                        if taxonomy_errors
                        else (
                            "arbitration_evidence_insufficient"
                            if len(reports) == 2
                            else "stage3_candidate_invalid_citation"
                        )
                    ),
                    attempted_retrieval_statuses=tuple(
                        delta.status for delta in evidence_phase.evidence_deltas
                    ),
                ),
                component_failures=analysis.component_failures,
                baseline_anchor=(
                    self._baseline_anchor if self._uses_baseline_preservation else None
                ),
                expected_baseline_anchor_hash=(
                    self._expected_baseline_anchor_hash
                    if self._uses_baseline_preservation
                    else None
                ),
                expected_taxonomy_structure_hash=(
                    self._expected_taxonomy_structure_hash
                    if self._uses_baseline_preservation
                    else None
                ),
            )
        if len(reports) != 2 and self._global_supervisor is not None:
            return Stage3WorkflowResult(
                reports=reports, final_decision=None, verification=None,
                evidence_deltas=evidence_phase.evidence_deltas,
                retrieval_source_request_ids=evidence_phase.retrieval_source_request_ids,
                budget_exhausted=evidence_phase.budget_exhausted,
                validity_report=evidence_phase.validity_report,
                readiness_report=evidence_phase.gap_report,
                classification_ledger_version=classification_ledger_version,
                component_failures=component_failures,
                supervision={"failed": True, "stop_reason": "incomplete_initial_teams", "events": []},
                unresolved=UnresolvedDecision(status=ResolutionStatus.UNRESOLVED,
                    dimensions=("stage3_components",),
                    missing_facts=("Both initial teams are required for global supervision.",),
                    stop_reason="supervisor_incomplete_initial_teams"),
            )
        if len(reports) != 2:
            final_decision = (
                _single_team_degraded_decision(reports[0])
                if len(reports) == 1
                else None
            )
            verification = (
                verify_stage3_decision(
                    final_decision,
                    _FrozenStage3EvidenceLedger(canonical_classification_view),
                    reports=reports,
                    readiness=evidence_phase.gap_report,
                )
                if final_decision is not None
                else None
            )
            if self._uses_baseline_preservation:
                outcome = self._apply_baseline_preservation(
                    reports=reports,
                    candidate=final_decision,
                    candidate_verification=verification,
                    view=canonical_classification_view,
                    component_failures=component_failures,
                    readiness=evidence_phase.gap_report,
                )
                return Stage3WorkflowResult(
                    reports=outcome.reports,
                    final_decision=outcome.final_decision,
                    verification=outcome.verification,
                    evidence_deltas=evidence_phase.evidence_deltas,
                    retrieval_source_request_ids=list(
                        evidence_phase.retrieval_source_request_ids
                    ),
                    budget_exhausted=evidence_phase.budget_exhausted,
                    validity_report=evidence_phase.validity_report,
                    readiness_report=evidence_phase.gap_report,
                    classification_ledger_version=classification_ledger_version,
                    unresolved=outcome.unresolved,
                    component_failures=outcome.component_failures,
                    pre_gate_candidate=final_decision,
                    pre_gate_verification=verification,
                    pre_gate_consensus_confidence=self._config.consensus_confidence,
                    baseline_anchor=self._baseline_anchor,
                    revision_certificates=outcome.certificates,
                    preservation_result=outcome.preservation_result,
                    revision_audit=outcome.revision_audit,
                    expected_baseline_anchor_hash=self._expected_baseline_anchor_hash,
                    expected_taxonomy_structure_hash=(
                        self._expected_taxonomy_structure_hash
                    ),
                )
            return Stage3WorkflowResult(
                reports=reports,
                final_decision=final_decision,
                verification=verification,
                evidence_deltas=evidence_phase.evidence_deltas,
                retrieval_source_request_ids=list(
                    evidence_phase.retrieval_source_request_ids
                ),
                budget_exhausted=evidence_phase.budget_exhausted,
                validity_report=evidence_phase.validity_report,
                readiness_report=evidence_phase.gap_report,
                classification_ledger_version=classification_ledger_version,
                unresolved=(
                    None
                    if final_decision is not None
                    else UnresolvedDecision(
                        status=ResolutionStatus.UNRESOLVED,
                        dimensions=("stage3_components",),
                        missing_facts=(
                            "No complete team passed causal consistency for a "
                            "prediction-preserving degraded decision.",
                        ),
                        stop_reason="stage3_component_failure",
                        attempted_retrieval_statuses=tuple(
                            delta.status for delta in evidence_phase.evidence_deltas
                        ),
                    )
                ),
                component_failures=analysis.component_failures,
            )
        boundary_challenge: BoundaryChallenge | None = None
        force_uncertain_fallback = False
        supervision = None
        if self._global_supervisor is not None:
            supervised = self._global_supervisor(reports, canonical_classification_view)
            reports = supervised.reports
            boundary_challenge = supervised.challenge
            supervision = supervised.audit()
            if supervised.failed:
                return Stage3WorkflowResult(
                    reports=reports, final_decision=None, verification=None,
                    evidence_deltas=evidence_phase.evidence_deltas,
                    retrieval_source_request_ids=evidence_phase.retrieval_source_request_ids,
                    budget_exhausted=evidence_phase.budget_exhausted,
                    validity_report=evidence_phase.validity_report,
                    readiness_report=evidence_phase.gap_report,
                    classification_ledger_version=classification_ledger_version,
                    boundary_challenge=boundary_challenge, supervision=supervision,
                    component_failures=component_failures,
                    unresolved=UnresolvedDecision(status=ResolutionStatus.UNRESOLVED,
                        dimensions=("stage3_components",),
                        missing_facts=("Global causal supervision did not complete; inspect supervision.events.",),
                        stop_reason=supervised.stop_reason),
                )
            force_uncertain_fallback = supervised.stop_reason == "evidence_gap"
        if self._global_supervisor is None and self._boundary_challenger is not None:
            try:
                boundary_challenge = self._boundary_challenger(
                    tuple(
                        Stage3TeamReport.model_validate(
                            report.model_dump(mode="python")
                        )
                        for report in reports
                    ),
                    _isolated_classification_view(canonical_classification_view),
                )
            except (StructuredOutputError, ModelTransportError) as error:
                component_failures = component_failures + (
                    _component_failure(
                        team_id="challenge",
                        role="boundary_challenger",
                        error=error,
                    ),
                )
                force_uncertain_fallback = True
        if boundary_challenge is not None:
            challenge_errors = _boundary_challenge_errors(
                boundary_challenge,
                reports,
                canonical_classification_view,
            )
            if challenge_errors:
                return Stage3WorkflowResult(
                    reports=reports,
                    final_decision=None,
                    verification=None,
                    evidence_deltas=evidence_phase.evidence_deltas,
                    retrieval_source_request_ids=list(
                        evidence_phase.retrieval_source_request_ids
                    ),
                    budget_exhausted=evidence_phase.budget_exhausted,
                    validity_report=evidence_phase.validity_report,
                    readiness_report=evidence_phase.gap_report,
                    classification_ledger_version=classification_ledger_version,
                    unresolved=_boundary_challenge_unresolved(
                        evidence_phase,
                        boundary_challenge,
                        stop_reason="boundary_challenger_invalid_citation",
                        missing_facts=challenge_errors,
                    ),
                    boundary_challenge=boundary_challenge,
                    supervision=supervision,
                    baseline_anchor=(
                        self._baseline_anchor
                        if self._uses_baseline_preservation
                        else None
                    ),
                    expected_baseline_anchor_hash=(
                        self._expected_baseline_anchor_hash
                        if self._uses_baseline_preservation
                        else None
                    ),
                    expected_taxonomy_structure_hash=(
                        self._expected_taxonomy_structure_hash
                        if self._uses_baseline_preservation
                        else None
                    ),
                )
            if boundary_challenge.action is BoundaryChallengeAction.EVIDENCE_REQUEST:
                # A non-authoritative challenger may record an evidence gap, but
                # ordinary uncertainty must not erase two valid team predictions.
                # Preserve the challenge in the audit and downgrade a consensus
                # result to the deterministic uncertain fallback.
                force_uncertain_fallback = True

        disagreement = build_stage3_disagreement_map(
            *reports,
            confidence_threshold=self._config.consensus_confidence,
            readiness=None,
            evidence_items=canonical_classification_view.items,
            taxonomy=canonical_classification_view.taxonomy,
        )
        if boundary_challenge is not None:
            disagreement = add_boundary_challenge_dimension(
                disagreement,
                boundary_challenge,
            )
        arbitration: Stage3ArbitrationDecision | None = None
        arbitration_evidence_ids: tuple[str, ...] | None = None
        unresolved: UnresolvedDecision | None = None
        final_decision: Stage3FinalDecision | None = None
        _, missing_ids = _resolve_relevant_evidence(
            canonical_classification_view,
            _stage3_candidate_ids(reports),
        )
        if missing_ids:
            unresolved = _arbitration_unresolved(
                disagreement,
                evidence_phase,
                missing_facts=tuple(
                    f"Candidate report cites unknown evidence_id {evidence_id!r}."
                    for evidence_id in missing_ids
                ),
            )
        elif not disagreement.requires_arbitration:
            final_decision = resolve_stage3_reports(
                *reports,
                confidence_threshold=self._config.consensus_confidence,
                evidence_items=canonical_classification_view.items,
                force_fallback=force_uncertain_fallback,
            )
        elif self._arbitrator is None:
            final_decision = resolve_stage3_reports(
                *reports,
                confidence_threshold=self._config.consensus_confidence,
                evidence_items=canonical_classification_view.items,
                force_fallback=True,
            )
        elif disagreement.requires_arbitration:
            relevant_evidence, _ = _resolve_relevant_evidence(
                canonical_classification_view,
                _stage3_relevant_ids(
                    reports,
                    disagreement,
                    boundary_challenge,
                ),
            )
            if supervision is not None:
                relevant_evidence = canonical_classification_view.items
            arbitration_evidence_ids = tuple(
                item.evidence_id for item in relevant_evidence
            )
            packet = Stage3ArbitrationPacket(
                supervision=supervision,
                domain_profile=canonical_classification_view.domain_profile,
                taxonomy=canonical_classification_view.taxonomy,
                disagreement=disagreement,
                team_a=AnonymousStage3TeamReport(
                    symptom=reports[0].symptom,
                    root_cause=reports[0].root_cause,
                    consistency=reports[0].consistency,
                ),
                team_b=AnonymousStage3TeamReport(
                    symptom=reports[1].symptom,
                    root_cause=reports[1].root_cause,
                    consistency=reports[1].consistency,
                ),
                classification_ledger_version=classification_ledger_version,
                relevant_evidence=relevant_evidence,
            )
            try:
                arbitration = self._arbitrator(packet)
            except (StructuredOutputError, ModelTransportError) as error:
                component_failures = component_failures + (
                    _component_failure(
                        team_id="arbitration",
                        role="stage3_arbitrator",
                        error=error,
                    ),
                )
                arbitration = Stage3ArbitrationDecision(
                    resolution_status=ResolutionStatus.UNRESOLVED,
                    rationale=(
                        "The targeted arbitration component failed after bounded "
                        "attempts; preserve valid candidates with deterministic "
                        "uncertain fallback."
                    ),
                    unresolved_dimensions=list(disagreement.dimensions),
                    missing_facts=[
                        "A valid targeted arbitration decision was unavailable."
                    ],
                )
            arbitration_citation_errors = tuple(
                f"Arbitration evidence_id {evidence_id!r} was not included in "
                "arbitration packet."
                for evidence_id in arbitration.supporting_evidence_ids
                if evidence_id not in arbitration_evidence_ids
            )
            arbitration_taxonomy_errors: tuple[str, ...] = ()
            if arbitration.resolution_status is ResolutionStatus.RESOLVED:
                arbitration_taxonomy_errors = tuple(
                    f"Arbitration returned a label outside the {dimension} taxonomy."
                    for dimension, label in (
                        ("symptom", arbitration.symptom_label),
                        ("root_cause", arbitration.root_cause_label),
                    )
                    if not label_valid(canonical_classification_view.taxonomy, dimension, label)
                )
            arbitration_scope_errors = _arbitration_scope_errors(
                disagreement,
                arbitration,
                reports,
            )
            if arbitration_citation_errors or arbitration_taxonomy_errors:
                unresolved = _arbitration_unresolved(
                    disagreement,
                    evidence_phase,
                    missing_facts=(
                        arbitration_citation_errors + arbitration_taxonomy_errors
                    ),
                    stop_reason=(
                        "arbitration_invalid_citation"
                        if arbitration_citation_errors
                        else "arbitration_invalid_taxonomy"
                    ),
                )
                arbitration = None
            elif arbitration_scope_errors:
                unresolved = _arbitration_unresolved(
                    disagreement,
                    evidence_phase,
                    missing_facts=tuple(
                        f"Arbitration authority violation: {error}"
                        for error in arbitration_scope_errors
                    ),
                    stop_reason="arbitration_authority_violation",
                )
                arbitration = None
            elif arbitration.resolution_status is ResolutionStatus.UNRESOLVED:
                final_decision = resolve_stage3_reports(
                    *reports,
                    arbitration=arbitration,
                    confidence_threshold=self._config.consensus_confidence,
                    evidence_items=canonical_classification_view.items,
                    force_fallback=True,
                )
            else:
                final_decision = resolve_stage3_reports(
                    *reports,
                    arbitration=arbitration,
                    confidence_threshold=self._config.consensus_confidence,
                    evidence_items=canonical_classification_view.items,
                )

        if (
            final_decision is None
            and unresolved is None
            and all(
                report.consistency.status is not ConsistencyStatus.CONSISTENT
                for report in reports
            )
        ):
            unresolved = _arbitration_unresolved(
                disagreement,
                evidence_phase,
                missing_facts=("No Stage 3 team passed causal consistency.",),
                stop_reason="no_consistent_stage3_candidate",
            )

        verification_arbitration = (
            arbitration
            if arbitration is not None
            and arbitration.resolution_status is ResolutionStatus.RESOLVED
            else None
        )

        verification = (
            verify_stage3_decision(
                final_decision,
                _FrozenStage3EvidenceLedger(canonical_classification_view),
                reports=reports,
                readiness=evidence_phase.gap_report,
                disagreement=disagreement,
                arbitration=verification_arbitration,
                arbitration_evidence_ids=(
                    arbitration_evidence_ids
                    if verification_arbitration is not None
                    else None
                ),
                boundary_challenge=boundary_challenge,
            )
            if final_decision is not None
            else None
        )
        pre_gate_candidate = final_decision
        pre_gate_verification = verification
        preservation_result: BaselinePreservationResult | None = None
        revision_certificates: tuple[LabelRevisionCertificate, ...] = ()
        revision_audit = (
            Stage3RevisionAudit(
                assessment_policy_version=HETEROGENEOUS_REVISION_ASSESSMENT_POLICY,
                cross_policy_version=TASK_SPECIFIC_REVISION_CROSS_POLICY,
            )
            if self._uses_baseline_preservation
            else None
        )
        preservation_allowed = unresolved is None or (
            unresolved.stop_reason == "no_consistent_stage3_candidate"
        )
        if self._uses_baseline_preservation and preservation_allowed:
            outcome = self._apply_baseline_preservation(
                reports=reports,
                candidate=pre_gate_candidate,
                candidate_verification=pre_gate_verification,
                view=canonical_classification_view,
                component_failures=component_failures,
                readiness=evidence_phase.gap_report,
                disagreement=disagreement,
                arbitration=verification_arbitration,
                arbitration_evidence_ids=(
                    arbitration_evidence_ids
                    if verification_arbitration is not None
                    else None
                ),
                boundary_challenge=boundary_challenge,
            )
            reports = outcome.reports
            final_decision = outcome.final_decision
            verification = outcome.verification
            component_failures = outcome.component_failures
            revision_certificates = outcome.certificates
            preservation_result = outcome.preservation_result
            revision_audit = outcome.revision_audit
            if outcome.unresolved is not None:
                unresolved = outcome.unresolved
            elif outcome.final_decision is not None:
                unresolved = None
        return Stage3WorkflowResult(
            reports=reports,
            final_decision=final_decision,
            verification=verification,
            evidence_deltas=evidence_phase.evidence_deltas,
            retrieval_source_request_ids=list(
                evidence_phase.retrieval_source_request_ids
            ),
            budget_exhausted=evidence_phase.budget_exhausted,
            validity_report=evidence_phase.validity_report,
            readiness_report=evidence_phase.gap_report,
            classification_ledger_version=classification_ledger_version,
            unresolved=unresolved,
            boundary_challenge=boundary_challenge,
            supervision=supervision,
            component_failures=component_failures,
            pre_gate_candidate=(
                pre_gate_candidate if self._uses_baseline_preservation else None
            ),
            pre_gate_verification=(
                pre_gate_verification if self._uses_baseline_preservation else None
            ),
            pre_gate_disagreement=(
                disagreement if self._uses_baseline_preservation else None
            ),
            pre_gate_arbitration=(
                verification_arbitration if self._uses_baseline_preservation else None
            ),
            pre_gate_arbitration_evidence_ids=(
                arbitration_evidence_ids
                if self._uses_baseline_preservation
                and verification_arbitration is not None
                else None
            ),
            pre_gate_consensus_confidence=(
                self._config.consensus_confidence
                if self._uses_baseline_preservation
                else None
            ),
            baseline_anchor=(
                self._baseline_anchor if self._uses_baseline_preservation else None
            ),
            revision_certificates=revision_certificates,
            preservation_result=preservation_result,
            expected_baseline_anchor_hash=(
                self._expected_baseline_anchor_hash
                if self._uses_baseline_preservation
                else None
            ),
            expected_taxonomy_structure_hash=(
                self._expected_taxonomy_structure_hash
                if self._uses_baseline_preservation
                else None
            ),
            revision_audit=revision_audit,
        )

    RootCauseReport,
