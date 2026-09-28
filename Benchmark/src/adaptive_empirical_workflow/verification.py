"""Deterministic verification of proposed workflow conclusions."""

from __future__ import annotations

import math
from collections.abc import Collection
from Benchmark.src.annotation_contracts import label_valid

from .contracts import (
    ArbitrationSource,
    BaselineAnchor,
    BaselinePreservationResult,
    BoundaryChallenge,
    BoundaryChallengeAction,
    DisagreementMap,
    EvidenceDimension,
    EvidenceItem,
    EvidenceReadinessReport,
    EvidenceView,
    Stage2AnalysisReport,
    Stage2ArbitrationDecision,
    Stage2FinalDecision,
    Stage2Verification,
    Stage3FinalDecision,
    Stage3ArbitrationDecision,
    Stage3ComponentFailure,
    Stage3TeamReport,
    Stage3RevisionAudit,
    Stage3Verification,
    LabelRevisionCertificate,
    TaxonomyStructure,
    normalize_revision_entailment_result,
    normalize_revision_falsification_result,
    revision_proposal_digest,
    HETEROGENEOUS_REVISION_ASSESSMENT_POLICY,
    TASK_SPECIFIC_REVISION_CROSS_POLICY,
)
from .baseline_preservation import (
    PreservationGateError,
    apply_baseline_preservation_gate,
    compose_baseline_preservation_decision,
)
from .capabilities import required_readiness_dimensions
from .evidence_capabilities import (
    assess_stage3_evidence_validity,
    supports_readiness_dimension,
)
from .ledger import EvidenceLedger
from .stage2_policy import stage2_acceptance_errors
from .stage3_composition import (
    boundary_challenge_semantic_errors,
    build_revision_proposals,
    compose_baseline_revision_certificates,
    stage3_correction_semantic_errors,
)
from .taxonomy_structure import route_boundary_card


def stage3_report_evidence_ids(
    report: Stage3TeamReport,
    *,
    dimensions: Collection[str] | None = None,
) -> tuple[str, ...]:
    """Return decision-owned report citations, deduplicated in first-use order."""

    requested = set(dimensions or ())
    include_all = dimensions is None
    include_symptom = include_all or any(
        "symptom" in dimension for dimension in requested
    )
    include_root = include_all or any(
        "root_cause" in dimension or dimension == "cause_review"
        for dimension in requested
    )
    include_consistency = include_all or any(
        "causal" in dimension or "consistency" in dimension for dimension in requested
    )
    if "consensus_gate" in requested:
        include_symptom = True
        include_root = True

    cited: list[str] = []

    def add_classification(report_part: object) -> None:
        cited.extend(getattr(report_part, "supporting_evidence_ids"))
        cited.extend(getattr(report_part, "counter_evidence_ids"))
        cited.extend(getattr(report_part, "boundary_evidence_ids"))

    if report.anchor is not None:
        if include_symptom:
            add_classification(report.anchor.symptom)
        if include_root:
            add_classification(report.anchor.root_cause)
        if include_all or (include_symptom and include_root):
            cited.extend(report.anchor.shared_supporting_evidence_ids)
            cited.extend(report.anchor.shared_counter_evidence_ids)
            cited.extend(report.anchor.shared_boundary_evidence_ids)
        for review in report.verifications:
            if (review.dimension is EvidenceDimension.SYMPTOM and include_symptom) or (
                review.dimension is EvidenceDimension.ROOT_CAUSE and include_root
            ):
                cited.extend(review.supporting_evidence_ids)
                cited.extend(review.counter_evidence_ids)
        for audit in report.correction_audit:
            if not audit.accepted:
                continue
            if (audit.dimension is EvidenceDimension.SYMPTOM and include_symptom) or (
                audit.dimension is EvidenceDimension.ROOT_CAUSE and include_root
            ):
                cited.extend(audit.supporting_evidence_ids)

    if include_symptom:
        add_classification(report.symptom)
    if include_root:
        add_classification(report.root_cause)
    if include_consistency:
        cited.extend(report.consistency.supporting_evidence_ids)
    return tuple(dict.fromkeys(cited))


def _classification_dimensions_are_complete(records: tuple[object, ...]) -> bool:
    dimensions = [getattr(record, "dimension", None) for record in records]
    return len(dimensions) == 2 and set(dimensions) == {
        EvidenceDimension.SYMPTOM,
        EvidenceDimension.ROOT_CAUSE,
    }


def stage3_report_provenance_errors(
    report: Stage3TeamReport,
    view: EvidenceView,
) -> tuple[str, ...]:
    """Validate one report against the canonical valid-only Stage 3 evidence."""

    anchored_fields_present = bool(
        report.anchor is not None or report.verifications or report.correction_audit
    )
    if not anchored_fields_present:
        known_ids = {item.evidence_id for item in view.items}
        return tuple(
            f"team {report.team_id} cites unknown evidence_id {evidence_id!r}"
            for evidence_id in stage3_report_evidence_ids(report)
            if evidence_id not in known_ids
        )

    structure_errors: list[str] = []
    if report.anchor is None:
        structure_errors.append(
            f"team {report.team_id} anchored report is missing its anchor"
        )
    if not _classification_dimensions_are_complete(report.verifications):
        structure_errors.append(
            f"team {report.team_id} anchored report verifications must contain "
            "exactly one symptom and one root cause record"
        )
    if not _classification_dimensions_are_complete(report.correction_audit):
        structure_errors.append(
            f"team {report.team_id} anchored report correction audit must contain "
            "exactly one symptom and one root cause record"
        )
    if structure_errors:
        return tuple(structure_errors)

    semantic_errors = stage3_correction_semantic_errors(report)

    validity = assess_stage3_evidence_validity(view)
    valid_ids = set(validity.valid_evidence_ids)
    evidence_by_id = {
        item.evidence_id: item for item in view.items if item.evidence_id in valid_ids
    }
    errors: list[str] = list(semantic_errors)
    all_ids = stage3_report_evidence_ids(report)
    errors.extend(
        f"team {report.team_id} cites unknown evidence_id {evidence_id!r}"
        for evidence_id in all_ids
        if evidence_id not in evidence_by_id
    )

    def validate_owned(
        evidence_ids: Collection[str],
        dimension: EvidenceDimension,
        owner: str,
    ) -> None:
        for evidence_id in evidence_ids:
            item = evidence_by_id.get(evidence_id)
            if item is not None and not supports_readiness_dimension(
                item,
                dimension,
                domain=view.domain_profile,
            ):
                errors.append(
                    f"team {report.team_id} {owner} {dimension.value} cites "
                    f"capability-incompatible evidence_id {evidence_id!r}"
                )

    def validate_task_relevant(evidence_ids: Collection[str], owner: str) -> None:
        for evidence_id in evidence_ids:
            item = evidence_by_id.get(evidence_id)
            if item is not None and not any(
                supports_readiness_dimension(
                    item,
                    dimension,
                    domain=view.domain_profile,
                )
                for dimension in (
                    EvidenceDimension.SYMPTOM,
                    EvidenceDimension.ROOT_CAUSE,
                )
            ):
                errors.append(
                    f"team {report.team_id} {owner} cites task-irrelevant "
                    f"evidence_id {evidence_id!r}"
                )

    assert report.anchor is not None
    for owner, report_part, dimension in (
        ("anchor", report.anchor.symptom, EvidenceDimension.SYMPTOM),
        ("anchor", report.anchor.root_cause, EvidenceDimension.ROOT_CAUSE),
        ("composed", report.symptom, EvidenceDimension.SYMPTOM),
        ("composed", report.root_cause, EvidenceDimension.ROOT_CAUSE),
    ):
        validate_owned(
            (
                *report_part.supporting_evidence_ids,
                *report_part.counter_evidence_ids,
                *report_part.boundary_evidence_ids,
            ),
            dimension,
            owner,
        )
    validate_task_relevant(
        (
            *report.anchor.shared_supporting_evidence_ids,
            *report.anchor.shared_counter_evidence_ids,
            *report.anchor.shared_boundary_evidence_ids,
        ),
        "anchor shared evidence",
    )
    validate_task_relevant(
        report.consistency.supporting_evidence_ids,
        "causal consistency",
    )
    for review in report.verifications:
        validate_owned(
            (*review.supporting_evidence_ids, *review.counter_evidence_ids),
            review.dimension,
            "verifier",
        )
    for audit in report.correction_audit:
        if audit.accepted:
            if not audit.supporting_evidence_ids:
                errors.append(
                    f"team {report.team_id} accepted correction audit for "
                    f"{audit.dimension.value} has no supporting evidence"
                )
            validate_owned(
                audit.supporting_evidence_ids,
                audit.dimension,
                "accepted correction audit",
            )
    return tuple(dict.fromkeys(errors))


def readiness_provenance_errors(
    readiness: EvidenceReadinessReport,
    evidence_items: tuple[EvidenceItem, ...],
    *,
    domain: str = "issta2024",
    insufficiency_is_error: bool = True,
) -> list[str]:
    """Validate readiness against one exact canonical evidence snapshot."""

    evidence_by_id = {item.evidence_id: item for item in evidence_items}
    errors: list[str] = []
    for dimension in readiness.dimensions:
        name = dimension.dimension.value
        if not dimension.sufficient:
            if insufficiency_is_error:
                errors.append(f"required readiness dimension {name} is insufficient")
        if dimension.sufficient and not dimension.confirmed_evidence_ids:
            errors.append(
                f"required readiness dimension {name} has no confirmed evidence"
            )
        errors.extend(
            f"required readiness dimension {name} cites unknown confirmed "
            f"evidence_id {evidence_id!r}"
            for evidence_id in dimension.confirmed_evidence_ids
            if evidence_id not in evidence_by_id
        )
        known_items = tuple(
            evidence_by_id[evidence_id]
            for evidence_id in dimension.confirmed_evidence_ids
            if evidence_id in evidence_by_id
        )
        capable_ids = {
            item.evidence_id
            for item in known_items
            if supports_readiness_dimension(
                item,
                dimension.dimension,
                domain=domain,
            )
        }
        if known_items and not capable_ids:
            errors.append(
                f"required readiness dimension {name} has no "
                "capability-linked confirmed evidence"
            )
        elif capable_ids:
            errors.extend(
                f"required readiness dimension {name} cites "
                "capability-incompatible confirmed evidence_id "
                f"{item.evidence_id!r}"
                for item in known_items
                if item.evidence_id not in capable_ids
            )
    return errors


def _readiness_structure_errors(
    readiness: EvidenceReadinessReport,
    *,
    domain: str,
    task: str,
) -> list[str]:
    """Validate the stage/domain readiness envelope before its evidence claims."""

    errors: list[str] = []
    if readiness.task != task:
        errors.append(
            f"readiness task {readiness.task!r} does not match "
            f"verification stage {task!r}"
        )
    required = required_readiness_dimensions(domain, task)
    reported = tuple(dimension.dimension for dimension in readiness.dimensions)
    if len(reported) != len(required) or set(reported) != set(required):
        errors.append(
            "readiness dimensions do not exactly match required dimensions for "
            f"{domain} {task}"
        )
    return errors


def _readiness_verification_errors(
    readiness: EvidenceReadinessReport,
    evidence_items: tuple[EvidenceItem, ...],
    *,
    domain: str,
    task: str,
    insufficiency_is_error: bool = True,
) -> list[str]:
    structure_errors = _readiness_structure_errors(
        readiness,
        domain=domain,
        task=task,
    )
    if structure_errors:
        return structure_errors
    return readiness_provenance_errors(
        readiness,
        evidence_items,
        domain=domain,
        insufficiency_is_error=insufficiency_is_error,
    )


def _resolution_accounting_errors(
    disagreement: DisagreementMap,
    resolved_dimensions: list[str],
) -> list[str]:
    listed = set(disagreement.dimensions)
    resolved = set(resolved_dimensions)
    errors = [
        f"arbitration resolved unlisted dimension {dimension!r}"
        for dimension in resolved_dimensions
        if dimension not in listed
    ]
    errors.extend(
        f"arbitration did not resolve active dimension {dimension!r}"
        for dimension in disagreement.dimensions
        if dimension not in resolved
    )
    return errors


def verify_stage2_decision(
    decision: Stage2FinalDecision,
    ledger: EvidenceLedger,
    reports: tuple[Stage2AnalysisReport, ...] = (),
    *,
    readiness: EvidenceReadinessReport | None = None,
    disagreement: DisagreementMap | None = None,
    arbitration: Stage2ArbitrationDecision | None = None,
    arbitration_evidence_ids: tuple[str, ...] | None = None,
) -> Stage2Verification:
    """Validate provenance while preserving the proposed decision label."""

    canonical_view = ledger.view()
    known_ids = {item.evidence_id for item in canonical_view.items}
    errors = [
        f"unknown evidence_id {evidence_id!r}"
        for evidence_id in decision.supporting_evidence_ids
        if evidence_id not in known_ids
    ]
    domain = canonical_view.domain_profile
    if readiness is not None:
        errors.extend(
            _readiness_verification_errors(
                readiness,
                canonical_view.items,
                domain=domain,
                task="stage2",
            )
        )
    if disagreement is not None and arbitration is not None:
        listed = set(disagreement.dimensions)
        errors.extend(
            _resolution_accounting_errors(
                disagreement,
                arbitration.resolved_dimensions,
            )
        )
        if (
            reports
            and "decision" not in listed
            and any(report.decision is not decision.decision for report in reports)
        ):
            errors.append("arbitration changed unlisted dimension 'decision'")
    if arbitration_evidence_ids is not None:
        allowed = set(arbitration_evidence_ids)
        errors.extend(
            f"evidence_id {evidence_id!r} was not included in arbitration packet"
            for evidence_id in decision.supporting_evidence_ids
            if evidence_id not in allowed
        )
    if decision.decision.value == "accepted_fault" and reports:
        report_errors = [
            stage2_acceptance_errors(domain, report.evidence_tests)
            for report in reports
        ]
        if not any(not candidate_errors for candidate_errors in report_errors):
            errors.extend(
                dict.fromkeys(
                    error
                    for candidate_errors in report_errors
                    for error in candidate_errors
                )
            )
    return Stage2Verification(
        decision=decision.decision,
        valid=not errors,
        errors=errors,
    )


def verify_stage3_decision(
    decision: Stage3FinalDecision,
    ledger: EvidenceLedger,
    *,
    reports: tuple[Stage3TeamReport, ...] = (),
    readiness: EvidenceReadinessReport | None = None,
    disagreement: DisagreementMap | None = None,
    arbitration: Stage3ArbitrationDecision | None = None,
    arbitration_evidence_ids: tuple[str, ...] | None = None,
    baseline_anchor: BaselineAnchor | None = None,
    pre_gate_candidate: Stage3FinalDecision | None = None,
    pre_gate_verification: Stage3Verification | None = None,
    revision_certificates: tuple[LabelRevisionCertificate, ...] | None = None,
    preservation_result: BaselinePreservationResult | None = None,
    revision_audit: Stage3RevisionAudit | None = None,
    taxonomy_structure: TaxonomyStructure | None = None,
    expected_baseline_config_hash: str | None = None,
    expected_baseline_predictions_sha256: str | None = None,
    expected_baseline_anchor_hash: str | None = None,
    expected_taxonomy_structure_hash: str | None = None,
    expected_revision_assessment_policy: str = (
        HETEROGENEOUS_REVISION_ASSESSMENT_POLICY
    ),
    expected_revision_cross_policy: str = TASK_SPECIFIC_REVISION_CROSS_POLICY,
    boundary_challenge: BoundaryChallenge | None = None,
    component_failures: tuple[Stage3ComponentFailure, ...] = (),
    consensus_confidence: float = 0.8,
) -> Stage3Verification:
    """Validate both taxonomy membership and cited evidence provenance."""

    def canonical_model(value, model_type, *, name: str):
        try:
            return model_type.model_validate(value.model_dump(mode="python"))
        except (AttributeError, ValueError) as error:
            raise ValueError(f"invalid canonical Stage 3 {name}: {error}") from error

    try:
        decision = canonical_model(
            decision, Stage3FinalDecision, name="post-gate decision"
        )
        reports = tuple(
            canonical_model(
                report,
                Stage3TeamReport,
                name="team report verifications correction audit",
            )
            for report in reports
        )
        if readiness is not None:
            readiness = canonical_model(
                readiness, EvidenceReadinessReport, name="readiness report"
            )
        if disagreement is not None:
            disagreement = canonical_model(
                disagreement, DisagreementMap, name="disagreement map"
            )
        if arbitration is not None:
            arbitration = canonical_model(
                arbitration, Stage3ArbitrationDecision, name="arbitration"
            )
        if boundary_challenge is not None:
            boundary_challenge = canonical_model(
                boundary_challenge, BoundaryChallenge, name="boundary challenge"
            )
        component_failures = tuple(
            canonical_model(failure, Stage3ComponentFailure, name="component failure")
            for failure in component_failures
        )
        if pre_gate_candidate is not None:
            pre_gate_candidate = canonical_model(
                pre_gate_candidate,
                Stage3FinalDecision,
                name="pre-gate candidate",
            )
        if pre_gate_verification is not None:
            pre_gate_verification = canonical_model(
                pre_gate_verification,
                Stage3Verification,
                name="pre-gate verification",
            )
    except ValueError as error:
        return Stage3Verification(
            symptom_label=str(getattr(decision, "symptom_label", "")),
            root_cause_label=str(getattr(decision, "root_cause_label", "")),
            valid=False,
            errors=(str(error),),
        )

    canonical_view = ledger.view()
    preservation_view = canonical_view
    anchored_reports = tuple(
        report
        for report in reports
        if report.anchor is not None or report.verifications or report.correction_audit
    )
    if anchored_reports:
        validity = assess_stage3_evidence_validity(canonical_view)
        valid_ids = set(validity.valid_evidence_ids)
        canonical_view = canonical_view.model_copy(
            update={
                "items": tuple(
                    item
                    for item in canonical_view.items
                    if item.evidence_id in valid_ids
                )
            }
        )
    taxonomy = canonical_view.taxonomy
    errors: list[str] = []
    if not label_valid(taxonomy, "symptom", decision.symptom_label):
        errors.append(f"unknown symptom label {decision.symptom_label!r}")
    if not label_valid(taxonomy, "root_cause", decision.root_cause_label):
        errors.append(f"unknown root_cause label {decision.root_cause_label!r}")
    known_ids = {item.evidence_id for item in canonical_view.items}
    errors.extend(
        f"unknown evidence_id {evidence_id!r}"
        for evidence_id in decision.supporting_evidence_ids
        if evidence_id not in known_ids
    )
    if readiness is not None:
        errors.extend(
            _readiness_verification_errors(
                readiness,
                canonical_view.items,
                domain=canonical_view.domain_profile,
                task="stage3",
                insufficiency_is_error=False,
            )
        )
    for report in reports:
        if report in anchored_reports:
            errors.extend(stage3_report_provenance_errors(report, canonical_view))
        for role_name, label, alternative_label in (
            (
                "symptom",
                report.symptom.label,
                report.symptom.alternative_label,
            ),
            (
                "root_cause",
                report.root_cause.label,
                report.root_cause.alternative_label,
            ),
        ):
            allowed_labels = taxonomy.get(role_name, [])
            if not label_valid(taxonomy, role_name, label):
                errors.append(
                    f"team {report.team_id} returned unknown {role_name} label {label!r}"
                )
            if not label_valid(taxonomy, role_name, alternative_label):
                errors.append(
                    f"team {report.team_id} returned unknown {role_name} alternative "
                    f"label {alternative_label!r}"
                )
    boundary_errors = (
        boundary_challenge_semantic_errors(
            boundary_challenge,
            reports,
            canonical_view,
        )
        if boundary_challenge is not None
        else ()
    )
    errors.extend(boundary_errors)
    verified_boundary_challenge = boundary_challenge if not boundary_errors else None
    if decision.source.value == "fallback_uncertain":
        selected_evidence_ids = set(decision.supporting_evidence_ids)
        backed_by_consistent_report = any(
            report.consistency.status.value == "consistent"
            and report.symptom.label == decision.symptom_label
            and report.root_cause.label == decision.root_cause_label
            and selected_evidence_ids.issubset(
                set(
                    (
                        *report.symptom.supporting_evidence_ids,
                        *report.root_cause.supporting_evidence_ids,
                        *report.consistency.supporting_evidence_ids,
                    )
                )
            )
            for report in reports
        )
        if not backed_by_consistent_report:
            errors.append(
                "fallback_uncertain is not backed by a consistent team report"
            )
    preservation_values = (
        baseline_anchor,
        revision_certificates,
        preservation_result,
        taxonomy_structure,
        expected_baseline_config_hash,
        expected_baseline_predictions_sha256,
        expected_baseline_anchor_hash,
        expected_taxonomy_structure_hash,
    )
    complete_preservation_surface = all(
        value is not None for value in preservation_values
    )
    arbitration_authority_decision = (
        pre_gate_candidate if complete_preservation_surface else decision
    )
    if disagreement is not None and arbitration is not None:
        listed = set(disagreement.dimensions)
        errors.extend(
            _resolution_accounting_errors(
                disagreement,
                arbitration.resolved_dimensions,
            )
        )
        symptom_authorized = "symptom_label" in listed
        root_authorized = "root_cause_label" in listed
        if arbitration_authority_decision is not None:
            if (
                reports
                and not symptom_authorized
                and any(
                    report.symptom.label != arbitration_authority_decision.symptom_label
                    for report in reports
                )
            ):
                errors.append("arbitration changed unlisted dimension 'symptom_label'")
            if (
                reports
                and not root_authorized
                and any(
                    report.root_cause.label
                    != arbitration_authority_decision.root_cause_label
                    for report in reports
                )
            ):
                errors.append(
                    "arbitration changed unlisted dimension 'root_cause_label'"
                )
    if arbitration_evidence_ids is not None:
        allowed = set(arbitration_evidence_ids)
        errors.extend(
            f"evidence_id {evidence_id!r} was not included in arbitration packet"
            for evidence_id in decision.supporting_evidence_ids
            if evidence_id not in allowed
        )
    if any(value is not None for value in preservation_values):
        if any(value is None for value in preservation_values):
            errors.append("partial Baseline preservation verification surface")
        else:
            assert baseline_anchor is not None
            assert revision_certificates is not None
            assert preservation_result is not None
            assert taxonomy_structure is not None
            assert expected_baseline_config_hash is not None
            assert expected_baseline_predictions_sha256 is not None
            assert expected_baseline_anchor_hash is not None
            assert expected_taxonomy_structure_hash is not None
            try:
                canonical_reports = tuple(
                    Stage3TeamReport.model_validate(report.model_dump(mode="python"))
                    for report in reports
                )
                canonical_disagreement = (
                    DisagreementMap.model_validate(
                        disagreement.model_dump(mode="python")
                    )
                    if disagreement is not None
                    else None
                )
                canonical_arbitration = (
                    Stage3ArbitrationDecision.model_validate(
                        arbitration.model_dump(mode="python")
                    )
                    if arbitration is not None
                    else None
                )
                canonical_boundary = (
                    BoundaryChallenge.model_validate(
                        verified_boundary_challenge.model_dump(mode="python")
                    )
                    if verified_boundary_challenge is not None
                    else None
                )
                canonical_failures = tuple(
                    Stage3ComponentFailure.model_validate(
                        failure.model_dump(mode="python")
                    )
                    for failure in component_failures
                )
            except (AttributeError, ValueError) as error:
                canonical_reports = ()
                canonical_disagreement = None
                canonical_arbitration = None
                canonical_boundary = None
                canonical_failures = ()
                errors.append(f"invalid canonical Stage 3 producer context: {error}")
            if (
                isinstance(consensus_confidence, bool)
                or not isinstance(consensus_confidence, (int, float))
                or not math.isfinite(float(consensus_confidence))
                or not 0.0 <= float(consensus_confidence) <= 1.0
            ):
                canonical_consensus_confidence = None
                errors.append(
                    "invalid canonical Stage 3 consensus confidence: "
                    f"{consensus_confidence!r}"
                )
            else:
                canonical_consensus_confidence = float(consensus_confidence)
            try:
                canonical_anchor = BaselineAnchor.model_validate(
                    baseline_anchor.model_dump(mode="python")
                )
                canonical_structure = TaxonomyStructure.model_validate(
                    taxonomy_structure.model_dump(mode="python")
                )
                canonical_input_certificates = tuple(
                    LabelRevisionCertificate.model_validate(
                        certificate.model_dump(mode="python")
                    )
                    for certificate in revision_certificates
                )
                canonical_preservation = BaselinePreservationResult.model_validate(
                    preservation_result.model_dump(mode="python")
                )
                proposal_source_reports = tuple(
                    report.model_copy(
                        update={
                            "baseline_revision_assessments": (),
                            "revision_consistency": (),
                        }
                    )
                    for report in canonical_reports
                )
                recomputed_proposals = build_revision_proposals(
                    anchor=canonical_anchor,
                    reports=proposal_source_reports,
                    structure=canonical_structure,
                    view=preservation_view,
                    boundary_challenge=canonical_boundary,
                )
                recomputed_certificates = compose_baseline_revision_certificates(
                    anchor=canonical_anchor,
                    reports=canonical_reports,
                    proposals=recomputed_proposals,
                    structure=canonical_structure,
                    view=preservation_view,
                    boundary_challenge=canonical_boundary,
                )
                canonical_revision_audit = (
                    Stage3RevisionAudit.model_validate(
                        revision_audit.model_dump(mode="python")
                    )
                    if revision_audit is not None
                    else None
                )
                preservation_inputs_valid = True
            except (AssertionError, AttributeError, TypeError, ValueError) as error:
                canonical_anchor = None
                canonical_structure = None
                canonical_input_certificates = ()
                canonical_preservation = None
                recomputed_proposals = ()
                recomputed_certificates = ()
                canonical_revision_audit = None
                preservation_inputs_valid = False
                errors.append(
                    f"invalid canonical Baseline preservation context: {error}"
                )
            if canonical_revision_audit is None:
                errors.append(
                    "revision audit policy is required for Baseline preservation"
                )
            elif (
                canonical_revision_audit.assessment_policy_version
                != expected_revision_assessment_policy
                or canonical_revision_audit.cross_policy_version
                != expected_revision_cross_policy
            ):
                errors.append(
                    "revision audit policy does not match the expected assessment/cross policy"
                )
            if (
                canonical_revision_audit is not None
                and preservation_inputs_valid
                and canonical_revision_audit.assessment_policy_version
                == expected_revision_assessment_policy
                and canonical_revision_audit.cross_policy_version
                == expected_revision_cross_policy
            ):
                proposal_by_digest = {
                    revision_proposal_digest(proposal): proposal
                    for proposal in recomputed_proposals
                }
                report_assessments = {
                    (
                        assessment.proposal_digest,
                        assessment.assessor_team_id,
                    ): assessment
                    for report in canonical_reports
                    for assessment in report.baseline_revision_assessments
                }
                for key, assessment in report_assessments.items():
                    proposal = proposal_by_digest.get(assessment.proposal_digest or "")
                    if proposal is None:
                        errors.append(
                            "raw assessment adapter cannot bind a canonical proposal"
                        )
                        continue
                    try:
                        card = route_boundary_card(
                            canonical_structure,
                            proposal.dimension,
                            proposal.baseline_label,
                            proposal.proposed_label,
                        )
                        if assessment.raw_entailment_result is not None:
                            normalized = normalize_revision_entailment_result(
                                result=assessment.raw_entailment_result,
                                proposal=proposal,
                                routed_card=card,
                            )
                        elif assessment.raw_falsification_result is not None:
                            normalized = normalize_revision_falsification_result(
                                result=assessment.raw_falsification_result,
                                proposal=proposal,
                                routed_card=card,
                            )
                        elif (
                            canonical_revision_audit.assessment_policy_version
                            == "heterogeneous-entailment-falsification-v1"
                        ):
                            raise ValueError(
                                "heterogeneous assessment lacks a task-owned raw result"
                            )
                        else:
                            continue
                    except (KeyError, TypeError, ValueError) as error:
                        errors.append(f"raw assessment adapter invalid: {error}")
                    else:
                        if normalized != assessment:
                            errors.append(
                                "raw assessment adapter does not match normalized assessment"
                            )
                audited_assessments = {
                    (attempt.proposal_digest, attempt.assessor_team_id): (
                        attempt.returned_assessment
                    )
                    for attempt in canonical_revision_audit.assessment_attempts
                    if attempt.returned_assessment is not None
                }
                if audited_assessments != report_assessments:
                    errors.append(
                        "raw assessment adapter audit does not match canonical reports"
                    )
                report_consistency = {
                    (record.proposal_digest, record.checker_team_id): record
                    for report in canonical_reports
                    for record in report.revision_consistency
                }
                audited_consistency = {
                    (attempt.proposal_digest, attempt.checker_team_id): (
                        attempt.returned_consistency
                    )
                    for attempt in canonical_revision_audit.consistency_attempts
                    if attempt.returned_consistency is not None
                }
                if audited_consistency != report_consistency:
                    errors.append(
                        "raw assessment adapter cross audit does not match canonical reports"
                    )
                if canonical_revision_audit.proposal_digests != tuple(
                    proposal_by_digest
                ):
                    errors.append(
                        "raw assessment adapter audit proposal set is not canonical"
                    )
            if (
                preservation_inputs_valid
                and recomputed_certificates != canonical_input_certificates
            ):
                errors.append(
                    "revision certificates do not match canonical team reports"
                )
            recomputed_pre_gate_verification = (
                verify_stage3_decision(
                    pre_gate_candidate,
                    ledger,
                    reports=canonical_reports,
                    readiness=readiness,
                    disagreement=canonical_disagreement,
                    arbitration=canonical_arbitration,
                    arbitration_evidence_ids=arbitration_evidence_ids,
                )
                if pre_gate_candidate is not None
                else None
            )
            if recomputed_pre_gate_verification != pre_gate_verification:
                errors.append(
                    "pre-gate verification does not match canonical producer reports"
                )
            force_fallback = (
                canonical_boundary is not None
                and canonical_boundary.action
                in {
                    BoundaryChallengeAction.SYMPTOM_REVIEW,
                    BoundaryChallengeAction.CAUSE_REVIEW,
                    BoundaryChallengeAction.EVIDENCE_REQUEST,
                }
                and canonical_arbitration is None
            ) or any(
                failure.role == "boundary_challenger" for failure in canonical_failures
            )
            from .disagreement import (
                resolve_single_stage3_report,
                resolve_stage3_reports,
            )

            canonical_candidate: Stage3FinalDecision | None
            if (
                pre_gate_candidate is not None
                and pre_gate_candidate.source is ArbitrationSource.TARGETED_ARBITRATION
                and canonical_arbitration is None
            ):
                errors.append(
                    "targeted-arbitration candidate requires canonical "
                    "arbitration authority"
                )
            if canonical_consensus_confidence is None:
                canonical_candidate = None
            elif len(canonical_reports) == 1:
                canonical_candidate = resolve_single_stage3_report(canonical_reports[0])
            elif len(canonical_reports) == 2:
                canonical_candidate = resolve_stage3_reports(
                    *canonical_reports,
                    arbitration=canonical_arbitration,
                    confidence_threshold=canonical_consensus_confidence,
                    evidence_items=canonical_view.items,
                    force_fallback=force_fallback,
                )
            else:
                canonical_candidate = None
            if canonical_candidate != pre_gate_candidate:
                errors.append(
                    "producer authority: pre-gate candidate does not match "
                    "canonical producer candidate"
                )
            try:
                if not preservation_inputs_valid:
                    raise PreservationGateError(
                        "canonical Baseline preservation context is invalid"
                    )
                assert canonical_anchor is not None
                assert canonical_structure is not None
                recomputed = apply_baseline_preservation_gate(
                    anchor=canonical_anchor,
                    candidate=pre_gate_candidate,
                    candidate_verification=recomputed_pre_gate_verification,
                    certificates=recomputed_certificates,
                    proposals=recomputed_proposals,
                    structure=canonical_structure,
                    view=preservation_view,
                    expected_source_config_hash=expected_baseline_config_hash,
                    expected_source_predictions_sha256=(
                        expected_baseline_predictions_sha256
                    ),
                    expected_anchor_hash=expected_baseline_anchor_hash,
                    expected_structure_hash=expected_taxonomy_structure_hash,
                )
            except PreservationGateError as error:
                errors.append(f"Baseline preservation gate invalid: {error}")
            else:
                if recomputed != canonical_preservation:
                    errors.append(
                        "Baseline preservation result does not match recomputed gate"
                    )
                expected_decision = compose_baseline_preservation_decision(
                    preservation=recomputed,
                    candidate=pre_gate_candidate,
                )
                if decision.model_dump(mode="json") != expected_decision.model_dump(
                    mode="json"
                ):
                    errors.append(
                        "post-gate decision does not match recomputed preservation semantics"
                    )
    return Stage3Verification(
        symptom_label=decision.symptom_label,
        root_cause_label=decision.root_cause_label,
        valid=not errors,
        errors=errors,
    )
