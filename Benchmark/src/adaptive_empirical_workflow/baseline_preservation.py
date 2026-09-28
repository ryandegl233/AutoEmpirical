"""Apply a conservative, evidence-verified gate over frozen Baseline labels."""

from __future__ import annotations

from collections.abc import Sequence

from .contracts import (
    BaselineAnchor,
    canonical_evidence_view_hash,
    BaselineDecisionAction,
    BaselineGateProvenance,
    BaselinePreservationResult,
    EvidenceDimension,
    EvidenceView,
    LabelRevisionCertificate,
    RevisionProposalEnvelope,
    revision_proposal_digest,
    ArbitrationSource,
    Stage3FinalDecision,
    Stage3Verification,
    TaxonomyStructure,
)
from .evidence_capabilities import (
    assess_stage3_evidence_validity,
    supports_readiness_dimension,
)
from .frozen_evidence_runtime import (
    authoritative_revision_support,
    frozen_evidence_may_be_support,
    frozen_evidence_runtime_enabled,
)
from .taxonomy_structure import _validate_against_official_taxonomy
from .taxonomy_structure import (
    materialize_official_boundary_cards,
    taxonomy_structure_hash,
)
from .baseline_anchor import baseline_anchor_hash


class PreservationGateError(ValueError):
    """Raised when a claimed revision certificate violates a safety boundary."""


BASELINE_PRESERVATION_POLICY_VERSION = "baseline-preservation-v1"


_BASELINE_ONLY_RATIONALE = (
    "The frozen Baseline labels were preserved without attributing current "
    "evidence or confidence."
)
_CANDIDATE_BACKED_RATIONALE = (
    "The independent Baseline preservation gate applied only complete dual-team "
    "revision certificates and retained the matching candidate provenance."
)


def compose_baseline_preservation_decision(
    *,
    preservation: BaselinePreservationResult,
    candidate: Stage3FinalDecision | None,
) -> Stage3FinalDecision:
    """Compose deterministic post-gate provenance without inventing support."""

    candidate_matches = candidate is not None and (
        candidate.symptom_label,
        candidate.root_cause_label,
    ) == (preservation.symptom_label, preservation.root_cause_label)
    certificate_citations = tuple(
        dict.fromkeys(
            evidence_id
            for certificate in preservation.applied_certificates
            for evidence_id in (
                *certificate.supporting_evidence_ids,
                *certificate.counter_evidence_ids,
            )
        )
    )
    if certificate_citations:
        provenance = BaselineGateProvenance.REVISION_CERTIFICATE
        confidence = None
        citations = certificate_citations
    elif candidate_matches:
        provenance = BaselineGateProvenance.PRE_GATE_CANDIDATE
        assert candidate is not None
        confidence = candidate.confidence
        citations = candidate.supporting_evidence_ids
    else:
        provenance = BaselineGateProvenance.PRESERVED
        confidence = None
        citations = ()
    return Stage3FinalDecision(
        symptom_label=preservation.symptom_label,
        root_cause_label=preservation.root_cause_label,
        confidence=confidence,
        rationale=(
            _CANDIDATE_BACKED_RATIONALE
            if candidate_matches or certificate_citations
            else _BASELINE_ONLY_RATIONALE
        ),
        supporting_evidence_ids=citations,
        source=ArbitrationSource.BASELINE_PRESERVATION_GATE,
        baseline_gate_provenance=provenance,
    )


def validate_baseline_preservation_context(
    *,
    anchor: BaselineAnchor,
    structure: TaxonomyStructure,
    view: EvidenceView,
    expected_source_config_hash: str,
    expected_source_predictions_sha256: str,
    expected_anchor_hash: str,
    expected_structure_hash: str,
) -> tuple[BaselineAnchor, TaxonomyStructure]:
    """Freeze and validate the record/artifact context before any model call."""

    anchor = _canonical_snapshot(anchor, BaselineAnchor, name="Baseline anchor")
    structure = _canonical_snapshot(
        structure, TaxonomyStructure, name="taxonomy structure"
    )
    if anchor.record_id != view.record_id:
        raise PreservationGateError(
            "Baseline anchor record_id does not match evidence view"
        )
    if anchor.source_config_hash != expected_source_config_hash:
        raise PreservationGateError(
            "Baseline anchor source config hash does not match expected manifest"
        )
    if anchor.source_predictions_sha256 != expected_source_predictions_sha256:
        raise PreservationGateError(
            "Baseline anchor source predictions hash does not match expected artifact"
        )
    if baseline_anchor_hash(anchor) != expected_anchor_hash:
        raise PreservationGateError(
            "Baseline anchor labels do not match the trusted record binding"
        )
    if taxonomy_structure_hash(structure) != expected_structure_hash:
        raise PreservationGateError(
            "taxonomy structure does not match the trusted artifact binding"
        )
    if structure.domain != view.domain_profile:
        raise PreservationGateError(
            "taxonomy structure domain does not match evidence view"
        )
    try:
        _validate_against_official_taxonomy(
            structure,
            view.taxonomy,
            require_complete_official_cards=(
                structure.domain == "ase2022"
                and structure.boundary_cards
                == materialize_official_boundary_cards(structure)
            ),
        )
    except ValueError as error:
        raise PreservationGateError(str(error)) from error
    if anchor.valid:
        for dimension, label in (
            ("symptom", anchor.symptom_label),
            ("root_cause", anchor.root_cause_label),
        ):
            if label not in view.taxonomy.get(dimension, ()):
                raise PreservationGateError(
                    f"anchor label outside {dimension} taxonomy"
                )
    return anchor, structure


def _canonical_snapshot(model, model_type, *, name: str):
    try:
        return model_type.model_validate(model.model_dump(mode="python"))
    except (AttributeError, ValueError) as error:
        raise PreservationGateError(f"invalid {name} contract: {error}") from error


def _dimension_values(
    *,
    dimension: EvidenceDimension,
    anchor: BaselineAnchor,
    candidate: Stage3FinalDecision,
) -> tuple[str, str]:
    if dimension is EvidenceDimension.SYMPTOM:
        assert anchor.symptom_label is not None
        return anchor.symptom_label, candidate.symptom_label
    assert anchor.root_cause_label is not None
    return anchor.root_cause_label, candidate.root_cause_label


def _validate_certificate(
    certificate: LabelRevisionCertificate,
    *,
    anchor: BaselineAnchor,
    candidate: Stage3FinalDecision | None,
    proposal: RevisionProposalEnvelope | None,
    structure: TaxonomyStructure,
    view: EvidenceView,
) -> bool:
    if proposal is not None:
        candidate_label = proposal.proposed_label
        baseline_label = proposal.baseline_label
        if certificate.proposal_digest != revision_proposal_digest(proposal):
            raise PreservationGateError("certificate proposal digest mismatch")
        if (
            proposal.dimension is not certificate.dimension
            or proposal.baseline_source_config_hash != anchor.source_config_hash
            or proposal.baseline_source_predictions_sha256
            != anchor.source_predictions_sha256
            or proposal.taxonomy_structure_hash != taxonomy_structure_hash(structure)
            or proposal.evidence_view_hash != canonical_evidence_view_hash(view)
        ):
            raise PreservationGateError("revision proposal trust binding mismatch")
        if not {
            *certificate.supporting_evidence_ids,
            *certificate.counter_evidence_ids,
        }.issubset(
            {
                *proposal.supporting_evidence_ids,
                *proposal.counter_evidence_ids,
            }
        ):
            raise PreservationGateError(
                "certificate citations exceed proposal authority"
            )
    else:
        if candidate is None:
            raise PreservationGateError("legacy certificate requires a candidate")
        baseline_label, candidate_label = _dimension_values(
            dimension=certificate.dimension,
            anchor=anchor,
            candidate=candidate,
        )
    if certificate.baseline_label != baseline_label:
        raise PreservationGateError("certificate baseline label does not match anchor")
    if certificate.proposed_label != candidate_label:
        raise PreservationGateError(
            "certificate proposed label does not match candidate label"
        )
    if any(team_id not in {"A", "B"} for team_id in certificate.supporting_team_ids):
        raise PreservationGateError("certificate contains an unknown supporting team")

    card = next(
        (
            candidate_card
            for candidate_card in structure.boundary_cards
            if candidate_card.card_id == certificate.boundary_card_id
        ),
        None,
    )
    if card is None:
        raise PreservationGateError("certificate references an unknown Boundary Card")
    if card.dimension != certificate.dimension.value:
        raise PreservationGateError(
            "Boundary Card does not match certificate dimension"
        )
    if set(card.labels) != {baseline_label, candidate_label}:
        raise PreservationGateError(
            "Boundary Card does not match certificate label pair"
        )
    criteria = {criterion.label: criterion for criterion in card.criteria}
    if (
        certificate.contradicted_baseline_condition
        not in criteria[baseline_label].exclusion_conditions
    ):
        raise PreservationGateError(
            "certificate baseline condition is not an exact Boundary Card exclusion"
        )
    if set(certificate.satisfied_proposed_conditions) != set(
        criteria[candidate_label].positive_conditions
    ):
        raise PreservationGateError(
            "certificate proposed conditions are not exact Boundary Card conditions"
        )

    valid_ids = set(assess_stage3_evidence_validity(view).valid_evidence_ids)
    items_by_id = {item.evidence_id: item for item in view.items}
    cited_ids = set(certificate.supporting_evidence_ids) | set(
        certificate.counter_evidence_ids
    )
    unknown_ids = cited_ids - valid_ids
    if unknown_ids:
        raise PreservationGateError(
            "certificate cites unknown evidence IDs: " + ", ".join(sorted(unknown_ids))
        )
    incapable_ids = {
        evidence_id
        for evidence_id in cited_ids
        if not supports_readiness_dimension(
            items_by_id[evidence_id],
            certificate.dimension,
            domain=view.domain_profile,
        )
    }
    if incapable_ids:
        raise PreservationGateError(
            "certificate cites incapable evidence IDs: "
            + ", ".join(sorted(incapable_ids))
        )
    if frozen_evidence_runtime_enabled(view):
        supporting = tuple(
            items_by_id[evidence_id]
            for evidence_id in certificate.supporting_evidence_ids
        )
        if (
            not supporting
            or not all(frozen_evidence_may_be_support(item) for item in supporting)
            or not any(
                authoritative_revision_support(item, view=view) for item in supporting
            )
        ):
            return False
    return set(certificate.supporting_team_ids) == {"A", "B"}


def _validate_candidate_fallback(
    *,
    candidate: Stage3FinalDecision,
    verification: Stage3Verification | None,
    view: EvidenceView,
) -> None:
    candidate = _canonical_snapshot(
        candidate, Stage3FinalDecision, name="Stage 3 candidate"
    )
    if verification is not None:
        verification = _canonical_snapshot(
            verification, Stage3Verification, name="Stage 3 verification"
        )
    if (
        verification is None
        or not verification.valid
        or verification.errors
        or verification.symptom_label != candidate.symptom_label
        or verification.root_cause_label != candidate.root_cause_label
    ):
        raise PreservationGateError(
            "unavailable Baseline requires a matching verified candidate"
        )
    if candidate.source not in {
        ArbitrationSource.DIRECT_CONSENSUS,
        ArbitrationSource.TARGETED_ARBITRATION,
    }:
        raise PreservationGateError(
            "unavailable Baseline requires a non-degraded verified candidate"
        )
    valid_ids = set(assess_stage3_evidence_validity(view).valid_evidence_ids)
    unknown_ids = set(candidate.supporting_evidence_ids) - valid_ids
    if unknown_ids:
        raise PreservationGateError(
            "verified candidate cites unknown evidence IDs: "
            + ", ".join(sorted(unknown_ids))
        )
    items_by_id = {item.evidence_id: item for item in view.items}
    for dimension in (EvidenceDimension.SYMPTOM, EvidenceDimension.ROOT_CAUSE):
        if not any(
            supports_readiness_dimension(
                items_by_id[evidence_id], dimension, domain=view.domain_profile
            )
            for evidence_id in candidate.supporting_evidence_ids
        ):
            raise PreservationGateError(
                "unavailable Baseline requires authoritative evidence for "
                f"{dimension.value}"
            )


def apply_baseline_preservation_gate(
    *,
    anchor: BaselineAnchor,
    candidate: Stage3FinalDecision | None,
    candidate_verification: Stage3Verification | None,
    certificates: Sequence[LabelRevisionCertificate],
    structure: TaxonomyStructure,
    view: EvidenceView,
    expected_source_config_hash: str,
    expected_source_predictions_sha256: str,
    expected_anchor_hash: str,
    expected_structure_hash: str,
    proposals: Sequence[RevisionProposalEnvelope] | None = None,
) -> BaselinePreservationResult:
    """Preserve each valid Baseline label unless a complete certificate revises it."""

    anchor, structure = validate_baseline_preservation_context(
        anchor=anchor,
        structure=structure,
        view=view,
        expected_source_config_hash=expected_source_config_hash,
        expected_source_predictions_sha256=expected_source_predictions_sha256,
        expected_anchor_hash=expected_anchor_hash,
        expected_structure_hash=expected_structure_hash,
    )
    certificates = tuple(
        _canonical_snapshot(
            certificate,
            LabelRevisionCertificate,
            name="revision certificate",
        )
        for certificate in certificates
    )
    candidate_bound_surface = proposals is not None
    canonical_proposals = tuple(
        _canonical_snapshot(
            proposal, RevisionProposalEnvelope, name="revision proposal"
        )
        for proposal in (proposals or ())
    )
    if len(canonical_proposals) != len(set(canonical_proposals)):
        raise PreservationGateError("revision proposals must be unique")
    proposal_by_digest = {
        revision_proposal_digest(proposal): proposal for proposal in canonical_proposals
    }

    if candidate is not None:
        candidate = _canonical_snapshot(
            candidate, Stage3FinalDecision, name="Stage 3 candidate"
        )
        for dimension, label in (
            ("symptom", candidate.symptom_label),
            ("root_cause", candidate.root_cause_label),
        ):
            if label not in view.taxonomy.get(dimension, ()):
                raise PreservationGateError(
                    f"candidate label outside {dimension} taxonomy"
                )
    elif certificates and not canonical_proposals:
        raise PreservationGateError("revision certificates require a candidate")
    certificates_by_dimension: dict[EvidenceDimension, LabelRevisionCertificate] = {}
    for certificate in certificates:
        if certificate.dimension in certificates_by_dimension:
            raise PreservationGateError(
                "multiple certificates claim the same dimension"
            )
        certificates_by_dimension[certificate.dimension] = certificate

    if not anchor.valid:
        if certificates:
            raise PreservationGateError(
                "cannot revise an unavailable Baseline with a certificate"
            )
        if candidate is None:
            raise PreservationGateError(
                "unavailable Baseline requires a matching verified candidate"
            )
        _validate_candidate_fallback(
            candidate=candidate,
            verification=candidate_verification,
            view=view,
        )
        assert candidate is not None
        return BaselinePreservationResult(
            record_id=anchor.record_id,
            symptom_label=candidate.symptom_label,
            root_cause_label=candidate.root_cause_label,
            symptom_action=BaselineDecisionAction.BASELINE_UNAVAILABLE,
            root_cause_action=BaselineDecisionAction.BASELINE_UNAVAILABLE,
            baseline_source_config_hash=anchor.source_config_hash,
            baseline_source_predictions_sha256=anchor.source_predictions_sha256,
            expected_baseline_anchor_hash=expected_anchor_hash,
            expected_taxonomy_structure_hash=expected_structure_hash,
        )

    assert anchor.symptom_label is not None and anchor.root_cause_label is not None
    labels = {
        EvidenceDimension.SYMPTOM: anchor.symptom_label,
        EvidenceDimension.ROOT_CAUSE: anchor.root_cause_label,
    }
    actions = {
        EvidenceDimension.SYMPTOM: BaselineDecisionAction.PRESERVED,
        EvidenceDimension.ROOT_CAUSE: BaselineDecisionAction.PRESERVED,
    }
    applied: list[LabelRevisionCertificate] = []
    for dimension, certificate in certificates_by_dimension.items():
        if candidate_bound_surface and certificate.proposal_digest is None:
            raise PreservationGateError(
                "candidate-bound gate rejects a legacy certificate without proposal digest"
            )
        proposal = (
            proposal_by_digest.get(certificate.proposal_digest)
            if certificate.proposal_digest is not None
            else None
        )
        if certificate.proposal_digest is not None and proposal is None:
            raise PreservationGateError(
                "candidate-bound certificate requires its exact revision proposal"
            )
        if _validate_certificate(
            certificate,
            anchor=anchor,
            candidate=candidate,
            proposal=proposal,
            structure=structure,
            view=view,
        ):
            labels[dimension] = certificate.proposed_label
            actions[dimension] = BaselineDecisionAction.REVISED
            applied.append(certificate)
    return BaselinePreservationResult(
        record_id=anchor.record_id,
        symptom_label=labels[EvidenceDimension.SYMPTOM],
        root_cause_label=labels[EvidenceDimension.ROOT_CAUSE],
        symptom_action=actions[EvidenceDimension.SYMPTOM],
        root_cause_action=actions[EvidenceDimension.ROOT_CAUSE],
        applied_certificates=tuple(applied),
        baseline_source_config_hash=anchor.source_config_hash,
        baseline_source_predictions_sha256=anchor.source_predictions_sha256,
        expected_baseline_anchor_hash=expected_anchor_hash,
        expected_taxonomy_structure_hash=expected_structure_hash,
    )
