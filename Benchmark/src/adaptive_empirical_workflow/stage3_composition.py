"""Deterministically compose one Stage 3 team candidate from frozen reviews."""

from __future__ import annotations

from collections.abc import Sequence
from Benchmark.src.annotation_contracts import label_valid, labels_equal

from .contracts import (
    BaselineAnchor,
    BaselineRevisionAssessment,
    baseline_revision_assessment_digest,
    canonical_evidence_view_hash,
    ConsistencyStatus,
    BoundaryChallenge,
    BoundaryChallengeAction,
    DimensionVerificationReport,
    ChainVerificationReport,
    EvidenceDimension,
    EvidenceView,
    JointAnchorReport,
    LabelRevisionCertificate,
    RevisionProposalEnvelope,
    RevisionCandidateSignal,
    RevisionCandidateSourceKind,
    revision_proposal_digest,
    RevisionBasis,
    RevisionAssessmentVerdict,
    RootCauseReport,
    SymptomReport,
    TeamComposedCandidate,
    TeamCorrectionAudit,
    Stage3TeamReport,
    stage3_team_report_digest,
    TaxonomyStructure,
    VerificationVerdict,
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
from .taxonomy_structure import taxonomy_structure_hash


REVISION_CANDIDATE_GRAPH_POLICY_VERSION = "evidence-bound-candidate-signals-v2"
REVISION_CROSS_CHECK_GRAPH_POLICY_VERSION = "dual-candidate-bound-cross-check-v1"


class CompositionError(ValueError):
    """Raised when a verifier result cannot safely alter a frozen anchor."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        dimension: EvidenceDimension | None = None,
        evidence_ids: Sequence[str] = (),
    ) -> None:
        super().__init__(message)
        self.code = code
        self.dimension = dimension
        self.evidence_ids = tuple(evidence_ids)


_PROPOSAL_DIMENSIONS = (
    EvidenceDimension.SYMPTOM,
    EvidenceDimension.ROOT_CAUSE,
)


def _revision_support_has_required_frozen_authority(
    view: EvidenceView,
    supporting_ids: Sequence[str],
) -> bool:
    """Require direct repair authority only when frozen evidence is enabled."""

    frozen_enabled = frozen_evidence_runtime_enabled(view)
    if not frozen_enabled:
        return True
    item_by_id = {item.evidence_id: item for item in view.items}
    supporting = tuple(
        item_by_id[evidence_id]
        for evidence_id in supporting_ids
        if evidence_id in item_by_id
    )
    return bool(
        supporting
        and all(frozen_evidence_may_be_support(item) for item in supporting)
        and any(authoritative_revision_support(item, view=view) for item in supporting)
    )


def _dimension_report(
    report: Stage3TeamReport, dimension: EvidenceDimension
) -> SymptomReport | RootCauseReport:
    return (
        report.symptom if dimension is EvidenceDimension.SYMPTOM else report.root_cause
    )


def stage3_correction_semantic_errors(
    report: Stage3TeamReport,
) -> tuple[str, ...]:
    """Replay the deterministic composer relation for one anchored team report."""

    if report.anchor is None:
        return (
            (
                f"team {report.team_id} correction semantics require a frozen joint anchor",
            )
            if report.verifications or report.correction_audit
            else ()
        )
    reviews = {review.dimension: review for review in report.verifications}
    audits = {audit.dimension: audit for audit in report.correction_audit}
    if set(reviews) != set(_PROPOSAL_DIMENSIONS) or set(audits) != set(
        _PROPOSAL_DIMENSIONS
    ):
        return (f"team {report.team_id} correction semantics are incomplete",)
    errors: list[str] = []
    for dimension in _PROPOSAL_DIMENSIONS:
        review = reviews[dimension]
        audit = audits[dimension]
        anchor_part = (
            report.anchor.symptom
            if dimension is EvidenceDimension.SYMPTOM
            else report.anchor.root_cause
        )
        composed_part = _dimension_report(report, dimension)
        if review.anchor_label != anchor_part.label:
            errors.append(
                f"team {report.team_id} {dimension.value} correction verifier "
                "anchor does not match the frozen joint anchor"
            )
            continue
        supporting_ids = tuple(review.supporting_evidence_ids)
        expected_part = (
            _correct_symptom(anchor_part, review, supporting_ids)
            if dimension is EvidenceDimension.SYMPTOM
            else _correct_root_cause(anchor_part, review, supporting_ids)
        )
        accepted = _applies_revision(review)
        expected_audit = _audit(
            review,
            final_label=expected_part.label,
            accepted=accepted,
            supporting_ids=supporting_ids,
        )
        if composed_part != expected_part:
            errors.append(
                f"team {report.team_id} {dimension.value} correction final does "
                "not match deterministic composition"
            )
        if audit != expected_audit:
            errors.append(
                f"team {report.team_id} {dimension.value} correction audit does "
                "not match deterministic composition"
            )
    return tuple(errors)


def _dimension_owned_report_evidence_ids(
    report: Stage3TeamReport,
    dimension: EvidenceDimension,
) -> tuple[str, ...]:
    cited: list[str] = []

    def add_part(part: SymptomReport | RootCauseReport) -> None:
        cited.extend(part.supporting_evidence_ids)
        cited.extend(part.counter_evidence_ids)
        cited.extend(part.boundary_evidence_ids)

    if report.anchor is not None:
        add_part(
            report.anchor.symptom
            if dimension is EvidenceDimension.SYMPTOM
            else report.anchor.root_cause
        )
        for review in report.verifications:
            if review.dimension is dimension:
                cited.extend(review.supporting_evidence_ids)
                cited.extend(review.counter_evidence_ids)
        for audit in report.correction_audit:
            if audit.dimension is dimension and audit.accepted:
                cited.extend(audit.supporting_evidence_ids)
    add_part(_dimension_report(report, dimension))
    return tuple(dict.fromkeys(cited))


def boundary_challenge_semantic_errors(
    challenge: BoundaryChallenge,
    reports: Sequence[Stage3TeamReport],
    view: EvidenceView,
) -> tuple[str, ...]:
    """Validate challenge citations against canonical team-owned evidence."""

    if len(reports) != 2 or {report.team_id for report in reports} != {"A", "B"}:
        return (
            "Boundary challenger requires exactly one canonical report "
            "from teams A and B.",
        )
    try:
        validity = assess_stage3_evidence_validity(view)
    except ValueError:
        return ("Boundary challenger received an invalid canonical evidence view.",)
    valid_ids = set(validity.valid_evidence_ids)
    evidence_by_id = {item.evidence_id: item for item in view.items}
    errors = [
        f"Boundary challenger cites unknown evidence_id {evidence_id!r}."
        for evidence_id in challenge.cited_evidence_ids
        if evidence_id not in evidence_by_id
    ]
    if errors:
        return tuple(errors)
    if challenge.action in (
        BoundaryChallengeAction.PASS,
        BoundaryChallengeAction.EVIDENCE_REQUEST,
    ):
        errors.extend(
            f"Boundary challenger cites noncanonical evidence_id {evidence_id!r}."
            for evidence_id in challenge.cited_evidence_ids
            if evidence_id not in valid_ids
        )
        return tuple(errors)
    dimension = (
        EvidenceDimension.SYMPTOM
        if challenge.action is BoundaryChallengeAction.SYMPTOM_REVIEW
        else EvidenceDimension.ROOT_CAUSE
    )
    relevant_ids = {
        evidence_id
        for report in reports
        for evidence_id in _dimension_owned_report_evidence_ids(report, dimension)
    }
    owner = "symptom" if dimension is EvidenceDimension.SYMPTOM else "cause"
    for evidence_id in challenge.cited_evidence_ids:
        item = evidence_by_id[evidence_id]
        if (
            evidence_id not in valid_ids
            or evidence_id not in relevant_ids
            or not supports_readiness_dimension(
                item,
                dimension,
                domain=view.domain_profile,
            )
        ):
            errors.append(
                f"Boundary challenger cites {owner}-irrelevant "
                f"evidence_id {evidence_id!r}."
            )
    return tuple(errors)


def _candidate_citations(
    report: Stage3TeamReport, dimension: EvidenceDimension
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    dimension_report = _dimension_report(report, dimension)
    supporting = tuple(
        sorted(
            {
                *dimension_report.supporting_evidence_ids,
                *dimension_report.boundary_evidence_ids,
            }
        )
    )
    return supporting, tuple(sorted(dimension_report.counter_evidence_ids))


def _reports_are_canonical_for_view(
    reports: Sequence[Stage3TeamReport],
    *,
    structure: TaxonomyStructure,
    view: EvidenceView,
    valid_by_id: dict[str, object],
) -> bool:
    if len(reports) != 2 or {report.team_id for report in reports} != {"A", "B"}:
        return False
    structure_labels = {
        dimension: {
            node.label for node in structure.nodes if node.dimension == dimension.value
        }
        for dimension in _PROPOSAL_DIMENSIONS
    }
    for dimension in _PROPOSAL_DIMENSIONS:
        if structure_labels[dimension] != set(view.taxonomy.get(dimension.value, ())):
            return False
    for report in reports:
        for dimension in _PROPOSAL_DIMENSIONS:
            dimension_report = _dimension_report(report, dimension)
            if dimension_report.label not in structure_labels[dimension]:
                return False
            supporting, counter = _candidate_citations(report, dimension)
            if (
                not supporting
                or any(
                    evidence_id in valid_by_id
                    and not frozen_evidence_may_be_support(
                        valid_by_id[evidence_id]  # type: ignore[arg-type]
                    )
                    for evidence_id in supporting
                )
                or any(
                    evidence_id not in valid_by_id
                    or not supports_readiness_dimension(
                        valid_by_id[evidence_id],  # type: ignore[arg-type]
                        dimension,
                        domain=view.domain_profile,
                    )
                    for evidence_id in (*supporting, *counter)
                )
            ):
                return False
        if any(
            evidence_id not in valid_by_id
            or not frozen_evidence_may_be_support(
                valid_by_id[evidence_id]  # type: ignore[arg-type]
            )
            for evidence_id in report.consistency.supporting_evidence_ids
        ):
            return False
    return True


def build_revision_proposals(
    *,
    anchor: BaselineAnchor,
    reports: Sequence[Stage3TeamReport],
    structure: TaxonomyStructure,
    view: EvidenceView,
    boundary_challenge: BoundaryChallenge | None = None,
) -> tuple[RevisionProposalEnvelope, ...]:
    """Build at most two evidence-bound candidate signals per Stage 3 dimension."""

    try:
        reports = tuple(
            Stage3TeamReport.model_validate(report.model_dump(mode="python"))
            for report in reports
        )
        if boundary_challenge is not None:
            boundary_challenge = BoundaryChallenge.model_validate(
                boundary_challenge.model_dump(mode="python")
            )
    except (AttributeError, ValueError):
        return ()
    if any(stage3_correction_semantic_errors(report) for report in reports):
        return ()
    if (
        not anchor.valid
        or anchor.record_id != view.record_id
        or structure.domain != view.domain_profile
    ):
        return ()
    try:
        valid_by_id = _valid_evidence_by_id(view)
    except CompositionError:
        return ()
    if len({item.evidence_id for item in view.items}) != len(
        view.items
    ) or not _reports_are_canonical_for_view(
        reports, structure=structure, view=view, valid_by_id=valid_by_id
    ):
        return ()
    if boundary_challenge is not None and boundary_challenge_semantic_errors(
        boundary_challenge,
        reports,
        view,
    ):
        return ()

    report_by_team = {report.team_id: report for report in reports}
    current_structure_hash = taxonomy_structure_hash(structure)
    current_view_hash = canonical_evidence_view_hash(view)
    proposals: list[RevisionProposalEnvelope] = []
    for dimension in _PROPOSAL_DIMENSIONS:
        baseline_label = _baseline_label(anchor, dimension)
        allowed = set(view.taxonomy.get(dimension.value, ()))
        if baseline_label not in allowed:
            return ()
        signal_parts: dict[
            str,
            dict[str, set[str] | set[RevisionCandidateSourceKind]],
        ] = {}

        def add_signal(
            *,
            report: Stage3TeamReport,
            label: str,
            source_kind: RevisionCandidateSourceKind,
            supporting_ids: Sequence[str],
            counter_ids: Sequence[str],
            require_authoritative_repair: bool,
        ) -> None:
            if label == baseline_label or label not in allowed:
                return
            support = tuple(sorted(set(supporting_ids)))
            counter = tuple(sorted(set(counter_ids)))
            if (
                not support
                or any(
                    evidence_id not in valid_by_id
                    or not frozen_evidence_may_be_support(
                        valid_by_id[evidence_id]  # type: ignore[arg-type]
                    )
                    or not supports_readiness_dimension(
                        valid_by_id[evidence_id],  # type: ignore[arg-type]
                        dimension,
                        domain=view.domain_profile,
                    )
                    for evidence_id in support
                )
                or any(
                    evidence_id not in valid_by_id
                    or not supports_readiness_dimension(
                        valid_by_id[evidence_id],  # type: ignore[arg-type]
                        dimension,
                        domain=view.domain_profile,
                    )
                    for evidence_id in counter
                )
            ):
                return
            authoritative = tuple(
                evidence_id
                for evidence_id in support
                if authoritative_revision_support(
                    valid_by_id[evidence_id],  # type: ignore[arg-type]
                    view=view,
                )
            )
            if require_authoritative_repair and not authoritative:
                return
            part = signal_parts.setdefault(
                label,
                {
                    "source_kinds": set(),
                    "team_ids": set(),
                    "report_digests": set(),
                    "supporting_evidence_ids": set(),
                    "counter_evidence_ids": set(),
                    "authoritative_repair_evidence_ids": set(),
                },
            )
            part["source_kinds"].add(source_kind)  # type: ignore[union-attr]
            part["team_ids"].add(report.team_id)  # type: ignore[union-attr]
            part["report_digests"].add(  # type: ignore[union-attr]
                stage3_team_report_digest(report)
            )
            part["supporting_evidence_ids"].update(support)  # type: ignore[union-attr]
            part["counter_evidence_ids"].update(counter)  # type: ignore[union-attr]
            part["authoritative_repair_evidence_ids"].update(  # type: ignore[union-attr]
                authoritative
            )

        challenge_ids: tuple[str, ...] = ()
        if boundary_challenge is not None:
            action_for_dimension = {
                EvidenceDimension.SYMPTOM: BoundaryChallengeAction.SYMPTOM_REVIEW,
                EvidenceDimension.ROOT_CAUSE: BoundaryChallengeAction.CAUSE_REVIEW,
            }[dimension]
            if boundary_challenge.action is action_for_dimension:
                challenge_ids = tuple(boundary_challenge.cited_evidence_ids)

        for report in reports:
            dimension_report = _dimension_report(report, dimension)
            supporting, counter = _candidate_citations(report, dimension)
            if dimension_report.label != baseline_label:
                add_signal(
                    report=report,
                    label=dimension_report.label,
                    source_kind=RevisionCandidateSourceKind.TEAM_FINAL,
                    supporting_ids=supporting,
                    counter_ids=counter,
                    require_authoritative_repair=False,
                )

            verification = next(
                (
                    review
                    for review in report.verifications
                    if review.dimension is dimension
                    and review.verdict is VerificationVerdict.REJECT
                ),
                None,
            )
            correction = next(
                (
                    audit
                    for audit in report.correction_audit
                    if audit.dimension is dimension and audit.accepted
                ),
                None,
            )
            if (
                verification is not None
                and correction is not None
                and verification.alternative_label == correction.proposed_label
                and correction.final_label == verification.alternative_label
            ):
                add_signal(
                    report=report,
                    label=verification.alternative_label,
                    source_kind=RevisionCandidateSourceKind.VERIFIER_CORRECTION,
                    supporting_ids=correction.supporting_evidence_ids,
                    counter_ids=verification.counter_evidence_ids,
                    require_authoritative_repair=False,
                )

            alternative_support = (
                *dimension_report.boundary_evidence_ids,
                *challenge_ids,
            )
            add_signal(
                report=report,
                label=dimension_report.alternative_label,
                source_kind=RevisionCandidateSourceKind.REPAIR_BACKED_ALTERNATIVE,
                supporting_ids=alternative_support,
                counter_ids=dimension_report.counter_evidence_ids,
                require_authoritative_repair=True,
            )

        exact_card_by_label = {
            label: cards[0]
            for label in signal_parts
            for cards in (
                tuple(
                    card
                    for card in structure.boundary_cards
                    if card.dimension == dimension.value
                    and set(card.labels) == {baseline_label, label}
                ),
            )
            if len(cards) == 1
        }
        ranked_labels = sorted(
            exact_card_by_label,
            key=lambda label: (
                -int(
                    RevisionCandidateSourceKind.REPAIR_BACKED_ALTERNATIVE
                    in signal_parts[label]["source_kinds"]
                ),
                -int(
                    bool(
                        {
                            RevisionCandidateSourceKind.VERIFIER_CORRECTION,
                            RevisionCandidateSourceKind.TEAM_FINAL,
                        }
                        & signal_parts[label]["source_kinds"]  # type: ignore[operator]
                    )
                ),
                -len(signal_parts[label]["team_ids"]),
                tuple(
                    sorted(
                        kind.value
                        for kind in signal_parts[label]["source_kinds"]  # type: ignore[union-attr]
                    )
                ),
                label,
            ),
        )
        for proposed_label in ranked_labels[:2]:
            card = exact_card_by_label[proposed_label]
            part = signal_parts[proposed_label]
            proposer_team_ids = tuple(
                team_id for team_id in ("A", "B") if team_id in part["team_ids"]
            )
            proposer_reports = tuple(
                report_by_team[team_id] for team_id in proposer_team_ids
            )
            supporting = tuple(sorted(part["supporting_evidence_ids"]))
            counter = tuple(sorted(part["counter_evidence_ids"]))
            signal = RevisionCandidateSignal(
                dimension=dimension,
                label=proposed_label,
                source_kinds=tuple(
                    sorted(part["source_kinds"], key=lambda kind: kind.value)  # type: ignore[arg-type,union-attr]
                ),
                team_ids=proposer_team_ids,
                report_digests=tuple(
                    stage3_team_report_digest(report) for report in proposer_reports
                ),
                supporting_evidence_ids=supporting,
                counter_evidence_ids=counter,
                authoritative_repair_evidence_ids=tuple(
                    sorted(part["authoritative_repair_evidence_ids"])
                ),
            )
            proposals.append(
                RevisionProposalEnvelope(
                    dimension=dimension,
                    baseline_label=baseline_label,
                    proposed_label=proposed_label,
                    boundary_card_id=card.card_id,
                    proposer_team_ids=proposer_team_ids,
                    proposer_report_digests=tuple(
                        stage3_team_report_digest(report) for report in proposer_reports
                    ),
                    supporting_evidence_ids=supporting,
                    counter_evidence_ids=counter,
                    candidate_signals=(signal,),
                    baseline_source_config_hash=anchor.source_config_hash,
                    baseline_source_predictions_sha256=(
                        anchor.source_predictions_sha256
                    ),
                    taxonomy_structure_hash=current_structure_hash,
                    evidence_view_hash=current_view_hash,
                )
            )
    return tuple(proposals)


def revision_proposal_matches_context(
    proposal: RevisionProposalEnvelope,
    *,
    anchor: BaselineAnchor,
    reports: Sequence[Stage3TeamReport],
    structure: TaxonomyStructure,
    view: EvidenceView,
    boundary_challenge: BoundaryChallenge | None = None,
) -> bool:
    """Rebuild the proposal trust root and require an exact canonical match."""

    return proposal in build_revision_proposals(
        anchor=anchor,
        reports=reports,
        structure=structure,
        view=view,
        boundary_challenge=boundary_challenge,
    )


def _valid_evidence_by_id(view: EvidenceView) -> dict[str, object]:
    try:
        validity = assess_stage3_evidence_validity(view)
    except ValueError as error:
        raise CompositionError(
            "invalid correction evidence view",
            code="invalid_correction_evidence_view",
        ) from error
    items_by_id = {item.evidence_id: item for item in view.items}
    return {
        evidence_id: items_by_id[evidence_id]
        for evidence_id in validity.valid_evidence_ids
    }


def _validated_correction_ids(
    review: DimensionVerificationReport,
    view: EvidenceView,
    valid_by_id: dict[str, object],
) -> tuple[str, ...]:
    cited = tuple(dict.fromkeys(review.supporting_evidence_ids))
    missing = [evidence_id for evidence_id in cited if evidence_id not in valid_by_id]
    if missing:
        raise CompositionError(
            f"unknown correction evidence IDs: {missing!r}",
            code="unknown_supporting_evidence",
            dimension=review.dimension,
            evidence_ids=missing,
        )
    incapable = [
        evidence_id
        for evidence_id in cited
        if not supports_readiness_dimension(
            valid_by_id[evidence_id],  # type: ignore[arg-type]
            review.dimension,
            domain=view.domain_profile,
        )
    ]
    if incapable:
        raise CompositionError(
            f"incapable correction evidence IDs: {incapable!r}",
            code="supporting_evidence_incapable_for_dimension",
            dimension=review.dimension,
            evidence_ids=incapable,
        )
    return cited


def _validated_counter_ids(
    review: DimensionVerificationReport,
    valid_by_id: dict[str, object],
) -> tuple[str, ...]:
    cited = tuple(dict.fromkeys(review.counter_evidence_ids))
    missing = [evidence_id for evidence_id in cited if evidence_id not in valid_by_id]
    if missing:
        raise CompositionError(
            f"unknown correction counter evidence IDs: {missing!r}",
            code="unknown_counter_evidence",
            dimension=review.dimension,
            evidence_ids=missing,
        )
    return cited


def _validate_review(
    review: DimensionVerificationReport,
    *,
    dimension: EvidenceDimension,
    anchor_label: str,
    view: EvidenceView,
    valid_by_id: dict[str, object],
) -> tuple[str, ...]:
    if review.dimension is not dimension:
        raise CompositionError(
            f"verification dimension {review.dimension.value!r} does not match "
            f"{dimension.value!r}",
            code="verification_dimension_mismatch",
            dimension=dimension,
        )
    if (not labels_equal(view.taxonomy, dimension.value, review.anchor_label, anchor_label)
        if view.taxonomy.get("annotation_modes") else review.anchor_label != anchor_label):
        raise CompositionError(
            f"verification anchor label {review.anchor_label!r} does not match "
            f"{anchor_label!r}",
            code="verification_anchor_label_mismatch",
            dimension=dimension,
        )
    supporting_ids = _validated_correction_ids(review, view, valid_by_id)
    _validated_counter_ids(review, valid_by_id)
    if review.verdict is VerificationVerdict.REJECT:
        allowed = view.taxonomy.get(dimension.value, ())
        if (
            review.alternative_label is None
            or labels_equal(view.taxonomy, dimension.value, review.alternative_label, anchor_label)
            or not label_valid(view.taxonomy, dimension.value, review.alternative_label)
        ):
            raise CompositionError(
                f"invalid {dimension.value} correction alternative label",
                code="invalid_correction_alternative_label",
                dimension=dimension,
            )
        if not supporting_ids:
            raise CompositionError(
                "rejected verification requires supporting evidence",
                code="rejected_verification_requires_supporting_evidence",
                dimension=dimension,
            )
        if (
            dimension is EvidenceDimension.ROOT_CAUSE
            and len(review.corrected_causal_chain) < 3
        ):
            raise CompositionError(
                "root-cause correction requires a three-step causal chain",
                code="root_cause_correction_requires_causal_chain",
                dimension=dimension,
            )
    return supporting_ids


def _audit(
    review: DimensionVerificationReport,
    *,
    final_label: str,
    accepted: bool,
    supporting_ids: tuple[str, ...],
) -> TeamCorrectionAudit:
    return TeamCorrectionAudit(
        dimension=review.dimension,
        anchor_label=review.anchor_label,
        proposed_label=review.alternative_label,
        final_label=final_label,
        accepted=accepted,
        reason=review.rationale,
        supporting_evidence_ids=supporting_ids if accepted else (),
    )


def _applies_revision(review: DimensionVerificationReport) -> bool:
    return review.verdict is VerificationVerdict.REJECT or (
        isinstance(review, ChainVerificationReport) and review.chain_check.action == "rewrite"
    )


def _chain_revision(anchor, review: ChainVerificationReport):
    revision = review.chain_check.revision
    if revision is None:
        return anchor
    ids = tuple(dict.fromkeys(c.evidence_id for c in revision.citations))
    update = dict(label=revision.label, supporting_evidence_ids=ids,
                  alternative_label=revision.alternative_label, boundary_reason=revision.boundary_reason,
                  boundary_evidence_ids=ids, unresolved_evidence_gaps=revision.unresolved_evidence_gaps,
                  confidence=min(anchor.confidence, review.confidence))
    if isinstance(anchor, SymptomReport):
        update["behavior_claim"] = revision.claim
    else:
        update.update(defect_mechanism=revision.claim, causal_chain=revision.causal_chain)
    return type(anchor).model_validate({**anchor.model_dump(), **update})


def _correct_symptom(
    anchor: SymptomReport,
    review: DimensionVerificationReport,
    supporting_ids: tuple[str, ...],
) -> SymptomReport:
    if isinstance(review, ChainVerificationReport):
        return _chain_revision(anchor, review)
    if review.verdict is not VerificationVerdict.REJECT:
        return anchor
    return anchor.model_copy(
        update={
            "label": review.alternative_label,
            "behavior_claim": review.corrected_claim,
            "supporting_evidence_ids": supporting_ids,
            "counter_evidence_ids": tuple(review.counter_evidence_ids),
            "alternative_label": anchor.label,
            "boundary_reason": review.rationale,
            "confidence": min(anchor.confidence, review.confidence),
        }
    )


def _correct_root_cause(
    anchor: RootCauseReport,
    review: DimensionVerificationReport,
    supporting_ids: tuple[str, ...],
) -> RootCauseReport:
    if isinstance(review, ChainVerificationReport):
        return _chain_revision(anchor, review)
    if review.verdict is not VerificationVerdict.REJECT:
        return anchor
    return anchor.model_copy(
        update={
            "label": review.alternative_label,
            "defect_mechanism": review.corrected_claim,
            "causal_chain": tuple(review.corrected_causal_chain),
            "supporting_evidence_ids": supporting_ids,
            "counter_evidence_ids": tuple(review.counter_evidence_ids),
            "alternative_label": anchor.label,
            "boundary_reason": review.rationale,
            "confidence": min(anchor.confidence, review.confidence),
        }
    )


def compose_stage3_team_candidate(
    team_id: str,
    anchor: JointAnchorReport,
    symptom_verification: DimensionVerificationReport,
    root_verification: DimensionVerificationReport,
    view: EvidenceView,
) -> TeamComposedCandidate:
    """Apply only valid, dimension-owned verifier corrections to an anchor."""

    valid_by_id = _valid_evidence_by_id(view)
    from .evidence_chain import validate_chain_audit
    for review in (symptom_verification, root_verification):
        if isinstance(review, ChainVerificationReport):
            validate_chain_audit(review.chain_check, anchor, review.dimension.value, view)
    symptom_ids = _validate_review(
        symptom_verification,
        dimension=EvidenceDimension.SYMPTOM,
        anchor_label=anchor.symptom.label,
        view=view,
        valid_by_id=valid_by_id,
    )
    root_ids = _validate_review(
        root_verification,
        dimension=EvidenceDimension.ROOT_CAUSE,
        anchor_label=anchor.root_cause.label,
        view=view,
        valid_by_id=valid_by_id,
    )
    symptom = _correct_symptom(anchor.symptom, symptom_verification, symptom_ids)
    root_cause = _correct_root_cause(anchor.root_cause, root_verification, root_ids)
    symptom_accepted = _applies_revision(symptom_verification)
    root_accepted = _applies_revision(root_verification)
    return TeamComposedCandidate(
        team_id=team_id,
        symptom=symptom,
        root_cause=root_cause,
        anchor=anchor,
        verifications=(symptom_verification, root_verification),
        correction_audit=(
            _audit(
                symptom_verification,
                final_label=symptom.label,
                accepted=symptom_accepted,
                supporting_ids=symptom_ids,
            ),
            _audit(
                root_verification,
                final_label=root_cause.label,
                accepted=root_accepted,
                supporting_ids=root_ids,
            ),
        ),
    )


def _baseline_label(anchor: BaselineAnchor, dimension: EvidenceDimension) -> str:
    label = (
        anchor.symptom_label
        if dimension is EvidenceDimension.SYMPTOM
        else anchor.root_cause_label
    )
    assert label is not None
    return label


def _assessment_is_valid(
    assessment: BaselineRevisionAssessment,
    *,
    anchor: BaselineAnchor,
    structure: TaxonomyStructure,
    view: EvidenceView,
    valid_by_id: dict[str, object],
) -> bool:
    dimension = assessment.dimension
    if assessment.verdict is not RevisionAssessmentVerdict.REVISE:
        return False
    baseline_label = _baseline_label(anchor, dimension)
    if assessment.baseline_label != baseline_label:
        return False
    if assessment.proposed_label not in view.taxonomy.get(dimension.value, ()):
        return False
    card = next(
        (
            candidate
            for candidate in structure.boundary_cards
            if candidate.card_id == assessment.boundary_card_id
        ),
        None,
    )
    if (
        card is None
        or card.dimension != dimension.value
        or set(card.labels) != {baseline_label, assessment.proposed_label}
    ):
        return False
    criteria = {criterion.label: criterion for criterion in card.criteria}
    if assessment.contradicted_baseline_condition not in criteria[
        baseline_label
    ].exclusion_conditions or not set(
        assessment.satisfied_proposed_conditions
    ).issubset(
        criteria[assessment.proposed_label].positive_conditions
    ):
        return False
    cited_ids = (
        *assessment.supporting_evidence_ids,
        *assessment.counter_evidence_ids,
    )
    if any(evidence_id not in valid_by_id for evidence_id in cited_ids):
        return False
    return all(
        supports_readiness_dimension(
            valid_by_id[evidence_id],  # type: ignore[arg-type]
            dimension,
            domain=view.domain_profile,
        )
        for evidence_id in cited_ids
    )


def _assessment_for_dimension(
    report: Stage3TeamReport,
    dimension: EvidenceDimension,
) -> BaselineRevisionAssessment | None:
    assessments = tuple(
        assessment
        for assessment in report.baseline_revision_assessments
        if assessment.dimension is dimension
    )
    return assessments[0] if len(assessments) == 1 else None


def compose_baseline_revision_certificates(
    *,
    anchor: BaselineAnchor,
    reports: Sequence[Stage3TeamReport],
    proposals: Sequence[RevisionProposalEnvelope] | None = None,
    structure: TaxonomyStructure,
    view: EvidenceView,
    boundary_challenge: BoundaryChallenge | None = None,
) -> tuple[LabelRevisionCertificate, ...]:
    """Compose only independently complete, evidence-valid Baseline revisions."""

    if not anchor.valid:
        return ()
    if anchor.record_id != view.record_id or structure.domain != view.domain_profile:
        return ()
    if any(
        _baseline_label(anchor, dimension) not in view.taxonomy.get(dimension.value, ())
        for dimension in (EvidenceDimension.SYMPTOM, EvidenceDimension.ROOT_CAUSE)
    ):
        return ()
    current_hashes = (
        anchor.source_config_hash,
        anchor.source_predictions_sha256,
        taxonomy_structure_hash(structure),
        canonical_evidence_view_hash(view),
    )
    if len(reports) != 2 or {report.team_id for report in reports} != {"A", "B"}:
        return ()
    if any(
        report.consistency.status is not ConsistencyStatus.CONSISTENT
        for report in reports
    ):
        return ()
    try:
        valid_by_id = _valid_evidence_by_id(view)
    except CompositionError:
        return ()
    if len({item.evidence_id for item in view.items}) != len(view.items):
        return ()

    if proposals is not None:
        return _compose_candidate_bound_certificates(
            anchor=anchor,
            reports=reports,
            proposals=proposals,
            structure=structure,
            view=view,
            boundary_challenge=boundary_challenge,
            valid_by_id=valid_by_id,
            current_hashes=current_hashes,
        )

    certificates: list[LabelRevisionCertificate] = []
    for dimension in (EvidenceDimension.SYMPTOM, EvidenceDimension.ROOT_CAUSE):
        assessments = tuple(
            _assessment_for_dimension(report, dimension) for report in reports
        )
        if any(assessment is None for assessment in assessments):
            continue
        team_a, team_b = assessments
        assert team_a is not None and team_b is not None
        if any(
            (
                assessment.baseline_source_config_hash,
                assessment.baseline_source_predictions_sha256,
                assessment.taxonomy_structure_hash,
                assessment.evidence_view_hash,
            )
            != current_hashes
            for assessment in (team_a, team_b)
        ):
            continue
        if not (
            _assessment_is_valid(
                team_a,
                anchor=anchor,
                structure=structure,
                view=view,
                valid_by_id=valid_by_id,
            )
            and _assessment_is_valid(
                team_b,
                anchor=anchor,
                structure=structure,
                view=view,
                valid_by_id=valid_by_id,
            )
        ):
            continue
        if (
            team_a.baseline_label,
            team_a.proposed_label,
            team_a.boundary_card_id,
            team_a.contradicted_baseline_condition,
            team_a.satisfied_proposed_conditions,
        ) != (
            team_b.baseline_label,
            team_b.proposed_label,
            team_b.boundary_card_id,
            team_b.contradicted_baseline_condition,
            team_b.satisfied_proposed_conditions,
        ):
            continue
        if any(
            (
                (
                    report.symptom.label
                    if dimension is EvidenceDimension.SYMPTOM
                    else report.root_cause.label
                )
                != assessment.proposed_label
                or not any(
                    record.dimension is dimension
                    and record.baseline_label == assessment.baseline_label
                    and record.proposed_label == assessment.proposed_label
                    and record.status is ConsistencyStatus.CONSISTENT
                    and record.assessment_digest
                    == baseline_revision_assessment_digest(assessment)
                    and record.boundary_card_id == assessment.boundary_card_id
                    and (
                        record.baseline_source_config_hash,
                        record.baseline_source_predictions_sha256,
                        record.taxonomy_structure_hash,
                        record.evidence_view_hash,
                    )
                    == current_hashes
                    and all(
                        evidence_id in valid_by_id
                        and supports_readiness_dimension(
                            valid_by_id[evidence_id],  # type: ignore[arg-type]
                            dimension,
                            domain=view.domain_profile,
                        )
                        for evidence_id in record.supporting_evidence_ids
                    )
                    for record in report.revision_consistency
                )
            )
            for report, assessment in zip(reports, (team_a, team_b), strict=True)
        ):
            continue
        combined_supporting_ids = tuple(
            dict.fromkeys(
                (
                    *team_a.supporting_evidence_ids,
                    *team_b.supporting_evidence_ids,
                )
            )
        )
        if not _revision_support_has_required_frozen_authority(
            view, combined_supporting_ids
        ):
            continue
        certificates.append(
            LabelRevisionCertificate(
                dimension=dimension,
                baseline_label=team_a.baseline_label,
                proposed_label=team_a.proposed_label,
                revision_basis=RevisionBasis.TAXONOMY_BOUNDARY,
                boundary_card_id=team_a.boundary_card_id,
                contradicted_baseline_condition=team_a.contradicted_baseline_condition,
                satisfied_proposed_conditions=team_a.satisfied_proposed_conditions,
                supporting_evidence_ids=combined_supporting_ids,
                counter_evidence_ids=tuple(
                    dict.fromkeys(
                        (*team_a.counter_evidence_ids, *team_b.counter_evidence_ids)
                    )
                ),
                supporting_team_ids=("A", "B"),
            )
        )
    return tuple(certificates)


def _compose_candidate_bound_certificates(
    *,
    anchor: BaselineAnchor,
    reports: Sequence[Stage3TeamReport],
    proposals: Sequence[RevisionProposalEnvelope],
    structure: TaxonomyStructure,
    view: EvidenceView,
    boundary_challenge: BoundaryChallenge | None,
    valid_by_id: dict[str, object],
    current_hashes: tuple[str, str, str, str],
) -> tuple[LabelRevisionCertificate, ...]:
    """Sign exact proposal-owned revisions after two opposite cross-checks."""

    canonical_proposals = tuple(proposals)
    if len(canonical_proposals) != len(set(canonical_proposals)):
        return ()
    if any(
        not revision_proposal_matches_context(
            proposal,
            anchor=anchor,
            reports=reports,
            structure=structure,
            view=view,
            boundary_challenge=boundary_challenge,
        )
        for proposal in canonical_proposals
    ):
        return ()
    report_by_team = {report.team_id: report for report in reports}
    eligible: list[LabelRevisionCertificate] = []
    for proposal in canonical_proposals:
        proposal_digest = revision_proposal_digest(proposal)
        assessments: dict[str, BaselineRevisionAssessment] = {}
        valid = True
        for team_id in ("A", "B"):
            matches = tuple(
                assessment
                for assessment in report_by_team[team_id].baseline_revision_assessments
                if assessment.proposal_digest == proposal_digest
            )
            if len(matches) != 1:
                valid = False
                break
            assessment = matches[0]
            if (
                assessment.assessor_team_id != team_id
                or assessment.dimension is not proposal.dimension
                or assessment.baseline_label != proposal.baseline_label
                or assessment.proposed_label != proposal.proposed_label
                or assessment.boundary_card_id != proposal.boundary_card_id
                or (
                    assessment.baseline_source_config_hash,
                    assessment.baseline_source_predictions_sha256,
                    assessment.taxonomy_structure_hash,
                    assessment.evidence_view_hash,
                )
                != current_hashes
                or not _assessment_is_valid(
                    assessment,
                    anchor=anchor,
                    structure=structure,
                    view=view,
                    valid_by_id=valid_by_id,
                )
                or not {
                    *assessment.supporting_evidence_ids,
                    *assessment.counter_evidence_ids,
                }.issubset(
                    {
                        *proposal.supporting_evidence_ids,
                        *proposal.counter_evidence_ids,
                    }
                )
            ):
                valid = False
                break
            assessments[team_id] = assessment
        if not valid:
            continue
        assessment_a = assessments["A"]
        assessment_b = assessments["B"]
        if (
            assessment_a.contradicted_baseline_condition,
            assessment_a.satisfied_proposed_conditions,
        ) != (
            assessment_b.contradicted_baseline_condition,
            assessment_b.satisfied_proposed_conditions,
        ):
            continue
        for checker_team_id, owner_team_id in (("A", "B"), ("B", "A")):
            owned = assessments[owner_team_id]
            matches = tuple(
                record
                for record in report_by_team[checker_team_id].revision_consistency
                if record.proposal_digest == proposal_digest
            )
            if len(matches) != 1:
                valid = False
                break
            record = matches[0]
            if (
                record.checker_team_id,
                record.assessment_owner_team_id,
                record.dimension,
                record.baseline_label,
                record.proposed_label,
                record.status,
                record.assessment_digest,
                record.boundary_card_id,
                record.supporting_evidence_ids,
                record.counter_evidence_ids,
                (
                    record.baseline_source_config_hash,
                    record.baseline_source_predictions_sha256,
                    record.taxonomy_structure_hash,
                    record.evidence_view_hash,
                ),
            ) != (
                checker_team_id,
                owner_team_id,
                proposal.dimension,
                proposal.baseline_label,
                proposal.proposed_label,
                ConsistencyStatus.CONSISTENT,
                baseline_revision_assessment_digest(owned),
                proposal.boundary_card_id,
                owned.supporting_evidence_ids,
                owned.counter_evidence_ids,
                current_hashes,
            ):
                valid = False
                break
        if not valid:
            continue
        combined_supporting_ids = tuple(
            dict.fromkeys(
                (
                    *assessment_a.supporting_evidence_ids,
                    *assessment_b.supporting_evidence_ids,
                )
            )
        )
        if not _revision_support_has_required_frozen_authority(
            view, combined_supporting_ids
        ):
            continue
        eligible.append(
            LabelRevisionCertificate(
                dimension=proposal.dimension,
                proposal_digest=proposal_digest,
                baseline_label=proposal.baseline_label,
                proposed_label=proposal.proposed_label,
                revision_basis=RevisionBasis.TAXONOMY_BOUNDARY,
                boundary_card_id=proposal.boundary_card_id,
                contradicted_baseline_condition=(
                    assessment_a.contradicted_baseline_condition
                ),
                satisfied_proposed_conditions=(
                    assessment_a.satisfied_proposed_conditions
                ),
                supporting_evidence_ids=combined_supporting_ids,
                counter_evidence_ids=tuple(
                    dict.fromkeys(
                        (
                            *assessment_a.counter_evidence_ids,
                            *assessment_b.counter_evidence_ids,
                        )
                    )
                ),
                supporting_team_ids=("A", "B"),
            )
        )
    ambiguous = {
        dimension
        for dimension in _PROPOSAL_DIMENSIONS
        if sum(certificate.dimension is dimension for certificate in eligible) > 1
    }
    return tuple(
        certificate
        for certificate in eligible
        if certificate.dimension not in ambiguous
    )
