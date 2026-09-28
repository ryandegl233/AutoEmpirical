from __future__ import annotations

import hashlib

import pytest

from Benchmark.src.adaptive_empirical_workflow.contracts import (
    BaselineAnchor,
    BaselineRevisionAssessment,
    baseline_revision_assessment_digest,
    canonical_evidence_view_hash,
    BoundaryCard,
    BoundaryCriterion,
    CausalConsistencyReport,
    ConsistencyStatus,
    DimensionVerificationReport,
    EvidenceDimension,
    EvidenceExplicitness,
    EvidenceItem,
    EvidenceSufficiency,
    EvidenceView,
    JointAnchorReport,
    RootCauseReport,
    RevisionAssessmentVerdict,
    RevisionConsistencyReport,
    SymptomReport,
    Stage3TeamReport,
    TaxonomyNode,
    TaxonomySemanticOrigin,
    TaxonomyStructure,
    VerificationVerdict,
)
from Benchmark.src.adaptive_empirical_workflow.stage3_composition import (
    CompositionError,
    compose_baseline_revision_certificates,
    compose_stage3_team_candidate,
)
from Benchmark.src.adaptive_empirical_workflow.taxonomy_structure import (
    taxonomy_structure_hash,
)


def _item(
    evidence_id: str, source_type: str, *, record_id: str = "record-1"
) -> EvidenceItem:
    content = f"Evidence {evidence_id} documents the classification."
    return EvidenceItem(
        evidence_id=evidence_id,
        record_id=record_id,
        source_type=source_type,
        source_uri=f"https://example.test/{evidence_id}",
        retrieved_at="2026-08-11T00:00:00Z",
        content=content,
        content_sha256=hashlib.sha256(content.encode()).hexdigest(),
        explicitness=EvidenceExplicitness.DIRECT,
    )


def _view(domain: str = "issta2024") -> EvidenceView:
    return EvidenceView(
        record_id="record-1",
        task="classify the defect",
        taxonomy={
            "symptom": ["Unexpected Output", "Crash"],
            "root_cause": ["API Misuse", "Incorrect Code Logic"],
        },
        domain_profile=domain,
        ledger_version=1,
        items=(
            _item("symptom-1", "issue_body"),
            _item("root-1", "code_context"),
            _item("counter-1", "issue_comment"),
            _item("quarantined-1", "code_context", record_id="other-record"),
        ),
    )


def _anchor() -> JointAnchorReport:
    return JointAnchorReport(
        symptom=SymptomReport(
            label="Unexpected Output",
            behavior_claim="The caller receives a result different from the request.",
            supporting_evidence_ids=["symptom-1"],
            alternative_label="Crash",
            boundary_reason="The observed behavior is a result, not process termination.",
            confidence=0.8,
            evidence_sufficiency=EvidenceSufficiency.SUFFICIENT,
        ),
        root_cause=RootCauseReport(
            label="API Misuse",
            defect_mechanism="The caller violates the API ordering precondition.",
            causal_chain=[
                "The caller creates a request.",
                "The API is called too early.",
                "The result differs.",
            ],
            supporting_evidence_ids=["root-1"],
            alternative_label="Incorrect Code Logic",
            boundary_reason="The anchor assigns the defect to caller ordering, not internal control flow.",
            confidence=0.7,
            evidence_sufficiency=EvidenceSufficiency.SUFFICIENT,
        ),
        causal_account="The invalid API ordering produces the observed unexpected result.",
        shared_supporting_evidence_ids=["symptom-1"],
    )


def _review(
    dimension: EvidenceDimension,
    verdict: VerificationVerdict,
    *,
    anchor_label: str | None = None,
    alternative_label: str | None = None,
    supporting: list[str] | None = None,
    counter: list[str] | None = None,
) -> DimensionVerificationReport:
    labels = _anchor().symptom.label, _anchor().root_cause.label
    anchor = anchor_label or labels[0 if dimension is EvidenceDimension.SYMPTOM else 1]
    payload: dict[str, object] = {
        "dimension": dimension,
        "verdict": verdict,
        "anchor_label": anchor,
        "rationale": "The verification explains whether the frozen anchor remains reliable.",
        "supporting_evidence_ids": supporting or [],
        "counter_evidence_ids": counter or [],
        "confidence": 0.6,
    }
    if verdict is VerificationVerdict.REJECT:
        payload.update(
            alternative_label=alternative_label
            or (
                "Crash"
                if dimension is EvidenceDimension.SYMPTOM
                else "Incorrect Code Logic"
            ),
            corrected_claim=(
                "The process terminates before returning a result."
                if dimension is EvidenceDimension.SYMPTOM
                else "An internal condition selects the wrong branch."
            ),
            corrected_causal_chain=(
                []
                if dimension is EvidenceDimension.SYMPTOM
                else [
                    "The condition is computed incorrectly.",
                    "The wrong branch executes.",
                    "The result differs.",
                ]
            ),
        )
    return DimensionVerificationReport.model_validate(payload)


def _accept_symptom() -> DimensionVerificationReport:
    return _review(EvidenceDimension.SYMPTOM, VerificationVerdict.ACCEPT)


def _insufficient_root() -> DimensionVerificationReport:
    return _review(
        EvidenceDimension.ROOT_CAUSE, VerificationVerdict.INSUFFICIENT_TO_REJECT
    )


def _valid_root_rejection(evidence_id: str = "root-1") -> DimensionVerificationReport:
    return _review(
        EvidenceDimension.ROOT_CAUSE,
        VerificationVerdict.REJECT,
        supporting=[evidence_id],
    )


def test_accept_and_insufficient_preserve_anchor_without_model_call() -> None:
    anchor = _anchor()

    report = compose_stage3_team_candidate(
        "A", anchor, _accept_symptom(), _insufficient_root(), _view("ase2022")
    )

    assert report.symptom == anchor.symptom
    assert report.root_cause == anchor.root_cause
    assert [audit.accepted for audit in report.correction_audit] == [False, False]
    assert [audit.final_label for audit in report.correction_audit] == [
        "Unexpected Output",
        "API Misuse",
    ]


def test_valid_owned_root_rejection_changes_only_root() -> None:
    anchor = _anchor()

    report = compose_stage3_team_candidate(
        "B", anchor, _accept_symptom(), _valid_root_rejection(), _view()
    )

    assert report.symptom == anchor.symptom
    assert report.root_cause.label == "Incorrect Code Logic"
    assert (
        report.root_cause.defect_mechanism
        == "An internal condition selects the wrong branch."
    )
    assert report.root_cause.causal_chain == (
        "The condition is computed incorrectly.",
        "The wrong branch executes.",
        "The result differs.",
    )
    assert report.root_cause.confidence == min(
        anchor.root_cause.confidence, _valid_root_rejection().confidence
    )
    assert report.correction_audit[1].accepted is True
    assert report.correction_audit[1].supporting_evidence_ids == ("root-1",)


@pytest.mark.parametrize("evidence_id", ["unknown-id", "quarantined-1", "symptom-1"])
def test_invalid_root_correction_fails_closed(evidence_id: str) -> None:
    with pytest.raises(CompositionError):
        compose_stage3_team_candidate(
            "A",
            _anchor(),
            _accept_symptom(),
            _valid_root_rejection(evidence_id),
            _view(),
        )


def test_duplicate_valid_and_quarantined_evidence_id_fails_as_composition_error() -> (
    None
):
    view = _view()
    duplicate = _item("root-1", "code_context", record_id="other-record")
    ambiguous_view = EvidenceView(
        record_id=view.record_id,
        task=view.task,
        taxonomy=view.model_dump(mode="python")["taxonomy"],
        domain_profile=view.domain_profile,
        ledger_version=view.ledger_version,
        items=view.items + (duplicate,),
    )

    with pytest.raises(CompositionError) as raised:
        compose_stage3_team_candidate(
            "A", _anchor(), _accept_symptom(), _valid_root_rejection(), ambiguous_view
        )

    assert raised.value.__cause__ is not None


@pytest.mark.parametrize(
    ("symptom", "root"),
    [
        (
            _review(EvidenceDimension.ROOT_CAUSE, VerificationVerdict.ACCEPT),
            _insufficient_root(),
        ),
        (
            _review(
                EvidenceDimension.SYMPTOM,
                VerificationVerdict.ACCEPT,
                anchor_label="Crash",
            ),
            _insufficient_root(),
        ),
    ],
)
def test_mismatched_dimension_or_anchor_label_fails_closed(
    symptom: DimensionVerificationReport, root: DimensionVerificationReport
) -> None:
    with pytest.raises(CompositionError):
        compose_stage3_team_candidate("A", _anchor(), symptom, root, _view())


@pytest.mark.parametrize(
    "dimension", [EvidenceDimension.SYMPTOM, EvidenceDimension.ROOT_CAUSE]
)
def test_unknown_alternative_fails_closed(dimension: EvidenceDimension) -> None:
    review = _review(
        dimension,
        VerificationVerdict.REJECT,
        alternative_label="Not In Taxonomy",
        supporting=[
            "symptom-1" if dimension is EvidenceDimension.SYMPTOM else "root-1"
        ],
    )
    symptom = review if dimension is EvidenceDimension.SYMPTOM else _accept_symptom()
    root = review if dimension is EvidenceDimension.ROOT_CAUSE else _insufficient_root()

    with pytest.raises(CompositionError):
        compose_stage3_team_candidate("A", _anchor(), symptom, root, _view())


def test_symptom_rejection_changes_only_symptom_and_deduplicates_supporting_ids() -> (
    None
):
    symptom = _review(
        EvidenceDimension.SYMPTOM, VerificationVerdict.REJECT, supporting=["symptom-1"]
    )
    anchor = _anchor()

    report = compose_stage3_team_candidate(
        "A", anchor, symptom, _insufficient_root(), _view()
    )

    assert report.symptom.label == "Crash"
    assert (
        report.symptom.behavior_claim
        == "The process terminates before returning a result."
    )
    assert report.root_cause == anchor.root_cause
    assert report.symptom.supporting_evidence_ids == ("symptom-1",)


def test_unknown_counter_evidence_fails_closed() -> None:
    root = _valid_root_rejection()
    root = root.model_copy(update={"counter_evidence_ids": ["unknown-id"]})

    with pytest.raises(CompositionError, match="counter"):
        compose_stage3_team_candidate("A", _anchor(), _accept_symptom(), root, _view())


def test_ase_issue_context_can_support_root_but_issta_issue_only_cannot() -> None:
    ase_report = compose_stage3_team_candidate(
        "A",
        _anchor(),
        _accept_symptom(),
        _valid_root_rejection("symptom-1"),
        _view("ase2022"),
    )
    assert ase_report.root_cause.label == "Incorrect Code Logic"

    with pytest.raises(CompositionError, match="incapable"):
        compose_stage3_team_candidate(
            "A",
            _anchor(),
            _accept_symptom(),
            _valid_root_rejection("symptom-1"),
            _view("issta2024"),
        )


def test_input_models_are_not_mutated_and_output_order_is_stable() -> None:
    anchor = _anchor()
    symptom = _accept_symptom()
    root = _valid_root_rejection()
    before = (anchor.model_dump(), symptom.model_dump(), root.model_dump())

    report = compose_stage3_team_candidate("A", anchor, symptom, root, _view())

    assert (anchor.model_dump(), symptom.model_dump(), root.model_dump()) == before
    assert [review.dimension for review in report.verifications] == [
        EvidenceDimension.SYMPTOM,
        EvidenceDimension.ROOT_CAUSE,
    ]
    assert [audit.dimension for audit in report.correction_audit] == [
        EvidenceDimension.SYMPTOM,
        EvidenceDimension.ROOT_CAUSE,
    ]


def _baseline(*, valid: bool = True) -> BaselineAnchor:
    return BaselineAnchor(
        record_id="record-1",
        valid=valid,
        symptom_label="Unexpected Output" if valid else None,
        root_cause_label="API Misuse" if valid else None,
        source_config_hash="a" * 64,
        source_predictions_sha256="b" * 64,
    )


def _structure() -> TaxonomyStructure:
    return TaxonomyStructure(
        schema_version=1,
        domain="issta2024",
        nodes=(
            TaxonomyNode(
                dimension="symptom",
                label="Unexpected Output",
                semantic_origin=TaxonomySemanticOrigin.OPERATIONAL_DEFINITION,
                definition="A returned result differs from the requested result.",
                abstraction_level="observable outcome",
                responsibility_scope="runtime result",
                concept_kind="outcome",
            ),
            TaxonomyNode(
                dimension="symptom",
                label="Crash",
                semantic_origin=TaxonomySemanticOrigin.OPERATIONAL_DEFINITION,
                definition="Execution terminates before returning a result.",
                abstraction_level="observable outcome",
                responsibility_scope="runtime process",
                concept_kind="outcome",
            ),
            TaxonomyNode(
                dimension="root_cause",
                label="API Misuse",
                semantic_origin=TaxonomySemanticOrigin.OPERATIONAL_DEFINITION,
                definition="A caller violates an API precondition.",
                abstraction_level="defect mechanism",
                responsibility_scope="API caller",
                concept_kind="mechanism",
            ),
            TaxonomyNode(
                dimension="root_cause",
                label="Incorrect Code Logic",
                semantic_origin=TaxonomySemanticOrigin.OPERATIONAL_DEFINITION,
                definition="Internal control flow selects an incorrect branch.",
                abstraction_level="defect mechanism",
                responsibility_scope="implementation",
                concept_kind="mechanism",
            ),
        ),
        boundary_cards=(
            BoundaryCard(
                card_id="symptom-output-crash",
                dimension="symptom",
                labels=("Unexpected Output", "Crash"),
                semantic_origin=TaxonomySemanticOrigin.OPERATIONAL_DEFINITION,
                decision_question="Does execution return a result?",
                observable_slots=("execution outcome",),
                criteria=(
                    BoundaryCriterion(
                        label="Unexpected Output",
                        positive_conditions=("a result is returned",),
                        exclusion_conditions=("execution terminates before returning",),
                    ),
                    BoundaryCriterion(
                        label="Crash",
                        positive_conditions=("execution terminates before returning",),
                        exclusion_conditions=("a result is returned",),
                    ),
                ),
            ),
            BoundaryCard(
                card_id="cause-api-logic",
                dimension="root_cause",
                labels=("API Misuse", "Incorrect Code Logic"),
                semantic_origin=TaxonomySemanticOrigin.OPERATIONAL_DEFINITION,
                decision_question="Who owns the failing mechanism?",
                observable_slots=("mechanism owner",),
                criteria=(
                    BoundaryCriterion(
                        label="API Misuse",
                        positive_conditions=("caller violates a precondition",),
                        exclusion_conditions=(
                            "internal branch selects the wrong path",
                        ),
                    ),
                    BoundaryCriterion(
                        label="Incorrect Code Logic",
                        positive_conditions=("internal branch selects the wrong path",),
                        exclusion_conditions=("caller violates a precondition",),
                    ),
                ),
            ),
        ),
    )


def _assessment(
    dimension: EvidenceDimension,
    *,
    supporting: tuple[str, ...],
    counter: tuple[str, ...] = (),
) -> BaselineRevisionAssessment:
    return BaselineRevisionAssessment(
        dimension=dimension,
        verdict=RevisionAssessmentVerdict.REVISE,
        baseline_label=(
            "Unexpected Output"
            if dimension is EvidenceDimension.SYMPTOM
            else "API Misuse"
        ),
        proposed_label=(
            "Crash"
            if dimension is EvidenceDimension.SYMPTOM
            else "Incorrect Code Logic"
        ),
        boundary_card_id=(
            "symptom-output-crash"
            if dimension is EvidenceDimension.SYMPTOM
            else "cause-api-logic"
        ),
        contradicted_baseline_condition=(
            "execution terminates before returning"
            if dimension is EvidenceDimension.SYMPTOM
            else "internal branch selects the wrong path"
        ),
        satisfied_proposed_conditions=(
            ("execution terminates before returning",)
            if dimension is EvidenceDimension.SYMPTOM
            else ("internal branch selects the wrong path",)
        ),
        supporting_evidence_ids=supporting,
        counter_evidence_ids=counter,
        baseline_source_config_hash="a" * 64,
        baseline_source_predictions_sha256="b" * 64,
        taxonomy_structure_hash=taxonomy_structure_hash(_structure()),
        evidence_view_hash=canonical_evidence_view_hash(_view()),
    )


def _team(
    team_id: str,
    assessments: tuple[BaselineRevisionAssessment, ...],
    *,
    consistent: bool = True,
) -> Stage3TeamReport:
    symptom = _anchor().symptom
    root_cause = _anchor().root_cause
    revision_consistency = []
    for assessment in assessments:
        if assessment.verdict is RevisionAssessmentVerdict.REVISE:
            if assessment.dimension is EvidenceDimension.SYMPTOM:
                symptom = symptom.model_copy(
                    update={"label": assessment.proposed_label}
                )
            else:
                root_cause = root_cause.model_copy(
                    update={"label": assessment.proposed_label}
                )
            revision_consistency.append(
                RevisionConsistencyReport(
                    dimension=assessment.dimension,
                    baseline_label=assessment.baseline_label,
                    proposed_label=assessment.proposed_label,
                    status=ConsistencyStatus.CONSISTENT,
                    rationale="The replacement boundary is causally supported by the cited evidence.",
                    supporting_evidence_ids=assessment.supporting_evidence_ids,
                    assessment_digest=baseline_revision_assessment_digest(assessment),
                    boundary_card_id=assessment.boundary_card_id,
                    baseline_source_config_hash=assessment.baseline_source_config_hash,
                    baseline_source_predictions_sha256=assessment.baseline_source_predictions_sha256,
                    taxonomy_structure_hash=assessment.taxonomy_structure_hash,
                    evidence_view_hash=assessment.evidence_view_hash,
                )
            )
    return Stage3TeamReport(
        team_id=team_id,
        symptom=symptom,
        root_cause=root_cause,
        consistency=CausalConsistencyReport(
            status=(
                ConsistencyStatus.CONSISTENT
                if consistent
                else ConsistencyStatus.CAUSE_REVIEW
            ),
            rationale="The proposed mechanism explains the observed behavior.",
            supporting_evidence_ids=["symptom-1"],
        ),
        baseline_revision_assessments=assessments,
        revision_consistency=tuple(revision_consistency),
    )


def test_composition_signs_dimension_certificates_only_after_independent_exact_agreement() -> (
    None
):
    reports = (
        _team(
            "A",
            (
                _assessment(EvidenceDimension.SYMPTOM, supporting=("symptom-1",)),
                _assessment(
                    EvidenceDimension.ROOT_CAUSE,
                    supporting=("root-1",),
                    counter=("root-1",),
                ),
            ),
        ),
        _team(
            "B",
            (
                _assessment(EvidenceDimension.SYMPTOM, supporting=("symptom-1",)),
                _assessment(
                    EvidenceDimension.ROOT_CAUSE,
                    supporting=("root-1",),
                    counter=("root-1",),
                ),
            ),
        ),
    )

    certificates = compose_baseline_revision_certificates(
        anchor=_baseline(), reports=reports, structure=_structure(), view=_view()
    )

    assert [
        (certificate.dimension, certificate.supporting_team_ids)
        for certificate in certificates
    ] == [
        (EvidenceDimension.SYMPTOM, ("A", "B")),
        (EvidenceDimension.ROOT_CAUSE, ("A", "B")),
    ]
    assert certificates[1].supporting_evidence_ids == ("root-1",)
    assert certificates[1].counter_evidence_ids == ("root-1",)


def test_composition_does_not_union_wash_an_invalid_team_citation() -> None:
    reports = (
        _team(
            "A", (_assessment(EvidenceDimension.SYMPTOM, supporting=("unknown-id",)),)
        ),
        _team(
            "B", (_assessment(EvidenceDimension.SYMPTOM, supporting=("symptom-1",)),)
        ),
    )

    certificates = compose_baseline_revision_certificates(
        anchor=_baseline(), reports=reports, structure=_structure(), view=_view()
    )

    assert certificates == ()


def test_composition_skips_an_unavailable_baseline() -> None:
    reports = (
        _team(
            "A", (_assessment(EvidenceDimension.SYMPTOM, supporting=("symptom-1",)),)
        ),
        _team(
            "B", (_assessment(EvidenceDimension.SYMPTOM, supporting=("symptom-1",)),)
        ),
    )
    assert (
        compose_baseline_revision_certificates(
            anchor=_baseline(valid=False),
            reports=reports,
            structure=_structure(),
            view=_view(),
        )
        == ()
    )


def test_revision_assessment_can_preserve_without_replacement_fields() -> None:
    assessment = BaselineRevisionAssessment(
        dimension=EvidenceDimension.SYMPTOM,
        verdict=RevisionAssessmentVerdict.PRESERVE,
        baseline_source_config_hash="a" * 64,
        baseline_source_predictions_sha256="b" * 64,
        taxonomy_structure_hash="c" * 64,
        evidence_view_hash="d" * 64,
    )

    assert assessment.proposed_label is None


def test_composition_rejects_assessment_detached_from_final_team_label() -> None:
    assessment = _assessment(EvidenceDimension.SYMPTOM, supporting=("symptom-1",))
    reports = (
        _team("A", (assessment,)).model_copy(update={"symptom": _anchor().symptom}),
        _team("B", (assessment,)),
    )

    assert (
        compose_baseline_revision_certificates(
            anchor=_baseline(), reports=reports, structure=_structure(), view=_view()
        )
        == ()
    )


def test_composition_rejects_stale_provenance_and_incapable_counter_per_team() -> None:
    stale = _assessment(
        EvidenceDimension.SYMPTOM, supporting=("symptom-1",)
    ).model_copy(update={"evidence_view_hash": "0" * 64})
    root_with_bad_counter = _assessment(
        EvidenceDimension.ROOT_CAUSE,
        supporting=("root-1",),
        counter=("counter-1",),
    )
    assert (
        compose_baseline_revision_certificates(
            anchor=_baseline(),
            reports=(_team("A", (stale,)), _team("B", (stale,))),
            structure=_structure(),
            view=_view(),
        )
        == ()
    )
    assert (
        compose_baseline_revision_certificates(
            anchor=_baseline(),
            reports=(
                _team("A", (root_with_bad_counter,)),
                _team("B", (root_with_bad_counter,)),
            ),
            structure=_structure(),
            view=_view(),
        )
        == ()
    )


@pytest.mark.parametrize(
    "field",
    [
        "assessment_digest",
        "boundary_card_id",
        "baseline_source_config_hash",
        "baseline_source_predictions_sha256",
        "taxonomy_structure_hash",
        "evidence_view_hash",
        "unknown_citation",
        "incapable_citation",
    ],
)
def test_composer_rejects_each_forged_revision_consistency_binding(field: str) -> None:
    assessment = _assessment(EvidenceDimension.SYMPTOM, supporting=("symptom-1",))
    reports = (_team("A", (assessment,)), _team("B", (assessment,)))
    original = reports[0].revision_consistency[0]
    if field == "unknown_citation":
        forged = original.model_copy(
            update={"supporting_evidence_ids": ("unknown-id",)}
        )
    elif field == "incapable_citation":
        forged = original.model_copy(update={"supporting_evidence_ids": ("root-1",)})
    else:
        forged = original.model_copy(update={field: "0" * 64})
    reports = (
        reports[0].model_copy(update={"revision_consistency": (forged,)}),
        reports[1],
    )

    assert (
        compose_baseline_revision_certificates(
            anchor=_baseline(), reports=reports, structure=_structure(), view=_view()
        )
        == ()
    )
