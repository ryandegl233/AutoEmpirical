from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from Benchmark.src.adaptive_empirical_workflow.baseline_preservation import (
    PreservationGateError,
    apply_baseline_preservation_gate as _apply_baseline_preservation_gate,
)
from Benchmark.src.adaptive_empirical_workflow.baseline_anchor import (
    baseline_anchor_hash,
)
from Benchmark.src.adaptive_empirical_workflow.contracts import (
    ArbitrationSource,
    BaselineAnchor,
    BaselineDecisionAction,
    BoundaryCard,
    BoundaryCriterion,
    EvidenceDimension,
    EvidenceExplicitness,
    EvidenceItem,
    EvidenceView,
    LabelRevisionCertificate,
    RevisionBasis,
    Stage3FinalDecision,
    Stage3Verification,
    TaxonomyNode,
    TaxonomySemanticOrigin,
    TaxonomyStructure,
)
from Benchmark.src.adaptive_empirical_workflow.taxonomy_structure import (
    taxonomy_structure_hash,
)


def apply_baseline_preservation_gate(**kwargs: object):
    """Supply the independently bound digests used by these isolated fixtures."""

    kwargs.setdefault("expected_anchor_hash", baseline_anchor_hash(kwargs["anchor"]))
    kwargs.setdefault(
        "expected_structure_hash", taxonomy_structure_hash(kwargs["structure"])
    )
    return _apply_baseline_preservation_gate(**kwargs)


def _item(
    evidence_id: str = "issue-1", source_type: str = "issue_body"
) -> EvidenceItem:
    content = "The runtime starts and then terminates with an exception."
    return EvidenceItem(
        evidence_id=evidence_id,
        record_id="r1",
        source_type=source_type,
        source_uri="https://example.test/issues/1",
        retrieved_at="2026-08-16T00:00:00Z",
        content=content,
        content_sha256=hashlib.sha256(content.encode()).hexdigest(),
        explicitness=EvidenceExplicitness.DIRECT,
    )


def _view(*items: EvidenceItem) -> EvidenceView:
    return EvidenceView(
        record_id="r1",
        task="Classify Stage 3.",
        taxonomy={
            "symptom": ["Crash", "Incorrect Functionality"],
            "root_cause": ["API Misuse", "Incorrect Code Logic"],
        },
        domain_profile="ase2022",
        ledger_version=len(items),
        items=items,
    )


def _node(label: str, dimension: str, neighbor: str) -> TaxonomyNode:
    return TaxonomyNode(
        label=label,
        dimension=dimension,
        definition=f"Definition for {label}.",
        semantic_origin=TaxonomySemanticOrigin.PAPER_DEFINITION,
        abstraction_level=(
            "observable_outcome" if dimension == "symptom" else "root_mechanism"
        ),
        responsibility_scope="runtime",
        concept_kind="outcome" if dimension == "symptom" else "mechanism",
        nearest_neighbors=(neighbor,),
    )


def _card(card_id: str, dimension: str, first: str, second: str) -> BoundaryCard:
    return BoundaryCard(
        card_id=card_id,
        dimension=dimension,
        labels=(first, second),
        semantic_origin=TaxonomySemanticOrigin.OPERATIONAL_DEFINITION,
        decision_question="Which directly supported condition separates the labels?",
        observable_slots=("direct_condition",),
        criteria=(
            BoundaryCriterion(
                label=first,
                positive_conditions=(f"condition for {first}",),
                exclusion_conditions=(f"condition for {second}",),
            ),
            BoundaryCriterion(
                label=second,
                positive_conditions=(f"condition for {second}",),
                exclusion_conditions=(f"condition for {first}",),
            ),
        ),
    )


def _structure() -> TaxonomyStructure:
    return TaxonomyStructure(
        schema_version=1,
        domain="ase2022",
        nodes=(
            _node("Crash", "symptom", "Incorrect Functionality"),
            _node("Incorrect Functionality", "symptom", "Crash"),
            _node("API Misuse", "root_cause", "Incorrect Code Logic"),
            _node("Incorrect Code Logic", "root_cause", "API Misuse"),
        ),
        boundary_cards=(
            _card(
                "symptom-crash-vs-functionality",
                "symptom",
                "Crash",
                "Incorrect Functionality",
            ),
            _card(
                "root-api-vs-logic",
                "root_cause",
                "API Misuse",
                "Incorrect Code Logic",
            ),
        ),
    )


def _anchor(*, valid: bool = True) -> BaselineAnchor:
    return BaselineAnchor(
        record_id="r1",
        valid=valid,
        symptom_label="Incorrect Functionality" if valid else None,
        root_cause_label="API Misuse" if valid else None,
        source_config_hash="a" * 64,
        source_predictions_sha256="b" * 64,
    )


def _candidate() -> Stage3FinalDecision:
    return Stage3FinalDecision(
        symptom_label="Crash",
        root_cause_label="Incorrect Code Logic",
        confidence=0.8,
        rationale="Two current teams produced this evidence-grounded candidate.",
        supporting_evidence_ids=["issue-1"],
        source=ArbitrationSource.DIRECT_CONSENSUS,
    )


def _verification(*, valid: bool = True) -> Stage3Verification:
    return Stage3Verification(
        symptom_label="Crash",
        root_cause_label="Incorrect Code Logic",
        valid=valid,
        errors=[] if valid else ["candidate verification failed"],
    )


def _certificate(
    dimension: EvidenceDimension = EvidenceDimension.SYMPTOM,
    *,
    teams: tuple[str, ...] = ("A", "B"),
    card_id: str = "symptom-crash-vs-functionality",
    evidence_ids: tuple[str, ...] = ("issue-1",),
) -> LabelRevisionCertificate:
    is_symptom = dimension is EvidenceDimension.SYMPTOM
    return LabelRevisionCertificate(
        dimension=dimension,
        baseline_label="Incorrect Functionality" if is_symptom else "API Misuse",
        proposed_label="Crash" if is_symptom else "Incorrect Code Logic",
        revision_basis=RevisionBasis.TAXONOMY_BOUNDARY,
        boundary_card_id=card_id,
        contradicted_baseline_condition=(
            "condition for Crash"
            if is_symptom
            else "condition for Incorrect Code Logic"
        ),
        satisfied_proposed_conditions=(
            (
                "condition for Crash"
                if is_symptom
                else "condition for Incorrect Code Logic"
            ),
        ),
        supporting_evidence_ids=evidence_ids,
        counter_evidence_ids=(),
        supporting_team_ids=teams,
    )


def test_valid_anchor_can_preserve_without_fabricating_a_candidate() -> None:
    result = apply_baseline_preservation_gate(
        anchor=_anchor(),
        candidate=None,
        candidate_verification=None,
        certificates=(),
        structure=_structure(),
        view=_view(_item()),
        expected_source_config_hash="a" * 64,
        expected_source_predictions_sha256="b" * 64,
    )

    assert result.symptom_label == "Incorrect Functionality"
    assert result.root_cause_label == "API Misuse"
    assert result.symptom_action is BaselineDecisionAction.PRESERVED
    assert result.root_cause_action is BaselineDecisionAction.PRESERVED


def test_invalid_anchor_cannot_fallback_without_a_verified_candidate() -> None:
    with pytest.raises(PreservationGateError, match="verified candidate"):
        apply_baseline_preservation_gate(
            anchor=_anchor(valid=False),
            candidate=None,
            candidate_verification=None,
            certificates=(),
            structure=_structure(),
            view=_view(_item()),
            expected_source_config_hash="a" * 64,
            expected_source_predictions_sha256="b" * 64,
        )


def test_no_certificate_preserves_each_valid_baseline_dimension() -> None:
    result = apply_baseline_preservation_gate(
        anchor=_anchor(),
        candidate=_candidate(),
        candidate_verification=None,
        certificates=(),
        structure=_structure(),
        view=_view(_item()),
        expected_source_config_hash="a" * 64,
        expected_source_predictions_sha256="b" * 64,
    )

    assert result.symptom_label == "Incorrect Functionality"
    assert result.root_cause_label == "API Misuse"
    assert result.symptom_action is BaselineDecisionAction.PRESERVED
    assert result.root_cause_action is BaselineDecisionAction.PRESERVED
    assert result.applied_certificates == ()


def test_single_team_proposal_is_not_a_revision_certificate() -> None:
    result = apply_baseline_preservation_gate(
        anchor=_anchor(),
        candidate=_candidate(),
        candidate_verification=None,
        certificates=(_certificate(teams=("A",)),),
        structure=_structure(),
        view=_view(_item()),
        expected_source_config_hash="a" * 64,
        expected_source_predictions_sha256="b" * 64,
    )

    assert result.symptom_label == "Incorrect Functionality"
    assert result.symptom_action is BaselineDecisionAction.PRESERVED
    assert result.applied_certificates == ()


def test_valid_two_team_certificate_revises_only_its_owned_dimension() -> None:
    certificate = _certificate()

    result = apply_baseline_preservation_gate(
        anchor=_anchor(),
        candidate=_candidate(),
        candidate_verification=None,
        certificates=(certificate,),
        structure=_structure(),
        view=_view(_item()),
        expected_source_config_hash="a" * 64,
        expected_source_predictions_sha256="b" * 64,
    )

    assert result.symptom_label == "Crash"
    assert result.root_cause_label == "API Misuse"
    assert result.symptom_action is BaselineDecisionAction.REVISED
    assert result.root_cause_action is BaselineDecisionAction.PRESERVED
    assert result.applied_certificates == (certificate,)


def test_invalid_baseline_uses_verified_candidate_without_fabricating_a_label() -> None:
    result = apply_baseline_preservation_gate(
        anchor=_anchor(valid=False),
        candidate=_candidate(),
        candidate_verification=_verification(),
        certificates=(),
        structure=_structure(),
        view=_view(_item()),
        expected_source_config_hash="a" * 64,
        expected_source_predictions_sha256="b" * 64,
    )

    assert (result.symptom_label, result.root_cause_label) == (
        "Crash",
        "Incorrect Code Logic",
    )
    assert result.symptom_action is BaselineDecisionAction.BASELINE_UNAVAILABLE
    assert result.root_cause_action is BaselineDecisionAction.BASELINE_UNAVAILABLE


@pytest.mark.parametrize(
    ("certificate", "message"),
    (
        (_certificate(card_id="unknown-card"), "unknown Boundary Card"),
        (
            _certificate(card_id="root-api-vs-logic"),
            "does not match certificate dimension",
        ),
        (_certificate(evidence_ids=("unknown-id",)), "unknown evidence IDs"),
        (
            _certificate(EvidenceDimension.ROOT_CAUSE, card_id="root-api-vs-logic"),
            "candidate label",
        ),
    ),
)
def test_claimed_certificate_fails_closed_when_provenance_or_scope_is_invalid(
    certificate: LabelRevisionCertificate, message: str
) -> None:
    candidate = _candidate()
    if certificate.dimension is EvidenceDimension.ROOT_CAUSE:
        candidate = candidate.model_copy(update={"root_cause_label": "API Misuse"})

    with pytest.raises(PreservationGateError, match=message):
        apply_baseline_preservation_gate(
            anchor=_anchor(),
            candidate=candidate,
            candidate_verification=None,
            certificates=(certificate,),
            structure=_structure(),
            view=_view(_item()),
            expected_source_config_hash="a" * 64,
            expected_source_predictions_sha256="b" * 64,
        )


def test_dimension_incapable_evidence_cannot_authorize_revision() -> None:
    with pytest.raises(PreservationGateError, match="incapable evidence IDs"):
        apply_baseline_preservation_gate(
            anchor=_anchor(),
            candidate=_candidate(),
            candidate_verification=None,
            certificates=(_certificate(),),
            structure=_structure(),
            view=_view(_item(source_type="source_code")),
            expected_source_config_hash="a" * 64,
            expected_source_predictions_sha256="b" * 64,
        )


def test_certificate_conditions_must_be_exact_boundary_card_conditions() -> None:
    certificate = _certificate().model_copy(
        update={"satisfied_proposed_conditions": ("unlisted invented condition",)}
    )

    with pytest.raises(PreservationGateError, match="proposed conditions"):
        apply_baseline_preservation_gate(
            anchor=_anchor(),
            candidate=_candidate(),
            candidate_verification=None,
            certificates=(certificate,),
            structure=_structure(),
            view=_view(_item()),
            expected_source_config_hash="a" * 64,
            expected_source_predictions_sha256="b" * 64,
        )


def test_gate_independently_rejects_anchor_label_outside_runtime_taxonomy() -> None:
    anchor = _anchor().model_copy(update={"symptom_label": "Unknown Symptom"})

    with pytest.raises(
        PreservationGateError, match="anchor label outside symptom taxonomy"
    ):
        apply_baseline_preservation_gate(
            anchor=anchor,
            candidate=_candidate(),
            candidate_verification=None,
            certificates=(),
            structure=_structure(),
            view=_view(_item()),
            expected_source_config_hash="a" * 64,
            expected_source_predictions_sha256="b" * 64,
        )


def test_invalid_baseline_rejects_unverified_uncertain_candidate() -> None:
    candidate = _candidate().model_copy(
        update={
            "source": ArbitrationSource.FALLBACK_UNCERTAIN,
            "supporting_evidence_ids": ["unknown-id"],
        }
    )

    with pytest.raises(PreservationGateError, match="verified candidate"):
        apply_baseline_preservation_gate(
            anchor=_anchor(valid=False),
            candidate=candidate,
            candidate_verification=_verification(),
            certificates=(),
            structure=_structure(),
            view=_view(_item()),
            expected_source_config_hash="a" * 64,
            expected_source_predictions_sha256="b" * 64,
        )


def test_invalid_baseline_accepts_only_matching_valid_verification() -> None:
    result = apply_baseline_preservation_gate(
        anchor=_anchor(valid=False),
        candidate=_candidate(),
        candidate_verification=_verification(),
        certificates=(),
        structure=_structure(),
        view=_view(_item()),
        expected_source_config_hash="a" * 64,
        expected_source_predictions_sha256="b" * 64,
    )

    assert result.symptom_action.value == "baseline_unavailable_fallback"
    assert result.root_cause_action.value == "baseline_unavailable_fallback"


def test_gate_binds_anchor_to_expected_artifact_and_config_hashes() -> None:
    with pytest.raises(PreservationGateError, match="source config hash"):
        apply_baseline_preservation_gate(
            anchor=_anchor(),
            candidate=_candidate(),
            candidate_verification=None,
            certificates=(),
            structure=_structure(),
            view=_view(_item()),
            expected_source_config_hash="c" * 64,
            expected_source_predictions_sha256="b" * 64,
        )


def test_direct_partial_structure_cannot_authorize_revision() -> None:
    partial = TaxonomyStructure(
        schema_version=1,
        domain="ase2022",
        nodes=(
            _node("Crash", "symptom", "Incorrect Functionality"),
            _node("Incorrect Functionality", "symptom", "Crash"),
        ),
        boundary_cards=(
            _card(
                "symptom-crash-vs-functionality",
                "symptom",
                "Crash",
                "Incorrect Functionality",
            ),
        ),
    )

    with pytest.raises(PreservationGateError, match="exactly match"):
        apply_baseline_preservation_gate(
            anchor=_anchor(),
            candidate=_candidate(),
            candidate_verification=None,
            certificates=(_certificate(),),
            structure=partial,
            view=_view(_item()),
            expected_source_config_hash="a" * 64,
            expected_source_predictions_sha256="b" * 64,
        )


def test_single_team_claim_still_rejects_invalid_card_and_citations() -> None:
    certificate = _certificate(
        teams=("A",), card_id="unknown-card", evidence_ids=("unknown-id",)
    )

    with pytest.raises(PreservationGateError, match="unknown Boundary Card"):
        apply_baseline_preservation_gate(
            anchor=_anchor(),
            candidate=_candidate(),
            candidate_verification=None,
            certificates=(certificate,),
            structure=_structure(),
            view=_view(_item()),
            expected_source_config_hash="a" * 64,
            expected_source_predictions_sha256="b" * 64,
        )


def test_certificate_requires_all_necessary_proposed_conditions() -> None:
    structure = _structure()
    symptom_card = structure.boundary_cards[0]
    proposed = symptom_card.criteria[0].model_copy(
        update={
            "positive_conditions": (
                "condition for Crash",
                "runtime termination is directly observed",
            )
        }
    )
    structure = structure.model_copy(
        update={
            "boundary_cards": (
                symptom_card.model_copy(
                    update={"criteria": (proposed, symptom_card.criteria[1])}
                ),
                structure.boundary_cards[1],
            )
        }
    )

    with pytest.raises(PreservationGateError, match="proposed conditions"):
        apply_baseline_preservation_gate(
            anchor=_anchor(),
            candidate=_candidate(),
            candidate_verification=None,
            certificates=(_certificate(),),
            structure=structure,
            view=_view(_item()),
            expected_source_config_hash="a" * 64,
            expected_source_predictions_sha256="b" * 64,
        )


def test_counter_evidence_must_support_certificate_dimension() -> None:
    certificate = _certificate().model_copy(
        update={"counter_evidence_ids": ("source-1",)}
    )

    with pytest.raises(PreservationGateError, match="incapable evidence IDs"):
        apply_baseline_preservation_gate(
            anchor=_anchor(),
            candidate=_candidate(),
            candidate_verification=None,
            certificates=(certificate,),
            structure=_structure(),
            view=_view(_item(), _item("source-1", "source_code")),
            expected_source_config_hash="a" * 64,
            expected_source_predictions_sha256="b" * 64,
        )


def test_gate_result_is_a_frozen_snapshot_of_applied_certificate() -> None:
    certificate = _certificate()
    result = apply_baseline_preservation_gate(
        anchor=_anchor(),
        candidate=_candidate(),
        candidate_verification=None,
        certificates=(certificate,),
        structure=_structure(),
        view=_view(_item()),
        expected_source_config_hash="a" * 64,
        expected_source_predictions_sha256="b" * 64,
    )
    before = result.model_dump_json()

    assert result.applied_certificates[0] is not certificate
    with pytest.raises(Exception, match="frozen"):
        certificate.boundary_card_id = "unknown-card"
    assert result.model_dump_json() == before


def test_taxonomy_structure_and_nested_cards_are_immutable() -> None:
    structure = _structure()

    with pytest.raises(Exception, match="frozen"):
        structure.boundary_cards[0].card_id = "changed-card"


def test_invalid_baseline_rejects_non_contract_zero_reference_candidate() -> None:
    candidate = SimpleNamespace(
        symptom_label="Crash",
        root_cause_label="Incorrect Code Logic",
        confidence=0.9,
        rationale="This object bypasses the required candidate contract entirely.",
        supporting_evidence_ids=[],
        source=ArbitrationSource.DIRECT_CONSENSUS,
    )
    with pytest.raises(PreservationGateError, match="candidate contract"):
        apply_baseline_preservation_gate(
            anchor=_anchor(valid=False),
            candidate=candidate,
            candidate_verification=_verification(),
            certificates=(),
            structure=_structure(),
            view=_view(_item()),
            expected_source_config_hash="a" * 64,
            expected_source_predictions_sha256="b" * 64,
        )


def test_correct_artifact_hashes_cannot_bind_tampered_anchor_labels() -> None:
    trusted = _anchor()
    tampered = trusted.model_copy(update={"symptom_label": "Crash"})
    with pytest.raises(PreservationGateError, match="trusted record binding"):
        _apply_baseline_preservation_gate(
            anchor=tampered,
            candidate=_candidate(),
            candidate_verification=None,
            certificates=(),
            structure=_structure(),
            view=_view(_item()),
            expected_source_config_hash="a" * 64,
            expected_source_predictions_sha256="b" * 64,
            expected_anchor_hash=baseline_anchor_hash(trusted),
            expected_structure_hash=taxonomy_structure_hash(_structure()),
        )


def test_complete_forged_boundary_card_cannot_replace_trusted_structure() -> None:
    trusted = _structure()
    forged_card = trusted.boundary_cards[0].model_copy(
        update={"decision_question": "Invented record-specific decision question?"}
    )
    forged = trusted.model_copy(
        update={
            "boundary_cards": (forged_card, trusted.boundary_cards[1]),
        }
    )
    with pytest.raises(PreservationGateError, match="trusted artifact binding"):
        _apply_baseline_preservation_gate(
            anchor=_anchor(),
            candidate=_candidate(),
            candidate_verification=None,
            certificates=(_certificate(),),
            structure=forged,
            view=_view(_item()),
            expected_source_config_hash="a" * 64,
            expected_source_predictions_sha256="b" * 64,
            expected_anchor_hash=baseline_anchor_hash(_anchor()),
            expected_structure_hash=taxonomy_structure_hash(trusted),
        )
