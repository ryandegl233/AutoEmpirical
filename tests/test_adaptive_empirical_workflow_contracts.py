from __future__ import annotations

import hashlib
from dataclasses import dataclass
from types import MappingProxyType

import pytest
from pydantic import BaseModel, ValidationError

from Benchmark.src.adaptive_empirical_workflow import contracts as contracts_module
from Benchmark.src.adaptive_empirical_workflow.contracts import (
    BoundaryChallenge,
    CausalConsistencyReport,
    ConsistencyStatus,
    DimensionVerificationReport,
    DimensionReadiness,
    EvidenceDimension,
    EvidenceExplicitness,
    EvidenceItem,
    EvidenceReadinessReport,
    EvidenceSufficiency,
    EvidenceRequest,
    EvidenceValidityIssue,
    EvidenceValidityReport,
    EvidenceView,
    FaultEvidenceAssessment,
    RepairCausalityAssessment,
    ResolutionStatus,
    ScopeBoundaryAssessment,
    SpecialistType,
    Stage2ArbitrationDecision,
    Stage2Decision,
    Stage3ArbitrationDecision,
    Stage3TeamReport,
    SymptomReport,
    RootCauseReport,
    TeamComposedCandidate,
    TeamCorrectionAudit,
    UnresolvedDecision,
    JointAnchorReport,
    VerificationVerdict,
)
from Benchmark.src.adaptive_empirical_workflow.ledger import EvidenceLedger


def test_revision_raw_contracts_are_task_distinct_and_hide_opaque_bindings() -> None:
    from Benchmark.src.adaptive_empirical_workflow import contracts

    assert hasattr(contracts, "RevisionEntailmentResult")
    assert hasattr(contracts, "RevisionFalsificationResult")
    entailment_schema = contracts.RevisionEntailmentResult.model_json_schema()
    falsification_schema = contracts.RevisionFalsificationResult.model_json_schema()

    assert entailment_schema != falsification_schema
    assert "condition_findings" in entailment_schema["properties"]
    assert "proposed_defeater_findings" in falsification_schema["properties"]
    forbidden = {
        "proposal_digest",
        "baseline_source_config_hash",
        "baseline_source_predictions_sha256",
        "taxonomy_structure_hash",
        "evidence_view_hash",
    }
    assert forbidden.isdisjoint(entailment_schema["properties"])
    assert forbidden.isdisjoint(falsification_schema["properties"])


def test_revision_raw_contracts_reject_each_others_schema() -> None:
    from Benchmark.src.adaptive_empirical_workflow import contracts

    entailment = {
        "condition_findings": [
            {
                "condition": "candidate condition",
                "status": "supported",
                "citation_ids": ["code-1"],
            }
        ],
        "baseline_exclusion_finding": {
            "condition": "baseline exclusion",
            "status": "supported",
            "citation_ids": ["code-1"],
        },
        "outcome": "entailed",
        "summary": "Direct repair evidence satisfies the candidate condition.",
    }
    falsification = {
        "baseline_survival_finding": {
            "condition": "baseline remains plausible",
            "status": "refuted",
            "citation_ids": ["code-1"],
        },
        "proposed_defeater_findings": [
            {
                "condition": "candidate defeater",
                "status": "refuted",
                "citation_ids": ["code-1"],
            }
        ],
        "strongest_competing_reading": {
            "label": "Baseline",
            "summary": "The strongest competing reading is contradicted by repair evidence.",
            "citation_ids": ["code-1"],
        },
        "outcome": "revision_survives",
        "summary": "The candidate survives the fixed falsification checks.",
    }

    contracts.RevisionEntailmentResult.model_validate(entailment)
    contracts.RevisionFalsificationResult.model_validate(falsification)
    with pytest.raises(ValidationError):
        contracts.RevisionEntailmentResult.model_validate(falsification)
    with pytest.raises(ValidationError):
        contracts.RevisionFalsificationResult.model_validate(entailment)


def _sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _item(
    evidence_id: str,
    content: str = "original evidence",
    *,
    metadata: dict[str, object] | None = None,
) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=evidence_id,
        record_id="record-1",
        source_type="issue",
        source_uri="https://example.test/issues/1",
        retrieved_at="2026-08-05T08:00:00Z",
        content=content,
        content_sha256=_sha256(content),
        explicitness=EvidenceExplicitness.DIRECT,
        metadata=metadata or {},
    )


def test_immutable_evidence_contracts_support_deep_copy_and_json_round_trip() -> None:
    item = _item("e-1").model_copy(
        update={
            "metadata": {
                "notes": ["stable"],
                "nested": {"count": 1},
                "enabled": True,
                "ratio": 0.5,
                "missing": None,
            }
        }
    )
    item = EvidenceItem.model_validate(item.model_dump(mode="python"))
    view = EvidenceView(
        record_id="record-1",
        task="stage3",
        taxonomy={
            "symptom": ["Crash"],
            "root_cause": ["Incorrect Code Logic"],
        },
        domain_profile="ase2022",
        ledger_version=1,
        items=(item,),
    )

    item_copy = item.model_copy(deep=True)
    view_copy = view.model_copy(deep=True)
    round_trip = EvidenceView.model_validate_json(view.model_dump_json())

    assert item_copy == item
    assert view_copy == view
    assert round_trip == view
    assert view.taxonomy["symptom"] == ("Crash",)
    assert view.items[0].model_dump(mode="python")["metadata"] == {
        "notes": ["stable"],
        "nested": {"count": 1},
        "enabled": True,
        "ratio": 0.5,
        "missing": None,
    }
    assert isinstance(view.taxonomy, MappingProxyType)
    assert isinstance(view.taxonomy["symptom"], tuple)
    assert isinstance(view.items[0].metadata, MappingProxyType)
    assert isinstance(view.items[0].metadata["notes"], tuple)


@dataclass
class _MutableDataclassMetadata:
    values: list[str]


class _MutableModelMetadata(BaseModel):
    values: list[str]


class _MutableCustomMetadata:
    def __init__(self) -> None:
        self.values = ["mutable"]


@pytest.mark.parametrize(
    "invalid_leaf",
    [
        _MutableDataclassMetadata(["mutable"]),
        _MutableModelMetadata(values=["mutable"]),
        _MutableCustomMetadata(),
        float("nan"),
        float("inf"),
    ],
)
def test_evidence_metadata_rejects_non_json_or_non_finite_leaves(
    invalid_leaf: object,
) -> None:
    with pytest.raises(ValidationError, match="JSON-compatible"):
        EvidenceItem(
            **_item("e-1").model_dump(exclude={"metadata"}),
            metadata={"invalid": invalid_leaf},
        )


@pytest.mark.parametrize(
    "invalid_collection",
    [frozenset({"unordered"}), range(3)],
)
def test_evidence_metadata_rejects_non_json_collection_inputs(
    invalid_collection: object,
) -> None:
    with pytest.raises(ValidationError, match="JSON-compatible"):
        EvidenceItem(
            **_item("e-1").model_dump(exclude={"metadata"}),
            metadata={"invalid": invalid_collection},
        )


def test_evidence_metadata_rejects_cycles_with_validation_error() -> None:
    cyclic: dict[str, object] = {}
    cyclic["self"] = cyclic

    with pytest.raises(ValidationError, match="cycle"):
        EvidenceItem(
            **_item("e-1").model_dump(exclude={"metadata"}),
            metadata=cyclic,
        )


def test_evidence_metadata_rejects_lone_surrogate_strings() -> None:
    with pytest.raises(ValidationError, match="Unicode"):
        EvidenceItem(
            **_item("e-1").model_dump(exclude={"metadata"}),
            metadata={"invalid": "\ud800"},
        )


def test_evidence_metadata_accepts_tuple_array_input_and_round_trips() -> None:
    item = EvidenceItem(
        **_item("e-1").model_dump(exclude={"metadata"}),
        metadata={"array": ("first", {"second": 2})},
    )

    assert item.model_dump(mode="json")["metadata"] == {
        "array": ["first", {"second": 2}]
    }
    assert EvidenceItem.model_validate_json(item.model_dump_json()) == item


def test_evidence_item_rejects_content_hash_mismatch() -> None:
    with pytest.raises(ValidationError, match="content_sha256"):
        EvidenceItem(
            evidence_id="e-1",
            record_id="record-1",
            source_type="issue",
            source_uri="https://example.test/issues/1",
            retrieved_at="2026-08-05T08:00:00Z",
            content="actual content",
            content_sha256="0" * 64,
            explicitness=EvidenceExplicitness.DIRECT,
        )


def test_ledger_is_append_only_and_versions_each_append() -> None:
    ledger = EvidenceLedger(
        record_id="record-1",
        task="Classify the supplied issue.",
        taxonomy={"decision": ["accepted_fault", "rejected_candidate"]},
        domain_profile="ase2022",
        initial_items=[_item("e-1")],
    )

    ledger.append(_item("e-2", "retrieved discussion"))

    assert ledger.version == 2
    assert ledger.get("e-1").content == "original evidence"
    assert ledger.get("e-2").content == "retrieved discussion"

    with pytest.raises(ValueError, match="already exists"):
        ledger.append(_item("e-1", "attempted replacement"))

    assert ledger.get("e-1").content == "original evidence"


def test_ledger_view_preserves_exact_evidence_and_version() -> None:
    ledger = EvidenceLedger(
        record_id="record-1",
        task="Classify the supplied issue.",
        taxonomy={"symptom": ["Crash"], "root_cause": ["Wrong Code Logic"]},
        domain_profile="issta2024",
        initial_items=[_item("e-1")],
    )

    view = ledger.view(["e-1"])

    assert view.record_id == "record-1"
    assert view.ledger_version == 1
    assert view.items[0].content == "original evidence"
    assert view.items[0].content_sha256 == _sha256("original evidence")


@pytest.mark.parametrize(
    "missing_fact,why_needed,expected_impact",
    [
        ("more context", "needed", "may help"),
        ("Get more information", "needed", "may help"),
        ("Whether a fault existed", "context", "may help"),
    ],
)
def test_evidence_request_rejects_vague_or_non_decisive_requests(
    missing_fact: str,
    why_needed: str,
    expected_impact: str,
) -> None:
    with pytest.raises(ValidationError):
        EvidenceRequest(
            request_id="req-1",
            missing_fact=missing_fact,
            why_needed=why_needed,
            target_specialist=SpecialistType.ISSUE_PR,
            target_source="linked issue",
            query="record-1",
            expected_decision_impact=expected_impact,
            max_items=3,
        )


def test_evidence_request_accepts_a_bounded_discriminating_question() -> None:
    request = EvidenceRequest(
        request_id="req-1",
        missing_fact="Whether the linked issue reports incorrect runtime behavior",
        why_needed=(
            "A confirmed failure distinguishes a fault repair from a feature change"
        ),
        target_specialist=SpecialistType.ISSUE_PR,
        target_source="linked issue and pull request",
        query="record-1 linked issue incorrect behavior",
        expected_decision_impact=(
            "A maintainer-confirmed failure supports accepted_fault; an enhancement "
            "request without failure evidence supports rejected_candidate"
        ),
        max_items=3,
    )

    assert request.target_specialist is SpecialistType.ISSUE_PR
    assert request.max_items == 3


def test_readiness_report_rejects_duplicate_dimensions() -> None:
    readiness = DimensionReadiness(
        dimension=EvidenceDimension.FAULT_EXISTENCE,
        sufficient=True,
        confirmed_evidence_ids=("e-1",),
        missing_facts=(),
        evidence_requests=(),
    )

    with pytest.raises(ValidationError, match="duplicate"):
        EvidenceReadinessReport(
            task="Classify the supplied issue.",
            dimensions=(readiness, readiness),
        )


@pytest.mark.parametrize(
    "sufficient,missing_facts,evidence_requests",
    [
        (True, ("Whether a fault existed",), ()),
        (False, (), ()),
    ],
)
def test_dimension_readiness_enforces_sufficient_and_missing_fact_rules(
    sufficient: bool,
    missing_facts: tuple[str, ...],
    evidence_requests: tuple[EvidenceRequest, ...],
) -> None:
    with pytest.raises(ValidationError):
        DimensionReadiness(
            dimension=EvidenceDimension.FAULT_EXISTENCE,
            sufficient=sufficient,
            confirmed_evidence_ids=(),
            missing_facts=missing_facts,
            evidence_requests=evidence_requests,
        )


def test_sufficient_readiness_requires_confirmed_evidence() -> None:
    with pytest.raises(ValidationError, match="confirmed evidence"):
        DimensionReadiness(
            dimension=EvidenceDimension.FAULT_EXISTENCE,
            sufficient=True,
            confirmed_evidence_ids=(),
            missing_facts=(),
            evidence_requests=(),
        )


def test_boundary_reports_expose_structured_boundary_evidence_ids() -> None:
    assert "boundary_evidence_ids" in SymptomReport.model_fields
    assert "boundary_evidence_ids" in RootCauseReport.model_fields


def test_evidence_capability_metadata_rejects_unknown_capability() -> None:
    with pytest.raises(ValidationError, match="evidence_capabilities"):
        _item(
            "e-capability",
            metadata={"evidence_capabilities": ["barcode_observation"]},
        )


def test_evidence_validity_report_rejects_duplicate_or_overlapping_ids() -> None:
    with pytest.raises(ValidationError, match="validity evidence ids must be unique"):
        EvidenceValidityReport(
            valid_evidence_ids=("issue-1", "issue-1"),
            capabilities_by_evidence_id={"issue-1": ("symptom_observation",)},
        )

    with pytest.raises(ValidationError, match="valid and quarantined evidence ids"):
        EvidenceValidityReport(
            valid_evidence_ids=("issue-1",),
            quarantined=(
                EvidenceValidityIssue(
                    evidence_id="issue-1",
                    reasons=("unknown_source_type",),
                ),
            ),
            capabilities_by_evidence_id={"issue-1": ("symptom_observation",)},
        )


def test_evidence_validity_report_requires_capabilities_for_exactly_valid_ids() -> None:
    with pytest.raises(ValidationError, match="capability ids must match valid"):
        EvidenceValidityReport(
            valid_evidence_ids=("issue-1",),
            capabilities_by_evidence_id={"other-issue": ("symptom_observation",)},
        )

    with pytest.raises(ValidationError, match="must not be empty"):
        EvidenceValidityReport(
            valid_evidence_ids=("issue-1",),
            capabilities_by_evidence_id={"issue-1": ()},
        )


def test_unresolved_decision_has_only_an_explicit_stop_contract() -> None:
    decision = UnresolvedDecision(
        status=ResolutionStatus.UNRESOLVED,
        dimensions=(EvidenceDimension.SYMPTOM,),
        missing_facts=("The observed externally visible behavior is not documented",),
        stop_reason="The available evidence cannot distinguish the symptom label.",
    )

    assert decision.status is ResolutionStatus.UNRESOLVED
    with pytest.raises(ValidationError, match="label"):
        UnresolvedDecision(
            status=ResolutionStatus.UNRESOLVED,
            dimensions=(EvidenceDimension.SYMPTOM,),
            missing_facts=(
                "The observed externally visible behavior is not documented",
            ),
            stop_reason="The available evidence cannot distinguish the symptom label.",
            label="Crash",
        )


def test_resolved_stage2_arbitration_requires_decision_and_no_gaps() -> None:
    decision = Stage2ArbitrationDecision(
        resolution_status=ResolutionStatus.RESOLVED,
        decision=Stage2Decision.ACCEPTED,
        confidence=0.86,
        rationale="The cited maintainer report establishes the prior faulty behavior.",
        supporting_evidence_ids=["issue-1"],
        resolved_dimensions=["decision"],
    )

    assert decision.decision is Stage2Decision.ACCEPTED
    with pytest.raises(ValidationError, match="unresolved dimensions or missing facts"):
        Stage2ArbitrationDecision(
            resolution_status=ResolutionStatus.RESOLVED,
            decision=Stage2Decision.ACCEPTED,
            confidence=0.86,
            rationale="The cited evidence supports the proposed Stage 2 decision.",
            supporting_evidence_ids=["issue-1"],
            resolved_dimensions=["decision"],
            missing_facts=["Whether maintainers considered the behavior faulty."],
        )


def test_unresolved_stage2_arbitration_forbids_a_forced_decision() -> None:
    decision = Stage2ArbitrationDecision(
        resolution_status=ResolutionStatus.UNRESOLVED,
        rationale="The cited material cannot distinguish repair from enhancement.",
        supporting_evidence_ids=["issue-1"],
        unresolved_dimensions=["decision"],
        missing_facts=["Whether the prior runtime behavior was considered faulty."],
    )

    assert decision.decision is None
    with pytest.raises(ValidationError, match="must not contain a decision"):
        Stage2ArbitrationDecision(
            resolution_status=ResolutionStatus.UNRESOLVED,
            decision=Stage2Decision.REJECTED,
            confidence=0.6,
            rationale="The evidence is insufficient, so no decision may be forced.",
            supporting_evidence_ids=["issue-1"],
            unresolved_dimensions=["decision"],
        )

    with pytest.raises(ValidationError, match="unresolved dimensions"):
        Stage2ArbitrationDecision(
            resolution_status=ResolutionStatus.UNRESOLVED,
            rationale="The evidence is insufficient to resolve the active dimension.",
            supporting_evidence_ids=["issue-1"],
            unresolved_dimensions=[],
            missing_facts=["The active dimension still lacks direct evidence."],
        )


def test_stage3_arbitration_enforces_resolved_and_unresolved_combinations() -> None:
    resolved = Stage3ArbitrationDecision(
        resolution_status=ResolutionStatus.RESOLVED,
        symptom_label="unexpected_rejection",
        root_cause_label="missing_null_check",
        confidence=0.83,
        rationale="The exact cited code and test establish both listed dimensions.",
        supporting_evidence_ids=["code-1", "test-1"],
        resolved_dimensions=["symptom_label", "root_cause_label"],
    )
    unresolved = Stage3ArbitrationDecision(
        resolution_status=ResolutionStatus.UNRESOLVED,
        rationale="The issue text does not establish a mechanism-specific root cause.",
        supporting_evidence_ids=["issue-1"],
        unresolved_dimensions=["unsupported_root_cause_specificity"],
        missing_facts=["Code or test evidence identifying the defect mechanism."],
    )

    assert resolved.root_cause_label == "missing_null_check"
    assert unresolved.symptom_label is None
    assert unresolved.root_cause_label is None
    with pytest.raises(ValidationError, match="must contain both labels"):
        Stage3ArbitrationDecision(
            resolution_status=ResolutionStatus.RESOLVED,
            symptom_label="unexpected_rejection",
            confidence=0.83,
            rationale="A resolved Stage 3 result cannot omit its root-cause label.",
            supporting_evidence_ids=["code-1"],
            resolved_dimensions=["symptom_label"],
        )


def test_t1_roles_own_only_one_named_test() -> None:
    expected = {
        "outcome",
        "claim",
        "supporting_evidence_ids",
        "counter_evidence_ids",
    }

    assert set(FaultEvidenceAssessment.model_fields) == expected
    assert set(ScopeBoundaryAssessment.model_fields) == expected
    assert set(RepairCausalityAssessment.model_fields) == expected
    assert "decision" not in ScopeBoundaryAssessment.model_fields


def test_boundary_challenger_cannot_replace_labels() -> None:
    assert "label" not in BoundaryChallenge.model_fields
    assert "symptom_label" not in BoundaryChallenge.model_fields
    assert "root_cause_label" not in BoundaryChallenge.model_fields
    assert "decision" not in BoundaryChallenge.model_fields


def test_boundary_challenger_allows_a_bounded_request_only_for_evidence_action() -> (
    None
):
    request = EvidenceRequest(
        request_id="boundary-observation",
        missing_fact="The exact execution phase of the observed failure is unknown",
        why_needed="Execution phase separates the two nearest symptom definitions.",
        target_specialist=SpecialistType.TEST_EVIDENCE,
        target_source="frozen test result",
        query="record-1 failure execution phase",
        expected_decision_impact=(
            "The observation would determine whether symptom-boundary review can finish."
        ),
        max_items=1,
    )

    challenge = BoundaryChallenge(
        action="evidence_request",
        rationale="The frozen snapshot lacks an execution-phase observation.",
        cited_evidence_ids=["e-1"],
        evidence_request=request,
    )

    assert challenge.evidence_request == request
    with pytest.raises(ValidationError, match="allowed only"):
        BoundaryChallenge(
            action="pass",
            rationale="No boundary review is required for the proposed labels.",
            cited_evidence_ids=["e-1"],
            evidence_request=request,
        )


def _symptom(label: str = "Unexpected Output") -> SymptomReport:
    return SymptomReport(
        label=label,
        behavior_claim="A valid request returns a result different from the requested outcome.",
        supporting_evidence_ids=["issue-1"],
        alternative_label="Crash",
        boundary_reason="The request completes with an incorrect result instead of terminating.",
        boundary_evidence_ids=["taxonomy-1"],
        confidence=0.9,
        evidence_sufficiency=EvidenceSufficiency.SUFFICIENT,
    )


def _root(label: str = "API Misuse") -> RootCauseReport:
    return RootCauseReport(
        label=label,
        defect_mechanism="The caller invokes the valid API in a semantically invalid order.",
        causal_chain=[
            "The caller creates a request.",
            "The caller invokes the API too early.",
            "The response differs from the requested outcome.",
        ],
        supporting_evidence_ids=["issue-1"],
        alternative_label="Incorrect Code Logic",
        boundary_reason="The failure originates in API call ordering rather than internal control flow.",
        boundary_evidence_ids=["taxonomy-1"],
        confidence=0.88,
        evidence_sufficiency=EvidenceSufficiency.SUFFICIENT,
    )


def _legacy_stage3_report_payload() -> dict[str, object]:
    return {
        "team_id": "A",
        "symptom": _symptom().model_dump(),
        "root_cause": _root().model_dump(),
        "consistency": CausalConsistencyReport(
            status=ConsistencyStatus.CONSISTENT,
            rationale="The call ordering accounts for the observed incorrect response.",
            supporting_evidence_ids=["issue-1"],
        ).model_dump(),
    }


def test_joint_anchor_and_verification_contracts_round_trip() -> None:
    anchor = JointAnchorReport(
        symptom=_symptom("Unexpected Output"),
        root_cause=_root("API Misuse"),
        causal_account="The caller uses a valid API in a semantically invalid order.",
        shared_supporting_evidence_ids=["issue-1"],
        shared_counter_evidence_ids=[],
        shared_boundary_evidence_ids=["taxonomy-1"],
    )
    review = DimensionVerificationReport(
        dimension=EvidenceDimension.ROOT_CAUSE,
        verdict=VerificationVerdict.REJECT,
        anchor_label="API Misuse",
        alternative_label="Incorrect Code Logic",
        rationale="The cited maintainer explanation identifies internal control flow.",
        supporting_evidence_ids=["comment-1"],
        counter_evidence_ids=["issue-1"],
        corrected_claim="The library executes the wrong internal branch.",
        corrected_causal_chain=[
            "condition is miscomputed",
            "wrong branch runs",
            "output differs",
        ],
        confidence=0.86,
    )

    assert JointAnchorReport.model_validate(anchor.model_dump()) == anchor
    assert DimensionVerificationReport.model_validate(review.model_dump()) == review


def test_old_stage3_team_report_remains_readable() -> None:
    restored = Stage3TeamReport.model_validate(_legacy_stage3_report_payload())

    assert restored.anchor is None
    assert restored.verifications == ()
    assert restored.correction_audit == ()


def test_dimension_verification_reject_requires_correction_evidence_and_chain() -> None:
    with pytest.raises(ValidationError, match="REJECT"):
        DimensionVerificationReport(
            dimension=EvidenceDimension.ROOT_CAUSE,
            verdict=VerificationVerdict.REJECT,
            anchor_label="API Misuse",
            alternative_label="API Misuse",
            rationale="The alternative must identify a materially different root cause.",
            corrected_claim="A corrected causal account is required for rejected anchors.",
            corrected_causal_chain=["one", "two"],
            confidence=0.5,
        )


def test_accepted_verification_forbids_alternative_and_corrected_content() -> None:
    with pytest.raises(ValidationError, match="ACCEPT"):
        DimensionVerificationReport(
            dimension=EvidenceDimension.SYMPTOM,
            verdict=VerificationVerdict.ACCEPT,
            anchor_label="Unexpected Output",
            alternative_label="Crash",
            rationale="The available evidence supports the anchored symptom classification.",
            corrected_claim="An accepted report cannot contain correction content.",
            confidence=0.9,
        )


def test_team_composed_candidate_requires_one_review_and_audit_per_dimension() -> None:
    anchor = JointAnchorReport(
        symptom=_symptom(),
        root_cause=_root(),
        causal_account="The API is invoked before its prerequisite state is established.",
        shared_supporting_evidence_ids=["issue-1"],
    )
    review = DimensionVerificationReport(
        dimension=EvidenceDimension.SYMPTOM,
        verdict=VerificationVerdict.ACCEPT,
        anchor_label="Unexpected Output",
        rationale="The issue description directly identifies the externally visible result.",
        confidence=0.9,
    )
    audit = TeamCorrectionAudit(
        dimension=EvidenceDimension.SYMPTOM,
        anchor_label="Unexpected Output",
        final_label="Unexpected Output",
        accepted=True,
        reason="The anchored symptom remains supported by the available evidence.",
    )

    with pytest.raises(ValidationError, match="symptom and root cause"):
        TeamComposedCandidate(
            team_id="A",
            symptom=_symptom(),
            root_cause=_root(),
            anchor=anchor,
            verifications=(review, review),
            correction_audit=(audit, audit),
        )


@pytest.mark.parametrize(
    "citation_field",
    [
        "supporting_evidence_ids",
        "counter_evidence_ids",
        "boundary_evidence_ids",
    ],
)
def test_symptom_report_rejects_duplicate_citation_ids(citation_field: str) -> None:
    payload = _symptom().model_dump()
    payload[citation_field] = ["duplicate-1", "duplicate-1"]

    with pytest.raises(ValidationError, match="duplicate evidence ids"):
        SymptomReport.model_validate(payload)


@pytest.mark.parametrize(
    "citation_field",
    [
        "supporting_evidence_ids",
        "counter_evidence_ids",
        "boundary_evidence_ids",
    ],
)
def test_root_cause_report_rejects_duplicate_citation_ids(
    citation_field: str,
) -> None:
    payload = _root().model_dump()
    payload[citation_field] = ["duplicate-1", "duplicate-1"]

    with pytest.raises(ValidationError, match="duplicate evidence ids"):
        RootCauseReport.model_validate(payload)


def test_cross_raw_contracts_are_task_specific_and_noninterchangeable() -> None:
    falsification_contract = getattr(
        contracts_module, "FalsificationCoverageCrossResult", None
    )
    entailment_contract = getattr(
        contracts_module, "EntailmentMappingCrossResult", None
    )

    assert falsification_contract is not None
    assert entailment_contract is not None
    assert falsification_contract is not entailment_contract
    falsification_payload = {
        "baseline_survival_finding": {
            "condition": "execution terminates without a result",
            "status": "confirmed",
            "citation_ids": ["issue-observation"],
        },
        "proposed_defeater_findings": [
            {
                "condition": "execution terminates without a result",
                "status": "confirmed",
                "citation_ids": ["issue-observation"],
            }
        ],
        "strongest_competing_reading_finding": {
            "label": "Runtime Termination",
            "status": "confirmed",
            "citation_ids": ["issue-observation"],
        },
        "status": "consistent",
        "rationale": "Every falsification finding is confirmed against fixed evidence.",
    }
    entailment_payload = {
        "condition_findings": [
            {
                "condition": "execution completes beyond its latency budget",
                "status": "confirmed",
                "citation_ids": ["issue-observation"],
            }
        ],
        "baseline_exclusion_finding": {
            "condition": "execution completes with a result",
            "status": "confirmed",
            "citation_ids": ["issue-observation"],
        },
        "status": "consistent",
        "rationale": "Every entailment mapping is confirmed against fixed evidence.",
    }

    falsification_contract.model_validate(falsification_payload)
    entailment_contract.model_validate(entailment_payload)
    with pytest.raises(ValidationError):
        entailment_contract.model_validate(falsification_payload)
    with pytest.raises(ValidationError):
        falsification_contract.model_validate(entailment_payload)
