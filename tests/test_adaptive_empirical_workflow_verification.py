from __future__ import annotations

import hashlib

import pytest

from Benchmark.src.adaptive_empirical_workflow.contracts import (
    ArbitrationSource,
    CausalConsistencyReport,
    ConsistencyStatus,
    DimensionReadiness,
    DimensionVerificationReport,
    DisagreementMap,
    EvidenceDimension,
    EvidenceExplicitness,
    EvidenceItem,
    EvidenceReadinessReport,
    EvidenceSufficiency,
    JointAnchorReport,
    ResolutionStatus,
    RootCauseReport,
    Stage2AnalysisReport,
    Stage2ArbitrationDecision,
    Stage2Decision,
    Stage2FinalDecision,
    Stage3ArbitrationDecision,
    Stage3FinalDecision,
    Stage3TeamReport,
    SymptomReport,
    TeamCorrectionAudit,
    TestOutcome as EvidenceTestOutcome,
    VerificationVerdict,
)
from Benchmark.src.adaptive_empirical_workflow.ledger import EvidenceLedger
from Benchmark.src.adaptive_empirical_workflow.verification import (
    readiness_provenance_errors,
    stage3_report_evidence_ids,
    verify_stage2_decision,
    verify_stage3_decision,
)


def _ledger() -> EvidenceLedger:
    content = "The code diff and regression test identify the null-handling failure."
    return EvidenceLedger(
        record_id="record-1",
        task="Classify the frozen record.",
        taxonomy={
            "decision": ["accepted_fault", "rejected_candidate"],
            "symptom": ["unexpected_rejection", "wrong_output"],
            "root_cause": ["missing_null_check", "incorrect_condition"],
        },
        domain_profile="issta2024",
        initial_items=[
            EvidenceItem(
                evidence_id="code-1",
                record_id="record-1",
                source_type="code_diff",
                source_uri="https://example.test/commit/1",
                retrieved_at="2026-08-05T11:00:00Z",
                content=content,
                content_sha256=hashlib.sha256(content.encode()).hexdigest(),
                explicitness=EvidenceExplicitness.DIRECT,
            )
        ],
    )


def _readiness(*, task: str, sufficient: bool) -> EvidenceReadinessReport:
    dimensions = (
        (
            EvidenceDimension.FAULT_EXISTENCE,
            EvidenceDimension.STUDY_SCOPE,
            EvidenceDimension.REPAIR_CAUSALITY,
        )
        if task == "stage2"
        else (EvidenceDimension.SYMPTOM, EvidenceDimension.ROOT_CAUSE)
    )
    return EvidenceReadinessReport(
        task=task,
        dimensions=tuple(
            DimensionReadiness(
                dimension=dimension,
                sufficient=sufficient,
                confirmed_evidence_ids=("code-1",) if sufficient else (),
                missing_facts=(
                    ()
                    if sufficient
                    else (f"Evidence for {dimension.value} is incomplete.",)
                ),
                evidence_requests=(),
            )
            for dimension in dimensions
        ),
    )


def _stage3_team(status: ConsistencyStatus) -> Stage3TeamReport:
    return Stage3TeamReport(
        team_id="A",
        symptom=SymptomReport(
            label="unexpected_rejection",
            behavior_claim="A valid request is rejected before normal processing begins.",
            supporting_evidence_ids=["code-1"],
            alternative_label="wrong_output",
            boundary_reason="The operation is prevented rather than completed incorrectly.",
            confidence=0.9,
            evidence_sufficiency=EvidenceSufficiency.SUFFICIENT,
        ),
        root_cause=RootCauseReport(
            label="missing_null_check",
            defect_mechanism="A nullable cache entry is used without a defensive guard.",
            causal_chain=[
                "The cache returns null.",
                "The handler assumes an entry exists.",
                "The request is rejected.",
            ],
            supporting_evidence_ids=["code-1"],
            alternative_label="incorrect_condition",
            boundary_reason="The condition is valid after null entries are excluded.",
            confidence=0.9,
            evidence_sufficiency=EvidenceSufficiency.SUFFICIENT,
        ),
        consistency=CausalConsistencyReport(
            status=status,
            rationale="The proposed mechanism does not fully explain the observed behavior.",
            supporting_evidence_ids=["code-1"],
        ),
    )


def _stage2_report(team_id: str) -> Stage2AnalysisReport:
    return Stage2AnalysisReport(
        team_id=team_id,
        decision=Stage2Decision.ACCEPTED,
        confidence=0.9,
        fault_claim="The previous behavior rejects a valid runtime request.",
        repair_claim="The patch restores that request and tests the behavior.",
        supporting_evidence_ids=["code-1"],
        evidence_tests={
            "fault_existence": EvidenceTestOutcome.PASS,
            "repair_causality": EvidenceTestOutcome.PASS,
            "scope_exclusion": EvidenceTestOutcome.PASS,
        },
        alternative_hypothesis="The change could instead be an enhancement.",
        decision_boundary="Prior faulty behavior separates repair from enhancement.",
        evidence_sufficiency=EvidenceSufficiency.SUFFICIENT,
    )


def test_stage2_verification_rejects_insufficient_required_readiness() -> None:
    decision = Stage2FinalDecision(
        decision=Stage2Decision.REJECTED,
        confidence=0.8,
        rationale="The frozen evidence does not establish a prior faulty behavior.",
        supporting_evidence_ids=["code-1"],
        source=ArbitrationSource.TARGETED_ARBITRATION,
    )

    verification = verify_stage2_decision(
        decision,
        _ledger(),
        readiness=_readiness(task="stage2", sufficient=False),
    )

    assert verification.valid is False
    assert "required readiness dimension fault_existence is insufficient" in (
        verification.errors
    )


def test_stage3_verification_rejects_fallback_without_consistent_report() -> None:
    decision = Stage3FinalDecision(
        symptom_label="unexpected_rejection",
        root_cause_label="missing_null_check",
        confidence=0.8,
        rationale="The cited evidence is proposed to resolve both Stage 3 labels.",
        supporting_evidence_ids=["code-1"],
        source=ArbitrationSource.FALLBACK_UNCERTAIN,
    )

    verification = verify_stage3_decision(
        decision,
        _ledger(),
        reports=(_stage3_team(ConsistencyStatus.CAUSE_REVIEW),),
    )

    assert verification.valid is False
    assert "fallback_uncertain is not backed by a consistent team report" in (
        verification.errors
    )


def test_stage3_verification_rejects_fallback_labels_from_failed_team() -> None:
    consistent = _stage3_team(ConsistencyStatus.CONSISTENT)
    failed = _stage3_team(ConsistencyStatus.CAUSE_REVIEW).model_copy(
        update={
            "team_id": "B",
            "symptom": consistent.symptom.model_copy(update={"label": "wrong_output"}),
            "root_cause": consistent.root_cause.model_copy(
                update={"label": "incorrect_condition"}
            ),
        }
    )
    decision = Stage3FinalDecision(
        symptom_label="wrong_output",
        root_cause_label="incorrect_condition",
        confidence=0.8,
        rationale="A forged fallback selects the consistency-failed team labels.",
        supporting_evidence_ids=["code-1"],
        source=ArbitrationSource.FALLBACK_UNCERTAIN,
    )

    verification = verify_stage3_decision(
        decision,
        _ledger(),
        reports=(consistent, failed),
    )

    assert verification.valid is False
    assert "fallback_uncertain is not backed by a consistent team report" in (
        verification.errors
    )


def test_stage3_verification_rejects_fallback_evidence_from_failed_team() -> None:
    ledger = _ledger()
    content = "Evidence cited only by the consistency-failed team."
    ledger.append(
        EvidenceItem(
            evidence_id="failed-only",
            record_id="record-1",
            source_type="code_diff",
            source_uri="https://example.test/commit/failed-only",
            retrieved_at="2026-08-05T11:01:00Z",
            content=content,
            content_sha256=hashlib.sha256(content.encode()).hexdigest(),
            explicitness=EvidenceExplicitness.DIRECT,
        )
    )
    consistent = _stage3_team(ConsistencyStatus.CONSISTENT)
    failed = _stage3_team(ConsistencyStatus.CAUSE_REVIEW).model_copy(
        update={
            "team_id": "B",
            "symptom": consistent.symptom.model_copy(
                update={"supporting_evidence_ids": ["failed-only"]}
            ),
            "root_cause": consistent.root_cause.model_copy(
                update={"supporting_evidence_ids": ["failed-only"]}
            ),
            "consistency": consistent.consistency.model_copy(
                update={
                    "status": ConsistencyStatus.CAUSE_REVIEW,
                    "supporting_evidence_ids": ["failed-only"],
                }
            ),
        }
    )
    decision = Stage3FinalDecision(
        symptom_label=consistent.symptom.label,
        root_cause_label=consistent.root_cause.label,
        confidence=0.8,
        rationale="A forged fallback launders evidence from the failed team.",
        supporting_evidence_ids=["failed-only"],
        source=ArbitrationSource.FALLBACK_UNCERTAIN,
    )

    verification = verify_stage3_decision(
        decision,
        ledger,
        reports=(consistent, failed),
    )

    assert verification.valid is False
    assert "fallback_uncertain is not backed by a consistent team report" in (
        verification.errors
    )


def test_stage3_verification_treats_insufficient_readiness_as_diagnostic() -> None:
    decision = Stage3FinalDecision(
        symptom_label="unexpected_rejection",
        root_cause_label="missing_null_check",
        confidence=0.64,
        rationale="The best-supported valid candidate is preserved with uncertainty.",
        supporting_evidence_ids=["code-1"],
        source=ArbitrationSource.FALLBACK_UNCERTAIN,
    )

    verification = verify_stage3_decision(
        decision,
        _ledger(),
        readiness=_readiness(task="stage3", sufficient=False),
        reports=(_stage3_team(ConsistencyStatus.CONSISTENT),),
    )

    assert verification.valid is True
    assert verification.errors == ()


def test_verification_rejects_arbitration_beyond_listed_dimensions() -> None:
    decision = Stage3FinalDecision(
        symptom_label="unexpected_rejection",
        root_cause_label="missing_null_check",
        confidence=0.8,
        rationale="The cited evidence is proposed to resolve the symptom boundary.",
        supporting_evidence_ids=["code-1"],
        source=ArbitrationSource.TARGETED_ARBITRATION,
    )
    arbitration = Stage3ArbitrationDecision(
        resolution_status=ResolutionStatus.RESOLVED,
        symptom_label="unexpected_rejection",
        root_cause_label="missing_null_check",
        confidence=0.8,
        rationale="The response improperly claims authority over an unlisted dimension.",
        supporting_evidence_ids=["code-1"],
        resolved_dimensions=["root_cause_label"],
    )
    disagreement = DisagreementMap(
        dimensions=["symptom_label"],
        details={"symptom_label": {"team_a": "a", "team_b": "b"}},
        requires_arbitration=True,
    )

    verification = verify_stage3_decision(
        decision,
        _ledger(),
        disagreement=disagreement,
        arbitration=arbitration,
    )

    assert verification.valid is False
    assert "arbitration resolved unlisted dimension 'root_cause_label'" in (
        verification.errors
    )


def test_verification_rejects_evidence_outside_arbitration_packet() -> None:
    ledger = _ledger()
    content = "A valid but unrelated ledger item that was not cited by either report."
    ledger.append(
        EvidenceItem(
            evidence_id="unrelated-1",
            record_id="record-1",
            source_type="issue_comment",
            source_uri="https://example.test/issues/1#unrelated",
            retrieved_at="2026-08-05T11:01:00Z",
            content=content,
            content_sha256=hashlib.sha256(content.encode()).hexdigest(),
            explicitness=EvidenceExplicitness.DIRECT,
        )
    )
    decision = Stage3FinalDecision(
        symptom_label="unexpected_rejection",
        root_cause_label="missing_null_check",
        confidence=0.8,
        rationale="The arbitrator improperly relied on an out-of-packet ledger item.",
        supporting_evidence_ids=["unrelated-1"],
        source=ArbitrationSource.TARGETED_ARBITRATION,
    )

    verification = verify_stage3_decision(
        decision,
        ledger,
        arbitration_evidence_ids=("code-1",),
    )

    assert verification.valid is False
    assert "evidence_id 'unrelated-1' was not included in arbitration packet" in (
        verification.errors
    )


def test_verification_rejects_silent_change_to_unlisted_stage3_label() -> None:
    decision = Stage3FinalDecision(
        symptom_label="unexpected_rejection",
        root_cause_label="incorrect_condition",
        confidence=0.8,
        rationale="The arbitrator silently changed the undisputed root-cause label.",
        supporting_evidence_ids=["code-1"],
        source=ArbitrationSource.TARGETED_ARBITRATION,
    )
    arbitration = Stage3ArbitrationDecision(
        resolution_status=ResolutionStatus.RESOLVED,
        symptom_label="unexpected_rejection",
        root_cause_label="incorrect_condition",
        confidence=0.8,
        rationale="The response claims only symptom authority but changes root cause.",
        supporting_evidence_ids=["code-1"],
        resolved_dimensions=["symptom_label"],
    )
    disagreement = DisagreementMap(
        dimensions=["symptom_label"],
        details={"symptom_label": {"team_a": "a", "team_b": "b"}},
        requires_arbitration=True,
    )
    team_a = _stage3_team(ConsistencyStatus.CONSISTENT)
    team_b = team_a.model_copy(update={"team_id": "B"})

    verification = verify_stage3_decision(
        decision,
        _ledger(),
        reports=(team_a, team_b),
        disagreement=disagreement,
        arbitration=arbitration,
    )

    assert verification.valid is False
    assert "arbitration changed unlisted dimension 'root_cause_label'" in (
        verification.errors
    )


def test_stage2_verification_rejects_silent_unanimous_decision_flip() -> None:
    decision = Stage2FinalDecision(
        decision=Stage2Decision.REJECTED,
        confidence=0.8,
        rationale="The arbitrator silently changed the unanimous Stage 2 decision.",
        supporting_evidence_ids=["code-1"],
        source=ArbitrationSource.TARGETED_ARBITRATION,
    )
    arbitration = Stage2ArbitrationDecision(
        resolution_status=ResolutionStatus.RESOLVED,
        decision=Stage2Decision.REJECTED,
        confidence=0.8,
        rationale="The response claims evidence authority but changes the decision.",
        supporting_evidence_ids=["code-1"],
        resolved_dimensions=["evidence_sufficiency"],
    )
    disagreement = DisagreementMap(
        dimensions=["evidence_sufficiency"],
        details={"evidence_sufficiency": {"team_a": "a", "team_b": "b"}},
        requires_arbitration=True,
    )

    verification = verify_stage2_decision(
        decision,
        _ledger(),
        (_stage2_report("A"), _stage2_report("B")),
        disagreement=disagreement,
        arbitration=arbitration,
    )

    assert verification.valid is False
    assert "arbitration changed unlisted dimension 'decision'" in verification.errors


def test_stage3_consensus_gate_does_not_authorize_label_changes() -> None:
    decision = Stage3FinalDecision(
        symptom_label="wrong_output",
        root_cause_label="incorrect_condition",
        confidence=0.8,
        rationale="The arbitrator changed labels while reviewing only a consensus gate.",
        supporting_evidence_ids=["code-1"],
        source=ArbitrationSource.TARGETED_ARBITRATION,
    )
    arbitration = Stage3ArbitrationDecision(
        resolution_status=ResolutionStatus.RESOLVED,
        symptom_label="wrong_output",
        root_cause_label="incorrect_condition",
        confidence=0.8,
        rationale="The response claims gate authority while changing both labels.",
        supporting_evidence_ids=["code-1"],
        resolved_dimensions=["consensus_gate"],
    )
    disagreement = DisagreementMap(
        dimensions=["consensus_gate"],
        details={"consensus_gate": {"team_a": ["low"], "team_b": ["low"]}},
        requires_arbitration=True,
    )
    team_a = _stage3_team(ConsistencyStatus.CONSISTENT)
    team_b = team_a.model_copy(update={"team_id": "B"})

    verification = verify_stage3_decision(
        decision,
        _ledger(),
        reports=(team_a, team_b),
        disagreement=disagreement,
        arbitration=arbitration,
    )

    assert verification.valid is False
    assert "arbitration changed unlisted dimension 'symptom_label'" in (
        verification.errors
    )
    assert "arbitration changed unlisted dimension 'root_cause_label'" in (
        verification.errors
    )


def _malformed_sufficient_readiness(
    task: str,
    confirmed_ids: tuple[str, ...],
) -> EvidenceReadinessReport:
    dimensions = (
        (
            EvidenceDimension.FAULT_EXISTENCE,
            EvidenceDimension.STUDY_SCOPE,
            EvidenceDimension.REPAIR_CAUSALITY,
        )
        if task == "stage2"
        else (EvidenceDimension.SYMPTOM, EvidenceDimension.ROOT_CAUSE)
    )
    return EvidenceReadinessReport.model_construct(
        task=task,
        dimensions=tuple(
            DimensionReadiness.model_construct(
                dimension=dimension,
                sufficient=True,
                confirmed_evidence_ids=confirmed_ids,
                missing_facts=(),
                evidence_requests=(),
            )
            for dimension in dimensions
        ),
    )


def test_stage2_verification_rechecks_empty_and_unknown_readiness_ids() -> None:
    decision = Stage2FinalDecision(
        decision=Stage2Decision.ACCEPTED,
        confidence=0.8,
        rationale="The proposed decision must still pass readiness provenance checks.",
        supporting_evidence_ids=["code-1"],
        source=ArbitrationSource.DIRECT_CONSENSUS,
    )

    for confirmed_ids in ((), ("unknown-readiness",)):
        verification = verify_stage2_decision(
            decision,
            _ledger(),
            readiness=_malformed_sufficient_readiness("stage2", confirmed_ids),
        )

        assert verification.valid is False
        assert any("confirmed evidence" in error for error in verification.errors)


def test_stage3_verification_rechecks_empty_and_unknown_readiness_ids() -> None:
    decision = Stage3FinalDecision(
        symptom_label="unexpected_rejection",
        root_cause_label="missing_null_check",
        confidence=0.8,
        rationale="The proposed labels must still pass readiness provenance checks.",
        supporting_evidence_ids=["code-1"],
        source=ArbitrationSource.DIRECT_CONSENSUS,
    )

    for confirmed_ids in ((), ("unknown-readiness",)):
        verification = verify_stage3_decision(
            decision,
            _ledger(),
            readiness=_malformed_sufficient_readiness("stage3", confirmed_ids),
        )

        assert verification.valid is False
        assert any("confirmed evidence" in error for error in verification.errors)


def test_shared_risk_dimension_does_not_authorize_stage3_relabel() -> None:
    decision = Stage3FinalDecision(
        symptom_label="wrong_output",
        root_cause_label="incorrect_condition",
        confidence=0.8,
        rationale="The arbitrator silently relabeled a unanimous shared-risk result.",
        supporting_evidence_ids=["code-1"],
        source=ArbitrationSource.TARGETED_ARBITRATION,
    )
    arbitration = Stage3ArbitrationDecision(
        resolution_status=ResolutionStatus.RESOLVED,
        symptom_label="wrong_output",
        root_cause_label="incorrect_condition",
        confidence=0.8,
        rationale="The response claims shared-risk authority for both label changes.",
        supporting_evidence_ids=["code-1"],
        resolved_dimensions=["shared_unsupported_inference"],
    )
    disagreement = DisagreementMap(
        dimensions=["shared_unsupported_inference"],
        details={"shared_unsupported_inference": {"evidence_ids": ["code-1"]}},
        requires_arbitration=True,
    )
    team_a = _stage3_team(ConsistencyStatus.CONSISTENT)
    team_b = team_a.model_copy(update={"team_id": "B"})

    verification = verify_stage3_decision(
        decision,
        _ledger(),
        reports=(team_a, team_b),
        disagreement=disagreement,
        arbitration=arbitration,
    )

    assert verification.valid is False
    assert "arbitration changed unlisted dimension 'symptom_label'" in (
        verification.errors
    )
    assert "arbitration changed unlisted dimension 'root_cause_label'" in (
        verification.errors
    )


def test_verification_rejects_partial_resolved_dimension_accounting() -> None:
    decision = Stage3FinalDecision(
        symptom_label="unexpected_rejection",
        root_cause_label="missing_null_check",
        confidence=0.8,
        rationale="The response did not account for every active disagreement.",
        supporting_evidence_ids=["code-1"],
        source=ArbitrationSource.TARGETED_ARBITRATION,
    )
    arbitration = Stage3ArbitrationDecision(
        resolution_status=ResolutionStatus.RESOLVED,
        symptom_label="unexpected_rejection",
        root_cause_label="missing_null_check",
        confidence=0.8,
        rationale="The response accounts for the symptom but omits root cause.",
        supporting_evidence_ids=["code-1"],
        resolved_dimensions=["symptom_label"],
    )
    disagreement = DisagreementMap(
        dimensions=["symptom_label", "root_cause_label"],
        details={
            "symptom_label": {"team_a": "a", "team_b": "b"},
            "root_cause_label": {"team_a": "a", "team_b": "b"},
        },
        requires_arbitration=True,
    )

    verification = verify_stage3_decision(
        decision,
        _ledger(),
        disagreement=disagreement,
        arbitration=arbitration,
    )

    assert verification.valid is False
    assert "arbitration did not resolve active dimension 'root_cause_label'" in (
        verification.errors
    )


def test_stage2_verification_rejects_partial_resolved_dimension_accounting() -> None:
    decision = Stage2FinalDecision(
        decision=Stage2Decision.ACCEPTED,
        confidence=0.8,
        rationale="The response did not account for every active disagreement.",
        supporting_evidence_ids=["code-1"],
        source=ArbitrationSource.TARGETED_ARBITRATION,
    )
    arbitration = Stage2ArbitrationDecision(
        resolution_status=ResolutionStatus.RESOLVED,
        decision=Stage2Decision.ACCEPTED,
        confidence=0.8,
        rationale="The response accounts for the decision but omits evidence tests.",
        supporting_evidence_ids=["code-1"],
        resolved_dimensions=["decision"],
    )
    disagreement = DisagreementMap(
        dimensions=["decision", "evidence_tests"],
        details={
            "decision": {"team_a": "a", "team_b": "b"},
            "evidence_tests": {"team_a": "a", "team_b": "b"},
        },
        requires_arbitration=True,
    )

    verification = verify_stage2_decision(
        decision,
        _ledger(),
        disagreement=disagreement,
        arbitration=arbitration,
    )

    assert verification.valid is False
    assert "arbitration did not resolve active dimension 'evidence_tests'" in (
        verification.errors
    )


def _capability_item(
    evidence_id: str,
    source_type: str,
    *,
    metadata: dict[str, object] | None = None,
) -> EvidenceItem:
    content = f"Capability evidence {evidence_id}."
    return EvidenceItem(
        evidence_id=evidence_id,
        record_id="record-1",
        source_type=source_type,
        source_uri=f"record:record-1#{evidence_id}",
        retrieved_at="2026-08-05T11:00:00Z",
        content=content,
        content_sha256=hashlib.sha256(content.encode()).hexdigest(),
        explicitness=EvidenceExplicitness.DIRECT,
        metadata=metadata or {},
    )


def _readiness_shape_ledger() -> EvidenceLedger:
    ledger = _ledger()
    ledger.append(_capability_item("observation", "issue_body"))
    ledger.append(_capability_item("scope", "record_summary"))
    return ledger


def _shape_readiness(
    *,
    task: str,
    dimensions: tuple[EvidenceDimension, ...],
) -> EvidenceReadinessReport:
    evidence_ids = {
        EvidenceDimension.FAULT_EXISTENCE: "observation",
        EvidenceDimension.STUDY_SCOPE: "scope",
        EvidenceDimension.REPAIR_CAUSALITY: "code-1",
        EvidenceDimension.SYMPTOM: "observation",
        EvidenceDimension.ROOT_CAUSE: "code-1",
    }
    return EvidenceReadinessReport(
        task=task,
        dimensions=tuple(
            DimensionReadiness(
                dimension=dimension,
                sufficient=True,
                confirmed_evidence_ids=(evidence_ids[dimension],),
                missing_facts=(),
                evidence_requests=(),
            )
            for dimension in dimensions
        ),
    )


@pytest.mark.parametrize(
    "readiness_task,dimensions,expected_error",
    [
        (
            "stage2",
            (
                EvidenceDimension.FAULT_EXISTENCE,
                EvidenceDimension.STUDY_SCOPE,
            ),
            "readiness dimensions do not exactly match required dimensions for "
            "issta2024 stage2",
        ),
        (
            "stage2",
            (
                EvidenceDimension.FAULT_EXISTENCE,
                EvidenceDimension.STUDY_SCOPE,
                EvidenceDimension.REPAIR_CAUSALITY,
                EvidenceDimension.SYMPTOM,
            ),
            "readiness dimensions do not exactly match required dimensions for "
            "issta2024 stage2",
        ),
        (
            "stage3",
            (
                EvidenceDimension.FAULT_EXISTENCE,
                EvidenceDimension.STUDY_SCOPE,
                EvidenceDimension.REPAIR_CAUSALITY,
            ),
            "readiness task 'stage3' does not match verification stage 'stage2'",
        ),
    ],
    ids=("missing-repair", "extra-symptom", "wrong-task"),
)
def test_stage2_verification_rejects_invalid_readiness_shape(
    readiness_task: str,
    dimensions: tuple[EvidenceDimension, ...],
    expected_error: str,
) -> None:
    decision = Stage2FinalDecision(
        decision=Stage2Decision.REJECTED,
        confidence=0.8,
        rationale="The verifier must reject malformed Stage 2 readiness structure.",
        supporting_evidence_ids=["code-1"],
        source=ArbitrationSource.DIRECT_CONSENSUS,
    )

    verification = verify_stage2_decision(
        decision,
        _readiness_shape_ledger(),
        readiness=_shape_readiness(
            task=readiness_task,
            dimensions=dimensions,
        ),
    )

    assert verification.valid is False
    assert expected_error in verification.errors


@pytest.mark.parametrize(
    "readiness_task,dimensions,expected_error",
    [
        (
            "stage3",
            (EvidenceDimension.ROOT_CAUSE,),
            "readiness dimensions do not exactly match required dimensions for "
            "issta2024 stage3",
        ),
        (
            "stage3",
            (
                EvidenceDimension.SYMPTOM,
                EvidenceDimension.ROOT_CAUSE,
                EvidenceDimension.FAULT_EXISTENCE,
            ),
            "readiness dimensions do not exactly match required dimensions for "
            "issta2024 stage3",
        ),
        (
            "stage2",
            (
                EvidenceDimension.SYMPTOM,
                EvidenceDimension.ROOT_CAUSE,
            ),
            "readiness task 'stage2' does not match verification stage 'stage3'",
        ),
    ],
    ids=("missing-symptom", "extra-fault", "wrong-task"),
)
def test_stage3_verification_rejects_invalid_readiness_shape(
    readiness_task: str,
    dimensions: tuple[EvidenceDimension, ...],
    expected_error: str,
) -> None:
    decision = Stage3FinalDecision(
        symptom_label="unexpected_rejection",
        root_cause_label="missing_null_check",
        confidence=0.8,
        rationale="The verifier must reject malformed Stage 3 readiness structure.",
        supporting_evidence_ids=["code-1"],
        source=ArbitrationSource.DIRECT_CONSENSUS,
    )

    verification = verify_stage3_decision(
        decision,
        _readiness_shape_ledger(),
        readiness=_shape_readiness(
            task=readiness_task,
            dimensions=dimensions,
        ),
    )

    assert verification.valid is False
    assert expected_error in verification.errors


@pytest.mark.parametrize(
    "dimension,evidence_id",
    [
        (EvidenceDimension.FAULT_EXISTENCE, "mechanism"),
        (EvidenceDimension.STUDY_SCOPE, "mechanism"),
        (EvidenceDimension.REPAIR_CAUSALITY, "observation"),
        (EvidenceDimension.SYMPTOM, "mechanism"),
        (EvidenceDimension.ROOT_CAUSE, "observation"),
    ],
)
def test_readiness_provenance_rejects_known_wrong_dimension_evidence(
    dimension: EvidenceDimension,
    evidence_id: str,
) -> None:
    evidence_items = (
        _capability_item("observation", "issue_body"),
        _capability_item("scope", "record_summary"),
        _capability_item("mechanism", "code_diff"),
    )
    readiness = EvidenceReadinessReport(
        task=(
            "stage3"
            if dimension in {EvidenceDimension.SYMPTOM, EvidenceDimension.ROOT_CAUSE}
            else "stage2"
        ),
        dimensions=(
            DimensionReadiness(
                dimension=dimension,
                sufficient=True,
                confirmed_evidence_ids=(evidence_id,),
                missing_facts=(),
                evidence_requests=(),
            ),
        ),
    )

    errors = readiness_provenance_errors(readiness, evidence_items)

    assert errors == [
        f"required readiness dimension {dimension.value} has no "
        "capability-linked confirmed evidence"
    ]


@pytest.mark.parametrize(
    "dimension,evidence_id",
    [
        (EvidenceDimension.FAULT_EXISTENCE, "observation"),
        (EvidenceDimension.STUDY_SCOPE, "scope"),
        (EvidenceDimension.REPAIR_CAUSALITY, "mechanism"),
        (EvidenceDimension.SYMPTOM, "observation"),
        (EvidenceDimension.ROOT_CAUSE, "mechanism"),
    ],
)
def test_readiness_provenance_allows_dimension_capable_evidence(
    dimension: EvidenceDimension,
    evidence_id: str,
) -> None:
    evidence_items = (
        _capability_item("observation", "issue_body"),
        _capability_item("scope", "record_summary"),
        _capability_item("mechanism", "code_diff"),
    )
    readiness = EvidenceReadinessReport(
        task="stage3" if dimension.value in {"symptom", "root_cause"} else "stage2",
        dimensions=(
            DimensionReadiness(
                dimension=dimension,
                sufficient=True,
                confirmed_evidence_ids=(evidence_id,),
                missing_facts=(),
                evidence_requests=(),
            ),
        ),
    )

    assert readiness_provenance_errors(readiness, evidence_items) == []


def test_ase_root_cause_readiness_accepts_issue_only_context() -> None:
    evidence_items = (_capability_item("observation", "issue_comments"),)
    readiness = EvidenceReadinessReport(
        task="stage3",
        dimensions=(
            DimensionReadiness(
                dimension=EvidenceDimension.ROOT_CAUSE,
                sufficient=True,
                confirmed_evidence_ids=("observation",),
                missing_facts=(),
                evidence_requests=(),
            ),
        ),
    )

    assert (
        readiness_provenance_errors(
            readiness,
            evidence_items,
            domain="ase2022",
        )
        == []
    )


def test_issta_root_cause_readiness_still_requires_mechanism_evidence() -> None:
    evidence_items = (_capability_item("observation", "issue_comments"),)
    readiness = EvidenceReadinessReport(
        task="stage3",
        dimensions=(
            DimensionReadiness(
                dimension=EvidenceDimension.ROOT_CAUSE,
                sufficient=True,
                confirmed_evidence_ids=("observation",),
                missing_facts=(),
                evidence_requests=(),
            ),
        ),
    )

    assert readiness_provenance_errors(
        readiness,
        evidence_items,
        domain="issta2024",
    ) == [
        "required readiness dimension root_cause has no "
        "capability-linked confirmed evidence"
    ]


def test_readiness_provenance_rejects_incompatible_id_mixed_with_valid_id() -> None:
    evidence_items = (
        _capability_item("observation", "issue_body"),
        _capability_item("mechanism", "code_diff"),
    )
    readiness = EvidenceReadinessReport(
        task="stage3",
        dimensions=(
            DimensionReadiness(
                dimension=EvidenceDimension.SYMPTOM,
                sufficient=True,
                confirmed_evidence_ids=("observation", "mechanism"),
                missing_facts=(),
                evidence_requests=(),
            ),
        ),
    )

    assert readiness_provenance_errors(readiness, evidence_items) == [
        "required readiness dimension symptom cites capability-incompatible "
        "confirmed evidence_id 'mechanism'"
    ]


def test_readiness_metadata_cannot_upgrade_issue_body_to_root_mechanism() -> None:
    evidence_items = (
        _capability_item(
            "issue",
            "issue_body",
            metadata={"evidence_capabilities": ["defect_mechanism"]},
        ),
    )
    readiness = EvidenceReadinessReport(
        task="stage3",
        dimensions=(
            DimensionReadiness(
                dimension=EvidenceDimension.ROOT_CAUSE,
                sufficient=True,
                confirmed_evidence_ids=("issue",),
                missing_facts=(),
                evidence_requests=(),
            ),
        ),
    )

    assert readiness_provenance_errors(readiness, evidence_items) == [
        "required readiness dimension root_cause has no capability-linked "
        "confirmed evidence"
    ]


@pytest.mark.parametrize(
    "confirmed_ids,expected_errors",
    [
        ((), []),
        (
            ("quarantined-or-unknown",),
            [
                "required readiness dimension symptom cites unknown confirmed "
                "evidence_id 'quarantined-or-unknown'"
            ],
        ),
        (
            ("mechanism",),
            [
                "required readiness dimension symptom has no "
                "capability-linked confirmed evidence"
            ],
        ),
    ],
)
def test_insufficient_readiness_still_validates_any_supplied_citations(
    confirmed_ids: tuple[str, ...],
    expected_errors: list[str],
) -> None:
    evidence_items = (
        _capability_item("observation", "issue_body"),
        _capability_item("mechanism", "code_diff"),
    )
    readiness = EvidenceReadinessReport(
        task="stage3",
        dimensions=(
            DimensionReadiness(
                dimension=EvidenceDimension.SYMPTOM,
                sufficient=False,
                confirmed_evidence_ids=confirmed_ids,
                missing_facts=("Direct symptom confirmation is still missing.",),
                evidence_requests=(),
            ),
        ),
    )

    assert (
        readiness_provenance_errors(
            readiness,
            evidence_items,
            insufficiency_is_error=False,
        )
        == expected_errors
    )


def test_stage2_verification_rejects_known_but_wrong_readiness_linkage() -> None:
    decision = Stage2FinalDecision(
        decision=Stage2Decision.ACCEPTED,
        confidence=0.8,
        rationale="The final decision must repeat readiness capability checks.",
        supporting_evidence_ids=["code-1"],
        source=ArbitrationSource.DIRECT_CONSENSUS,
    )
    readiness = EvidenceReadinessReport(
        task="stage2",
        dimensions=tuple(
            DimensionReadiness(
                dimension=dimension,
                sufficient=True,
                confirmed_evidence_ids=("code-1",),
                missing_facts=(),
                evidence_requests=(),
            )
            for dimension in (
                EvidenceDimension.FAULT_EXISTENCE,
                EvidenceDimension.STUDY_SCOPE,
                EvidenceDimension.REPAIR_CAUSALITY,
            )
        ),
    )

    verification = verify_stage2_decision(
        decision,
        _ledger(),
        readiness=readiness,
    )

    assert verification.valid is False
    assert any("capability-linked" in error for error in verification.errors)


def test_stage3_verification_rejects_swapped_readiness_linkage() -> None:
    decision = Stage3FinalDecision(
        symptom_label="unexpected_rejection",
        root_cause_label="missing_null_check",
        confidence=0.8,
        rationale="The final labels must repeat readiness capability checks.",
        supporting_evidence_ids=["code-1"],
        source=ArbitrationSource.DIRECT_CONSENSUS,
    )
    readiness = EvidenceReadinessReport(
        task="stage3",
        dimensions=(
            DimensionReadiness(
                dimension=EvidenceDimension.SYMPTOM,
                sufficient=True,
                confirmed_evidence_ids=("code-1",),
                missing_facts=(),
                evidence_requests=(),
            ),
            DimensionReadiness(
                dimension=EvidenceDimension.ROOT_CAUSE,
                sufficient=True,
                confirmed_evidence_ids=("code-1",),
                missing_facts=(),
                evidence_requests=(),
            ),
        ),
    )

    verification = verify_stage3_decision(
        decision,
        _ledger(),
        readiness=readiness,
    )

    assert verification.valid is False
    assert any(
        "symptom" in error and "capability-linked" in error
        for error in verification.errors
    )


def _anchored_ledger(*, domain: str = "issta2024") -> EvidenceLedger:
    return EvidenceLedger(
        record_id="record-1",
        task="Classify the frozen record.",
        taxonomy={
            "symptom": ["unexpected_rejection", "wrong_output"],
            "root_cause": ["missing_null_check", "incorrect_condition"],
        },
        domain_profile=domain,
        initial_items=[
            _capability_item("symptom-1", "issue_comments"),
            _capability_item("root-1", "code_diff"),
            _capability_item("audit-symptom", "test_result"),
        ],
    )


def _dimension_verification(
    dimension: EvidenceDimension,
    evidence_id: str,
) -> DimensionVerificationReport:
    return DimensionVerificationReport(
        dimension=dimension,
        verdict=VerificationVerdict.ACCEPT,
        anchor_label=(
            "unexpected_rejection"
            if dimension is EvidenceDimension.SYMPTOM
            else "missing_null_check"
        ),
        rationale="The frozen evidence independently supports the anchor label.",
        supporting_evidence_ids=[evidence_id],
        confidence=0.84,
    )


def _correction_audit(
    dimension: EvidenceDimension,
    *,
    accepted: bool = False,
    evidence_ids: list[str] | None = None,
) -> TeamCorrectionAudit:
    label = (
        "unexpected_rejection"
        if dimension is EvidenceDimension.SYMPTOM
        else "missing_null_check"
    )
    return TeamCorrectionAudit(
        dimension=dimension,
        anchor_label=label,
        final_label=label,
        accepted=accepted,
        reason="The frozen evidence independently supports the anchor label.",
        supporting_evidence_ids=(evidence_ids or []) if accepted else [],
    )


def _anchored_stage3_team() -> Stage3TeamReport:
    symptom = _stage3_team(ConsistencyStatus.CONSISTENT).symptom.model_copy(
        update={
            "supporting_evidence_ids": ["symptom-1"],
            "counter_evidence_ids": ["symptom-1"],
            "boundary_evidence_ids": ["symptom-1"],
        }
    )
    root = _stage3_team(ConsistencyStatus.CONSISTENT).root_cause.model_copy(
        update={
            "supporting_evidence_ids": ["root-1"],
            "counter_evidence_ids": ["root-1"],
            "boundary_evidence_ids": ["root-1"],
        }
    )
    return Stage3TeamReport(
        team_id="A",
        symptom=symptom,
        root_cause=root,
        consistency=CausalConsistencyReport(
            status=ConsistencyStatus.CONSISTENT,
            rationale="The symptom and mechanism form one evidence-backed causal chain.",
            supporting_evidence_ids=["root-1"],
        ),
        anchor=JointAnchorReport(
            symptom=symptom,
            root_cause=root,
            causal_account="The missing null guard directly explains the rejected request.",
            shared_supporting_evidence_ids=["symptom-1"],
            shared_counter_evidence_ids=["root-1"],
            shared_boundary_evidence_ids=["symptom-1"],
        ),
        verifications=(
            _dimension_verification(EvidenceDimension.SYMPTOM, "symptom-1"),
            _dimension_verification(EvidenceDimension.ROOT_CAUSE, "root-1"),
        ),
        correction_audit=(
            _correction_audit(EvidenceDimension.SYMPTOM),
            _correction_audit(EvidenceDimension.ROOT_CAUSE),
        ),
    )


def _anchored_decision(report: Stage3TeamReport) -> Stage3FinalDecision:
    return Stage3FinalDecision(
        symptom_label=report.symptom.label,
        root_cause_label=report.root_cause.label,
        confidence=0.82,
        rationale="The complete anchored report supplies the final bounded decision.",
        supporting_evidence_ids=[
            report.symptom.supporting_evidence_ids[0],
            report.root_cause.supporting_evidence_ids[0],
        ],
        source=ArbitrationSource.DIRECT_CONSENSUS,
    )


def _with_anchored_citation(
    report: Stage3TeamReport,
    location: str,
    evidence_id: str,
) -> Stage3TeamReport:
    assert report.anchor is not None
    if location == "anchor_nested":
        anchor = report.anchor.model_copy(
            update={
                "symptom": report.anchor.symptom.model_copy(
                    update={"counter_evidence_ids": [evidence_id]}
                )
            }
        )
        return report.model_copy(update={"anchor": anchor})
    if location == "anchor_shared":
        return report.model_copy(
            update={
                "anchor": report.anchor.model_copy(
                    update={"shared_boundary_evidence_ids": [evidence_id]}
                )
            }
        )
    if location in {"symptom_verifier", "root_verifier"}:
        index = 0 if location == "symptom_verifier" else 1
        reviews = list(report.verifications)
        reviews[index] = reviews[index].model_copy(
            update={"counter_evidence_ids": [evidence_id]}
        )
        return report.model_copy(update={"verifications": tuple(reviews)})
    audits = list(report.correction_audit)
    audits[0] = audits[0].model_copy(
        update={"accepted": True, "supporting_evidence_ids": [evidence_id]}
    )
    return report.model_copy(update={"correction_audit": tuple(audits)})


@pytest.mark.parametrize(
    "location",
    [
        "anchor_nested",
        "anchor_shared",
        "symptom_verifier",
        "root_verifier",
        "accepted_audit",
    ],
)
def test_final_verification_rejects_unknown_anchored_provenance(
    location: str,
) -> None:
    report = _with_anchored_citation(
        _anchored_stage3_team(),
        location,
        "unknown-id",
    )

    verification = verify_stage3_decision(
        _anchored_decision(report),
        _anchored_ledger(),
        reports=(report,),
        readiness=_readiness(task="stage3", sufficient=False),
    )

    assert verification.valid is False
    assert any("unknown-id" in error for error in verification.errors)


@pytest.mark.parametrize(
    "dimension,evidence_id",
    [
        (EvidenceDimension.SYMPTOM, "root-1"),
        (EvidenceDimension.ROOT_CAUSE, "symptom-1"),
    ],
)
def test_final_verification_rejects_incapable_verifier_citation(
    dimension: EvidenceDimension,
    evidence_id: str,
) -> None:
    report = _anchored_stage3_team()
    reviews = list(report.verifications)
    index = 0 if dimension is EvidenceDimension.SYMPTOM else 1
    reviews[index] = _dimension_verification(dimension, evidence_id)
    report = report.model_copy(update={"verifications": tuple(reviews)})

    verification = verify_stage3_decision(
        _anchored_decision(report),
        _anchored_ledger(),
        reports=(report,),
    )

    assert verification.valid is False
    assert any(
        dimension.value in error and evidence_id in error
        for error in verification.errors
    )


@pytest.mark.parametrize(
    "domain,valid",
    [("ase2022", True), ("issta2024", False)],
)
def test_root_verifier_uses_domain_specific_issue_context_capability(
    domain: str,
    valid: bool,
) -> None:
    report = _anchored_stage3_team()
    reviews = list(report.verifications)
    reviews[1] = _dimension_verification(EvidenceDimension.ROOT_CAUSE, "symptom-1")
    report = report.model_copy(update={"verifications": tuple(reviews)})

    verification = verify_stage3_decision(
        _anchored_decision(report),
        _anchored_ledger(domain=domain),
        reports=(report,),
    )

    assert verification.valid is valid


@pytest.mark.parametrize(
    "field",
    ["verifications", "correction_audit"],
)
def test_anchored_report_requires_both_dimension_records(field: str) -> None:
    report = _anchored_stage3_team()
    report = report.model_copy(update={field: getattr(report, field)[:1]})

    verification = verify_stage3_decision(
        _anchored_decision(report),
        _anchored_ledger(),
        reports=(report,),
    )

    assert verification.valid is False
    assert any(field.replace("_", " ") in error for error in verification.errors)


def test_legacy_stage3_report_keeps_optional_anchored_fields_empty() -> None:
    report = _stage3_team(ConsistencyStatus.CONSISTENT)
    verification = verify_stage3_decision(
        _anchored_decision(report),
        _ledger(),
        reports=(report,),
    )

    assert report.anchor is None
    assert report.verifications == ()
    assert report.correction_audit == ()
    assert verification.valid is True


def test_dimension_filtered_provenance_includes_accepted_correction_only_for_owner() -> (
    None
):
    report = _anchored_stage3_team()
    audits = list(report.correction_audit)
    audits[0] = _correction_audit(
        EvidenceDimension.SYMPTOM,
        accepted=True,
        evidence_ids=["audit-symptom"],
    )
    report = report.model_copy(update={"correction_audit": tuple(audits)})

    symptom_ids = stage3_report_evidence_ids(
        report,
        dimensions=("symptom_label",),
    )
    root_ids = stage3_report_evidence_ids(
        report,
        dimensions=("root_cause_label",),
    )

    assert "audit-symptom" in symptom_ids
    assert "audit-symptom" not in root_ids
    assert "root-1" not in symptom_ids


def test_accepted_correction_audit_requires_supporting_provenance() -> None:
    report = _anchored_stage3_team()
    audits = list(report.correction_audit)
    audits[0] = _correction_audit(EvidenceDimension.SYMPTOM, accepted=True)
    report = report.model_copy(update={"correction_audit": tuple(audits)})

    verification = verify_stage3_decision(
        _anchored_decision(report),
        _anchored_ledger(),
        reports=(report,),
    )

    assert verification.valid is False
    assert any(
        "accepted correction audit" in error and "supporting" in error
        for error in verification.errors
    )


def _legacy_custom_source_ledger() -> EvidenceLedger:
    content = "Legacy adapters supplied this frozen, source-addressable evidence."
    return EvidenceLedger(
        record_id="record-1",
        task="Classify the frozen record.",
        taxonomy={
            "symptom": ["unexpected_rejection", "wrong_output"],
            "root_cause": ["missing_null_check", "incorrect_condition"],
        },
        domain_profile="legacy-domain",
        initial_items=[
            EvidenceItem(
                evidence_id="code-1",
                record_id="record-1",
                source_type="legacy_custom_source",
                source_uri="legacy:record-1#code-1",
                retrieved_at="2026-08-05T11:00:00Z",
                content=content,
                content_sha256=hashlib.sha256(content.encode()).hexdigest(),
                explicitness=EvidenceExplicitness.DIRECT,
            )
        ],
    )


def test_legacy_final_verification_preserves_known_custom_source_semantics() -> None:
    report = _stage3_team(ConsistencyStatus.CONSISTENT)

    verification = verify_stage3_decision(
        _anchored_decision(report),
        _legacy_custom_source_ledger(),
        reports=(report,),
    )

    assert report.anchor is None
    assert report.verifications == ()
    assert report.correction_audit == ()
    assert verification.valid is True
    assert verification.errors == ()


def test_anchored_final_verification_rejects_known_custom_source() -> None:
    legacy = _stage3_team(ConsistencyStatus.CONSISTENT)
    report = legacy.model_copy(
        update={
            "anchor": JointAnchorReport(
                symptom=legacy.symptom,
                root_cause=legacy.root_cause,
                causal_account="The legacy evidence is proposed as one anchored causal account.",
                shared_supporting_evidence_ids=["code-1"],
            ),
            "verifications": (
                _dimension_verification(EvidenceDimension.SYMPTOM, "code-1"),
                _dimension_verification(EvidenceDimension.ROOT_CAUSE, "code-1"),
            ),
            "correction_audit": (
                _correction_audit(EvidenceDimension.SYMPTOM),
                _correction_audit(EvidenceDimension.ROOT_CAUSE),
            ),
        }
    )

    verification = verify_stage3_decision(
        _anchored_decision(report),
        _legacy_custom_source_ledger(),
        reports=(report,),
    )

    assert verification.valid is False
    assert any("code-1" in error for error in verification.errors)
