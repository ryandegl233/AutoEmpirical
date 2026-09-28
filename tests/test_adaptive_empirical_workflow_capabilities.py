from __future__ import annotations

import hashlib

import pytest
from pydantic import ValidationError

from Benchmark.src.adaptive_empirical_workflow.capabilities import (
    AnalystRole,
    build_role_task,
    capability_for,
    required_readiness_dimensions,
)
from Benchmark.src.adaptive_empirical_workflow.contracts import (
    ConsistencyStatus,
    EvidenceExplicitness,
    EvidenceDimension,
    EvidenceItem,
    EvidenceSufficiency,
    EvidenceView,
    RootCauseReport,
    Stage2AnalysisReport,
    Stage2Decision,
    SymptomReport,
    TestOutcome as EvidenceTestOutcome,
)
from Benchmark.src.adaptive_empirical_workflow.evidence_capabilities import (
    assess_stage3_evidence_validity,
    mechanism_capable,
)


def _view() -> EvidenceView:
    content = "The runtime panics when the outgoing queue is full."
    return EvidenceView(
        record_id="record-1",
        task="Classify this confirmed container-runtime bug fix.",
        taxonomy={
            "symptom": ["Runtime Daemon Crash", "Wrong Container Behavior"],
            "root_cause": ["Improper Exception Handling", "Wrong Code Logic"],
        },
        domain_profile="issta2024",
        ledger_version=3,
        items=(
            EvidenceItem(
                evidence_id="message-1",
                record_id="record-1",
                source_type="commit_message",
                source_uri="https://example.test/commit/1",
                retrieved_at="2026-08-05T08:00:00Z",
                content=content,
                content_sha256=hashlib.sha256(content.encode()).hexdigest(),
                explicitness=EvidenceExplicitness.DIRECT,
            ),
        ),
    )


def _view_with_item(
    *,
    domain_profile: str = "issta2024",
    evidence_record_id: str = "record-1",
    source_type: str,
    metadata: dict[str, object] | None = None,
) -> EvidenceView:
    content = "The request is rejected when an entry is missing."
    return EvidenceView(
        record_id="record-1",
        task="Classify this confirmed bug fix.",
        taxonomy={
            "symptom": ["Unexpected rejection"],
            "root_cause": ["Missing null check"],
        },
        domain_profile=domain_profile,
        ledger_version=1,
        items=(
            EvidenceItem(
                evidence_id="issue-1",
                record_id=evidence_record_id,
                source_type=source_type,
                source_uri="https://example.test/issues/1",
                retrieved_at="2026-08-05T08:00:00Z",
                content=content,
                content_sha256=hashlib.sha256(content.encode()).hexdigest(),
                explicitness=EvidenceExplicitness.DIRECT,
                metadata=metadata or {},
            ),
        ),
    )


def _ase_issue_body_view() -> EvidenceView:
    return _view_with_item(
        domain_profile="ase2022",
        source_type="issue_body",
        metadata={"evidence_capabilities": ["symptom_observation"]},
    )


def test_stage3_validity_quarantines_unknown_source_even_with_metadata_capability() -> (
    None
):
    view = _view_with_item(
        source_type="invented_source",
        metadata={"evidence_capabilities": ["defect_mechanism"]},
    )

    report = assess_stage3_evidence_validity(view)

    assert report.valid_evidence_ids == ()
    assert report.quarantined[0].reasons == ("unknown_source_type",)


def test_stage3_validity_quarantines_evidence_from_another_record() -> None:
    report = assess_stage3_evidence_validity(
        _view_with_item(
            evidence_record_id="other-record",
            source_type="issue_body",
        )
    )

    assert report.valid_evidence_ids == ()
    assert report.quarantined[0].reasons == ("record_id_mismatch",)


def test_stage3_validity_keeps_ase_issue_root_context_without_upgrading_mechanism() -> (
    None
):
    report = assess_stage3_evidence_validity(_ase_issue_body_view())

    assert report.valid_evidence_ids == ("issue-1",)
    assert report.capabilities_by_evidence_id["issue-1"] == (
        "root_cause_context",
        "symptom_observation",
    )


def test_stage3_validity_metadata_cannot_upgrade_issue_body_to_defect_mechanism() -> (
    None
):
    report = assess_stage3_evidence_validity(
        _view_with_item(
            domain_profile="ase2022",
            source_type="issue_body",
            metadata={
                "evidence_capabilities": [
                    "symptom_observation",
                    "defect_mechanism",
                ]
            },
        )
    )

    assert report.capabilities_by_evidence_id["issue-1"] == (
        "root_cause_context",
        "symptom_observation",
    )


def test_downstream_capability_helper_cannot_reexpand_issue_body_mechanism() -> None:
    item = _view_with_item(
        source_type="issue_body",
        metadata={"evidence_capabilities": ["defect_mechanism"]},
    ).items[0]

    assert mechanism_capable(item) is False


def test_stage3_validity_keeps_issta_code_defect_mechanism() -> None:
    report = assess_stage3_evidence_validity(_view_with_item(source_type="code_diff"))

    assert report.capabilities_by_evidence_id["issue-1"] == (
        "defect_mechanism",
        "root_cause_context",
    )


def test_stage2_acceptance_requires_fault_and_scope_evidence() -> None:
    with pytest.raises(ValidationError, match="scope_exclusion=pass"):
        Stage2AnalysisReport(
            team_id="A",
            decision=Stage2Decision.ACCEPTED,
            confidence=0.81,
            fault_claim="The pre-fix implementation returned duplicate limits.",
            repair_claim="The change prevents pre-sized slices from being appended.",
            supporting_evidence_ids=["diff-1"],
            counter_evidence_ids=[],
            evidence_tests={
                "fault_existence": EvidenceTestOutcome.PASS,
                "repair_causality": EvidenceTestOutcome.PASS,
                "scope_exclusion": EvidenceTestOutcome.UNKNOWN,
            },
            alternative_hypothesis="The change could be a cleanup.",
            decision_boundary="Reject if no incorrect pre-fix behavior is confirmed.",
            evidence_sufficiency=EvidenceSufficiency.SUFFICIENT,
        )


def test_stage2_report_preserves_competing_hypothesis_and_boundary() -> None:
    report = Stage2AnalysisReport(
        team_id="A",
        decision=Stage2Decision.REJECTED,
        confidence=0.72,
        fault_claim="No observable incorrect behavior is established.",
        repair_claim="The patch only changes wording in an error message.",
        supporting_evidence_ids=["issue-1"],
        counter_evidence_ids=["title-1"],
        evidence_tests={
            "fault_existence": EvidenceTestOutcome.FAIL,
            "repair_causality": EvidenceTestOutcome.FAIL,
            "scope_exclusion": EvidenceTestOutcome.PASS,
        },
        alternative_hypothesis="The word fix may suggest a repair.",
        decision_boundary=(
            "Accept only if linked discussion shows the typo changed behavior."
        ),
        evidence_sufficiency=EvidenceSufficiency.SUFFICIENT,
    )

    assert report.alternative_hypothesis.startswith("The word fix")
    assert report.decision_boundary.startswith("Accept only")


def test_symptom_report_rejects_root_cause_content() -> None:
    with pytest.raises(ValidationError, match="root_cause"):
        SymptomReport.model_validate(
            {
                "label": "Runtime Daemon Crash",
                "behavior_claim": "The daemon panics when forwarding packets.",
                "supporting_evidence_ids": ["message-1"],
                "alternative_label": "Wrong Container Behavior",
                "boundary_reason": "The evidence explicitly states panic.",
                "confidence": 0.9,
                "evidence_sufficiency": "sufficient",
                "root_cause": "Improper Exception Handling",
            }
        )


def test_root_cause_report_requires_a_pre_fix_causal_chain() -> None:
    with pytest.raises(ValidationError, match="causal_chain"):
        RootCauseReport(
            label="Improper Exception Handling",
            defect_mechanism="A transient buffer error is treated as fatal.",
            causal_chain=["The patch catches the error."],
            supporting_evidence_ids=["diff-1"],
            alternative_label="Wrong Code Logic",
            boundary_reason="The defect is specifically error handling.",
            confidence=0.86,
            evidence_sufficiency=EvidenceSufficiency.SUFFICIENT,
        )


def test_role_capabilities_have_distinct_inputs_tools_and_reasoning() -> None:
    stage2 = capability_for(AnalystRole.STAGE2_FAULT_VERIFIER)
    symptom = capability_for(AnalystRole.SYMPTOM_ANALYST)
    cause = capability_for(AnalystRole.ROOT_CAUSE_ANALYST)
    checker = capability_for(AnalystRole.CAUSAL_CONSISTENCY_CHECKER)

    assert "fault_existence" in stage2.reasoning_steps
    assert "issue_pr" in stage2.allowed_specialists

    assert symptom.evidence_focus == "behavior"
    assert "root_cause_report" in symptom.forbidden_inputs
    assert "code_context" not in symptom.allowed_specialists

    assert cause.evidence_focus == "mechanism"
    assert "symptom_report" in cause.forbidden_inputs
    assert "code_context" in cause.allowed_specialists

    assert checker.may_classify is False
    assert checker.allowed_outputs == tuple(
        status.value for status in ConsistencyStatus
    )


def test_joint_anchor_and_role_owned_verifier_capabilities_are_dimension_scoped() -> None:
    anchor = capability_for(AnalystRole.JOINT_ANCHOR)
    symptom = capability_for(AnalystRole.SYMPTOM_VERIFIER)
    root_cause = capability_for(AnalystRole.ROOT_CAUSE_VERIFIER)

    assert anchor.may_classify is True
    assert anchor.allowed_outputs == ("joint_anchor_report",)
    assert symptom.may_classify is True
    assert symptom.evidence_focus == "symptom_verification"
    assert symptom.allowed_outputs == ("symptom_verification_report",)
    assert root_cause.may_classify is True
    assert root_cause.evidence_focus == "root_cause_verification"
    assert root_cause.allowed_outputs == ("root_cause_verification_report",)


def test_evidence_readiness_capability_cannot_classify_or_consume_peer_data() -> None:
    readiness = capability_for(AnalystRole.EVIDENCE_READINESS)

    assert readiness.may_classify is False
    assert readiness.allowed_outputs == ("evidence_readiness_report",)
    assert "peer_report" in readiness.forbidden_inputs
    assert "gold_annotation" in readiness.forbidden_inputs


def test_role_owned_stage2_and_boundary_capabilities_are_non_classifying() -> None:
    fault = capability_for(AnalystRole.FAULT_EVIDENCE_ANALYST)
    scope = capability_for(AnalystRole.SCOPE_BOUNDARY_ANALYST)
    repair = capability_for(AnalystRole.REPAIR_CAUSALITY_ANALYST)
    challenger = capability_for(AnalystRole.BOUNDARY_CHALLENGER)

    assert fault.reasoning_steps == ("fault_existence",)
    assert scope.reasoning_steps == ("study_scope",)
    assert repair.reasoning_steps == ("repair_causality",)
    assert {role.allowed_outputs for role in (fault, scope, repair)} == {
        ("fault_evidence_assessment",),
        ("scope_boundary_assessment",),
        ("repair_causality_assessment",),
    }
    assert all(role.may_classify is False for role in (fault, scope, repair))
    assert challenger.may_classify is False
    assert challenger.allowed_outputs == (
        "pass",
        "symptom_review",
        "cause_review",
        "evidence_request",
    )


@pytest.mark.parametrize(
    ("domain", "task", "expected"),
    [
        (
            "ase2022",
            "stage2",
            (EvidenceDimension.FAULT_EXISTENCE, EvidenceDimension.STUDY_SCOPE),
        ),
        (
            "issta2024",
            "stage2",
            (
                EvidenceDimension.FAULT_EXISTENCE,
                EvidenceDimension.STUDY_SCOPE,
                EvidenceDimension.REPAIR_CAUSALITY,
            ),
        ),
        (
            "ase2022",
            "stage3",
            (EvidenceDimension.SYMPTOM, EvidenceDimension.ROOT_CAUSE),
        ),
    ],
)
def test_required_readiness_dimensions_match_the_task_policy(
    domain: str,
    task: str,
    expected: tuple[EvidenceDimension, ...],
) -> None:
    assert required_readiness_dimensions(domain, task) == expected


def test_required_readiness_dimensions_rejects_unsupported_tasks() -> None:
    with pytest.raises(ValueError, match="unsupported readiness task 'stage9'"):
        required_readiness_dimensions("ase2022", "stage9")


def test_role_task_keeps_exact_ledger_evidence_while_setting_role_focus() -> None:
    task = build_role_task(
        AnalystRole.SYMPTOM_ANALYST,
        team_id="A",
        evidence_view=_view(),
    )

    assert task.team_id == "A"
    assert task.capability.evidence_focus == "behavior"
    assert task.evidence_view.ledger_version == 3
    assert task.evidence_view.items[0].content == (
        "The runtime panics when the outgoing queue is full."
    )
    assert "peer_report" not in task.model_dump()
