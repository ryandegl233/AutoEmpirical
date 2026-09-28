from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from Benchmark.src.adaptive_empirical_workflow.audit import (
    workflow_audit_json,
    workflow_audit_record,
)
from Benchmark.src.adaptive_empirical_workflow.contracts import (
    CausalConsistencyReport,
    DimensionVerificationReport,
    EvidenceDimension,
    EvidenceValidityReport,
    JointAnchorReport,
    RootCauseReport,
    Stage3ComponentFailure,
    Stage3TeamReport,
    SymptomReport,
    TeamCorrectionAudit,
    VerificationVerdict,
)
from Benchmark.src.adaptive_empirical_workflow.controller import Stage3WorkflowResult
from Benchmark.src.adaptive_empirical_workflow.ledger import EvidenceLedger
from Benchmark.src.adaptive_empirical_workflow.workflow import AdaptiveWorkflowResult


def _symptom(label: str = "Crash") -> SymptomReport:
    return SymptomReport(
        label=label,
        behavior_claim="A valid request terminates before normal processing completes.",
        supporting_evidence_ids=["issue-1"],
        alternative_label="Incorrect Functionality",
        boundary_reason="The process terminates instead of returning an incorrect value.",
        confidence=0.9,
        evidence_sufficiency="sufficient",
    )


def _root(label: str = "Incorrect Code Logic") -> RootCauseReport:
    return RootCauseReport(
        label=label,
        defect_mechanism="A reversed condition selects the rejecting control-flow branch.",
        causal_chain=[
            "the condition is evaluated",
            "the wrong branch is selected",
            "the valid request is rejected",
        ],
        supporting_evidence_ids=["code-1"],
        alternative_label="API Misuse",
        boundary_reason="The defect is internal branch logic rather than caller API usage.",
        confidence=0.88,
        evidence_sufficiency="sufficient",
    )


def _anchored_report() -> Stage3TeamReport:
    symptom = _symptom()
    root = _root()
    anchor = JointAnchorReport(
        symptom=symptom,
        root_cause=root,
        causal_account="The reversed branch condition directly causes the observed termination.",
        shared_supporting_evidence_ids=["issue-1", "code-1"],
    )
    return Stage3TeamReport(
        team_id="A",
        symptom=symptom,
        root_cause=root,
        consistency=CausalConsistencyReport(
            status="consistent",
            rationale="The internal branch error causally explains the observed termination.",
            supporting_evidence_ids=["code-1"],
        ),
        anchor=anchor,
        verifications=(
            DimensionVerificationReport(
                dimension=EvidenceDimension.SYMPTOM,
                verdict=VerificationVerdict.ACCEPT,
                anchor_label=symptom.label,
                rationale="The issue evidence independently supports the anchored symptom.",
                supporting_evidence_ids=["issue-1"],
                confidence=0.87,
            ),
            DimensionVerificationReport(
                dimension=EvidenceDimension.ROOT_CAUSE,
                verdict=VerificationVerdict.ACCEPT,
                anchor_label=root.label,
                rationale="The code evidence independently supports the anchored mechanism.",
                supporting_evidence_ids=["code-1"],
                confidence=0.86,
            ),
        ),
        correction_audit=(
            TeamCorrectionAudit(
                dimension=EvidenceDimension.SYMPTOM,
                anchor_label=symptom.label,
                final_label=symptom.label,
                accepted=False,
                reason="The symptom verifier preserved the anchored label.",
            ),
            TeamCorrectionAudit(
                dimension=EvidenceDimension.ROOT_CAUSE,
                anchor_label=root.label,
                final_label=root.label,
                accepted=False,
                reason="The root-cause verifier preserved the anchored label.",
            ),
        ),
    )


def _audit_record(report: Stage3TeamReport) -> dict[str, object]:
    ledger = EvidenceLedger(
        record_id="record-1",
        task="stage3",
        taxonomy={
            "symptom": ["Crash", "Incorrect Functionality"],
            "root_cause": ["Incorrect Code Logic", "API Misuse"],
        },
        domain_profile="ase2022",
    )
    stage3 = Stage3WorkflowResult(
        reports=(report,),
        final_decision=None,
        verification=None,
        evidence_deltas=(),
        retrieval_source_request_ids=[],
        budget_exhausted=False,
        validity_report=EvidenceValidityReport(
            valid_evidence_ids=(),
            capabilities_by_evidence_id={},
        ),
        component_failures=(
            Stage3ComponentFailure(
                team_id="B",
                role="joint_anchor",
                error_type="StructuredOutputError",
                message="structured output failed schema validation after bounded retries",
                schema_name="JointAnchorReport",
                validation_summary={"fields": ["symptom.label"]},
            ),
        ),
    )
    result = AdaptiveWorkflowResult(stage2=None, stage3=stage3, stop_reason=None)
    record = workflow_audit_record(ledger, result)

    assert json.loads(workflow_audit_json(ledger, result)) == record
    return record


def test_workflow_audit_round_trips_anchored_reports_without_raw_completion() -> None:
    report = _anchored_report()

    record = _audit_record(report)

    serialized = record["stage3"]["reports"][0]
    assert serialized["anchor"] == report.anchor.model_dump(mode="json")
    assert [item["dimension"] for item in serialized["verifications"]] == [
        "symptom",
        "root_cause",
    ]
    assert [item["dimension"] for item in serialized["correction_audit"]] == [
        "symptom",
        "root_cause",
    ]
    assert serialized["symptom"] == report.symptom.model_dump(mode="json")
    assert serialized["root_cause"] == report.root_cause.model_dump(mode="json")
    assert serialized["consistency"] == report.consistency.model_dump(mode="json")
    failure = record["stage3"]["component_failures"][0]
    assert set(failure) == {
        "team_id",
        "role",
        "error_type",
        "message",
        "schema_name",
        "validation_summary",
    }
    assert "raw" not in json.dumps(failure).lower()


def test_workflow_audit_keeps_legacy_reports_without_synthesizing_anchor() -> None:
    anchored = _anchored_report()
    legacy = Stage3TeamReport(
        team_id=anchored.team_id,
        symptom=anchored.symptom,
        root_cause=anchored.root_cause,
        consistency=anchored.consistency,
    )

    record = _audit_record(legacy)
    serialized = record["stage3"]["reports"][0]

    assert serialized["anchor"] is None
    assert serialized["verifications"] == []
    assert serialized["correction_audit"] == []


@pytest.mark.parametrize("field", ("verifications", "correction_audit"))
@pytest.mark.parametrize("shape", ("reversed", "partial"))
def test_anchored_report_requires_ordered_complete_dimension_records(
    field: str,
    shape: str,
) -> None:
    payload = _anchored_report().model_dump(mode="python")
    records = payload[field]
    payload[field] = list(reversed(records)) if shape == "reversed" else records[:1]

    with pytest.raises(ValidationError, match="symptom and root cause in order"):
        Stage3TeamReport.model_validate(payload)


@pytest.mark.parametrize("field", ("verifications", "correction_audit"))
def test_legacy_report_rejects_anchored_dimension_records_without_anchor(
    field: str,
) -> None:
    payload = _anchored_report().model_dump(mode="python")
    payload["anchor"] = None
    other_field = (
        "correction_audit" if field == "verifications" else "verifications"
    )
    payload[other_field] = []

    with pytest.raises(ValidationError, match="anchor is required"):
        Stage3TeamReport.model_validate(payload)
