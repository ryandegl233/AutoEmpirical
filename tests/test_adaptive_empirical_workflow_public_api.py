from __future__ import annotations

import hashlib
import json

from Benchmark.src import adaptive_empirical_workflow as workflow_api
from Benchmark.src.adaptive_empirical_workflow.audit import workflow_audit_json
from Benchmark.src.adaptive_empirical_workflow.contracts import (
    ArbitrationSource,
    EvidenceExplicitness,
    EvidenceItem,
    Stage2Decision,
    Stage2FinalDecision,
    Stage2Verification,
)
from Benchmark.src.adaptive_empirical_workflow.controller import (
    Stage2WorkflowResult,
)
from Benchmark.src.adaptive_empirical_workflow.ledger import EvidenceLedger
from Benchmark.src.adaptive_empirical_workflow.workflow import (
    AdaptiveEmpiricalWorkflow,
)


def _ledger() -> EvidenceLedger:
    content = "The issue confirms a bug and the patch adds a regression test."
    item = EvidenceItem(
        evidence_id="issue-1",
        record_id="record-1",
        source_type="issue",
        source_uri="https://example.test/issues/1",
        retrieved_at="2026-08-05T12:00:00Z",
        content=content,
        content_sha256=hashlib.sha256(content.encode()).hexdigest(),
        explicitness=EvidenceExplicitness.DIRECT,
    )
    return EvidenceLedger(
        record_id="record-1",
        task="Run the adaptive empirical workflow.",
        taxonomy={
            "decision": ["accepted_fault", "rejected_candidate"],
            "symptom": ["unexpected_rejection"],
            "root_cause": ["missing_null_check"],
        },
        domain_profile="ase2022",
        initial_items=[item],
    )


def test_recoverable_specialist_error_is_part_of_public_specialist_contract() -> None:
    assert issubclass(workflow_api.RecoverableSpecialistError, RuntimeError)


def test_anchor_verification_contracts_are_part_of_public_api() -> None:
    assert workflow_api.BaselineAnchor.__name__ == "BaselineAnchor"
    assert callable(workflow_api.load_baseline_anchors)
    assert workflow_api.LabelRevisionCertificate.__name__ == "LabelRevisionCertificate"
    assert callable(workflow_api.apply_baseline_preservation_gate)
    assert workflow_api.VerificationVerdict.REJECT.value == "reject"
    assert workflow_api.JointAnchorReport.__name__ == "JointAnchorReport"
    assert (
        workflow_api.DimensionVerificationReport.__name__
        == "DimensionVerificationReport"
    )
    assert workflow_api.TeamCorrectionAudit.__name__ == "TeamCorrectionAudit"
    assert workflow_api.TeamComposedCandidate.__name__ == "TeamComposedCandidate"


def _stage2_result(
    decision: Stage2Decision,
    *,
    verified: bool = True,
) -> Stage2WorkflowResult:
    final = Stage2FinalDecision(
        decision=decision,
        confidence=0.9,
        rationale="The two independent reports satisfy the configured evidence gate.",
        supporting_evidence_ids=["issue-1"],
        source=ArbitrationSource.DIRECT_CONSENSUS,
    )
    verification = Stage2Verification(
        decision=decision,
        valid=verified,
        errors=[] if verified else ["verification failed"],
    )
    return Stage2WorkflowResult(
        reports=(),  # type: ignore[arg-type]
        final_decision=final,
        verification=verification,
        evidence_deltas=(),
        retrieval_source_request_ids=[],
        budget_exhausted=False,
    )


class _Stage2Runner:
    def __init__(self, result: Stage2WorkflowResult) -> None:
        self.result = result
        self.calls = 0

    def run(self, ledger: EvidenceLedger) -> Stage2WorkflowResult:
        self.calls += 1
        return self.result


class _Stage3Runner:
    def __init__(self) -> None:
        self.calls = 0

    def run(self, ledger: EvidenceLedger) -> object:
        self.calls += 1
        return {"stage": 3}


def test_stage3_runs_only_after_verified_stage2_acceptance() -> None:
    stage2 = _Stage2Runner(_stage2_result(Stage2Decision.ACCEPTED))
    stage3 = _Stage3Runner()

    result = AdaptiveEmpiricalWorkflow(stage2=stage2, stage3=stage3).run(_ledger())

    assert stage2.calls == 1
    assert stage3.calls == 1
    assert result.stage3 == {"stage": 3}
    assert result.stop_reason is None


def test_stage2_rejection_stops_before_stage3() -> None:
    stage3 = _Stage3Runner()
    result = AdaptiveEmpiricalWorkflow(
        stage2=_Stage2Runner(_stage2_result(Stage2Decision.REJECTED)),
        stage3=stage3,
    ).run(_ledger())

    assert stage3.calls == 0
    assert result.stage3 is None
    assert result.stop_reason == "stage2_rejected_candidate"


def test_failed_verification_stops_before_stage3() -> None:
    stage3 = _Stage3Runner()
    result = AdaptiveEmpiricalWorkflow(
        stage2=_Stage2Runner(_stage2_result(Stage2Decision.ACCEPTED, verified=False)),
        stage3=stage3,
    ).run(_ledger())

    assert stage3.calls == 0
    assert result.stop_reason == "stage2_verification_failed"


def test_audit_json_preserves_exact_evidence_and_gate_outcome() -> None:
    ledger = _ledger()
    result = AdaptiveEmpiricalWorkflow(
        stage2=_Stage2Runner(_stage2_result(Stage2Decision.REJECTED)),
        stage3=_Stage3Runner(),
    ).run(ledger)

    audit = json.loads(workflow_audit_json(ledger, result))

    assert audit["record_id"] == "record-1"
    assert audit["ledger_version"] == 1
    assert (
        audit["evidence_items"][0]["content_sha256"]
        == ledger.get("issue-1").content_sha256
    )
    assert audit["stage2"]["final_decision"]["decision"] == "rejected_candidate"
    assert audit["stop_reason"] == "stage2_rejected_candidate"
