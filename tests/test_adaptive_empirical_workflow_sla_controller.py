from __future__ import annotations

import hashlib

import pytest

from Benchmark.src.adaptive_empirical_workflow.agents import (
    ModelTransportError,
    StructuredOutputError,
)
from Benchmark.src.adaptive_empirical_workflow.contracts import (
    ArbitrationSource,
    BaselineAnchor,
    EvidenceDimension,
    EvidenceExplicitness,
    EvidenceItem,
    EvidenceSufficiency,
    EvidenceView,
    RootCauseReport,
    SlaConditionalArbitration,
    SlaDimensionArbitration,
    SlaDimensionVerification,
    SlaJointDiagnosis,
    SlaJointVerification,
    SlaVerifierVerdict,
    SymptomReport,
    validate_sla_arbitration_choice,
    validate_sla_conditional_arbitration,
    validate_sla_joint_verification,
)
from Benchmark.src.adaptive_empirical_workflow.sla_budget import (
    GlobalSlaBudget,
    SlaBudgetConfig,
    SlaBudgetExhausted,
)
from Benchmark.src.adaptive_empirical_workflow.sla_controller import (
    SlaDecisionStatus,
    TargetedSlaController,
)


def _item(evidence_id: str, source_type: str) -> EvidenceItem:
    content = f"Evidence for {evidence_id}."
    return EvidenceItem(
        evidence_id=evidence_id,
        record_id="record-1",
        source_type=source_type,
        source_uri=f"https://example.test/{evidence_id}",
        retrieved_at="2026-08-22T00:00:00Z",
        content=content,
        content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        explicitness=EvidenceExplicitness.DIRECT,
    )


def evidence_view(*ids: str) -> EvidenceView:
    known = ids or ("observation", "mechanism")
    source_types = {
        "observation": "issue_body",
        "mechanism": "code_diff",
    }
    return EvidenceView(
        record_id="record-1",
        task="Classify the supplied issue.",
        taxonomy={
            "symptom": ["Crash", "Incorrect Functionality", "Poor Performance"],
            "root_cause": ["API Misuse", "Incorrect Code Logic"],
        },
        domain_profile="ase2022",
        ledger_version=1,
        items=tuple(
            _item(item_id, source_types.get(item_id, "issue_body")) for item_id in known
        ),
    )


def evidence_view_with_items(items: tuple[EvidenceItem, ...]) -> EvidenceView:
    return EvidenceView(
        record_id="record-1",
        task="Classify the supplied issue.",
        taxonomy={
            "symptom": ["Crash", "Incorrect Functionality", "Poor Performance"],
            "root_cause": ["API Misuse", "Incorrect Code Logic"],
        },
        domain_profile="ase2022",
        ledger_version=1,
        items=items,
    )


def anchor() -> BaselineAnchor:
    return BaselineAnchor(
        record_id="record-1",
        valid=True,
        symptom_label="Crash",
        root_cause_label="Incorrect Code Logic",
        source_config_hash="a" * 64,
        source_predictions_sha256="b" * 64,
    )


def diagnosis() -> SlaJointDiagnosis:
    return SlaJointDiagnosis(
        symptom=SymptomReport(
            label="Crash",
            behavior_claim="The request terminates instead of completing with a result.",
            supporting_evidence_ids=("observation",),
            alternative_label="Incorrect Functionality",
            boundary_reason="The observed termination separates this from an incorrect output.",
            confidence=0.9,
            evidence_sufficiency=EvidenceSufficiency.SUFFICIENT,
        ),
        root_cause=RootCauseReport(
            label="API Misuse",
            defect_mechanism="The caller invokes an API before its required state exists.",
            causal_chain=(
                "The caller starts the request.",
                "The caller invokes the API too early.",
                "The request terminates without a result.",
            ),
            supporting_evidence_ids=("mechanism",),
            alternative_label="Incorrect Code Logic",
            boundary_reason="The failure is caused by invalid API ordering rather than internal logic.",
            confidence=0.9,
            evidence_sufficiency=EvidenceSufficiency.SUFFICIENT,
        ),
        symptom_matches_baseline=True,
        root_cause_matches_baseline=False,
    )


def verification(
    symptom: SlaVerifierVerdict = SlaVerifierVerdict.ACCEPT_CANDIDATE,
    root_cause: SlaVerifierVerdict = SlaVerifierVerdict.ACCEPT_CANDIDATE,
) -> SlaJointVerification:
    def dimension_result(
        dimension: EvidenceDimension,
        verdict: SlaVerifierVerdict,
        evidence_id: str,
        rationale: str,
    ) -> SlaDimensionVerification:
        return SlaDimensionVerification(
            dimension=dimension,
            verdict=verdict,
            supporting_evidence_ids=(
                (evidence_id,) if verdict is SlaVerifierVerdict.ACCEPT_CANDIDATE else ()
            ),
            counter_evidence_ids=(
                () if verdict is SlaVerifierVerdict.ACCEPT_CANDIDATE else (evidence_id,)
            ),
            rationale=rationale,
        )

    return SlaJointVerification(
        symptom=dimension_result(
            EvidenceDimension.SYMPTOM,
            symptom,
            "observation",
            "The observed termination supports the candidate symptom.",
        ),
        root_cause=dimension_result(
            EvidenceDimension.ROOT_CAUSE,
            root_cause,
            "mechanism",
            "The code change supports the candidate root cause.",
        ),
    )


def budget() -> object:
    return GlobalSlaBudget(SlaBudgetConfig()).start_record()


class CountingSlaAgents:
    def __init__(
        self,
        *,
        review: SlaJointVerification | None = None,
        diagnosis_error: Exception | None = None,
        verifier_error: Exception | None = None,
    ) -> None:
        self.roles: list[str] = []
        self._review = review or verification()
        self._diagnosis_error = diagnosis_error
        self._verifier_error = verifier_error

    def sla_joint_diagnosis(
        self, view: EvidenceView, baseline: BaselineAnchor
    ) -> SlaJointDiagnosis:
        self.roles.append("sla_joint_diagnosis")
        if self._diagnosis_error is not None:
            raise self._diagnosis_error
        return diagnosis()

    def sla_joint_verifier(
        self,
        view: EvidenceView,
        baseline: BaselineAnchor,
        candidate: SlaJointDiagnosis,
    ) -> SlaJointVerification:
        self.roles.append("sla_joint_verifier")
        if self._verifier_error is not None:
            raise self._verifier_error
        return self._review

    def sla_conditional_arbitrator(
        self,
        view: EvidenceView,
        baseline: BaselineAnchor,
        candidate: SlaJointDiagnosis,
        review: SlaJointVerification,
    ) -> SlaConditionalArbitration:
        self.roles.append("sla_conditional_arbitrator")
        return SlaConditionalArbitration(
            root_cause=SlaDimensionArbitration(
                dimension=EvidenceDimension.ROOT_CAUSE,
                selected_label="API Misuse",
                supporting_evidence_ids=("mechanism",),
                rationale="The fixed root-cause pair is resolved by code evidence.",
            )
        )


class TelemetryCountingSlaAgents(CountingSlaAgents):
    def __init__(self) -> None:
        super().__init__()
        self.telemetry_record_ids: list[str | None] = []

    def telemetry(self, *, record_id: str | None = None) -> dict[str, object]:
        self.telemetry_record_ids.append(record_id)
        return {
            "calls": [
                {
                    "record_id": record_id,
                    "role": "sla_joint_diagnosis",
                    "latency_seconds": 0.25,
                    "attempts": 2,
                    "network_attempts": 3,
                    "network_attempt_details": [
                        {
                            "status": "retryable_error",
                            "retryable": True,
                            "cause_type": "timeout_error",
                            "http_status": None,
                            "retry_after_seconds": None,
                        }
                    ],
                }
            ]
        }


def test_sla_verifier_rejects_unknown_dimension_citations() -> None:
    view = evidence_view("known")
    report = verification()

    with pytest.raises(ValueError, match="exact evidence view"):
        validate_sla_joint_verification(
            report.model_copy(
                update={
                    "symptom": report.symptom.model_copy(
                        update={"supporting_evidence_ids": ("unknown",)}
                    )
                }
            ),
            view,
        )


def test_sla_arbitrator_cannot_invent_a_third_label() -> None:
    with pytest.raises(ValueError, match="fixed candidate pair"):
        validate_sla_arbitration_choice(
            choice="Poor Performance",
            candidate="Crash",
            baseline="Incorrect Functionality",
        )


@pytest.mark.parametrize(
    ("verdict", "field_name"),
    (
        (SlaVerifierVerdict.ACCEPT_CANDIDATE, "supporting_evidence_ids"),
        (SlaVerifierVerdict.PRESERVE_BASELINE, "counter_evidence_ids"),
        (SlaVerifierVerdict.UNRESOLVED, "counter_evidence_ids"),
    ),
)
def test_sla_verifier_requires_verdict_appropriate_dimension_evidence(
    verdict: SlaVerifierVerdict,
    field_name: str,
) -> None:
    report = verification(symptom=verdict)
    empty = report.symptom.model_copy(update={field_name: ()})

    with pytest.raises(ValueError, match="verdict-appropriate"):
        validate_sla_joint_verification(
            report.model_copy(update={"symptom": empty}), evidence_view()
        )


def test_single_dimension_arbitration_requires_total_owned_evidence() -> None:
    report = SlaConditionalArbitration(
        root_cause=SlaDimensionArbitration(
            dimension=EvidenceDimension.ROOT_CAUSE,
            selected_label="API Misuse",
            supporting_evidence_ids=("mechanism",),
            rationale="The mechanism evidence resolves the root-cause conflict.",
        )
    )

    validate_sla_conditional_arbitration(
        report,
        view=evidence_view(),
        diagnosis=diagnosis(),
        verification=verification(root_cause=SlaVerifierVerdict.PRESERVE_BASELINE),
        baseline=anchor(),
    )

    with pytest.raises(ValueError, match="exactly the conflicting dimensions"):
        validate_sla_conditional_arbitration(
            SlaConditionalArbitration(
                symptom=SlaDimensionArbitration(
                    dimension=EvidenceDimension.SYMPTOM,
                    selected_label="Crash",
                    supporting_evidence_ids=("observation",),
                    rationale="The observation supports the symptom choice.",
                )
            ),
            view=evidence_view(),
            diagnosis=diagnosis(),
            verification=verification(root_cause=SlaVerifierVerdict.PRESERVE_BASELINE),
            baseline=anchor(),
        )


def test_matching_diagnosis_and_verifier_use_exactly_two_calls() -> None:
    calls = CountingSlaAgents()

    result = TargetedSlaController(calls).run(evidence_view(), anchor(), budget())

    assert calls.roles == ["sla_joint_diagnosis", "sla_joint_verifier"]
    assert result.call_count == 2
    assert result.final_decision.root_cause_label == "API Misuse"
    assert (
        result.decision_status[EvidenceDimension.ROOT_CAUSE]
        is SlaDecisionStatus.MODEL_VERIFIED
    )


def test_controller_returns_record_scoped_attempt_and_provider_error_audit() -> None:
    agents = TelemetryCountingSlaAgents()

    result = TargetedSlaController(agents).run(evidence_view(), anchor(), budget())

    assert agents.telemetry_record_ids == ["record-1"]
    assert len(result.call_audit) == 1
    audit = result.call_audit[0]
    assert audit.role == "sla_joint_diagnosis"
    assert audit.schema_attempts == 2
    assert audit.network_attempts == 3
    assert audit.provider_errors[0].cause_type == "timeout_error"


def test_provider_error_audit_retains_terminal_transport_failure_without_attempt_detail() -> (
    None
):
    errors = TargetedSlaController._provider_errors(
        {
            "network_attempt_details": [],
            "transport_failures": [
                {
                    "retryable": False,
                    "cause_type": "remote_disconnected",
                    "http_status": None,
                    "retry_after_seconds": None,
                }
            ],
        }
    )

    assert errors[0].status == "transport_error"
    assert errors[0].cause_type == "remote_disconnected"


def test_conflict_uses_one_arbitrator_and_never_a_fourth_call() -> None:
    calls = CountingSlaAgents(
        review=verification(root_cause=SlaVerifierVerdict.PRESERVE_BASELINE)
    )

    result = TargetedSlaController(calls).run(evidence_view(), anchor(), budget())

    assert calls.roles == [
        "sla_joint_diagnosis",
        "sla_joint_verifier",
        "sla_conditional_arbitrator",
    ]
    assert result.call_count == 3
    assert (
        result.decision_status[EvidenceDimension.ROOT_CAUSE]
        is SlaDecisionStatus.ARBITRATED
    )


def test_unresolved_dimension_falls_back_without_overriding_accepted_dimension() -> (
    None
):
    calls = CountingSlaAgents(
        review=verification(root_cause=SlaVerifierVerdict.UNRESOLVED)
    )

    result = TargetedSlaController(calls).run(evidence_view(), anchor(), budget())

    assert result.final_decision.symptom_label == "Crash"
    assert result.final_decision.root_cause_label == "Incorrect Code Logic"
    assert (
        result.decision_status[EvidenceDimension.SYMPTOM]
        is SlaDecisionStatus.BASELINE_AGREEMENT
    )
    assert (
        result.decision_status[EvidenceDimension.ROOT_CAUSE]
        is SlaDecisionStatus.UNRESOLVED_FALLBACK
    )
    assert result.call_count == 2


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (
            ModelTransportError("network unavailable"),
            SlaDecisionStatus.TRANSPORT_FALLBACK,
        ),
        (StructuredOutputError("schema invalid"), SlaDecisionStatus.SCHEMA_FALLBACK),
    ],
)
def test_recoverable_role_failure_preserves_both_baseline_dimensions(
    error: Exception, status: SlaDecisionStatus
) -> None:
    calls = CountingSlaAgents(verifier_error=error)

    result = TargetedSlaController(calls).run(evidence_view(), anchor(), budget())

    assert result.final_decision.symptom_label == "Crash"
    assert result.final_decision.root_cause_label == "Incorrect Code Logic"
    assert set(result.decision_status.values()) == {status}
    assert result.fallback is True


def test_budget_exhaustion_before_verifier_never_starts_verifier() -> None:
    class ExhaustedBudget:
        def require_request_budget(self) -> float:
            raise SlaBudgetExhausted("no time left")

        def remaining_seconds(self) -> float:
            return 0.0

    calls = CountingSlaAgents()

    result = TargetedSlaController(calls).run(
        evidence_view(), anchor(), ExhaustedBudget()
    )

    assert calls.roles == []
    assert result.call_count == 0
    assert set(result.decision_status.values()) == {SlaDecisionStatus.BUDGET_FALLBACK}


@pytest.mark.parametrize(
    "view",
    (
        evidence_view_with_items(()),
        evidence_view_with_items((_item("bad", "unrecognized_source"),)),
    ),
)
def test_transport_fallback_without_valid_evidence_has_no_citations(
    view: EvidenceView,
) -> None:
    calls = CountingSlaAgents(
        diagnosis_error=ModelTransportError("network unavailable")
    )

    result = TargetedSlaController(calls).run(view, anchor(), budget())

    assert result.final_decision.supporting_evidence_ids == ()
    assert result.final_decision.confidence is None
    assert result.final_decision.source is ArbitrationSource.SLA_FALLBACK
    assert set(result.decision_status.values()) == {
        SlaDecisionStatus.TRANSPORT_FALLBACK
    }


def test_transport_fallback_cites_only_valid_capable_evidence() -> None:
    view = evidence_view_with_items(
        (
            _item("bad", "unrecognized_source"),
            _item("observation", "issue_body"),
        )
    )
    calls = CountingSlaAgents(
        diagnosis_error=ModelTransportError("network unavailable")
    )

    result = TargetedSlaController(calls).run(view, anchor(), budget())

    assert result.final_decision.supporting_evidence_ids == ("observation",)
