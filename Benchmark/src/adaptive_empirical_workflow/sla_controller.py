"""Isolated two-call targeted SLA controller with typed per-dimension fallback."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Protocol

from .agents import ModelTransportError, StructuredOutputError, record_telemetry_scope
from .contracts import (
    ArbitrationSource,
    BaselineAnchor,
    EvidenceDimension,
    EvidenceView,
    SlaDecisionStatus,
    SlaConditionalArbitration,
    SlaJointDiagnosis,
    SlaJointVerification,
    SlaVerifierVerdict,
    Stage3FinalDecision,
    validate_sla_conditional_arbitration,
    validate_sla_joint_diagnosis,
    validate_sla_joint_verification,
)
from .evidence_capabilities import (
    assess_stage3_evidence_validity,
    supports_readiness_dimension,
)
from .sla_budget import RecordSlaBudget, SlaBudgetExhausted


class TargetedSlaAgents(Protocol):
    def sla_joint_diagnosis(
        self, view: EvidenceView, anchor: BaselineAnchor
    ) -> SlaJointDiagnosis: ...

    def sla_joint_verifier(
        self,
        view: EvidenceView,
        anchor: BaselineAnchor,
        diagnosis: SlaJointDiagnosis,
    ) -> SlaJointVerification: ...

    def sla_conditional_arbitrator(
        self,
        view: EvidenceView,
        anchor: BaselineAnchor,
        diagnosis: SlaJointDiagnosis,
        verification: SlaJointVerification,
    ) -> SlaConditionalArbitration: ...

    def telemetry(self, *, record_id: str | None = None) -> dict[str, object]: ...


@dataclass(frozen=True)
class SlaProviderErrorAudit:
    status: str
    retryable: bool
    cause_type: str | None
    http_status: int | None
    retry_after_seconds: float | None


@dataclass(frozen=True)
class SlaCallAudit:
    role: str
    latency_seconds: float
    schema_attempts: int
    network_attempts: int
    provider_errors: tuple[SlaProviderErrorAudit, ...]


@dataclass(frozen=True)
class TargetedSlaResult:
    final_decision: Stage3FinalDecision
    decision_status: Mapping[EvidenceDimension, SlaDecisionStatus]
    diagnosis: SlaJointDiagnosis | None
    verification: SlaJointVerification | None
    arbitration: SlaConditionalArbitration | None
    call_count: int
    fallback: bool
    budget_remaining_seconds: float
    call_audit: tuple[SlaCallAudit, ...] = ()


_DIMENSIONS = (EvidenceDimension.SYMPTOM, EvidenceDimension.ROOT_CAUSE)
_FALLBACK_STATUSES = frozenset(
    {
        SlaDecisionStatus.TRANSPORT_FALLBACK,
        SlaDecisionStatus.SCHEMA_FALLBACK,
        SlaDecisionStatus.BUDGET_FALLBACK,
        SlaDecisionStatus.UNRESOLVED_FALLBACK,
    }
)


class TargetedSlaController:
    """Run the bounded targeted SLA call graph without Stage3Controller coupling."""

    def __init__(self, agents: TargetedSlaAgents) -> None:
        self._agents = agents

    def run(
        self,
        view: EvidenceView,
        anchor: BaselineAnchor,
        budget: RecordSlaBudget,
    ) -> TargetedSlaResult:
        with record_telemetry_scope(view.record_id):
            result = self._run(view, anchor, budget)
            return replace(result, call_audit=self._call_audit(view.record_id))

    def _run(
        self,
        view: EvidenceView,
        anchor: BaselineAnchor,
        budget: RecordSlaBudget,
    ) -> TargetedSlaResult:
        self._validate_inputs(view, anchor)
        calls = 0
        diagnosis: SlaJointDiagnosis | None = None
        verification: SlaJointVerification | None = None
        arbitration: SlaConditionalArbitration | None = None
        try:
            budget.require_request_budget()
            calls += 1
            diagnosis = self._agents.sla_joint_diagnosis(view, anchor)
            validate_sla_joint_diagnosis(diagnosis, view, anchor)
            budget.require_request_budget()
            calls += 1
            verification = self._agents.sla_joint_verifier(view, anchor, diagnosis)
            validate_sla_joint_verification(verification, view)
        except (
            ModelTransportError,
            StructuredOutputError,
            SlaBudgetExhausted,
        ) as error:
            return self._fallback(
                view=view,
                anchor=anchor,
                status=self._status_for(error),
                diagnosis=diagnosis,
                verification=verification,
                arbitration=None,
                call_count=calls,
                budget=budget,
            )

        assert diagnosis is not None and verification is not None
        labels, statuses, conflicts = self._resolve_verification(
            diagnosis, verification, anchor
        )
        if conflicts:
            try:
                budget.require_request_budget()
                calls += 1
                arbitration = self._agents.sla_conditional_arbitrator(
                    view, anchor, diagnosis, verification
                )
                validate_sla_conditional_arbitration(
                    arbitration,
                    view=view,
                    diagnosis=diagnosis,
                    verification=verification,
                    baseline=anchor,
                )
            except (
                ModelTransportError,
                StructuredOutputError,
                SlaBudgetExhausted,
            ) as error:
                for dimension in conflicts:
                    labels[dimension] = self._baseline_label(anchor, dimension)
                    statuses[dimension] = self._status_for(error)
            else:
                for dimension in conflicts:
                    labels[dimension] = self._arbitration_label(arbitration, dimension)
                    statuses[dimension] = SlaDecisionStatus.ARBITRATED

        citations = self._decision_citations(view, diagnosis, verification, arbitration)
        result = TargetedSlaResult(
            final_decision=Stage3FinalDecision(
                symptom_label=labels[EvidenceDimension.SYMPTOM],
                root_cause_label=labels[EvidenceDimension.ROOT_CAUSE],
                confidence=0.9,
                rationale="The targeted SLA controller applied independently validated dimension decisions.",
                supporting_evidence_ids=citations,
                source=(
                    ArbitrationSource.TARGETED_ARBITRATION
                    if arbitration is not None
                    else ArbitrationSource.DIRECT_CONSENSUS
                ),
            ),
            decision_status=statuses,
            diagnosis=diagnosis,
            verification=verification,
            arbitration=arbitration,
            call_count=calls,
            fallback=any(status in _FALLBACK_STATUSES for status in statuses.values()),
            budget_remaining_seconds=budget.remaining_seconds(),
        )
        assert result.call_count <= 3
        return result

    def _call_audit(self, record_id: str) -> tuple[SlaCallAudit, ...]:
        telemetry = getattr(self._agents, "telemetry", None)
        if not callable(telemetry):
            return ()
        payload = telemetry(record_id=record_id)
        calls = payload.get("calls") if isinstance(payload, dict) else None
        if not isinstance(calls, list):
            return ()
        return tuple(
            audit
            for call in calls
            if isinstance(call, dict)
            and (audit := self._call_audit_entry(call)) is not None
        )

    @staticmethod
    def _call_audit_entry(call: Mapping[str, object]) -> SlaCallAudit | None:
        role = call.get("role")
        latency = call.get("latency_seconds")
        attempts = call.get("attempts")
        network_attempts = call.get("network_attempts")
        if (
            not isinstance(role, str)
            or not role.startswith("sla_")
            or isinstance(latency, bool)
            or not isinstance(latency, (int, float))
            or isinstance(attempts, bool)
            or not isinstance(attempts, int)
            or attempts < 1
            or isinstance(network_attempts, bool)
            or not isinstance(network_attempts, int)
            or network_attempts < 0
        ):
            return None
        return SlaCallAudit(
            role=role,
            latency_seconds=float(latency),
            schema_attempts=attempts,
            network_attempts=network_attempts,
            provider_errors=TargetedSlaController._provider_errors(call),
        )

    @staticmethod
    def _provider_errors(
        call: Mapping[str, object],
    ) -> tuple[SlaProviderErrorAudit, ...]:
        raw_errors = call.get("network_attempt_details")
        if not isinstance(raw_errors, list) or not raw_errors:
            raw_errors = call.get("transport_failures")
        if not isinstance(raw_errors, list):
            return ()
        errors: list[SlaProviderErrorAudit] = []
        for raw_error in raw_errors:
            if not isinstance(raw_error, Mapping):
                continue
            status = raw_error.get("status")
            if status == "success":
                continue
            cause_type = raw_error.get("cause_type")
            http_status = raw_error.get("http_status")
            retry_after = raw_error.get("retry_after_seconds")
            errors.append(
                SlaProviderErrorAudit(
                    status=status if isinstance(status, str) else "transport_error",
                    retryable=bool(raw_error.get("retryable", False)),
                    cause_type=cause_type if isinstance(cause_type, str) else None,
                    http_status=(
                        http_status
                        if isinstance(http_status, int)
                        and not isinstance(http_status, bool)
                        else None
                    ),
                    retry_after_seconds=(
                        float(retry_after)
                        if isinstance(retry_after, (int, float))
                        and not isinstance(retry_after, bool)
                        else None
                    ),
                )
            )
        return tuple(errors)

    @staticmethod
    def _validate_inputs(view: EvidenceView, anchor: BaselineAnchor) -> None:
        if (
            not anchor.valid
            or anchor.symptom_label is None
            or anchor.root_cause_label is None
        ):
            raise ValueError("targeted SLA controller requires a valid Baseline anchor")
        if anchor.record_id != view.record_id:
            raise ValueError("Baseline anchor record_id must match the evidence view")
        for dimension, label in (
            ("symptom", anchor.symptom_label),
            ("root_cause", anchor.root_cause_label),
        ):
            if label not in view.taxonomy.get(dimension, ()):
                raise ValueError(
                    f"Baseline {dimension} label is outside the evidence taxonomy"
                )

    @staticmethod
    def _status_for(error: Exception) -> SlaDecisionStatus:
        if isinstance(error, ModelTransportError):
            return SlaDecisionStatus.TRANSPORT_FALLBACK
        if isinstance(error, StructuredOutputError):
            return SlaDecisionStatus.SCHEMA_FALLBACK
        return SlaDecisionStatus.BUDGET_FALLBACK

    def _fallback(
        self,
        *,
        view: EvidenceView,
        anchor: BaselineAnchor,
        status: SlaDecisionStatus,
        diagnosis: SlaJointDiagnosis | None,
        verification: SlaJointVerification | None,
        arbitration: SlaConditionalArbitration | None,
        call_count: int,
        budget: RecordSlaBudget,
    ) -> TargetedSlaResult:
        citations = self._fallback_citations(view)
        no_valid_evidence = not citations
        return TargetedSlaResult(
            final_decision=Stage3FinalDecision(
                symptom_label=anchor.symptom_label,
                root_cause_label=anchor.root_cause_label,
                confidence=None if no_valid_evidence else 0.0,
                rationale=(
                    "The targeted SLA controller preserved Baseline labels after a typed recoverable failure."
                    if not no_valid_evidence
                    else "The targeted SLA controller preserved Baseline labels because no valid evidence was available after a typed recoverable failure."
                ),
                supporting_evidence_ids=citations,
                source=(
                    ArbitrationSource.FALLBACK_UNCERTAIN
                    if not no_valid_evidence
                    else ArbitrationSource.SLA_FALLBACK
                ),
            ),
            decision_status={dimension: status for dimension in _DIMENSIONS},
            diagnosis=diagnosis,
            verification=verification,
            arbitration=arbitration,
            call_count=call_count,
            fallback=True,
            budget_remaining_seconds=budget.remaining_seconds(),
        )

    @staticmethod
    def _baseline_label(anchor: BaselineAnchor, dimension: EvidenceDimension) -> str:
        label = (
            anchor.symptom_label
            if dimension is EvidenceDimension.SYMPTOM
            else anchor.root_cause_label
        )
        assert label is not None
        return label

    def _resolve_verification(
        self,
        diagnosis: SlaJointDiagnosis,
        verification: SlaJointVerification,
        anchor: BaselineAnchor,
    ) -> tuple[
        dict[EvidenceDimension, str],
        dict[EvidenceDimension, SlaDecisionStatus],
        set[EvidenceDimension],
    ]:
        labels: dict[EvidenceDimension, str] = {}
        statuses: dict[EvidenceDimension, SlaDecisionStatus] = {}
        conflicts: set[EvidenceDimension] = set()
        for dimension, candidate, verdict in (
            (
                EvidenceDimension.SYMPTOM,
                diagnosis.symptom.label,
                verification.symptom.verdict,
            ),
            (
                EvidenceDimension.ROOT_CAUSE,
                diagnosis.root_cause.label,
                verification.root_cause.verdict,
            ),
        ):
            baseline = self._baseline_label(anchor, dimension)
            if verdict is SlaVerifierVerdict.ACCEPT_CANDIDATE:
                labels[dimension] = candidate
                statuses[dimension] = (
                    SlaDecisionStatus.BASELINE_AGREEMENT
                    if candidate == baseline
                    else SlaDecisionStatus.MODEL_VERIFIED
                )
            elif verdict is SlaVerifierVerdict.PRESERVE_BASELINE:
                labels[dimension] = baseline
                statuses[dimension] = SlaDecisionStatus.BASELINE_AGREEMENT
                if candidate != baseline:
                    conflicts.add(dimension)
            else:
                labels[dimension] = baseline
                statuses[dimension] = SlaDecisionStatus.UNRESOLVED_FALLBACK
        return labels, statuses, conflicts

    @staticmethod
    def _arbitration_label(
        arbitration: SlaConditionalArbitration, dimension: EvidenceDimension
    ) -> str:
        choice = (
            arbitration.symptom
            if dimension is EvidenceDimension.SYMPTOM
            else arbitration.root_cause
        )
        assert choice is not None
        return choice.selected_label

    @staticmethod
    def _fallback_citations(view: EvidenceView) -> tuple[str, ...]:
        validity = assess_stage3_evidence_validity(view)
        valid_ids = set(validity.valid_evidence_ids)
        for item in view.items:
            if item.evidence_id not in valid_ids:
                continue
            if any(
                supports_readiness_dimension(
                    item, dimension, domain=view.domain_profile
                )
                for dimension in _DIMENSIONS
            ):
                return (item.evidence_id,)
        return ()

    def _decision_citations(
        self,
        view: EvidenceView,
        diagnosis: SlaJointDiagnosis,
        verification: SlaJointVerification,
        arbitration: SlaConditionalArbitration | None,
    ) -> tuple[str, ...]:
        cited = [
            *verification.symptom.supporting_evidence_ids,
            *verification.root_cause.supporting_evidence_ids,
        ]
        if arbitration is not None:
            for choice in (arbitration.symptom, arbitration.root_cause):
                if choice is not None:
                    cited.extend(choice.supporting_evidence_ids)
        if not cited:
            cited.extend(diagnosis.symptom.supporting_evidence_ids)
            cited.extend(diagnosis.root_cause.supporting_evidence_ids)
        if not cited:
            cited.extend(self._fallback_citations(view))
        return tuple(dict.fromkeys(cited))
