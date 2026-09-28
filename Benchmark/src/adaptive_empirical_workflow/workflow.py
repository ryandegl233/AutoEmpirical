"""Public composition API for the complete adaptive workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .contracts import Stage2Decision
from .controller import Stage2WorkflowResult
from .ledger import EvidenceLedger


class Stage2Runner(Protocol):
    def run(self, ledger: EvidenceLedger) -> Stage2WorkflowResult: ...


class Stage3Runner(Protocol):
    def run(self, ledger: EvidenceLedger) -> object: ...


@dataclass(frozen=True)
class AdaptiveWorkflowResult:
    stage2: Stage2WorkflowResult | None
    stage3: object | None
    stop_reason: str | None


class AdaptiveEmpiricalWorkflow:
    """Gates Stage 3 on a verified Stage 2 fault decision."""

    def __init__(self, *, stage2: Stage2Runner, stage3: Stage3Runner) -> None:
        self._stage2 = stage2
        self._stage3 = stage3

    def run(self, ledger: EvidenceLedger) -> AdaptiveWorkflowResult:
        stage2 = self._stage2.run(ledger)
        if stage2.final_decision is None:
            return AdaptiveWorkflowResult(
                stage2=stage2,
                stage3=None,
                stop_reason="stage2_unresolved",
            )
        if stage2.verification is None or not stage2.verification.valid:
            return AdaptiveWorkflowResult(
                stage2=stage2,
                stage3=None,
                stop_reason="stage2_verification_failed",
            )
        if stage2.final_decision.decision is Stage2Decision.REJECTED:
            return AdaptiveWorkflowResult(
                stage2=stage2,
                stage3=None,
                stop_reason="stage2_rejected_candidate",
            )
        return AdaptiveWorkflowResult(
            stage2=stage2,
            stage3=self._stage3.run(ledger),
            stop_reason=None,
        )
