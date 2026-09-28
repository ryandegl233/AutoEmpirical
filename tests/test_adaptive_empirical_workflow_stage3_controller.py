from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import replace
from types import MappingProxyType

import pytest

from Benchmark.src.adaptive_empirical_workflow import frozen_evidence_runtime
from Benchmark.src.adaptive_empirical_workflow import (
    verification as verification_module,
)

from Benchmark.src.adaptive_empirical_workflow.audit import workflow_audit_record
from Benchmark.src.adaptive_empirical_workflow.baseline_preservation import (
    compose_baseline_preservation_decision,
)
from Benchmark.src.adaptive_empirical_workflow.baseline_anchor import (
    baseline_anchor_hash,
)
from Benchmark.src.adaptive_empirical_workflow.agents import (
    ModelTransportError,
    StructuredOutputError,
)
from Benchmark.src.adaptive_empirical_workflow.contracts import (
    ArbitrationSource,
    BaselineAnchor,
    BaselineDecisionAction,
    BaselineRevisionAssessment,
    RevisionAssessmentVerdict,
    RevisionEntailmentResult,
    RevisionFalsificationResult,
    normalize_revision_entailment_result,
    normalize_revision_falsification_result,
    revision_raw_result_digest,
    baseline_revision_assessment_digest,
    canonical_evidence_view_hash,
    BoundaryCard,
    BoundaryCriterion,
    BoundaryChallenge,
    BoundaryChallengeAction,
    CausalConsistencyReport,
    ConsistencyStatus,
    DisagreementMap,
    DimensionReadiness,
    DimensionVerificationReport,
    EvidenceDelta,
    EvidenceDimension,
    EvidenceExplicitness,
    EvidenceFact,
    EvidenceItem,
    EvidenceReadinessReport,
    EvidenceRequest,
    EvidenceSufficiency,
    JointAnchorReport,
    RetrievalStatus,
    RevisionConsistencyReport,
    RevisionProposalEnvelope,
    revision_proposal_digest,
    ResolutionStatus,
    RootCauseReport,
    SpecialistType,
    Stage3ArbitrationDecision,
    Stage3RevisionAudit,
    Stage3TeamReport,
    SymptomReport,
    TaxonomyNode,
    TaxonomySemanticOrigin,
    TaxonomyStructure,
    VerificationVerdict,
)
from Benchmark.src.adaptive_empirical_workflow.controller import (
    Stage3Controller,
    Stage3WorkflowConfig,
    _boundary_challenge_errors,
)
from Benchmark.src.adaptive_empirical_workflow.domains import build_record_runtime
from Benchmark.src.adaptive_empirical_workflow.frozen_evidence_graph import (
    EvidenceAuthority,
    FrozenEvidenceGraphBundle,
    FrozenGraphNode,
    FrozenNodeType,
    FrozenRecordGraph,
    frozen_graph_node_id,
)
from Benchmark.src.adaptive_empirical_workflow.frozen_evidence_runtime import (
    load_registered_frozen_evidence_runtime,
    project_frozen_evidence_for_record,
)
from Benchmark.src.adaptive_empirical_workflow.ledger import EvidenceLedger
from Benchmark.src.adaptive_empirical_workflow.specialists import SpecialistRegistry
from Benchmark.src.adaptive_empirical_workflow.stage3_composition import (
    build_revision_proposals,
)
from Benchmark.src.adaptive_empirical_workflow.taxonomy_structure import (
    taxonomy_structure_hash,
)
from Benchmark.src.adaptive_empirical_workflow.workflow import AdaptiveWorkflowResult
from Benchmark.src.adaptive_empirical_workflow.verification import (
    verify_stage3_decision,
)


def _ledger(metadata: dict[str, object] | None = None) -> EvidenceLedger:
    content = (
        "A null cache entry causes request rejection; the patch skips null entries."
    )
    item = EvidenceItem(
        evidence_id="code-1",
        record_id="record-1",
        source_type="test_result",
        source_uri="https://example.test/commit/1",
        retrieved_at="2026-08-05T11:00:00Z",
        content=content,
        content_sha256=hashlib.sha256(content.encode()).hexdigest(),
        explicitness=EvidenceExplicitness.DIRECT,
        metadata={
            "evidence_capabilities": [
                "symptom_observation",
                "defect_mechanism",
            ],
            **(metadata or {}),
        },
    )
    return EvidenceLedger(
        record_id="record-1",
        task="Classify the symptom and root cause.",
        taxonomy={
            "symptom": ["unexpected_rejection", "wrong_output"],
            "root_cause": ["missing_null_check", "incorrect_condition"],
        },
        domain_profile="issta2024",
        initial_items=[item],
    )


def _issue_only_ledger() -> EvidenceLedger:
    content = "Maintainer: requests with null cache entries are unexpectedly rejected."
    item = EvidenceItem(
        evidence_id="code-1",
        record_id="record-1",
        source_type="issue_body",
        source_uri="https://example.test/issues/1",
        retrieved_at="2026-08-05T11:00:00Z",
        content=content,
        content_sha256=hashlib.sha256(content.encode()).hexdigest(),
        explicitness=EvidenceExplicitness.DIRECT,
    )
    mechanism_content = "The code context shows a missing null-entry guard."
    mechanism = EvidenceItem(
        evidence_id="code-2",
        record_id="record-1",
        source_type="code_diff",
        source_uri="https://example.test/commit/1",
        retrieved_at="2026-08-05T11:01:00Z",
        content=mechanism_content,
        content_sha256=hashlib.sha256(mechanism_content.encode()).hexdigest(),
        explicitness=EvidenceExplicitness.DIRECT,
    )
    return EvidenceLedger(
        record_id="record-1",
        task="Classify the symptom and root cause.",
        taxonomy={
            "symptom": ["unexpected_rejection", "wrong_output"],
            "root_cause": ["missing_null_check", "incorrect_condition"],
        },
        domain_profile="issta2024",
        initial_items=[item, mechanism],
    )


def _ase_issue_only_ledger() -> EvidenceLedger:
    content = "Maintainer: the unsupported API combination causes the crash."
    item = EvidenceItem(
        evidence_id="code-1",
        record_id="record-1",
        source_type="issue_comments",
        source_uri="https://example.test/issues/1",
        retrieved_at="2026-08-05T11:00:00Z",
        content=content,
        content_sha256=hashlib.sha256(content.encode()).hexdigest(),
        explicitness=EvidenceExplicitness.DIRECT,
    )
    return EvidenceLedger(
        record_id="record-1",
        task="Classify the symptom and root cause.",
        taxonomy={
            "symptom": ["unexpected_rejection", "wrong_output"],
            "root_cause": ["missing_null_check", "incorrect_condition"],
        },
        domain_profile="ase2022",
        initial_items=[item],
    )


def _stage3_validity_ledger(*, include_valid: bool) -> EvidenceLedger:
    items: list[EvidenceItem] = []
    if include_valid:
        content = "Maintainer: requests with null cache entries are rejected."
        items.append(
            EvidenceItem(
                evidence_id="issue-1",
                record_id="record-1",
                source_type="issue_body",
                source_uri="https://example.test/issues/1",
                retrieved_at="2026-08-05T11:00:00Z",
                content=content,
                content_sha256=hashlib.sha256(content.encode()).hexdigest(),
                explicitness=EvidenceExplicitness.DIRECT,
            )
        )
    invalid_content = "Untrusted material must remain visible only to audit."
    items.append(
        EvidenceItem(
            evidence_id="invalid-1",
            record_id="record-1",
            source_type="invented_source",
            source_uri="https://example.test/untrusted/1",
            retrieved_at="2026-08-05T11:01:00Z",
            content=invalid_content,
            content_sha256=hashlib.sha256(invalid_content.encode()).hexdigest(),
            explicitness=EvidenceExplicitness.DIRECT,
            metadata={"evidence_capabilities": ["symptom_observation"]},
        )
    )
    return EvidenceLedger(
        record_id="record-1",
        task="Classify the symptom and root cause.",
        taxonomy={
            "symptom": ["unexpected_rejection", "wrong_output"],
            "root_cause": ["missing_null_check", "incorrect_condition"],
        },
        domain_profile="ase2022",
        initial_items=items,
    )


def _symptom(label: str = "unexpected_rejection") -> SymptomReport:
    return SymptomReport(
        label=label,
        behavior_claim="A valid request is rejected instead of being processed.",
        supporting_evidence_ids=["code-1"],
        alternative_label=(
            "wrong_output"
            if label == "unexpected_rejection"
            else "unexpected_rejection"
        ),
        boundary_reason="The operation is prevented rather than completed incorrectly.",
        boundary_evidence_ids=["code-1"],
        confidence=0.9,
        evidence_sufficiency=EvidenceSufficiency.SUFFICIENT,
    )


def _root(label: str = "missing_null_check") -> RootCauseReport:
    return RootCauseReport(
        label=label,
        defect_mechanism="A nullable cache entry is dereferenced without a guard.",
        causal_chain=[
            "The cache returns a null entry.",
            "The handler treats the entry as populated.",
            "The request follows the rejection branch.",
        ],
        supporting_evidence_ids=["code-1"],
        alternative_label=(
            "incorrect_condition"
            if label == "missing_null_check"
            else "missing_null_check"
        ),
        boundary_reason="The condition is valid once null entries are excluded.",
        boundary_evidence_ids=["code-1"],
        confidence=0.88,
        evidence_sufficiency=EvidenceSufficiency.SUFFICIENT,
    )


def _consistent() -> CausalConsistencyReport:
    return CausalConsistencyReport(
        status=ConsistencyStatus.CONSISTENT,
        rationale=(
            "The missing guard explains how the null entry causes request rejection."
        ),
        supporting_evidence_ids=["code-1"],
    )


def _readiness_report(*, sufficient: bool = True) -> EvidenceReadinessReport:
    return EvidenceReadinessReport(
        task="stage3",
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
            for dimension in (
                EvidenceDimension.SYMPTOM,
                EvidenceDimension.ROOT_CAUSE,
            )
        ),
    )


def _ready(task: str, view: object) -> EvidenceReadinessReport:
    assert task == "stage3"
    root_evidence_id = next(
        (item.evidence_id for item in view.items if item.source_type == "code_diff"),
        "code-1",
    )
    report = _readiness_report()
    root = report.dimensions[1].model_copy(
        update={"confirmed_evidence_ids": (root_evidence_id,)}
    )
    return report.model_copy(update={"dimensions": (report.dimensions[0], root)})


def test_symptom_and_root_cause_are_independent_inputs_to_checker() -> None:
    events: list[tuple[str, str]] = []
    classification_views: list[object] = []
    readiness_events: list[int] = []
    analyst_barrier = threading.Barrier(4)
    checker_barrier = threading.Barrier(2)
    event_lock = threading.Lock()

    def readiness(task: str, view: object) -> EvidenceReadinessReport:
        assert not events
        readiness_events.append(view.ledger_version)
        return _readiness_report()

    def symptom_analyst(team_id: str, view: object) -> SymptomReport:
        analyst_barrier.wait(timeout=2)
        with event_lock:
            classification_views.append(view)
            events.append((team_id, "symptom"))
        return _symptom()

    def root_analyst(team_id: str, view: object) -> RootCauseReport:
        analyst_barrier.wait(timeout=2)
        with event_lock:
            classification_views.append(view)
            events.append((team_id, "root"))
        return _root()

    def checker(
        team_id: str,
        symptom: SymptomReport,
        root: RootCauseReport,
        view: object,
    ) -> CausalConsistencyReport:
        with event_lock:
            assert len(events) == 4
            classification_views.append(view)
        checker_barrier.wait(timeout=2)
        with event_lock:
            events.append((team_id, "checker"))
        return _consistent()

    result = Stage3Controller(
        readiness=readiness,
        symptom_analyst=symptom_analyst,
        root_cause_analyst=root_analyst,
        consistency_checker=checker,
    ).run(_ledger())

    assert set(events[:4]) == {
        ("A", "symptom"),
        ("A", "root"),
        ("B", "symptom"),
        ("B", "root"),
    }
    assert set(events[4:]) == {("A", "checker"), ("B", "checker")}
    assert readiness_events == [1]
    assert len({id(view) for view in classification_views}) == 6
    assert all(view == classification_views[0] for view in classification_views)
    assert {view.ledger_version for view in classification_views} == {1}
    assert result.classification_ledger_version == 1
    assert result.readiness_report == _readiness_report()
    assert result.unresolved is None
    assert result.verification is not None
    assert result.verification.valid is True


def test_consistency_checker_cannot_emit_or_overwrite_classification_labels() -> None:
    assert "label" not in CausalConsistencyReport.model_fields
    assert "symptom_label" not in CausalConsistencyReport.model_fields
    assert "root_cause_label" not in CausalConsistencyReport.model_fields


def test_boundary_challenger_pass_preserves_direct_consensus() -> None:
    challenge_views: list[object] = []

    def challenger(reports: tuple[object, ...], view: object) -> BoundaryChallenge:
        challenge_views.append(view)
        assert len(reports) == 2
        return BoundaryChallenge(
            action=BoundaryChallengeAction.PASS,
            rationale="The proposed labels respect the nearest taxonomy boundaries.",
            cited_evidence_ids=["code-1"],
        )

    result = Stage3Controller(
        readiness=_ready,
        symptom_analyst=lambda team_id, view: _symptom(),
        root_cause_analyst=lambda team_id, view: _root(),
        consistency_checker=lambda team_id, symptom, root, view: _consistent(),
        boundary_challenger=challenger,
    ).run(_ledger())

    assert len(challenge_views) == 1
    assert challenge_views[0].ledger_version == 1
    assert result.boundary_challenge is not None
    assert result.boundary_challenge.action is BoundaryChallengeAction.PASS
    assert result.final_decision is not None
    assert result.unresolved is None


@pytest.mark.parametrize(
    "error",
    (
        StructuredOutputError(
            "SECRET_RAW_CHALLENGER_OUTPUT",
            schema_name="BoundaryChallenge",
            validation_summary={"fields": ["__root__"], "codes": ["value_error"]},
        ),
        ModelTransportError("SECRET_CHALLENGER_PROVIDER_FAILURE", attempts=6),
    ),
    ids=("structured-output", "model-transport"),
)
def test_boundary_challenger_operational_failure_preserves_verified_candidates(
    error: Exception,
) -> None:
    def challenger(reports: tuple[object, ...], view: object) -> BoundaryChallenge:
        assert len(reports) == 2
        raise error

    result = Stage3Controller(
        readiness=_ready,
        symptom_analyst=lambda team_id, view: _symptom(),
        root_cause_analyst=lambda team_id, view: _root(),
        consistency_checker=lambda team_id, symptom, root, view: _consistent(),
        boundary_challenger=challenger,
    ).run(_ledger())

    assert len(result.reports) == 2
    assert result.final_decision is not None
    assert result.final_decision.symptom_label == "unexpected_rejection"
    assert result.final_decision.root_cause_label == "missing_null_check"
    assert result.final_decision.source.value == "fallback_uncertain"
    assert result.verification is not None and result.verification.valid is True
    assert result.unresolved is None
    assert result.boundary_challenge is None
    failure = result.component_failures[-1]
    assert failure.team_id == "challenge"
    assert failure.role == "boundary_challenger"
    assert failure.error_type == type(error).__name__
    assert "SECRET" not in repr(failure.model_dump(mode="python"))


def test_boundary_challenger_generic_runtime_error_propagates() -> None:
    def challenger(reports: tuple[object, ...], view: object) -> BoundaryChallenge:
        raise RuntimeError("generic challenger bug")

    with pytest.raises(RuntimeError, match="generic challenger bug"):
        Stage3Controller(
            readiness=_ready,
            symptom_analyst=lambda team_id, view: _symptom(),
            root_cause_analyst=lambda team_id, view: _root(),
            consistency_checker=lambda team_id, symptom, root, view: _consistent(),
            boundary_challenger=challenger,
        ).run(_ledger())


def _root_with_evidence(
    evidence_id: str,
    label: str = "missing_null_check",
) -> RootCauseReport:
    return _root(label).model_copy(
        update={
            "supporting_evidence_ids": [evidence_id],
            "boundary_evidence_ids": [evidence_id],
        }
    )


def test_boundary_pass_cannot_add_symptom_evidence_to_root_disagreement_packet() -> (
    None
):
    packets: list[object] = []

    def arbitrator(packet: object) -> Stage3ArbitrationDecision:
        packets.append(packet)
        assert "root_cause_label" in packet.disagreement.dimensions
        assert [item.evidence_id for item in packet.relevant_evidence] == ["code-2"]
        return Stage3ArbitrationDecision(
            resolution_status=ResolutionStatus.RESOLVED,
            symptom_label="unexpected_rejection",
            root_cause_label="missing_null_check",
            confidence=0.85,
            rationale="The mechanism evidence resolves only the root-cause disagreement.",
            supporting_evidence_ids=["code-2"],
            resolved_dimensions=list(packet.disagreement.dimensions),
        )

    result = Stage3Controller(
        readiness=_ready,
        symptom_analyst=lambda team_id, view: _symptom(),
        root_cause_analyst=lambda team_id, view: _root_with_evidence(
            "code-2",
            "missing_null_check" if team_id == "A" else "incorrect_condition",
        ),
        consistency_checker=lambda team_id, symptom, root, view: _consistent(),
        boundary_challenger=lambda reports, view: BoundaryChallenge(
            action=BoundaryChallengeAction.PASS,
            rationale="No additional boundary review is requested.",
            cited_evidence_ids=["code-1"],
        ),
        arbitrator=arbitrator,
    ).run(_issue_only_ledger())

    assert len(packets) == 1
    assert result.final_decision is not None


def test_boundary_pass_cannot_add_cause_evidence_to_symptom_disagreement_packet() -> (
    None
):
    packets: list[object] = []

    def arbitrator(packet: object) -> Stage3ArbitrationDecision:
        packets.append(packet)
        assert "symptom_label" in packet.disagreement.dimensions
        assert [item.evidence_id for item in packet.relevant_evidence] == ["code-1"]
        return Stage3ArbitrationDecision(
            resolution_status=ResolutionStatus.RESOLVED,
            symptom_label="unexpected_rejection",
            root_cause_label="missing_null_check",
            confidence=0.85,
            rationale="The observation evidence resolves only the symptom disagreement.",
            supporting_evidence_ids=["code-1"],
            resolved_dimensions=list(packet.disagreement.dimensions),
        )

    result = Stage3Controller(
        readiness=_ready,
        symptom_analyst=lambda team_id, view: _symptom(
            "unexpected_rejection" if team_id == "A" else "wrong_output"
        ),
        root_cause_analyst=lambda team_id, view: _root_with_evidence("code-2"),
        consistency_checker=lambda team_id, symptom, root, view: _consistent(),
        boundary_challenger=lambda reports, view: BoundaryChallenge(
            action=BoundaryChallengeAction.PASS,
            rationale="No additional boundary review is requested.",
            cited_evidence_ids=["code-2"],
        ),
        arbitrator=arbitrator,
    ).run(_issue_only_ledger())

    assert len(packets) == 1
    assert result.final_decision is not None


def test_boundary_challenger_review_triggers_bounded_non_relabeling_arbitration() -> (
    None
):
    packets: list[object] = []

    def arbitrator(packet: object) -> Stage3ArbitrationDecision:
        packets.append(packet)
        assert packet.disagreement.dimensions == ("symptom_review",)
        assert [item.evidence_id for item in packet.relevant_evidence] == ["code-1"]
        return Stage3ArbitrationDecision(
            resolution_status=ResolutionStatus.RESOLVED,
            symptom_label="unexpected_rejection",
            root_cause_label="missing_null_check",
            confidence=0.85,
            rationale="The cited observation supports preserving the unanimous labels.",
            supporting_evidence_ids=["code-1"],
            resolved_dimensions=["symptom_review"],
        )

    result = Stage3Controller(
        readiness=_ready,
        symptom_analyst=lambda team_id, view: _symptom(),
        root_cause_analyst=lambda team_id, view: _root(),
        consistency_checker=lambda team_id, symptom, root, view: _consistent(),
        boundary_challenger=lambda reports, view: BoundaryChallenge(
            action=BoundaryChallengeAction.SYMPTOM_REVIEW,
            rationale="The observation level is close to the competing symptom label.",
            cited_evidence_ids=["code-1"],
        ),
        arbitrator=arbitrator,
    ).run(_ledger())

    assert len(packets) == 1
    assert result.final_decision is not None
    assert result.final_decision.symptom_label == "unexpected_rejection"
    assert result.final_decision.root_cause_label == "missing_null_check"


def test_boundary_cause_review_includes_only_owned_mechanism_evidence() -> None:
    packets: list[object] = []

    def arbitrator(packet: object) -> Stage3ArbitrationDecision:
        packets.append(packet)
        assert packet.disagreement.dimensions == ("cause_review",)
        assert [item.evidence_id for item in packet.relevant_evidence] == ["code-2"]
        return Stage3ArbitrationDecision(
            resolution_status=ResolutionStatus.RESOLVED,
            symptom_label="unexpected_rejection",
            root_cause_label="missing_null_check",
            confidence=0.85,
            rationale="The mechanism evidence supports preserving the proposed cause.",
            supporting_evidence_ids=["code-2"],
            resolved_dimensions=["cause_review"],
        )

    result = Stage3Controller(
        readiness=_ready,
        symptom_analyst=lambda team_id, view: _symptom(),
        root_cause_analyst=lambda team_id, view: _root_with_evidence("code-2"),
        consistency_checker=lambda team_id, symptom, root, view: _consistent(),
        boundary_challenger=lambda reports, view: BoundaryChallenge(
            action=BoundaryChallengeAction.CAUSE_REVIEW,
            rationale="The local mechanism is close to a higher-level cause boundary.",
            cited_evidence_ids=["code-2"],
        ),
        arbitrator=arbitrator,
    ).run(_issue_only_ledger())

    assert len(packets) == 1
    assert result.final_decision is not None


@pytest.mark.parametrize(
    ("action", "evidence_id", "expected_dimension"),
    [
        (BoundaryChallengeAction.SYMPTOM_REVIEW, "code-2", "symptom-irrelevant"),
        (BoundaryChallengeAction.CAUSE_REVIEW, "code-1", "cause-irrelevant"),
    ],
)
def test_boundary_review_rejects_known_wrong_dimension_evidence_before_arbitration(
    action: BoundaryChallengeAction,
    evidence_id: str,
    expected_dimension: str,
) -> None:
    arbitrator_calls: list[object] = []
    result = Stage3Controller(
        readiness=_ready,
        symptom_analyst=lambda team_id, view: _symptom(),
        root_cause_analyst=lambda team_id, view: _root_with_evidence("code-2"),
        consistency_checker=lambda team_id, symptom, root, view: _consistent(),
        boundary_challenger=lambda reports, view: BoundaryChallenge(
            action=action,
            rationale="The proposed label requires a dimension-owned boundary check.",
            cited_evidence_ids=[evidence_id],
        ),
        arbitrator=lambda packet: arbitrator_calls.append(packet),
    ).run(_issue_only_ledger())

    assert result.final_decision is None
    assert result.unresolved is not None
    assert expected_dimension in result.unresolved.missing_facts[0]
    assert arbitrator_calls == []


def test_boundary_cause_review_cannot_reexpand_issue_body_metadata() -> None:
    view = _issue_only_ledger().view()
    issue = view.items[0].model_copy(
        update={"metadata": {"evidence_capabilities": ["defect_mechanism"]}}
    )
    tampered_view = view.model_copy(update={"items": (issue, view.items[1])})
    report = Stage3TeamReport(
        team_id="A",
        symptom=_symptom(),
        root_cause=_root_with_evidence("code-1"),
        consistency=_consistent(),
    )
    challenge = BoundaryChallenge(
        action=BoundaryChallengeAction.CAUSE_REVIEW,
        rationale="Metadata cannot grant mechanism authority to issue prose.",
        cited_evidence_ids=["code-1"],
    )

    assert _boundary_challenge_errors(
        challenge,
        (report, report.model_copy(update={"team_id": "B"})),
        tampered_view,
    ) == ("Boundary challenger cites cause-irrelevant evidence_id 'code-1'.",)


def test_ase_cause_challenge_accepts_issue_context_owned_by_root_reports() -> None:
    report = Stage3TeamReport(
        team_id="A",
        symptom=_symptom(),
        root_cause=_root(),
        consistency=_consistent(),
    )
    challenge = BoundaryChallenge(
        action=BoundaryChallengeAction.CAUSE_REVIEW,
        rationale="The maintainer explanation supports reviewing the nearest cause.",
        cited_evidence_ids=["code-1"],
    )

    assert (
        _boundary_challenge_errors(
            challenge,
            (report, report.model_copy(update={"team_id": "B"})),
            _ase_issue_only_ledger().view(),
        )
        == ()
    )


def test_boundary_review_unauthorized_relabel_fails_closed() -> None:
    def arbitrator(packet: object) -> Stage3ArbitrationDecision:
        return Stage3ArbitrationDecision(
            resolution_status=ResolutionStatus.RESOLVED,
            symptom_label="wrong_output",
            root_cause_label="missing_null_check",
            confidence=0.85,
            rationale="The challenger review alone cannot replace a unanimous label.",
            supporting_evidence_ids=["code-1"],
            resolved_dimensions=list(packet.disagreement.dimensions),
        )

    result = Stage3Controller(
        readiness=_ready,
        symptom_analyst=lambda team_id, view: _symptom(),
        root_cause_analyst=lambda team_id, view: _root(),
        consistency_checker=lambda team_id, symptom, root, view: _consistent(),
        boundary_challenger=lambda reports, view: BoundaryChallenge(
            action=BoundaryChallengeAction.SYMPTOM_REVIEW,
            rationale="The observation level is close to the competing symptom label.",
            cited_evidence_ids=["code-1"],
        ),
        arbitrator=arbitrator,
    ).run(_ledger())

    assert result.final_decision is None
    assert result.verification is None
    assert result.unresolved is not None
    assert result.unresolved.stop_reason == "arbitration_authority_violation"


def test_boundary_challenger_unknown_id_fails_closed_without_arbitration() -> None:
    arbitrator_calls: list[object] = []
    challenge = BoundaryChallenge(
        action=BoundaryChallengeAction.CAUSE_REVIEW,
        rationale="The proposed mechanism requires review against its nearest boundary.",
        cited_evidence_ids=["unknown-id"],
    )
    result = Stage3Controller(
        readiness=_ready,
        symptom_analyst=lambda team_id, view: _symptom(),
        root_cause_analyst=lambda team_id, view: _root(),
        consistency_checker=lambda team_id, symptom, root, view: _consistent(),
        boundary_challenger=lambda reports, view: challenge,
        arbitrator=lambda packet: arbitrator_calls.append(packet),
    ).run(_ledger())

    assert result.final_decision is None
    assert result.verification is None
    assert result.unresolved is not None
    assert result.unresolved.stop_reason == "boundary_challenger_invalid_citation"
    assert arbitrator_calls == []


def test_boundary_challenger_evidence_request_keeps_a_fallback_prediction() -> None:
    arbitrator_calls: list[object] = []
    challenge = BoundaryChallenge(
        action=BoundaryChallengeAction.EVIDENCE_REQUEST,
        rationale="A frozen execution-phase observation is missing for this boundary.",
        cited_evidence_ids=["code-1"],
    )

    result = Stage3Controller(
        readiness=_ready,
        symptom_analyst=lambda team_id, view: _symptom(),
        root_cause_analyst=lambda team_id, view: _root(),
        consistency_checker=lambda team_id, symptom, root, view: _consistent(),
        boundary_challenger=lambda reports, view: challenge,
        arbitrator=lambda packet: arbitrator_calls.append(packet),
    ).run(_ledger())

    assert result.final_decision is not None
    assert result.final_decision.symptom_label == "unexpected_rejection"
    assert result.final_decision.root_cause_label == "missing_null_check"
    assert result.final_decision.source.value == "fallback_uncertain"
    assert result.verification is not None
    assert result.verification.valid is True
    assert result.unresolved is None
    assert result.boundary_challenge == challenge
    assert arbitrator_calls == []


def test_stage3_disagreement_invokes_anonymized_targeted_arbitration() -> None:
    calls: list[object] = []
    classification_views: list[object] = []

    def symptom_analyst(team_id: str, view: object) -> SymptomReport:
        classification_views.append(view)
        return _symptom("unexpected_rejection" if team_id == "A" else "wrong_output")

    def arbitrator(packet: object) -> Stage3ArbitrationDecision:
        calls.append(packet)
        assert not hasattr(packet.team_a, "team_id")
        assert not hasattr(packet.team_a, "perspective")
        assert not hasattr(packet.team_b, "perspective")
        assert packet.disagreement.dimensions == ("symptom_label",)
        assert packet.classification_ledger_version == 1
        assert packet.relevant_evidence == classification_views[0].items
        return Stage3ArbitrationDecision(
            resolution_status=ResolutionStatus.RESOLVED,
            symptom_label="unexpected_rejection",
            root_cause_label="missing_null_check",
            confidence=0.85,
            rationale="Observed behavior is prevention, while the causal mechanism is null handling.",
            supporting_evidence_ids=["code-1"],
            resolved_dimensions=["symptom_label"],
        )

    result = Stage3Controller(
        readiness=_ready,
        symptom_analyst=symptom_analyst,
        root_cause_analyst=lambda team_id, view: _root(),
        consistency_checker=lambda team_id, symptom, root, view: _consistent(),
        arbitrator=arbitrator,
    ).run(_ledger())

    assert len(calls) == 1
    assert result.final_decision is not None
    assert result.final_decision.symptom_label == "unexpected_rejection"
    assert result.verification is not None
    assert result.verification.valid is True


def test_matching_specific_root_cause_risk_invokes_review_on_nominal_consensus() -> (
    None
):
    packets: list[object] = []

    def arbitrator(packet: object) -> Stage3ArbitrationDecision:
        packets.append(packet)
        return Stage3ArbitrationDecision(
            resolution_status=ResolutionStatus.UNRESOLVED,
            rationale="The cited issue text cannot establish the asserted defect mechanism.",
            supporting_evidence_ids=["code-1"],
            unresolved_dimensions=["unsupported_root_cause_specificity"],
            missing_facts=["Code or test evidence identifying the mechanism."],
        )

    result = Stage3Controller(
        readiness=_ready,
        symptom_analyst=lambda team_id, view: _symptom(),
        root_cause_analyst=lambda team_id, view: _root(),
        consistency_checker=lambda team_id, symptom, root, view: _consistent(),
        arbitrator=arbitrator,
    ).run(_issue_only_ledger())

    assert len(packets) == 1
    assert "unsupported_root_cause_specificity" in (packets[0].disagreement.dimensions)
    assert result.final_decision is not None
    assert result.final_decision.source.value == "fallback_uncertain"
    assert result.verification is not None
    assert result.verification.valid is True
    assert result.unresolved is None


def test_stage3_arbitrator_may_leave_case_unresolved() -> None:
    def symptom(team_id: str, view: object) -> SymptomReport:
        return _symptom("unexpected_rejection" if team_id == "A" else "wrong_output")

    def arbitrator(packet: object) -> Stage3ArbitrationDecision:
        assert tuple(item.evidence_id for item in packet.relevant_evidence) == (
            "code-1",
        )
        return Stage3ArbitrationDecision(
            resolution_status=ResolutionStatus.UNRESOLVED,
            rationale="The cited frozen evidence cannot distinguish the symptom boundary.",
            supporting_evidence_ids=["code-1"],
            unresolved_dimensions=["symptom_label"],
            missing_facts=["Observable runtime impact independent of the patch."],
        )

    result = Stage3Controller(
        readiness=_ready,
        symptom_analyst=symptom,
        root_cause_analyst=lambda team_id, view: _root(),
        consistency_checker=lambda team_id, symptom, root, view: _consistent(),
        arbitrator=arbitrator,
    ).run(_ledger())

    assert result.final_decision is not None
    assert result.final_decision.symptom_label == "unexpected_rejection"
    assert result.final_decision.root_cause_label == "missing_null_check"
    assert result.final_decision.confidence == 0.88
    assert result.final_decision.source.value == "fallback_uncertain"
    assert result.verification is not None
    assert result.verification.valid is True
    assert result.unresolved is None


@pytest.mark.parametrize(
    "citation_id,append_known_item",
    [
        ("unknown-arbitration-evidence", False),
        ("known-but-outside-packet", True),
    ],
)
def test_arbitrator_evidence_outside_packet_fails_closed_without_fallback(
    citation_id: str,
    append_known_item: bool,
) -> None:
    ledger = _ledger()
    if append_known_item:
        ledger.append(
            ledger.get("code-1").model_copy(
                update={
                    "evidence_id": citation_id,
                    "source_uri": "https://example.test/commit/1#unrelated",
                }
            )
        )

    def symptom(team_id: str, view: object) -> SymptomReport:
        return _symptom("unexpected_rejection" if team_id == "A" else "wrong_output")

    def arbitrator(packet: object) -> Stage3ArbitrationDecision:
        packet_ids = tuple(item.evidence_id for item in packet.relevant_evidence)
        assert packet_ids == ("code-1",)
        assert citation_id not in packet_ids
        return Stage3ArbitrationDecision(
            resolution_status=ResolutionStatus.UNRESOLVED,
            rationale="The frozen packet cannot distinguish the symptom boundary.",
            supporting_evidence_ids=[citation_id],
            unresolved_dimensions=["symptom_label"],
            missing_facts=["A packet-owned direct symptom observation is required."],
        )

    result = Stage3Controller(
        readiness=_ready,
        symptom_analyst=symptom,
        root_cause_analyst=lambda team_id, view: _root(),
        consistency_checker=lambda team_id, symptom, root, view: _consistent(),
        arbitrator=arbitrator,
    ).run(ledger)

    assert result.final_decision is None
    assert result.verification is None
    assert result.unresolved is not None
    assert result.unresolved.stop_reason == "arbitration_invalid_citation"
    assert citation_id in " ".join(result.unresolved.missing_facts)


def test_stage3_fallback_prefers_direct_evidence_after_coverage_and_consistency_tie() -> (
    None
):
    ledger = _ledger()
    inferred = ledger.get("code-1").model_copy(
        update={
            "evidence_id": "inferred-2",
            "source_uri": "https://example.test/issues/1#inferred",
            "explicitness": EvidenceExplicitness.INFERRED,
        }
    )
    ledger.append(inferred)

    def evidence_id(team_id: str) -> str:
        return "inferred-2" if team_id == "A" else "code-1"

    def symptom(team_id: str, view: object) -> SymptomReport:
        return _symptom(
            "unexpected_rejection" if team_id == "A" else "wrong_output"
        ).model_copy(
            update={
                "supporting_evidence_ids": [evidence_id(team_id)],
                "boundary_evidence_ids": [evidence_id(team_id)],
            }
        )

    def root(team_id: str, view: object) -> RootCauseReport:
        return _root().model_copy(
            update={
                "supporting_evidence_ids": [evidence_id(team_id)],
                "boundary_evidence_ids": [evidence_id(team_id)],
            }
        )

    def checker(
        team_id: str,
        symptom_report: SymptomReport,
        root_report: RootCauseReport,
        view: object,
    ) -> CausalConsistencyReport:
        return _consistent().model_copy(
            update={"supporting_evidence_ids": [evidence_id(team_id)]}
        )

    result = Stage3Controller(
        readiness=_ready,
        symptom_analyst=symptom,
        root_cause_analyst=root,
        consistency_checker=checker,
    ).run(ledger)

    assert result.final_decision is not None
    assert result.final_decision.symptom_label == "wrong_output"
    assert result.final_decision.root_cause_label == "missing_null_check"
    assert result.final_decision.source.value == "fallback_uncertain"
    assert result.final_decision.supporting_evidence_ids == ("code-1",)
    assert result.verification is not None
    assert result.verification.valid is True


def test_unknown_stage3_citation_fails_closed_before_arbitrator() -> None:
    calls: list[object] = []

    def symptom(team_id: str, view: object) -> SymptomReport:
        report = _symptom("unexpected_rejection" if team_id == "A" else "wrong_output")
        return report.model_copy(
            update={"supporting_evidence_ids": ["missing-evidence"]}
        )

    result = Stage3Controller(
        readiness=_ready,
        symptom_analyst=symptom,
        root_cause_analyst=lambda team_id, view: _root(),
        consistency_checker=lambda team_id, symptom, root, view: _consistent(),
        arbitrator=lambda packet: calls.append(packet),
    ).run(_ledger())

    assert calls == []
    assert result.final_decision is None
    assert result.unresolved is not None
    assert "missing-evidence" in " ".join(result.unresolved.missing_facts)


@pytest.mark.parametrize("role_name", ["symptom", "root_cause"])
def test_unknown_stage3_boundary_citation_fails_before_arbitrator(
    role_name: str,
) -> None:
    calls: list[object] = []

    def symptom(team_id: str, view: object) -> SymptomReport:
        report = _symptom()
        if role_name == "symptom":
            return report.model_copy(
                update={"boundary_evidence_ids": ["missing-boundary"]}
            )
        return report

    def root(team_id: str, view: object) -> RootCauseReport:
        report = _root()
        if role_name == "root_cause":
            return report.model_copy(
                update={"boundary_evidence_ids": ["missing-boundary"]}
            )
        return report

    def arbitrator(packet: object) -> Stage3ArbitrationDecision:
        calls.append(packet)
        return Stage3ArbitrationDecision(
            resolution_status=ResolutionStatus.UNRESOLVED,
            rationale="The candidate boundary evidence cannot be resolved safely.",
            unresolved_dimensions=list(packet.disagreement.dimensions),
            missing_facts=["The cited boundary evidence is unavailable."],
        )

    result = Stage3Controller(
        readiness=_ready,
        symptom_analyst=symptom,
        root_cause_analyst=root,
        consistency_checker=lambda team_id, symptom, root, view: _consistent(),
        arbitrator=arbitrator,
    ).run(_ledger())

    assert calls == []
    assert result.final_decision is None
    assert result.verification is None
    assert result.unresolved is not None
    assert "missing-boundary" in " ".join(result.unresolved.missing_facts)


@pytest.mark.parametrize("confirmed_ids", [(), ("unknown-readiness",)])
def test_invalid_stage3_readiness_citations_are_diagnostic_only(
    confirmed_ids: tuple[str, ...],
) -> None:
    calls: list[str] = []
    malformed = EvidenceReadinessReport.model_construct(
        task="stage3",
        dimensions=(
            DimensionReadiness.model_construct(
                dimension=EvidenceDimension.SYMPTOM,
                sufficient=True,
                confirmed_evidence_ids=confirmed_ids,
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

    def record(role: str, report: object) -> object:
        calls.append(role)
        return report

    result = Stage3Controller(
        readiness=lambda task, view: malformed,
        symptom_analyst=lambda team_id, view: record("symptom", _symptom()),
        root_cause_analyst=lambda team_id, view: record("root", _root()),
        consistency_checker=lambda team_id, symptom, root, view: record(
            "checker", _consistent()
        ),
    ).run(_ledger())

    assert sorted(calls) == ["checker", "checker", "root", "root", "symptom", "symptom"]
    assert result.final_decision is not None
    if confirmed_ids:
        assert result.readiness_report == malformed
    else:
        assert result.readiness_report is None


def test_swapped_stage3_readiness_is_diagnostic_only() -> None:
    calls: list[str] = []
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

    def symptom(team_id: str, view: object) -> SymptomReport:
        calls.append("symptom")
        return _symptom()

    def root(team_id: str, view: object) -> RootCauseReport:
        calls.append("root")
        return _root()

    result = Stage3Controller(
        readiness=lambda task, view: readiness,
        symptom_analyst=symptom,
        root_cause_analyst=root,
        consistency_checker=lambda team_id, symptom, root, view: _consistent(),
    ).run(_issue_only_ledger())

    assert sorted(calls) == ["root", "root", "symptom", "symptom"]
    assert len(result.reports) == 2
    assert result.readiness_report == readiness


@pytest.mark.parametrize(
    "risk_kind,expected_dimension",
    [
        ("shared_inference", "shared_unsupported_inference"),
        ("boundary", "unsupported_symptom_boundary"),
        ("root_specificity", "unsupported_root_cause_specificity"),
    ],
)
def test_shared_risk_only_arbitration_relabel_fails_closed(
    risk_kind: str,
    expected_dimension: str,
) -> None:
    ledger = _ledger()
    if risk_kind == "shared_inference":
        item = ledger.get("code-1").model_copy(
            update={"explicitness": EvidenceExplicitness.INFERRED}
        )
        ledger = EvidenceLedger(
            record_id="record-1",
            task="Classify the symptom and root cause.",
            taxonomy={
                "symptom": ["unexpected_rejection", "wrong_output"],
                "root_cause": ["missing_null_check", "incorrect_condition"],
            },
            domain_profile="issta2024",
            initial_items=[item],
        )
    elif risk_kind == "root_specificity":
        ledger = _issue_only_ledger()

    def symptom(team_id: str, view: object) -> SymptomReport:
        report = _symptom()
        if risk_kind == "evidence_sufficiency" and team_id == "B":
            return report.model_copy(
                update={
                    "evidence_sufficiency": EvidenceSufficiency.INSUFFICIENT,
                    "unresolved_evidence_gaps": [
                        "The symptom observation is incomplete."
                    ],
                }
            )
        if risk_kind == "boundary":
            return report.model_copy(
                update={
                    "boundary_reason": "The labels appear plausibly separated.",
                    "boundary_evidence_ids": [],
                }
            )
        return report

    def arbitrator(packet: object) -> Stage3ArbitrationDecision:
        assert expected_dimension in packet.disagreement.dimensions
        return Stage3ArbitrationDecision(
            resolution_status=ResolutionStatus.RESOLVED,
            symptom_label=(
                "wrong_output"
                if risk_kind in {"shared_inference", "evidence_sufficiency", "boundary"}
                else "unexpected_rejection"
            ),
            root_cause_label=(
                "incorrect_condition"
                if risk_kind in {"shared_inference", "root_specificity"}
                else "missing_null_check"
            ),
            confidence=0.81,
            rationale="The response silently changes labels under shared-risk authority.",
            supporting_evidence_ids=["code-1"],
            resolved_dimensions=list(packet.disagreement.dimensions),
        )

    result = Stage3Controller(
        readiness=_ready,
        symptom_analyst=symptom,
        root_cause_analyst=lambda team_id, view: _root(),
        consistency_checker=lambda team_id, symptom, root, view: _consistent(),
        arbitrator=arbitrator,
    ).run(ledger)

    assert result.final_decision is None
    assert result.verification is None
    assert result.unresolved is not None
    assert result.unresolved.stop_reason == "arbitration_authority_violation"


def test_partial_stage3_resolution_fails_closed_as_authority_violation() -> None:
    def symptom(team_id: str, view: object) -> SymptomReport:
        return _symptom("unexpected_rejection" if team_id == "A" else "wrong_output")

    def root(team_id: str, view: object) -> RootCauseReport:
        return _root("missing_null_check" if team_id == "A" else "incorrect_condition")

    def arbitrator(packet: object) -> Stage3ArbitrationDecision:
        assert set(packet.disagreement.dimensions) == {
            "symptom_label",
            "root_cause_label",
        }
        return Stage3ArbitrationDecision(
            resolution_status=ResolutionStatus.RESOLVED,
            symptom_label="unexpected_rejection",
            root_cause_label="missing_null_check",
            confidence=0.82,
            rationale="The response accounts for only one of two active dimensions.",
            supporting_evidence_ids=["code-1"],
            resolved_dimensions=["symptom_label"],
        )

    result = Stage3Controller(
        readiness=_ready,
        symptom_analyst=symptom,
        root_cause_analyst=root,
        consistency_checker=lambda team_id, symptom, root, view: _consistent(),
        arbitrator=arbitrator,
    ).run(_ledger())

    assert result.final_decision is None
    assert result.verification is None
    assert result.unresolved is not None
    assert result.unresolved.stop_reason == "arbitration_authority_violation"
    assert "authority violation" in " ".join(result.unresolved.missing_facts)


def test_unlisted_stage3_unresolved_dimension_fails_closed_as_authority_violation() -> (
    None
):
    def symptom(team_id: str, view: object) -> SymptomReport:
        return _symptom("unexpected_rejection" if team_id == "A" else "wrong_output")

    def arbitrator(packet: object) -> Stage3ArbitrationDecision:
        return Stage3ArbitrationDecision(
            resolution_status=ResolutionStatus.UNRESOLVED,
            rationale="The response names a dimension outside its packet authority.",
            unresolved_dimensions=["invented_dimension"],
            missing_facts=["The symptom label disagreement remains unresolved."],
        )

    result = Stage3Controller(
        readiness=_ready,
        symptom_analyst=symptom,
        root_cause_analyst=lambda team_id, view: _root(),
        consistency_checker=lambda team_id, symptom, root, view: _consistent(),
        arbitrator=arbitrator,
    ).run(_ledger())

    assert result.final_decision is None
    assert result.verification is None
    assert result.unresolved is not None
    assert result.unresolved.stop_reason == "arbitration_authority_violation"
    assert "authority violation" in " ".join(result.unresolved.missing_facts)


def test_invented_stage3_alternative_fails_closed_before_arbitration() -> None:
    calls: list[object] = []

    def symptom(team_id: str, view: object) -> SymptomReport:
        report = _symptom()
        object.__setattr__(report, "alternative_label", "plausible_but_invented")
        object.__setattr__(report, "boundary_evidence_ids", ["code-1"])
        return report

    def arbitrator(packet: object) -> Stage3ArbitrationDecision:
        calls.append(packet)
        assert "unsupported_symptom_boundary" in packet.disagreement.dimensions
        return Stage3ArbitrationDecision(
            resolution_status=ResolutionStatus.UNRESOLVED,
            rationale="The alternative is not present in the frozen taxonomy.",
            unresolved_dimensions=["unsupported_symptom_boundary"],
            missing_facts=["A taxonomy-valid alternative boundary is required."],
        )

    result = Stage3Controller(
        readiness=_ready,
        symptom_analyst=symptom,
        root_cause_analyst=lambda team_id, view: _root(),
        consistency_checker=lambda team_id, symptom, root, view: _consistent(),
        arbitrator=arbitrator,
    ).run(_ledger())

    assert calls == []
    assert result.final_decision is None
    assert result.verification is None
    assert result.unresolved is not None
    assert result.unresolved.stop_reason == "stage3_candidate_invalid_taxonomy"
    assert "alternative" in " ".join(result.unresolved.missing_facts)


def _shared_risk_item(
    evidence_id: str,
    source_type: str,
    *,
    explicitness: EvidenceExplicitness = EvidenceExplicitness.DIRECT,
) -> EvidenceItem:
    content = f"Shared-risk packet evidence {evidence_id}."
    return EvidenceItem(
        evidence_id=evidence_id,
        record_id="record-1",
        source_type=source_type,
        source_uri=f"record:record-1#{evidence_id}",
        retrieved_at="2026-08-05T11:10:00Z",
        content=content,
        content_sha256=hashlib.sha256(content.encode()).hexdigest(),
        explicitness=explicitness,
    )


def test_symptom_only_shared_inference_packet_excludes_root_and_consistency() -> None:
    items = (
        _shared_risk_item(
            "sym-a", "issue_body", explicitness=EvidenceExplicitness.INFERRED
        ),
        _shared_risk_item(
            "sym-b", "issue_body", explicitness=EvidenceExplicitness.INFERRED
        ),
        _shared_risk_item("sym-boundary", "issue_body"),
        _shared_risk_item("root-direct", "code_diff"),
        _shared_risk_item("consistency-direct", "issue_body"),
    )
    ledger = EvidenceLedger(
        record_id="record-1",
        task="Classify the symptom and root cause.",
        taxonomy={
            "symptom": ["unexpected_rejection", "wrong_output"],
            "root_cause": ["missing_null_check", "incorrect_condition"],
        },
        domain_profile="issta2024",
        initial_items=items,
    )
    captured: dict[str, object] = {}

    def readiness(task: str, view: object) -> EvidenceReadinessReport:
        return EvidenceReadinessReport(
            task="stage3",
            dimensions=(
                DimensionReadiness(
                    dimension=EvidenceDimension.SYMPTOM,
                    sufficient=True,
                    confirmed_evidence_ids=("sym-boundary",),
                    missing_facts=(),
                    evidence_requests=(),
                ),
                DimensionReadiness(
                    dimension=EvidenceDimension.ROOT_CAUSE,
                    sufficient=True,
                    confirmed_evidence_ids=("root-direct",),
                    missing_facts=(),
                    evidence_requests=(),
                ),
            ),
        )

    def symptom(team_id: str, view: object) -> SymptomReport:
        return _symptom().model_copy(
            update={
                "supporting_evidence_ids": [f"sym-{team_id.lower()}"],
                "counter_evidence_ids": ["sym-boundary"],
                "boundary_evidence_ids": ["sym-boundary"],
            }
        )

    def root(team_id: str, view: object) -> RootCauseReport:
        return _root().model_copy(
            update={
                "supporting_evidence_ids": ["root-direct"],
                "boundary_evidence_ids": ["root-direct"],
            }
        )

    def consistency(
        team_id: str,
        symptom: SymptomReport,
        root: RootCauseReport,
        view: object,
    ) -> CausalConsistencyReport:
        return _consistent().model_copy(
            update={"supporting_evidence_ids": ["consistency-direct"]}
        )

    def arbitrator(packet: object) -> Stage3ArbitrationDecision:
        captured["details"] = packet.disagreement.details[
            "shared_unsupported_inference"
        ]
        captured["ids"] = tuple(item.evidence_id for item in packet.relevant_evidence)
        return Stage3ArbitrationDecision(
            resolution_status=ResolutionStatus.UNRESOLVED,
            rationale="The symptom claim remains supported only by inference.",
            unresolved_dimensions=["shared_unsupported_inference"],
            missing_facts=["Direct symptom evidence supporting the shared claim."],
        )

    Stage3Controller(
        readiness=readiness,
        symptom_analyst=symptom,
        root_cause_analyst=root,
        consistency_checker=consistency,
        arbitrator=arbitrator,
    ).run(ledger)

    assert captured["details"] == {
        "dimensions": ("symptom",),
        "evidence_ids": ("sym-a", "sym-b"),
    }
    assert captured["ids"] == ("sym-a", "sym-boundary", "sym-b")


def test_root_only_shared_inference_packet_excludes_symptom_and_consistency() -> None:
    items = (
        _shared_risk_item("symptom-direct", "issue_body"),
        _shared_risk_item(
            "root-a", "code_diff", explicitness=EvidenceExplicitness.INFERRED
        ),
        _shared_risk_item(
            "root-b", "code_diff", explicitness=EvidenceExplicitness.INFERRED
        ),
        _shared_risk_item("root-boundary", "code_diff"),
        _shared_risk_item("consistency-direct", "issue_body"),
    )
    ledger = EvidenceLedger(
        record_id="record-1",
        task="Classify the symptom and root cause.",
        taxonomy={
            "symptom": ["unexpected_rejection", "wrong_output"],
            "root_cause": ["missing_null_check", "incorrect_condition"],
        },
        domain_profile="issta2024",
        initial_items=items,
    )
    captured: dict[str, object] = {}

    def readiness(task: str, view: object) -> EvidenceReadinessReport:
        return EvidenceReadinessReport(
            task="stage3",
            dimensions=(
                DimensionReadiness(
                    dimension=EvidenceDimension.SYMPTOM,
                    sufficient=True,
                    confirmed_evidence_ids=("symptom-direct",),
                    missing_facts=(),
                    evidence_requests=(),
                ),
                DimensionReadiness(
                    dimension=EvidenceDimension.ROOT_CAUSE,
                    sufficient=True,
                    confirmed_evidence_ids=("root-boundary",),
                    missing_facts=(),
                    evidence_requests=(),
                ),
            ),
        )

    def symptom(team_id: str, view: object) -> SymptomReport:
        return _symptom().model_copy(
            update={
                "supporting_evidence_ids": ["symptom-direct"],
                "boundary_evidence_ids": ["symptom-direct"],
            }
        )

    def root(team_id: str, view: object) -> RootCauseReport:
        return _root().model_copy(
            update={
                "supporting_evidence_ids": [f"root-{team_id.lower()}"],
                "counter_evidence_ids": ["root-boundary"],
                "boundary_evidence_ids": ["root-boundary"],
            }
        )

    def consistency(
        team_id: str,
        symptom: SymptomReport,
        root: RootCauseReport,
        view: object,
    ) -> CausalConsistencyReport:
        return _consistent().model_copy(
            update={"supporting_evidence_ids": ["consistency-direct"]}
        )

    def arbitrator(packet: object) -> Stage3ArbitrationDecision:
        captured["details"] = packet.disagreement.details[
            "shared_unsupported_inference"
        ]
        captured["ids"] = tuple(item.evidence_id for item in packet.relevant_evidence)
        return Stage3ArbitrationDecision(
            resolution_status=ResolutionStatus.UNRESOLVED,
            rationale="The root-cause claim remains supported only by inference.",
            unresolved_dimensions=["shared_unsupported_inference"],
            missing_facts=["Direct mechanism evidence supporting the shared claim."],
        )

    Stage3Controller(
        readiness=readiness,
        symptom_analyst=symptom,
        root_cause_analyst=root,
        consistency_checker=consistency,
        arbitrator=arbitrator,
    ).run(ledger)

    assert captured["details"] == {
        "dimensions": ("root_cause",),
        "evidence_ids": ("root-a", "root-b"),
    }
    assert captured["ids"] == ("root-a", "root-boundary", "root-b")


def test_all_stage3_consumers_are_isolated_from_view_and_item_replacement() -> None:
    mutation_attempted = threading.Event()
    observed: list[tuple[str, str, int, tuple[str, ...], object]] = []
    issued_views: list[object] = []

    def observe(role: str, team_id: str, view: object) -> None:
        observed.append(
            (
                role,
                team_id,
                view.ledger_version,
                tuple(view.taxonomy["symptom"]),
                view.items[0].metadata,
            )
        )

    def symptom_analyst(team_id: str, view: object) -> SymptomReport:
        if team_id == "A":
            object.__setattr__(view, "ledger_version", 999)
            object.__setattr__(view, "taxonomy", {"symptom": ["tampered"]})
            object.__setattr__(view.items[0], "metadata", {"tampered": True})
            mutation_attempted.set()
        else:
            assert mutation_attempted.wait(timeout=2)
        observe("symptom", team_id, view)
        return _symptom("unexpected_rejection" if team_id == "A" else "wrong_output")

    def root_analyst(team_id: str, view: object) -> RootCauseReport:
        assert mutation_attempted.wait(timeout=2)
        observe("root", team_id, view)
        return _root()

    def checker(
        team_id: str,
        symptom: SymptomReport,
        root: RootCauseReport,
        view: object,
    ) -> CausalConsistencyReport:
        observe("checker", team_id, view)
        return _consistent()

    def arbitrator(packet: object) -> Stage3ArbitrationDecision:
        canonical_view = issued_views[-1]
        assert packet.classification_ledger_version == 1
        assert packet.relevant_evidence[0] is not canonical_view.items[0]
        assert (
            packet.relevant_evidence[0].metadata is not canonical_view.items[0].metadata
        )
        assert (
            packet.relevant_evidence[0].metadata["notes"]
            is not canonical_view.items[0].metadata["notes"]
        )
        assert tuple(packet.taxonomy["symptom"]) == (
            "unexpected_rejection",
            "wrong_output",
        )
        assert "injected" not in packet.taxonomy
        assert tuple(item.evidence_id for item in packet.relevant_evidence) == (
            "code-1",
        )
        assert tuple(packet.relevant_evidence[0].metadata["notes"]) == ("stable",)
        object.__setattr__(
            packet.relevant_evidence[0],
            "metadata",
            {"notes": ["arbitrator-tampered"]},
        )
        assert tuple(canonical_view.items[0].metadata["notes"]) == ("stable",)
        return Stage3ArbitrationDecision(
            resolution_status=ResolutionStatus.RESOLVED,
            symptom_label="unexpected_rejection",
            root_cause_label="missing_null_check",
            confidence=0.85,
            rationale="The unchanged frozen evidence resolves the symptom disagreement.",
            supporting_evidence_ids=["code-1"],
            resolved_dimensions=["symptom_label"],
        )

    ledger = _ledger(metadata={"notes": ["stable"]})
    original_view = ledger.view

    def tracked_view(evidence_ids: object = None) -> object:
        view = original_view(evidence_ids)
        issued_views.append(view)
        return view

    ledger.view = tracked_view
    result = Stage3Controller(
        readiness=_ready,
        symptom_analyst=symptom_analyst,
        root_cause_analyst=root_analyst,
        consistency_checker=checker,
        arbitrator=arbitrator,
    ).run(ledger)

    mutating_observation = next(
        item for item in observed if item[:2] == ("symptom", "A")
    )
    assert mutating_observation[2:] == (
        999,
        ("tampered",),
        {"tampered": True},
    )
    assert all(
        item[2:]
        == (
            1,
            ("unexpected_rejection", "wrong_output"),
            {
                "evidence_capabilities": (
                    "symptom_observation",
                    "defect_mechanism",
                ),
                "notes": ("stable",),
            },
        )
        for item in observed
        if item[:2] != ("symptom", "A")
    )
    assert result.classification_ledger_version == 1
    assert result.verification is not None
    assert result.verification.valid is True


def test_nonconsistent_team_report_cannot_pass_direct_consensus() -> None:
    def checker(
        team_id: str,
        symptom: SymptomReport,
        root: RootCauseReport,
        view: object,
    ) -> CausalConsistencyReport:
        if team_id == "A":
            return _consistent()
        return CausalConsistencyReport(
            status=ConsistencyStatus.CAUSE_REVIEW,
            rationale="The proposed mechanism does not yet explain the observed rejection.",
            supporting_evidence_ids=["code-1"],
        )

    result = Stage3Controller(
        readiness=_ready,
        symptom_analyst=lambda team_id, view: _symptom(),
        root_cause_analyst=lambda team_id, view: _root(),
        consistency_checker=checker,
    ).run(_ledger())

    assert result.final_decision is not None
    assert result.final_decision.source.value == "fallback_uncertain"
    assert result.verification is not None
    assert result.verification.valid is True
    assert result.unresolved is None


def test_matching_nonconsistent_reports_still_expose_an_arbitration_target() -> None:
    packets: list[object] = []

    def checker(
        team_id: str,
        symptom: SymptomReport,
        root: RootCauseReport,
        view: object,
    ) -> CausalConsistencyReport:
        return CausalConsistencyReport(
            status=ConsistencyStatus.CAUSE_REVIEW,
            rationale="The mechanism does not yet explain the observed rejection.",
            supporting_evidence_ids=["code-1"],
        )

    def arbitrator(packet: object) -> Stage3ArbitrationDecision:
        packets.append(packet)
        return Stage3ArbitrationDecision(
            resolution_status=ResolutionStatus.RESOLVED,
            symptom_label="unexpected_rejection",
            root_cause_label="missing_null_check",
            confidence=0.82,
            rationale="The shared evidence resolves the causal-consistency gate.",
            supporting_evidence_ids=["code-1"],
            resolved_dimensions=["failed_causal_consistency"],
        )

    Stage3Controller(
        readiness=_ready,
        symptom_analyst=lambda team_id, view: _symptom(),
        root_cause_analyst=lambda team_id, view: _root(),
        consistency_checker=checker,
        arbitrator=arbitrator,
    ).run(_ledger())

    assert packets[0].disagreement.dimensions == ("failed_causal_consistency",)
    assert packets[0].disagreement.requires_arbitration is True


def test_nonconsistent_reports_without_arbitrator_stop_as_unresolved() -> None:
    def checker(
        team_id: str,
        symptom: SymptomReport,
        root: RootCauseReport,
        view: object,
    ) -> CausalConsistencyReport:
        return CausalConsistencyReport(
            status=ConsistencyStatus.CAUSE_REVIEW,
            rationale="The mechanism does not yet explain the observed rejection.",
            supporting_evidence_ids=["code-1"],
        )

    result = Stage3Controller(
        readiness=_ready,
        symptom_analyst=lambda team_id, view: _symptom(),
        root_cause_analyst=lambda team_id, view: _root(),
        consistency_checker=checker,
    ).run(_ledger())

    assert result.final_decision is None
    assert result.verification is None
    assert result.unresolved is not None
    assert result.unresolved.stop_reason == "no_consistent_stage3_candidate"
    assert result.unresolved.dimensions == ("failed_causal_consistency",)
    assert "No Stage 3 team passed causal consistency." in (
        result.unresolved.missing_facts
    )


def test_nonconsistent_reports_with_unresolved_arbitration_stop_explicitly() -> None:
    def checker(
        team_id: str,
        symptom: SymptomReport,
        root: RootCauseReport,
        view: object,
    ) -> CausalConsistencyReport:
        return CausalConsistencyReport(
            status=ConsistencyStatus.SYMPTOM_REVIEW,
            rationale="The observed behavior still needs a boundary review.",
            supporting_evidence_ids=["code-1"],
        )

    def arbitrator(packet: object) -> Stage3ArbitrationDecision:
        return Stage3ArbitrationDecision(
            resolution_status=ResolutionStatus.UNRESOLVED,
            rationale="The packet cannot establish a consistent Stage 3 candidate.",
            unresolved_dimensions=list(packet.disagreement.dimensions),
            missing_facts=["A direct observation distinguishing the boundary."],
        )

    result = Stage3Controller(
        readiness=_ready,
        symptom_analyst=lambda team_id, view: _symptom(),
        root_cause_analyst=lambda team_id, view: _root(),
        consistency_checker=checker,
        arbitrator=arbitrator,
    ).run(_ledger())

    assert result.final_decision is None
    assert result.verification is None
    assert result.unresolved is not None
    assert result.unresolved.stop_reason == "no_consistent_stage3_candidate"
    assert result.unresolved.dimensions == ("failed_causal_consistency",)
    assert "No Stage 3 team passed causal consistency." in (
        result.unresolved.missing_facts
    )


def test_nonconsistent_reports_with_operational_arbitrator_failure_stop_explicitly() -> (
    None
):
    def checker(
        team_id: str,
        symptom: SymptomReport,
        root: RootCauseReport,
        view: object,
    ) -> CausalConsistencyReport:
        return CausalConsistencyReport(
            status=ConsistencyStatus.EVIDENCE_REQUEST,
            rationale="Additional evidence is required to establish the causal chain.",
            supporting_evidence_ids=["code-1"],
        )

    def arbitrator(packet: object) -> Stage3ArbitrationDecision:
        raise ModelTransportError("provider unavailable", attempts=6)

    result = Stage3Controller(
        readiness=_ready,
        symptom_analyst=lambda team_id, view: _symptom(),
        root_cause_analyst=lambda team_id, view: _root(),
        consistency_checker=checker,
        arbitrator=arbitrator,
    ).run(_ledger())

    assert result.final_decision is None
    assert result.verification is None
    assert result.unresolved is not None
    assert result.unresolved.stop_reason == "no_consistent_stage3_candidate"
    assert result.unresolved.dimensions == ("failed_causal_consistency",)
    assert "No Stage 3 team passed causal consistency." in (
        result.unresolved.missing_facts
    )
    assert result.component_failures[-1].role == "stage3_arbitrator"


def test_stage3_candidate_labels_outside_taxonomy_fail_closed() -> None:
    calls: list[object] = []

    def arbitrator(packet: object) -> Stage3ArbitrationDecision:
        calls.append(packet)
        return Stage3ArbitrationDecision(
            resolution_status=ResolutionStatus.RESOLVED,
            symptom_label="invented_symptom",
            root_cause_label="missing_null_check",
            confidence=0.8,
            rationale="The response preserves the proposed label for verification.",
            supporting_evidence_ids=["code-1"],
            resolved_dimensions=list(packet.disagreement.dimensions),
        )

    result = Stage3Controller(
        readiness=_ready,
        symptom_analyst=lambda team_id, view: _symptom("invented_symptom"),
        root_cause_analyst=lambda team_id, view: _root(),
        consistency_checker=lambda team_id, symptom, root, view: _consistent(),
        arbitrator=arbitrator,
    ).run(_ledger())

    assert calls == []
    assert result.final_decision is None
    assert result.verification is None
    assert result.unresolved is not None
    assert result.unresolved.stop_reason == "stage3_candidate_invalid_taxonomy"
    assert "invented_symptom" in " ".join(result.unresolved.missing_facts)


def test_stage3_gap_does_not_stop_both_teams_or_prediction() -> None:
    calls: list[tuple[str, str]] = []

    result = Stage3Controller(
        readiness=lambda task, view: _readiness_report(sufficient=False),
        symptom_analyst=lambda team, view: calls.append((team, "symptom"))
        or _symptom(),
        root_cause_analyst=lambda team, view: calls.append((team, "root")) or _root(),
        consistency_checker=lambda team, symptom, root, view: _consistent(),
    ).run(_ledger())

    assert sorted(calls) == [
        ("A", "root"),
        ("A", "symptom"),
        ("B", "root"),
        ("B", "symptom"),
    ]
    assert result.final_decision is not None
    assert result.readiness_report.dimensions[0].sufficient is False


@pytest.mark.parametrize("failed_team, surviving_team", [("A", "B"), ("B", "A")])
def test_single_complete_team_produces_verified_degraded_prediction(
    failed_team: str,
    surviving_team: str,
) -> None:
    def root(team_id: str, view: object) -> RootCauseReport:
        if team_id == failed_team:
            raise ValueError("root schema rejected completion")
        return _root()

    result = Stage3Controller(
        readiness=_ready,
        symptom_analyst=lambda team_id, view: _symptom(),
        root_cause_analyst=root,
        consistency_checker=lambda team_id, symptom, root, view: _consistent(),
    ).run(_ledger())

    assert [report.team_id for report in result.reports] == [surviving_team]
    assert result.final_decision is not None
    assert result.final_decision.symptom_label == "unexpected_rejection"
    assert result.final_decision.root_cause_label == "missing_null_check"
    assert result.final_decision.source.value == "single_team_degraded"
    assert result.verification is not None and result.verification.valid is True
    assert result.component_failures[0].team_id == failed_team
    assert result.component_failures[0].role == "root_cause_analyst"


def test_complementary_partial_teams_are_not_stitched_into_prediction() -> None:
    def symptom(team_id: str, view: object) -> SymptomReport:
        if team_id == "B":
            raise ValueError("symptom schema rejected completion")
        return _symptom()

    def root(team_id: str, view: object) -> RootCauseReport:
        if team_id == "A":
            raise ValueError("root schema rejected completion")
        return _root()

    result = Stage3Controller(
        readiness=_ready,
        symptom_analyst=symptom,
        root_cause_analyst=root,
        consistency_checker=lambda team_id, symptom, root, view: _consistent(),
    ).run(_ledger())

    assert result.reports == ()
    assert result.final_decision is None
    assert result.verification is None
    assert result.unresolved is not None
    assert result.unresolved.stop_reason == "stage3_component_failure"


def test_total_dual_team_failure_remains_invalid() -> None:
    def root(team_id: str, view: object) -> RootCauseReport:
        raise ValueError("root schema rejected completion")

    result = Stage3Controller(
        readiness=_ready,
        symptom_analyst=lambda team_id, view: _symptom(),
        root_cause_analyst=root,
        consistency_checker=lambda team_id, symptom, root, view: _consistent(),
    ).run(_ledger())

    assert result.reports == ()
    assert result.final_decision is None
    assert result.verification is None
    assert result.unresolved is not None
    assert result.unresolved.stop_reason == "stage3_component_failure"


def test_surviving_checker_failure_cannot_produce_degraded_prediction() -> None:
    def root(team_id: str, view: object) -> RootCauseReport:
        if team_id == "B":
            raise ValueError("root schema rejected completion")
        return _root()

    def checker(
        team_id: str,
        symptom: SymptomReport,
        root: RootCauseReport,
        view: object,
    ) -> CausalConsistencyReport:
        raise ValueError("consistency schema rejected completion")

    result = Stage3Controller(
        readiness=_ready,
        symptom_analyst=lambda team_id, view: _symptom(),
        root_cause_analyst=root,
        consistency_checker=checker,
    ).run(_ledger())

    assert result.final_decision is None
    assert result.verification is None
    assert result.unresolved is not None
    assert result.unresolved.stop_reason == "stage3_component_failure"
    assert {failure.role for failure in result.component_failures} == {
        "root_cause_analyst",
        "causal_consistency_checker",
    }


def test_asymmetric_checker_only_failure_preserves_other_verified_team() -> None:
    def checker(
        team_id: str,
        symptom: SymptomReport,
        root: RootCauseReport,
        view: object,
    ) -> CausalConsistencyReport:
        if team_id == "A":
            raise ValueError("consistency schema rejected completion")
        return _consistent()

    result = Stage3Controller(
        readiness=_ready,
        symptom_analyst=lambda team_id, view: _symptom(),
        root_cause_analyst=lambda team_id, view: _root(),
        consistency_checker=checker,
    ).run(_ledger())

    assert [report.team_id for report in result.reports] == ["B"]
    assert result.final_decision is not None
    assert result.final_decision.source.value == "single_team_degraded"
    assert result.final_decision.symptom_label == "unexpected_rejection"
    assert result.final_decision.root_cause_label == "missing_null_check"
    assert result.verification is not None and result.verification.valid is True

    assert result.component_failures[0].role == "causal_consistency_checker"
    assert result.component_failures[0].team_id == "A"


@pytest.mark.parametrize(
    ("role", "field"),
    [
        ("symptom", "supporting_evidence_ids"),
        ("symptom", "counter_evidence_ids"),
        ("symptom", "boundary_evidence_ids"),
        ("root_cause", "supporting_evidence_ids"),
        ("root_cause", "counter_evidence_ids"),
        ("root_cause", "boundary_evidence_ids"),
        ("consistency", "supporting_evidence_ids"),
    ],
)
def test_single_team_degraded_rejects_unknown_complete_report_citation_union(
    role: str,
    field: str,
) -> None:
    def root(team_id: str, view: object) -> RootCauseReport:
        if team_id == "B":
            raise ValueError("root schema rejected completion")
        return _root()

    def symptom(team_id: str, view: object) -> SymptomReport:
        report = _symptom()
        return (
            report.model_copy(update={field: ["unknown"]})
            if role == "symptom"
            else report
        )

    def surviving_root() -> RootCauseReport:
        report = _root()
        return (
            report.model_copy(update={field: ["unknown"]})
            if role == "root_cause"
            else report
        )

    def checker(
        team_id: str,
        symptom_report: SymptomReport,
        root_report: RootCauseReport,
        view: object,
    ) -> CausalConsistencyReport:
        report = _consistent()
        return (
            report.model_copy(update={field: ["unknown"]})
            if role == "consistency"
            else report
        )

    result = Stage3Controller(
        readiness=_ready,
        symptom_analyst=symptom,
        root_cause_analyst=lambda team_id, view: (
            root(team_id, view) if team_id == "B" else surviving_root()
        ),
        consistency_checker=checker,
    ).run(_ledger())

    assert [report.team_id for report in result.reports] == ["A"]
    assert result.final_decision is None
    assert result.verification is None
    assert result.unresolved is not None
    assert result.unresolved.stop_reason == "stage3_candidate_invalid_citation"
    assert result.unresolved.missing_facts == (
        "Candidate report cites unknown evidence_id 'unknown'.",
    )


def test_single_team_degraded_taxonomy_invalid_label_fails_closed() -> None:
    def root(team_id: str, view: object) -> RootCauseReport:
        if team_id == "B":
            raise ValueError("root schema rejected completion")
        return _root()

    result = Stage3Controller(
        readiness=_ready,
        symptom_analyst=lambda team_id, view: _symptom().model_copy(
            update={"label": "invented"}
        ),
        root_cause_analyst=root,
        consistency_checker=lambda team_id, symptom, root, view: _consistent(),
    ).run(_ledger())

    assert result.final_decision is None
    assert result.verification is None
    assert result.unresolved is not None
    assert result.unresolved.stop_reason == "stage3_candidate_invalid_taxonomy"
    assert "invented" in " ".join(result.unresolved.missing_facts)


def test_component_failure_audit_is_sanitized_without_raw_completion() -> None:
    raw_completion = '{"label":"SECRET_RAW_COMPLETION"}'

    def root(team_id: str, view: object) -> RootCauseReport:
        if team_id == "B":
            raise ValueError(raw_completion)
        return _root()

    result = Stage3Controller(
        readiness=_ready,
        symptom_analyst=lambda team_id, view: _symptom(),
        root_cause_analyst=root,
        consistency_checker=lambda team_id, symptom, root, view: _consistent(),
    ).run(_ledger())
    audit = workflow_audit_record(
        _ledger(),
        AdaptiveWorkflowResult(stage2=None, stage3=result, stop_reason=None),
    )

    failure = audit["stage3"]["component_failures"][0]
    assert failure["team_id"] == "B"
    assert failure["role"] == "root_cause_analyst"
    assert failure["error_type"] == "ValueError"
    assert raw_completion not in str(audit)


def test_component_failure_preserves_sanitized_schema_metadata() -> None:
    def root(team_id: str, view: object) -> RootCauseReport:
        if team_id == "B":
            raise StructuredOutputError(
                "schema output invalid",
                schema_name="RootCauseReport",
                validation_summary={"fields": ["label"], "codes": ["missing"]},
            )
        return _root()

    result = Stage3Controller(
        readiness=_ready,
        symptom_analyst=lambda team_id, view: _symptom(),
        root_cause_analyst=root,
        consistency_checker=lambda team_id, symptom, root, view: _consistent(),
    ).run(_ledger())

    failure = result.component_failures[0]
    assert failure.schema_name == "RootCauseReport"
    assert failure.validation_summary == {
        "fields": ("label",),
        "codes": ("missing",),
    }


def test_component_failure_drops_unrecognized_schema_name() -> None:
    secret = "SECRET_RAW_MODEL_COMPLETION"

    def root(team_id: str, view: object) -> RootCauseReport:
        if team_id == "B":
            raise StructuredOutputError(
                "schema output invalid",
                schema_name=secret,
            )
        return _root()

    result = Stage3Controller(
        readiness=_ready,
        symptom_analyst=lambda team_id, view: _symptom(),
        root_cause_analyst=root,
        consistency_checker=lambda team_id, symptom, root, view: _consistent(),
    ).run(_ledger())

    failure = result.component_failures[0]
    assert failure.schema_name is None
    assert secret not in str(failure.model_dump(mode="json"))


def test_component_failure_summary_drops_adversarial_raw_keys_and_values() -> None:
    secret = "SECRET_RAW_MODEL_COMPLETION"

    def root(team_id: str, view: object) -> RootCauseReport:
        if team_id == "B":
            raise StructuredOutputError(
                "schema output invalid",
                schema_name="RootCauseReport",
                validation_summary={
                    "fields": ["label", secret],
                    "codes": ["missing", secret],
                    "raw_completion": secret,
                    "content": {"prompt": secret},
                    secret: secret,
                },
            )
        return _root()

    result = Stage3Controller(
        readiness=_ready,
        symptom_analyst=lambda team_id, view: _symptom(),
        root_cause_analyst=root,
        consistency_checker=lambda team_id, symptom, root, view: _consistent(),
    ).run(_ledger())
    audit = workflow_audit_record(
        _ledger(),
        AdaptiveWorkflowResult(stage2=None, stage3=result, stop_reason=None),
    )

    failure = audit["stage3"]["component_failures"][0]
    assert failure["validation_summary"] == {
        "fields": ["label"],
        "codes": ["missing"],
    }
    assert secret not in str(audit)
    assert "raw_completion" not in str(audit)
    assert "prompt" not in str(audit)


def test_stage3_quarantined_items_never_reach_any_consumer() -> None:
    observed: list[object] = []
    request = EvidenceRequest(
        request_id="gap-1",
        missing_fact="Whether runtime behavior confirms the reported rejection symptom",
        why_needed="The readiness diagnostic records the missing runtime observation.",
        target_specialist=SpecialistType.ISSUE_PR,
        target_source="frozen issue history",
        query="runtime rejection symptom",
        expected_decision_impact=(
            "A matching observation strengthens symptom support; its absence remains "
            "a non-blocking Stage 3 evidence gap."
        ),
        max_items=1,
    )

    def readiness(task: str, view: object) -> EvidenceReadinessReport:
        observed.append(view)
        return EvidenceReadinessReport(
            task="stage3",
            dimensions=(
                DimensionReadiness(
                    dimension=EvidenceDimension.SYMPTOM,
                    sufficient=False,
                    confirmed_evidence_ids=(),
                    missing_facts=(
                        "A direct runtime symptom observation is unavailable.",
                    ),
                    evidence_requests=(request,),
                ),
                DimensionReadiness(
                    dimension=EvidenceDimension.ROOT_CAUSE,
                    sufficient=True,
                    confirmed_evidence_ids=("issue-1",),
                    missing_facts=(),
                    evidence_requests=(),
                ),
            ),
        )

    def specialist(
        request: EvidenceRequest,
        view: object,
    ) -> EvidenceDelta:
        observed.append(view)
        return EvidenceDelta(
            request_id=request.request_id,
            specialist=request.target_specialist,
            status=RetrievalStatus.UNAVAILABLE,
        )

    def symptom(team: str, view: object) -> SymptomReport:
        observed.append(view)
        return _symptom().model_copy(
            update={
                "supporting_evidence_ids": ["issue-1"],
                "boundary_evidence_ids": ["issue-1"],
            }
        )

    def root(team: str, view: object) -> RootCauseReport:
        observed.append(view)
        return _root().model_copy(
            update={
                "supporting_evidence_ids": ["issue-1"],
                "boundary_evidence_ids": ["issue-1"],
            }
        )

    def checker(
        team: str,
        symptom_report: SymptomReport,
        root_report: RootCauseReport,
        view: object,
    ) -> CausalConsistencyReport:
        observed.append(view)
        return _consistent().model_copy(update={"supporting_evidence_ids": ["issue-1"]})

    def challenger(reports: tuple[object, ...], view: object) -> BoundaryChallenge:
        observed.append(view)
        return BoundaryChallenge(
            action=BoundaryChallengeAction.PASS,
            rationale="The valid issue evidence supports the proposed boundaries.",
            cited_evidence_ids=["issue-1"],
        )

    ledger = _stage3_validity_ledger(include_valid=True)
    result = Stage3Controller(
        readiness=readiness,
        symptom_analyst=symptom,
        root_cause_analyst=root,
        consistency_checker=checker,
        boundary_challenger=challenger,
        specialists=SpecialistRegistry({SpecialistType.ISSUE_PR: specialist}),
    ).run(ledger)

    assert result.validity_report.valid_evidence_ids == ("issue-1",)
    assert result.validity_report.quarantined[0].evidence_id == "invalid-1"
    assert all(
        tuple(item.evidence_id for item in view.items) == ("issue-1",)
        for view in observed
    )
    audit = workflow_audit_record(
        ledger,
        AdaptiveWorkflowResult(stage2=None, stage3=result, stop_reason=None),
    )
    assert [item["evidence_id"] for item in audit["evidence_items"]] == [
        "issue-1",
        "invalid-1",
    ]
    assert (
        audit["stage3"]["validity_report"]["quarantined"][0]["evidence_id"]
        == "invalid-1"
    )


def test_zero_valid_stage3_evidence_stops_before_analysts() -> None:
    analyst_calls: list[str] = []

    result = Stage3Controller(
        readiness=lambda task, view: _readiness_report(sufficient=False),
        symptom_analyst=lambda team, view: analyst_calls.append("symptom"),
        root_cause_analyst=lambda team, view: analyst_calls.append("root"),
        consistency_checker=lambda team, symptom, root, view: analyst_calls.append(
            "checker"
        ),
    ).run(_stage3_validity_ledger(include_valid=False))

    assert analyst_calls == []
    assert result.reports == ()
    assert result.final_decision is None
    assert result.validity_report.valid_evidence_ids == ()
    assert result.unresolved is not None
    assert result.unresolved.stop_reason == "no_valid_stage3_evidence"


@pytest.mark.parametrize(
    "readiness_error",
    [
        StructuredOutputError(
            "readiness schema retries exhausted",
            schema_name="EvidenceReadinessReport",
        ),
        ModelTransportError("readiness transport retries exhausted", attempts=3),
    ],
)
def test_stage3_recoverable_readiness_failure_is_diagnostic_and_preserves_prediction(
    readiness_error: Exception,
) -> None:
    analyst_calls: list[tuple[str, str]] = []

    def readiness(task: str, view: object) -> EvidenceReadinessReport:
        raise readiness_error

    def symptom(team_id: str, view: object) -> SymptomReport:
        analyst_calls.append((team_id, "symptom"))
        return _symptom()

    def root(team_id: str, view: object) -> RootCauseReport:
        analyst_calls.append((team_id, "root_cause"))
        return _root()

    result = Stage3Controller(
        readiness=readiness,
        symptom_analyst=symptom,
        root_cause_analyst=root,
        consistency_checker=lambda team_id, symptom, root, view: _consistent(),
    ).run(_ledger())

    assert sorted(analyst_calls) == [
        ("A", "root_cause"),
        ("A", "symptom"),
        ("B", "root_cause"),
        ("B", "symptom"),
    ]
    assert result.final_decision is not None
    assert result.readiness_report is not None
    assert all(
        dimension.sufficient is False
        for dimension in result.readiness_report.dimensions
    )
    assert all(
        "readiness assessment unavailable" in dimension.missing_facts[0]
        for dimension in result.readiness_report.dimensions
    )
    audit = workflow_audit_record(
        _ledger(),
        AdaptiveWorkflowResult(stage2=None, stage3=result, stop_reason=None),
    )
    assert all(
        dimension["sufficient"] is False
        for dimension in audit["stage3"]["readiness_report"]["dimensions"]
    )


def test_stage3_programmer_error_in_readiness_remains_strict() -> None:
    with pytest.raises(AssertionError, match="programmer defect"):
        Stage3Controller(
            readiness=lambda task, view: (_ for _ in ()).throw(
                AssertionError("programmer defect")
            ),
            symptom_analyst=lambda team_id, view: _symptom(),
            root_cause_analyst=lambda team_id, view: _root(),
            consistency_checker=lambda team_id, symptom, root, view: _consistent(),
        ).run(_ledger())


def test_stage3_readiness_rechecks_after_appended_evidence_before_classifying() -> None:
    readiness_versions: list[int] = []
    readiness_evidence_ids: list[tuple[str, ...]] = []
    classification_versions: list[int] = []
    classification_evidence_ids: list[tuple[str, ...]] = []
    request = EvidenceRequest(
        request_id="taxonomy-1",
        missing_fact="Whether the taxonomy source defines the observed failure symptom",
        why_needed="A definition is needed to establish that symptom evidence is ready.",
        target_specialist=SpecialistType.TAXONOMY_KNOWLEDGE,
        target_source="supplied taxonomy",
        query="record-1 symptom definition",
        expected_decision_impact=(
            "A matching definition makes symptom evidence ready; absence leaves the "
            "classification unresolved without choosing a label."
        ),
        max_items=1,
    )

    def readiness(task: str, view: object) -> EvidenceReadinessReport:
        readiness_versions.append(view.ledger_version)
        readiness_evidence_ids.append(tuple(item.evidence_id for item in view.items))
        if view.ledger_version > 1:
            return _readiness_report()
        report = _readiness_report(sufficient=False)
        symptom = report.dimensions[0].model_copy(
            update={"evidence_requests": (request,)}
        )
        return report.model_copy(update={"dimensions": (symptom, report.dimensions[1])})

    def specialist(request: EvidenceRequest, view: object) -> EvidenceDelta:
        content = "The supplied taxonomy defines unexpected rejection as a symptom."
        item = EvidenceItem(
            evidence_id="taxonomy-1",
            record_id="record-1",
            source_type="issue_body",
            source_uri="record:record-1#taxonomy",
            retrieved_at="2026-08-05T11:05:00Z",
            content=content,
            content_sha256=hashlib.sha256(content.encode()).hexdigest(),
            explicitness=EvidenceExplicitness.DIRECT,
        )
        return EvidenceDelta(
            request_id=request.request_id,
            specialist=request.target_specialist,
            status=RetrievalStatus.FOUND,
            items=[item],
        )

    def symptom(team_id: str, view: object) -> SymptomReport:
        classification_versions.append(view.ledger_version)
        classification_evidence_ids.append(
            tuple(item.evidence_id for item in view.items)
        )
        return _symptom()

    def root(team_id: str, view: object) -> RootCauseReport:
        classification_versions.append(view.ledger_version)
        classification_evidence_ids.append(
            tuple(item.evidence_id for item in view.items)
        )
        return _root()

    def checker(
        team_id: str,
        symptom: SymptomReport,
        root: RootCauseReport,
        view: object,
    ) -> CausalConsistencyReport:
        classification_versions.append(view.ledger_version)
        classification_evidence_ids.append(
            tuple(item.evidence_id for item in view.items)
        )
        return _consistent()

    result = Stage3Controller(
        readiness=readiness,
        symptom_analyst=symptom,
        root_cause_analyst=root,
        consistency_checker=checker,
        specialists=SpecialistRegistry({SpecialistType.TAXONOMY_KNOWLEDGE: specialist}),
    ).run(_ledger())

    assert readiness_versions == [1, 2]
    assert readiness_evidence_ids == [
        ("code-1",),
        ("code-1", "taxonomy-1"),
    ]
    assert classification_versions == [2, 2, 2, 2, 2, 2]
    assert classification_evidence_ids == [("code-1", "taxonomy-1")] * 6
    assert result.classification_ledger_version == 2
    assert result.validity_report.valid_evidence_ids == ("code-1", "taxonomy-1")
    assert [delta.request_id for delta in result.evidence_deltas] == ["taxonomy-1"]


def test_stage3_retrieval_is_idempotent_for_the_same_frozen_passage() -> None:
    requests = tuple(
        EvidenceRequest(
            request_id=f"passage-request-{index}",
            missing_fact=(
                "Whether the same frozen issue passage confirms the rejection symptom"
            ),
            why_needed=(
                "The passage is independently requested by two readiness dimensions."
            ),
            target_specialist=SpecialistType.ISSUE_PR,
            target_source="issue body",
            query=f"rejection symptom confirmation {index}",
            expected_decision_impact=(
                "A matching frozen passage strengthens the cited symptom evidence; "
                "absence leaves the diagnostic gap unchanged."
            ),
            max_items=1,
        )
        for index in (1, 2)
    )
    readiness_calls = 0

    def readiness(task: str, view: object) -> EvidenceReadinessReport:
        nonlocal readiness_calls
        readiness_calls += 1
        if readiness_calls > 1:
            return _readiness_report()
        report = _readiness_report(sufficient=False)
        symptom = report.dimensions[0].model_copy(
            update={"evidence_requests": requests}
        )
        return report.model_copy(update={"dimensions": (symptom, report.dimensions[1])})

    def specialist(request: EvidenceRequest, view: object) -> EvidenceDelta:
        content = "The same frozen passage reports that valid requests are rejected."
        item = EvidenceItem(
            evidence_id="source-shared-passage",
            record_id="record-1",
            source_type="issue_body",
            source_uri="https://example.test/issues/1",
            retrieved_at="2026-08-05T11:05:00Z",
            content=content,
            content_sha256=hashlib.sha256(content.encode()).hexdigest(),
            explicitness=EvidenceExplicitness.DIRECT,
            metadata={
                "request_id": request.request_id,
                "query": request.query,
                "frozen_source_type": "issue_body",
                "frozen_source_sha256": hashlib.sha256(content.encode()).hexdigest(),
                "passage_index": 0,
            },
        )
        return EvidenceDelta(
            request_id=request.request_id,
            specialist=request.target_specialist,
            status=RetrievalStatus.FOUND,
            items=[item],
        )

    ledger = _ledger()
    result = Stage3Controller(
        readiness=readiness,
        symptom_analyst=lambda team, view: _symptom(),
        root_cause_analyst=lambda team, view: _root(),
        consistency_checker=lambda team, symptom, root, view: _consistent(),
        specialists=SpecialistRegistry({SpecialistType.ISSUE_PR: specialist}),
    ).run(ledger)

    assert ledger.evidence_ids == ("code-1", "source-shared-passage")
    assert ledger.version == 2
    assert result.validity_report.valid_evidence_ids == (
        "code-1",
        "source-shared-passage",
    )
    assert [delta.request_id for delta in result.evidence_deltas] == [
        "passage-request-1",
        "passage-request-2",
    ]


def test_stage3_retrieval_rejects_a_conflicting_duplicate_evidence_id() -> None:
    requests = tuple(
        EvidenceRequest(
            request_id=f"collision-request-{index}",
            missing_fact="Whether the frozen issue passage confirms the rejection symptom",
            why_needed="Two independent requests exercise evidence identity admission.",
            target_specialist=SpecialistType.ISSUE_PR,
            target_source="issue body",
            query=f"rejection evidence variant {index}",
            expected_decision_impact=(
                "A matching passage strengthens the symptom evidence; a conflicting "
                "identity must fail closed before classification."
            ),
            max_items=1,
        )
        for index in (1, 2)
    )

    def readiness(task: str, view: object) -> EvidenceReadinessReport:
        report = _readiness_report(sufficient=False)
        symptom = report.dimensions[0].model_copy(
            update={"evidence_requests": requests}
        )
        return report.model_copy(update={"dimensions": (symptom, report.dimensions[1])})

    def specialist(request: EvidenceRequest, view: object) -> EvidenceDelta:
        content = f"Conflicting frozen content for {request.request_id}."
        return EvidenceDelta(
            request_id=request.request_id,
            specialist=request.target_specialist,
            status=RetrievalStatus.FOUND,
            items=[
                EvidenceItem(
                    evidence_id="source-collision",
                    record_id="record-1",
                    source_type="issue_body",
                    source_uri="https://example.test/issues/1",
                    retrieved_at="2026-08-05T11:05:00Z",
                    content=content,
                    content_sha256=hashlib.sha256(content.encode()).hexdigest(),
                    explicitness=EvidenceExplicitness.DIRECT,
                )
            ],
        )

    with pytest.raises(ValueError, match="conflicts with existing evidence"):
        Stage3Controller(
            readiness=readiness,
            symptom_analyst=lambda team, view: _symptom(),
            root_cause_analyst=lambda team, view: _root(),
            consistency_checker=lambda team, symptom, root, view: _consistent(),
            specialists=SpecialistRegistry({SpecialistType.ISSUE_PR: specialist}),
        ).run(_ledger())


def test_stage3_retrieval_quarantines_wrong_record_without_losing_valid_items() -> None:
    request = EvidenceRequest(
        request_id="mixed-records-1",
        missing_fact="Whether issue history confirms the reported rejection symptom",
        why_needed="The gap diagnostic requests a frozen same-record confirmation.",
        target_specialist=SpecialistType.ISSUE_PR,
        target_source="frozen issue history",
        query="rejection symptom confirmation",
        expected_decision_impact=(
            "A same-record confirmation strengthens the symptom claim; unavailable or "
            "invalid material remains a non-blocking diagnostic gap."
        ),
        max_items=2,
    )
    readiness_views: list[tuple[str, ...]] = []
    analyst_views: list[tuple[str, ...]] = []
    analyst_calls: list[tuple[str, str]] = []

    def readiness(task: str, view: object) -> EvidenceReadinessReport:
        readiness_views.append(tuple(item.evidence_id for item in view.items))
        report = _readiness_report(sufficient=False)
        symptom = report.dimensions[0].model_copy(
            update={"evidence_requests": (request,)}
        )
        return report.model_copy(update={"dimensions": (symptom, report.dimensions[1])})

    def retrieved_item(evidence_id: str, record_id: str) -> EvidenceItem:
        content = f"Retrieved confirmation for {record_id}."
        return EvidenceItem(
            evidence_id=evidence_id,
            record_id=record_id,
            source_type="issue_body",
            source_uri=f"https://example.test/issues/{record_id}",
            retrieved_at="2026-08-05T11:05:00Z",
            content=content,
            content_sha256=hashlib.sha256(content.encode()).hexdigest(),
            explicitness=EvidenceExplicitness.DIRECT,
        )

    def specialist(request: EvidenceRequest, view: object) -> EvidenceDelta:
        return EvidenceDelta(
            request_id=request.request_id,
            specialist=request.target_specialist,
            status=RetrievalStatus.FOUND,
            items=[
                retrieved_item("issue-2", "record-1"),
                retrieved_item("wrong-record-1", "record-elsewhere"),
            ],
        )

    def symptom(team: str, view: object) -> SymptomReport:
        analyst_calls.append((team, "symptom"))
        analyst_views.append(tuple(item.evidence_id for item in view.items))
        return _symptom()

    def root(team: str, view: object) -> RootCauseReport:
        analyst_calls.append((team, "root"))
        analyst_views.append(tuple(item.evidence_id for item in view.items))
        return _root()

    ledger = _ledger()
    result = Stage3Controller(
        readiness=readiness,
        symptom_analyst=symptom,
        root_cause_analyst=root,
        consistency_checker=lambda team, symptom, root, view: _consistent(),
        specialists=SpecialistRegistry({SpecialistType.ISSUE_PR: specialist}),
    ).run(ledger)

    assert sorted(analyst_calls) == [
        ("A", "root"),
        ("A", "symptom"),
        ("B", "root"),
        ("B", "symptom"),
    ]
    assert ledger.evidence_ids == ("code-1", "issue-2")
    assert readiness_views == [("code-1",), ("code-1", "issue-2")]
    assert analyst_views == [("code-1", "issue-2")] * 4
    assert result.validity_report.valid_evidence_ids == ("code-1", "issue-2")
    assert result.validity_report.quarantined[0].evidence_id == "wrong-record-1"
    assert result.validity_report.quarantined[0].reasons == ("record_id_mismatch",)


@pytest.mark.parametrize(
    "config",
    [
        Stage3WorkflowConfig(max_retrieval_rounds=0),
        Stage3WorkflowConfig(max_specialist_calls=0),
    ],
    ids=["retrieval-rounds", "specialist-calls"],
)
def test_stage3_retrieval_budget_exhaustion_keeps_both_teams_running(
    config: Stage3WorkflowConfig,
) -> None:
    request = EvidenceRequest(
        request_id="budget-gap-1",
        missing_fact="Whether a frozen execution trace confirms the rejection symptom",
        why_needed="The diagnostic records the absent direct execution observation.",
        target_specialist=SpecialistType.TEST_EVIDENCE,
        target_source="frozen execution evidence",
        query="execution rejection symptom",
        expected_decision_impact=(
            "A matching trace strengthens the symptom evidence; exhausted retrieval "
            "leaves a recorded non-blocking gap."
        ),
        max_items=1,
    )
    calls: list[tuple[str, str]] = []

    def readiness(task: str, view: object) -> EvidenceReadinessReport:
        report = _readiness_report(sufficient=False)
        symptom = report.dimensions[0].model_copy(
            update={"evidence_requests": (request,)}
        )
        return report.model_copy(update={"dimensions": (symptom, report.dimensions[1])})

    result = Stage3Controller(
        readiness=readiness,
        symptom_analyst=lambda team, view: calls.append((team, "symptom"))
        or _symptom(),
        root_cause_analyst=lambda team, view: calls.append((team, "root")) or _root(),
        consistency_checker=lambda team, symptom, root, view: _consistent(),
        config=config,
    ).run(_ledger())

    assert sorted(calls) == [
        ("A", "root"),
        ("A", "symptom"),
        ("B", "root"),
        ("B", "symptom"),
    ]
    assert result.budget_exhausted is True
    assert result.readiness_report.dimensions[0].sufficient is False


def _anchor(team_id: str) -> JointAnchorReport:
    return JointAnchorReport(
        symptom=_symptom(),
        root_cause=_root(),
        causal_account=(
            f"Team {team_id} links the missing null guard to the observed rejection."
        ),
        shared_supporting_evidence_ids=["code-1"],
    )


def _verification(
    dimension: EvidenceDimension,
    *,
    anchor_label: str | None = None,
    verdict: VerificationVerdict = VerificationVerdict.ACCEPT,
) -> DimensionVerificationReport:
    expected_label = (
        "unexpected_rejection"
        if dimension is EvidenceDimension.SYMPTOM
        else "missing_null_check"
    )
    alternative_label = (
        "wrong_output"
        if dimension is EvidenceDimension.SYMPTOM
        else "incorrect_condition"
    )
    return DimensionVerificationReport(
        dimension=dimension,
        verdict=verdict,
        anchor_label=anchor_label or expected_label,
        alternative_label=(
            alternative_label if verdict is VerificationVerdict.REJECT else None
        ),
        rationale="The independently reviewed evidence supports this bounded verdict.",
        supporting_evidence_ids=["code-1"],
        corrected_claim=(
            "The request completes with an incorrect observable output."
            if verdict is VerificationVerdict.REJECT
            else None
        ),
        corrected_causal_chain=(
            [
                "The null entry reaches an incorrect condition.",
                "The condition selects the wrong processing branch.",
                "The request produces the observed failure.",
            ]
            if verdict is VerificationVerdict.REJECT
            and dimension is EvidenceDimension.ROOT_CAUSE
            else []
        ),
        confidence=0.84,
    )


def _anchored_controller(**overrides: object) -> Stage3Controller:
    options: dict[str, object] = {
        "readiness": _ready,
        "joint_anchor": lambda team_id, view: _anchor(team_id),
        "symptom_verifier": lambda team_id, anchor, view: _verification(
            EvidenceDimension.SYMPTOM
        ),
        "root_cause_verifier": lambda team_id, anchor, view: _verification(
            EvidenceDimension.ROOT_CAUSE
        ),
        "consistency_checker": (lambda team_id, symptom, root, view: _consistent()),
    }
    options.update(overrides)
    return Stage3Controller(**options)


def _preservation_structure() -> TaxonomyStructure:
    def node(label: str, dimension: str, neighbor: str) -> TaxonomyNode:
        return TaxonomyNode(
            label=label,
            dimension=dimension,
            definition=f"Definition for {label}.",
            semantic_origin=TaxonomySemanticOrigin.OPERATIONAL_DEFINITION,
            abstraction_level=(
                "observable_outcome" if dimension == "symptom" else "root_mechanism"
            ),
            responsibility_scope="runtime",
            concept_kind="outcome" if dimension == "symptom" else "mechanism",
            nearest_neighbors=(neighbor,),
        )

    def card(card_id: str, dimension: str, first: str, second: str) -> BoundaryCard:
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

    return TaxonomyStructure(
        schema_version=1,
        domain="issta2024",
        nodes=(
            node("unexpected_rejection", "symptom", "wrong_output"),
            node("wrong_output", "symptom", "unexpected_rejection"),
            node("missing_null_check", "root_cause", "incorrect_condition"),
            node("incorrect_condition", "root_cause", "missing_null_check"),
        ),
        boundary_cards=(
            card("symptom-boundary", "symptom", "unexpected_rejection", "wrong_output"),
            card(
                "root-boundary",
                "root_cause",
                "missing_null_check",
                "incorrect_condition",
            ),
        ),
    )


def _baseline_anchor() -> BaselineAnchor:
    return BaselineAnchor(
        record_id="record-1",
        valid=True,
        symptom_label="unexpected_rejection",
        root_cause_label="missing_null_check",
        source_config_hash="a" * 64,
        source_predictions_sha256="b" * 64,
    )


def _preservation_controller(**overrides: object) -> Stage3Controller:
    options: dict[str, object] = {
        "baseline_anchor": _baseline_anchor(),
        "taxonomy_structure": _preservation_structure(),
        "baseline_revision_assessor": lambda team, anchor, structure, view, dimension: None,
        "baseline_revision_consistency": lambda team, assessment, label, structure, view: None,
        "expected_baseline_config_hash": "a" * 64,
        "expected_baseline_predictions_sha256": "b" * 64,
        "expected_baseline_anchor_hash": baseline_anchor_hash(_baseline_anchor()),
        "expected_taxonomy_structure_hash": taxonomy_structure_hash(
            _preservation_structure()
        ),
    }
    options.update(overrides)
    if "expected_baseline_anchor_hash" not in overrides:
        options["expected_baseline_anchor_hash"] = baseline_anchor_hash(
            options["baseline_anchor"]
        )
    if "expected_taxonomy_structure_hash" not in overrides:
        options["expected_taxonomy_structure_hash"] = taxonomy_structure_hash(
            options["taxonomy_structure"]
        )
    return _anchored_controller(**options)


def _sealed_preservation_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[object, str, str, BaselineAnchor, TaxonomyStructure]:
    record_id = "issta2024:owner/repo:92"
    commit = "8" * 40
    content = "- choose_wrong_branch()\n+ choose_correct_branch()"
    content_sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
    repair = FrozenGraphNode.model_construct(
        node_id=frozen_graph_node_id(
            record_id=record_id,
            node_type=FrozenNodeType.CHANGED_CODE,
            canonical_uri=f"https://github.com/owner/repo/blob/{commit}/src/core.py",
            immutable_ref=commit,
            content_sha256=content_sha256,
        ),
        record_id=record_id,
        node_type=FrozenNodeType.CHANGED_CODE,
        canonical_uri=f"https://github.com/owner/repo/blob/{commit}/src/core.py",
        repository="owner/repo",
        relation_depth=1,
        intrinsic_parent_node_id=None,
        retrieved_at="2026-08-17T00:00:00+00:00",
        remote_updated_at="2026-08-16T00:00:00+00:00",
        immutable_ref=commit,
        raw_blob_sha256="a" * 64,
        content_sha256=content_sha256,
        content=content,
        evidence_capabilities=("defect_mechanism",),
        authority=EvidenceAuthority.DIRECT,
        metadata=MappingProxyType({}),
    )
    graph = FrozenRecordGraph.model_construct(
        schema_version="ase-frozen-record-graph-v1",
        domain="issta2024",
        record_id=record_id,
        repository="owner/repo",
        seed_node_id=repair.node_id,
        nodes=(repair,),
        edges=(),
        capture_status="captured",
        unavailable_reasons=(),
        graph_sha256="b" * 64,
    )
    manifest = type("Manifest", (), {"bundle_merkle_root": "c" * 64})()
    bundle = FrozenEvidenceGraphBundle.model_construct(
        domain="issta2024",
        bundle_id="controller-candidate-red-v1",
        split_id="controller-candidate-red",
        policy=type("Policy", (), {"policy_id": "one-hop-v2"})(),
        graphs=MappingProxyType({record_id: graph}),
        record_ids=(record_id,),
        manifest=manifest,
        trust_manifest_sha256="d" * 64,
        trust_manifest_relative_path="registered/manifest.json",
    )
    monkeypatch.setattr(
        frozen_evidence_runtime,
        "load_bound_frozen_evidence_graph_for_domain",
        lambda domain, repository_root=None: bundle,
    )
    projection = project_frozen_evidence_for_record(
        load_registered_frozen_evidence_runtime("issta2024"), record_id
    )
    runtime = build_record_runtime(
        {"record_id": record_id, "title": "Synthetic controller candidate recall"},
        taxonomy={
            "symptom": ["unexpected_rejection", "wrong_output"],
            "root_cause": ["missing_null_check", "incorrect_condition"],
        },
        domain="issta2024",
        frozen_evidence_projection=projection,
    )
    symptom_id = next(
        item.evidence_id
        for item in runtime.ledger.view().items
        if item.evidence_id != repair.node_id
    )
    anchor = _baseline_anchor().model_copy(update={"record_id": record_id})
    structure = _preservation_structure()
    return runtime, symptom_id, repair.node_id, anchor, structure


@pytest.mark.parametrize(
    ("challenge_mode", "expected_calls"),
    (("valid", 2), ("missing", 0), ("invalid", 0), ("component_failure", 0)),
)
def test_real_run_routes_exact_boundary_challenge_into_preservation_builder(
    monkeypatch: pytest.MonkeyPatch,
    challenge_mode: str,
    expected_calls: int,
) -> None:
    runtime, symptom_id, repair_id, anchor, structure = _sealed_preservation_runtime(
        monkeypatch
    )
    assessment_proposals: list[RevisionProposalEnvelope] = []

    def joint_anchor(team_id: str, view: object) -> JointAnchorReport:
        return JointAnchorReport(
            symptom=_symptom().model_copy(
                update={
                    "supporting_evidence_ids": (symptom_id,),
                    "boundary_evidence_ids": (),
                }
            ),
            root_cause=_root().model_copy(
                update={
                    "supporting_evidence_ids": (repair_id,),
                    "boundary_evidence_ids": (),
                }
            ),
            causal_account="The immutable repair explains the observed issue behavior.",
            shared_supporting_evidence_ids=(repair_id,),
        )

    def verify_dimension(
        dimension: EvidenceDimension,
        evidence_id: str,
    ) -> DimensionVerificationReport:
        return _verification(dimension).model_copy(
            update={"supporting_evidence_ids": (evidence_id,)}
        )

    def challenger(reports: object, view: object) -> BoundaryChallenge:
        if challenge_mode == "component_failure":
            raise StructuredOutputError(
                "boundary challenge invalid",
                schema_name="BoundaryChallenge",
                validation_summary={"codes": ["invalid"]},
            )
        return BoundaryChallenge(
            action=(
                BoundaryChallengeAction.CAUSE_REVIEW
                if challenge_mode in {"valid", "invalid"}
                else BoundaryChallengeAction.PASS
            ),
            rationale="Route only a validated, team-nominated root-cause alternative.",
            cited_evidence_ids=(
                (repair_id,)
                if challenge_mode == "valid"
                else (("unknown-id",) if challenge_mode == "invalid" else ())
            ),
        )

    def assess(
        team_id: str,
        proposal: RevisionProposalEnvelope,
        baseline: BaselineAnchor,
        card: BoundaryCard,
        view: object,
    ) -> BaselineRevisionAssessment:
        assessment_proposals.append(proposal)
        return _heterogeneous_candidate_revision_assessment(
            team_id,
            proposal,
            baseline,
            card,
            view,
            falsified=team_id == "B",
        )

    controller = Stage3Controller(
        readiness=lambda task, view: EvidenceReadinessReport(
            task="stage3",
            dimensions=(
                DimensionReadiness(
                    dimension=EvidenceDimension.SYMPTOM,
                    sufficient=True,
                    confirmed_evidence_ids=(symptom_id,),
                    missing_facts=(),
                    evidence_requests=(),
                ),
                DimensionReadiness(
                    dimension=EvidenceDimension.ROOT_CAUSE,
                    sufficient=True,
                    confirmed_evidence_ids=(repair_id,),
                    missing_facts=(),
                    evidence_requests=(),
                ),
            ),
        ),
        joint_anchor=joint_anchor,
        symptom_verifier=lambda team, report, view: verify_dimension(
            EvidenceDimension.SYMPTOM, symptom_id
        ),
        root_cause_verifier=lambda team, report, view: verify_dimension(
            EvidenceDimension.ROOT_CAUSE, repair_id
        ),
        consistency_checker=lambda team, symptom, root, view: _consistent().model_copy(
            update={"supporting_evidence_ids": (repair_id,)}
        ),
        boundary_challenger=challenger,
        baseline_anchor=anchor,
        taxonomy_structure=structure,
        baseline_revision_assessor=assess,
        baseline_revision_consistency=lambda *args: pytest.fail(
            "preserve assessments must not invoke cross consistency"
        ),
        expected_baseline_config_hash=anchor.source_config_hash,
        expected_baseline_predictions_sha256=anchor.source_predictions_sha256,
        expected_baseline_anchor_hash=baseline_anchor_hash(anchor),
        expected_taxonomy_structure_hash=taxonomy_structure_hash(structure),
    )

    result = controller.run(runtime.ledger)

    assert len(assessment_proposals) == expected_calls
    if challenge_mode == "valid":
        assert {proposal.proposed_label for proposal in assessment_proposals} == {
            "incorrect_condition"
        }
        assert result.boundary_challenge is not None
        assert result.verification is not None and result.verification.valid is True
    elif challenge_mode == "invalid":
        assert result.unresolved is not None
        assert result.unresolved.stop_reason == "boundary_challenger_invalid_citation"
    elif challenge_mode == "component_failure":
        assert any(
            failure.role == "boundary_challenger"
            for failure in result.component_failures
        )


def _candidate_revision_assessment(
    team_id: str,
    proposal: RevisionProposalEnvelope,
    anchor: BaselineAnchor,
    card: BoundaryCard,
    view: object,
) -> BaselineRevisionAssessment:
    return BaselineRevisionAssessment(
        assessor_team_id=team_id,
        proposal_digest=revision_proposal_digest(proposal),
        dimension=proposal.dimension,
        verdict=RevisionAssessmentVerdict.REVISE,
        baseline_label=proposal.baseline_label,
        proposed_label=proposal.proposed_label,
        boundary_card_id=card.card_id,
        contradicted_baseline_condition="condition for wrong_output",
        satisfied_proposed_conditions=("condition for wrong_output",),
        supporting_evidence_ids=("code-1",),
        counter_evidence_ids=("code-1",),
        baseline_source_config_hash=anchor.source_config_hash,
        baseline_source_predictions_sha256=anchor.source_predictions_sha256,
        taxonomy_structure_hash=proposal.taxonomy_structure_hash,
        evidence_view_hash=proposal.evidence_view_hash,
    )


def _heterogeneous_candidate_revision_assessment(
    team_id: str,
    proposal: RevisionProposalEnvelope,
    anchor: BaselineAnchor,
    card: BoundaryCard,
    view: object,
    *,
    falsified: bool = False,
) -> BaselineRevisionAssessment:
    criteria = {criterion.label: criterion for criterion in card.criteria}
    proposed = criteria[proposal.proposed_label]
    baseline = criteria[proposal.baseline_label]
    citation = proposal.supporting_evidence_ids[0]
    if team_id == "A":
        return normalize_revision_entailment_result(
            result=RevisionEntailmentResult.model_validate(
                {
                    "condition_findings": [
                        {
                            "condition": condition,
                            "status": "supported",
                            "citation_ids": [citation],
                        }
                        for condition in proposed.positive_conditions
                    ],
                    "baseline_exclusion_finding": {
                        "condition": baseline.exclusion_conditions[0],
                        "status": "supported",
                        "citation_ids": [citation],
                    },
                    "outcome": "entailed",
                    "summary": "The direct observation entails the complete candidate boundary.",
                }
            ),
            proposal=proposal,
            routed_card=card,
        )
    return normalize_revision_falsification_result(
        result=RevisionFalsificationResult.model_validate(
            {
                "baseline_survival_finding": {
                    "condition": baseline.positive_conditions[0],
                    "status": "supported" if falsified else "refuted",
                    "citation_ids": [citation],
                },
                "proposed_defeater_findings": [
                    {
                        "condition": condition,
                        "status": "refuted",
                        "citation_ids": [citation],
                    }
                    for condition in proposed.exclusion_conditions
                ],
                "strongest_competing_reading": {
                    "label": proposal.baseline_label,
                    "summary": "The Baseline is the strongest exact-pair competing reading.",
                    "citation_ids": [citation],
                },
                "outcome": ("revision_falsified" if falsified else "revision_survives"),
                "summary": (
                    "The Baseline survives and falsifies the proposed revision."
                    if falsified
                    else "The proposed revision survives every fixed falsification check."
                ),
            }
        ),
        proposal=proposal,
        routed_card=card,
    )


def _candidate_revision_consistency(
    checker_team_id: str,
    owner_team_id: str,
    proposal: RevisionProposalEnvelope,
    assessment: BaselineRevisionAssessment,
    card: BoundaryCard,
    view: object,
) -> RevisionConsistencyReport:
    return RevisionConsistencyReport(
        checker_team_id=checker_team_id,
        assessment_owner_team_id=owner_team_id,
        proposal_digest=revision_proposal_digest(proposal),
        dimension=assessment.dimension,
        baseline_label=assessment.baseline_label,
        proposed_label=assessment.proposed_label,
        status=ConsistencyStatus.CONSISTENT,
        rationale="The exact cited evidence satisfies the replacement boundary.",
        supporting_evidence_ids=assessment.supporting_evidence_ids,
        counter_evidence_ids=assessment.counter_evidence_ids,
        assessment_digest=baseline_revision_assessment_digest(assessment),
        assessment_owner_task=assessment.assessor_task,
        assessment_raw_result_digest=assessment.raw_result_digest,
        boundary_card_id=card.card_id,
        baseline_source_config_hash=assessment.baseline_source_config_hash,
        baseline_source_predictions_sha256=(
            assessment.baseline_source_predictions_sha256
        ),
        taxonomy_structure_hash=assessment.taxonomy_structure_hash,
        evidence_view_hash=assessment.evidence_view_hash,
    )


def test_partial_preservation_surface_fails_closed_at_construction() -> None:
    with pytest.raises(ValueError, match="partial Baseline preservation"):
        _anchored_controller(baseline_anchor=_baseline_anchor())


def test_invalid_preservation_context_stops_before_readiness_or_any_model_call() -> (
    None
):
    calls: list[str] = []
    result = _preservation_controller(
        readiness=lambda task, view: calls.append("readiness") or _ready(task, view),
        joint_anchor=lambda team_id, view: calls.append("anchor") or _anchor(team_id),
        expected_baseline_anchor_hash="f" * 64,
    ).run(_ledger())

    assert calls == []
    assert result.final_decision is None
    assert result.unresolved is not None
    assert result.unresolved.stop_reason == "baseline_preservation_context_invalid"


def test_matching_valid_baseline_skips_assessment_and_preserves_both_labels() -> None:
    calls: list[object] = []
    result = _preservation_controller(
        baseline_revision_assessor=lambda *args: calls.append(args),
    ).run(_ledger())

    assert calls == []
    assert result.pre_gate_candidate is not None
    assert result.pre_gate_candidate.symptom_label == "unexpected_rejection"
    assert result.baseline_anchor == _baseline_anchor()
    assert result.revision_certificates == ()
    assert result.preservation_result is not None
    assert result.preservation_result.symptom_action is BaselineDecisionAction.PRESERVED
    assert (
        result.preservation_result.root_cause_action is BaselineDecisionAction.PRESERVED
    )
    assert result.final_decision is not None
    assert result.final_decision.source.value == "baseline_preservation_gate"
    assert result.pre_gate_candidate is not None
    assert result.final_decision.confidence == result.pre_gate_candidate.confidence
    assert (
        result.final_decision.supporting_evidence_ids
        == result.pre_gate_candidate.supporting_evidence_ids
    )
    assert result.verification is not None and result.verification.valid is True


@pytest.mark.parametrize(
    "mutation",
    ("composed_final", "verifier_anchor", "audit_accepted", "audit_support"),
)
def test_real_controller_correction_copy_fails_candidate_builder_and_verifier(
    mutation: str,
) -> None:
    ledger = _ledger()
    result = _preservation_controller(
        root_cause_verifier=lambda team_id, anchor, view: _verification(
            EvidenceDimension.ROOT_CAUSE,
            verdict=VerificationVerdict.REJECT,
        )
    ).run(ledger)
    assert result.final_decision is not None
    report = result.reports[0]
    root_review = report.verifications[1]
    root_audit = report.correction_audit[1]
    if mutation == "composed_final":
        report = report.model_copy(update={"root_cause": report.anchor.root_cause})
    elif mutation == "verifier_anchor":
        report = report.model_copy(
            update={
                "verifications": (
                    report.verifications[0],
                    root_review.model_copy(
                        update={"anchor_label": "incorrect_condition"}
                    ),
                )
            }
        )
    elif mutation == "audit_accepted":
        report = report.model_copy(
            update={
                "correction_audit": (
                    report.correction_audit[0],
                    root_audit.model_copy(update={"accepted": False}),
                )
            }
        )
    else:
        report = report.model_copy(
            update={
                "correction_audit": (
                    report.correction_audit[0],
                    root_audit.model_copy(update={"supporting_evidence_ids": ()}),
                )
            }
        )
    forged_reports = (report, result.reports[1])

    assert (
        build_revision_proposals(
            anchor=_baseline_anchor(),
            reports=forged_reports,
            structure=_preservation_structure(),
            view=ledger.view(),
        )
        == ()
    )
    verification = verify_stage3_decision(
        result.final_decision,
        ledger,
        reports=forged_reports,
    )
    assert verification.valid is False
    assert any("correction" in error for error in verification.errors)


@pytest.mark.parametrize(
    "verdict",
    [RevisionAssessmentVerdict.PRESERVE, RevisionAssessmentVerdict.INSUFFICIENT],
)
def test_rawless_assessment_success_is_downgraded_to_typed_failure(
    verdict: RevisionAssessmentVerdict,
) -> None:
    def joint_anchor(team_id: str, view: object) -> JointAnchorReport:
        return _anchor(team_id).model_copy(update={"symptom": _symptom("wrong_output")})

    def assess(
        team_id: str,
        proposal: RevisionProposalEnvelope,
        anchor: BaselineAnchor,
        card: BoundaryCard,
        view: object,
    ) -> BaselineRevisionAssessment:
        return BaselineRevisionAssessment(
            assessor_team_id=team_id,
            proposal_digest=revision_proposal_digest(proposal),
            dimension=proposal.dimension,
            verdict=verdict,
            baseline_source_config_hash=anchor.source_config_hash,
            baseline_source_predictions_sha256=anchor.source_predictions_sha256,
            taxonomy_structure_hash=proposal.taxonomy_structure_hash,
            evidence_view_hash=proposal.evidence_view_hash,
        )

    result = _preservation_controller(
        joint_anchor=joint_anchor,
        symptom_verifier=lambda team_id, anchor, view: _verification(
            EvidenceDimension.SYMPTOM,
            anchor_label="wrong_output",
        ),
        baseline_revision_assessor=assess,
    ).run(_ledger())

    assert result.revision_audit is not None
    assert len(result.revision_audit.proposal_entries) == 1
    assert len(result.revision_audit.assessment_attempts) == 2
    assert all(
        attempt.operational_failure
        for attempt in result.revision_audit.assessment_attempts
    ), repr(result.revision_audit)
    assert len(result.component_failures) == 2
    assert {failure.error_type for failure in result.component_failures} == {
        "StructuredOutputError"
    }
    assert all(
        failure.validation_summary["codes"] == ("policy_downgrade",)
        for failure in result.component_failures
    )
    assert (
        result.revision_audit.assessment_policy_version
        == "heterogeneous-entailment-falsification-v1"
    )
    assert result.revision_audit.cross_policy_version == (
        "opposite-task-specific-raw-bound-v3"
    )
    assert result.revision_audit.dual_revise_proposal_digests == ()
    assert result.revision_audit.consistency_attempts == ()
    assert result.revision_audit.issued_certificate_digests == ()
    assert result.revision_audit.applied_certificate_digests == ()
    assert result.revision_audit.failed_proposal_digests == (
        result.revision_audit.proposal_digests[0],
    )
    assert (
        result.revision_audit.__class__.model_validate_json(
            result.revision_audit.model_dump_json()
        )
        == result.revision_audit
    )


def test_preserved_baseline_is_not_rechecked_as_pre_gate_arbitration_output() -> None:
    def joint_anchor(team_id: str, view: object) -> JointAnchorReport:
        return _anchor(team_id).model_copy(update={"symptom": _symptom("wrong_output")})

    result = _preservation_controller(
        joint_anchor=joint_anchor,
        symptom_verifier=lambda team_id, anchor, view: _verification(
            EvidenceDimension.SYMPTOM,
            anchor_label="wrong_output",
        ),
        boundary_challenger=lambda reports, view: BoundaryChallenge(
            action=BoundaryChallengeAction.SYMPTOM_REVIEW,
            rationale="The unanimous symptom remains close to the Baseline boundary.",
            cited_evidence_ids=["code-1"],
        ),
        arbitrator=lambda packet: Stage3ArbitrationDecision(
            resolution_status=ResolutionStatus.RESOLVED,
            symptom_label="wrong_output",
            root_cause_label="missing_null_check",
            confidence=0.84,
            rationale="The packet supports the unanimous pre-gate symptom label.",
            supporting_evidence_ids=["code-1"],
            resolved_dimensions=list(packet.disagreement.dimensions),
        ),
    ).run(_ledger())

    assert result.pre_gate_candidate is not None
    assert result.pre_gate_candidate.source is ArbitrationSource.TARGETED_ARBITRATION
    assert result.pre_gate_candidate.symptom_label == "wrong_output"
    assert result.pre_gate_verification is not None
    assert result.pre_gate_verification.valid is True
    assert result.preservation_result is not None
    assert result.preservation_result.symptom_action is BaselineDecisionAction.PRESERVED
    assert result.final_decision is not None
    assert result.final_decision.symptom_label == "unexpected_rejection"
    assert result.verification is not None
    assert result.verification.valid is True
    assert result.unresolved is None


def test_preservation_verifier_still_rejects_pre_gate_arbitration_overreach() -> None:
    result = _preservation_controller().run(_ledger())
    assert result.pre_gate_candidate is not None
    assert result.pre_gate_verification is not None
    assert result.preservation_result is not None
    forged_candidate = result.pre_gate_candidate.model_copy(
        update={
            "symptom_label": "wrong_output",
            "confidence": 0.84,
            "rationale": "The arbitrator changed a label outside its listed authority.",
            "source": ArbitrationSource.TARGETED_ARBITRATION,
        }
    )
    disagreement = DisagreementMap(
        dimensions=("unsupported_symptom_boundary",),
        details={"unsupported_symptom_boundary": {"team_a": {}}},
        requires_arbitration=True,
    )
    arbitration = Stage3ArbitrationDecision(
        resolution_status=ResolutionStatus.RESOLVED,
        symptom_label="wrong_output",
        root_cause_label="missing_null_check",
        confidence=0.84,
        rationale="The arbitrator changed a label outside its listed authority.",
        supporting_evidence_ids=["code-1"],
        resolved_dimensions=list(disagreement.dimensions),
    )
    forged_pre_verification = result.pre_gate_verification.model_copy(
        update={
            "symptom_label": "wrong_output",
            "valid": True,
            "errors": (),
        }
    )
    forged_final = compose_baseline_preservation_decision(
        preservation=result.preservation_result,
        candidate=forged_candidate,
    )

    verification = verify_stage3_decision(
        forged_final,
        _ledger(),
        reports=result.reports,
        disagreement=disagreement,
        arbitration=arbitration,
        arbitration_evidence_ids=("code-1",),
        baseline_anchor=result.baseline_anchor,
        pre_gate_candidate=forged_candidate,
        pre_gate_verification=forged_pre_verification,
        revision_certificates=result.revision_certificates,
        preservation_result=result.preservation_result,
        revision_audit=result.revision_audit,
        taxonomy_structure=_preservation_structure(),
        expected_baseline_config_hash="a" * 64,
        expected_baseline_predictions_sha256="b" * 64,
        expected_baseline_anchor_hash=baseline_anchor_hash(_baseline_anchor()),
        expected_taxonomy_structure_hash=taxonomy_structure_hash(
            _preservation_structure()
        ),
    )

    assert verification.valid is False
    assert "arbitration changed unlisted dimension 'symptom_label'" in (
        verification.errors
    )
    assert any(
        "pre-gate verification does not match" in error for error in verification.errors
    )


def test_dual_team_revision_assesses_only_changed_dimension_and_revises_it() -> None:
    assessment_calls: list[tuple[str, EvidenceDimension, str]] = []
    consistency_calls: list[tuple[str, EvidenceDimension]] = []

    def joint_anchor(team_id: str, view: object) -> JointAnchorReport:
        return _anchor(team_id).model_copy(update={"symptom": _symptom("wrong_output")})

    def symptom_verifier(
        team_id: str, anchor: JointAnchorReport, view: object
    ) -> DimensionVerificationReport:
        return _verification(
            EvidenceDimension.SYMPTOM,
            anchor_label="wrong_output",
        )

    def assess(
        team_id: str,
        proposal: RevisionProposalEnvelope,
        anchor: BaselineAnchor,
        card: BoundaryCard,
        view: object,
    ) -> BaselineRevisionAssessment:
        assessment_calls.append((team_id, proposal.dimension, view.model_dump_json()))
        return _heterogeneous_candidate_revision_assessment(
            team_id, proposal, anchor, card, view
        )

    def consistency(
        checker_team_id: str,
        owner_team_id: str,
        proposal: RevisionProposalEnvelope,
        assessment: BaselineRevisionAssessment,
        card: BoundaryCard,
        view: object,
    ) -> RevisionConsistencyReport:
        consistency_calls.append((checker_team_id, assessment.dimension))
        return RevisionConsistencyReport(
            checker_team_id=checker_team_id,
            assessment_owner_team_id=owner_team_id,
            proposal_digest=revision_proposal_digest(proposal),
            dimension=assessment.dimension,
            baseline_label=assessment.baseline_label,
            proposed_label=assessment.proposed_label,
            status=ConsistencyStatus.CONSISTENT,
            rationale="The cited direct evidence satisfies the exact proposed boundary.",
            supporting_evidence_ids=("code-1",),
            counter_evidence_ids=("code-1",),
            assessment_digest=baseline_revision_assessment_digest(assessment),
            assessment_owner_task=assessment.assessor_task,
            assessment_raw_result_digest=assessment.raw_result_digest,
            boundary_card_id=card.card_id,
            baseline_source_config_hash=assessment.baseline_source_config_hash,
            baseline_source_predictions_sha256=(
                assessment.baseline_source_predictions_sha256
            ),
            taxonomy_structure_hash=assessment.taxonomy_structure_hash,
            evidence_view_hash=assessment.evidence_view_hash,
        )

    result = _preservation_controller(
        joint_anchor=joint_anchor,
        symptom_verifier=symptom_verifier,
        baseline_revision_assessor=assess,
        baseline_revision_consistency=consistency,
    ).run(_ledger())

    assert {(team, dimension) for team, dimension, _ in assessment_calls} == {
        ("A", EvidenceDimension.SYMPTOM),
        ("B", EvidenceDimension.SYMPTOM),
    }
    assert len({snapshot for _, _, snapshot in assessment_calls}) == 1
    assert set(consistency_calls) == {
        ("A", EvidenceDimension.SYMPTOM),
        ("B", EvidenceDimension.SYMPTOM),
    }
    assert len(result.revision_certificates) == 1
    assert result.preservation_result is not None
    assert result.preservation_result.symptom_action is BaselineDecisionAction.REVISED
    assert (
        result.preservation_result.root_cause_action is BaselineDecisionAction.PRESERVED
    )
    assert result.final_decision is not None
    assert result.final_decision.symptom_label == "wrong_output"
    assert result.final_decision.root_cause_label == "missing_null_check"
    assert result.verification is not None and result.verification.valid is True

    stripped_reports = tuple(
        report.model_copy(
            update={
                "baseline_revision_assessments": (),
                "revision_consistency": (),
            }
        )
        for report in result.reports
    )
    forged_verification = verify_stage3_decision(
        result.final_decision,
        _ledger(),
        reports=stripped_reports,
        baseline_anchor=result.baseline_anchor,
        pre_gate_candidate=result.pre_gate_candidate,
        pre_gate_verification=result.pre_gate_verification,
        revision_certificates=result.revision_certificates,
        preservation_result=result.preservation_result,
        revision_audit=result.revision_audit,
        taxonomy_structure=_preservation_structure(),
        expected_baseline_config_hash="a" * 64,
        expected_baseline_predictions_sha256="b" * 64,
        expected_baseline_anchor_hash=baseline_anchor_hash(_baseline_anchor()),
        expected_taxonomy_structure_hash=taxonomy_structure_hash(
            _preservation_structure()
        ),
    )
    assert forged_verification.valid is False
    assert any(
        "certificates do not match canonical team reports" in error
        for error in forged_verification.errors
    )


def test_revision_assessor_runtime_error_propagates() -> None:
    def fail(*args: object) -> BaselineRevisionAssessment:
        raise RuntimeError("programming defect")

    with pytest.raises(RuntimeError, match="programming defect"):
        _preservation_controller(
            joint_anchor=lambda team_id, view: _anchor(team_id).model_copy(
                update={"symptom": _symptom("wrong_output")}
            ),
            symptom_verifier=lambda team_id, anchor, view: _verification(
                EvidenceDimension.SYMPTOM, anchor_label="wrong_output"
            ),
            baseline_revision_assessor=fail,
        ).run(_ledger())


def test_revision_schema_failure_is_sanitized_and_preserves_baseline() -> None:
    def fail(*args: object) -> BaselineRevisionAssessment:
        raise StructuredOutputError(
            "SECRET_RAW_ASSESSMENT",
            schema_name="BaselineRevisionAssessment",
            validation_summary={"fields": ["verdict"], "codes": ["missing"]},
        )

    result = _preservation_controller(
        joint_anchor=lambda team_id, view: _anchor(team_id).model_copy(
            update={"symptom": _symptom("wrong_output")}
        ),
        symptom_verifier=lambda team_id, anchor, view: _verification(
            EvidenceDimension.SYMPTOM, anchor_label="wrong_output"
        ),
        baseline_revision_assessor=fail,
    ).run(_ledger())

    assert result.final_decision is not None
    assert result.final_decision.symptom_label == "unexpected_rejection"
    assert result.final_decision.confidence is None
    assert result.final_decision.supporting_evidence_ids == ()
    assert result.revision_certificates == ()
    assert len(result.component_failures) == 2
    assert all("SECRET" not in failure.message for failure in result.component_failures)
    assert result.revision_audit is not None
    assert (
        result.revision_audit.assessment_policy_version
        == "heterogeneous-entailment-falsification-v1"
    )
    assert result.revision_audit.cross_policy_version == (
        "opposite-task-specific-raw-bound-v3"
    )


def test_revision_consistency_transport_failure_preserves_without_certificate() -> None:
    def joint_anchor(team_id: str, view: object) -> JointAnchorReport:
        return _anchor(team_id).model_copy(update={"symptom": _symptom("wrong_output")})

    def assess(
        team_id: str,
        proposal: RevisionProposalEnvelope,
        anchor: BaselineAnchor,
        card: BoundaryCard,
        view: object,
    ) -> BaselineRevisionAssessment:
        return _heterogeneous_candidate_revision_assessment(
            team_id, proposal, anchor, card, view
        )

    def fail_consistency(*args: object) -> RevisionConsistencyReport:
        raise ModelTransportError("SECRET_PROVIDER", attempts=1)

    result = _preservation_controller(
        joint_anchor=joint_anchor,
        symptom_verifier=lambda team_id, anchor, view: _verification(
            EvidenceDimension.SYMPTOM, anchor_label="wrong_output"
        ),
        baseline_revision_assessor=assess,
        baseline_revision_consistency=fail_consistency,
    ).run(_ledger())

    assert result.final_decision is not None
    assert result.final_decision.symptom_label == "unexpected_rejection"
    assert result.revision_certificates == ()
    assert result.preservation_result is not None
    assert result.preservation_result.applied_certificates == ()
    assert result.preservation_result.symptom_action is BaselineDecisionAction.PRESERVED
    assert len(result.component_failures) == 2
    assert all(
        failure.role.startswith("baseline_revision_consistency_symptom-boundary")
        for failure in result.component_failures
    )
    assert all("SECRET" not in failure.message for failure in result.component_failures)


def test_rawless_legacy_dual_team_path_is_rejected_as_policy_downgrade() -> None:
    def joint_anchor(team_id: str, view: object) -> JointAnchorReport:
        return _anchor(team_id).model_copy(update={"symptom": _symptom("wrong_output")})

    def assess(
        team_id: str,
        proposal: RevisionProposalEnvelope,
        anchor: BaselineAnchor,
        card: BoundaryCard,
        view: object,
    ) -> BaselineRevisionAssessment:
        return _candidate_revision_assessment(team_id, proposal, anchor, card, view)

    def consistency(
        checker_team_id: str,
        owner_team_id: str,
        proposal: RevisionProposalEnvelope,
        assessment: BaselineRevisionAssessment,
        card: BoundaryCard,
        view: object,
    ) -> RevisionConsistencyReport:
        return _candidate_revision_consistency(
            checker_team_id, owner_team_id, proposal, assessment, card, view
        )

    result = _preservation_controller(
        joint_anchor=joint_anchor,
        symptom_verifier=lambda team_id, anchor, view: _verification(
            EvidenceDimension.SYMPTOM, anchor_label="wrong_output"
        ),
        baseline_revision_assessor=assess,
        baseline_revision_consistency=consistency,
    ).run(_ledger())

    assert result.revision_certificates == ()
    assert result.preservation_result is not None
    assert result.preservation_result.applied_certificates == ()
    assert result.preservation_result.symptom_action is BaselineDecisionAction.PRESERVED
    assert result.final_decision is not None
    assert result.final_decision.symptom_label == "unexpected_rejection"
    assert len(result.component_failures) == 2
    assert result.revision_audit is not None
    assert (
        result.revision_audit.assessment_policy_version
        == "heterogeneous-entailment-falsification-v1"
    )
    assert result.revision_audit.cross_policy_version == (
        "opposite-task-specific-raw-bound-v3"
    )


def test_entailment_plus_falsified_revision_runs_no_cross_and_preserves_baseline() -> (
    None
):
    cross_calls: list[object] = []

    def joint_anchor(team_id: str, view: object) -> JointAnchorReport:
        return _anchor(team_id).model_copy(update={"symptom": _symptom("wrong_output")})

    def assess(
        team_id: str,
        proposal: RevisionProposalEnvelope,
        anchor: BaselineAnchor,
        card: BoundaryCard,
        view: object,
    ) -> BaselineRevisionAssessment:
        return _heterogeneous_candidate_revision_assessment(
            team_id,
            proposal,
            anchor,
            card,
            view,
            falsified=team_id == "B",
        )

    result = _preservation_controller(
        joint_anchor=joint_anchor,
        symptom_verifier=lambda team_id, anchor, view: _verification(
            EvidenceDimension.SYMPTOM, anchor_label="wrong_output"
        ),
        baseline_revision_assessor=assess,
        baseline_revision_consistency=lambda *args: cross_calls.append(args),
    ).run(_ledger())

    assert cross_calls == []
    assert result.revision_certificates == ()
    assert result.final_decision is not None
    assert result.final_decision.symptom_label == "unexpected_rejection"
    assert result.revision_audit is not None
    assert (
        result.revision_audit.assessment_policy_version
        == "heterogeneous-entailment-falsification-v1"
    )
    assert result.revision_audit.dual_revise_proposal_digests == ()


def test_entailment_plus_surviving_revision_runs_exactly_two_opposite_crosses() -> None:
    cross_calls: list[tuple[str, str]] = []

    def joint_anchor(team_id: str, view: object) -> JointAnchorReport:
        return _anchor(team_id).model_copy(update={"symptom": _symptom("wrong_output")})

    def assess(
        team_id: str,
        proposal: RevisionProposalEnvelope,
        anchor: BaselineAnchor,
        card: BoundaryCard,
        view: object,
    ) -> BaselineRevisionAssessment:
        return _heterogeneous_candidate_revision_assessment(
            team_id, proposal, anchor, card, view
        )

    def consistency(
        checker_team_id: str,
        owner_team_id: str,
        proposal: RevisionProposalEnvelope,
        assessment: BaselineRevisionAssessment,
        card: BoundaryCard,
        view: object,
    ) -> RevisionConsistencyReport:
        cross_calls.append((checker_team_id, owner_team_id))
        return _candidate_revision_consistency(
            checker_team_id, owner_team_id, proposal, assessment, card, view
        )

    result = _preservation_controller(
        joint_anchor=joint_anchor,
        symptom_verifier=lambda team_id, anchor, view: _verification(
            EvidenceDimension.SYMPTOM, anchor_label="wrong_output"
        ),
        baseline_revision_assessor=assess,
        baseline_revision_consistency=consistency,
    ).run(_ledger())

    assert set(cross_calls) == {("A", "B"), ("B", "A")}
    assert len(cross_calls) == 2
    assert result.revision_audit is not None
    assert (
        result.revision_audit.assessment_policy_version
        == "heterogeneous-entailment-falsification-v1"
    )


def test_valid_baseline_preserves_without_candidate_after_bounded_component_failure() -> (
    None
):
    result = _preservation_controller(
        joint_anchor=lambda team_id, view: (_ for _ in ()).throw(
            ModelTransportError("SECRET_PROVIDER", attempts=2)
        )
    ).run(_ledger())

    assert result.pre_gate_candidate is None
    assert result.final_decision is not None
    assert result.final_decision.symptom_label == "unexpected_rejection"
    assert result.final_decision.root_cause_label == "missing_null_check"
    assert result.final_decision.confidence is None
    assert result.final_decision.supporting_evidence_ids == ()
    assert result.verification is not None and result.verification.valid is True
    assert result.unresolved is None
    assert result.revision_audit is not None
    assert (
        result.revision_audit.assessment_policy_version
        == "heterogeneous-entailment-falsification-v1"
    )
    assert result.revision_audit.cross_policy_version == (
        "opposite-task-specific-raw-bound-v3"
    )


def test_preserve_only_verifier_rejects_mismatched_resolution_accounting() -> None:
    result = _preservation_controller(
        joint_anchor=lambda team_id, view: (_ for _ in ()).throw(
            ModelTransportError("provider unavailable", attempts=2)
        )
    ).run(_ledger())
    assert result.pre_gate_candidate is None
    assert result.final_decision is not None
    assert result.verification is not None and result.verification.valid is True
    assert result.preservation_result is not None
    disagreement = DisagreementMap(
        dimensions=("symptom_label",),
        details={"symptom_label": {"team_a": "a", "team_b": "b"}},
        requires_arbitration=True,
    )
    arbitration = Stage3ArbitrationDecision(
        resolution_status=ResolutionStatus.RESOLVED,
        symptom_label="unexpected_rejection",
        root_cause_label="missing_null_check",
        confidence=0.84,
        rationale="This packet resolves a different dimension than the active one.",
        supporting_evidence_ids=["code-1"],
        resolved_dimensions=["root_cause_label"],
    )

    verification = verify_stage3_decision(
        result.final_decision,
        _ledger(),
        reports=result.reports,
        disagreement=disagreement,
        arbitration=arbitration,
        arbitration_evidence_ids=("code-1",),
        baseline_anchor=result.baseline_anchor,
        pre_gate_candidate=result.pre_gate_candidate,
        pre_gate_verification=result.pre_gate_verification,
        revision_certificates=result.revision_certificates,
        preservation_result=result.preservation_result,
        revision_audit=result.revision_audit,
        taxonomy_structure=_preservation_structure(),
        expected_baseline_config_hash="a" * 64,
        expected_baseline_predictions_sha256="b" * 64,
        expected_baseline_anchor_hash=baseline_anchor_hash(_baseline_anchor()),
        expected_taxonomy_structure_hash=taxonomy_structure_hash(
            _preservation_structure()
        ),
        component_failures=result.component_failures,
    )

    assert verification.valid is False
    assert verification.errors == (
        "arbitration resolved unlisted dimension 'root_cause_label'",
        "arbitration did not resolve active dimension 'symptom_label'",
    )


@pytest.mark.parametrize(
    "forged_update",
    (
        {"confidence": 0.01},
        {"supporting_evidence_ids": ["code-1"]},
        {"supporting_evidence_ids": ["foreign-id"]},
    ),
)
def test_preserve_only_verifier_rejects_forged_confidence_and_citations(
    forged_update: dict[str, object],
) -> None:
    result = _preservation_controller(
        joint_anchor=lambda team_id, view: (_ for _ in ()).throw(
            ModelTransportError("provider unavailable", attempts=2)
        )
    ).run(_ledger())
    assert result.final_decision is not None
    assert result.preservation_result is not None

    verification = verify_stage3_decision(
        result.final_decision.model_copy(update=forged_update),
        _ledger(),
        reports=result.reports,
        baseline_anchor=result.baseline_anchor,
        pre_gate_candidate=result.pre_gate_candidate,
        pre_gate_verification=result.pre_gate_verification,
        revision_certificates=result.revision_certificates,
        preservation_result=result.preservation_result,
        taxonomy_structure=_preservation_structure(),
        expected_baseline_config_hash="a" * 64,
        expected_baseline_predictions_sha256="b" * 64,
        expected_baseline_anchor_hash=baseline_anchor_hash(_baseline_anchor()),
        expected_taxonomy_structure_hash=taxonomy_structure_hash(
            _preservation_structure()
        ),
    )

    assert verification.valid is False
    assert any("post-gate decision" in error for error in verification.errors)


def test_invalid_baseline_accepts_only_verified_authoritative_candidate() -> None:
    unavailable = _baseline_anchor().model_copy(
        update={"valid": False, "symptom_label": None, "root_cause_label": None}
    )
    result = _preservation_controller(baseline_anchor=unavailable).run(_ledger())

    assert result.final_decision is not None
    assert result.preservation_result is not None
    assert (
        result.preservation_result.symptom_action
        is BaselineDecisionAction.BASELINE_UNAVAILABLE
    )
    assert result.verification is not None and result.verification.valid is True

    forged_verification = verify_stage3_decision(
        result.final_decision,
        _ledger(),
        reports=(),
        baseline_anchor=result.baseline_anchor,
        pre_gate_candidate=result.pre_gate_candidate,
        pre_gate_verification=result.pre_gate_verification,
        revision_certificates=result.revision_certificates,
        preservation_result=result.preservation_result,
        taxonomy_structure=_preservation_structure(),
        expected_baseline_config_hash="a" * 64,
        expected_baseline_predictions_sha256="b" * 64,
        expected_baseline_anchor_hash=baseline_anchor_hash(unavailable),
        expected_taxonomy_structure_hash=taxonomy_structure_hash(
            _preservation_structure()
        ),
    )
    assert forged_verification.valid is False
    assert any("producer authority" in error for error in forged_verification.errors)


def test_independent_verifier_rejects_forged_post_gate_result_and_audit_round_trips() -> (
    None
):
    result = _preservation_controller().run(_ledger())
    assert result.final_decision is not None
    assert result.preservation_result is not None
    forged = result.final_decision.model_copy(update={"symptom_label": "wrong_output"})
    verification = verify_stage3_decision(
        forged,
        _ledger(),
        reports=result.reports,
        baseline_anchor=result.baseline_anchor,
        pre_gate_candidate=result.pre_gate_candidate,
        pre_gate_verification=result.pre_gate_verification,
        revision_certificates=result.revision_certificates,
        preservation_result=result.preservation_result,
        taxonomy_structure=_preservation_structure(),
        expected_baseline_config_hash="a" * 64,
        expected_baseline_predictions_sha256="b" * 64,
        expected_baseline_anchor_hash=baseline_anchor_hash(_baseline_anchor()),
        expected_taxonomy_structure_hash=taxonomy_structure_hash(
            _preservation_structure()
        ),
    )
    assert verification.valid is False
    assert any("post-gate decision" in error for error in verification.errors)

    audit = workflow_audit_record(
        _ledger(),
        AdaptiveWorkflowResult(stage2=None, stage3=result, stop_reason="stage3_only"),
    )
    serialized = json.loads(json.dumps(audit))
    assert serialized["stage3"]["pre_gate_candidate"]["symptom_label"] == (
        "unexpected_rejection"
    )
    assert serialized["stage3"]["baseline_anchor"]["source_config_hash"] == "a" * 64
    assert serialized["stage3"]["preservation_result"]["symptom_action"] == "preserved"
    assert serialized["stage3"][
        "expected_baseline_anchor_hash"
    ] == baseline_anchor_hash(_baseline_anchor())
    assert serialized["stage3"][
        "expected_taxonomy_structure_hash"
    ] == taxonomy_structure_hash(_preservation_structure())


def test_independent_verifier_recomputes_raw_assessment_adapter_from_canonical_card() -> (
    None
):
    def joint_anchor(team_id: str, view: object) -> JointAnchorReport:
        return _anchor(team_id).model_copy(update={"symptom": _symptom("wrong_output")})

    result = _preservation_controller(
        joint_anchor=joint_anchor,
        symptom_verifier=lambda team_id, anchor, view: _verification(
            EvidenceDimension.SYMPTOM, anchor_label="wrong_output"
        ),
        baseline_revision_assessor=_heterogeneous_candidate_revision_assessment,
        baseline_revision_consistency=_candidate_revision_consistency,
    ).run(_ledger())
    assert result.final_decision is not None
    assert result.preservation_result is not None
    assert result.revision_audit is not None
    assessment = result.reports[0].baseline_revision_assessments[0]
    assert assessment.raw_entailment_result is not None
    forged_raw = assessment.raw_entailment_result.model_copy(
        update={
            "condition_findings": (
                assessment.raw_entailment_result.condition_findings[0].model_copy(
                    update={"condition": "caller-forged condition outside the card"}
                ),
            )
        }
    )
    forged_assessment = assessment.model_copy(
        update={
            "raw_entailment_result": forged_raw,
            "raw_result_digest": revision_raw_result_digest(forged_raw),
        }
    )
    forged_reports = (
        result.reports[0].model_copy(
            update={"baseline_revision_assessments": (forged_assessment,)}
        ),
        result.reports[1],
    )

    verification = verify_stage3_decision(
        result.final_decision,
        _ledger(),
        reports=forged_reports,
        baseline_anchor=result.baseline_anchor,
        pre_gate_candidate=result.pre_gate_candidate,
        pre_gate_verification=result.pre_gate_verification,
        revision_certificates=result.revision_certificates,
        preservation_result=result.preservation_result,
        revision_audit=result.revision_audit,
        taxonomy_structure=_preservation_structure(),
        expected_baseline_config_hash="a" * 64,
        expected_baseline_predictions_sha256="b" * 64,
        expected_baseline_anchor_hash=baseline_anchor_hash(_baseline_anchor()),
        expected_taxonomy_structure_hash=taxonomy_structure_hash(
            _preservation_structure()
        ),
    )

    assert verification.valid is False
    assert any("raw assessment adapter" in error for error in verification.errors)


def test_preservation_result_tree_rejects_post_verification_mutation() -> None:
    def anchor_with_unvalidated_nested_list(
        team_id: str, view: object
    ) -> JointAnchorReport:
        anchor = _anchor(team_id)
        symptom = anchor.symptom.model_copy(
            update={"supporting_evidence_ids": ["code-1"]}
        )
        return anchor.model_copy(update={"symptom": symptom})

    result = _preservation_controller(
        joint_anchor=anchor_with_unvalidated_nested_list
    ).run(_ledger())
    assert result.final_decision is not None
    before = workflow_audit_record(
        _ledger(),
        AdaptiveWorkflowResult(stage2=None, stage3=result, stop_reason="stage3_only"),
    )
    with pytest.raises(Exception, match="frozen"):
        result.final_decision.rationale = "Mutated after verification and audit."
    with pytest.raises(Exception, match="frozen"):
        result.pre_gate_candidate.symptom_label = "wrong_output"
    with pytest.raises(Exception, match="frozen"):
        result.reports[0].baseline_revision_assessments = ()
    with pytest.raises(AttributeError):
        result.final_decision.supporting_evidence_ids.append(
            "post-verification-mutation"
        )
    assert result.reports[0].symptom.supporting_evidence_ids == ("code-1",)
    with pytest.raises(AttributeError):
        result.reports[0].symptom.supporting_evidence_ids.append(
            "post-verification-report-mutation"
        )
    with pytest.raises(Exception, match="frozen"):
        result.validity_report.task = "tampered"
    with pytest.raises(TypeError):
        result.validity_report.capabilities_by_evidence_id["code-1"] = (
            "symptom_observation",
            "injected",
        )
    assert result.readiness_report is not None
    with pytest.raises(Exception, match="frozen"):
        result.readiness_report.task = "tampered"
    after = workflow_audit_record(
        _ledger(),
        AdaptiveWorkflowResult(stage2=None, stage3=result, stop_reason="stage3_only"),
    )
    assert after == before


def test_preservation_verifier_rejects_noncanonical_anchor_without_raising() -> None:
    result = _preservation_controller().run(_ledger())
    assert result.final_decision is not None
    forged_anchor = _baseline_anchor().model_copy(update={"root_cause_label": None})

    verification = verify_stage3_decision(
        result.final_decision,
        _ledger(),
        reports=result.reports,
        baseline_anchor=forged_anchor,
        pre_gate_candidate=result.pre_gate_candidate,
        pre_gate_verification=result.pre_gate_verification,
        revision_certificates=result.revision_certificates,
        preservation_result=result.preservation_result,
        taxonomy_structure=_preservation_structure(),
        expected_baseline_config_hash="a" * 64,
        expected_baseline_predictions_sha256="b" * 64,
        expected_baseline_anchor_hash=baseline_anchor_hash(_baseline_anchor()),
        expected_taxonomy_structure_hash=taxonomy_structure_hash(
            _preservation_structure()
        ),
    )

    assert verification.valid is False
    assert any("Baseline preservation" in error for error in verification.errors)


def test_result_boundary_deep_freezes_retrieval_delta_and_fact_collections() -> None:
    result = _preservation_controller().run(_ledger())
    item = _ledger().view().items[0]
    fact = EvidenceFact(
        claim="The null cache entry directly triggers request rejection.",
        evidence_ids=["code-1"],
        explicitness=EvidenceExplicitness.DIRECT,
    )
    delta = EvidenceDelta(
        request_id="request-1",
        specialist=SpecialistType.CODE_CONTEXT,
        status=RetrievalStatus.FOUND,
        items=[item],
        facts=[fact],
        diagnostics={"trace": {"attempts": [1, 2]}},
    )
    forged = delta.model_copy(
        update={
            "items": [item],
            "facts": [fact],
            "diagnostics": {"trace": {"attempts": [1, 2]}},
        }
    )

    canonical_result = replace(result, evidence_deltas=(forged,))
    canonical_delta = canonical_result.evidence_deltas[0]

    with pytest.raises(Exception, match="frozen"):
        canonical_delta.request_id = "tampered"
    with pytest.raises(AttributeError):
        canonical_delta.items.append(item)
    with pytest.raises(AttributeError):
        canonical_delta.facts[0].evidence_ids.append("injected")
    with pytest.raises(TypeError):
        canonical_delta.diagnostics["trace"] = {"attempts": [3]}
    with pytest.raises(AttributeError):
        canonical_delta.diagnostics["trace"]["attempts"].append(3)


@pytest.mark.parametrize(
    "candidate_update",
    (
        {"confidence": 0.01},
        {"rationale": "A caller-authored rationale that was never produced."},
        {"supporting_evidence_ids": ("code-1", "code-1")},
        {"source": ArbitrationSource.FALLBACK_UNCERTAIN},
    ),
)
def test_verifier_reconstructs_exact_pre_gate_candidate_from_producer_policy(
    candidate_update: dict[str, object],
) -> None:
    result = _preservation_controller().run(_ledger())
    assert result.pre_gate_candidate is not None
    assert result.preservation_result is not None
    forged_candidate = result.pre_gate_candidate.model_copy(update=candidate_update)
    forged_pre_verification = verify_stage3_decision(
        forged_candidate,
        _ledger(),
        reports=result.reports,
    )
    forged_final = compose_baseline_preservation_decision(
        preservation=result.preservation_result,
        candidate=forged_candidate,
    )

    verification = verify_stage3_decision(
        forged_final,
        _ledger(),
        reports=result.reports,
        baseline_anchor=result.baseline_anchor,
        pre_gate_candidate=forged_candidate,
        pre_gate_verification=forged_pre_verification,
        revision_certificates=result.revision_certificates,
        preservation_result=result.preservation_result,
        taxonomy_structure=_preservation_structure(),
        expected_baseline_config_hash="a" * 64,
        expected_baseline_predictions_sha256="b" * 64,
        expected_baseline_anchor_hash=baseline_anchor_hash(_baseline_anchor()),
        expected_taxonomy_structure_hash=taxonomy_structure_hash(
            _preservation_structure()
        ),
    )

    assert verification.valid is False
    assert any("canonical producer candidate" in error for error in verification.errors)


def test_verifier_canonicalizes_string_boundary_action_before_producer_resolution() -> (
    None
):
    result = _preservation_controller().run(_ledger())
    assert result.final_decision is not None
    boundary = BoundaryChallenge(
        action=BoundaryChallengeAction.PASS,
        rationale="The frozen proposals satisfy the current decision boundary.",
    )
    forged_boundary = boundary.model_copy(update={"action": "evidence_request"})

    verification = verify_stage3_decision(
        result.final_decision,
        _ledger(),
        reports=result.reports,
        disagreement=result.pre_gate_disagreement,
        arbitration=result.pre_gate_arbitration,
        arbitration_evidence_ids=result.pre_gate_arbitration_evidence_ids,
        baseline_anchor=result.baseline_anchor,
        pre_gate_candidate=result.pre_gate_candidate,
        pre_gate_verification=result.pre_gate_verification,
        revision_certificates=result.revision_certificates,
        preservation_result=result.preservation_result,
        taxonomy_structure=_preservation_structure(),
        expected_baseline_config_hash="a" * 64,
        expected_baseline_predictions_sha256="b" * 64,
        expected_baseline_anchor_hash=baseline_anchor_hash(_baseline_anchor()),
        expected_taxonomy_structure_hash=taxonomy_structure_hash(
            _preservation_structure()
        ),
        boundary_challenge=forged_boundary,
        component_failures=result.component_failures,
        consensus_confidence=0.8,
    )

    assert verification.valid is False
    assert any("canonical producer candidate" in error for error in verification.errors)


def test_independent_verifier_rejects_unknown_pass_challenge_before_reconstruction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _preservation_controller().run(_ledger())
    assert result.final_decision is not None
    reconstructed_boundaries: list[BoundaryChallenge | None] = []
    original_builder = verification_module.build_revision_proposals

    def capture_builder(*args: object, **kwargs: object) -> object:
        reconstructed_boundaries.append(kwargs.get("boundary_challenge"))
        return original_builder(*args, **kwargs)

    monkeypatch.setattr(
        verification_module, "build_revision_proposals", capture_builder
    )
    forged_boundary = BoundaryChallenge(
        action=BoundaryChallengeAction.PASS,
        rationale="A caller-forged PASS cites evidence rejected by the producer.",
        cited_evidence_ids=("unknown-id",),
    )

    verification = verify_stage3_decision(
        result.final_decision,
        _ledger(),
        reports=result.reports,
        disagreement=result.pre_gate_disagreement,
        arbitration=result.pre_gate_arbitration,
        arbitration_evidence_ids=result.pre_gate_arbitration_evidence_ids,
        baseline_anchor=result.baseline_anchor,
        pre_gate_candidate=result.pre_gate_candidate,
        pre_gate_verification=result.pre_gate_verification,
        revision_certificates=result.revision_certificates,
        preservation_result=result.preservation_result,
        taxonomy_structure=_preservation_structure(),
        expected_baseline_config_hash="a" * 64,
        expected_baseline_predictions_sha256="b" * 64,
        expected_baseline_anchor_hash=baseline_anchor_hash(_baseline_anchor()),
        expected_taxonomy_structure_hash=taxonomy_structure_hash(
            _preservation_structure()
        ),
        boundary_challenge=forged_boundary,
        component_failures=result.component_failures,
        consensus_confidence=0.8,
    )

    assert verification.valid is False
    assert any("unknown-id" in error for error in verification.errors)
    assert reconstructed_boundaries == [None]


@pytest.mark.parametrize(
    ("team_ids", "valid"),
    (
        ((), False),
        (("A",), False),
        (("A", "A"), False),
        (("A", "C"), False),
        (("A", "B"), True),
    ),
)
def test_independent_verifier_requires_exact_challenge_team_context(
    team_ids: tuple[str, ...],
    valid: bool,
) -> None:
    result = _preservation_controller().run(_ledger())
    assert result.pre_gate_candidate is not None
    reports = tuple(
        result.reports[index].model_copy(update={"team_id": team_id})
        for index, team_id in enumerate(team_ids)
    )
    challenge = BoundaryChallenge(
        action=BoundaryChallengeAction.PASS,
        rationale="A challenge exists only after canonical teams A and B complete.",
    )

    verification = verify_stage3_decision(
        result.pre_gate_candidate,
        _ledger(),
        reports=reports,
        boundary_challenge=challenge,
    )

    assert verification.valid is valid
    if valid:
        assert verification.errors == ()
    else:
        assert any("teams A and B" in error for error in verification.errors)


def test_verifier_canonicalizes_semantically_valid_string_decision_source() -> None:
    result = _preservation_controller().run(_ledger())
    assert result.final_decision is not None
    forged_final = result.final_decision.model_copy(
        update={"source": "baseline_preservation_gate"}
    )

    verification = verify_stage3_decision(
        forged_final,
        _ledger(),
        reports=result.reports,
        disagreement=result.pre_gate_disagreement,
        arbitration=result.pre_gate_arbitration,
        arbitration_evidence_ids=result.pre_gate_arbitration_evidence_ids,
        baseline_anchor=result.baseline_anchor,
        pre_gate_candidate=result.pre_gate_candidate,
        pre_gate_verification=result.pre_gate_verification,
        revision_certificates=result.revision_certificates,
        preservation_result=result.preservation_result,
        revision_audit=result.revision_audit,
        taxonomy_structure=_preservation_structure(),
        expected_baseline_config_hash="a" * 64,
        expected_baseline_predictions_sha256="b" * 64,
        expected_baseline_anchor_hash=baseline_anchor_hash(_baseline_anchor()),
        expected_taxonomy_structure_hash=taxonomy_structure_hash(
            _preservation_structure()
        ),
        component_failures=result.component_failures,
    )

    assert verification.valid is True


@pytest.mark.parametrize(
    ("resolution_status", "expected_valid"),
    (("resolved", False), ("invented_status", False)),
)
def test_verifier_canonicalizes_string_arbitration_status_without_raising(
    resolution_status: str, expected_valid: bool
) -> None:
    result = _preservation_controller().run(_ledger())
    assert result.final_decision is not None
    arbitration = Stage3ArbitrationDecision(
        resolution_status=ResolutionStatus.RESOLVED,
        symptom_label="unexpected_rejection",
        root_cause_label="missing_null_check",
        confidence=0.88,
        rationale="The supplied packet supports the same bounded consensus labels.",
        supporting_evidence_ids=("code-1",),
        resolved_dimensions=("symptom_label",),
    ).model_copy(update={"resolution_status": resolution_status})

    verification = verify_stage3_decision(
        result.final_decision,
        _ledger(),
        reports=result.reports,
        arbitration=arbitration,
        baseline_anchor=result.baseline_anchor,
        pre_gate_candidate=result.pre_gate_candidate,
        pre_gate_verification=result.pre_gate_verification,
        revision_certificates=result.revision_certificates,
        preservation_result=result.preservation_result,
        taxonomy_structure=_preservation_structure(),
        expected_baseline_config_hash="a" * 64,
        expected_baseline_predictions_sha256="b" * 64,
        expected_baseline_anchor_hash=baseline_anchor_hash(_baseline_anchor()),
        expected_taxonomy_structure_hash=taxonomy_structure_hash(
            _preservation_structure()
        ),
        component_failures=result.component_failures,
    )

    assert verification.valid is expected_valid
    if not expected_valid:
        expected_error = (
            "producer authority" if resolution_status == "resolved" else "arbitration"
        )
        assert any(expected_error in error for error in verification.errors)


@pytest.mark.parametrize("invalid_confidence", (float("nan"), -0.1, 1.1, "0.8"))
def test_verifier_rejects_invalid_producer_confidence_without_raising(
    invalid_confidence: object,
) -> None:
    result = _preservation_controller().run(_ledger())
    assert result.final_decision is not None

    verification = verify_stage3_decision(
        result.final_decision,
        _ledger(),
        reports=result.reports,
        disagreement=result.pre_gate_disagreement,
        arbitration=result.pre_gate_arbitration,
        arbitration_evidence_ids=result.pre_gate_arbitration_evidence_ids,
        baseline_anchor=result.baseline_anchor,
        pre_gate_candidate=result.pre_gate_candidate,
        pre_gate_verification=result.pre_gate_verification,
        revision_certificates=result.revision_certificates,
        preservation_result=result.preservation_result,
        taxonomy_structure=_preservation_structure(),
        expected_baseline_config_hash="a" * 64,
        expected_baseline_predictions_sha256="b" * 64,
        expected_baseline_anchor_hash=baseline_anchor_hash(_baseline_anchor()),
        expected_taxonomy_structure_hash=taxonomy_structure_hash(
            _preservation_structure()
        ),
        boundary_challenge=result.boundary_challenge,
        component_failures=result.component_failures,
        consensus_confidence=invalid_confidence,  # type: ignore[arg-type]
    )

    assert verification.valid is False
    assert any("consensus confidence" in error for error in verification.errors)


@pytest.mark.parametrize(
    "forged_update",
    (
        {"confidence": 0.01},
        {"supporting_evidence_ids": []},
        {"supporting_evidence_ids": ["foreign-id"]},
    ),
)
def test_candidate_backed_gate_verifier_binds_confidence_and_exact_citations(
    forged_update: dict[str, object],
) -> None:
    result = _preservation_controller().run(_ledger())
    assert result.final_decision is not None
    assert result.preservation_result is not None
    verification = verify_stage3_decision(
        result.final_decision.model_copy(update=forged_update),
        _ledger(),
        reports=result.reports,
        baseline_anchor=result.baseline_anchor,
        pre_gate_candidate=result.pre_gate_candidate,
        pre_gate_verification=result.pre_gate_verification,
        revision_certificates=result.revision_certificates,
        preservation_result=result.preservation_result,
        taxonomy_structure=_preservation_structure(),
        expected_baseline_config_hash="a" * 64,
        expected_baseline_predictions_sha256="b" * 64,
        expected_baseline_anchor_hash=baseline_anchor_hash(_baseline_anchor()),
        expected_taxonomy_structure_hash=taxonomy_structure_hash(
            _preservation_structure()
        ),
    )

    assert verification.valid is False
    assert any("post-gate decision" in error for error in verification.errors)


def test_arbitration_invalid_citation_is_a_hard_stop_before_preservation() -> None:
    def symptom_verifier(
        team_id: str,
        anchor_report: JointAnchorReport,
        view: object,
    ) -> DimensionVerificationReport:
        return _verification(
            EvidenceDimension.SYMPTOM,
            verdict=(
                VerificationVerdict.REJECT
                if team_id == "A"
                else VerificationVerdict.ACCEPT
            ),
        )

    result = _preservation_controller(
        symptom_verifier=symptom_verifier,
        arbitrator=lambda packet: Stage3ArbitrationDecision(
            resolution_status=ResolutionStatus.RESOLVED,
            symptom_label="wrong_output",
            root_cause_label="missing_null_check",
            confidence=0.82,
            rationale="The response cites evidence outside the frozen arbitration packet.",
            supporting_evidence_ids=["unknown-arbitration-evidence"],
            resolved_dimensions=list(packet.disagreement.dimensions),
        ),
    ).run(_ledger())

    assert result.final_decision is None
    assert result.verification is None
    assert result.preservation_result is None
    assert result.unresolved is not None
    assert result.unresolved.stop_reason == "arbitration_invalid_citation"


@pytest.mark.parametrize(
    ("symptom_label", "root_label", "expected_stop"),
    (
        (
            "unexpected_rejection",
            "invented_root",
            "arbitration_invalid_taxonomy",
        ),
        (
            "wrong_output",
            "missing_null_check",
            "arbitration_authority_violation",
        ),
    ),
)
def test_arbitration_taxonomy_and_authority_violations_cannot_be_preserved(
    symptom_label: str,
    root_label: str,
    expected_stop: str,
) -> None:
    def root_verifier(
        team_id: str,
        anchor_report: JointAnchorReport,
        view: object,
    ) -> DimensionVerificationReport:
        return _verification(
            EvidenceDimension.ROOT_CAUSE,
            verdict=(
                VerificationVerdict.REJECT
                if team_id == "A"
                else VerificationVerdict.ACCEPT
            ),
        )

    result = _preservation_controller(
        root_cause_verifier=root_verifier,
        arbitrator=lambda packet: Stage3ArbitrationDecision(
            resolution_status=ResolutionStatus.RESOLVED,
            symptom_label=symptom_label,
            root_cause_label=root_label,
            confidence=0.82,
            rationale="The arbitration response violates a frozen decision boundary.",
            supporting_evidence_ids=["code-1"],
            resolved_dimensions=list(packet.disagreement.dimensions),
        ),
    ).run(_ledger())

    assert result.final_decision is None
    assert result.preservation_result is None
    assert result.unresolved is not None
    assert result.unresolved.stop_reason == expected_stop


def test_targeted_candidate_verification_requires_canonical_arbitration_authority() -> (
    None
):
    def symptom_verifier(
        team_id: str,
        anchor_report: JointAnchorReport,
        view: object,
    ) -> DimensionVerificationReport:
        return _verification(
            EvidenceDimension.SYMPTOM,
            verdict=(
                VerificationVerdict.REJECT
                if team_id == "A"
                else VerificationVerdict.ACCEPT
            ),
        )

    result = _preservation_controller(
        symptom_verifier=symptom_verifier,
        arbitrator=lambda packet: Stage3ArbitrationDecision(
            resolution_status=ResolutionStatus.RESOLVED,
            symptom_label="wrong_output",
            root_cause_label="missing_null_check",
            confidence=0.82,
            rationale="The packet-owned evidence resolves the symptom disagreement.",
            supporting_evidence_ids=["code-1"],
            resolved_dimensions=list(packet.disagreement.dimensions),
        ),
    ).run(_ledger())
    assert result.pre_gate_candidate is not None
    assert result.pre_gate_candidate.source.value == "targeted_arbitration"
    assert result.final_decision is not None
    assert result.preservation_result is not None

    verification = verify_stage3_decision(
        result.final_decision,
        _ledger(),
        reports=result.reports,
        baseline_anchor=result.baseline_anchor,
        pre_gate_candidate=result.pre_gate_candidate,
        pre_gate_verification=result.pre_gate_verification,
        revision_certificates=result.revision_certificates,
        preservation_result=result.preservation_result,
        taxonomy_structure=_preservation_structure(),
        expected_baseline_config_hash="a" * 64,
        expected_baseline_predictions_sha256="b" * 64,
        expected_baseline_anchor_hash=baseline_anchor_hash(_baseline_anchor()),
        expected_taxonomy_structure_hash=taxonomy_structure_hash(
            _preservation_structure()
        ),
    )
    assert verification.valid is False
    assert any("targeted-arbitration" in error for error in verification.errors)


@pytest.mark.parametrize(
    "error",
    (
        StructuredOutputError(
            "SECRET_RAW_ARBITRATION_OUTPUT",
            schema_name="Stage3ArbitrationDecision",
            validation_summary={"fields": ["__root__"], "codes": ["value_error"]},
        ),
        ModelTransportError("SECRET_PROVIDER_FAILURE", attempts=6),
    ),
    ids=("structured-output", "model-transport"),
)
def test_arbitrator_operational_failure_uses_verified_candidate_fallback(
    error: Exception,
) -> None:
    def root_verifier(
        team_id: str,
        anchor_report: JointAnchorReport,
        view: object,
    ) -> DimensionVerificationReport:
        return _verification(
            EvidenceDimension.ROOT_CAUSE,
            verdict=(
                VerificationVerdict.REJECT
                if team_id == "A"
                else VerificationVerdict.ACCEPT
            ),
        )

    def arbitrator(packet: object) -> Stage3ArbitrationDecision:
        assert packet.disagreement.dimensions == ("root_cause_label",)
        assert tuple(item.evidence_id for item in packet.relevant_evidence) == (
            "code-1",
        )
        raise error

    result = _anchored_controller(
        root_cause_verifier=root_verifier,
        arbitrator=arbitrator,
    ).run(_ledger())

    assert len(result.reports) == 2
    assert result.final_decision is not None
    assert result.final_decision.symptom_label
    assert result.final_decision.root_cause_label
    assert result.final_decision.source.value == "fallback_uncertain"
    assert result.verification is not None and result.verification.valid is True
    assert result.unresolved is None
    failure = result.component_failures[-1]
    assert failure.team_id == "arbitration"
    assert failure.role == "stage3_arbitrator"
    assert failure.error_type == type(error).__name__
    assert "SECRET" not in failure.message


def test_arbitrator_failure_cannot_fallback_a_taxonomy_invalid_candidate() -> None:
    arbitrator_calls: list[object] = []

    def anchor(team_id: str, view: object) -> JointAnchorReport:
        report = _anchor(team_id)
        return (
            report.model_copy(
                update={
                    "symptom": report.symptom.model_copy(
                        update={"label": "invented_symptom"}
                    )
                }
            )
            if team_id == "A"
            else report
        )

    def symptom_verifier(
        team_id: str,
        anchor_report: JointAnchorReport,
        view: object,
    ) -> DimensionVerificationReport:
        return _verification(
            EvidenceDimension.SYMPTOM,
            anchor_label=anchor_report.symptom.label,
        )

    def arbitrator(packet: object) -> Stage3ArbitrationDecision:
        arbitrator_calls.append(packet)
        raise ModelTransportError("provider unavailable", attempts=6)

    result = _anchored_controller(
        joint_anchor=anchor,
        symptom_verifier=symptom_verifier,
        arbitrator=arbitrator,
    ).run(_ledger())

    assert len(result.reports) == 2
    assert arbitrator_calls == []
    assert result.final_decision is None
    assert result.verification is None
    assert result.unresolved is not None
    assert result.unresolved.stop_reason == "stage3_candidate_invalid_taxonomy"
    assert all(
        failure.role != "stage3_arbitrator" for failure in result.component_failures
    )


def test_both_anchors_and_all_verifiers_start_before_either_is_released() -> None:
    anchor_barrier = threading.Barrier(2)
    verifier_barrier = threading.Barrier(4)
    calls: list[tuple[str, str]] = []
    captured_views: list[object] = []
    lock = threading.Lock()

    def anchor(team_id: str, view: object) -> JointAnchorReport:
        with lock:
            calls.append((team_id, "anchor"))
            captured_views.append(view)
        anchor_barrier.wait(timeout=2)
        return _anchor(team_id)

    def verify(
        team_id: str,
        anchor_report: JointAnchorReport,
        view: object,
        *,
        dimension: EvidenceDimension,
    ) -> DimensionVerificationReport:
        with lock:
            calls.append((team_id, dimension.value))
            captured_views.append(view)
        verifier_barrier.wait(timeout=2)
        return _verification(dimension)

    result = _anchored_controller(
        joint_anchor=anchor,
        symptom_verifier=lambda team_id, anchor_report, view: verify(
            team_id,
            anchor_report,
            view,
            dimension=EvidenceDimension.SYMPTOM,
        ),
        root_cause_verifier=lambda team_id, anchor_report, view: verify(
            team_id,
            anchor_report,
            view,
            dimension=EvidenceDimension.ROOT_CAUSE,
        ),
    ).run(_ledger())

    assert set(calls[:2]) == {("A", "anchor"), ("B", "anchor")}
    assert set(calls[2:]) == {
        ("A", "symptom"),
        ("A", "root_cause"),
        ("B", "symptom"),
        ("B", "root_cause"),
    }
    assert len(captured_views) == 6
    assert len({view.model_dump_json() for view in captured_views}) == 1
    assert len({id(view) for view in captured_views}) == 6
    assert [report.team_id for report in result.reports] == ["A", "B"]
    assert result.verification is not None and result.verification.valid is True


def test_anchored_consumers_cannot_mutate_other_or_canonical_views() -> None:
    mutated = threading.Event()
    observations: list[tuple[str, int, tuple[str, ...], object]] = []

    def anchor(team_id: str, view: object) -> JointAnchorReport:
        if team_id == "A":
            object.__setattr__(view, "ledger_version", 999)
            object.__setattr__(view, "taxonomy", {"symptom": ["tampered"]})
            object.__setattr__(view.items[0], "metadata", {"tampered": True})
            mutated.set()
        else:
            assert mutated.wait(timeout=2)
        observations.append(
            (
                f"anchor-{team_id}",
                view.ledger_version,
                tuple(view.taxonomy["symptom"]),
                view.items[0].metadata,
            )
        )
        return _anchor(team_id)

    def verifier(
        team_id: str,
        anchor_report: JointAnchorReport,
        view: object,
        dimension: EvidenceDimension,
    ) -> DimensionVerificationReport:
        observations.append(
            (
                f"{dimension.value}-{team_id}",
                view.ledger_version,
                tuple(view.taxonomy["symptom"]),
                view.items[0].metadata,
            )
        )
        return _verification(dimension)

    ledger = _ledger(metadata={"notes": ["stable"]})
    result = _anchored_controller(
        joint_anchor=anchor,
        symptom_verifier=lambda team, anchor_report, view: verifier(
            team, anchor_report, view, EvidenceDimension.SYMPTOM
        ),
        root_cause_verifier=lambda team, anchor_report, view: verifier(
            team, anchor_report, view, EvidenceDimension.ROOT_CAUSE
        ),
    ).run(ledger)

    mutating = next(item for item in observations if item[0] == "anchor-A")
    assert mutating == ("anchor-A", 999, ("tampered",), {"tampered": True})
    assert all(
        observation[1:]
        == (
            1,
            ("unexpected_rejection", "wrong_output"),
            {
                "evidence_capabilities": (
                    "symptom_observation",
                    "defect_mechanism",
                ),
                "notes": ("stable",),
            },
        )
        for observation in observations
        if observation[0] != "anchor-A"
    )
    canonical = ledger.view()
    assert canonical.ledger_version == 1
    assert tuple(canonical.taxonomy["symptom"]) == (
        "unexpected_rejection",
        "wrong_output",
    )
    assert canonical.items[0].metadata["notes"] == ("stable",)
    assert result.verification is not None and result.verification.valid is True


@pytest.mark.parametrize("failed_team", ["A", "B"])
def test_anchor_failure_preserves_the_other_complete_team(failed_team: str) -> None:
    secret = "RAW-ANCHOR-COMPLETION"

    def anchor(team_id: str, view: object) -> JointAnchorReport:
        if team_id == failed_team:
            raise ValueError(secret)
        return _anchor(team_id)

    result = _anchored_controller(joint_anchor=anchor).run(_ledger())

    surviving_team = "B" if failed_team == "A" else "A"
    assert [report.team_id for report in result.reports] == [surviving_team]
    assert result.final_decision is not None
    assert result.final_decision.source.value == "single_team_degraded"
    failure = result.component_failures[0]
    assert (failure.team_id, failure.role) == (failed_team, "joint_anchor")
    assert secret not in failure.message


def test_complementary_verifier_failures_are_not_stitched_across_teams() -> None:
    def symptom_verifier(
        team_id: str, anchor: JointAnchorReport, view: object
    ) -> DimensionVerificationReport:
        if team_id == "B":
            raise ValueError("B symptom verifier failed")
        return _verification(EvidenceDimension.SYMPTOM)

    def root_verifier(
        team_id: str, anchor: JointAnchorReport, view: object
    ) -> DimensionVerificationReport:
        if team_id == "A":
            raise ValueError("A root verifier failed")
        return _verification(EvidenceDimension.ROOT_CAUSE)

    result = _anchored_controller(
        symptom_verifier=symptom_verifier,
        root_cause_verifier=root_verifier,
    ).run(_ledger())

    assert result.reports == ()
    assert result.final_decision is None
    assert result.verification is None
    assert {
        (failure.team_id, failure.role) for failure in result.component_failures
    } == {
        ("A", "root_cause_verifier"),
        ("B", "symptom_verifier"),
    }


def test_composer_failure_is_team_local_and_consistency_uses_composed_labels() -> None:
    checker_inputs: list[tuple[str, str, str]] = []

    def symptom_verifier(
        team_id: str, anchor: JointAnchorReport, view: object
    ) -> DimensionVerificationReport:
        if team_id == "A":
            return _verification(
                EvidenceDimension.SYMPTOM,
                anchor_label="wrong_output",
            )
        return _verification(
            EvidenceDimension.SYMPTOM,
            verdict=VerificationVerdict.REJECT,
        )

    def checker(
        team_id: str,
        symptom: SymptomReport,
        root: RootCauseReport,
        view: object,
    ) -> CausalConsistencyReport:
        checker_inputs.append((team_id, symptom.label, root.label))
        return _consistent()

    result = _anchored_controller(
        symptom_verifier=symptom_verifier,
        consistency_checker=checker,
    ).run(_ledger())

    assert [report.team_id for report in result.reports] == ["B"]
    assert checker_inputs == [("B", "wrong_output", "missing_null_check")]
    assert {
        (failure.team_id, failure.role) for failure in result.component_failures
    } == {("A", "team_composer")}


def test_composer_capability_mismatch_audit_uses_opaque_code_and_identifiers() -> None:
    content = "The maintainer identifies the missing guard as the defect mechanism."
    ledger = EvidenceLedger(
        record_id="capability-1",
        task="Classify the symptom and root cause.",
        taxonomy={
            "symptom": ["unexpected_rejection", "wrong_output"],
            "root_cause": ["missing_null_check", "incorrect_condition"],
        },
        domain_profile="ase2022",
        initial_items=[
            EvidenceItem(
                evidence_id="feg-node-test",
                record_id="capability-1",
                source_type="maintainer_confirmation",
                source_uri="https://example.test/issues/1#maintainer",
                retrieved_at="2026-08-05T11:00:00Z",
                content=content,
                content_sha256=hashlib.sha256(content.encode()).hexdigest(),
                explicitness=EvidenceExplicitness.DIRECT,
                metadata={"evidence_capabilities": ["defect_mechanism"]},
            )
        ],
    )

    def anchor(team_id: str, view: object) -> JointAnchorReport:
        return _anchor(team_id).model_copy(
            update={
                "symptom": _symptom().model_copy(
                    update={
                        "supporting_evidence_ids": ["feg-node-test"],
                        "boundary_evidence_ids": ["feg-node-test"],
                    }
                ),
                "root_cause": _root().model_copy(
                    update={
                        "supporting_evidence_ids": ["feg-node-test"],
                        "boundary_evidence_ids": ["feg-node-test"],
                    }
                ),
                "shared_supporting_evidence_ids": ["feg-node-test"],
            }
        )

    result = _anchored_controller(
        joint_anchor=anchor,
        symptom_verifier=lambda team_id, anchor, view: _verification(
            EvidenceDimension.SYMPTOM
        ).model_copy(update={"supporting_evidence_ids": ["feg-node-test"]}),
        root_cause_verifier=lambda team_id, anchor, view: _verification(
            EvidenceDimension.ROOT_CAUSE
        ).model_copy(update={"supporting_evidence_ids": ["feg-node-test"]}),
    ).run(ledger)

    failure = result.component_failures[0].model_dump(mode="json")
    assert failure["error_type"] == "CompositionError"
    assert failure["validation_summary"] == {
        "codes": ["supporting_evidence_incapable_for_dimension"],
        "fields": ["symptom", "evidence_id.feg_node_test"],
    }


def test_controller_composes_when_verifier_outputs_use_dimension_capable_evidence() -> (
    None
):
    mechanism_content = "The maintainer identifies the missing guard mechanism."
    symptom_content = "The runtime observation records the rejected request."
    ledger = EvidenceLedger(
        record_id="capability-2",
        task="Classify the symptom and root cause.",
        taxonomy={
            "symptom": ["unexpected_rejection", "wrong_output"],
            "root_cause": ["missing_null_check", "incorrect_condition"],
        },
        domain_profile="ase2022",
        initial_items=[
            EvidenceItem(
                evidence_id="feg-node-test",
                record_id="capability-2",
                source_type="maintainer_confirmation",
                source_uri="https://example.test/issues/2#maintainer",
                retrieved_at="2026-08-05T11:00:00Z",
                content=mechanism_content,
                content_sha256=hashlib.sha256(mechanism_content.encode()).hexdigest(),
                explicitness=EvidenceExplicitness.DIRECT,
                metadata={"evidence_capabilities": ["defect_mechanism"]},
            ),
            EvidenceItem(
                evidence_id="symptom-node-test",
                record_id="capability-2",
                source_type="runtime_observation",
                source_uri="https://example.test/issues/2#runtime",
                retrieved_at="2026-08-05T11:01:00Z",
                content=symptom_content,
                content_sha256=hashlib.sha256(symptom_content.encode()).hexdigest(),
                explicitness=EvidenceExplicitness.DIRECT,
            ),
        ],
    )

    def anchor(team_id: str, view: object) -> JointAnchorReport:
        return _anchor(team_id).model_copy(
            update={
                "symptom": _symptom().model_copy(
                    update={
                        "supporting_evidence_ids": ["symptom-node-test"],
                        "boundary_evidence_ids": ["symptom-node-test"],
                    }
                ),
                "root_cause": _root().model_copy(
                    update={
                        "supporting_evidence_ids": ["feg-node-test"],
                        "boundary_evidence_ids": ["feg-node-test"],
                    }
                ),
                "shared_supporting_evidence_ids": ["symptom-node-test"],
            }
        )

    result = _anchored_controller(
        joint_anchor=anchor,
        symptom_verifier=lambda team_id, anchor, view: _verification(
            EvidenceDimension.SYMPTOM
        ).model_copy(update={"supporting_evidence_ids": ["symptom-node-test"]}),
        root_cause_verifier=lambda team_id, anchor, view: _verification(
            EvidenceDimension.ROOT_CAUSE
        ).model_copy(update={"supporting_evidence_ids": ["feg-node-test"]}),
    ).run(ledger)

    assert [report.team_id for report in result.reports] == ["A", "B"]
    assert not any(
        failure.role == "team_composer" for failure in result.component_failures
    )


def test_checker_failure_is_team_local_after_composition() -> None:
    def checker(
        team_id: str,
        symptom: SymptomReport,
        root: RootCauseReport,
        view: object,
    ) -> CausalConsistencyReport:
        if team_id == "B":
            raise ValueError("B consistency failed")
        return _consistent()

    result = _anchored_controller(consistency_checker=checker).run(_ledger())

    assert [report.team_id for report in result.reports] == ["A"]
    assert result.final_decision is not None
    assert result.component_failures[0].role == "causal_consistency_checker"
    assert result.component_failures[0].team_id == "B"


def test_two_failed_anchors_produce_no_default_labels() -> None:
    def anchor(team_id: str, view: object) -> JointAnchorReport:
        raise ValueError(f"{team_id} anchor failed")

    result = _anchored_controller(joint_anchor=anchor).run(_ledger())

    assert result.reports == ()
    assert result.final_decision is None
    assert result.verification is None
    assert result.unresolved is not None
    assert result.unresolved.stop_reason == "stage3_component_failure"


def test_anchored_teams_still_run_with_a_readiness_gap() -> None:
    calls: list[str] = []

    result = _anchored_controller(
        readiness=lambda task, view: _readiness_report(sufficient=False),
        joint_anchor=lambda team_id, view: calls.append(team_id) or _anchor(team_id),
    ).run(_ledger())

    assert sorted(calls) == ["A", "B"]
    assert result.final_decision is not None
    assert result.readiness_report is not None
    assert result.readiness_report.dimensions[0].sufficient is False


def test_zero_valid_evidence_stops_before_anchored_roles() -> None:
    calls: list[str] = []

    result = _anchored_controller(
        readiness=lambda task, view: _readiness_report(sufficient=False),
        joint_anchor=lambda team_id, view: calls.append(team_id) or _anchor(team_id),
    ).run(_stage3_validity_ledger(include_valid=False))

    assert calls == []
    assert result.reports == ()
    assert result.unresolved is not None
    assert result.unresolved.stop_reason == "no_valid_stage3_evidence"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"joint_anchor": lambda team_id, view: _anchor(team_id)},
        {
            "joint_anchor": lambda team_id, view: _anchor(team_id),
            "symptom_verifier": None,
            "root_cause_verifier": lambda team_id, anchor, view: _verification(
                EvidenceDimension.ROOT_CAUSE
            ),
        },
        {
            "joint_anchor": lambda team_id, view: _anchor(team_id),
            "symptom_verifier": lambda team_id, anchor, view: _verification(
                EvidenceDimension.SYMPTOM
            ),
            "root_cause_verifier": object(),
        },
    ],
)
def test_partial_or_noncallable_anchored_constructor_surface_fails_closed(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="partial baseline-anchored Stage 3"):
        Stage3Controller(
            readiness=_ready,
            symptom_analyst=lambda team_id, view: _symptom(),
            root_cause_analyst=lambda team_id, view: _root(),
            consistency_checker=lambda team_id, symptom, root, view: _consistent(),
            **kwargs,
        )


def test_partial_legacy_surface_fails_even_with_complete_anchored_surface() -> None:
    with pytest.raises(ValueError, match="partial legacy Stage 3"):
        _anchored_controller(
            symptom_analyst=lambda team_id, view: _symptom(),
            root_cause_analyst=None,
        )


def test_composed_provenance_and_labels_reach_challenger_and_arbitrator() -> None:
    checker_inputs: list[tuple[str, str, str]] = []
    challenger_labels: list[tuple[str, str]] = []
    arbitration_labels: list[tuple[str, str]] = []

    def symptom_verifier(
        team_id: str, anchor: JointAnchorReport, view: object
    ) -> DimensionVerificationReport:
        return _verification(
            EvidenceDimension.SYMPTOM,
            verdict=(
                VerificationVerdict.REJECT
                if team_id == "A"
                else VerificationVerdict.ACCEPT
            ),
        )

    def checker(
        team_id: str,
        symptom: SymptomReport,
        root: RootCauseReport,
        view: object,
    ) -> CausalConsistencyReport:
        checker_inputs.append((team_id, symptom.label, root.label))
        return _consistent()

    def challenger(
        reports: tuple[Stage3TeamReport, Stage3TeamReport], view: object
    ) -> BoundaryChallenge:
        challenger_labels.extend(
            (report.symptom.label, report.root_cause.label) for report in reports
        )
        return BoundaryChallenge(
            action=BoundaryChallengeAction.PASS,
            rationale="The composed labels stay within the frozen taxonomy.",
            cited_evidence_ids=["code-1"],
        )

    def arbitrator(packet: object) -> Stage3ArbitrationDecision:
        arbitration_labels.extend(
            [
                (packet.team_a.symptom.label, packet.team_a.root_cause.label),
                (packet.team_b.symptom.label, packet.team_b.root_cause.label),
            ]
        )
        return Stage3ArbitrationDecision(
            resolution_status=ResolutionStatus.RESOLVED,
            symptom_label="wrong_output",
            root_cause_label="missing_null_check",
            confidence=0.82,
            rationale="The corrected symptom has the stronger direct evidence support.",
            supporting_evidence_ids=["code-1"],
            resolved_dimensions=list(packet.disagreement.dimensions),
        )

    result = _anchored_controller(
        symptom_verifier=symptom_verifier,
        consistency_checker=checker,
        boundary_challenger=challenger,
        arbitrator=arbitrator,
    ).run(_ledger())

    assert checker_inputs == [
        ("A", "wrong_output", "missing_null_check"),
        ("B", "unexpected_rejection", "missing_null_check"),
    ]
    assert challenger_labels == [
        ("wrong_output", "missing_null_check"),
        ("unexpected_rejection", "missing_null_check"),
    ]
    assert arbitration_labels == challenger_labels
    report_a = result.reports[0]
    assert report_a.anchor == _anchor("A")
    assert report_a.verifications[0].verdict is VerificationVerdict.REJECT
    assert report_a.correction_audit[0].accepted is True
    assert report_a.correction_audit[0].final_label == "wrong_output"
    assert result.final_decision is not None
    assert result.final_decision.symptom_label == "wrong_output"


def test_anchored_candidate_unknown_precheck_runs_before_challenger_and_arbitrator() -> (
    None
):
    challenger_calls: list[object] = []
    arbitrator_calls: list[object] = []

    def anchor(team_id: str, view: object) -> JointAnchorReport:
        report = _anchor(team_id)
        return report.model_copy(
            update={"shared_counter_evidence_ids": ["unknown-anchor"]}
        )

    result = _anchored_controller(
        joint_anchor=anchor,
        boundary_challenger=lambda reports, view: challenger_calls.append(reports)
        or BoundaryChallenge(
            action=BoundaryChallengeAction.PASS,
            rationale="The candidate appears within the frozen boundaries.",
        ),
        arbitrator=lambda packet: arbitrator_calls.append(packet),
    ).run(_ledger())

    assert challenger_calls == []
    assert arbitrator_calls == []
    assert result.final_decision is None
    assert result.verification is None
    assert result.unresolved is not None
    assert result.unresolved.stop_reason == "arbitration_evidence_insufficient"
    assert "unknown-anchor" in " ".join(result.unresolved.missing_facts)


def test_single_complete_anchored_team_never_splices_peer_components() -> None:
    def anchor(team_id: str, view: object) -> JointAnchorReport:
        report = _anchor(team_id)
        if team_id == "A":
            return report.model_copy(update={"symptom": _symptom("wrong_output")})
        return report

    def symptom_verifier(
        team_id: str,
        anchor_report: JointAnchorReport,
        view: object,
    ) -> DimensionVerificationReport:
        return _verification(
            EvidenceDimension.SYMPTOM,
            anchor_label=anchor_report.symptom.label,
        )

    def root_verifier(
        team_id: str,
        anchor_report: JointAnchorReport,
        view: object,
    ) -> DimensionVerificationReport:
        if team_id == "B":
            raise ValueError("B root verifier failed schema validation")
        return _verification(
            EvidenceDimension.ROOT_CAUSE,
            verdict=VerificationVerdict.REJECT,
        )

    result = _anchored_controller(
        joint_anchor=anchor,
        symptom_verifier=symptom_verifier,
        root_cause_verifier=root_verifier,
    ).run(_ledger())

    assert [report.team_id for report in result.reports] == ["A"]
    assert result.final_decision is not None
    assert result.final_decision.source.value == "single_team_degraded"
    assert result.final_decision.symptom_label == "wrong_output"
    assert result.final_decision.root_cause_label == "incorrect_condition"


def test_arbitrator_cannot_change_agreed_dimension_fails_closed() -> None:
    def root_verifier(
        team_id: str,
        anchor_report: JointAnchorReport,
        view: object,
    ) -> DimensionVerificationReport:
        return _verification(
            EvidenceDimension.ROOT_CAUSE,
            verdict=(
                VerificationVerdict.REJECT
                if team_id == "A"
                else VerificationVerdict.ACCEPT
            ),
        )

    def arbitrator(packet: object) -> Stage3ArbitrationDecision:
        assert packet.disagreement.dimensions == ("root_cause_label",)
        return Stage3ArbitrationDecision(
            resolution_status=ResolutionStatus.RESOLVED,
            symptom_label="wrong_output",
            root_cause_label="missing_null_check",
            confidence=0.81,
            rationale="The response improperly changes the already agreed symptom.",
            supporting_evidence_ids=["code-1"],
            resolved_dimensions=["root_cause_label"],
        )

    result = _anchored_controller(
        root_cause_verifier=root_verifier,
        arbitrator=arbitrator,
    ).run(_ledger())

    assert result.final_decision is None
    assert result.verification is None
    assert result.unresolved is not None
    assert result.unresolved.stop_reason == "arbitration_authority_violation"
    assert "authority violation" in " ".join(result.unresolved.missing_facts)


def test_arbitrator_taxonomy_escape_fails_closed_without_fallback() -> None:
    def arbitrator(packet: object) -> Stage3ArbitrationDecision:
        assert packet.disagreement.dimensions == ("symptom_label",)
        return Stage3ArbitrationDecision(
            resolution_status=ResolutionStatus.RESOLVED,
            symptom_label="invented_symptom",
            root_cause_label="missing_null_check",
            confidence=0.81,
            rationale="The response improperly escapes the supplied symptom taxonomy.",
            supporting_evidence_ids=["code-1"],
            resolved_dimensions=["symptom_label"],
        )

    result = Stage3Controller(
        readiness=_ready,
        symptom_analyst=lambda team_id, view: _symptom(
            "unexpected_rejection" if team_id == "A" else "wrong_output"
        ),
        root_cause_analyst=lambda team_id, view: _root(),
        consistency_checker=lambda team_id, symptom, root, view: _consistent(),
        arbitrator=arbitrator,
    ).run(_ledger())

    assert result.final_decision is None
    assert result.verification is None
    assert result.unresolved is not None
    assert result.unresolved.stop_reason == "arbitration_invalid_taxonomy"
    assert "outside the symptom taxonomy" in " ".join(result.unresolved.missing_facts)


def _append_stage3_item(
    ledger: EvidenceLedger,
    evidence_id: str,
    source_type: str,
) -> None:
    content = f"Frozen Stage 3 evidence for {evidence_id}."
    ledger.append(
        EvidenceItem(
            evidence_id=evidence_id,
            record_id="record-1",
            source_type=source_type,
            source_uri=f"record:record-1#{evidence_id}",
            retrieved_at="2026-08-05T11:05:00Z",
            content=content,
            content_sha256=hashlib.sha256(content.encode()).hexdigest(),
            explicitness=EvidenceExplicitness.DIRECT,
        )
    )


def test_anchored_arbitration_packet_is_dimension_minimal_and_keeps_correction() -> (
    None
):
    ledger = _ledger()
    for evidence_id, source_type in (
        ("symptom-anchor", "issue_body"),
        ("symptom-correction", "test_result"),
        ("root-anchor", "code_diff"),
        ("consistency-only", "code_diff"),
    ):
        _append_stage3_item(ledger, evidence_id, source_type)
    captured_ids: list[tuple[str, ...]] = []

    def anchor(team_id: str, view: object) -> JointAnchorReport:
        return JointAnchorReport(
            symptom=_symptom().model_copy(
                update={
                    "supporting_evidence_ids": ["symptom-anchor"],
                    "boundary_evidence_ids": ["symptom-anchor"],
                }
            ),
            root_cause=_root().model_copy(
                update={
                    "supporting_evidence_ids": ["root-anchor"],
                    "boundary_evidence_ids": ["root-anchor"],
                }
            ),
            causal_account="The frozen symptom and cause form one bounded account.",
            shared_supporting_evidence_ids=["symptom-anchor"],
        )

    def symptom_verifier(
        team_id: str,
        anchor_report: JointAnchorReport,
        view: object,
    ) -> DimensionVerificationReport:
        if team_id == "A":
            return _verification(
                EvidenceDimension.SYMPTOM,
                verdict=VerificationVerdict.REJECT,
            ).model_copy(update={"supporting_evidence_ids": ["symptom-correction"]})
        return _verification(EvidenceDimension.SYMPTOM).model_copy(
            update={"supporting_evidence_ids": ["symptom-anchor"]}
        )

    def root_verifier(
        team_id: str,
        anchor_report: JointAnchorReport,
        view: object,
    ) -> DimensionVerificationReport:
        return _verification(EvidenceDimension.ROOT_CAUSE).model_copy(
            update={"supporting_evidence_ids": ["root-anchor"]}
        )

    def arbitrator(packet: object) -> Stage3ArbitrationDecision:
        ids = tuple(item.evidence_id for item in packet.relevant_evidence)
        captured_ids.append(ids)
        return Stage3ArbitrationDecision(
            resolution_status=ResolutionStatus.UNRESOLVED,
            rationale="The symptom correction remains uncertain within the packet.",
            unresolved_dimensions=list(packet.disagreement.dimensions),
            missing_facts=["A stronger direct symptom comparison is still required."],
        )

    result = _anchored_controller(
        joint_anchor=anchor,
        symptom_verifier=symptom_verifier,
        root_cause_verifier=root_verifier,
        consistency_checker=lambda team, symptom, root, view: _consistent().model_copy(
            update={"supporting_evidence_ids": ["consistency-only"]}
        ),
        arbitrator=arbitrator,
    ).run(ledger)

    assert captured_ids == [("symptom-anchor", "symptom-correction")]
    assert "root-anchor" not in captured_ids[0]
    assert "consistency-only" not in captured_ids[0]
    assert result.final_decision is not None


def test_boundary_review_can_use_dimension_owned_verifier_provenance() -> None:
    ledger = _ledger()
    _append_stage3_item(ledger, "verifier-observation", "test_result")
    packets: list[object] = []

    def arbitrator(packet: object) -> Stage3ArbitrationDecision:
        packets.append(packet)
        assert tuple(item.evidence_id for item in packet.relevant_evidence) == (
            "code-1",
            "verifier-observation",
        )
        return Stage3ArbitrationDecision(
            resolution_status=ResolutionStatus.UNRESOLVED,
            rationale="The symptom boundary remains uncertain after bounded review.",
            unresolved_dimensions=list(packet.disagreement.dimensions),
            missing_facts=["A discriminating runtime symptom comparison is required."],
        )

    result = _anchored_controller(
        symptom_verifier=lambda team, anchor, view: _verification(
            EvidenceDimension.SYMPTOM
        ).model_copy(update={"supporting_evidence_ids": ["verifier-observation"]}),
        boundary_challenger=lambda reports, view: BoundaryChallenge(
            action=BoundaryChallengeAction.SYMPTOM_REVIEW,
            rationale="The verifier observation warrants a bounded symptom review.",
            cited_evidence_ids=["verifier-observation"],
        ),
        arbitrator=arbitrator,
    ).run(ledger)

    assert len(packets) == 1
    assert result.final_decision is not None
    assert result.final_decision.source.value == "fallback_uncertain"


@pytest.mark.parametrize("revision_audit", [None, Stage3RevisionAudit()])
def test_full_preservation_verifier_rejects_missing_or_legacy_revision_audit(
    revision_audit: Stage3RevisionAudit | None,
) -> None:
    result = _preservation_controller().run(_ledger())
    assert result.final_decision is not None
    assert result.preservation_result is not None

    verification = verify_stage3_decision(
        result.final_decision,
        _ledger(),
        reports=result.reports,
        baseline_anchor=result.baseline_anchor,
        pre_gate_candidate=result.pre_gate_candidate,
        pre_gate_verification=result.pre_gate_verification,
        revision_certificates=result.revision_certificates,
        preservation_result=result.preservation_result,
        revision_audit=revision_audit,
        taxonomy_structure=_preservation_structure(),
        expected_baseline_config_hash="a" * 64,
        expected_baseline_predictions_sha256="b" * 64,
        expected_baseline_anchor_hash=baseline_anchor_hash(_baseline_anchor()),
        expected_taxonomy_structure_hash=taxonomy_structure_hash(
            _preservation_structure()
        ),
    )

    assert verification.valid is False
    assert any("revision audit policy" in error for error in verification.errors)
