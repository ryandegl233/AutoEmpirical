from __future__ import annotations

import hashlib
import threading

import pytest

from Benchmark.src.adaptive_empirical_workflow.agents import StructuredOutputError
from Benchmark.src.adaptive_empirical_workflow.contracts import (
    DimensionReadiness,
    EvidenceDelta,
    EvidenceDimension,
    EvidenceExplicitness,
    EvidenceItem,
    EvidenceReadinessReport,
    EvidenceRequest,
    EvidenceSufficiency,
    FaultEvidenceAssessment,
    RepairCausalityAssessment,
    RetrievalStatus,
    ResolutionStatus,
    ScopeBoundaryAssessment,
    SpecialistType,
    Stage2AnalysisReport,
    Stage2ArbitrationDecision,
    Stage2Decision,
    Stage2ScopeExclusion,
    TestOutcome as EvidenceTestOutcome,
)
from Benchmark.src.adaptive_empirical_workflow.controller import (
    Stage2Controller,
    Stage2WorkflowConfig,
    Stage2WorkflowResult,
)
from Benchmark.src.adaptive_empirical_workflow.ledger import EvidenceLedger
from Benchmark.src.adaptive_empirical_workflow.specialists import SpecialistRegistry


def test_stage2_recoverable_readiness_failure_remains_strict() -> None:
    analyst_calls: list[str] = []

    with pytest.raises(StructuredOutputError):
        Stage2Controller(
            readiness=lambda task, view: (_ for _ in ()).throw(
                StructuredOutputError(
                    "readiness schema retries exhausted",
                    schema_name="EvidenceReadinessReport",
                )
            ),
            analyst=lambda team_id, view: analyst_calls.append(team_id),
        ).run(_ledger())

    assert analyst_calls == []


def _ledger(domain: str = "ase2022") -> EvidenceLedger:
    return EvidenceLedger(
        record_id="record-1",
        task="Verify whether the candidate repairs a fault.",
        taxonomy={"decision": ["accepted_fault", "rejected_candidate"]},
        domain_profile=domain,
    )


def _request(team_id: str) -> EvidenceRequest:
    return EvidenceRequest(
        request_id=f"{team_id}-issue",
        missing_fact="Whether maintainers confirm the prior runtime behavior is incorrect",
        why_needed="The intended behavior separates a repair from an enhancement.",
        target_specialist=SpecialistType.ISSUE_PR,
        target_source="linked issue discussion",
        query="record-1 intended runtime behavior",
        expected_decision_impact=(
            "Confirmation of incorrect prior behavior supports accepted_fault; "
            "an enhancement request without failure evidence supports rejection."
        ),
        max_items=2,
    )


def _report(
    team_id: str,
    *,
    decision: Stage2Decision = Stage2Decision.ACCEPTED,
    sufficient: bool = True,
    requests: list[EvidenceRequest] | None = None,
    repair_outcome: EvidenceTestOutcome = EvidenceTestOutcome.PASS,
    fault_outcome: EvidenceTestOutcome | None = None,
) -> Stage2AnalysisReport:
    outcomes = {
        "fault_existence": EvidenceTestOutcome.PASS,
        "repair_causality": repair_outcome,
        "scope_exclusion": EvidenceTestOutcome.PASS,
    }
    if decision is Stage2Decision.REJECTED:
        outcomes["fault_existence"] = EvidenceTestOutcome.FAIL
    if fault_outcome is not None:
        outcomes["fault_existence"] = fault_outcome
    return Stage2AnalysisReport(
        team_id=team_id,
        decision=decision,
        confidence=0.9,
        fault_claim="The previous behavior rejects a valid runtime request.",
        repair_claim="The patch restores that request and covers it with a test.",
        supporting_evidence_ids=["issue-1"],
        evidence_tests=outcomes,
        alternative_hypothesis="The change could instead add unsupported behavior.",
        decision_boundary="Prior incorrect behavior distinguishes repair from feature work.",
        evidence_sufficiency=(
            EvidenceSufficiency.SUFFICIENT
            if sufficient
            else EvidenceSufficiency.INSUFFICIENT
        ),
        unresolved_evidence_gaps=(
            [] if sufficient else ["Maintainer intent is not yet established."]
        ),
        evidence_requests=requests or [],
    )


def _issue_item() -> EvidenceItem:
    content = "Maintainer: this rejection is a bug and should be fixed."
    return EvidenceItem(
        evidence_id="issue-1",
        record_id="record-1",
        source_type="issue_comment",
        source_uri="https://example.test/issues/1",
        retrieved_at="2026-08-05T10:00:00Z",
        content=content,
        content_sha256=hashlib.sha256(content.encode()).hexdigest(),
        explicitness=EvidenceExplicitness.DIRECT,
    )


def _unrelated_item() -> EvidenceItem:
    content = "An unrelated ledger note that neither candidate report cites."
    return EvidenceItem(
        evidence_id="unrelated-1",
        record_id="record-1",
        source_type="issue_comment",
        source_uri="https://example.test/issues/1#unrelated",
        retrieved_at="2026-08-05T10:01:00Z",
        content=content,
        content_sha256=hashlib.sha256(content.encode()).hexdigest(),
        explicitness=EvidenceExplicitness.DIRECT,
    )


def _code_item() -> EvidenceItem:
    content = "The code diff adds the missing guard in the request handler."
    return EvidenceItem(
        evidence_id="code-1",
        record_id="record-1",
        source_type="code_diff",
        source_uri="https://example.test/commit/1",
        retrieved_at="2026-08-05T10:02:00Z",
        content=content,
        content_sha256=hashlib.sha256(content.encode()).hexdigest(),
        explicitness=EvidenceExplicitness.DIRECT,
    )


def _readiness_report(
    *,
    sufficient: bool = True,
    requests: tuple[EvidenceRequest, ...] = (),
    include_repair: bool = False,
    repair_evidence_id: str = "issue-1",
) -> EvidenceReadinessReport:
    dimensions = [
        DimensionReadiness(
            dimension=EvidenceDimension.FAULT_EXISTENCE,
            sufficient=sufficient,
            confirmed_evidence_ids=("issue-1",) if sufficient else (),
            missing_facts=(
                () if sufficient else ("Maintainer intent is not yet established.",)
            ),
            evidence_requests=requests,
        ),
        DimensionReadiness(
            dimension=EvidenceDimension.STUDY_SCOPE,
            sufficient=True,
            confirmed_evidence_ids=("issue-1",),
            missing_facts=(),
            evidence_requests=(),
        ),
    ]
    if include_repair:
        dimensions.append(
            DimensionReadiness(
                dimension=EvidenceDimension.REPAIR_CAUSALITY,
                sufficient=True,
                confirmed_evidence_ids=(repair_evidence_id,),
                missing_facts=(),
                evidence_requests=(),
            )
        )
    return EvidenceReadinessReport(
        task="stage2",
        dimensions=tuple(dimensions),
    )


def _ready(task: str, view: object) -> EvidenceReadinessReport:
    assert task == "stage2"
    is_issta = view.domain_profile == "issta2024"
    return _readiness_report(
        include_repair=is_issta,
        repair_evidence_id="code-1" if is_issta else "issue-1",
    )


def _fault_assessment(
    evidence_id: str = "issue-1",
    outcome: EvidenceTestOutcome = EvidenceTestOutcome.PASS,
) -> FaultEvidenceAssessment:
    return FaultEvidenceAssessment(
        outcome=outcome,
        claim="The frozen issue evidence establishes a concrete runtime failure.",
        supporting_evidence_ids=[evidence_id],
        counter_evidence_ids=[],
    )


def _scope_assessment(
    evidence_id: str = "issue-1",
    outcome: EvidenceTestOutcome = EvidenceTestOutcome.PASS,
) -> ScopeBoundaryAssessment:
    return ScopeBoundaryAssessment(
        outcome=outcome,
        claim="The reported runtime failure is included by the study policy.",
        supporting_evidence_ids=[evidence_id],
        counter_evidence_ids=[],
    )


def _repair_assessment(
    evidence_id: str = "code-1",
    outcome: EvidenceTestOutcome = EvidenceTestOutcome.PASS,
) -> RepairCausalityAssessment:
    return RepairCausalityAssessment(
        outcome=outcome,
        claim="The frozen code change directly repairs the documented mechanism.",
        supporting_evidence_ids=[evidence_id],
        counter_evidence_ids=[],
    )


@pytest.mark.parametrize(
    "role_kwargs",
    [
        {"repair_causality_analyst": lambda view: _repair_assessment()},
        {"fault_evidence_analyst": lambda view: _fault_assessment()},
        {"scope_boundary_analyst": lambda view: _scope_assessment()},
        {
            "fault_evidence_analyst": lambda view: _fault_assessment(),
            "repair_causality_analyst": lambda view: _repair_assessment(),
        },
        {
            "scope_boundary_analyst": lambda view: _scope_assessment(),
            "repair_causality_analyst": lambda view: _repair_assessment(),
        },
    ],
)
def test_stage2_controller_rejects_every_partial_role_owned_surface(
    role_kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="role-owned analysts must provide"):
        Stage2Controller(readiness=_ready, **role_kwargs)


@pytest.mark.parametrize(
    "role_kwargs",
    [
        {"fault_evidence_analyst": lambda view: _fault_assessment()},
        {"scope_boundary_analyst": lambda view: _scope_assessment()},
        {"repair_causality_analyst": lambda view: _repair_assessment()},
        {
            "fault_evidence_analyst": lambda view: _fault_assessment(),
            "scope_boundary_analyst": lambda view: _scope_assessment(),
        },
        {
            "fault_evidence_analyst": lambda view: _fault_assessment(),
            "scope_boundary_analyst": lambda view: _scope_assessment(),
            "repair_causality_analyst": lambda view: _repair_assessment(),
        },
    ],
)
def test_stage2_controller_rejects_legacy_mixed_with_any_role_surface(
    role_kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="legacy analyst cannot be mixed"):
        Stage2Controller(
            analyst=lambda team_id, view: _report(team_id),
            readiness=_ready,
            **role_kwargs,
        )


@pytest.mark.parametrize(
    ("domain", "include_repair", "message"),
    [
        ("ase2022", True, "ase2022 role-owned mode must omit"),
        ("issta2024", False, "issta2024 role-owned mode requires"),
    ],
)
def test_stage2_controller_rejects_illegal_domain_role_combination_before_readiness(
    domain: str,
    include_repair: bool,
    message: str,
) -> None:
    def must_not_run(*args: object) -> object:
        raise AssertionError(
            "invalid domain role wiring must fail before any role call"
        )

    controller = Stage2Controller(
        fault_evidence_analyst=must_not_run,
        scope_boundary_analyst=must_not_run,
        repair_causality_analyst=must_not_run if include_repair else None,
        readiness=must_not_run,
    )

    with pytest.raises(ValueError, match=message):
        controller.run(_ledger(domain))


def test_role_owned_ase_composition_skips_repair_and_returns_one_policy_report() -> (
    None
):
    calls: list[tuple[str, int]] = []

    def fault(view: object) -> FaultEvidenceAssessment:
        calls.append(("fault", view.ledger_version))
        return _fault_assessment()

    def scope(view: object) -> ScopeBoundaryAssessment:
        calls.append(("scope", view.ledger_version))
        return _scope_assessment()

    ledger = _ledger()
    ledger.append(_issue_item())
    result = Stage2Controller(
        fault_evidence_analyst=fault,
        scope_boundary_analyst=scope,
        readiness=_ready,
    ).run(ledger)

    assert sorted(calls) == [("fault", 1), ("scope", 1)]
    assert len(result.reports) == 1
    assert result.reports[0].team_id == "policy_composition"
    assert result.final_decision is not None
    assert result.final_decision.decision is Stage2Decision.ACCEPTED
    assert result.final_decision.source.value == "policy_composition"
    assert result.role_assessments is not None
    assert result.role_assessments.repair_causality is None


def test_role_owned_issta_composition_requires_repair_assessment() -> None:
    calls: list[str] = []
    ledger = _ledger("issta2024")
    ledger.append(_issue_item())
    ledger.append(_code_item())

    result = Stage2Controller(
        fault_evidence_analyst=lambda view: (
            calls.append("fault") or _fault_assessment()
        ),
        scope_boundary_analyst=lambda view: (
            calls.append("scope") or _scope_assessment()
        ),
        repair_causality_analyst=lambda view: (
            calls.append("repair") or _repair_assessment()
        ),
        readiness=_ready,
    ).run(ledger)

    assert sorted(calls) == ["fault", "repair", "scope"]
    assert result.final_decision is not None
    assert result.final_decision.decision is Stage2Decision.ACCEPTED


def test_role_owned_stage2_unknown_or_cross_role_citation_fails_closed() -> None:
    arbitrator_calls: list[object] = []
    ledger = _ledger()
    ledger.append(_issue_item())

    result = Stage2Controller(
        fault_evidence_analyst=lambda view: _fault_assessment("unknown-id"),
        scope_boundary_analyst=lambda view: _scope_assessment(),
        readiness=_ready,
        arbitrator=lambda packet: arbitrator_calls.append(packet),
    ).run(ledger)

    assert result.final_decision is None
    assert result.verification is None
    assert result.unresolved is not None
    assert result.unresolved.stop_reason == "stage2_role_assessment_invalid"
    assert "unknown evidence_id" in result.unresolved.missing_facts[0]
    assert arbitrator_calls == []


def test_teams_analyze_the_same_frozen_view_before_any_cross_team_step() -> None:
    observed: list[tuple[str, int, tuple[str, ...]]] = []
    readiness_events: list[int] = []
    barrier = threading.Barrier(2)

    def readiness(task: str, view: object) -> EvidenceReadinessReport:
        assert not observed
        readiness_events.append(view.ledger_version)
        return _readiness_report()

    def analyst(team_id: str, view: object) -> Stage2AnalysisReport:
        barrier.wait(timeout=2)
        observed.append(
            (
                team_id,
                view.ledger_version,
                tuple(item.evidence_id for item in view.items),
            )
        )
        return _report(team_id)

    ledger = _ledger()
    ledger.append(_issue_item())
    result = Stage2Controller(analyst=analyst, readiness=readiness).run(ledger)

    assert sorted(observed) == [
        ("A", 1, ("issue-1",)),
        ("B", 1, ("issue-1",)),
    ]
    assert readiness_events == [1]
    assert result.classification_ledger_version == 1
    assert result.readiness_report == _readiness_report()
    assert result.unresolved is None
    assert result.verification is not None
    assert result.verification.valid is True


def test_duplicate_readiness_requests_trigger_one_retrieval_before_classification() -> (
    None
):
    analysis_versions: list[tuple[str, int]] = []
    readiness_versions: list[int] = []
    specialist_calls: list[str] = []

    def analyst(team_id: str, view: object) -> Stage2AnalysisReport:
        analysis_versions.append((team_id, view.ledger_version))
        return _report(team_id)

    def readiness(task: str, view: object) -> EvidenceReadinessReport:
        readiness_versions.append(view.ledger_version)
        if view.ledger_version == 0:
            return _readiness_report(
                sufficient=False,
                requests=(_request("A"), _request("B")),
            )
        return _readiness_report()

    def specialist(request: EvidenceRequest, view: object) -> EvidenceDelta:
        specialist_calls.append(request.request_id)
        return EvidenceDelta(
            request_id=request.request_id,
            specialist=SpecialistType.ISSUE_PR,
            status=RetrievalStatus.FOUND,
            items=[_issue_item()],
        )

    result = Stage2Controller(
        analyst=analyst,
        readiness=readiness,
        specialists=SpecialistRegistry({SpecialistType.ISSUE_PR: specialist}),
    ).run(_ledger())

    assert specialist_calls == ["A-issue"]
    assert readiness_versions == [0, 1]
    assert analysis_versions == [("A", 1), ("B", 1)]
    assert result.retrieval_source_request_ids == [("A-issue", "B-issue")]
    assert result.verification is not None
    assert result.verification.valid is True


def test_same_round_readiness_keeps_distinct_query_and_bound_requests() -> None:
    calls: list[tuple[str, str, int]] = []
    first = _request("A").model_copy(update={"query": "allocator leak", "max_items": 1})
    second = _request("B").model_copy(update={"query": "null guard", "max_items": 3})

    def readiness(task: str, view: object) -> EvidenceReadinessReport:
        return _readiness_report(sufficient=False, requests=(first, second))

    def absent(request: EvidenceRequest, view: object) -> EvidenceDelta:
        calls.append((request.request_id, request.query, request.max_items))
        return EvidenceDelta(
            request_id=request.request_id,
            specialist=request.target_specialist,
            status=RetrievalStatus.ABSENT,
        )

    Stage2Controller(
        analyst=lambda team_id, view: _report(team_id),
        readiness=readiness,
        specialists=SpecialistRegistry({SpecialistType.ISSUE_PR: absent}),
    ).run(_ledger())

    assert calls == [
        ("A-issue", "allocator leak", 1),
        ("B-issue", "null guard", 3),
    ]


def test_later_readiness_round_retries_fact_with_new_query_and_bound() -> None:
    calls: list[tuple[str, int]] = []
    first = _request("A").model_copy(update={"query": "allocator leak", "max_items": 1})
    second = _request("B").model_copy(update={"query": "null guard", "max_items": 3})

    def readiness(task: str, view: object) -> EvidenceReadinessReport:
        if view.ledger_version == 0:
            return _readiness_report(sufficient=False, requests=(first,))
        if view.ledger_version == 1:
            return _readiness_report(sufficient=False, requests=(second,))
        return _readiness_report()

    def specialist(request: EvidenceRequest, view: object) -> EvidenceDelta:
        calls.append((request.query, request.max_items))
        number = len(calls)
        content = f"Retrieved evidence passage {number}."
        item = EvidenceItem(
            evidence_id=f"issue-{number}",
            record_id="record-1",
            source_type="issue_comment",
            source_uri=f"https://example.test/issues/1#comment-{number}",
            retrieved_at="2026-08-05T10:00:00Z",
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

    Stage2Controller(
        analyst=lambda team_id, view: _report(team_id),
        readiness=readiness,
        specialists=SpecialistRegistry({SpecialistType.ISSUE_PR: specialist}),
    ).run(_ledger())

    assert calls == [("allocator leak", 1), ("null guard", 3)]


def test_valid_json_from_both_teams_still_invokes_arbitration_on_disagreement() -> None:
    arbitration_calls: list[object] = []
    classification_views: list[object] = []

    def analyst(team_id: str, view: object) -> Stage2AnalysisReport:
        classification_views.append(view)
        decision = (
            Stage2Decision.ACCEPTED if team_id == "A" else Stage2Decision.REJECTED
        )
        return _report(team_id, decision=decision)

    def arbitrator(packet: object) -> Stage2ArbitrationDecision:
        arbitration_calls.append(packet)
        assert not hasattr(packet.report_a, "team_id")
        assert not hasattr(packet.report_b, "team_id")
        assert packet.disagreement.requires_arbitration is True
        assert packet.classification_ledger_version == 2
        assert tuple(item.evidence_id for item in packet.relevant_evidence) == (
            "issue-1",
        )
        assert packet.relevant_evidence[0].content == (
            "Maintainer: this rejection is a bug and should be fixed."
        )
        return Stage2ArbitrationDecision(
            resolution_status=ResolutionStatus.RESOLVED,
            decision=Stage2Decision.REJECTED,
            confidence=0.86,
            rationale="Fault existence is not established by the shared evidence.",
            supporting_evidence_ids=["issue-1"],
            resolved_dimensions=list(packet.disagreement.dimensions),
        )

    ledger = _ledger()
    ledger.append(_issue_item())
    ledger.append(_unrelated_item())
    result = Stage2Controller(
        analyst=analyst,
        readiness=_ready,
        arbitrator=arbitrator,
    ).run(ledger)

    assert len(arbitration_calls) == 1
    assert len({id(view) for view in classification_views}) == 2
    assert classification_views[0] == classification_views[1]
    assert {view.ledger_version for view in classification_views} == {2}
    assert result.final_decision is not None
    assert result.final_decision.decision is Stage2Decision.REJECTED
    assert result.verification is not None
    assert result.verification.valid is True


def test_stage2_disagreement_without_arbitrator_returns_unresolved() -> None:
    def analyst(team_id: str, view: object) -> Stage2AnalysisReport:
        return _report(
            team_id,
            decision=(
                Stage2Decision.ACCEPTED if team_id == "A" else Stage2Decision.REJECTED
            ),
        )

    ledger = _ledger()
    ledger.append(_issue_item())
    result = Stage2Controller(analyst=analyst, readiness=_ready).run(ledger)

    assert result.final_decision is None
    assert result.verification is None
    assert result.unresolved is not None
    assert result.unresolved.stop_reason == "arbitration_evidence_insufficient"
    assert result.unresolved.attempted_retrieval_statuses == ()


def test_unknown_stage2_citation_fails_closed_before_arbitrator() -> None:
    calls: list[object] = []

    def analyst(team_id: str, view: object) -> Stage2AnalysisReport:
        report = _report(
            team_id,
            decision=(
                Stage2Decision.ACCEPTED if team_id == "A" else Stage2Decision.REJECTED
            ),
        )
        return report.model_copy(
            update={"supporting_evidence_ids": ["missing-evidence"]}
        )

    ledger = _ledger()
    ledger.append(_issue_item())
    result = Stage2Controller(
        analyst=analyst,
        readiness=_ready,
        arbitrator=lambda packet: calls.append(packet),
    ).run(ledger)

    assert calls == []
    assert result.final_decision is None
    assert result.verification is None
    assert result.unresolved is not None
    assert "missing-evidence" in " ".join(result.unresolved.missing_facts)


@pytest.mark.parametrize("confirmed_ids", [(), ("unknown-readiness",)])
def test_invalid_stage2_readiness_citations_stop_before_classification(
    confirmed_ids: tuple[str, ...],
) -> None:
    analyst_calls: list[str] = []
    malformed = EvidenceReadinessReport.model_construct(
        task="stage2",
        dimensions=(
            DimensionReadiness.model_construct(
                dimension=EvidenceDimension.FAULT_EXISTENCE,
                sufficient=True,
                confirmed_evidence_ids=confirmed_ids,
                missing_facts=(),
                evidence_requests=(),
            ),
            DimensionReadiness(
                dimension=EvidenceDimension.STUDY_SCOPE,
                sufficient=True,
                confirmed_evidence_ids=("issue-1",),
                missing_facts=(),
                evidence_requests=(),
            ),
        ),
    )
    ledger = _ledger()
    ledger.append(_issue_item())

    result = Stage2Controller(
        analyst=lambda team_id, view: analyst_calls.append(team_id),
        readiness=lambda task, view: malformed,
    ).run(ledger)

    assert analyst_calls == []
    assert result.final_decision is None
    assert result.verification is None
    assert result.unresolved is not None
    assert EvidenceDimension.FAULT_EXISTENCE in result.unresolved.dimensions


def test_wrong_dimension_stage2_readiness_stops_before_classification() -> None:
    analyst_calls: list[str] = []
    readiness = EvidenceReadinessReport(
        task="stage2",
        dimensions=(
            DimensionReadiness(
                dimension=EvidenceDimension.FAULT_EXISTENCE,
                sufficient=True,
                confirmed_evidence_ids=("issue-1",),
                missing_facts=(),
                evidence_requests=(),
            ),
            DimensionReadiness(
                dimension=EvidenceDimension.STUDY_SCOPE,
                sufficient=True,
                confirmed_evidence_ids=("code-1",),
                missing_facts=(),
                evidence_requests=(),
            ),
        ),
    )
    ledger = _ledger()
    ledger.append(_issue_item())
    ledger.append(_code_item())

    result = Stage2Controller(
        analyst=lambda team_id, view: (
            analyst_calls.append(team_id) or _report(team_id)
        ),
        readiness=lambda task, view: readiness,
    ).run(ledger)

    assert analyst_calls == []
    assert result.final_decision is None
    assert result.verification is None
    assert result.unresolved is not None
    assert EvidenceDimension.STUDY_SCOPE in result.unresolved.dimensions


def test_partial_stage2_resolution_fails_closed() -> None:
    def analyst(team_id: str, view: object) -> Stage2AnalysisReport:
        return _report(
            team_id,
            decision=(
                Stage2Decision.ACCEPTED if team_id == "A" else Stage2Decision.REJECTED
            ),
        )

    def arbitrator(packet: object) -> Stage2ArbitrationDecision:
        assert {"decision", "evidence_tests"}.issubset(
            set(packet.disagreement.dimensions)
        )
        return Stage2ArbitrationDecision(
            resolution_status=ResolutionStatus.RESOLVED,
            decision=Stage2Decision.REJECTED,
            confidence=0.82,
            rationale="The evidence resolves only one of several active dimensions.",
            supporting_evidence_ids=["issue-1"],
            resolved_dimensions=["decision"],
        )

    ledger = _ledger()
    ledger.append(_issue_item())
    result = Stage2Controller(
        analyst=analyst,
        readiness=_ready,
        arbitrator=arbitrator,
    ).run(ledger)

    assert result.final_decision is None
    assert result.verification is None
    assert result.unresolved is not None
    assert set(result.unresolved.dimensions) == set(["decision", "evidence_tests"])


def test_unlisted_stage2_unresolved_dimension_is_not_emitted() -> None:
    def analyst(team_id: str, view: object) -> Stage2AnalysisReport:
        return _report(
            team_id,
            decision=(
                Stage2Decision.ACCEPTED if team_id == "A" else Stage2Decision.REJECTED
            ),
        )

    def arbitrator(packet: object) -> Stage2ArbitrationDecision:
        return Stage2ArbitrationDecision(
            resolution_status=ResolutionStatus.UNRESOLVED,
            rationale="The response names a dimension outside its packet authority.",
            unresolved_dimensions=["invented_dimension"],
            missing_facts=["The listed packet dimensions remain unresolved."],
        )

    ledger = _ledger()
    ledger.append(_issue_item())
    result = Stage2Controller(
        analyst=analyst,
        readiness=_ready,
        arbitrator=arbitrator,
    ).run(ledger)

    assert result.unresolved is not None
    assert "invented_dimension" not in result.unresolved.dimensions
    assert set(result.unresolved.dimensions) == {"decision", "evidence_tests"}


def test_classification_consumers_are_isolated_from_view_and_item_replacement() -> None:
    mutation_attempted = threading.Event()
    observed: dict[str, tuple[int, tuple[str, ...], tuple[str, ...]]] = {}
    issued_views: list[object] = []

    def analyst(team_id: str, view: object) -> Stage2AnalysisReport:
        if team_id == "A":
            object.__setattr__(view, "ledger_version", 999)
            object.__setattr__(view, "taxonomy", {"decision": ["tampered"]})
            object.__setattr__(view.items[0], "metadata", {"notes": ["tampered"]})
            mutation_attempted.set()
        else:
            assert mutation_attempted.wait(timeout=2)
        observed[team_id] = (
            view.ledger_version,
            tuple(view.taxonomy["decision"]),
            tuple(view.items[0].metadata["notes"]),
        )
        return _report(
            team_id,
            decision=(
                Stage2Decision.ACCEPTED if team_id == "A" else Stage2Decision.REJECTED
            ),
        )

    def arbitrator(packet: object) -> Stage2ArbitrationDecision:
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
        assert tuple(packet.taxonomy["decision"]) == (
            "accepted_fault",
            "rejected_candidate",
        )
        assert tuple(packet.relevant_evidence[0].metadata["notes"]) == ("stable",)
        object.__setattr__(
            packet.relevant_evidence[0],
            "metadata",
            {"notes": ["arbitrator-tampered"]},
        )
        assert tuple(canonical_view.items[0].metadata["notes"]) == ("stable",)
        return Stage2ArbitrationDecision(
            resolution_status=ResolutionStatus.RESOLVED,
            decision=Stage2Decision.REJECTED,
            confidence=0.86,
            rationale="The untouched canonical evidence does not establish a fault.",
            supporting_evidence_ids=["issue-1"],
            resolved_dimensions=["decision"],
        )

    ledger = _ledger()
    ledger.append(_issue_item().model_copy(update={"metadata": {"notes": ["stable"]}}))
    original_view = ledger.view

    def tracked_view(evidence_ids: object = None) -> object:
        view = original_view(evidence_ids)
        issued_views.append(view)
        return view

    ledger.view = tracked_view
    result = Stage2Controller(
        analyst=analyst,
        readiness=_ready,
        arbitrator=arbitrator,
    ).run(ledger)

    assert observed["A"] == (999, ("tampered",), ("tampered",))
    assert observed["B"] == (
        1,
        ("accepted_fault", "rejected_candidate"),
        ("stable",),
    )
    assert tuple(ledger.view().items[0].metadata["notes"]) == ("stable",)
    assert result.classification_ledger_version == 1


def test_stage2_result_preserves_positional_scope_exclusion_slot() -> None:
    exclusion = Stage2ScopeExclusion(
        rule_id="legacy-rule",
        boundary="The legacy positional scope boundary remains supported.",
        evidence_id="issue-1",
        source_uri="https://example.test/issues/1",
    )

    result = Stage2WorkflowResult((), None, None, (), [], False, exclusion)

    assert result.scope_exclusion is exclusion
    assert result.readiness_report is None


def test_issta_verifier_rejects_arbitrated_acceptance_without_repair_support() -> None:
    def analyst(team_id: str, view: object) -> Stage2AnalysisReport:
        return _report(
            team_id,
            decision=(
                Stage2Decision.ACCEPTED if team_id == "A" else Stage2Decision.REJECTED
            ),
            fault_outcome=EvidenceTestOutcome.PASS,
            repair_outcome=(
                EvidenceTestOutcome.UNKNOWN
                if team_id == "A"
                else EvidenceTestOutcome.FAIL
            ),
        )

    def arbitrator(packet: object) -> Stage2ArbitrationDecision:
        return Stage2ArbitrationDecision(
            resolution_status=ResolutionStatus.RESOLVED,
            decision=Stage2Decision.ACCEPTED,
            confidence=0.8,
            rationale=(
                "The issue describes a fault, but the supplied commit has no "
                "demonstrated causal repair."
            ),
            supporting_evidence_ids=["issue-1"],
            resolved_dimensions=["decision", "evidence_tests"],
        )

    ledger = _ledger("issta2024")
    ledger.append(_issue_item())
    ledger.append(_code_item())
    result = Stage2Controller(
        analyst=analyst,
        readiness=_ready,
        arbitrator=arbitrator,
    ).run(ledger)

    assert result.verification is not None
    assert result.verification.valid is False
    assert "issta2024 acceptance requires repair_causality=pass" in (
        result.verification.errors
    )


def test_specialist_call_budget_stops_additional_distinct_requests() -> None:
    calls: list[str] = []
    analysis_calls = 0

    def analyst(team_id: str, view: object) -> Stage2AnalysisReport:
        nonlocal analysis_calls
        analysis_calls += 1
        return _report(team_id)

    requests = tuple(
        _request(team_id).model_copy(
            update={
                "missing_fact": (
                    f"Whether maintainers confirm team {team_id} specific behavior"
                )
            }
        )
        for team_id in ("A", "B")
    )

    def readiness(task: str, view: object) -> EvidenceReadinessReport:
        return _readiness_report(sufficient=False, requests=requests)

    def unavailable(request: EvidenceRequest, view: object) -> EvidenceDelta:
        calls.append(request.request_id)
        return EvidenceDelta(
            request_id=request.request_id,
            specialist=request.target_specialist,
            status=RetrievalStatus.ABSENT,
        )

    result = Stage2Controller(
        analyst=analyst,
        readiness=readiness,
        specialists=SpecialistRegistry({SpecialistType.ISSUE_PR: unavailable}),
        config=Stage2WorkflowConfig(max_specialist_calls=1),
    ).run(_ledger())

    assert calls == ["A-issue"]
    assert analysis_calls == 0
    assert result.budget_exhausted is True
    assert result.final_decision is None
    assert result.unresolved is not None


def test_unavailable_request_is_not_repeated_after_other_evidence_arrives() -> None:
    calls: list[SpecialistType] = []

    def request_for(team_id: str, specialist: SpecialistType) -> EvidenceRequest:
        return _request(team_id).model_copy(
            update={
                "request_id": f"{team_id}-{specialist.value}",
                "target_specialist": specialist,
                "missing_fact": (
                    "Whether the linked source confirms the exact applied repair"
                ),
            }
        )

    def readiness(task: str, view: object) -> EvidenceReadinessReport:
        requests = [request_for("A", SpecialistType.COMMIT_HISTORY)]
        if view.ledger_version == 0:
            requests.append(request_for("A", SpecialistType.ISSUE_PR))
        return _readiness_report(
            sufficient=False,
            requests=tuple(requests),
        )

    def commit_specialist(request: EvidenceRequest, view: object) -> EvidenceDelta:
        calls.append(SpecialistType.COMMIT_HISTORY)
        return EvidenceDelta(
            request_id=request.request_id,
            specialist=SpecialistType.COMMIT_HISTORY,
            status=RetrievalStatus.UNAVAILABLE,
        )

    def issue_specialist(request: EvidenceRequest, view: object) -> EvidenceDelta:
        calls.append(SpecialistType.ISSUE_PR)
        return EvidenceDelta(
            request_id=request.request_id,
            specialist=SpecialistType.ISSUE_PR,
            status=RetrievalStatus.FOUND,
            items=[_issue_item()],
        )

    Stage2Controller(
        analyst=lambda team_id, view: _report(team_id),
        readiness=readiness,
        specialists=SpecialistRegistry(
            {
                SpecialistType.COMMIT_HISTORY: commit_specialist,
                SpecialistType.ISSUE_PR: issue_specialist,
            }
        ),
    ).run(_ledger())

    assert calls.count(SpecialistType.COMMIT_HISTORY) == 1
    assert calls.count(SpecialistType.ISSUE_PR) == 1


def test_no_progress_stops_readiness_without_classification_or_budget_flag() -> None:
    readiness_calls = 0
    analyst_calls = 0

    def readiness(task: str, view: object) -> EvidenceReadinessReport:
        nonlocal readiness_calls
        readiness_calls += 1
        return _readiness_report(
            sufficient=False,
            requests=(_request("A"),),
        )

    def analyst(team_id: str, view: object) -> Stage2AnalysisReport:
        nonlocal analyst_calls
        analyst_calls += 1
        return _report(team_id)

    def absent(request: EvidenceRequest, view: object) -> EvidenceDelta:
        return EvidenceDelta(
            request_id=request.request_id,
            specialist=request.target_specialist,
            status=RetrievalStatus.ABSENT,
        )

    result = Stage2Controller(
        analyst=analyst,
        readiness=readiness,
        specialists=SpecialistRegistry({SpecialistType.ISSUE_PR: absent}),
    ).run(_ledger())

    assert readiness_calls == 1
    assert analyst_calls == 0
    assert result.unresolved is not None
    assert result.budget_exhausted is False
    assert result.classification_ledger_version is None


def test_ase_pull_request_is_rejected_before_any_agent_call() -> None:
    ledger = _ledger()
    ledger.append(
        _issue_item().model_copy(
            update={
                "source_type": "record_summary",
                "source_uri": "https://github.com/tensorflow/tfjs/pull/123",
            }
        )
    )
    calls = 0

    def analyst(team_id: str, view: object) -> Stage2AnalysisReport:
        nonlocal calls
        calls += 1
        return _report(team_id)

    result = Stage2Controller(
        analyst=analyst,
        readiness=lambda task, view: (_ for _ in ()).throw(
            AssertionError("scope gate must run before readiness")
        ),
    ).run(ledger)

    assert calls == 0
    assert result.final_decision is not None
    assert result.final_decision.decision is Stage2Decision.REJECTED
    assert result.verification is not None
    assert result.verification.valid is True
    assert result.reports == ()
    assert result.final_decision.source.value == "deterministic_scope_gate"
    assert result.scope_exclusion is not None
    assert result.scope_exclusion.rule_id == "ase2022_issue_artifact_only"
    assert result.scope_exclusion.evidence_id == "issue-1"
    assert result.scope_exclusion.source_uri.endswith("/pull/123")
    assert result.readiness_report is None
    assert result.classification_ledger_version is None
    assert result.unresolved is None


def test_ase_issue_still_uses_independent_agents() -> None:
    calls: list[str] = []

    def analyst(team_id: str, view: object) -> Stage2AnalysisReport:
        calls.append(team_id)
        return _report(team_id)

    ledger = _ledger()
    ledger.append(_issue_item())
    result = Stage2Controller(analyst=analyst, readiness=_ready).run(ledger)

    assert sorted(calls) == ["A", "B"]
    assert result.final_decision is not None
    assert result.final_decision.decision is Stage2Decision.ACCEPTED


def test_ase_issue_with_linked_pull_request_evidence_still_uses_agents() -> None:
    calls: list[str] = []

    def analyst(team_id: str, view: object) -> Stage2AnalysisReport:
        calls.append(team_id)
        return _report(team_id)

    ledger = _ledger()
    ledger.append(_issue_item().model_copy(update={"source_type": "record_summary"}))
    ledger.append(
        _issue_item().model_copy(
            update={
                "evidence_id": "linked-pr",
                "source_type": "linked_pull_request",
                "source_uri": "https://github.com/tensorflow/tfjs/pull/123",
            }
        )
    )
    result = Stage2Controller(analyst=analyst, readiness=_ready).run(ledger)

    assert sorted(calls) == ["A", "B"]
    assert result.scope_exclusion is None


def test_issta_pull_request_still_uses_independent_agents() -> None:
    calls: list[str] = []

    def analyst(team_id: str, view: object) -> Stage2AnalysisReport:
        calls.append(team_id)
        return _report(team_id)

    ledger = _ledger(domain="issta2024")
    ledger.append(
        _issue_item().model_copy(
            update={"source_uri": "https://github.com/containerd/containerd/pull/123"}
        )
    )
    ledger.append(_code_item())
    result = Stage2Controller(analyst=analyst, readiness=_ready).run(ledger)

    assert sorted(calls) == ["A", "B"]
    assert result.final_decision is not None
    assert result.final_decision.decision is Stage2Decision.ACCEPTED
