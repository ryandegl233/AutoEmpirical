from __future__ import annotations

import hashlib

import pytest

from Benchmark.src.adaptive_empirical_workflow.contracts import (
    ArbitrationSource,
    CausalConsistencyReport,
    ConsistencyStatus,
    DimensionReadiness,
    DimensionVerificationReport,
    EvidenceDimension,
    EvidenceExplicitness,
    EvidenceItem,
    EvidenceReadinessReport,
    EvidenceSufficiency,
    JointAnchorReport,
    RootCauseReport,
    ResolutionStatus,
    Stage2AnalysisReport,
    Stage2ArbitrationDecision,
    Stage2Decision,
    Stage3TeamReport,
    SymptomReport,
    TeamCorrectionAudit,
    TestOutcome as EvidenceTestOutcome,
    VerificationVerdict,
)
from Benchmark.src.adaptive_empirical_workflow.disagreement import (
    build_stage2_disagreement_map,
    build_stage3_disagreement_map,
    resolve_stage2_reports,
    resolve_stage3_reports,
)
from Benchmark.src.adaptive_empirical_workflow.ledger import EvidenceLedger
from Benchmark.src.adaptive_empirical_workflow.verification import (
    verify_stage2_decision,
)


def _ledger() -> EvidenceLedger:
    content = "The patch restores the rejected request and adds a regression test."
    item = EvidenceItem(
        evidence_id="patch-1",
        record_id="record-1",
        source_type="patch",
        source_uri="https://example.test/commit/1",
        retrieved_at="2026-08-05T09:00:00Z",
        content=content,
        content_sha256=hashlib.sha256(content.encode()).hexdigest(),
        explicitness=EvidenceExplicitness.DIRECT,
    )
    return EvidenceLedger(
        record_id="record-1",
        task="Verify whether the change repairs a fault.",
        taxonomy={"decision": ["accepted_fault", "rejected_candidate"]},
        domain_profile="ase2022",
        initial_items=[item],
    )


def _report(
    team_id: str,
    *,
    decision: Stage2Decision = Stage2Decision.ACCEPTED,
    confidence: float = 0.9,
    sufficiency: EvidenceSufficiency = EvidenceSufficiency.SUFFICIENT,
    gaps: list[str] | None = None,
) -> Stage2AnalysisReport:
    outcomes = {
        "fault_existence": EvidenceTestOutcome.PASS,
        "repair_causality": EvidenceTestOutcome.PASS,
        "scope_exclusion": EvidenceTestOutcome.PASS,
    }
    if decision is Stage2Decision.REJECTED:
        outcomes["fault_existence"] = EvidenceTestOutcome.FAIL
    return Stage2AnalysisReport(
        team_id=team_id,
        decision=decision,
        confidence=confidence,
        fault_claim="The previous behavior rejects a valid runtime request.",
        repair_claim="The patch restores that request and tests the behavior.",
        supporting_evidence_ids=["patch-1"],
        evidence_tests=outcomes,
        alternative_hypothesis="The change could instead be a new feature.",
        decision_boundary="A demonstrated prior failure separates repair from feature work.",
        evidence_sufficiency=sufficiency,
        unresolved_evidence_gaps=gaps or [],
    )


def _stage3_team(team_id: str) -> Stage3TeamReport:
    return Stage3TeamReport(
        team_id=team_id,
        symptom=SymptomReport(
            label="unexpected_rejection",
            behavior_claim="A valid request is rejected before it can be processed.",
            supporting_evidence_ids=["issue-1"],
            alternative_label="wrong_output",
            boundary_reason="The operation is prevented rather than completed incorrectly.",
            confidence=0.9,
            evidence_sufficiency=EvidenceSufficiency.SUFFICIENT,
        ),
        root_cause=RootCauseReport(
            label="missing_null_check",
            defect_mechanism="A nullable cache entry is used without a defensive guard.",
            causal_chain=[
                "The cache returns a null entry.",
                "The handler assumes the entry exists.",
                "The request follows the rejection path.",
            ],
            supporting_evidence_ids=["issue-1"],
            alternative_label="incorrect_condition",
            boundary_reason="The conditional is correct after null entries are excluded.",
            confidence=0.88,
            evidence_sufficiency=EvidenceSufficiency.SUFFICIENT,
        ),
        consistency=CausalConsistencyReport(
            status=ConsistencyStatus.CONSISTENT,
            rationale="The missing guard explains how the null entry causes rejection.",
            supporting_evidence_ids=["issue-1"],
        ),
    )


def _stage3_readiness() -> EvidenceReadinessReport:
    return EvidenceReadinessReport(
        task="stage3",
        dimensions=tuple(
            DimensionReadiness(
                dimension=dimension,
                sufficient=True,
                confirmed_evidence_ids=("issue-1",),
                missing_facts=(),
                evidence_requests=(),
            )
            for dimension in (EvidenceDimension.SYMPTOM, EvidenceDimension.ROOT_CAUSE)
        ),
    )


def _insufficient_stage3_readiness() -> EvidenceReadinessReport:
    return EvidenceReadinessReport(
        task="stage3",
        dimensions=tuple(
            DimensionReadiness(
                dimension=dimension,
                sufficient=False,
                confirmed_evidence_ids=(),
                missing_facts=(f"Missing {dimension.value} confirmation.",),
                evidence_requests=(),
            )
            for dimension in (EvidenceDimension.SYMPTOM, EvidenceDimension.ROOT_CAUSE)
        ),
    )


def _evidence_item(
    evidence_id: str = "issue-1",
    *,
    source_type: str = "issue_body",
    explicitness: EvidenceExplicitness = EvidenceExplicitness.DIRECT,
    metadata: dict[str, object] | None = None,
) -> EvidenceItem:
    content = f"Evidence {evidence_id}: requests with null entries are rejected."
    return EvidenceItem(
        evidence_id=evidence_id,
        record_id="record-1",
        source_type=source_type,
        source_uri=f"https://example.test/evidence/{evidence_id}",
        retrieved_at="2026-08-05T09:00:00Z",
        content=content,
        content_sha256=hashlib.sha256(content.encode()).hexdigest(),
        explicitness=explicitness,
        metadata=metadata or {},
    )


def _issue_item() -> EvidenceItem:
    return _evidence_item()


def _stage3_taxonomy() -> dict[str, list[str]]:
    return {
        "symptom": ["unexpected_rejection", "wrong_output"],
        "root_cause": ["missing_null_check", "incorrect_condition"],
    }


def test_disagreement_map_localizes_decision_and_sufficiency_differences() -> None:
    disagreement = build_stage2_disagreement_map(
        _report("A"),
        _report(
            "B",
            decision=Stage2Decision.REJECTED,
            sufficiency=EvidenceSufficiency.INSUFFICIENT,
            gaps=["No issue discussion confirms the intended behavior."],
        ),
    )

    assert set(disagreement.dimensions) == {
        "decision",
        "evidence_tests",
        "evidence_sufficiency",
        "unresolved_evidence_gaps",
    }
    assert disagreement.requires_arbitration is True


def test_two_high_confidence_sufficient_reports_can_resolve_directly() -> None:
    result = resolve_stage2_reports(_report("A"), _report("B"))

    assert result.source is ArbitrationSource.DIRECT_CONSENSUS
    assert result.decision is Stage2Decision.ACCEPTED
    assert result.supporting_evidence_ids == ["patch-1"]


def test_stage3_gap_alone_does_not_create_relabel_authority() -> None:
    team_a = _stage3_team("A")
    team_b = _stage3_team("B")
    for team in (team_a, team_b):
        object.__setattr__(
            team,
            "symptom",
            team.symptom.model_copy(
                update={
                    "boundary_evidence_ids": ["issue-1"],
                    "confidence": 0.61,
                    "evidence_sufficiency": EvidenceSufficiency.INSUFFICIENT,
                    "unresolved_evidence_gaps": [
                        "The runtime reproduction is unavailable."
                    ],
                }
            ),
        )
        object.__setattr__(
            team,
            "root_cause",
            team.root_cause.model_copy(update={"boundary_evidence_ids": ["issue-1"]}),
        )
    evidence = _evidence_item(
        source_type="test_result",
        metadata={
            "evidence_capabilities": [
                "symptom_observation",
                "defect_mechanism",
            ]
        },
    )

    disagreement = build_stage3_disagreement_map(
        team_a,
        team_b,
        readiness=_insufficient_stage3_readiness(),
        evidence_items=(evidence,),
        taxonomy=_stage3_taxonomy(),
    )

    assert disagreement.dimensions == ()
    assert disagreement.requires_arbitration is False


def test_stage3_weak_consensus_preserves_labels_without_raising_confidence() -> None:
    team_a = _stage3_team("A")
    team_b = _stage3_team("B")
    team_a = team_a.model_copy(
        update={
            "symptom": team_a.symptom.model_copy(
                update={
                    "confidence": 0.61,
                    "unresolved_evidence_gaps": ["Runtime confirmation is absent."],
                }
            )
        }
    )
    team_b = team_b.model_copy(
        update={"symptom": team_b.symptom.model_copy(update={"confidence": 0.72})}
    )

    result = resolve_stage3_reports(team_a, team_b)

    assert result is not None
    assert result.symptom_label == "unexpected_rejection"
    assert result.root_cause_label == "missing_null_check"
    assert result.confidence == 0.61
    assert result.source is ArbitrationSource.FALLBACK_UNCERTAIN
    assert ArbitrationSource("fallback_uncertain") is result.source


@pytest.mark.parametrize(
    "status",
    (
        ConsistencyStatus.SYMPTOM_REVIEW,
        ConsistencyStatus.CAUSE_REVIEW,
        ConsistencyStatus.EVIDENCE_REQUEST,
    ),
)
def test_stage3_fallback_requires_at_least_one_consistent_team(
    status: ConsistencyStatus,
) -> None:
    teams = tuple(
        _stage3_team(team_id).model_copy(
            update={
                "consistency": _stage3_team(team_id).consistency.model_copy(
                    update={"status": status}
                )
            }
        )
        for team_id in ("A", "B")
    )

    result = resolve_stage3_reports(teams[0], teams[1], force_fallback=True)

    assert result is None


def test_stage3_fallback_selects_only_from_consistent_teams() -> None:
    team_a = _stage3_team("A")
    team_a = team_a.model_copy(
        update={
            "symptom": team_a.symptom.model_copy(
                update={"supporting_evidence_ids": ["consistent-only"]}
            ),
            "root_cause": team_a.root_cause.model_copy(
                update={"supporting_evidence_ids": ["consistent-only"]}
            ),
            "consistency": team_a.consistency.model_copy(
                update={"supporting_evidence_ids": ["consistent-only"]}
            ),
        }
    )
    team_b = _stage3_team("B")
    team_b = team_b.model_copy(
        update={
            "symptom": team_b.symptom.model_copy(update={"label": "wrong_output"}),
            "root_cause": team_b.root_cause.model_copy(
                update={"label": "incorrect_condition"}
            ),
            "consistency": team_b.consistency.model_copy(
                update={"status": ConsistencyStatus.CAUSE_REVIEW}
            ),
        }
    )

    result = resolve_stage3_reports(
        team_a,
        team_b,
        evidence_items=(_issue_item(),),
        force_fallback=True,
    )

    assert result is not None
    assert result.source is ArbitrationSource.FALLBACK_UNCERTAIN
    assert result.symptom_label == team_a.symptom.label
    assert result.root_cause_label == team_a.root_cause.label
    assert result.supporting_evidence_ids == ("consistent-only",)


def test_matching_labels_do_not_bypass_arbitration_when_evidence_is_insufficient() -> (
    None
):
    report_a = _report(
        "A",
        sufficiency=EvidenceSufficiency.INSUFFICIENT,
        gaps=["The regression test result is unavailable."],
    )
    report_b = _report(
        "B",
        sufficiency=EvidenceSufficiency.INSUFFICIENT,
        gaps=["The regression test result is unavailable."],
    )
    result = resolve_stage2_reports(
        report_a,
        report_b,
    )
    disagreement = build_stage2_disagreement_map(report_a, report_b)

    assert result is None
    assert disagreement.dimensions == ("consensus_gate",)
    assert disagreement.requires_arbitration is True


def test_matching_specific_root_causes_without_mechanism_evidence_trigger_risk() -> (
    None
):
    disagreement = build_stage3_disagreement_map(
        _stage3_team("A"),
        _stage3_team("B"),
        readiness=_stage3_readiness(),
        evidence_items=(_issue_item(),),
    )

    assert "unsupported_root_cause_specificity" in disagreement.dimensions
    assert disagreement.requires_arbitration is True


def test_stage3_shared_risk_map_covers_patch_symptom_and_failed_causality() -> None:
    team_a = _stage3_team("A")
    team_b = _stage3_team("B")
    failed = CausalConsistencyReport(
        status=ConsistencyStatus.CAUSE_REVIEW,
        rationale="The proposed mechanism does not explain the observed rejection.",
        supporting_evidence_ids=["issue-1"],
    )
    team_a = team_a.model_copy(update={"consistency": failed})
    team_b = team_b.model_copy(update={"consistency": failed})
    patch_content = (
        "The patch changes runtime error and failure handling in the request path."
    )
    patch_item = EvidenceItem(
        evidence_id="issue-1",
        record_id="record-1",
        source_type="patch",
        source_uri="https://example.test/commit/1",
        retrieved_at="2026-08-05T09:00:00Z",
        content=patch_content,
        content_sha256=hashlib.sha256(patch_content.encode()).hexdigest(),
        explicitness=EvidenceExplicitness.DIRECT,
    )

    disagreement = build_stage3_disagreement_map(
        team_a,
        team_b,
        readiness=_stage3_readiness(),
        evidence_items=(patch_item,),
    )

    assert "patch_only_symptom_evidence" in disagreement.dimensions
    assert "failed_causal_consistency" in disagreement.dimensions


def test_stage3_shared_risk_map_covers_missing_owned_evidence() -> None:
    team_a = _stage3_team("A")
    team_b = _stage3_team("B")
    missing_symptom = team_a.symptom.model_copy(
        update={"supporting_evidence_ids": ["missing-symptom"]}
    )
    missing_root = team_a.root_cause.model_copy(
        update={"supporting_evidence_ids": ["missing-root"]}
    )
    team_a = team_a.model_copy(
        update={"symptom": missing_symptom, "root_cause": missing_root}
    )
    team_b = team_b.model_copy(
        update={"symptom": missing_symptom, "root_cause": missing_root}
    )

    disagreement = build_stage3_disagreement_map(
        team_a,
        team_b,
        readiness=_stage3_readiness(),
        taxonomy=_stage3_taxonomy(),
        evidence_items=(_issue_item(),),
    )

    assert "missing_symptom_evidence" in disagreement.dimensions
    assert "missing_root_cause_evidence" in disagreement.dimensions
    assert "unknown_cited_evidence" in disagreement.dimensions


def test_stage3_shared_risk_map_covers_unsupported_neighbor_boundary() -> None:
    team_a = _stage3_team("A")
    team_b = _stage3_team("B")
    symptom = team_a.symptom.model_copy(
        update={
            "boundary_reason": (
                "No cited evidence separates these neighboring taxonomy labels."
            )
        }
    )
    team_a = team_a.model_copy(update={"symptom": symptom})
    team_b = team_b.model_copy(update={"symptom": symptom})

    disagreement = build_stage3_disagreement_map(
        team_a,
        team_b,
        readiness=_stage3_readiness(),
        taxonomy=_stage3_taxonomy(),
        evidence_items=(_issue_item(),),
    )

    assert "unsupported_symptom_boundary" in disagreement.dimensions


def test_nominal_consensus_on_same_inferred_claim_triggers_shared_risk() -> None:
    inferred = _issue_item().model_copy(
        update={"explicitness": EvidenceExplicitness.INFERRED}
    )

    disagreement = build_stage3_disagreement_map(
        _stage3_team("A"),
        _stage3_team("B"),
        readiness=_stage3_readiness(),
        evidence_items=(inferred,),
    )

    assert "shared_unsupported_inference" in disagreement.dimensions


def test_stage2_same_normalized_claim_with_different_inferred_ids_triggers_risk() -> (
    None
):
    report_a = _report("A").model_copy(
        update={
            "fault_claim": "The previous behavior rejects a valid runtime request.",
            "supporting_evidence_ids": ["inferred-a"],
        }
    )
    report_b = _report("B").model_copy(
        update={
            "fault_claim": "  THE previous behavior rejects a valid runtime request!  ",
            "supporting_evidence_ids": ["inferred-b"],
        }
    )

    disagreement = build_stage2_disagreement_map(
        report_a,
        report_b,
        evidence_items=(
            _evidence_item("inferred-a", explicitness=EvidenceExplicitness.INFERRED),
            _evidence_item("inferred-b", explicitness=EvidenceExplicitness.INFERRED),
        ),
    )

    assert "shared_unsupported_inference" in disagreement.dimensions


def test_stage2_unrelated_direct_evidence_does_not_launder_shared_inference() -> None:
    supporting_ids = ["inferred-fault", "direct-taxonomy"]
    report_a = _report("A").model_copy(
        update={"supporting_evidence_ids": supporting_ids}
    )
    report_b = _report("B").model_copy(
        update={"supporting_evidence_ids": supporting_ids}
    )

    disagreement = build_stage2_disagreement_map(
        report_a,
        report_b,
        evidence_items=(
            _evidence_item(
                "inferred-fault", explicitness=EvidenceExplicitness.INFERRED
            ),
            _evidence_item("direct-taxonomy", source_type="taxonomy"),
        ),
    )

    assert "shared_unsupported_inference" in disagreement.dimensions


def test_stage3_same_claim_with_different_inferred_ids_triggers_risk() -> None:
    team_a = _stage3_team("A")
    team_b = _stage3_team("B")
    team_a = team_a.model_copy(
        update={
            "symptom": team_a.symptom.model_copy(
                update={"supporting_evidence_ids": ["inferred-a"]}
            ),
            "root_cause": team_a.root_cause.model_copy(
                update={"supporting_evidence_ids": ["inferred-a"]}
            ),
        }
    )
    team_b = team_b.model_copy(
        update={
            "symptom": team_b.symptom.model_copy(
                update={
                    "behavior_claim": (
                        "  A VALID request is rejected before it can be processed! "
                    ),
                    "supporting_evidence_ids": ["inferred-b"],
                }
            ),
            "root_cause": team_b.root_cause.model_copy(
                update={"supporting_evidence_ids": ["inferred-b"]}
            ),
        }
    )

    disagreement = build_stage3_disagreement_map(
        team_a,
        team_b,
        taxonomy=_stage3_taxonomy(),
        evidence_items=(
            _evidence_item("inferred-a", explicitness=EvidenceExplicitness.INFERRED),
            _evidence_item("inferred-b", explicitness=EvidenceExplicitness.INFERRED),
        ),
    )

    assert "shared_unsupported_inference" in disagreement.dimensions


def test_stage3_unrelated_direct_consistency_evidence_does_not_launder_claim() -> None:
    inferred = _evidence_item(
        "inferred-claim", explicitness=EvidenceExplicitness.INFERRED
    )
    direct = _evidence_item("direct-unrelated", source_type="taxonomy")
    teams: list[Stage3TeamReport] = []
    for team_id in ("A", "B"):
        team = _stage3_team(team_id)
        teams.append(
            team.model_copy(
                update={
                    "symptom": team.symptom.model_copy(
                        update={"supporting_evidence_ids": ["inferred-claim"]}
                    ),
                    "root_cause": team.root_cause.model_copy(
                        update={"supporting_evidence_ids": ["inferred-claim"]}
                    ),
                    "consistency": team.consistency.model_copy(
                        update={"supporting_evidence_ids": ["direct-unrelated"]}
                    ),
                }
            )
        )

    disagreement = build_stage3_disagreement_map(
        teams[0],
        teams[1],
        taxonomy=_stage3_taxonomy(),
        evidence_items=(inferred, direct),
    )

    assert "shared_unsupported_inference" in disagreement.dimensions


def test_static_code_and_documentation_cannot_support_observable_symptom() -> None:
    for source_type in ("source_code", "documentation"):
        disagreement = build_stage3_disagreement_map(
            _stage3_team("A"),
            _stage3_team("B"),
            taxonomy=_stage3_taxonomy(),
            evidence_items=(_evidence_item(source_type=source_type),),
        )

        assert "missing_symptom_observation" in disagreement.dimensions


def test_adversarial_near_name_sources_do_not_gain_capabilities() -> None:
    barcode = build_stage3_disagreement_map(
        _stage3_team("A"),
        _stage3_team("B"),
        taxonomy=_stage3_taxonomy(),
        evidence_items=(_evidence_item(source_type="barcode_scan"),),
    )
    dispatch = build_stage3_disagreement_map(
        _stage3_team("A"),
        _stage3_team("B"),
        taxonomy=_stage3_taxonomy(),
        evidence_items=(_evidence_item(source_type="dispatch_log"),),
    )

    assert "unsupported_root_cause_specificity" in barcode.dimensions
    assert "missing_symptom_observation" in dispatch.dimensions
    assert "patch_only_symptom_evidence" not in dispatch.dimensions


def test_metadata_cannot_give_code_diff_observation_authority() -> None:
    item = _evidence_item(
        source_type="code_diff",
        metadata={"evidence_capabilities": ["symptom_observation"]},
    )
    team_a = _stage3_team("A")
    team_b = _stage3_team("B")
    for team in (team_a, team_b):
        object.__setattr__(
            team.symptom,
            "boundary_evidence_ids",
            ["issue-1"],
        )
        object.__setattr__(
            team.root_cause,
            "boundary_evidence_ids",
            ["issue-1"],
        )

    disagreement = build_stage3_disagreement_map(
        team_a,
        team_b,
        taxonomy=_stage3_taxonomy(),
        evidence_items=(item,),
    )

    assert "missing_symptom_observation" in disagreement.dimensions


def test_boundary_requires_taxonomy_member_and_direct_dimension_evidence() -> None:
    team_a = _stage3_team("A")
    team_b = _stage3_team("B")
    for team in (team_a, team_b):
        object.__setattr__(team.symptom, "alternative_label", "invented_neighbor")
        object.__setattr__(team.symptom, "boundary_evidence_ids", ["issue-1"])
        object.__setattr__(team.root_cause, "boundary_evidence_ids", ["issue-1"])

    disagreement = build_stage3_disagreement_map(
        team_a,
        team_b,
        taxonomy=_stage3_taxonomy(),
        evidence_items=(_issue_item(),),
    )

    assert "unsupported_symptom_boundary" in disagreement.dimensions


def test_plausible_boundary_prose_without_structured_evidence_is_unsupported() -> None:
    teams = (_stage3_team("A"), _stage3_team("B"))
    for team in teams:
        object.__setattr__(
            team.symptom,
            "boundary_reason",
            "The evidence appears to separate the two plausible labels.",
        )
        object.__setattr__(team.symptom, "boundary_evidence_ids", [])

    disagreement = build_stage3_disagreement_map(
        *teams,
        taxonomy=_stage3_taxonomy(),
        evidence_items=(_issue_item(),),
    )

    assert "unsupported_symptom_boundary" in disagreement.dimensions


def test_direct_dimension_boundary_evidence_avoids_keyword_false_positive() -> None:
    observation = _evidence_item("observation", source_type="issue_body")
    mechanism = _evidence_item("mechanism", source_type="code_diff")
    teams = (_stage3_team("A"), _stage3_team("B"))
    for team in teams:
        object.__setattr__(
            team.symptom,
            "supporting_evidence_ids",
            ["observation"],
        )
        object.__setattr__(
            team.symptom,
            "boundary_evidence_ids",
            ["observation"],
        )
        object.__setattr__(
            team.symptom,
            "boundary_reason",
            "The alternative is not supported because the request is rejected.",
        )
        object.__setattr__(
            team.root_cause,
            "supporting_evidence_ids",
            ["mechanism"],
        )
        object.__setattr__(
            team.root_cause,
            "boundary_evidence_ids",
            ["mechanism"],
        )

    disagreement = build_stage3_disagreement_map(
        *teams,
        taxonomy=_stage3_taxonomy(),
        evidence_items=(observation, mechanism),
    )

    assert "unsupported_symptom_boundary" not in disagreement.dimensions
    assert "unsupported_root_cause_boundary" not in disagreement.dimensions


def test_stage2_disagreement_map_fails_closed_on_readiness_and_unknown_citations() -> (
    None
):
    readiness = EvidenceReadinessReport(
        task="stage2",
        dimensions=(
            DimensionReadiness(
                dimension=EvidenceDimension.FAULT_EXISTENCE,
                sufficient=False,
                confirmed_evidence_ids=(),
                missing_facts=("Fault existence is not established.",),
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
    report_a = _report("A").model_copy(
        update={"supporting_evidence_ids": ["missing-evidence"]}
    )
    report_b = _report("B").model_copy(
        update={"supporting_evidence_ids": ["missing-evidence"]}
    )

    disagreement = build_stage2_disagreement_map(
        report_a,
        report_b,
        readiness=readiness,
        evidence_items=(_issue_item(),),
    )

    assert "insufficient_required_readiness" in disagreement.dimensions
    assert "unknown_cited_evidence" in disagreement.dimensions


def test_empty_frozen_snapshot_still_rejects_unknown_citations() -> None:
    stage2 = build_stage2_disagreement_map(
        _report("A"),
        _report("B"),
        evidence_items=(),
    )
    stage3 = build_stage3_disagreement_map(
        _stage3_team("A"),
        _stage3_team("B"),
        evidence_items=(),
    )

    assert "unknown_cited_evidence" in stage2.dimensions
    assert "unknown_cited_evidence" in stage3.dimensions


def test_ase_accepted_consensus_ignores_repair_only_gap_and_lower_confidence() -> None:
    report_a = _report(
        "A",
        confidence=0.72,
        gaps=["No commit, pull request, patch, or regression test is available."],
    )
    report_b = _report(
        "B",
        confidence=0.72,
        gaps=["No commit, pull request, patch, or regression test is available."],
    )
    repair_unknown = {
        **report_a.evidence_tests,
        "repair_causality": EvidenceTestOutcome.UNKNOWN,
    }
    repair_failed = {
        **report_b.evidence_tests,
        "repair_causality": EvidenceTestOutcome.FAIL,
    }
    report_a = report_a.model_copy(update={"evidence_tests": repair_unknown})
    report_b = report_b.model_copy(update={"evidence_tests": repair_failed})

    result = resolve_stage2_reports(
        report_a,
        report_b,
        domain="ase2022",
    )
    disagreement = build_stage2_disagreement_map(
        report_a,
        report_b,
        domain="ase2022",
    )

    assert result is not None
    assert result.decision is Stage2Decision.ACCEPTED
    assert result.source is ArbitrationSource.DIRECT_CONSENSUS
    assert disagreement.requires_arbitration is False


def test_arbitrated_decision_preserves_its_source_and_selected_evidence() -> None:
    arbitration = Stage2ArbitrationDecision(
        resolution_status=ResolutionStatus.RESOLVED,
        decision=Stage2Decision.REJECTED,
        confidence=0.84,
        rationale="The patch evidence does not establish that prior behavior was faulty.",
        supporting_evidence_ids=["patch-1"],
        resolved_dimensions=["decision"],
    )
    result = resolve_stage2_reports(
        _report("A"),
        _report("B", decision=Stage2Decision.REJECTED),
        arbitration=arbitration,
    )

    assert result is not None
    assert result.source is ArbitrationSource.TARGETED_ARBITRATION
    assert result.decision is Stage2Decision.REJECTED


def test_verifier_rejects_unknown_evidence_without_changing_the_label() -> None:
    arbitration = Stage2ArbitrationDecision(
        resolution_status=ResolutionStatus.RESOLVED,
        decision=Stage2Decision.REJECTED,
        confidence=0.84,
        rationale="The competing hypothesis better explains the available patch.",
        supporting_evidence_ids=["missing-evidence"],
        resolved_dimensions=["decision"],
    )
    result = resolve_stage2_reports(
        _report("A"),
        _report("B", decision=Stage2Decision.REJECTED),
        arbitration=arbitration,
    )

    verification = verify_stage2_decision(result, _ledger())

    assert verification.valid is False
    assert verification.decision is Stage2Decision.REJECTED
    assert "unknown evidence_id 'missing-evidence'" in verification.errors


def _anchored_disagreement_team(
    team_id: str,
    *,
    symptom_verdict: VerificationVerdict,
) -> Stage3TeamReport:
    base = _stage3_team(team_id)
    symptom_review = DimensionVerificationReport(
        dimension=EvidenceDimension.SYMPTOM,
        verdict=symptom_verdict,
        anchor_label=base.symptom.label,
        alternative_label=(
            "wrong_output" if symptom_verdict is VerificationVerdict.REJECT else None
        ),
        rationale="The verifier records a bounded view without owning arbitration.",
        supporting_evidence_ids=["issue-1"],
        counter_evidence_ids=(
            ["issue-1"] if symptom_verdict is VerificationVerdict.REJECT else []
        ),
        corrected_claim=(
            "The request completes with an incorrect observable result."
            if symptom_verdict is VerificationVerdict.REJECT
            else None
        ),
        confidence=0.82,
    )
    root_review = DimensionVerificationReport(
        dimension=EvidenceDimension.ROOT_CAUSE,
        verdict=VerificationVerdict.ACCEPT,
        anchor_label=base.root_cause.label,
        rationale="The verifier independently confirms the frozen root cause.",
        supporting_evidence_ids=["issue-1"],
        confidence=0.82,
    )
    return base.model_copy(
        update={
            "anchor": JointAnchorReport(
                symptom=base.symptom,
                root_cause=base.root_cause,
                causal_account="The missing guard explains the observable rejection path.",
                shared_supporting_evidence_ids=["issue-1"],
            ),
            "verifications": (symptom_review, root_review),
            "correction_audit": (
                TeamCorrectionAudit(
                    dimension=EvidenceDimension.SYMPTOM,
                    anchor_label=base.symptom.label,
                    proposed_label=symptom_review.alternative_label,
                    final_label=base.symptom.label,
                    accepted=False,
                    reason="The composed label remains frozen for this direct test.",
                ),
                TeamCorrectionAudit(
                    dimension=EvidenceDimension.ROOT_CAUSE,
                    anchor_label=base.root_cause.label,
                    final_label=base.root_cause.label,
                    accepted=False,
                    reason="The anchor root cause remains unchanged.",
                ),
            ),
        }
    )


def test_anchor_or_verifier_opposition_does_not_create_arbitration_dimension() -> None:
    disagreement = build_stage3_disagreement_map(
        _anchored_disagreement_team(
            "A",
            symptom_verdict=VerificationVerdict.ACCEPT,
        ),
        _anchored_disagreement_team(
            "B",
            symptom_verdict=VerificationVerdict.REJECT,
        ),
    )

    assert disagreement.dimensions == ()
    assert disagreement.requires_arbitration is False


def test_disagreement_unknown_scan_includes_nested_anchor_provenance() -> None:
    team_a = _anchored_disagreement_team(
        "A",
        symptom_verdict=VerificationVerdict.ACCEPT,
    )
    assert team_a.anchor is not None
    team_a = team_a.model_copy(
        update={
            "anchor": team_a.anchor.model_copy(
                update={"shared_counter_evidence_ids": ["unknown-anchor"]}
            )
        }
    )

    disagreement = build_stage3_disagreement_map(
        team_a,
        _anchored_disagreement_team(
            "B",
            symptom_verdict=VerificationVerdict.ACCEPT,
        ),
        evidence_items=(_issue_item(),),
    )

    assert disagreement.details["unknown_cited_evidence"] == {
        "evidence_ids": ("unknown-anchor",)
    }
