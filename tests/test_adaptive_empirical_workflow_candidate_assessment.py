from __future__ import annotations

import hashlib
import json

import pytest
from pydantic import ValidationError

from Benchmark.src.adaptive_empirical_workflow.agents import (
    StructuredModelClient,
    StructuredOutputError,
    StructuredRoleAgents,
)
from Benchmark.src.adaptive_empirical_workflow.contracts import (
    BaselineAnchor,
    BaselineRevisionAssessment,
    BoundaryCard,
    EvidenceDimension,
    EvidenceExplicitness,
    EvidenceItem,
    EvidenceView,
    FalsificationCoverageCrossResult,
    RevisionAssessmentVerdict,
    RevisionEntailmentResult,
    RevisionFalsificationResult,
    RevisionConsistencyReport,
    RevisionProposalEnvelope,
    baseline_revision_assessment_digest,
    canonical_evidence_view_hash,
    normalize_revision_entailment_result,
    normalize_revision_falsification_result,
    revision_proposal_digest,
)


def _view() -> EvidenceView:
    content = "Execution completes but exceeds the documented latency budget."
    unrelated_content = "An internal branch is described without an observable outcome."
    return EvidenceView(
        record_id="synthetic-record",
        task="Classify the frozen evidence.",
        taxonomy={
            "symptom": ["Runtime Termination", "Slow Completion"],
            "root_cause": ["Faulty Branch", "Interface Misuse"],
        },
        domain_profile="synthetic-domain",
        ledger_version=1,
        items=(
            EvidenceItem(
                evidence_id="issue-observation",
                record_id="synthetic-record",
                source_type="issue_body",
                source_uri="https://example.test/observations/1",
                retrieved_at="2026-08-16T00:00:00Z",
                content=content,
                content_sha256=hashlib.sha256(content.encode()).hexdigest(),
                explicitness=EvidenceExplicitness.DIRECT,
            ),
            EvidenceItem(
                evidence_id="unrelated-code",
                record_id="synthetic-record",
                source_type="code_context",
                source_uri="https://example.test/code/1",
                retrieved_at="2026-08-16T00:00:00Z",
                content=unrelated_content,
                content_sha256=hashlib.sha256(unrelated_content.encode()).hexdigest(),
                explicitness=EvidenceExplicitness.DIRECT,
            ),
        ),
    )


def _anchor() -> BaselineAnchor:
    return BaselineAnchor(
        record_id="synthetic-record",
        valid=True,
        symptom_label="Runtime Termination",
        root_cause_label="Faulty Branch",
        source_config_hash="a" * 64,
        source_predictions_sha256="b" * 64,
    )


def _card() -> BoundaryCard:
    return BoundaryCard.model_validate(
        {
            "card_id": "runtime-termination--slow-completion",
            "dimension": "symptom",
            "labels": ["Runtime Termination", "Slow Completion"],
            "semantic_origin": "operational_definition",
            "decision_question": "Does execution terminate or finish slowly?",
            "observable_slots": ["completion outcome", "elapsed duration"],
            "criteria": [
                {
                    "label": "Runtime Termination",
                    "positive_conditions": ["execution terminates without a result"],
                    "exclusion_conditions": ["execution completes with a result"],
                },
                {
                    "label": "Slow Completion",
                    "positive_conditions": [
                        "execution completes beyond its latency budget"
                    ],
                    "exclusion_conditions": ["execution terminates without a result"],
                },
            ],
        }
    )


def _proposal() -> RevisionProposalEnvelope:
    view = _view()
    return RevisionProposalEnvelope(
        dimension=EvidenceDimension.SYMPTOM,
        baseline_label="Runtime Termination",
        proposed_label="Slow Completion",
        boundary_card_id=_card().card_id,
        proposer_team_ids=("B",),
        proposer_report_digests=("c" * 64,),
        supporting_evidence_ids=("issue-observation",),
        baseline_source_config_hash="a" * 64,
        baseline_source_predictions_sha256="b" * 64,
        taxonomy_structure_hash="d" * 64,
        evidence_view_hash=canonical_evidence_view_hash(view),
    )


def _raw_assessment(
    team_id: str,
    *,
    revises: bool = True,
) -> RevisionEntailmentResult | RevisionFalsificationResult:
    if team_id == "A":
        return RevisionEntailmentResult.model_validate(
            {
                "condition_findings": [
                    {
                        "condition": "execution completes beyond its latency budget",
                        "status": "supported" if revises else "refuted",
                        "citation_ids": ["issue-observation"],
                    }
                ],
                "baseline_exclusion_finding": {
                    "condition": "execution completes with a result",
                    "status": "supported" if revises else "refuted",
                    "citation_ids": ["issue-observation"],
                },
                "outcome": "entailed" if revises else "not_entailed",
                "summary": "The bounded evidence is mapped to every exact card condition.",
            }
        )
    return RevisionFalsificationResult.model_validate(
        {
            "baseline_survival_finding": {
                "condition": "execution terminates without a result",
                "status": "refuted" if revises else "supported",
                "citation_ids": ["issue-observation"],
            },
            "proposed_defeater_findings": [
                {
                    "condition": "execution terminates without a result",
                    "status": "refuted" if revises else "supported",
                    "citation_ids": ["issue-observation"],
                }
            ],
            "strongest_competing_reading": {
                "label": "Runtime Termination",
                "summary": "The exact Baseline label is the strongest competing reading.",
                "citation_ids": ["issue-observation"],
            },
            "outcome": "revision_survives" if revises else "revision_falsified",
            "summary": "The revision was checked against every fixed falsifier.",
        }
    )


def _assessment(
    team_id: str = "A", *, revises: bool = True
) -> BaselineRevisionAssessment:
    proposal = _proposal()
    raw = _raw_assessment(team_id, revises=revises)
    if team_id == "A":
        return normalize_revision_entailment_result(
            result=raw,
            proposal=proposal,
            routed_card=_card(),
        )
    return normalize_revision_falsification_result(
        result=raw,
        proposal=proposal,
        routed_card=_card(),
    )


def _repair_causality_fixture() -> (
    tuple[EvidenceView, BaselineAnchor, BoundaryCard, RevisionProposalEnvelope]
):
    content = (
        "The maintainer repair replaces an incompatible dependency version. "
        "BUILD and package files are edited only to apply that replacement."
    )
    view = EvidenceView(
        record_id="repair-causality-record",
        task="Classify the frozen repair mechanism.",
        taxonomy={
            "symptom": ["Build Failure", "Crash"],
            "root_cause": ["Misconfiguration", "Dependency Error"],
        },
        domain_profile="ase2022",
        ledger_version=1,
        items=(
            EvidenceItem(
                evidence_id="maintainer-repair",
                record_id="repair-causality-record",
                source_type="issue_body",
                source_uri="https://example.test/issues/2",
                retrieved_at="2026-08-17T00:00:00Z",
                content=content,
                content_sha256=hashlib.sha256(content.encode()).hexdigest(),
                explicitness=EvidenceExplicitness.DIRECT,
            ),
        ),
    )
    anchor = BaselineAnchor(
        record_id=view.record_id,
        valid=True,
        symptom_label="Build Failure",
        root_cause_label="Misconfiguration",
        source_config_hash="1" * 64,
        source_predictions_sha256="2" * 64,
    )
    card = BoundaryCard.model_validate(
        {
            "card_id": "misconfiguration--dependency-error",
            "dimension": "root_cause",
            "labels": ["Misconfiguration", "Dependency Error"],
            "semantic_origin": "operational_definition",
            "decision_question": "Is the pre-fault cause configuration or dependency incompatibility?",
            "observable_slots": ["pre-fault cause", "repair mechanism"],
            "criteria": [
                {
                    "label": "Misconfiguration",
                    "positive_conditions": [
                        "invalid project configuration causes the failure"
                    ],
                    "exclusion_conditions": [
                        "an incompatible external dependency causes the failure"
                    ],
                },
                {
                    "label": "Dependency Error",
                    "positive_conditions": [
                        "an incompatible external dependency causes the failure"
                    ],
                    "exclusion_conditions": [
                        "project configuration alone causes the failure"
                    ],
                },
            ],
        }
    )
    proposal = RevisionProposalEnvelope(
        dimension=EvidenceDimension.ROOT_CAUSE,
        baseline_label="Misconfiguration",
        proposed_label="Dependency Error",
        boundary_card_id=card.card_id,
        proposer_team_ids=("A",),
        proposer_report_digests=("3" * 64,),
        supporting_evidence_ids=("maintainer-repair",),
        baseline_source_config_hash=anchor.source_config_hash,
        baseline_source_predictions_sha256=anchor.source_predictions_sha256,
        taxonomy_structure_hash="4" * 64,
        evidence_view_hash=canonical_evidence_view_hash(view),
    )
    return view, anchor, card, proposal


def test_entailment_uses_direct_causal_mechanism_without_taxonomy_word_match() -> None:
    """Dropping mechanism-over-wording guidance must leave the proposal insufficient."""

    view, anchor, card, proposal = _repair_causality_fixture()

    def transport(system: str, _user: str) -> str:
        follows_causal_mechanism = (
            "does not need to repeat the proposed taxonomy label verbatim" in system
        )
        status = "supported" if follows_causal_mechanism else "insufficient"
        outcome = "entailed" if follows_causal_mechanism else "insufficient"
        return json.dumps(
            {
                "condition_findings": [
                    {
                        "condition": "an incompatible external dependency causes the failure",
                        "status": status,
                        "citation_ids": (
                            ["maintainer-repair"] if follows_causal_mechanism else []
                        ),
                    }
                ],
                "baseline_exclusion_finding": {
                    "condition": "an incompatible external dependency causes the failure",
                    "status": status,
                    "citation_ids": (
                        ["maintainer-repair"] if follows_causal_mechanism else []
                    ),
                },
                "outcome": outcome,
                "summary": "The exact causal mechanism is tested against the card boundary.",
            }
        )

    result = StructuredRoleAgents(
        StructuredModelClient(transport, max_schema_retries=0)
    ).baseline_revision_assessment("A", proposal, anchor, card, view)

    assert result.verdict is RevisionAssessmentVerdict.REVISE


def test_falsification_does_not_confuse_repair_surface_with_prefault_cause() -> None:
    """Dropping causal-direction guidance must incorrectly preserve configuration."""

    view, anchor, card, proposal = _repair_causality_fixture()

    def transport(system: str, _user: str) -> str:
        separates_repair_surface = all(
            phrase in system
            for phrase in (
                "pre-fault causal mechanism",
                "files, tools, or configuration surfaces touched by the repair",
            )
        )
        refutation_status = "refuted" if separates_repair_surface else "supported"
        outcome = (
            "revision_survives" if separates_repair_surface else "revision_falsified"
        )
        return json.dumps(
            {
                "baseline_survival_finding": {
                    "condition": "invalid project configuration causes the failure",
                    "status": refutation_status,
                    "citation_ids": ["maintainer-repair"],
                },
                "proposed_defeater_findings": [
                    {
                        "condition": "project configuration alone causes the failure",
                        "status": refutation_status,
                        "citation_ids": ["maintainer-repair"],
                    }
                ],
                "strongest_competing_reading": {
                    "label": "Misconfiguration",
                    "summary": "The repair surface is considered without treating it as the pre-fault cause.",
                    "citation_ids": ["maintainer-repair"],
                },
                "outcome": outcome,
                "summary": "Causal direction is separated from the artifacts used to apply the repair.",
            }
        )

    result = StructuredRoleAgents(
        StructuredModelClient(transport, max_schema_retries=0)
    ).baseline_revision_assessment("B", proposal, anchor, card, view)

    assert result.verdict is RevisionAssessmentVerdict.REVISE


def test_candidate_bound_assessment_prompts_isolate_proposer_identity_and_full_card_graph() -> (
    None
):
    proposal = _proposal()
    card = _card()
    captured: dict[str, tuple[str, dict[str, object]]] = {}

    def transport(system: str, user: str) -> str:
        payload = json.loads(user)
        captured[payload["team_id"]] = (system, payload)
        return _raw_assessment(payload["team_id"]).model_dump_json()

    agents = StructuredRoleAgents(StructuredModelClient(transport))
    for team_id in ("A", "B"):
        assert agents.baseline_revision_assessment(
            team_id,
            proposal,
            _anchor(),
            card,
            _view().model_copy(deep=True),
        ) == _assessment(team_id)

    system_a, prompt_a = captured["A"]
    system_b, prompt_b = captured["B"]
    context_a = prompt_a["context"]
    context_b = prompt_b["context"]
    assert context_a["proposal"] == context_b["proposal"]
    assert context_a["boundary_card"] == context_b["boundary_card"]
    assert set(context_a["required_findings"]) == {
        "proposed_positive_conditions",
        "baseline_exclusion_condition",
        "allowed_positive_citation_ids",
        "allowed_exclusion_citation_ids",
    }
    assert set(context_b["required_findings"]) == {
        "baseline_survival_condition",
        "proposed_defeater_conditions",
        "strongest_competing_reading_label",
        "allowed_citation_ids",
    }
    assert "proposal_digest" not in json.dumps(context_a)
    assert context_a["proposal"]["baseline_label"] == "Runtime Termination"
    assert context_a["proposal"]["proposed_label"] == "Slow Completion"
    assert context_a["proposal"]["supporting_evidence_ids"] == ["issue-observation"]
    assert context_a["boundary_card"]["card_id"] == card.card_id
    serialized = json.dumps([context_a, context_b], ensure_ascii=False)
    assert "proposer_team_ids" not in serialized
    assert "proposer_report_digests" not in serialized
    assert "Faulty Branch" not in serialized
    assert "Interface Misuse" not in serialized
    assert [item["evidence_id"] for item in prompt_a["evidence_view"]["items"]] == [
        "issue-observation"
    ]
    assert "gold" not in (system_a + system_b + serialized).lower()
    assert "ROLE: revision_entailment" in system_a
    assert "ROLE: revision_falsification" in system_b


@pytest.mark.parametrize("team_id", ["A", "B"])
def test_prompt_keeps_identity_and_provenance_for_nonrevision_verdicts(
    team_id: str,
) -> None:
    proposal = _proposal()
    result = StructuredRoleAgents(
        StructuredModelClient(
            lambda _system, _user: _raw_assessment(
                team_id, revises=False
            ).model_dump_json(),
            max_schema_retries=0,
        )
    ).baseline_revision_assessment(team_id, proposal, _anchor(), _card(), _view())

    expected_verdict = (
        RevisionAssessmentVerdict.INSUFFICIENT
        if team_id == "A"
        else RevisionAssessmentVerdict.PRESERVE
    )
    assert result.verdict is expected_verdict
    assert result.assessor_team_id == team_id
    assert result.proposal_digest == revision_proposal_digest(proposal)
    assert result.dimension is proposal.dimension
    assert result.baseline_source_config_hash == proposal.baseline_source_config_hash
    assert (
        result.baseline_source_predictions_sha256
        == proposal.baseline_source_predictions_sha256
    )
    assert result.taxonomy_structure_hash == proposal.taxonomy_structure_hash
    assert result.evidence_view_hash == proposal.evidence_view_hash
    assert result.proposed_label is None
    assert result.boundary_card_id is None


@pytest.mark.parametrize(
    "forged_field",
    [
        "assessor_team_id",
        "proposal_digest",
        "dimension",
        "baseline_source_config_hash",
        "baseline_source_predictions_sha256",
        "taxonomy_structure_hash",
        "evidence_view_hash",
    ],
)
def test_model_visible_raw_contract_omits_program_owned_bindings(
    forged_field: str,
) -> None:
    captured_system = ""

    def transport(system: str, _user: str) -> str:
        nonlocal captured_system
        captured_system = system
        payload = _raw_assessment("A").model_dump(mode="json")
        payload[forged_field] = "forged"
        return json.dumps(payload)

    with pytest.raises(StructuredOutputError):
        StructuredRoleAgents(
            StructuredModelClient(transport, max_schema_retries=0)
        ).baseline_revision_assessment("A", _proposal(), _anchor(), _card(), _view())
    assert forged_field not in RevisionEntailmentResult.model_fields
    assert "proposal_digest" not in captured_system


def test_prompt_allows_owned_evidence_overlap_when_proposal_counter_pool_is_empty() -> (
    None
):
    proposal = _proposal()
    assert proposal.counter_evidence_ids == ()

    result = StructuredRoleAgents(
        StructuredModelClient(
            lambda _system, _user: _raw_assessment("B").model_dump_json(),
            max_schema_retries=0,
        )
    ).baseline_revision_assessment("B", proposal, _anchor(), _card(), _view())

    assert result.verdict is RevisionAssessmentVerdict.REVISE
    assert result.supporting_evidence_ids == ("issue-observation",)
    assert result.counter_evidence_ids == ("issue-observation",)


def test_falsification_prompt_supplies_exact_ordered_required_findings() -> None:
    captured_context: dict[str, object] = {}

    def transport(_system: str, user: str) -> str:
        context = json.loads(user)["context"]
        captured_context.update(context)
        required = context["required_findings"]
        citation_id = required["allowed_citation_ids"][0]
        return json.dumps(
            {
                "baseline_survival_finding": {
                    "condition": required["baseline_survival_condition"],
                    "status": "refuted",
                    "citation_ids": [citation_id],
                },
                "proposed_defeater_findings": [
                    {
                        "condition": condition,
                        "status": "refuted",
                        "citation_ids": [citation_id],
                    }
                    for condition in required["proposed_defeater_conditions"]
                ],
                "strongest_competing_reading": {
                    "label": required["strongest_competing_reading_label"],
                    "summary": "The exact Baseline remains the strongest competing reading.",
                    "citation_ids": [citation_id],
                },
                "outcome": "revision_survives",
                "summary": "Every required falsification finding was checked in order.",
            }
        )

    result = StructuredRoleAgents(
        StructuredModelClient(transport, max_schema_retries=0)
    ).baseline_revision_assessment("B", _proposal(), _anchor(), _card(), _view())

    required = captured_context["required_findings"]
    assert required == {
        "baseline_survival_condition": "execution terminates without a result",
        "proposed_defeater_conditions": ["execution terminates without a result"],
        "strongest_competing_reading_label": "Runtime Termination",
        "allowed_citation_ids": ["issue-observation"],
    }
    assert result.verdict is RevisionAssessmentVerdict.REVISE


def test_entailment_prompt_supplies_exact_ordered_required_findings() -> None:
    captured_context: dict[str, object] = {}

    def transport(_system: str, user: str) -> str:
        context = json.loads(user)["context"]
        captured_context.update(context)
        required = context["required_findings"]
        positive_citation = required["allowed_positive_citation_ids"][0]
        exclusion_citation = required["allowed_exclusion_citation_ids"][0]
        return json.dumps(
            {
                "condition_findings": [
                    {
                        "condition": condition,
                        "status": "supported",
                        "citation_ids": [positive_citation],
                    }
                    for condition in required["proposed_positive_conditions"]
                ],
                "baseline_exclusion_finding": {
                    "condition": required["baseline_exclusion_condition"],
                    "status": "supported",
                    "citation_ids": [exclusion_citation],
                },
                "outcome": "entailed",
                "summary": "Every exact entailment finding is supported in order.",
            }
        )

    result = StructuredRoleAgents(
        StructuredModelClient(transport, max_schema_retries=0)
    ).baseline_revision_assessment("A", _proposal(), _anchor(), _card(), _view())

    required = captured_context["required_findings"]
    assert required == {
        "proposed_positive_conditions": [
            "execution completes beyond its latency budget"
        ],
        "baseline_exclusion_condition": "execution completes with a result",
        "allowed_positive_citation_ids": ["issue-observation"],
        "allowed_exclusion_citation_ids": ["issue-observation"],
    }
    assert "baseline_survival_condition" not in required
    assert result.verdict is RevisionAssessmentVerdict.REVISE


def test_revision_assessment_contract_keeps_nonrevision_verdicts_proposal_bound() -> (
    None
):
    proposal = _proposal()
    preserved = BaselineRevisionAssessment(
        assessor_team_id="A",
        proposal_digest=revision_proposal_digest(proposal),
        dimension=proposal.dimension,
        verdict=RevisionAssessmentVerdict.PRESERVE,
        baseline_source_config_hash=proposal.baseline_source_config_hash,
        baseline_source_predictions_sha256=(
            proposal.baseline_source_predictions_sha256
        ),
        taxonomy_structure_hash=proposal.taxonomy_structure_hash,
        evidence_view_hash=proposal.evidence_view_hash,
    )
    assert preserved.proposed_label is None
    assert preserved.boundary_card_id is None

    forged = preserved.model_dump(mode="python")
    forged["proposed_label"] = proposal.proposed_label
    with pytest.raises(ValidationError, match="revision fields"):
        BaselineRevisionAssessment.model_validate(forged)


@pytest.mark.parametrize(
    "mutation",
    [
        "opaque_binding",
        "wrong_condition",
        "missing_condition",
        "unknown_positive_citation",
        "counter_only_positive_citation",
        "unsupported_positive",
        "wrong_baseline_exclusion",
        "outcome_mismatch",
    ],
)
def test_candidate_bound_assessment_repairs_every_forged_binding(
    mutation: str,
) -> None:
    valid_raw = _raw_assessment("A")
    invalid = valid_raw.model_dump(mode="json")
    if mutation == "opaque_binding":
        invalid["proposal_digest"] = "0" * 64
    elif mutation == "wrong_condition":
        invalid["condition_findings"][0]["condition"] = "invented condition"
    elif mutation == "missing_condition":
        invalid["condition_findings"] = []
    elif mutation == "unknown_positive_citation":
        invalid["condition_findings"][0]["citation_ids"] = ["not-in-view"]
    elif mutation == "counter_only_positive_citation":
        view = _view().model_copy(
            update={
                "items": (
                    _view().items[0],
                    _view().items[1].model_copy(update={"source_type": "issue_body"}),
                )
            }
        )
        proposal = _proposal().model_copy(
            update={
                "counter_evidence_ids": ("unrelated-code",),
                "evidence_view_hash": canonical_evidence_view_hash(view),
            }
        )
        invalid["condition_findings"][0]["citation_ids"] = ["unrelated-code"]
    elif mutation == "unsupported_positive":
        invalid["condition_findings"][0]["status"] = "not_supported"
    elif mutation == "wrong_baseline_exclusion":
        invalid["baseline_exclusion_finding"]["condition"] = "invented exclusion"
    elif mutation == "outcome_mismatch":
        invalid["outcome"] = "not_entailed"
    else:  # pragma: no cover - exhaustive parametrization guard
        raise AssertionError(mutation)
    proposal = locals().get("proposal", _proposal())
    if mutation == "counter_only_positive_citation":
        valid_raw = RevisionEntailmentResult.model_validate(
            valid_raw.model_dump(mode="json")
        )
    replies = iter((json.dumps(invalid), valid_raw.model_dump_json()))
    result = StructuredRoleAgents(
        StructuredModelClient(
            lambda _system, _user: next(replies), max_schema_retries=1
        )
    ).baseline_revision_assessment(
        "A", proposal, _anchor(), _card(), locals().get("view", _view())
    )
    expected = normalize_revision_entailment_result(
        result=valid_raw,
        proposal=proposal,
        routed_card=_card(),
    )
    assert result == expected


def test_candidate_bound_revise_can_use_one_candidate_fact_for_both_sides() -> None:
    assessment = _assessment("A")
    assert assessment.supporting_evidence_ids == ("issue-observation",)
    assert assessment.counter_evidence_ids == ("issue-observation",)

    result = StructuredRoleAgents(
        StructuredModelClient(
            lambda _system, _user: _raw_assessment("A").model_dump_json()
        )
    ).baseline_revision_assessment("A", _proposal(), _anchor(), _card(), _view())
    assert result == assessment


def test_cross_consistency_binds_opposite_owner_and_exact_candidate_citations() -> None:
    proposal = _proposal()
    assessment = _assessment("B")
    valid_raw = FalsificationCoverageCrossResult.model_validate(
        {
            "baseline_survival_finding": {
                "condition": "execution terminates without a result",
                "status": "confirmed",
                "citation_ids": ["issue-observation"],
            },
            "proposed_defeater_findings": [
                {
                    "condition": "execution terminates without a result",
                    "status": "confirmed",
                    "citation_ids": ["issue-observation"],
                }
            ],
            "strongest_competing_reading_finding": {
                "label": "Runtime Termination",
                "status": "confirmed",
                "citation_ids": ["issue-observation"],
            },
            "status": "consistent",
            "rationale": "Every falsification finding is confirmed from its owner citations.",
        }
    )
    captured: dict[str, object] = {}
    captured_system = ""

    def transport(system: str, user: str) -> str:
        nonlocal captured_system
        captured_system = system
        captured.update(json.loads(user))
        return valid_raw.model_dump_json()

    result = StructuredRoleAgents(
        StructuredModelClient(transport)
    ).baseline_revision_consistency(
        checker_team_id="A",
        assessment_owner_team_id="B",
        proposal=proposal,
        assessment=assessment,
        routed_card=_card(),
        view=_view(),
    )
    assert result.status is valid_raw.status
    assert result.proposal_digest == revision_proposal_digest(proposal)
    assert result.assessment_digest == baseline_revision_assessment_digest(assessment)
    assert result.assessment_owner_task == assessment.assessor_task
    assert result.assessment_raw_result_digest == assessment.raw_result_digest
    assert "EVIDENCE-FIRST" in captured_system
    assert captured["context"]["cross_check_input"]["assessment_owner_task"] == (
        "revision_falsification"
    )
    assert "proposal_digest" not in json.dumps(captured["context"])

    forged = result.model_dump(mode="python")
    forged["counter_evidence_ids"] = ()
    with pytest.raises(ValidationError, match="contradiction citations"):
        RevisionConsistencyReport.model_validate(forged)


def test_cross_consistency_rejects_self_check_before_transport() -> None:
    called = False

    def transport(_system: str, _user: str) -> str:
        nonlocal called
        called = True
        return "{}"

    with pytest.raises(ValueError, match="cross-check"):
        StructuredRoleAgents(
            StructuredModelClient(transport)
        ).baseline_revision_consistency(
            checker_team_id="B",
            assessment_owner_team_id="B",
            proposal=_proposal(),
            assessment=_assessment("B"),
            routed_card=_card(),
            view=_view(),
        )
    assert called is False


def test_cross_consistency_rejects_assessment_with_unrouted_condition() -> None:
    called = False
    forged = _assessment("B").model_copy(
        update={
            "contradicted_baseline_condition": "a forged condition outside the card"
        }
    )

    def transport(_system: str, _user: str) -> str:
        nonlocal called
        called = True
        return "{}"

    with pytest.raises(ValueError, match="conditions"):
        StructuredRoleAgents(
            StructuredModelClient(transport)
        ).baseline_revision_consistency(
            checker_team_id="A",
            assessment_owner_team_id="B",
            proposal=_proposal(),
            assessment=forged,
            routed_card=_card(),
            view=_view(),
        )
    assert called is False


def test_cross_consistency_rejects_unknown_counter_citation_before_transport() -> None:
    called = False
    proposal = _proposal().model_copy(update={"counter_evidence_ids": ("not-in-view",)})
    assessment = _assessment("B").model_copy(
        update={
            "proposal_digest": revision_proposal_digest(proposal),
            "counter_evidence_ids": ("not-in-view",),
        }
    )

    def transport(_system: str, _user: str) -> str:
        nonlocal called
        called = True
        return "{}"

    with pytest.raises(ValueError, match="valid and owned-capable"):
        StructuredRoleAgents(
            StructuredModelClient(transport)
        ).baseline_revision_consistency(
            checker_team_id="A",
            assessment_owner_team_id="B",
            proposal=proposal,
            assessment=assessment,
            routed_card=_card(),
            view=_view(),
        )
    assert called is False
