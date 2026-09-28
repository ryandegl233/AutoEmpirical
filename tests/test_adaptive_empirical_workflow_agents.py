from __future__ import annotations

import hashlib
import json
import math
import re

import pytest

from Benchmark.src.adaptive_empirical_workflow.agents import (
    ModelCallOptions,
    ModelTransportError,
    ModelTransportResponse,
    StructuredModelClient,
    StructuredOutputError,
    StructuredRoleAgents,
)
from Benchmark.src.adaptive_empirical_workflow.capabilities import (
    AnalystRole,
    DEFAULT_STAGE3_TEAM_PERSPECTIVES,
)
from Benchmark.src.adaptive_empirical_workflow.contracts import (
    AnonymousStage2Report,
    AnonymousStage3TeamReport,
    BaselineAnchor,
    BaselineRevisionAssessment,
    baseline_revision_assessment_digest,
    canonical_evidence_view_hash,
    BoundaryChallenge,
    BoundaryChallengeAction,
    CausalConsistencyReport,
    ConsistencyStatus,
    DimensionVerificationReport,
    DisagreementMap,
    EvidenceExplicitness,
    EvidenceDimension,
    EvidenceItem,
    EvidenceView,
    EvidenceSufficiency,
    FaultEvidenceAssessment,
    JointAnchorReport,
    RepairCausalityAssessment,
    RoleModelPolicy,
    RootCauseReport,
    SlaConditionalArbitration,
    SlaDimensionArbitration,
    SlaJointDiagnosis,
    SlaJointVerification,
    SlaVerifierVerdict,
    ScopeBoundaryAssessment,
    ResolutionStatus,
    Stage2AnalysisReport,
    Stage2ArbitrationPacket,
    Stage2Decision,
    Stage3ArbitrationPacket,
    Stage3AgentPolicy,
    Stage3TeamReport,
    SymptomReport,
    TaxonomyStructure,
    RevisionAssessmentVerdict,
    RevisionEntailmentResult,
    RevisionFalsificationResult,
    EntailmentMappingCrossResult,
    FalsificationCoverageCrossResult,
    normalize_revision_entailment_result,
    normalize_revision_falsification_result,
    RevisionConsistencyReport,
    RevisionCrossCheckResult,
    RevisionProposalEnvelope,
    revision_proposal_digest,
    ThinkingMode,
    VerificationVerdict,
)
from Benchmark.src.adaptive_empirical_workflow.domains import taxonomy_guidance
from Benchmark.src.adaptive_empirical_workflow.taxonomy_structure import (
    taxonomy_structure_hash,
)


def _view() -> EvidenceView:
    content = "A valid request is rejected before the patch."
    return EvidenceView(
        record_id="record-1",
        task="Verify whether this candidate repairs a fault.",
        taxonomy={"decision": ["accepted_fault", "rejected_candidate"]},
        domain_profile="ase2022",
        ledger_version=1,
        items=(
            EvidenceItem(
                evidence_id="issue-body",
                record_id="record-1",
                source_type="issue_body",
                source_uri="https://example.test/issues/1",
                retrieved_at="2026-08-05T12:30:00Z",
                content=content,
                content_sha256=hashlib.sha256(content.encode()).hexdigest(),
                explicitness=EvidenceExplicitness.DIRECT,
            ),
        ),
    )


def _production_taxonomy_view() -> EvidenceView:
    return _view().model_copy(
        update={
            "taxonomy": {
                "decision": ["accepted_fault", "rejected_candidate"],
                "symptom": ["Crash", "Incorrect Functionality"],
                "root_cause": ["Incorrect Code Logic", "API Misuse"],
            }
        }
    )


def _stage2_payload() -> dict[str, object]:
    return {
        "team_id": "A",
        "decision": "accepted_fault",
        "confidence": 0.91,
        "fault_claim": "A valid runtime request is rejected before the patch.",
        "repair_claim": "The patch changes that rejection and restores processing.",
        "supporting_evidence_ids": ["issue-body"],
        "counter_evidence_ids": [],
        "evidence_tests": {
            "fault_existence": "pass",
            "repair_causality": "pass",
            "scope_exclusion": "pass",
        },
        "alternative_hypothesis": "The change could instead add unsupported behavior.",
        "decision_boundary": "Prior incorrect behavior separates repair from feature work.",
        "evidence_sufficiency": "sufficient",
        "unresolved_evidence_gaps": [],
        "evidence_requests": [],
    }


def _readiness_payload(
    dimensions: list[str] | None = None,
) -> dict[str, object]:
    return {
        "task": "stage2",
        "dimensions": [
            {
                "dimension": dimension,
                "sufficient": True,
                "confirmed_evidence_ids": ["issue-body"],
                "missing_facts": [],
                "evidence_requests": [],
            }
            for dimension in (dimensions or ["fault_existence", "study_scope"])
        ],
    }


def _assessment_payload() -> dict[str, object]:
    return {
        "outcome": "pass",
        "claim": "The frozen evidence establishes the role-owned named test.",
        "supporting_evidence_ids": ["issue-body"],
        "counter_evidence_ids": [],
    }


@pytest.mark.parametrize(
    ("method_name", "role_name", "owned_test", "unowned_tests", "schema"),
    [
        (
            "fault_evidence_analyst",
            "fault_evidence_analyst",
            "FAULT-EXISTENCE TEST",
            ("study_scope", "repair_causality"),
            FaultEvidenceAssessment,
        ),
        (
            "scope_boundary_analyst",
            "scope_boundary_analyst",
            "STUDY-SCOPE TEST",
            ("fault_existence", "repair_causality"),
            ScopeBoundaryAssessment,
        ),
        (
            "repair_causality_analyst",
            "repair_causality_analyst",
            "REPAIR-CAUSALITY TEST",
            ("fault_existence", "study_scope"),
            RepairCausalityAssessment,
        ),
    ],
)
def test_stage2_role_prompts_own_one_test_and_expose_no_decision_schema(
    method_name: str,
    role_name: str,
    owned_test: str,
    unowned_tests: tuple[str, str],
    schema: type[object],
) -> None:
    captured: dict[str, str] = {}
    view = _production_taxonomy_view()
    assessment = _assessment_payload()
    if method_name == "repair_causality_analyst":
        patch_content = "- reject(request)\n+ process(request)"
        patch = view.items[0].model_copy(update={
            "evidence_id": "code-diff", "source_type": "code_diff", "content": patch_content,
            "content_sha256": hashlib.sha256(patch_content.encode()).hexdigest(),
        })
        view = view.model_copy(update={"items": (*view.items, patch)})
        assessment["supporting_evidence_ids"] = ["code-diff"]

    def transport(system: str, user: str) -> str:
        captured["system"] = system
        captured["user"] = user
        return json.dumps(assessment)

    method = getattr(
        StructuredRoleAgents(StructuredModelClient(transport)),
        method_name,
    )
    result = method(view)

    assert isinstance(result, schema)
    assert role_name in captured["system"]
    assert owned_test in captured["system"]
    assert "cannot assign accepted/rejected" in captured["system"]
    output_contract = captured["system"].split("OUTPUT CONTRACT:\n", 1)[1]
    assert '"decision"' not in output_contract
    assert '"symptom_label"' not in output_contract
    assert '"root_cause_label"' not in output_contract
    assert "A valid request is rejected before the patch." in captured["user"]
    assert "SYMPTOM:" not in captured["system"]
    assert "ROOT_CAUSE:" not in captured["system"]
    assert "Crash" not in captured["system"]
    assert "Incorrect Functionality" not in captured["system"]
    assert "Incorrect Code Logic" not in captured["system"]
    assert "API Misuse" not in captured["system"]
    normalized_prompt = re.sub(r"[\s-]+", "_", captured["system"].lower())
    assert all(test not in normalized_prompt for test in unowned_tests)


def test_boundary_challenger_prompt_is_non_classifying_and_anonymized() -> None:
    captured: dict[str, str] = {}

    def transport(system: str, user: str) -> str:
        captured["system"] = system
        captured["user"] = user
        return json.dumps(
            {
                "action": "pass",
                "rationale": "The proposed labels respect the nearest boundaries.",
                "cited_evidence_ids": ["issue-body"],
            }
        )

    symptom = SymptomReport(
        label="observed_failure",
        behavior_claim="A valid request is rejected before the patch is applied.",
        supporting_evidence_ids=["issue-body"],
        alternative_label="wrong_output",
        boundary_reason="The request is blocked rather than completed incorrectly.",
        boundary_evidence_ids=["issue-body"],
        confidence=0.9,
        evidence_sufficiency=EvidenceSufficiency.SUFFICIENT,
    )
    root = RootCauseReport(
        label="incorrect_condition",
        defect_mechanism="A valid request is routed through the rejection condition.",
        causal_chain=[
            "Request arrives.",
            "Condition misfires.",
            "Request is rejected.",
        ],
        supporting_evidence_ids=["issue-body"],
        alternative_label="missing_guard",
        boundary_reason="The evidence identifies a wrong branch condition.",
        boundary_evidence_ids=["issue-body"],
        confidence=0.9,
        evidence_sufficiency=EvidenceSufficiency.SUFFICIENT,
    )
    consistency = CausalConsistencyReport(
        status=ConsistencyStatus.CONSISTENT,
        rationale="The proposed local mechanism explains the observed rejection.",
        supporting_evidence_ids=["issue-body"],
    )
    reports = (
        Stage3TeamReport(
            team_id="A", symptom=symptom, root_cause=root, consistency=consistency
        ),
        Stage3TeamReport(
            team_id="B", symptom=symptom, root_cause=root, consistency=consistency
        ),
    )

    challenge = StructuredRoleAgents(
        StructuredModelClient(transport)
    ).boundary_challenger(reports, _view())

    assert challenge.action is BoundaryChallengeAction.PASS
    assert "boundary_challenger" in captured["system"]
    output_contract = captured["system"].split("OUTPUT CONTRACT:\n", 1)[1]
    assert '"label"' not in output_contract
    assert '"symptom_label"' not in output_contract
    assert '"root_cause_label"' not in output_contract
    payload = json.loads(captured["user"])
    proposed = payload["context"]["proposed_anonymized_reports"]
    assert len(proposed) == 2
    assert all("team_id" not in report for report in proposed)


def test_structured_client_accepts_one_json_markdown_fence() -> None:
    raw = "```json\n" + json.dumps(_stage2_payload()) + "\n```"
    client = StructuredModelClient(lambda system, user: raw)

    report = StructuredRoleAgents(client).stage2_analyst("A", _view())

    assert report.decision is Stage2Decision.ACCEPTED
    assert report.supporting_evidence_ids == ["issue-body"]


def test_schema_failure_is_retried_with_validation_feedback() -> None:
    prompts: list[str] = []
    responses = iter(
        [
            '{"decision":"accepted_fault"}',
            json.dumps(_stage2_payload()),
        ]
    )

    def transport(system: str, user: str) -> str:
        prompts.append(user)
        return next(responses)

    client = StructuredModelClient(transport, max_schema_retries=1)
    report = StructuredRoleAgents(client).stage2_analyst("A", _view())

    assert report.team_id == "A"
    assert len(prompts) == 2
    assert "Validation failed" in prompts[1]
    assert '{"decision":"accepted_fault"}' in prompts[1]
    telemetry = client.telemetry()
    assert telemetry["schema_retry_calls"] == 1
    assert telemetry["first_pass_success_rate"] == 0.0
    assert telemetry["calls"][0]["validation_errors"][0]["fields"]


def test_unscoped_structured_telemetry_keeps_normal_call_shape() -> None:
    client = StructuredModelClient(lambda _system, _user: json.dumps(_stage2_payload()))

    StructuredRoleAgents(client).stage2_analyst("A", _view())

    assert "record_id" not in client.telemetry()["calls"][0]


def test_ase_accepts_real_fault_without_repair_evidence_on_first_pass() -> None:
    payload = _stage2_payload()
    payload["repair_claim"] = (
        "No repair artifact is available; this does not negate the issue fault."
    )
    payload["evidence_tests"]["repair_causality"] = "unknown"
    responses = 0

    def transport(system: str, user: str) -> str:
        nonlocal responses
        responses += 1
        return json.dumps(payload)

    report = StructuredRoleAgents(StructuredModelClient(transport)).stage2_analyst(
        "A", _view()
    )

    assert report.decision is Stage2Decision.ACCEPTED
    assert responses == 1


def test_issta_acceptance_still_requires_causal_repair() -> None:
    payload = _stage2_payload()
    payload["repair_claim"] = "The supplied commit has no demonstrated repair."
    payload["evidence_tests"]["repair_causality"] = "unknown"
    view = _view().model_copy(update={"domain_profile": "issta2024"})

    try:
        StructuredRoleAgents(
            StructuredModelClient(
                lambda system, user: json.dumps(payload),
                max_schema_retries=0,
            )
        ).stage2_analyst("A", view)
    except StructuredOutputError:
        pass
    else:
        raise AssertionError("ISSTA acceptance must require causal repair")


def test_role_prompt_contains_exact_evidence_but_no_gold_annotation() -> None:
    captured: dict[str, str] = {}

    def transport(system: str, user: str) -> str:
        captured["system"] = system
        captured["user"] = user
        return json.dumps(_stage2_payload())

    StructuredRoleAgents(StructuredModelClient(transport)).stage2_analyst("A", _view())

    assert "stage2_fault_verifier" in captured["system"]
    assert "gold_annotation" in captured["system"]
    assert "under 80 words" in captured["system"]
    assert "Do not restate evidence verbatim" in captured["system"]
    assert "A valid request is rejected before the patch." in captured["user"]
    prompt_payload = json.loads(captured["user"])
    assert "taxonomy" not in prompt_payload["evidence_view"]
    assert "domain_profile" not in prompt_payload["evidence_view"]
    assert "role_task" not in prompt_payload
    assert "output_schema" not in prompt_payload
    assert '"gold"' not in captured["user"].lower()


def test_readiness_agent_uses_required_dimensions_and_non_classifying_prompt() -> None:
    captured: dict[str, str] = {}

    def transport(system: str, user: str) -> str:
        captured["system"] = system
        captured["user"] = user
        return json.dumps(_readiness_payload())

    report = StructuredRoleAgents(StructuredModelClient(transport)).evidence_readiness(
        "stage2", _view()
    )

    assert [item.dimension for item in report.dimensions] == [
        EvidenceDimension.FAULT_EXISTENCE,
        EvidenceDimension.STUDY_SCOPE,
    ]
    assert (
        "cannot assign accepted/rejected, symptom, or root-cause labels"
        in captured["system"]
    )
    payload = json.loads(captured["user"])
    assert payload["evidence_view"]["task"] == "stage2"
    assert payload["context"]["task"] == "stage2"
    assert payload["context"]["required_dimensions"] == [
        "fault_existence",
        "study_scope",
    ]
    assert '"gold"' not in captured["user"].lower()


def test_readiness_agent_receives_record_local_source_availability() -> None:
    captured: dict[str, str] = {}
    item_payload = _view().items[0].model_dump(mode="python")
    item_payload["metadata"] = {
        "captured_frozen_source_types": ["issue_body", "issue_comments"],
        "unavailable_frozen_source_types": [
            "changed_files",
            "code_diff",
            "commit_history",
        ],
    }
    view = _view().model_copy(
        update={"items": (EvidenceItem.model_validate(item_payload),)}
    )

    def transport(system: str, user: str) -> str:
        captured["system"] = system
        captured["user"] = user
        payload = _readiness_payload(["symptom", "root_cause"])
        payload["task"] = "stage3"
        return json.dumps(payload)

    StructuredRoleAgents(StructuredModelClient(transport)).evidence_readiness(
        "stage3", view
    )

    context = json.loads(captured["user"])["context"]
    assert context["captured_frozen_source_types"] == [
        "issue_body",
        "issue_comments",
    ]
    assert context["unavailable_frozen_source_types"] == [
        "changed_files",
        "code_diff",
        "commit_history",
    ]
    assert "Do not request unavailable frozen source types" in captured["system"]
    assert "ASE2022 ROOT-CAUSE READINESS" in captured["system"]
    assert "maintainer explanations" in captured["system"]


def test_readiness_retries_requests_for_uncaptured_specialist_sources() -> None:
    prompts: list[str] = []
    item_payload = _view().items[0].model_dump(mode="python")
    item_payload["metadata"] = {
        "captured_frozen_source_types": ["issue_body", "issue_comments"],
        "unavailable_frozen_source_types": [
            "changed_files",
            "code_diff",
            "commit_history",
        ],
    }
    view = _view().model_copy(
        update={"items": (EvidenceItem.model_validate(item_payload),)}
    )
    invalid = _readiness_payload()
    invalid["dimensions"][0]["sufficient"] = False
    invalid["dimensions"][0]["confirmed_evidence_ids"] = []
    invalid["dimensions"][0]["missing_facts"] = ["Need a patch."]
    invalid["dimensions"][0]["evidence_requests"] = [
        {
            "request_id": "req-code",
            "missing_fact": "The patch mechanism is unknown.",
            "why_needed": "It may distinguish a code defect from user error.",
            "target_specialist": "code_context",
            "target_source": "code diff",
            "query": "patch mechanism",
            "expected_decision_impact": "A patch supports code defect; no patch supports user error.",
            "max_items": 3,
        }
    ]
    responses = iter((json.dumps(invalid), json.dumps(_readiness_payload())))

    def transport(system: str, user: str) -> str:
        prompts.append(user)
        return next(responses)

    report = StructuredRoleAgents(
        StructuredModelClient(transport, max_schema_retries=1)
    ).evidence_readiness("stage2", view)

    assert report.dimensions[0].sufficient is True
    assert len(prompts) == 2
    assert "uncaptured frozen sources" in prompts[1]


def test_readiness_prompt_excludes_stage3_taxonomy_boundaries() -> None:
    captured: dict[str, str] = {}

    def transport(system: str, user: str) -> str:
        captured["system"] = system
        return json.dumps(_readiness_payload())

    StructuredRoleAgents(StructuredModelClient(transport)).evidence_readiness(
        "stage2", _view()
    )

    assert "tutorial or example itself" not in captured["system"]


def test_stage2_readiness_receives_scope_policy_without_stage3_boundaries() -> None:
    captured: dict[str, str] = {}

    def transport(system: str, user: str) -> str:
        captured["system"] = system
        return json.dumps(_readiness_payload())

    StructuredRoleAgents(StructuredModelClient(transport)).evidence_readiness(
        "stage2", _view()
    )

    assert "STAGE 2 DOMAIN POLICY (ASE2022)" in captured["system"]
    assert "study-eligible fault case" in captured["system"]
    assert "cannot assign accepted/rejected, symptom, or root-cause labels" in (
        captured["system"]
    )
    assert "motivating user-observed problem" not in captured["system"]
    assert "uncaught runtime exception" not in captured["system"]


def test_readiness_agent_rejects_semantic_dimension_mismatches() -> None:
    client = StructuredModelClient(
        lambda system, user: json.dumps(
            _readiness_payload(["fault_existence", "repair_causality"])
        ),
        max_schema_retries=0,
    )

    with pytest.raises(
        StructuredOutputError,
        match="model output failed EvidenceReadinessReport validation after 1 attempt",
    ) as caught:
        StructuredRoleAgents(client).evidence_readiness("stage2", _view())

    assert isinstance(caught.value.__cause__, ValueError)
    assert "readiness dimensions must exactly match required dimensions" in str(
        caught.value.__cause__
    )


def test_readiness_agent_retries_semantic_task_mismatch_with_feedback() -> None:
    prompts: list[str] = []
    wrong_task = _readiness_payload(["symptom", "root_cause"])
    wrong_task["task"] = "stage2"
    corrected = _readiness_payload(["symptom", "root_cause"])
    corrected["task"] = "stage3"
    responses = iter((json.dumps(wrong_task), json.dumps(corrected)))

    def transport(system: str, user: str) -> str:
        prompts.append(user)
        return next(responses)

    client = StructuredModelClient(transport, max_schema_retries=1)
    report = StructuredRoleAgents(client).evidence_readiness("stage3", _view())

    assert report.task == "stage3"
    assert len(prompts) == 2
    assert "readiness report task must match requested task" in prompts[1]
    assert client.telemetry()["schema_retry_calls"] == 1


def test_readiness_repairs_capability_incompatible_confirmed_evidence() -> None:
    prompts: list[str] = []
    invalid = _readiness_payload(["symptom", "root_cause"])
    invalid["task"] = "stage3"
    invalid["dimensions"][0]["confirmed_evidence_ids"] = ["feg-node-test"]
    invalid["dimensions"][1]["confirmed_evidence_ids"] = ["comment-1"]
    corrected = json.loads(json.dumps(invalid))
    corrected["dimensions"][0]["confirmed_evidence_ids"] = ["symptom-node-test"]
    responses = iter((json.dumps(invalid), json.dumps(corrected)))

    def transport(_system: str, user: str) -> str:
        prompts.append(user)
        return next(responses)

    report = StructuredRoleAgents(
        StructuredModelClient(transport, max_schema_retries=1)
    ).evidence_readiness("stage3", _ase_capability_split_view())

    assert report.dimensions[0].confirmed_evidence_ids == ("symptom-node-test",)
    assert len(prompts) == 2
    initial_context = json.loads(prompts[0])["context"]
    assert initial_context["confirmed_evidence_eligibility"]["feg-node-test"] == [
        "root_cause"
    ]
    assert initial_context["confirmed_evidence_eligibility"]["symptom-node-test"] == [
        "symptom",
        "root_cause",
    ]
    assert "capability-incompatible confirmed evidence_id" in prompts[1]


def test_readiness_agent_retries_invalid_frozen_source_target() -> None:
    prompts: list[str] = []
    request = {
        "request_id": "req-code-context",
        "missing_fact": "The local code mechanism behind the reported failure is unknown.",
        "why_needed": "The root-cause readiness decision requires direct local mechanism evidence.",
        "target_specialist": "code_context",
        "target_source": (
            "https://github.com/tensorflow/tfjs/blob/master/"
            "tfjs-layers/src/engine/training.ts"
        ),
        "query": "training failure mechanism",
        "expected_decision_impact": (
            "A matching frozen code passage would establish whether a specific "
            "mechanism can be classified."
        ),
        "max_items": 2,
    }
    wrong_target = {
        "task": "stage3",
        "dimensions": [
            {
                "dimension": "symptom",
                "sufficient": True,
                "confirmed_evidence_ids": ["issue-body"],
                "missing_facts": [],
                "evidence_requests": [],
            },
            {
                "dimension": "root_cause",
                "sufficient": False,
                "confirmed_evidence_ids": [],
                "missing_facts": [request["missing_fact"]],
                "evidence_requests": [request],
            },
        ],
    }
    corrected = json.loads(json.dumps(wrong_target))
    corrected["dimensions"][1]["evidence_requests"][0]["target_source"] = "code context"
    responses = iter((json.dumps(wrong_target), json.dumps(corrected)))

    def transport(system: str, user: str) -> str:
        prompts.append(user)
        return next(responses)

    report = StructuredRoleAgents(
        StructuredModelClient(transport, max_schema_retries=1)
    ).evidence_readiness("stage3", _view())

    assert len(prompts) == 2
    initial_payload = json.loads(prompts[0])
    assert (
        "code context"
        in initial_payload["context"]["allowed_evidence_targets"]["code_context"]
    )
    assert "does not match code_context frozen sources" in prompts[1]
    assert report.dimensions[1].evidence_requests[0].target_source == "code context"


def test_readiness_agent_rejects_duplicate_dimensions() -> None:
    client = StructuredModelClient(
        lambda system, user: json.dumps(
            _readiness_payload(["fault_existence", "fault_existence"])
        ),
        max_schema_retries=0,
    )

    with pytest.raises(
        StructuredOutputError,
        match="model output failed EvidenceReadinessReport validation after 1 attempt",
    ):
        StructuredRoleAgents(client).evidence_readiness("stage2", _view())


def test_ase_stage2_prompt_has_definitions_without_stage3_boundaries() -> None:
    captured: dict[str, str] = {}
    view = _view().model_copy(
        update={
            "taxonomy": {
                "decision": ["accepted_fault", "rejected_candidate"],
                "symptom": ["Document Error", "Incorrect Functionality"],
                "root_cause": ["Confused Document", "Incorrect Code Logic"],
            }
        }
    )

    def transport(system: str, user: str) -> str:
        captured["system"] = system
        captured["user"] = user
        return json.dumps(_stage2_payload())

    StructuredRoleAgents(StructuredModelClient(transport)).stage2_analyst("A", view)

    assert "Document Error: Faults in official documents" in captured["system"]
    assert (
        "Incorrect Functionality: The system runs without crashing"
        in captured["system"]
    )
    assert "tutorial or example itself" not in captured["system"]
    assert captured["system"].find("TAXONOMY") < captured["system"].find(
        "OUTPUT CONTRACT"
    )
    assert (
        "missing commit, pull request, patch, or regression test" in captured["system"]
    )
    assert "must not by itself cause rejection" in captured["system"]
    assert "API Misuse" in captured["system"]
    assert "Cross-platform App Framework Incompatibility" in captured["system"]
    assert "study-eligible fault case" in captured["system"]
    assert "maintainer closure" in captured["system"]
    assert "open-ended usage or capability question" in captured["system"]
    assert "no observed failure or concrete limitation" in captured["system"]
    assert "internal CI, test, release, or dependency maintenance" in captured["system"]
    assert "motivating user-observed problem" not in captured["system"]
    assert "uncaught runtime exception" not in captured["system"]
    assert "Untimely Update over Dependency Error" not in captured["system"]


def test_stage2_schema_exposes_all_required_evidence_test_keys() -> None:
    schema = Stage2AnalysisReport.model_json_schema()
    evidence_tests = schema["$defs"]["Stage2EvidenceTests"]

    assert set(evidence_tests["required"]) == {
        "fault_existence",
        "repair_causality",
        "scope_exclusion",
    }
    assert set(evidence_tests["properties"]) == {
        "fault_existence",
        "repair_causality",
        "scope_exclusion",
    }


def test_ase_stage3_guidance_uses_neutral_taxonomy_comparison_without_record_shaped_precedence() -> (
    None
):
    guidance = taxonomy_guidance(
        "ase2022",
        _production_taxonomy_view().taxonomy,
        include_stage3_boundaries=True,
    )

    assert "Crash: Functionality is terminated unexpectedly" in guidance
    assert "Incorrect Code Logic:" in guidance
    assert "Compare the record-local evidence against every supplied label" in guidance
    assert "nearest competing label" in guidance
    assert "Do not apply a global label precedence" in guidance
    for record_shaped_phrase in (
        "motivating user-observed problem",
        "uncaught runtime exception",
        "long-running memory exhaustion",
        "Untimely Update over Dependency Error",
        "maintainer's API-semantics explanation",
        "Inconsistent Modules over Cross-platform App Framework",
        "undefined or empty imported module",
    ):
        assert record_shaped_phrase not in guidance


def test_role_call_uses_configured_output_limit_and_telemetry_records_it() -> None:
    received: list[int | None] = []

    def transport(
        system: str,
        user: str,
        *,
        max_tokens: int | None = None,
        thinking_enabled: bool | None = None,
    ) -> str:
        received.append(max_tokens)
        return json.dumps(_stage2_payload())

    client = StructuredModelClient(transport)
    StructuredRoleAgents(client, role_max_tokens=1400).stage2_analyst("A", _view())

    assert received == [1400]
    telemetry = client.telemetry()
    assert telemetry["total_calls"] == 1
    assert telemetry["total_attempts"] == 1
    assert telemetry["calls"][0]["role"] == "stage2_fault_verifier"
    assert telemetry["calls"][0]["latency_seconds"] >= 0
    assert telemetry["calls"][0]["max_tokens"] == 1400
    assert telemetry["calls"][0]["prompt_characters"] > 0
    assert telemetry["calls"][0]["completion_characters"] > 0
    assert telemetry["estimated_prompt_tokens"] > 0
    assert telemetry["estimated_completion_tokens"] > 0


def test_all_stage3_profile_enables_thinking_with_high_accuracy_ceilings() -> None:
    policy = Stage3AgentPolicy.from_profile("all-stage3")

    assert policy.resolve("stage2_fault_verifier", "A").thinking == (
        ThinkingMode.DISABLED
    )
    assert policy.resolve("symptom_analyst", "A") == RoleModelPolicy(
        thinking=ThinkingMode.ENABLED,
        max_tokens=2200,
    )
    assert policy.resolve("root_cause_analyst", "B") == RoleModelPolicy(
        thinking=ThinkingMode.ENABLED,
        max_tokens=2800,
    )
    assert policy.resolve("causal_consistency_checker", "A").max_tokens == 1800
    assert policy.resolve("boundary_challenger", None).max_tokens == 1800
    assert policy.resolve("stage3_arbitrator", None) == RoleModelPolicy(
        thinking=ThinkingMode.ENABLED,
        max_tokens=2600,
    )


def test_named_thinking_profiles_and_override_precedence_are_resolved() -> None:
    causal = Stage3AgentPolicy.from_profile("causal")
    assert causal.resolve("symptom_analyst", "A").thinking == ThinkingMode.DISABLED
    assert causal.resolve("root_cause_analyst", "A").thinking == ThinkingMode.ENABLED
    assert causal.resolve("causal_consistency_checker", "B").thinking == (
        ThinkingMode.ENABLED
    )
    assert causal.resolve("boundary_challenger", None).thinking == ThinkingMode.ENABLED
    assert causal.resolve("stage3_arbitrator", None).thinking == ThinkingMode.ENABLED

    off = Stage3AgentPolicy.from_profile("off")
    assert off.resolve("root_cause_analyst", "A").thinking == ThinkingMode.DISABLED
    assert off.resolve("stage2_fault_verifier", "A").thinking == (ThinkingMode.DISABLED)

    overridden = Stage3AgentPolicy.from_profile(
        "all-stage3",
        role_overrides={
            "root_cause_analyst": RoleModelPolicy(
                thinking=ThinkingMode.ENABLED,
                max_tokens=2400,
            )
        },
        role_team_overrides={
            "root_cause_analyst@B": RoleModelPolicy(
                thinking=ThinkingMode.DISABLED,
                max_tokens=2500,
            )
        },
    )
    assert overridden.resolve("root_cause_analyst", "B") == RoleModelPolicy(
        thinking=ThinkingMode.DISABLED,
        max_tokens=2500,
    )
    assert overridden.resolve(
        "root_cause_analyst",
        "B",
        invocation_override=RoleModelPolicy(
            thinking=ThinkingMode.ENABLED,
            max_tokens=2700,
        ),
    ) == RoleModelPolicy(thinking=ThinkingMode.ENABLED, max_tokens=2700)


def test_policy_rejects_dead_non_stage3_team_override() -> None:
    with pytest.raises(ValueError, match="does not accept a team override"):
        Stage3AgentPolicy.from_profile(
            "all-stage3",
            role_team_overrides={
                "evidence_readiness@A": RoleModelPolicy(
                    thinking=ThinkingMode.ENABLED,
                    max_tokens=1900,
                )
            },
        )


def test_option_transport_receives_resolved_stage3_policy_and_team_telemetry() -> None:
    received: list[ModelCallOptions] = []

    def transport(
        system: str,
        user: str,
        *,
        options: ModelCallOptions,
    ) -> str:
        received.append(options)
        return json.dumps(_stage3_team().root_cause.model_dump(mode="json"))

    agents = StructuredRoleAgents(StructuredModelClient(transport))
    agents.root_cause_analyst("B", _stage3_view())

    assert received == [
        ModelCallOptions(
            role="root_cause_analyst",
            team_id="B",
            perspective="falsification_first",
            max_tokens=2800,
            thinking_enabled=True,
        )
    ]
    call = agents.telemetry()["calls"][0]
    assert call["team_id"] == "B"
    assert call["perspective"] == "falsification_first"
    assert call["thinking_enabled"] is True
    assert call["max_tokens"] == 2800


def test_schema_retry_switches_thinking_off_and_records_actual_provider_usage() -> None:
    received: list[ModelCallOptions] = []
    responses = iter(
        (
            '{"invalid":true}',
            ModelTransportResponse(
                json.dumps(_stage3_team().root_cause.model_dump(mode="json")),
                prompt_tokens=321,
                completion_tokens=123,
                reasoning_characters=456,
            ),
        )
    )

    def transport(
        system: str,
        user: str,
        *,
        options: ModelCallOptions,
    ) -> str:
        received.append(options)
        return next(responses)

    agents = StructuredRoleAgents(
        StructuredModelClient(transport, max_schema_retries=1)
    )
    agents.root_cause_analyst("A", _stage3_view())

    assert len(received) == 2
    assert [options.thinking_enabled for options in received] == [True, False]
    assert [options.max_tokens for options in received] == [2800, 2800]
    call = agents.telemetry()["calls"][0]
    assert call["thinking_enabled"] is True
    assert call["attempt_policies"] == [
        {"thinking_enabled": True, "max_tokens": 2800},
        {"thinking_enabled": False, "max_tokens": 2800},
    ]
    assert call["actual_prompt_tokens"] == 321
    assert call["actual_completion_tokens"] == 123
    assert call["reasoning_characters"] == 456
    assert call["estimated_prompt_tokens"] > 0
    assert call["estimated_completion_tokens"] > 0


def test_empty_thinking_response_disables_thinking_for_structured_retry() -> None:
    received: list[ModelCallOptions] = []

    def transport(
        system: str,
        user: str,
        *,
        options: ModelCallOptions,
    ) -> str:
        received.append(options)
        if len(received) == 1:
            return ""
        return json.dumps(_stage3_team().root_cause.model_dump(mode="json"))

    agents = StructuredRoleAgents(
        StructuredModelClient(transport, max_schema_retries=1)
    )
    agents.root_cause_analyst("B", _stage3_view())

    assert [options.max_tokens for options in received] == [2800, 2800]
    assert [options.thinking_enabled for options in received] == [True, False]
    call = agents.telemetry()["calls"][0]
    assert call["thinking_enabled"] is True
    assert call["attempt_policies"] == [
        {"thinking_enabled": True, "max_tokens": 2800},
        {"thinking_enabled": False, "max_tokens": 2800},
    ]


def test_empty_nonthinking_response_preserves_output_budget_on_retry() -> None:
    received: list[ModelCallOptions] = []

    def transport(
        system: str,
        user: str,
        *,
        options: ModelCallOptions,
    ) -> str:
        received.append(options)
        if len(received) == 1:
            return ""
        return json.dumps(_stage2_payload())

    agents = StructuredRoleAgents(
        StructuredModelClient(transport, max_schema_retries=1)
    )
    agents.stage2_analyst("A", _view())

    assert [options.max_tokens for options in received] == [1600, 1600]


def test_label_thinking_profile_enables_label_owners_and_stage3_arbitrator() -> None:
    policy = Stage3AgentPolicy.from_profile("label-thinking")

    assert policy.resolve("symptom_analyst", "A") == RoleModelPolicy(
        thinking=ThinkingMode.ENABLED,
        max_tokens=32768,
    )
    assert policy.resolve("root_cause_analyst", "B") == RoleModelPolicy(
        thinking=ThinkingMode.ENABLED,
        max_tokens=32768,
    )
    assert (
        policy.resolve("causal_consistency_checker", "A").thinking
        == ThinkingMode.DISABLED
    )
    assert policy.resolve("boundary_challenger", None).thinking == ThinkingMode.DISABLED
    assert policy.resolve("stage3_arbitrator", None) == RoleModelPolicy(
        thinking=ThinkingMode.ENABLED,
        max_tokens=32768,
    )


def test_unsupported_provider_omits_thinking_from_explicit_options() -> None:
    received: list[ModelCallOptions] = []

    def transport(
        system: str,
        user: str,
        *,
        options: ModelCallOptions,
    ) -> str:
        received.append(options)
        return json.dumps(_stage3_team().symptom.model_dump(mode="json"))

    StructuredRoleAgents(
        StructuredModelClient(transport),
        provider_supports_thinking=False,
    ).symptom_analyst("A", _stage3_view())

    assert received[0].thinking_enabled is None


def test_partial_option_aware_transport_fails_closed_for_thinking_call() -> None:
    def transport(system: str, user: str, *, max_tokens: int) -> str:
        return json.dumps(_stage3_team().symptom.model_dump(mode="json"))

    with pytest.raises(TypeError, match="partial option-aware transport"):
        StructuredRoleAgents(StructuredModelClient(transport)).symptom_analyst(
            "A", _stage3_view()
        )


def test_max_tokens_only_transport_fails_closed_when_thinking_is_unsupported() -> None:
    def transport(system: str, user: str, *, max_tokens: int) -> str:
        return json.dumps(_stage3_team().symptom.model_dump(mode="json"))

    with pytest.raises(TypeError, match="partial option-aware transport"):
        StructuredRoleAgents(
            StructuredModelClient(transport),
            provider_supports_thinking=False,
        ).symptom_analyst("A", _stage3_view())


def test_max_tokens_only_transport_fails_closed_for_provider_default_thinking() -> None:
    def transport(system: str, user: str, *, max_tokens: int) -> str:
        return json.dumps(_stage3_team().symptom.model_dump(mode="json"))

    policy = Stage3AgentPolicy.from_profile(
        "all-stage3",
        role_overrides={
            "symptom_analyst": RoleModelPolicy(
                thinking=ThinkingMode.PROVIDER_DEFAULT,
                max_tokens=2200,
            )
        },
    )
    with pytest.raises(TypeError, match="partial option-aware transport"):
        StructuredRoleAgents(
            StructuredModelClient(transport),
            agent_policy=policy,
        ).symptom_analyst("A", _stage3_view())


def test_generic_kwargs_transport_fails_closed_without_explicit_options() -> None:
    called = False

    def transport(system: str, user: str, **kwargs: object) -> str:
        nonlocal called
        called = True
        return json.dumps(_stage3_team().symptom.model_dump(mode="json"))

    with pytest.raises(TypeError, match="generic kwargs transport"):
        StructuredRoleAgents(StructuredModelClient(transport)).symptom_analyst(
            "A", _stage3_view()
        )
    assert called is False


def test_var_keyword_named_options_is_not_an_explicit_options_parameter() -> None:
    called = False

    def transport(system: str, user: str, **options: object) -> str:
        nonlocal called
        called = True
        return json.dumps(_stage3_team().symptom.model_dump(mode="json"))

    with pytest.raises(TypeError, match="generic kwargs transport"):
        StructuredRoleAgents(StructuredModelClient(transport)).symptom_analyst(
            "A", _stage3_view()
        )
    assert called is False


def test_two_positional_legacy_transport_may_name_first_parameter_options() -> None:
    received: dict[str, str] = {}

    def transport(options: str, user: str) -> str:
        received["system"] = options
        received["user"] = user
        return json.dumps(_stage3_team().symptom.model_dump(mode="json"))

    report = StructuredRoleAgents(StructuredModelClient(transport)).symptom_analyst(
        "A", _stage3_view()
    )

    assert report.label == "Crash"
    assert "symptom_analyst" in received["system"]
    assert json.loads(received["user"])["team_id"] == "A"


def test_arbitration_prompt_contains_only_disputed_fields_and_evidence() -> None:
    captured: dict[str, str] = {}

    def transport(
        system: str,
        user: str,
        *,
        max_tokens: int,
        thinking_enabled: bool | None,
    ) -> str:
        captured["system"] = system
        captured["user"] = user
        return json.dumps(
            {
                "resolution_status": "resolved",
                "decision": "accepted_fault",
                "confidence": 0.8,
                "rationale": "The demonstrated prior failure and causal repair satisfy the boundary.",
                "supporting_evidence_ids": ["issue-body"],
                "resolved_dimensions": ["decision"],
            }
        )

    common = {
        key: value
        for key, value in _stage2_payload().items()
        if key not in {"team_id", "evidence_requests"}
    }
    report_a = AnonymousStage2Report.model_validate(common)
    report_b = report_a.model_copy(
        update={
            "decision": Stage2Decision.REJECTED,
            "fault_claim": "This large non-disputed field must not be repeated.",
        }
    )
    disagreement = DisagreementMap(
        dimensions=["decision"],
        details={
            "decision": {
                "team_a": "accepted_fault",
                "team_b": "rejected_candidate",
            }
        },
        requires_arbitration=True,
    )

    StructuredRoleAgents(StructuredModelClient(transport)).stage2_arbitrator(
        Stage2ArbitrationPacket(
            domain_profile="ase2022",
            taxonomy={
                "decision": ["accepted_fault", "rejected_candidate"],
                "symptom": ["Document Error", "Incorrect Functionality"],
                "root_cause": ["Confused Document", "Incorrect Code Logic"],
            },
            disagreement=disagreement,
            report_a=report_a,
            report_b=report_b,
            classification_ledger_version=1,
            relevant_evidence=_view().items,
        )
    )

    payload = json.loads(captured["user"])
    assert payload["disagreement"] == disagreement.model_dump(mode="json")
    assert payload["candidates"]["team_a"]["decision"] == "accepted_fault"
    assert payload["candidates"]["team_b"]["decision"] == "rejected_candidate"
    assert "packet" not in payload
    assert (
        payload["candidates"]["team_b"]["fault_claim"]
        == "This large non-disputed field must not be repeated."
    )
    assert "repair_claim" in payload["candidates"]["team_a"]
    assert payload["evidence_snapshot"] == {
        "ledger_version": 1,
        "items": [item.model_dump(mode="json") for item in _view().items],
    }
    assert "Resolve only the listed disagreement dimensions" in captured["system"]
    assert "exact cited evidence" in captured["system"]
    assert "UNRESOLVED" in captured["system"]


def test_transport_failure_records_network_attempts_before_propagating() -> None:
    def transport(
        system: str,
        user: str,
        *,
        max_tokens: int,
        thinking_enabled: bool | None,
    ) -> str:
        raise ModelTransportError("rate limit exhausted", attempts=3)

    client = StructuredModelClient(transport)

    try:
        StructuredRoleAgents(client).stage2_analyst("A", _view())
    except ModelTransportError:
        pass
    else:
        raise AssertionError("transport error should propagate")

    telemetry = client.telemetry()
    assert telemetry["total_calls"] == 1
    assert telemetry["total_attempts"] == 3
    assert telemetry["calls"][0]["network_attempts"] == 3
    assert telemetry["calls"][0]["status"] == "transport_error"

    success_client = StructuredModelClient(
        lambda system, user, max_tokens, thinking_enabled: json.dumps(_stage2_payload())
    )
    StructuredRoleAgents(success_client).stage2_analyst("A", _view())
    success_characters = success_client.telemetry()["calls"][0]["prompt_characters"]
    assert telemetry["calls"][0]["prompt_characters"] == 3 * success_characters


def test_successful_provider_retries_are_visible_in_structured_client_telemetry() -> (
    None
):
    retry_events = (
        {
            "attempt": 1,
            "status": "retryable_error",
            "latency_seconds": 0.25,
            "retryable": True,
            "cause_type": "remote_disconnected",
            "http_status": None,
            "retry_after_seconds": None,
        },
        {
            "attempt": 2,
            "status": "success",
            "latency_seconds": 0.5,
            "retryable": False,
            "cause_type": None,
            "http_status": None,
            "retry_after_seconds": None,
        },
    )

    def transport(
        system: str,
        user: str,
        *,
        max_tokens: int,
        thinking_enabled: bool | None,
    ) -> str:
        return ModelTransportResponse(
            json.dumps(_stage2_payload()),
            network_attempts=2,
            network_attempt_details=retry_events,
        )

    client = StructuredModelClient(transport)
    StructuredRoleAgents(client).stage2_analyst("A", _view())

    telemetry = client.telemetry()
    assert telemetry["provider_retry_calls"] == 1
    assert telemetry["provider_retry_count"] == 1
    call = telemetry["calls"][0]
    assert call["network_attempt_details"] == list(retry_events)
    assert call["network_attempts"] == 2


def test_optional_revision_budget_recovers_after_schema_and_transport_failures() -> (
    None
):
    replies = iter(
        (
            "{}",
            ModelTransportError(
                "SECRET_TRANSIENT_PROVIDER_DETAIL",
                attempts=1,
                retryable=True,
                cause_type="http_error",
                http_status=429,
            ),
            json.dumps(_stage2_payload()),
        )
    )
    prompts: list[str] = []

    def transport(
        system: str,
        user: str,
        *,
        options: ModelCallOptions,
    ) -> str:
        prompts.append(user)
        reply = next(replies)
        if isinstance(reply, Exception):
            raise reply
        return reply

    client = StructuredModelClient(transport, max_schema_retries=2)
    result = client.complete(
        system_prompt="Return JSON.",
        user_prompt="Assess the optional revision.",
        schema=Stage2AnalysisReport,
        options=ModelCallOptions(
            role="baseline_revision_assessment",
            team_id="B",
            perspective="falsification_first",
            max_tokens=32768,
            thinking_enabled=None,
            share_transport_schema_budget=True,
        ),
    )

    assert result.decision is Stage2Decision.ACCEPTED
    telemetry = client.telemetry()
    assert telemetry["total_attempts"] == 3
    assert telemetry["schema_validation_failure_count"] == 1
    assert telemetry["transport_retry_count"] == 1
    call = telemetry["calls"][0]
    assert call["status"] == "valid"
    assert call["attempts"] == 3
    assert call["network_attempts"] == 3
    assert call["schema_validation_failure_count"] == 1
    assert call["transport_retry_count"] == 1
    assert call["transport_error_count"] == 1
    assert "SECRET_TRANSIENT_PROVIDER_DETAIL" not in json.dumps(call)
    assert prompts[0] != prompts[1]
    assert prompts[1] == prompts[2]


def test_optional_revision_budget_exhaustion_propagates_typed_transport_error() -> None:
    calls = 0

    def transport(
        system: str,
        user: str,
        *,
        options: ModelCallOptions,
    ) -> str:
        nonlocal calls
        calls += 1
        raise ModelTransportError(
            "SECRET_PROVIDER_DETAIL",
            attempts=1,
            retryable=True,
            cause_type="http_error",
            http_status=429,
        )

    client = StructuredModelClient(transport, max_schema_retries=2)
    with pytest.raises(ModelTransportError):
        client.complete(
            system_prompt="Return JSON.",
            user_prompt="Assess the optional revision.",
            schema=Stage2AnalysisReport,
            options=ModelCallOptions(
                role="baseline_revision_consistency",
                team_id="A",
                perspective="evidence_first",
                max_tokens=32768,
                thinking_enabled=None,
                share_transport_schema_budget=True,
            ),
        )

    assert calls == 3
    telemetry = client.telemetry()
    assert telemetry["total_attempts"] == 3
    assert telemetry["transport_retry_count"] == 2
    call = telemetry["calls"][0]
    assert call["status"] == "transport_error"
    assert call["transport_error_count"] == 3
    assert call["transport_retry_count"] == 2
    assert "SECRET_PROVIDER_DETAIL" not in json.dumps(call)


def test_optional_revision_retries_typed_retryable_transport_with_backoff() -> None:
    replies = iter(
        (
            ModelTransportError(
                "rate limited",
                attempts=1,
                retryable=True,
                cause_type="http_error",
                http_status=429,
            ),
            json.dumps(_stage2_payload()),
        )
    )
    delays: list[float] = []

    def transport(
        system: str,
        user: str,
        *,
        options: ModelCallOptions,
    ) -> str:
        reply = next(replies)
        if isinstance(reply, Exception):
            raise reply
        return reply

    client = StructuredModelClient(
        transport,
        max_schema_retries=2,
        retry_delay_seconds=0.25,
        waiter=delays.append,
    )
    result = client.complete(
        system_prompt="Return JSON.",
        user_prompt="Assess the optional revision.",
        schema=Stage2AnalysisReport,
        options=ModelCallOptions(
            role="baseline_revision_consistency",
            team_id="A",
            perspective="evidence_first",
            max_tokens=32768,
            thinking_enabled=None,
            share_transport_schema_budget=True,
        ),
    )

    assert result.decision is Stage2Decision.ACCEPTED
    assert delays == [0.25]
    call = client.telemetry()["calls"][0]
    assert call["transport_retry_count"] == 1
    assert call["transport_failures"] == [
        {
            "retryable": True,
            "cause_type": "http_error",
            "http_status": 429,
            "retry_after_seconds": None,
        }
    ]


@pytest.mark.parametrize(
    ("retry_after_seconds", "expected"),
    (
        (0.5, 0.5),
        (float("nan"), None),
        (math.inf, None),
        (-1.0, None),
    ),
)
def test_transport_error_normalizes_retry_after_to_finite_nonnegative_seconds(
    retry_after_seconds: float,
    expected: float | None,
) -> None:
    error = ModelTransportError(
        "rate limited",
        retryable=True,
        cause_type="http_error",
        http_status=429,
        retry_after_seconds=retry_after_seconds,
    )

    assert error.retry_after_seconds == expected


def test_shared_transport_budget_records_actual_overspent_attempts_without_retry() -> (
    None
):
    calls = 0

    def transport(
        system: str,
        user: str,
        *,
        options: ModelCallOptions,
    ) -> str:
        nonlocal calls
        calls += 1
        raise ModelTransportError(
            "provider retried before client admission",
            attempts=5,
            retryable=True,
            cause_type="http_error",
            http_status=429,
        )

    client = StructuredModelClient(transport, max_schema_retries=2)
    with pytest.raises(ModelTransportError):
        client.complete(
            system_prompt="Return JSON.",
            user_prompt="Assess the optional revision.",
            schema=Stage2AnalysisReport,
            options=ModelCallOptions(
                role="baseline_revision_consistency",
                team_id="A",
                perspective="evidence_first",
                max_tokens=32768,
                thinking_enabled=None,
                share_transport_schema_budget=True,
            ),
        )

    call = client.telemetry()["calls"][0]
    assert calls == 1
    assert call["network_attempts"] == 5
    assert call["network_budget_limit"] == 3
    assert call["network_budget_overspent"] == 2
    assert call["transport_retry_count"] == 0


def test_optional_revision_does_not_retry_nonretryable_typed_transport() -> None:
    calls = 0

    def transport(
        system: str,
        user: str,
        *,
        options: ModelCallOptions,
    ) -> str:
        nonlocal calls
        calls += 1
        raise ModelTransportError(
            "bad request",
            attempts=1,
            retryable=False,
            cause_type="http_error",
            http_status=400,
        )

    client = StructuredModelClient(transport, max_schema_retries=2)
    with pytest.raises(ModelTransportError) as raised:
        client.complete(
            system_prompt="Return JSON.",
            user_prompt="Assess the optional revision.",
            schema=Stage2AnalysisReport,
            options=ModelCallOptions(
                role="baseline_revision_consistency",
                team_id="A",
                perspective="evidence_first",
                max_tokens=32768,
                thinking_enabled=None,
                share_transport_schema_budget=True,
            ),
        )

    assert calls == 1
    assert raised.value.retryable is False
    assert raised.value.http_status == 400


def test_optional_revision_budget_does_not_swallow_generic_runtime_error() -> None:
    def transport(
        system: str,
        user: str,
        *,
        options: ModelCallOptions,
    ) -> str:
        raise RuntimeError("programming error")

    client = StructuredModelClient(transport, max_schema_retries=2)
    with pytest.raises(RuntimeError, match="programming error"):
        client.complete(
            system_prompt="Return JSON.",
            user_prompt="Assess the optional revision.",
            schema=Stage2AnalysisReport,
            options=ModelCallOptions(
                role="baseline_revision_assessment",
                team_id="A",
                perspective="evidence_first",
                max_tokens=32768,
                thinking_enabled=None,
                share_transport_schema_budget=True,
            ),
        )


@pytest.mark.parametrize(
    ("role", "expected"),
    (
        ("baseline_revision_assessment", True),
        ("baseline_revision_consistency", True),
        ("joint_anchor", False),
        ("stage3_arbitrator", False),
    ),
)
def test_only_optional_revision_roles_share_transport_schema_budget(
    role: str,
    expected: bool,
) -> None:
    agents = StructuredRoleAgents(StructuredModelClient(lambda system, user: "{}"))

    options = agents._model_call_options(
        role,
        team_id="A",
        perspective="evidence_first",
    )

    assert options.share_transport_schema_budget is expected


def _stage3_view() -> EvidenceView:
    return _view().model_copy(
        update={
            "taxonomy": {
                "symptom": ["Crash", "Poor Performance"],
                "root_cause": ["Incorrect Code Logic", "API Misuse"],
            }
        }
    )


def _stage3_team() -> Stage3TeamReport:
    return Stage3TeamReport(
        team_id="A",
        symptom=SymptomReport(
            label="Crash",
            behavior_claim="A valid request terminates processing unexpectedly.",
            supporting_evidence_ids=["issue-body"],
            alternative_label="Poor Performance",
            boundary_reason="Execution terminates rather than merely slowing down.",
            boundary_evidence_ids=["issue-body"],
            confidence=0.9,
            evidence_sufficiency=EvidenceSufficiency.SUFFICIENT,
        ),
        root_cause=RootCauseReport(
            label="Incorrect Code Logic",
            defect_mechanism="An incorrect branch rejects the valid request.",
            causal_chain=[
                "The request enters the handler.",
                "The condition selects rejection.",
                "Processing terminates.",
            ],
            supporting_evidence_ids=["issue-body"],
            alternative_label="API Misuse",
            boundary_reason="The defect is internal logic rather than API usage.",
            boundary_evidence_ids=["issue-body"],
            confidence=0.9,
            evidence_sufficiency=EvidenceSufficiency.SUFFICIENT,
        ),
        consistency=CausalConsistencyReport(
            status=ConsistencyStatus.CONSISTENT,
            rationale="The incorrect branch directly explains request termination.",
            supporting_evidence_ids=["issue-body"],
        ),
    )


def _baseline_revision_structure() -> TaxonomyStructure:
    return TaxonomyStructure.model_validate(
        {
            "schema_version": 1,
            "domain": "ase2022",
            "nodes": [
                {
                    "dimension": "symptom",
                    "label": "Crash",
                    "definition": "Execution terminates before returning a result.",
                    "semantic_origin": "operational_definition",
                    "abstraction_level": "observable outcome",
                    "responsibility_scope": "runtime process",
                    "concept_kind": "outcome",
                },
                {
                    "dimension": "symptom",
                    "label": "Poor Performance",
                    "definition": "Execution completes slower than expected.",
                    "semantic_origin": "operational_definition",
                    "abstraction_level": "observable outcome",
                    "responsibility_scope": "runtime process",
                    "concept_kind": "outcome",
                },
            ],
            "boundary_cards": [
                {
                    "card_id": "crash-performance",
                    "dimension": "symptom",
                    "labels": ["Crash", "Poor Performance"],
                    "semantic_origin": "operational_definition",
                    "decision_question": "Does execution return a result?",
                    "observable_slots": ["execution outcome"],
                    "criteria": [
                        {
                            "label": "Crash",
                            "positive_conditions": [
                                "execution terminates before returning"
                            ],
                            "exclusion_conditions": ["execution completes slowly"],
                        },
                        {
                            "label": "Poor Performance",
                            "positive_conditions": ["execution completes slowly"],
                            "exclusion_conditions": [
                                "execution terminates before returning"
                            ],
                        },
                    ],
                }
            ],
        }
    )


def _baseline_revision_proposal(view: EvidenceView) -> RevisionProposalEnvelope:
    structure = _baseline_revision_structure()
    return RevisionProposalEnvelope(
        dimension=EvidenceDimension.SYMPTOM,
        baseline_label="Crash",
        proposed_label="Poor Performance",
        boundary_card_id=structure.boundary_cards[0].card_id,
        proposer_team_ids=("B",),
        proposer_report_digests=("c" * 64,),
        supporting_evidence_ids=("issue-body",),
        baseline_source_config_hash="a" * 64,
        baseline_source_predictions_sha256="b" * 64,
        taxonomy_structure_hash=taxonomy_structure_hash(structure),
        evidence_view_hash=canonical_evidence_view_hash(view),
    )


def _baseline_revision_assessment(
    team_id: str, view: EvidenceView
) -> BaselineRevisionAssessment:
    proposal = _baseline_revision_proposal(view)
    return BaselineRevisionAssessment(
        assessor_team_id=team_id,
        proposal_digest=revision_proposal_digest(proposal),
        dimension=EvidenceDimension.SYMPTOM,
        verdict=RevisionAssessmentVerdict.REVISE,
        baseline_label=proposal.baseline_label,
        proposed_label=proposal.proposed_label,
        boundary_card_id=proposal.boundary_card_id,
        contradicted_baseline_condition="execution completes slowly",
        satisfied_proposed_conditions=("execution completes slowly",),
        supporting_evidence_ids=proposal.supporting_evidence_ids,
        counter_evidence_ids=proposal.supporting_evidence_ids,
        baseline_source_config_hash=proposal.baseline_source_config_hash,
        baseline_source_predictions_sha256=(
            proposal.baseline_source_predictions_sha256
        ),
        taxonomy_structure_hash=proposal.taxonomy_structure_hash,
        evidence_view_hash=proposal.evidence_view_hash,
    )


def _heterogeneous_revision_assessment(
    team_id: str, view: EvidenceView
) -> BaselineRevisionAssessment:
    proposal = _baseline_revision_proposal(view)
    card = _baseline_revision_structure().boundary_cards[0]
    if team_id == "A":
        return normalize_revision_entailment_result(
            result=RevisionEntailmentResult.model_validate(
                {
                    "condition_findings": [
                        {
                            "condition": "execution completes slowly",
                            "status": "supported",
                            "citation_ids": ["issue-body"],
                        }
                    ],
                    "baseline_exclusion_finding": {
                        "condition": "execution completes slowly",
                        "status": "supported",
                        "citation_ids": ["issue-body"],
                    },
                    "outcome": "entailed",
                    "summary": "The direct observation entails the fixed candidate condition.",
                }
            ),
            proposal=proposal,
            routed_card=card,
        )
    return normalize_revision_falsification_result(
        result=RevisionFalsificationResult.model_validate(
            {
                "baseline_survival_finding": {
                    "condition": "execution terminates before returning",
                    "status": "refuted",
                    "citation_ids": ["issue-body"],
                },
                "proposed_defeater_findings": [
                    {
                        "condition": "execution terminates before returning",
                        "status": "refuted",
                        "citation_ids": ["issue-body"],
                    }
                ],
                "strongest_competing_reading": {
                    "label": "Crash",
                    "summary": "Crash is the strongest exact-pair competing reading here.",
                    "citation_ids": ["issue-body"],
                },
                "outcome": "revision_survives",
                "summary": "The candidate survives all fixed adversarial checks in the card.",
            }
        ),
        proposal=proposal,
        routed_card=card,
    )


def _task_specific_cross_raw(
    owner_team_id: str,
) -> EntailmentMappingCrossResult | FalsificationCoverageCrossResult:
    if owner_team_id == "A":
        return EntailmentMappingCrossResult.model_validate(
            {
                "condition_findings": [
                    {
                        "condition": "execution completes slowly",
                        "status": "confirmed",
                        "citation_ids": ["issue-body"],
                    }
                ],
                "baseline_exclusion_finding": {
                    "condition": "execution completes slowly",
                    "status": "confirmed",
                    "citation_ids": ["issue-body"],
                },
                "status": "consistent",
                "rationale": "Every exact entailment mapping is confirmed by its owner citation.",
            }
        )
    return FalsificationCoverageCrossResult.model_validate(
        {
            "baseline_survival_finding": {
                "condition": "execution terminates before returning",
                "status": "confirmed",
                "citation_ids": ["issue-body"],
            },
            "proposed_defeater_findings": [
                {
                    "condition": "execution terminates before returning",
                    "status": "confirmed",
                    "citation_ids": ["issue-body"],
                }
            ],
            "strongest_competing_reading_finding": {
                "label": "Crash",
                "status": "confirmed",
                "citation_ids": ["issue-body"],
            },
            "status": "consistent",
            "rationale": "Every exact falsification finding is confirmed by its owner citation.",
        }
    )


def test_stage3_team_report_allows_distinct_candidate_assessments_per_dimension() -> (
    None
):
    first = _baseline_revision_assessment("A", _stage3_view())
    second = first.model_copy(update={"proposal_digest": "e" * 64})
    payload = _stage3_team().model_dump(mode="python")
    payload["baseline_revision_assessments"] = [first, second]

    report = Stage3TeamReport.model_validate(payload)

    assert len(report.baseline_revision_assessments) == 2


def test_stage3_team_report_rejects_duplicate_candidate_assessment() -> None:
    assessment = _baseline_revision_assessment("A", _stage3_view())
    payload = _stage3_team().model_dump(mode="python")
    payload["baseline_revision_assessments"] = [assessment, assessment]

    with pytest.raises(ValueError, match="duplicate a proposal"):
        Stage3TeamReport.model_validate(payload)


def test_stage3_team_report_rejects_duplicate_consistency_for_same_proposal() -> None:
    assessment = _baseline_revision_assessment("B", _stage3_view())
    first = RevisionConsistencyReport(
        checker_team_id="A",
        assessment_owner_team_id="B",
        proposal_digest=assessment.proposal_digest,
        dimension=assessment.dimension,
        baseline_label="Crash",
        proposed_label="Poor Performance",
        status=ConsistencyStatus.CONSISTENT,
        rationale="The exact candidate evidence satisfies the routed boundary.",
        supporting_evidence_ids=("issue-body",),
        counter_evidence_ids=("issue-body",),
        assessment_digest=baseline_revision_assessment_digest(assessment),
        boundary_card_id="crash-performance",
        baseline_source_config_hash="a" * 64,
        baseline_source_predictions_sha256="b" * 64,
        taxonomy_structure_hash=assessment.taxonomy_structure_hash,
        evidence_view_hash=assessment.evidence_view_hash,
    )
    second = first.model_copy(
        update={"checker_team_id": "B", "assessment_owner_team_id": "A"}
    )
    payload = _stage3_team().model_dump(mode="python")
    payload["revision_consistency"] = [first, second]

    with pytest.raises(ValueError, match="duplicate a proposal"):
        Stage3TeamReport.model_validate(payload)


def test_baseline_revision_assessment_prompts_are_task_distinct_and_dimension_scoped() -> (
    None
):
    captured: dict[str, tuple[str, dict[str, object]]] = {}
    view = _stage3_view()
    proposal = _baseline_revision_proposal(view)
    structure = _baseline_revision_structure()

    def transport(system: str, user: str) -> str:
        decoded = json.loads(user)
        captured[decoded["team_id"]] = (system, decoded)
        if decoded["team_id"] == "A":
            return RevisionEntailmentResult.model_validate(
                {
                    "condition_findings": [
                        {
                            "condition": "execution completes slowly",
                            "status": "supported",
                            "citation_ids": ["issue-body"],
                        }
                    ],
                    "baseline_exclusion_finding": {
                        "condition": "execution completes slowly",
                        "status": "supported",
                        "citation_ids": ["issue-body"],
                    },
                    "outcome": "entailed",
                    "summary": "The observed completion behavior entails the proposed condition.",
                }
            ).model_dump_json()
        return RevisionFalsificationResult.model_validate(
            {
                "baseline_survival_finding": {
                    "condition": "execution terminates before returning",
                    "status": "refuted",
                    "citation_ids": ["issue-body"],
                },
                "proposed_defeater_findings": [
                    {
                        "condition": "execution terminates before returning",
                        "status": "refuted",
                        "citation_ids": ["issue-body"],
                    }
                ],
                "strongest_competing_reading": {
                    "label": "Crash",
                    "summary": "Crash is the strongest competitor but the cited outcome refutes it.",
                    "citation_ids": ["issue-body"],
                },
                "outcome": "revision_survives",
                "summary": "The revision survives every fixed adversarial boundary check.",
            }
        ).model_dump_json()

    anchor = BaselineAnchor(
        record_id="record-1",
        valid=True,
        symptom_label="Crash",
        root_cause_label="Incorrect Code Logic",
        source_config_hash="a" * 64,
        source_predictions_sha256="b" * 64,
    )
    agents = StructuredRoleAgents(StructuredModelClient(transport))
    for team_id in ("A", "B"):
        result = agents.baseline_revision_assessment(
            team_id,
            proposal,
            anchor,
            structure.boundary_cards[0],
            view.model_copy(deep=True),
        )
        assert result.verdict is RevisionAssessmentVerdict.REVISE
        assert result.assessor_team_id == team_id
        assert result.proposal_digest == revision_proposal_digest(proposal)

    system_a, prompt_a = captured["A"]
    system_b, prompt_b = captured["B"]
    assert "ROLE: revision_entailment" in system_a
    assert "ROLE: revision_falsification" in system_b
    assert "condition_findings" in system_a
    assert "proposed_defeater_findings" not in system_a
    assert "proposed_defeater_findings" in system_b
    assert "condition_findings" not in system_b
    assert prompt_a["context"]["boundary_card"]["card_id"] == "crash-performance"
    assert "proposer_team_ids" not in json.dumps(prompt_a["context"])
    assert "proposal_digest" not in json.dumps(prompt_a["context"])
    assert "proposal_digest" not in json.dumps(prompt_b["context"])
    assert "source_config_hash" not in json.dumps(prompt_a["context"])
    assert "source_config_hash" not in json.dumps(prompt_b["context"])
    assert "crash-performance" not in system_a
    assert "gold" not in (system_a + json.dumps(prompt_a)).lower()


@pytest.mark.parametrize("team_id", ["A", "B"])
def test_revision_raw_semantic_error_repairs_before_normalized_assessment(
    team_id: str,
) -> None:
    view = _stage3_view()
    proposal = _baseline_revision_proposal(view)
    structure = _baseline_revision_structure()
    if team_id == "A":
        invalid = {
            "condition_findings": [
                {
                    "condition": "invented condition outside the exact card",
                    "status": "supported",
                    "citation_ids": ["issue-body"],
                }
            ],
            "baseline_exclusion_finding": {
                "condition": "execution completes slowly",
                "status": "supported",
                "citation_ids": ["issue-body"],
            },
            "outcome": "entailed",
            "summary": "The invented condition appears supported but is not card-owned.",
        }
        valid = invalid | {
            "condition_findings": [
                {
                    "condition": "execution completes slowly",
                    "status": "supported",
                    "citation_ids": ["issue-body"],
                }
            ]
        }
    else:
        invalid = {
            "baseline_survival_finding": {
                "condition": "execution terminates before returning",
                "status": "refuted",
                "citation_ids": ["issue-body"],
            },
            "proposed_defeater_findings": [
                {
                    "condition": "invented candidate defeater",
                    "status": "refuted",
                    "citation_ids": ["issue-body"],
                }
            ],
            "strongest_competing_reading": {
                "label": "Crash",
                "summary": "Crash is the strongest competing reading in the exact pair.",
                "citation_ids": ["issue-body"],
            },
            "outcome": "revision_survives",
            "summary": "The candidate appears to survive but skipped a fixed defeater.",
        }
        valid = invalid | {
            "proposed_defeater_findings": [
                {
                    "condition": "execution terminates before returning",
                    "status": "refuted",
                    "citation_ids": ["issue-body"],
                }
            ]
        }
    replies = iter((json.dumps(invalid), json.dumps(valid)))
    anchor = BaselineAnchor(
        record_id="record-1",
        valid=True,
        symptom_label="Crash",
        root_cause_label="Incorrect Code Logic",
        source_config_hash="a" * 64,
        source_predictions_sha256="b" * 64,
    )

    result = StructuredRoleAgents(
        StructuredModelClient(lambda system, user: next(replies), max_schema_retries=1)
    ).baseline_revision_assessment(
        team_id, proposal, anchor, structure.boundary_cards[0], view
    )

    assert result.verdict is RevisionAssessmentVerdict.REVISE
    assert result.assessor_team_id == team_id


@pytest.mark.parametrize(
    "key",
    [
        "ground truth",
        "ground.truth",
        "groundTruth",
        "symptomLabel",
        "rootCauseLabel",
        "expected answer",
        "label answer",
    ],
)
def test_revision_prompt_rejects_nested_answer_bearing_evidence_metadata(
    key: str,
) -> None:
    called = False

    def transport(system: str, user: str) -> str:
        nonlocal called
        called = True
        return "{}"

    item_payload = _stage3_view().items[0].model_dump(mode="python")
    item_payload["metadata"] = {"nested": {key: "Crash"}}
    unsafe_view = EvidenceView.model_validate(
        _stage3_view().model_dump(mode="python") | {"items": [item_payload]}
    )
    anchor = BaselineAnchor(
        record_id="record-1",
        valid=True,
        symptom_label="Crash",
        root_cause_label="Incorrect Code Logic",
        source_config_hash="a" * 64,
        source_predictions_sha256="b" * 64,
    )

    with pytest.raises(ValueError, match="forbidden answer-bearing"):
        StructuredRoleAgents(
            StructuredModelClient(transport)
        ).baseline_revision_assessment(
            "A",
            _baseline_revision_proposal(unsafe_view),
            anchor,
            _baseline_revision_structure().boundary_cards[0],
            unsafe_view,
        )

    assert called is False


def test_normal_target_source_metadata_reaches_a_role_transport() -> None:
    called = False
    item_payload = _stage3_view().items[0].model_dump(mode="python")
    item_payload["metadata"] = {
        "target_source": "code_diff",
        "reference_url": "https://example.test",
    }
    view = EvidenceView.model_validate(
        _stage3_view().model_dump(mode="python") | {"items": [item_payload]}
    )

    def transport(system: str, user: str) -> str:
        nonlocal called
        called = True
        return json.dumps(_stage3_team().symptom.model_dump(mode="json"))

    StructuredRoleAgents(StructuredModelClient(transport)).symptom_analyst("A", view)
    assert called is True


def _sla_anchor() -> BaselineAnchor:
    return BaselineAnchor(
        record_id="record-1",
        valid=True,
        symptom_label="Crash",
        root_cause_label="Incorrect Code Logic",
        source_config_hash="a" * 64,
        source_predictions_sha256="b" * 64,
    )


def _sla_diagnosis() -> SlaJointDiagnosis:
    team = _stage3_team()
    return SlaJointDiagnosis(
        symptom=team.symptom,
        root_cause=team.root_cause.model_copy(
            update={
                "label": "API Misuse",
                "alternative_label": "Incorrect Code Logic",
            }
        ),
        symptom_matches_baseline=True,
        root_cause_matches_baseline=False,
    )


def _sla_verification() -> SlaJointVerification:
    return SlaJointVerification(
        symptom={
            "dimension": "symptom",
            "verdict": "accept_candidate",
            "supporting_evidence_ids": ["issue-body"],
            "rationale": "The direct evidence supports the observed symptom label.",
        },
        root_cause={
            "dimension": "root_cause",
            "verdict": "preserve_baseline",
            "counter_evidence_ids": ["issue-body"],
            "rationale": "The direct evidence supports preserving the Baseline cause.",
        },
    )


def _sla_arbitration() -> SlaConditionalArbitration:
    return SlaConditionalArbitration(
        root_cause=SlaDimensionArbitration(
            dimension=EvidenceDimension.ROOT_CAUSE,
            selected_label="Incorrect Code Logic",
            supporting_evidence_ids=["issue-body"],
            rationale="The exact evidence supports the fixed root-cause choice.",
        )
    )


class _CapturingSlaTransport:
    def __init__(self, responses: list[str]) -> None:
        self._responses = iter(responses)
        self.calls: list[tuple[str, str, ModelCallOptions]] = []

    def __call__(self, system: str, user: str, *, options: ModelCallOptions) -> str:
        self.calls.append((system, user, options))
        return next(self._responses)


def test_sla_roles_disable_thinking_use_bounded_tokens_and_isolate_prompts() -> None:
    diagnosis = _sla_diagnosis()
    transport = _CapturingSlaTransport(
        [
            diagnosis.model_dump_json(),
            _sla_verification().model_dump_json(),
            _sla_arbitration().model_dump_json(),
        ]
    )
    item = _stage3_view().items[0].model_dump(mode="python")
    item["metadata"] = {"error_cluster": "must-not-reach-model"}
    view = EvidenceView.model_validate(
        _stage3_view().model_dump(mode="python") | {"items": [item]}
    )
    agents = StructuredRoleAgents(StructuredModelClient(transport))

    returned_diagnosis = agents.sla_joint_diagnosis(view, _sla_anchor())
    returned_verification = agents.sla_joint_verifier(
        view, _sla_anchor(), returned_diagnosis
    )
    agents.sla_conditional_arbitrator(
        view, _sla_anchor(), returned_diagnosis, returned_verification
    )

    assert [
        (options.role, options.thinking_enabled, options.max_tokens)
        for _, _, options in transport.calls
    ] == [
        ("sla_joint_diagnosis", False, 2400),
        ("sla_joint_verifier", False, 1800),
        ("sla_conditional_arbitrator", False, 1800),
    ]
    prompts = "\n".join(system + user for system, user, _ in transport.calls).lower()
    assert "crash" in prompts
    assert "incorrect code logic" in prompts
    assert "gold" not in prompts
    assert "expected_answer" not in prompts
    assert "error_cluster" not in prompts
    diagnosis_prompt = transport.calls[0][0] + transport.calls[0][1]
    assert "Crash" in diagnosis_prompt
    assert "Poor Performance" in diagnosis_prompt
    assert "Incorrect Code Logic" in diagnosis_prompt
    assert "API Misuse" in diagnosis_prompt
    assert "error_cluster" not in diagnosis_prompt
    assert "must-not-reach-model" not in diagnosis_prompt
    verifier_prompt = transport.calls[1][1]
    assert diagnosis.symptom.behavior_claim not in verifier_prompt
    assert diagnosis.root_cause.defect_mechanism not in verifier_prompt
    assert diagnosis.root_cause.causal_chain[0] in verifier_prompt


@pytest.mark.parametrize("conflict_dimension", ("symptom", "root_cause"))
def test_sla_arbitrator_prompt_serializes_exactly_one_conflicting_dimension(
    conflict_dimension: str,
) -> None:
    base = _sla_diagnosis()
    if conflict_dimension == "symptom":
        candidate = base.model_copy(
            update={
                "symptom": base.symptom.model_copy(
                    update={"label": "Poor Performance"}
                ),
                "root_cause": base.root_cause.model_copy(
                    update={"label": "Incorrect Code Logic"}
                ),
                "symptom_matches_baseline": False,
                "root_cause_matches_baseline": True,
            }
        )
        verification = _sla_verification().model_copy(
            update={
                "symptom": _sla_verification().symptom.model_copy(
                    update={
                        "verdict": SlaVerifierVerdict.PRESERVE_BASELINE,
                        "supporting_evidence_ids": (),
                        "counter_evidence_ids": ("issue-body",),
                    }
                ),
                "root_cause": _sla_verification().root_cause.model_copy(
                    update={
                        "verdict": SlaVerifierVerdict.ACCEPT_CANDIDATE,
                        "supporting_evidence_ids": ("issue-body",),
                        "counter_evidence_ids": (),
                    }
                ),
            }
        )
        arbitration = SlaConditionalArbitration(
            symptom=SlaDimensionArbitration(
                dimension=EvidenceDimension.SYMPTOM,
                selected_label="Crash",
                supporting_evidence_ids=("issue-body",),
                rationale="The exact observation supports the Baseline symptom.",
            )
        )
    else:
        candidate = base.model_copy(
            update={
                "root_cause": base.root_cause.model_copy(
                    update={"label": "API Misuse"}
                ),
                "root_cause_matches_baseline": False,
            }
        )
        verification = _sla_verification()
        arbitration = _sla_arbitration()
    transport = _CapturingSlaTransport([arbitration.model_dump_json()])

    StructuredRoleAgents(StructuredModelClient(transport)).sla_conditional_arbitrator(
        _stage3_view(), _sla_anchor(), candidate, verification
    )

    context = json.loads(transport.calls[0][1])["context"]
    assert context["conflicting_dimensions"] == [conflict_dimension]
    assert set(context["conflicting_candidate_baseline_pairs"]) == {conflict_dimension}


def test_sla_diagnosis_repairs_false_persisted_baseline_comparison() -> None:
    valid = _sla_diagnosis()
    invalid = valid.model_copy(update={"symptom_matches_baseline": False})
    transport = _CapturingSlaTransport(
        [invalid.model_dump_json(), valid.model_dump_json()]
    )

    result = StructuredRoleAgents(
        StructuredModelClient(transport, max_schema_retries=1)
    ).sla_joint_diagnosis(_stage3_view(), _sla_anchor())

    assert result.symptom_matches_baseline is True
    assert len(transport.calls) == 2


@pytest.mark.parametrize(
    ("role", "invalid_payload"),
    [
        (
            "diagnosis",
            lambda: _sla_diagnosis()
            .model_copy(
                update={
                    "symptom": _sla_diagnosis().symptom.model_copy(
                        update={"supporting_evidence_ids": ("unknown",)}
                    )
                }
            )
            .model_dump_json(),
        ),
        (
            "diagnosis",
            lambda: _sla_diagnosis()
            .model_copy(
                update={
                    "symptom": _sla_diagnosis().symptom.model_copy(
                        update={"label": "invented symptom"}
                    )
                }
            )
            .model_dump_json(),
        ),
        (
            "arbitration",
            lambda: _sla_arbitration()
            .model_copy(
                update={
                    "root_cause": _sla_arbitration().root_cause.model_copy(
                        update={"selected_label": "third label"}
                    )
                }
            )
            .model_dump_json(),
        ),
    ],
)
def test_sla_semantic_errors_consume_exactly_one_schema_repair(
    role: str,
    invalid_payload: object,
) -> None:
    diagnosis = _sla_diagnosis()
    verification = _sla_verification()
    valid = (
        diagnosis.model_dump_json()
        if role == "diagnosis"
        else _sla_arbitration().model_dump_json()
    )
    transport = _CapturingSlaTransport([invalid_payload(), valid])  # type: ignore[operator]
    agents = StructuredRoleAgents(
        StructuredModelClient(transport, max_schema_retries=5)
    )

    if role == "diagnosis":
        result = agents.sla_joint_diagnosis(_stage3_view(), _sla_anchor())
        assert result == diagnosis
    else:
        result = agents.sla_conditional_arbitrator(
            _stage3_view(), _sla_anchor(), diagnosis, verification
        )
        assert result == _sla_arbitration()
    assert len(transport.calls) == 2
    assert agents.telemetry()["calls"][0]["attempts"] == 2


def test_sla_second_semantic_error_raises_the_role_schema_name() -> None:
    invalid = (
        _sla_diagnosis()
        .model_copy(
            update={
                "symptom": _sla_diagnosis().symptom.model_copy(
                    update={"supporting_evidence_ids": ("unknown",)}
                )
            }
        )
        .model_dump_json()
    )
    transport = _CapturingSlaTransport([invalid, invalid])

    with pytest.raises(StructuredOutputError) as error:
        StructuredRoleAgents(
            StructuredModelClient(transport, max_schema_retries=5)
        ).sla_joint_diagnosis(_stage3_view(), _sla_anchor())

    assert error.value.schema_name == "SlaJointDiagnosis"
    assert len(transport.calls) == 2


def test_sla_diagnosis_repairs_dimension_incapable_citation() -> None:
    content = "The implementation contains a branch but not the observed outcome."
    incapable_item = EvidenceItem(
        evidence_id="code-context",
        record_id="record-1",
        source_type="code_context",
        source_uri="https://example.test/code",
        retrieved_at="2026-08-22T00:00:00Z",
        content=content,
        content_sha256=hashlib.sha256(content.encode()).hexdigest(),
        explicitness=EvidenceExplicitness.DIRECT,
    )
    view = _stage3_view().model_copy(
        update={"items": _stage3_view().items + (incapable_item,)}
    )
    invalid = (
        _sla_diagnosis()
        .model_copy(
            update={
                "symptom": _sla_diagnosis().symptom.model_copy(
                    update={"supporting_evidence_ids": ("code-context",)}
                )
            }
        )
        .model_dump_json()
    )
    transport = _CapturingSlaTransport([invalid, _sla_diagnosis().model_dump_json()])

    result = StructuredRoleAgents(
        StructuredModelClient(transport, max_schema_retries=5)
    ).sla_joint_diagnosis(view, _sla_anchor())

    assert result == _sla_diagnosis()
    assert len(transport.calls) == 2


@pytest.mark.parametrize(
    "mutation",
    [
        "citation",
        "incapable_citation",
    ],
)
def test_revision_consistency_agent_rejects_unbound_output_before_accepting_repair(
    mutation: str,
) -> None:
    view = _stage3_view()
    incapable_content = "Internal code context describes an implementation branch."
    view = view.model_copy(
        update={
            "items": view.items
            + (
                EvidenceItem(
                    evidence_id="code-context",
                    record_id="record-1",
                    source_type="code_context",
                    source_uri="https://example.test/code",
                    retrieved_at="2026-08-16T00:00:00Z",
                    content=incapable_content,
                    content_sha256=hashlib.sha256(
                        incapable_content.encode()
                    ).hexdigest(),
                    explicitness=EvidenceExplicitness.DIRECT,
                ),
            )
        }
    )
    structure = _baseline_revision_structure()
    proposal = _baseline_revision_proposal(view)
    assessment = _heterogeneous_revision_assessment("B", view)
    valid = _task_specific_cross_raw("B")
    invalid = valid
    if mutation == "citation":
        invalid = valid.model_copy(
            update={
                "strongest_competing_reading_finding": (
                    valid.strongest_competing_reading_finding.model_copy(
                        update={"citation_ids": ("unknown-id",)}
                    )
                )
            }
        )
    if mutation == "incapable_citation":
        invalid = valid.model_copy(
            update={
                "strongest_competing_reading_finding": (
                    valid.strongest_competing_reading_finding.model_copy(
                        update={"citation_ids": ("code-context",)}
                    )
                )
            }
        )
    replies = iter((invalid.model_dump_json(), valid.model_dump_json()))
    result = StructuredRoleAgents(
        StructuredModelClient(lambda system, user: next(replies), max_schema_retries=1)
    ).baseline_revision_consistency(
        checker_team_id="A",
        assessment_owner_team_id="B",
        proposal=proposal,
        assessment=assessment,
        routed_card=structure.boundary_cards[0],
        view=view,
    )
    assert result.status is valid.status
    assert result.supporting_evidence_ids == assessment.supporting_evidence_ids
    assert result.assessment_digest == baseline_revision_assessment_digest(assessment)


def test_revision_consistency_prompt_supplies_every_exact_validator_binding() -> None:
    view = _stage3_view()
    structure = _baseline_revision_structure()
    proposal = _baseline_revision_proposal(view)
    assessment = _heterogeneous_revision_assessment("B", view)
    valid = _task_specific_cross_raw("B")
    captured: dict[str, object] = {}

    def transport(system: str, user: str) -> str:
        captured.update(json.loads(user))
        return valid.model_dump_json()

    client = StructuredModelClient(transport)
    result = StructuredRoleAgents(client).baseline_revision_consistency(
        checker_team_id="A",
        assessment_owner_team_id="B",
        proposal=proposal,
        assessment=assessment,
        routed_card=structure.boundary_cards[0],
        view=view,
    )

    assert result.status is ConsistencyStatus.CONSISTENT
    assert captured["context"]["cross_check_input"] == {
        "assessment_owner_task": "revision_falsification",
        "raw_assessment": assessment.raw_falsification_result.model_dump(mode="json"),
        "dimension": "symptom",
        "baseline_label": "Crash",
        "proposed_label": "Poor Performance",
        "boundary_card_id": "crash-performance",
        "supporting_evidence_ids": ["issue-body"],
        "counter_evidence_ids": ["issue-body"],
        "normalized_conditions": ["execution completes slowly"],
    }
    assert client.telemetry()["calls"][0]["attempts"] == 1


@pytest.mark.parametrize(
    ("checker_team_id", "owner_team_id", "expected_target"),
    (
        ("A", "B", "AUDIT TARGET: falsification coverage"),
        ("B", "A", "AUDIT TARGET: entailment mapping"),
    ),
)
def test_revision_cross_prompt_binds_and_audits_the_opposite_raw_task(
    checker_team_id: str,
    owner_team_id: str,
    expected_target: str,
) -> None:
    view = _stage3_view()
    structure = _baseline_revision_structure()
    proposal = _baseline_revision_proposal(view)
    assessment = _heterogeneous_revision_assessment(owner_team_id, view)
    raw_cross = _task_specific_cross_raw(owner_team_id)
    valid = RevisionConsistencyReport(
        checker_team_id=checker_team_id,
        assessment_owner_team_id=owner_team_id,
        proposal_digest=revision_proposal_digest(proposal),
        dimension=assessment.dimension,
        baseline_label=assessment.baseline_label,
        proposed_label=assessment.proposed_label,
        status=ConsistencyStatus.CONSISTENT,
        rationale=raw_cross.rationale,
        supporting_evidence_ids=assessment.supporting_evidence_ids,
        counter_evidence_ids=assessment.counter_evidence_ids,
        assessment_digest=baseline_revision_assessment_digest(assessment),
        assessment_owner_task=assessment.assessor_task,
        assessment_raw_result_digest=assessment.raw_result_digest,
        boundary_card_id=proposal.boundary_card_id,
        baseline_source_config_hash=proposal.baseline_source_config_hash,
        baseline_source_predictions_sha256=(
            proposal.baseline_source_predictions_sha256
        ),
        taxonomy_structure_hash=proposal.taxonomy_structure_hash,
        evidence_view_hash=proposal.evidence_view_hash,
    )
    captured: dict[str, str] = {}

    def transport(system: str, user: str) -> str:
        captured.update(system=system, user=user)
        return raw_cross.model_dump_json()

    result = StructuredRoleAgents(
        StructuredModelClient(transport)
    ).baseline_revision_consistency(
        checker_team_id,
        owner_team_id,
        proposal,
        assessment,
        structure.boundary_cards[0],
        view,
    )

    assert result == valid
    assert expected_target in captured["system"]
    context = json.loads(captured["user"])["context"]
    assert context["cross_check_input"]["assessment_owner_task"] == (
        assessment.assessor_task.value
    )
    assert "proposal_digest" not in json.dumps(context["cross_check_input"])
    assert "source_config_hash" not in json.dumps(context["cross_check_input"])


@pytest.mark.parametrize(
    ("checker_team_id", "owner_team_id"),
    (("A", "B"), ("B", "A")),
)
def test_revision_cross_prompt_keeps_opaque_bindings_out_of_model_output(
    checker_team_id: str,
    owner_team_id: str,
) -> None:
    view = _stage3_view()
    structure = _baseline_revision_structure()
    proposal = _baseline_revision_proposal(view)
    assessment = _heterogeneous_revision_assessment(owner_team_id, view)
    raw_cross = _task_specific_cross_raw(owner_team_id)
    calls: list[str] = []

    def instruction_following_transport(system: str, user: str) -> str:
        calls.append(system)
        payload = raw_cross.model_dump(mode="json")
        if "do not output opaque" not in system.lower():
            payload["proposal_digest"] = "copied-opaque-value"
        return json.dumps(payload)

    result = StructuredRoleAgents(
        StructuredModelClient(instruction_following_transport)
    ).baseline_revision_consistency(
        checker_team_id,
        owner_team_id,
        proposal,
        assessment,
        structure.boundary_cards[0],
        view,
    )

    assert result.status is ConsistencyStatus.CONSISTENT
    assert len(calls) == 1
    assert "context.expected_bindings" not in calls[0]
    assert "do not output opaque" in calls[0].lower()


class _CapturedStage3RolePrompts:
    """Runs real role adapters while retaining their prompt boundary."""

    def __init__(self) -> None:
        self.system_by_team: dict[str, str] = {}
        self.user_by_team: dict[str, str] = {}
        self.agents = StructuredRoleAgents(StructuredModelClient(self._transport))

    def _transport(self, system: str, user: str) -> str:
        team_id = json.loads(user)["team_id"]
        self.system_by_team[team_id] = system
        self.user_by_team[team_id] = user
        if "ROLE: symptom_analyst" in system:
            return json.dumps(_stage3_team().symptom.model_dump(mode="json"))
        if "ROLE: root_cause_analyst" in system:
            return json.dumps(_stage3_team().root_cause.model_dump(mode="json"))
        return json.dumps(_stage3_team().consistency.model_dump(mode="json"))

    def system_for(self, team_id: str) -> str:
        return self.system_by_team[team_id]

    def evidence_payload(self, team_id: str) -> dict[str, object]:
        return json.loads(self.user_by_team[team_id])["evidence_view"]


def _run_stage3_role(
    captured: _CapturedStage3RolePrompts,
    role_method: str,
    team_id: str,
) -> None:
    team = _stage3_team()
    if role_method == "consistency_checker":
        captured.agents.consistency_checker(
            team_id,
            team.symptom,
            team.root_cause,
            _stage3_view(),
        )
        return
    getattr(captured.agents, role_method)(team_id, _stage3_view())


@pytest.mark.parametrize("role_method", ["symptom_analyst", "root_cause_analyst"])
def test_stage3_team_prompts_use_distinct_domain_independent_perspectives(
    role_method: str,
) -> None:
    captured = _CapturedStage3RolePrompts()

    _run_stage3_role(captured, role_method, "A")
    _run_stage3_role(captured, role_method, "B")

    team_a = captured.system_for("A")
    team_b = captured.system_for("B")
    assert "EVIDENCE-FIRST" in team_a
    assert "FALSIFICATION-FIRST" not in team_a
    assert "FALSIFICATION-FIRST" in team_b
    assert "EVIDENCE-FIRST" not in team_b
    assert captured.evidence_payload("A") == captured.evidence_payload("B")


@pytest.mark.parametrize("role_method", ["symptom_analyst", "root_cause_analyst"])
@pytest.mark.parametrize("team_id", ["A", "B"])
def test_stage3_analyst_perspectives_preserve_role_contracts(
    role_method: str,
    team_id: str,
) -> None:
    captured = _CapturedStage3RolePrompts()

    _run_stage3_role(captured, role_method, team_id)

    system = captured.system_for(team_id)
    output_contract = system.split("OUTPUT CONTRACT:\n", 1)[1]
    assert "Cite exact evidence IDs." in system
    assert "gold_annotation" in system
    assert "alternative_label" in system
    assert "OWNED LABEL SET" in system
    assert '"label"' in output_contract
    assert '"alternative_label"' in output_contract
    assert "Return one JSON object matching this contract exactly." in system


@pytest.mark.parametrize("role_method", ["symptom_analyst", "root_cause_analyst"])
def test_stage3_analyst_evidence_first_prompt_requires_substantive_evidence_work(
    role_method: str,
) -> None:
    captured = _CapturedStage3RolePrompts()

    _run_stage3_role(captured, role_method, "A")

    system = captured.system_for("A")
    assert "Inventory the cited facts before proposing any label" in system
    assert "Separate direct observations from inferences" in system
    assert "nearest legal alternative only after that inventory" in system
    assert "cited evidence tier" in system


@pytest.mark.parametrize("role_method", ["symptom_analyst", "root_cause_analyst"])
def test_stage3_analyst_falsification_first_prompt_requires_substantive_boundary_work(
    role_method: str,
) -> None:
    captured = _CapturedStage3RolePrompts()

    _run_stage3_role(captured, role_method, "B")

    system = captured.system_for("B")
    assert "leading hypothesis" in system
    assert "nearest legal alternative" in system
    assert "facts would falsify" in system
    assert "counterevidence" in system
    assert "boundary that survives the falsification attempt" in system


def test_stage3_consistency_prompts_use_distinct_perspectives_and_keep_boundaries() -> (
    None
):
    captured = _CapturedStage3RolePrompts()

    _run_stage3_role(captured, "consistency_checker", "A")
    _run_stage3_role(captured, "consistency_checker", "B")

    team_a = captured.system_for("A")
    team_b = captured.system_for("B")
    assert "EVIDENCE-FIRST" in team_a
    assert "FALSIFICATION-FIRST" not in team_a
    assert "FALSIFICATION-FIRST" in team_b
    assert "EVIDENCE-FIRST" not in team_b
    assert captured.evidence_payload("A") == captured.evidence_payload("B")


@pytest.mark.parametrize("team_id", ["A", "B"])
def test_stage3_checker_perspectives_preserve_schema_and_non_label_authority(
    team_id: str,
) -> None:
    captured = _CapturedStage3RolePrompts()

    _run_stage3_role(captured, "consistency_checker", team_id)

    system = captured.system_for(team_id)
    output_contract = system.split("OUTPUT CONTRACT:\n", 1)[1]
    assert "ROLE: causal_consistency_checker" in system
    assert "You must not assign or replace any classification label." in system
    assert "Cite exact evidence IDs." in system
    assert "gold_annotation" in system
    assert "legal alternatives supplied in the reports" in system
    assert '"status"' in output_contract
    assert '"label"' not in output_contract
    assert "Return one JSON object matching this contract exactly." in system


def test_stage3_checker_evidence_first_prompt_requires_positive_chain_validation() -> (
    None
):
    captured = _CapturedStage3RolePrompts()

    _run_stage3_role(captured, "consistency_checker", "A")

    system = captured.system_for("A")
    assert "Validate the cited positive causal chain" in system
    assert "cited evidence tier" in system
    assert "consistency status, rationale, citations, and evidence requests" in system


def test_stage3_checker_falsification_first_prompt_uses_schema_compatible_outcomes() -> (
    None
):
    captured = _CapturedStage3RolePrompts()

    _run_stage3_role(captured, "consistency_checker", "B")

    system = captured.system_for("B")
    assert "temporal, causal, and granularity counterexamples" in system
    assert "consistency status, rationale, citations, and evidence requests" in system
    assert "lower confidence" not in system
    assert "record the unresolved evidence gap" not in system


def test_default_stage3_team_perspectives_are_immutable() -> None:
    with pytest.raises(TypeError):
        DEFAULT_STAGE3_TEAM_PERSPECTIVES["A"] = DEFAULT_STAGE3_TEAM_PERSPECTIVES["B"]


def test_unknown_stage3_team_id_fails_closed() -> None:
    with pytest.raises(ValueError, match="unknown Stage 3 team"):
        StructuredRoleAgents(
            StructuredModelClient(lambda system, user: "{}")
        ).symptom_analyst("C", _stage3_view())


def test_root_cause_analyst_retries_label_from_wrong_taxonomy_dimension() -> None:
    prompts: list[str] = []
    wrong = _stage3_team().root_cause.model_dump(mode="json")
    wrong["label"] = "Crash"
    wrong["alternative_label"] = "Poor Performance"
    correct = _stage3_team().root_cause.model_dump(mode="json")
    responses = iter((json.dumps(wrong), json.dumps(correct)))

    def transport(system: str, user: str) -> str:
        prompts.append(user)
        return next(responses)

    report = StructuredRoleAgents(
        StructuredModelClient(transport, max_schema_retries=1)
    ).root_cause_analyst("A", _stage3_view())

    assert report.label == "Incorrect Code Logic"
    assert len(prompts) == 2
    assert "root_cause taxonomy" in prompts[1]


def test_stage3_role_prompts_state_exact_owned_label_set() -> None:
    captured: dict[str, str] = {}

    def symptom_transport(system: str, user: str) -> str:
        captured["symptom"] = system
        return json.dumps(_stage3_team().symptom.model_dump(mode="json"))

    def root_transport(system: str, user: str) -> str:
        captured["root"] = system
        return json.dumps(_stage3_team().root_cause.model_dump(mode="json"))

    StructuredRoleAgents(StructuredModelClient(symptom_transport)).symptom_analyst(
        "A", _stage3_view()
    )
    StructuredRoleAgents(StructuredModelClient(root_transport)).root_cause_analyst(
        "A", _stage3_view()
    )

    assert 'OWNED LABEL SET: ["Crash", "Poor Performance"]' in captured["symptom"]
    assert 'OWNED LABEL SET: ["Incorrect Code Logic", "API Misuse"]' in captured["root"]
    assert "Never use a label from the other taxonomy dimension" in captured["root"]


def test_stage3_role_prompt_requires_structured_boundary_citations() -> None:
    captured: dict[str, str] = {}

    def transport(system: str, user: str) -> str:
        captured["system"] = system
        return json.dumps(_stage3_team().symptom.model_dump(mode="json"))

    report = StructuredRoleAgents(StructuredModelClient(transport)).symptom_analyst(
        "A", _stage3_view()
    )

    assert report.boundary_evidence_ids == ("issue-body",)
    assert (
        "boundary_evidence_ids must cite direct, dimension-relevant evidence"
        in captured["system"]
    )


def test_ase_root_prompt_accepts_frozen_issue_and_maintainer_cause_evidence() -> None:
    captured: dict[str, str] = {}

    def transport(system: str, user: str) -> str:
        captured["system"] = system
        return json.dumps(_stage3_team().root_cause.model_dump(mode="json"))

    StructuredRoleAgents(StructuredModelClient(transport)).root_cause_analyst(
        "A", _stage3_view()
    )

    assert "frozen issue body" in captured["system"]
    assert "maintainer discussion" in captured["system"]
    assert "legitimate root-cause evidence" in captured["system"]
    assert (
        "Do not require unavailable code, patch, commit history" in captured["system"]
    )


def test_issta_root_prompt_retains_code_mechanism_semantics() -> None:
    captured: dict[str, str] = {}
    view = _stage3_view().model_copy(update={"domain_profile": "issta2024"})

    def transport(system: str, user: str) -> str:
        captured["system"] = system
        return json.dumps(_stage3_team().root_cause.model_dump(mode="json"))

    StructuredRoleAgents(StructuredModelClient(transport)).root_cause_analyst("A", view)

    assert "invert_patch" in captured["system"]
    assert "pre-fix defect mechanism" in captured["system"]
    assert "frozen issue body" not in captured["system"]


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
def test_stage3_role_retries_unknown_citation_against_exact_view(
    role: str,
    field: str,
) -> None:
    team = _stage3_team()
    valid_report = getattr(team, role)
    invalid_report = valid_report.model_copy(update={field: ["unknown-evidence"]})
    responses = iter(
        (
            json.dumps(invalid_report.model_dump(mode="json")),
            json.dumps(valid_report.model_dump(mode="json")),
        )
    )
    prompts: list[str] = []

    def transport(system: str, user: str) -> str:
        prompts.append(user)
        return next(responses)

    agents = StructuredRoleAgents(
        StructuredModelClient(transport, max_schema_retries=1)
    )
    if role == "symptom":
        result = agents.symptom_analyst("A", _stage3_view())
    elif role == "root_cause":
        result = agents.root_cause_analyst("A", _stage3_view())
    else:
        result = agents.consistency_checker(
            "A", team.symptom, team.root_cause, _stage3_view()
        )

    assert getattr(result, field) == getattr(valid_report, field)
    assert len(prompts) == 2
    assert "unknown-evidence" in prompts[1]
    assert "not present in the exact evidence view" in prompts[1]


@pytest.mark.parametrize("role", ["symptom", "root_cause", "consistency"])
def test_stage3_role_accepts_known_citations_from_exact_view(role: str) -> None:
    team = _stage3_team()
    calls = 0

    def transport(system: str, user: str) -> str:
        nonlocal calls
        calls += 1
        return json.dumps(getattr(team, role).model_dump(mode="json"))

    agents = StructuredRoleAgents(StructuredModelClient(transport))
    if role == "symptom":
        agents.symptom_analyst("A", _stage3_view())
    elif role == "root_cause":
        agents.root_cause_analyst("A", _stage3_view())
    else:
        agents.consistency_checker("A", team.symptom, team.root_cause, _stage3_view())

    assert calls == 1


def test_stage3_arbitration_candidates_include_boundary_evidence_ids() -> None:
    captured: dict[str, str] = {}
    report = _stage3_team()
    disagreement = DisagreementMap(
        dimensions=["unsupported_symptom_boundary"],
        details={"unsupported_symptom_boundary": {"team_a": {}}},
        requires_arbitration=True,
    )

    def transport(system: str, user: str) -> str:
        captured["system"] = system
        captured["user"] = user
        return json.dumps(
            {
                "resolution_status": ResolutionStatus.UNRESOLVED.value,
                "rationale": "The cited evidence cannot resolve the symptom boundary.",
                "unresolved_dimensions": ["unsupported_symptom_boundary"],
                "missing_facts": [
                    "Direct evidence distinguishing the labels is absent."
                ],
            }
        )

    StructuredRoleAgents(StructuredModelClient(transport)).stage3_arbitrator(
        Stage3ArbitrationPacket(
            domain_profile="ase2022",
            taxonomy=_stage3_view().taxonomy,
            disagreement=disagreement,
            team_a=AnonymousStage3TeamReport(
                symptom=report.symptom,
                root_cause=report.root_cause,
                consistency=report.consistency,
            ),
            team_b=AnonymousStage3TeamReport(
                symptom=report.symptom,
                root_cause=report.root_cause,
                consistency=report.consistency,
            ),
            classification_ledger_version=1,
            relevant_evidence=_stage3_view().items,
        )
    )

    payload = json.loads(captured["user"])
    assert payload["candidates"]["team_a"]["symptom"]["boundary_evidence_ids"] == [
        "issue-body"
    ]
    assert payload["candidates"]["team_a"]["root_cause"]["boundary_evidence_ids"] == [
        "issue-body"
    ]
    assert (
        "cause_review does not authorize changing root_cause_label"
        in captured["system"]
    )


def _ase_view() -> EvidenceView:
    content = "A caller receives unexpected output because the API is misused."
    return EvidenceView(
        record_id="ase-1",
        task="Classify the confirmed defect.",
        taxonomy={
            "symptom": ["Unexpected Output", "Crash"],
            "root_cause": ["API Misuse", "Incorrect Code Logic"],
        },
        domain_profile="ase2022",
        ledger_version=1,
        items=(
            EvidenceItem(
                evidence_id="comment-1",
                record_id="ase-1",
                source_type="issue_comments",
                source_uri="https://example.test/issues/1#comment-1",
                retrieved_at="2026-08-11T12:00:00Z",
                content=content,
                content_sha256=hashlib.sha256(content.encode()).hexdigest(),
                explicitness=EvidenceExplicitness.DIRECT,
            ),
        ),
    )


def _ase_capability_split_view() -> EvidenceView:
    mechanism_content = "A maintainer identifies the API contract defect mechanism."
    symptom_content = "A runtime observation records the unexpected output."
    anchor_content = "The anchor records the observed behavior and its cause."
    return EvidenceView(
        record_id="ase-capability-1",
        task="Classify the confirmed defect.",
        taxonomy={
            "symptom": ["Unexpected Output", "Crash"],
            "root_cause": ["API Misuse", "Incorrect Code Logic"],
        },
        domain_profile="ase2022",
        ledger_version=1,
        items=(
            EvidenceItem(
                evidence_id="comment-1",
                record_id="ase-capability-1",
                source_type="test_result",
                source_uri="https://example.test/issues/1#anchor",
                retrieved_at="2026-08-11T11:59:00Z",
                content=anchor_content,
                content_sha256=hashlib.sha256(anchor_content.encode()).hexdigest(),
                explicitness=EvidenceExplicitness.DIRECT,
            ),
            EvidenceItem(
                evidence_id="feg-node-test",
                record_id="ase-capability-1",
                source_type="maintainer_confirmation",
                source_uri="https://example.test/issues/1#maintainer",
                retrieved_at="2026-08-11T12:00:00Z",
                content=mechanism_content,
                content_sha256=hashlib.sha256(mechanism_content.encode()).hexdigest(),
                explicitness=EvidenceExplicitness.DIRECT,
                metadata={"evidence_capabilities": ["defect_mechanism"]},
            ),
            EvidenceItem(
                evidence_id="symptom-node-test",
                record_id="ase-capability-1",
                source_type="runtime_observation",
                source_uri="https://example.test/issues/1#runtime",
                retrieved_at="2026-08-11T12:01:00Z",
                content=symptom_content,
                content_sha256=hashlib.sha256(symptom_content.encode()).hexdigest(),
                explicitness=EvidenceExplicitness.DIRECT,
            ),
        ),
    )


def _valid_joint_anchor_payload() -> dict[str, object]:
    return {
        "symptom": {
            "label": "Unexpected Output",
            "behavior_claim": "The caller receives an unexpected result before the fix.",
            "supporting_evidence_ids": ["comment-1"],
            "counter_evidence_ids": ["comment-1"],
            "alternative_label": "Crash",
            "boundary_reason": "The evidence records a result rather than a terminated process.",
            "boundary_evidence_ids": ["comment-1"],
            "confidence": 0.82,
            "evidence_sufficiency": "sufficient",
        },
        "root_cause": {
            "label": "API Misuse",
            "defect_mechanism": "The caller invokes the API with an invalid contract assumption.",
            "causal_chain": [
                "The caller selects the API.",
                "The call violates its contract.",
                "The API returns an unexpected result.",
            ],
            "supporting_evidence_ids": ["comment-1"],
            "counter_evidence_ids": ["comment-1"],
            "alternative_label": "Incorrect Code Logic",
            "boundary_reason": "The evidence identifies caller use rather than an internal branch defect.",
            "boundary_evidence_ids": ["comment-1"],
            "confidence": 0.84,
            "evidence_sufficiency": "sufficient",
        },
        "causal_account": "The invalid API use produces the unexpected result described by the report.",
        "shared_supporting_evidence_ids": ["comment-1"],
        "shared_counter_evidence_ids": ["comment-1"],
        "shared_boundary_evidence_ids": ["comment-1"],
    }


def _anchor() -> JointAnchorReport:
    return JointAnchorReport.model_validate(_valid_joint_anchor_payload())


def _valid_root_verification_payload() -> dict[str, object]:
    return {
        "dimension": "root_cause",
        "verdict": "accept",
        "anchor_label": "API Misuse",
        "alternative_label": None,
        "rationale": "The cited record supports the anchor mechanism without a stronger alternative.",
        "supporting_evidence_ids": ["comment-1"],
        "counter_evidence_ids": ["comment-1"],
        "confidence": 0.85,
    }


def test_ase_root_cause_prompt_separates_caller_api_contract_from_internal_logic() -> (
    None
):
    """Removing the responsibility rubric must regress this to generic logic."""

    def transport(system: str, _user: str) -> str:
        applies_responsibility_boundary = all(
            phrase in system
            for phrase in (
                "caller violates an external API contract",
                "error catching or propagation is itself causal",
                "own algorithm or control flow",
            )
        )
        label = (
            "API Misuse" if applies_responsibility_boundary else "Incorrect Code Logic"
        )
        alternative = (
            "Incorrect Code Logic" if applies_responsibility_boundary else "API Misuse"
        )
        return json.dumps(
            {
                "label": label,
                "defect_mechanism": (
                    "The caller dereferences an optional API result without satisfying "
                    "the API contract."
                ),
                "causal_chain": [
                    "The caller invokes an external API.",
                    "The API returns no result under its documented contract.",
                    "The caller dereferences the absent result and fails.",
                ],
                "supporting_evidence_ids": ["comment-1"],
                "counter_evidence_ids": [],
                "alternative_label": alternative,
                "boundary_reason": (
                    "The causal responsibility belongs to caller-side API use, not an "
                    "independent internal branch defect."
                ),
                "boundary_evidence_ids": ["comment-1"],
                "confidence": 0.86,
                "evidence_sufficiency": "sufficient",
                "unresolved_evidence_gaps": [],
                "evidence_requests": [],
            }
        )

    result = StructuredRoleAgents(
        StructuredModelClient(transport, max_schema_retries=0)
    ).root_cause_analyst("A", _ase_view())

    assert result.label == "API Misuse"
    assert result.alternative_label == "Incorrect Code Logic"


def _recording_agents(
    payload: dict[str, object],
) -> tuple[StructuredRoleAgents, list[ModelCallOptions]]:
    calls: list[ModelCallOptions] = []

    def transport(system: str, user: str, options: ModelCallOptions) -> str:
        calls.append(options)
        return json.dumps(payload)

    return StructuredRoleAgents(StructuredModelClient(transport)), calls


def _retrying_agents(
    invalid: dict[str, object], valid: dict[str, object]
) -> tuple[StructuredRoleAgents, list[str]]:
    prompts: list[str] = []
    responses = iter((json.dumps(invalid), json.dumps(valid)))

    def transport(system: str, user: str) -> str:
        prompts.append(user)
        return next(responses)

    return (
        StructuredRoleAgents(StructuredModelClient(transport, max_schema_retries=1)),
        prompts,
    )


def test_joint_anchor_receives_both_taxonomies_and_team_perspective() -> None:
    agents, calls = _recording_agents(_valid_joint_anchor_payload())

    report = agents.joint_anchor("B", _ase_view())

    assert report.symptom.label == "Unexpected Output"
    assert report.root_cause.label == "API Misuse"
    assert calls[0].role == "joint_anchor"
    assert calls[0].team_id == "B"
    assert calls[0].perspective == "falsification_first"


@pytest.mark.parametrize(
    (
        "singleton_dimension",
        "singleton_label",
        "singleton_instruction",
        "multi_instruction",
    ),
    [
        (
            "symptom",
            "Unexpected Output",
            "The symptom label and alternative_label must both use the only member "
            'of this exact SYMPTOM LABEL SET: ["Unexpected Output"]; they may '
            "therefore be identical.",
            "The root-cause label and alternative_label must be distinct members "
            "of this exact ROOT-CAUSE LABEL SET",
        ),
        (
            "root_cause",
            "API Misuse",
            "The root-cause label and alternative_label must both use the only member "
            'of this exact ROOT-CAUSE LABEL SET: ["API Misuse"]; they may '
            "therefore be identical.",
            "The symptom label and alternative_label must be distinct members "
            "of this exact SYMPTOM LABEL SET",
        ),
    ],
)
def test_joint_anchor_prompt_gives_satisfiable_singleton_label_instructions(
    singleton_dimension: str,
    singleton_label: str,
    singleton_instruction: str,
    multi_instruction: str,
) -> None:
    payload = _valid_joint_anchor_payload()
    classification = payload[singleton_dimension]
    assert isinstance(classification, dict)
    classification["alternative_label"] = singleton_label
    taxonomy = {
        "symptom": ["Unexpected Output", "Crash"],
        "root_cause": ["API Misuse", "Incorrect Code Logic"],
    }
    taxonomy[singleton_dimension] = [singleton_label]
    view = _ase_view().model_copy(update={"taxonomy": taxonomy})
    captured: dict[str, str] = {}

    def transport(system: str, user: str) -> str:
        captured["system"] = system
        return json.dumps(payload)

    StructuredRoleAgents(StructuredModelClient(transport)).joint_anchor("A", view)

    assert singleton_instruction in captured["system"]
    assert multi_instruction in captured["system"]


def test_joint_singleton_label_instruction_does_not_leak_to_other_roles() -> None:
    captured = _CapturedStage3RolePrompts()

    _run_stage3_role(captured, "symptom_analyst", "A")

    assert "JOINT LABEL OWNERSHIP" not in captured.system_for("A")
    assert "they may therefore be identical" not in captured.system_for("A")


@pytest.mark.parametrize(
    "field",
    ["supporting_evidence_ids", "counter_evidence_ids"],
)
def test_verifier_retries_unknown_exact_view_citations(field: str) -> None:
    invalid = _valid_root_verification_payload()
    invalid[field] = ["unknown-id"]
    agents, prompts = _retrying_agents(invalid, _valid_root_verification_payload())

    result = agents.root_cause_verifier("A", _anchor(), _ase_view())

    assert result.supporting_evidence_ids == ("comment-1",)
    assert len(prompts) == 2
    assert "not present in the exact evidence view" in prompts[1]


def test_symptom_verifier_repairs_defect_mechanism_only_supporting_evidence() -> None:
    invalid = _valid_root_verification_payload()
    invalid.update(
        {
            "dimension": "symptom",
            "anchor_label": "Unexpected Output",
            "supporting_evidence_ids": ["feg-node-test"],
        }
    )
    valid = dict(invalid)
    valid["supporting_evidence_ids"] = ["symptom-node-test"]
    agents, prompts = _retrying_agents(invalid, valid)

    result = agents.symptom_verifier(
        "A",
        _anchor(),
        _ase_capability_split_view(),
    )

    assert result.supporting_evidence_ids == ("symptom-node-test",)
    assert len(prompts) == 2
    assert "does not support the symptom dimension" in prompts[1]


def test_ase_root_verifier_accepts_defect_mechanism_only_supporting_evidence() -> None:
    payload = _valid_root_verification_payload()
    payload["supporting_evidence_ids"] = ["feg-node-test"]
    agents, prompts = _retrying_agents(payload, payload)

    result = agents.root_cause_verifier(
        "A",
        _anchor(),
        _ase_capability_split_view(),
    )

    assert result.supporting_evidence_ids == ("feg-node-test",)
    assert len(prompts) == 1


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("symptom", "supporting_evidence_ids", ["unknown-id"]),
        ("symptom", "counter_evidence_ids", ["unknown-id"]),
        ("symptom", "boundary_evidence_ids", ["unknown-id"]),
        ("root_cause", "supporting_evidence_ids", ["unknown-id"]),
        ("root_cause", "counter_evidence_ids", ["unknown-id"]),
        ("root_cause", "boundary_evidence_ids", ["unknown-id"]),
        ("shared", "shared_supporting_evidence_ids", ["unknown-id"]),
        ("shared", "shared_counter_evidence_ids", ["unknown-id"]),
        ("shared", "shared_boundary_evidence_ids", ["unknown-id"]),
    ],
)
def test_joint_anchor_retries_all_unknown_citations(
    section: str, field: str, value: list[str]
) -> None:
    invalid = _valid_joint_anchor_payload()
    target = invalid[section] if section != "shared" else invalid
    assert isinstance(target, dict)
    target[field] = value
    agents, prompts = _retrying_agents(invalid, _valid_joint_anchor_payload())

    report = agents.joint_anchor("A", _ase_view())

    assert report.shared_supporting_evidence_ids == ("comment-1",)
    assert len(prompts) == 2
    assert "not present in the exact evidence view" in prompts[1]


def test_joint_anchor_repairs_dimension_incapable_positive_citations() -> None:
    invalid = _valid_joint_anchor_payload()
    invalid_symptom = invalid["symptom"]
    assert isinstance(invalid_symptom, dict)
    invalid_symptom["supporting_evidence_ids"] = ["feg-node-test"]
    invalid_symptom["boundary_evidence_ids"] = ["feg-node-test"]

    valid = _valid_joint_anchor_payload()
    valid_symptom = valid["symptom"]
    assert isinstance(valid_symptom, dict)
    valid_symptom["supporting_evidence_ids"] = ["symptom-node-test"]
    valid_symptom["counter_evidence_ids"] = ["symptom-node-test"]
    valid_symptom["boundary_evidence_ids"] = ["symptom-node-test"]
    agents, prompts = _retrying_agents(invalid, valid)

    report = agents.joint_anchor("A", _ase_capability_split_view())

    assert report.symptom.supporting_evidence_ids == ("symptom-node-test",)
    assert len(prompts) == 2
    first_context = json.loads(prompts[0])["context"]
    assert first_context["evidence_eligibility"] == {
        "comment-1": ["root_cause", "symptom"],
        "feg-node-test": ["root_cause"],
        "symptom-node-test": ["root_cause", "symptom"],
    }
    assert "does not support the symptom dimension" in prompts[1]


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("symptom", "label", "API Misuse"),
        ("root_cause", "label", "Unexpected Output"),
    ],
)
def test_joint_anchor_retries_labels_from_the_wrong_taxonomy_dimension(
    section: str, field: str, value: str
) -> None:
    invalid = _valid_joint_anchor_payload()
    target = invalid[section]
    assert isinstance(target, dict)
    target[field] = value
    agents, prompts = _retrying_agents(invalid, _valid_joint_anchor_payload())

    report = agents.joint_anchor("A", _ase_view())

    assert report.symptom.alternative_label == "Crash"
    assert report.root_cause.alternative_label == "Incorrect Code Logic"
    assert len(prompts) == 2


@pytest.mark.parametrize(
    ("section", "label"),
    [
        ("symptom", "Unexpected Output"),
        ("root_cause", "API Misuse"),
    ],
)
def test_joint_anchor_allows_the_only_taxonomy_label_as_its_alternative(
    section: str,
    label: str,
) -> None:
    payload = _valid_joint_anchor_payload()
    classification = payload[section]
    assert isinstance(classification, dict)
    classification["alternative_label"] = label
    taxonomy = {
        "symptom": ["Unexpected Output", "Crash"],
        "root_cause": ["API Misuse", "Incorrect Code Logic"],
    }
    taxonomy[section] = [label]
    view = _ase_view().model_copy(update={"taxonomy": taxonomy})
    agents, calls = _recording_agents(payload)

    report = agents.joint_anchor("A", view)

    assert getattr(report, section).label == label
    assert getattr(report, section).alternative_label == label
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("section", "label", "outside_label"),
    [
        ("symptom", "Unexpected Output", "Crash"),
        ("root_cause", "API Misuse", "Incorrect Code Logic"),
    ],
)
def test_joint_anchor_retries_alternative_outside_singleton_taxonomy(
    section: str,
    label: str,
    outside_label: str,
) -> None:
    invalid = _valid_joint_anchor_payload()
    invalid_classification = invalid[section]
    assert isinstance(invalid_classification, dict)
    invalid_classification["alternative_label"] = outside_label
    valid = _valid_joint_anchor_payload()
    valid_classification = valid[section]
    assert isinstance(valid_classification, dict)
    valid_classification["alternative_label"] = label
    taxonomy = {
        "symptom": ["Unexpected Output", "Crash"],
        "root_cause": ["API Misuse", "Incorrect Code Logic"],
    }
    taxonomy[section] = [label]
    view = _ase_view().model_copy(update={"taxonomy": taxonomy})
    agents, prompts = _retrying_agents(invalid, valid)

    report = agents.joint_anchor("A", view)

    assert getattr(report, section).alternative_label == label
    assert len(prompts) == 2
    assert "must belong to the supplied" in prompts[1]


@pytest.mark.parametrize(
    ("section", "label"),
    [
        ("symptom", "Unexpected Output"),
        ("root_cause", "API Misuse"),
    ],
)
def test_joint_anchor_retries_identical_labels_when_taxonomy_has_alternatives(
    section: str,
    label: str,
) -> None:
    invalid = _valid_joint_anchor_payload()
    classification = invalid[section]
    assert isinstance(classification, dict)
    classification["alternative_label"] = label
    agents, prompts = _retrying_agents(invalid, _valid_joint_anchor_payload())

    report = agents.joint_anchor("A", _ase_view())

    assert report.symptom.alternative_label == "Crash"
    assert report.root_cause.alternative_label == "Incorrect Code Logic"
    assert len(prompts) == 2
    assert "alternative_label must be distinct" in prompts[1]


@pytest.mark.parametrize(
    ("method_name", "dimension", "anchor_label", "invalid_field", "invalid_value"),
    [
        (
            "symptom_verifier",
            EvidenceDimension.SYMPTOM,
            "Unexpected Output",
            "dimension",
            EvidenceDimension.ROOT_CAUSE.value,
        ),
        (
            "root_cause_verifier",
            EvidenceDimension.ROOT_CAUSE,
            "API Misuse",
            "anchor_label",
            "Incorrect Code Logic",
        ),
    ],
)
def test_verifier_retries_wrong_owned_dimension_or_anchor_label(
    method_name: str,
    dimension: EvidenceDimension,
    anchor_label: str,
    invalid_field: str,
    invalid_value: str,
) -> None:
    valid = _valid_root_verification_payload()
    valid["dimension"] = dimension.value
    valid["anchor_label"] = anchor_label
    invalid = dict(valid)
    invalid[invalid_field] = invalid_value
    agents, prompts = _retrying_agents(invalid, valid)

    result = getattr(agents, method_name)("A", _anchor(), _ase_view())

    assert result.dimension is dimension
    assert result.anchor_label == anchor_label
    assert len(prompts) == 2


def test_root_cause_verifier_retries_reject_with_unowned_alternative_label() -> None:
    valid = _valid_root_verification_payload()
    invalid = {
        **valid,
        "verdict": VerificationVerdict.REJECT.value,
        "alternative_label": "Unexpected Output",
        "corrected_claim": "The internal branch misroutes a valid API request.",
        "corrected_causal_chain": [
            "A valid request arrives.",
            "The branch misroutes the request.",
            "The result is unexpectedly returned.",
        ],
    }
    agents, prompts = _retrying_agents(invalid, valid)

    result = agents.root_cause_verifier("A", _anchor(), _ase_view())

    assert result.verdict is VerificationVerdict.ACCEPT
    assert len(prompts) == 2


@pytest.mark.parametrize(
    ("method_name", "dimension", "anchor_label"),
    (
        ("symptom_verifier", "symptom", "Unexpected Output"),
        ("root_cause_verifier", "root_cause", "API Misuse"),
    ),
)
def test_verifier_system_prompt_explains_each_verdict_json_shape(
    method_name: str,
    dimension: str,
    anchor_label: str,
) -> None:
    captured: dict[str, str] = {}
    payload = {
        **_valid_root_verification_payload(),
        "dimension": dimension,
        "anchor_label": anchor_label,
        "verdict": VerificationVerdict.INSUFFICIENT_TO_REJECT.value,
    }

    def transport(system: str, user: str) -> str:
        captured["system"] = system
        return json.dumps(payload)

    getattr(StructuredRoleAgents(StructuredModelClient(transport)), method_name)(
        "A",
        _anchor(),
        _ase_view(),
    )

    prompt = captured["system"]
    assert 'For verdict "accept" or "insufficient_to_reject"' in prompt
    assert '"alternative_label": null' in prompt
    assert '"corrected_claim": null' in prompt
    assert '"corrected_causal_chain": []' in prompt
    assert 'For verdict "reject"' in prompt
    assert "distinct from anchor_label" in prompt
    assert "supporting_evidence_ids must be a non-empty JSON array" in prompt
    assert "corrected_causal_chain must always be a JSON array of strings" in prompt
    assert "never a string or object" in prompt
    if dimension == "root_cause":
        assert "at least 3 JSON strings" in prompt
    else:
        assert "For symptom reject" not in prompt


def test_root_verifier_prompt_allows_five_step_reject_chain() -> None:
    captured: dict[str, str] = {}
    chain = [
        "The request reaches the API boundary.",
        "The caller supplies an incompatible argument.",
        "The API follows the incompatible branch.",
        "The branch produces the wrong internal value.",
        "The caller observes the unexpected output.",
    ]
    payload = {
        **_valid_root_verification_payload(),
        "verdict": VerificationVerdict.REJECT.value,
        "alternative_label": "Incorrect Code Logic",
        "corrected_claim": "An internal branch computes the wrong value for this request.",
        "corrected_causal_chain": chain,
    }

    def transport(system: str, user: str) -> str:
        captured["system"] = system
        return json.dumps(payload)

    result = StructuredRoleAgents(StructuredModelClient(transport)).root_cause_verifier(
        "A", _anchor(), _ase_view()
    )

    assert result.verdict is VerificationVerdict.REJECT
    assert result.corrected_causal_chain == tuple(chain)
    assert "at least 3 JSON strings" in captured["system"]
    assert "exactly 3 or 4" not in captured["system"]


def test_root_verifier_prompt_does_not_forbid_contract_valid_empty_chain_steps() -> (
    None
):
    captured: dict[str, str] = {}
    payload = {
        **_valid_root_verification_payload(),
        "verdict": VerificationVerdict.REJECT.value,
        "alternative_label": "Incorrect Code Logic",
        "corrected_claim": "An internal branch computes the wrong value for this request.",
        "corrected_causal_chain": ["", "", ""],
    }

    def transport(system: str, user: str) -> str:
        captured["system"] = system
        return json.dumps(payload)

    result = StructuredRoleAgents(StructuredModelClient(transport)).root_cause_verifier(
        "A", _anchor(), _ase_view()
    )

    assert result.corrected_causal_chain == ("", "", "")
    assert "non-empty JSON strings" not in captured["system"]
    assert "at least 3 JSON strings" in captured["system"]


def test_symptom_verifier_prompt_does_not_force_empty_reject_chain() -> None:
    captured: dict[str, str] = {}
    payload = {
        **_valid_root_verification_payload(),
        "dimension": EvidenceDimension.SYMPTOM.value,
        "anchor_label": "Unexpected Output",
        "verdict": VerificationVerdict.REJECT.value,
        "alternative_label": "Crash",
        "corrected_claim": "The process terminates instead of returning a result.",
        "corrected_causal_chain": ["The process terminates before producing output."],
    }

    def transport(system: str, user: str) -> str:
        captured["system"] = system
        return json.dumps(payload)

    result = StructuredRoleAgents(StructuredModelClient(transport)).symptom_verifier(
        "A", _anchor(), _ase_view()
    )

    assert result.verdict is VerificationVerdict.REJECT
    assert result.corrected_causal_chain == (
        "The process terminates before producing output.",
    )
    assert "For symptom reject" not in captured["system"]
    assert (
        "corrected_causal_chain must always be a JSON array of strings"
        in captured["system"]
    )


def test_verifier_verdict_shape_guidance_does_not_leak_to_other_roles() -> None:
    captured: dict[str, str] = {}

    def transport(system: str, user: str) -> str:
        captured["system"] = system
        return json.dumps(_valid_joint_anchor_payload())

    StructuredRoleAgents(StructuredModelClient(transport)).joint_anchor(
        "A", _ase_view()
    )

    assert "VERIFIER VERDICT SHAPES" not in captured["system"]


def test_verifier_schema_retry_keeps_verdict_shape_guidance() -> None:
    invalid = {
        **_valid_root_verification_payload(),
        "verdict": "reject",
        "alternative_label": "Incorrect Code Logic",
        "corrected_claim": "The internal branch misroutes the valid request.",
        "corrected_causal_chain": "request -> wrong branch -> wrong result",
    }
    insufficient = {
        **_valid_root_verification_payload(),
        "verdict": VerificationVerdict.INSUFFICIENT_TO_REJECT.value,
    }
    responses = iter((json.dumps(invalid), json.dumps(insufficient)))
    systems: list[str] = []

    def transport(system: str, user: str) -> str:
        systems.append(system)
        return next(responses)

    result = StructuredRoleAgents(
        StructuredModelClient(transport, max_schema_retries=1)
    ).root_cause_verifier("A", _anchor(), _ase_view())

    assert result.verdict is VerificationVerdict.INSUFFICIENT_TO_REJECT
    assert len(systems) == 2
    assert systems[0] == systems[1]
    assert all(
        "corrected_causal_chain must always be a JSON array of strings" in system
        and "at least 3 JSON strings" in system
        for system in systems
    )


def test_verifier_prompt_contains_only_own_anonymized_anchor_context() -> None:
    captured: dict[str, str] = {}

    def transport(system: str, user: str) -> str:
        captured["system"] = system
        captured["user"] = user
        return json.dumps(_valid_root_verification_payload())

    StructuredRoleAgents(StructuredModelClient(transport)).root_cause_verifier(
        "B", _anchor(), _ase_view()
    )

    prompt = (captured["system"] + captured["user"]).lower()
    context = json.loads(captured["user"])["context"]
    assert context["owned_dimension"] == "root_cause"
    assert "team_id" not in context["anchor"]
    assert "falsification-first" in captured["system"].lower()
    assert all(
        prohibited not in prompt
        for prohibited in (
            "peer_report",
            "peer team",
            "gold",
            "development",
            "label frequency",
        )
    )


def test_joint_anchor_team_prompts_remain_isolated_and_use_distinct_perspectives() -> (
    None
):
    captured: dict[str, tuple[str, str]] = {}

    def transport(system: str, user: str) -> str:
        captured[json.loads(user)["team_id"]] = (system, user)
        return json.dumps(_valid_joint_anchor_payload())

    agents = StructuredRoleAgents(StructuredModelClient(transport))
    agents.joint_anchor("A", _ase_view())
    agents.joint_anchor("B", _ase_view())

    assert "evidence-first" in captured["A"][0].lower()
    assert "falsification-first" in captured["B"][0].lower()
    assert (
        json.loads(captured["A"][1])["evidence_view"]
        == json.loads(captured["B"][1])["evidence_view"]
    )


@pytest.mark.parametrize("owner_team_id", ["A", "B"])
def test_revision_cross_uses_owner_task_specific_raw_contract(
    owner_team_id: str,
) -> None:
    view = _stage3_view()
    structure = _baseline_revision_structure()
    proposal = _baseline_revision_proposal(view)
    assessment = _heterogeneous_revision_assessment(owner_team_id, view)
    if owner_team_id == "A":
        raw_cross = EntailmentMappingCrossResult.model_validate(
            {
                "condition_findings": [
                    {
                        "condition": "execution completes slowly",
                        "status": "confirmed",
                        "citation_ids": ["issue-body"],
                    }
                ],
                "baseline_exclusion_finding": {
                    "condition": "execution completes slowly",
                    "status": "confirmed",
                    "citation_ids": ["issue-body"],
                },
                "status": "consistent",
                "rationale": "Every exact entailment mapping is confirmed by its owner citation.",
            }
        )
    else:
        raw_cross = FalsificationCoverageCrossResult.model_validate(
            {
                "baseline_survival_finding": {
                    "condition": "execution terminates before returning",
                    "status": "confirmed",
                    "citation_ids": ["issue-body"],
                },
                "proposed_defeater_findings": [
                    {
                        "condition": "execution terminates before returning",
                        "status": "confirmed",
                        "citation_ids": ["issue-body"],
                    }
                ],
                "strongest_competing_reading_finding": {
                    "label": "Crash",
                    "status": "confirmed",
                    "citation_ids": ["issue-body"],
                },
                "status": "consistent",
                "rationale": "Every exact falsification finding is confirmed by its owner citation.",
            }
        )

    result = StructuredRoleAgents(
        StructuredModelClient(lambda _system, _user: raw_cross.model_dump_json())
    ).baseline_revision_consistency(
        checker_team_id="B" if owner_team_id == "A" else "A",
        assessment_owner_team_id=owner_team_id,
        proposal=proposal,
        assessment=assessment,
        routed_card=structure.boundary_cards[0],
        view=view,
    )

    assert result.status is ConsistencyStatus.CONSISTENT
    assert result.assessment_owner_task == assessment.assessor_task
    assert result.assessment_raw_result_digest == assessment.raw_result_digest


@pytest.mark.parametrize("owner_team_id", ["A", "B"])
def test_revision_cross_rejects_swapped_task_schema(owner_team_id: str) -> None:
    view = _stage3_view()
    proposal = _baseline_revision_proposal(view)
    assessment = _heterogeneous_revision_assessment(owner_team_id, view)
    swapped = _task_specific_cross_raw("B" if owner_team_id == "A" else "A")

    with pytest.raises(StructuredOutputError):
        StructuredRoleAgents(
            StructuredModelClient(
                lambda _system, _user: swapped.model_dump_json(),
                max_schema_retries=0,
            )
        ).baseline_revision_consistency(
            checker_team_id="B" if owner_team_id == "A" else "A",
            assessment_owner_team_id=owner_team_id,
            proposal=proposal,
            assessment=assessment,
            routed_card=_baseline_revision_structure().boundary_cards[0],
            view=view,
        )


@pytest.mark.parametrize(
    ("owner_team_id", "missing_field"),
    [
        ("A", "condition_findings"),
        ("A", "baseline_exclusion_finding"),
        ("B", "baseline_survival_finding"),
        ("B", "proposed_defeater_findings"),
        ("B", "strongest_competing_reading_finding"),
    ],
)
def test_revision_cross_rejects_missing_task_specific_finding(
    owner_team_id: str,
    missing_field: str,
) -> None:
    view = _stage3_view()
    proposal = _baseline_revision_proposal(view)
    assessment = _heterogeneous_revision_assessment(owner_team_id, view)
    payload = _task_specific_cross_raw(owner_team_id).model_dump(mode="json")
    payload.pop(missing_field)

    with pytest.raises(StructuredOutputError):
        StructuredRoleAgents(
            StructuredModelClient(
                lambda _system, _user: json.dumps(payload), max_schema_retries=0
            )
        ).baseline_revision_consistency(
            checker_team_id="B" if owner_team_id == "A" else "A",
            assessment_owner_team_id=owner_team_id,
            proposal=proposal,
            assessment=assessment,
            routed_card=_baseline_revision_structure().boundary_cards[0],
            view=view,
        )
