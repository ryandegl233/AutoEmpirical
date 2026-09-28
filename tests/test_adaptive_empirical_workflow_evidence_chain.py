import hashlib
import json

import pytest

from Benchmark.src.adaptive_empirical_workflow.contracts import EvidenceItem, EvidenceView, JointAnchorReport
from Benchmark.src.ase2022_llm_baseline import ROOT_CAUSE_DEFINITIONS, SYMPTOM_DEFINITIONS


@pytest.mark.parametrize("mode", ["S0", "A-v1", "A", "B", "AB"])
def test_comment_policy_reaches_all_stage3_roles_without_changing_controls(mode):
    from Benchmark.src.adaptive_empirical_workflow.agents import StructuredModelClient, StructuredRoleAgents
    from Benchmark.src.adaptive_empirical_workflow.capabilities import AnalystRole
    from Benchmark.src.adaptive_empirical_workflow.explainable_tools import ExplainableTools
    from Benchmark.src.adaptive_empirical_workflow.formal_reasoning import FormalExplainableTools
    from Benchmark.src.adaptive_empirical_workflow.contracts import Stage3ArbitrationPacket
    view, anchor = sample()
    systems = []
    class Captured(Exception):
        pass
    def transport(system, user, *, options):
        systems.append(system)
        if options.role == "explainable_tool_query":
            assert ("ROOT-CAUSE DECISION POLICY v3" in system) == (mode in {"A", "AB"})
            return "{}"
        raise Captured()
    tool = FormalExplainableTools() if mode in {"A", "AB"} else ExplainableTools() if mode == "A-v1" else None
    agents = StructuredRoleAgents(StructuredModelClient(transport), explainable_tools=tool,
                                 chain_checker=mode in {"B", "AB"})
    active = mode in {"A", "B", "AB"}
    for role in (AnalystRole.JOINT_ANCHOR, AnalystRole.ROOT_CAUSE_ANALYST,
                 AnalystRole.CAUSAL_CONSISTENCY_CHECKER, AnalystRole.BOUNDARY_CHALLENGER):
        with pytest.raises(Captured):
            agents._role(role, team_id="A", view=view, schema=JointAnchorReport)
        assert ("ROOT-CAUSE DECISION POLICY v3" in systems[-1]) == active
    # Use the existing complete arbitration fixture to exercise the actual prompt path.
    from tests.test_adaptive_empirical_workflow_agents import _stage3_team
    from Benchmark.src.adaptive_empirical_workflow.contracts import AnonymousStage3TeamReport, DisagreementMap
    team = _stage3_team()
    anonymous = AnonymousStage3TeamReport(symptom=team.symptom, root_cause=team.root_cause, consistency=team.consistency)
    packet = Stage3ArbitrationPacket(domain_profile="ase2022", taxonomy=view.taxonomy,
        disagreement=DisagreementMap(dimensions=["root_cause_label"], details={}, requires_arbitration=True),
        team_a=anonymous, team_b=anonymous, classification_ledger_version=1, relevant_evidence=view.items)
    with pytest.raises(Captured):
        agents.stage3_arbitrator(packet)
    assert ("ROOT-CAUSE DECISION POLICY v3" in systems[-1]) == active
    assert "Preserve each unanimous candidate label" in systems[-1]
    if mode in {"B", "AB"}:
        with pytest.raises(Captured):
            agents.root_cause_verifier("A", anchor, view)
        assert "ROOT-CAUSE DECISION POLICY v3" in systems[-1]
    with pytest.raises(Captured):
        agents._role(AnalystRole.FAULT_EVIDENCE_ANALYST, team_id="A",
                     view=view.model_copy(update={"task": "stage2"}), schema=JointAnchorReport)
    assert "ROOT-CAUSE DECISION POLICY v3" not in systems[-1]


def sample():
    text = "The output differs. Have you tried another decoder? STATE: open"
    view = EvidenceView(record_id="r", task="stage3", domain_profile="ase2022", ledger_version=1,
                        taxonomy={"symptom": list(SYMPTOM_DEFINITIONS), "root_cause": list(ROOT_CAUSE_DEFINITIONS)},
                        items=(EvidenceItem(evidence_id="e", record_id="r", source_type="issue_body",
                                            source_uri="https://example.test/issues/1", retrieved_at="snapshot",
                                            content=text, content_sha256=hashlib.sha256(text.encode()).hexdigest(),
                                            explicitness="direct"),))
    common = {"supporting_evidence_ids": ["e"], "boundary_evidence_ids": ["e"],
              "boundary_reason": "The observed output differs without an execution crash.",
              "confidence": 0.9, "evidence_sufficiency": "sufficient"}
    anchor = JointAnchorReport(
        symptom={**common, "label": "Incorrect Functionality", "alternative_label": "Crash",
                 "behavior_claim": "The application returns different output values."},
        root_cause={**common, "label": "API Misuse", "alternative_label": "Incorrect Code Logic",
                    "defect_mechanism": "The custom decoder violates the API input contract.",
                    "causal_chain": ["The input is decoded using a custom decoder.",
                                     "The decoder violates the required API contract.",
                                     "The output differs because that contract was violated."]},
        causal_account="The application returns different output values because of API misuse.",
        shared_supporting_evidence_ids=["e"],
    )
    return view, anchor


def finding(target, status="entailed", quote="The output differs.", kind="observation"):
    return {"target_id": target, "status": status, "rule_id": "source_entailment",
            "citations": [{"evidence_id": "e", "quote": quote, "kind": kind}],
            "rationale": "This finding must be checked against the cited current evidence.",
            "missing_evidence": "The required API contract has not been supplied." if status == "unknown" else ""}


def test_checker_requires_complete_claim_and_link_coverage_and_real_quotes():
    from Benchmark.src.adaptive_empirical_workflow.evidence_chain import ChainAudit, check_targets, validate_chain_audit
    view, anchor = sample()
    targets = check_targets(anchor.root_cause, "root_cause", anchor.causal_account)
    assert "link:0" in targets and "label_link" in targets and "joint_account" in targets
    audit = ChainAudit(action="unresolved", findings=[finding(k, "unknown") for k in targets], confidence=0.5)
    validate_chain_audit(audit, anchor, "root_cause", view)
    with pytest.raises(ValueError, match="coverage"):
        validate_chain_audit(audit.model_copy(update={"findings": audit.findings[:-1]}), anchor, "root_cause", view)
    with pytest.raises(ValueError, match="quote"):
        invalid = audit.model_dump(mode="json")
        invalid["findings"][0]["citations"][0]["quote"] = "fabricated source"
        validate_chain_audit(ChainAudit.model_validate(invalid), anchor, "root_cause", view)
    with pytest.raises(ValueError, match="pass"):
        validate_chain_audit(audit.model_copy(update={"action": "pass"}), anchor, "root_cause", view)


def test_checker_rejects_workflow_status_as_entailment():
    from Benchmark.src.adaptive_empirical_workflow.evidence_chain import ChainAudit, check_targets, validate_chain_audit
    view, anchor = sample()
    targets = check_targets(anchor.root_cause, "root_cause", anchor.causal_account)
    audit = ChainAudit(action="unresolved", findings=[
        finding(k, "entailed" if k == "label" else "unknown", "STATE: open", "workflow_state") for k in targets
    ], confidence=0.5)
    with pytest.raises(ValueError, match="workflow"):
        validate_chain_audit(audit, anchor, "root_cause", view)


def test_formal_a_does_not_accept_unknown_as_a_submitted_fact():
    from Benchmark.src.adaptive_empirical_workflow.formal_reasoning import FormalQuery, FormalExplainableTools
    view, _ = sample()
    query = FormalQuery.model_validate({
        "atoms": [{"atom_id": "o1", "kind": "workflow_state", "statement": "The issue is open.",
                   "citations": [{"evidence_id": "e", "quote": "STATE: open", "kind": "workflow_state"}]}],
        "candidates": [{"dimension": "root_cause", "label": "Unknown", "premise_ids": ["o1"],
                        "relation": "definition_match", "assessment": "supported",
                        "reason": "The issue status is open.", "missing_evidence": ""}],
    })
    with pytest.raises(ValueError, match="Unknown"):
        FormalExplainableTools().evaluate(view, query)
    candidate = query.candidates[0].model_copy(update={"label": "API Misuse"})
    with pytest.raises(ValueError, match="workflow"):
        FormalExplainableTools().evaluate(view, query.model_copy(update={"candidates": (candidate,)}))


def test_formal_a_keeps_reported_causes_distinct_from_observed_facts():
    from Benchmark.src.adaptive_empirical_workflow.formal_reasoning import FormalQuery, FormalExplainableTools
    view, _ = sample()
    query = FormalQuery.model_validate({
        "atoms": [{"atom_id": "s1", "kind": "suggestion", "statement": "Another decoder is suggested.",
                   "citations": [{"evidence_id": "e", "quote": "Have you tried another decoder?", "kind": "suggestion"}]}],
        "candidates": [{"dimension": "root_cause", "label": "API Misuse", "premise_ids": ["s1"],
                        "relation": "causal_link", "assessment": "unknown",
                        "reason": "A suggestion does not establish a contract violation.",
                        "missing_evidence": "The actual API contract and evidence of its violation."}],
    })
    result = FormalExplainableTools().run(view, query)
    assert result["candidate_checks"][0]["state"] == "unknown"
    assert result["candidate_checks"][0]["missing_evidence"]
    assert "cause:Unknown" not in json.dumps(result)
    candidate = query.candidates[0].model_copy(update={"assessment": "supported", "missing_evidence": ""})
    with pytest.raises(ValueError, match="suggestion"):
        FormalExplainableTools().evaluate(view, query.model_copy(update={"candidates": (candidate,)}))


def test_checker_rewrite_is_applied_and_survives_serialization():
    from Benchmark.src.adaptive_empirical_workflow.evidence_chain import ChainAudit, CandidateRevision, check_targets
    from Benchmark.src.adaptive_empirical_workflow.agents import StructuredModelClient, StructuredRoleAgents
    from Benchmark.src.adaptive_empirical_workflow.contracts import (ChainVerificationReport, DimensionVerificationReport,
                                                                   TeamComposedCandidate, Stage3TeamReport)
    from Benchmark.src.adaptive_empirical_workflow.stage3_composition import (compose_stage3_team_candidate,
                                                                           stage3_correction_semantic_errors)
    view, anchor = sample()
    revision = CandidateRevision(label=anchor.root_cause.label,
        claim="The source reports differing output; the exact mechanism is not established.",
        causal_chain=["The source reports differing output values.", "A different decoder is suggested for investigation.",
                      "The suggestion does not demonstrate a violated API contract."],
        alternative_label="Incorrect Code Logic", boundary_reason="The explanation now preserves the stated uncertainty.",
        citations=[{"evidence_id": "e", "quote": "The output differs.", "kind": "observation"}],
        unresolved_evidence_gaps=["Actual API contract and evidence of a violation."])
    # This fixture tests the adapter, not semantic accuracy of a model judgment.
    audit = ChainAudit(action="rewrite", confidence=0.6, revision=revision,
        findings=[finding(k, "entailed" if k == "label" else "unknown")
                  for k in check_targets(anchor.root_cause, "root_cause", anchor.causal_account)],
        revision_findings=[finding(k) for k in check_targets(revision, "root_cause")])
    calls = []
    def transport(system, user, *, options):
        calls.append(options.role)
        context = json.loads(user)["context"]
        assert "anchor" not in context and "confidence" not in json.dumps(context)
        assert "module_a" not in context
        assert "FORMAL EVIDENCE POLICY" in system
        return audit.model_dump_json()
    agents = StructuredRoleAgents(StructuredModelClient(transport), chain_checker=True)
    root_review = agents.root_cause_verifier("A", anchor, view)
    assert calls == ["root_cause_chain_checker"]
    symptom_review = DimensionVerificationReport(dimension="symptom", verdict="accept",
        anchor_label=anchor.symptom.label, rationale="The observable output claim is supported.", confidence=0.8)
    composed = compose_stage3_team_candidate("A", anchor, symptom_review, root_review, view)
    replay = TeamComposedCandidate.model_validate_json(composed.model_dump_json())
    assert isinstance(replay.verifications[1], ChainVerificationReport)
    assert replay.root_cause.label == anchor.root_cause.label
    assert replay.root_cause.defect_mechanism == revision.claim
    assert replay.correction_audit[1].accepted
    assert replay.root_cause.unresolved_evidence_gaps == revision.unresolved_evidence_gaps
    report = Stage3TeamReport(
        team_id="A", symptom=replay.symptom, root_cause=replay.root_cause, anchor=anchor,
        verifications=replay.verifications, correction_audit=replay.correction_audit,
        consistency={"status": "consistent", "rationale": "Fixture used to validate correction serialization and replay.",
                     "supporting_evidence_ids": ["e"]})
    report = Stage3TeamReport.model_validate_json(report.model_dump_json())
    assert isinstance(report.verifications[1], ChainVerificationReport)
    assert stage3_correction_semantic_errors(report) == ()
    assert agents.telemetry()["module_b_calls"][0]["audit"]["action"] == "rewrite"


def test_checker_unresolved_is_logged_without_legacy_fallback():
    from Benchmark.src.adaptive_empirical_workflow.evidence_chain import ChainAudit, CheckerUnresolved, check_targets
    from Benchmark.src.adaptive_empirical_workflow.agents import StructuredModelClient, StructuredRoleAgents
    view, anchor = sample()
    audit = ChainAudit(action="unresolved", confidence=0.3, findings=[finding(k, "unknown")
        for k in check_targets(anchor.root_cause, "root_cause", anchor.causal_account)])
    calls = []
    def transport(system, user, *, options):
        calls.append(options.role)
        return audit.model_dump_json()
    agents = StructuredRoleAgents(StructuredModelClient(transport), chain_checker=True)
    with pytest.raises(CheckerUnresolved):
        agents.root_cause_verifier("A", anchor, view)
    assert calls == ["root_cause_chain_checker"]
    assert agents.telemetry()["module_b_calls"][0]["audit"]["action"] == "unresolved"


def test_formal_reported_support_and_unknown_revision_guard():
    from Benchmark.src.adaptive_empirical_workflow.formal_reasoning import FormalQuery, FormalExplainableTools
    from Benchmark.src.adaptive_empirical_workflow.evidence_chain import ChainAudit, check_targets, validate_chain_audit
    view, anchor = sample()
    query = FormalQuery.model_validate({"atoms": [{"atom_id": "a1", "kind": "source_assertion",
        "statement": "The source reports differing output.", "citations": [{"evidence_id": "e",
        "quote": "The output differs.", "kind": "source_assertion"}]}], "candidates": [{
        "dimension": "symptom", "label": "Incorrect Functionality", "premise_ids": ["a1"],
        "relation": "definition_match", "assessment": "supported", "reason": "The stated observation matches differing output."}]})
    assert FormalExplainableTools().evaluate(view, query)["candidate_checks"][0]["state"] == "reported_support"
    unknown = anchor.model_copy(update={"root_cause": anchor.root_cause.model_copy(update={"label": "Unknown"})})
    audit = ChainAudit(action="pass", confidence=0.5, findings=[finding(k)
        for k in check_targets(unknown.root_cause, "root_cause", unknown.causal_account)])
    with pytest.raises(ValueError, match="comparisons"):
        validate_chain_audit(audit, unknown, "root_cause", view)


def test_checker_candidate_gaps_are_explicit_but_disproof_is_distinct():
    from Benchmark.src.adaptive_empirical_workflow.evidence_chain import ChainAudit, CandidateReview, check_targets, validate_chain_audit
    view, anchor = sample()
    candidate = dict(label="Crash", status="insufficient", citations=[
        {"evidence_id": "e", "quote": "The output differs.", "kind": "observation"}],
        rationale="This fixture distinguishes missing evidence from a claimed contradiction.")
    with pytest.raises(ValueError, match="missing_evidence"):
        CandidateReview.model_validate(candidate)
    audit = ChainAudit(action="unresolved", confidence=0.5, findings=[finding(k, "unknown")
        for k in check_targets(anchor.symptom, "symptom", anchor.causal_account)],
        candidates=[{**candidate, "missing_evidence": ""}])
    with pytest.raises(ValueError, match="missing discriminating"):
        validate_chain_audit(audit, anchor, "symptom", view)
    contradicted = CandidateReview(**{**candidate, "status": "contradicted", "missing_evidence": ""})
    validate_chain_audit(audit.model_copy(update={"candidates": (contradicted,)}), anchor, "symptom", view)


def test_checker_relabel_applies_only_a_fully_audited_revision():
    from Benchmark.src.adaptive_empirical_workflow.evidence_chain import ChainAudit, CandidateRevision, check_targets, validate_chain_audit, verification_from_audit
    from Benchmark.src.adaptive_empirical_workflow.stage3_composition import _correct_symptom
    view, anchor = sample()
    wrong = anchor.model_copy(update={"symptom": anchor.symptom.model_copy(update={"label": "Crash"})})
    revision = CandidateRevision(label="Incorrect Functionality", claim="The application returns differing output values.",
        alternative_label="Crash", boundary_reason="The reported behavior is a difference in output values.",
        citations=[{"evidence_id": "e", "quote": "The output differs.", "kind": "observation"}])
    audit = ChainAudit(action="relabel", confidence=0.6, revision=revision,
        findings=[finding(k, "unknown") for k in check_targets(wrong.symptom, "symptom", wrong.causal_account)],
        revision_findings=[finding(k) for k in check_targets(revision, "symptom")])
    validate_chain_audit(audit, wrong, "symptom", view)
    review = verification_from_audit(audit, wrong, "symptom")
    final = _correct_symptom(wrong.symptom, review, review.supporting_evidence_ids)
    assert final.label == revision.label and final.behavior_claim == revision.claim
    with pytest.raises(ValueError, match="coverage"):
        validate_chain_audit(audit.model_copy(update={"revision_findings": audit.revision_findings[:-1]}), wrong, "symptom", view)


def test_checker_output_schema_excludes_unknown_and_other_dimension_candidates():
    from Benchmark.src.adaptive_empirical_workflow.evidence_chain import checker_output_schema, check_targets
    view, anchor = sample()
    schema = checker_output_schema(view, "root_cause")
    payload = dict(action="unresolved", confidence=0.5, findings=[finding(k, "unknown")
        for k in check_targets(anchor.root_cause, "root_cause", anchor.causal_account)],
        candidates=[dict(label="Unknown", status="insufficient", citations=[],
                        rationale="Specific causes cannot yet be distinguished.", missing_evidence="The actual API contract.")])
    for invalid in ("Unknown", "Crash"):
        payload["candidates"][0]["label"] = invalid
        with pytest.raises(ValueError, match="candidates.0.label"):
            schema.model_validate(payload)
    payload["candidates"][0]["label"] = "API Misuse"
    assert schema.model_validate(payload).candidates[0].label == "API Misuse"


def test_checker_retains_failed_raw_output_and_specific_error_per_attempt():
    from Benchmark.src.adaptive_empirical_workflow.evidence_chain import check_targets
    from Benchmark.src.adaptive_empirical_workflow.agents import StructuredModelClient, StructuredRoleAgents
    view, anchor = sample()
    payload = dict(action="pass", confidence=0.5, findings=[finding(k)
        for k in check_targets(anchor.symptom, "symptom", anchor.causal_account)])
    bad = json.loads(json.dumps(payload)); bad['findings'][0]['citations'][0]['quote'] = 'invented quote'
    replies = iter([json.dumps(bad), json.dumps(payload)])
    agents = StructuredRoleAgents(StructuredModelClient(lambda system, user, *, options: next(replies), max_schema_retries=1,
                                                       retry_delay_seconds=0), chain_checker=True)
    agents.symptom_verifier("A", anchor, view)
    attempts = agents.telemetry(record_id="r")["module_b_attempts"]
    assert len(attempts) == 2
    assert [entry["status"] for entry in attempts] == ["invalid", "valid"]
    assert "invented quote" in attempts[0]["raw_response"]
    assert "quote" in attempts[0]["error"] and "label" in attempts[0]["error"]
    assert all(entry["record_id"] == "r" and entry["team_id"] == "A" for entry in attempts)
