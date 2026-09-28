"""Control-flow checks use synthetic evidence, not evaluation labels."""
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from Benchmark.src.adaptive_empirical_workflow.global_supervisor import (
    SupervisorDecision, run_supervision, validate_decision,
)
from Benchmark.src.adaptive_empirical_workflow.contracts import BoundaryChallenge
from tests.test_adaptive_empirical_workflow_agents import _stage3_view, _stage3_team


def teams():
    a = _stage3_team()
    return a, a.model_copy(update={"team_id": "B"})


def decision(view, action, targets=(), missing=False, reports=None):
    reports = reports or teams()
    return SupervisorDecision.model_validate(dict(
        facts=[dict(fact_id="f", kind="observation", evidence_id=view.items[0].evidence_id,
            quote=view.items[0].content, interpretation="Synthetic observed event.")],
        links=[dict(team=t, dimension=d, candidate_label=getattr(next(r for r in reports if r.team_id == t),d).label, premise_ids=["f"], conclusion="Synthetic candidate claim",
            warrant="Synthetic test warrant from the cited event to the current candidate.",
            status="missing" if missing else "supported")
            for t in ("A", "B") for d in ("symptom", "root_cause")],
        disputes=[], challenger_addresses_disputes=True, challenger_assessment="Inspect the specified causal premise.",
        action=action, targets=targets, instruction="Check the causal premise using only the cited original source.",
        missing_facts=["The source does not determine the causal distinction."] if action == "evidence_gap" else [],
    ))


def test_controller_reworks_only_target_and_rechallenges_with_retained_history():
    view = _stage3_view()
    script = iter([("rework_teams", ("B",)), ("challenge", ()), ("challenge", ()), ("arbitrate", ())])
    rebuilt, challenged, states = [], [], []

    def review(v, state, reports, actions):
        states.append(json.loads(json.dumps(state)))
        action, targets = next(script)
        return decision(v, action, targets)

    def rebuild(t, v, feedback):
        rebuilt.append((t, feedback["instruction"]))
        return _stage3_team().model_copy(update={"team_id": t})

    def challenge(reports, v, feedback):
        challenged.append(feedback)
        return BoundaryChallenge(action="pass", rationale="The specific premise was checked.", cited_evidence_ids=[v.items[0].evidence_id])

    out = run_supervision(view, teams(), review=review, rebuild_team=rebuild, challenge=challenge)
    assert not out.failed and out.stop_reason == "arbitrate"
    assert [t for t, _ in rebuilt] == ["B"]
    assert len(challenged) == 2
    assert states[1]["remaining_team_reworks"] == {"A": 1, "B": 0}
    assert states[1]["history"][1]["kind"] == "team_rework"
    assert "rework_teams" not in states[-1]["available_actions"]
    assert len([e for e in out.events if e["kind"] == "review"]) == 4


def test_budget_rejects_repeated_rework_and_failure_is_not_pass():
    view = _stage3_view()
    out = run_supervision(view, teams(),
        review=lambda v, *_: decision(v, "rework_teams", ("A",)),
        rebuild_team=lambda t, *_: _stage3_team(), challenge=lambda *_: pytest.fail("not reached"))
    assert out.failed and out.stop_reason == "supervisor_component_failure"
    assert sum(e["kind"] == "team_rework" for e in out.events) == 1
    assert "budget" in out.events[-1]["message"]


def test_terminal_requires_challenger_and_missing_links_cannot_pass():
    view = _stage3_view()
    with pytest.raises(ValueError, match="unavailable"):
        validate_decision(decision(view, "arbitrate"), view, teams(), ["challenge"])
    with pytest.raises(ValueError, match="gaps"):
        validate_decision(decision(view, "arbitrate", missing=True), view, teams(), ["arbitrate"])
    validate_decision(decision(view, "evidence_gap", missing=True), view, teams(), ["evidence_gap"])


def test_exact_quotes_real_disputes_and_hypothesis_boundaries():
    view = _stage3_view()
    good = decision(view, "challenge")
    bad = good.model_copy(update={"facts": (good.facts[0].model_copy(update={"quote": "invented evidence"}),)})
    with pytest.raises(ValueError, match="exact"):
        validate_decision(bad, view, teams(), ["challenge"])
    bad = good.model_copy(update={"facts": (good.facts[0].model_copy(update={"kind": "hypothesis"}),)})
    with pytest.raises(ValueError, match="non-hypothetical"):
        validate_decision(bad, view, teams(), ["challenge"])
    a, b = teams()
    b = b.model_copy(update={"root_cause": b.root_cause.model_copy(update={"label": "API Misuse"})})
    with pytest.raises(ValueError, match="actual differing"):
        validate_decision(decision(view, "challenge", reports=(a,b)), view, (a, b), ["challenge"])


def test_case_states_are_isolated_under_concurrency():
    def run(record_id):
        view = _stage3_view().model_copy(update={"record_id": record_id})
        seen = []
        def review(v, state, *_):
            seen.append(len(state["history"]))
            return decision(v, "challenge" if not state["history"] else "arbitrate")
        out = run_supervision(view, teams(), review=review, rebuild_team=lambda *_: pytest.fail("no rework"),
            challenge=lambda *_: BoundaryChallenge(action="pass", rationale=record_id))
        return seen, out
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(run, ["case-one", "case-two"]))
    assert all(seen == [0, 2] and not out.failed for seen, out in results)
    assert results[0][1].challenge.rationale == "case-one"
    assert results[1][1].challenge.rationale == "case-two"


def test_role_transport_temperature_and_targeted_challenger_prompt():
    from Benchmark.src.adaptive_empirical_workflow.agents import StructuredModelClient, StructuredRoleAgents
    view = _stage3_view()
    calls = []
    def transport(system, user, *, options):
        calls.append((options, user))
        if options.role == "global_supervisor":
            state = json.loads(user)["context"]
            return decision(view, "challenge" if not state["history"] else "arbitrate").model_dump_json()
        assert options.role == "boundary_challenger"
        assert "supervisor_feedback" in user
        return BoundaryChallenge(action="pass", rationale="The requested premise is supported.", cited_evidence_ids=["issue-body"]).model_dump_json()
    agents = StructuredRoleAgents(StructuredModelClient(transport, max_schema_retries=0), global_supervisor=True)
    out = agents.supervise_framework(teams(), view)
    assert not out.failed
    assert [(o.role, o.temperature) for o, _ in calls] == [
        ("global_supervisor", .3), ("boundary_challenger", 0), ("global_supervisor", .3)]
    assert agents._supervisor_feedback.get() is None
    assert [c["temperature"] for c in agents.telemetry()["calls"]] == [.3, 0, .3]


def test_rework_feedback_reaches_v4_prequery_and_is_reset():
    from Benchmark.src.adaptive_empirical_workflow.agents import StructuredModelClient, StructuredRoleAgents
    from Benchmark.src.adaptive_empirical_workflow.a_decision import RuleDecisionTools
    from tests.test_adaptive_empirical_workflow_evidence_chain import sample
    from tests.test_adaptive_empirical_workflow_a_decision import decision_query
    view, anchor = sample()
    calls = []
    def transport(system, user, *, options):
        calls.append((options.role, json.loads(user)))
        return decision_query().model_dump_json() if options.role == "explainable_tool_query" else anchor.model_dump_json()
    agents = StructuredRoleAgents(StructuredModelClient(transport, max_schema_retries=0), explainable_tools=RuleDecisionTools())
    token = agents._supervisor_feedback.set({"instruction": "Re-examine the usage requirement from the original evidence."})
    try:
        agents.joint_anchor("A", view)
    finally:
        agents._supervisor_feedback.reset(token)
    agents.joint_anchor("B", view)
    assert [r for r, _ in calls] == ["explainable_tool_query", "joint_anchor"] * 2
    assert all("supervisor_feedback" in p["context"] for _, p in calls[:2])
    assert all("supervisor_feedback" not in p["context"] for _, p in calls[2:])


def test_pooled_gemini_wire_payload_uses_role_temperature(monkeypatch):
    from Benchmark.src import ase2022_llm_baseline as baseline
    sent = []
    class Response:
        status = 200
        def read(self):
            return b'{"choices":[{"message":{"content":"{}"}}]}'
    class Connection:
        def __init__(self, *args, **kwargs): pass
        def request(self, method, path, body, headers): sent.append(json.loads(body))
        def getresponse(self): return Response()
        def close(self): pass
    monkeypatch.setattr(baseline.http.client, "HTTPSConnection", Connection)
    pool = baseline.PooledChatCompletionClient(base_url="https://example.test/v1", max_connections=1)
    for temp in (.3, 0):
        pool.call_model(system_prompt="audit", user_prompt="source", model="test", api_key="test",
                        wire_api="chat", temperature=temp)
    pool.close()
    assert [p["temperature"] for p in sent] == [.3, 0]


def test_supervisor_rejects_answer_metadata_before_transport():
    from Benchmark.src.adaptive_empirical_workflow.agents import StructuredModelClient, StructuredRoleAgents
    view = _stage3_view()
    bad = view.items[0].model_copy(update={"metadata": {"ground_truth": "secret"}})
    view = view.model_copy(update={"items": (bad,)})
    def transport(system, user, *, options): pytest.fail("No transport call permitted")
    out = StructuredRoleAgents(StructuredModelClient(transport), global_supervisor=True).supervise_framework(teams(), view)
    assert out.failed and "forbidden" in out.events[-1]["message"]


def test_supervised_controller_delivers_causal_agenda_to_arbitration_and_audit():
    from Benchmark.src.adaptive_empirical_workflow.controller import Stage3Controller
    from Benchmark.src.adaptive_empirical_workflow.contracts import Stage3ArbitrationDecision
    from Benchmark.src.adaptive_empirical_workflow.global_supervisor import pending_consistency, Dispute
    from tests.test_adaptive_empirical_workflow_stage3_controller import _ledger, _ready, _symptom, _root
    packets = []

    def supervise(reports, view):
        assert all(r.consistency.status.value == "cause_review" for r in reports)
        def review(v, state, current, actions):
            d = decision(v, "challenge" if not state["history"] else "arbitrate", reports=current)
            return d.model_copy(update={"disputes": (Dispute(dimension="root_cause",
                candidates=("missing_null_check", "incorrect_condition"),
                question="Does the evidence identify a missing guard or a wrong condition?",
                evidence_ids=("code-1",), status="resolved", explanation="Synthetic comparison of the current alternatives."),)})
        return run_supervision(view, reports, review=review, rebuild_team=lambda *_: pytest.fail("no rework"),
            challenge=lambda *_: BoundaryChallenge(action="pass", rationale="The actual alternatives were checked.", cited_evidence_ids=["code-1"]))

    def arbitrate(packet):
        packets.append(packet)
        return Stage3ArbitrationDecision(resolution_status="unresolved",
            rationale="Synthetic arbitration preserves candidates with uncertainty.",
            unresolved_dimensions=list(packet.disagreement.dimensions), missing_facts=["Synthetic unresolved distinction."])

    result = Stage3Controller(readiness=_ready,
        symptom_analyst=lambda *_: _symptom(),
        root_cause_analyst=lambda t, _: _root("missing_null_check" if t == "A" else "incorrect_condition"),
        consistency_checker=pending_consistency, global_supervisor=supervise,
        boundary_challenger=lambda *_: pytest.fail("old endpoint challenger must not run"),
        arbitrator=arbitrate).run(_ledger())
    assert len(packets) == 1
    assert packets[0].supervision["stop_reason"] == "arbitrate"
    assert result.supervision["events"][-1]["decision"]["disputes"][0]["candidates"] == ["missing_null_check", "incorrect_condition"]
    assert result.final_decision is not None
    assert result.verification.valid


def test_supervised_controller_does_not_bypass_failed_audit():
    from Benchmark.src.adaptive_empirical_workflow.controller import Stage3Controller
    from Benchmark.src.adaptive_empirical_workflow.global_supervisor import pending_consistency, SupervisionOutcome
    from tests.test_adaptive_empirical_workflow_stage3_controller import _ledger, _ready, _symptom, _root
    result = Stage3Controller(readiness=_ready, symptom_analyst=lambda *_: _symptom(),
        root_cause_analyst=lambda *_: _root(), consistency_checker=pending_consistency,
        global_supervisor=lambda reports, _: SupervisionOutcome(reports, failed=True, stop_reason="simulated_failure"),
        arbitrator=lambda *_: pytest.fail("No unaudited arbitration")).run(_ledger())
    assert result.final_decision is None
    assert result.unresolved.stop_reason == "simulated_failure"
    assert result.supervision["failed"]
