"""Offline regressions for transport, truncation budgets and corrective feedback."""
import json
import ssl
import urllib.error

import pytest

from Benchmark.src import ase2022_llm_baseline as transport
from Benchmark.src.adaptive_empirical_workflow.agents import StructuredModelClient, StructuredRoleAgents
from Benchmark.src.adaptive_empirical_workflow.a_decision import RuleDecisionTools
from Benchmark.src.adaptive_empirical_workflow.capabilities import AnalystRole
from Benchmark.src.adaptive_empirical_workflow.contracts import (
    CausalConsistencyReport, JointAnchorReport, RootCauseReport, Stage3TeamReport, SymptomReport,
)
from tests.test_adaptive_empirical_workflow_agents import _view
from tests.test_adaptive_empirical_workflow_a_decision import decision_query
from tests.test_adaptive_empirical_workflow_evidence_chain import sample


def test_ssl_eof_is_retried_but_certificate_errors_are_not():
    attempts = []
    def call():
        attempts.append(1)
        if len(attempts) == 1:
            raise ssl.SSLEOFError(8, 'unexpected EOF')
        return 'ok'
    assert transport.call_model_with_retries(max_retries=1, retry_delay_seconds=0, model_call=call) == ('ok', 2)
    assert transport.model_transport_error_details(ssl.SSLCertVerificationError(1, 'bad cert'))['retryable'] is False


def test_retry_after_is_respected_even_without_backoff(monkeypatch):
    delays, attempts = [], []
    monkeypatch.setattr(transport.time, 'sleep', delays.append)
    def call():
        attempts.append(1)
        if len(attempts) == 1:
            raise urllib.error.HTTPError('https://example.test', 429, 'limited', {'Retry-After': '25'}, None)
        return 'ok'
    assert transport.call_model_with_retries(max_retries=1, retry_delay_seconds=0, model_call=call) == ('ok', 2)
    assert delays == [25.0]


@pytest.mark.parametrize('bad_id', ['missing-source', 'unowned-source'])
def test_boundary_invalid_citation_gets_corrective_retry(bad_id):
    symptom = SymptomReport(label='observed_failure', behavior_claim='A valid request is rejected.',
        supporting_evidence_ids=['issue-body'], alternative_label='wrong_output',
        boundary_reason='The request is blocked before producing output.', boundary_evidence_ids=['issue-body'],
        confidence=.9, evidence_sufficiency='sufficient')
    root = RootCauseReport(label='incorrect_condition', defect_mechanism='The wrong condition rejects the request.',
        causal_chain=['Request arrives.', 'Condition misfires.', 'Request is rejected.'],
        supporting_evidence_ids=['issue-body'], alternative_label='missing_guard',
        boundary_reason='A wrong condition is reported.', boundary_evidence_ids=['issue-body'],
        confidence=.9, evidence_sufficiency='sufficient')
    consistency = CausalConsistencyReport(status='consistent', rationale='The condition explains the rejection.',
        supporting_evidence_ids=['issue-body'])
    reports = tuple(Stage3TeamReport(team_id=t, symptom=symptom, root_cause=root, consistency=consistency) for t in ['A', 'B'])
    prompts = []
    def call(system, user):
        prompts.append(user)
        return json.dumps(dict(action='symptom_review', rationale='Please recheck the observed symptom boundary.',
            cited_evidence_ids=[bad_id if len(prompts) == 1 else 'issue-body']))
    view = _view()
    view = view.model_copy(update={'items': (*view.items, view.items[0].model_copy(update={'evidence_id': 'unowned-source'}))})
    result = StructuredRoleAgents(StructuredModelClient(call, max_schema_retries=1)).boundary_challenger(reports, view)
    assert len(prompts) == 2
    assert bad_id in prompts[1] and 'VALIDATION ERROR' in prompts[1]
    assert list(result.cited_evidence_ids) == ['issue-body']


def test_formal_query_has_sufficient_output_budget_and_manifest():
    view, anchor = sample()
    replies = iter([decision_query().model_dump_json(), anchor.model_dump_json()])
    options_seen = []
    def call(system, user, *, options):
        options_seen.append(options)
        return next(replies)
    tool = RuleDecisionTools()
    agents = StructuredRoleAgents(StructuredModelClient(call), explainable_tools=tool)
    agents._role(AnalystRole.JOINT_ANCHOR, team_id='A', view=view, schema=JointAnchorReport)
    assert options_seen[0].max_tokens == 32768
    assert tool.manifest()['query_max_tokens'] == options_seen[0].max_tokens


def test_counterevidence_feedback_identifies_the_failed_review():
    view, _ = sample()
    query = decision_query()
    query = query.model_copy(update={'root_reviews': (query.root_reviews[0].model_copy(update={'counterevidence_kind': 'retraction'}),)})
    with pytest.raises(ValueError, match='API Misuse.*retraction'):
        RuleDecisionTools().validate_query(view, query)


def test_bad_quotation_is_rejected_and_feedback_identifies_source():
    view, _ = sample()
    query = decision_query()
    atom = query.atoms[0]
    bad = atom.citations[0].model_copy(update={'quote': 'This fabricated quotation is absent.'})
    query = query.model_copy(update={'atoms': (atom.model_copy(update={'citations': (bad,)}),)})
    with pytest.raises(ValueError, match="evidence_id='e'"):
        RuleDecisionTools().validate_query(view, query)
