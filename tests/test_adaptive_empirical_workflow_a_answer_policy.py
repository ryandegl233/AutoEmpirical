"""The answer-first experiment changes prompts, retaining v4's evidence gates."""
import pytest

from tests.test_adaptive_empirical_workflow_a_decision import decision_query
from tests.test_adaptive_empirical_workflow_evidence_chain import sample


def test_answer_policy_reaches_query_and_downstream_roles_with_same_gate():
    from Benchmark.src.adaptive_empirical_workflow.a_answer_policy import AnswerDecisionTools
    from Benchmark.src.adaptive_empirical_workflow.a_decision import RuleDecisionTools
    from Benchmark.src.adaptive_empirical_workflow.agents import StructuredModelClient, StructuredRoleAgents
    from Benchmark.src.adaptive_empirical_workflow.capabilities import AnalystRole
    from Benchmark.src.adaptive_empirical_workflow.contracts import JointAnchorReport
    from Benchmark.scripts.run_adaptive_empirical_workflow import build_parser

    assert build_parser().parse_args(['--domain', 'ase2022', '--a-protocol', 'rules-v5']).a_protocol == 'rules-v5'
    tool = AnswerDecisionTools()
    assert tool.query_schema is RuleDecisionTools.query_schema
    assert tool.validate_query.__func__ is RuleDecisionTools.validate_query
    assert tool.manifest()['version'] == 'module-a-v5-answer-policy'
    assert tool.manifest()['decision_policy_sha256'] != RuleDecisionTools().manifest()['decision_policy_sha256']
    view, anchor = sample()
    bad = anchor.model_copy(update={'root_cause': anchor.root_cause.model_copy(update={'label': 'Unknown'})})
    replies = iter([decision_query().model_dump_json(), bad.model_dump_json(), anchor.model_dump_json()])
    prompts = []
    def transport(system, user, *, options):
        prompts.append((options.role, system))
        return next(replies)
    agents = StructuredRoleAgents(StructuredModelClient(transport, max_schema_retries=1, retry_delay_seconds=0), explainable_tools=tool)
    result = agents._role(AnalystRole.JOINT_ANCHOR, team_id='A', view=view, schema=JointAnchorReport)
    assert result.root_cause.label == 'API Misuse'
    assert [role for role, _ in prompts] == ['explainable_tool_query', 'joint_anchor', 'joint_anchor']
    assert all(tool.decision_policy in prompt for _, prompt in prompts)


def test_answer_policy_preserves_abstention_and_source_boundary():
    from Benchmark.src.adaptive_empirical_workflow.a_answer_policy import AnswerDecisionTools
    tool = AnswerDecisionTools()
    view, _ = sample()
    query = decision_query(form='hypothesis', choice='Unknown', unknown_basis='no_supported_category')
    query = query.model_copy(update={'root_reviews': (query.root_reviews[0].model_copy(update={
        'missing_category_fact': 'No observation identifies which component produced the different output.'}),)})
    assert tool.validate_query(view, query)['root_gate']['unknown_allowed']
    forced = query.model_copy(update={'root_selection': query.root_selection.model_copy(update={'label': 'API Misuse', 'unknown_basis': None})})
    with pytest.raises(ValueError, match='supported'):
        tool.validate_query(view, forced)
    atom = query.atoms[0]
    fabricated = atom.model_copy(update={'citations': (atom.citations[0].model_copy(update={'quote': 'This quote is not in the source.'}),)})
    with pytest.raises(ValueError):
        tool.validate_query(view, query.model_copy(update={'atoms': (fabricated,)}))
