import importlib.util
import json
import hashlib
from types import SimpleNamespace

import pytest

from Benchmark.src.adaptive_empirical_workflow.agents import StructuredRoleAgents
from Benchmark.src.adaptive_empirical_workflow.contracts import EvidenceItem, EvidenceView

def item(**kwargs):
    return EvidenceItem(**kwargs,content_sha256=hashlib.sha256(kwargs['content'].encode()).hexdigest())


def view():
    return EvidenceView(record_id='case-x', task='Classify this report using the provided definitions.',
        domain_profile='ase2022', ledger_version=1,
        taxonomy={'symptom': ['Crash', 'Incorrect Functionality'],
                  'root_cause': ['Incorrect Code Logic', 'Misconfiguration', 'Unknown']},
        items=(item(evidence_id='body', record_id='case-x',
            source_type='issue_body', source_uri='https://github.com/a/b/issues/1',
            retrieved_at='2026-09-22T00:00:00Z', explicitness='direct',
            content='TITLE: Buffer failure\n\nBrowser loading fails.\n\nSee https://github.com/a/b/issues/1#issuecomment-2'),
            item(evidence_id='comment', record_id='case-x',
            source_type='issue_comment', source_uri='https://github.com/a/b/issues/1#issuecomment-2',
            retrieved_at='2026-09-22T00:00:00Z', explicitness='direct',
            metadata={'source_author':'reporter','source_published_at':'2026-09-21T00:00:00Z'},
            content='A new version restores loading. The repair mechanism is not described.')))


def test_natural_language_task_does_not_drop_stage3_rules():
    from Benchmark.src.adaptive_empirical_workflow.capabilities import AnalystRole
    from Benchmark.src.adaptive_empirical_workflow.contracts import DimensionVerificationReport
    captured=[]
    def complete(**kwargs):
        captured.append(kwargs['system_prompt'])
        raise RuntimeError('captured')
    agents = StructuredRoleAgents(SimpleNamespace(complete=complete),
        explainable_tools=SimpleNamespace(query_prompt='rules',decision_policy='RULE_MARKER'))
    with pytest.raises(RuntimeError,match='captured'):
        agents._role(AnalystRole.ROOT_CAUSE_VERIFIER,team_id='A',view=view(),schema=DimensionVerificationReport)
    assert 'RULE_MARKER' in captured[0]
    assert agents._root_cause_decision_policy('ase2022', 'stage2') == ''


def load_module(name):
    full = 'Benchmark.src.adaptive_empirical_workflow.' + name
    assert importlib.util.find_spec(full) is not None, f'{name} not implemented'
    return importlib.import_module(full)


def test_source_graph_has_only_explicit_edges_and_keeps_original_evidence():
    module = load_module('source_graph')
    v = view()
    graph = module.build_source_graph(v)
    assert graph == module.build_source_graph(v)
    assert {e['relation'] for e in graph['edges']} <= {'contains','links_to','reply_to','quotes','snapshot_of'}
    assert any(e['relation']=='links_to' for e in graph['edges'])
    context = module.source_context(v, 'Buffer browser', mode='graph')
    assert context['retrieval']['seed_ids']
    assert all(n.get('evidence_id') in {'body','comment',None} for n in context['nodes'])
    assert tuple(i.evidence_id for i in v.items)==('body','comment')
    flat = module.source_context(v,'Buffer browser',mode='flat')
    assert flat['retrieval']==context['retrieval']
    assert json.loads(flat['serialized_graph'])['nodes']==context['nodes']
    assert json.loads(flat['serialized_graph'])['edges']==context['edges']


def test_grounding_requires_category_bridge_and_real_quotes():
    module = load_module('experiment_two')
    v=view()
    review={'label':'Incorrect Code Logic','causal_form':'observed_connection',
        'same_failure':True,'category_match':True,'support_atom_ids':['O1'],
        'counterevidence_atom_ids':[],'counterevidence_kind':'none',
        'reasoning':'A possible browser adaptation failure explains the error.',
        'missing_category_fact':'','mechanism_detail_gaps':[],
        'category_evidence':[], 'category_warrant':'',
        'mechanism_claim':'The patch repaired browser adaptation logic.',
        'mechanism_status':'inferred','mechanism_evidence':[],
        'mechanism_gap':'The exact repair mechanism is not reported.'}
    citation={'evidence_id':'body','quote':'Browser loading fails.','kind':'observation'}
    q=module.GroundedDecisionQuery.model_validate({'atoms':[{'atom_id':'O1','kind':'observation',
        'statement':'Browser loading fails.','citations':[citation]}], 'root_reviews':[review],
        'root_selection':{'label':'Incorrect Code Logic','unknown_basis':None,
            'reasoning':'The candidate is selected for the browser failure.','comparison_atom_ids':['O1']}})
    with pytest.raises(ValueError,match='category'):
        module.GroundedDecisionTools().validate_query(v,q)
    data=q.model_dump(mode='json')
    data['root_reviews'][0].update(category_evidence=[{**citation,'quote':'invented code diff'}],
        category_warrant='This quoted observation allegedly distinguishes this category.')
    with pytest.raises(ValueError,match='quote'):
        module.GroundedDecisionTools().validate_query(v,module.GroundedDecisionQuery.model_validate(data))


def test_experiment_two_arms_are_explicit_and_budget_matched():
    module=load_module('experiment_two')
    assert module.arm_options('E00')=={'rule_checks':False,'source_graph_mode':'off'}
    assert module.arm_options('E11')=={'rule_checks':True,'source_graph_mode':'graph'}
    with pytest.raises(ValueError): module.arm_options('invalid')


def test_supervisor_rejects_missing_or_unresolved_claims():
    from tests.test_adaptive_empirical_workflow_global_supervisor import teams, decision, _stage3_view
    from Benchmark.src.adaptive_empirical_workflow.experiment_two import supervisor_schema, supervisor_targets, validate_supervisor_claims
    v = _stage3_view(); reports = teams()
    data = decision(v, 'arbitrate').model_dump(mode='json')
    data['claim_findings'] = [dict(target_id=k, status='entailed', rule_id='source_entailment',
        citations=[dict(evidence_id=v.items[0].evidence_id, quote=v.items[0].content, kind='observation')],
        rationale='Synthetic finding checks the transport and coverage contract.', missing_evidence='')
        for k in supervisor_targets(reports)]
    good = supervisor_schema().model_validate(data)
    validate_supervisor_claims(good, v, reports)
    with pytest.raises(ValueError, match='every current'):
        validate_supervisor_claims(good.model_copy(update={'claim_findings':good.claim_findings[:-1]}), v, reports)
    bad = good.claim_findings[0].model_copy(update={'status':'unknown','missing_evidence':'Connecting evidence is absent.'})
    with pytest.raises(ValueError, match='Unresolved explanation'):
        validate_supervisor_claims(good.model_copy(update={'claim_findings':(bad,*good.claim_findings[1:])}),v,reports)


def test_final_arbitration_audits_every_sentence_and_preserves_rewrites():
    from tests.test_adaptive_empirical_workflow_evidence_chain import sample, finding
    from tests.test_adaptive_empirical_workflow_agents import _stage3_team
    from Benchmark.src.adaptive_empirical_workflow.agents import StructuredModelClient
    from Benchmark.src.adaptive_empirical_workflow.contracts import AnonymousStage3TeamReport, Stage3ArbitrationPacket, DisagreementMap, Stage3ArbitrationDecision
    v, anchor = sample()
    team = _stage3_team()
    claim = 'The source reports differing output; the exact mechanism is not established.'
    root = team.root_cause.model_copy(update={'defect_mechanism':claim})
    anon = AnonymousStage3TeamReport(symptom=team.symptom, root_cause=root, consistency=team.consistency)
    packet = Stage3ArbitrationPacket(domain_profile='ase2022',taxonomy=v.taxonomy,
        disagreement=DisagreementMap(dimensions=['root_cause_label'],details={},requires_arbitration=True),
        team_a=anon,team_b=anon,classification_ledger_version=1,relevant_evidence=v.items)
    captures=[]
    def transport(system,user,*,options):
        captures.append((json.loads(user),options))
        return json.dumps(dict(resolution_status='resolved',symptom_label=team.symptom.label,
            root_cause_label=root.label,confidence=.7,rationale=claim,rationale_sentences=[claim],
            rationale_findings=[finding('rationale:0')],supporting_evidence_ids=['e'],
            resolved_dimensions=['root_cause_label']))
    agents=StructuredRoleAgents(StructuredModelClient(transport,max_schema_retries=0),experiment_two_arm='E11')
    result=agents.stage3_arbitrator(packet)
    # The final verifier revalidates against the strict canonical schema.
    canonical=Stage3ArbitrationDecision.model_validate(result.model_dump(mode='python'))
    assert canonical.rationale==claim
    from Benchmark.src.adaptive_empirical_workflow.experiment_two import arbitration_schema
    audited=arbitration_schema(Stage3ArbitrationDecision).model_validate(
        agents.telemetry()['final_claim_audits'][0]['audit'])
    assert audited.rationale_findings[0].target_id=='rationale:0'
    assert result.rationale==claim
    payload,options=captures[0]
    assert payload['candidates']['team_a']['root_cause']['defect_mechanism']==claim
    assert 'source_navigation' in payload
    assert payload['evidence_snapshot']['items']==[i.model_dump(mode='json') for i in v.items]
    assert options.max_tokens==8192
    from Benchmark.src.adaptive_empirical_workflow.experiment_two import validate_final_claims
    with pytest.raises(ValueError,match='exactly join'):
        validate_final_claims(audited.model_copy(update={'rationale':claim+' The patch fixes the API.'}),v)
    with pytest.raises(ValueError,match='coverage'):
        validate_final_claims(audited.model_copy(update={'rationale_findings':()}),v)


def test_actual_verifier_caps_and_full_payload_match_across_arms():
    for arm in ['E00','E10','E01','E11']:
        agents=StructuredRoleAgents(SimpleNamespace(),experiment_two_arm=arm)
        assert agents._model_call_options('root_cause_verifier',team_id='A',perspective=None).max_tokens==8192
        payload=json.loads(agents._payload(team_id='A',view=view(),context={}))
        assert payload['evidence_view']['items']==[i.model_dump(mode='json') for i in view().items]
        assert ('source_navigation' in payload.get('context',{}))==(arm in ['E01','E11'])


def test_runner_freezes_inputs_and_retains_invalid_completed_cases(tmp_path):
    from Benchmark.scripts import run_experiment_two as runner
    source=tmp_path/'input.txt';source.write_text('original')
    protocol={'file_hashes':{str(source):runner.digest(source)}}
    runner.verify_frozen(protocol)
    source.write_text('changed')
    with pytest.raises(ValueError,match='changed'): runner.verify_frozen(protocol)
    output=tmp_path/'evaluation48/E00/case01';output.mkdir(parents=True)
    path=output/'predictions_test.jsonl'
    path.write_text(json.dumps({'record_id':'r','stage3_valid':False})+'\n')
    assert runner.completed_row(tmp_path,'E00',1,'r')['stage3_valid'] is False
    path.write_text('{partial')
    with pytest.raises(ValueError): runner.completed_row(tmp_path,'E00',1,'r')


def test_runner_never_sends_gold_to_workers(tmp_path):
    from Benchmark.scripts import run_experiment_two as runner
    protocol=dict(inputs={k:k for k in ['cohort','split','examples','example_source','supplemental']},
        record_ids=['r'],model='fake',graph_serialization='graph',gold_path='SECRET_GOLD')
    cmd=runner.command(protocol,tmp_path/'batch','E11',1,dry=True)
    assert '--dry-run' in cmd and 'SECRET_GOLD' not in cmd
    assert cmd[cmd.index('--record-ids')+1]=='r'
    assert '--source-graph-mode' in cmd


def test_scoring_preserves_pairing_and_invalid_uncertainty(tmp_path):
    from Benchmark.scripts import run_experiment_two as runner
    gold=tmp_path/'gold.jsonl'
    gold.write_text(json.dumps(dict(record_id='r',symptom='Crash',root_cause='Unknown'))+'\n')
    for arm,valid in [('E00',False),('E10',True)]:
        output=tmp_path/'evaluation48'/arm/'case01';output.mkdir(parents=True)
        (output/'predictions_fake.jsonl').write_text(json.dumps(dict(record_id='r',stage3_valid=valid,
            symptom_prediction='Crash',root_cause_prediction='Unknown'))+'\n')
    protocol=dict(gold_path=str(gold),gold_sha256=runner.digest(gold),arms=['E00','E10'],cases=[1],record_ids=['r'])
    runner.score(protocol,tmp_path)
    result=runner.read(tmp_path/'analysis/summary.json')
    assert result['arms']['E00']['invalid_cases']==[1]
    assert result['arms']['E10']['joint_correct']==1
    comparison=result['comparisons']['E10-E00']
    assert comparison['gain']==1 and comparison['loss']==0
    assert comparison['invalid_label_sensitivity_bounds']==[0,1]


def test_e11_supervisor_transport_has_graph_and_claims():
    from tests.test_adaptive_empirical_workflow_global_supervisor import teams, decision, _stage3_view
    from Benchmark.src.adaptive_empirical_workflow.agents import StructuredModelClient
    from Benchmark.src.adaptive_empirical_workflow.contracts import BoundaryChallenge
    v=_stage3_view(); captured=[]
    def transport(system,user,*,options):
        payload=json.loads(user);captured.append(options.role)
        assert 'source_navigation' in payload['context']
        if options.role=='boundary_challenger':
            return BoundaryChallenge(action='pass',rationale='The synthetic premise was checked.',cited_evidence_ids=['issue-body']).model_dump_json()
        assert options.max_tokens==12288
        state=payload['context']
        data=decision(v,'challenge' if not state['history'] else 'arbitrate').model_dump(mode='json')
        data['claim_findings']=[dict(target_id=k,status='entailed',rule_id='source_entailment',
            citations=[dict(evidence_id=v.items[0].evidence_id,quote=v.items[0].content,kind='observation')],
            rationale='Synthetic transport validation, not a semantic judgment.',missing_evidence='')
            for k in state['claim_targets']]
        return json.dumps(data)
    agents=StructuredRoleAgents(StructuredModelClient(transport,max_schema_retries=0),
        global_supervisor=True,experiment_two_arm='E11')
    result=agents.supervise_framework(teams(),v)
    assert not result.failed
    assert captured==['global_supervisor','boundary_challenger','global_supervisor']
