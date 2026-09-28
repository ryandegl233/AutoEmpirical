"""Rule completeness, abstention, provenance and downstream enforcement."""
import hashlib
import pytest
from tests.test_adaptive_empirical_workflow_evidence_chain import sample


def proof_fixture():
    from Benchmark.src.adaptive_empirical_workflow.rule_proof import ProofQuery
    view, anchor = sample()
    sentences = ['The API requires a positive size.', 'The application passed a negative size.',
                 'The negative size violates that requirement.', 'That call produced the reported error.']
    content = ' '.join(sentences)
    item = view.items[0].model_copy(update={'content': content, 'content_sha256': hashlib.sha256(content.encode()).hexdigest()})
    view = view.model_copy(update={'items': (item,)})
    ids = ['usage_requirement', 'actual_usage', 'violation', 'failure_link']
    q = ProofQuery.model_validate(dict(atoms=[dict(atom_id=f'a{i}', kind='observation', statement=s,
        citations=[dict(evidence_id=item.evidence_id, quote=s, kind='observation')]) for i,s in enumerate(sentences)],
        root_reviews=[dict(rule_id='R01', label='API Misuse', route='observed_connection',
            conditions=[dict(condition_id=c, status='supported', atom_ids=[f'a{i}'], explanation=sentences[i]) for i,c in enumerate(ids)],
            mechanism_detail_gaps=[])],
        root_selection=dict(label='API Misuse', unknown_basis=None, reasoning='The recorded use violates the stated requirement and causes this failure.', comparison_atom_ids=['a0','a1','a2','a3'])))
    return view, anchor, q


def test_complete_chain_and_missing_premise_have_different_outcomes():
    from Benchmark.src.adaptive_empirical_workflow.rule_proof import ProofDecisionTools
    view, _, q = proof_fixture(); tool = ProofDecisionTools()
    gate = tool.validate_query(view,q)['root_gate']
    assert gate['allowed_labels'] == ['API Misuse']
    review = q.root_reviews[0]
    # Removing a required step is an invalid output, not an automatic Unknown.
    with pytest.raises(ValueError, match='usage_requirement'):
        tool.validate_query(view,q.model_copy(update={'root_reviews':(review.model_copy(update={'conditions':review.conditions[1:]}),)}))
    missing = review.conditions[0].model_copy(update={'status':'missing','atom_ids':(), 'explanation':'The API usage requirement is absent from the current evidence.'})
    incomplete = q.model_copy(update={'root_reviews':(review.model_copy(update={'conditions':(missing,*review.conditions[1:])}),)})
    with pytest.raises(ValueError, match='supported'):
        tool.validate_query(view,incomplete)
    unknown = incomplete.model_copy(update={'root_selection':q.root_selection.model_copy(update={'label':'Unknown','unknown_basis':'no_supported_category'})})
    result = tool.validate_query(view,unknown)
    assert result['root_gate']['allowed_labels'] == ['Unknown']
    assert result['rule_proofs'][0]['missing_condition_ids'] == ['usage_requirement']


@pytest.mark.parametrize('corruption', ['rule','duplicate','foreign_atom','quote','suggestion'])
def test_proof_rejects_untrusted_or_disconnected_premises(corruption):
    from Benchmark.src.adaptive_empirical_workflow.rule_proof import ProofDecisionTools
    view, _, q = proof_fixture(); data = q.model_dump(mode='json'); review=data['root_reviews'][0]
    if corruption=='rule': review['rule_id']='R99'
    if corruption=='duplicate': review['conditions'][1]['condition_id']='usage_requirement'
    if corruption=='foreign_atom': review['conditions'][0]['atom_ids']=['another-case:a0']
    if corruption=='quote': data['atoms'][0]['citations'][0]['quote']='A fabricated API contract.'
    if corruption=='suggestion':
        data['atoms'][0]['kind']='suggestion'; data['atoms'][0]['citations'][0]['kind']='suggestion'
    with pytest.raises(ValueError): ProofDecisionTools().validate_query(view,type(q).model_validate(data))


def test_program_derives_counterevidence_and_rejects_unsupported_final_label():
    from Benchmark.src.adaptive_empirical_workflow.rule_proof import ProofDecisionTools
    view, _, q = proof_fixture(); r=q.root_reviews[0]
    contradicted=r.conditions[2].model_copy(update={'status':'contradicted'})
    q=q.model_copy(update={'root_reviews':(r.model_copy(update={'conditions':(*r.conditions[:2],contradicted,r.conditions[3])}),),
                          'root_selection':q.root_selection.model_copy(update={'label':'Unknown','unknown_basis':'no_supported_category'})})
    result=ProofDecisionTools().validate_query(view,q)
    assert result['root_gate']['reviews'][0]['counterevidence_kind']=='contradiction'
    assert result['rule_proofs'][0]['contradicted_condition_ids']==['violation']


def test_anchor_cannot_switch_to_an_unproved_specific_label():
    from Benchmark.src.adaptive_empirical_workflow.rule_proof import ProofDecisionTools
    from Benchmark.src.adaptive_empirical_workflow.agents import StructuredModelClient, StructuredRoleAgents
    from Benchmark.src.adaptive_empirical_workflow.capabilities import AnalystRole
    from Benchmark.src.adaptive_empirical_workflow.contracts import JointAnchorReport
    view, anchor, q=proof_fixture()
    bad=anchor.model_copy(update={'root_cause':anchor.root_cause.model_copy(update={'label':'Incorrect Code Logic'})})
    replies=iter([q.model_dump_json(),bad.model_dump_json(),anchor.model_dump_json()]); seen=[]
    def transport(system,user,*,options):
        seen.append(options.role); return next(replies)
    agents=StructuredRoleAgents(StructuredModelClient(transport,max_schema_retries=1,retry_delay_seconds=0),explainable_tools=ProofDecisionTools())
    result=agents._role(AnalystRole.JOINT_ANCHOR,team_id='A',view=view,schema=JointAnchorReport)
    assert result.root_cause.label=='API Misuse'
    assert seen==['explainable_tool_query','joint_anchor','joint_anchor']


def test_reported_cause_does_not_require_observed_contract_or_patch():
    from Benchmark.src.adaptive_empirical_workflow.rule_proof import ProofDecisionTools, RULES
    from Benchmark.src.adaptive_empirical_workflow.explainable_tools import ROOT_CAUSE_DEFINITIONS
    view, _, q=proof_fixture()
    text='The discussion attributes this failure to incorrect API usage.'
    item=view.items[0].model_copy(update={'content':text,'content_sha256':hashlib.sha256(text.encode()).hexdigest()})
    view=view.model_copy(update={'items':(item,)})
    data=q.model_dump(mode='json')
    data['atoms']=[dict(atom_id='a0',kind='source_assertion',statement=text,
        citations=[dict(evidence_id='e',quote=text,kind='source_assertion')])]
    data['root_reviews'][0].update(route='direct_report',conditions=[dict(condition_id=c,status='supported',atom_ids=['a0'],explanation=text)
        for c in RULES['R01']['routes']['direct_report']],mechanism_detail_gaps=['The internal mechanism has not been reproduced.'])
    data['root_selection']['comparison_atom_ids']=['a0']
    result=ProofDecisionTools().validate_query(view,type(q).model_validate(data))
    assert result['root_gate']['allowed_labels']==['API Misuse']
    assert {r['label'] for r in RULES.values()} == set(ROOT_CAUSE_DEFINITIONS)-{'Unknown'}
    assert all('failure_link' in route for r in RULES.values() for route in r['routes'].values())


@pytest.mark.parametrize('stale', [False,True])
def test_arbitration_rejects_unproved_labels_and_stale_gates(stale):
    from Benchmark.src.adaptive_empirical_workflow.rule_proof import ProofDecisionTools
    from Benchmark.src.adaptive_empirical_workflow.agents import StructuredRoleAgents, StructuredModelClient
    from Benchmark.src.adaptive_empirical_workflow.contracts import Stage3ArbitrationPacket, AnonymousStage3TeamReport, DisagreementMap
    from tests.test_adaptive_empirical_workflow_agents import _stage3_team
    import json
    view, _, q=proof_fixture(); tool=ProofDecisionTools(); team=_stage3_team()
    anon=AnonymousStage3TeamReport(symptom=team.symptom,root_cause=team.root_cause,consistency=team.consistency)
    packet=Stage3ArbitrationPacket(domain_profile='ase2022',taxonomy=view.taxonomy,
        disagreement=DisagreementMap(dimensions=['root_cause_label'],details={},requires_arbitration=True),
        team_a=anon,team_b=anon,classification_ledger_version=view.ledger_version,relevant_evidence=view.items)
    bad=dict(resolution_status='resolved',symptom_label='Crash',root_cause_label='Incorrect Code Logic',confidence=.8,
             rationale='The selected label must have a complete proof in the current evidence.',supporting_evidence_ids=['e'],resolved_dimensions=['root_cause_label'])
    good=(dict(resolution_status='unresolved',rationale='No current proof is available to support this final root label.',unresolved_dimensions=['root_cause_label'])
          if stale else {**bad,'root_cause_label':'API Misuse'})
    replies=iter([json.dumps(bad),json.dumps(good)]); calls=[]
    def transport(s,u,*,options): calls.append(options.role);return next(replies)
    agents=StructuredRoleAgents(StructuredModelClient(transport,max_schema_retries=1,retry_delay_seconds=0),explainable_tools=tool)
    agents._module_a_calls=[dict(record_id=view.record_id,team_id=t,ledger_version=view.ledger_version+(1 if stale else 0),result=tool.validate_query(view,q)) for t in ('A','B')]
    result=agents.stage3_arbitrator(packet)
    assert result.root_cause_label==(None if stale else 'API Misuse')
    assert len(calls)==2


def test_broken_query_retries_with_condition_error_not_automatic_unknown():
    from Benchmark.src.adaptive_empirical_workflow.rule_proof import ProofDecisionTools
    from Benchmark.src.adaptive_empirical_workflow.agents import StructuredRoleAgents, StructuredModelClient
    from Benchmark.src.adaptive_empirical_workflow.capabilities import AnalystRole
    from Benchmark.src.adaptive_empirical_workflow.contracts import JointAnchorReport
    view,anchor,q=proof_fixture(); r=q.root_reviews[0]
    broken=q.model_copy(update={'root_reviews':(r.model_copy(update={'conditions':r.conditions[1:]}),)})
    replies=iter([broken.model_dump_json(),q.model_dump_json(),anchor.model_dump_json()]); users=[]
    def transport(s,u,*,options):users.append(u);return next(replies)
    agents=StructuredRoleAgents(StructuredModelClient(transport,max_schema_retries=1,retry_delay_seconds=0),explainable_tools=ProofDecisionTools())
    result=agents._role(AnalystRole.JOINT_ANCHOR,team_id='A',view=view,schema=JointAnchorReport)
    assert result.root_cause.label=='API Misuse'
    assert 'usage_requirement' in users[1]
    attempts=agents.telemetry(record_id=view.record_id)['module_a_attempts']
    assert [a['status'] for a in attempts]==['invalid','valid']


def test_verifier_cannot_replace_anchor_with_an_unproved_specific_cause():
    from Benchmark.src.adaptive_empirical_workflow.rule_proof import ProofDecisionTools
    from Benchmark.src.adaptive_empirical_workflow.agents import StructuredRoleAgents, StructuredModelClient
    from Benchmark.src.adaptive_empirical_workflow.capabilities import AnalystRole
    from Benchmark.src.adaptive_empirical_workflow.contracts import JointAnchorReport, DimensionVerificationReport
    view,anchor,q=proof_fixture()
    reject=DimensionVerificationReport(dimension='root_cause',verdict='reject',anchor_label='API Misuse',
        alternative_label='Incorrect Code Logic',rationale='A different specific cause is proposed without the required proof.',
        corrected_claim='The implementation uses an incorrect branch.',corrected_causal_chain=['The call starts.','A branch is selected.','The failure occurs.'],
        supporting_evidence_ids=['e'],confidence=.8)
    accept=reject.model_copy(update={'verdict':type(reject.verdict)('accept'),'alternative_label':None,'corrected_claim':None,'corrected_causal_chain':()})
    replies=iter([q.model_dump_json(),anchor.model_dump_json(),reject.model_dump_json(),accept.model_dump_json()])
    agents=StructuredRoleAgents(StructuredModelClient(lambda s,u,*,options:next(replies),max_schema_retries=1,retry_delay_seconds=0),explainable_tools=ProofDecisionTools())
    agents._role(AnalystRole.JOINT_ANCHOR,team_id='A',view=view,schema=JointAnchorReport)
    assert agents.root_cause_verifier('A',anchor,view).verdict.value=='accept'
