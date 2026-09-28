"""Program-enforced evidence gates; semantic judgments remain explicitly audited."""
import pytest

from tests.test_adaptive_empirical_workflow_evidence_chain import sample


def decision_query(form="direct_report", choice="API Misuse", unknown_basis=None):
    from Benchmark.src.adaptive_empirical_workflow.a_decision import DecisionQuery
    return DecisionQuery.model_validate(dict(
        atoms=[dict(atom_id="a1", kind="source_assertion", statement="The discussion attributes this failure to API misuse.",
                    citations=[dict(evidence_id="e", quote="The output differs.", kind="source_assertion")])],
        root_reviews=[dict(label="API Misuse", causal_form=form, same_failure=True, category_match=True,
                           support_atom_ids=["a1"], counterevidence_atom_ids=[], counterevidence_kind="none",
                           reasoning="The quoted assertion maps to the supplied category.", missing_category_fact="",
                           mechanism_detail_gaps=["The internal sequence is not reproduced."])],
        root_selection=dict(label=choice, unknown_basis=unknown_basis,
                            reasoning="The category is supported at the reported-cause level.", comparison_atom_ids=["a1"])))


def test_direct_report_is_not_demoted_for_missing_mechanism():
    from Benchmark.src.adaptive_empirical_workflow.a_decision import RuleDecisionTools
    view, _ = sample(); tool = RuleDecisionTools()
    result = tool.validate_query(view, decision_query())
    assert result['root_gate']['supported_labels'] == ['API Misuse']
    assert result['root_gate']['unknown_allowed'] is False
    with pytest.raises(ValueError, match="Unknown"):
        tool.validate_query(view, decision_query(choice="Unknown", unknown_basis="no_supported_category"))


def test_hypothesis_and_absence_do_not_force_a_specific_label():
    from Benchmark.src.adaptive_empirical_workflow.a_decision import RuleDecisionTools
    view, _ = sample(); tool = RuleDecisionTools()
    q = decision_query(form="hypothesis", choice="Unknown", unknown_basis="no_supported_category")
    review = q.root_reviews[0].model_copy(update={"missing_category_fact": "A direct cause statement or observed causal connection."})
    q = q.model_copy(update={"root_reviews": (review,)})
    assert tool.validate_query(view, q)['root_gate']['unknown_allowed']
    with pytest.raises(ValueError, match="supported"):
        tool.validate_query(view, q.model_copy(update={"root_selection": q.root_selection.model_copy(update={"label": "API Misuse", "unknown_basis": None})}))
    with pytest.raises(ValueError, match="decision"):
        tool.validate_query(view, type(q)())


def test_unknown_conflict_requires_two_supported_causes_with_cited_comparison():
    from Benchmark.src.adaptive_empirical_workflow.a_decision import RuleDecisionTools
    view, _ = sample(); q = decision_query(); tool = RuleDecisionTools()
    other = q.root_reviews[0].model_copy(update={"label": "Incorrect Code Logic"})
    selected = q.root_selection.model_copy(update={"label": "Unknown", "unknown_basis": "unresolved_supported_conflict"})
    # This fixture checks structural comparisons, not whether one quote semantically supports both labels.
    conflict = q.model_copy(update={"root_reviews": (*q.root_reviews, other), "root_selection": selected})
    assert tool.validate_query(view, conflict)['root_gate']['unknown_allowed']
    with pytest.raises(ValueError, match="comparison"):
        tool.validate_query(view, conflict.model_copy(update={"root_selection": selected.model_copy(update={"comparison_atom_ids": ()})}))


def test_retraction_requires_actual_current_counterevidence():
    from Benchmark.src.adaptive_empirical_workflow.a_decision import RuleDecisionTools
    view, _ = sample(); q = decision_query(); tool = RuleDecisionTools()
    q = q.model_copy(update={"root_reviews": (q.root_reviews[0].model_copy(update={"counterevidence_kind": "retraction"}),)})
    with pytest.raises(ValueError, match="counterevidence"):
        tool.validate_query(view, q)


def test_schema_separates_specific_candidates_from_final_unknown():
    from Benchmark.src.adaptive_empirical_workflow.a_decision import DecisionQuery
    definitions = DecisionQuery.model_json_schema()['$defs']
    assert 'Unknown' not in definitions['RootReview']['properties']['label']['enum']
    assert 'Unknown' in definitions['RootSelection']['properties']['label']['enum']


def test_final_anchor_cannot_ignore_a_validated_unknown_gate():
    from Benchmark.src.adaptive_empirical_workflow.agents import StructuredModelClient, StructuredRoleAgents
    from Benchmark.src.adaptive_empirical_workflow.a_decision import RuleDecisionTools
    from Benchmark.src.adaptive_empirical_workflow.capabilities import AnalystRole
    from Benchmark.src.adaptive_empirical_workflow.contracts import JointAnchorReport
    view, anchor = sample(); seen = []
    bad = anchor.model_copy(update={"root_cause": anchor.root_cause.model_copy(update={"label": "Unknown"})})
    replies = iter([decision_query().model_dump_json(), bad.model_dump_json(), anchor.model_dump_json()])
    def transport(system, user, *, options):
        seen.append(options.role)
        return next(replies)
    agents = StructuredRoleAgents(StructuredModelClient(transport, max_schema_retries=1, retry_delay_seconds=0), explainable_tools=RuleDecisionTools())
    result = agents._role(AnalystRole.JOINT_ANCHOR, team_id="A", view=view, schema=JointAnchorReport)
    assert result.root_cause.label == 'API Misuse'
    assert seen == ['explainable_tool_query', 'joint_anchor', 'joint_anchor']
    assert agents.telemetry(record_id=view.record_id)['module_a_calls'][0]['result']['root_gate']['unknown_allowed'] is False


def test_source_examples_load_but_do_not_fabricate_mining_targets(tmp_path):
    import csv
    import hashlib
    import json
    from Benchmark.src.adaptive_empirical_workflow.a_decision import RuleDecisionTools, MinedDecisionTools, DecisionQuery
    source = tmp_path/'source.csv'; bundle = tmp_path/'examples.json'
    text = 'Perhaps a decoder issue caused the different output.'
    with source.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['record_id','issue_url','body'])
        writer.writeheader()
        for i in range(2):writer.writerow(dict(record_id=f'example-{i}',issue_url=f'https://training.test/{i}',body=text))
    examples=[]
    for i in range(2):
        examples.append(dict(record_id=f'example-{i}',query=dict(atoms=[dict(atom_id='h',kind='source_assertion',statement=text,
            citations=[dict(evidence_id=f'example-{i}:body',quote=text,kind='source_assertion')])],
            candidates=[dict(dimension='root_cause',label='Incorrect Code Logic',premise_ids=['h'],relation='causal_link',
                             assessment='unknown',reason='The source only proposes a tentative decoder hypothesis.',missing_evidence='The actual causal link.')])) )
    bundle.write_text(json.dumps(dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),examples=examples)),encoding='utf-8')
    for cls in (RuleDecisionTools, MinedDecisionTools):
        tool = cls.from_files(bundle, source)
        result = tool.run(sample()[0], DecisionQuery())
        assert len(result['examples']) == 2
        if cls is MinedDecisionTools:
            assert result['recommendations']['training_records'] == 0
            assert result['recommendations']['candidate_range'] == []
            assert result['recommendations']['status'] == 'insufficient_source_supported_causes'


def test_apriori_counts_and_cart_path_use_only_source_supported_examples():
    from Benchmark.src.adaptive_empirical_workflow.formal_reasoning import FormalQuery, FormalExplainableTools
    from Benchmark.src.adaptive_empirical_workflow.source_mining import recommend
    from tests.test_explainable_tools import view
    examples = []
    for i in range(6):
        label, term = ('Device Incompatibility', 'chipset') if i < 3 else ('Browser Incompatibility', 'browser')
        text = f'The report attributes failure to {term} incompatibility.'
        v = view(record_id=f'train-{i}', content=text)
        v = v.model_copy(update={'items': (v.items[0].model_copy(update={'source_uri': f'https://training.test/issues/{i}'}),)})
        query = FormalQuery.model_validate(dict(atoms=[dict(atom_id='a',kind='source_assertion',statement=text,
            citations=[dict(evidence_id='body',quote=text,kind='source_assertion')])], candidates=[dict(
                dimension='root_cause',label=label,premise_ids=['a'],relation='causal_link',assessment='supported',
                reason='The report explicitly attributes this failure to that incompatibility.',missing_evidence='')]))
        FormalExplainableTools().evaluate(v, query)
        examples.append((v, query))
    result = recommend(examples, view(content='The chipset fails.'))
    assert result['status'] == 'available'
    rule = next(r for r in result['association_rules'] if r['antecedent'] == ['chipset'])
    assert (rule['support_count'],rule['antecedent_count'],rule['total_records']) == (3,3,6)
    assert (rule['support'],rule['confidence'],rule['lift']) == (0.5,1.0,2.0)
    assert result['tree_path'] and result['candidate_range'] == ['Device Incompatibility']
    assert recommend(examples, view(content='unseenwordxyz'))['status'] == 'no_feature_coverage'
    with pytest.raises(ValueError, match='overlap'):
        recommend(examples, examples[0][0])
    # Merely hypothesized causes do not become supervised targets or association consequents.
    weak = [(v,q.model_copy(update={'candidates': (q.candidates[0].model_copy(update={'assessment':'unknown','missing_evidence':'No cause was established.'}),)})) for v,q in examples]
    assert recommend(weak,view())['training_records'] == 0


def test_legacy_verifier_cannot_bypass_unknown_gate():
    from Benchmark.src.adaptive_empirical_workflow.agents import StructuredRoleAgents, StructuredModelClient
    from Benchmark.src.adaptive_empirical_workflow.a_decision import RuleDecisionTools
    from Benchmark.src.adaptive_empirical_workflow.contracts import JointAnchorReport, DimensionVerificationReport
    from Benchmark.src.adaptive_empirical_workflow.capabilities import AnalystRole
    view, anchor = sample()
    reject = DimensionVerificationReport(dimension='root_cause', verdict='reject', anchor_label='API Misuse',
        alternative_label='Unknown', rationale='The mechanism has not been independently reproduced.',
        corrected_claim='The reported failure has not been reproduced.',
        corrected_causal_chain=['The output differs.','A decoder is suggested.','The mechanism is not reproduced.'],
        supporting_evidence_ids=['e'], confidence=0.8)
    accept = reject.model_copy(update={'verdict': type(reject.verdict)('accept'), 'alternative_label':None,
                                       'corrected_claim':None,'corrected_causal_chain':()})
    replies = iter([decision_query().model_dump_json(),anchor.model_dump_json(),reject.model_dump_json(),accept.model_dump_json()])
    agents = StructuredRoleAgents(StructuredModelClient(lambda s,u,*,options: next(replies),max_schema_retries=1,retry_delay_seconds=0),explainable_tools=RuleDecisionTools())
    agents._role(AnalystRole.JOINT_ANCHOR,team_id='A',view=view,schema=JointAnchorReport)
    assert agents.root_cause_verifier('A',anchor,view).verdict.value == 'accept'
    assert len(agents.telemetry(record_id=view.record_id)['module_a_attempts']) == 1
