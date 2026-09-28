"""V4-based rule completeness checks. Semantic evidence interpretation remains fallible."""
import json
from pathlib import Path
from typing import Literal

from pydantic import Field

from .a_decision import DECISION_POLICY, DecisionQuery, RootReview, RootSelection, RuleDecisionTools, SpecificRootLabel
from .evidence_chain import ChainModel, validate_citations
from .explainable_tools import ROOT_CAUSE_DEFINITIONS, _sha
from .formal_reasoning import FormalQuery, QUERY_PROMPT


# Operational decompositions of the existing taxonomy, not rules learned from case labels.
OBSERVED_CONDITIONS = {
    'API Misuse': dict(usage_requirement='The relevant API usage requirement is stated in allowed evidence.',
        actual_usage='The actual API usage in this failure is recorded.',
        violation='That actual usage conflicts with the stated requirement.'),
    'Browser Incompatibility': dict(browser_behavior='Browser-specific behavior or a browser support limitation is evidenced; naming a browser alone is insufficient.'),
    'Confused Document': dict(document_content='The relevant documented instruction or omission is identified.',
        document_defect='Its error, ambiguity or misleading effect is evidenced.'),
    'Cross-platform App Framework Incompatibility': dict(framework='The relevant cross-platform app environment is identified.',
        incompatibility='An incompatibility involving that framework is evidenced, beyond merely using it.'),
    'Data/Model Inaccessibility': dict(access_failure='A model/data access or loading failure, path problem, CORS or fetch failure is evidenced.'),
    'Dependency Error': dict(dependency='The relevant dependency is identified.',
        dependency_problem='Its missing, redundant, conflicting, vulnerable or incompatible state is evidenced.'),
    'Device Incompatibility': dict(device_constraint='A hardware, OS or device-specific support limitation is evidenced, beyond merely naming the device.'),
    'Import Error': dict(import_problem='A missing, wrong, duplicate or conflicting import is evidenced.'),
    'Improper Exception Handling': dict(handling_problem='Missing, suspicious, unclear or misleading exception handling is evidenced; an exception occurring alone is insufficient.'),
    'Improper Model Attribute': dict(attribute='The relevant model/tensor attribute, size, parameter or property is identified.',
        inappropriate_attribute='Why that attribute is inappropriate is supported by the allowed evidence.'),
    'Incompatibilitty between 3rd-party DL Library and TF.js': dict(libraries='The third-party DL library and TensorFlow.js are identified.',
        incompatibility='Their incompatibility is evidenced, beyond merely using both.'),
    'Inconsistent Modules': dict(modules='The compared TensorFlow.js modules are identified.',
        inconsistency='An inconsistency in their behavior or implementation is evidenced.'),
    'Incorrect Code Logic': dict(faulty_operation='Faulty implementation, algorithm, memory management or environment adaptation is evidenced at category level. An unexplained symptom alone is insufficient; an exact faulty line is not required.'),
    'Misconfiguration': dict(configuration='The relevant environment, build, bundler, backend or runtime setting is identified.',
        incorrect_setting='Why the setting is incorrect is supported, beyond merely naming the environment.'),
    'Unimplemented Operator': dict(operator='The required operator/function is identified.',
        unsupported='Its lack of implementation or support is evidenced.'),
    'Untimely Update': dict(version_lag='An outdated package, delayed update or version lag is evidenced; version numbers alone are insufficient.'),
    'WebGL Limits': dict(inherent_limit='An inherent WebGL limitation is evidenced; a stack trace in WebGL alone is insufficient.'),
}
RULES = {f'R{i:02d}': dict(label=label, definition=definition,
    routes={
        'direct_report': dict(attribution='An affirmative source assertion attributes THIS failure to a cause; preserve uncertainty and author scope.',
            category_condition=definition,
            failure_link='The attribution concerns and explains the current failure at category level.'),
        'observed_connection': {**OBSERVED_CONDITIONS[label],
            'failure_link':'The cited observations connect the category condition to THIS failure, without adding an unobserved premise.'}})
    for i, (label, definition) in enumerate(((k,v) for k,v in ROOT_CAUSE_DEFINITIONS.items() if k!='Unknown'), 1)}


class ConditionCheck(ChainModel):
    condition_id: str = Field(min_length=1)
    status: Literal['supported', 'missing', 'contradicted']
    atom_ids: tuple[str, ...] = Field(max_length=4)
    explanation: str = Field(min_length=10, max_length=800)


class RuleProof(ChainModel):
    rule_id: Literal[tuple(RULES)]
    label: SpecificRootLabel
    route: Literal['direct_report', 'observed_connection']
    conditions: tuple[ConditionCheck, ...] = Field(min_length=1, max_length=8)
    mechanism_detail_gaps: tuple[str, ...] = Field(default=(), max_length=6)


class ProofQuery(FormalQuery):
    root_reviews: tuple[RuleProof, ...] = Field(default=(), max_length=18)
    root_selection: RootSelection | None = None


PROOF_POLICY = DECISION_POLICY + (
    '\nRULE COMPLETENESS v6. Submit root_reviews as rule proofs under the current schema. '
    'For each plausible specific category choose its fixed rule_id and either direct_report '
    'or observed_connection. Cover EVERY condition of that route exactly once, including '
    'failure_link. Never invent a rule or omit a missing condition. Each condition is '
    'supported, contradicted, or missing, with an explanation connecting quoted atom IDs '
    'to its supplied meaning. Supported/contradicted requires current factual evidence; '
    'suggestions and workflow status cannot establish a condition. Missing may cite context '
    'but must identify the unavailable necessary fact. A complete proof requires ALL '
    'conditions supported. The program derives support and counterevidence; do not submit '
    'category_match, same_failure or counterevidence_kind. Use at most 12 distinct support '
    'atoms per proof. Evaluate actual competing evidence, including substantive counterevidence. '
    'Direct reported causes remain available without a patch or demonstrated mechanism; '
    'do not require an observed-route proof in addition to a valid reported-route proof. '
    'For an unsupported candidate, consider both routes before abstaining; explain why '
    'the unsubmitted route does not supply the missing premise. Unknown with no supported '
    'category requires identifiable missing/contradicted conditions. Unknown with supported '
    'competing categories requires the v4 cited conflict comparison. A missing mechanism '
    'detail is not automatically a missing category condition. A validation error identifies '
    'a broken argument, not a root cause: repair the cited condition; never fabricate '
    'evidence or select Unknown solely to satisfy JSON validation. Downstream root labels '
    'must belong to context.module_a_root_gate.allowed_labels; arbitration may use the '
    'union of available teams\' allowed labels or return UNRESOLVED. This is a structural '
    'rule check, not an independent certification of semantic truth.'
)


class ProofDecisionTools(RuleDecisionTools):
    query_schema = ProofQuery
    enforce_proof_labels = True
    decision_policy = PROOF_POLICY
    query_prompt = QUERY_PROMPT + '\n' + PROOF_POLICY + '\nFIXED RULE CATALOG:\n' + json.dumps(RULES, ensure_ascii=False) + (
        '\nUse candidates ONLY for symptoms. Frozen examples retain their old method schema; '
        'your current-case output must use ProofQuery. Copy short exact source quotations.'
    )

    def manifest(self):
        return {**super().manifest(), 'version':'module-a-v6-rule-proof',
            'proof_implementation_sha256':_sha(Path(__file__).read_bytes()),
            'rule_catalog_sha256':_sha(json.dumps(RULES,sort_keys=True).encode())}

    def validate_query(self, view, query):
        if not isinstance(query, ProofQuery):
            raise ValueError('a real rule-proof query requires the proof schema')
        return super().validate_query(view, query)

    def evaluate(self, view, query):
        if not isinstance(query, ProofQuery):
            return super().evaluate(view, query)  # Frozen FormalQuery method examples.
        # Check all original provenance constraints before interpreting condition references.
        super().evaluate(view, FormalQuery(atoms=query.atoms, candidates=query.candidates))
        atoms = {a.atom_id:a for a in query.atoms}
        reviews, proofs = [], []
        for proof in query.root_reviews:
            rule = RULES.get(proof.rule_id)
            if rule is None or rule['label'] != proof.label:
                raise ValueError(f'{proof.rule_id}: rule must match a fixed category')
            expected = rule['routes'][proof.route]
            conditions = {c.condition_id:c for c in proof.conditions}
            if len(conditions) != len(proof.conditions) or set(conditions) != set(expected):
                raise ValueError(f'{proof.rule_id}: cover required conditions exactly once; missing={sorted(set(expected)-set(conditions))}; unexpected={sorted(set(conditions)-set(expected))}')
            support, counter, missing = set(), set(), []
            for c in proof.conditions:
                ids = set(c.atom_ids)
                if len(ids)!=len(c.atom_ids) or ids-atoms.keys():
                    raise ValueError(f'{proof.rule_id}.{c.condition_id}: require unique current-case atom IDs')
                for atom_id in ids:
                    validate_citations(atoms[atom_id].citations, view, 'root_cause')
                if c.status=='missing':
                    missing.append(c.condition_id)
                    continue
                if not ids or any(atoms[i].kind in {'suggestion','workflow_state'} for i in ids):
                    raise ValueError(f'{proof.rule_id}.{c.condition_id}: {c.status} requires factual premises, not suggestions/status')
                if c.condition_id=='attribution' and c.status=='supported' and not any(atoms[i].kind=='source_assertion' for i in ids):
                    raise ValueError(f'{proof.rule_id}.attribution: direct_report requires a quoted source assertion')
                (support if c.status=='supported' else counter).update(ids)
            all_supported = all(c.status=='supported' for c in proof.conditions)
            category_match = all(c.status=='supported' for c in proof.conditions if c.condition_id!='failure_link')
            # Preserve the v4 routes and predicate gate; derive the booleans from conditions.
            route_kind = 'source_assertion' if proof.route=='direct_report' else 'observation'
            if all_supported and not any(atoms[i].kind==route_kind for i in support):
                raise ValueError(f'{proof.rule_id}: {proof.route} requires {route_kind} evidence')
            gaps = '; '.join(f'{proof.rule_id}.{c.condition_id}: {c.explanation}' for c in proof.conditions if c.status=='missing')
            reviews.append(RootReview(label=proof.label,
                causal_form=proof.route if any(atoms[i].kind==route_kind for i in support) else 'hypothesis',
                same_failure=conditions['failure_link'].status=='supported', category_match=category_match,
                support_atom_ids=tuple(sorted(support)), counterevidence_atom_ids=tuple(sorted(counter)),
                counterevidence_kind='contradiction' if counter else 'none',
                reasoning=('; '.join(f'{c.condition_id}: {c.explanation}' for c in proof.conditions))[:1000],
                missing_category_fact=gaps[:800], mechanism_detail_gaps=proof.mechanism_detail_gaps))
            proofs.append({**proof.model_dump(mode='json'), 'complete':all_supported,
                'missing_condition_ids':missing,
                'contradicted_condition_ids':[c.condition_id for c in proof.conditions if c.status=='contradicted']})
        result = super().evaluate(view, DecisionQuery(atoms=query.atoms, candidates=query.candidates,
            root_reviews=tuple(reviews), root_selection=query.root_selection))
        result['rule_proofs'] = proofs
        if 'root_gate' in result:
            gate = result['root_gate']
            gate['allowed_labels'] = gate['supported_labels'] + (['Unknown'] if gate['unknown_allowed'] else [])
            gate['rule_proofs'] = proofs
        return result
