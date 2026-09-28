"""Module A v4: explicit pre-decision predicates and a bounded Unknown gate.

The program checks consistency of predicates, not the truth of their model interpretation.
No gold labels or evaluation examples are loaded here.
"""
from pathlib import Path
from typing import Literal

from pydantic import Field

from .evidence_chain import ChainModel, ROOT_CAUSE_DECISION_POLICY, validate_citations
from .formal_reasoning import FormalExplainableTools, FormalQuery, QUERY_PROMPT
from .explainable_tools import ROOT_CAUSE_DEFINITIONS, _sha

DECISION_POLICY = ROOT_CAUSE_DECISION_POLICY + (
    "\nPRE-DECISION GATE v4: Before producing a root label, complete root_reviews and "
    "root_selection. Inspect source wording, category fit, relevance to THIS failure, "
    "and actual counterevidence separately. direct_report means an affirmative causal "
    "assertion; 'maybe', a question, or a troubleshooting proposal is hypothesis. "
    "observed_connection requires a documented causal connection, not just an error message. "
    "Set category_match for support at CATEGORY level; do not silently demand an entire "
    "mechanism. Cite the source atoms behind every assessment. Missing mechanism details "
    "belong in mechanism_detail_gaps, never in missing_category_fact. "
    "A relevant, category-matching direct report or observed connection with no substantive "
    "counterevidence supports a specific category. If exactly one category is supported, "
    "select it. With multiple supported categories, compare their actual evidence; choose "
    "the best-supported or explain a real unresolved conflict. Unknown is permitted only "
    "with no_supported_category or unresolved_supported_conflict. Imagined alternatives, "
    "issue closure, missing patch or unnamed author do not create a supported conflict. "
    "Do not alter a true predicate merely to make Unknown pass. Qualify reported causes "
    "instead of inventing mechanisms. The program will reject contradictory selections. "
    "Downstream context.module_a_root_gate retains this check: a disallowed Unknown "
    "cannot be introduced by an anchor or a verifier. Semantic judgments remain yours; "
    "the tool does not certify their truth."
)


SpecificRootLabel = Literal[tuple(label for label in ROOT_CAUSE_DEFINITIONS if label != 'Unknown')]
RootLabel = Literal[tuple(ROOT_CAUSE_DEFINITIONS)]


class RootReview(ChainModel):
    label: SpecificRootLabel
    causal_form: Literal['direct_report', 'observed_connection', 'hypothesis', 'none']
    same_failure: bool
    category_match: bool
    support_atom_ids: tuple[str, ...] = Field(max_length=12)
    counterevidence_atom_ids: tuple[str, ...] = Field(max_length=12)
    counterevidence_kind: Literal['none', 'contradiction', 'retraction']
    reasoning: str = Field(min_length=10, max_length=1000)
    missing_category_fact: str = Field(max_length=800)
    mechanism_detail_gaps: tuple[str, ...] = Field(max_length=6)


class RootSelection(ChainModel):
    label: RootLabel
    unknown_basis: Literal['no_supported_category', 'unresolved_supported_conflict'] | None
    reasoning: str = Field(min_length=10, max_length=1200)
    comparison_atom_ids: tuple[str, ...] = Field(max_length=24)


class DecisionQuery(FormalQuery):
    root_reviews: tuple[RootReview, ...] = Field(default=(), max_length=18)
    root_selection: RootSelection | None = None


class RuleDecisionTools(FormalExplainableTools):
    query_schema = DecisionQuery
    decision_policy = DECISION_POLICY
    query_prompt = QUERY_PROMPT + '\n' + DECISION_POLICY + (
        '\nUse candidates ONLY for symptom comparisons; use root_reviews for causes. '
        'The frozen method examples use the older schema; your response must use the '
        'current output contract. root_selection and all review predicates are mandatory '
        'for a real query. Keep quotations short and verbatim. Tool recommendations, when '
        'present, are search hints, never evidence proving this case or a closed label set.'
    )

    def manifest(self):
        return {**super().manifest(), 'version': 'module-a-v4-rules',
                'decision_implementation_sha256': _sha(Path(__file__).read_bytes()),
                'decision_policy_sha256': _sha(self.query_prompt.encode())}

    def validate_query(self, view, query):
        if query.root_selection is None or not query.root_reviews:
            raise ValueError('a real pre-decision query requires root reviews and a decision')
        return self.evaluate(view, query)

    def evaluate(self, view, query):
        # Preserve the two frozen source-only demonstrations in their original schema.
        result = super().evaluate(view, query)
        if not isinstance(query, DecisionQuery):
            return result
        if query.root_selection is None and not query.root_reviews:
            return result  # Empty preparation call; never accepted by validate_query.
        if query.root_selection is None or not query.root_reviews:
            raise ValueError('root reviews and decision must be supplied together')
        if any(c.dimension == 'root_cause' for c in query.candidates):
            raise ValueError('root-cause comparisons belong in root_reviews, not candidates')
        atoms = {a.atom_id: a for a in query.atoms}

        def premises(ids):
            if len(set(ids)) != len(ids) or set(ids) - atoms.keys():
                raise ValueError('decision premises require unique current atom IDs')
            selected = [atoms[key] for key in ids]
            for atom in selected:
                validate_citations(atom.citations, view, 'root_cause')
            return selected

        supported, seen = [], set()
        reviews = []
        for review in query.root_reviews:
            if (review.label == 'Unknown' or review.label not in ROOT_CAUSE_DEFINITIONS
                    or review.label not in view.taxonomy['root_cause'] or review.label in seen):
                raise ValueError('root reviews require distinct legal specific labels')
            seen.add(review.label)
            support = premises(review.support_atom_ids)
            counter = premises(review.counterevidence_atom_ids)
            if bool(counter) != (review.counterevidence_kind != 'none'):
                raise ValueError('counterevidence kind and actual cited premises must agree: '
                                 f'label={review.label!r}, counterevidence_kind={review.counterevidence_kind!r}, '
                                 f'counterevidence_atom_ids={list(review.counterevidence_atom_ids)!r}. '
                                 'Use kind=none when no counterevidence is cited; otherwise cite actual current counterevidence atoms.')
            if any(a.kind in {'suggestion', 'workflow_state'} for a in counter):
                raise ValueError('counterevidence cannot consist of suggestions or issue status')
            if review.causal_form == 'direct_report' and not any(a.kind == 'source_assertion' for a in support):
                raise ValueError('direct_report requires a source assertion')
            if review.causal_form == 'observed_connection' and not any(a.kind == 'observation' for a in support):
                raise ValueError('observed_connection requires observed evidence: '
                                 f'label={review.label!r}, support_atom_ids={list(review.support_atom_ids)!r}, '
                                 f'kinds={[a.kind for a in support]!r}. '
                                 'Check the source before choosing causal_form; a reported assertion is not an observed connection.')
            eligible = (review.causal_form in {'direct_report', 'observed_connection'}
                        and review.same_failure and review.category_match and not counter)
            if eligible:
                if review.missing_category_fact.strip():
                    raise ValueError('matched supported category cannot also lack a category fact; separate mechanism gaps')
                supported.append(review.label)
            elif not counter and not review.missing_category_fact.strip():
                raise ValueError('unsupported category requires a specific missing category fact')
            reviews.append({**review.model_dump(mode='json'), 'category_supported': eligible})
        selected = query.root_selection
        premises(selected.comparison_atom_ids)
        if selected.label == 'Unknown':
            if supported:
                if len(supported) < 2 or selected.unknown_basis != 'unresolved_supported_conflict':
                    raise ValueError('Unknown cannot replace a supported specific category without a supported conflict')
                compared = set(selected.comparison_atom_ids)
                if any(not compared.intersection(r.support_atom_ids) for r in query.root_reviews if r.label in supported):
                    raise ValueError('Unknown conflict comparison must cite the supporting premises of each competing cause')
            elif selected.unknown_basis != 'no_supported_category':
                raise ValueError('Unknown with no supported category must explain no_supported_category')
        elif selected.label not in supported or selected.unknown_basis is not None:
            raise ValueError('specific selection must be supported and cannot carry an Unknown basis')
        result['root_gate'] = dict(selected_label=selected.label, supported_labels=supported,
            unknown_allowed=selected.label == 'Unknown', selection=selected.model_dump(mode='json'),
            reviews=reviews, semantic_status='model_predicates_checked_for_consistency_not_independent_truth')
        return result

    def run(self, view, query):
        return {**super().run(view, query), 'version': self.manifest()['version'], 'guidance': self.query_prompt}


class MinedDecisionTools(RuleDecisionTools):
    def manifest(self):
        from .source_mining import mining_manifest
        return {**super().manifest(), 'version': 'module-a-v4-tools', 'source_mining': mining_manifest()}

    def run(self, view, query):
        from .source_mining import recommend
        result = super().run(view, query)
        # The library is immutable approved examples, never current/evaluation records.
        result['recommendations'] = recommend(self._examples, view)
        return result
