"""A v5: category-level inference before abstention; unchanged v4 schema/validators."""
from pathlib import Path

from .a_decision import RuleDecisionTools
from .evidence_chain import ROOT_CAUSE_DECISION_POLICY
from .explainable_tools import _sha
from .formal_reasoning import QUERY_PROMPT


ANSWER_POLICY = ROOT_CAUSE_DECISION_POLICY + (
    '\nCATEGORY-LEVEL ANSWER POLICY v5. Your task is to choose the best-supported '
    'root-cause CATEGORY from the supplied taxonomy, not to certify a complete physical '
    'mechanism. Make a supported decision whenever the visible record permits one. '
    'Unknown is an exceptional outcome of comparison, never a shortcut around analysis.\n'
    '1. Build root_reviews from quoted current-case facts. Separate direct_report '
    '(an affirmative source attribution), observed_connection (a category-level connection '
    'supported by recorded observations and the supplied definition), hypothesis '
    '(a proposed explanation still needing an unobserved premise), and none. '
    'An observed_connection may be INFERRED by you: the source need not explicitly name '
    'the cause or label. In reasoning write: cited facts -> defining category condition '
    '-> label; then identify the closest alternative and why its evidence is weaker. '
    'Keep source atoms as observations/assertions; put your inference in reasoning, '
    'never fabricate an inferential quotation.\n'
    '2. Before rejecting a category, inspect the recorded operations, error context, '
    'success/failure contrasts, affected scope, and relevant comments together. Ask '
    'whether these establish its defining condition without inventing an API contract, '
    'an unseen implementation, an unperformed test, or an undocumented event. If yes, '
    'use observed_connection, same_failure=true and category_match=true. You may '
    'state an inferred category with a qualified mechanism. A symptom name alone, '
    'a component merely appearing in a stack trace, or a speculation alone is insufficient.\n'
    '3. Do not equate absence of an explicit diagnosis with absence of category evidence. '
    'Do not demand a maintainer response, code diff, independent reproduction, exact '
    'faulty line, or complete memory/cache/implementation explanation unless that fact '
    'is actually needed to distinguish the competing CATEGORY definitions. Unresolved '
    'implementation details go in mechanism_detail_gaps. A tentative mechanism in '
    'a comment does not erase separately recorded observations; evaluate those facts '
    'on their own. Conversely, evidence from another commenter\'s different setup must '
    'not silently become a demonstrated fact about the original failure.\n'
    '4. Treat category_match as a category-level judgment, not certainty about every '
    'causal step. A relevant direct report or observed_connection with category fit '
    'and no substantive counterevidence supports a specific label under the existing '
    'gate. Preserve the distinction between reported, inferred, and demonstrated '
    'causes in your explanation. Do not upgrade questions or suggestions into facts.\n'
    '5. Compare candidate evidence before abstaining. Prefer the category whose '
    'defining condition is supported most directly by the record over alternatives '
    'that need additional unobserved premises. Mere possibility, label overlap, or '
    'multiple conceivable mechanisms does not establish an unresolved conflict. '
    'Do not use any broad category as a catch-all for all unexplained failures.\n'
    '6. Before selecting Unknown, revisit the strongest specific candidate. In '
    'root_selection.reasoning explicitly state its best cited evidence, why the '
    'facts-plus-definition inference still fails, and the decisive missing category '
    'fact. For no_supported_category, each rejected review must name that concrete '
    'missing fact in missing_category_fact; generic "unconfirmed cause", "no fix", '
    'or "no definitive proof" is not an adequate explanation. For '
    'unresolved_supported_conflict, compare the actual supporting premises of the '
    'competing labels and explain why their defining conditions cannot be distinguished. '
    'If the record really lacks that information, retain Unknown. Never invent evidence '
    'or force a label merely to reduce abstention.\n'
    '7. Complete root_reviews and root_selection before the final answer. Existing '
    'citation, predicate and selection checks all remain binding. Once a specific '
    'category is supported, downstream roles must not switch to Unknown just because '
    'a mechanism is uncertain. Qualify the explanation instead. Respect role ownership '
    'and arbitration authority. The program checks consistency of your predicates, '
    'not their independent semantic truth.'
)


class AnswerDecisionTools(RuleDecisionTools):
    decision_policy = ANSWER_POLICY
    query_prompt = QUERY_PROMPT + '\n' + ANSWER_POLICY + (
        '\nUse candidates ONLY for symptom comparisons and root_reviews for causes. '
        'Use the current output contract, not the older schema of the frozen method '
        'examples. root_selection and every review predicate are mandatory for a real '
        'query. Keep quotations short and verbatim. Example conclusions illustrate a '
        'method; they do not prescribe an abstention rate or prove the current case.'
    )

    def manifest(self):
        return {**super().manifest(), 'version': 'module-a-v5-answer-policy',
                'answer_policy_implementation_sha256': _sha(Path(__file__).read_bytes())}
