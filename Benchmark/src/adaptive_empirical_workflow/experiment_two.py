"""Approved experiment-two interventions. No case-specific rules or gold labels."""
import hashlib
from pathlib import Path
from typing import Literal

from pydantic import Field, create_model
from .a_decision import DecisionQuery, RootReview, RuleDecisionTools
from .evidence_chain import Citation, ChainFinding, check_targets, validate_citations, _validate_findings

VERSION = 'experiment-two-v1'
RULES = '''EXECUTABLE RULE UNION U1-U7:
U1 Distinguish source observations, source assertions, suggestions and model inference.
U2 Preserve certainty, time and scope. Recovery after a change does not prove its mechanism.
U3 Missing evidence is not refutation; unrefuted speculation is not support.
U4 Apply supplied definitions without additional mandatory conditions or keyword shortcuts.
U5 Separate supported category attribution from proof of a complete mechanism. No universal patch requirement.
U6 Compare actual supported candidates; Unknown requires a stated unresolved category distinction.
U7 Repair only unsupported claims and their dependents; preserve independently supported conclusions.
For each proposed mechanism ask whether evidence establishes it, or it merely explains the observation.
Source quotes and well-formed fields do not certify semantic truth. Feedback and consensus are not evidence.
'''


def arm_options(arm):
    if arm not in {'E00','E10','E01','E11'}: raise ValueError('Unknown experiment-two arm')
    return dict(rule_checks=arm[1]=='1',source_graph_mode='graph' if arm[2]=='1' else 'off')


class GroundedRootReview(RootReview):
    category_evidence: tuple[Citation,...] = Field(max_length=6)
    category_warrant: str = Field(max_length=1400)
    mechanism_claim: str = Field(min_length=10,max_length=1400)
    mechanism_status: Literal['observed','reported','inferred','unknown']
    mechanism_evidence: tuple[Citation,...] = Field(max_length=6)
    mechanism_gap: str = Field(max_length=1000)


class GroundedDecisionQuery(DecisionQuery):
    root_reviews: tuple[GroundedRootReview,...] = Field(default=(),max_length=18)


class GroundedDecisionTools(RuleDecisionTools):
    query_schema = GroundedDecisionQuery
    query_prompt = RuleDecisionTools.query_prompt+'\n'+RULES+'''
For every root review cite category_evidence and explain category_warrant: what connects the
observations to this category and distinguishes the closest alternative? Merely satisfying a
definition IF an unproven mechanism were true is not support. State mechanism_claim separately,
its observed/reported/inferred/unknown status, evidence and remaining gap. An inferred mechanism
must remain qualified; do not upgrade it through category_match or observed_connection.
Category attribution can be supported even with incomplete mechanism details; do not default to Unknown.
'''

    def evaluate(self,view,query):
        result = super().evaluate(view,query)
        if not isinstance(query,GroundedDecisionQuery): return result
        for r in query.root_reviews:
            validate_citations(r.category_evidence,view,'root_cause')
            validate_citations(r.mechanism_evidence,view,'root_cause')
            if r.causal_form in {'observed_connection','direct_report'} and r.category_match:
                if not r.category_evidence or len(r.category_warrant.strip())<10:
                    raise ValueError('Supported category requires quoted category evidence and a discriminating warrant')
            if r.mechanism_status in {'observed','reported'}:
                required='observation' if r.mechanism_status=='observed' else 'source_assertion'
                if not any(c.kind==required for c in r.mechanism_evidence):
                    raise ValueError('Established/reported mechanism requires matching source citations')
            elif not r.mechanism_gap.strip():
                raise ValueError('Inferred/unknown mechanism must state the missing connecting evidence')
        result['mechanism_grounding']=[r.model_dump(mode='json') for r in query.root_reviews]
        return result

    def manifest(self):
        return {**super().manifest(),'version':VERSION+'-grounding',
                'grounding_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def supervisor_targets(reports):
    return {f'{r.team_id}/{d}/{key}':value for r in reports for d in ('symptom','root_cause')
            for key,value in check_targets(getattr(r,d),d).items()}


def supervisor_schema():
    from .global_supervisor import SupervisorDecision
    return create_model('GroundedSupervisorDecision',__base__=SupervisorDecision,
        claim_findings=(tuple[ChainFinding,...],Field(min_length=1,max_length=160)))


def validate_supervisor_claims(value,view,reports):
    targets=supervisor_targets(reports)
    actual=[f.target_id for f in value.claim_findings]
    if len(actual)!=len(set(actual)) or set(actual)!=set(targets):
        raise ValueError('Supervisor must cover every current claim/link target exactly once')
    for r in reports:
        for d in ('symptom','root_cause'):
            prefix=f'{r.team_id}/{d}/'
            fs=[f.model_copy(update={'target_id':f.target_id[len(prefix):]})
                for f in value.claim_findings if f.target_id.startswith(prefix)]
            _validate_findings(fs,check_targets(getattr(r,d),d),view,d)
    if value.action=='arbitrate' and any(f.status!='entailed' for f in value.claim_findings):
        raise ValueError('Unresolved explanation claims cannot be marked ready for arbitration')


def arbitration_schema(base):
    return create_model('GroundedFinalDecision',__base__=base,
        rationale_sentences=(tuple[str,...],Field(min_length=1,max_length=16)),
        rationale_findings=(tuple[ChainFinding,...],Field(min_length=1,max_length=16)))


def validate_final_claims(result,view):
    if result.rationale != ' '.join(result.rationale_sentences):
        raise ValueError('Final rationale must exactly join its audited sentences')
    targets={f'rationale:{i}':s for i,s in enumerate(result.rationale_sentences)}
    _validate_findings(result.rationale_findings,targets,view,None)
    status=getattr(result.resolution_status,'value',result.resolution_status)
    if str(status).lower()=='resolved' and any(f.status!='entailed' for f in result.rationale_findings):
        raise ValueError('Resolved final rationale cannot contain unsupported assertions')


def manifest(arm,graph_mode=None):
    opts=arm_options(arm)
    if graph_mode is not None: opts['source_graph_mode']=graph_mode
    here=Path(__file__).parent
    return dict(version=VERSION,arm=arm,**opts,
        policy_delivery='explicit-stage3-role-v1', verifier_max_tokens=8192,
        supervisor_max_tokens=12288,query_max_tokens=8192,arbitrator_max_tokens=8192,
        semantic_truth_verified=False,
        source_sha256={n:hashlib.sha256((here/n).read_bytes()).hexdigest() for n in
            ['experiment_two.py','source_graph.py','evidence_chain.py','a_decision.py','stage3_composition.py','agents.py']})
