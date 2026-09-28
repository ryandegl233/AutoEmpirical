"""Module A v2: quoted atoms -> explicit candidate comparisons -> evidence gaps.

This validates a model's argument structure, not the truth of its interpretation.
It learns no rules or labels from evaluation data.
"""
from pathlib import Path
from typing import Literal

from pydantic import Field

from .evidence_chain import ChainModel, Citation, FORMAL_POLICY, validate_citations
from .explainable_tools import ExplainableTools, ROOT_CAUSE_DEFINITIONS, SYMPTOM_DEFINITIONS, _sha

VERSION = "module-a-formal-few-shot-v3-comment-policy"
QUERY_PROMPT = FORMAL_POLICY + (
    "\nExtract quoted source atoms, preserving their type and uncertainty. Then compare "
    "plausible specific candidates with the supplied definitions. Each comparison must "
    "name premise atom IDs, relation, assessment, reason and any missing discriminator. "
    "Root-cause comparisons use causal_link and explain the causal connection, not word "
    "matching. Never submit Unknown as a candidate fact. An unknown candidate assessment "
    "is not a final Unknown label. Check all plausible competing candidates without "
    "inventing requirements. Source assertions may support a reported cause at that "
    "strength; they are not automatically demonstrations. Do not impose a universal "
    "patch/reproduction requirement. Example conclusions are method demonstrations, "
    "not adjudicated answers. Example IDs cannot prove the current case. Return JSON."
)


class EvidenceAtom(ChainModel):
    atom_id: str = Field(min_length=1)
    kind: Literal["observation", "source_assertion", "suggestion", "workflow_state"]
    statement: str = Field(min_length=5, max_length=1200)
    citations: tuple[Citation, ...] = Field(min_length=1, max_length=6)


class CandidateCheck(ChainModel):
    dimension: Literal["symptom", "root_cause"]
    label: str = Field(min_length=1)
    premise_ids: tuple[str, ...] = Field(default=(), max_length=24)
    relation: Literal["definition_match", "causal_link"]
    assessment: Literal["supported", "contradicted", "unknown"]
    reason: str = Field(min_length=10, max_length=1600)
    missing_evidence: str = Field(default="", max_length=1600)


class FormalQuery(ChainModel):
    atoms: tuple[EvidenceAtom, ...] = Field(default=(), max_length=24)
    candidates: tuple[CandidateCheck, ...] = Field(default=(), max_length=24)


class FormalExample(ChainModel):
    record_id: str = Field(min_length=1)
    query: FormalQuery


class FormalBundle(ChainModel):
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    examples: tuple[FormalExample, ...] = Field(min_length=1, max_length=8)


class FormalExplainableTools(ExplainableTools):
    query_max_tokens = 32768
    query_schema = FormalQuery
    bundle_schema = FormalBundle
    query_prompt = QUERY_PROMPT

    def manifest(self):
        return {**super().manifest(), "version": VERSION,
                "formal_implementation_sha256": _sha(Path(__file__).read_bytes()),
                "formal_policy_sha256": _sha(FORMAL_POLICY.encode()), "query_max_tokens": self.query_max_tokens}

    def evaluate(self, view, query: FormalQuery):
        if view.domain_profile != "ase2022" or view.taxonomy.get("annotation_modes"):
            raise ValueError("formal Module A requires ASE2022 single-label taxonomy")
        if len({item.evidence_id for item in view.items}) != len(view.items):
            raise ValueError("evidence IDs must be unique")
        atoms = {atom.atom_id: atom for atom in query.atoms}
        if len(atoms) != len(query.atoms):
            raise ValueError("atom IDs must be unique")
        citation_checks = []
        for atom in query.atoms:
            if any(c.kind != atom.kind for c in atom.citations):
                raise ValueError(f"atom and quotation kinds must agree: atom_id={atom.atom_id!r}, "
                                 f"atom kind={atom.kind!r}, citation kinds={[c.kind for c in atom.citations]!r}. "
                                 "Check the source and use a consistent evidence type.")
            citation_checks.extend({'atom_id': atom.atom_id, **check}
                                   for check in validate_citations(atom.citations, view, None))
        checks, seen = [], set()
        for candidate in query.candidates:
            dimension, label = candidate.dimension, candidate.label
            if label == "Unknown":
                raise ValueError("Unknown is an unresolved comparison outcome, never a submitted fact")
            definitions = SYMPTOM_DEFINITIONS if dimension == "symptom" else ROOT_CAUSE_DEFINITIONS
            if label not in view.taxonomy.get(dimension, ()) or label not in definitions:
                raise ValueError("candidate must use a supplied legal label")
            if (dimension, label) in seen:
                raise ValueError("compare each candidate once, incorporating conflicting evidence")
            seen.add((dimension, label))
            if len(set(candidate.premise_ids)) != len(candidate.premise_ids) or set(candidate.premise_ids) - atoms.keys():
                raise ValueError("premises must reference unique current atom IDs")
            premises = [atoms[key] for key in candidate.premise_ids]
            for atom in premises:
                validate_citations(atom.citations, view, dimension)
            kinds = {atom.kind for atom in premises}
            if candidate.assessment != "unknown":
                if not premises:
                    raise ValueError("a supported/contradicted candidate requires cited premises")
                if kinds <= {"workflow_state"}:
                    raise ValueError("workflow state cannot establish a defect label")
                if dimension == "root_cause" and kinds <= {"suggestion", "workflow_state"}:
                    raise ValueError("suggestion cannot establish a root cause")
            if candidate.assessment == "unknown" and not candidate.missing_evidence.strip():
                raise ValueError("unknown comparison requires missing discriminating evidence")
            if dimension == "root_cause" and candidate.relation != "causal_link":
                raise ValueError("root-cause comparisons require a causal_link argument")
            state = candidate.assessment
            if state == "supported":
                state = "reported_support" if "source_assertion" in kinds else "conditional_support"
            checks.append({**candidate.model_dump(mode="json"), "state": state,
                           "definition": definitions[label], "semantic_status": "model_interpretation"})
        return {"atoms": [atom.model_dump(mode="json") for atom in query.atoms],
                "candidate_checks": checks, "citation_validation": citation_checks}

    def run(self, view, query):
        self.check_evaluation_records([{"record_id": view.record_id, "issue_url": item.source_uri} for item in view.items]
                                      or [{"record_id": view.record_id}])
        examples = [{"record_id": example.record_id,
                     "source_items": [{"evidence_id": item.evidence_id, "source_uri": item.source_uri,
                                       "content": item.content, "content_sha256": item.content_sha256} for item in example.items],
                     "query": example_query.model_dump(mode="json"), **self.evaluate(example, example_query)}
                    for example, example_query in self._examples]
        return {"version": VERSION, "guidance": QUERY_PROMPT,
                "knowledge_status": "configured" if examples else "not_configured",
                "examples": examples, **self.evaluate(view, query)}
