"""Validated message contracts shared by workflow components."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)
from typing_extensions import TypedDict
from Benchmark.src.annotation_contracts import label_valid
from .evidence_chain import ChainAudit


def _deep_freeze(value: Any) -> Any:
    if type(value) is dict or isinstance(value, MappingProxyType):
        return MappingProxyType(
            {key: _deep_freeze(item) for key, item in value.items()}
        )
    if type(value) in (list, tuple):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _validate_json_string(value: str, *, path: str) -> None:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(
            "metadata values must be JSON-compatible; "
            f"{path} contains invalid Unicode"
        ) from error


def _validate_json_value(
    value: Any,
    *,
    path: str,
    active_container_ids: set[int] | None = None,
) -> None:
    if value is None or type(value) in (bool, int):
        return
    if type(value) is str:
        _validate_json_string(value, path=path)
        return
    if type(value) is float:
        if math.isfinite(value):
            return
        raise ValueError(
            f"metadata values must be JSON-compatible; {path} is non-finite"
        )
    is_object = type(value) is dict or isinstance(value, MappingProxyType)
    is_array = type(value) in (list, tuple)
    if is_object or is_array:
        active = active_container_ids if active_container_ids is not None else set()
        identity = id(value)
        if identity in active:
            raise ValueError(
                "metadata values must be JSON-compatible; " f"{path} contains a cycle"
            )
        active.add(identity)
        try:
            if is_object:
                for key, item in value.items():
                    if type(key) is not str:
                        raise ValueError(
                            "metadata values must be JSON-compatible; "
                            f"{path} contains a non-string key"
                        )
                    _validate_json_string(key, path=f"{path} key")
                    _validate_json_value(
                        item,
                        path=f"{path}.{key}",
                        active_container_ids=active,
                    )
            else:
                for index, item in enumerate(value):
                    _validate_json_value(
                        item,
                        path=f"{path}[{index}]",
                        active_container_ids=active,
                    )
        finally:
            active.remove(identity)
        return
    raise ValueError(
        "metadata values must be JSON-compatible; "
        f"{path} has unsupported type {type(value).__name__}"
    )


def _deep_thaw(value: Any) -> Any:
    if type(value) is dict or isinstance(value, MappingProxyType):
        return {key: _deep_thaw(item) for key, item in value.items()}
    if type(value) in (list, tuple):
        return [_deep_thaw(item) for item in value]
    return value


class StrictModel(BaseModel):
    """Base model that rejects undeclared fields."""

    model_config = ConfigDict(extra="forbid")


class FrozenStrictModel(StrictModel):
    """Strict contract whose validated state cannot change after construction."""

    model_config = ConfigDict(extra="forbid", frozen=True)


_VALIDATION_SUMMARY_IDENTIFIER = re.compile(
    r"^(?:__root__|[a-z][a-z0-9_]*(?:\.(?:[a-z][a-z0-9_]*|\d+))*)$"
)
_VALIDATION_SUMMARY_FORBIDDEN_TERMS = frozenset(
    {"content", "completion", "output", "prompt", "raw", "response", "secret"}
)
_KNOWN_STRUCTURED_OUTPUT_SCHEMAS = frozenset(
    {
        "BoundaryChallenge",
        "CausalConsistencyReport",
        "EvidenceReadinessReport",
        "FaultEvidenceAssessment",
        "IsstaStage2AnalysisReport",
        "RepairCausalityAssessment",
        "RootCauseReport",
        "ScopeBoundaryAssessment",
        "SlaConditionalArbitration",
        "SlaJointDiagnosis",
        "SlaJointVerification",
        "Stage2AnalysisReport",
        "Stage2ArbitrationDecision",
        "Stage3ArbitrationDecision",
        "SymptomReport",
    }
)


def _safe_validation_summary_value(value: object) -> str | None:
    if not isinstance(value, str) or len(value) > 96:
        return None
    if not _VALIDATION_SUMMARY_IDENTIFIER.fullmatch(value):
        return None
    if any(term in value for term in _VALIDATION_SUMMARY_FORBIDDEN_TERMS):
        return None
    return value


def sanitize_validation_summary(value: object) -> dict[str, list[str]]:
    """Retain only bounded structural validation identifiers for audit."""

    if not isinstance(value, Mapping):
        return {}
    sanitized: dict[str, list[str]] = {}
    for key in ("fields", "codes"):
        values = value.get(key)
        if not isinstance(values, (list, tuple)):
            continue
        safe_values = list(
            dict.fromkeys(
                candidate
                for item in values[:16]
                if (candidate := _safe_validation_summary_value(item)) is not None
            )
        )
        if safe_values:
            sanitized[key] = safe_values
    return sanitized


def sanitize_schema_name(value: object) -> str | None:
    """Return only a schema identity owned by this workflow."""

    if isinstance(value, str) and value in _KNOWN_STRUCTURED_OUTPUT_SCHEMAS:
        return value
    return None


class EvidenceExplicitness(str, Enum):
    DIRECT = "direct"
    INFERRED = "inferred"


class TeamPerspective(str, Enum):
    """The domain-independent reasoning policy for one Stage 3 team."""

    EVIDENCE_FIRST = "evidence_first"
    FALSIFICATION_FIRST = "falsification_first"


class ThinkingMode(str, Enum):
    """Provider-neutral thinking behavior for one model role."""

    ENABLED = "enabled"
    DISABLED = "disabled"
    PROVIDER_DEFAULT = "provider_default"


class RoleModelPolicy(StrictModel):
    """Resolved thinking and output ceiling for one model invocation."""

    thinking: ThinkingMode
    max_tokens: int = Field(ge=1)


_STAGE3_ROLE_TOKEN_CEILINGS = {
    "symptom_analyst": 2200,
    "root_cause_analyst": 2800,
    "joint_anchor": 32768,
    "symptom_verifier": 2200,
    "root_cause_verifier": 32768,
    "causal_consistency_checker": 1800,
    "boundary_challenger": 1800,
    "stage3_arbitrator": 2600,
    "baseline_revision_assessment": 2200,
    "baseline_revision_consistency": 1800,
}
_OPTIONAL_REVISION_ROLES = frozenset(
    {"baseline_revision_assessment", "baseline_revision_consistency"}
)
_NON_STAGE3_MODEL_ROLES = (
    "stage2_fault_verifier",
    "fault_evidence_analyst",
    "scope_boundary_analyst",
    "repair_causality_analyst",
    "evidence_readiness",
    "stage2_arbitrator",
)
_CAUSAL_THINKING_ROLES = frozenset(
    {
        "root_cause_analyst",
        "joint_anchor",
        "root_cause_verifier",
        "causal_consistency_checker",
        "boundary_challenger",
        "stage3_arbitrator",
    }
)
_STAGE3_TEAM_SCOPED_ROLES = frozenset(
    {
        "symptom_analyst",
        "root_cause_analyst",
        "joint_anchor",
        "symptom_verifier",
        "root_cause_verifier",
        "causal_consistency_checker",
        "baseline_revision_assessment",
        "baseline_revision_consistency",
    }
)


class Stage3AgentPolicy(StrictModel):
    """Named Stage 3 policy with explicit role and role/team overrides."""

    profile: Literal["off", "causal", "label-thinking", "all-stage3"]
    provider_default: RoleModelPolicy
    profile_policies: dict[str, RoleModelPolicy]
    role_overrides: dict[str, RoleModelPolicy] = Field(default_factory=dict)
    role_team_overrides: dict[str, RoleModelPolicy] = Field(default_factory=dict)

    @field_validator("role_team_overrides")
    @classmethod
    def validate_role_team_overrides(
        cls,
        overrides: dict[str, RoleModelPolicy],
    ) -> dict[str, RoleModelPolicy]:
        for target in overrides:
            role, marker, team_id = target.partition("@")
            if (
                not marker
                or role not in _STAGE3_TEAM_SCOPED_ROLES
                or team_id not in {"A", "B"}
            ):
                raise ValueError(f"{role or target} does not accept a team override")
        return overrides

    @classmethod
    def from_profile(
        cls,
        profile: (
            Literal["off", "causal", "label-thinking", "all-stage3"] | str
        ) = "all-stage3",
        *,
        default_max_tokens: int = 1600,
        role_overrides: Mapping[str, RoleModelPolicy] | None = None,
        role_team_overrides: Mapping[str, RoleModelPolicy] | None = None,
    ) -> Stage3AgentPolicy:
        if default_max_tokens < 1:
            raise ValueError("default_max_tokens must be positive")
        if profile not in {"off", "causal", "label-thinking", "all-stage3"}:
            raise ValueError(f"unknown Stage 3 thinking profile: {profile}")
        policies: dict[str, RoleModelPolicy] = {}
        for role in _NON_STAGE3_MODEL_ROLES:
            policies[role] = RoleModelPolicy(
                thinking=ThinkingMode.DISABLED,
                max_tokens=default_max_tokens,
            )
        for role, ceiling in _STAGE3_ROLE_TOKEN_CEILINGS.items():
            label_thinking_role = role in {
                "symptom_analyst",
                "root_cause_analyst",
                "joint_anchor",
                "root_cause_verifier",
                "stage3_arbitrator",
            }
            enabled = (
                (profile == "all-stage3" and role not in _OPTIONAL_REVISION_ROLES)
                or (profile == "causal" and role in _CAUSAL_THINKING_ROLES)
                or (profile == "label-thinking" and label_thinking_role)
            )
            if profile == "label-thinking" and label_thinking_role:
                ceiling = 32768
            policies[role] = RoleModelPolicy(
                thinking=(ThinkingMode.ENABLED if enabled else ThinkingMode.DISABLED),
                max_tokens=ceiling,
            )
        return cls(
            profile=profile,
            provider_default=RoleModelPolicy(
                thinking=ThinkingMode.PROVIDER_DEFAULT,
                max_tokens=default_max_tokens,
            ),
            profile_policies=policies,
            role_overrides=dict(role_overrides or {}),
            role_team_overrides=dict(role_team_overrides or {}),
        )

    def resolve(
        self,
        role: str,
        team_id: str | None,
        *,
        invocation_override: RoleModelPolicy | None = None,
    ) -> RoleModelPolicy:
        """Resolve invocation > role/team > role > profile > provider default."""

        if invocation_override is not None:
            return invocation_override
        if team_id is not None:
            role_team = self.role_team_overrides.get(f"{role}@{team_id}")
            if role_team is not None:
                return role_team
        role_override = self.role_overrides.get(role)
        if role_override is not None:
            return role_override
        return self.profile_policies.get(role, self.provider_default)


@dataclass(frozen=True)
class Stage3TeamPolicy:
    """Frozen identity-to-perspective assignment consumed by role adapters."""

    team_id: str
    perspective: TeamPerspective


class EvidenceDimension(str, Enum):
    FAULT_EXISTENCE = "fault_existence"
    STUDY_SCOPE = "study_scope"
    REPAIR_CAUSALITY = "repair_causality"
    SYMPTOM = "symptom"
    ROOT_CAUSE = "root_cause"


class TaxonomySemanticOrigin(str, Enum):
    """Whether taxonomy semantics come from the paper or an operational model."""

    PAPER_DEFINITION = "paper_definition"
    OPERATIONAL_DEFINITION = "operational_definition"


class TaxonomyNode(FrozenStrictModel):
    """One official label plus explicitly sourced structural semantics."""

    label: str = Field(min_length=1)
    dimension: Literal["symptom", "root_cause"]
    definition: str = Field(min_length=1)
    definition_semantic_origin: TaxonomySemanticOrigin
    structure_semantic_origin: TaxonomySemanticOrigin
    abstraction_level: str = Field(min_length=1)
    responsibility_scope: str = Field(min_length=1)
    concept_kind: Literal[
        "outcome", "mechanism", "trigger", "responsibility", "context"
    ]
    parents: tuple[str, ...] = ()
    children: tuple[str, ...] = ()
    nearest_neighbors: tuple[str, ...] = ()
    exclusion_rules: tuple[str, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_semantic_origin(cls, value: object) -> object:
        """Normalize old node-wide provenance without emitting it downstream."""

        if not isinstance(value, Mapping) or "semantic_origin" not in value:
            return value
        if {
            "definition_semantic_origin",
            "structure_semantic_origin",
        }.intersection(value):
            raise ValueError("taxonomy node cannot mix legacy and field provenance")
        normalized = dict(value)
        origin = normalized.pop("semantic_origin")
        normalized["definition_semantic_origin"] = origin
        normalized["structure_semantic_origin"] = (
            TaxonomySemanticOrigin.OPERATIONAL_DEFINITION
        )
        return normalized

    @model_validator(mode="after")
    def validate_relationship_lists(self) -> "TaxonomyNode":
        for field_name in (
            "parents",
            "children",
            "nearest_neighbors",
            "exclusion_rules",
        ):
            values = getattr(self, field_name)
            if len(values) != len(set(values)) or any(not value for value in values):
                raise ValueError(
                    f"taxonomy node {field_name} must be unique and non-empty"
                )
        if self.label in {*self.parents, *self.children, *self.nearest_neighbors}:
            raise ValueError("taxonomy node cannot reference itself")
        return self


class BoundaryCriterion(FrozenStrictModel):
    """Necessary and excluding conditions for one side of a label boundary."""

    label: str = Field(min_length=1)
    positive_conditions: tuple[str, ...] = Field(min_length=1)
    exclusion_conditions: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_conditions(self) -> "BoundaryCriterion":
        for field_name in ("positive_conditions", "exclusion_conditions"):
            values = getattr(self, field_name)
            if len(values) != len(set(values)) or any(not value for value in values):
                raise ValueError(f"boundary {field_name} must be unique and non-empty")
        return self


class BoundaryCard(FrozenStrictModel):
    """A record-independent decision rule for two neighboring official labels."""

    card_id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9-]*$")
    dimension: Literal["symptom", "root_cause"]
    labels: tuple[str, str]
    semantic_origin: TaxonomySemanticOrigin
    decision_question: str = Field(min_length=1)
    observable_slots: tuple[str, ...] = Field(min_length=1)
    criteria: tuple[BoundaryCriterion, BoundaryCriterion]

    @model_validator(mode="after")
    def validate_card_shape(self) -> "BoundaryCard":
        if self.labels[0] == self.labels[1]:
            raise ValueError("boundary card labels must be distinct")
        if len(self.observable_slots) != len(set(self.observable_slots)) or any(
            not value for value in self.observable_slots
        ):
            raise ValueError("boundary observable slots must be unique and non-empty")
        if {criterion.label for criterion in self.criteria} != set(self.labels):
            raise ValueError("boundary criteria labels must exactly match card labels")
        return self


class TaxonomyStructure(FrozenStrictModel):
    """Versioned, record-independent structure layered over an official taxonomy."""

    schema_version: Literal[1]
    domain: str = Field(min_length=1)
    nodes: tuple[TaxonomyNode, ...] = Field(min_length=1)
    boundary_cards: tuple[BoundaryCard, ...] = ()

    @model_validator(mode="after")
    def validate_unique_identities(self) -> "TaxonomyStructure":
        node_keys = [(node.dimension, node.label) for node in self.nodes]
        if len(node_keys) != len(set(node_keys)):
            raise ValueError("taxonomy structure contains duplicate nodes")
        card_ids = [card.card_id for card in self.boundary_cards]
        if len(card_ids) != len(set(card_ids)):
            raise ValueError("taxonomy structure contains duplicate boundary card IDs")
        return self


class BaselineAnchor(FrozenStrictModel):
    """One frozen Baseline decision and its artifact-level provenance."""

    record_id: str = Field(min_length=1)
    valid: bool
    symptom_label: str | None = Field(default=None, min_length=1)
    root_cause_label: str | None = Field(default=None, min_length=1)
    source_config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_predictions_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_prediction_shape(self) -> "BaselineAnchor":
        labels = (self.symptom_label, self.root_cause_label)
        if self.valid and any(label is None for label in labels):
            raise ValueError("valid Baseline anchor requires both Stage 3 labels")
        if not self.valid and any(label is not None for label in labels):
            raise ValueError("invalid Baseline anchor cannot contain Stage 3 labels")
        return self


class RevisionCandidateSourceKind(str, Enum):
    """Canonical producer-owned source of one bounded revision candidate."""

    TEAM_FINAL = "team_final"
    VERIFIER_CORRECTION = "verifier_correction"
    REPAIR_BACKED_ALTERNATIVE = "repair_backed_alternative"


class RevisionCandidateSignal(FrozenStrictModel):
    """Evidence and report provenance that nominate one exact taxonomy label."""

    dimension: EvidenceDimension
    label: str = Field(min_length=1)
    source_kinds: tuple[RevisionCandidateSourceKind, ...] = Field(min_length=1)
    team_ids: tuple[str, ...] = Field(min_length=1, max_length=2)
    report_digests: tuple[str, ...] = Field(min_length=1, max_length=2)
    supporting_evidence_ids: tuple[str, ...] = Field(min_length=1)
    counter_evidence_ids: tuple[str, ...] = ()
    authoritative_repair_evidence_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_canonical_signal(self) -> "RevisionCandidateSignal":
        canonical_kinds = tuple(
            sorted(set(self.source_kinds), key=lambda kind: kind.value)
        )
        if self.source_kinds != canonical_kinds:
            raise ValueError("candidate source kinds must be unique and canonical")
        if self.team_ids not in {("A",), ("B",), ("A", "B")}:
            raise ValueError(
                "candidate signal team IDs must be canonical A/B identities"
            )
        if len(self.report_digests) != len(self.team_ids):
            raise ValueError("candidate signal report digests must align with team IDs")
        if any(
            re.fullmatch(r"[0-9a-f]{64}", digest) is None
            for digest in self.report_digests
        ):
            raise ValueError("candidate signal report digests must be SHA-256 values")
        for field_name in (
            "supporting_evidence_ids",
            "counter_evidence_ids",
            "authoritative_repair_evidence_ids",
        ):
            values = getattr(self, field_name)
            if values != tuple(sorted(set(values))) or any(
                not value for value in values
            ):
                raise ValueError(f"{field_name} must be unique, non-empty, and sorted")
        if not set(self.authoritative_repair_evidence_ids).issubset(
            self.supporting_evidence_ids
        ):
            raise ValueError("authoritative repair evidence must be positive support")
        return self


class RevisionProposalEnvelope(FrozenStrictModel):
    """Canonical, evidence-owned request to assess one Baseline challenger."""

    dimension: EvidenceDimension
    baseline_label: str = Field(min_length=1)
    proposed_label: str = Field(min_length=1)
    boundary_card_id: str = Field(min_length=1)
    proposer_team_ids: tuple[str, ...] = Field(min_length=1, max_length=2)
    proposer_report_digests: tuple[str, ...] = Field(min_length=1, max_length=2)
    supporting_evidence_ids: tuple[str, ...] = Field(min_length=1)
    counter_evidence_ids: tuple[str, ...] = ()
    candidate_signals: tuple[RevisionCandidateSignal, ...] = Field(min_length=1)
    baseline_source_config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    baseline_source_predictions_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    taxonomy_structure_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_view_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="before")
    @classmethod
    def bind_legacy_team_final_signal(cls, value: object) -> object:
        """Upgrade pre-v2 envelopes while keeping provenance inside the digest."""

        if not isinstance(value, Mapping) or "candidate_signals" in value:
            return value
        upgraded = dict(value)
        required = (
            "dimension",
            "proposed_label",
            "proposer_team_ids",
            "proposer_report_digests",
            "supporting_evidence_ids",
        )
        if any(field_name not in upgraded for field_name in required):
            return value
        upgraded["candidate_signals"] = (
            {
                "dimension": upgraded["dimension"],
                "label": upgraded["proposed_label"],
                "source_kinds": (RevisionCandidateSourceKind.TEAM_FINAL,),
                "team_ids": upgraded["proposer_team_ids"],
                "report_digests": upgraded["proposer_report_digests"],
                "supporting_evidence_ids": upgraded["supporting_evidence_ids"],
                "counter_evidence_ids": upgraded.get("counter_evidence_ids", ()),
                "authoritative_repair_evidence_ids": (),
            },
        )
        return upgraded

    @model_validator(mode="after")
    def validate_canonical_identity(self) -> "RevisionProposalEnvelope":
        if self.dimension not in {
            EvidenceDimension.SYMPTOM,
            EvidenceDimension.ROOT_CAUSE,
        }:
            raise ValueError("revision proposal must own one Stage 3 dimension")
        if self.baseline_label == self.proposed_label:
            raise ValueError("revision proposal must challenge the Baseline label")
        if self.proposer_team_ids not in {("A",), ("B",), ("A", "B")}:
            raise ValueError("proposal team IDs must be canonical A/B identities")
        if len(self.proposer_report_digests) != len(self.proposer_team_ids):
            raise ValueError("proposal report digests must align with proposer teams")
        if any(
            re.fullmatch(r"[0-9a-f]{64}", digest) is None
            for digest in self.proposer_report_digests
        ):
            raise ValueError("proposal report digests must be SHA-256 values")
        if len(self.proposer_report_digests) != len(set(self.proposer_report_digests)):
            raise ValueError("proposal report digests must be unique")
        for field_name in (
            "supporting_evidence_ids",
            "counter_evidence_ids",
        ):
            values = getattr(self, field_name)
            if len(values) != len(set(values)) or any(not value for value in values):
                raise ValueError(f"{field_name} must be unique and non-empty")
        if any(
            signal.dimension is not self.dimension
            or signal.label != self.proposed_label
            for signal in self.candidate_signals
        ):
            raise ValueError(
                "candidate signals must own the proposal dimension and label"
            )
        signal_team_ids = tuple(
            team_id
            for team_id in ("A", "B")
            if any(team_id in signal.team_ids for signal in self.candidate_signals)
        )
        signal_report_digests = tuple(
            digest
            for team_id in signal_team_ids
            for signal in self.candidate_signals
            for signal_team_id, digest in zip(signal.team_ids, signal.report_digests)
            if signal_team_id == team_id
        )
        if (
            signal_team_ids != self.proposer_team_ids
            or signal_report_digests != self.proposer_report_digests
        ):
            raise ValueError("proposal ownership must exactly match candidate signals")
        signal_support = tuple(
            sorted(
                {
                    evidence_id
                    for signal in self.candidate_signals
                    for evidence_id in signal.supporting_evidence_ids
                }
            )
        )
        signal_counter = tuple(
            sorted(
                {
                    evidence_id
                    for signal in self.candidate_signals
                    for evidence_id in signal.counter_evidence_ids
                }
            )
        )
        if (
            signal_support != self.supporting_evidence_ids
            or signal_counter != self.counter_evidence_ids
        ):
            raise ValueError("proposal evidence must exactly match candidate signals")
        return self


def revision_proposal_digest(proposal: RevisionProposalEnvelope) -> str:
    """Hash the complete canonical proposal without interpreting opaque bindings."""

    return hashlib.sha256(
        json.dumps(
            proposal.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


class RevisionBasis(str, Enum):
    """The independently verifiable mechanism authorizing a Baseline revision."""

    TAXONOMY_BOUNDARY = "taxonomy_boundary"


class BaselineDecisionAction(str, Enum):
    PRESERVED = "preserved"
    REVISED = "revised"
    BASELINE_UNAVAILABLE = "baseline_unavailable_fallback"


class BaselineGateProvenance(str, Enum):
    """Exact authority used to construct a post-preservation decision."""

    PRESERVED = "preserved"
    PRE_GATE_CANDIDATE = "pre_gate_candidate"
    REVISION_CERTIFICATE = "revision_certificate"


class LabelRevisionCertificate(FrozenStrictModel):
    """A record-level, dimension-owned claim that may revise one Baseline label."""

    dimension: EvidenceDimension
    proposal_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    baseline_label: str = Field(min_length=1)
    proposed_label: str = Field(min_length=1)
    revision_basis: RevisionBasis
    boundary_card_id: str = Field(min_length=1)
    contradicted_baseline_condition: str = Field(min_length=10)
    satisfied_proposed_conditions: tuple[str, ...] = Field(min_length=1)
    supporting_evidence_ids: tuple[str, ...] = Field(min_length=1)
    counter_evidence_ids: tuple[str, ...] = ()
    supporting_team_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_revision_shape(self) -> "LabelRevisionCertificate":
        if self.dimension not in {
            EvidenceDimension.SYMPTOM,
            EvidenceDimension.ROOT_CAUSE,
        }:
            raise ValueError("revision certificate must own one Stage 3 dimension")
        if self.baseline_label == self.proposed_label:
            raise ValueError("revision certificate must propose a different label")
        for field_name in (
            "satisfied_proposed_conditions",
            "supporting_evidence_ids",
            "counter_evidence_ids",
            "supporting_team_ids",
        ):
            values = getattr(self, field_name)
            if len(values) != len(set(values)) or any(not value for value in values):
                raise ValueError(f"{field_name} must be unique and non-empty")
        return self


class RevisionAssessmentVerdict(str, Enum):
    PRESERVE = "preserve"
    REVISE = "revise"
    INSUFFICIENT = "insufficient"


class RevisionFindingStatus(str, Enum):
    """Evidence status for one fixed Boundary Card condition."""

    SUPPORTED = "supported"
    REFUTED = "refuted"
    INSUFFICIENT = "insufficient"


class RevisionConditionFinding(FrozenStrictModel):
    """One model-visible finding bound to a literal card condition."""

    condition: str = Field(min_length=1)
    status: RevisionFindingStatus
    citation_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_citations(self) -> "RevisionConditionFinding":
        if len(self.citation_ids) != len(set(self.citation_ids)) or any(
            not value for value in self.citation_ids
        ):
            raise ValueError("finding citations must be unique and non-empty")
        if (
            self.status is not RevisionFindingStatus.INSUFFICIENT
            and not self.citation_ids
        ):
            raise ValueError("determinate finding requires at least one citation")
        return self


class RevisionEntailmentOutcome(str, Enum):
    ENTAILED = "entailed"
    NOT_ENTAILED = "not_entailed"
    INSUFFICIENT = "insufficient"


class RevisionEntailmentResult(FrozenStrictModel):
    """Raw Team A output for positive candidate entailment."""

    condition_findings: tuple[RevisionConditionFinding, ...] = Field(min_length=1)
    baseline_exclusion_finding: RevisionConditionFinding
    outcome: RevisionEntailmentOutcome
    summary: str = Field(min_length=20, max_length=800)

    @model_validator(mode="after")
    def outcome_matches_findings(self) -> "RevisionEntailmentResult":
        statuses = tuple(
            finding.status
            for finding in (*self.condition_findings, self.baseline_exclusion_finding)
        )
        if self.outcome is RevisionEntailmentOutcome.ENTAILED and any(
            status is not RevisionFindingStatus.SUPPORTED for status in statuses
        ):
            raise ValueError("entailed requires every fixed condition to be supported")
        if self.outcome is RevisionEntailmentOutcome.NOT_ENTAILED and not any(
            status is RevisionFindingStatus.REFUTED for status in statuses
        ):
            raise ValueError("not_entailed requires at least one refuted condition")
        if self.outcome is RevisionEntailmentOutcome.INSUFFICIENT and not any(
            status is RevisionFindingStatus.INSUFFICIENT for status in statuses
        ):
            raise ValueError("insufficient requires at least one insufficient finding")
        return self


class RevisionFalsificationOutcome(str, Enum):
    REVISION_SURVIVES = "revision_survives"
    REVISION_FALSIFIED = "revision_falsified"
    INSUFFICIENT = "insufficient"


class RevisionCompetingReading(FrozenStrictModel):
    label: str = Field(min_length=1)
    summary: str = Field(min_length=20, max_length=800)
    citation_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_citations(self) -> "RevisionCompetingReading":
        if len(self.citation_ids) != len(set(self.citation_ids)) or any(
            not value for value in self.citation_ids
        ):
            raise ValueError("competing-reading citations must be unique and non-empty")
        return self


class RevisionFalsificationResult(FrozenStrictModel):
    """Raw Team B output for adversarial candidate falsification."""

    baseline_survival_finding: RevisionConditionFinding
    proposed_defeater_findings: tuple[RevisionConditionFinding, ...] = Field(
        min_length=1
    )
    strongest_competing_reading: RevisionCompetingReading
    outcome: RevisionFalsificationOutcome
    summary: str = Field(min_length=20, max_length=800)

    @model_validator(mode="after")
    def outcome_matches_findings(self) -> "RevisionFalsificationResult":
        baseline_status = self.baseline_survival_finding.status
        defeater_statuses = tuple(
            finding.status for finding in self.proposed_defeater_findings
        )
        if self.outcome is RevisionFalsificationOutcome.REVISION_SURVIVES and (
            baseline_status is not RevisionFindingStatus.REFUTED
            or any(
                status is not RevisionFindingStatus.REFUTED
                for status in defeater_statuses
            )
        ):
            raise ValueError(
                "revision_survives requires refuting Baseline survival and every defeater"
            )
        if self.outcome is RevisionFalsificationOutcome.REVISION_FALSIFIED and not (
            baseline_status is RevisionFindingStatus.SUPPORTED
            or any(
                status is RevisionFindingStatus.SUPPORTED
                for status in defeater_statuses
            )
        ):
            raise ValueError(
                "revision_falsified requires a surviving Baseline or supported defeater"
            )
        if self.outcome is RevisionFalsificationOutcome.INSUFFICIENT and not (
            baseline_status is RevisionFindingStatus.INSUFFICIENT
            or any(
                status is RevisionFindingStatus.INSUFFICIENT
                for status in defeater_statuses
            )
        ):
            raise ValueError("insufficient requires at least one insufficient finding")
        return self


class RevisionAssessorTask(str, Enum):
    ENTAILMENT = "revision_entailment"
    FALSIFICATION = "revision_falsification"


HETEROGENEOUS_REVISION_ASSESSMENT_POLICY = "heterogeneous-entailment-falsification-v1"
TASK_SPECIFIC_REVISION_CROSS_POLICY = "opposite-task-specific-raw-bound-v3"
LEGACY_REVISION_ASSESSMENT_POLICY = "legacy-homogeneous-v1"
LEGACY_REVISION_CROSS_POLICY = "legacy-shared-cross-v1"


def revision_raw_result_digest(
    result: RevisionEntailmentResult | RevisionFalsificationResult,
) -> str:
    task = (
        RevisionAssessorTask.ENTAILMENT
        if isinstance(result, RevisionEntailmentResult)
        else RevisionAssessorTask.FALSIFICATION
    )
    return hashlib.sha256(
        json.dumps(
            {"task": task.value, "result": result.model_dump(mode="json")},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


class BaselineRevisionAssessment(FrozenStrictModel):
    """One team's structured, dimension-owned basis for replacing a Baseline label."""

    assessor_team_id: str | None = Field(default=None, min_length=1)
    proposal_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    dimension: EvidenceDimension
    verdict: RevisionAssessmentVerdict
    baseline_label: str | None = Field(default=None, min_length=1)
    proposed_label: str | None = Field(default=None, min_length=1)
    boundary_card_id: str | None = Field(default=None, min_length=1)
    contradicted_baseline_condition: str | None = Field(default=None, min_length=10)
    satisfied_proposed_conditions: tuple[str, ...] = ()
    supporting_evidence_ids: tuple[str, ...] = ()
    counter_evidence_ids: tuple[str, ...] = ()
    baseline_source_config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    baseline_source_predictions_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    taxonomy_structure_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_view_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    assessor_task: RevisionAssessorTask | None = None
    raw_result_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    raw_entailment_result: RevisionEntailmentResult | None = None
    raw_falsification_result: RevisionFalsificationResult | None = None

    @model_validator(mode="after")
    def validate_revision_shape(self) -> "BaselineRevisionAssessment":
        if self.assessor_team_id not in {None, "A", "B"}:
            raise ValueError("revision assessor team must be A or B")
        if self.dimension not in {
            EvidenceDimension.SYMPTOM,
            EvidenceDimension.ROOT_CAUSE,
        }:
            raise ValueError(
                "baseline revision assessment must own one Stage 3 dimension"
            )
        raw_values = (
            self.assessor_task,
            self.raw_result_digest,
            self.raw_entailment_result,
            self.raw_falsification_result,
        )
        if any(value is not None for value in raw_values):
            if self.assessor_task is None or self.raw_result_digest is None:
                raise ValueError("raw assessment provenance must be complete")
            expected_task = (
                RevisionAssessorTask.ENTAILMENT
                if self.assessor_team_id == "A"
                else RevisionAssessorTask.FALSIFICATION
            )
            if self.assessor_task is not expected_task:
                raise ValueError("assessor task must match the fixed A/B task identity")
            raw_result = (
                self.raw_entailment_result
                if self.assessor_task is RevisionAssessorTask.ENTAILMENT
                else self.raw_falsification_result
            )
            other_result = (
                self.raw_falsification_result
                if self.assessor_task is RevisionAssessorTask.ENTAILMENT
                else self.raw_entailment_result
            )
            if raw_result is None or other_result is not None:
                raise ValueError(
                    "assessment must carry exactly its task-owned raw result"
                )
            if revision_raw_result_digest(raw_result) != self.raw_result_digest:
                raise ValueError(
                    "raw assessment digest must match the task-owned result"
                )
        revision_fields = (
            self.baseline_label,
            self.proposed_label,
            self.boundary_card_id,
            self.contradicted_baseline_condition,
        )
        if self.verdict is not RevisionAssessmentVerdict.REVISE:
            if any(value is not None for value in revision_fields) or any(
                (
                    self.satisfied_proposed_conditions,
                    self.supporting_evidence_ids,
                    self.counter_evidence_ids,
                )
            ):
                raise ValueError(
                    "preserve and insufficient assessments cannot carry revision fields"
                )
            return self
        if (
            any(value is None for value in revision_fields)
            or not self.satisfied_proposed_conditions
            or not self.supporting_evidence_ids
            or (self.proposal_digest is not None and not self.counter_evidence_ids)
        ):
            raise ValueError(
                "candidate-bound revise assessment requires replacement, Boundary "
                "Card, conditions, proposed-support citations, and Baseline-"
                "contradiction citations"
            )
        if self.baseline_label == self.proposed_label:
            raise ValueError(
                "baseline revision assessment must propose a different label"
            )
        for field_name in (
            "satisfied_proposed_conditions",
            "supporting_evidence_ids",
            "counter_evidence_ids",
        ):
            values = getattr(self, field_name)
            if len(values) != len(set(values)) or any(not value for value in values):
                raise ValueError(f"{field_name} must be unique and non-empty")
        return self


def _ordered_unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def normalize_revision_entailment_result(
    *,
    result: RevisionEntailmentResult,
    proposal: RevisionProposalEnvelope,
    routed_card: BoundaryCard,
) -> BaselineRevisionAssessment:
    """Validate and bind Team A raw entailment to an opaque canonical assessment."""

    criteria = {criterion.label: criterion for criterion in routed_card.criteria}
    proposed = criteria[proposal.proposed_label]
    baseline = criteria[proposal.baseline_label]
    if tuple(finding.condition for finding in result.condition_findings) != (
        proposed.positive_conditions
    ):
        raise ValueError("entailment must cover every proposed card condition exactly")
    if result.baseline_exclusion_finding.condition not in baseline.exclusion_conditions:
        raise ValueError("entailment must check one exact Baseline exclusion condition")
    owned = {*proposal.supporting_evidence_ids, *proposal.counter_evidence_ids}
    all_citations = {
        citation
        for finding in (*result.condition_findings, result.baseline_exclusion_finding)
        for citation in finding.citation_ids
    }
    if not all_citations.issubset(owned):
        raise ValueError("entailment citations must belong to the exact proposal view")
    if any(
        not set(finding.citation_ids).issubset(proposal.supporting_evidence_ids)
        for finding in result.condition_findings
    ):
        raise ValueError("counter-only evidence cannot positively entail a condition")
    revise = result.outcome is RevisionEntailmentOutcome.ENTAILED
    return BaselineRevisionAssessment(
        assessor_team_id="A",
        proposal_digest=revision_proposal_digest(proposal),
        dimension=proposal.dimension,
        verdict=(
            RevisionAssessmentVerdict.REVISE
            if revise
            else RevisionAssessmentVerdict.INSUFFICIENT
        ),
        baseline_label=proposal.baseline_label if revise else None,
        proposed_label=proposal.proposed_label if revise else None,
        boundary_card_id=proposal.boundary_card_id if revise else None,
        contradicted_baseline_condition=(
            result.baseline_exclusion_finding.condition if revise else None
        ),
        satisfied_proposed_conditions=(
            tuple(finding.condition for finding in result.condition_findings)
            if revise
            else ()
        ),
        supporting_evidence_ids=(
            _ordered_unique(
                citation
                for finding in result.condition_findings
                for citation in finding.citation_ids
            )
            if revise
            else ()
        ),
        counter_evidence_ids=(
            result.baseline_exclusion_finding.citation_ids if revise else ()
        ),
        baseline_source_config_hash=proposal.baseline_source_config_hash,
        baseline_source_predictions_sha256=proposal.baseline_source_predictions_sha256,
        taxonomy_structure_hash=proposal.taxonomy_structure_hash,
        evidence_view_hash=proposal.evidence_view_hash,
        assessor_task=RevisionAssessorTask.ENTAILMENT,
        raw_result_digest=revision_raw_result_digest(result),
        raw_entailment_result=result,
    )


def normalize_revision_falsification_result(
    *,
    result: RevisionFalsificationResult,
    proposal: RevisionProposalEnvelope,
    routed_card: BoundaryCard,
) -> BaselineRevisionAssessment:
    """Validate and bind Team B raw falsification to an opaque canonical assessment."""

    criteria = {criterion.label: criterion for criterion in routed_card.criteria}
    proposed = criteria[proposal.proposed_label]
    baseline = criteria[proposal.baseline_label]
    if result.baseline_survival_finding.condition not in baseline.positive_conditions:
        raise ValueError(
            "falsification must check one exact Baseline survival condition"
        )
    if tuple(finding.condition for finding in result.proposed_defeater_findings) != (
        proposed.exclusion_conditions
    ):
        raise ValueError("falsification must cover every proposed defeater exactly")
    if result.strongest_competing_reading.label != proposal.baseline_label:
        raise ValueError("strongest competing reading must be the exact Baseline label")
    owned = {*proposal.supporting_evidence_ids, *proposal.counter_evidence_ids}
    all_citations = {
        *result.baseline_survival_finding.citation_ids,
        *result.strongest_competing_reading.citation_ids,
        *(
            citation
            for finding in result.proposed_defeater_findings
            for citation in finding.citation_ids
        ),
    }
    if not all_citations.issubset(owned):
        raise ValueError(
            "falsification citations must belong to the exact proposal view"
        )
    revise = result.outcome is RevisionFalsificationOutcome.REVISION_SURVIVES
    return BaselineRevisionAssessment(
        assessor_team_id="B",
        proposal_digest=revision_proposal_digest(proposal),
        dimension=proposal.dimension,
        verdict=(
            RevisionAssessmentVerdict.REVISE
            if revise
            else (
                RevisionAssessmentVerdict.PRESERVE
                if result.outcome is RevisionFalsificationOutcome.REVISION_FALSIFIED
                else RevisionAssessmentVerdict.INSUFFICIENT
            )
        ),
        baseline_label=proposal.baseline_label if revise else None,
        proposed_label=proposal.proposed_label if revise else None,
        boundary_card_id=proposal.boundary_card_id if revise else None,
        contradicted_baseline_condition=(
            baseline.exclusion_conditions[0] if revise else None
        ),
        satisfied_proposed_conditions=(proposed.positive_conditions if revise else ()),
        supporting_evidence_ids=(
            _ordered_unique(
                (
                    *result.strongest_competing_reading.citation_ids,
                    *(
                        citation
                        for finding in result.proposed_defeater_findings
                        for citation in finding.citation_ids
                    ),
                )
            )
            if revise
            else ()
        ),
        counter_evidence_ids=(
            result.baseline_survival_finding.citation_ids if revise else ()
        ),
        baseline_source_config_hash=proposal.baseline_source_config_hash,
        baseline_source_predictions_sha256=proposal.baseline_source_predictions_sha256,
        taxonomy_structure_hash=proposal.taxonomy_structure_hash,
        evidence_view_hash=proposal.evidence_view_hash,
        assessor_task=RevisionAssessorTask.FALSIFICATION,
        raw_result_digest=revision_raw_result_digest(result),
        raw_falsification_result=result,
    )


class CrossFindingStatus(str, Enum):
    CONFIRMED = "confirmed"
    DISPUTED = "disputed"
    INSUFFICIENT = "insufficient"


class CrossConditionFinding(FrozenStrictModel):
    condition: str = Field(min_length=1)
    status: CrossFindingStatus
    citation_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def citations_are_unique(self) -> "CrossConditionFinding":
        if len(self.citation_ids) != len(set(self.citation_ids)) or any(
            not value for value in self.citation_ids
        ):
            raise ValueError("cross finding citations must be unique and non-empty")
        return self


class CrossCompetingReadingFinding(FrozenStrictModel):
    label: str = Field(min_length=1)
    status: CrossFindingStatus
    citation_ids: tuple[str, ...] = Field(min_length=1)


class EntailmentMappingCrossResult(FrozenStrictModel):
    """Team B's model-visible audit of Team A's entailment mapping."""

    condition_findings: tuple[CrossConditionFinding, ...] = Field(min_length=1)
    baseline_exclusion_finding: CrossConditionFinding
    status: ConsistencyStatus
    rationale: str = Field(min_length=20, max_length=800)

    @model_validator(mode="after")
    def status_matches_findings(self) -> "EntailmentMappingCrossResult":
        statuses = tuple(
            finding.status
            for finding in (*self.condition_findings, self.baseline_exclusion_finding)
        )
        if self.status is ConsistencyStatus.CONSISTENT and any(
            status is not CrossFindingStatus.CONFIRMED for status in statuses
        ):
            raise ValueError("consistent entailment cross requires confirmed findings")
        return self


class FalsificationCoverageCrossResult(FrozenStrictModel):
    """Team A's model-visible audit of Team B's falsification coverage."""

    baseline_survival_finding: CrossConditionFinding
    proposed_defeater_findings: tuple[CrossConditionFinding, ...] = Field(min_length=1)
    strongest_competing_reading_finding: CrossCompetingReadingFinding
    status: ConsistencyStatus
    rationale: str = Field(min_length=20, max_length=800)

    @model_validator(mode="after")
    def status_matches_findings(self) -> "FalsificationCoverageCrossResult":
        statuses = (
            self.baseline_survival_finding.status,
            *(finding.status for finding in self.proposed_defeater_findings),
            self.strongest_competing_reading_finding.status,
        )
        if self.status is ConsistencyStatus.CONSISTENT and any(
            status is not CrossFindingStatus.CONFIRMED for status in statuses
        ):
            raise ValueError(
                "consistent falsification cross requires confirmed findings"
            )
        return self


class RevisionCrossCheckResult(FrozenStrictModel):
    """Legacy shared cross result retained only for JSON migration compatibility."""

    status: ConsistencyStatus
    rationale: str = Field(min_length=20, max_length=800)
    supporting_evidence_ids: tuple[str, ...] = Field(min_length=1)
    counter_evidence_ids: tuple[str, ...] = ()


class RevisionConsistencyReport(FrozenStrictModel):
    """A team-local causal check bound to one requested Baseline revision."""

    checker_team_id: str | None = Field(default=None, min_length=1)
    assessment_owner_team_id: str | None = Field(default=None, min_length=1)
    proposal_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    dimension: EvidenceDimension
    baseline_label: str = Field(min_length=1)
    proposed_label: str = Field(min_length=1)
    status: ConsistencyStatus
    rationale: str = Field(min_length=20)
    supporting_evidence_ids: tuple[str, ...] = Field(min_length=1)
    counter_evidence_ids: tuple[str, ...] = ()
    assessment_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    assessment_owner_task: RevisionAssessorTask | None = None
    assessment_raw_result_digest: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    boundary_card_id: str = Field(min_length=1)
    baseline_source_config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    baseline_source_predictions_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    taxonomy_structure_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_view_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_cross_check_identity(self) -> "RevisionConsistencyReport":
        if (self.assessment_owner_task is None) != (
            self.assessment_raw_result_digest is None
        ):
            raise ValueError("cross consistency raw owner provenance must be complete")
        cross_bindings = (
            self.checker_team_id,
            self.assessment_owner_team_id,
            self.proposal_digest,
        )
        if any(value is not None for value in cross_bindings):
            if any(value is None for value in cross_bindings):
                raise ValueError("cross consistency requires every identity binding")
            if self.checker_team_id not in {"A", "B"} or (
                self.assessment_owner_team_id not in {"A", "B"}
            ):
                raise ValueError("cross consistency team identities must be A or B")
            if self.checker_team_id == self.assessment_owner_team_id:
                raise ValueError("cross consistency cannot self-check an assessment")
        if len(self.counter_evidence_ids) != len(set(self.counter_evidence_ids)) or any(
            not value for value in self.counter_evidence_ids
        ):
            raise ValueError(
                "cross consistency counter evidence IDs must be unique and non-empty"
            )
        if self.proposal_digest is not None and not self.counter_evidence_ids:
            raise ValueError(
                "candidate-bound cross consistency requires Baseline-contradiction "
                "citations"
            )
        return self


def revision_consistency_report_digest(report: RevisionConsistencyReport) -> str:
    return hashlib.sha256(
        json.dumps(
            report.model_dump(mode="json", exclude_none=True),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


class RevisionAssessmentAuditAttempt(FrozenStrictModel):
    """Sanitized execution outcome for one proposal/team assessment call."""

    proposal_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    dimension: EvidenceDimension
    boundary_card_id: str = Field(min_length=1)
    assessor_team_id: Literal["A", "B"]
    returned_assessment_digest: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    returned_assessment: BaselineRevisionAssessment | None = None
    verdict: RevisionAssessmentVerdict | None = None
    operational_failure: bool

    @model_validator(mode="after")
    def return_or_failure_is_explicit(self) -> "RevisionAssessmentAuditAttempt":
        returned = self.returned_assessment_digest is not None
        if returned == self.operational_failure:
            raise ValueError(
                "assessment attempt must record exactly one return or operational failure"
            )
        if returned != (self.verdict is not None):
            raise ValueError(
                "returned assessment digest and verdict must appear together"
            )
        if returned != (self.returned_assessment is not None):
            raise ValueError("returned assessment and digest must appear together")
        if self.returned_assessment is not None:
            assessment = self.returned_assessment
            identity_mismatch = (
                baseline_revision_assessment_digest(assessment)
                != self.returned_assessment_digest
                or assessment.assessor_team_id != self.assessor_team_id
                or assessment.proposal_digest != self.proposal_digest
                or assessment.dimension != self.dimension
                or assessment.verdict != self.verdict
            )
            revision_route_mismatch = (
                assessment.verdict is RevisionAssessmentVerdict.REVISE
                and assessment.boundary_card_id != self.boundary_card_id
            )
            if identity_mismatch or revision_route_mismatch:
                raise ValueError(
                    "returned assessment must match its complete audit attempt"
                )
        return self


class RevisionConsistencyAuditAttempt(FrozenStrictModel):
    """Sanitized execution outcome for one cross-team consistency call."""

    proposal_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    dimension: EvidenceDimension
    boundary_card_id: str = Field(min_length=1)
    checker_team_id: Literal["A", "B"]
    assessment_owner_team_id: Literal["A", "B"]
    returned_consistency_digest: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    returned_consistency: RevisionConsistencyReport | None = None
    status: ConsistencyStatus | None = None
    operational_failure: bool

    @model_validator(mode="after")
    def return_or_failure_is_explicit(self) -> "RevisionConsistencyAuditAttempt":
        if self.checker_team_id == self.assessment_owner_team_id:
            raise ValueError("cross consistency audit cannot self-check an assessment")
        returned = self.returned_consistency_digest is not None
        if returned == self.operational_failure:
            raise ValueError(
                "consistency attempt must record exactly one return or operational failure"
            )
        if returned != (self.status is not None):
            raise ValueError(
                "returned consistency digest and status must appear together"
            )
        if returned != (self.returned_consistency is not None):
            raise ValueError(
                "returned consistency report and digest must appear together"
            )
        if self.returned_consistency is not None and (
            revision_consistency_report_digest(self.returned_consistency)
            != self.returned_consistency_digest
            or self.returned_consistency.checker_team_id != self.checker_team_id
            or self.returned_consistency.assessment_owner_team_id
            != self.assessment_owner_team_id
            or self.returned_consistency.proposal_digest != self.proposal_digest
            or self.returned_consistency.dimension != self.dimension
            or self.returned_consistency.boundary_card_id != self.boundary_card_id
            or self.returned_consistency.status != self.status
        ):
            raise ValueError(
                "returned consistency report must match its complete audit attempt"
            )
        return self


class RevisionProposalAuditEntry(FrozenStrictModel):
    """Canonical proposal identity used to verify every downstream audit edge."""

    proposal_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    dimension: EvidenceDimension
    boundary_card_id: str = Field(min_length=1)
    baseline_label: str = Field(min_length=1)
    proposed_label: str = Field(min_length=1)

    @model_validator(mode="after")
    def owns_one_real_revision(self) -> "RevisionProposalAuditEntry":
        if self.dimension not in {
            EvidenceDimension.SYMPTOM,
            EvidenceDimension.ROOT_CAUSE,
        }:
            raise ValueError(
                "revision proposal audit entry must own a Stage 3 dimension"
            )
        if self.baseline_label == self.proposed_label:
            raise ValueError("revision proposal audit entry must change the label")
        return self


class RevisionCertificateAuditEntry(FrozenStrictModel):
    """Exact proposal route claimed by one issued certificate digest."""

    certificate_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    proposal_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    dimension: EvidenceDimension
    boundary_card_id: str = Field(min_length=1)
    baseline_label: str = Field(min_length=1)
    proposed_label: str = Field(min_length=1)


class Stage3RevisionAudit(FrozenStrictModel):
    """Canonical, deeply frozen audit of the complete preservation revision funnel."""

    assessment_policy_version: Literal[
        "legacy-homogeneous-v1", "heterogeneous-entailment-falsification-v1"
    ] = "legacy-homogeneous-v1"
    cross_policy_version: Literal[
        "legacy-shared-cross-v1", "opposite-task-specific-raw-bound-v3"
    ] = "legacy-shared-cross-v1"
    proposal_digests: tuple[str, ...] = ()
    routed_card_ids: tuple[str, ...] = ()
    proposal_entries: tuple[RevisionProposalAuditEntry, ...] = ()
    assessment_attempts: tuple[RevisionAssessmentAuditAttempt, ...] = ()
    dual_revise_proposal_digests: tuple[str, ...] = ()
    consistency_attempts: tuple[RevisionConsistencyAuditAttempt, ...] = ()
    consistency_pass_proposal_digests: tuple[str, ...] = ()
    issued_certificate_digests: tuple[str, ...] = ()
    issued_certificate_entries: tuple[RevisionCertificateAuditEntry, ...] = ()
    applied_certificate_digests: tuple[str, ...] = ()
    ambiguous_certified_dimensions: tuple[EvidenceDimension, ...] = ()
    failed_proposal_digests: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_revision_funnel(self) -> "Stage3RevisionAudit":
        returned_assessments = tuple(
            attempt.returned_assessment
            for attempt in self.assessment_attempts
            if attempt.returned_assessment is not None
        )
        if (
            self.assessment_policy_version
            == "heterogeneous-entailment-falsification-v1"
        ):
            if self.cross_policy_version != "opposite-task-specific-raw-bound-v3":
                raise ValueError(
                    "heterogeneous assessment policy requires task-specific cross policy"
                )
            if any(
                assessment.assessor_task is None or assessment.raw_result_digest is None
                for assessment in returned_assessments
            ):
                raise ValueError(
                    "heterogeneous assessment policy requires raw task provenance"
                )
        elif any(
            assessment.assessor_task is not None for assessment in returned_assessments
        ):
            raise ValueError(
                "legacy assessment policy cannot contain heterogeneous raw results"
            )
        digest_fields = (
            "proposal_digests",
            "dual_revise_proposal_digests",
            "consistency_pass_proposal_digests",
            "issued_certificate_digests",
            "applied_certificate_digests",
            "failed_proposal_digests",
        )
        for field_name in digest_fields:
            values = getattr(self, field_name)
            if len(values) != len(set(values)):
                raise ValueError(f"{field_name} must be unique")
            if any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in values):
                raise ValueError(f"{field_name} must contain canonical sha256 digests")
        if len(self.routed_card_ids) != len(set(self.routed_card_ids)) or any(
            not card_id for card_id in self.routed_card_ids
        ):
            raise ValueError("routed card IDs must be unique and non-empty")
        if len(self.proposal_digests) != len(self.routed_card_ids):
            raise ValueError("every proposal must have exactly one routed card")
        if len(self.proposal_entries) != len(self.proposal_digests):
            raise ValueError(
                "every proposal must have exactly one canonical proposal entry"
            )
        if tuple(entry.proposal_digest for entry in self.proposal_entries) != (
            self.proposal_digests
        ) or tuple(entry.boundary_card_id for entry in self.proposal_entries) != (
            self.routed_card_ids
        ):
            raise ValueError(
                "proposal digests and routed cards must match canonical proposal entries"
            )
        proposals = set(self.proposal_digests)
        entries = {entry.proposal_digest: entry for entry in self.proposal_entries}
        referenced = {
            attempt.proposal_digest for attempt in self.assessment_attempts
        } | {attempt.proposal_digest for attempt in self.consistency_attempts}
        referenced |= set(self.dual_revise_proposal_digests)
        referenced |= set(self.consistency_pass_proposal_digests)
        referenced |= set(self.failed_proposal_digests)
        if not referenced.issubset(proposals):
            raise ValueError(
                "every revision audit attempt must reference a listed proposal"
            )
        for attempt in (*self.assessment_attempts, *self.consistency_attempts):
            entry = entries.get(attempt.proposal_digest)
            if entry is None or (
                attempt.dimension != entry.dimension
                or attempt.boundary_card_id != entry.boundary_card_id
            ):
                raise ValueError(
                    "revision attempt must match its canonical proposal entry"
                )
            returned = (
                attempt.returned_assessment
                if isinstance(attempt, RevisionAssessmentAuditAttempt)
                else attempt.returned_consistency
            )
            carries_revision_pair = (
                not isinstance(attempt, RevisionAssessmentAuditAttempt)
                or attempt.verdict is RevisionAssessmentVerdict.REVISE
            )
            if returned is not None and (
                carries_revision_pair
                and (
                    returned.baseline_label != entry.baseline_label
                    or returned.proposed_label != entry.proposed_label
                )
            ):
                raise ValueError(
                    "returned revision record must match proposal entry label pair"
                )
        assessment_keys = [
            (attempt.proposal_digest, attempt.assessor_team_id)
            for attempt in self.assessment_attempts
        ]
        if len(assessment_keys) != len(set(assessment_keys)):
            raise ValueError("assessment attempts cannot duplicate proposal/team")
        consistency_keys = [
            (attempt.proposal_digest, attempt.checker_team_id)
            for attempt in self.consistency_attempts
        ]
        if len(consistency_keys) != len(set(consistency_keys)):
            raise ValueError("consistency attempts cannot duplicate proposal/checker")
        returned_assessment_digests = {
            (attempt.proposal_digest, attempt.assessor_team_id): (
                attempt.returned_assessment_digest
            )
            for attempt in self.assessment_attempts
            if not attempt.operational_failure
        }
        returned_assessment_records = {
            (attempt.proposal_digest, attempt.assessor_team_id): (
                attempt.returned_assessment
            )
            for attempt in self.assessment_attempts
            if attempt.returned_assessment is not None
        }
        for attempt in self.consistency_attempts:
            owner_assessment = returned_assessment_records.get(
                (attempt.proposal_digest, attempt.assessment_owner_team_id)
            )
            if attempt.returned_consistency is not None and (
                attempt.returned_consistency.assessment_digest
                != returned_assessment_digests.get(
                    (attempt.proposal_digest, attempt.assessment_owner_team_id)
                )
                or (
                    attempt.returned_consistency.assessment_owner_task,
                    attempt.returned_consistency.assessment_raw_result_digest,
                )
                != (
                    getattr(owner_assessment, "assessor_task", None),
                    getattr(owner_assessment, "raw_result_digest", None),
                )
            ):
                raise ValueError(
                    "cross consistency must bind the audited owner assessment digest"
                )
        derived_dual_revise = tuple(
            digest
            for digest in self.proposal_digests
            if {
                attempt.assessor_team_id
                for attempt in self.assessment_attempts
                if attempt.proposal_digest == digest
                and not attempt.operational_failure
                and attempt.verdict is RevisionAssessmentVerdict.REVISE
            }
            == {"A", "B"}
        )
        if self.dual_revise_proposal_digests != derived_dual_revise:
            raise ValueError(
                "dual revise proposals must be derived from returned A/B revise assessments"
            )
        if any(
            attempt.proposal_digest not in set(derived_dual_revise)
            for attempt in self.consistency_attempts
        ):
            raise ValueError(
                "cross consistency attempts require a dual revise proposal"
            )
        derived_consistency_pass = tuple(
            digest
            for digest in self.proposal_digests
            if {
                (attempt.checker_team_id, attempt.assessment_owner_team_id)
                for attempt in self.consistency_attempts
                if attempt.proposal_digest == digest
                and not attempt.operational_failure
                and attempt.status is ConsistencyStatus.CONSISTENT
            }
            == {("A", "B"), ("B", "A")}
        )
        if self.consistency_pass_proposal_digests != derived_consistency_pass:
            raise ValueError(
                "consistency pass proposals must be derived from both opposite cross checks"
            )
        derived_failed = tuple(
            digest
            for digest in self.proposal_digests
            if any(
                attempt.proposal_digest == digest and attempt.operational_failure
                for attempt in (*self.assessment_attempts, *self.consistency_attempts)
            )
        )
        if self.failed_proposal_digests != derived_failed:
            raise ValueError(
                "failed proposal digests must exactly identify operational failures"
            )
        if set(derived_failed) & set(derived_consistency_pass):
            raise ValueError("failed proposal cannot also have a consistency pass")
        if (
            tuple(entry.certificate_digest for entry in self.issued_certificate_entries)
            != self.issued_certificate_digests
        ):
            raise ValueError(
                "issued certificate digests must match certificate audit entries"
            )
        if len(self.issued_certificate_entries) != len(
            {entry.certificate_digest for entry in self.issued_certificate_entries}
        ):
            raise ValueError("issued certificate audit entries must be unique")
        for certificate in self.issued_certificate_entries:
            proposal = entries.get(certificate.proposal_digest)
            if certificate.proposal_digest not in set(derived_consistency_pass):
                raise ValueError("issued certificate requires a consistency pass")
            if proposal is None or (
                certificate.dimension != proposal.dimension
                or certificate.boundary_card_id != proposal.boundary_card_id
                or certificate.baseline_label != proposal.baseline_label
                or certificate.proposed_label != proposal.proposed_label
            ):
                raise ValueError(
                    "issued certificate must match the complete canonical proposal chain"
                )
        if not set(self.applied_certificate_digests).issubset(
            self.issued_certificate_digests
        ):
            raise ValueError(
                "applied certificates must be a subset of issued certificates"
            )
        if any(
            dimension not in {EvidenceDimension.SYMPTOM, EvidenceDimension.ROOT_CAUSE}
            for dimension in self.ambiguous_certified_dimensions
        ) or len(self.ambiguous_certified_dimensions) != len(
            set(self.ambiguous_certified_dimensions)
        ):
            raise ValueError(
                "ambiguous certified dimensions must be unique Stage 3 dimensions"
            )
        derived_ambiguous = tuple(
            dimension
            for dimension in (EvidenceDimension.SYMPTOM, EvidenceDimension.ROOT_CAUSE)
            if sum(
                entry.dimension is dimension
                and entry.proposal_digest in set(derived_consistency_pass)
                for entry in self.proposal_entries
            )
            > 1
        )
        if self.ambiguous_certified_dimensions != derived_ambiguous:
            raise ValueError(
                "ambiguous certified dimensions must be derived from multiple consistency passes"
            )
        return self


class BaselinePreservationResult(FrozenStrictModel):
    """Decision labels plus an explicit per-dimension preservation audit."""

    record_id: str = Field(min_length=1)
    symptom_label: str = Field(min_length=1)
    root_cause_label: str = Field(min_length=1)
    symptom_action: BaselineDecisionAction
    root_cause_action: BaselineDecisionAction
    applied_certificates: tuple[LabelRevisionCertificate, ...] = ()
    baseline_source_config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    baseline_source_predictions_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_baseline_anchor_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_taxonomy_structure_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class SpecialistType(str, Enum):
    COMMIT_HISTORY = "commit_history"
    ISSUE_PR = "issue_pr"
    CODE_CONTEXT = "code_context"
    TEST_EVIDENCE = "test_evidence"
    SIMILAR_CASE = "similar_case"
    TAXONOMY_KNOWLEDGE = "taxonomy_knowledge"


class RetrievalStatus(str, Enum):
    FOUND = "found"
    ABSENT = "absent"
    UNAVAILABLE = "unavailable"
    INVALID = "invalid"


class Stage2Decision(str, Enum):
    ACCEPTED = "accepted_fault"
    REJECTED = "rejected_candidate"


class TestOutcome(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


class FaultEvidenceAssessment(StrictModel):
    """Role-owned outcome for the fault-existence test only."""

    outcome: TestOutcome
    claim: str = Field(min_length=1)
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    counter_evidence_ids: list[str] = Field(default_factory=list)


class ScopeBoundaryAssessment(StrictModel):
    """Role-owned outcome for the paper-specific study-scope test only."""

    outcome: TestOutcome
    claim: str = Field(min_length=1)
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    counter_evidence_ids: list[str] = Field(default_factory=list)


class RepairCausalityAssessment(StrictModel):
    """Role-owned outcome for the repair-causality test only."""

    outcome: TestOutcome
    claim: str = Field(min_length=1)
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    counter_evidence_ids: list[str] = Field(default_factory=list)


class Stage2RoleAssessments(StrictModel):
    """Typed role results; domain policy decides whether repair is required."""

    fault_evidence: FaultEvidenceAssessment
    scope_boundary: ScopeBoundaryAssessment
    repair_causality: RepairCausalityAssessment | None = None


class Stage2EvidenceTests(TypedDict):
    """Named Stage 2 tests so the JSON contract exposes every required key."""

    fault_existence: TestOutcome
    repair_causality: TestOutcome
    scope_exclusion: TestOutcome


class EvidenceSufficiency(str, Enum):
    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"


class ConsistencyStatus(str, Enum):
    CONSISTENT = "consistent"
    SYMPTOM_REVIEW = "symptom_review"
    CAUSE_REVIEW = "cause_review"
    EVIDENCE_REQUEST = "evidence_request"


class ArbitrationSource(str, Enum):
    DIRECT_CONSENSUS = "direct_consensus"
    TARGETED_ARBITRATION = "targeted_arbitration"
    FALLBACK_UNCERTAIN = "fallback_uncertain"
    SINGLE_TEAM_DEGRADED = "single_team_degraded"
    DETERMINISTIC_SCOPE_GATE = "deterministic_scope_gate"
    POLICY_COMPOSITION = "policy_composition"
    BASELINE_PRESERVATION_GATE = "baseline_preservation_gate"
    SLA_FALLBACK = "sla_fallback"


class EvidenceItem(StrictModel):
    """One immutable, source-addressable item in an evidence ledger."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str = Field(min_length=1)
    record_id: str = Field(min_length=1)
    source_type: str = Field(min_length=1)
    source_uri: str = Field(min_length=1)
    retrieved_at: str = Field(min_length=1)
    content: str = Field(min_length=1)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    explicitness: EvidenceExplicitness
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Recursively JSON-compatible metadata: null, bool, int, finite "
            "float, string, arrays, and string-keyed objects only."
        ),
    )

    @field_validator("metadata", mode="before")
    @classmethod
    def metadata_is_json_value(cls, value: Any) -> Any:
        _validate_json_value(value, path="metadata")
        return value

    @model_validator(mode="after")
    def content_hash_matches(self) -> "EvidenceItem":
        observed = hashlib.sha256(self.content.encode("utf-8")).hexdigest()
        if self.content_sha256 != observed:
            raise ValueError("content_sha256 does not match content")
        _validate_json_value(self.metadata, path="metadata")
        capabilities = self.metadata.get("evidence_capabilities")
        if capabilities is not None:
            allowed_capabilities = {
                "symptom_observation",
                "defect_mechanism",
                "study_scope",
            }
            if type(capabilities) not in (list, tuple) or not capabilities:
                raise ValueError(
                    "metadata.evidence_capabilities must be a non-empty array"
                )
            if any(
                type(capability) is not str or capability not in allowed_capabilities
                for capability in capabilities
            ):
                raise ValueError(
                    "metadata.evidence_capabilities contains an unknown capability"
                )
            if len(capabilities) != len(set(capabilities)):
                raise ValueError(
                    "metadata.evidence_capabilities cannot contain duplicates"
                )
        for flag in (
            "observable_impact",
            "mechanism_evidence",
            "study_scope_evidence",
        ):
            if flag in self.metadata and type(self.metadata[flag]) is not bool:
                raise ValueError(f"metadata.{flag} must be a boolean")
        object.__setattr__(self, "metadata", _deep_freeze(self.metadata))
        return self

    @field_serializer("metadata")
    def serialize_metadata(self, value: Mapping[str, Any]) -> dict[str, Any]:
        return _deep_thaw(value)

    def __deepcopy__(self, memo: dict[int, object] | None = None) -> "EvidenceItem":
        copied = type(self).model_validate(self.model_dump(mode="python"))
        if memo is not None:
            memo[id(self)] = copied
        return copied


class EvidenceView(StrictModel):
    """A role-specific, versioned view of exact ledger items."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    record_id: str
    task: str
    taxonomy: dict[str, list[str] | dict[str, str]]
    domain_profile: str
    ledger_version: int = Field(ge=0)
    items: tuple[EvidenceItem, ...]

    @model_validator(mode="after")
    def freeze_nested_collections(self) -> "EvidenceView":
        object.__setattr__(self, "taxonomy", _deep_freeze(self.taxonomy))
        rebuilt_items: list[EvidenceItem] = []
        for item in self.items:
            _validate_json_value(item.metadata, path="metadata")
            rebuilt_items.append(
                EvidenceItem.model_validate(item.model_dump(mode="python"))
            )
        object.__setattr__(
            self,
            "items",
            tuple(rebuilt_items),
        )
        return self

    @field_serializer("taxonomy")
    def serialize_taxonomy(
        self, value: Mapping[str, Any]
    ) -> dict[str, list[str] | dict[str, str]]:
        return _deep_thaw(value)

    def __deepcopy__(self, memo: dict[int, object] | None = None) -> "EvidenceView":
        copied = type(self).model_validate(self.model_dump(mode="python"))
        if memo is not None:
            memo[id(self)] = copied
        return copied


def canonical_evidence_view_hash(view: EvidenceView) -> str:
    """Hash the exact record/task/ledger evidence snapshot used for a revision."""

    payload = {
        "record_id": view.record_id,
        "task": view.task,
        "domain_profile": view.domain_profile,
        "taxonomy": view.model_dump(mode="json")["taxonomy"],
        "ledger_version": view.ledger_version,
        "items": [
            {
                "evidence_id": item.evidence_id,
                "record_id": item.record_id,
                "source_type": item.source_type,
                "source_uri": item.source_uri,
                "retrieved_at": item.retrieved_at,
                "content_sha256": item.content_sha256,
                "explicitness": item.explicitness.value,
                "metadata": item.model_dump(mode="json")["metadata"],
            }
            for item in view.items
        ],
    }
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def baseline_revision_assessment_digest(assessment: BaselineRevisionAssessment) -> str:
    return hashlib.sha256(
        json.dumps(
            assessment.model_dump(mode="json", exclude_none=True),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


class EvidenceValidityIssue(FrozenStrictModel):
    """Deterministic reason an evidence item cannot enter Stage 3."""

    evidence_id: str = Field(min_length=1)
    reasons: tuple[str, ...] = Field(min_length=1)


class EvidenceValidityReport(FrozenStrictModel):
    """Canonical Stage 3 evidence admission outcome."""

    task: Literal["stage3"] = "stage3"
    valid_evidence_ids: tuple[str, ...]
    quarantined: tuple[EvidenceValidityIssue, ...] = ()
    capabilities_by_evidence_id: Mapping[str, tuple[str, ...]]

    @model_validator(mode="after")
    def ids_are_disjoint_and_unique(self) -> "EvidenceValidityReport":
        valid = self.valid_evidence_ids
        rejected = tuple(issue.evidence_id for issue in self.quarantined)
        if len(valid) != len(set(valid)) or len(rejected) != len(set(rejected)):
            raise ValueError("validity evidence ids must be unique")
        if set(valid) & set(rejected):
            raise ValueError("valid and quarantined evidence ids must be disjoint")
        if set(self.capabilities_by_evidence_id) != set(valid):
            raise ValueError("capability ids must match valid evidence ids")
        if any(
            not capabilities
            for capabilities in self.capabilities_by_evidence_id.values()
        ):
            raise ValueError("capability tuples must not be empty")
        object.__setattr__(
            self,
            "capabilities_by_evidence_id",
            _deep_freeze(self.capabilities_by_evidence_id),
        )
        return self

    @field_serializer("capabilities_by_evidence_id")
    def serialize_capabilities(
        self, value: Mapping[str, tuple[str, ...]]
    ) -> dict[str, list[str]]:
        return _deep_thaw(value)


class EvidenceRequest(FrozenStrictModel):
    """A bounded request for one fact that can discriminate decisions."""

    request_id: str = Field(min_length=1)
    missing_fact: str = Field(min_length=20)
    why_needed: str = Field(min_length=20)
    target_specialist: SpecialistType
    target_source: str = Field(min_length=3)
    query: str = Field(min_length=3)
    expected_decision_impact: str = Field(min_length=40)
    max_items: int = Field(ge=1, le=20)

    @model_validator(mode="after")
    def request_is_specific(self) -> "EvidenceRequest":
        normalized = " ".join(self.missing_fact.lower().split())
        vague_phrases = (
            "more context",
            "more information",
            "get more",
            "additional context",
            "additional information",
        )
        if any(phrase in normalized for phrase in vague_phrases):
            raise ValueError("missing_fact must name a specific fact")
        return self


class DimensionReadiness(FrozenStrictModel):
    """Evidence sufficiency for one decision-relevant dimension."""

    dimension: EvidenceDimension
    sufficient: bool
    confirmed_evidence_ids: tuple[str, ...]
    missing_facts: tuple[str, ...]
    evidence_requests: tuple[EvidenceRequest, ...]

    @model_validator(mode="after")
    def validate_readiness(self) -> "DimensionReadiness":
        if self.sufficient and (self.missing_facts or self.evidence_requests):
            raise ValueError(
                "sufficient dimensions cannot retain missing facts or evidence requests"
            )
        if self.sufficient and not self.confirmed_evidence_ids:
            raise ValueError("sufficient dimensions must cite confirmed evidence")
        if not self.sufficient and not self.missing_facts:
            raise ValueError(
                "insufficient dimensions must explain at least one missing fact"
            )
        return self


class EvidenceReadinessReport(FrozenStrictModel):
    """Readiness status across all decision-relevant dimensions."""

    task: str = Field(min_length=1)
    dimensions: tuple[DimensionReadiness, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def dimensions_are_unique(self) -> "EvidenceReadinessReport":
        dimensions = [readiness.dimension for readiness in self.dimensions]
        if len(dimensions) != len(set(dimensions)):
            raise ValueError("readiness report cannot contain duplicate dimensions")
        return self


class ResolutionStatus(str, Enum):
    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"


class UnresolvedDecision(FrozenStrictModel):
    """A controlled stop when the available evidence cannot support a decision."""

    status: Literal[ResolutionStatus.UNRESOLVED]
    dimensions: tuple[EvidenceDimension | str, ...] = Field(min_length=1)
    missing_facts: tuple[str, ...] = Field(min_length=1)
    stop_reason: str = Field(min_length=1)
    attempted_retrieval_statuses: tuple[RetrievalStatus, ...] = Field(
        default_factory=tuple
    )


class EvidenceFact(FrozenStrictModel):
    """A specialist claim grounded only in returned evidence."""

    claim: str = Field(min_length=10)
    evidence_ids: tuple[str, ...] = Field(min_length=1)
    explicitness: EvidenceExplicitness


class EvidenceDelta(FrozenStrictModel):
    """Non-classifying evidence returned by a retrieval specialist."""

    request_id: str = Field(min_length=1)
    specialist: SpecialistType
    status: RetrievalStatus
    items: tuple[EvidenceItem, ...] = Field(default_factory=tuple)
    facts: tuple[EvidenceFact, ...] = Field(default_factory=tuple)
    counterfacts: tuple[EvidenceFact, ...] = Field(default_factory=tuple)
    diagnostics: Mapping[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_retrieval_result(self) -> "EvidenceDelta":
        if self.status is RetrievalStatus.FOUND and not self.items:
            raise ValueError("found retrieval must include at least one item")
        returned_ids = {item.evidence_id for item in self.items}
        referenced_ids = {
            evidence_id
            for fact in (*self.facts, *self.counterfacts)
            for evidence_id in fact.evidence_ids
        }
        if not referenced_ids.issubset(returned_ids):
            raise ValueError("facts must reference evidence returned in this delta")
        _validate_json_value(self.diagnostics, path="diagnostics")
        object.__setattr__(self, "diagnostics", _deep_freeze(self.diagnostics))
        return self

    @field_serializer("diagnostics")
    def serialize_diagnostics(self, value: Mapping[str, Any]) -> dict[str, Any]:
        return _deep_thaw(value)


@dataclass(frozen=True)
class Stage3EvidencePhaseResult:
    """Audit state from bounded Stage 3 retrieval and deterministic admission."""

    validity_report: EvidenceValidityReport
    gap_report: EvidenceReadinessReport
    evidence_deltas: tuple[EvidenceDelta, ...]
    retrieval_source_request_ids: tuple[tuple[str, ...], ...]
    budget_exhausted: bool
    final_ledger_version: int


class Stage2AnalysisReport(StrictModel):
    """Independent Stage 2 fault-verification result."""

    team_id: str = Field(min_length=1)
    decision: Stage2Decision
    confidence: float = Field(ge=0.0, le=1.0)
    fault_claim: str = Field(min_length=10)
    repair_claim: str = Field(min_length=10)
    supporting_evidence_ids: list[str] = Field(min_length=1)
    counter_evidence_ids: list[str] = Field(default_factory=list)
    evidence_tests: Stage2EvidenceTests
    alternative_hypothesis: str = Field(min_length=10)
    decision_boundary: str = Field(min_length=10)
    evidence_sufficiency: EvidenceSufficiency
    unresolved_evidence_gaps: list[str] = Field(default_factory=list)
    evidence_requests: list[EvidenceRequest] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_evidence_tests(self) -> "Stage2AnalysisReport":
        from .stage2_policy import stage2_acceptance_errors

        required = {"fault_existence", "repair_causality", "scope_exclusion"}
        if set(self.evidence_tests) != required:
            raise ValueError(
                "evidence_tests must contain fault_existence, "
                "repair_causality, and scope_exclusion"
            )
        if self.decision is Stage2Decision.ACCEPTED:
            errors = stage2_acceptance_errors(
                "ase2022",
                self.evidence_tests,
            )
            if errors:
                raise ValueError("; ".join(errors))
        return self


class IsstaStage2AnalysisReport(Stage2AnalysisReport):
    """Stage 2 report with the commit-repair gate required by ISSTA2024."""

    @model_validator(mode="after")
    def validate_issta_policy(self) -> "IsstaStage2AnalysisReport":
        from .stage2_policy import stage2_acceptance_errors

        if self.decision is Stage2Decision.ACCEPTED:
            errors = stage2_acceptance_errors(
                "issta2024",
                self.evidence_tests,
            )
            if errors:
                raise ValueError("; ".join(errors))
        return self


class SymptomReport(FrozenStrictModel):
    """Behavior-only Stage 3 symptom analysis."""

    label: str = Field(min_length=1)
    behavior_claim: str = Field(min_length=10)
    supporting_evidence_ids: tuple[str, ...] = Field(min_length=1)
    counter_evidence_ids: tuple[str, ...] = Field(default_factory=tuple)
    alternative_label: str = Field(min_length=1)
    boundary_reason: str = Field(min_length=10)
    boundary_evidence_ids: tuple[str, ...] = Field(default_factory=tuple)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_sufficiency: EvidenceSufficiency
    unresolved_evidence_gaps: tuple[str, ...] = Field(default_factory=tuple)
    evidence_requests: tuple[EvidenceRequest, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def citation_ids_are_unique(self) -> "SymptomReport":
        for field_name in (
            "supporting_evidence_ids",
            "counter_evidence_ids",
            "boundary_evidence_ids",
        ):
            _require_unique_evidence_ids(
                getattr(self, field_name), field_name=field_name
            )
        return self


class RootCauseReport(FrozenStrictModel):
    """Mechanism-only Stage 3 root-cause analysis."""

    label: str = Field(min_length=1)
    defect_mechanism: str = Field(min_length=10)
    causal_chain: tuple[str, ...] = Field(min_length=3)
    supporting_evidence_ids: tuple[str, ...] = Field(min_length=1)
    counter_evidence_ids: tuple[str, ...] = Field(default_factory=tuple)
    alternative_label: str = Field(min_length=1)
    boundary_reason: str = Field(min_length=10)
    boundary_evidence_ids: tuple[str, ...] = Field(default_factory=tuple)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_sufficiency: EvidenceSufficiency
    unresolved_evidence_gaps: tuple[str, ...] = Field(default_factory=tuple)
    evidence_requests: tuple[EvidenceRequest, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def citation_ids_are_unique(self) -> "RootCauseReport":
        for field_name in (
            "supporting_evidence_ids",
            "counter_evidence_ids",
            "boundary_evidence_ids",
        ):
            _require_unique_evidence_ids(
                getattr(self, field_name), field_name=field_name
            )
        return self


class SlaVerifierVerdict(str, Enum):
    """The bounded verdict an SLA verifier may return for one dimension."""

    ACCEPT_CANDIDATE = "accept_candidate"
    PRESERVE_BASELINE = "preserve_baseline"
    UNRESOLVED = "unresolved"


class SlaDecisionStatus(str, Enum):
    """Auditable outcome for one targeted SLA classification dimension."""

    MODEL_VERIFIED = "model_verified"
    ARBITRATED = "arbitrated"
    BASELINE_AGREEMENT = "baseline_agreement"
    TRANSPORT_FALLBACK = "transport_fallback"
    SCHEMA_FALLBACK = "schema_fallback"
    BUDGET_FALLBACK = "budget_fallback"
    UNRESOLVED_FALLBACK = "unresolved_fallback"


class SlaJointDiagnosis(FrozenStrictModel):
    """One fast-path diagnosis containing both independently auditable dimensions."""

    symptom: SymptomReport
    root_cause: RootCauseReport
    symptom_matches_baseline: bool
    root_cause_matches_baseline: bool


class SlaDimensionVerification(FrozenStrictModel):
    """A verifier's bounded disposition for exactly one classification dimension."""

    dimension: EvidenceDimension
    verdict: SlaVerifierVerdict
    supporting_evidence_ids: tuple[str, ...] = ()
    counter_evidence_ids: tuple[str, ...] = ()
    rationale: str = Field(min_length=10)

    @model_validator(mode="after")
    def citation_ids_are_disjoint_and_unique(self) -> "SlaDimensionVerification":
        _require_unique_evidence_ids(
            self.supporting_evidence_ids, field_name="supporting_evidence_ids"
        )
        _require_unique_evidence_ids(
            self.counter_evidence_ids, field_name="counter_evidence_ids"
        )
        if set(self.supporting_evidence_ids) & set(self.counter_evidence_ids):
            raise ValueError("supporting and counter evidence ids must be disjoint")
        return self


class SlaJointVerification(FrozenStrictModel):
    """A single SLA verifier response with one result per decision dimension."""

    symptom: SlaDimensionVerification
    root_cause: SlaDimensionVerification


class SlaDimensionArbitration(FrozenStrictModel):
    """One conflict-only choice with evidence owned by that dimension."""

    dimension: EvidenceDimension
    selected_label: str = Field(min_length=1)
    supporting_evidence_ids: tuple[str, ...] = Field(min_length=1)
    rationale: str = Field(min_length=10)

    @model_validator(mode="after")
    def supporting_ids_are_unique(self) -> "SlaDimensionArbitration":
        _require_unique_evidence_ids(
            self.supporting_evidence_ids, field_name="supporting_evidence_ids"
        )
        return self


class SlaConditionalArbitration(FrozenStrictModel):
    """Conflict-only selections between fixed candidate and Baseline pairs."""

    symptom: SlaDimensionArbitration | None = None
    root_cause: SlaDimensionArbitration | None = None

    @model_validator(mode="after")
    def at_least_one_conflict_is_resolved(self) -> "SlaConditionalArbitration":
        if self.symptom is None and self.root_cause is None:
            raise ValueError("SLA arbitration must resolve a conflicting dimension")
        return self


def _validate_sla_citations(
    evidence_ids: Sequence[str],
    *,
    view: EvidenceView,
    dimension: EvidenceDimension,
    owner: str,
) -> None:
    """Require citations from this exact, valid, dimension-capable evidence view."""

    from .evidence_capabilities import (
        assess_stage3_evidence_validity,
        supports_readiness_dimension,
    )

    valid_ids = set(assess_stage3_evidence_validity(view).valid_evidence_ids)
    by_id = {item.evidence_id: item for item in view.items}
    for evidence_id in evidence_ids:
        if evidence_id not in valid_ids:
            raise ValueError(
                f"{owner} citation {evidence_id!r} is not in the exact evidence view"
            )
        if not supports_readiness_dimension(
            by_id[evidence_id], dimension, domain=view.domain_profile
        ):
            raise ValueError(
                f"{owner} citation {evidence_id!r} is not capability-compatible with "
                f"{dimension.value}"
            )


def _validate_sla_label(*, label: str, view: EvidenceView, dimension: str) -> None:
    if not label_valid(view.taxonomy, dimension, label):
        raise ValueError(f"label {label!r} is outside the {dimension} taxonomy")


def validate_sla_joint_diagnosis(
    report: SlaJointDiagnosis,
    view: EvidenceView,
    baseline: BaselineAnchor | None = None,
) -> None:
    """Validate diagnosis labels and citations against the canonical SLA view."""

    _validate_sla_label(label=report.symptom.label, view=view, dimension="symptom")
    _validate_sla_label(
        label=report.root_cause.label, view=view, dimension="root_cause"
    )
    for owner, dimension, part in (
        ("SLA symptom diagnosis", EvidenceDimension.SYMPTOM, report.symptom),
        ("SLA root-cause diagnosis", EvidenceDimension.ROOT_CAUSE, report.root_cause),
    ):
        supporting = part.supporting_evidence_ids
        counter = part.counter_evidence_ids
        if set(supporting) & set(counter):
            raise ValueError(
                f"{owner} supporting and counter evidence ids must be disjoint"
            )
        _validate_sla_citations(
            (*supporting, *counter, *part.boundary_evidence_ids),
            view=view,
            dimension=dimension,
            owner=owner,
        )
    if baseline is not None:
        assert baseline.symptom_label is not None
        assert baseline.root_cause_label is not None
        expected_comparison = {
            "symptom_matches_baseline": (
                report.symptom.label == baseline.symptom_label
            ),
            "root_cause_matches_baseline": (
                report.root_cause.label == baseline.root_cause_label
            ),
        }
        for field_name, expected in expected_comparison.items():
            if getattr(report, field_name) is not expected:
                raise ValueError(
                    f"SLA diagnosis {field_name} does not match the fixed Baseline"
                )


def validate_sla_joint_verification(
    report: SlaJointVerification, view: EvidenceView
) -> None:
    """Validate the exact verifier shape and its dimension-owned citations."""

    expected = (
        ("symptom", EvidenceDimension.SYMPTOM, report.symptom),
        ("root_cause", EvidenceDimension.ROOT_CAUSE, report.root_cause),
    )
    for name, dimension, part in expected:
        if part.dimension is not dimension:
            raise ValueError(
                f"SLA {name} verifier must own the {dimension.value} dimension"
            )
        _validate_sla_citations(
            (*part.supporting_evidence_ids, *part.counter_evidence_ids),
            view=view,
            dimension=dimension,
            owner=f"SLA {name} verifier",
        )
        if (
            part.verdict is SlaVerifierVerdict.ACCEPT_CANDIDATE
            and not part.supporting_evidence_ids
        ):
            raise ValueError(
                f"SLA {name} verifier lacks verdict-appropriate supporting evidence"
            )
        if (
            part.verdict is SlaVerifierVerdict.PRESERVE_BASELINE
            and not part.counter_evidence_ids
        ):
            raise ValueError(
                f"SLA {name} verifier lacks verdict-appropriate counter evidence"
            )
        if part.verdict is SlaVerifierVerdict.UNRESOLVED and not (
            *part.supporting_evidence_ids,
            *part.counter_evidence_ids,
        ):
            raise ValueError(
                f"SLA {name} verifier lacks verdict-appropriate unresolved evidence"
            )


def validate_sla_arbitration_choice(
    *, choice: str, candidate: str, baseline: str
) -> None:
    """Forbid a conditional arbitrator from widening its fixed candidate pair."""

    if choice not in {candidate, baseline}:
        raise ValueError(
            "SLA arbitration choice must belong to the fixed candidate pair"
        )


def validate_sla_conditional_arbitration(
    report: SlaConditionalArbitration,
    *,
    view: EvidenceView,
    diagnosis: SlaJointDiagnosis,
    verification: SlaJointVerification,
    baseline: BaselineAnchor,
) -> None:
    """Require exactly one evidence-owned choice for every conflicting dimension."""

    assert baseline.symptom_label is not None and baseline.root_cause_label is not None
    candidates = {
        EvidenceDimension.SYMPTOM: diagnosis.symptom.label,
        EvidenceDimension.ROOT_CAUSE: diagnosis.root_cause.label,
    }
    baselines = {
        EvidenceDimension.SYMPTOM: baseline.symptom_label,
        EvidenceDimension.ROOT_CAUSE: baseline.root_cause_label,
    }
    verdicts = {
        EvidenceDimension.SYMPTOM: verification.symptom.verdict,
        EvidenceDimension.ROOT_CAUSE: verification.root_cause.verdict,
    }
    expected = {
        dimension
        for dimension in (EvidenceDimension.SYMPTOM, EvidenceDimension.ROOT_CAUSE)
        if verdicts[dimension] is SlaVerifierVerdict.PRESERVE_BASELINE
        and candidates[dimension] != baselines[dimension]
    }
    reported = {
        dimension
        for dimension, choice in (
            (EvidenceDimension.SYMPTOM, report.symptom),
            (EvidenceDimension.ROOT_CAUSE, report.root_cause),
        )
        if choice is not None
    }
    if reported != expected:
        raise ValueError(
            "SLA arbitration must contain exactly the conflicting dimensions"
        )
    for dimension, choice in (
        (EvidenceDimension.SYMPTOM, report.symptom),
        (EvidenceDimension.ROOT_CAUSE, report.root_cause),
    ):
        if choice is None:
            continue
        if choice.dimension is not dimension:
            raise ValueError(
                f"SLA arbitrator choice must own the {dimension.value} dimension"
            )
        validate_sla_arbitration_choice(
            choice=choice.selected_label,
            candidate=candidates[dimension],
            baseline=baselines[dimension],
        )
        _validate_sla_citations(
            choice.supporting_evidence_ids,
            view=view,
            dimension=dimension,
            owner=f"SLA {dimension.value} arbitrator",
        )


class VerificationVerdict(str, Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    INSUFFICIENT_TO_REJECT = "insufficient_to_reject"


def _require_unique_evidence_ids(ids: Sequence[str], *, field_name: str) -> None:
    if len(ids) != len(set(ids)):
        raise ValueError(f"{field_name} cannot contain duplicate evidence ids")


class JointAnchorReport(FrozenStrictModel):
    symptom: SymptomReport
    root_cause: RootCauseReport
    causal_account: str = Field(min_length=20)
    shared_supporting_evidence_ids: tuple[str, ...] = Field(min_length=1)
    shared_counter_evidence_ids: tuple[str, ...] = Field(default_factory=tuple)
    shared_boundary_evidence_ids: tuple[str, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def evidence_ids_are_unique(self) -> "JointAnchorReport":
        for field_name in (
            "shared_supporting_evidence_ids",
            "shared_counter_evidence_ids",
            "shared_boundary_evidence_ids",
        ):
            _require_unique_evidence_ids(
                getattr(self, field_name), field_name=field_name
            )
        return self


class DimensionVerificationReport(FrozenStrictModel):
    dimension: EvidenceDimension
    verdict: VerificationVerdict
    anchor_label: str = Field(min_length=1)
    alternative_label: str | None = Field(default=None, min_length=1)
    rationale: str = Field(min_length=20)
    supporting_evidence_ids: tuple[str, ...] = Field(default_factory=tuple)
    counter_evidence_ids: tuple[str, ...] = Field(default_factory=tuple)
    corrected_claim: str | None = Field(default=None, min_length=10)
    corrected_causal_chain: tuple[str, ...] = Field(default_factory=tuple)
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_verdict(self) -> "DimensionVerificationReport":
        _require_unique_evidence_ids(
            self.supporting_evidence_ids, field_name="supporting_evidence_ids"
        )
        _require_unique_evidence_ids(
            self.counter_evidence_ids, field_name="counter_evidence_ids"
        )
        corrected_content = self.corrected_claim is not None or bool(
            self.corrected_causal_chain
        )
        if self.verdict is VerificationVerdict.REJECT:
            if (
                self.alternative_label is None
                or self.alternative_label == self.anchor_label
                or self.corrected_claim is None
                or not self.supporting_evidence_ids
            ):
                raise ValueError(
                    "REJECT requires a different alternative label, corrected claim, and supporting evidence"
                )
            if (
                self.dimension is EvidenceDimension.ROOT_CAUSE
                and len(self.corrected_causal_chain) < 3
            ):
                raise ValueError(
                    "REJECT root-cause verification requires a three-step corrected causal chain"
                )
            return self
        if self.alternative_label is not None or corrected_content:
            raise ValueError(
                f"{self.verdict.name} verification cannot contain alternative or corrected content"
            )
        return self


class ChainVerificationReport(DimensionVerificationReport):
    """B audit; legacy verifier schemas and S0 prompts remain unchanged."""
    chain_check: ChainAudit

    @model_validator(mode="after")
    def checker_transition(self) -> "ChainVerificationReport":
        expected = {"pass": VerificationVerdict.ACCEPT,
                    "rewrite": VerificationVerdict.INSUFFICIENT_TO_REJECT,
                    "relabel": VerificationVerdict.REJECT}
        audit = self.chain_check
        if audit.action not in expected or self.verdict is not expected[audit.action]:
            raise ValueError("checker action and adapter verdict disagree; unresolved cannot compose")
        if audit.action in {"rewrite", "relabel"}:
            if audit.revision is None:
                raise ValueError("checker revision is required")
            if audit.action == "rewrite" and audit.revision.label != self.anchor_label:
                raise ValueError("rewrite cannot change the anchor label")
            if audit.action == "relabel" and (
                self.alternative_label != audit.revision.label
                or self.corrected_claim != audit.revision.claim
                or self.corrected_causal_chain != audit.revision.causal_chain
            ):
                raise ValueError("legacy correction fields must match the checker revision")
        return self


class TeamCorrectionAudit(FrozenStrictModel):
    dimension: EvidenceDimension
    anchor_label: str
    proposed_label: str | None = None
    final_label: str
    accepted: bool
    reason: str = Field(min_length=1)
    supporting_evidence_ids: tuple[str, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def evidence_ids_are_unique(self) -> "TeamCorrectionAudit":
        _require_unique_evidence_ids(
            self.supporting_evidence_ids, field_name="supporting_evidence_ids"
        )
        return self


class TeamComposedCandidate(StrictModel):
    team_id: str = Field(min_length=1)
    symptom: SymptomReport
    root_cause: RootCauseReport
    anchor: JointAnchorReport
    verifications: tuple[ChainVerificationReport | DimensionVerificationReport, ChainVerificationReport | DimensionVerificationReport]
    correction_audit: tuple[TeamCorrectionAudit, TeamCorrectionAudit]

    @model_validator(mode="after")
    def covers_each_classification_dimension_once(self) -> "TeamComposedCandidate":
        expected = {EvidenceDimension.SYMPTOM, EvidenceDimension.ROOT_CAUSE}
        verification_dimensions = {review.dimension for review in self.verifications}
        audit_dimensions = {audit.dimension for audit in self.correction_audit}
        if verification_dimensions != expected:
            raise ValueError(
                "verifications must contain symptom and root cause once each"
            )
        if audit_dimensions != expected:
            raise ValueError(
                "correction audit must contain symptom and root cause once each"
            )
        return self


class DisagreementMap(FrozenStrictModel):
    """Machine-readable dimensions on which two frozen reports differ."""

    dimensions: tuple[str, ...]
    details: Mapping[str, Mapping[str, Any]]
    requires_arbitration: bool

    @model_validator(mode="after")
    def freeze_details(self) -> "DisagreementMap":
        object.__setattr__(self, "details", _deep_freeze(self.details))
        return self

    @field_serializer("details")
    def serialize_details(
        self, value: Mapping[str, Mapping[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        return _deep_thaw(value)


class Stage2ArbitrationDecision(StrictModel):
    """A targeted judgment over the disagreement map, not a fresh analysis."""

    resolution_status: ResolutionStatus
    decision: Stage2Decision | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    rationale: str = Field(min_length=20)
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    resolved_dimensions: list[str] = Field(default_factory=list)
    unresolved_dimensions: list[str] = Field(default_factory=list)
    missing_facts: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_resolution(self) -> "Stage2ArbitrationDecision":
        if len(self.resolved_dimensions) != len(set(self.resolved_dimensions)):
            raise ValueError("resolved dimensions cannot contain duplicates")
        if len(self.unresolved_dimensions) != len(set(self.unresolved_dimensions)):
            raise ValueError("unresolved dimensions cannot contain duplicates")
        if self.resolution_status is ResolutionStatus.RESOLVED:
            if self.decision is None or self.confidence is None:
                raise ValueError("resolved Stage 2 arbitration must contain a decision")
            if not self.supporting_evidence_ids or not self.resolved_dimensions:
                raise ValueError(
                    "resolved Stage 2 arbitration must cite evidence and dimensions"
                )
            if self.unresolved_dimensions or self.missing_facts:
                raise ValueError(
                    "resolved arbitration cannot contain unresolved dimensions or missing facts"
                )
            return self
        if self.decision is not None:
            raise ValueError(
                "unresolved Stage 2 arbitration must not contain a decision"
            )
        if self.confidence is not None or self.resolved_dimensions:
            raise ValueError(
                "unresolved Stage 2 arbitration must not contain resolved output"
            )
        if not self.unresolved_dimensions:
            raise ValueError(
                "unresolved arbitration must contain unresolved dimensions"
            )
        return self


class Stage2FinalDecision(StrictModel):
    """Resolved Stage 2 decision awaiting deterministic verification."""

    decision: Stage2Decision
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=20)
    supporting_evidence_ids: list[str] = Field(min_length=1)
    source: ArbitrationSource


class Stage2Verification(StrictModel):
    """Verification result that preserves the proposed classification."""

    decision: Stage2Decision
    valid: bool
    errors: list[str] = Field(default_factory=list)


class Stage2ScopeExclusion(StrictModel):
    """Auditable match produced by a deterministic artifact-scope rule."""

    rule_id: str = Field(min_length=1)
    boundary: str = Field(min_length=10)
    evidence_id: str = Field(min_length=1)
    source_uri: str = Field(min_length=1)


class AnonymousStage2Report(StrictModel):
    """Decision-relevant report content with team identity removed."""

    decision: Stage2Decision
    confidence: float = Field(ge=0.0, le=1.0)
    fault_claim: str
    repair_claim: str
    supporting_evidence_ids: list[str]
    counter_evidence_ids: list[str]
    evidence_tests: Stage2EvidenceTests
    alternative_hypothesis: str
    decision_boundary: str
    evidence_sufficiency: EvidenceSufficiency
    unresolved_evidence_gaps: list[str]


class Stage2ArbitrationPacket(StrictModel):
    """An anonymized, disagreement-focused input for the arbitrator."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    domain_profile: str = Field(min_length=1)
    taxonomy: dict[str, list[str] | dict[str, str]]
    disagreement: DisagreementMap
    report_a: AnonymousStage2Report
    report_b: AnonymousStage2Report
    classification_ledger_version: int | None = Field(default=None, ge=0)
    relevant_evidence: tuple[EvidenceItem, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def freeze_taxonomy(self) -> "Stage2ArbitrationPacket":
        object.__setattr__(self, "taxonomy", _deep_freeze(self.taxonomy))
        return self

    @field_serializer("taxonomy")
    def serialize_taxonomy(
        self, value: Mapping[str, Any]
    ) -> dict[str, list[str] | dict[str, str]]:
        return _deep_thaw(value)

    def __deepcopy__(
        self, memo: dict[int, object] | None = None
    ) -> "Stage2ArbitrationPacket":
        copied = type(self).model_validate(self.model_dump(mode="python"))
        if memo is not None:
            memo[id(self)] = copied
        return copied


class CausalConsistencyReport(FrozenStrictModel):
    """Checks causal fit without owning either classification label."""

    status: ConsistencyStatus
    rationale: str = Field(min_length=20)
    supporting_evidence_ids: tuple[str, ...] = Field(min_length=1)
    evidence_requests: tuple[EvidenceRequest, ...] = Field(default_factory=tuple)


class BoundaryChallengeAction(str, Enum):
    PASS = "pass"
    SYMPTOM_REVIEW = "symptom_review"
    CAUSE_REVIEW = "cause_review"
    EVIDENCE_REQUEST = "evidence_request"


class BoundaryChallenge(FrozenStrictModel):
    """Non-classifying Stage 3 boundary review over frozen proposals."""

    action: BoundaryChallengeAction
    rationale: str = Field(min_length=1)
    cited_evidence_ids: tuple[str, ...] = Field(default_factory=tuple)
    evidence_request: EvidenceRequest | None = None

    @model_validator(mode="after")
    def request_belongs_only_to_evidence_action(self) -> "BoundaryChallenge":
        if (
            self.evidence_request is not None
            and self.action is not BoundaryChallengeAction.EVIDENCE_REQUEST
        ):
            raise ValueError(
                "evidence_request is allowed only for the evidence_request action"
            )
        return self


class Stage3TeamReport(FrozenStrictModel):
    """Frozen Stage 3 outputs for one independent team."""

    team_id: str = Field(min_length=1)
    symptom: SymptomReport
    root_cause: RootCauseReport
    consistency: CausalConsistencyReport
    anchor: JointAnchorReport | None = None
    verifications: tuple[ChainVerificationReport | DimensionVerificationReport, ...] = Field(
        default_factory=tuple
    )
    correction_audit: tuple[TeamCorrectionAudit, ...] = Field(default_factory=tuple)
    baseline_revision_assessments: tuple[BaselineRevisionAssessment, ...] = Field(
        default_factory=tuple
    )
    revision_consistency: tuple[RevisionConsistencyReport, ...] = Field(
        default_factory=tuple
    )

    @model_validator(mode="after")
    def anchored_records_are_complete_and_ordered(self) -> "Stage3TeamReport":
        assessment_keys = [
            (assessment.dimension, assessment.proposal_digest)
            for assessment in self.baseline_revision_assessments
        ]
        if len(assessment_keys) != len(set(assessment_keys)):
            raise ValueError(
                "baseline revision assessments cannot duplicate a proposal"
            )
        revision_keys = [
            (
                record.dimension,
                record.proposal_digest,
            )
            for record in self.revision_consistency
        ]
        if len(revision_keys) != len(set(revision_keys)):
            raise ValueError("revision consistency records cannot duplicate a proposal")
        has_dimension_records = bool(self.verifications or self.correction_audit)
        if self.anchor is None:
            if has_dimension_records:
                raise ValueError(
                    "anchor is required when verification or correction records exist"
                )
            return self

        expected = (EvidenceDimension.SYMPTOM, EvidenceDimension.ROOT_CAUSE)
        verification_dimensions = tuple(
            review.dimension for review in self.verifications
        )
        audit_dimensions = tuple(audit.dimension for audit in self.correction_audit)
        if verification_dimensions != expected or audit_dimensions != expected:
            raise ValueError("anchored reports require symptom and root cause in order")
        return self


def stage3_team_report_digest(report: Stage3TeamReport) -> str:
    """Hash the canonical pre-revision classification projection of a team report."""

    payload = report.model_dump(mode="json")
    payload.pop("baseline_revision_assessments")
    payload.pop("revision_consistency")

    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


class Stage3ComponentFailure(FrozenStrictModel):
    """Sanitized operational failure for one named Stage 3 team component."""

    team_id: str = Field(min_length=1)
    role: str = Field(min_length=1)
    error_type: str = Field(min_length=1)
    message: str = Field(min_length=1)
    schema_name: str | None = Field(default=None, min_length=1)
    validation_summary: Mapping[str, tuple[str, ...]] = Field(default_factory=dict)

    @field_validator("schema_name", mode="before")
    @classmethod
    def sanitize_schema_name_value(cls, value: object) -> str | None:
        return sanitize_schema_name(value)

    @field_validator("validation_summary", mode="before")
    @classmethod
    def sanitize_validation_summary_value(cls, value: object) -> dict[str, list[str]]:
        return sanitize_validation_summary(value)

    @model_validator(mode="after")
    def freeze_validation_summary(self) -> "Stage3ComponentFailure":
        object.__setattr__(
            self, "validation_summary", _deep_freeze(self.validation_summary)
        )
        return self

    @field_serializer("validation_summary")
    def serialize_validation_summary(
        self, value: Mapping[str, Any]
    ) -> dict[str, list[str] | dict[str, str]]:
        return _deep_thaw(value)


class AnonymousStage3TeamReport(StrictModel):
    """Stage 3 team report with its identity removed."""

    symptom: SymptomReport
    root_cause: RootCauseReport
    consistency: CausalConsistencyReport


class Stage3ArbitrationDecision(FrozenStrictModel):
    resolution_status: ResolutionStatus
    symptom_label: str | None = Field(default=None, min_length=1)
    root_cause_label: str | None = Field(default=None, min_length=1)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    rationale: str = Field(min_length=20)
    supporting_evidence_ids: tuple[str, ...] = Field(default_factory=tuple)
    resolved_dimensions: tuple[str, ...] = Field(default_factory=tuple)
    unresolved_dimensions: tuple[str, ...] = Field(default_factory=tuple)
    missing_facts: tuple[str, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def validate_resolution(self) -> "Stage3ArbitrationDecision":
        if len(self.resolved_dimensions) != len(set(self.resolved_dimensions)):
            raise ValueError("resolved dimensions cannot contain duplicates")
        if len(self.unresolved_dimensions) != len(set(self.unresolved_dimensions)):
            raise ValueError("unresolved dimensions cannot contain duplicates")
        if self.resolution_status is ResolutionStatus.RESOLVED:
            if (
                self.symptom_label is None
                or self.root_cause_label is None
                or self.confidence is None
            ):
                raise ValueError(
                    "resolved Stage 3 arbitration must contain both labels and confidence"
                )
            if not self.supporting_evidence_ids or not self.resolved_dimensions:
                raise ValueError(
                    "resolved Stage 3 arbitration must cite evidence and dimensions"
                )
            if self.unresolved_dimensions or self.missing_facts:
                raise ValueError(
                    "resolved arbitration cannot contain unresolved dimensions or missing facts"
                )
            return self
        if self.symptom_label is not None or self.root_cause_label is not None:
            raise ValueError("unresolved Stage 3 arbitration must not contain labels")
        if self.confidence is not None or self.resolved_dimensions:
            raise ValueError(
                "unresolved Stage 3 arbitration must not contain resolved output"
            )
        if not self.unresolved_dimensions:
            raise ValueError(
                "unresolved arbitration must contain unresolved dimensions"
            )
        return self


class Stage3FinalDecision(FrozenStrictModel):
    symptom_label: str = Field(min_length=1)
    root_cause_label: str = Field(min_length=1)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    rationale: str = Field(min_length=20)
    supporting_evidence_ids: tuple[str, ...] = Field(default_factory=tuple)
    source: ArbitrationSource
    baseline_gate_provenance: BaselineGateProvenance | None = None

    @model_validator(mode="after")
    def require_provenance_for_ordinary_decisions(self) -> "Stage3FinalDecision":
        if self.source is ArbitrationSource.SLA_FALLBACK:
            if (
                self.confidence is not None
                or self.supporting_evidence_ids
                or self.baseline_gate_provenance is not None
            ):
                raise ValueError(
                    "SLA fallback decisions cannot claim confidence, citations, or Baseline gate provenance"
                )
            return self
        if self.source is ArbitrationSource.BASELINE_PRESERVATION_GATE:
            if self.baseline_gate_provenance is BaselineGateProvenance.PRESERVED:
                if self.confidence is not None or self.supporting_evidence_ids:
                    raise ValueError(
                        "preserved Baseline gate decisions cannot claim confidence or citations"
                    )
                return self
            if (
                self.baseline_gate_provenance
                is BaselineGateProvenance.REVISION_CERTIFICATE
            ):
                if self.confidence is not None or not self.supporting_evidence_ids:
                    raise ValueError(
                        "certificate-backed Baseline gate decisions require citations without confidence"
                    )
                return self
            if (
                self.baseline_gate_provenance
                is BaselineGateProvenance.PRE_GATE_CANDIDATE
            ):
                if self.confidence is None or not self.supporting_evidence_ids:
                    raise ValueError(
                        "pre-gate-backed Baseline decisions require confidence and citations"
                    )
                return self
            if (self.confidence is None) != (not self.supporting_evidence_ids):
                raise ValueError(
                    "Baseline gate confidence and citations must both be present or absent"
                )
            return self
        if self.baseline_gate_provenance is not None:
            raise ValueError("ordinary decisions cannot claim Baseline gate provenance")
        if self.confidence is None or not self.supporting_evidence_ids:
            raise ValueError(
                "ordinary Stage 3 decisions require confidence and supporting evidence"
            )
        return self


class Stage3Verification(FrozenStrictModel):
    symptom_label: str
    root_cause_label: str
    valid: bool
    errors: tuple[str, ...] = Field(default_factory=tuple)


class Stage3ArbitrationPacket(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    domain_profile: str = Field(min_length=1)
    taxonomy: dict[str, list[str] | dict[str, str]]
    disagreement: DisagreementMap
    team_a: AnonymousStage3TeamReport
    team_b: AnonymousStage3TeamReport
    classification_ledger_version: int | None = Field(default=None, ge=0)
    relevant_evidence: tuple[EvidenceItem, ...] = Field(default_factory=tuple)

    supervision: dict[str, Any] | None = None

    @model_validator(mode="after")
    def freeze_taxonomy(self) -> "Stage3ArbitrationPacket":
        object.__setattr__(self, "taxonomy", _deep_freeze(self.taxonomy))
        return self

    @field_serializer("taxonomy")
    def serialize_taxonomy(
        self, value: Mapping[str, Any]
    ) -> dict[str, list[str] | dict[str, str]]:
        return _deep_thaw(value)

    def __deepcopy__(
        self, memo: dict[int, object] | None = None
    ) -> "Stage3ArbitrationPacket":
        copied = type(self).model_validate(self.model_dump(mode="python"))
        if memo is not None:
            memo[id(self)] = copied
        return copied
