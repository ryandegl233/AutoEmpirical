"""Declarative capability boundaries for each expert role."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType

from pydantic import BaseModel, ConfigDict, Field
from Benchmark.src.annotation_contracts import NEW_PAPER_DOMAINS

from .contracts import (
    ConsistencyStatus,
    EvidenceDimension,
    EvidenceView,
    SpecialistType,
    Stage3TeamPolicy,
    TeamPerspective,
)


class AnalystRole(str, Enum):
    STAGE2_FAULT_VERIFIER = "stage2_fault_verifier"
    FAULT_EVIDENCE_ANALYST = "fault_evidence_analyst"
    SCOPE_BOUNDARY_ANALYST = "scope_boundary_analyst"
    REPAIR_CAUSALITY_ANALYST = "repair_causality_analyst"
    EVIDENCE_READINESS = "evidence_readiness"
    SYMPTOM_ANALYST = "symptom_analyst"
    ROOT_CAUSE_ANALYST = "root_cause_analyst"
    JOINT_ANCHOR = "joint_anchor"
    SYMPTOM_VERIFIER = "symptom_verifier"
    ROOT_CAUSE_VERIFIER = "root_cause_verifier"
    CAUSAL_CONSISTENCY_CHECKER = "causal_consistency_checker"
    BOUNDARY_CHALLENGER = "boundary_challenger"
    SLA_JOINT_DIAGNOSIS = "sla_joint_diagnosis"
    SLA_JOINT_VERIFIER = "sla_joint_verifier"
    SLA_CONDITIONAL_ARBITRATOR = "sla_conditional_arbitrator"


@dataclass(frozen=True)
class CapabilitySpec:
    role: AnalystRole
    objective: str
    evidence_focus: str
    reasoning_steps: tuple[str, ...]
    allowed_specialists: tuple[str, ...]
    forbidden_inputs: tuple[str, ...]
    allowed_outputs: tuple[str, ...]
    may_classify: bool


class RoleTask(BaseModel):
    """Exact ledger view plus the capability boundary for one isolated role."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    team_id: str = Field(min_length=1)
    capability: CapabilitySpec
    evidence_view: EvidenceView


_CAPABILITIES = {
    AnalystRole.STAGE2_FAULT_VERIFIER: CapabilitySpec(
        role=AnalystRole.STAGE2_FAULT_VERIFIER,
        objective=(
            "Apply the supplied paper-specific policy to screen a candidate "
            "for a study-eligible fault case."
        ),
        evidence_focus="fault_scope_and_domain_specific_repair_requirement",
        reasoning_steps=(
            "normalize_fault_and_nonfault_signals",
            "construct_competing_hypotheses",
            "fault_existence",
            "scope_exclusion",
            "repair_causality_when_required_or_available",
            "apply_domain_specific_acceptance_policy",
        ),
        allowed_specialists=(
            SpecialistType.ISSUE_PR.value,
            SpecialistType.COMMIT_HISTORY.value,
            SpecialistType.CODE_CONTEXT.value,
            SpecialistType.TEST_EVIDENCE.value,
            SpecialistType.TAXONOMY_KNOWLEDGE.value,
        ),
        forbidden_inputs=("peer_report", "gold_annotation"),
        allowed_outputs=("stage2_analysis_report",),
        may_classify=True,
    ),
    AnalystRole.FAULT_EVIDENCE_ANALYST: CapabilitySpec(
        role=AnalystRole.FAULT_EVIDENCE_ANALYST,
        objective="Assess only whether frozen evidence establishes a fault.",
        evidence_focus="fault_existence",
        reasoning_steps=("fault_existence",),
        allowed_specialists=(),
        forbidden_inputs=(
            "peer_report",
            "gold_annotation",
            "final_decision",
            "stage3_labels",
        ),
        allowed_outputs=("fault_evidence_assessment",),
        may_classify=False,
    ),
    AnalystRole.SCOPE_BOUNDARY_ANALYST: CapabilitySpec(
        role=AnalystRole.SCOPE_BOUNDARY_ANALYST,
        objective="Apply only the paper-specific study inclusion boundary.",
        evidence_focus="study_scope",
        reasoning_steps=("study_scope",),
        allowed_specialists=(),
        forbidden_inputs=(
            "peer_report",
            "gold_annotation",
            "fault_reinterpretation",
            "final_decision",
            "stage3_labels",
        ),
        allowed_outputs=("scope_boundary_assessment",),
        may_classify=False,
    ),
    AnalystRole.REPAIR_CAUSALITY_ANALYST: CapabilitySpec(
        role=AnalystRole.REPAIR_CAUSALITY_ANALYST,
        objective="Assess only whether frozen change evidence repairs the fault.",
        evidence_focus="repair_causality",
        reasoning_steps=("repair_causality",),
        allowed_specialists=(),
        forbidden_inputs=(
            "peer_report",
            "gold_annotation",
            "final_decision",
            "stage3_labels",
        ),
        allowed_outputs=("repair_causality_assessment",),
        may_classify=False,
    ),
    AnalystRole.EVIDENCE_READINESS: CapabilitySpec(
        role=AnalystRole.EVIDENCE_READINESS,
        objective=(
            "Assess whether frozen evidence supports every required decision "
            "dimension and request only specific discriminating evidence."
        ),
        evidence_focus="decision_relevant_evidence_sufficiency",
        reasoning_steps=(
            "identify_required_dimensions",
            "assess_confirmed_evidence",
            "identify_specific_missing_facts",
            "request_discriminating_evidence_only",
        ),
        allowed_specialists=(
            SpecialistType.ISSUE_PR.value,
            SpecialistType.COMMIT_HISTORY.value,
            SpecialistType.CODE_CONTEXT.value,
            SpecialistType.TEST_EVIDENCE.value,
        ),
        forbidden_inputs=("peer_report", "gold_annotation"),
        allowed_outputs=("evidence_readiness_report",),
        may_classify=False,
    ),
    AnalystRole.SYMPTOM_ANALYST: CapabilitySpec(
        role=AnalystRole.SYMPTOM_ANALYST,
        objective="Classify only the externally observable pre-fix behavior.",
        evidence_focus="behavior",
        reasoning_steps=(
            "extract_expected_and_actual_behavior",
            "locate_failure_phase",
            "identify_observable_object",
            "contrast_nearest_symptom_labels",
        ),
        allowed_specialists=(
            SpecialistType.ISSUE_PR.value,
            SpecialistType.TEST_EVIDENCE.value,
            SpecialistType.TAXONOMY_KNOWLEDGE.value,
        ),
        forbidden_inputs=("root_cause_report", "peer_report", "gold_annotation"),
        allowed_outputs=("symptom_report",),
        may_classify=True,
    ),
    AnalystRole.ROOT_CAUSE_ANALYST: CapabilitySpec(
        role=AnalystRole.ROOT_CAUSE_ANALYST,
        objective="Classify only the pre-fix defect mechanism.",
        evidence_focus="mechanism",
        reasoning_steps=(
            "invert_patch",
            "build_pre_fix_causal_chain",
            "locate_responsibility",
            "contrast_domain_specific_root_cause_labels",
        ),
        allowed_specialists=(
            SpecialistType.COMMIT_HISTORY.value,
            SpecialistType.CODE_CONTEXT.value,
            SpecialistType.TEST_EVIDENCE.value,
            SpecialistType.SIMILAR_CASE.value,
            SpecialistType.TAXONOMY_KNOWLEDGE.value,
        ),
        forbidden_inputs=("symptom_report", "peer_report", "gold_annotation"),
        allowed_outputs=("root_cause_report",),
        may_classify=True,
    ),
    AnalystRole.JOINT_ANCHOR: CapabilitySpec(
        role=AnalystRole.JOINT_ANCHOR,
        objective=(
            "Establish one evidence-grounded symptom and root-cause anchor "
            "from the exact frozen evidence view."
        ),
        evidence_focus="joint_symptom_and_root_cause",
        reasoning_steps=(
            "inventory_joint_evidence",
            "separate_observed_behavior_from_mechanism",
            "contrast_nearest_legal_alternatives",
            "record_shared_causal_account",
        ),
        allowed_specialists=(SpecialistType.TAXONOMY_KNOWLEDGE.value,),
        forbidden_inputs=(),
        allowed_outputs=("joint_anchor_report",),
        may_classify=True,
    ),
    AnalystRole.SYMPTOM_VERIFIER: CapabilitySpec(
        role=AnalystRole.SYMPTOM_VERIFIER,
        objective=(
            "Verify only the symptom label in the supplied anonymized team "
            "anchor against the exact frozen evidence view."
        ),
        evidence_focus="symptom_verification",
        reasoning_steps=(
            "inspect_symptom_anchor",
            "test_nearest_symptom_alternative",
            "return_symptom_verdict",
        ),
        allowed_specialists=(SpecialistType.TAXONOMY_KNOWLEDGE.value,),
        forbidden_inputs=(),
        allowed_outputs=("symptom_verification_report",),
        may_classify=True,
    ),
    AnalystRole.ROOT_CAUSE_VERIFIER: CapabilitySpec(
        role=AnalystRole.ROOT_CAUSE_VERIFIER,
        objective=(
            "Verify only the root-cause label in the supplied anonymized team "
            "anchor against the exact frozen evidence view."
        ),
        evidence_focus="root_cause_verification",
        reasoning_steps=(
            "inspect_root_cause_anchor",
            "test_nearest_root_cause_alternative",
            "return_root_cause_verdict",
        ),
        allowed_specialists=(SpecialistType.TAXONOMY_KNOWLEDGE.value,),
        forbidden_inputs=(),
        allowed_outputs=("root_cause_verification_report",),
        may_classify=True,
    ),
    AnalystRole.CAUSAL_CONSISTENCY_CHECKER: CapabilitySpec(
        role=AnalystRole.CAUSAL_CONSISTENCY_CHECKER,
        objective=(
            "Check temporal, causal, evidence, and granularity consistency "
            "without assigning labels."
        ),
        evidence_focus="symptom_cause_pair",
        reasoning_steps=(
            "check_pre_fix_temporal_alignment",
            "check_causal_plausibility",
            "check_evidence_chain",
            "check_label_granularity",
        ),
        allowed_specialists=(),
        forbidden_inputs=("gold_annotation",),
        allowed_outputs=tuple(status.value for status in ConsistencyStatus),
        may_classify=False,
    ),
    AnalystRole.BOUNDARY_CHALLENGER: CapabilitySpec(
        role=AnalystRole.BOUNDARY_CHALLENGER,
        objective=(
            "Challenge only taxonomy boundaries, causal granularity, and "
            "unsupported specificity in proposed anonymized reports."
        ),
        evidence_focus="stage3_boundary_review",
        reasoning_steps=(
            "nearest_competing_label",
            "symptom_execution_phase_and_observation_level",
            "local_mechanism_vs_high_level_cause",
            "unsupported_specificity",
        ),
        allowed_specialists=(),
        forbidden_inputs=("gold_annotation", "replacement_label", "final_decision"),
        allowed_outputs=(
            "pass",
            "symptom_review",
            "cause_review",
            "evidence_request",
        ),
        may_classify=False,
    ),
    AnalystRole.SLA_JOINT_DIAGNOSIS: CapabilitySpec(
        role=AnalystRole.SLA_JOINT_DIAGNOSIS,
        objective=(
            "Produce bounded symptom and root-cause candidates from the exact "
            "frozen evidence view and supplied Baseline labels."
        ),
        evidence_focus="joint_symptom_and_root_cause",
        reasoning_steps=(
            "separate_observable_behavior_from_pre_fix_mechanism",
            "select_one_taxonomy_label_per_dimension",
            "cite_dimension_capable_evidence",
        ),
        allowed_specialists=(),
        forbidden_inputs=("gold_annotation", "expected_answer", "error_cluster"),
        allowed_outputs=("sla_joint_diagnosis",),
        may_classify=True,
    ),
    AnalystRole.SLA_JOINT_VERIFIER: CapabilitySpec(
        role=AnalystRole.SLA_JOINT_VERIFIER,
        objective=(
            "Independently verify each supplied SLA candidate against the exact "
            "frozen evidence and the Baseline pair."
        ),
        evidence_focus="dimension_level_candidate_verification",
        reasoning_steps=(
            "inspect_canonical_candidate_projection",
            "compare_candidate_with_baseline",
            "return_one_bounded_verdict_per_dimension",
        ),
        allowed_specialists=(),
        forbidden_inputs=(
            "gold_annotation",
            "expected_answer",
            "error_cluster",
            "diagnosis_free_text_rationale",
        ),
        allowed_outputs=("sla_joint_verification",),
        may_classify=True,
    ),
    AnalystRole.SLA_CONDITIONAL_ARBITRATOR: CapabilitySpec(
        role=AnalystRole.SLA_CONDITIONAL_ARBITRATOR,
        objective=(
            "Resolve only an explicit conflicting SLA candidate-versus-Baseline "
            "label pair without widening the available choices."
        ),
        evidence_focus="fixed_candidate_baseline_pairs",
        reasoning_steps=(
            "inspect_conflicting_fixed_pairs",
            "select_only_candidate_or_baseline",
            "cite_exact_supporting_evidence",
        ),
        allowed_specialists=(),
        forbidden_inputs=("gold_annotation", "expected_answer", "error_cluster"),
        allowed_outputs=("sla_conditional_arbitration",),
        may_classify=True,
    ),
}


DEFAULT_STAGE3_TEAM_PERSPECTIVES = MappingProxyType(
    {
        "A": TeamPerspective.EVIDENCE_FIRST,
        "B": TeamPerspective.FALSIFICATION_FIRST,
    }
)


def stage3_team_policy(team_id: str) -> Stage3TeamPolicy:
    """Return the declared perspective for one supported isolated Stage 3 team."""

    try:
        perspective = DEFAULT_STAGE3_TEAM_PERSPECTIVES[team_id]
    except KeyError as error:
        raise ValueError(f"unknown Stage 3 team {team_id!r}") from error
    return Stage3TeamPolicy(team_id=team_id, perspective=perspective)


def render_stage3_perspective_instructions(
    policy: Stage3TeamPolicy,
    role: AnalystRole,
) -> str:
    """Render a schema-compatible perspective without changing role authority."""

    if role is AnalystRole.CAUSAL_CONSISTENCY_CHECKER:
        return _render_consistency_perspective(policy)
    if role not in (
        AnalystRole.SYMPTOM_ANALYST,
        AnalystRole.ROOT_CAUSE_ANALYST,
        AnalystRole.JOINT_ANCHOR,
        AnalystRole.SYMPTOM_VERIFIER,
        AnalystRole.ROOT_CAUSE_VERIFIER,
    ):
        raise ValueError(f"unsupported Stage 3 perspective role {role.value!r}")

    if policy.perspective is TeamPerspective.EVIDENCE_FIRST:
        return (
            "TEAM PERSPECTIVE: EVIDENCE-FIRST\n"
            "Inventory the cited facts before proposing any label or consistency "
            "finding. Separate direct observations from inferences, compare the "
            "nearest legal alternative only after that inventory, and limit every "
            "claim's strength to the cited evidence tier."
        )
    if policy.perspective is TeamPerspective.FALSIFICATION_FIRST:
        return (
            "TEAM PERSPECTIVE: FALSIFICATION-FIRST\n"
            "Form a leading hypothesis, name the nearest legal alternative, state "
            "what facts would falsify it, and search the frozen evidence for "
            "counterevidence. Explain the boundary that survives the falsification "
            "attempt. If the boundary cannot be distinguished, lower confidence "
            "and record the unresolved evidence gap."
        )
    raise ValueError(f"unsupported Stage 3 perspective {policy.perspective!r}")


def _render_consistency_perspective(policy: Stage3TeamPolicy) -> str:
    """Keep perspective work within the non-label consistency schema."""

    shared_outcome = (
        "Report the outcome only through the existing consistency status, rationale, "
        "citations, and evidence requests. Use the legal alternatives supplied in "
        "the reports only to assess consistency; do not name or replace labels."
    )
    if policy.perspective is TeamPerspective.EVIDENCE_FIRST:
        return (
            "TEAM PERSPECTIVE: EVIDENCE-FIRST\n"
            "Validate the cited positive causal chain and its cited evidence tier. "
            "Separate direct evidence from inference before checking temporal, causal, "
            f"and granularity consistency. {shared_outcome}"
        )
    if policy.perspective is TeamPerspective.FALSIFICATION_FIRST:
        return (
            "TEAM PERSPECTIVE: FALSIFICATION-FIRST\n"
            "Search the proposed symptom/cause pair for temporal, causal, and "
            "granularity counterexamples against the legal alternatives supplied in "
            f"the reports. {shared_outcome}"
        )
    raise ValueError(f"unsupported Stage 3 perspective {policy.perspective!r}")


def required_readiness_dimensions(
    domain: str,
    task: str,
) -> tuple[EvidenceDimension, ...]:
    if task == "stage2" and (domain == "ase2022" or domain in NEW_PAPER_DOMAINS):
        return (
            EvidenceDimension.FAULT_EXISTENCE,
            EvidenceDimension.STUDY_SCOPE,
        )
    if task == "stage2" and domain == "issta2024":
        return (
            EvidenceDimension.FAULT_EXISTENCE,
            EvidenceDimension.STUDY_SCOPE,
            EvidenceDimension.REPAIR_CAUSALITY,
        )
    if task == "stage3":
        return (
            EvidenceDimension.SYMPTOM,
            EvidenceDimension.ROOT_CAUSE,
        )
    raise ValueError(f"unsupported readiness task {task!r}")


def capability_for(role: AnalystRole) -> CapabilitySpec:
    return _CAPABILITIES[role]


def build_role_task(
    role: AnalystRole,
    *,
    team_id: str,
    evidence_view: EvidenceView,
) -> RoleTask:
    return RoleTask(
        team_id=team_id,
        capability=capability_for(role),
        evidence_view=evidence_view,
    )


def render_system_prompt(
    role: AnalystRole,
    *,
    include_dataset_guidance: bool = True,
) -> str:
    """Render the declarative capability as provider-neutral instructions."""

    capability = capability_for(role)
    steps = "\n".join(
        f"{index}. {step}"
        for index, step in enumerate(capability.reasoning_steps, start=1)
    )
    tools = ", ".join(capability.allowed_specialists) or "none"
    forbidden = ", ".join(capability.forbidden_inputs) or "none"
    outputs = ", ".join(capability.allowed_outputs)
    classification_rule = (
        "You may assign only the label dimension owned by this role."
        if capability.may_classify
        else "You must not assign or replace any classification label."
    )
    dataset_guidance = (
        "Do not infer a gold label from dataset membership. "
        if include_dataset_guidance
        else ""
    )
    return (
        f"ROLE: {capability.role.value}\n"
        f"OBJECTIVE: {capability.objective}\n"
        f"EVIDENCE FOCUS: {capability.evidence_focus}\n"
        f"REASONING PIPELINE:\n{steps}\n"
        f"ALLOWED SPECIALISTS: {tools}\n"
        f"FORBIDDEN INPUTS: {forbidden}\n"
        f"ALLOWED OUTPUTS: {outputs}\n"
        f"{classification_rule}\n"
        "Cite exact evidence IDs. Keep direct evidence distinct from inference. "
        f"{dataset_guidance}"
        "Keep each free-text output field under 80 words. "
        "Use only 3-4 concise causal-chain steps when that field is present. "
        "Do not restate evidence verbatim; cite its evidence ID instead."
    )
