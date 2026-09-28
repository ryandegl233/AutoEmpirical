"""Explicit evidence capabilities shared by readiness and disagreement checks."""

from __future__ import annotations

import re
from Benchmark.src.annotation_contracts import NEW_PAPER_DOMAINS

from .contracts import (
    EvidenceDimension,
    EvidenceItem,
    EvidenceValidityIssue,
    EvidenceValidityReport,
    EvidenceView,
)

SYMPTOM_OBSERVATION = "symptom_observation"
DEFECT_MECHANISM = "defect_mechanism"
STUDY_SCOPE = "study_scope"
ROOT_CAUSE_CONTEXT = "root_cause_context"

_OBSERVATION_SOURCE_TYPES = frozenset(
    {
        "developer_confirmation",
        "execution_log",
        "failure_report",
        "issue_body",
        "issue_comment",
        "issue_comments",
        "maintainer_confirmation",
        "record_summary",
        "runtime_observation",
        "test_result",
    }
)
_MECHANISM_SOURCE_TYPES = frozenset(
    {
        "changed_files",
        "code_context",
        "code_diff",
        "commit",
        "commit_diff",
        "commit_history",
        "commit_message",
        "developer_confirmation",
        "linked_pull_request",
        "maintainer_confirmation",
        "patch",
        "pull_request",
        "regression_test",
        "repair",
        "repair_commit",
        "source_code",
        "test",
        "test_result",
    }
)
_STUDY_SCOPE_SOURCE_TYPES = frozenset(
    {
        "developer_confirmation",
        "issue_body",
        "issue_comment",
        "issue_comments",
        "linked_pull_request",
        "maintainer_confirmation",
        "pull_request",
        "record_summary",
    }
)
PATCH_SOURCE_TYPES = frozenset(
    {"code_diff", "commit", "commit_diff", "patch", "repair_commit"}
)

_DIMENSION_CAPABILITY = {
    EvidenceDimension.FAULT_EXISTENCE: SYMPTOM_OBSERVATION,
    EvidenceDimension.STUDY_SCOPE: STUDY_SCOPE,
    EvidenceDimension.REPAIR_CAUSALITY: DEFECT_MECHANISM,
    EvidenceDimension.SYMPTOM: SYMPTOM_OBSERVATION,
    EvidenceDimension.ROOT_CAUSE: DEFECT_MECHANISM,
}


def normalized_source_type(value: str) -> str:
    """Normalize only separators/case before exact capability-set lookup."""

    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def declared_capabilities(item: EvidenceItem) -> frozenset[str]:
    capabilities = item.metadata.get("evidence_capabilities", ())
    return frozenset(capabilities if isinstance(capabilities, tuple) else ())


def _source_capabilities(source_type: str) -> frozenset[str]:
    capabilities: set[str] = set()
    if source_type in _OBSERVATION_SOURCE_TYPES:
        capabilities.update({SYMPTOM_OBSERVATION, ROOT_CAUSE_CONTEXT})
    if source_type in _MECHANISM_SOURCE_TYPES:
        capabilities.update({DEFECT_MECHANISM, ROOT_CAUSE_CONTEXT})
    if source_type in _STUDY_SCOPE_SOURCE_TYPES:
        capabilities.add(STUDY_SCOPE)
    return frozenset(capabilities)


def _stage3_source_capabilities(item: EvidenceItem) -> frozenset[str]:
    """Return source-owned capabilities, narrowed (never expanded) by metadata."""

    source_capabilities = _source_capabilities(normalized_source_type(item.source_type))
    if "evidence_capabilities" not in item.metadata:
        return source_capabilities

    narrowed = source_capabilities & declared_capabilities(item)
    if ROOT_CAUSE_CONTEXT in source_capabilities and narrowed & {
        SYMPTOM_OBSERVATION,
        DEFECT_MECHANISM,
    }:
        narrowed = narrowed | {ROOT_CAUSE_CONTEXT}
    return frozenset(narrowed)


def assess_stage3_evidence_validity(view: EvidenceView) -> EvidenceValidityReport:
    """Admit only known, task-relevant source evidence to Stage 3."""

    valid_evidence_ids: list[str] = []
    quarantined: list[EvidenceValidityIssue] = []
    capabilities_by_evidence_id: dict[str, tuple[str, ...]] = {}
    for item in view.items:
        if item.record_id != view.record_id:
            quarantined.append(
                EvidenceValidityIssue(
                    evidence_id=item.evidence_id,
                    reasons=("record_id_mismatch",),
                )
            )
            continue

        source_capabilities = _source_capabilities(
            normalized_source_type(item.source_type)
        )
        if not source_capabilities:
            quarantined.append(
                EvidenceValidityIssue(
                    evidence_id=item.evidence_id,
                    reasons=("unknown_source_type",),
                )
            )
            continue

        capabilities = _stage3_source_capabilities(item)
        if not capabilities & {
            SYMPTOM_OBSERVATION,
            DEFECT_MECHANISM,
            ROOT_CAUSE_CONTEXT,
        }:
            quarantined.append(
                EvidenceValidityIssue(
                    evidence_id=item.evidence_id,
                    reasons=("no_stage3_capability",),
                )
            )
            continue

        valid_evidence_ids.append(item.evidence_id)
        capabilities_by_evidence_id[item.evidence_id] = tuple(sorted(capabilities))

    return EvidenceValidityReport(
        valid_evidence_ids=tuple(valid_evidence_ids),
        quarantined=tuple(quarantined),
        capabilities_by_evidence_id=capabilities_by_evidence_id,
    )


def evidence_capabilities(item: EvidenceItem) -> frozenset[str]:
    """Return source-owned capabilities, optionally narrowed by metadata."""

    return _stage3_source_capabilities(item)


def observation_capable(item: EvidenceItem) -> bool:
    return SYMPTOM_OBSERVATION in evidence_capabilities(item)


def mechanism_capable(item: EvidenceItem) -> bool:
    return DEFECT_MECHANISM in evidence_capabilities(item)


def supports_readiness_dimension(
    item: EvidenceItem,
    dimension: EvidenceDimension,
    *,
    domain: str = "issta2024",
) -> bool:
    if dimension is EvidenceDimension.ROOT_CAUSE and (domain == "ase2022" or domain in NEW_PAPER_DOMAINS):
        return ROOT_CAUSE_CONTEXT in evidence_capabilities(item)
    return _DIMENSION_CAPABILITY[dimension] in evidence_capabilities(item)
