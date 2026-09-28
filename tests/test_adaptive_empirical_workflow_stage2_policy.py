from __future__ import annotations

import pytest

from Benchmark.src.adaptive_empirical_workflow.contracts import (
    Stage2Decision,
    TestOutcome as EvidenceTestOutcome,
)
from Benchmark.src.adaptive_empirical_workflow.stage2_policy import (
    artifact_scope_exclusion,
    compose_stage2_decision,
)


@pytest.mark.parametrize(
    "source_uri",
    [
        "https://github.com/org/repo/pull/123",
        "https://github.com/org/repo/pull/123/files",
        "https://www.github.com/org/repo/pull/123",
        "github.com/org/repo/pull/123",
        "https://api.github.com/repos/org/repo/pulls/123",
    ],
)
def test_ase_scope_gate_recognizes_github_pull_request_locators(
    source_uri: str,
) -> None:
    assert artifact_scope_exclusion("ase2022", source_uri) is not None


@pytest.mark.parametrize(
    "source_uri",
    [
        "https://github.com/org/repo/issues/123",
        "https://example.test/github.com/org/repo/pull/123",
        "record:case-1#initial-summary",
    ],
)
def test_ase_scope_gate_does_not_reject_non_pull_request_locators(
    source_uri: str,
) -> None:
    assert artifact_scope_exclusion("ase2022", source_uri) is None


def test_scope_gate_is_disabled_outside_ase() -> None:
    assert (
        artifact_scope_exclusion(
            "issta2024",
            "https://github.com/org/repo/pull/123",
        )
        is None
    )


@pytest.mark.parametrize(
    ("domain", "fault", "scope", "repair", "expected"),
    [
        (
            "ase2022",
            EvidenceTestOutcome.PASS,
            EvidenceTestOutcome.PASS,
            EvidenceTestOutcome.UNKNOWN,
            Stage2Decision.ACCEPTED,
        ),
        (
            "ase2022",
            EvidenceTestOutcome.FAIL,
            EvidenceTestOutcome.PASS,
            EvidenceTestOutcome.PASS,
            Stage2Decision.REJECTED,
        ),
        (
            "ase2022",
            EvidenceTestOutcome.PASS,
            EvidenceTestOutcome.FAIL,
            EvidenceTestOutcome.PASS,
            Stage2Decision.REJECTED,
        ),
        (
            "issta2024",
            EvidenceTestOutcome.PASS,
            EvidenceTestOutcome.PASS,
            EvidenceTestOutcome.PASS,
            Stage2Decision.ACCEPTED,
        ),
        (
            "issta2024",
            EvidenceTestOutcome.PASS,
            EvidenceTestOutcome.PASS,
            EvidenceTestOutcome.UNKNOWN,
            Stage2Decision.REJECTED,
        ),
        (
            "issta2024",
            EvidenceTestOutcome.PASS,
            EvidenceTestOutcome.PASS,
            EvidenceTestOutcome.FAIL,
            Stage2Decision.REJECTED,
        ),
    ],
)
def test_stage2_decision_is_composed_from_domain_truth_table(
    domain: str,
    fault: EvidenceTestOutcome,
    scope: EvidenceTestOutcome,
    repair: EvidenceTestOutcome,
    expected: Stage2Decision,
) -> None:
    tests = {
        "fault_existence": fault,
        "scope_exclusion": scope,
        "repair_causality": repair,
    }

    assert compose_stage2_decision(domain, tests) is expected
