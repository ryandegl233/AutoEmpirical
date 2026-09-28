from __future__ import annotations

import hashlib
import json

import pytest

import Benchmark.src.adaptive_empirical_workflow as workflow_package
from Benchmark.src.adaptive_empirical_workflow import evaluation as evaluation_module
from Benchmark.src.adaptive_empirical_workflow.evaluation import (
    baseline_preservation_diagnostics,
    paired_stage3_comparison,
    stage3_team_diagnostics,
    targeted_sla_diagnostics,
)


def test_targeted_sla_diagnostics_reports_execution_and_paired_recovery() -> None:
    records = _gold("r1", "r2", "r3")
    baseline = {
        "r1": {"symptom_label": "Crash", "root_cause_label": "Incorrect Code Logic"},
        "r2": {
            "symptom_label": "Incorrect Functionality",
            "root_cause_label": "API Misuse",
        },
        "r3": {"symptom_label": "Crash", "root_cause_label": "Incorrect Code Logic"},
    }
    rows = [
        {
            "record_id": "r1",
            "stage3_valid": True,
            "symptom_prediction": "Crash",
            "root_cause_prediction": "Incorrect Code Logic",
            "call_count": 2,
            "fallback": False,
            "audit": {
                "targeted_sla": {
                    "decision_status": {
                        "symptom": "verified",
                        "root_cause": "verified",
                    },
                    "arbitrated": False,
                    "call_audit": [
                        {
                            "latency_seconds": 1.0,
                            "schema_attempts": 1,
                            "network_attempts": 1,
                            "provider_errors": [],
                        }
                    ],
                }
            },
        },
        {
            "record_id": "r2",
            "stage3_valid": True,
            "symptom_prediction": "Crash",
            "root_cause_prediction": "Incorrect Code Logic",
            "call_count": 3,
            "fallback": False,
            "audit": {
                "targeted_sla": {
                    "decision_status": {
                        "symptom": "arbitrated",
                        "root_cause": "arbitrated",
                    },
                    "arbitrated": True,
                    "call_audit": [
                        {
                            "latency_seconds": 3.0,
                            "schema_attempts": 2,
                            "network_attempts": 3,
                            "provider_errors": [
                                {
                                    "cause_type": "http_error",
                                    "status": "retryable_error",
                                }
                            ],
                        }
                    ],
                }
            },
        },
        {
            "record_id": "r3",
            "stage3_valid": True,
            "symptom_prediction": "Incorrect Functionality",
            "root_cause_prediction": "API Misuse",
            "call_count": 0,
            "fallback": True,
            "audit": {
                "targeted_sla": {
                    "decision_status": {
                        "symptom": "budget_fallback",
                        "root_cause": "budget_fallback",
                    },
                    "arbitrated": False,
                    "call_audit": [],
                }
            },
        },
    ]

    metrics = targeted_sla_diagnostics(records, rows, baseline)

    assert metrics["coverage"] == 1.0
    assert metrics["completed_model_decisions"] == 2
    assert metrics["role_call_count"] == 5
    assert metrics["model_call_count"] == 4
    assert metrics["provider_request_count"] == 4
    assert metrics["arbitrated_count"] == 1
    assert metrics["fallback_counts"] == {"budget_fallback": 2}
    assert metrics["provider_error_distribution"] == {"http_error": 1}
    assert metrics["call_latency_seconds"] == {"p50": 2.0, "p95": 2.9}
    assert metrics["evaluation_status"] == "evaluated"
    assert metrics["recovery"] == 1
    assert metrics["harm"] == 1
    assert metrics["net_recovery"] == 0


def test_targeted_sla_diagnostics_keeps_execution_metrics_for_evidence_only_rows() -> (
    None
):
    metrics = targeted_sla_diagnostics(
        [{"record_id": "r1"}],
        [
            {
                "record_id": "r1",
                "stage3_valid": True,
                "symptom_prediction": "Crash",
                "root_cause_prediction": "Incorrect Code Logic",
                "call_count": 0,
                "fallback": True,
                "audit": {
                    "targeted_sla": {
                        "decision_status": {
                            "symptom": "transport_fallback",
                            "root_cause": "transport_fallback",
                        },
                        "arbitrated": False,
                        "call_audit": [],
                    }
                },
            }
        ],
        {"r1": {"symptom_label": "Crash", "root_cause_label": "Incorrect Code Logic"}},
    )

    assert metrics["evaluation_status"] == "not_evaluated"
    assert metrics["fallback_counts"] == {"transport_fallback": 2}
    assert "recovery" not in metrics


def test_baseline_preservation_metrics_use_fixed_denominator_and_paired_actions() -> (
    None
):
    records = _gold("r1", "r2", "r3")

    def assessment(verdict: str, dimension: str, proposed: str | None = None):
        return {
            "dimension": dimension,
            "verdict": verdict,
            "proposed_label": proposed,
        }

    def row(
        record_id: str,
        *,
        baseline: tuple[str | None, str | None, bool],
        pre: tuple[str, str],
        actions: tuple[str, str],
        certificates: tuple[str, ...] = (),
        reports: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        symptom, root, valid = baseline
        certificate_rows = [
            {"dimension": dimension, "supporting_team_ids": ["A", "B"]}
            for dimension in certificates
        ]
        return {
            "record_id": record_id,
            "stage3_valid": True,
            "symptom_prediction": "Crash",
            "root_cause_prediction": "Incorrect Code Logic",
            "audit": {
                "stage3": {
                    "baseline_anchor": {
                        "valid": valid,
                        "symptom_label": symptom,
                        "root_cause_label": root,
                    },
                    "pre_gate_candidate": {
                        "symptom_label": pre[0],
                        "root_cause_label": pre[1],
                    },
                    "preservation_result": {
                        "symptom_action": actions[0],
                        "root_cause_action": actions[1],
                        "applied_certificates": certificate_rows,
                    },
                    "revision_certificates": certificate_rows,
                    "reports": reports or [],
                    "component_failures": [],
                    "validity_report": {"quarantined": []},
                    "unresolved": None,
                }
            },
        }

    rows = [
        row(
            "r1",
            baseline=("Crash", "Incorrect Code Logic", True),
            pre=("Incorrect Functionality", "Incorrect Code Logic"),
            actions=("preserved", "preserved"),
            reports=[
                {
                    "team_id": "A",
                    "baseline_revision_assessments": [
                        assessment("preserve", "symptom")
                    ],
                    "revision_consistency": [],
                },
                {
                    "team_id": "B",
                    "baseline_revision_assessments": [
                        assessment("insufficient", "symptom")
                    ],
                    "revision_consistency": [],
                },
            ],
        ),
        row(
            "r2",
            baseline=("Incorrect Functionality", "API Misuse", True),
            pre=("Crash", "Incorrect Code Logic"),
            actions=("revised", "revised"),
            certificates=("symptom", "root_cause"),
            reports=[
                {
                    "team_id": team,
                    "baseline_revision_assessments": [
                        assessment("revise", "symptom", "Crash"),
                        assessment("revise", "root_cause", "Incorrect Code Logic"),
                    ],
                    "revision_consistency": [
                        {"dimension": "symptom", "status": "consistent"},
                        {"dimension": "root_cause", "status": "consistent"},
                    ],
                }
                for team in ("A", "B")
            ],
        ),
        row(
            "r3",
            baseline=(None, None, False),
            pre=("Crash", "API Misuse"),
            actions=(
                "baseline_unavailable_fallback",
                "baseline_unavailable_fallback",
            ),
        ),
    ]

    metrics = baseline_preservation_diagnostics(records, rows)

    assert metrics["n"] == 3
    assert metrics["baseline"]["joint"] == {
        "correct_count": 1,
        "accuracy": pytest.approx(1 / 3),
    }
    assert metrics["baseline"]["coverage"] == pytest.approx(2 / 3)
    assert metrics["baseline"]["invalid_count"] == 1
    assert metrics["pre_gate"]["joint"]["accuracy"] == pytest.approx(1 / 3)
    assert metrics["pre_gate"]["coverage"] == 1.0
    assert metrics["pre_gate"]["invalid_count"] == 0
    assert metrics["post_gate"]["joint"]["accuracy"] == 1.0
    assert metrics["post_gate"]["coverage"] == 1.0
    assert metrics["post_gate"]["invalid_count"] == 0
    assert metrics["actions"]["symptom"] == {
        "preserved": 1,
        "revised": 1,
        "baseline_unavailable_fallback": 1,
    }
    assert metrics["certificates"] == {
        "attempted_count": 4,
        "issued_count": 2,
        "applied_count": 2,
        "revision_precision": 1.0,
    }
    assert metrics["paired"] == {
        "baseline_harm_prevented_count": 1,
        "baseline_correct_to_final_wrong_count": 0,
        "baseline_wrong_to_final_correct_count": 1,
        "net_corrections": 1,
    }
    assert metrics["roles"]["assessment_verdict_counts"] == {
        "insufficient": 1,
        "preserve": 1,
        "revise": 4,
    }
    assert metrics["roles"]["revision_agreement_count"] == 2
    assert metrics["roles"]["consistency_pass_count"] == 4
    assert metrics["safety"]["hardstop_preserved_count"] == 0


def test_preservation_safety_metrics_classify_structured_hardstops_and_uncertainty() -> (
    None
):
    def audit_row(
        record_id: str,
        *,
        valid: bool,
        stop_reason: str | None,
        actions: tuple[str, str] | None = None,
        verification_errors: list[str] | None = None,
        component_failures: list[dict[str, str]] | None = None,
    ) -> dict[str, object]:
        preservation = (
            {
                "symptom_action": actions[0],
                "root_cause_action": actions[1],
                "applied_certificates": [],
            }
            if actions
            else None
        )
        return {
            "record_id": record_id,
            "stage3_valid": valid,
            "symptom_prediction": "Crash" if valid else None,
            "root_cause_prediction": "Incorrect Code Logic" if valid else None,
            "audit": {
                "stage3": {
                    "baseline_anchor": {
                        "valid": True,
                        "symptom_label": "Crash",
                        "root_cause_label": "Incorrect Code Logic",
                    },
                    "pre_gate_candidate": {
                        "symptom_label": "Crash",
                        "root_cause_label": "Incorrect Code Logic",
                    },
                    "preservation_result": preservation,
                    "revision_certificates": [],
                    "reports": [],
                    "verification": {
                        "valid": not verification_errors,
                        "errors": verification_errors or [],
                    },
                    "component_failures": component_failures or [],
                    "validity_report": {"quarantined": []},
                    "unresolved": (
                        {"stop_reason": stop_reason, "missing_facts": []}
                        if stop_reason
                        else None
                    ),
                }
            },
        }

    rows = [
        audit_row(
            "r1",
            valid=False,
            stop_reason="stage3_candidate_invalid_citation",
            verification_errors=["unknown evidence citation missing-id"],
        ),
        audit_row(
            "r2",
            valid=False,
            stop_reason="baseline_preservation_gate_failed",
            verification_errors=["root citation is not capable for this dimension"],
            component_failures=[
                {"error": "expected anchor digest does not match trusted digest"}
            ],
        ),
        audit_row(
            "r3",
            valid=True,
            stop_reason="no_consistent_stage3_candidate",
            actions=("preserved", "preserved"),
        ),
        audit_row(
            "r4",
            valid=True,
            stop_reason=None,
            actions=("preserved", "preserved"),
        ),
    ]

    metrics = baseline_preservation_diagnostics(_gold("r1", "r2", "r3", "r4"), rows)

    assert metrics["post_gate"]["coverage"] == 0.5
    assert metrics["post_gate"]["invalid_count"] == 2
    assert metrics["safety"] == {
        "unknown_citation_count": 1,
        "incapable_citation_count": 1,
        "trust_or_context_failure_count": 1,
        "hardstop_count": 2,
        "digest_mismatch_count": 1,
        "ordinary_uncertainty_preserved_count": 1,
        "hardstop_preserved_count": 0,
    }


def _gold(*ids: str) -> list[dict[str, str]]:
    return [
        {
            "record_id": record_id,
            "decision": "accepted_fault",
            "symptom": "Crash",
            "root_cause": "Incorrect Code Logic",
        }
        for record_id in ids
    ]


def test_error_category_diagnostics_is_exported_by_workflow_package() -> None:
    assert (
        workflow_package.stage3_error_category_diagnostics
        is evaluation_module.stage3_error_category_diagnostics
    )


def _row(record_id: str, *, correct: bool, team_a: bool = True, team_b: bool = True):
    symptom = "Crash" if correct else "Incorrect Functionality"
    root = "Incorrect Code Logic" if correct else "API Misuse"
    return {
        "record_id": record_id,
        "stage3_valid": True,
        "symptom_prediction": symptom,
        "root_cause_prediction": root,
        "audit": {
            "stage3": {
                "reports": [
                    {
                        "team_id": "A",
                        "symptom": {
                            "label": "Crash" if team_a else "Incorrect Functionality"
                        },
                        "root_cause": {
                            "label": "Incorrect Code Logic" if team_a else "API Misuse"
                        },
                    },
                    {
                        "team_id": "B",
                        "symptom": {
                            "label": "Crash" if team_b else "Incorrect Functionality"
                        },
                        "root_cause": {
                            "label": "Incorrect Code Logic" if team_b else "API Misuse"
                        },
                    },
                ],
                "final_decision": {"source": "direct_consensus"},
            }
        },
    }


def _prediction(
    record_id: str,
    *,
    symptom: str | None,
    root_cause: str | None,
    stage3_valid: bool = True,
    stage3_audit: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "record_id": record_id,
        "stage3_valid": stage3_valid,
        "symptom_prediction": symptom,
        "root_cause_prediction": root_cause,
        "audit": {"stage3": stage3_audit or {}},
    }


def test_error_category_diagnostics_reports_recovery_and_baseline_harm() -> None:
    records = _gold("r1", "r2", "r3")
    baseline = [
        _prediction(
            "r1", symptom="Incorrect Functionality", root_cause="Incorrect Code Logic"
        ),
        _prediction("r2", symptom="Crash", root_cause="API Misuse"),
        _prediction("r3", symptom="Crash", root_cause="Incorrect Code Logic"),
    ]
    candidate = [
        _prediction("r1", symptom="Crash", root_cause="Incorrect Code Logic"),
        _prediction("r2", symptom="Incorrect Functionality", root_cause="API Misuse"),
        _prediction("r3", symptom="Crash", root_cause="Incorrect Code Logic"),
    ]

    metrics = evaluation_module.stage3_error_category_diagnostics(
        records,
        baseline,
        candidate,
        [
            {"record_id": "r1", "dimension": "symptom", "category": "S1"},
            {"record_id": "r2", "dimension": "root_cause", "category": "R2"},
        ],
    )

    assert metrics["n"] == 3
    assert metrics["categories"]["S1"] == {
        "dimension": "symptom",
        "n": 1,
        "recovery_count": 1,
        "recovery_rate": 1.0,
        "recovered_record_ids": ["r1"],
        "still_wrong_record_ids": [],
    }
    assert metrics["categories"]["R2"] == {
        "dimension": "root_cause",
        "n": 1,
        "recovery_count": 0,
        "recovery_rate": 0.0,
        "recovered_record_ids": [],
        "still_wrong_record_ids": ["r2"],
    }
    assert metrics["baseline_preservation"]["symptom"] == {
        "baseline_correct_count": 2,
        "preserved_correct_count": 1,
        "harm_count": 1,
        "preservation_rate": 0.5,
        "harmed_record_ids": ["r2"],
    }
    assert metrics["baseline_preservation"]["root_cause"] == {
        "baseline_correct_count": 2,
        "preserved_correct_count": 2,
        "harm_count": 0,
        "preservation_rate": 1.0,
        "harmed_record_ids": [],
    }
    assert metrics["baseline_preservation"]["joint"] == {
        "baseline_correct_count": 1,
        "preserved_correct_count": 1,
        "harm_count": 0,
        "preservation_rate": 1.0,
        "harmed_record_ids": [],
    }
    assert metrics["baseline_errors"]["joint"] == {
        "baseline_error_count": 2,
        "recovery_count": 1,
        "recovery_rate": 0.5,
        "recovered_record_ids": ["r1"],
        "still_wrong_record_ids": ["r2"],
    }


@pytest.mark.parametrize(
    ("categories", "message"),
    (
        (
            [{"record_id": "r1", "dimension": "symptom", "category": "S1"}],
            "uncategorized baseline errors",
        ),
        (
            [
                {"record_id": "r1", "dimension": "symptom", "category": "S1"},
                {"record_id": "r1", "dimension": "symptom", "category": "S2"},
                {"record_id": "r2", "dimension": "root_cause", "category": "R2"},
            ],
            "duplicate error category",
        ),
        (
            [
                {"record_id": "r1", "dimension": "symptom", "category": "S1"},
                {"record_id": "r2", "dimension": "root_cause", "category": "R2"},
                {"record_id": "r3", "dimension": "symptom", "category": "S1"},
            ],
            "category marks a baseline-correct prediction",
        ),
        (
            [
                {"record_id": "r1", "dimension": "symptom", "category": "S1"},
                {"record_id": "r2", "dimension": "root_cause", "category": "R2"},
                {"record_id": "unknown", "dimension": "symptom", "category": "S1"},
            ],
            "unknown category record_id",
        ),
        (
            [
                {"record_id": "r1", "dimension": "decision", "category": "S1"},
                {"record_id": "r2", "dimension": "root_cause", "category": "R2"},
            ],
            "invalid error category dimension",
        ),
    ),
)
def test_error_category_diagnostics_fails_closed_on_invalid_audit_manifest(
    categories: list[dict[str, str]], message: str
) -> None:
    records = _gold("r1", "r2", "r3")
    baseline = [
        _prediction(
            "r1", symptom="Incorrect Functionality", root_cause="Incorrect Code Logic"
        ),
        _prediction("r2", symptom="Crash", root_cause="API Misuse"),
        _prediction("r3", symptom="Crash", root_cause="Incorrect Code Logic"),
    ]
    candidate = list(baseline)

    with pytest.raises(ValueError, match=message):
        evaluation_module.stage3_error_category_diagnostics(
            records, baseline, candidate, categories
        )


def test_error_category_diagnostics_exposes_instrumentation_coverage() -> None:
    records = _gold("r1", "r2")
    baseline = [
        _prediction(
            "r1", symptom="Incorrect Functionality", root_cause="Incorrect Code Logic"
        ),
        _prediction("r2", symptom="Crash", root_cause="API Misuse"),
    ]
    candidate = [
        _prediction(
            "r1",
            symptom="Crash",
            root_cause="Incorrect Code Logic",
            stage3_audit={
                "evidence_deltas": [
                    {"request_id": "q1", "status": "found"},
                    {"request_id": "q2", "status": "absent"},
                ],
                "boundary_card_ids": ["symptom/crash-vs-initialization"],
            },
        ),
        _prediction(
            "r2",
            symptom="Crash",
            root_cause="API Misuse",
            stage3_audit={"evidence_deltas": [], "boundary_card_ids": []},
        ),
    ]

    metrics = evaluation_module.stage3_error_category_diagnostics(
        records,
        baseline,
        candidate,
        [
            {"record_id": "r1", "dimension": "symptom", "category": "S1"},
            {"record_id": "r2", "dimension": "root_cause", "category": "R2"},
        ],
    )

    assert metrics["evidence_expansion"] == {
        "instrumented": True,
        "instrumented_record_count": 2,
        "missing_record_count": 0,
        "request_count": 2,
        "found_count": 1,
        "hit_rate": 0.5,
        "records_with_found_evidence_count": 1,
    }
    assert metrics["boundary_cards"] == {
        "instrumented": True,
        "instrumented_record_count": 2,
        "missing_record_count": 0,
        "usage_count": 1,
        "record_usage_count": 1,
        "used_card_ids": ["symptom/crash-vs-initialization"],
    }


def test_error_category_diagnostics_does_not_report_missing_instrumentation_as_zero() -> (
    None
):
    records = _gold("r1")
    baseline = [
        _prediction(
            "r1", symptom="Incorrect Functionality", root_cause="Incorrect Code Logic"
        )
    ]
    candidate = [_prediction("r1", symptom="Crash", root_cause="Incorrect Code Logic")]

    metrics = evaluation_module.stage3_error_category_diagnostics(
        records,
        baseline,
        candidate,
        [{"record_id": "r1", "dimension": "symptom", "category": "S1"}],
    )

    assert metrics["evidence_expansion"] == {
        "instrumented": False,
        "instrumented_record_count": 0,
        "missing_record_count": 1,
        "request_count": 0,
        "found_count": 0,
        "hit_rate": None,
        "records_with_found_evidence_count": 0,
    }
    assert metrics["boundary_cards"] == {
        "instrumented": False,
        "instrumented_record_count": 0,
        "missing_record_count": 1,
        "usage_count": 0,
        "record_usage_count": 0,
        "used_card_ids": [],
    }


def test_revision_funnel_uses_canonical_audit_and_counts_failed_calls() -> None:
    records = _gold("r1", "r2")
    rows = [
        _prediction(
            "r1",
            symptom="Crash",
            root_cause="Incorrect Code Logic",
            stage3_audit={
                "baseline_anchor": {
                    "valid": True,
                    "symptom_label": "Crash",
                    "root_cause_label": "API Misuse",
                },
                "pre_gate_candidate": {
                    "symptom_label": "Crash",
                    "root_cause_label": "Incorrect Code Logic",
                },
                "preservation_result": {
                    "symptom_action": "preserved",
                    "root_cause_action": "preserved",
                    "applied_certificates": [],
                },
                "revision_certificates": [],
                "reports": [],
                "component_failures": [],
                "revision_audit": {
                    "proposal_digests": ["a" * 64],
                    "routed_card_ids": ["root/api-vs-logic"],
                    "proposal_entries": [
                        {
                            "proposal_digest": "a" * 64,
                            "dimension": "root_cause",
                            "boundary_card_id": "root/api-vs-logic",
                            "baseline_label": "API Misuse",
                            "proposed_label": "Incorrect Code Logic",
                        }
                    ],
                    "assessment_attempts": [
                        {
                            "proposal_digest": "a" * 64,
                            "dimension": "root_cause",
                            "boundary_card_id": "root/api-vs-logic",
                            "assessor_team_id": "A",
                            "returned_assessment_digest": None,
                            "returned_assessment": None,
                            "verdict": None,
                            "operational_failure": True,
                        },
                        {
                            "proposal_digest": "a" * 64,
                            "dimension": "root_cause",
                            "boundary_card_id": "root/api-vs-logic",
                            "assessor_team_id": "B",
                            "returned_assessment_digest": None,
                            "verdict": None,
                            "operational_failure": True,
                        },
                    ],
                    "dual_revise_proposal_digests": [],
                    "consistency_attempts": [],
                    "consistency_pass_proposal_digests": [],
                    "issued_certificate_digests": [],
                    "issued_certificate_entries": [],
                    "applied_certificate_digests": [],
                    "ambiguous_certified_dimensions": [],
                    "failed_proposal_digests": ["a" * 64],
                },
            },
        ),
        _prediction(
            "r2",
            symptom="Crash",
            root_cause="Incorrect Code Logic",
            stage3_audit={
                "baseline_anchor": {
                    "valid": True,
                    "symptom_label": "Crash",
                    "root_cause_label": "Incorrect Code Logic",
                },
                "pre_gate_candidate": {
                    "symptom_label": "Crash",
                    "root_cause_label": "Incorrect Code Logic",
                },
                "preservation_result": {
                    "symptom_action": "preserved",
                    "root_cause_action": "preserved",
                    "applied_certificates": [],
                },
                "revision_certificates": [],
                "reports": [],
                "component_failures": [],
                "revision_audit": {
                    "proposal_digests": [],
                    "routed_card_ids": [],
                    "assessment_attempts": [],
                    "dual_revise_proposal_digests": [],
                    "consistency_attempts": [],
                    "consistency_pass_proposal_digests": [],
                    "issued_certificate_digests": [],
                    "applied_certificate_digests": [],
                    "ambiguous_certified_dimensions": [],
                    "failed_proposal_digests": [],
                },
            },
        ),
    ]

    metrics = baseline_preservation_diagnostics(records, rows)

    assert metrics["revision_funnel"] == {
        "instrumented": True,
        "instrumented_record_count": 2,
        "missing_record_count": 0,
        "proposal_count": 1,
        "routed_card_count": 1,
        "missing_route_count": 0,
        "assessment_attempt_count": 2,
        "assessment_return_count": 0,
        "dual_assessed_count": 0,
        "dual_revise_count": 0,
        "cross_attempt_count": 0,
        "cross_return_count": 0,
        "cross_pass_count": 0,
        "issued_count": 0,
        "applied_count": 0,
        "ambiguous_dimension_count": 0,
        "operational_failure_count": 2,
        "failed_proposal_count": 1,
        "applied_help_count": 0,
        "applied_harm_count": 0,
        "per_card": {
            "root/api-vs-logic": {
                "proposal_count": 1,
                "assessment_attempt_count": 2,
                "dual_revise_count": 0,
                "cross_attempt_count": 0,
                "issued_count": 0,
                "applied_count": 0,
                "help_count": 0,
                "harm_count": 0,
            }
        },
    }


def test_boundary_card_instrumentation_uses_revision_audit_without_legacy_field() -> (
    None
):
    records = _gold("r1")
    baseline = [
        _prediction("r1", symptom="Incorrect Functionality", root_cause="API Misuse")
    ]
    candidate = [
        _prediction(
            "r1",
            symptom="Crash",
            root_cause="Incorrect Code Logic",
            stage3_audit={
                "evidence_deltas": [],
                "revision_audit": {
                    "proposal_digests": ["a" * 64],
                    "routed_card_ids": ["root/api-vs-logic"],
                    "proposal_entries": [
                        {
                            "proposal_digest": "a" * 64,
                            "dimension": "root_cause",
                            "boundary_card_id": "root/api-vs-logic",
                            "baseline_label": "API Misuse",
                            "proposed_label": "Incorrect Code Logic",
                        }
                    ],
                    "assessment_attempts": [],
                    "dual_revise_proposal_digests": [],
                    "consistency_attempts": [],
                    "consistency_pass_proposal_digests": [],
                    "issued_certificate_digests": [],
                    "issued_certificate_entries": [],
                    "applied_certificate_digests": [],
                    "ambiguous_certified_dimensions": [],
                    "failed_proposal_digests": [],
                },
            },
        )
    ]

    metrics = evaluation_module.stage3_error_category_diagnostics(
        records,
        baseline,
        candidate,
        [
            {"record_id": "r1", "dimension": "symptom", "category": "S1"},
            {"record_id": "r1", "dimension": "root_cause", "category": "R2"},
        ],
    )

    assert metrics["boundary_cards"] == {
        "instrumented": True,
        "instrumented_record_count": 1,
        "missing_record_count": 0,
        "usage_count": 1,
        "record_usage_count": 1,
        "used_card_ids": ["root/api-vs-logic"],
    }


def test_revision_funnel_rejects_certificate_that_does_not_match_audited_proposal() -> (
    None
):
    def digest(value: dict[str, object]) -> str:
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    assessments = {
        team: {
            "assessor_team_id": team,
            "proposal_digest": "a" * 64,
            "dimension": "root_cause",
            "verdict": "revise",
            "baseline_label": "API Misuse",
            "proposed_label": "Incorrect Code Logic",
            "boundary_card_id": "root/api-vs-logic",
            "contradicted_baseline_condition": "The baseline condition is contradicted.",
            "satisfied_proposed_conditions": ["The proposed condition is satisfied."],
            "supporting_evidence_ids": ["e1"],
            "counter_evidence_ids": ["e1"],
            "baseline_source_config_hash": "1" * 64,
            "baseline_source_predictions_sha256": "2" * 64,
            "taxonomy_structure_hash": "3" * 64,
            "evidence_view_hash": "4" * 64,
        }
        for team in ("A", "B")
    }
    consistencies = {
        checker: {
            "checker_team_id": checker,
            "assessment_owner_team_id": owner,
            "proposal_digest": "a" * 64,
            "dimension": "root_cause",
            "baseline_label": "API Misuse",
            "proposed_label": "Incorrect Code Logic",
            "status": "consistent",
            "rationale": "The opposite assessment is causally and taxonomically consistent.",
            "supporting_evidence_ids": ["e1"],
            "counter_evidence_ids": ["e1"],
            "assessment_digest": digest(assessments[owner]),
            "boundary_card_id": "root/api-vs-logic",
            "baseline_source_config_hash": "1" * 64,
            "baseline_source_predictions_sha256": "2" * 64,
            "taxonomy_structure_hash": "3" * 64,
            "evidence_view_hash": "4" * 64,
        }
        for checker, owner in (("A", "B"), ("B", "A"))
    }
    forged_certificate = {
        "dimension": "root_cause",
        "proposal_digest": "b" * 64,
        "baseline_label": "API Misuse",
        "proposed_label": "Incorrect Code Logic",
        "revision_basis": "taxonomy_boundary",
        "boundary_card_id": "root/wrong-card",
        "contradicted_baseline_condition": "The baseline condition is contradicted.",
        "satisfied_proposed_conditions": ["The proposed condition is satisfied."],
        "supporting_evidence_ids": ["e1"],
        "counter_evidence_ids": ["e1"],
        "supporting_team_ids": ["A", "B"],
    }
    certificate_digest = digest(forged_certificate)
    audit = {
        "proposal_digests": ["a" * 64],
        "routed_card_ids": ["root/api-vs-logic"],
        "proposal_entries": [
            {
                "proposal_digest": "a" * 64,
                "dimension": "root_cause",
                "boundary_card_id": "root/api-vs-logic",
                "baseline_label": "API Misuse",
                "proposed_label": "Incorrect Code Logic",
            }
        ],
        "assessment_attempts": [
            {
                "proposal_digest": "a" * 64,
                "dimension": "root_cause",
                "boundary_card_id": "root/api-vs-logic",
                "assessor_team_id": team,
                "returned_assessment_digest": digest(assessments[team]),
                "returned_assessment": assessments[team],
                "verdict": "revise",
                "operational_failure": False,
            }
            for team in ("A", "B")
        ],
        "dual_revise_proposal_digests": ["a" * 64],
        "consistency_attempts": [
            {
                "proposal_digest": "a" * 64,
                "dimension": "root_cause",
                "boundary_card_id": "root/api-vs-logic",
                "checker_team_id": checker,
                "assessment_owner_team_id": owner,
                "returned_consistency_digest": digest(consistencies[checker]),
                "returned_consistency": consistencies[checker],
                "status": "consistent",
                "operational_failure": False,
            }
            for checker, owner in (("A", "B"), ("B", "A"))
        ],
        "consistency_pass_proposal_digests": ["a" * 64],
        "issued_certificate_digests": [certificate_digest],
        "issued_certificate_entries": [
            {
                "certificate_digest": certificate_digest,
                "proposal_digest": "a" * 64,
                "dimension": "root_cause",
                "boundary_card_id": "root/api-vs-logic",
                "baseline_label": "API Misuse",
                "proposed_label": "Incorrect Code Logic",
            }
        ],
        "applied_certificate_digests": [certificate_digest],
        "ambiguous_certified_dimensions": [],
        "failed_proposal_digests": [],
    }
    rows = [
        _prediction(
            "r1",
            symptom="Crash",
            root_cause="Incorrect Code Logic",
            stage3_audit={
                "baseline_anchor": {
                    "valid": True,
                    "symptom_label": "Crash",
                    "root_cause_label": "API Misuse",
                },
                "pre_gate_candidate": {
                    "symptom_label": "Crash",
                    "root_cause_label": "Incorrect Code Logic",
                },
                "reports": [
                    {
                        "team_id": team,
                        "baseline_revision_assessments": [assessments[team]],
                        "revision_consistency": [consistencies[team]],
                    }
                    for team in ("A", "B")
                ],
                "revision_certificates": [forged_certificate],
                "preservation_result": {
                    "symptom_action": "preserved",
                    "root_cause_action": "revised",
                    "applied_certificates": [forged_certificate],
                },
                "component_failures": [],
                "revision_audit": audit,
            },
        )
    ]

    with pytest.raises(ValueError, match="canonical proposal chain"):
        baseline_preservation_diagnostics(_gold("r1"), rows)


def _anchored_report(
    team_id: str,
    *,
    anchor: tuple[str, str] | None,
    composed: tuple[str, str],
    accepted: tuple[bool, bool] = (False, False),
) -> dict[str, object]:
    report: dict[str, object] = {
        "team_id": team_id,
        "symptom": {"label": composed[0]},
        "root_cause": {"label": composed[1]},
        "correction_audit": [
            {"dimension": "symptom", "accepted": accepted[0]},
            {"dimension": "root_cause", "accepted": accepted[1]},
        ],
    }
    if anchor is not None:
        report["anchor"] = {
            "symptom": {"label": anchor[0]},
            "root_cause": {"label": anchor[1]},
        }
    return report


def _anchored_row(
    record_id: str,
    reports: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "record_id": record_id,
        "stage3_valid": True,
        "symptom_prediction": "Crash",
        "root_cause_prediction": "Incorrect Code Logic",
        "audit": {
            "stage3": {
                "reports": reports,
                "final_decision": {"source": "direct_consensus"},
            }
        },
    }


def test_paired_comparison_rejects_duplicate_and_mismatched_record_sets() -> None:
    with pytest.raises(ValueError, match="duplicate record_id"):
        paired_stage3_comparison(
            [_row("r1", correct=True), _row("r1", correct=True)],
            [_row("r1", correct=True)],
            _gold("r1"),
        )

    with pytest.raises(ValueError, match="duplicate record_id in gold"):
        paired_stage3_comparison(
            [_row("r1", correct=True)],
            [_row("r1", correct=True)],
            _gold("r1", "r1"),
        )

    with pytest.raises(ValueError, match="record sets differ"):
        paired_stage3_comparison(
            [_row("r1", correct=True)],
            [_row("r2", correct=True)],
            _gold("r1", "r2"),
        )


def test_paired_comparison_reports_hand_checked_counts_and_exact_test() -> None:
    records = _gold("r1", "r2", "r3", "r4", "r5", "r6")
    candidate = [
        _row("r1", correct=True),
        _row("r2", correct=True),
        _row("r3", correct=True),
        _row("r4", correct=False),
        _row("r5", correct=True),
        _row("r6", correct=False),
    ]
    baseline = [
        _row("r1", correct=False),
        _row("r2", correct=False),
        _row("r3", correct=False),
        _row("r4", correct=True),
        _row("r5", correct=True),
        _row("r6", correct=False),
    ]

    result = paired_stage3_comparison(
        candidate,
        baseline,
        records,
        bootstrap_samples=500,
        bootstrap_seed=17,
    )

    assert (result["wins"], result["losses"], result["ties"]) == (3, 1, 2)
    assert result["joint_accuracy_delta"] == pytest.approx(2 / 6)
    assert result["mcnemar_exact_two_sided_p"] == pytest.approx(0.625)
    assert result["bootstrap"]["seed"] == 17
    assert result["bootstrap"]["samples"] == 500
    assert result == paired_stage3_comparison(
        candidate,
        baseline,
        records,
        bootstrap_samples=500,
        bootstrap_seed=17,
    )


def test_exact_mcnemar_five_to_zero_is_ten_points_but_not_significant() -> None:
    ids = tuple(f"r{index}" for index in range(1, 51))
    result = paired_stage3_comparison(
        [_row(record_id, correct=index <= 5) for index, record_id in enumerate(ids, 1)],
        [_row(record_id, correct=False) for record_id in ids],
        _gold(*ids),
        bootstrap_samples=100,
        bootstrap_seed=2,
    )
    assert result["wins"] == 5
    assert result["losses"] == 0
    assert result["joint_accuracy_delta"] == pytest.approx(0.10)
    assert result["mcnemar_exact_two_sided_p"] == pytest.approx(0.0625)


def test_team_diagnostics_reports_standalone_accuracy_and_error_overlap() -> None:
    records = _gold("r1", "r2", "r3")
    rows = [
        _row("r1", correct=True, team_a=True, team_b=True),
        _row("r2", correct=True, team_a=True, team_b=False),
        _row("r3", correct=False, team_a=False, team_b=False),
    ]

    result = stage3_team_diagnostics(records, rows)

    assert result["team_a"]["joint_accuracy"] == pytest.approx(2 / 3)
    assert result["team_b"]["joint_accuracy"] == pytest.approx(1 / 3)
    assert result["label_agreement_count"] == 2
    assert result["both_wrong_count"] == 1
    assert result["oracle_union_joint_accuracy"] == pytest.approx(2 / 3)
    assert result["resolution_source_counts"] == {"direct_consensus": 3}
    assert result["direct_certificate_accuracy"] == pytest.approx(2 / 3)


def test_team_diagnostics_reports_challenger_yield_and_arbitration_effect() -> None:
    records = _gold("rescued", "harmed", "pass")
    rescued = _row("rescued", correct=True, team_a=False, team_b=False)
    harmed = _row("harmed", correct=False, team_a=True, team_b=False)
    passed = _row("pass", correct=True, team_a=True, team_b=True)
    rescued["audit"]["stage3"].update(
        {
            "boundary_challenge": {"action": "cause_review"},
            "final_decision": {"source": "targeted_arbitration"},
        }
    )
    harmed["audit"]["stage3"].update(
        {
            "boundary_challenge": {"action": "symptom_review"},
            "final_decision": {"source": "targeted_arbitration"},
        }
    )
    passed["audit"]["stage3"]["boundary_challenge"] = {"action": "pass"}

    result = stage3_team_diagnostics(records, [rescued, harmed, passed])

    assert result["challenger_action_counts"] == {
        "cause_review": 1,
        "pass": 1,
        "symptom_review": 1,
    }
    assert result["challenger_yield_count"] == 2
    assert result["challenger_yield_rate"] == pytest.approx(2 / 3)
    assert result["arbitration_flip_help_count"] == 1
    assert result["arbitration_flip_harm_count"] == 1


def test_team_diagnostics_uses_fixed_denominator_when_a_record_never_reaches_teams() -> (
    None
):
    records = _gold("r1", "r2")
    rows = [
        _row("r1", correct=True),
        {
            "record_id": "r2",
            "stage3_valid": False,
            "symptom_prediction": None,
            "root_cause_prediction": None,
            "audit": {"stage3": {"reports": []}},
        },
    ]

    result = stage3_team_diagnostics(records, rows)

    assert result["both_team_reach_count"] == 1
    assert result["both_team_reach_rate"] == 0.5
    assert result["team_a"]["joint_accuracy"] == 0.5
    assert result["team_b"]["joint_accuracy"] == 0.5


def test_team_diagnostics_counts_one_report_as_degraded_reach_only() -> None:
    row = _row("r1", correct=True)
    row["audit"]["stage3"]["reports"] = [row["audit"]["stage3"]["reports"][1]]
    row["audit"]["stage3"]["final_decision"] = {"source": "single_team_degraded"}

    result = stage3_team_diagnostics(_gold("r1"), [row])

    assert result["single_team_degraded_reach_count"] == 1
    assert result["both_team_reach_count"] == 0
    assert result["team_a"]["joint_accuracy"] == 0.0
    assert result["team_b"]["joint_accuracy"] == 1.0


def test_team_diagnostics_rejects_duplicate_or_extra_team_reports() -> None:
    row = _row("r1", correct=True)
    reports = row["audit"]["stage3"]["reports"]
    reports.append(dict(reports[0]))

    with pytest.raises(ValueError, match="exactly teams A and B"):
        stage3_team_diagnostics(_gold("r1"), [row])


def test_anchor_and_correction_metrics_use_fixed_gold_denominator() -> None:
    wrong = ("Incorrect Functionality", "API Misuse")
    correct = ("Crash", "Incorrect Code Logic")
    rows = [
        _anchored_row(
            "r1",
            [
                _anchored_report(
                    "A", anchor=wrong, composed=correct, accepted=(True, False)
                ),
                _anchored_report("B", anchor=correct, composed=correct),
            ],
        ),
        _anchored_row(
            "r2",
            [
                _anchored_report(
                    "A", anchor=correct, composed=wrong, accepted=(True, True)
                ),
                _anchored_report("B", anchor=None, composed=wrong),
            ],
        ),
        _anchored_row(
            "r3",
            [
                _anchored_report(
                    "A",
                    anchor=wrong,
                    composed=("Unexpected Output", "API Misuse"),
                ),
                _anchored_report("B", anchor=wrong, composed=wrong),
            ],
        ),
    ]

    metrics = stage3_team_diagnostics(_gold("r1", "r2", "r3"), rows)

    assert metrics["team_a"]["anchor_reach_count"] == 3
    assert metrics["team_a"]["anchor_reach_rate"] == 1.0
    assert metrics["team_a"]["anchor_joint_correct_count"] == 1
    assert metrics["team_a"]["anchor_joint_accuracy"] == pytest.approx(1 / 3)
    assert metrics["team_a"]["composed_joint_correct_count"] == 1
    assert metrics["team_a"]["composed_joint_accuracy"] == pytest.approx(1 / 3)
    assert metrics["team_a"]["joint_correct_count"] == 1
    assert metrics["team_a"]["joint_accuracy"] == pytest.approx(1 / 3)
    assert metrics["team_b"]["anchor_reach_count"] == 2
    assert metrics["team_b"]["anchor_reach_rate"] == pytest.approx(2 / 3)
    assert metrics["anchor_oracle_union_joint_accuracy"] == pytest.approx(2 / 3)
    assert metrics["composed_oracle_union_joint_accuracy"] == pytest.approx(1 / 3)
    assert metrics["oracle_union_joint_accuracy"] == pytest.approx(1 / 3)
    assert metrics["anchor_preservation_count"] == 2
    assert metrics["anchor_preservation_rate"] == pytest.approx(2 / 5)
    assert metrics["corrections"] == {
        "accepted_count": 3,
        "help_count": 1,
        "harm_count": 1,
        "neutral_changed_count": 1,
        "net_help_count": 0,
        "directional_precision": pytest.approx(0.5),
    }


def test_anchor_metrics_keep_missing_team_and_anchor_in_gold_denominator() -> None:
    correct = ("Crash", "Incorrect Code Logic")
    rows = [
        _anchored_row("r1", [_anchored_report("A", anchor=correct, composed=correct)]),
        {
            "record_id": "r2",
            "audit": {"stage3": {"reports": []}},
        },
    ]

    metrics = stage3_team_diagnostics(_gold("r1", "r2"), rows)

    assert metrics["team_a"]["anchor_reach_count"] == 1
    assert metrics["team_a"]["anchor_joint_accuracy"] == 0.5
    assert metrics["team_b"]["anchor_reach_count"] == 0
    assert metrics["team_b"]["composed_joint_accuracy"] == 0.0
    assert metrics["anchor_oracle_union_joint_accuracy"] == 0.5
    assert metrics["composed_oracle_union_joint_accuracy"] == 0.5


def test_legacy_team_audit_has_zero_anchor_metrics_and_unchanged_aliases() -> None:
    metrics = stage3_team_diagnostics(
        _gold("r1"),
        [_row("r1", correct=True, team_a=True, team_b=False)],
    )

    assert metrics["team_a"]["anchor_reach_count"] == 0
    assert metrics["team_a"]["anchor_joint_correct_count"] == 0
    assert metrics["team_a"]["anchor_joint_accuracy"] == 0.0
    assert metrics["team_a"]["composed_joint_accuracy"] == 1.0
    assert metrics["team_a"]["joint_accuracy"] == 1.0
    assert metrics["anchor_oracle_union_joint_accuracy"] == 0.0
    assert metrics["composed_oracle_union_joint_accuracy"] == 1.0
    assert metrics["oracle_union_joint_accuracy"] == 1.0
    assert metrics["anchor_preservation_count"] == 0
    assert metrics["anchor_preservation_rate"] is None
    assert metrics["corrections"] == {
        "accepted_count": 0,
        "help_count": 0,
        "harm_count": 0,
        "neutral_changed_count": 0,
        "net_help_count": 0,
        "directional_precision": None,
    }


@pytest.mark.parametrize(
    "report_update",
    (
        {"anchor": []},
        {"anchor": {"symptom": {"label": "Crash"}}},
        {"correction_audit": {}},
        {"correction_audit": [{"dimension": "symptom", "accepted": "yes"}]},
    ),
)
def test_team_diagnostics_rejects_malformed_anchored_report_fields(
    report_update: dict[str, object],
) -> None:
    correct = ("Crash", "Incorrect Code Logic")
    row = _anchored_row(
        "r1",
        [
            {
                **_anchored_report("A", anchor=correct, composed=correct),
                **report_update,
            }
        ],
    )

    with pytest.raises(ValueError, match="malformed Stage 3 team report"):
        stage3_team_diagnostics(_gold("r1"), [row])
