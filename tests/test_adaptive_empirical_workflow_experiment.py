from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from Benchmark.src.adaptive_empirical_workflow.experiment import (
    ProgressSnapshot,
    _write_ordered_jsonl,
    evaluate_experiment,
    run_adaptive_record,
    run_records,
    run_targeted_sla_record,
    select_records,
)
from Benchmark.src.adaptive_empirical_workflow.contracts import (
    BaselineAnchor,
    CausalConsistencyReport,
    ConsistencyStatus,
    DimensionReadiness,
    DimensionVerificationReport,
    EvidenceDimension,
    EvidenceReadinessReport,
    EvidenceRequest,
    EvidenceSufficiency,
    FaultEvidenceAssessment,
    JointAnchorReport,
    RepairCausalityAssessment,
    ResolutionStatus,
    RootCauseReport,
    SlaDecisionStatus,
    ScopeBoundaryAssessment,
    SpecialistType,
    Stage2AnalysisReport,
    Stage2Decision,
    Stage3ArbitrationDecision,
    Stage3ArbitrationPacket,
    SymptomReport,
    TestOutcome as EvidenceTestOutcome,
    VerificationVerdict,
)
from Benchmark.src.adaptive_empirical_workflow.sla_budget import (
    GlobalSlaBudget,
    SlaBudgetConfig,
    SlaBudgetExhausted,
)
from Benchmark.src.adaptive_empirical_workflow.sla_controller import (
    TargetedSlaController,
)


def _sla_anchor(record_id: str) -> BaselineAnchor:
    return BaselineAnchor(
        record_id=record_id,
        valid=True,
        symptom_label="Crash",
        root_cause_label="Incorrect Code Logic",
        source_config_hash="a" * 64,
        source_predictions_sha256="b" * 64,
    )


class _VerifiedSlaController:
    def __init__(self) -> None:
        self.views: list[object] = []

    def run(self, view: object, anchor: object, budget: object) -> object:
        del anchor, budget
        self.views.append(view)
        return SimpleNamespace(
            final_decision=SimpleNamespace(
                symptom_label="Crash",
                root_cause_label="API Misuse",
            ),
            decision_status={
                EvidenceDimension.SYMPTOM: SlaDecisionStatus.MODEL_VERIFIED,
                EvidenceDimension.ROOT_CAUSE: SlaDecisionStatus.MODEL_VERIFIED,
            },
            call_count=2,
            fallback=False,
            budget_remaining_seconds=123.0,
            diagnosis=SimpleNamespace(
                symptom_matches_baseline=True,
                root_cause_matches_baseline=False,
            ),
            verification=SimpleNamespace(
                symptom=SimpleNamespace(
                    verdict=SimpleNamespace(value="accept_candidate"),
                    supporting_evidence_ids=("issue-body",),
                    counter_evidence_ids=(),
                ),
                root_cause=SimpleNamespace(
                    verdict=SimpleNamespace(value="accept_candidate"),
                    supporting_evidence_ids=("issue-body",),
                    counter_evidence_ids=(),
                ),
            ),
            arbitration=None,
        )


class _TelemetrySlaController(_VerifiedSlaController):
    def __init__(self) -> None:
        super().__init__()
        self._calls: list[dict[str, object]] = []
        self._agents = SimpleNamespace(  # noqa: SLF001 - mirrors the controller seam.
            telemetry=lambda: {"calls": list(self._calls)}
        )

    def run(self, view: object, anchor: object, budget: object) -> object:
        result = super().run(view, anchor, budget)
        call_audit = [
            {
                "role": "sla_joint_diagnosis",
                "latency_seconds": 0.25,
                "schema_attempts": 1,
                "network_attempts": 1,
                "provider_errors": [],
            },
            {
                "role": "sla_joint_verifier",
                "latency_seconds": 0.5,
                "schema_attempts": 1,
                "network_attempts": 1,
                "provider_errors": [],
            },
        ]
        self._calls.extend(
            [
                {"role": entry["role"], "latency_seconds": entry["latency_seconds"]}
                for entry in call_audit
            ]
        )
        return SimpleNamespace(**result.__dict__, call_audit=tuple(call_audit))


class _ArbitratedSlaController(_VerifiedSlaController):
    def run(self, view: object, anchor: object, budget: object) -> object:
        result = super().run(view, anchor, budget)
        return SimpleNamespace(
            **{
                **result.__dict__,
                "arbitration": SimpleNamespace(
                    symptom=None,
                    root_cause=SimpleNamespace(
                        selected_label="API Misuse",
                        supporting_evidence_ids=("issue-body",),
                        rationale="Bounded rationale is not persisted.",
                    ),
                ),
            }
        )


class _InterleavingTelemetrySlaController(_VerifiedSlaController):
    """Exposes the old shared telemetry shape while returning per-record audits."""

    def __init__(self) -> None:
        super().__init__()
        self._calls: list[dict[str, object]] = []
        self._lock = threading.Lock()
        self._barrier = threading.Barrier(2)
        self._second_returned = threading.Event()
        self._agents = SimpleNamespace(  # noqa: SLF001 - regression seam.
            telemetry=lambda: {"calls": list(self._calls)}
        )

    def run(self, view: object, anchor: object, budget: object) -> object:
        result = super().run(view, anchor, budget)
        record_id = view.record_id
        audit = {
            "role": f"sla_{record_id}",
            "latency_seconds": 0.25,
            "schema_attempts": 2,
            "network_attempts": 3,
            "provider_errors": [
                {
                    "status": "retryable_error",
                    "retryable": False,
                    "cause_type": "timeout_error",
                    "http_status": None,
                    "retry_after_seconds": None,
                }
            ],
        }
        with self._lock:
            self._calls.append(dict(audit))
        self._barrier.wait(timeout=2)
        if record_id == "sla-first":
            assert self._second_returned.wait(timeout=2)
        else:
            self._second_returned.set()
        return SimpleNamespace(**result.__dict__, call_audit=(audit,))


def _sla_record(record_id: str = "sla-record") -> dict[str, str]:
    return {
        "record_id": record_id,
        "title": "Request crashes",
        "body": "Calling the API before initialization terminates the request.",
    }


def _sla_taxonomy() -> dict[str, list[str]]:
    return {
        "symptom": ["Crash"],
        "root_cause": ["Incorrect Code Logic", "API Misuse"],
    }


def test_targeted_sla_record_serializes_verified_bounded_audit() -> None:
    controller = _VerifiedSlaController()
    record = _sla_record()
    row = run_targeted_sla_record(
        record,
        domain="ase2022",
        taxonomy=_sla_taxonomy(),
        controller=controller,
        baseline_anchor=_sla_anchor(record["record_id"]),
        record_budget=GlobalSlaBudget(SlaBudgetConfig()).start_record(),
    )

    assert controller.views[0].record_id == record["record_id"]
    assert row["stage3_valid"] is True
    assert row["symptom_prediction"] == "Crash"
    assert row["root_cause_prediction"] == "API Misuse"
    assert row["decision_status"] == {
        "symptom": "model_verified",
        "root_cause": "model_verified",
    }
    assert row["call_count"] == 2
    assert row["fallback"] is False
    assert row["budget_remaining_seconds"] == 123.0
    assert "role_timings" in row["audit"]["targeted_sla"]
    assert row["audit"]["targeted_sla"]["baseline_comparison"] == {
        "symptom_matches_baseline": True,
        "root_cause_matches_baseline": False,
    }
    assert row["audit"]["targeted_sla"]["verifier_evidence"] == {
        "symptom": {
            "verdict": "accept_candidate",
            "supporting_evidence_ids": ["issue-body"],
            "counter_evidence_ids": [],
        },
        "root_cause": {
            "verdict": "accept_candidate",
            "supporting_evidence_ids": ["issue-body"],
            "counter_evidence_ids": [],
        },
    }
    assert "rationale" not in json.dumps(row["audit"]["targeted_sla"])


def test_targeted_sla_record_serializes_dimension_owned_arbitration_evidence() -> None:
    record = _sla_record()

    row = run_targeted_sla_record(
        record,
        domain="ase2022",
        taxonomy=_sla_taxonomy(),
        controller=_ArbitratedSlaController(),
        baseline_anchor=_sla_anchor(record["record_id"]),
        record_budget=GlobalSlaBudget(SlaBudgetConfig()).start_record(),
    )

    assert row["audit"]["targeted_sla"]["arbitration_evidence"] == {
        "root_cause": {
            "selected_label": "API Misuse",
            "supporting_evidence_ids": ["issue-body"],
        }
    }
    assert "rationale" not in json.dumps(row["audit"]["targeted_sla"])


def test_targeted_sla_record_serializes_only_its_new_role_timings() -> None:
    controller = _TelemetrySlaController()
    record = _sla_record()

    row = run_targeted_sla_record(
        record,
        domain="ase2022",
        taxonomy=_sla_taxonomy(),
        controller=controller,
        baseline_anchor=_sla_anchor(record["record_id"]),
        record_budget=GlobalSlaBudget(SlaBudgetConfig()).start_record(),
    )

    assert row["role_timings"] == {
        "sla_joint_diagnosis": 0.25,
        "sla_joint_verifier": 0.5,
    }


def test_targeted_sla_rows_keep_concurrent_call_audits_record_scoped(
    tmp_path: Path,
) -> None:
    controller = _InterleavingTelemetrySlaController()
    records = [_sla_record("sla-first"), _sla_record("sla-second")]
    budgets = {
        record["record_id"]: GlobalSlaBudget(SlaBudgetConfig()).start_record()
        for record in records
    }

    def runner(record: dict[str, str]) -> dict[str, object]:
        return run_targeted_sla_record(
            record,
            domain="ase2022",
            taxonomy=_sla_taxonomy(),
            controller=controller,
            baseline_anchor=_sla_anchor(record["record_id"]),
            record_budget=budgets[record["record_id"]],
        )

    rows = run_records(
        records,
        predictions_path=tmp_path / "concurrent-audits.jsonl",
        config_hash="targeted-sla-config",
        record_runner=runner,
        resume=False,
        concurrency=2,
    )

    for row in rows:
        record_id = row["record_id"]
        expected_role = f"sla_{record_id}"
        assert row["role_timings"] == {expected_role: 0.25}
        assert row["call_audit"] == [
            {
                "role": expected_role,
                "latency_seconds": 0.25,
                "schema_attempts": 2,
                "network_attempts": 3,
                "provider_errors": [
                    {
                        "status": "retryable_error",
                        "retryable": False,
                        "cause_type": "timeout_error",
                        "http_status": None,
                        "retry_after_seconds": None,
                    }
                ],
            }
        ]
        assert row["audit"]["targeted_sla"]["call_audit"] == row["call_audit"]


def test_targeted_sla_budget_rejection_writes_typed_baseline_fallback() -> None:
    class NeverCalledSlaAgents:
        def sla_joint_diagnosis(self, *_args: object) -> object:
            pytest.fail("budget rejection must not invoke a model role")

    record = _sla_record()
    # Construct a valid record budget, then exhaust global admission before execution.
    clock = [0.0]
    global_budget = GlobalSlaBudget(
        SlaBudgetConfig(global_seconds=1.0), clock=lambda: clock[0]
    )
    record_budget = global_budget.start_record()
    clock[0] = 2.0
    row = run_targeted_sla_record(
        record,
        domain="ase2022",
        taxonomy=_sla_taxonomy(),
        controller=TargetedSlaController(NeverCalledSlaAgents()),
        baseline_anchor=_sla_anchor(record["record_id"]),
        record_budget=record_budget,
    )

    assert row["record_id"] == record["record_id"]
    assert row["stage3_valid"] is True
    assert row["symptom_prediction"] == "Crash"
    assert row["root_cause_prediction"] == "Incorrect Code Logic"
    assert row["decision_status"] == {
        "symptom": "budget_fallback",
        "root_cause": "budget_fallback",
    }
    assert row["fallback"] is True


def test_run_records_serializes_sla_budget_rejections_with_failure_factory(
    tmp_path: Path,
) -> None:
    records = _records()[:2]
    failures: list[str] = []

    def rejected_runner(_record: dict[str, str]) -> dict[str, object]:
        raise SlaBudgetExhausted("global SLA budget exhausted")

    def fallback_row(
        record: dict[str, str], error: SlaBudgetExhausted
    ) -> dict[str, object]:
        failures.append(str(error))
        return {
            "record_id": record["record_id"],
            "stage3_valid": True,
            "fallback": True,
            "decision_status": {
                "symptom": "budget_fallback",
                "root_cause": "budget_fallback",
            },
        }

    output = tmp_path / "targeted-sla.jsonl"
    rows = run_records(
        records,
        predictions_path=output,
        config_hash="targeted-sla-config",
        record_runner=rejected_runner,
        resume=False,
        concurrency=2,
        failure_row_factory=fallback_row,
    )

    assert [row["record_id"] for row in rows] == [
        "accepted-1",
        "accepted-2",
    ]
    assert len(failures) == 2
    assert all(row["fallback"] is True for row in rows)
    assert len(output.read_text(encoding="utf-8").splitlines()) == 2


def test_run_records_flushes_four_worker_targeted_sla_cohort(
    tmp_path: Path,
) -> None:
    records = [{"record_id": f"sla-{index}"} for index in range(14)]
    lock = threading.Lock()
    release = threading.Event()
    entered = threading.Event()
    active = 0
    maximum_active = 0
    first_wave = 0

    def runner(record: dict[str, str]) -> dict[str, object]:
        nonlocal active, first_wave, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
            first_wave += 1
            if first_wave == 4:
                entered.set()
        if first_wave <= 4:
            assert release.wait(timeout=2)
        with lock:
            active -= 1
        return {"record_id": record["record_id"], "stage3_valid": True}

    def release_workers() -> None:
        assert entered.wait(timeout=2)
        release.set()

    releaser = threading.Thread(target=release_workers)
    releaser.start()
    output = tmp_path / "targeted-sla-cohort.jsonl"
    snapshots: list[ProgressSnapshot] = []
    simulated_clock = [0.0]

    def advancing_clock() -> float:
        current = simulated_clock[0]
        simulated_clock[0] += 60.0
        return current

    rows = run_records(
        records,
        predictions_path=output,
        config_hash="targeted-sla-config",
        record_runner=runner,
        resume=False,
        concurrency=4,
        clock=advancing_clock,
        progress_callback=snapshots.append,
    )
    releaser.join(timeout=2)

    assert maximum_active == 4
    assert [row["record_id"] for row in rows] == [
        record["record_id"] for record in records
    ]
    assert len(output.read_text(encoding="utf-8").splitlines()) == 14
    assert snapshots[-1].elapsed_seconds == 900.0
    assert snapshots[-1].elapsed_seconds < 1680.0


def _records() -> list[dict[str, str]]:
    return [
        {
            "record_id": "accepted-1",
            "decision": "accepted_fault",
            "symptom": "Crash",
            "root_cause": "Incorrect Code Logic",
        },
        {
            "record_id": "accepted-2",
            "decision": "accepted_fault",
            "symptom": "Poor Performance",
            "root_cause": "API Misuse",
        },
        {
            "record_id": "rejected-1",
            "decision": "rejected_candidate",
            "symptom": "",
            "root_cause": "",
        },
    ]


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([{"record_id": "accepted-1"}, {"record_id": "accepted-1"}], "duplicate"),
        (
            [
                {"record_id": ""},
                {"record_id": "accepted-2"},
                {"record_id": "rejected-1"},
            ],
            "missing record_id",
        ),
        ([{"record_id": "accepted-1"}], "record set mismatch"),
        (
            [
                {"record_id": "accepted-1"},
                {"record_id": "accepted-2"},
                {"record_id": "rejected-1"},
                {"record_id": "extra"},
            ],
            "record set mismatch",
        ),
    ],
)
def test_evaluation_rejects_non_exact_prediction_record_sets(
    rows: list[dict[str, object]], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        evaluate_experiment(_records(), rows)


def test_evaluation_rejects_empty_gold_record_id() -> None:
    records = _records()
    records[0]["record_id"] = ""
    rows = [
        {"record_id": ""},
        {"record_id": "accepted-2"},
        {"record_id": "rejected-1"},
    ]
    with pytest.raises(ValueError, match="missing record_id"):
        evaluate_experiment(records, rows)


def test_metrics_count_invalid_and_stage2_misses_against_full_denominators() -> None:
    rows = [
        {
            "record_id": "accepted-1",
            "stage2_prediction": "accepted_fault",
            "stage2_valid": True,
            "symptom_prediction": "Crash",
            "root_cause_prediction": "Incorrect Code Logic",
            "stage3_valid": True,
        },
        {
            "record_id": "accepted-2",
            "stage2_prediction": "rejected_candidate",
            "stage2_valid": True,
            "symptom_prediction": None,
            "root_cause_prediction": None,
            "stage3_valid": False,
        },
        {
            "record_id": "rejected-1",
            "stage2_prediction": None,
            "stage2_valid": False,
            "symptom_prediction": None,
            "root_cause_prediction": None,
            "stage3_valid": False,
        },
    ]

    metrics = evaluate_experiment(_records(), rows)

    assert metrics["stage2"]["accuracy"] == pytest.approx(1 / 3)
    assert metrics["stage2"]["invalid_count"] == 1
    assert metrics["stage3"]["invalid_count"] == 0
    assert metrics["stage3"]["joint_accuracy"] == pytest.approx(1 / 2)
    assert metrics["stage3"]["missed_by_stage2_count"] == 1
    assert metrics["end_to_end"]["exact_match_accuracy"] == pytest.approx(1 / 3)


def test_metrics_count_invalid_stage3_after_valid_stage2_acceptance() -> None:
    record = {
        "record_id": "accepted",
        "decision": "accepted_fault",
        "symptom": "Crash",
        "root_cause": "Incorrect Code Logic",
    }
    row = {
        "record_id": "accepted",
        "stage2_prediction": "accepted_fault",
        "stage2_valid": True,
        "symptom_prediction": None,
        "root_cause_prediction": None,
        "stage3_valid": False,
    }

    metrics = evaluate_experiment([record], [row])

    assert metrics["stage3"]["invalid_count"] == 1


def test_stage3_metrics_keep_unresolved_rows_in_the_fixed_denominator() -> None:
    records = _records()[:2]
    rows = [
        {
            "record_id": "accepted-1",
            "stage2_prediction": None,
            "stage2_valid": False,
            "symptom_prediction": "Crash",
            "root_cause_prediction": "Incorrect Code Logic",
            "stage3_valid": True,
            "stop_reason": "stage3_only",
            "audit": {
                "stage2": None,
                "stage3": {
                    "final_decision": {"source": "direct_consensus"},
                    "evidence_deltas": [],
                    "budget_exhausted": False,
                    "unresolved": None,
                },
            },
        },
        {
            "record_id": "accepted-2",
            "stage2_prediction": None,
            "stage2_valid": False,
            "symptom_prediction": None,
            "root_cause_prediction": None,
            "stage3_valid": False,
            "stop_reason": "stage3_only",
            "audit": {
                "stage2": None,
                "stage3": {
                    "final_decision": None,
                    "evidence_deltas": [
                        {"status": "found"},
                        {"status": "absent"},
                        {"status": "unavailable"},
                    ],
                    "budget_exhausted": True,
                    "unresolved": {
                        "dimensions": ["symptom", "root_cause"],
                        "stop_reason": "evidence_readiness_budget_exhausted",
                    },
                },
            },
        },
    ]

    metrics = evaluate_experiment(records, rows, stage="stage3")

    assert set(metrics) == {"n", "stage3"}
    assert metrics["stage3"]["n"] == 2
    assert metrics["stage3"]["resolved_count"] == 1
    assert metrics["stage3"]["coverage"] == 0.5
    assert metrics["stage3"]["joint_accuracy"] == 0.5
    assert metrics["stage3"]["selective_joint_accuracy"] == 1.0
    assert metrics["stage3"]["unresolved_count"] == 1
    assert metrics["stage3"]["stop_reason_counts"] == {
        "evidence_readiness_budget_exhausted": 1
    }
    assert metrics["stage3"]["unresolved_dimension_counts"] == {
        "root_cause": 1,
        "symptom": 1,
    }
    assert metrics["stage3"]["retrieval_request_count"] == 3
    assert metrics["stage3"]["retrieval_status_counts"] == {
        "absent": 1,
        "found": 1,
        "unavailable": 1,
    }
    assert metrics["stage3"]["budget_exhausted_count"] == 1
    assert metrics["stage3"]["budget_exhausted_rate"] == 0.5
    assert metrics["stage3"]["resolution_source_counts"] == {
        "direct_consensus": 1,
        "fallback_uncertain": 0,
        "single_team_degraded": 0,
        "targeted_arbitration": 0,
    }
    assert metrics["stage3"]["resolution_source_accuracies"] == {
        "direct_consensus": 1.0,
        "fallback_uncertain": None,
        "single_team_degraded": None,
        "targeted_arbitration": None,
    }


def test_experiment_metrics_attach_preservation_diagnostics_only_when_instrumented() -> (
    None
):
    record = {
        "record_id": "r1",
        "decision": "accepted_fault",
        "symptom": "Crash",
        "root_cause": "Incorrect Code Logic",
    }
    row = {
        "record_id": "r1",
        "stage3_valid": True,
        "symptom_prediction": "Crash",
        "root_cause_prediction": "Incorrect Code Logic",
        "audit": {
            "stage3": {
                "baseline_anchor": {
                    "valid": True,
                    "symptom_label": "Crash",
                    "root_cause_label": "Incorrect Code Logic",
                },
                "pre_gate_candidate": {
                    "symptom_label": "Incorrect Functionality",
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
                "validity_report": {"quarantined": []},
                "unresolved": None,
            }
        },
    }

    instrumented = evaluate_experiment([record], [row], stage="stage3")
    legacy = evaluate_experiment(
        [record],
        [{**row, "audit": {"stage3": {}}}],
        stage="stage3",
    )

    assert instrumented["stage3"]["baseline_preservation"]["paired"] == {
        "baseline_harm_prevented_count": 1,
        "baseline_correct_to_final_wrong_count": 0,
        "baseline_wrong_to_final_correct_count": 0,
        "net_corrections": 0,
    }
    assert "baseline_preservation" not in legacy["stage3"]


def test_stage3_metrics_count_uncertain_fallback_as_a_prediction() -> None:
    record = _records()[0]
    row = {
        "record_id": record["record_id"],
        "stage2_prediction": None,
        "stage2_valid": False,
        "symptom_prediction": "Crash",
        "root_cause_prediction": "Incorrect Code Logic",
        "stage3_valid": True,
        "stop_reason": "stage3_only",
        "audit": {
            "stage2": None,
            "stage3": {
                "final_decision": {"source": "fallback_uncertain"},
                "evidence_deltas": [],
                "budget_exhausted": False,
                "unresolved": None,
            },
        },
    }

    metrics = evaluate_experiment([record], [row], stage="stage3")

    assert metrics["stage3"]["resolution_source_counts"] == {
        "direct_consensus": 0,
        "fallback_uncertain": 1,
        "single_team_degraded": 0,
        "targeted_arbitration": 0,
    }
    assert metrics["stage3"]["resolution_source_accuracies"] == {
        "direct_consensus": None,
        "fallback_uncertain": 1.0,
        "single_team_degraded": None,
        "targeted_arbitration": None,
    }


def test_stage3_metrics_count_single_team_degraded_as_a_prediction() -> None:
    record = _records()[0]
    row = {
        "record_id": record["record_id"],
        "stage2_prediction": None,
        "stage2_valid": False,
        "symptom_prediction": "Crash",
        "root_cause_prediction": "Incorrect Code Logic",
        "stage3_valid": True,
        "stop_reason": "stage3_only",
        "audit": {
            "stage2": None,
            "stage3": {
                "final_decision": {"source": "single_team_degraded"},
                "evidence_deltas": [],
                "budget_exhausted": False,
                "unresolved": None,
            },
        },
    }

    metrics = evaluate_experiment([record], [row], stage="stage3")

    assert metrics["stage3"]["resolution_source_counts"] == {
        "direct_consensus": 0,
        "fallback_uncertain": 0,
        "single_team_degraded": 1,
        "targeted_arbitration": 0,
    }
    assert metrics["stage3"]["resolution_source_accuracies"] == {
        "direct_consensus": None,
        "fallback_uncertain": None,
        "single_team_degraded": 1.0,
        "targeted_arbitration": None,
    }


def test_selective_metrics_are_undefined_when_no_stage3_row_resolves() -> None:
    record = _records()[0]
    row = {
        "record_id": record["record_id"],
        "stage2_prediction": None,
        "stage2_valid": False,
        "symptom_prediction": None,
        "root_cause_prediction": None,
        "stage3_valid": False,
        "stop_reason": "structured_output_invalid",
        "audit": None,
    }

    metrics = evaluate_experiment([record], [row], stage="stage3")

    assert metrics["stage3"]["coverage"] == 0.0
    assert metrics["stage3"]["selective_symptom_accuracy"] is None
    assert metrics["stage3"]["selective_root_cause_accuracy"] is None
    assert metrics["stage3"]["selective_joint_accuracy"] is None
    assert metrics["stage3"]["joint_accuracy"] == 0.0
    assert metrics["stage3"]["stop_reason_counts"] == {"structured_output_invalid": 1}


def test_metrics_name_verification_failures_from_structured_audit_state() -> None:
    record = _records()[0]
    row = {
        "record_id": record["record_id"],
        "stage2_prediction": None,
        "stage2_valid": False,
        "symptom_prediction": None,
        "root_cause_prediction": None,
        "stage3_valid": False,
        "stop_reason": "stage3_only",
        "audit": {
            "stage2": None,
            "stage3": {
                "final_decision": {"source": "direct_consensus"},
                "verification": {"valid": False},
                "evidence_deltas": [],
                "budget_exhausted": False,
                "unresolved": None,
            },
        },
    }

    metrics = evaluate_experiment([record], [row], stage="stage3")

    assert metrics["stage3"]["stop_reason_counts"] == {"stage3_verification_failed": 1}


def test_stage2_metrics_report_selective_and_resolution_source_accuracy() -> None:
    records = _records()[:2]
    rows = [
        {
            "record_id": "accepted-1",
            "stage2_prediction": "accepted_fault",
            "stage2_valid": True,
            "audit": {
                "stage2": {
                    "final_decision": {"source": "targeted_arbitration"},
                    "evidence_deltas": [{"status": "found"}],
                    "budget_exhausted": False,
                    "unresolved": None,
                }
            },
        },
        {
            "record_id": "accepted-2",
            "stage2_prediction": None,
            "stage2_valid": False,
            "stop_reason": "stage2_unresolved",
            "audit": None,
        },
    ]

    metrics = evaluate_experiment(records, rows, stage="stage2")

    assert set(metrics) == {"n", "stage2"}
    assert metrics["stage2"]["n"] == 2
    assert metrics["stage2"]["resolved_count"] == 1
    assert metrics["stage2"]["coverage"] == 0.5
    assert metrics["stage2"]["accuracy"] == 0.5
    assert metrics["stage2"]["selective_accuracy"] == 1.0
    assert metrics["stage2"]["resolution_source_counts"] == {
        "direct_consensus": 0,
        "policy_composition": 0,
        "targeted_arbitration": 1,
    }
    assert metrics["stage2"]["resolution_source_accuracies"] == {
        "direct_consensus": None,
        "policy_composition": None,
        "targeted_arbitration": 1.0,
    }


def test_stage2_metrics_preserve_policy_composition_as_its_own_source() -> None:
    record = _records()[0]
    row = {
        "record_id": record["record_id"],
        "stage2_prediction": "accepted_fault",
        "stage2_valid": True,
        "audit": {
            "stage2": {
                "final_decision": {"source": "policy_composition"},
                "evidence_deltas": [],
                "budget_exhausted": False,
                "unresolved": None,
            }
        },
    }

    metrics = evaluate_experiment([record], [row], stage="stage2")

    assert metrics["stage2"]["resolution_source_counts"] == {
        "direct_consensus": 0,
        "policy_composition": 1,
        "targeted_arbitration": 0,
    }
    assert metrics["stage2"]["resolution_source_accuracies"] == {
        "direct_consensus": None,
        "policy_composition": 1.0,
        "targeted_arbitration": None,
    }


def test_resume_reuses_matching_rows_and_runs_only_missing_records(
    tmp_path: Path,
) -> None:
    output = tmp_path / "predictions.jsonl"
    output.write_text(
        json.dumps(
            {
                "record_id": "accepted-1",
                "config_hash": "same-config",
                "stage2_prediction": "accepted_fault",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    calls: list[str] = []

    def runner(record: dict[str, str]) -> dict[str, object]:
        calls.append(record["record_id"])
        return {
            "record_id": record["record_id"],
            "config_hash": "same-config",
            "stage2_prediction": "rejected_candidate",
        }

    rows = run_records(
        _records()[:2],
        predictions_path=output,
        config_hash="same-config",
        record_runner=runner,
        resume=True,
    )

    assert calls == ["accepted-2"]
    assert [row["record_id"] for row in rows] == [
        "accepted-1",
        "accepted-2",
    ]
    assert len(output.read_text(encoding="utf-8").splitlines()) == 2


def test_resume_reorders_partial_predictions_into_selected_record_order(
    tmp_path: Path,
) -> None:
    output = tmp_path / "predictions.jsonl"
    output.write_text(
        json.dumps(
            {
                "record_id": "accepted-2",
                "config_hash": "same-config",
                "stage2_prediction": "accepted_fault",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows = run_records(
        _records()[:2],
        predictions_path=output,
        config_hash="same-config",
        record_runner=lambda record: {
            "record_id": record["record_id"],
            "stage2_prediction": "accepted_fault",
        },
        resume=True,
    )

    expected_ids = ["accepted-1", "accepted-2"]
    persisted_ids = [
        json.loads(line)["record_id"]
        for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["record_id"] for row in rows] == expected_ids
    assert persisted_ids == expected_ids


def test_ordered_finalize_uses_a_short_temporary_name_near_windows_max_path(
    tmp_path: Path,
) -> None:
    parent = tmp_path
    while len(str(parent)) < 215:
        remaining = 215 - len(str(parent)) - 1
        component_length = min(40, remaining)
        if component_length < 1:
            break
        parent /= "d" * component_length
        parent.mkdir()
    assert len(str(parent)) == 215
    output = parent / ("p" * 25 + ".jsonl")

    rows = run_records(
        _records()[:2],
        predictions_path=output,
        config_hash="same-config",
        record_runner=lambda record: {
            "record_id": record["record_id"],
            "stage2_prediction": "accepted_fault",
        },
        resume=False,
        concurrency=2,
    )

    persisted_ids = [
        json.loads(line)["record_id"]
        for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert persisted_ids == [row["record_id"] for row in rows]
    assert persisted_ids == ["accepted-1", "accepted-2"]


def test_ordered_finalize_preserves_partial_file_when_atomic_replace_is_denied(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "predictions.jsonl"
    partial = '{"record_id": "completed-before-finalize"}\n'
    output.write_text(partial, encoding="utf-8")

    def deny_replace(source: Path, target: Path) -> None:
        raise PermissionError(f"replace denied: {source} -> {target}")

    monkeypatch.setattr(Path, "replace", deny_replace)

    with pytest.raises(PermissionError, match="replace denied"):
        _write_ordered_jsonl(
            output,
            [{"record_id": "ordered", "config_hash": "same-config"}],
        )

    assert output.read_text(encoding="utf-8") == partial
    assert list(tmp_path.glob(".aew-*.tmp")) == []


def test_resume_rejects_rows_from_another_configuration(tmp_path: Path) -> None:
    output = tmp_path / "predictions.jsonl"
    output.write_text(
        json.dumps(
            {
                "record_id": "accepted-1",
                "config_hash": "old-config",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="config_hash"):
        run_records(
            _records()[:1],
            predictions_path=output,
            config_hash="new-config",
            record_runner=lambda record: {},
            resume=True,
        )


def test_resume_rejects_prediction_ids_outside_selected_records(
    tmp_path: Path,
) -> None:
    output = tmp_path / "predictions.jsonl"
    output.write_text(
        json.dumps({"record_id": "not-selected", "config_hash": "same-config"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown record_id in existing predictions"):
        run_records(
            _records()[:1],
            predictions_path=output,
            config_hash="same-config",
            record_runner=lambda record: {"record_id": record["record_id"]},
            resume=True,
        )


@pytest.mark.parametrize(
    ("existing_id", "selected_id"),
    [
        (True, "True"),
        (1, "1"),
        (1.5, "1.5"),
        (["1"], "['1']"),
        (None, "selected"),
        ("", "selected"),
    ],
)
def test_resume_rejects_existing_record_ids_that_are_not_native_nonempty_strings(
    tmp_path: Path,
    existing_id: object,
    selected_id: str,
) -> None:
    output = tmp_path / "predictions.jsonl"
    original = (
        json.dumps({"record_id": existing_id, "config_hash": "same-config"}) + "\n"
    )
    output.write_text(original, encoding="utf-8")

    with pytest.raises(ValueError, match="record_id must be a non-empty string"):
        run_records(
            [{"record_id": selected_id}],
            predictions_path=output,
            config_hash="same-config",
            record_runner=lambda _record: pytest.fail(
                "invalid existing row must fail before record execution"
            ),
            resume=True,
        )

    assert output.read_text(encoding="utf-8") == original


@pytest.mark.parametrize("record_id", [True, 1, 1.5, ["1"], None, ""])
def test_run_records_rejects_input_record_ids_that_are_not_native_nonempty_strings(
    tmp_path: Path,
    record_id: object,
) -> None:
    with pytest.raises(ValueError, match="record_id"):
        run_records(
            [{"record_id": record_id}],  # type: ignore[list-item]
            predictions_path=tmp_path / "predictions.jsonl",
            config_hash="same-config",
            record_runner=lambda _record: pytest.fail(
                "invalid input ID must fail before record execution"
            ),
            resume=False,
        )


@pytest.mark.parametrize("record_id", [True, 1, 1.5, ["1"], None, ""])
def test_run_records_rejects_result_ids_that_are_not_native_nonempty_strings(
    tmp_path: Path,
    record_id: object,
) -> None:
    with pytest.raises(ValueError, match="record_id"):
        run_records(
            [{"record_id": "selected"}],
            predictions_path=tmp_path / "predictions.jsonl",
            config_hash="same-config",
            record_runner=lambda _record: {"record_id": record_id},
            resume=False,
        )


@pytest.mark.parametrize(
    ("result", "record_count", "message"),
    [
        ({}, 1, "missing record_id"),
        ({"record_id": "not-selected"}, 1, "unknown record_id in record result"),
        ({"record_id": "accepted-2"}, 2, "duplicate or mismatched record_id"),
    ],
)
def test_run_records_rejects_non_exact_result_record_ids(
    tmp_path: Path,
    result: dict[str, object],
    record_count: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        run_records(
            _records()[:record_count],
            predictions_path=tmp_path / "predictions.jsonl",
            config_hash="same-config",
            record_runner=lambda _record: result,
            resume=False,
        )


def test_run_records_rejects_duplicate_result_record_ids(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicate or mismatched record_id"):
        run_records(
            _records()[:2],
            predictions_path=tmp_path / "predictions.jsonl",
            config_hash="same-config",
            record_runner=lambda _record: {"record_id": "accepted-1"},
            resume=False,
            concurrency=1,
        )


def test_record_selection_preserves_cohort_order_and_validates_ids() -> None:
    selected = select_records(
        _records(),
        record_ids=["rejected-1", "accepted-1"],
        limit=1,
    )

    assert [row["record_id"] for row in selected] == ["accepted-1"]
    with pytest.raises(ValueError, match="unknown record_ids"):
        select_records(_records(), record_ids=["missing"], limit=None)


class _DeterministicAgents:
    def __init__(self) -> None:
        self.readiness_tasks: list[str] = []
        self.stage3_arbitrations: list[tuple[str, ...]] = []

    def evidence_readiness(self, task: str, view: object) -> EvidenceReadinessReport:
        self.readiness_tasks.append(task)
        dimensions = (
            (EvidenceDimension.FAULT_EXISTENCE, EvidenceDimension.STUDY_SCOPE)
            if task == "stage2"
            else (EvidenceDimension.SYMPTOM, EvidenceDimension.ROOT_CAUSE)
        )
        if task == "stage3":
            mechanism = next(
                (item for item in view.items if item.source_type == "code_diff"),
                None,
            )
            if mechanism is None:
                return EvidenceReadinessReport(
                    task=task,
                    dimensions=(
                        DimensionReadiness(
                            dimension=EvidenceDimension.SYMPTOM,
                            sufficient=True,
                            confirmed_evidence_ids=(view.items[0].evidence_id,),
                            missing_facts=(),
                            evidence_requests=(),
                        ),
                        DimensionReadiness(
                            dimension=EvidenceDimension.ROOT_CAUSE,
                            sufficient=False,
                            confirmed_evidence_ids=(),
                            missing_facts=(
                                "The pre-fix defect mechanism requires code evidence.",
                            ),
                            evidence_requests=(
                                EvidenceRequest(
                                    request_id="stage3-code-context",
                                    missing_fact="The pre-fix defect mechanism in code.",
                                    why_needed=(
                                        "Root-cause readiness requires mechanism-capable "
                                        "evidence."
                                    ),
                                    target_specialist=SpecialistType.CODE_CONTEXT,
                                    target_source="code diff",
                                    query="reject request process",
                                    expected_decision_impact=(
                                        "Code evidence can establish root-cause readiness."
                                    ),
                                    max_items=2,
                                ),
                            ),
                        ),
                    ),
                )
            return EvidenceReadinessReport(
                task=task,
                dimensions=(
                    DimensionReadiness(
                        dimension=EvidenceDimension.SYMPTOM,
                        sufficient=True,
                        confirmed_evidence_ids=(view.items[0].evidence_id,),
                        missing_facts=(),
                        evidence_requests=(),
                    ),
                    DimensionReadiness(
                        dimension=EvidenceDimension.ROOT_CAUSE,
                        sufficient=True,
                        confirmed_evidence_ids=(mechanism.evidence_id,),
                        missing_facts=(),
                        evidence_requests=(),
                    ),
                ),
            )
        return EvidenceReadinessReport(
            task=task,
            dimensions=tuple(
                DimensionReadiness(
                    dimension=dimension,
                    sufficient=True,
                    confirmed_evidence_ids=(view.items[0].evidence_id,),
                    missing_facts=(),
                    evidence_requests=(),
                )
                for dimension in dimensions
            ),
        )

    @staticmethod
    def stage2_analyst(team_id: str, view: object) -> Stage2AnalysisReport:
        evidence_id = view.items[0].evidence_id
        return Stage2AnalysisReport(
            team_id=team_id,
            decision=Stage2Decision.ACCEPTED,
            confidence=0.9,
            fault_claim="The pre-fix implementation rejects a valid request.",
            repair_claim="The supplied change restores processing of that request.",
            supporting_evidence_ids=[evidence_id],
            evidence_tests={
                "fault_existence": EvidenceTestOutcome.PASS,
                "repair_causality": EvidenceTestOutcome.PASS,
                "scope_exclusion": EvidenceTestOutcome.PASS,
            },
            alternative_hypothesis="The change could instead add a new feature.",
            decision_boundary="Incorrect prior behavior distinguishes repair from feature work.",
            evidence_sufficiency=EvidenceSufficiency.SUFFICIENT,
        )

    @staticmethod
    def symptom_analyst(team_id: str, view: object) -> SymptomReport:
        return SymptomReport(
            label="Crash",
            behavior_claim="The valid request terminates processing unexpectedly.",
            supporting_evidence_ids=[view.items[0].evidence_id],
            alternative_label="Poor Performance",
            boundary_reason="Execution terminates rather than merely slowing down.",
            boundary_evidence_ids=[view.items[0].evidence_id],
            confidence=0.9,
            evidence_sufficiency=EvidenceSufficiency.SUFFICIENT,
        )

    @staticmethod
    def root_cause_analyst(team_id: str, view: object) -> RootCauseReport:
        return RootCauseReport(
            label="Incorrect Code Logic",
            defect_mechanism="The handler follows an incorrect rejection branch.",
            causal_chain=[
                "The request enters the handler.",
                "An incorrect condition selects rejection.",
                "Processing terminates unexpectedly.",
            ],
            supporting_evidence_ids=[view.items[0].evidence_id],
            alternative_label="API Misuse",
            boundary_reason="The defect is internal control flow rather than API usage.",
            boundary_evidence_ids=[view.items[0].evidence_id],
            confidence=0.9,
            evidence_sufficiency=EvidenceSufficiency.SUFFICIENT,
        )

    @staticmethod
    def consistency_checker(
        team_id: str,
        symptom: SymptomReport,
        root: RootCauseReport,
        view: object,
    ) -> CausalConsistencyReport:
        return CausalConsistencyReport(
            status=ConsistencyStatus.CONSISTENT,
            rationale="The incorrect rejection branch directly explains termination.",
            supporting_evidence_ids=[view.items[0].evidence_id],
        )

    def stage3_arbitrator(
        self, packet: Stage3ArbitrationPacket
    ) -> Stage3ArbitrationDecision:
        self.stage3_arbitrations.append(tuple(packet.disagreement.dimensions))
        return Stage3ArbitrationDecision(
            resolution_status=ResolutionStatus.RESOLVED,
            symptom_label="Crash",
            root_cause_label="Incorrect Code Logic",
            confidence=0.85,
            rationale=(
                "The bounded record evidence supports the listed shared-risk review."
            ),
            supporting_evidence_ids=[packet.relevant_evidence[0].evidence_id],
            resolved_dimensions=list(packet.disagreement.dimensions),
        )


class _NeverCalledAgents:
    @staticmethod
    def _fail(*args: object, **kwargs: object) -> object:
        raise AssertionError("scope-excluded records must not call an agent")

    stage2_analyst = _fail
    evidence_readiness = _fail
    stage2_arbitrator = _fail
    symptom_analyst = _fail
    root_cause_analyst = _fail
    consistency_checker = _fail
    stage3_arbitrator = _fail


class _RoleOwnedStage2Agents(_DeterministicAgents):
    @staticmethod
    def stage2_analyst(*args: object) -> object:
        raise AssertionError("normal role-owned wiring must not call stage2_analyst")

    @staticmethod
    def fault_evidence_analyst(view: object) -> FaultEvidenceAssessment:
        return FaultEvidenceAssessment(
            outcome=EvidenceTestOutcome.PASS,
            claim="The issue reports a concrete failure before any repair.",
            supporting_evidence_ids=[view.items[0].evidence_id],
            counter_evidence_ids=[],
        )

    @staticmethod
    def scope_boundary_analyst(view: object) -> ScopeBoundaryAssessment:
        return ScopeBoundaryAssessment(
            outcome=EvidenceTestOutcome.PASS,
            claim="The issue report is included by the ASE2022 study policy.",
            supporting_evidence_ids=[view.items[0].evidence_id],
            counter_evidence_ids=[],
        )

    @staticmethod
    def repair_causality_analyst(view: object) -> RepairCausalityAssessment:
        raise AssertionError("ASE2022 minimal wiring must not call repair analysis")


@pytest.mark.parametrize(
    "surface",
    [
        ("stage2_analyst", "fault_evidence_analyst"),
        ("stage2_analyst", "scope_boundary_analyst"),
        ("stage2_analyst", "repair_causality_analyst"),
        ("fault_evidence_analyst",),
        ("scope_boundary_analyst",),
        ("repair_causality_analyst",),
        ("fault_evidence_analyst", "repair_causality_analyst"),
        ("scope_boundary_analyst", "repair_causality_analyst"),
    ],
)
def test_experiment_rejects_partial_role_adapter_instead_of_falling_back_to_legacy(
    surface: tuple[str, ...],
) -> None:
    def unused(*args: object) -> object:
        raise AssertionError("partial adapters must fail before agent invocation")

    attributes = {name: unused for name in surface}
    attributes["evidence_readiness"] = unused
    agents = SimpleNamespace(**attributes)

    with pytest.raises(ValueError, match="partial role-owned Stage 2 adapter surface"):
        run_adaptive_record(
            {
                "record_id": "partial-adapter",
                "title": "A concrete runtime failure",
                "body": "The request fails before the patch.",
            },
            domain="ase2022",
            taxonomy={"decision": ["accepted_fault", "rejected_candidate"]},
            agents=agents,
            stage="stage2",
            retrieved_at="2026-08-05T13:00:00Z",
        )


def test_experiment_rejects_noncallable_new_role_attribute_as_partial_surface() -> None:
    agents = SimpleNamespace(
        stage2_analyst=lambda team_id, view: _DeterministicAgents.stage2_analyst(
            team_id, view
        ),
        fault_evidence_analyst=None,
        evidence_readiness=_DeterministicAgents().evidence_readiness,
    )

    with pytest.raises(ValueError, match="partial role-owned Stage 2 adapter surface"):
        run_adaptive_record(
            {
                "record_id": "noncallable-partial-adapter",
                "title": "A concrete runtime failure",
                "body": "The request fails before the patch.",
            },
            domain="ase2022",
            taxonomy={"decision": ["accepted_fault", "rejected_candidate"]},
            agents=agents,
            stage="stage2",
            retrieved_at="2026-08-05T13:00:00Z",
        )


def test_experiment_requires_repair_role_for_issta_role_owned_adapter() -> None:
    delegate = _RoleOwnedStage2Agents()
    agents = SimpleNamespace(
        stage2_analyst=delegate.stage2_analyst,
        fault_evidence_analyst=delegate.fault_evidence_analyst,
        scope_boundary_analyst=delegate.scope_boundary_analyst,
        evidence_readiness=delegate.evidence_readiness,
    )

    with pytest.raises(ValueError, match="partial role-owned Stage 2 adapter surface"):
        run_adaptive_record(
            {
                "record_id": "issta-partial-adapter",
                "commit_message": "Fix request rejection",
                "code_diff": "- reject(request)\n+ process(request)",
            },
            domain="issta2024",
            taxonomy={"decision": ["accepted_fault", "rejected_candidate"]},
            agents=agents,
            stage="stage2",
            retrieved_at="2026-08-05T13:00:00Z",
        )


def test_ase_pull_request_scope_gate_stops_full_workflow_without_agent_calls() -> None:
    record = {
        "record_id": "pull-request-1",
        "issue_url": "https://github.com/tensorflow/tfjs/pull/123",
        "title": "Internal maintenance",
        "body": "Update the release workflow.",
    }
    taxonomy = {
        "decision": ["accepted_fault", "rejected_candidate"],
        "symptom": ["Crash"],
        "root_cause": ["Incorrect Code Logic"],
    }

    row = run_adaptive_record(
        record,
        domain="ase2022",
        taxonomy=taxonomy,
        agents=_NeverCalledAgents(),
        stage="all",
        retrieved_at="2026-08-05T13:00:00Z",
    )

    assert row["stage2_prediction"] == "rejected_candidate"
    assert row["stage2_valid"] is True
    assert row["stage3_valid"] is False
    assert row["stop_reason"] == "stage2_rejected_candidate"
    assert row["audit"]["stage3"] is None
    assert row["audit"]["stage2"]["reports"] == []
    assert row["audit"]["stage2"]["final_decision"]["source"] == (
        "deterministic_scope_gate"
    )
    assert row["audit"]["stage2"]["scope_exclusion"] == {
        "rule_id": "ase2022_issue_artifact_only",
        "boundary": ("ASE2022 studies issue reports; pull requests are outside scope."),
        "evidence_id": row["audit"]["evidence_items"][0]["evidence_id"],
        "source_uri": record["issue_url"],
    }


def test_normal_experiment_wiring_uses_role_owned_stage2_without_legacy_calls() -> None:
    record = {
        "record_id": "record-1",
        "title": "Valid request terminates",
        "body": "The request is rejected before the patch.",
    }
    row = run_adaptive_record(
        record,
        domain="ase2022",
        taxonomy={"decision": ["accepted_fault", "rejected_candidate"]},
        agents=_RoleOwnedStage2Agents(),
        stage="stage2",
        retrieved_at="2026-08-05T13:00:00Z",
    )

    assert row["stage2_prediction"] == "accepted_fault"
    assert row["stage2_valid"] is True
    assert len(row["audit"]["stage2"]["reports"]) == 1
    assert row["audit"]["stage2"]["reports"][0]["team_id"] == ("policy_composition")
    assert row["audit"]["stage2"]["final_decision"]["source"] == ("policy_composition")
    assert row["audit"]["stage2"]["role_assessments"]["repair_causality"] is None


def test_adaptive_record_runner_wires_both_stages_without_gold_leakage() -> None:
    record = {
        "record_id": "record-1",
        "title": "Valid request terminates",
        "body": "The request is rejected before the patch.",
        "code_diff": "- reject(request)\n+ process(request)",
        "decision": "SECRET_GOLD_DECISION",
        "symptom": "SECRET_GOLD_SYMPTOM",
        "root_cause": "SECRET_GOLD_CAUSE",
    }
    taxonomy = {
        "decision": ["accepted_fault", "rejected_candidate"],
        "symptom": ["Crash", "Poor Performance"],
        "root_cause": ["Incorrect Code Logic", "API Misuse"],
    }

    agents = _DeterministicAgents()
    row = run_adaptive_record(
        record,
        domain="ase2022",
        taxonomy=taxonomy,
        agents=agents,
        stage="all",
        retrieved_at="2026-08-05T13:00:00Z",
    )

    assert row["stage2_prediction"] == "accepted_fault"
    assert row["stage2_valid"] is True
    assert row["symptom_prediction"] == "Crash"
    assert row["root_cause_prediction"] == "Incorrect Code Logic"
    assert row["stage3_valid"] is True
    assert agents.stage3_arbitrations == [
        (
            "unsupported_root_cause_boundary",
            "unsupported_root_cause_specificity",
        )
    ]
    assert agents.readiness_tasks == ["stage2", "stage3", "stage3"]
    assert row["audit"]["stage2"]["classification_ledger_version"] == 1
    assert row["audit"]["stage3"]["classification_ledger_version"] == 2
    assert row["audit"]["stage2"]["readiness_report"]["task"] == "stage2"
    assert row["audit"]["stage3"]["readiness_report"]["task"] == "stage3"
    assert "SECRET_GOLD" not in json.dumps(row["audit"])


def test_stage3_record_runner_never_requires_or_calls_stage2_agents() -> None:
    record = {
        "record_id": "positive-1",
        "title": "Valid request terminates",
        "body": "The request is rejected before the patch.",
        "code_diff": "- reject(request)\n+ process(request)",
        "decision": "SECRET_GOLD_DECISION",
        "symptom": "SECRET_GOLD_SYMPTOM",
        "root_cause": "SECRET_GOLD_CAUSE",
    }
    taxonomy = {
        "decision": ["accepted_fault", "rejected_candidate"],
        "symptom": ["Crash", "Poor Performance"],
        "root_cause": ["Incorrect Code Logic", "API Misuse"],
    }
    delegate = _DeterministicAgents()
    observed_payloads: list[object] = []

    def capture(callable_: object) -> object:
        def wrapped(*args: object) -> object:
            observed_payloads.extend(args)
            return callable_(*args)

        return wrapped

    agents = SimpleNamespace(
        evidence_readiness=capture(delegate.evidence_readiness),
        symptom_analyst=capture(delegate.symptom_analyst),
        root_cause_analyst=capture(delegate.root_cause_analyst),
        consistency_checker=capture(delegate.consistency_checker),
        stage3_arbitrator=capture(delegate.stage3_arbitrator),
    )

    row = run_adaptive_record(
        record,
        domain="ase2022",
        taxonomy=taxonomy,
        agents=agents,
        stage="stage3",
        retrieved_at="2026-08-05T13:00:00Z",
    )

    assert row["stage2_prediction"] is None
    assert row["stage2_valid"] is False
    assert row["stage3_valid"] is True
    assert row["audit"]["stage2"] is None
    serialized_requests = json.dumps(
        [
            (
                payload.model_dump(mode="json")
                if hasattr(payload, "model_dump")
                else payload
            )
            for payload in observed_payloads
        ],
        default=str,
    )
    assert "SECRET_GOLD" not in serialized_requests
    assert "SECRET_GOLD" not in json.dumps(row["audit"])


def test_stage3_record_runner_exposes_the_internal_failure_reason() -> None:
    record = {
        "record_id": "positive-1",
        "title": "Valid request terminates",
        "body": "The request is rejected before the patch.",
        "code_diff": "- reject(request)\n+ process(request)",
    }
    taxonomy = {
        "decision": ["accepted_fault", "rejected_candidate"],
        "symptom": ["Crash", "Poor Performance"],
        "root_cause": ["Incorrect Code Logic", "API Misuse"],
    }
    delegate = _DeterministicAgents()

    def invalid_symptom(team_id: str, view: object) -> SymptomReport:
        return delegate.symptom_analyst(team_id, view).model_copy(
            update={"supporting_evidence_ids": ["quarantined-or-unknown"]}
        )

    agents = SimpleNamespace(
        evidence_readiness=delegate.evidence_readiness,
        symptom_analyst=invalid_symptom,
        root_cause_analyst=delegate.root_cause_analyst,
        consistency_checker=delegate.consistency_checker,
        stage3_arbitrator=delegate.stage3_arbitrator,
    )

    row = run_adaptive_record(
        record,
        domain="ase2022",
        taxonomy=taxonomy,
        agents=agents,
        stage="stage3",
        retrieved_at="2026-08-05T13:00:00Z",
    )

    internal_reason = row["audit"]["stage3"]["unresolved"]["stop_reason"]
    assert internal_reason == "arbitration_evidence_insufficient"
    assert row["stop_reason"] == internal_reason
    assert row["stop_reason"] != "stage3_only"


def test_stage3_no_consistent_candidate_is_visible_in_row_and_metrics() -> None:
    record = {
        "record_id": "positive-1",
        "title": "Valid request terminates",
        "body": "The request is rejected before the patch.",
        "code_diff": "- reject(request)\n+ process(request)",
        "decision": "accepted_fault",
        "symptom": "Crash",
        "root_cause": "Incorrect Code Logic",
    }
    taxonomy = {
        "decision": ["accepted_fault", "rejected_candidate"],
        "symptom": ["Crash", "Poor Performance"],
        "root_cause": ["Incorrect Code Logic", "API Misuse"],
    }
    delegate = _DeterministicAgents()

    def failed_consistency(
        team_id: str,
        symptom: SymptomReport,
        root_cause: RootCauseReport,
        view: object,
    ) -> CausalConsistencyReport:
        return CausalConsistencyReport(
            status=ConsistencyStatus.CAUSE_REVIEW,
            rationale="The proposed mechanism does not explain the observed crash.",
            supporting_evidence_ids=[view.items[0].evidence_id],
        )

    agents = SimpleNamespace(
        evidence_readiness=delegate.evidence_readiness,
        symptom_analyst=delegate.symptom_analyst,
        root_cause_analyst=delegate.root_cause_analyst,
        consistency_checker=failed_consistency,
    )

    row = run_adaptive_record(
        record,
        domain="ase2022",
        taxonomy=taxonomy,
        agents=agents,
        stage="stage3",
        retrieved_at="2026-08-05T13:00:00Z",
    )
    metrics = evaluate_experiment([record], [row], stage="stage3")

    assert row["stage3_valid"] is False
    assert row["stop_reason"] == "no_consistent_stage3_candidate"
    assert "failed_causal_consistency" in (
        row["audit"]["stage3"]["unresolved"]["dimensions"]
    )
    assert metrics["stage3"]["stop_reason_counts"] == {
        "no_consistent_stage3_candidate": 1
    }
    assert (
        metrics["stage3"]["unresolved_dimension_counts"]["failed_causal_consistency"]
        == 1
    )


def test_experiment_row_preserves_verified_single_team_degraded_labels() -> None:
    record = {
        "record_id": "degraded-1",
        "title": "Valid request terminates",
        "body": "The request is rejected before the patch.",
        "code_diff": "- reject(request)\n+ process(request)",
    }
    taxonomy = {
        "decision": ["accepted_fault", "rejected_candidate"],
        "symptom": ["Crash", "Poor Performance"],
        "root_cause": ["Incorrect Code Logic", "API Misuse"],
    }
    delegate = _DeterministicAgents()

    def failed_root(team_id: str, view: object) -> RootCauseReport:
        if team_id == "B":
            raise ValueError("schema failure")
        return delegate.root_cause_analyst(team_id, view)

    row = run_adaptive_record(
        record,
        domain="ase2022",
        taxonomy=taxonomy,
        agents=SimpleNamespace(
            evidence_readiness=delegate.evidence_readiness,
            symptom_analyst=delegate.symptom_analyst,
            root_cause_analyst=failed_root,
            consistency_checker=delegate.consistency_checker,
            stage3_arbitrator=delegate.stage3_arbitrator,
        ),
        stage="stage3",
        retrieved_at="2026-08-05T13:00:00Z",
    )

    assert row["stage3_valid"] is True
    assert row["symptom_prediction"] == "Crash"
    assert row["root_cause_prediction"] == "Incorrect Code Logic"
    assert row["audit"]["stage3"]["final_decision"]["source"] == (
        "single_team_degraded"
    )


def test_experiment_api_rejects_unknown_stage_modes() -> None:
    with pytest.raises(ValueError, match="stage must be"):
        evaluate_experiment([], [], stage="unknown")
    with pytest.raises(ValueError, match="stage must be"):
        run_adaptive_record(
            {"record_id": "record-1"},
            domain="ase2022",
            taxonomy={
                "decision": ["accepted_fault", "rejected_candidate"],
                "symptom": ["Crash"],
                "root_cause": ["Incorrect Code Logic"],
            },
            agents=SimpleNamespace(),
            stage="unknown",
        )


def test_record_workers_run_concurrently_but_results_keep_cohort_order(
    tmp_path: Path,
) -> None:
    release_first = threading.Event()
    snapshots: list[ProgressSnapshot] = []

    def runner(record: dict[str, str]) -> dict[str, object]:
        if record["record_id"] == "accepted-1":
            if not release_first.wait(timeout=2):
                raise AssertionError("second record did not complete first")
        return {
            "record_id": record["record_id"],
            "stage2_prediction": record["decision"],
        }

    def capture(snapshot: ProgressSnapshot) -> None:
        snapshots.append(snapshot)
        if snapshot.record_id == "accepted-2":
            release_first.set()

    output = tmp_path / "parallel.jsonl"
    rows = run_records(
        _records()[:2],
        predictions_path=output,
        config_hash="parallel-config",
        record_runner=runner,
        resume=False,
        concurrency=2,
        progress_callback=capture,
    )

    expected_ids = ["accepted-1", "accepted-2"]
    completion_ids = [
        snapshot.record_id for snapshot in snapshots if snapshot.record_id
    ]
    persisted_ids = [
        json.loads(line)["record_id"]
        for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert completion_ids == ["accepted-2", "accepted-1"]
    assert [row["record_id"] for row in rows] == expected_ids
    assert persisted_ids == expected_ids
    assert snapshots[0].status == "starting"
    assert snapshots[-1].status == "completed"
    assert snapshots[-1].completed == 2
    assert snapshots[-1].total == 2
    assert snapshots[-1].eta_seconds == 0.0


class _AnchoredAndLegacyStage3Agents(_DeterministicAgents):
    def __init__(self) -> None:
        super().__init__()
        self.anchor_calls: list[str] = []
        self.legacy_stage3_calls: list[tuple[str, str]] = []

    def joint_anchor(self, team_id: str, view: object) -> JointAnchorReport:
        self.anchor_calls.append(team_id)
        return JointAnchorReport(
            symptom=super().symptom_analyst(team_id, view),
            root_cause=super().root_cause_analyst(team_id, view),
            causal_account=(
                "The incorrect rejection branch directly explains the termination."
            ),
            shared_supporting_evidence_ids=[view.items[0].evidence_id],
        )

    @staticmethod
    def symptom_verifier(
        team_id: str, anchor: JointAnchorReport, view: object
    ) -> DimensionVerificationReport:
        return DimensionVerificationReport(
            dimension=EvidenceDimension.SYMPTOM,
            verdict=VerificationVerdict.ACCEPT,
            anchor_label=anchor.symptom.label,
            rationale="Independent symptom evidence confirms the anchored label.",
            supporting_evidence_ids=[],
            confidence=0.88,
        )

    @staticmethod
    def root_cause_verifier(
        team_id: str, anchor: JointAnchorReport, view: object
    ) -> DimensionVerificationReport:
        return DimensionVerificationReport(
            dimension=EvidenceDimension.ROOT_CAUSE,
            verdict=VerificationVerdict.ACCEPT,
            anchor_label=anchor.root_cause.label,
            rationale="Independent mechanism evidence confirms the anchored label.",
            supporting_evidence_ids=[],
            confidence=0.87,
        )

    def symptom_analyst(self, team_id: str, view: object) -> SymptomReport:
        self.legacy_stage3_calls.append((team_id, "symptom"))
        return super().symptom_analyst(team_id, view)

    def root_cause_analyst(self, team_id: str, view: object) -> RootCauseReport:
        self.legacy_stage3_calls.append((team_id, "root_cause"))
        return super().root_cause_analyst(team_id, view)


def _stage3_record() -> dict[str, str]:
    return {
        "record_id": "anchored-stage3",
        "title": "Valid request terminates",
        "body": "The request is rejected before the patch.",
        "code_diff": "- reject(request)\n+ process(request)",
    }


def _stage3_kwargs() -> dict[str, object]:
    return {
        "domain": "ase2022",
        "taxonomy": {
            "decision": ["accepted_fault", "rejected_candidate"],
            "symptom": ["Crash", "Poor Performance"],
            "root_cause": ["Incorrect Code Logic", "API Misuse"],
        },
        "stage": "stage3",
        "retrieved_at": "2026-08-05T13:00:00Z",
    }


def test_normal_experiment_prefers_anchored_roles_over_legacy_analysts() -> None:
    agents = _AnchoredAndLegacyStage3Agents()

    row = run_adaptive_record(_stage3_record(), agents=agents, **_stage3_kwargs())

    assert sorted(agents.anchor_calls) == ["A", "B"]
    assert agents.legacy_stage3_calls == []
    assert row["stage3_valid"] is True
    assert row["symptom_prediction"] == "Crash"
    assert row["root_cause_prediction"] == "Incorrect Code Logic"
    reports = row["audit"]["stage3"]["reports"]
    assert [report["team_id"] for report in reports] == ["A", "B"]
    assert all(report["anchor"] is not None for report in reports)
    assert all(len(report["verifications"]) == 2 for report in reports)
    assert all(len(report["correction_audit"]) == 2 for report in reports)


def test_legacy_only_stage3_adapter_keeps_legacy_report_shape() -> None:
    row = run_adaptive_record(
        _stage3_record(), agents=_DeterministicAgents(), **_stage3_kwargs()
    )

    assert row["stage3_valid"] is True
    assert row["symptom_prediction"] == "Crash"
    assert row["root_cause_prediction"] == "Incorrect Code Logic"
    reports = row["audit"]["stage3"]["reports"]
    assert [report["team_id"] for report in reports] == ["A", "B"]
    assert all(report["anchor"] is None for report in reports)
    assert all(report["verifications"] == [] for report in reports)
    assert all(report["correction_audit"] == [] for report in reports)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda agents: setattr(agents, "root_cause_verifier", None),
        lambda agents: setattr(agents, "symptom_verifier", object()),
        lambda agents: setattr(agents, "joint_anchor", None),
    ],
)
def test_partial_new_role_surface_fails_closed(mutate: object) -> None:
    agents = _AnchoredAndLegacyStage3Agents()
    mutate(agents)

    with pytest.raises(ValueError, match="partial baseline-anchored Stage 3"):
        run_adaptive_record(_stage3_record(), agents=agents, **_stage3_kwargs())


def test_mixed_complete_new_and_partial_legacy_surface_fails_closed() -> None:
    delegate = _AnchoredAndLegacyStage3Agents()
    agents = SimpleNamespace(
        evidence_readiness=delegate.evidence_readiness,
        joint_anchor=delegate.joint_anchor,
        symptom_verifier=delegate.symptom_verifier,
        root_cause_verifier=delegate.root_cause_verifier,
        symptom_analyst=delegate.symptom_analyst,
        consistency_checker=delegate.consistency_checker,
    )

    with pytest.raises(ValueError, match="partial legacy Stage 3"):
        run_adaptive_record(_stage3_record(), agents=agents, **_stage3_kwargs())


def test_stage3_adapter_without_consistency_checker_fails_closed() -> None:
    delegate = _AnchoredAndLegacyStage3Agents()
    agents = SimpleNamespace(
        evidence_readiness=delegate.evidence_readiness,
        joint_anchor=delegate.joint_anchor,
        symptom_verifier=delegate.symptom_verifier,
        root_cause_verifier=delegate.root_cause_verifier,
    )

    with pytest.raises(ValueError, match="consistency_checker"):
        run_adaptive_record(_stage3_record(), agents=agents, **_stage3_kwargs())
