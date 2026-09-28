from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor
from email.message import Message
import hashlib
import http.client
import io
import json
import shutil
import subprocess
import threading
import time
import urllib.error
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest
from pydantic import BaseModel

import Benchmark.src.adaptive_empirical_workflow.frozen_evidence_graph as graph_module
from Benchmark.src.adaptive_empirical_workflow import sla_budget as sla_budget_module
from Benchmark.scripts.run_adaptive_empirical_workflow import (
    ConsoleProgress,
    _code_state,
    _targeted_provider_request,
    build_parser,
    run_cli,
)
from Benchmark.scripts import run_adaptive_empirical_workflow as runner_script
from Benchmark.src.adaptive_empirical_workflow.capabilities import (
    AnalystRole,
    render_system_prompt,
)
from Benchmark.src.adaptive_empirical_workflow.contracts import (
    baseline_revision_assessment_digest,
    BaselineRevisionAssessment,
    RevisionConsistencyReport,
    RoleModelPolicy,
    Stage3AgentPolicy,
    ThinkingMode,
)
from Benchmark.src.adaptive_empirical_workflow.experiment import ProgressSnapshot
from Benchmark.src.adaptive_empirical_workflow.sla_budget import SlaBudgetExhausted
from Benchmark.src.adaptive_empirical_workflow.agents import (
    baseline_revision_prompt_hashes,
    ModelCallOptions,
    ModelTransportError,
    ModelTransportResponse,
    StructuredModelClient,
)
from Benchmark.src.ase2022_llm_baseline import (
    _is_retryable_network_error,
    model_transport_error_details,
)


def _write_inputs(
    tmp_path: Path,
    *,
    record_id: str = "record-1",
) -> tuple[Path, Path]:
    cohort = tmp_path / "cohort.csv"
    with cohort.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "record_id",
                "title",
                "body",
                "code_diff",
                "decision",
                "symptom",
                "root_cause",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "record_id": record_id,
                "title": "Request rejected",
                "body": "A valid request is rejected.",
                "code_diff": "- reject(request)\n+ process(request)",
                "decision": "accepted_fault",
                "symptom": "Crash",
                "root_cause": "Incorrect Code Logic",
            }
        )
    taxonomy = tmp_path / "taxonomy.json"
    taxonomy.write_text(
        json.dumps(
            {
                "symptom": ["Crash", "Incorrect Functionality"],
                "root_cause": ["Incorrect Code Logic", "API Misuse"],
            }
        ),
        encoding="utf-8",
    )
    return cohort, taxonomy


def _write_ordering_inputs(tmp_path: Path) -> tuple[Path, Path]:
    cohort, taxonomy = _write_inputs(tmp_path)
    fieldnames = [
        "record_id",
        "title",
        "body",
        "code_diff",
        "decision",
        "symptom",
        "root_cause",
    ]
    with cohort.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        for index in (2, 3):
            writer.writerow(
                {
                    "record_id": f"record-{index}",
                    "title": f"Request {index} rejected",
                    "body": "A valid request is rejected.",
                    "code_diff": "- reject(request)\n+ process(request)",
                    "decision": "accepted_fault",
                    "symptom": "Crash",
                    "root_cause": "Incorrect Code Logic",
                }
            )
    return cohort, taxonomy


def _write_preservation_inputs(
    tmp_path: Path, *, baseline_symptom: str = "Crash"
) -> tuple[Path, Path, Path]:
    source_split = Path("Benchmark/configs/splits/ase2022_stage3_contaminated_dev50")
    split_root = tmp_path / "ase2022_stage3_contaminated_dev50"
    split_root.mkdir(exist_ok=True)
    shutil.copy2(
        source_split / "development_runner_cohort.csv",
        split_root / "development_runner_cohort.csv",
    )
    shutil.copy2(
        source_split / "split_manifest.json", split_root / "split_manifest.json"
    )
    cohort = split_root / "development_runner_cohort.csv"
    taxonomy = Path(
        "Benchmark/inputs/ase2022_issue_only_holdout_seed20260806/"
        "ase2022_issue_only_holdout_taxonomy.json"
    )
    anchors = tmp_path / "synthetic-baseline.jsonl"
    ids = [row["record_id"] for row in csv.DictReader(cohort.open(encoding="utf-8"))]
    rows = [{"record_id": rid, "config_hash": "a" * 64, "invalid": False,
             "final_prediction": {"symptom": baseline_symptom,
                                  "root_cause": "Incorrect Code Logic"}} for rid in ids]
    anchors.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    from Benchmark.scripts import prepare_baseline_trust_root as prepare
    original_root = prepare.ROOT
    try:
        prepare.ROOT = tmp_path
        prepare.register_baseline_trust_root(anchors, output=_trust_path(anchors),
            artifact_id="synthetic-dev50-test-only", domain="ase2022",
            baseline_code_sha256="b" * 64)
    finally:
        prepare.ROOT = original_root
    trust = json.loads(_trust_path(anchors).read_text(encoding="utf-8"))
    trust["status"] = "active"
    _trust_path(anchors).write_text(json.dumps(trust), encoding="utf-8")
    _TEST_TRUST_DIGESTS["trust.json"] = anchor_module._canonical_sha256(trust)
    return cohort, taxonomy, anchors


from Benchmark.src.adaptive_empirical_workflow import baseline_anchor as anchor_module
_TEST_TRUST_DIGESTS: dict[str, str] = {}


@pytest.fixture(autouse=True)
def synthetic_baseline_registration(tmp_path, monkeypatch):
    """Register only synthetic local fixtures; retain all production hash/path checks."""
    _TEST_TRUST_DIGESTS.clear()
    monkeypatch.setattr(anchor_module, "_repository_root", lambda: tmp_path)
    monkeypatch.setattr(anchor_module, "_registered_baseline_sha", _TEST_TRUST_DIGESTS.get)


def _trust_path(anchors: Path) -> Path:
    return anchors.with_name("trust.json")


def test_targeted_sla30_rejects_incompatible_cli_surface_before_model_calls(
    tmp_path: Path,
) -> None:
    cohort, taxonomy = _write_inputs(tmp_path)
    base = [
        "--domain",
        "ase2022",
        "--cohort-path",
        str(cohort),
        "--taxonomy-path",
        str(taxonomy),
        "--output-dir",
        str(tmp_path / "targeted"),
        "--execution-profile",
        "targeted-sla30",
        "--dry-run",
    ]
    cases = [
        (base, "targeted-sla30 requires --stage stage3"),
        ([*base, "--stage", "stage3"], "targeted-sla30 requires explicit --record-ids"),
        (
            [*base, "--stage", "stage3", "--record-ids", "record-1"],
            "targeted-sla30 requires --baseline-anchor-path",
        ),
        (
            [
                *base,
                "--stage",
                "stage3",
                "--record-ids",
                "record-1",
                "--baseline-anchor-path",
                "baseline.jsonl",
            ],
            "targeted-sla30 requires --provider micu",
        ),
    ]
    for arguments, message in cases:
        with pytest.raises(ValueError, match=message):
            run_cli(arguments)


def test_targeted_sla30_rejects_preservation_bad_budgets_and_wrong_scheduling(
    tmp_path: Path,
) -> None:
    cohort, taxonomy = _write_inputs(tmp_path)
    base = [
        "--domain",
        "ase2022",
        "--stage",
        "stage3",
        "--provider",
        "micu",
        "--model",
        "deepseek-v4-flash",
        "--cohort-path",
        str(cohort),
        "--taxonomy-path",
        str(taxonomy),
        "--output-dir",
        str(tmp_path / "targeted"),
        "--execution-profile",
        "targeted-sla30",
        "--record-ids",
        "record-1",
        "--baseline-anchor-path",
        "baseline.jsonl",
        "--dry-run",
    ]
    for arguments, message in (
        ([*base, "--baseline-preservation"], "cannot be combined"),
        ([*base, "--global-deadline-seconds", "0"], "global_seconds"),
        ([*base, "--record-budget-seconds", "0"], "record_seconds"),
        ([*base, "--request-timeout-seconds", "0"], "request_seconds"),
        ([*base, "--concurrency", "3"], "concurrency=4"),
        ([*base, "--provider-max-inflight", "5"], "between 1 and 4"),
    ):
        with pytest.raises(ValueError, match=message):
            run_cli(arguments)


def test_targeted_sla30_dry_run_records_the_isolated_sla_manifest(
    tmp_path: Path,
) -> None:
    cohort, taxonomy = _write_inputs(tmp_path)
    anchors = tmp_path / "baseline.jsonl"
    anchors.write_text(
        json.dumps(
            {
                "record_id": "record-1",
                "invalid": False,
                "config_hash": "a" * 64,
                "final_prediction": {
                    "symptom": "Crash",
                    "root_cause": "Incorrect Code Logic",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = run_cli(
        [
            "--domain",
            "ase2022",
            "--stage",
            "stage3",
            "--provider",
            "micu",
            "--model",
            "deepseek-v4-flash",
            "--cohort-path",
            str(cohort),
            "--taxonomy-path",
            str(taxonomy),
            "--output-dir",
            str(tmp_path / "targeted"),
            "--execution-profile",
            "targeted-sla30",
            "--record-ids",
            "record-1",
            "--baseline-anchor-path",
            str(anchors),
            "--dry-run",
        ]
    )

    manifest = json.loads(Path(summary["manifest_path"]).read_text(encoding="utf-8"))
    assert {
        key: manifest[key]
        for key in (
            "execution_profile",
            "development_only",
            "baseline_preservation_protocol",
            "sla_policy_version",
            "global_deadline_seconds",
            "record_budget_seconds",
            "request_timeout_seconds",
            "concurrency",
            "provider_max_inflight",
        )
    } == {
        "execution_profile": "targeted-sla30",
        "development_only": True,
        "baseline_preservation_protocol": False,
        "sla_policy_version": "targeted-sla30-v1",
        "global_deadline_seconds": 1680.0,
        "record_budget_seconds": 240.0,
        "request_timeout_seconds": 75.0,
        "concurrency": 4,
        "provider_max_inflight": 4,
    }
    assert manifest["targeted_sla_baseline"]["record_anchor_digests"]
    assert manifest["max_network_retries"] == 1
    assert manifest["max_schema_retries"] == 1


def test_default_profile_keeps_the_pre_profile_config_identity_surface(
    tmp_path: Path,
) -> None:
    cohort, taxonomy = _write_inputs(tmp_path)

    summary = run_cli(
        [
            "--domain",
            "ase2022",
            "--cohort-path",
            str(cohort),
            "--taxonomy-path",
            str(taxonomy),
            "--output-dir",
            str(tmp_path / "normal"),
            "--dry-run",
        ]
    )

    manifest = json.loads(Path(summary["manifest_path"]).read_text(encoding="utf-8"))
    assert "execution_profile" not in manifest["resolved_config"]
    assert "execution_profile" not in manifest


def test_targeted_provider_retry_reuses_one_budget_and_fresh_admission() -> None:
    class Budget:
        def __init__(self) -> None:
            self.requests = 0

        def require_request_budget(self) -> float:
            self.requests += 1
            return 12.0

        def remaining_seconds(self) -> float:
            return 12.0

    budget = Budget()
    calls: list[float] = []
    responses = [
        ModelTransportError(
            "retry",
            retryable=True,
            cause_type="connection_error",
        ),
        '{"ok": true}',
    ]

    def call_once(request: object, _completion_lease: object) -> str:
        assert isinstance(request, sla_budget_module.SlaRequestDeadline)
        calls.append(request.timeout_seconds)
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    response = _targeted_provider_request(
        record_budget=budget,
        admission=threading.BoundedSemaphore(1),
        call_once=call_once,
    )

    assert str(response) == '{"ok": true}'
    assert calls == [12.0, 12.0]
    assert budget.requests == 2
    assert response.network_attempts == 2
    assert [event["status"] for event in response.network_attempt_details] == [
        "retryable_error",
        "success",
    ]


def test_targeted_provider_retry_exhaustion_and_admission_expiry_are_typed() -> None:
    class ExhaustingBudget:
        def __init__(self, *, expires: bool = False) -> None:
            self.requests = 0
            self.expires = expires

        def require_request_budget(self) -> float:
            self.requests += 1
            if self.expires:
                from Benchmark.src.adaptive_empirical_workflow.sla_budget import (
                    SlaBudgetExhausted,
                )

                raise SlaBudgetExhausted("expired while waiting for admission")
            return 8.0

        def remaining_seconds(self) -> float:
            return 8.0

    retry_budget = ExhaustingBudget()
    with pytest.raises(ModelTransportError) as failure:
        _targeted_provider_request(
            record_budget=retry_budget,
            admission=threading.BoundedSemaphore(1),
            call_once=lambda _request, _completion_lease: (_ for _ in ()).throw(
                ModelTransportError(
                    "retry", retryable=True, cause_type="connection_error"
                )
            ),
        )
    assert failure.value.attempts == 2
    assert retry_budget.requests == 2

    expired_budget = ExhaustingBudget(expires=True)
    with pytest.raises(Exception, match="expired while waiting") as expired:
        _targeted_provider_request(
            record_budget=expired_budget,
            admission=threading.BoundedSemaphore(1),
            call_once=lambda _request, _completion_lease: pytest.fail(
                "provider must not be called"
            ),
        )
    assert expired.value.__class__.__name__ == "SlaBudgetExhausted"


def test_targeted_provider_admission_worker_lifetime() -> None:
    class Budget:
        def __init__(self, admission_started: threading.Event | None = None) -> None:
            self.admission_started = admission_started

        def require_request_deadline(self) -> sla_budget_module.SlaRequestDeadline:
            return sla_budget_module.SlaRequestDeadline(
                timeout_seconds=5.0,
                deadline_monotonic=time.monotonic() + 5.0,
            )

        def remaining_seconds(self) -> float:
            if self.admission_started is not None:
                self.admission_started.set()
            return 5.0

    admission = threading.BoundedSemaphore(1)
    first_worker_started = threading.Event()
    release_first_worker = threading.Event()
    first_worker_finished = threading.Event()
    second_admission_started = threading.Event()
    second_request_entered = threading.Event()
    request_count = 0

    def call_once(
        _deadline: sla_budget_module.SlaRequestDeadline,
        completion_lease: object | None = None,
    ) -> str:
        nonlocal request_count
        request_count += 1
        if request_count == 1:
            if completion_lease is not None:
                completion_lease.transfer_to_worker()

            def late_worker() -> None:
                first_worker_started.set()
                try:
                    assert release_first_worker.wait(timeout=5.0)
                finally:
                    if completion_lease is not None:
                        completion_lease.release_from_worker()
                    first_worker_finished.set()

            worker = threading.Thread(target=late_worker, daemon=True)
            worker.start()
            assert first_worker_started.wait(timeout=5.0)
            raise ModelTransportError(
                "simulated caller timeout",
                retryable=False,
                cause_type="timeout_error",
            )
        second_request_entered.set()
        return '{"ok": true}'

    def attempt(budget: Budget) -> ModelTransportResponse:
        return _targeted_provider_request(
            record_budget=budget,
            admission=admission,
            retry_budget=sla_budget_module.RoleNetworkRetryBudget(max_retries=0),
            call_once=call_once,
        )

    executor = ThreadPoolExecutor(max_workers=2)
    try:
        first = executor.submit(attempt, Budget())
        assert first_worker_started.wait(timeout=5.0)
        with pytest.raises(ModelTransportError):
            first.result(timeout=5.0)

        reacquired = admission.acquire(blocking=False)
        if reacquired:
            admission.release()
        assert not reacquired

        second = executor.submit(attempt, Budget(second_admission_started))
        assert second_admission_started.wait(timeout=5.0)
        assert not second_request_entered.is_set()

        release_first_worker.set()
        assert first_worker_finished.wait(timeout=5.0)
        assert str(second.result(timeout=5.0)) == '{"ok": true}'
    finally:
        release_first_worker.set()
        executor.shutdown(wait=True)

    assert request_count == 2
    assert second_request_entered.is_set()
    assert admission.acquire(timeout=0.5)
    admission.release()


def test_targeted_role_keeps_one_network_retry_across_schema_repair() -> None:
    class Answer(BaseModel):
        ok: bool

    record_budget = sla_budget_module.GlobalSlaBudget(
        sla_budget_module.SlaBudgetConfig()
    ).start_record()
    retry_budget = sla_budget_module.RoleNetworkRetryBudget(max_retries=1)
    provider_results: list[str | Exception] = [
        '{"wrong": true}',
        ModelTransportError(
            "repair request disconnected",
            retryable=True,
            cause_type="connection_error",
        ),
        '{"ok": true}',
    ]
    provider_calls = 0

    def transport(_system: str, _user: str, *, options: ModelCallOptions) -> str:
        del options

        def call_once(_request: object, _completion_lease: object) -> str:
            nonlocal provider_calls
            provider_calls += 1
            result = provider_results.pop(0)
            if isinstance(result, Exception):
                raise result
            return result

        return _targeted_provider_request(
            record_budget=record_budget,
            retry_budget=retry_budget,
            admission=threading.BoundedSemaphore(1),
            call_once=call_once,
        )

    result = StructuredModelClient(transport, max_schema_retries=1).complete(
        system_prompt="system",
        user_prompt="user",
        schema=Answer,
        options=ModelCallOptions(
            role="sla_joint_diagnosis",
            team_id=None,
            perspective=None,
            max_tokens=10,
            thinking_enabled=False,
        ),
    )

    assert result.ok is True
    assert provider_calls == 3
    assert retry_budget.used_retries == 1


def test_targeted_role_never_gets_a_second_network_retry_after_schema_repair() -> None:
    class Answer(BaseModel):
        ok: bool

    record_budget = sla_budget_module.GlobalSlaBudget(
        sla_budget_module.SlaBudgetConfig()
    ).start_record()
    retry_budget = sla_budget_module.RoleNetworkRetryBudget(max_retries=1)
    provider_results: list[str | Exception] = [
        ModelTransportError(
            "first request disconnected",
            retryable=True,
            cause_type="connection_error",
        ),
        '{"wrong": true}',
        ModelTransportError(
            "repair request disconnected",
            retryable=True,
            cause_type="connection_error",
        ),
        '{"ok": true}',
    ]
    provider_calls = 0

    def transport(_system: str, _user: str, *, options: ModelCallOptions) -> str:
        del options

        def call_once(_request: object, _completion_lease: object) -> str:
            nonlocal provider_calls
            provider_calls += 1
            result = provider_results.pop(0)
            if isinstance(result, Exception):
                raise result
            return result

        return _targeted_provider_request(
            record_budget=record_budget,
            retry_budget=retry_budget,
            admission=threading.BoundedSemaphore(1),
            call_once=call_once,
        )

    with pytest.raises(ModelTransportError):
        StructuredModelClient(transport, max_schema_retries=1).complete(
            system_prompt="system",
            user_prompt="user",
            schema=Answer,
            options=ModelCallOptions(
                role="sla_joint_diagnosis",
                team_id=None,
                perspective=None,
                max_tokens=10,
                thinking_enabled=False,
            ),
        )

    assert provider_calls == 3
    assert retry_budget.used_retries == 1
    assert provider_results == ['{"ok": true}']


def test_admission_budget_expiry_is_not_counted_as_a_provider_request() -> None:
    exhausted = SlaBudgetExhausted("expired after admission")

    class Budget:
        def require_request_budget(self) -> float:
            raise exhausted

        def remaining_seconds(self) -> float:
            return 1.0

    client = StructuredModelClient(
        lambda _system, _user, *, options: _targeted_provider_request(
            record_budget=Budget(),
            admission=threading.BoundedSemaphore(1),
            call_once=lambda _request, _completion_lease: pytest.fail(
                "provider must not be called"
            ),
        ),
        max_schema_retries=1,
    )

    with pytest.raises(SlaBudgetExhausted) as raised:
        client.complete(
            system_prompt="ROLE: sla_joint_diagnosis",
            user_prompt="Return JSON.",
            schema=RevisionConsistencyReport,
            options=ModelCallOptions(
                role="sla_joint_diagnosis",
                team_id=None,
                perspective=None,
                max_tokens=100,
                thinking_enabled=False,
            ),
        )

    assert raised.value is exhausted
    call = client.telemetry()["calls"][0]
    assert call["network_attempts"] == 0
    assert call["network_attempt_details"] == []


def test_retry_then_admission_budget_expiry_preserves_actual_request_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cohort, taxonomy = _write_inputs(tmp_path)
    anchors = tmp_path / "baseline.jsonl"
    anchors.write_text(
        json.dumps(
            {
                "record_id": "record-1",
                "invalid": False,
                "config_hash": "a" * 64,
                "final_prediction": {
                    "symptom": "Crash",
                    "root_cause": "Incorrect Code Logic",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class Budget:
        def __init__(self) -> None:
            self.checks = 0

        def require_request_budget(self) -> float:
            self.checks += 1
            if self.checks == 3:
                raise SlaBudgetExhausted("retry budget exhausted after admission")
            return 10.0

        def remaining_seconds(self) -> float:
            return 0.0 if self.checks >= 3 else 10.0

        def require_request_deadline(self) -> sla_budget_module.SlaRequestDeadline:
            timeout_seconds = self.require_request_budget()
            return sla_budget_module.SlaRequestDeadline(
                timeout_seconds=timeout_seconds,
                deadline_monotonic=time.monotonic() + timeout_seconds,
            )

    budget = Budget()

    class GlobalBudget:
        def __init__(self, _config: object) -> None:
            pass

        def start_record(self) -> Budget:
            return budget

    provider_calls = 0

    def retryable_provider_failure(_system: str, _user: str) -> str:
        nonlocal provider_calls
        provider_calls += 1
        raise ModelTransportError(
            "retryable provider failure",
            retryable=True,
            cause_type="connection_error",
        )

    monkeypatch.setattr(runner_script, "GlobalSlaBudget", GlobalBudget)
    summary = run_cli(
        [
            "--domain",
            "ase2022",
            "--stage",
            "stage3",
            "--provider",
            "micu",
            "--model",
            "deepseek-v4-flash",
            "--cohort-path",
            str(cohort),
            "--taxonomy-path",
            str(taxonomy),
            "--output-dir",
            str(tmp_path / "retry-budget"),
            "--execution-profile",
            "targeted-sla30",
            "--record-ids",
            "record-1",
            "--baseline-anchor-path",
            str(anchors),
            "--no-progress",
            "--no-resume",
        ],
        transport_override=retryable_provider_failure,
    )

    row = json.loads(
        Path(summary["predictions_path"]).read_text(encoding="utf-8").strip()
    )
    assert provider_calls == 1
    assert row["decision_status"] == {
        "symptom": "budget_fallback",
        "root_cause": "budget_fallback",
    }
    assert len(row["call_audit"]) == 1
    call_audit = row["call_audit"][0]
    assert call_audit["role"] == "sla_joint_diagnosis"
    assert call_audit["latency_seconds"] >= 0.0
    assert call_audit["schema_attempts"] == 1
    assert call_audit["network_attempts"] == 1
    assert call_audit["provider_errors"] == [
        {
            "status": "retryable_error",
            "retryable": True,
            "cause_type": "connection_error",
            "http_status": None,
            "retry_after_seconds": None,
        }
    ]
    metrics = summary["metrics"]["stage3"]
    assert metrics["role_call_count"] == 1
    assert metrics["provider_request_count"] == 1
    assert metrics["provider_error_distribution"] == {"connection_error": 1}


def test_targeted_sla30_runs_only_the_bounded_record_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cohort, taxonomy = _write_inputs(tmp_path)
    anchors = tmp_path / "baseline.jsonl"
    anchors.write_text(
        json.dumps(
            {
                "record_id": "record-1",
                "invalid": False,
                "config_hash": "a" * 64,
                "final_prediction": {
                    "symptom": "Crash",
                    "root_cause": "Incorrect Code Logic",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    calls: list[str] = []

    def targeted(record: dict[str, str], **_: object) -> dict[str, object]:
        calls.append(record["record_id"])
        return {
            "record_id": record["record_id"],
            "stage2_prediction": None,
            "stage2_valid": False,
            "symptom_prediction": "Crash",
            "root_cause_prediction": "Incorrect Code Logic",
            "stage3_valid": True,
            "stop_reason": "targeted_sla30",
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
                            "role": "sla_joint_diagnosis",
                            "latency_seconds": 1.0,
                            "schema_attempts": 1,
                            "network_attempts": 1,
                            "provider_errors": [],
                        },
                        {
                            "role": "sla_joint_verifier",
                            "latency_seconds": 1.0,
                            "schema_attempts": 1,
                            "network_attempts": 1,
                            "provider_errors": [],
                        },
                    ],
                }
            },
        }

    monkeypatch.setattr(runner_script, "run_targeted_sla_record", targeted)
    monkeypatch.setattr(
        runner_script,
        "run_adaptive_record",
        lambda *_args, **_kwargs: pytest.fail("normal record path was selected"),
    )
    summary = run_cli(
        [
            "--domain",
            "ase2022",
            "--stage",
            "stage3",
            "--provider",
            "micu",
            "--model",
            "deepseek-v4-flash",
            "--cohort-path",
            str(cohort),
            "--taxonomy-path",
            str(taxonomy),
            "--output-dir",
            str(tmp_path / "targeted-run"),
            "--execution-profile",
            "targeted-sla30",
            "--record-ids",
            "record-1",
            "--baseline-anchor-path",
            str(anchors),
            "--no-progress",
        ],
        transport_override=lambda *_args, **_kwargs: pytest.fail(
            "no model call expected"
        ),
    )

    assert calls == ["record-1"]
    metrics = summary["metrics"]["stage3"]
    assert metrics["coverage"] == 1.0
    assert metrics["completed_model_decisions"] == 1
    assert metrics["model_call_count"] == 2


@pytest.mark.parametrize(
    ("provider", "model", "api_key_name", "expected_thinking"),
    [
        ("micu", "deepseek-v4-flash-0731", "MICU_API_KEY", False),
        ("gemini", "gemini-3.7-flash", "GOOGLE_API_KEY", None),
    ],
)
def test_targeted_sla30_uses_provider_compatible_streaming_fresh_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    model: str,
    api_key_name: str,
    expected_thinking: bool | None,
) -> None:
    cohort, taxonomy = _write_inputs(tmp_path)
    anchors = tmp_path / "baseline.jsonl"
    anchors.write_text(
        json.dumps(
            {
                "record_id": "record-1",
                "invalid": False,
                "config_hash": "a" * 64,
                "final_prediction": {
                    "symptom": "Crash",
                    "root_cause": "Incorrect Code Logic",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    captured_requests: list[dict[str, object]] = []
    captured_client: dict[str, StructuredModelClient] = {}
    real_agents = runner_script.StructuredRoleAgents

    def capture_agents(
        client: StructuredModelClient, **kwargs: object
    ) -> object:
        captured_client["client"] = client
        return real_agents(client, **kwargs)

    def fresh_transport(**kwargs: object) -> str:
        captured_requests.append(kwargs)
        return '{"ok": true}'

    class Answer(BaseModel):
        ok: bool

    def targeted(record: dict[str, str], **_: object) -> dict[str, object]:
        answer = captured_client["client"].complete(
            system_prompt="system",
            user_prompt="user",
            schema=Answer,
            options=ModelCallOptions(
                role="sla_joint_diagnosis",
                team_id=None,
                perspective=None,
                max_tokens=2400,
                thinking_enabled=False,
            ),
        )
        assert answer.ok is True
        return {
            "record_id": record["record_id"],
            "stage2_prediction": None,
            "stage2_valid": False,
            "symptom_prediction": "Crash",
            "root_cause_prediction": "Incorrect Code Logic",
            "stage3_valid": True,
            "stop_reason": "targeted_sla30",
            "call_count": 1,
            "fallback": False,
            "audit": {
                "targeted_sla": {
                    "decision_status": {
                        "symptom": "verified",
                        "root_cause": "verified",
                    },
                    "call_audit": [],
                }
            },
        }

    monkeypatch.setenv(api_key_name, "test-only-key")
    monkeypatch.setattr(runner_script, "StructuredRoleAgents", capture_agents)
    monkeypatch.setattr(runner_script, "call_fresh_chat_completion", fresh_transport)
    monkeypatch.setattr(runner_script, "run_targeted_sla_record", targeted)

    run_cli(
        [
            "--domain",
            "ase2022",
            "--stage",
            "stage3",
            "--provider",
            provider,
            "--model",
            model,
            "--cohort-path",
            str(cohort),
            "--taxonomy-path",
            str(taxonomy),
            "--output-dir",
            str(tmp_path / "streaming-targeted-run"),
            "--execution-profile",
            "targeted-sla30",
            "--record-ids",
            "record-1",
            "--baseline-anchor-path",
            str(anchors),
            "--no-progress",
            "--no-resume",
        ]
    )

    assert len(captured_requests) == 1
    assert captured_requests[0]["stream"] is True
    assert "thinking_enabled" in captured_requests[0]
    assert captured_requests[0]["thinking_enabled"] is expected_thinking


def test_targeted_sla30_binds_deadline_before_manifest_and_measures_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cohort, taxonomy = _write_inputs(tmp_path)
    anchors = tmp_path / "baseline.jsonl"
    anchors.write_text(
        json.dumps(
            {
                "record_id": "record-1",
                "invalid": False,
                "config_hash": "a" * 64,
                "final_prediction": {
                    "symptom": "Crash",
                    "root_cause": "Incorrect Code Logic",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    clock = [100.0]
    metrics_writes: list[Path] = []
    original_manifest_writer = runner_script.write_run_manifest
    original_json_writer = runner_script._write_json

    def write_manifest(*args: object, **kwargs: object) -> Path:
        result = original_manifest_writer(*args, **kwargs)
        clock[0] += 5.0
        return result

    def write_json(*args: object, **kwargs: object) -> None:
        metrics_writes.append(Path(args[0]))
        original_json_writer(*args, **kwargs)
        clock[0] += 7.0

    monkeypatch.setattr(runner_script.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(runner_script, "write_run_manifest", write_manifest)
    monkeypatch.setattr(runner_script, "_write_json", write_json)
    monkeypatch.setattr(
        runner_script,
        "run_targeted_sla_record",
        lambda record, **_kwargs: {
            "record_id": record["record_id"],
            "stage3_valid": True,
            "symptom_prediction": "Crash",
            "root_cause_prediction": "Incorrect Code Logic",
            "call_count": 0,
            "fallback": True,
            "audit": {
                "targeted_sla": {
                    "decision_status": {
                        "symptom": "budget_fallback",
                        "root_cause": "budget_fallback",
                    },
                    "call_audit": [],
                }
            },
        },
    )

    summary = run_cli(
        [
            "--domain",
            "ase2022",
            "--stage",
            "stage3",
            "--provider",
            "micu",
            "--model",
            "deepseek-v4-flash",
            "--cohort-path",
            str(cohort),
            "--taxonomy-path",
            str(taxonomy),
            "--output-dir",
            str(tmp_path / "timed"),
            "--execution-profile",
            "targeted-sla30",
            "--record-ids",
            "record-1",
            "--baseline-anchor-path",
            str(anchors),
            "--no-progress",
        ],
        transport_override=lambda *_args, **_kwargs: pytest.fail(
            "record boundary is replaced"
        ),
    )

    manifest = json.loads(Path(summary["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["deadline_started_at"]
    assert manifest["deadline_clock_policy"] == "process_monotonic-v1"
    assert summary["metrics"]["stage3"]["run_wall_time_seconds"] >= 5.0
    assert metrics_writes == [Path(summary["metrics_path"])]


def test_metrics_json_atomic_replace_preserves_previous_file_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "metrics.json"
    previous = '{"status":"previous"}\n'
    output.write_text(previous, encoding="utf-8")

    def deny_replace(source: object, target: object) -> None:
        raise PermissionError(f"replace denied: {source} -> {target}")

    monkeypatch.setattr(runner_script.os, "replace", deny_replace)

    with pytest.raises(PermissionError, match="replace denied"):
        runner_script._write_json(output, {"status": "final"})

    assert output.read_text(encoding="utf-8") == previous
    assert list(tmp_path.glob(".aew-metrics-*.tmp")) == []


def test_metrics_json_fsyncs_same_directory_temp_before_atomic_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "metrics.json"
    fsynced: list[int] = []
    replacements: list[tuple[Path, Path]] = []
    original_replace = runner_script.os.replace

    def track_fsync(descriptor: int) -> None:
        fsynced.append(descriptor)

    def track_replace(source: object, target: object) -> None:
        source_path = Path(source)
        target_path = Path(target)
        assert fsynced
        assert source_path.parent == target_path.parent == tmp_path
        replacements.append((source_path, target_path))
        original_replace(source, target)

    monkeypatch.setattr(runner_script.os, "fsync", track_fsync)
    monkeypatch.setattr(runner_script.os, "replace", track_replace)

    runner_script._write_json(output, {"status": "final"})

    assert json.loads(output.read_text(encoding="utf-8")) == {"status": "final"}
    assert len(fsynced) == 1
    assert len(replacements) == 1


def test_targeted_sla30_rejects_resume_that_would_reset_the_deadline(
    tmp_path: Path,
) -> None:
    cohort, taxonomy = _write_inputs(tmp_path)
    anchors = tmp_path / "baseline.jsonl"
    anchors.write_text(
        json.dumps(
            {
                "record_id": "record-1",
                "invalid": False,
                "config_hash": "a" * 64,
                "final_prediction": {
                    "symptom": "Crash",
                    "root_cause": "Incorrect Code Logic",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    arguments = [
        "--domain",
        "ase2022",
        "--stage",
        "stage3",
        "--provider",
        "micu",
        "--model",
        "deepseek-v4-flash",
        "--cohort-path",
        str(cohort),
        "--taxonomy-path",
        str(taxonomy),
        "--output-dir",
        str(tmp_path / "resume"),
        "--execution-profile",
        "targeted-sla30",
        "--record-ids",
        "record-1",
        "--baseline-anchor-path",
        str(anchors),
        "--dry-run",
    ]
    run_cli([*arguments, "--no-resume"])

    with pytest.raises(ValueError, match="does not support resume"):
        run_cli(arguments)


def test_frozen_evidence_preflight_fails_before_any_model_call_when_unregistered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(graph_module, "_REGISTERED_TRUST_ROOTS", MappingProxyType({}))
    cohort, taxonomy = _write_inputs(tmp_path)
    model_calls = 0

    def transport(*args: object, **kwargs: object) -> str:
        nonlocal model_calls
        model_calls += 1
        return "{}"

    with pytest.raises(ValueError, match="trust root|registered"):
        run_cli(
            [
                "--domain",
                "ase2022",
                "--stage",
                "stage3",
                "--cohort-path",
                str(cohort),
                "--taxonomy-path",
                str(taxonomy),
                "--output-dir",
                str(tmp_path / "out"),
                "--frozen-evidence",
                "--no-progress",
            ],
            transport_override=transport,
        )

    assert model_calls == 0


def _fake_registered_bundle(
    *, cohort: Path, split_manifest: Path, merkle: str = "d" * 64
) -> SimpleNamespace:
    return SimpleNamespace(
        domain="ase2022",
        bundle_id="ase2022-dev50-frozen-v1",
        split_id="ase2022-dev50",
        record_ids=("record-1",),
        graphs={"record-1": object()},
        trust_manifest_sha256="e" * 64,
        trust_manifest_relative_path="registered/frozen_evidence_manifest.json",
        policy=SimpleNamespace(
            schema_version="ase-frozen-capture-policy-v1",
            policy_id="github-one-hop-v1",
            runtime_network_forbidden=True,
        ),
        manifest=SimpleNamespace(
            runner_cohort_sha256=hashlib.sha256(cohort.read_bytes()).hexdigest(),
            split_manifest_sha256=hashlib.sha256(
                split_manifest.read_bytes()
            ).hexdigest(),
            bundle_merkle_root=merkle,
            capture_policy=SimpleNamespace(sha256="f" * 64),
            retrieval_tool=SimpleNamespace(
                name="bounded-github-capture",
                version="v1",
                module_sha256="a" * 64,
            ),
            network_policy="offline_capture_only_runtime_network_forbidden",
            gold_accessed=False,
        ),
    )


def test_frozen_evidence_dry_run_binds_registered_bundle_and_projection_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cohort, taxonomy = _write_inputs(tmp_path)
    split_manifest = tmp_path / "split_manifest.json"
    split_manifest.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        runner_script,
        "load_domain_inputs",
        lambda *args, **kwargs: SimpleNamespace(
            profile=SimpleNamespace(
                default_cohort_path=str(cohort), default_taxonomy_path=str(taxonomy)
            ),
            records=[
                {
                    "record_id": "record-1",
                    "title": "Request rejected",
                    "body": "A valid request is rejected.",
                }
            ],
            taxonomy={
                "decision": ["accepted_fault", "rejected_candidate"],
                "symptom": ["Crash", "Incorrect Functionality"],
                "root_cause": ["Incorrect Code Logic", "API Misuse"],
            },
        ),
    )
    monkeypatch.setattr(
        runner_script,
        "load_registered_frozen_evidence_runtime",
        lambda domain: _fake_registered_bundle(
            cohort=cohort, split_manifest=split_manifest
        ),
    )

    summary = run_cli(
        [
            "--domain",
            "ase2022",
            "--stage",
            "stage3",
            "--cohort-path",
            str(cohort),
            "--taxonomy-path",
            str(taxonomy),
            "--split-manifest",
            str(split_manifest),
            "--output-dir",
            str(tmp_path / "out"),
            "--frozen-evidence",
            "--dry-run",
            "--no-progress",
        ],
        transport_override=lambda *args, **kwargs: pytest.fail(
            "dry-run must not call a model"
        ),
    )

    manifest = json.loads(Path(summary["manifest_path"]).read_text(encoding="utf-8"))
    binding = manifest["resolved_config"]["frozen_evidence"]
    assert binding["enabled"] is True
    assert binding["trust_manifest_sha256"] == "e" * 64
    assert binding["bundle_merkle_root"] == "d" * 64
    assert binding["capture_policy_sha256"] == "f" * 64
    assert binding["retrieval_tool"] == {
        "name": "bounded-github-capture",
        "version": "v1",
        "module_sha256": "a" * 64,
    }
    assert binding["projection_policy_sha256"] == (
        runner_script.FROZEN_EVIDENCE_PROJECTION_POLICY_HASH
    )
    assert binding["runtime_network_forbidden"] is True


def test_frozen_evidence_resume_rejects_changed_bundle_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cohort, taxonomy = _write_inputs(tmp_path)
    split_manifest = tmp_path / "split_manifest.json"
    split_manifest.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        runner_script,
        "load_domain_inputs",
        lambda *args, **kwargs: SimpleNamespace(
            profile=SimpleNamespace(
                default_cohort_path=str(cohort), default_taxonomy_path=str(taxonomy)
            ),
            records=[{"record_id": "record-1", "body": "failure"}],
            taxonomy={
                "decision": ["accepted_fault", "rejected_candidate"],
                "symptom": ["Crash"],
                "root_cause": ["Incorrect Code Logic"],
            },
        ),
    )
    current = {"merkle": "d" * 64}
    monkeypatch.setattr(
        runner_script,
        "load_registered_frozen_evidence_runtime",
        lambda domain: _fake_registered_bundle(
            cohort=cohort,
            split_manifest=split_manifest,
            merkle=current["merkle"],
        ),
    )
    args = [
        "--domain",
        "ase2022",
        "--stage",
        "stage3",
        "--cohort-path",
        str(cohort),
        "--taxonomy-path",
        str(taxonomy),
        "--split-manifest",
        str(split_manifest),
        "--output-dir",
        str(tmp_path / "out"),
        "--arm-id",
        "frozen-arm",
        "--frozen-evidence",
        "--dry-run",
        "--no-progress",
    ]
    run_cli(args)
    current["merkle"] = "9" * 64

    with pytest.raises(ValueError, match="configuration mismatch"):
        run_cli(args)


def test_frozen_evidence_fake_smoke_projects_exact_record_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cohort, taxonomy = _write_inputs(tmp_path)
    split_manifest = tmp_path / "split_manifest.json"
    split_manifest.write_text("{}", encoding="utf-8")
    bundle = _fake_registered_bundle(cohort=cohort, split_manifest=split_manifest)
    monkeypatch.setattr(
        runner_script,
        "load_domain_inputs",
        lambda *args, **kwargs: SimpleNamespace(
            profile=SimpleNamespace(
                default_cohort_path=str(cohort), default_taxonomy_path=str(taxonomy)
            ),
            records=[
                {
                    "record_id": "record-1",
                    "title": "Request rejected",
                    "body": "A valid request is rejected.",
                }
            ],
            taxonomy={
                "decision": ["accepted_fault", "rejected_candidate"],
                "symptom": ["Crash", "Incorrect Functionality"],
                "root_cause": ["Incorrect Code Logic", "API Misuse"],
            },
        ),
    )
    monkeypatch.setattr(
        runner_script,
        "load_registered_frozen_evidence_runtime",
        lambda domain: bundle,
    )
    projected_item = SimpleNamespace(evidence_id="feg-node-1")
    monkeypatch.setattr(
        runner_script,
        "project_frozen_evidence_for_record",
        lambda actual_bundle, record_id: SimpleNamespace(
            record_id=record_id,
            items=(projected_item,),
            audit={
                "enabled": True,
                "available": True,
                "bundle_id": bundle.bundle_id,
                "graph_sha256": "c" * 64,
                "projection_policy_sha256": (
                    runner_script.FROZEN_EVIDENCE_PROJECTION_POLICY_HASH
                ),
                "node_count": 1,
                "nodes": (
                    {
                        "evidence_id": "feg-node-1",
                        "node_type": "changed_code",
                        "authority": "direct",
                        "citation_usage": "support_or_counter",
                        "capabilities": ("defect_mechanism",),
                        "revision_authoritative": True,
                        "source_uri": "https://github.com/owner/repo/blob/sha/a.py",
                        "content_sha256": "1" * 64,
                    },
                ),
                "unavailable_reasons": (),
            },
        ),
    )
    captured: dict[str, object] = {}

    def fake_record(record: dict[str, str], **kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {
            "record_id": record["record_id"],
            "stage2_prediction": None,
            "stage2_valid": False,
            "symptom_prediction": "Crash",
            "root_cause_prediction": "Incorrect Code Logic",
            "stage3_valid": True,
            "stop_reason": "stage3_only",
            "audit": {
                "stage3": {
                    "final_decision": {"supporting_evidence_ids": ["feg-node-1"]}
                }
            },
        }

    monkeypatch.setattr(runner_script, "run_adaptive_record", fake_record)
    monkeypatch.setattr(
        runner_script,
        "evaluate_experiment",
        lambda *args, **kwargs: {"stage3": {"joint_accuracy": 1.0}},
    )
    monkeypatch.setattr(runner_script, "stage3_team_diagnostics", lambda *args: {})

    summary = run_cli(
        [
            "--domain",
            "ase2022",
            "--stage",
            "stage3",
            "--cohort-path",
            str(cohort),
            "--taxonomy-path",
            str(taxonomy),
            "--split-manifest",
            str(split_manifest),
            "--output-dir",
            str(tmp_path / "out"),
            "--frozen-evidence",
            "--no-progress",
        ],
        transport_override=lambda *args, **kwargs: pytest.fail(
            "runtime frozen evidence smoke must not access network transport"
        ),
    )

    assert captured["frozen_evidence_projection"].items == (projected_item,)
    row = json.loads(Path(summary["predictions_path"]).read_text().splitlines()[0])
    assert row["audit"]["frozen_evidence"]["nodes"][0]["node_type"] == ("changed_code")
    assert row["audit"]["frozen_evidence"]["nodes"][0]["cited"] is True
    assert "raw_blob" not in json.dumps(row["audit"]["frozen_evidence"])
    assert summary["metrics"]["frozen_evidence"]["node_type_counts"] == {
        "changed_code": 1
    }
    assert summary["metrics"]["frozen_evidence"]["cited_node_count"] == 1


def test_preservation_dry_run_binds_trusted_artifacts_and_revision_policy(
    tmp_path: Path,
) -> None:
    cohort, taxonomy, anchors = _write_preservation_inputs(tmp_path)

    summary = run_cli(
        [
            "--domain",
            "ase2022",
            "--stage",
            "stage3",
            "--cohort-path",
            str(cohort),
            "--taxonomy-path",
            str(taxonomy),
            "--output-dir",
            str(tmp_path / "out"),
            "--baseline-preservation",
            "--baseline-anchor-path",
            str(anchors),
            "--baseline-trust-manifest",
            str(_trust_path(anchors)),
            "--limit",
            "1",
            "--dry-run",
            "--no-progress",
        ],
        transport_override=lambda *args, **kwargs: pytest.fail(
            "dry-run must not call the model"
        ),
    )

    manifest = json.loads(Path(summary["manifest_path"]).read_text(encoding="utf-8"))
    preservation = manifest["baseline_preservation"]
    assert preservation["enabled"] is True
    trust = json.loads(_trust_path(anchors).read_text(encoding="utf-8"))
    assert preservation["source_config_hash"] == trust["source_config_hash"]
    assert (
        preservation["source_predictions_sha256"]
        == hashlib.sha256(anchors.read_bytes()).hexdigest()
    )
    assert len(preservation["selected_anchor_set_sha256"]) == 64
    assert len(preservation["anchor_digest_map_sha256"]) == 64
    assert len(preservation["taxonomy_structure_sha256"]) == 64
    assert len(preservation["bound_corpus_sidecar_sha256"]) == 64
    assert preservation["policy_version"] == "baseline-preservation-v1"
    assert preservation["official_boundary_generator_version"] == (
        "official-definition-pair-v1"
    )
    assert len(preservation["materialized_boundary_cards_sha256"]) == 64
    assert preservation["candidate_graph_policy"] == (
        "evidence-bound-candidate-signals-v2"
    )
    assert preservation["cross_check_graph_policy"] == (
        "dual-candidate-bound-cross-check-v1"
    )
    assert preservation["revision_assessment_policy"] == (
        "heterogeneous-entailment-falsification-v1"
    )
    assert preservation["revision_cross_policy"] == (
        "opposite-task-specific-raw-bound-v3"
    )
    assert set(preservation["revision_schema_hashes"]) == {
        "falsification_coverage_cross",
        "entailment_mapping_cross",
        "revision_entailment",
        "revision_falsification",
    }
    assert all(
        len(digest) == 64 for digest in preservation["revision_schema_hashes"].values()
    )
    first_id = next(csv.DictReader(cohort.open(encoding="utf-8")))["record_id"]
    assert set(preservation["record_anchor_digests"]) == {first_id}
    assert set(manifest["resolved_config"]["prompt_hashes"]) >= {
        "falsification_coverage_cross",
        "entailment_mapping_cross",
        "revision_entailment",
        "revision_falsification",
    }
    assert {
        key: manifest["resolved_config"]["prompt_hashes"][key]
        for key in (
            "falsification_coverage_cross",
            "entailment_mapping_cross",
            "revision_entailment",
            "revision_falsification",
        )
    } == baseline_revision_prompt_hashes()
    forbidden_keys = {"gt", "gold", "groundtruth", "restrictedgoldpath", "reasoningraw"}

    def keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return {
                "".join(
                    character for character in str(key).lower() if character.isalnum()
                )
                for key in value
            }.union(*(keys(item) for item in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(item) for item in value))
        return set()

    assert keys(manifest).isdisjoint(forbidden_keys)
    assert {
        "cohort_path",
        "taxonomy_path",
        "predictions_path",
        "metrics_path",
    }.isdisjoint(manifest)
    assert str(tmp_path) not in json.dumps(manifest)


def test_preservation_resume_rejects_legacy_candidate_graph_policy(
    tmp_path: Path,
) -> None:
    cohort, taxonomy, anchors = _write_preservation_inputs(tmp_path)
    common = [
        "--domain",
        "ase2022",
        "--stage",
        "stage3",
        "--cohort-path",
        str(cohort),
        "--taxonomy-path",
        str(taxonomy),
        "--output-dir",
        str(tmp_path / "out"),
        "--experiment-id",
        "preservation",
        "--split-id",
        "dev",
        "--arm-id",
        "a2",
        "--run-id",
        "run-1",
        "--baseline-preservation",
        "--baseline-anchor-path",
        str(anchors),
        "--baseline-trust-manifest",
        str(_trust_path(anchors)),
        "--limit",
        "1",
        "--dry-run",
        "--no-progress",
    ]
    first = run_cli(common)
    manifest_path = Path(first["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    legacy_config = manifest["resolved_config"]
    legacy_config["baseline_preservation"][
        "candidate_graph_policy"
    ] = "evidence-bound-candidate-signals-v1"
    manifest["config_hash"] = runner_script._config_hash(legacy_config)
    manifest["resolved_config"] = legacy_config
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="existing run configuration mismatch"):
        run_cli(
            common,
            transport_override=lambda *args, **kwargs: pytest.fail(
                "legacy policy resume must fail before model calls"
            ),
        )


def test_preservation_requires_active_split_before_model_calls(tmp_path: Path) -> None:
    cohort, taxonomy, anchors = _write_preservation_inputs(tmp_path)
    (cohort.parent / "split_manifest.json").unlink()

    with pytest.raises(ValueError, match="requires an active split manifest"):
        run_cli(
            [
                "--domain",
                "ase2022",
                "--stage",
                "stage3",
                "--cohort-path",
                str(cohort),
                "--taxonomy-path",
                str(taxonomy),
                "--baseline-preservation",
                "--baseline-anchor-path",
                str(anchors),
                "--baseline-trust-manifest",
                str(_trust_path(anchors)),
            ],
            transport_override=lambda *args, **kwargs: pytest.fail(
                "missing split must fail before model calls"
            ),
        )


def test_cli_rejects_self_signed_baseline_manifest_before_model_calls(
    tmp_path: Path,
) -> None:
    cohort, taxonomy, anchors = _write_preservation_inputs(tmp_path)
    self_signed = tmp_path / "self-signed-trust.json"
    self_signed.write_bytes(_trust_path(anchors).read_bytes())

    with pytest.raises(ValueError, match="pre-registered trust root"):
        run_cli(
            [
                "--domain",
                "ase2022",
                "--stage",
                "stage3",
                "--cohort-path",
                str(cohort),
                "--taxonomy-path",
                str(taxonomy),
                "--baseline-preservation",
                "--baseline-anchor-path",
                str(anchors),
                "--baseline-trust-manifest",
                str(self_signed),
            ],
            transport_override=lambda *args, **kwargs: pytest.fail(
                "self-signed trust must fail before model calls"
            ),
        )


def test_registered_dev50_split_and_trust_root_dry_run_without_model_calls(
    tmp_path: Path,
) -> None:
    cohort, taxonomy, anchors = _write_preservation_inputs(tmp_path)
    split_root = cohort.parent
    summary = run_cli(
        [
            "--domain",
            "ase2022",
            "--stage",
            "stage3",
            "--cohort-path",
            str(split_root / "development_runner_cohort.csv"),
            "--taxonomy-path",
            (
                "Benchmark/inputs/ase2022_issue_only_holdout_seed20260806/"
                "ase2022_issue_only_holdout_taxonomy.json"
            ),
            "--split-manifest",
            str(split_root / "split_manifest.json"),
            "--baseline-preservation",
            "--baseline-anchor-path",
            str(anchors),
            "--baseline-trust-manifest",
            str(_trust_path(anchors)),
            "--output-dir",
            str(tmp_path / "trusted-dev50-dry-run"),
            "--dry-run",
            "--no-resume",
            "--no-progress",
        ],
        transport_override=lambda *args, **kwargs: pytest.fail(
            "trusted dry-run must make zero model calls"
        ),
    )

    assert summary["record_count"] == 50
    manifest = json.loads(Path(summary["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["baseline_preservation"]["baseline_trust_artifact_id"] == (
        "synthetic-dev50-test-only"
    )
    assert (
        len(manifest["baseline_preservation"]["baseline_trust_manifest_sha256"]) == 64
    )
    assert str(tmp_path) not in json.dumps(manifest)


def test_preservation_flag_requires_anchor_before_any_transport_call(
    tmp_path: Path,
) -> None:
    cohort, taxonomy, _ = _write_preservation_inputs(tmp_path)

    with pytest.raises(ValueError, match="baseline-anchor"):
        run_cli(
            [
                "--domain",
                "ase2022",
                "--stage",
                "stage3",
                "--cohort-path",
                str(cohort),
                "--taxonomy-path",
                str(taxonomy),
                "--output-dir",
                str(tmp_path / "out"),
                "--baseline-preservation",
            ],
            transport_override=lambda *args, **kwargs: pytest.fail(
                "invalid preservation input must fail before model calls"
            ),
        )


def test_cli_passes_loader_owned_anchor_and_expected_digests_per_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cohort, taxonomy, anchors = _write_preservation_inputs(tmp_path)
    captured: list[dict[str, object]] = []

    def fake_run(record: dict[str, str], **kwargs: object) -> dict[str, object]:
        captured.append(kwargs)
        return {
            "record_id": record["record_id"],
            "stage2_prediction": None,
            "stage2_valid": False,
            "symptom_prediction": None,
            "root_cause_prediction": None,
            "stage3_valid": False,
            "stop_reason": "no_consistent_stage3_candidate",
            "audit": None,
        }

    monkeypatch.setattr(runner_script, "run_adaptive_record", fake_run)
    summary = run_cli(
        [
            "--domain",
            "ase2022",
            "--stage",
            "stage3",
            "--cohort-path",
            str(cohort),
            "--taxonomy-path",
            str(taxonomy),
            "--output-dir",
            str(tmp_path / "out"),
            "--baseline-preservation",
            "--baseline-anchor-path",
            str(anchors),
            "--baseline-trust-manifest",
            str(_trust_path(anchors)),
            "--limit",
            "1",
            "--no-progress",
            "--no-resume",
        ],
        transport_override=lambda *args, **kwargs: pytest.fail(
            "CLI wiring test replaces the record execution boundary"
        ),
    )

    manifest = json.loads(Path(summary["manifest_path"]).read_text(encoding="utf-8"))
    assert len(captured) == 1
    record_id = next(csv.DictReader(cohort.open(encoding="utf-8")))["record_id"]
    trust = json.loads(_trust_path(anchors).read_text(encoding="utf-8"))
    assert captured[0]["baseline_anchor"].record_id == record_id
    assert captured[0]["expected_baseline_config_hash"] == trust["source_config_hash"]
    assert (
        captured[0]["expected_baseline_predictions_sha256"]
        == hashlib.sha256(anchors.read_bytes()).hexdigest()
    )
    assert (
        captured[0]["expected_baseline_anchor_hash"]
        == manifest["baseline_preservation"]["record_anchor_digests"][record_id]
    )
    assert (
        captured[0]["expected_taxonomy_structure_hash"]
        == manifest["baseline_preservation"]["taxonomy_structure_sha256"]
    )


@pytest.mark.parametrize(
    "malformation",
    ("duplicate", "missing", "invalid_label", "mixed_config"),
)
def test_cli_rejects_untrusted_anchor_artifacts_before_model_calls(
    tmp_path: Path,
    malformation: str,
) -> None:
    cohort, taxonomy, anchors = _write_preservation_inputs(tmp_path)
    valid = json.loads(anchors.read_text(encoding="utf-8").splitlines()[0])
    if malformation == "duplicate":
        rows = [valid, valid]
    elif malformation == "missing":
        rows = [{**valid, "record_id": "another-record"}]
    elif malformation == "invalid_label":
        rows = [
            {
                **valid,
                "final_prediction": {
                    "symptom": "Not An Official Label",
                    "root_cause": "Incorrect Code Logic",
                },
            }
        ]
    else:
        rows = [valid, {**valid, "record_id": "extra", "config_hash": "b" * 64}]
    untrusted = tmp_path / "modified-baseline.jsonl"
    untrusted.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        run_cli(
            [
                "--domain",
                "ase2022",
                "--stage",
                "stage3",
                "--cohort-path",
                str(cohort),
                "--taxonomy-path",
                str(taxonomy),
                "--output-dir",
                str(tmp_path / "out"),
                "--baseline-preservation",
                "--baseline-anchor-path",
                str(untrusted),
                "--baseline-trust-manifest",
                str(_trust_path(anchors)),
            ],
            transport_override=lambda *args, **kwargs: pytest.fail(
                "untrusted anchors must fail before model calls"
            ),
        )


def test_registered_trust_rejects_copied_changed_anchor_before_model_calls(
    tmp_path: Path,
) -> None:
    cohort, taxonomy, anchors = _write_preservation_inputs(tmp_path)
    changed = tmp_path / "changed-anchor.jsonl"
    changed.write_text(anchors.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    common = [
        "--domain",
        "ase2022",
        "--stage",
        "stage3",
        "--cohort-path",
        str(cohort),
        "--taxonomy-path",
        str(taxonomy),
        "--output-dir",
        str(tmp_path / "out"),
        "--experiment-id",
        "preservation",
        "--split-id",
        "dev",
        "--arm-id",
        "a2",
        "--run-id",
        "run-1",
        "--baseline-preservation",
        "--baseline-anchor-path",
        str(changed),
        "--baseline-trust-manifest",
        str(_trust_path(anchors)),
        "--no-progress",
    ]
    with pytest.raises(ValueError, match="artifact path"):
        run_cli(
            common,
            transport_override=lambda *args, **kwargs: pytest.fail(
                "resume mismatch must fail before model calls"
            ),
        )


def test_code_state_hash_changes_when_dirty_content_changes(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    source = tmp_path / "Benchmark" / "src" / "module.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=tmp_path, check=True)

    source.write_text("VALUE = 2\n", encoding="utf-8")
    first = _code_state(tmp_path)
    source.write_text("VALUE = 3\n", encoding="utf-8")
    second = _code_state(tmp_path)

    assert first["dirty_source_state_sha256"] != second["dirty_source_state_sha256"]


def _write_stage3_selection_inputs(
    tmp_path: Path,
    *,
    decisions: tuple[str, ...] = ("rejected_candidate", "accepted_fault"),
) -> tuple[Path, Path]:
    cohort, taxonomy = _write_inputs(tmp_path)
    with cohort.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "record_id",
                "title",
                "body",
                "code_diff",
                "decision",
                "symptom",
                "root_cause",
            ],
        )
        writer.writeheader()
        for index, decision in enumerate(decisions, start=1):
            writer.writerow(
                {
                    "record_id": f"record-{index}",
                    "title": "Request rejected",
                    "body": "A valid request is rejected.",
                    "code_diff": "- reject(request)\n+ process(request)",
                    "decision": decision,
                    "symptom": "Crash" if decision == "accepted_fault" else "",
                    "root_cause": (
                        "Incorrect Code Logic" if decision == "accepted_fault" else ""
                    ),
                }
            )
    return cohort, taxonomy


def test_incomplete_http_response_is_retryable() -> None:
    assert _is_retryable_network_error(http.client.IncompleteRead(b"partial"))


def test_adaptive_transport_preserves_provider_usage() -> None:
    class ProviderText(str):
        prompt_tokens = 123
        completion_tokens = 45
        reasoning_characters = 678

    response = runner_script._model_transport_response(
        ProviderText('{"ok":true}'),
        attempts=3,
    )

    assert response == '{"ok":true}'
    assert response.network_attempts == 3
    assert response.prompt_tokens == 123
    assert response.completion_tokens == 45
    assert response.reasoning_characters == 678


def test_default_transport_fails_closed_for_optional_http_400_without_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cohort, taxonomy = _write_inputs(tmp_path)
    provider_calls = 0
    observed: ModelTransportError | None = None

    def fake_retry_call(**kwargs: object) -> tuple[str, int]:
        nonlocal provider_calls
        provider_calls += 1
        raise urllib.error.HTTPError(
            "https://example.invalid",
            400,
            "bad request",
            hdrs=None,
            fp=None,
        )

    def fake_record_runner(
        record: dict[str, str],
        *,
        agents: object,
        **kwargs: object,
    ) -> dict[str, object]:
        nonlocal observed
        with pytest.raises(ModelTransportError) as raised:
            agents._client.complete(
                system_prompt="ROLE: baseline_revision_consistency",
                user_prompt="Return JSON.",
                schema=RevisionConsistencyReport,
                options=ModelCallOptions(
                    role="baseline_revision_consistency",
                    team_id="A",
                    perspective="evidence_first",
                    max_tokens=32768,
                    thinking_enabled=None,
                    share_transport_schema_budget=True,
                ),
            )
        observed = raised.value
        return {
            "record_id": record["record_id"],
            "stage2_prediction": None,
            "stage2_valid": False,
            "symptom_prediction": "Crash",
            "root_cause_prediction": "Incorrect Code Logic",
            "stage3_valid": True,
            "stop_reason": "stage3_only",
            "audit": None,
        }

    monkeypatch.setattr(
        runner_script,
        "resolve_run_config",
        lambda *args, **kwargs: {
            "api_key": "not-a-real-key",
            "base_url": "https://example.invalid",
        },
    )
    monkeypatch.setattr(runner_script, "call_model_with_retries", fake_retry_call)
    monkeypatch.setattr(runner_script, "run_adaptive_record", fake_record_runner)

    run_cli(
        [
            "--domain",
            "ase2022",
            "--stage",
            "stage3",
            "--cohort-path",
            str(cohort),
            "--taxonomy-path",
            str(taxonomy),
            "--output-dir",
            str(tmp_path / "transport-400-output"),
            "--max-schema-retries",
            "5",
            "--no-progress",
            "--no-resume",
        ]
    )

    assert provider_calls == 1
    assert observed is not None
    assert observed.retryable is False
    assert observed.cause_type == "http_error"
    assert observed.http_status == 400


@pytest.mark.parametrize(
    ("retry_after", "expected"),
    (("0.5", 0.5), ("NaN", None), ("inf", None), ("-1", None)),
)
def test_http_retry_after_parser_rejects_nonfinite_or_negative_values(
    retry_after: str,
    expected: float | None,
) -> None:
    headers = Message()
    headers["Retry-After"] = retry_after
    error = urllib.error.HTTPError(
        "https://example.invalid",
        429,
        "rate limited",
        hdrs=headers,
        fp=None,
    )

    assert model_transport_error_details(error)["retry_after_seconds"] == expected


@pytest.mark.parametrize("keyword_only", (True, False))
def test_run_cli_transport_override_preserves_expanded_option_signatures(
    tmp_path: Path,
    keyword_only: bool,
) -> None:
    cohort, taxonomy = _write_inputs(tmp_path)
    received: list[tuple[int, bool | None]] = []
    valid = json.dumps(
        {
            "dimension": "symptom",
            "baseline_label": "Crash",
            "proposed_label": "Incorrect Functionality",
            "status": "consistent",
            "rationale": "The exact evidence supports this replacement boundary.",
            "supporting_evidence_ids": ["evidence-1"],
            "assessment_digest": "e" * 64,
            "boundary_card_id": "symptom-boundary",
            "baseline_source_config_hash": "a" * 64,
            "baseline_source_predictions_sha256": "b" * 64,
            "taxonomy_structure_hash": "c" * 64,
            "evidence_view_hash": "d" * 64,
        }
    )

    if keyword_only:

        def transport(
            system: str,
            user: str,
            *,
            max_tokens: int,
            thinking_enabled: bool | None,
        ) -> str:
            received.append((max_tokens, thinking_enabled))
            return valid

    else:

        def transport(
            system: str,
            user: str,
            max_tokens: int,
            thinking_enabled: bool | None,
        ) -> str:
            received.append((max_tokens, thinking_enabled))
            return valid

    def record_runner(
        record: dict[str, str],
        *,
        agents: object,
        **kwargs: object,
    ) -> dict[str, object]:
        result = agents._client.complete(
            system_prompt="ROLE: baseline_revision_consistency",
            user_prompt="Return JSON.",
            schema=RevisionConsistencyReport,
            options=ModelCallOptions(
                role="baseline_revision_consistency",
                team_id="A",
                perspective="evidence_first",
                max_tokens=321,
                thinking_enabled=False,
                share_transport_schema_budget=True,
            ),
        )
        assert result.status.value == "consistent"
        return {
            "record_id": record["record_id"],
            "stage2_prediction": None,
            "stage2_valid": False,
            "symptom_prediction": "Crash",
            "root_cause_prediction": "Incorrect Code Logic",
            "stage3_valid": True,
            "stop_reason": "stage3_only",
            "audit": None,
        }

    original_record_runner = runner_script.run_adaptive_record
    runner_script.run_adaptive_record = record_runner
    try:
        run_cli(
            [
                "--domain",
                "ase2022",
                "--stage",
                "stage3",
                "--cohort-path",
                str(cohort),
                "--taxonomy-path",
                str(taxonomy),
                "--output-dir",
                str(tmp_path / f"expanded-options-{keyword_only}"),
                "--no-progress",
                "--no-resume",
            ],
            transport_override=transport,
        )
    finally:
        runner_script.run_adaptive_record = original_record_runner

    assert received == [(321, False)]


def test_default_runner_transport_drops_sensitive_exception_chain_and_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cohort, taxonomy = _write_inputs(tmp_path)
    secret = "SECRET_PROVIDER_URL_BODY_PROMPT_KEY"
    captured: ModelTransportError | None = None
    captured_telemetry: dict[str, object] | None = None

    def fake_retry_call(**kwargs: object) -> tuple[str, int]:
        raise urllib.error.HTTPError(
            f"https://example.invalid/{secret}",
            400,
            secret,
            hdrs=None,
            fp=None,
        )

    def capture_runner(
        record: dict[str, str],
        *,
        agents: object,
        **kwargs: object,
    ) -> dict[str, object]:
        nonlocal captured, captured_telemetry
        with pytest.raises(ModelTransportError) as raised:
            agents._client.complete(
                system_prompt="ROLE: baseline_revision_consistency",
                user_prompt=secret,
                schema=RevisionConsistencyReport,
                options=ModelCallOptions(
                    role="baseline_revision_consistency",
                    team_id="A",
                    perspective="evidence_first",
                    max_tokens=321,
                    thinking_enabled=False,
                    share_transport_schema_budget=True,
                ),
            )
        captured = raised.value
        captured_telemetry = agents._client.telemetry()
        return {
            "record_id": record["record_id"],
            "stage2_prediction": None,
            "stage2_valid": False,
            "symptom_prediction": "Crash",
            "root_cause_prediction": "Incorrect Code Logic",
            "stage3_valid": True,
            "stop_reason": "stage3_only",
            "audit": None,
        }

    monkeypatch.setattr(
        runner_script,
        "resolve_run_config",
        lambda *args, **kwargs: {
            "api_key": secret,
            "base_url": f"https://example.invalid/{secret}",
        },
    )
    monkeypatch.setattr(runner_script, "call_model_with_retries", fake_retry_call)
    monkeypatch.setattr(runner_script, "run_adaptive_record", capture_runner)

    summary = run_cli(
        [
            "--domain",
            "ase2022",
            "--stage",
            "stage3",
            "--cohort-path",
            str(cohort),
            "--taxonomy-path",
            str(taxonomy),
            "--output-dir",
            str(tmp_path / "secret-chain-output"),
            "--no-progress",
            "--no-resume",
        ]
    )

    assert captured is not None
    assert captured.__cause__ is None
    assert captured.__context__ is None
    assert secret not in repr(captured)
    assert captured_telemetry is not None
    assert secret not in json.dumps(captured_telemetry)
    assert secret not in Path(summary["predictions_path"]).read_text(encoding="utf-8")


def test_consistency_provider_admission_is_global_across_records(
    tmp_path: Path,
) -> None:
    cohort, taxonomy = _write_stage3_selection_inputs(tmp_path)
    active = 0
    max_active = 0
    lock = threading.Lock()
    record_barrier = threading.Barrier(2)
    first_call_entered = threading.Event()
    release_first_call = threading.Event()
    completed_crosses: set[tuple[str, str]] = set()
    valid = json.dumps(
        {
            "dimension": "symptom",
            "baseline_label": "Crash",
            "proposed_label": "Incorrect Functionality",
            "status": "consistent",
            "rationale": "The exact evidence supports this replacement boundary.",
            "supporting_evidence_ids": ["evidence-1"],
            "assessment_digest": "e" * 64,
            "boundary_card_id": "symptom-boundary",
            "baseline_source_config_hash": "a" * 64,
            "baseline_source_predictions_sha256": "b" * 64,
            "taxonomy_structure_hash": "c" * 64,
            "evidence_view_hash": "d" * 64,
        }
    )

    def transport(
        system: str,
        user: str,
        *,
        options: ModelCallOptions,
    ) -> str:
        nonlocal active, max_active
        assert options.role == "baseline_revision_consistency"
        with lock:
            active += 1
            max_active = max(max_active, active)
            concurrent = active > 1
        if concurrent:
            with lock:
                active -= 1
            raise ModelTransportError(
                "admission exceeded",
                attempts=1,
                retryable=False,
                cause_type="http_error",
                http_status=429,
            )
        first_call_entered.set()
        release_first_call.wait(timeout=0.2)
        with lock:
            active -= 1
        return valid

    def record_runner(
        record: dict[str, str],
        *,
        agents: object,
        **kwargs: object,
    ) -> dict[str, object]:
        record_barrier.wait(timeout=2)
        for team_id in ("A", "B"):
            report = agents._client.complete(
                system_prompt="ROLE: baseline_revision_consistency",
                user_prompt="Return JSON.",
                schema=RevisionConsistencyReport,
                options=ModelCallOptions(
                    role="baseline_revision_consistency",
                    team_id=team_id,
                    perspective="evidence_first",
                    max_tokens=32768,
                    thinking_enabled=None,
                    share_transport_schema_budget=True,
                ),
            )
            assert report.status.value == "consistent"
            completed_crosses.add((record["record_id"], team_id))
        return {
            "record_id": record["record_id"],
            "stage2_prediction": None,
            "stage2_valid": False,
            "symptom_prediction": "Crash",
            "root_cause_prediction": "Incorrect Code Logic",
            "stage3_valid": True,
            "stop_reason": "stage3_only",
            "audit": None,
        }

    original_record_runner = runner_script.run_adaptive_record
    runner_script.run_adaptive_record = record_runner
    try:
        summary = run_cli(
            [
                "--domain",
                "ase2022",
                "--stage",
                "all",
                "--cohort-path",
                str(cohort),
                "--taxonomy-path",
                str(taxonomy),
                "--output-dir",
                str(tmp_path / "consistency-admission-output"),
                "--concurrency",
                "2",
                "--no-progress",
                "--no-resume",
            ],
            transport_override=transport,
        )
    finally:
        runner_script.run_adaptive_record = original_record_runner
        release_first_call.set()

    assert first_call_entered.is_set()
    assert max_active == 1
    assert completed_crosses == {
        ("record-1", "A"),
        ("record-1", "B"),
        ("record-2", "A"),
        ("record-2", "B"),
    }
    assert all(
        json.loads(line)["stage3_valid"] is True
        for line in Path(summary["predictions_path"])
        .read_text(encoding="utf-8")
        .splitlines()
    )


@pytest.mark.parametrize(
    ("role", "schema_attempt", "shares_budget", "expected_retries"),
    (
        ("baseline_revision_assessment", 1, True, 0),
        ("baseline_revision_assessment", 2, True, 0),
        ("baseline_revision_consistency", 1, True, 0),
        ("baseline_revision_consistency", 2, True, 0),
        ("joint_anchor", 2, False, 5),
        ("symptom_verifier", 2, False, 5),
        ("root_cause_verifier", 2, False, 5),
        ("stage3_arbitrator", 2, False, 5),
    ),
)
def test_network_retry_boundary_is_bounded_only_for_optional_revision_roles(
    role: str,
    schema_attempt: int,
    shares_budget: bool,
    expected_retries: int,
) -> None:
    options = ModelCallOptions(
        role=role,
        team_id="A",
        perspective="evidence_first",
        max_tokens=32768,
        thinking_enabled=True,
        schema_attempt=schema_attempt,
        share_transport_schema_budget=shares_budget,
    )

    retry_limit = getattr(
        runner_script,
        "_network_retry_limit",
        lambda *, options, configured_max_retries: configured_max_retries,
    )

    assert retry_limit(options=options, configured_max_retries=5) == expected_retries


def test_revision_schema_repairs_cannot_nest_full_network_retry_budget() -> None:
    options = ModelCallOptions(
        role="baseline_revision_consistency",
        team_id="A",
        perspective="evidence_first",
        max_tokens=32768,
        thinking_enabled=None,
        share_transport_schema_budget=True,
    )
    retry_limit = getattr(
        runner_script,
        "_network_retry_limit",
        lambda *, options, configured_max_retries: configured_max_retries,
    )
    responses = iter(
        (
            '{"invalid":true}',
            '{"still_invalid":true}',
            json.dumps(
                {
                    "dimension": "symptom",
                    "baseline_label": "Crash",
                    "proposed_label": "Incorrect Functionality",
                    "status": "consistent",
                    "rationale": "The exact evidence supports this replacement boundary.",
                    "supporting_evidence_ids": ["evidence-1"],
                    "assessment_digest": "e" * 64,
                    "boundary_card_id": "symptom-boundary",
                    "baseline_source_config_hash": "a" * 64,
                    "baseline_source_predictions_sha256": "b" * 64,
                    "taxonomy_structure_hash": "c" * 64,
                    "evidence_view_hash": "d" * 64,
                }
            ),
        )
    )

    def transport(
        system: str,
        user: str,
        *,
        options: ModelCallOptions,
    ) -> ModelTransportResponse:
        retries = retry_limit(options=options, configured_max_retries=5)
        return ModelTransportResponse(next(responses), network_attempts=retries + 1)

    client = StructuredModelClient(transport, max_schema_retries=2)
    result = client.complete(
        system_prompt="Return JSON.",
        user_prompt="Assess the revision.",
        schema=RevisionConsistencyReport,
        options=options,
    )

    assert result.status.value == "consistent"
    assert client.telemetry()["calls"][0]["attempts"] == 3
    assert client.telemetry()["calls"][0]["network_attempts"] == 3


def test_run_cli_default_transport_applies_retry_limit_per_schema_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cohort, taxonomy = _write_inputs(tmp_path)
    observed: dict[str, list[int]] = {
        "joint_anchor": [],
        "baseline_revision_consistency": [],
    }
    valid = json.dumps(
        {
            "dimension": "symptom",
            "baseline_label": "Crash",
            "proposed_label": "Incorrect Functionality",
            "status": "consistent",
            "rationale": "The exact evidence supports this replacement boundary.",
            "supporting_evidence_ids": ["evidence-1"],
            "assessment_digest": "e" * 64,
            "boundary_card_id": "symptom-boundary",
            "baseline_source_config_hash": "a" * 64,
            "baseline_source_predictions_sha256": "b" * 64,
            "taxonomy_structure_hash": "c" * 64,
            "evidence_view_hash": "d" * 64,
        }
    )
    replies = {
        "joint_anchor": iter(('{"invalid":true}', valid)),
        "baseline_revision_consistency": iter(
            (
                '{"invalid":true}',
                urllib.error.HTTPError(
                    "https://example.invalid",
                    429,
                    "rate limited",
                    hdrs=None,
                    fp=None,
                ),
                valid,
            )
        ),
    }

    def fake_retry_call(*, max_retries: int, system_prompt: str, **kwargs: object):
        role = system_prompt.removeprefix("ROLE: ")
        observed[role].append(max_retries)
        reply = next(replies[role])
        if isinstance(reply, Exception):
            raise reply
        return reply, 1

    def fake_record_runner(
        record: dict[str, str],
        *,
        agents: object,
        **kwargs: object,
    ) -> dict[str, object]:
        client = agents._client
        for role in ("joint_anchor", "baseline_revision_consistency"):
            client.complete(
                system_prompt=f"ROLE: {role}",
                user_prompt="Return the requested JSON object.",
                schema=RevisionConsistencyReport,
                options=ModelCallOptions(
                    role=role,
                    team_id="A",
                    perspective="evidence_first",
                    max_tokens=32768,
                    thinking_enabled=True,
                    share_transport_schema_budget=(
                        role == "baseline_revision_consistency"
                    ),
                ),
            )
        return {
            "record_id": record["record_id"],
            "stage2_prediction": None,
            "stage2_valid": False,
            "symptom_prediction": "Crash",
            "root_cause_prediction": "Incorrect Code Logic",
            "stage3_valid": True,
            "stop_reason": "stage3_only",
            "audit": None,
        }

    monkeypatch.setattr(
        runner_script,
        "resolve_run_config",
        lambda *args, **kwargs: {
            "api_key": "not-a-real-key",
            "base_url": "https://example.invalid",
        },
    )
    monkeypatch.setattr(runner_script, "call_model_with_retries", fake_retry_call)
    monkeypatch.setattr(runner_script, "run_adaptive_record", fake_record_runner)

    run_cli(
        [
            "--domain",
            "ase2022",
            "--stage",
            "stage3",
            "--cohort-path",
            str(cohort),
            "--taxonomy-path",
            str(taxonomy),
            "--output-dir",
            str(tmp_path / "runner-transport-output"),
            "--max-schema-retries",
            "2",
            "--max-network-retries",
            "5",
            "--retry-delay-seconds",
            "0",
            "--no-progress",
            "--no-resume",
        ]
    )

    assert observed == {
        "joint_anchor": [5, 5],
        "baseline_revision_consistency": [0, 0, 0],
    }


def test_resume_telemetry_accumulates_reasoning_character_counts() -> None:
    def telemetry(reasoning_characters: int) -> dict[str, object]:
        return {
            "total_calls": 0,
            "total_attempts": 0,
            "total_latency_seconds": 0.0,
            "estimated_prompt_tokens": 0,
            "estimated_completion_tokens": 0,
            "actual_prompt_tokens": 0,
            "actual_completion_tokens": 0,
            "reasoning_characters": reasoning_characters,
            "calls": [],
        }

    merged = runner_script._merge_llm_telemetry(
        telemetry(120),
        telemetry(34),
    )

    assert merged["reasoning_characters"] == 154


def test_resume_telemetry_keeps_schema_and_transport_retries_distinct() -> None:
    previous = {
        "total_calls": 1,
        "total_attempts": 2,
        "total_latency_seconds": 1.0,
        "estimated_prompt_tokens": 10,
        "estimated_completion_tokens": 2,
        "actual_prompt_tokens": None,
        "actual_completion_tokens": None,
        "reasoning_characters": 0,
        "calls": [
            {
                "status": "valid",
                "attempts": 2,
                "network_attempts": 2,
                "validation_errors": [{"error_type": "ValidationError"}],
            }
        ],
    }
    current = {
        "total_calls": 1,
        "total_attempts": 2,
        "total_latency_seconds": 1.0,
        "estimated_prompt_tokens": 10,
        "estimated_completion_tokens": 2,
        "actual_prompt_tokens": None,
        "actual_completion_tokens": None,
        "reasoning_characters": 0,
        "calls": [
            {
                "status": "valid",
                "attempts": 2,
                "network_attempts": 2,
                "validation_errors": [],
                "schema_validation_failure_count": 0,
                "transport_error_count": 1,
                "transport_retry_count": 1,
            }
        ],
    }

    merged = runner_script._merge_llm_telemetry(previous, current)

    assert merged["schema_retry_calls"] == 1
    assert merged["schema_validation_failure_count"] == 1
    assert merged["transport_retry_calls"] == 1
    assert merged["transport_retry_count"] == 1
    assert merged["transport_error_count"] == 1


def test_preservation_compute_metrics_isolate_new_revision_role_cost() -> None:
    telemetry = {
        "calls": [
            {
                "role": "baseline_revision_assessment",
                "attempts": 2,
                "latency_seconds": 1.25,
                "estimated_prompt_tokens": 100,
                "estimated_completion_tokens": 20,
                "actual_prompt_tokens": 90,
                "actual_completion_tokens": 18,
                "schema_validation_failure_count": 1,
                "transport_error_count": 1,
                "transport_retry_count": 1,
            },
            {
                "role": "baseline_revision_consistency",
                "attempts": 1,
                "latency_seconds": 0.75,
                "estimated_prompt_tokens": 80,
                "estimated_completion_tokens": 10,
                "actual_prompt_tokens": None,
                "actual_completion_tokens": None,
            },
            {
                "role": "joint_anchor",
                "attempts": 1,
                "latency_seconds": 9.0,
                "estimated_prompt_tokens": 999,
                "estimated_completion_tokens": 999,
                "actual_prompt_tokens": 999,
                "actual_completion_tokens": 999,
            },
        ]
    }

    assert runner_script._preservation_compute_metrics(telemetry) == {
        "assessment_calls": 1,
        "consistency_calls": 1,
        "total_calls": 2,
        "total_attempts": 3,
        "total_latency_seconds": 2.0,
        "estimated_prompt_tokens": 180,
        "estimated_completion_tokens": 30,
        "actual_prompt_tokens": 90,
        "actual_completion_tokens": 18,
        "schema_validation_failure_count": 1,
        "transport_error_count": 1,
        "transport_retry_count": 1,
    }


def test_preservation_fake_transport_runs_revision_certificate_call_graph(
    tmp_path: Path,
) -> None:
    cohort, taxonomy, anchors = _write_preservation_inputs(
        tmp_path, baseline_symptom="Incorrect Functionality"
    )
    calls: list[tuple[str, str | None]] = []
    user_payloads: list[dict[str, object]] = []

    def transport(
        system: str,
        user: str,
        *,
        options: ModelCallOptions,
    ) -> str:
        payload = json.loads(user)
        user_payloads.append(payload)
        calls.append((options.role, options.team_id))
        view_payload = payload.get("evidence_view", payload.get("evidence_snapshot"))
        assert isinstance(view_payload, dict)
        items = view_payload["items"]
        symptom_id = next(
            item["evidence_id"] for item in items if item["source_type"] != "code_diff"
        )
        root_id = items[-1]["evidence_id"]
        if options.role == "evidence_readiness":
            return json.dumps(
                {
                    "task": "stage3",
                    "dimensions": [
                        {
                            "dimension": "symptom",
                            "sufficient": True,
                            "confirmed_evidence_ids": [symptom_id],
                            "missing_facts": [],
                            "evidence_requests": [],
                        },
                        {
                            "dimension": "root_cause",
                            "sufficient": True,
                            "confirmed_evidence_ids": [root_id],
                            "missing_facts": [],
                            "evidence_requests": [],
                        },
                    ],
                }
            )
        if options.role == "joint_anchor":
            return json.dumps(
                {
                    "symptom": {
                        "label": "Crash",
                        "behavior_claim": "The request terminates with a runtime exception.",
                        "supporting_evidence_ids": [symptom_id],
                        "counter_evidence_ids": [],
                        "alternative_label": "Incorrect Functionality",
                        "boundary_reason": "Termination is primary, not a continued wrong result.",
                        "confidence": 0.9,
                        "evidence_sufficiency": "sufficient",
                    },
                    "root_cause": {
                        "label": "Incorrect Code Logic",
                        "defect_mechanism": "An incorrect internal branch routes to failure.",
                        "causal_chain": [
                            "The request enters the branch.",
                            "The incorrect branch routes to failure.",
                            "The request terminates.",
                        ],
                        "supporting_evidence_ids": [root_id],
                        "counter_evidence_ids": [],
                        "alternative_label": "API Misuse",
                        "boundary_reason": "The defect is internal rather than caller misuse.",
                        "confidence": 0.9,
                        "evidence_sufficiency": "sufficient",
                    },
                    "causal_account": "The incorrect internal branch directly causes termination.",
                    "shared_supporting_evidence_ids": [symptom_id, root_id],
                }
            )
        if options.role in {"symptom_verifier", "root_cause_verifier"}:
            dimension = payload["context"]["owned_dimension"]
            anchor = payload["context"]["anchor"]
            owned = anchor["symptom" if dimension == "symptom" else "root_cause"]
            return json.dumps(
                {
                    "dimension": dimension,
                    "verdict": "accept",
                    "anchor_label": owned["label"],
                    "alternative_label": None,
                    "rationale": "The exact frozen evidence independently supports this label.",
                    "supporting_evidence_ids": owned["supporting_evidence_ids"],
                    "counter_evidence_ids": [],
                    "confidence": 0.88,
                }
            )
        if options.role == "causal_consistency_checker":
            return json.dumps(
                {
                    "status": "consistent",
                    "rationale": "The incorrect branch directly explains the observed termination.",
                    "supporting_evidence_ids": [root_id],
                    "evidence_requests": [],
                }
            )
        if options.role == "boundary_challenger":
            return json.dumps(
                {
                    "action": "pass",
                    "rationale": "The labels satisfy the supplied taxonomy boundaries.",
                    "cited_evidence_ids": [symptom_id, root_id],
                }
            )
        if options.role == "stage3_arbitrator":
            return json.dumps(
                {
                    "resolution_status": "resolved",
                    "symptom_label": "Crash",
                    "root_cause_label": "Incorrect Code Logic",
                    "confidence": 0.88,
                    "rationale": "Both isolated teams and the exact evidence support this pair.",
                    "supporting_evidence_ids": [symptom_id, root_id],
                    "resolved_dimensions": payload["disagreement"]["dimensions"],
                }
            )
        if options.role == "baseline_revision_assessment":
            assert options.max_tokens == 2200
            assert options.thinking_enabled is False
            context = payload["context"]
            proposal = context["proposal"]
            assert proposal["dimension"] == "symptom"
            citation = proposal["supporting_evidence_ids"][0]
            if options.team_id == "A":
                required = context["required_findings"]
                return json.dumps(
                    {
                        "condition_findings": [
                            {
                                "condition": condition,
                                "status": "supported",
                                "citation_ids": [citation],
                            }
                            for condition in required["proposed_positive_conditions"]
                        ],
                        "baseline_exclusion_finding": {
                            "condition": required["baseline_exclusion_condition"],
                            "status": "supported",
                            "citation_ids": [citation],
                        },
                        "outcome": "entailed",
                        "summary": "The direct evidence entails every fixed candidate condition.",
                    }
                )
            return json.dumps(
                {
                    "baseline_survival_finding": {
                        "condition": context["required_findings"][
                            "baseline_survival_condition"
                        ],
                        "status": "refuted",
                        "citation_ids": [citation],
                    },
                    "proposed_defeater_findings": [
                        {
                            "condition": condition,
                            "status": "refuted",
                            "citation_ids": [citation],
                        }
                        for condition in context["required_findings"][
                            "proposed_defeater_conditions"
                        ]
                    ],
                    "strongest_competing_reading": {
                        "label": context["required_findings"][
                            "strongest_competing_reading_label"
                        ],
                        "summary": "The Baseline is the strongest exact-pair competing reading.",
                        "citation_ids": [citation],
                    },
                    "outcome": "revision_survives",
                    "summary": "The revision survives every fixed adversarial boundary check.",
                }
            )
        if options.role == "baseline_revision_consistency":
            assert options.max_tokens == 1800
            assert options.thinking_enabled is False
            context = payload["context"]
            cross_input = context["cross_check_input"]
            raw_assessment = cross_input["raw_assessment"]
            if cross_input["assessment_owner_task"] == "revision_entailment":
                return json.dumps(
                    {
                        "condition_findings": [
                            {
                                "condition": finding["condition"],
                                "status": "confirmed",
                                "citation_ids": finding["citation_ids"],
                            }
                            for finding in raw_assessment["condition_findings"]
                        ],
                        "baseline_exclusion_finding": {
                            "condition": raw_assessment["baseline_exclusion_finding"][
                                "condition"
                            ],
                            "status": "confirmed",
                            "citation_ids": raw_assessment[
                                "baseline_exclusion_finding"
                            ]["citation_ids"],
                        },
                        "status": "consistent",
                        "rationale": "Every exact entailment mapping is confirmed from owner citations.",
                    }
                )
            return json.dumps(
                {
                    "baseline_survival_finding": {
                        "condition": raw_assessment["baseline_survival_finding"][
                            "condition"
                        ],
                        "status": "confirmed",
                        "citation_ids": raw_assessment["baseline_survival_finding"][
                            "citation_ids"
                        ],
                    },
                    "proposed_defeater_findings": [
                        {
                            "condition": finding["condition"],
                            "status": "confirmed",
                            "citation_ids": finding["citation_ids"],
                        }
                        for finding in raw_assessment["proposed_defeater_findings"]
                    ],
                    "strongest_competing_reading_finding": {
                        "label": raw_assessment["strongest_competing_reading"]["label"],
                        "status": "confirmed",
                        "citation_ids": raw_assessment["strongest_competing_reading"][
                            "citation_ids"
                        ],
                    },
                    "status": "consistent",
                    "rationale": "Every exact falsification finding is confirmed from owner citations.",
                }
            )
        raise AssertionError(f"unexpected role: {options.role}\n{system}")

    summary = run_cli(
        [
            "--domain",
            "ase2022",
            "--stage",
            "stage3",
            "--cohort-path",
            str(cohort),
            "--taxonomy-path",
            str(taxonomy),
            "--output-dir",
            str(tmp_path / "preservation-output"),
            "--baseline-preservation",
            "--baseline-anchor-path",
            str(anchors),
            "--baseline-trust-manifest",
            str(_trust_path(anchors)),
            "--record-ids",
            "ase2022_towards_understanding_the_faults_of:698e88d7ae56b0e1",
            "--no-progress",
        ],
        transport_override=transport,
    )

    role_counts = {
        role: sum(call_role == role for call_role, _ in calls)
        for role in {call_role for call_role, _ in calls}
    }
    assert role_counts["baseline_revision_assessment"] == 2
    assert role_counts["baseline_revision_consistency"] == 2
    diagnostics = summary["metrics"]["stage3"]["baseline_preservation"]
    assert summary["metrics"]["evaluation"]["status"] == "not_evaluated"
    assert "joint_accuracy" not in summary["metrics"]["stage3"]
    assert "paired" not in diagnostics
    assert diagnostics["actions"]["symptom"]["revised"] == 1
    assert diagnostics["actions"]["root_cause"]["preserved"] == 1
    assert diagnostics["certificates"]["issued_count"] == 1
    assert diagnostics["compute"]["assessment_calls"] == 2
    assert diagnostics["compute"]["consistency_calls"] == 2
    prediction = json.loads(
        Path(summary["predictions_path"]).read_text(encoding="utf-8").splitlines()[0]
    )
    assert prediction["symptom_prediction"] == "Crash"
    assert prediction["root_cause_prediction"] == "Incorrect Code Logic"

    def normalized_keys(value: object) -> set[str]:
        found: set[str] = set()
        if isinstance(value, dict):
            for key, nested in value.items():
                found.add(
                    "".join(
                        character
                        for character in str(key).lower()
                        if character.isalnum()
                    )
                )
                found.update(normalized_keys(nested))
        elif isinstance(value, list):
            for nested in value:
                found.update(normalized_keys(nested))
        return found

    all_keys = set().union(*(normalized_keys(payload) for payload in user_payloads))
    assert not all_keys.intersection({"gt", "gold", "groundtruth", "expectedanswer"})


def test_cli_accepts_exactly_the_three_supported_stage_modes() -> None:
    parser = build_parser()

    for stage in ("stage2", "stage3", "all"):
        assert parser.parse_args(["--domain", "ase2022", "--stage", stage]).stage == (
            stage
        )
    with pytest.raises(SystemExit):
        parser.parse_args(["--domain", "ase2022", "--stage", "unknown"])


def test_stage3_cli_oracle_selects_only_gold_positive_records_and_isolates_resume(
    tmp_path: Path,
) -> None:
    cohort, taxonomy = _write_stage3_selection_inputs(tmp_path)
    output = tmp_path / "output"
    common = [
        "--domain",
        "ase2022",
        "--cohort-path",
        str(cohort),
        "--taxonomy-path",
        str(taxonomy),
        "--output-dir",
        str(output),
        "--dry-run",
    ]

    all_summary = run_cli([*common, "--stage", "all"])
    stage3_summary = run_cli([*common, "--stage", "stage3"])

    assert all_summary["record_count"] == 2
    assert stage3_summary["record_count"] == 1
    assert all_summary["config_hash"] != stage3_summary["config_hash"]
    manifest = json.loads(
        Path(stage3_summary["manifest_path"]).read_text(encoding="utf-8")
    )
    assert manifest["record_ids"] == ["record-2"]

    Path(stage3_summary["predictions_path"]).write_text(
        json.dumps(
            {
                "record_id": "record-2",
                "config_hash": all_summary["config_hash"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="config_hash mismatch"):
        run_cli(
            [argument for argument in common if argument != "--dry-run"]
            + ["--stage", "stage3", "--no-progress"],
            transport_override=lambda system, user: (_ for _ in ()).throw(
                AssertionError("incompatible resume rows must fail before model calls")
            ),
        )


def test_stage3_cli_handles_an_all_negative_oracle_cohort(tmp_path: Path) -> None:
    cohort, taxonomy = _write_stage3_selection_inputs(
        tmp_path,
        decisions=("rejected_candidate",),
    )

    summary = run_cli(
        [
            "--domain",
            "ase2022",
            "--cohort-path",
            str(cohort),
            "--taxonomy-path",
            str(taxonomy),
            "--output-dir",
            str(tmp_path / "negative-output"),
            "--stage",
            "stage3",
            "--dry-run",
        ]
    )

    assert summary["record_count"] == 0
    manifest = json.loads(Path(summary["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["record_ids"] == []


def test_stage3_split_manifest_runs_all_evidence_only_records_without_decision_oracle(
    tmp_path: Path,
) -> None:
    split_root = Path(
        "Benchmark/configs/splits/ase2022_stage3_uncontaminated_seed20260816_revision4"
    )
    validation_summary = run_cli(
        [
            "--domain",
            "ase2022",
            "--cohort-path",
            str(split_root / "validation_runner_cohort.csv"),
            "--taxonomy-path",
            "Benchmark/configs/ase2022_taxonomy_structure_v1.json",
            "--split-manifest",
            str(split_root / "split_manifest.json"),
            "--output-dir",
            str(tmp_path / "dry-run"),
            "--stage",
            "stage3",
            "--dry-run",
        ]
    )

    final_summary = run_cli(
        [
            "--domain",
            "ase2022",
            "--cohort-path",
            str(split_root / "final_runner_cohort.csv"),
            "--taxonomy-path",
            "Benchmark/configs/ase2022_taxonomy_structure_v1.json",
            "--split-manifest",
            str(split_root / "split_manifest.json"),
            "--output-dir",
            str(tmp_path / "final-dry-run"),
            "--stage",
            "stage3",
            "--dry-run",
        ]
    )

    assert validation_summary["record_count"] == 8
    assert final_summary["record_count"] == 60


def test_stage3_cli_rejects_revoked_split(tmp_path: Path) -> None:
    split_root = Path(
        "Benchmark/configs/splits/ase2022_stage3_uncontaminated_seed20260816"
    )
    with pytest.raises(ValueError, match="revoked"):
        run_cli(
            [
                "--domain",
                "ase2022",
                "--cohort-path",
                str(split_root / "validation_runner_cohort.csv"),
                "--taxonomy-path",
                "Benchmark/configs/ase2022_taxonomy_structure_v1.json",
                "--split-manifest",
                str(split_root / "split_manifest.json"),
                "--output-dir",
                str(tmp_path / "revoked"),
                "--stage",
                "stage3",
                "--dry-run",
            ]
        )


def test_stage3_split_manifest_rejects_wrong_domain_taxonomy(tmp_path: Path) -> None:
    split_root = Path(
        "Benchmark/configs/splits/ase2022_stage3_uncontaminated_seed20260816_revision4"
    )
    with pytest.raises(ValueError, match="taxonomy.*hash|domain"):
        run_cli(
            [
                "--domain",
                "ase2022",
                "--cohort-path",
                str(split_root / "validation_runner_cohort.csv"),
                "--taxonomy-path",
                "Benchmark/results/issta2024_bugs_in_pods_baseline/code_diff_repaired/issta2024_taxonomy.json",
                "--split-manifest",
                str(split_root / "split_manifest.json"),
                "--output-dir",
                str(tmp_path / "wrong-taxonomy"),
                "--stage",
                "stage3",
                "--dry-run",
            ]
        )


def test_dry_run_validates_inputs_and_writes_manifest_without_api_key(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    cohort, taxonomy = _write_inputs(tmp_path)
    output = tmp_path / "output"
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API", raising=False)

    summary = run_cli(
        [
            "--domain",
            "ase2022",
            "--cohort-path",
            str(cohort),
            "--taxonomy-path",
            str(taxonomy),
            "--output-dir",
            str(output),
            "--provider",
            "deepseek",
            "--model",
            "deepseek-v4-flash",
            "--stage",
            "all",
            "--dry-run",
        ]
    )

    manifest = json.loads(Path(summary["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["domain"] == "ase2022"
    assert manifest["record_count"] == 1
    assert manifest["stage"] == "all"
    assert manifest["prompt_version"] == "causal-responsibility-revision-v25"
    assert manifest["role_max_tokens"] == 1600
    assert manifest["agent_policy"]["profile"] == "label-thinking"
    assert manifest["agent_policy"]["provider_supports_thinking"] is True
    assert manifest["agent_policy"]["resolved"]["root_cause_analyst@B"] == {
        "effective_thinking_enabled": True,
        "max_tokens": 32768,
        "thinking": "enabled",
    }
    assert summary["agent_policy"] == manifest["agent_policy"]
    assert manifest["agent_policy"]["team_perspectives"] == {
        "A": "evidence_first",
        "B": "falsification_first",
    }
    serialized_perspectives = json.dumps(
        manifest["agent_policy"]["team_perspectives"],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert (
        manifest["resolved_config"]["prompt_hashes"]["team_perspective_policy"]
        == hashlib.sha256(serialized_perspectives).hexdigest()
    )
    assert manifest["gold_fields_excluded_from_agent_prompts"] == [
        "decision",
        "root_cause",
        "symptom",
    ]
    assert not Path(summary["predictions_path"]).exists()


def test_micu_provider_uses_dedicated_credentials_and_normalized_v1_base_url() -> None:
    config = runner_script._resolve_adaptive_provider_config(
        {
            "MICU_API_KEY": "micu-test-key",
            "MICU_BASE_URL": "https://www.micuapi.ai/",
            "DEEPSEEK_API_KEY": "must-not-be-used",
            "OPENAI_API_KEY": "must-not-be-used",
        },
        provider="micu",
        base_url_override=None,
    )

    assert config == {
        "api_key": "micu-test-key",
        "base_url": "https://www.micuapi.ai/v1",
        "user_agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:149.0) "
            "Gecko/20100101 Firefox/149.0"
        ),
    }
    with pytest.raises(SystemExit, match="Missing MICU_API_KEY"):
        runner_script._resolve_adaptive_provider_config(
            {
                "DEEPSEEK_API_KEY": "must-not-be-used",
                "OPENAI_API_KEY": "must-not-be-used",
            },
            provider="micu",
            base_url_override=None,
        )


@pytest.mark.parametrize(
    ("base_url", "message"),
    [
        ("http://www.micuapi.ai", "HTTPS"),
        ("https://attacker.example", "official MICU host"),
        ("https://user@www.micuapi.ai", "userinfo"),
        ("https://www.micuapi.ai:444", "port"),
        ("https://www.micuapi.ai/v1?redirect=1", "query or fragment"),
        ("https://www.micuapi.ai/custom", "base path"),
    ],
)
def test_micu_provider_rejects_credential_redirecting_base_urls(
    base_url: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        runner_script._resolve_adaptive_provider_config(
            {"MICU_API_KEY": "micu-test-key"},
            provider="micu",
            base_url_override=base_url,
        )


def test_micu_provider_accepts_documented_preferred_host() -> None:
    config = runner_script._resolve_adaptive_provider_config(
        {"MICU_API_KEY": "micu-test-key"},
        provider="micu",
        base_url_override="https://api-slb.micuapi.ai/v1/",
    )

    assert config["base_url"] == "https://api-slb.micuapi.ai/v1"


@pytest.mark.parametrize("alias", ["gemini", "gemeni", "genemi"])
def test_gemini_provider_aliases_are_canonicalized(alias: str) -> None:
    assert runner_script._canonical_provider(alias) == "gemini"


def test_gemini_provider_uses_official_credentials_and_endpoint() -> None:
    config = runner_script._resolve_adaptive_provider_config(
        {
            "GOOGLE_API_KEY": "google-test-key",
            "GEMINI_API_KEY": "gemini-fallback-key",
            "OPENAI_API_KEY": "must-not-be-used",
        },
        provider="gemini",
        base_url_override=None,
    )

    assert config == {
        "api_key": "google-test-key",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
    }


@pytest.mark.parametrize(
    ("base_url", "message"),
    [
        ("http://generativelanguage.googleapis.com/v1beta/openai", "HTTPS"),
        ("https://attacker.example/v1beta/openai", "official Gemini host"),
        (
            "https://user@generativelanguage.googleapis.com/v1beta/openai",
            "userinfo",
        ),
        (
            "https://generativelanguage.googleapis.com:444/v1beta/openai",
            "port",
        ),
        (
            "https://generativelanguage.googleapis.com/v1beta/openai?redirect=1",
            "query or fragment",
        ),
        (
            "https://generativelanguage.googleapis.com/v1beta",
            "base path",
        ),
    ],
)
def test_gemini_provider_rejects_credential_redirecting_base_urls(
    base_url: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        runner_script._resolve_adaptive_provider_config(
            {"GEMINI_API_KEY": "gemini-test-key"},
            provider="gemini",
            base_url_override=base_url,
        )


def test_gemini_provider_requires_an_explicit_model() -> None:
    with pytest.raises(SystemExit, match="--provider gemini requires an explicit --model"):
        runner_script._resolve_adaptive_model("gemini", None)


def test_targeted_sla30_accepts_gemini_provider() -> None:
    args = build_parser().parse_args(
        [
            "--domain",
            "ase2022",
            "--stage",
            "stage3",
            "--provider",
            "gemini",
            "--model",
            "gemini-3.6-flash",
            "--execution-profile",
            "targeted-sla30",
            "--baseline-anchor-path",
            "baseline.jsonl",
            "--record-ids",
            "record-1",
        ]
    )

    config = runner_script._targeted_sla_config(
        args,
        concurrency_explicit=False,
        provider_max_inflight_explicit=False,
    )

    assert config is not None
    assert args.concurrency == 4
    assert args.provider_max_inflight == 4


def test_gemini_dry_run_canonicalizes_alias_without_leaking_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cohort, taxonomy = _write_inputs(tmp_path)
    secret = "gemini-secret-must-not-appear"
    monkeypatch.setenv("GEMINI_API_KEY", secret)

    summary = run_cli(
        [
            "--domain",
            "ase2022",
            "--cohort-path",
            str(cohort),
            "--taxonomy-path",
            str(taxonomy),
            "--output-dir",
            str(tmp_path / "gemini-dry-run"),
            "--provider",
            "gemeni",
            "--model",
            "gemini-3.6-flash",
            "--dry-run",
        ]
    )

    manifest_text = Path(summary["manifest_path"]).read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    assert manifest["provider"] == "gemini"
    assert manifest["model"] == "gemini-3.6-flash"
    assert manifest["backend"] == (
        "https://generativelanguage.googleapis.com/v1beta/openai"
    )
    assert manifest["agent_policy"]["provider_supports_thinking"] is False
    assert secret not in manifest_text


def test_gemini_transport_policy_uses_pool_and_omits_deepseek_thinking() -> None:
    assert runner_script._uses_pooled_provider_client("gemini", targeted_sla=False)
    assert not runner_script._uses_pooled_provider_client("gemini", targeted_sla=True)
    assert runner_script._provider_thinking_option("gemini", True) is None
    assert runner_script._provider_thinking_option("gemini", False) is None
    assert runner_script._provider_thinking_option("micu", True) is True


def test_micu_dry_run_records_provider_without_loading_or_leaking_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cohort, taxonomy = _write_inputs(tmp_path)
    secret = "micu-secret-must-not-appear"
    monkeypatch.setenv("MICU_API_KEY", secret)

    summary = run_cli(
        [
            "--domain",
            "ase2022",
            "--cohort-path",
            str(cohort),
            "--taxonomy-path",
            str(taxonomy),
            "--output-dir",
            str(tmp_path / "micu-dry-run"),
            "--provider",
            "micu",
            "--model",
            "deepseek-v4-flash",
            "--base-url",
            "https://www.micuapi.ai/",
            "--stage",
            "all",
            "--dry-run",
        ]
    )

    manifest_text = Path(summary["manifest_path"]).read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    assert manifest["provider"] == "micu"
    assert manifest["model"] == "deepseek-v4-flash"
    assert manifest["backend"] == "https://www.micuapi.ai/v1"
    assert manifest["agent_policy"]["provider_supports_thinking"] is True
    assert secret not in manifest_text


def test_micu_provider_requires_explicit_model_before_writing_run(
    tmp_path: Path,
) -> None:
    cohort, taxonomy = _write_inputs(tmp_path)

    with pytest.raises(SystemExit, match="--model"):
        run_cli(
            [
                "--domain",
                "ase2022",
                "--cohort-path",
                str(cohort),
                "--taxonomy-path",
                str(taxonomy),
                "--output-dir",
                str(tmp_path / "micu-missing-model"),
                "--provider",
                "micu",
                "--dry-run",
            ]
        )


def test_micu_default_transport_forwards_dedicated_http_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cohort, taxonomy = _write_inputs(tmp_path)
    observed: dict[str, object] = {}

    class FakePooledChatCompletionClient:
        def __init__(
            self,
            *,
            base_url: str,
            max_connections: int,
        ) -> None:
            observed["pool_base_url"] = base_url
            observed["pool_max_connections"] = max_connections
            observed["pool_instance"] = self

        def call_model(self, **_kwargs: object) -> str:
            raise AssertionError("fake retry boundary should not call the HTTP client")

        def close(self) -> None:
            observed["pool_closed"] = True

    def fake_retry_call(**kwargs: object) -> tuple[str, int]:
        observed.update(kwargs)
        return (
            json.dumps(
                {
                    "dimension": "symptom",
                    "baseline_label": "Crash",
                    "proposed_label": "Incorrect Functionality",
                    "status": "consistent",
                    "rationale": "The exact evidence supports this boundary.",
                    "supporting_evidence_ids": ["evidence-1"],
                    "assessment_digest": "e" * 64,
                    "boundary_card_id": "symptom-boundary",
                    "baseline_source_config_hash": "a" * 64,
                    "baseline_source_predictions_sha256": "b" * 64,
                    "taxonomy_structure_hash": "c" * 64,
                    "evidence_view_hash": "d" * 64,
                }
            ),
            1,
        )

    def fake_record_runner(
        record: dict[str, str],
        *,
        agents: object,
        **kwargs: object,
    ) -> dict[str, object]:
        agents._client.complete(
            system_prompt="ROLE: provider_wiring_probe",
            user_prompt="Return the requested JSON object.",
            schema=RevisionConsistencyReport,
            options=ModelCallOptions(
                role="joint_anchor",
                team_id="A",
                perspective="evidence_first",
                max_tokens=1200,
                thinking_enabled=True,
            ),
        )
        return {
            "record_id": record["record_id"],
            "stage2_prediction": None,
            "stage2_valid": False,
            "symptom_prediction": "Crash",
            "root_cause_prediction": "Incorrect Code Logic",
            "stage3_valid": True,
            "stop_reason": "stage3_only",
            "audit": None,
        }

    monkeypatch.setenv("MICU_API_KEY", "micu-test-key")
    monkeypatch.setattr(
        runner_script,
        "PooledChatCompletionClient",
        FakePooledChatCompletionClient,
    )
    monkeypatch.setattr(runner_script, "call_model_with_retries", fake_retry_call)
    monkeypatch.setattr(runner_script, "run_adaptive_record", fake_record_runner)

    run_cli(
        [
            "--domain",
            "ase2022",
            "--stage",
            "stage3",
            "--provider",
            "micu",
            "--model",
            "deepseek-v4-flash",
            "--cohort-path",
            str(cohort),
            "--taxonomy-path",
            str(taxonomy),
            "--output-dir",
            str(tmp_path / "micu-provider-wiring"),
            "--provider-max-inflight",
            "2",
            "--no-progress",
            "--no-resume",
        ]
    )

    assert observed["api_key"] == "micu-test-key"
    assert observed["base_url"] == "https://www.micuapi.ai/v1"
    assert observed["user_agent"] == (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:149.0) "
        "Gecko/20100101 Firefox/149.0"
    )
    assert observed["model"] == "deepseek-v4-flash"
    assert observed["pool_base_url"] == "https://www.micuapi.ai/v1"
    assert observed["pool_max_connections"] == 2
    assert observed["model_call"] == observed["pool_instance"].call_model
    assert observed["pool_closed"] is True


def test_cli_role_output_limit_is_configurable(tmp_path: Path) -> None:
    cohort, taxonomy = _write_inputs(tmp_path)

    summary = run_cli(
        [
            "--domain",
            "ase2022",
            "--cohort-path",
            str(cohort),
            "--taxonomy-path",
            str(taxonomy),
            "--output-dir",
            str(tmp_path / "output"),
            "--role-max-tokens",
            "900",
            "--dry-run",
        ]
    )

    manifest = json.loads(Path(summary["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["role_max_tokens"] == 900


def test_cli_policy_overrides_are_resolved_and_change_config_hash(
    tmp_path: Path,
) -> None:
    cohort, taxonomy = _write_inputs(tmp_path)
    common = [
        "--domain",
        "ase2022",
        "--cohort-path",
        str(cohort),
        "--taxonomy-path",
        str(taxonomy),
        "--output-dir",
        str(tmp_path / "policy-output"),
        "--dry-run",
    ]

    off = run_cli([*common, "--thinking-profile", "off"])
    overridden = run_cli(
        [
            *common,
            "--thinking-profile",
            "all-stage3",
            "--thinking-override",
            "root_cause_analyst=enabled",
            "--thinking-override",
            "root_cause_analyst@B=disabled",
            "--max-tokens-override",
            "root_cause_analyst@B=2500",
        ]
    )

    assert off["config_hash"] != overridden["config_hash"]
    manifest = json.loads(Path(overridden["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["agent_policy"]["resolved"]["root_cause_analyst@A"] == {
        "effective_thinking_enabled": True,
        "max_tokens": 2800,
        "thinking": "enabled",
    }
    assert manifest["agent_policy"]["resolved"]["root_cause_analyst@B"] == {
        "effective_thinking_enabled": False,
        "max_tokens": 2500,
        "thinking": "disabled",
    }


@pytest.mark.parametrize("profile", ("off", "causal", "label-thinking", "all-stage3"))
def test_optional_revision_roles_keep_independent_nonthinking_budgets(
    profile: str,
) -> None:
    policy = Stage3AgentPolicy.from_profile(profile)

    for team_id in ("A", "B"):
        assessment = policy.resolve("baseline_revision_assessment", team_id)
        consistency = policy.resolve("baseline_revision_consistency", team_id)
        assert assessment.thinking is ThinkingMode.DISABLED
        assert assessment.max_tokens == 2200
        assert consistency.thinking is ThinkingMode.DISABLED
        assert consistency.max_tokens == 1800


@pytest.mark.parametrize(
    ("profile", "joint", "symptom", "root"),
    (
        ("off", ThinkingMode.DISABLED, ThinkingMode.DISABLED, ThinkingMode.DISABLED),
        ("causal", ThinkingMode.ENABLED, ThinkingMode.DISABLED, ThinkingMode.ENABLED),
        (
            "label-thinking",
            ThinkingMode.ENABLED,
            ThinkingMode.DISABLED,
            ThinkingMode.ENABLED,
        ),
        (
            "all-stage3",
            ThinkingMode.ENABLED,
            ThinkingMode.ENABLED,
            ThinkingMode.ENABLED,
        ),
    ),
)
def test_anchored_role_thinking_profile_matrix(
    profile: str,
    joint: ThinkingMode,
    symptom: ThinkingMode,
    root: ThinkingMode,
) -> None:
    policy = Stage3AgentPolicy.from_profile(profile)

    assert policy.resolve("joint_anchor", "A").thinking is joint
    assert policy.resolve("joint_anchor", "B").max_tokens == 32768
    assert policy.resolve("symptom_verifier", "A").thinking is symptom
    assert policy.resolve("symptom_verifier", "A").max_tokens == 2200
    assert policy.resolve("root_cause_verifier", "A").thinking is root
    assert policy.resolve("root_cause_verifier", "B").max_tokens == 32768


def test_anchored_role_and_team_overrides_keep_resolution_precedence() -> None:
    role_override = RoleModelPolicy(
        thinking=ThinkingMode.DISABLED,
        max_tokens=9000,
    )
    team_override = RoleModelPolicy(
        thinking=ThinkingMode.ENABLED,
        max_tokens=9100,
    )
    invocation_override = RoleModelPolicy(
        thinking=ThinkingMode.DISABLED,
        max_tokens=9200,
    )
    policy = Stage3AgentPolicy.from_profile(
        "off",
        role_overrides={"joint_anchor": role_override},
        role_team_overrides={"joint_anchor@B": team_override},
    )

    assert policy.resolve("joint_anchor", "A") == role_override
    assert policy.resolve("joint_anchor", "B") == team_override
    assert (
        policy.resolve("joint_anchor", "B", invocation_override=invocation_override)
        == invocation_override
    )


@pytest.mark.parametrize(
    "target",
    (
        "joint_anchor",
        "joint_anchor@A",
        "symptom_verifier",
        "symptom_verifier@B",
        "root_cause_verifier",
        "root_cause_verifier@A",
    ),
)
def test_each_anchored_policy_override_changes_config_hash(
    tmp_path: Path,
    target: str,
) -> None:
    cohort, taxonomy = _write_inputs(tmp_path)
    common = [
        "--domain",
        "ase2022",
        "--cohort-path",
        str(cohort),
        "--taxonomy-path",
        str(taxonomy),
        "--output-dir",
        str(tmp_path / "anchored-policy"),
        "--dry-run",
    ]

    baseline = run_cli(common)
    overridden = run_cli([*common, "--max-tokens-override", f"{target}=1234"])

    assert overridden["config_hash"] != baseline["config_hash"]


@pytest.mark.parametrize("target", ("verifier", "joint_anchor@C"))
def test_cli_rejects_unknown_anchored_policy_targets(
    tmp_path: Path,
    target: str,
) -> None:
    cohort, taxonomy = _write_inputs(tmp_path)

    with pytest.raises(ValueError, match="unknown Stage 3 policy"):
        run_cli(
            [
                "--domain",
                "ase2022",
                "--cohort-path",
                str(cohort),
                "--taxonomy-path",
                str(taxonomy),
                "--output-dir",
                str(tmp_path / "unknown-anchored-policy"),
                "--thinking-override",
                f"{target}=enabled",
                "--dry-run",
            ]
        )


def test_cli_defaults_to_label_owner_thinking_profile(tmp_path: Path) -> None:
    cohort, taxonomy = _write_inputs(tmp_path)
    summary = run_cli(
        [
            "--domain",
            "ase2022",
            "--stage",
            "stage3",
            "--cohort-path",
            str(cohort),
            "--taxonomy-path",
            str(taxonomy),
            "--output-dir",
            str(tmp_path / "runs"),
            "--dry-run",
        ]
    )

    assert summary["agent_policy"]["profile"] == "label-thinking"
    assert summary["agent_policy"]["resolved"]["root_cause_analyst@B"] == {
        "thinking": "enabled",
        "effective_thinking_enabled": True,
        "max_tokens": 32768,
    }
    assert summary["agent_policy"]["resolved"]["stage3_arbitrator"] == {
        "thinking": "enabled",
        "effective_thinking_enabled": True,
        "max_tokens": 32768,
    }
    assert build_parser().parse_args(["--domain", "ase2022"]).max_network_retries == 5


def test_cli_uses_collision_safe_run_identity_and_treats_concurrency_as_execution_only(
    tmp_path: Path,
) -> None:
    cohort, taxonomy = _write_inputs(tmp_path)
    common = [
        "--domain",
        "ase2022",
        "--stage",
        "stage3",
        "--cohort-path",
        str(cohort),
        "--taxonomy-path",
        str(taxonomy),
        "--output-dir",
        str(tmp_path / "runs"),
        "--experiment-id",
        "dual-team-v1",
        "--split-id",
        "dev",
        "--arm-id",
        "a3-all-thinking",
        "--run-id",
        "run-01",
        "--dry-run",
    ]

    first = run_cli([*common, "--concurrency", "1"])
    second = run_cli([*common, "--concurrency", "4"])

    expected = tmp_path / "runs" / "dual-team-v1" / "dev" / "a3-all-thinking" / "run-01"
    assert Path(first["manifest_path"]) == expected / "run_manifest.json"
    assert Path(first["predictions_path"]).name == (
        "predictions_deepseek-v4-flash.jsonl"
    )
    assert Path(first["metrics_path"]).name == "metrics_deepseek-v4-flash.json"
    assert first["config_hash"] == second["config_hash"]
    manifest = json.loads(Path(second["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["resolved_config"]["concurrency"] == 4
    assert manifest["experiment_id"] == "dual-team-v1"
    assert manifest["arm_id"] == "a3-all-thinking"


def test_run_cli_parallel_completion_keeps_limited_cohort_order_in_predictions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cohort, taxonomy = _write_ordering_inputs(tmp_path)
    release_first = threading.Event()
    snapshots: list[ProgressSnapshot] = []

    def capture(snapshot: ProgressSnapshot) -> None:
        snapshots.append(snapshot)
        if snapshot.record_id == "record-2":
            release_first.set()

    def fake_run_adaptive_record(
        record: dict[str, str], **_kwargs: object
    ) -> dict[str, object]:
        if record["record_id"] == "record-1":
            if not release_first.wait(timeout=2):
                raise AssertionError("record-2 did not complete first")
        return {
            "record_id": record["record_id"],
            "stage2_prediction": "accepted_fault",
            "stage2_valid": True,
            "symptom_prediction": None,
            "root_cause_prediction": None,
            "stage3_valid": False,
            "audit": None,
        }

    monkeypatch.setattr(runner_script, "ConsoleProgress", lambda: capture)
    monkeypatch.setattr(
        runner_script,
        "run_adaptive_record",
        fake_run_adaptive_record,
    )
    summary = run_cli(
        [
            "--domain",
            "ase2022",
            "--stage",
            "stage2",
            "--cohort-path",
            str(cohort),
            "--taxonomy-path",
            str(taxonomy),
            "--output-dir",
            str(tmp_path / "runs"),
            "--limit",
            "2",
            "--concurrency",
            "2",
            "--no-resume",
        ],
        transport_override=lambda *_args, **_kwargs: pytest.fail(
            "record stub must avoid model calls"
        ),
    )

    persisted_ids = [
        json.loads(line)["record_id"]
        for line in Path(summary["predictions_path"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    completion_ids = [
        snapshot.record_id for snapshot in snapshots if snapshot.record_id
    ]
    assert completion_ids == ["record-2", "record-1"]
    assert persisted_ids == ["record-1", "record-2"]
    assert summary["record_count"] == 2
    assert summary["metrics"]["n"] == 2


def test_run_cli_resume_rejects_numeric_existing_id_matching_by_string_coercion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cohort, taxonomy = _write_inputs(tmp_path, record_id="1")

    def fake_run_adaptive_record(
        record: dict[str, str], **_kwargs: object
    ) -> dict[str, object]:
        return {
            "record_id": record["record_id"],
            "stage2_prediction": "accepted_fault",
            "stage2_valid": True,
            "symptom_prediction": None,
            "root_cause_prediction": None,
            "stage3_valid": False,
            "audit": None,
        }

    monkeypatch.setattr(
        runner_script,
        "run_adaptive_record",
        fake_run_adaptive_record,
    )
    args = [
        "--domain",
        "ase2022",
        "--stage",
        "stage2",
        "--cohort-path",
        str(cohort),
        "--taxonomy-path",
        str(taxonomy),
        "--output-dir",
        str(tmp_path / "runs"),
        "--no-progress",
    ]
    no_model_transport = lambda *_args, **_kwargs: pytest.fail(
        "record stub must avoid model calls"
    )
    summary = run_cli(args, transport_override=no_model_transport)
    predictions = Path(summary["predictions_path"])
    row = json.loads(predictions.read_text(encoding="utf-8"))
    row["record_id"] = 1
    malformed = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
    predictions.write_text(malformed, encoding="utf-8")
    monkeypatch.setattr(
        runner_script,
        "run_adaptive_record",
        lambda *_args, **_kwargs: pytest.fail(
            "invalid resume row must fail before record execution"
        ),
    )

    with pytest.raises(ValueError, match="record_id must be a non-empty string"):
        run_cli(args, transport_override=no_model_transport)

    assert predictions.read_text(encoding="utf-8") == malformed


def test_cli_refuses_to_overwrite_same_run_identity_with_different_algorithm(
    tmp_path: Path,
) -> None:
    cohort, taxonomy = _write_inputs(tmp_path)
    common = [
        "--domain",
        "ase2022",
        "--stage",
        "stage3",
        "--cohort-path",
        str(cohort),
        "--taxonomy-path",
        str(taxonomy),
        "--output-dir",
        str(tmp_path / "runs"),
        "--experiment-id",
        "dual-team-v1",
        "--split-id",
        "dev",
        "--arm-id",
        "a3",
        "--run-id",
        "run-01",
        "--dry-run",
    ]
    run_cli([*common, "--thinking-profile", "off"])

    with pytest.raises(ValueError, match="existing run configuration mismatch"):
        run_cli([*common, "--thinking-profile", "all-stage3"])


@pytest.mark.parametrize(
    "target",
    (
        "fault_evidence_analyst@A",
        "scope_boundary_analyst@B",
        "repair_causality_analyst@A",
        "evidence_readiness@B",
        "stage2_fault_verifier@A",
    ),
)
def test_cli_rejects_dead_non_stage3_team_override(
    tmp_path: Path,
    target: str,
) -> None:
    cohort, taxonomy = _write_inputs(tmp_path)

    with pytest.raises(ValueError, match="does not accept a team override"):
        run_cli(
            [
                "--domain",
                "ase2022",
                "--cohort-path",
                str(cohort),
                "--taxonomy-path",
                str(taxonomy),
                "--output-dir",
                str(tmp_path / "dead-override"),
                "--thinking-override",
                f"{target}=enabled",
                "--dry-run",
            ]
        )


def test_manifest_serializes_only_effective_policy_invocation_targets(
    tmp_path: Path,
) -> None:
    cohort, taxonomy = _write_inputs(tmp_path)
    summary = run_cli(
        [
            "--domain",
            "ase2022",
            "--cohort-path",
            str(cohort),
            "--taxonomy-path",
            str(taxonomy),
            "--output-dir",
            str(tmp_path / "effective-policy"),
            "--dry-run",
        ]
    )

    resolved = summary["agent_policy"]["resolved"]
    assert resolved["baseline_revision_assessment@A"] == {
        "effective_thinking_enabled": False,
        "max_tokens": 2200,
        "thinking": "disabled",
    }
    assert resolved["baseline_revision_assessment@B"] == {
        "effective_thinking_enabled": False,
        "max_tokens": 2200,
        "thinking": "disabled",
    }
    assert resolved["baseline_revision_consistency@A"] == {
        "effective_thinking_enabled": False,
        "max_tokens": 1800,
        "thinking": "disabled",
    }
    assert resolved["baseline_revision_consistency@B"] == {
        "effective_thinking_enabled": False,
        "max_tokens": 1800,
        "thinking": "disabled",
    }
    assert set(resolved) == {
        "stage2_fault_verifier@A",
        "stage2_fault_verifier@B",
        "fault_evidence_analyst@fault_evidence",
        "scope_boundary_analyst@scope_boundary",
        "repair_causality_analyst@repair_causality",
        "evidence_readiness@evidence_readiness",
        "stage2_arbitrator",
        "symptom_analyst@A",
        "symptom_analyst@B",
        "root_cause_analyst@A",
        "root_cause_analyst@B",
        "joint_anchor@A",
        "joint_anchor@B",
        "symptom_verifier@A",
        "symptom_verifier@B",
        "root_cause_verifier@A",
        "root_cause_verifier@B",
        "causal_consistency_checker@A",
        "causal_consistency_checker@B",
        "baseline_revision_assessment@A",
        "baseline_revision_assessment@B",
        "baseline_revision_consistency@A",
        "baseline_revision_consistency@B",
        "boundary_challenger",
        "stage3_arbitrator",
    }
    assert resolved["fault_evidence_analyst@fault_evidence"] == {
        "effective_thinking_enabled": False,
        "max_tokens": 1600,
        "thinking": "disabled",
    }
    assert resolved["joint_anchor@A"] == {
        "effective_thinking_enabled": True,
        "max_tokens": 32768,
        "thinking": "enabled",
    }
    assert resolved["symptom_verifier@B"] == {
        "effective_thinking_enabled": False,
        "max_tokens": 2200,
        "thinking": "disabled",
    }
    assert resolved["root_cause_verifier@A"] == {
        "effective_thinking_enabled": True,
        "max_tokens": 32768,
        "thinking": "enabled",
    }

    manifest = json.loads(Path(summary["manifest_path"]).read_text(encoding="utf-8"))
    prompt_hashes = manifest["resolved_config"]["prompt_hashes"]
    for role in (
        AnalystRole.JOINT_ANCHOR,
        AnalystRole.SYMPTOM_VERIFIER,
        AnalystRole.ROOT_CAUSE_VERIFIER,
    ):
        assert (
            prompt_hashes[role.value]
            == hashlib.sha256(
                render_system_prompt(
                    role,
                    include_dataset_guidance=False,
                ).encode("utf-8")
            ).hexdigest()
        )


def test_unsupported_provider_preserves_declared_anchored_policy(
    tmp_path: Path,
) -> None:
    cohort, taxonomy = _write_inputs(tmp_path)

    summary = run_cli(
        [
            "--domain",
            "ase2022",
            "--cohort-path",
            str(cohort),
            "--taxonomy-path",
            str(taxonomy),
            "--output-dir",
            str(tmp_path / "unsupported-thinking"),
            "--provider",
            "proxy",
            "--dry-run",
        ]
    )

    resolved = summary["agent_policy"]["resolved"]["joint_anchor@A"]
    assert resolved["thinking"] == "enabled"
    assert resolved["effective_thinking_enabled"] is None


def test_record_subset_and_limit_are_reflected_in_config_hash(
    tmp_path: Path,
) -> None:
    cohort, taxonomy = _write_inputs(tmp_path)
    output = tmp_path / "output"

    first = run_cli(
        [
            "--domain",
            "ase2022",
            "--cohort-path",
            str(cohort),
            "--taxonomy-path",
            str(taxonomy),
            "--output-dir",
            str(output),
            "--dry-run",
        ]
    )
    second = run_cli(
        [
            "--domain",
            "ase2022",
            "--cohort-path",
            str(cohort),
            "--taxonomy-path",
            str(taxonomy),
            "--output-dir",
            str(output),
            "--limit",
            "0",
            "--dry-run",
        ]
    )

    assert first["config_hash"] != second["config_hash"]
    assert first["record_count"] == 1
    assert second["record_count"] == 0


def test_cli_executes_complete_pipeline_with_only_http_boundary_replaced(
    tmp_path: Path,
) -> None:
    cohort, taxonomy = _write_inputs(tmp_path)
    output = tmp_path / "output"

    captured_options: list[ModelCallOptions] = []
    captured_system_prompts: dict[str, str] = {}

    def transport(
        system: str,
        user: str,
        *,
        options: ModelCallOptions,
    ) -> str:
        captured_options.append(options)
        captured_system_prompts.setdefault(options.role, system)
        payload = json.loads(user)
        if "joint_anchor" in system:
            items = payload["evidence_view"]["items"]
            symptom_id = next(
                item["evidence_id"]
                for item in items
                if item["source_type"] != "code_diff"
            )
            root_id = next(
                item["evidence_id"]
                for item in items
                if item["source_type"] == "code_diff"
            )
            return json.dumps(
                {
                    "symptom": {
                        "label": "Crash",
                        "behavior_claim": "The request terminates processing unexpectedly.",
                        "supporting_evidence_ids": [symptom_id],
                        "counter_evidence_ids": [],
                        "alternative_label": "Incorrect Functionality",
                        "boundary_reason": "The process terminates instead of returning a wrong value.",
                        "confidence": 0.9,
                        "evidence_sufficiency": "sufficient",
                    },
                    "root_cause": {
                        "label": "Incorrect Code Logic",
                        "defect_mechanism": "An incorrect branch rejects the valid request.",
                        "causal_chain": [
                            "The request enters the handler.",
                            "The condition selects rejection.",
                            "Processing terminates.",
                        ],
                        "supporting_evidence_ids": [root_id],
                        "counter_evidence_ids": [],
                        "alternative_label": "API Misuse",
                        "boundary_reason": "The branch defect is internal rather than caller misuse.",
                        "confidence": 0.9,
                        "evidence_sufficiency": "sufficient",
                    },
                    "causal_account": (
                        "The incorrect internal branch directly causes request termination."
                    ),
                    "shared_supporting_evidence_ids": [symptom_id, root_id],
                }
            )
        if "symptom_verifier" in system or "root_cause_verifier" in system:
            dimension = payload["context"]["owned_dimension"]
            anchor = payload["context"]["anchor"]
            evidence_id = (
                anchor["symptom"]["supporting_evidence_ids"][0]
                if dimension == "symptom"
                else anchor["root_cause"]["supporting_evidence_ids"][0]
            )
            anchor_label = (
                anchor["symptom"]["label"]
                if dimension == "symptom"
                else anchor["root_cause"]["label"]
            )
            return json.dumps(
                {
                    "dimension": dimension,
                    "verdict": "accept",
                    "anchor_label": anchor_label,
                    "alternative_label": None,
                    "rationale": (
                        "The frozen evidence independently supports the anchored label."
                    ),
                    "supporting_evidence_ids": [evidence_id],
                    "counter_evidence_ids": [],
                    "confidence": 0.87,
                }
            )
        if "evidence_snapshot" in payload:
            evidence_id = payload["evidence_snapshot"]["items"][0]["evidence_id"]
            return json.dumps(
                {
                    "resolution_status": "resolved",
                    "symptom_label": "Crash",
                    "root_cause_label": "Incorrect Code Logic",
                    "confidence": 0.85,
                    "rationale": (
                        "The bounded evidence supports the listed shared-risk review."
                    ),
                    "supporting_evidence_ids": [evidence_id],
                    "resolved_dimensions": payload["disagreement"]["dimensions"],
                }
            )
        evidence_id = payload["evidence_view"]["items"][0]["evidence_id"]
        team_id = payload["team_id"]
        if "evidence_readiness" in system:
            task = payload["context"]["task"]
            if task == "stage3":
                mechanism = next(
                    (
                        item
                        for item in payload["evidence_view"]["items"]
                        if item["source_type"] == "code_diff"
                    ),
                    None,
                )
                if mechanism is None:
                    return json.dumps(
                        {
                            "task": task,
                            "dimensions": [
                                {
                                    "dimension": "symptom",
                                    "sufficient": True,
                                    "confirmed_evidence_ids": [evidence_id],
                                    "missing_facts": [],
                                    "evidence_requests": [],
                                },
                                {
                                    "dimension": "root_cause",
                                    "sufficient": False,
                                    "confirmed_evidence_ids": [],
                                    "missing_facts": [
                                        "The defect mechanism requires code evidence."
                                    ],
                                    "evidence_requests": [
                                        {
                                            "request_id": "stage3-code-context",
                                            "missing_fact": (
                                                "The pre-fix defect mechanism in code."
                                            ),
                                            "why_needed": (
                                                "Root-cause readiness requires "
                                                "mechanism-capable evidence."
                                            ),
                                            "target_specialist": "code_context",
                                            "target_source": "code diff",
                                            "query": "reject request process",
                                            "expected_decision_impact": (
                                                "Code evidence can establish root-cause "
                                                "readiness."
                                            ),
                                            "max_items": 2,
                                        }
                                    ],
                                },
                            ],
                        }
                    )
                return json.dumps(
                    {
                        "task": task,
                        "dimensions": [
                            {
                                "dimension": "symptom",
                                "sufficient": True,
                                "confirmed_evidence_ids": [evidence_id],
                                "missing_facts": [],
                                "evidence_requests": [],
                            },
                            {
                                "dimension": "root_cause",
                                "sufficient": True,
                                "confirmed_evidence_ids": [mechanism["evidence_id"]],
                                "missing_facts": [],
                                "evidence_requests": [],
                            },
                        ],
                    }
                )
            return json.dumps(
                {
                    "task": task,
                    "dimensions": [
                        {
                            "dimension": dimension,
                            "sufficient": True,
                            "confirmed_evidence_ids": [evidence_id],
                            "missing_facts": [],
                            "evidence_requests": [],
                        }
                        for dimension in payload["context"]["required_dimensions"]
                    ],
                }
            )
        if "fault_evidence_analyst" in system:
            return json.dumps(
                {
                    "outcome": "pass",
                    "claim": "A valid request is rejected before the patch.",
                    "supporting_evidence_ids": [evidence_id],
                    "counter_evidence_ids": [],
                }
            )
        if "scope_boundary_analyst" in system:
            return json.dumps(
                {
                    "outcome": "pass",
                    "claim": "The issue report is included by the ASE2022 policy.",
                    "supporting_evidence_ids": [evidence_id],
                    "counter_evidence_ids": [],
                }
            )
        if "symptom_analyst" in system:
            return json.dumps(
                {
                    "label": "Crash",
                    "behavior_claim": "The request terminates processing unexpectedly.",
                    "supporting_evidence_ids": [evidence_id],
                    "counter_evidence_ids": [],
                    "alternative_label": "Crash",
                    "boundary_reason": "The supplied taxonomy has one applicable behavior.",
                    "confidence": 0.9,
                    "evidence_sufficiency": "sufficient",
                    "unresolved_evidence_gaps": [],
                    "evidence_requests": [],
                }
            )
        if "root_cause_analyst" in system:
            return json.dumps(
                {
                    "label": "Incorrect Code Logic",
                    "defect_mechanism": "An incorrect branch rejects the valid request.",
                    "causal_chain": [
                        "The request enters the handler.",
                        "The condition selects rejection.",
                        "Processing terminates.",
                    ],
                    "supporting_evidence_ids": [evidence_id],
                    "counter_evidence_ids": [],
                    "alternative_label": "Incorrect Code Logic",
                    "boundary_reason": "The supplied taxonomy has one applicable mechanism.",
                    "confidence": 0.9,
                    "evidence_sufficiency": "sufficient",
                    "unresolved_evidence_gaps": [],
                    "evidence_requests": [],
                }
            )
        if "causal_consistency_checker" in system:
            return json.dumps(
                {
                    "status": "consistent",
                    "rationale": "The incorrect rejection branch directly explains termination.",
                    "supporting_evidence_ids": [evidence_id],
                    "evidence_requests": [],
                }
            )
        if "boundary_challenger" in system:
            return json.dumps(
                {
                    "action": "pass",
                    "rationale": "The proposed labels respect the supplied boundaries.",
                    "cited_evidence_ids": [evidence_id],
                }
            )
        raise AssertionError(f"unexpected system prompt: {system}")

    summary = run_cli(
        [
            "--domain",
            "ase2022",
            "--cohort-path",
            str(cohort),
            "--taxonomy-path",
            str(taxonomy),
            "--output-dir",
            str(output),
            "--no-progress",
        ],
        transport_override=transport,
    )

    assert summary["metrics"]["stage2"]["accuracy"] == 1.0
    assert summary["metrics"]["stage3"]["joint_accuracy"] == 1.0
    assert summary["metrics"]["stage3"]["dual_team"]["both_team_reach_rate"] == 1.0
    assert summary["metrics"]["stage3"]["dual_team"]["team_a"]["joint_accuracy"] == 1.0
    assert summary["metrics"]["stage3"]["dual_team"]["team_b"]["joint_accuracy"] == 1.0
    assert summary["metrics"]["llm_telemetry"]["total_calls"] == 15
    assert (
        summary["metrics"]["stage2"]["resolution_source_counts"]["policy_composition"]
        == 1
    )
    assert {call["role"] for call in summary["metrics"]["llm_telemetry"]["calls"]} == {
        "boundary_challenger",
        "evidence_readiness",
        "fault_evidence_analyst",
        "scope_boundary_analyst",
        "joint_anchor",
        "symptom_verifier",
        "root_cause_verifier",
        "causal_consistency_checker",
        "stage3_arbitrator",
    }
    stage3_options = {
        (option.role, option.team_id): option for option in captured_options
    }
    assert stage3_options[("joint_anchor", "A")].max_tokens == 32768
    assert stage3_options[("symptom_verifier", "A")].thinking_enabled is False
    assert stage3_options[("symptom_verifier", "A")].max_tokens == 2200
    assert stage3_options[("root_cause_verifier", "B")].thinking_enabled is True
    assert stage3_options[("root_cause_verifier", "B")].max_tokens == 32768
    assert stage3_options[("root_cause_verifier", "B")].perspective == (
        "falsification_first"
    )
    assert stage3_options[("causal_consistency_checker", "A")].max_tokens == 1800
    assert stage3_options[("causal_consistency_checker", "A")].thinking_enabled is False
    assert stage3_options[("boundary_challenger", None)].max_tokens == 1800
    assert stage3_options[("boundary_challenger", None)].thinking_enabled is False
    assert stage3_options[("stage3_arbitrator", None)].max_tokens == 32768
    assert stage3_options[("stage3_arbitrator", None)].thinking_enabled is True
    manifest = json.loads(Path(summary["manifest_path"]).read_text(encoding="utf-8"))
    prompt_hashes = manifest["resolved_config"]["prompt_hashes"]
    for role in (
        AnalystRole.JOINT_ANCHOR,
        AnalystRole.SYMPTOM_VERIFIER,
        AnalystRole.ROOT_CAUSE_VERIFIER,
    ):
        actual_base_prompt = render_system_prompt(
            role,
            include_dataset_guidance=False,
        )
        assert actual_base_prompt in captured_system_prompts[role.value]
        assert render_system_prompt(role) not in captured_system_prompts[role.value]
        assert (
            prompt_hashes[role.value]
            == hashlib.sha256(actual_base_prompt.encode("utf-8")).hexdigest()
        )
    prediction = json.loads(
        Path(summary["predictions_path"]).read_text(encoding="utf-8").splitlines()[0]
    )
    assert prediction["stage2_prediction"] == "accepted_fault"
    assert prediction["symptom_prediction"] == "Crash"
    assert prediction["audit"]["ledger_version"] == 2

    resumed = run_cli(
        [
            "--domain",
            "ase2022",
            "--cohort-path",
            str(cohort),
            "--taxonomy-path",
            str(taxonomy),
            "--output-dir",
            str(output),
            "--no-progress",
        ],
        transport_override=lambda system, user: (_ for _ in ()).throw(
            AssertionError("a complete resume must not call the model")
        ),
    )
    assert resumed["metrics"]["llm_telemetry"]["total_calls"] == 15
    assert resumed["metrics"]["cumulative_wall_time_seconds"] > 0


def test_invalid_model_output_is_recorded_and_scored_instead_of_aborting(
    tmp_path: Path,
) -> None:
    cohort, taxonomy = _write_inputs(tmp_path)
    output = tmp_path / "output"

    summary = run_cli(
        [
            "--domain",
            "ase2022",
            "--cohort-path",
            str(cohort),
            "--taxonomy-path",
            str(taxonomy),
            "--output-dir",
            str(output),
            "--max-schema-retries",
            "0",
            "--no-progress",
        ],
        transport_override=lambda system, user: '{"invalid":true}',
    )

    assert summary["metrics"]["stage2"]["invalid_count"] == 1
    row = json.loads(
        Path(summary["predictions_path"]).read_text(encoding="utf-8").splitlines()[0]
    )
    assert row["stage2_valid"] is False
    assert row["stage3_valid"] is False
    assert row["error"]["type"] == "StructuredOutputError"
    assert "invalid" not in row["error"]["message"]
    assert "input_value" not in row["error"]["message"]


def test_transport_failure_is_distinguished_from_structured_output_failure(
    tmp_path: Path,
) -> None:
    cohort, taxonomy = _write_inputs(tmp_path)

    def transport(system: str, user: str) -> str:
        raise ModelTransportError("connection reset", attempts=6)

    summary = run_cli(
        [
            "--domain",
            "ase2022",
            "--cohort-path",
            str(cohort),
            "--taxonomy-path",
            str(taxonomy),
            "--output-dir",
            str(tmp_path / "output"),
            "--no-progress",
        ],
        transport_override=transport,
    )

    row = json.loads(
        Path(summary["predictions_path"]).read_text(encoding="utf-8").splitlines()[0]
    )
    assert row["stop_reason"] == "transport_error"
    assert row["error"]["type"] == "ModelTransportError"
    assert row["error"]["code"] == "model_transport_failed"
    assert row["error"]["message"] == "Model transport failed during record execution."
    assert "connection reset" not in str(row["error"])


def test_cli_does_not_persist_arbitrary_structured_error_text(tmp_path: Path) -> None:
    cohort, taxonomy = _write_inputs(tmp_path)
    secret = "SECRET_RAW_MODEL_COMPLETION"

    def transport(system: str, user: str) -> str:
        from Benchmark.src.adaptive_empirical_workflow.agents import (
            StructuredOutputError,
        )

        raise StructuredOutputError(secret, schema_name=secret)

    summary = run_cli(
        [
            "--domain",
            "ase2022",
            "--cohort-path",
            str(cohort),
            "--taxonomy-path",
            str(taxonomy),
            "--output-dir",
            str(tmp_path / "output"),
            "--no-progress",
        ],
        transport_override=transport,
    )

    row = json.loads(
        Path(summary["predictions_path"]).read_text(encoding="utf-8").splitlines()[0]
    )
    assert row["error"] == {
        "type": "StructuredOutputError",
        "code": "structured_output_invalid",
        "message": "Structured model output failed validation.",
    }
    assert secret not in str(row)


def test_console_progress_displays_elapsed_time_and_eta() -> None:
    stream = io.StringIO()
    progress = ConsoleProgress(stream=stream, width=10)

    progress(
        ProgressSnapshot(
            completed=1,
            total=2,
            record_id="paper:record-1",
            status="running",
            elapsed_seconds=30,
            eta_seconds=30,
            resumed_count=0,
        )
    )
    progress(
        ProgressSnapshot(
            completed=2,
            total=2,
            record_id="paper:record-2",
            status="completed",
            elapsed_seconds=60,
            eta_seconds=0,
            resumed_count=0,
        )
    )

    output = stream.getvalue()
    assert "1/2" in output
    assert "elapsed 30s" in output
    assert "ETA 30s" in output
    assert "record-1" in output
    assert output.endswith("\n")
