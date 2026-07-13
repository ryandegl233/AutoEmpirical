from __future__ import annotations

import json
from pathlib import Path

import pytest

from Benchmark.scripts.run_ase2022_stage2_filter_baseline import resolve_run_config
from Benchmark.src import ase2022_stage2_filter_baseline as baseline


def _write_prompts(path: Path) -> None:
    rows = [
        {
            "record_id": "p",
            "ground_truth": baseline.ACCEPTED_FAULT,
            "system_prompt": "system",
            "user_prompt": "positive",
        },
        {
            "record_id": "n",
            "ground_truth": baseline.REJECTED_CANDIDATE,
            "system_prompt": "system",
            "user_prompt": "negative",
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_run_filter_prompts_persists_metrics_and_resumes(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    metrics = tmp_path / "metrics.json"
    _write_prompts(prompts)
    calls: list[str] = []

    def caller(**kwargs: object) -> str:
        user_prompt = str(kwargs["user_prompt"])
        calls.append(user_prompt)
        decision = (
            baseline.ACCEPTED_FAULT if user_prompt == "positive" else baseline.REJECTED_CANDIDATE
        )
        return json.dumps({"decision": decision})

    first = baseline.run_filter_prompts(
        prompts_path=prompts,
        predictions_path=predictions,
        metrics_path=metrics,
        model="fake-model",
        model_caller=caller,
        max_retries=0,
    )
    second = baseline.run_filter_prompts(
        prompts_path=prompts,
        predictions_path=predictions,
        metrics_path=metrics,
        model="fake-model",
        model_caller=caller,
        max_retries=0,
    )

    assert calls == ["positive", "negative"]
    assert first["accuracy"] == 1.0
    assert first["processed_count"] == 2
    assert second["resumed_count"] == 2
    assert second["processed_count"] == 0
    assert len(predictions.read_text(encoding="utf-8").splitlines()) == 2


def test_run_filter_prompts_persists_exhausted_failure_as_invalid(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    metrics = tmp_path / "metrics.json"
    _write_prompts(prompts)
    attempts = 0

    def failing_caller(**kwargs: object) -> str:
        nonlocal attempts
        attempts += 1
        raise TimeoutError("timeout")

    result = baseline.run_filter_prompts(
        prompts_path=prompts,
        predictions_path=predictions,
        metrics_path=metrics,
        model="fake-model",
        model_caller=failing_caller,
        max_retries=1,
        retry_delay_seconds=0,
        limit=1,
    )

    row = json.loads(predictions.read_text(encoding="utf-8").strip())
    assert attempts == 2
    assert row["invalid"] is True
    assert row["request_failed"] is True
    assert row["attempts"] == 2
    assert result["invalid_count"] == 1


def test_run_filter_prompts_does_not_retry_non_transient_failure(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts.jsonl"
    _write_prompts(prompts)
    attempts = 0

    def failing_caller(**kwargs: object) -> str:
        nonlocal attempts
        attempts += 1
        raise ValueError("invalid API configuration")

    baseline.run_filter_prompts(
        prompts_path=prompts,
        predictions_path=tmp_path / "predictions.jsonl",
        metrics_path=tmp_path / "metrics.json",
        model="fake-model",
        model_caller=failing_caller,
        max_retries=3,
        retry_delay_seconds=0,
        limit=1,
    )

    assert attempts == 1


def test_run_filter_prompts_rejects_duplicate_resume_records(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    _write_prompts(prompts)
    duplicate = {"record_id": "p", "model": "fake-model", "invalid": False}
    predictions.write_text(json.dumps(duplicate) + "\n" + json.dumps(duplicate) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate record_id"):
        baseline.run_filter_prompts(
            prompts_path=prompts,
            predictions_path=predictions,
            metrics_path=tmp_path / "metrics.json",
            model="fake-model",
            model_caller=lambda **kwargs: "",
        )


def test_resolve_run_config_accepts_self_credentials() -> None:
    config = resolve_run_config({"SELF_BASE_URL": "https://example.test/v1", "SELF_API": "key"})
    assert config["base_url"] == "https://example.test/v1"
    assert config["api_key"] == "key"
