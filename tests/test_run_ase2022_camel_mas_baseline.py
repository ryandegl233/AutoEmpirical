from __future__ import annotations

import logging
import json
from io import StringIO

import pytest

from Benchmark.scripts import run_ase2022_camel_mas_baseline as runner
from Benchmark.scripts.run_ase2022_camel_mas_baseline import (
    CamelNoiseFilter,
    ConsoleProgress,
    _load_optional_json,
    resolve_model,
    resolve_run_config,
    validate_max_turns,
)


def test_resolve_run_config_prefers_self_credentials() -> None:
    config = resolve_run_config(
        {
            "SELF_BASE_URL": "https://proxy.example/v1",
            "BASE_URL": "https://fallback.example/v1",
            "SELF_API": "self-key",
            "OPENAI_API_KEY": "openai-key",
        }
    )

    assert config == {
        "base_url": "https://proxy.example/v1",
        "api_key": "self-key",
    }


def test_resolve_run_config_supports_official_deepseek_api() -> None:
    config = resolve_run_config(
        {"DEEPSEEK_API_KEY": "deepseek-key"},
        provider="deepseek",
    )

    assert config == {
        "base_url": "https://api.deepseek.com",
        "api_key": "deepseek-key",
    }


def test_resolve_run_config_allows_deepseek_endpoint_override() -> None:
    config = resolve_run_config(
        {
            "DEEPSEEK_API_KEY": "deepseek-key",
            "DEEPSEEK_BASE_URL": "https://deepseek-proxy.example/v1",
        },
        provider="deepseek",
    )

    assert config["base_url"] == "https://deepseek-proxy.example/v1"


def test_resolve_model_uses_provider_default_and_honors_explicit_model() -> None:
    assert resolve_model("proxy", None) == "claude-3-5-sonnet-20241022"
    assert resolve_model("deepseek", None) == "deepseek-v4-flash"
    assert resolve_model("deepseek", "deepseek-chat") == "deepseek-chat"


def test_resolve_run_config_requires_endpoint_and_key() -> None:
    with pytest.raises(SystemExit, match="Missing SELF_BASE_URL"):
        resolve_run_config({"SELF_API": "key"})
    with pytest.raises(SystemExit, match="Missing SELF_API"):
        resolve_run_config({"SELF_BASE_URL": "https://proxy.example/v1"})
    with pytest.raises(SystemExit, match="Missing DEEPSEEK_API_KEY"):
        resolve_run_config({}, provider="deepseek")


def test_load_optional_json_reports_missing_control_and_loads_existing(tmp_path) -> None:
    missing = tmp_path / "missing.json"
    existing = tmp_path / "metrics.json"
    existing.write_text('{"accuracy": 0.5}', encoding="utf-8")

    assert _load_optional_json(missing) is None
    assert _load_optional_json(existing) == {"accuracy": 0.5}


def test_validate_max_turns_requires_positive_value() -> None:
    assert validate_max_turns(5) == 5
    with pytest.raises(ValueError, match="max_turns must be positive"):
        validate_max_turns(0)


def test_runner_uses_ten_as_default_society_turn_cap() -> None:
    assert getattr(runner, "DEFAULT_MAX_TURNS", None) == 10


def test_runner_defaults_to_evidence_anchored_society() -> None:
    assert getattr(runner, "DEFAULT_SOCIETY_MODE", None) == "evidence_anchored"


def test_strict_json_defaults_are_profile_specific() -> None:
    from Benchmark.scripts import run_issta2024_camel_mas_baseline as issta

    ase_args = runner.build_parser(runner.ASE2022_PROFILE).parse_args([])
    issta_args = runner.build_parser(issta.PROFILE).parse_args([])

    assert ase_args.require_valid_json is False
    assert issta_args.require_valid_json is True


def test_issta_allow_invalid_explicitly_disables_strict_json() -> None:
    from Benchmark.scripts import run_issta2024_camel_mas_baseline as issta

    args = runner.build_parser(issta.PROFILE).parse_args(["--allow-invalid"])

    assert args.require_valid_json is False


def test_strict_json_setting_changes_config_hash() -> None:
    records = [{"record_id": "r1"}]
    taxonomy = {"symptom": ["Crash"], "root_cause": ["Cause"]}

    permissive = runner.build_config_hash(
        "model",
        "stage2",
        records,
        taxonomy,
        0.0,
        require_valid_json=False,
    )
    strict = runner.build_config_hash(
        "model",
        "stage2",
        records,
        taxonomy,
        0.0,
        require_valid_json=True,
    )

    assert permissive != strict


def test_stage_metrics_record_strict_json_mode(
    tmp_path,
    monkeypatch,
) -> None:
    cohort = [
        {
            "record_id": "r1",
            "decision": "accepted_fault",
            "symptom": "Crash",
            "root_cause": "Cause",
        }
    ]
    taxonomy = {"symptom": ["Crash"], "root_cause": ["Cause"]}
    observed: dict[str, object] = {}

    monkeypatch.setattr(runner, "_load_env_file", lambda _path: None)
    monkeypatch.setattr(runner, "load_unified_cohort", lambda _path: cohort)
    monkeypatch.setattr(runner, "_load_taxonomy", lambda _path: taxonomy)
    monkeypatch.setattr(
        runner,
        "resolve_run_config",
        lambda *_args, **_kwargs: {
            "base_url": "https://example.invalid",
            "api_key": "key",
        },
    )
    monkeypatch.setattr(
        runner,
        "make_runner_factories",
        lambda *_args, **_kwargs: ("society", "finalizer"),
    )
    monkeypatch.setattr(runner, "configure_camel_logging", lambda *_args: None)

    def run_records(*_args, **kwargs):
        observed["require_valid_json"] = kwargs["require_valid_json"]
        return [
            {
                "record_id": "r1",
                "invalid": False,
                "final_prediction": {"decision": "accepted_fault"},
                "society": {},
            }
        ]

    monkeypatch.setattr(runner, "run_stage_records", run_records)

    runner.run_profile(
        runner.ASE2022_PROFILE,
        [
            "--stage",
            "stage2",
            "--model",
            "model",
            "--output-dir",
            str(tmp_path),
            "--require-valid-json",
            "--no-progress",
        ],
    )

    metrics_path = next(tmp_path.glob("*stage2_metrics*.json"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert observed["require_valid_json"] is True
    assert metrics["require_valid_json"] is True


def test_society_artifact_prefix_isolates_native_and_anchored_outputs() -> None:
    assert runner.society_artifact_prefix("native") == "ase2022_camel_society"
    assert (
        runner.society_artifact_prefix("evidence_anchored")
        == "ase2022_camel_evidence_anchored"
    )
    assert (
        runner.society_artifact_prefix("native", study_slug="issta2024")
        == "issta2024_camel_society"
    )
    assert (
        runner.society_artifact_prefix(
            "evidence_anchored", study_slug="issta2024"
        )
        == "issta2024_camel_evidence_anchored"
    )


def test_runner_builds_society_and_finalizer_with_identical_model_config(
    monkeypatch,
) -> None:
    calls: list[tuple[str, tuple, dict]] = []

    def society_factory(*args, **kwargs):
        calls.append(("society", args, kwargs))
        return "society-factory"

    def finalizer_factory(*args, **kwargs):
        calls.append(("finalizer", args, kwargs))
        return "finalizer-factory"

    monkeypatch.setattr(runner, "make_camel_society_factory", society_factory)
    monkeypatch.setattr(
        runner,
        "make_camel_agent_factory",
        finalizer_factory,
        raising=False,
    )

    factories = runner.make_runner_factories(
        "deepseek-chat",
        "secret",
        "https://api.deepseek.com",
        temperature=0.0,
        max_retries=2,
        timeout=120.0,
        society_mode="evidence_anchored",
    )

    assert factories == ("society-factory", "finalizer-factory")
    assert calls[0][1] == calls[1][1]
    assert calls[0][2]["society_mode"] == "evidence_anchored"
    assert "society_mode" not in calls[1][2]
    for key in ("temperature", "max_retries", "timeout"):
        assert calls[0][2][key] == calls[1][2][key]


@pytest.mark.parametrize(
    "message",
    [
        "Unknown model 'claude': context window size not defined. Defaulting to 999_999_999.",
        "Format validation error: invalid JSON. Attempting fallback with JSON format.",
    ],
)
def test_camel_noise_filter_suppresses_only_known_recoverable_warnings(
    message: str,
) -> None:
    record = logging.LogRecord("camel", logging.WARNING, "", 0, message, (), None)
    assert CamelNoiseFilter().filter(record) is False


def test_camel_noise_filter_keeps_real_errors_and_other_warnings() -> None:
    error = logging.LogRecord(
        "camel", logging.ERROR, "", 0, "Fallback attempt also failed", (), None
    )
    warning = logging.LogRecord(
        "camel", logging.WARNING, "", 0, "Connection retry exhausted", (), None
    )

    assert CamelNoiseFilter().filter(error) is True
    assert CamelNoiseFilter().filter(warning) is True


def test_console_progress_renders_stage_count_percentage_eta_and_completion() -> None:
    stream = StringIO()
    times = iter([0.0, 10.0, 20.0])
    progress = ConsoleProgress(
        "Stage 2",
        stream=stream,
        clock=lambda: next(times),
        width=10,
    )

    progress(1, 2, "paper:record-1", "completed")
    progress(2, 2, "paper:record-2", "completed")

    output = stream.getvalue()
    assert "Stage 2 [█████░░░░░]" in output
    assert "1/2" in output
    assert "50.0%" in output
    assert "ETA 10s" in output
    assert "2/2" in output
    assert output.endswith("\n")
