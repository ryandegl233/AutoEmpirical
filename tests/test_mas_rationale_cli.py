from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from Benchmark.scripts import run_ase2022_camel_mas_baseline as runner
from Benchmark.scripts import run_issta2024_camel_mas_baseline as issta_runner
from Benchmark.src import issta2024_bugs_in_pods_baseline as issta
from Benchmark.src.mas_rationale import CONTRACT_VERSION


@pytest.mark.parametrize("profile", [runner.ASE2022_PROFILE, issta_runner.PROFILE])
def test_rationale_cli_is_opt_in_and_rejects_unknown_modes(profile) -> None:
    parser = runner.build_parser(profile)
    assert getattr(parser.parse_args([]), "explanation_mode", None) == "label_only"
    assert parser.parse_args(
        ["--explanation-mode", "evidence_rationale"]
    ).explanation_mode == "evidence_rationale"
    with pytest.raises(SystemExit):
        parser.parse_args(["--explanation-mode", "freeform"])


@pytest.mark.parametrize(
    ("society_mode", "legacy_prefix"),
    [
        ("native", "ase2022_camel_society"),
        ("evidence_anchored", "ase2022_camel_evidence_anchored"),
    ],
)
def test_rationale_artifact_prefix_cannot_overwrite_label_only_outputs(
    society_mode: str, legacy_prefix: str
) -> None:
    assert runner.society_artifact_prefix(society_mode) == legacy_prefix
    assert runner.society_artifact_prefix(
        society_mode, explanation_mode="label_only"
    ) == legacy_prefix
    assert runner.society_artifact_prefix(
        society_mode, explanation_mode="evidence_rationale"
    ) == legacy_prefix + f"_rationale_v{CONTRACT_VERSION}"


def test_rationale_factories_enable_transport_capture_for_both_roles(monkeypatch) -> None:
    options = []

    def external_factory(_model, _key, _url, **kwargs):
        options.append(kwargs)
        return object()

    monkeypatch.setattr(runner, "make_camel_society_factory", external_factory)
    monkeypatch.setattr(runner, "make_camel_agent_factory", external_factory)
    runner.make_runner_factories(
        "model", "secret", "https://example.invalid", temperature=0.0,
        max_retries=0, timeout=None, capture_trace=True,
    )
    assert len(options) == 2
    assert all(option["capture_trace"] is True for option in options)

    options.clear()
    runner.make_runner_factories(
        "model", "secret", "https://example.invalid", temperature=0.0,
        max_retries=0, timeout=None,
    )
    assert all("capture_trace" not in option for option in options)


def _response(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        msg=SimpleNamespace(content=content), terminated=False, info={}
    )


class _LocalAgent:
    def __init__(self, content: str) -> None:
        self.content = content

    def step(self, _message):
        return _response(self.content)


class _LocalSociety:
    def __init__(self, content: str) -> None:
        self.specified_task_prompt = None
        self.assistant_agent = _LocalAgent(content)
        self.user_agent = _LocalAgent(
            "Instruction: Classify the supplied evidence.\n"
            "Input: USE_IMMUTABLE_EVIDENCE"
        )

    def init_chat(self, init_msg_content=None):
        return SimpleNamespace(content=init_msg_content or "Classify this record.")

    def step(self, message):
        return self.assistant_agent.step(message), self.user_agent.step(message)


def _inputs(tmp_path: Path, profile):
    taxonomy = (
        issta.build_issta2024_taxonomy()
        if profile.study_slug == "issta2024"
        else {"symptom": ["Crash"], "root_cause": ["Incorrect Code Logic"]}
    )
    record = {
        "record_id": "r1",
        "paper_id": profile.study_slug,
        "issue_url": "https://example.invalid/commit/1",
        "title": "Repair null pointer crash",
        "body": "The process crashes when the pointer is null.",
        "comments": "The null check fixes the crash.",
        "state": "closed",
        "created_at": "2026-01-01",
        "source_project": "runc",
        "changed_files": '["main.go"]',
        "code_diff": "+ if pointer == nil { return }",
        "decision": "accepted_fault",
        "symptom": taxonomy["symptom"][0],
        "root_cause": taxonomy["root_cause"][0],
    }
    cohort_path = tmp_path / "cohort.csv"
    with cohort_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(record))
        writer.writeheader()
        writer.writerow(record)
    taxonomy_path = tmp_path / "taxonomy.json"
    taxonomy_path.write_text(json.dumps(taxonomy), encoding="utf-8")
    return record, cohort_path, taxonomy_path


def _prediction(record, stage: str, *, rationale: bool, study_slug: str):
    labels = (
        {"decision": "accepted_fault"}
        if stage == "stage2"
        else {"symptom": record["symptom"], "root_cause": record["root_cause"]}
    )
    result = dict(labels)
    if rationale:
        evidence = (
            {"field": "code_diff", "quote": "pointer == nil"}
            if study_slug == "issta2024"
            else {"field": "body", "quote": "process crashes"}
        )
        for dimension in labels:
            result[f"{dimension}_rationale"] = {
                "reason": "The reported crash and null guard support this label.",
                "evidence_refs": [evidence],
                "evidence_limitations": "The full execution trace is unavailable.",
            }
    return result


def test_rationale_manifest_binds_input_contents_and_current_source(tmp_path) -> None:
    _, cohort_path, taxonomy_path = _inputs(tmp_path, runner.ASE2022_PROFILE)
    args = runner.build_parser().parse_args([
        "--model", "local-test-model", "--explanation-mode", "evidence_rationale"
    ])
    paths = runner.ExperimentPaths(
        cohort_path, taxonomy_path, tmp_path / "outputs",
        tmp_path / "missing-stage2.json", tmp_path / "missing-stage3.json",
    )

    def write_manifest():
        return runner._write_rationale_manifest(
            args, paths, artifact_prefix="ase2022_camel_society_rationale_v1",
            slug="local-test-model", backend_id="proxy:https://example.invalid",
            temperature=0.0, stage2_path=tmp_path / "stage2.jsonl",
        )

    first_path, first_id = write_manifest()
    manifest = json.loads(first_path.read_text(encoding="utf-8"))
    assert len(manifest["code"]["git_revision"]) == 40
    source = "Benchmark/scripts/run_ase2022_camel_mas_baseline.py"
    assert manifest["code"]["source_sha256"][source] == hashlib.sha256(
        (runner.REPO_ROOT / source).read_bytes()
    ).hexdigest()
    cohort_path.write_bytes(cohort_path.read_bytes() + b"\n")
    second_path, second_id = write_manifest()
    assert first_id != second_id
    assert first_path != second_path
    assert first_path.is_file() and second_path.is_file()


def test_rationale_run_identity_changes_cache_hash_but_preserves_legacy_hash() -> None:
    assert runner._bind_run_identity("legacy-hash", None) == "legacy-hash"
    first = runner._bind_run_identity("stage-hash", "manifest-one")
    second = runner._bind_run_identity("stage-hash", "manifest-two")
    assert len(first) == 64
    assert first != second


@pytest.mark.parametrize(
    ("stage", "suffix", "payload"),
    [
        ("stage2", "stage2_predictions_model.jsonl", '{"run_id":"previous"}\n'),
        ("stage2", "stage2_metrics_model.json", '{"run_id":"previous"}'),
        ("stage3", "stage3_predictions_model.jsonl", '{"record_id":"unknown"}\n'),
        ("all", "end_to_end_metrics_model.json", '{"run_id":"previous"}'),
        ("all", "stage2_predictions_model.jsonl", 'incomplete JSON\n'),
    ],
)
def test_rationale_preflight_preserves_files_from_other_or_unknown_runs(
    tmp_path, stage, suffix, payload
) -> None:
    path = tmp_path / f"prefix_{suffix}"
    path.write_text(payload, encoding="utf-8")
    original = path.read_bytes()
    with pytest.raises(ValueError, match="new --output-dir"):
        runner._preflight_rationale_outputs(
            tmp_path, artifact_prefix="prefix", slug="model", stage=stage, run_id="current"
        )
    assert path.read_bytes() == original


def test_stage3_preflight_allows_reading_stage2_results_from_a_separate_invocation(
    tmp_path,
) -> None:
    path = tmp_path / "prefix_stage2_predictions_model.jsonl"
    path.write_text('{"run_id":"stage2-run"}\n', encoding="utf-8")
    runner._preflight_rationale_outputs(
        tmp_path, artifact_prefix="prefix", slug="model", stage="stage3", run_id="stage3-run"
    )
    assert path.read_text(encoding="utf-8") == '{"run_id":"stage2-run"}\n'


@pytest.mark.parametrize("profile", [runner.ASE2022_PROFILE, issta_runner.PROFILE])
@pytest.mark.parametrize("society_mode", ["native", "evidence_anchored"])
@pytest.mark.parametrize("stage", ["stage2", "stage3", "all"])
def test_cli_runs_rationale_contract_without_reusing_legacy_artifacts(
    tmp_path, monkeypatch, profile, society_mode, stage
) -> None:
    record, cohort_path, taxonomy_path = _inputs(tmp_path, profile)
    output_dir = tmp_path / "outputs"
    monkeypatch.setattr(runner, "_load_env_file", lambda _path: None)
    monkeypatch.setenv("SELF_API", "test-secret-never-persist")
    monkeypatch.setenv("SELF_BASE_URL", "https://example.invalid/v1")
    stages = ["stage2", "stage3"] if stage == "all" else [stage]
    responses: list[str] = []
    provider_options = []

    def society_factory(_task):
        return _LocalSociety(responses.pop(0))

    def local_factories(*_args, **kwargs):
        provider_options.append(kwargs)
        return society_factory, None

    monkeypatch.setattr(runner, "make_runner_factories", local_factories)
    argv = [
        "--provider", "proxy", "--model", "local-test-model",
        "--stage", stage, "--society-mode", society_mode,
        "--cohort-path", str(cohort_path), "--taxonomy-path", str(taxonomy_path),
        "--output-dir", str(output_dir), "--max-turns", "1",
        "--max-retries", "0", "--no-progress", "--require-valid-json",
        "--single-llm-stage2-metrics", str(tmp_path / "missing-control2.json"),
        "--single-llm-stage3-metrics", str(tmp_path / "missing-control3.json"),
    ]
    responses.extend(json.dumps(_prediction(
        record, current, rationale=False, study_slug=profile.study_slug
    )) for current in stages)
    runner.run_profile(profile, argv)
    legacy_files = {path: path.read_bytes() for path in output_dir.iterdir()}
    assert len(legacy_files) == (5 if stage == "all" else 2)
    assert all("rationale" not in path.name for path in legacy_files)
    legacy_hashes = {
        current: json.loads(next(output_dir.glob(
            f"*_{current}_metrics_*.json"
        )).read_text(encoding="utf-8"))["config_hash"]
        for current in stages
    }

    responses.extend(json.dumps(_prediction(
        record, current, rationale=True, study_slug=profile.study_slug
    )) for current in stages)
    runner.run_profile(profile, [*argv, "--explanation-mode", "evidence_rationale"])

    assert not responses
    assert provider_options[0].get("capture_trace", False) is False
    assert provider_options[1]["capture_trace"] is True
    assert all(path.read_bytes() == original for path, original in legacy_files.items())
    for current in stages:
        path = next(output_dir.glob(f"*_rationale_v{CONTRACT_VERSION}_{current}_predictions_*.jsonl"))
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 1
        row = rows[0]
        assert row["invalid"] is False
        assert row["explanation_mode"] == "evidence_rationale"
        assert row["config_hash"] != legacy_hashes[current]
        assert row["final_prediction"] == _prediction(
            record, current, rationale=False, study_slug=profile.study_slug
        )
        assert row["society"]["parsed_output"] == _prediction(
            record, current, rationale=True, study_slug=profile.study_slug
        )
        assert row["explanation_audit"]["status"] == "schema_and_citations_valid"
        assert row["explanation_audit"]["citations"][0]["field"] == (
            "code_diff" if profile.study_slug == "issta2024" else "body"
        )
    metrics_files = list(output_dir.glob(f"*_rationale_v{CONTRACT_VERSION}_*metrics_*.json"))
    assert len(metrics_files) == (3 if stage == "all" else 1)
    for path in metrics_files:
        metrics = json.loads(path.read_text(encoding="utf-8"))
        assert metrics["explanation_mode"] == "evidence_rationale"
        assert metrics["explanation_contract_version"] == CONTRACT_VERSION

    manifest_files = list(output_dir.glob(f"*_rationale_v{CONTRACT_VERSION}_run_manifest_*.json"))
    assert len(manifest_files) == 1
    manifest_text = manifest_files[0].read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    assert len(manifest["run_id"]) == 64
    assert manifest["config"]["explanation_mode"] == "evidence_rationale"
    assert manifest["config"]["stage"] == stage
    assert manifest["inputs"]["cohort"]["sha256"] == hashlib.sha256(
        cohort_path.read_bytes()
    ).hexdigest()
    assert manifest["inputs"]["taxonomy"]["sha256"] == hashlib.sha256(
        taxonomy_path.read_bytes()
    ).hexdigest()
    assert "api_key" not in manifest_text
    assert "test-secret-never-persist" not in manifest_text
    assert len(manifest["code"]["git_revision"]) == 40
    for source in (
        "Benchmark/scripts/run_ase2022_camel_mas_baseline.py",
        "Benchmark/scripts/run_issta2024_camel_mas_baseline.py",
        "Benchmark/src/ase2022_camel_mas_baseline.py",
        "Benchmark/src/ase2022_llm_baseline.py",
        "Benchmark/src/issta2024_bugs_in_pods_baseline.py",
    ):
        assert manifest["code"]["source_sha256"][source] == hashlib.sha256(
            (runner.REPO_ROOT / source).read_bytes()
        ).hexdigest()

    # A second invocation must reuse only the validated rationale checkpoint.
    runner.run_profile(profile, [*argv, "--explanation-mode", "evidence_rationale"])
    assert len(list(output_dir.glob(f"*_rationale_v{CONTRACT_VERSION}_run_manifest_*.json"))) == 1

    # A changed source identity must preserve earlier results for review.
    code_root = tmp_path / "changed-source"
    source_path = code_root / "Benchmark/scripts/run_ase2022_camel_mas_baseline.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("# different implementation revision\n", encoding="utf-8")
    monkeypatch.setattr(runner, "REPO_ROOT", code_root)
    original_files = {path: path.read_bytes() for path in output_dir.iterdir()}
    responses.extend(json.dumps(_prediction(
        record, current, rationale=True, study_slug=profile.study_slug
    )) for current in stages)
    with pytest.raises(ValueError, match="new --output-dir"):
        runner.run_profile(profile, [*argv, "--explanation-mode", "evidence_rationale"])
    assert len(responses) == len(stages)
    assert {path: path.read_bytes() for path in output_dir.iterdir()} == original_files
