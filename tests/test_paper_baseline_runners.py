from __future__ import annotations

import importlib.util
import json
import csv
from types import SimpleNamespace

import pytest

from Benchmark.scripts import run_ase2022_camel_mas_baseline as cli
from Benchmark.src import ase2022_camel_mas_baseline as mas
from Benchmark.src import ase2022_llm_baseline as single


def taxonomy(mode="multi_label"):
    return {
        "symptom": [] if mode == "free_text" else ["Crash", "Slow"],
        "root_cause": ["Logic", "Configuration"],
        "annotation_modes": {"symptom": mode, "root_cause": "single_label"},
    }


def test_mas_accepts_native_multilabel_and_rejects_unknown_atomic_label():
    check = mas._taxonomy_validator(taxonomy())
    check(mas.Stage3SocietyOutput(symptom="Slow || Crash", root_cause="Logic"))
    with pytest.raises(ValueError, match="symptom"):
        check(mas.Stage3SocietyOutput(symptom="Crash || Invented", root_cause="Logic"))


def test_mas_resume_validates_native_free_text_but_rejects_blank():
    row = {"record_id": "a", "invalid": False, "final_prediction": {
        "symptom": "Vehicle unexpectedly turns left.", "root_cause": "Logic"}}
    mas.validate_final_prediction(row, "stage3", taxonomy("free_text"))
    row["final_prediction"]["symptom"] = " "
    with pytest.raises(ValueError, match="symptom"):
        mas.validate_final_prediction(row, "stage3", taxonomy("free_text"))


def test_cli_preserves_free_text_contract_and_rejects_empty_closed_taxonomy(tmp_path):
    path = tmp_path / "taxonomy.json"
    path.write_text(json.dumps(taxonomy("free_text")), encoding="utf-8")
    assert cli._load_taxonomy(path) == taxonomy("free_text")
    path.write_text(json.dumps({"symptom": [], "root_cause": ["Logic"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="taxonomy"):
        cli._load_taxonomy(path)


@pytest.mark.parametrize("payload", [
    {"symptom": ["Crash"], "root_cause": ["Logic"], "annotation_modes": {"symptom": "invented_mode"}},
    {"symptom": ["Crash", "Slow"], "root_cause": ["Logic"], "annotation_modes": {"symptom": "constant"}},
    {"symptom": ["Crash", "Crash"], "root_cause": ["Logic"], "annotation_modes": {"symptom": "multi_label"}},
])
def test_cli_rejects_invalid_native_taxonomy_before_provider_calls(tmp_path, payload):
    path = tmp_path / "taxonomy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        cli._load_taxonomy(path)


def test_mas_saved_multilabel_contract_remains_a_string():
    row = {"record_id": "a", "invalid": False, "final_prediction": {
        "symptom": ["Crash", "Slow"], "root_cause": "Logic"}}
    with pytest.raises(ValueError, match="symptom"):
        mas.validate_final_prediction(row, "stage3", taxonomy())


def test_mas_native_metrics_compare_label_sets_and_skip_free_text_accuracy():
    gold = [{"record_id": "a", "decision": "accepted_fault", "symptom": "Crash || Slow", "root_cause": "Logic"}]
    preds = [{"record_id": "a", "invalid": False, "final_prediction": {
        "symptom": "Slow || Crash", "root_cause": "Logic"}}]
    metrics = mas.evaluate_stage3(gold, preds, taxonomy=taxonomy())
    assert metrics["symptom_set_exact_match"] == 1.0
    assert metrics["symptom_micro_f1"] == 1.0
    gold[0]["symptom"] = "Vehicle turns left."
    preds[0]["final_prediction"]["symptom"] = "Unexpected vehicle rotation."
    metrics = mas.evaluate_stage3(gold, preds, taxonomy=taxonomy("free_text"))
    assert metrics["symptom_accuracy"] is None
    assert metrics["joint_accuracy"] is None
    assert metrics["root_cause_accuracy"] == 1.0


def test_mas_native_end_to_end_uses_scored_dimensions_only():
    gold = [{"record_id": "a", "decision": "accepted_fault", "symptom": "Vehicle turns left.", "root_cause": "Logic"}]
    s2 = [{"record_id": "a", "invalid": False, "final_prediction": {"decision": "accepted_fault"}}]
    s3 = [{"record_id": "a", "invalid": False, "final_prediction": {"symptom": "Unexpected rotation.", "root_cause": "Logic"}}]
    metrics = mas.evaluate_end_to_end(gold, s2, s3, taxonomy=taxonomy("free_text"))
    assert metrics["scored_dimensions"] == ["root_cause"]
    assert metrics["complete_correct_count"] == 1


def test_single_native_parser_canonicalizes_sets_and_rejects_unknowns():
    parsed = single.parse_prediction_text('{"symptom":"Slow || Crash","root_cause":"Logic"}', taxonomy())
    assert parsed["invalid"] is False
    assert parsed["symptom"] == "Crash || Slow"
    assert single.parse_prediction_text('{"symptom":"Crash || Invented","root_cause":"Logic"}', taxonomy())["invalid"]
    assert not single.parse_prediction_text('{"symptom":"Vehicle turns left.","root_cause":"Logic"}', taxonomy("free_text"))["invalid"]


@pytest.mark.parametrize("name", ["run_paper_camel_mas_baseline", "run_paper_llm_baseline"])
def test_generic_runner_module_exists(name):
    assert importlib.util.find_spec("Benchmark.scripts." + name) is not None


def test_generic_runner_defaults_share_versioned_prepared_root():
    from Benchmark.scripts import run_paper_camel_mas_baseline as mas_runner
    from Benchmark.scripts import run_paper_llm_baseline as single_runner
    from Benchmark.src.paper_benchmark import DEFAULT_OUTPUT_ROOT

    assert mas_runner.DEFAULT_PREPARED_ROOT == str(DEFAULT_OUTPUT_ROOT)
    assert single_runner.build_parser().parse_args(["--domain", "icse2023"]).prepared_root == str(DEFAULT_OUTPUT_ROOT)


@pytest.mark.parametrize("stage", ["stage2", "stage3"])
def test_single_rejects_prompt_evidence_not_bound_to_current_projection(tmp_path, monkeypatch, stage):
    from Benchmark.scripts import run_paper_llm_baseline as runner

    base, _ = _prepared_fixture(tmp_path, "icse2023")
    path = base / f"{stage}_prompts.jsonl"
    prompts = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    prompts[0]["user_prompt"] += "\n\ncomments:\nnot_fetched"
    path.write_text("\n".join(json.dumps(row) for row in prompts), encoding="utf-8")
    monkeypatch.setattr(runner, "_load_env_file", lambda path: None)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    def unexpected_provider_call(**kwargs):
        pytest.fail("Prompt projection must be checked before calling any provider")
    monkeypatch.setattr(runner, "run_filter_prompts", unexpected_provider_call)
    monkeypatch.setattr(runner, "run_llm_prompts", unexpected_provider_call)
    with pytest.raises(ValueError, match="prompt evidence"):
        runner.main(["--domain", "icse2023", "--prepared-root", str(tmp_path),
                     "--model", "test-model", "--stage", stage])
    assert not (base / "single_llm").exists()


def test_single_runner_refuses_wrong_paper_or_duplicate_sample_ids(tmp_path):
    from Benchmark.scripts.run_paper_llm_baseline import _load_rows

    path = tmp_path / "sample.csv"
    path.write_text("record_id,paper_id\na,unexpected\n", encoding="utf-8")
    with pytest.raises(ValueError, match="paper_id"):
        _load_rows(path, "expected")
    path.write_text("record_id,paper_id\na,expected\na,expected\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        _load_rows(path, "expected")


def test_single_native_run_scores_missing_and_invalid_predictions_in_denominator(tmp_path, monkeypatch):
    prompts = tmp_path / "prompts.jsonl"
    rows = [{"record_id": str(index), "paper_id": "test", "issue_url": "", "system_prompt": "classify", "user_prompt": "observed failure"} for index in (1, 2)]
    prompts.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    outputs = iter(['{"symptom":"Slow || Crash","root_cause":"Logic"}', '{"symptom":"Invented","root_cause":"Logic"}'])
    monkeypatch.setattr(single, "call_model_with_retries", lambda **kwargs: (next(outputs), 1))
    metrics = single.run_llm_prompts(
        prompts, [{"record_id": str(index), "symptom": "Crash || Slow", "root_cause": "Logic"} for index in (1, 2)],
        taxonomy(), tmp_path / "predictions.jsonl", tmp_path / "metrics.json",
        model="test", api_key="test", base_url="https://example.invalid", resume=False,
    )
    assert metrics["n"] == 2
    assert metrics["invalid_count"] == 1
    assert metrics["symptom_set_exact_match"] == 0.5
    assert metrics["root_cause_accuracy"] == 0.5


DOMAINS = ("ase2022", "issta2024", "fse2021", "icse2021", "icse2022", "icse2023", "icse2024")


def _prepared_fixture(tmp_path, domain):
    from Benchmark.src import paper_benchmark as papers

    profile = papers.get_paper_profile(domain)
    tax = papers.build_taxonomy(domain)
    symptom = "Vehicle unexpectedly turns left." if profile.symptom_mode == "free_text" else tax["symptom"][0]
    if profile.symptom_mode == "multi_label":
        symptom = " || ".join(sorted(tax["symptom"][:2]))
    labels = {"symptom": symptom, "root_cause": tax["root_cause"][0]}
    records = [
        {"record_id": "a", "paper_id": profile.paper_id, "issue_url": "https://example.invalid/1",
         "title": "Fault", "body": "The observed failure is recorded.", "decision": "accepted_fault", **labels},
        {"record_id": "b", "paper_id": profile.paper_id, "issue_url": "https://example.invalid/2",
         "title": "Request", "body": "This is a feature request.", "decision": "rejected_candidate", "symptom": "", "root_cause": ""},
    ]
    base = tmp_path / domain
    base.mkdir()
    for filename, rows in (("cohort.csv", records), ("stage3_sample.csv", records[:1])):
        with (base / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(rows)
    (base / "taxonomy.json").write_text(json.dumps(tax), encoding="utf-8")
    for stage, rows in (("stage2", records), ("stage3", records[:1])):
        system = papers.build_stage2_system_prompt(domain) if stage == "stage2" else papers.build_stage3_system_prompt(domain, tax)
        prompts = [{"record_id": row["record_id"], "paper_id": profile.paper_id, "issue_url": row["issue_url"],
                    "ground_truth": row["decision"] if stage == "stage2" else labels,
                    "system_prompt": system, "user_prompt": papers.build_user_prompt(row)} for row in rows]
        (base / f"{stage}_prompts.jsonl").write_text("\n".join(json.dumps(row) for row in prompts), encoding="utf-8")
    return base, labels


@pytest.mark.parametrize("domain", DOMAINS)
def test_all_paper_single_cli_uses_native_metrics_and_bound_manifest(tmp_path, monkeypatch, domain):
    from Benchmark.scripts import run_paper_llm_baseline as runner
    from Benchmark.src import ase2022_stage2_filter_baseline as filter_runner

    base, labels = _prepared_fixture(tmp_path, domain)
    monkeypatch.setattr(runner, "_load_env_file", lambda path: None)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(filter_runner, "call_model", lambda **kwargs: json.dumps({
        "decision": "rejected_candidate" if "feature request" in kwargs["user_prompt"] else "accepted_fault"}))
    monkeypatch.setattr(single, "call_model_with_retries", lambda **kwargs: (json.dumps(labels), 1))
    runner.main(["--domain", domain, "--prepared-root", str(tmp_path), "--model", "test-model"])
    stage2 = json.loads(next((base / "single_llm").glob("*stage2_filter_metrics_*.json")).read_text(encoding="utf-8"))
    stage3 = json.loads(next((base / "single_llm").glob("*stage3_llm_metrics_*.json")).read_text(encoding="utf-8"))
    assert stage2["n"] == 2
    assert stage2["accuracy"] == 1.0
    assert stage3["n"] == 1
    assert stage3["root_cause_accuracy"] == 1.0
    if domain in {"fse2021", "icse2022"}:
        assert stage3["symptom_accuracy"] is None
    if domain == "icse2021":
        assert stage3["symptom_micro_f1"] == 1.0
    manifest = json.loads(next((base / "single_llm").glob("*stage3_llm_run_manifest_*.json")).read_text(encoding="utf-8"))
    assert manifest["run_id"] == stage3["run_id"]
    assert "Benchmark/configs/paper_codebooks_v1.json" in manifest["code_sha256"]
    runner.main(["--domain", domain, "--prepared-root", str(tmp_path), "--model", "test-model"])
    resumed = json.loads(next((base / "single_llm").glob("*stage3_llm_metrics_*.json")).read_text(encoding="utf-8"))
    assert resumed["resumed_count"] == 1
    assert resumed["processed_count"] == 0


@pytest.mark.parametrize("domain", DOMAINS)
def test_all_paper_mas_cli_retains_native_rationales_and_metrics(tmp_path, monkeypatch, domain):
    from Benchmark.scripts import run_paper_camel_mas_baseline as runner

    base, labels = _prepared_fixture(tmp_path, domain)
    monkeypatch.setattr(cli, "_load_env_file", lambda path: None)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    class Agent:
        def __init__(self, text):
            self.text = text

        def step(self, prompt, response_format=None):
            return SimpleNamespace(msgs=[SimpleNamespace(content=self.text, parsed=None)])

    class Society:
        def __init__(self, task):
            negative = "This is a feature request." in task
            body = "This is a feature request." if negative else "The observed failure is recorded."
            rationale = {"reason": "The supplied body supports this decision.",
                         "evidence_refs": [{"field": "body", "quote": body}],
                         "evidence_limitations": "This test response is not a scientific result."}
            result = ({"decision": "rejected_candidate" if negative else "accepted_fault", "decision_rationale": rationale}
                      if "decision_rationale" in task else {**labels, "symptom_rationale": rationale, "root_cause_rationale": rationale})
            self.user_agent = Agent("Instruction: Use the supplied source evidence.")
            self.assistant_agent = Agent(json.dumps(result))
            self.specified_task_prompt = None

        def init_chat(self, init_msg_content=None):
            return SimpleNamespace(content=init_msg_content or "initial")

    monkeypatch.setattr(cli, "make_runner_factories", lambda *args, **kwargs: (Society, None))
    runner.main(["--domain", domain, "--prepared-root", str(tmp_path), "--model", "test-model", "--max-turns", "1", "--no-progress"])
    stage3 = json.loads(next((base / "mas").glob("*stage3_metrics_*.json")).read_text(encoding="utf-8"))
    assert stage3["final"]["root_cause_accuracy"] == 1.0
    if domain in {"fse2021", "icse2022"}:
        assert stage3["final"]["symptom_accuracy"] is None
    if domain == "icse2021":
        assert stage3["final"]["symptom_micro_f1"] == 1.0
    row = json.loads(next((base / "mas").glob("*stage3_predictions_*.jsonl")).read_text(encoding="utf-8"))
    assert row["explanation_mode"] == "evidence_rationale"
    assert row["explanation_audit"]["status"] == "schema_and_citations_valid"
    assert len(row["explanation_audit"]["citations"]) == 2
    manifest = json.loads(next((base / "mas").glob("*run_manifest_*.json")).read_text(encoding="utf-8"))
    assert "Benchmark/configs/paper_codebooks_v1.json" in manifest["code"]["source_sha256"]


def test_generic_mas_help_explains_domain_without_requiring_one(capsys):
    from Benchmark.scripts.run_paper_camel_mas_baseline import main

    with pytest.raises(SystemExit) as result:
        main(["--help"])
    assert result.value.code == 0
    output = capsys.readouterr().out
    assert "--domain" in output
    assert "--explanation-mode" in output


def test_single_no_resume_preserves_existing_run_evidence(tmp_path, monkeypatch):
    from Benchmark.scripts import run_paper_llm_baseline as runner

    base, labels = _prepared_fixture(tmp_path, "fse2021")
    monkeypatch.setattr(runner, "_load_env_file", lambda path: None)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(single, "call_model_with_retries", lambda **kwargs: (json.dumps(labels), 1))
    args = ["--domain", "fse2021", "--prepared-root", str(tmp_path), "--model", "test-model", "--stage", "stage3"]
    runner.main(args)
    output = next((base / "single_llm").glob("*predictions_*.jsonl"))
    before = output.read_bytes()
    with pytest.raises(ValueError, match="new --output-dir"):
        runner.main(args + ["--no-resume"])
    assert output.read_bytes() == before


def test_single_manifest_rejects_changed_input_without_overwriting(tmp_path):
    from Benchmark.scripts.run_paper_llm_baseline import _prepare_manifest

    source, manifest = tmp_path / "input.json", tmp_path / "manifest.json"
    source.write_text("original", encoding="utf-8")
    _prepare_manifest(manifest, [source], [], {"model": "test"})
    before = manifest.read_bytes()
    source.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="new --output-dir"):
        _prepare_manifest(manifest, [source], [], {"model": "test"})
    assert manifest.read_bytes() == before
