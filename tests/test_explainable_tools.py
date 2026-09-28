"""Module A: evidence boundaries, three-valued rules, and the real role hook."""
import csv
import hashlib
import json

import pytest

from Benchmark.src.adaptive_empirical_workflow.contracts import EvidenceItem, EvidenceView
from Benchmark.src.ase2022_llm_baseline import ROOT_CAUSE_DEFINITIONS, SYMPTOM_DEFINITIONS


def view(record_id="current", content="The system runs slowly. It terminated unexpectedly."):
    return EvidenceView(
        record_id=record_id, task="Classify", domain_profile="ase2022", ledger_version=1,
        taxonomy={"symptom": list(SYMPTOM_DEFINITIONS), "root_cause": list(ROOT_CAUSE_DEFINITIONS)},
        items=(EvidenceItem(evidence_id="body", record_id=record_id, source_type="issue_body",
                            source_uri="https://example.test/issues/1", retrieved_at="2026-09-15T00:00:00Z",
                            content=content, content_sha256=hashlib.sha256(content.encode()).hexdigest(), explicitness="direct"),),
    )


def fact(condition, quote, status="supported"):
    return {"condition": condition, "status": status, "evidence_id": "body", "quote": quote}


def test_rules_preserve_unknown_conflict_and_competing_candidates():
    from Benchmark.src.adaptive_empirical_workflow.explainable_tools import ExplainableTools, ToolQuery
    tool = ExplainableTools()
    result = tool.run(view(), ToolQuery(facts=[
        fact("performance_problem", "runs slowly"),
        fact("unexpected_termination", "terminated unexpectedly"),
    ]))
    rows = {row["label"]: row for row in result["decision_table"]}
    assert rows["Crash"]["state"] == "supported"
    assert rows["Poor Performance"]["state"] == "unknown"  # running condition is missing
    assert rows["Incorrect Functionality"]["state"] == "unknown"
    assert result["knowledge_status"] == "not_configured"
    assert result["examples"] == []
    both = tool.run(view(), ToolQuery(facts=[
        fact("system_runs", "The system runs"), fact("performance_problem", "runs slowly"),
        fact("unexpected_termination", "terminated unexpectedly"),
    ]))
    assert {r["label"] for r in both["decision_table"] if r["state"] == "supported"} == {"Crash", "Poor Performance"}
    conflict = tool.run(view(), ToolQuery(facts=[
        fact("unexpected_termination", "terminated unexpectedly"),
        fact("unexpected_termination", "The system runs", "contradicted"),
    ]))
    assert next(r for r in conflict["decision_table"] if r["label"] == "Crash")["state"] == "conflict"
    with pytest.raises(ValueError, match="quote"):
        tool.run(view(), ToolQuery(facts=[fact("performance_problem", "fabricated")]))
    with pytest.raises(ValueError):
        tool.run(view(), ToolQuery(facts=[fact("hidden_answer", "runs slowly")]))
    narrowed = view().items[0].model_copy(update={"metadata": {"evidence_capabilities": ["study_scope"]}})
    with pytest.raises(ValueError, match="eligible"):
        tool.run(view().model_copy(update={"items": (narrowed,)}), ToolQuery(facts=[fact("performance_problem", "runs slowly")]))


def write_examples(tmp_path):
    source = tmp_path / "examples.csv"
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["record_id", "issue_url", "body", "comments"])
        writer.writeheader()
        writer.writerow({"record_id": "example-1", "issue_url": "https://example.test/issues/2",
                         "body": "The system runs slowly.", "comments": ""})
    bundle = {"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "examples": [
        {"record_id": "example-1", "query": {"facts": [
            {**fact("performance_problem", "runs slowly"), "evidence_id": "example-1:body"}
        ]}}
    ]}
    path = tmp_path / "examples.json"
    path.write_text(json.dumps(bundle), encoding="utf-8")
    return source, path, bundle


def test_examples_are_source_bound_and_disjoint(tmp_path):
    from Benchmark.src.adaptive_empirical_workflow.explainable_tools import ExplainableTools, ToolQuery
    source, path, bundle = write_examples(tmp_path)
    tool = ExplainableTools.from_files(path, source)
    result = tool.run(view(), ToolQuery())
    assert result["knowledge_status"] == "configured"
    assert result["examples"][0]["source_items"][0]["content"] == "The system runs slowly."
    assert "expected_label" not in json.dumps(result)
    first = result["examples"][0]["facts"][0]
    assert first["source"]["start"] == len("The system ")
    path.write_text(json.dumps({**bundle, "source_sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        ExplainableTools.from_files(path, source)
    path.write_text(json.dumps(bundle), encoding="utf-8")
    with pytest.raises(ValueError, match="overlap"):
        tool.check_evaluation_records([{"record_id": "example-1"}])
    with pytest.raises(ValueError, match="overlap"):
        tool.check_evaluation_records([{"record_id": "alias", "issue_url": "https://EXAMPLE.test/issues/2/?x=y#comment"}])
    with pytest.raises(ValueError, match="overlap"):
        tool.run(view("example-1"), ToolQuery())
    bundle["examples"][0]["root_cause"] = "hidden answer"
    path.write_text(json.dumps(bundle), encoding="utf-8")
    with pytest.raises(ValueError):
        ExplainableTools.from_files(path, source)
    # Even a matching content hash cannot authorize an answer-bearing source column.
    source.write_text("record_id,body,root_cause\nexample-1,text,SECRET\n", encoding="utf-8")
    bundle["examples"][0].pop("root_cause")
    bundle["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    path.write_text(json.dumps(bundle), encoding="utf-8")
    with pytest.raises(ValueError, match="fields"):
        ExplainableTools.from_files(path, source)


def test_role_hook_is_opt_in_and_leaves_verifier_alone():
    from Benchmark.src.adaptive_empirical_workflow.agents import StructuredRoleAgents
    from Benchmark.src.adaptive_empirical_workflow.capabilities import AnalystRole
    from Benchmark.src.adaptive_empirical_workflow.contracts import SymptomReport
    from Benchmark.src.adaptive_empirical_workflow.explainable_tools import ExplainableTools, ToolQuery

    class Client:
        def __init__(self):
            self.calls = []

        def complete(self, **kwargs):
            self.calls.append(kwargs)
            if kwargs["schema"] is ToolQuery:
                query = ToolQuery(facts=[fact("performance_problem", "runs slowly")])
                kwargs["result_validator"](query)
                return query
            return None  # Only capture the role boundary; existing tests cover report validation.

        def telemetry(self, **kwargs):
            return {"calls": len(self.calls)}

    off, on = Client(), Client()
    plain = StructuredRoleAgents(off)
    enabled = StructuredRoleAgents(on, explainable_tools=ExplainableTools())
    for agents in (plain, enabled):
        agents._role(AnalystRole.SYMPTOM_ANALYST, team_id="A", view=view(), schema=SymptomReport)
    assert len(off.calls) == 1 and len(on.calls) == 2
    assert "module_a" not in off.calls[0]["user_prompt"]
    payload = json.loads(on.calls[1]["user_prompt"])
    assert payload["context"]["module_a"]["decision_table"]
    assert enabled.telemetry(record_id="current")["module_a_calls"][0]["query"]["facts"]
    assert enabled.telemetry(record_id="different")["module_a_calls"] == []
    on.calls.clear()
    off.calls.clear()
    for agents in (plain, enabled):
        agents._role(AnalystRole.SYMPTOM_VERIFIER, team_id="A", view=view(), schema=SymptomReport)
    assert len(on.calls) == 1
    assert on.calls[0]["system_prompt"] == off.calls[0]["system_prompt"]
    assert on.calls[0]["user_prompt"] == off.calls[0]["user_prompt"]


def test_cli_rejects_unsupported_module_a_modes():
    from Benchmark.scripts.run_adaptive_empirical_workflow import build_parser, run_cli
    assert build_parser().parse_args(["--domain", "ase2022", "--module-a"]).module_a
    with pytest.raises(ValueError, match="Module A"):
        run_cli(["--domain", "ase2022", "--stage", "stage2", "--module-a", "--dry-run"])


def test_structured_query_retry_then_valid_classification():
    from Benchmark.src.adaptive_empirical_workflow.agents import StructuredModelClient, StructuredRoleAgents
    from Benchmark.src.adaptive_empirical_workflow.explainable_tools import ExplainableTools
    calls = []

    def transport(system, user, *, options):
        calls.append(options.role)
        if options.role == "explainable_tool_query":
            return json.dumps({"facts": [fact("unexpected_termination",
                                              "fabricated" if len(calls) == 1 else "terminated unexpectedly")]})
        assert "module_a" in user and "fabricated" not in user
        return json.dumps({"label": "Crash", "behavior_claim": "The program terminated unexpectedly.",
                           "supporting_evidence_ids": ["body"], "alternative_label": "Poor Performance",
                           "boundary_reason": "Termination is explicit; the performance observation may coexist.",
                           "boundary_evidence_ids": ["body"], "confidence": 0.7,
                           "evidence_sufficiency": "sufficient"})

    agents = StructuredRoleAgents(StructuredModelClient(transport, max_schema_retries=1, retry_delay_seconds=0),
                                  explainable_tools=ExplainableTools())
    report = agents.symptom_analyst("A", view())
    assert report.label == "Crash"
    assert calls == ["explainable_tool_query", "explainable_tool_query", "symptom_analyst"]


def test_cli_rejects_answer_columns_before_domain_loader(tmp_path, monkeypatch):
    from Benchmark.scripts import run_adaptive_empirical_workflow as runner
    source = tmp_path / "bad.csv"
    source.write_text("record_id,issue_url,body,human_answer\nx,https://example.test/1,text,SECRET\n", encoding="utf-8")
    def forbidden_loader(*args, **kwargs):
        pytest.fail("general domain loader must not see an unapproved cohort")
    monkeypatch.setattr(runner, "load_domain_inputs", forbidden_loader)
    with pytest.raises(ValueError, match="fields"):
        runner.run_cli(["--domain", "ase2022", "--stage", "stage3", "--module-a", "--cohort-path", str(source), "--dry-run"])


@pytest.mark.parametrize("enable_a,enable_b", [(False, False), (True, False), (False, True), (True, True)])
def test_evidence_only_run_finishes_without_requesting_gold(tmp_path, monkeypatch, enable_a, enable_b):
    from types import SimpleNamespace
    from Benchmark.scripts import run_adaptive_empirical_workflow as runner
    source, _, _ = write_examples(tmp_path)
    records = list(csv.DictReader(source.open(encoding="utf-8")))
    taxonomy = {"symptom": list(SYMPTOM_DEFINITIONS), "root_cause": list(ROOT_CAUSE_DEFINITIONS)}
    taxonomy_path = tmp_path / "taxonomy.json"
    taxonomy_path.write_text(json.dumps(taxonomy), encoding="utf-8")
    split = tmp_path / "split.json"
    split.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(runner, "load_domain_inputs", lambda *a, **k:
                        SimpleNamespace(records=records, taxonomy=taxonomy, profile=None))
    def run_record(record, **kwargs):
        agents = kwargs["agents"]
        assert (agents._explainable_tools is not None) == enable_a
        assert agents._chain_checker_enabled == enable_b
        return {"record_id": record["record_id"], "stage3_valid": False, "stop_reason": "test_unresolved"}
    monkeypatch.setattr(runner, "run_adaptive_record", run_record)
    result = runner.run_cli([
        "--domain", "ase2022", "--stage", "stage3", "--cohort-path", str(source),
        "--taxonomy-path", str(taxonomy_path), "--split-manifest", str(split),
        "--output-dir", str(tmp_path / "runs"), "--no-progress",
        *(["--module-a"] if enable_a else []),
        *(["--module-b"] if enable_b else []),
    ], transport_override=lambda *a, **k: pytest.fail("No model call needed for the metrics regression"))
    assert result["metrics"]["evaluation"]["status"] == "not_evaluated"
    assert result["metrics"]["stage3"]["n"] == 1
    assert result["metrics"]["stage3"]["invalid_count"] == 1
    assert "joint_accuracy" not in result["metrics"]["stage3"]
    assert "dual_team" not in result["metrics"]["stage3"]  # This helper includes GT-dependent accuracy.
