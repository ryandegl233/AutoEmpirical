from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from Benchmark.src import ase2022_camel_mas_baseline as mas
from Benchmark.src import mas_rationale


TAXONOMY = {"symptom": ["Crash"], "root_cause": ["API Misuse", "Unknown"]}
RECORD = {
    "record_id": "diagnostic:1", "paper_id": "fixture", "issue_url": "https://example.test/1",
    "title": "Failure at startup", "body": "Call failed: TypeError at startup.",
    "comments": "Fixed by awaiting the Promise.", "state": "closed", "created_at": "2021-01-01",
    "decision": "accepted_fault", "symptom": "Crash", "root_cause": "API Misuse",
}


def explanation(field="body", quote="TypeError"):
    return {"reason": "The cited event supports the selected label.",
            "evidence_refs": [{"field": field, "quote": quote}], "evidence_limitations": ""}


def prediction(stage="stage3"):
    if stage == "stage2":
        return {"decision": "accepted_fault", "decision_rationale": explanation()}
    return {"symptom": "Crash", "root_cause": "API Misuse",
            "symptom_rationale": explanation(),
            "root_cause_rationale": explanation("comments", "awaiting")}


def response(text):
    message = SimpleNamespace(content=text)
    return SimpleNamespace(msg=message, msgs=[message], terminated=False, info={})


class Agent:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.prompts = []

    def step(self, prompt, response_format=None):
        self.prompts.append(prompt)
        output = next(self.outputs)
        if isinstance(output, Exception):
            raise output
        return response(output)


class Society:
    def __init__(self, outputs):
        self.assistant_agent = Agent(outputs)
        self.user_agent = Agent(["Instruction: Classify this record.\nInput: INVENTED CLAIM"] * 5)
        self.specified_task_prompt = ""

    def init_chat(self, init_msg_content=None):
        return SimpleNamespace(content=init_msg_content or "initial task")

    def step(self, message):
        return self.assistant_agent.step(message), self.user_agent.step(message)


def run(outputs, stage="stage3", mode="evidence_anchored", **options):
    society = Society(outputs)
    row = mas.run_roleplaying_society_record(
        RECORD, stage, TAXONOMY, "test-model", lambda task: society,
        society_mode=mode, explanation_mode="evidence_rationale", **options,
    )
    return row, society


@pytest.mark.parametrize("stage", ["stage2", "stage3"])
@pytest.mark.parametrize("mode", ["native", "evidence_anchored"])
@pytest.mark.parametrize("wrapper", ["{}", "```json\n{}\n```", "Solution: {}\nNext request."])
def test_rationale_turn_preserves_labels_and_exact_citation_source(stage, mode, wrapper):
    raw = wrapper.format(json.dumps(prediction(stage)))
    row, _ = run([raw], stage=stage, mode=mode, max_turns=1)
    assert row["invalid"] is False
    assert row["final_prediction"] == ({"decision": "accepted_fault"} if stage == "stage2" else {"symptom": "Crash", "root_cause": "API Misuse"})
    assert row["society"]["raw_final_output"] == raw
    assert row["society"]["parsed_output"] == prediction(stage)
    audit = row["explanation_audit"]
    assert audit["status"] == "schema_and_citations_valid"
    assert audit["semantic_support"] == "not_verified"
    assert audit["source"] == "society_assistant"
    citation = audit["citations"][0]
    assert (citation["field"], citation["quote"], citation["start_char"], citation["end_char"]) == ("body", "TypeError", 13, 22)
    assert "root_cause" not in audit["evidence_fields"]
    assert "INVENTED CLAIM" not in json.dumps(audit["evidence_fields"])


@pytest.mark.parametrize("bad_reference", [
    {"field": "body", "quote": "invented root cause"},
    {"field": "root_cause", "quote": "API Misuse"},
    {"field": "body", "quote": "The issue is caused by misunderstanding"},
    {"field": "comments", "quote": "TypeError"},
    {"field": "body", "quote": "   "},
])
def test_unsupported_citation_is_recorded_and_repaired(bad_reference):
    bad = prediction(); bad["root_cause_rationale"]["evidence_refs"] = [bad_reference]
    bad_raw, good_raw = json.dumps(bad), json.dumps(prediction())
    row, _ = run([bad_raw, good_raw], max_turns=2)
    assert row["invalid"] is False
    assert row["society"]["turn_count"] == 2
    first, second = row["society"]["turns"]
    assert first["assistant"]["content"] == bad_raw
    assert first["assistant"]["parse_error"]
    assert second["is_format_repair"]
    assert "root_cause_rationale" in second["repair_feedback"]
    assert "without explanation" not in second["repair_feedback"]


@pytest.mark.parametrize("field", sorted(mas_rationale.CITABLE_FIELDS))
@pytest.mark.parametrize("marker", [
    "not_fetched", "not_available_in_source", "comments_unavailable_in_source",
    "  NOT_FETCHED \n",
])
def test_source_availability_marker_is_never_a_technical_citation(field, marker):
    payload = prediction("stage2")
    payload["decision_rationale"] = explanation(field, marker.strip())
    value = mas_rationale.Stage2RationaleOutput.model_validate(payload)
    with pytest.raises(ValueError, match="source availability marker"):
        mas_rationale.audit_citations(value, {field: marker}, "stage2")


@pytest.mark.parametrize("marker", ["no_comments_in_source", "[]", " NO_COMMENTS_IN_SOURCE "])
def test_known_empty_comments_are_not_a_technical_citation(marker):
    payload = prediction("stage2")
    payload["decision_rationale"] = explanation("comments", marker.strip())
    value = mas_rationale.Stage2RationaleOutput.model_validate(payload)
    with pytest.raises(ValueError, match="source availability marker"):
        mas_rationale.audit_citations(value, {"comments": marker}, "stage2")


@pytest.mark.parametrize("field,content,quote", [
    ("comments", "The error message is not_fetched after retry.", "not_fetched"),
    ("body", "The function unexpectedly returns [].", "[]"),
    ("body", "[]", "[]"),
])
def test_real_source_mentions_of_markers_remain_exact_citable_text(field, content, quote):
    payload = prediction("stage2")
    payload["decision_rationale"] = explanation(field, quote)
    value = mas_rationale.Stage2RationaleOutput.model_validate(payload)
    _, citations = mas_rationale.audit_citations(value, {field: content}, "stage2")
    assert citations[0]["quote"] == quote


def test_source_marker_citation_is_repaired_before_society_accepts_result():
    record = dict(RECORD, comments="not_fetched")
    bad = prediction("stage2")
    bad["decision_rationale"] = explanation("comments", "not_fetched")
    society = Society([json.dumps(bad), json.dumps(prediction("stage2"))])
    row = mas.run_roleplaying_society_record(
        record, "stage2", TAXONOMY, "test-model", lambda task: society,
        explanation_mode="evidence_rationale", max_turns=2,
    )
    assert row["invalid"] is False
    assert "source availability marker" in row["society"]["turns"][0]["assistant"]["parse_error"]
    assert row["society"]["turns"][1]["is_format_repair"]
    assert row["explanation_audit"]["citations"][0]["field"] == "body"


@pytest.mark.parametrize("change", ["missing", "blank", "bad_label", "empty_without_limit"])
def test_incomplete_explanation_never_counts_as_valid(change):
    payload = prediction()
    if change == "missing":
        del payload["symptom_rationale"]
    elif change == "blank":
        payload["symptom_rationale"]["reason"] = " \n "
    elif change == "bad_label":
        payload["root_cause"] = "unlisted category"
    else:
        payload["root_cause_rationale"]["evidence_refs"] = []
    row, _ = run([json.dumps(payload)], max_turns=1)
    assert row["invalid"] is True
    assert row["final_prediction"] == {}
    assert row["explanation_audit"]["status"] == "invalid"
    assert row["society"]["turns"][0]["assistant"]["content"] == json.dumps(payload)


def test_unknown_can_explain_missing_evidence_without_fabricating_a_quote():
    payload = prediction(); payload["root_cause"] = "Unknown"
    payload["root_cause_rationale"] = {"reason": "No mechanism has been confirmed.",
        "evidence_refs": [], "evidence_limitations": "No diagnosis or tested repair is available."}
    row, _ = run([json.dumps(payload)], max_turns=1)
    assert not row["invalid"]
    assert row["final_prediction"]["root_cause"] == "Unknown"
    assert row["explanation_audit"]["rationales"]["root_cause"]["evidence_refs"] == []


def test_forced_finalizer_retains_every_attempt_and_uses_same_citation_validator():
    bad = prediction(); bad["root_cause_rationale"]["evidence_refs"][0]["quote"] = "fabrication"
    finalizer = Agent([json.dumps(bad), json.dumps(prediction())])
    row, _ = run(["not a classification"], max_turns=1,
        finalizer_factory=lambda role, system: finalizer, finalizer_max_retries=1)
    assert row["invalid"] is False
    assert row["output_source"] == "forced_finalizer"
    assert row["explanation_audit"]["source"] == "forced_finalizer"
    attempts = row["society"]["forced_finalizer"]["attempt_trace"]
    assert [a["raw_output"] for a in attempts] == [json.dumps(bad), json.dumps(prediction())]
    assert attempts[0]["error"] and not attempts[1]["error"]
    assert all("root_cause_rationale" in a["prompt"] for a in attempts)


def test_rationale_run_identity_changes_with_evidence_but_not_gold():
    args = ("model", "stage3", [RECORD], TAXONOMY, 0.0)
    legacy = mas.build_config_hash(*args)
    assert legacy == mas.build_config_hash(*args, explanation_mode="label_only")
    diagnostic = mas.build_config_hash(*args, explanation_mode="evidence_rationale")
    assert diagnostic != legacy
    changed = dict(RECORD, body="A different issue")
    assert diagnostic != mas.build_config_hash("model", "stage3", [changed], TAXONOMY, 0.0, explanation_mode="evidence_rationale")
    changed_gold = dict(RECORD, root_cause="Unknown")
    assert diagnostic == mas.build_config_hash("model", "stage3", [changed_gold], TAXONOMY, 0.0, explanation_mode="evidence_rationale")


def test_resume_rejects_missing_rationale_even_with_matching_hash(tmp_path):
    row, _ = run([json.dumps(prediction())], max_turns=1)
    row.update(config_hash="hash", model="test-model")
    del row["explanation_audit"]
    output = tmp_path / "predictions.jsonl"
    output.write_text(json.dumps(row) + "\n", encoding="utf-8")
    calls = []
    def rerun(record):
        calls.append(record["record_id"])
        replacement, _ = run([json.dumps(prediction())], max_turns=1)
        return replacement
    results = mas.run_stage_records([RECORD], output, "stage3", "test-model", "hash", rerun,
        explanation_mode="evidence_rationale")
    assert calls == ["diagnostic:1"]
    assert results[0]["explanation_audit"]["status"] == "schema_and_citations_valid"


def test_v2_does_not_resume_saved_v1_rationale_audit():
    row, _ = run([json.dumps(prediction())], max_turns=1)
    row.update(config_hash="hash", explanation_contract_version=1)
    assert mas_rationale.CONTRACT_VERSION == 2
    assert not mas._complete_result(row, "stage3", "test-model", "hash", "evidence_rationale")


def test_failed_diagnostic_row_is_saved_even_when_validity_is_required(tmp_path):
    row, _ = run(["bad response"], max_turns=1)
    output = tmp_path / "failed.jsonl"
    with pytest.raises(ValueError):
        mas.run_stage_records([RECORD], output, "stage3", "test-model", "hash", lambda record: row,
            explanation_mode="evidence_rationale", require_valid_json=True, taxonomy=TAXONOMY)
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["invalid"] is True
    assert saved["society"]["turns"][0]["assistant"]["content"] == "bad response"


@pytest.mark.parametrize("tampering", ["reason", "source_turn", "raw_output"])
def test_resume_rejects_audit_not_bound_to_original_model_response(tampering):
    row, _ = run([json.dumps(prediction())], max_turns=1)
    row["config_hash"] = "hash"
    if tampering == "reason":
        changed = "A human supplied this explanation later."
        row["society"]["parsed_output"]["symptom_rationale"]["reason"] = changed
        row["explanation_audit"]["rationales"]["symptom"]["reason"] = changed
    elif tampering == "source_turn":
        row["society"]["turns"][0]["assistant"]["content"] = "different response"
    else:
        row["society"]["raw_final_output"] = "different response"
    assert not mas._complete_result(row, "stage3", "test-model", "hash", "evidence_rationale")
