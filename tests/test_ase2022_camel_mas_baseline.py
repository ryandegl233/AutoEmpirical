from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from Benchmark.src import ase2022_camel_mas_baseline as mas


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _record(record_id: str, **extra: str) -> dict[str, str]:
    return {
        "record_id": record_id,
        "paper_id": mas.ASE2022_PAPER_ID,
        "issue_url": f"https://example.test/{record_id}",
        "title": f"title {record_id}",
        "body": f"body {record_id}",
        "comments": "[]",
        "state": "closed",
        "created_at": "2021-01-01T00:00:00Z",
        **extra,
    }


def test_build_unified_cohort_uses_stage3_positives_and_stage2_negatives(
    tmp_path: Path,
) -> None:
    stage2_path = tmp_path / "stage2_sample.csv"
    stage3_path = tmp_path / "stage3_sample.csv"
    _write_csv(
        stage2_path,
        [
            _record("old-positive", decision=mas.ACCEPTED_FAULT),
            _record("n1", decision=mas.REJECTED_CANDIDATE),
            _record("n2", decision=mas.REJECTED_CANDIDATE),
        ],
    )
    _write_csv(
        stage3_path,
        [
            _record("p1", symptom="Crash", root_cause="Incorrect Code Logic"),
            _record("p2", symptom="Poor Performance", root_cause="WebGL Limits"),
        ],
    )

    cohort = mas.build_unified_cohort(
        stage2_sample_path=stage2_path,
        stage3_sample_path=stage3_path,
        positives=2,
        negatives=2,
    )

    assert [row["record_id"] for row in cohort] == ["p1", "p2", "n1", "n2"]
    assert [row["decision"] for row in cohort] == [
        mas.ACCEPTED_FAULT,
        mas.ACCEPTED_FAULT,
        mas.REJECTED_CANDIDATE,
        mas.REJECTED_CANDIDATE,
    ]
    assert cohort[0]["symptom"] == "Crash"
    assert cohort[2]["symptom"] == ""
    assert "old-positive" not in repr(cohort)


def test_build_unified_cohort_rejects_positive_without_stage3_labels(
    tmp_path: Path,
) -> None:
    stage2_path = tmp_path / "stage2_sample.csv"
    stage3_path = tmp_path / "stage3_sample.csv"
    _write_csv(stage2_path, [_record("n", decision=mas.REJECTED_CANDIDATE)])
    _write_csv(stage3_path, [_record("p", symptom="", root_cause="Cause")])

    with pytest.raises(ValueError, match="Stage 3 labels"):
        mas.build_unified_cohort(stage2_path, stage3_path, positives=1, negatives=1)


def test_parse_role_output_accepts_markdown_fenced_json() -> None:
    parsed = mas.parse_role_output(
        '```json\n{"decision":"accepted_fault"}\n```',
        mas.Stage2ProposerOutput,
    )

    assert parsed["value"].decision == mas.ACCEPTED_FAULT
    assert parsed["format_normalized"] is True
    assert parsed["output_format"] == "markdown_fenced_json"


def test_parse_role_output_rejects_extra_fields_and_invalid_labels() -> None:
    with pytest.raises(ValueError):
        mas.parse_role_output(
            '{"decision":"accepted_fault","reason":"extra"}',
            mas.Stage2ProposerOutput,
        )
    with pytest.raises(ValueError):
        mas.parse_role_output('{"decision":"maybe"}', mas.Stage2ProposerOutput)


def test_parse_society_prediction_accepts_camel_wrapper() -> None:
    parsed = mas.parse_society_prediction(
        'Solution: {"decision":"accepted_fault"}\nNext request.',
        mas.Stage2SocietyOutput,
    )

    assert parsed["value"].decision == mas.ACCEPTED_FAULT
    assert parsed["format_normalized"] is True
    assert parsed["output_format"] == "camel_wrapped_json"


def test_parse_society_prediction_accepts_fenced_json_inside_camel_wrapper() -> None:
    parsed = mas.parse_society_prediction(
        "Solution:\n```json\n"
        '{"symptom":"Crash","root_cause":"Incorrect Code Logic"}\n'
        "```\nNext request.",
        mas.Stage3SocietyOutput,
    )

    assert parsed["value"].symptom == "Crash"
    assert parsed["value"].root_cause == "Incorrect Code Logic"
    assert parsed["output_format"] == "camel_wrapped_json"


def test_parse_society_prediction_rejects_wrapper_without_valid_schema() -> None:
    with pytest.raises(ValueError, match="no valid Society prediction"):
        mas.parse_society_prediction(
            'Solution: {"decision":"maybe"}\nNext request.',
            mas.Stage2SocietyOutput,
        )


def test_society_task_excludes_gold_fields_and_includes_taxonomy() -> None:
    task = mas.build_society_task(
        _record(
            "r1",
            decision=mas.ACCEPTED_FAULT,
            symptom="LEAKED_SYMPTOM",
            root_cause="LEAKED_CAUSE",
        ),
        "stage3",
        {
            "symptom": ["Crash"],
            "root_cause": ["Incorrect Code Logic"],
        },
    )

    assert "LEAKED_SYMPTOM" not in task
    assert "LEAKED_CAUSE" not in task
    assert "Crash" in task
    assert "Incorrect Code Logic" in task
    assert "title r1" in task


class _FakeAgent:
    def __init__(self, responses: list[str], prompts: list[str]) -> None:
        self._responses = responses
        self._prompts = prompts

    def step(self, prompt: str, response_format=None):
        self._prompts.append(prompt)
        content = self._responses.pop(0)
        return SimpleNamespace(msgs=[SimpleNamespace(content=content, parsed=None)])


class _TrackedFinalizerAgent(_FakeAgent):
    def __init__(
        self,
        responses: list[str],
        prompts: list[str],
        *,
        tokens_per_request: int = 100,
    ) -> None:
        super().__init__(responses, prompts)
        self.tokens_per_request = tokens_per_request
        self._mas_request_stats = {
            "api_request_count": 0,
            "usage_observed_request_count": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "latency_seconds": 0.0,
        }

    def step(self, prompt: str, response_format=None):
        stats = self._mas_request_stats
        stats["api_request_count"] += 1
        stats["usage_observed_request_count"] += 1
        stats["prompt_tokens"] += self.tokens_per_request - 20
        stats["completion_tokens"] += 20
        stats["total_tokens"] += self.tokens_per_request
        return super().step(prompt, response_format=response_format)


def _society_response(content: str, *, terminated: bool = False):
    return SimpleNamespace(
        msg=SimpleNamespace(content=content),
        terminated=terminated,
        info={"termination_reasons": ["test"] if terminated else []},
    )


class _FakeSociety:
    def __init__(
        self,
        turns: list[tuple[SimpleNamespace, SimpleNamespace]],
        *,
        specified_task_prompt: str = "specified task",
    ) -> None:
        self._turns = list(turns)
        self.specified_task_prompt = specified_task_prompt
        self.step_inputs: list[object] = []

    def init_chat(self):
        return SimpleNamespace(content="initial task")

    def step(self, input_msg):
        self.step_inputs.append(input_msg)
        return self._turns.pop(0)


class _AnchoredFakeSociety:
    def __init__(self, user_agent: _FakeAgent, assistant_agent: _FakeAgent) -> None:
        self.user_agent = user_agent
        self.assistant_agent = assistant_agent
        self.specified_task_prompt = None

    def init_chat(self, init_msg_content=None):
        return SimpleNamespace(content=init_msg_content or "initial task")


def test_split_user_instruction_discards_generated_input() -> None:
    instruction, discarded = mas._split_user_instruction(
        "Instruction: Analyze the exact shared record.\n\n"
        "Input: Issue Title: Model loss becomes NaN after ten epochs"
    )

    assert instruction == "Analyze the exact shared record."
    assert "Model loss becomes NaN" in discarded
    assert "Model loss becomes NaN" not in instruction


def test_evidence_anchored_turn_removes_fabricated_input_from_assistant_prompt() -> None:
    user_prompts: list[object] = []
    assistant_prompts: list[str] = []
    user_agent = _FakeAgent(
        [
            "Instruction: Decide whether the shared record is a fault.\n"
            "Input: Issue Title: Model loss becomes NaN after ten epochs"
        ],
        user_prompts,
    )
    assistant_agent = _FakeAgent(
        ['{"decision":"accepted_fault"}'],
        assistant_prompts,
    )
    society = _AnchoredFakeSociety(user_agent, assistant_agent)
    record = _record(
        "r1",
        title="[Codelab]: Making Predictions from 2D Data",
        body="script executes before body and document.body is null",
        decision="LEAKED_DECISION",
    )

    row = mas.run_roleplaying_society_record(
        record,
        stage="stage2",
        taxonomy={"symptom": ["Crash"], "root_cause": ["Cause"]},
        model="test-model",
        society_factory=lambda _task: society,
        society_mode="evidence_anchored",
        max_turns=1,
    )

    assistant_input = assistant_prompts[0]
    assert "[Codelab]: Making Predictions from 2D Data" in assistant_input
    assert "document.body is null" in assistant_input
    assert "r1" in assistant_input
    assert "Model loss becomes NaN" not in assistant_input
    assert "LEAKED_DECISION" not in assistant_input
    assert row["architecture"] == "camel_roleplaying_evidence_anchored"
    assert row["society_mode"] == "evidence_anchored"
    assert row["society"]["immutable_evidence_sha256"] in assistant_input
    turn = row["society"]["turns"][0]
    assert turn["user_instruction"] == "Decide whether the shared record is a fault."
    assert "Model loss becomes NaN" in turn["discarded_user_input"]
    assert turn["assistant_input"] == assistant_input


def test_society_record_stops_immediately_on_first_valid_json() -> None:
    society = _FakeSociety(
        [
            (
                _society_response(
                    'Solution: {"decision":"accepted_fault"}\nNext request.'
                ),
                _society_response("CAMEL_TASK_DONE"),
            )
        ]
    )

    row = mas.run_roleplaying_society_record(
        _record("r1"),
        stage="stage2",
        taxonomy={"symptom": ["Crash"], "root_cause": ["Cause"]},
        model="test-model",
        society_factory=lambda _task: society,
        society_mode="native",
        max_turns=5,
    )

    assert row["final_prediction"] == {"decision": mas.ACCEPTED_FAULT}
    assert row["invalid"] is False
    assert row["society"]["turn_count"] == 1
    assert row["society"]["stop_reason"] == "valid_json"
    assert row["society"]["final_answer_turn"] == 1
    assert row["society"]["specified_task_prompt"] == "specified task"


def test_society_record_accepts_a_paper_specific_task_builder() -> None:
    captured_tasks: list[str] = []
    society = _FakeSociety(
        [
            (
                _society_response(
                    'Solution: {"decision":"accepted_fault"}\nNext request.'
                ),
                _society_response("CAMEL_TASK_DONE"),
            )
        ]
    )

    def task_builder(record, stage, taxonomy):
        assert record["record_id"] == "r1"
        assert stage == "stage2"
        assert taxonomy["symptom"] == ["Crash"]
        return "CUSTOM ISSTA CONTAINER RUNTIME TASK"

    row = mas.run_roleplaying_society_record(
        _record("r1"),
        stage="stage2",
        taxonomy={"symptom": ["Crash"], "root_cause": ["Cause"]},
        model="test-model",
        society_factory=lambda task: captured_tasks.append(task) or society,
        society_mode="native",
        max_turns=1,
        task_builder=task_builder,
    )

    assert captured_tasks == ["CUSTOM ISSTA CONTAINER RUNTIME TASK"]
    assert row["invalid"] is False


def test_society_record_ignores_early_done_and_repairs_invalid_json() -> None:
    society = _FakeSociety(
        [
            (
                _society_response("Solution: classification is accepted."),
                _society_response("CAMEL_TASK_DONE"),
            ),
            (
                _society_response('{"decision":"accepted_fault"}'),
                _society_response("CAMEL_TASK_DONE"),
            ),
        ]
    )

    row = mas.run_roleplaying_society_record(
        _record("r1"),
        stage="stage2",
        taxonomy={"symptom": ["Crash"], "root_cause": ["Cause"]},
        model="test-model",
        society_factory=lambda _task: society,
        society_mode="native",
        max_turns=5,
    )

    assert row["final_prediction"] == {"decision": mas.ACCEPTED_FAULT}
    assert row["invalid"] is False
    assert row["society"]["turn_count"] == 2
    assert row["society"]["stop_reason"] == "valid_json"
    assert row["society"]["final_answer_turn"] == 2
    assert row["society"]["ignored_completion_signal_count"] == 1
    assert row["society"]["turns"][0]["is_format_repair"] is False
    assert row["society"]["turns"][1]["is_format_repair"] is True
    repair_feedback = row["society"]["turns"][1]["repair_feedback"]
    assert '"decision"' in repair_feedback
    assert "accepted_fault" in repair_feedback
    assert "rejected_candidate" in repair_feedback
    assert "no valid Society prediction" in repair_feedback
    assert "You are the AI User" in repair_feedback
    assert "issue a new Instruction to the AI Assistant" in repair_feedback
    assert "Do not answer the classification yourself" in repair_feedback
    assert society.step_inputs[1].content == repair_feedback


def test_society_record_marks_missing_valid_answer_invalid() -> None:
    society = _FakeSociety(
        [
            (
                _society_response("Solution: classification is unclear."),
                _society_response("CAMEL_TASK_DONE"),
            )
        ]
    )

    row = mas.run_roleplaying_society_record(
        _record("r1"),
        stage="stage2",
        taxonomy={"symptom": ["Crash"], "root_cause": ["Cause"]},
        model="test-model",
        society_factory=lambda _task: society,
        society_mode="native",
        max_turns=1,
    )

    assert row["final_prediction"] == {}
    assert row["invalid"] is True
    assert row["error"] == "no valid Assistant prediction after 1 turns"
    assert row["society"]["stop_reason"] == "max_turns"
    assert row["society"]["turns"][0]["assistant"]["parse_error"]


def test_stage3_society_validates_taxonomy_and_does_not_leak_gold() -> None:
    tasks: list[str] = []
    society = _FakeSociety(
        [
            (
                _society_response(
                    'Solution: {"symptom":"Crash",'
                    '"root_cause":"Incorrect Code Logic"}\nNext request.'
                ),
                _society_response("CAMEL_TASK_DONE"),
            )
        ]
    )

    def factory(task: str):
        tasks.append(task)
        return society

    row = mas.run_roleplaying_society_record(
        _record(
            "r1",
            decision=mas.ACCEPTED_FAULT,
            symptom="LEAKED_SYMPTOM",
            root_cause="LEAKED_CAUSE",
        ),
        stage="stage3",
        taxonomy={
            "symptom": ["Crash"],
            "root_cause": ["Incorrect Code Logic"],
        },
        model="test-model",
        society_factory=factory,
        society_mode="native",
        max_turns=5,
    )

    assert row["final_prediction"] == {
        "symptom": "Crash",
        "root_cause": "Incorrect Code Logic",
    }
    assert "LEAKED_SYMPTOM" not in tasks[0]
    assert "LEAKED_CAUSE" not in tasks[0]


def test_stage3_society_repairs_unsupported_taxonomy_label_without_gold_leak() -> None:
    society = _FakeSociety(
        [
            (
                _society_response(
                    '{"symptom":"Crash","root_cause":"Null Pointer"}'
                ),
                _society_response("CAMEL_TASK_DONE"),
            ),
            (
                _society_response(
                    '{"symptom":"Crash",'
                    '"root_cause":"Incorrect Code Logic"}'
                ),
                _society_response("CAMEL_TASK_DONE"),
            ),
        ]
    )

    row = mas.run_roleplaying_society_record(
        _record(
            "r1",
            decision=mas.ACCEPTED_FAULT,
            symptom="LEAKED_SYMPTOM",
            root_cause="LEAKED_CAUSE",
        ),
        stage="stage3",
        taxonomy={
            "symptom": ["Crash", "Document Error"],
            "root_cause": ["Incorrect Code Logic", "Confused Document"],
        },
        model="test-model",
        society_factory=lambda _task: society,
        society_mode="native",
        max_turns=5,
    )

    assert row["final_prediction"] == {
        "symptom": "Crash",
        "root_cause": "Incorrect Code Logic",
    }
    assert row["society"]["turn_count"] == 2
    repair_feedback = row["society"]["turns"][1]["repair_feedback"]
    assert "unsupported root_cause label: Null Pointer" in repair_feedback
    assert '"Crash"' in repair_feedback
    assert '"Incorrect Code Logic"' in repair_feedback
    assert '"Confused Document"' in repair_feedback
    assert "LEAKED_SYMPTOM" not in repair_feedback
    assert "LEAKED_CAUSE" not in repair_feedback


def test_society_record_exhausts_turn_cap_after_only_invalid_answers() -> None:
    society = _FakeSociety(
        [
            (
                _society_response("The issue looks like a crash."),
                _society_response("CAMEL_TASK_DONE"),
            ),
            (
                _society_response("I confirm the previous analysis."),
                _society_response("CAMEL_TASK_DONE"),
            ),
        ]
    )

    row = mas.run_roleplaying_society_record(
        _record("r1"),
        stage="stage2",
        taxonomy={"symptom": ["Crash"], "root_cause": ["Cause"]},
        model="test-model",
        society_factory=lambda _task: society,
        society_mode="native",
        max_turns=2,
    )

    assert len(society.step_inputs) == 2
    assert row["invalid"] is True
    assert row["society"]["stop_reason"] == "max_turns"
    assert row["society"]["ignored_completion_signal_count"] == 2
    assert row["society"]["final_parse_error"]
    assert row["error"] == "no valid Assistant prediction after 2 turns"


def test_society_uses_fresh_forced_finalizer_after_turn_cap() -> None:
    society = _FakeSociety(
        [
            (
                _society_response("The issue appears to be a fault."),
                _society_response("Continue the analysis."),
            ),
            (
                _society_response("I agree, but here is no JSON."),
                _society_response("CAMEL_TASK_DONE"),
            ),
        ]
    )
    finalizer_prompts: list[str] = []
    finalizer = _FakeAgent(
        ['{"decision":"accepted_fault"}'],
        finalizer_prompts,
    )
    created: list[tuple[str, str]] = []

    def finalizer_factory(role: str, system_message: str):
        created.append((role, system_message))
        return finalizer

    row = mas.run_roleplaying_society_record(
        _record("r1", decision="LEAKED_DECISION"),
        stage="stage2",
        taxonomy={"symptom": ["Crash"], "root_cause": ["Cause"]},
        model="test-model",
        society_factory=lambda _task: society,
        society_mode="native",
        finalizer_factory=finalizer_factory,
        finalizer_max_retries=2,
        max_turns=2,
    )

    assert row["final_prediction"] == {"decision": mas.ACCEPTED_FAULT}
    assert row["invalid"] is False
    assert row["output_source"] == "forced_finalizer"
    assert row["society"]["turn_count"] == 2
    assert row["society"]["final_answer_turn"] is None
    assert row["society"]["stop_reason"] == "forced_finalizer_json"
    assert row["society"]["forced_finalization_attempted"] is True
    assert row["society"]["forced_finalizer"]["attempts"] == 1
    assert created[0][0] == "forced_finalizer"
    combined_prompt = created[0][1] + "\n" + finalizer_prompts[0]
    assert "title r1" in combined_prompt
    assert "body r1" in combined_prompt
    assert "I agree, but here is no JSON." in combined_prompt
    assert "LEAKED_DECISION" not in combined_prompt


def test_stage3_forced_finalizer_retries_unsupported_label_then_succeeds() -> None:
    society = _FakeSociety(
        [
            (
                _society_response("No structured answer."),
                _society_response("CAMEL_TASK_DONE"),
            )
        ]
    )
    finalizer = _FakeAgent(
        [
            '{"symptom":"Crash","root_cause":"Null Pointer"}',
            '{"symptom":"Crash","root_cause":"Incorrect Code Logic"}',
        ],
        [],
    )

    row = mas.run_roleplaying_society_record(
        _record("r1"),
        stage="stage3",
        taxonomy={
            "symptom": ["Crash"],
            "root_cause": ["Incorrect Code Logic"],
        },
        model="test-model",
        society_factory=lambda _task: society,
        society_mode="native",
        finalizer_factory=lambda _role, _system: finalizer,
        finalizer_max_retries=1,
        max_turns=1,
    )

    assert row["final_prediction"] == {
        "symptom": "Crash",
        "root_cause": "Incorrect Code Logic",
    }
    metadata = row["society"]["forced_finalizer"]
    assert metadata["attempts"] == 2
    assert "unsupported root_cause label: Null Pointer" in metadata["errors"][0]


def test_forced_finalizer_failure_preserves_errors_and_invalid_result() -> None:
    society = _FakeSociety(
        [
            (
                _society_response("No structured answer."),
                _society_response("CAMEL_TASK_DONE"),
            )
        ]
    )
    finalizer = _FakeAgent(
        ["Still prose.", "Still not JSON."],
        [],
    )

    row = mas.run_roleplaying_society_record(
        _record("r1"),
        stage="stage2",
        taxonomy={"symptom": ["Crash"], "root_cause": ["Cause"]},
        model="test-model",
        society_factory=lambda _task: society,
        society_mode="native",
        finalizer_factory=lambda _role, _system: finalizer,
        finalizer_max_retries=1,
        max_turns=1,
    )

    assert row["final_prediction"] == {}
    assert row["invalid"] is True
    assert row["output_source"] == ""
    assert row["society"]["stop_reason"] == "forced_finalizer_failed"
    metadata = row["society"]["forced_finalizer"]
    assert metadata["attempts"] == 2
    assert metadata["raw_output"] == "Still not JSON."
    assert len(metadata["errors"]) == 2
    assert "invalid JSON" in row["error"]


def test_forced_finalizer_cost_is_included_in_society_totals() -> None:
    society = _FakeSociety(
        [
            (
                _society_response("No structured answer."),
                _society_response("CAMEL_TASK_DONE"),
            )
        ]
    )
    finalizer = _TrackedFinalizerAgent(
        ["Still prose.", '{"decision":"accepted_fault"}'],
        [],
        tokens_per_request=120,
    )

    row = mas.run_roleplaying_society_record(
        _record("r1"),
        stage="stage2",
        taxonomy={"symptom": ["Crash"], "root_cause": ["Cause"]},
        model="test-model",
        society_factory=lambda _task: society,
        society_mode="native",
        finalizer_factory=lambda _role, _system: finalizer,
        finalizer_max_retries=1,
        max_turns=1,
    )

    assert row["society"]["forced_finalizer"]["api_request_count"] == 2
    assert row["society"]["api_request_count"] == 2
    assert row["society"]["token_usage"]["total_tokens"] == 240
    finalizer_stats = row["society"]["role_request_stats"]["forced_finalizer"]
    assert {
        key: value
        for key, value in finalizer_stats.items()
        if key != "latency_seconds"
    } == {
        "api_request_count": 2,
        "usage_observed_request_count": 2,
        "prompt_tokens": 200,
        "completion_tokens": 40,
        "total_tokens": 240,
    }
    assert finalizer_stats["latency_seconds"] >= 0.0


def test_native_camel_society_factory_preserves_roleplaying_defaults(monkeypatch) -> None:
    import camel.models
    import camel.societies
    import openai

    captured: dict[str, object] = {}

    def fake_role_playing(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(specified_task_prompt="specified")

    completions = SimpleNamespace(create=lambda **_kwargs: None)
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions),
        beta=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )
    monkeypatch.setattr(camel.societies, "RolePlaying", fake_role_playing)
    monkeypatch.setattr(
        camel.models.ModelFactory,
        "create",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(openai, "OpenAI", lambda **_kwargs: fake_client)

    factory = mas.make_camel_society_factory(
        "test-model",
        "test-key",
        "https://example.test/v1",
        society_mode="native",
    )
    society = factory("original task")

    assert captured["assistant_role_name"] == "Software Fault Analyst"
    assert captured["user_role_name"] == "Empirical Software Researcher"
    assert captured["task_prompt"] == "original task"
    assert "with_task_specify" not in captured
    assert "with_task_planner" not in captured
    assert "with_critic_in_the_loop" not in captured
    assert set(society._mas_role_request_stats) == {
        "task_specifier",
        "ai_user",
        "ai_assistant",
    }


def test_default_evidence_anchored_factory_disables_task_specifier(monkeypatch) -> None:
    import camel.models
    import camel.societies
    import openai

    captured: dict[str, object] = {}
    backend_calls: list[dict[str, object]] = []

    def fake_role_playing(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(specified_task_prompt=None)

    completions = SimpleNamespace(create=lambda **_kwargs: None)
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions),
        beta=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )
    monkeypatch.setattr(camel.societies, "RolePlaying", fake_role_playing)
    monkeypatch.setattr(
        camel.models.ModelFactory,
        "create",
        lambda **kwargs: backend_calls.append(kwargs) or object(),
    )
    monkeypatch.setattr(openai, "OpenAI", lambda **_kwargs: fake_client)

    factory = mas.make_camel_society_factory(
        "test-model",
        "test-key",
        "https://example.test/v1",
    )
    society = factory("immutable original task")

    assert getattr(mas, "DEFAULT_SOCIETY_MODE", None) == "evidence_anchored"
    assert captured["with_task_specify"] is False
    assert captured["task_prompt"] == "immutable original task"
    assert len(backend_calls) == 2
    assert set(society._mas_role_request_stats) == {"ai_user", "ai_assistant"}


def test_escape_task_for_camel_specifier_survives_two_format_passes() -> None:
    from camel.prompts import TextPrompt

    original = '{"symptom":"Crash","root_cause":"Incorrect Code Logic"}'
    escaped = mas._escape_task_for_camel_specifier(original)
    first_pass = TextPrompt(
        "Task: {task}\nAssistant: {assistant_role}"
    ).format(task=escaped)
    second_pass = first_pass.format(assistant_role="Software Fault Analyst")

    assert original in second_pass


def test_stage2_record_runs_fixed_roles_without_leaking_gold() -> None:
    prompts: list[str] = []
    created_roles: list[str] = []
    responses = {
        "proposer": ['{"decision":"accepted_fault"}'],
        "critic": [
            '{"verdict":"revise","suggested_decision":"rejected_candidate",'
            '"evidence":["feature request"],"reason":"No observed failure"}'
        ],
        "judge": [
            '{"decision":"rejected_candidate","evidence":["feature request"],'
            '"reason":"The record requests a capability"}'
        ],
    }

    def factory(role: str, _system_message: str):
        created_roles.append(role)
        return _FakeAgent(responses[role], prompts)

    row = mas.run_three_role_record(
        _record(
            "r1",
            decision=mas.ACCEPTED_FAULT,
            symptom="LEAKED_SYMPTOM",
            root_cause="LEAKED_CAUSE",
        ),
        stage="stage2",
        taxonomy={"symptom": ["Crash"], "root_cause": ["Cause"]},
        model="test-model",
        agent_factory=factory,
        max_retries=0,
        backend_id="deepseek:https://api.deepseek.com",
    )

    assert created_roles == ["proposer", "critic", "judge"]
    assert row["final_prediction"] == {"decision": mas.REJECTED_CANDIDATE}
    assert row["backend_id"] == "deepseek:https://api.deepseek.com"
    assert row["invalid"] is False
    all_prompts = "\n".join(prompts)
    assert "LEAKED_SYMPTOM" not in all_prompts
    assert "LEAKED_CAUSE" not in all_prompts
    assert mas.ACCEPTED_FAULT not in prompts[0]


def test_stage3_all_roles_receive_the_exact_taxonomy() -> None:
    systems: dict[str, str] = {}
    responses = {
        "proposer": ['{"symptom":"Crash","root_cause":"Incorrect Code Logic"}'],
        "critic": [
            '{"verdict":"uphold","suggested_symptom":"Crash",'
            '"suggested_root_cause":"Incorrect Code Logic","evidence":["exception"],'
            '"reason":"Labels match the evidence"}'
        ],
        "judge": [
            '{"symptom":"Crash","root_cause":"Incorrect Code Logic",'
            '"evidence":["exception"],"reason":"Labels match the evidence"}'
        ],
    }

    def factory(role: str, system_message: str):
        systems[role] = system_message
        return _FakeAgent(responses[role], [])

    row = mas.run_three_role_record(
        _record("r1", symptom="LEAKED_SYMPTOM", root_cause="LEAKED_CAUSE"),
        stage="stage3",
        taxonomy={
            "symptom": ["Crash"],
            "root_cause": ["Incorrect Code Logic"],
        },
        model="test-model",
        agent_factory=factory,
        max_retries=0,
    )

    assert row["invalid"] is False
    for role in ("proposer", "critic", "judge"):
        assert "Crash" in systems[role]
        assert "Incorrect Code Logic" in systems[role]


def test_role_is_retried_after_invalid_format() -> None:
    responses = {
        "proposer": ["not json", '{"decision":"accepted_fault"}'],
        "critic": [
            '{"verdict":"uphold","suggested_decision":"accepted_fault",'
            '"evidence":["crash"],"reason":"Observed failure"}'
        ],
        "judge": [
            '{"decision":"accepted_fault","evidence":["crash"],'
            '"reason":"Observed failure"}'
        ],
    }

    def factory(role: str, _system_message: str):
        return _FakeAgent(responses[role], [])

    row = mas.run_three_role_record(
        _record("r1", decision=mas.ACCEPTED_FAULT),
        stage="stage2",
        taxonomy={"symptom": ["Crash"], "root_cause": ["Cause"]},
        model="test-model",
        agent_factory=factory,
        max_retries=1,
    )

    assert row["roles"]["proposer"]["attempts"] == 2
    assert row["invalid"] is False


def test_exhausted_role_failure_preserves_attempt_metadata() -> None:
    responses = {"proposer": ["not json", "still not json"]}

    def factory(role: str, _system_message: str):
        return _FakeAgent(responses[role], [])

    row = mas.run_three_role_record(
        _record("r1", decision=mas.ACCEPTED_FAULT),
        stage="stage2",
        taxonomy={"symptom": ["Crash"], "root_cause": ["Cause"]},
        model="test-model",
        agent_factory=factory,
        max_retries=1,
    )

    assert row["invalid"] is True
    assert row["roles"]["proposer"]["attempts"] == 2
    assert row["roles"]["proposer"]["output_format"] == "invalid"
    assert len(row["roles"]["proposer"]["errors"]) == 2


def test_end_to_end_metrics_count_stage2_rejection_as_missed_positive() -> None:
    cohort = [
        _record(
            "p-correct",
            decision=mas.ACCEPTED_FAULT,
            symptom="Crash",
            root_cause="Cause",
        ),
        _record(
            "p-rejected",
            decision=mas.ACCEPTED_FAULT,
            symptom="Crash",
            root_cause="Cause",
        ),
        _record("n-fp", decision=mas.REJECTED_CANDIDATE, symptom="", root_cause=""),
    ]
    stage2_rows = [
        {"record_id": "p-correct", "invalid": False, "final_prediction": {"decision": mas.ACCEPTED_FAULT}},
        {"record_id": "p-rejected", "invalid": False, "final_prediction": {"decision": mas.REJECTED_CANDIDATE}},
        {"record_id": "n-fp", "invalid": False, "final_prediction": {"decision": mas.ACCEPTED_FAULT}},
    ]
    stage3_rows = [
        {
            "record_id": "p-correct",
            "invalid": False,
            "final_prediction": {"symptom": "Crash", "root_cause": "Cause"},
        },
        {
            "record_id": "n-fp",
            "invalid": False,
            "final_prediction": {"symptom": "Crash", "root_cause": "Cause"},
        },
    ]

    metrics = mas.evaluate_end_to_end(cohort, stage2_rows, stage3_rows)

    assert metrics["gold_positive_count"] == 2
    assert metrics["entered_stage3_count"] == 2
    assert metrics["complete_correct_count"] == 1
    assert metrics["missed_by_stage2_count"] == 1
    assert metrics["stage2_invalid_count"] == 0
    assert metrics["stage3_invalid_after_entry_count"] == 0
    assert metrics["total_api_requests"] == 0
    assert metrics["total_tokens"] == 0
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5


def test_select_stage3_records_includes_all_positives_and_accepted_negatives() -> None:
    cohort = [
        _record("p1", decision=mas.ACCEPTED_FAULT, symptom="Crash", root_cause="Cause"),
        _record("p2", decision=mas.ACCEPTED_FAULT, symptom="Crash", root_cause="Cause"),
        _record("n1", decision=mas.REJECTED_CANDIDATE, symptom="", root_cause=""),
        _record("n2", decision=mas.REJECTED_CANDIDATE, symptom="", root_cause=""),
    ]
    stage2_rows = [
        {"record_id": "p1", "invalid": False, "final_prediction": {"decision": mas.REJECTED_CANDIDATE}},
        {"record_id": "p2", "invalid": False, "final_prediction": {"decision": mas.ACCEPTED_FAULT}},
        {"record_id": "n1", "invalid": False, "final_prediction": {"decision": mas.ACCEPTED_FAULT}},
        {"record_id": "n2", "invalid": False, "final_prediction": {"decision": mas.REJECTED_CANDIDATE}},
    ]

    selected = mas.select_stage3_records(cohort, stage2_rows)

    assert [row["record_id"] for row in selected] == ["p1", "p2", "n1"]


def test_select_requested_records_supports_precise_smoke_subset() -> None:
    cohort = [_record("p1"), _record("p2"), _record("n1")]

    selected = mas.select_requested_records(cohort, ["p1", "n1"], limit=None)

    assert [row["record_id"] for row in selected] == ["p1", "n1"]
    with pytest.raises(ValueError, match="unknown record_ids"):
        mas.select_requested_records(cohort, ["missing"], limit=None)


def test_collaboration_metrics_report_improvements_and_degradations() -> None:
    cohort = [
        _record("improved", decision=mas.ACCEPTED_FAULT),
        _record("degraded", decision=mas.REJECTED_CANDIDATE),
    ]
    predictions = [
        {
            "record_id": "improved",
            "invalid": False,
            "roles": {
                "proposer": {"parsed_output": {"decision": mas.REJECTED_CANDIDATE}, "attempts": 1, "latency_seconds": 1.0, "token_usage": {"total_tokens": 10}},
                "critic": {"parsed_output": {"verdict": "revise", "suggested_decision": mas.ACCEPTED_FAULT}, "attempts": 1, "latency_seconds": 2.0, "token_usage": {"total_tokens": 20}},
                "judge": {"parsed_output": {"decision": mas.ACCEPTED_FAULT}, "attempts": 1, "latency_seconds": 3.0, "token_usage": {"total_tokens": 30}},
            },
            "final_prediction": {"decision": mas.ACCEPTED_FAULT},
        },
        {
            "record_id": "degraded",
            "invalid": False,
            "roles": {
                "proposer": {"parsed_output": {"decision": mas.REJECTED_CANDIDATE}, "attempts": 1, "latency_seconds": 1.0, "token_usage": {}},
                "critic": {"parsed_output": {"verdict": "revise", "suggested_decision": mas.ACCEPTED_FAULT}, "attempts": 2, "latency_seconds": 2.0, "token_usage": {}},
                "judge": {"parsed_output": {"decision": mas.ACCEPTED_FAULT}, "attempts": 1, "latency_seconds": 3.0, "token_usage": {}},
            },
            "final_prediction": {"decision": mas.ACCEPTED_FAULT},
        },
    ]

    metrics = mas.evaluate_collaboration(cohort, predictions, stage="stage2")

    assert metrics["proposer_judge_change_count"] == 2
    assert metrics["critic_disagreement_count"] == 2
    assert metrics["judge_improvement_count"] == 1
    assert metrics["judge_degradation_count"] == 1
    assert metrics["total_role_attempts"] == 7
    assert metrics["total_api_requests"] == 7
    assert metrics["total_role_latency_seconds"] == 12.0
    assert metrics["total_tokens"] == 60


def test_run_stage_records_resumes_only_complete_matching_rows(tmp_path: Path) -> None:
    cohort = [_record("r1"), _record("r2"), _record("r3")]
    output = tmp_path / "predictions.jsonl"
    complete = {
        "record_id": "r1",
        "stage": "stage2",
        "model": "model",
        "config_hash": "hash",
        "roles": {"proposer": {}, "critic": {}, "judge": {}},
        "final_prediction": {"decision": mas.ACCEPTED_FAULT},
        "invalid": False,
    }
    incomplete = {
        **complete,
        "record_id": "r2",
        "roles": {"proposer": {}},
    }
    wrong_hash = {**complete, "record_id": "r3", "config_hash": "old"}
    output.write_text(
        "\n".join(json.dumps(row) for row in [complete, incomplete, wrong_hash]) + "\n",
        encoding="utf-8",
    )
    called: list[str] = []
    progress: list[tuple[int, int, str, str]] = []

    def runner(record: dict[str, str]) -> dict[str, object]:
        called.append(record["record_id"])
        return {
            **complete,
            "record_id": record["record_id"],
        }

    rows = mas.run_stage_records(
        cohort,
        predictions_path=output,
        stage="stage2",
        model="model",
        config_hash="hash",
        record_runner=runner,
        resume=True,
        progress_callback=lambda current, total, record_id, status: progress.append(
            (current, total, record_id, status)
        ),
    )

    assert called == ["r2", "r3"]
    assert [row["record_id"] for row in rows] == ["r1", "r2", "r3"]
    saved = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["record_id"] for row in saved] == ["r1", "r2", "r3"]
    assert progress == [
        (1, 3, "r1", "resumed"),
        (2, 3, "r2", "completed"),
        (3, 3, "r3", "completed"),
    ]


def test_run_stage_records_checkpoints_each_completed_record(tmp_path: Path) -> None:
    cohort = [_record("r1"), _record("r2"), _record("interrupt")]
    output = tmp_path / "predictions.jsonl"

    def runner(record: dict[str, str]) -> dict[str, object]:
        if record["record_id"] == "interrupt":
            raise KeyboardInterrupt
        return {
            "record_id": record["record_id"],
            "roles": {"proposer": {}, "critic": {}, "judge": {}},
            "final_prediction": {"decision": mas.ACCEPTED_FAULT},
            "invalid": False,
        }

    with pytest.raises(KeyboardInterrupt):
        mas.run_stage_records(
            cohort,
            predictions_path=output,
            stage="stage2",
            model="model",
            config_hash="hash",
            record_runner=runner,
            resume=False,
        )

    saved = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["record_id"] for row in saved] == ["r1", "r2"]


def test_validate_final_prediction_rejects_invalid_stage2_decision() -> None:
    row = {
        "record_id": "bad-stage2",
        "invalid": False,
        "final_prediction": {"decision": "maybe"},
    }

    with pytest.raises(ValueError, match="bad-stage2"):
        mas.validate_final_prediction(
            row,
            stage="stage2",
            taxonomy={"symptom": ["Crash"], "root_cause": ["Cause"]},
        )


def test_validate_final_prediction_rejects_stage3_label_outside_taxonomy() -> None:
    row = {
        "record_id": "bad-stage3",
        "invalid": False,
        "final_prediction": {
            "symptom": "Unknown",
            "root_cause": "Cause",
        },
    }

    with pytest.raises(ValueError, match="bad-stage3"):
        mas.validate_final_prediction(
            row,
            stage="stage3",
            taxonomy={"symptom": ["Crash"], "root_cause": ["Cause"]},
        )


def test_strict_stage_records_do_not_write_invalid_row_and_resume_retries_it(
    tmp_path: Path,
) -> None:
    cohort = [_record("valid"), _record("retry")]
    output = tmp_path / "strict_predictions.jsonl"
    taxonomy = {"symptom": ["Crash"], "root_cause": ["Cause"]}

    def first_runner(record: dict[str, str]) -> dict[str, object]:
        return {
            "record_id": record["record_id"],
            "roles": {"proposer": {}, "critic": {}, "judge": {}},
            "final_prediction": (
                {"decision": mas.ACCEPTED_FAULT}
                if record["record_id"] == "valid"
                else {}
            ),
            "invalid": record["record_id"] == "retry",
        }

    with pytest.raises(ValueError, match="retry"):
        mas.run_stage_records(
            cohort,
            predictions_path=output,
            stage="stage2",
            model="model",
            config_hash="strict-hash",
            record_runner=first_runner,
            resume=False,
            require_valid_json=True,
            taxonomy=taxonomy,
        )

    saved_after_failure = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["record_id"] for row in saved_after_failure] == ["valid"]

    called: list[str] = []

    def retry_runner(record: dict[str, str]) -> dict[str, object]:
        called.append(record["record_id"])
        return {
            "record_id": record["record_id"],
            "roles": {"proposer": {}, "critic": {}, "judge": {}},
            "final_prediction": {"decision": mas.REJECTED_CANDIDATE},
            "invalid": False,
        }

    rows = mas.run_stage_records(
        cohort,
        predictions_path=output,
        stage="stage2",
        model="model",
        config_hash="strict-hash",
        record_runner=retry_runner,
        resume=True,
        require_valid_json=True,
        taxonomy=taxonomy,
    )

    assert called == ["retry"]
    assert [row["record_id"] for row in rows] == ["valid", "retry"]


def test_prepare_artifacts_writes_cohort_prompts_taxonomy_and_manifest(tmp_path: Path) -> None:
    stage2_path = tmp_path / "stage2_sample.csv"
    stage3_path = tmp_path / "stage3_sample.csv"
    output_dir = tmp_path / "out"
    _write_csv(
        stage2_path,
        [
            _record("n1", decision=mas.REJECTED_CANDIDATE),
            _record("n2", decision=mas.REJECTED_CANDIDATE),
        ],
    )
    _write_csv(
        stage3_path,
        [
            _record("p1", symptom="Crash", root_cause="Incorrect Code Logic"),
            _record("p2", symptom="Poor Performance", root_cause="WebGL Limits"),
        ],
    )

    paths = mas.prepare_artifacts(
        stage2_sample_path=stage2_path,
        stage3_sample_path=stage3_path,
        output_dir=output_dir,
        positives=2,
        negatives=2,
    )

    cohort = list(csv.DictReader(paths["cohort"].open(encoding="utf-8")))
    taxonomy = json.loads(paths["taxonomy"].read_text(encoding="utf-8"))
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    stage2_prompts = [
        json.loads(line)
        for line in paths["stage2_prompts"].read_text(encoding="utf-8").splitlines()
    ]
    stage3_prompts = [
        json.loads(line)
        for line in paths["stage3_prompts"].read_text(encoding="utf-8").splitlines()
    ]

    assert len(cohort) == 4
    assert taxonomy == {
        "symptom": sorted(mas.stage3_baseline.SYMPTOM_DEFINITIONS),
        "root_cause": sorted(mas.stage3_baseline.ROOT_CAUSE_DEFINITIONS),
    }
    assert manifest["counts"] == {mas.ACCEPTED_FAULT: 2, mas.REJECTED_CANDIDATE: 2}
    assert manifest["cohort_sha256"]
    assert len(stage2_prompts) == 4
    assert len(stage3_prompts) == 2
    assert all("ground_truth" in row for row in stage2_prompts)
    assert all("ground_truth" not in row["user_prompt"] for row in stage2_prompts)
    assert all("ground_truth" not in row for row in stage3_prompts)


def test_config_hash_is_stable_and_changes_with_experiment_inputs() -> None:
    records = [_record("r1"), _record("r2")]
    taxonomy = {"symptom": ["Crash"], "root_cause": ["Cause"]}

    first = mas.build_config_hash("model", "stage2", records, taxonomy, 0.0)
    second = mas.build_config_hash("model", "stage2", records, taxonomy, 0.0)
    changed_model = mas.build_config_hash("other", "stage2", records, taxonomy, 0.0)
    changed_order = mas.build_config_hash("model", "stage2", list(reversed(records)), taxonomy, 0.0)
    changed_backend = mas.build_config_hash(
        "model",
        "stage2",
        records,
        taxonomy,
        0.0,
        backend_id="deepseek:https://api.deepseek.com",
    )
    changed_turns = mas.build_config_hash(
        "model", "stage2", records, taxonomy, 0.0, max_turns=4
    )

    assert first == second
    assert first != changed_model
    assert first != changed_order
    assert first != changed_backend
    assert first != changed_turns


def test_society_default_turn_cap_is_ten_and_affects_config_hash() -> None:
    records = [_record("r1")]
    taxonomy = {"symptom": ["Crash"], "root_cause": ["Cause"]}

    default_hash = mas.build_config_hash(
        "model", "stage2", records, taxonomy, 0.0
    )
    five_turn_hash = mas.build_config_hash(
        "model", "stage2", records, taxonomy, 0.0, max_turns=5
    )
    ten_turn_hash = mas.build_config_hash(
        "model", "stage2", records, taxonomy, 0.0, max_turns=10
    )

    assert getattr(mas, "DEFAULT_MAX_TURNS", None) == 10
    assert default_hash == ten_turn_hash
    assert default_hash != five_turn_hash


def test_config_hash_separates_native_and_evidence_anchored_modes() -> None:
    records = [_record("r1")]
    taxonomy = {"symptom": ["Crash"], "root_cause": ["Cause"]}

    anchored = mas.build_config_hash(
        "model",
        "stage2",
        records,
        taxonomy,
        0.0,
        society_mode="evidence_anchored",
    )
    native = mas.build_config_hash(
        "model",
        "stage2",
        records,
        taxonomy,
        0.0,
        society_mode="native",
    )

    assert anchored != native


def test_society_diagnostics_report_turns_stops_cost_and_formats() -> None:
    predictions = [
        {
            "record_id": "r1",
            "invalid": False,
            "society": {
                "turn_count": 1,
                "max_turns": 5,
                "stop_reason": "valid_json",
                "final_answer_turn": 1,
                "format_normalized": True,
                "api_request_count": 3,
                "token_usage": {"total_tokens": 30},
                "latency_seconds": 2.0,
                "role_request_stats": {
                    "task_specifier": {
                        "api_request_count": 1,
                        "usage_observed_request_count": 1,
                    }
                },
            },
        },
        {
            "record_id": "r2",
            "invalid": True,
            "society": {
                "turn_count": 5,
                "max_turns": 5,
                "stop_reason": "max_turns",
                "final_answer_turn": None,
                "format_normalized": False,
                "api_request_count": 11,
                "token_usage": {"total_tokens": 110},
                "latency_seconds": 8.0,
                "role_request_stats": {
                    "ai_assistant": {
                        "api_request_count": 5,
                        "usage_observed_request_count": 4,
                    }
                },
            },
        },
    ]

    metrics = mas.evaluate_society_diagnostics(predictions)

    assert metrics["n"] == 2
    assert metrics["valid_count"] == 1
    assert metrics["invalid_count"] == 1
    assert metrics["total_turn_count"] == 6
    assert metrics["average_turn_count"] == 3.0
    assert metrics["early_stop_count"] == 1
    assert metrics["stop_reason_counts"] == {
        "valid_json": 1,
        "max_turns": 1,
    }
    assert metrics["final_answer_turn_counts"] == {"1": 1, "none": 1}
    assert metrics["total_api_requests"] == 14
    assert metrics["total_tokens"] == 140
    assert metrics["total_latency_seconds"] == 10.0
    assert metrics["format_normalized_count"] == 1
    assert metrics["token_usage_incomplete_role_count"] == 1


def test_run_stage_records_resumes_complete_society_row(tmp_path: Path) -> None:
    cohort = [_record("r1")]
    output = tmp_path / "society_predictions.jsonl"
    complete = {
        "record_id": "r1",
        "stage": "stage2",
        "model": "model",
        "config_hash": "hash",
        "architecture": "camel_roleplaying_society",
        "society": {
            "task_prompt": "task",
            "specified_task_prompt": "specified",
            "turn_count": 1,
            "turns": [{"turn": 1}],
            "stop_reason": "camel_task_done",
        },
        "final_prediction": {"decision": mas.ACCEPTED_FAULT},
        "invalid": False,
    }
    output.write_text(json.dumps(complete) + "\n", encoding="utf-8")
    called: list[str] = []

    rows = mas.run_stage_records(
        cohort,
        predictions_path=output,
        stage="stage2",
        model="model",
        config_hash="hash",
        record_runner=lambda record: called.append(record["record_id"]),
        resume=True,
    )

    assert called == []
    assert rows == [complete]


def test_run_stage_records_resumes_complete_evidence_anchored_row(
    tmp_path: Path,
) -> None:
    cohort = [_record("r1")]
    output = tmp_path / "anchored_predictions.jsonl"
    complete = {
        "record_id": "r1",
        "stage": "stage2",
        "model": "model",
        "config_hash": "anchored-hash",
        "architecture": "camel_roleplaying_evidence_anchored",
        "society_mode": "evidence_anchored",
        "society": {
            "task_prompt": "task",
            "specified_task_prompt": "",
            "immutable_evidence_sha256": "digest",
            "turn_count": 1,
            "turns": [
                {
                    "turn": 1,
                    "user_instruction": "inspect evidence",
                    "discarded_user_input": "invented input",
                    "assistant_input": "immutable task",
                }
            ],
            "stop_reason": "valid_json",
        },
        "final_prediction": {"decision": mas.ACCEPTED_FAULT},
        "invalid": False,
    }
    output.write_text(json.dumps(complete) + "\n", encoding="utf-8")
    called: list[str] = []

    rows = mas.run_stage_records(
        cohort,
        predictions_path=output,
        stage="stage2",
        model="model",
        config_hash="anchored-hash",
        record_runner=lambda record: called.append(record["record_id"]),
        resume=True,
    )

    assert called == []
    assert rows == [complete]


def test_counting_openai_client_counts_internal_requests_and_tokens() -> None:
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=4, total_tokens=14)

    def create(**_kwargs):
        return SimpleNamespace(usage=usage)

    def parse(**_kwargs):
        return SimpleNamespace(usage=usage)

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        beta=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse))),
    )
    counting = mas._CountingOpenAIClient(client)

    counting.chat.completions.create(model="test")
    counting.beta.chat.completions.parse(model="test")

    latency = counting.request_stats.pop("latency_seconds")
    assert latency >= 0.0
    assert counting.request_stats == {
        "api_request_count": 2,
        "usage_observed_request_count": 2,
        "prompt_tokens": 20,
        "completion_tokens": 8,
        "total_tokens": 28,
    }
