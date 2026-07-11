import http.client
import json

from Benchmark.src import ase2022_llm_baseline as baseline


TAXONOMY = {
    "symptom": ["Crash"],
    "root_cause": ["Incorrect Code Logic"],
}


def _prompt(record_id: str) -> dict[str, str]:
    return {
        "record_id": record_id,
        "paper_id": "paper",
        "issue_url": f"https://example.test/{record_id}",
        "system_prompt": "system",
        "user_prompt": "user",
    }


def _example(record_id: str) -> dict[str, str]:
    return {
        "record_id": record_id,
        "symptom": "Crash",
        "root_cause": "Incorrect Code Logic",
    }


def _run(tmp_path, monkeypatch, prompts, examples, fake_call, **kwargs):
    prompts_path = tmp_path / "prompts.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    metrics_path = tmp_path / "metrics.json"
    prompts_path.write_text(
        "\n".join(json.dumps(prompt) for prompt in prompts) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(baseline, "call_model", fake_call)
    metrics = baseline.run_llm_prompts(
        prompts_path=prompts_path,
        examples=examples,
        taxonomy=TAXONOMY,
        predictions_path=predictions_path,
        metrics_path=metrics_path,
        model="test-model",
        api_key="key",
        base_url="https://example.test/v1",
        retry_delay_seconds=0,
        **kwargs,
    )
    rows = [json.loads(line) for line in predictions_path.read_text().splitlines()]
    return metrics, rows, predictions_path


def test_retries_remote_disconnect_then_succeeds(tmp_path, monkeypatch) -> None:
    attempts = 0

    def fake_call(**_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise http.client.RemoteDisconnected("connection closed")
        return '{"symptom":"Crash","root_cause":"Incorrect Code Logic"}'

    metrics, rows, _ = _run(
        tmp_path,
        monkeypatch,
        [_prompt("r1")],
        [_example("r1")],
        fake_call,
        max_retries=2,
    )

    assert attempts == 2
    assert metrics["valid_count"] == 1
    assert rows[0]["attempts"] == 2


def test_resume_skips_existing_prediction(tmp_path, monkeypatch) -> None:
    prompts = [_prompt("r1"), _prompt("r2")]
    calls = []

    def fake_call(**kwargs):
        calls.append(kwargs)
        return '{"symptom":"Crash","root_cause":"Incorrect Code Logic"}'

    prompts_path = tmp_path / "prompts.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    metrics_path = tmp_path / "metrics.json"
    prompts_path.write_text(
        "\n".join(json.dumps(prompt) for prompt in prompts) + "\n",
        encoding="utf-8",
    )
    predictions_path.write_text(
        json.dumps(
            {
                "record_id": "r1",
                "paper_id": "paper",
                "issue_url": "https://example.test/r1",
                "model": "test-model",
                "raw_output": "{}",
                "latency_seconds": 1.0,
                "symptom": "Crash",
                "root_cause": "Incorrect Code Logic",
                "invalid": False,
                "error": "",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(baseline, "call_model", fake_call)

    metrics = baseline.run_llm_prompts(
        prompts_path=prompts_path,
        examples=[_example("r1"), _example("r2")],
        taxonomy=TAXONOMY,
        predictions_path=predictions_path,
        metrics_path=metrics_path,
        model="test-model",
        api_key="key",
        base_url="https://example.test/v1",
        resume=True,
        max_retries=0,
        retry_delay_seconds=0,
    )
    rows = [json.loads(line) for line in predictions_path.read_text().splitlines()]

    assert len(calls) == 1
    assert [row["record_id"] for row in rows] == ["r1", "r2"]
    assert metrics["n"] == 2
    assert metrics["valid_count"] == 2


def test_resume_retries_previous_network_failure(tmp_path, monkeypatch) -> None:
    prompts = [_prompt("r1")]

    def disconnected(**_kwargs):
        raise http.client.RemoteDisconnected("connection closed")

    first_metrics, first_rows, predictions_path = _run(
        tmp_path,
        monkeypatch,
        prompts,
        [_example("r1")],
        disconnected,
        max_retries=0,
    )

    assert first_metrics["invalid_count"] == 1
    assert first_rows[0]["request_failed"] is True

    calls = 0

    def succeeds(**_kwargs):
        nonlocal calls
        calls += 1
        return '{"symptom":"Crash","root_cause":"Incorrect Code Logic"}'

    monkeypatch.setattr(baseline, "call_model", succeeds)
    metrics = baseline.run_llm_prompts(
        prompts_path=tmp_path / "prompts.jsonl",
        examples=[_example("r1")],
        taxonomy=TAXONOMY,
        predictions_path=predictions_path,
        metrics_path=tmp_path / "metrics.json",
        model="test-model",
        api_key="key",
        base_url="https://example.test/v1",
        resume=True,
        max_retries=0,
        retry_delay_seconds=0,
    )

    assert calls == 1
    assert metrics["valid_count"] == 1
