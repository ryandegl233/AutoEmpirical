from __future__ import annotations

import csv
import http.client
import json
import os
import random
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd


ASE2022_PAPER_ID = "ase2022_towards_understanding_the_faults_of"
TEXT_FIELDS = ("title", "state", "created_at", "body", "comments")

SYMPTOM_DEFINITIONS = {
    "Build & Initialization Failure": (
        "Failures during installation, build, compilation, or backend/runtime initialization."
    ),
    "Crash": "Functionality is terminated unexpectedly, usually with errors or exceptions.",
    "Document Error": (
        "Faults in official documents, tutorials, examples, links, or instructions."
    ),
    "Incorrect Functionality": (
        "The system runs without crashing but produces incorrect, inconsistent, null, NaN, "
        "or inaccurate results."
    ),
    "Poor Performance": (
        "The system runs but is slow, hangs, leaks memory, or consumes abnormal resources."
    ),
}

ROOT_CAUSE_DEFINITIONS = {
    "API Misuse": "The issue is caused by misunderstanding or incorrectly using an API.",
    "Browser Incompatibility": (
        "The issue is caused by browser-specific behavior or browser support limitations."
    ),
    "Confused Document": (
        "The issue is caused by unclear, wrong, or misleading documentation."
    ),
    "Cross-platform App Framework Incompatibility": (
        "The issue involves incompatibility with frameworks such as React Native or other "
        "cross-platform app environments."
    ),
    "Data/Model Inaccessibility": (
        "The issue is caused by inaccessible model/data files, path problems, CORS, fetch "
        "failures, or loading failures."
    ),
    "Dependency Error": (
        "The issue is caused by missing, redundant, conflicting, vulnerable, or incompatible "
        "dependencies."
    ),
    "Device Incompatibility": (
        "The issue is caused by specific hardware, OS, or device support limitations."
    ),
    "Import Error": "The issue is caused by missing, wrong, duplicate, or conflicting imports.",
    "Improper Exception Handling": (
        "The issue involves missing, suspicious, unclear, or misleading exception handling."
    ),
    "Improper Model Attribute": (
        "The issue is caused by inappropriate model/tensor attributes, model size, model "
        "parameters, or tensor/model properties."
    ),
    "Incompatibilitty between 3rd-party DL Library and TF.js": (
        "The issue is caused by incompatibility between a third-party deep learning library "
        "and TensorFlow.js. Use this exact label string, including its spelling."
    ),
    "Inconsistent Modules": (
        "The issue is caused by inconsistent behavior or implementation across TensorFlow.js "
        "modules."
    ),
    "Incorrect Code Logic": (
        "The issue is caused by faulty implementation logic, algorithm logic, memory "
        "management, or environment adaptation."
    ),
    "Misconfiguration": (
        "The issue is caused by incorrect environment, build, bundler, backend, or runtime "
        "configuration."
    ),
    "Unimplemented Operator": (
        "The issue is caused by an operator or function not yet supported or implemented."
    ),
    "Unknown": "The root cause cannot be determined from the available issue evidence.",
    "Untimely Update": (
        "The issue is caused by outdated packages, delayed updates, or version lag."
    ),
    "WebGL Limits": "The issue is caused by inherent WebGL limitations.",
}


def evidence_chars(example: dict[str, str]) -> int:
    return sum(len(str(example.get(field, "")).strip()) for field in ("title", "body", "comments"))


def add_length_fields(example: dict[str, str]) -> dict[str, str | int]:
    enriched = dict(example)
    enriched["body_chars"] = len(str(example.get("body", "")).strip())
    enriched["comments_chars"] = len(str(example.get("comments", "")).strip())
    enriched["evidence_chars"] = evidence_chars(example)
    return enriched


def _clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def load_ase2022_examples(
    stage2_path: str | Path = "Dataset/stage2.csv",
    stage3_path: str | Path = "Dataset/stage3.csv",
) -> list[dict[str, str]]:
    stage2 = pd.read_csv(stage2_path, low_memory=False)
    stage3 = pd.read_csv(stage3_path, low_memory=False)
    stage2 = stage2[stage2["paper_id"] == ASE2022_PAPER_ID]
    stage3 = stage3[stage3["paper_id"] == ASE2022_PAPER_ID]

    merged = stage3[
        ["record_id", "paper_id", "issue_url", "symptom", "root_cause"]
    ].merge(
        stage2[["record_id", "paper_id", "issue_url", *TEXT_FIELDS]],
        on=["record_id", "paper_id", "issue_url"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(stage3):
        raise ValueError(
            f"Matched {len(merged)} of {len(stage3)} ASE2022 Stage 3 rows to Stage 2"
        )

    examples = []
    for _, row in merged.sort_values("record_id").iterrows():
        examples.append(
            {
                "record_id": str(row["record_id"]),
                "paper_id": str(row["paper_id"]),
                "issue_url": str(row["issue_url"]),
                "title": _clean_text(row["title"]),
                "state": _clean_text(row["state"]),
                "created_at": _clean_text(row["created_at"]),
                "body": _clean_text(row["body"]),
                "comments": _clean_text(row["comments"]),
                "symptom": _clean_text(row["symptom"]),
                "root_cause": _clean_text(row["root_cause"]),
            }
        )
    return examples


def build_taxonomy(examples: list[dict[str, str]]) -> dict[str, list[str]]:
    return {
        "symptom": sorted({example["symptom"] for example in examples if example["symptom"]}),
        "root_cause": sorted(
            {example["root_cause"] for example in examples if example["root_cause"]}
        ),
    }


def select_fixed_sample(
    examples: list[dict[str, str]],
    sample_size: int = 50,
    seed: int = 20260708,
    min_evidence_chars: int = 500,
) -> list[dict[str, str | int]]:
    eligible = [example for example in examples if evidence_chars(example) >= min_evidence_chars]
    pool = eligible if len(eligible) >= sample_size else examples

    if sample_size >= len(pool):
        return [add_length_fields(row) for row in sorted(pool, key=lambda row: row["record_id"])]

    by_symptom: dict[str, list[dict[str, str]]] = {}
    for example in pool:
        by_symptom.setdefault(example["symptom"], []).append(example)

    rng = random.Random(seed)
    selected: list[dict[str, str]] = []
    for symptom in sorted(by_symptom):
        group = sorted(by_symptom[symptom], key=lambda row: row["record_id"])
        selected.append(rng.choice(group))

    remaining = [
        example
        for example in pool
        if example["record_id"] not in {row["record_id"] for row in selected}
    ]
    rng.shuffle(remaining)
    selected.extend(remaining[: sample_size - len(selected)])
    return [add_length_fields(row) for row in sorted(selected, key=lambda row: row["record_id"])]


def _taxonomy_section(values: list[str], definitions: dict[str, str]) -> str:
    return "\n\n".join(f"{value}:\n{definitions[value]}" for value in values)


def build_system_prompt(taxonomy: dict[str, list[str]]) -> str:
    return f"""You are an expert in analyzing bugs in JavaScript-based deep learning systems.

Given an issue report from TensorFlow.js or related JavaScript-based deep learning projects, classify the bug according to the ASE 2022 empirical-study taxonomy.

Analyze the issue report and classify it into:
1. symptom: what the fault looks like
2. root_cause: why the fault occurs

You must choose exactly one symptom label and exactly one root_cause label from the allowed labels below.

## Symptom Taxonomy

{_taxonomy_section(taxonomy["symptom"], SYMPTOM_DEFINITIONS)}

## Root Cause Taxonomy

{_taxonomy_section(taxonomy["root_cause"], ROOT_CAUSE_DEFINITIONS)}

## Output Format

Return ONLY strict JSON with no explanation:

{{
  "symptom": "<one exact symptom label>",
  "root_cause": "<one exact root_cause label>"
}}

## Classification Guidelines

1. For symptom, focus on the observable behavior.
2. For root_cause, focus on the underlying technical reason.
3. If several labels seem possible, choose the most central one.
4. Use only the exact labels listed above.
5. Do not invent new labels.
"""


def build_user_prompt(example: dict[str, str]) -> str:
    return f"""Please carefully read the following issue report and provide the classification result:

## Issue Report
### [Title]:
{example["title"]}

### [State]:
{example["state"]}

### [Created At]:
{example["created_at"]}

### [Body]:
{example["body"]}

### [Other Comments]:
{example["comments"]}
"""


def build_anchored_prompt(example: dict[str, str], taxonomy: dict[str, list[str]]) -> str:
    return f"{build_system_prompt(taxonomy)}\n\n{build_user_prompt(example)}"


def write_sample_csv(examples: list[dict[str, str]], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "record_id",
        "paper_id",
        "issue_url",
        "symptom",
        "root_cause",
        "title",
        "state",
        "created_at",
        "body",
        "comments",
        "body_chars",
        "comments_chars",
        "evidence_chars",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(add_length_fields(example) for example in examples)
    return path


def write_prompt_jsonl(
    examples: list[dict[str, str]],
    taxonomy: dict[str, list[str]],
    output_path: str | Path,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        system_prompt = build_system_prompt(taxonomy)
        for example in examples:
            row = {
                "record_id": example["record_id"],
                "paper_id": example["paper_id"],
                "issue_url": example["issue_url"],
                "system_prompt": system_prompt,
                "user_prompt": build_user_prompt(example),
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def parse_prediction_text(
    text: str,
    taxonomy: dict[str, list[str]],
) -> dict[str, object]:
    cleaned = text.strip()
    format_repaired = False
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
        format_repaired = True

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        return {
            "symptom": "",
            "root_cause": "",
            "invalid": True,
            "format_repaired": format_repaired,
            "error": str(exc),
        }

    symptom = str(payload.get("symptom", "")).strip()
    root_cause = str(payload.get("root_cause", "")).strip()
    invalid = symptom not in taxonomy["symptom"] or root_cause not in taxonomy["root_cause"]
    return {
        "symptom": symptom,
        "root_cause": root_cause,
        "invalid": invalid,
        "format_repaired": format_repaired,
        "error": "" if not invalid else "label_not_in_taxonomy",
    }


def evaluate_predictions(
    examples: list[dict[str, str]],
    predictions: list[dict[str, object]],
) -> dict[str, float | int]:
    by_record_id = {prediction["record_id"]: prediction for prediction in predictions}
    valid_examples = []
    invalid_count = 0
    for example in examples:
        prediction = by_record_id.get(example["record_id"])
        if not prediction or prediction.get("invalid"):
            invalid_count += 1
            continue
        valid_examples.append((example, prediction))

    n = len(examples)
    valid_n = len(valid_examples)
    symptom_correct = sum(
        example["symptom"] == prediction.get("symptom")
        for example, prediction in valid_examples
    )
    root_cause_correct = sum(
        example["root_cause"] == prediction.get("root_cause")
        for example, prediction in valid_examples
    )
    both_correct = sum(
        example["symptom"] == prediction.get("symptom")
        and example["root_cause"] == prediction.get("root_cause")
        for example, prediction in valid_examples
    )
    denominator = valid_n or 1
    return {
        "n": n,
        "valid_count": valid_n,
        "invalid_count": invalid_count,
        "invalid_rate": invalid_count / n if n else 0.0,
        "symptom_accuracy": symptom_correct / denominator,
        "root_cause_accuracy": root_cause_correct / denominator,
        "both_exact_match_accuracy": both_correct / denominator,
    }


def call_chat_completion(
    system_prompt: str,
    user_prompt: str,
    model: str,
    api_key: str,
    base_url: str,
    timeout_seconds: int = 120,
    temperature: float = 0.0,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        data = json.loads(response.read().decode("utf-8"))
    return str(data["choices"][0]["message"]["content"])


def _extract_responses_text(data: dict[str, object]) -> str:
    if isinstance(data.get("output_text"), str):
        return str(data["output_text"])
    output = data.get("output")
    if isinstance(output, list):
        chunks: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                text = part.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        if chunks:
            return "\n".join(chunks)
    raise KeyError("No text content found in responses API payload")


def call_responses_api(
    system_prompt: str,
    user_prompt: str,
    model: str,
    api_key: str,
    base_url: str,
    timeout_seconds: int = 120,
    responses_path: str = "/responses",
    http_client: str = "urllib",
) -> str:
    url = base_url.rstrip("/") + "/" + responses_path.strip("/")
    payload = {
        "model": model,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    if http_client == "powershell":
        data = _post_json_with_powershell(
            url=url,
            payload=payload,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
        )
        return _extract_responses_text(data)

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        data = json.loads(response.read().decode("utf-8"))
    return _extract_responses_text(data)


def _post_json_with_powershell(
    url: str,
    payload: dict[str, object],
    api_key: str,
    timeout_seconds: int,
) -> dict[str, object]:
    env = os.environ.copy()
    env["AE_LLM_API_KEY"] = api_key
    env["AE_LLM_URL"] = url
    script = r"""
$body = [Console]::In.ReadToEnd()
$headers = @{
  Authorization = "Bearer $env:AE_LLM_API_KEY"
  "Content-Type" = "application/json"
  Accept = "application/json"
}
$response = Invoke-RestMethod -Method Post -Uri $env:AE_LLM_URL -Headers $headers -Body $body
$response | ConvertTo-Json -Depth 30
"""
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        input=json.dumps(payload, ensure_ascii=False),
        capture_output=True,
        encoding="utf-8",
        env=env,
        timeout=timeout_seconds,
        check=False,
    )
    if completed.returncode != 0:
        error = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"PowerShell HTTP request failed: {error}")
    return json.loads(completed.stdout)


def call_model(
    system_prompt: str,
    user_prompt: str,
    model: str,
    api_key: str,
    base_url: str,
    wire_api: str,
    responses_path: str = "/responses",
    http_client: str = "urllib",
) -> str:
    if wire_api == "responses":
        return call_responses_api(
            system_prompt,
            user_prompt,
            model,
            api_key,
            base_url,
            responses_path=responses_path,
            http_client=http_client,
        )
    if wire_api in {"chat_completions", "chat"}:
        return call_chat_completion(system_prompt, user_prompt, model, api_key, base_url)
    raise ValueError(f"Unsupported WIRE_API: {wire_api}")


def _is_retryable_network_error(exc: BaseException) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in {408, 425, 429, 500, 502, 503, 504}
    return isinstance(
        exc,
        (
            http.client.RemoteDisconnected,
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
        ),
    )


def call_model_with_retries(
    *,
    max_retries: int,
    retry_delay_seconds: float,
    **call_kwargs: object,
) -> tuple[str, int]:
    attempts = 0
    while True:
        attempts += 1
        try:
            return call_model(**call_kwargs), attempts
        except Exception as exc:
            if attempts > max_retries or not _is_retryable_network_error(exc):
                setattr(exc, "attempts", attempts)
                raise
            if retry_delay_seconds:
                time.sleep(retry_delay_seconds * (2 ** (attempts - 1)))


def load_existing_predictions(
    path: str | Path,
    model: str,
    allowed_record_ids: set[str],
) -> dict[str, dict[str, object]]:
    predictions_path = Path(path)
    if not predictions_path.exists():
        return {}

    existing: dict[str, dict[str, object]] = {}
    for line_number, line in enumerate(
        predictions_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            prediction = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSONL in {predictions_path} at line {line_number}: {exc}"
            ) from exc
        record_id = str(prediction.get("record_id", ""))
        if (
            prediction.get("model") == model
            and record_id in allowed_record_ids
            and not prediction.get("request_failed", False)
        ):
            existing[record_id] = prediction
    return existing


def run_llm_prompts(
    prompts_path: str | Path,
    examples: list[dict[str, str]],
    taxonomy: dict[str, list[str]],
    predictions_path: str | Path,
    metrics_path: str | Path,
    model: str,
    api_key: str,
    base_url: str,
    wire_api: str = "chat_completions",
    responses_path: str = "/responses",
    http_client: str = "urllib",
    limit: int | None = None,
    sleep_seconds: float = 0.0,
    resume: bool = True,
    max_retries: int = 3,
    retry_delay_seconds: float = 2.0,
) -> dict[str, float | int | str]:
    prompts = [
        json.loads(line)
        for line in Path(prompts_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if limit is not None:
        prompts = prompts[:limit]

    predictions_output = Path(predictions_path)
    predictions_output.parent.mkdir(parents=True, exist_ok=True)
    prompt_record_ids = {str(row["record_id"]) for row in prompts}
    existing_by_record_id = (
        load_existing_predictions(predictions_output, model, prompt_record_ids)
        if resume
        else {}
    )
    parsed_predictions: list[dict[str, object]] = [
        existing_by_record_id[str(row["record_id"])]
        for row in prompts
        if str(row["record_id"]) in existing_by_record_id
    ]
    pending_prompts = [
        row for row in prompts if str(row["record_id"]) not in existing_by_record_id
    ]
    output_mode = "a" if resume and predictions_output.exists() else "w"
    with predictions_output.open(output_mode, encoding="utf-8") as handle:
        for row in pending_prompts:
            started = time.perf_counter()
            attempts = 0
            request_failed = False
            try:
                raw_output, attempts = call_model_with_retries(
                    max_retries=max_retries,
                    retry_delay_seconds=retry_delay_seconds,
                    system_prompt=row["system_prompt"],
                    user_prompt=row["user_prompt"],
                    model=model,
                    api_key=api_key,
                    base_url=base_url,
                    wire_api=wire_api,
                    responses_path=responses_path,
                    http_client=http_client,
                )
                parsed = parse_prediction_text(raw_output, taxonomy)
            except (
                http.client.RemoteDisconnected,
                urllib.error.URLError,
                TimeoutError,
                ConnectionError,
                KeyError,
                ValueError,
                RuntimeError,
                subprocess.SubprocessError,
            ) as exc:
                attempts = int(getattr(exc, "attempts", attempts or 1))
                request_failed = True
                raw_output = ""
                parsed = {
                    "symptom": "",
                    "root_cause": "",
                    "invalid": True,
                    "error": str(exc),
                }

            prediction = {
                "record_id": row["record_id"],
                "paper_id": row["paper_id"],
                "issue_url": row["issue_url"],
                "model": model,
                "raw_output": raw_output,
                "latency_seconds": round(time.perf_counter() - started, 3),
                "attempts": attempts,
                "request_failed": request_failed,
                **parsed,
            }
            parsed_predictions.append(prediction)
            handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            handle.flush()
            if sleep_seconds:
                time.sleep(sleep_seconds)

    selected_examples = [
        example
        for example in examples
        if example["record_id"] in {prediction["record_id"] for prediction in parsed_predictions}
    ]
    metrics = evaluate_predictions(selected_examples, parsed_predictions)
    metrics_with_context = {
        "model": model,
        "base_url": base_url,
        "wire_api": wire_api,
        "responses_path": responses_path if wire_api == "responses" else "",
        "http_client": http_client,
        "resumed_count": len(existing_by_record_id),
        "processed_count": len(pending_prompts),
        "max_retries": max_retries,
        **metrics,
    }
    metrics_output = Path(metrics_path)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.write_text(
        json.dumps(metrics_with_context, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return metrics_with_context
