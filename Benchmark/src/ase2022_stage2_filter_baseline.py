from __future__ import annotations

import csv
import json
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Callable

import pandas as pd

from Benchmark.src.ase2022_llm_baseline import _is_retryable_network_error, call_model


ASE2022_PAPER_ID = "ase2022_towards_understanding_the_faults_of"
ACCEPTED_FAULT = "accepted_fault"
REJECTED_CANDIDATE = "rejected_candidate"
ALLOWED_DECISIONS = {ACCEPTED_FAULT, REJECTED_CANDIDATE}
SAFE_TEXT_FIELDS = ("title", "body", "comments", "state", "created_at")


def _clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _paper_rows(path: str | Path, stage_name: str) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    required = {"record_id", "paper_id"}
    missing_columns = required - set(frame.columns)
    if missing_columns:
        raise ValueError(f"{stage_name} is missing columns: {sorted(missing_columns)}")
    frame = frame[frame["paper_id"] == ASE2022_PAPER_ID].copy()
    record_ids = frame["record_id"].fillna("").astype(str).str.strip()
    if (record_ids == "").any():
        raise ValueError(f"{stage_name} contains missing record_id")
    if record_ids.duplicated().any():
        duplicates = sorted(record_ids[record_ids.duplicated(keep=False)].unique())
        raise ValueError(f"{stage_name} contains duplicate record_id: {duplicates[:5]}")
    frame["record_id"] = record_ids
    return frame


def load_ase2022_stage2_filter_examples(
    stage1_path: str | Path = "Dataset/stage1.csv",
    stage2_path: str | Path = "Dataset/stage2.csv",
) -> list[dict[str, str]]:
    stage1 = _paper_rows(stage1_path, "Stage 1")
    stage2 = _paper_rows(stage2_path, "Stage 2")
    stage2_ids = set(stage2["record_id"])
    stage1_ids = set(stage1["record_id"])
    absent = sorted(stage2_ids - stage1_ids)
    if absent:
        raise ValueError(f"Stage 2 record_ids absent from Stage 1: {absent[:5]}")

    for field in ("issue_url", *SAFE_TEXT_FIELDS):
        if field not in stage1.columns:
            raise ValueError(f"Stage 1 is missing columns: ['{field}']")

    examples: list[dict[str, str]] = []
    for _, row in stage1.sort_values("record_id").iterrows():
        record_id = str(row["record_id"])
        example = {
            "record_id": record_id,
            "paper_id": ASE2022_PAPER_ID,
            "issue_url": _clean_text(row["issue_url"]),
            **{field: _clean_text(row[field]) for field in SAFE_TEXT_FIELDS},
            "decision": ACCEPTED_FAULT if record_id in stage2_ids else REJECTED_CANDIDATE,
        }
        examples.append(example)
    return examples


def _has_usable_evidence(example: dict[str, str]) -> bool:
    return any(str(example.get(field, "")).strip() for field in ("title", "body", "comments"))


def select_balanced_sample(
    examples: list[dict[str, str]],
    per_class: int = 50,
    seed: int = 20260711,
) -> list[dict[str, str]]:
    if per_class <= 0:
        raise ValueError("per_class must be positive")
    selected: list[dict[str, str]] = []
    for offset, decision in enumerate((ACCEPTED_FAULT, REJECTED_CANDIDATE)):
        pool = [
            example
            for example in examples
            if example.get("decision") == decision and _has_usable_evidence(example)
        ]
        if len(pool) < per_class:
            raise ValueError(
                f"{decision}: requested {per_class} records but only {len(pool)} are eligible"
            )
        rng = random.Random(seed + offset)
        selected.extend(rng.sample(sorted(pool, key=lambda row: row["record_id"]), per_class))
    return sorted(selected, key=lambda row: row["record_id"])


def build_system_prompt() -> str:
    return """You are screening GitHub issue records for the ASE2022 empirical study of faults in JavaScript-based deep-learning systems.

Decide whether the issue provides evidence of a real, identifiable fault in a JavaScript deep-learning framework, third-party library, or application.

Retain reports of observable failures, incorrect behavior, crashes, build or initialization failures, performance faults, or documentation faults. Reject feature requests, general usage questions without fault evidence, irrelevant discussions, unclear or insufficient reports, and keyword matches that are not errors.

Return ONLY strict JSON with exactly one key named decision. Its value must be accepted_fault or rejected_candidate. Do not add explanations, Markdown, or other keys.
"""


def build_user_prompt(example: dict[str, str]) -> str:
    return f"""Classify this candidate issue.

Title:
{example.get("title", "")}

State:
{example.get("state", "")}

Created At:
{example.get("created_at", "")}

Body:
{example.get("body", "")}

Comments:
{example.get("comments", "")}
"""


def parse_filter_response(raw_text: str) -> dict[str, object]:
    raw = raw_text.strip() if isinstance(raw_text, str) else ""
    normalized = raw
    format_normalized = False
    output_format = "strict_json"
    try:
        parsed = json.loads(normalized)
    except (json.JSONDecodeError, TypeError, AttributeError) as strict_exc:
        fenced = re.fullmatch(
            r"```(?:json)?[ \t]*\r?\n(?P<body>[\s\S]*?)\r?\n```",
            raw,
            flags=re.IGNORECASE,
        )
        if not fenced:
            return {
                "decision": "",
                "invalid": True,
                "error": f"invalid JSON: {strict_exc}",
                "normalized_output": raw,
                "format_normalized": False,
                "output_format": "invalid",
            }
        normalized = fenced.group("body").strip()
        format_normalized = True
        output_format = "markdown_fenced_json"
        try:
            parsed = json.loads(normalized)
        except (json.JSONDecodeError, TypeError, AttributeError) as fenced_exc:
            return {
                "decision": "",
                "invalid": True,
                "error": f"invalid JSON inside Markdown fence: {fenced_exc}",
                "normalized_output": normalized,
                "format_normalized": True,
                "output_format": output_format,
            }

    metadata = {
        "normalized_output": normalized,
        "format_normalized": format_normalized,
        "output_format": output_format,
    }
    if not isinstance(parsed, dict):
        return {
            "decision": "",
            "invalid": True,
            "error": "output must be a JSON object",
            **metadata,
        }
    if set(parsed) != {"decision"}:
        return {
            "decision": "",
            "invalid": True,
            "error": "output must contain exactly the decision key",
            **metadata,
        }
    decision = parsed["decision"]
    if decision not in ALLOWED_DECISIONS:
        return {
            "decision": "",
            "invalid": True,
            "error": "unsupported decision label",
            **metadata,
        }
    return {"decision": decision, "invalid": False, "error": "", **metadata}


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate_filter_predictions(
    examples: list[dict[str, str]],
    predictions: list[dict[str, object]],
) -> dict[str, int | float]:
    expected = {example["record_id"]: example["decision"] for example in examples}
    if len(expected) != len(examples):
        raise ValueError("examples contain duplicate record_id")
    prediction_by_id = {str(row["record_id"]): row for row in predictions}
    if len(prediction_by_id) != len(predictions):
        raise ValueError("predictions contain duplicate record_id")

    tp = fp = tn = fn = invalid_count = 0
    for record_id, truth in expected.items():
        prediction = prediction_by_id.get(record_id)
        if prediction is None or prediction.get("invalid", False):
            invalid_count += 1
            continue
        decision = prediction.get("decision")
        if decision not in ALLOWED_DECISIONS:
            invalid_count += 1
        elif truth == ACCEPTED_FAULT and decision == ACCEPTED_FAULT:
            tp += 1
        elif truth == ACCEPTED_FAULT:
            fn += 1
        elif decision == ACCEPTED_FAULT:
            fp += 1
        else:
            tn += 1

    n = len(examples)
    valid_count = tp + fp + tn + fn
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    specificity = _safe_ratio(tn, tn + fp)
    return {
        "n": n,
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "valid_denominator": valid_count,
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "accuracy": _safe_ratio(tp + tn, valid_count),
        "precision": precision,
        "recall": recall,
        "f1": _safe_ratio(2 * precision * recall, precision + recall),
        "specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2,
        "invalid_rate": _safe_ratio(invalid_count, n),
    }


def write_sample_csv(
    examples: list[dict[str, str]], output_path: str | Path
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "record_id",
        "paper_id",
        "issue_url",
        *SAFE_TEXT_FIELDS,
        "decision",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(examples)
    return path


def write_prompts_jsonl(
    examples: list[dict[str, str]], output_path: str | Path
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    system_prompt = build_system_prompt()
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            row = {
                "record_id": example["record_id"],
                "ground_truth": example["decision"],
                "system_prompt": system_prompt,
                "user_prompt": build_user_prompt(example),
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def _decision_counts(examples: list[dict[str, str]]) -> dict[str, int]:
    counts = Counter(example["decision"] for example in examples)
    unsupported = set(counts) - ALLOWED_DECISIONS
    if unsupported:
        raise ValueError(f"unsupported decisions: {sorted(unsupported)}")
    return {decision: counts.get(decision, 0) for decision in (ACCEPTED_FAULT, REJECTED_CANDIDATE)}


def write_manifest(
    all_examples: list[dict[str, str]],
    selected_examples: list[dict[str, str]],
    output_path: str | Path,
    *,
    seed: int,
    per_class: int,
    stage1_path: str | Path,
    stage2_path: str | Path,
    mode: str,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "task": "ase2022_stage2_filter",
        "mode": mode,
        "seed": seed,
        "per_class": per_class,
        "stage1_path": str(stage1_path),
        "stage2_path": str(stage2_path),
        "all_counts": _decision_counts(all_examples),
        "selected_counts": _decision_counts(selected_examples),
        "selected_count": len(selected_examples),
        "label_semantics": {
            ACCEPTED_FAULT: "record_id is present in the ASE2022 Stage 2 dataset",
            REJECTED_CANDIDATE: "record_id is absent from the ASE2022 Stage 2 dataset",
        },
        "model_input_fields": list(SAFE_TEXT_FIELDS),
    }
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _load_prompt_rows(path: str | Path, limit: int | None) -> list[dict[str, str]]:
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if limit is not None:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        rows = rows[:limit]
    record_ids = [str(row["record_id"]) for row in rows]
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("prompts contain duplicate record_id")
    return rows


def _load_filter_predictions(path: Path, model: str) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    existing: dict[str, dict[str, object]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid prediction JSONL at line {line_number}: {exc}") from exc
        if row.get("model") != model:
            continue
        record_id = str(row.get("record_id", ""))
        if record_id in existing:
            raise ValueError(f"predictions contain duplicate record_id: {record_id}")
        existing[record_id] = row
    return existing


def run_filter_prompts(
    *,
    prompts_path: str | Path,
    predictions_path: str | Path,
    metrics_path: str | Path,
    model: str,
    api_key: str = "",
    base_url: str = "",
    wire_api: str = "chat_completions",
    responses_path: str = "/responses",
    http_client: str = "urllib",
    limit: int | None = None,
    sleep_seconds: float = 0.0,
    resume: bool = True,
    max_retries: int = 3,
    retry_delay_seconds: float = 2.0,
    model_caller: Callable[..., str] | None = None,
) -> dict[str, object]:
    prompts = _load_prompt_rows(prompts_path, limit)
    predictions_output = Path(predictions_path)
    predictions_output.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_filter_predictions(predictions_output, model) if resume else {}
    allowed_ids = {str(row["record_id"]) for row in prompts}
    existing = {record_id: row for record_id, row in existing.items() if record_id in allowed_ids}
    pending = [row for row in prompts if str(row["record_id"]) not in existing]
    caller = model_caller or call_model
    output_mode = "a" if resume and predictions_output.exists() else "w"

    new_rows: list[dict[str, object]] = []
    with predictions_output.open(output_mode, encoding="utf-8") as handle:
        for prompt in pending:
            started = time.perf_counter()
            attempts = 0
            raw_output = ""
            request_failed = False
            parsed: dict[str, object]
            while True:
                attempts += 1
                try:
                    raw_output = caller(
                        system_prompt=prompt["system_prompt"],
                        user_prompt=prompt["user_prompt"],
                        model=model,
                        api_key=api_key,
                        base_url=base_url,
                        wire_api=wire_api,
                        responses_path=responses_path,
                        http_client=http_client,
                    )
                    parsed = parse_filter_response(raw_output)
                    break
                except Exception as exc:
                    if attempts > max_retries or not _is_retryable_network_error(exc):
                        request_failed = True
                        parsed = {
                            "decision": "",
                            "invalid": True,
                            "error": str(exc),
                            "normalized_output": "",
                            "format_normalized": False,
                            "output_format": "request_failed",
                        }
                        break
                    if retry_delay_seconds:
                        time.sleep(retry_delay_seconds * (2 ** (attempts - 1)))
            prediction = {
                "record_id": str(prompt["record_id"]),
                "ground_truth": prompt["ground_truth"],
                "model": model,
                "raw_output": raw_output,
                "latency_seconds": round(time.perf_counter() - started, 3),
                "attempts": attempts,
                "request_failed": request_failed,
                **parsed,
            }
            new_rows.append(prediction)
            handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            handle.flush()
            if sleep_seconds:
                time.sleep(sleep_seconds)

    all_by_id = {**existing, **{str(row["record_id"]): row for row in new_rows}}
    ordered_predictions = [all_by_id[str(prompt["record_id"])] for prompt in prompts]
    examples = [
        {"record_id": str(prompt["record_id"]), "decision": str(prompt["ground_truth"])}
        for prompt in prompts
    ]
    metrics = evaluate_filter_predictions(examples, ordered_predictions)
    metrics_with_context: dict[str, object] = {
        "model": model,
        "task": "ase2022_stage2_filter",
        "wire_api": wire_api,
        "http_client": http_client,
        "resumed_count": len(existing),
        "processed_count": len(new_rows),
        "max_retries": max_retries,
        **metrics,
    }
    metrics_output = Path(metrics_path)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.write_text(
        json.dumps(metrics_with_context, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return metrics_with_context
