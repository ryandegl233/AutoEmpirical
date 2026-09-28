"""Build a human review index over frozen MAS/Self predictions, without model calls.

Reads the standard-library-only index contract below, not current production
code. Missing fixed-cohort IDs are explicit rows; invalid outputs are not dropped.
The index summarizes saved explanations, never infers why a model was wrong.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


COLUMNS = (
    "domain", "engine", "stage", "record_id", "issue_url", "record_status",
    "duplicate_prediction_id", "gt_decision", "pred_decision", "gt_symptom",
    "pred_symptom", "gt_root_cause", "pred_root_cause", "saved_valid",
    "native_schema_valid", "decision_correct", "symptom_mode", "symptom_scored",
    "symptom_correct", "root_cause_mode", "root_cause_scored", "root_cause_correct",
    "joint_correct", "decision_reason", "symptom_reason", "root_cause_reason",
    "final_reason", "recorded_error_or_unresolved", "stop_reason",
    "mas_citation_count", "self_ledger_count", "prediction_path",
    "prediction_jsonl_line", "prediction_location", "audit_json_pointer",
    "cohort_path", "cohort_record_number", "filter_metric_target",
)


def read_cohort(path: Path) -> list[dict[str, str]]:
    csv.field_size_limit(2**31 - 1)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [{key: (value or "").strip() for key, value in row.items()}
                for row in csv.DictReader(handle)]


def display(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def concise(value: Any, limit: int = 600) -> str:
    text = " ".join(display(value).split())
    return text if len(text) <= limit else text[:limit] + " [excerpt; see full audit]"


def mode(taxonomy: dict, dimension: str) -> str:
    value = taxonomy.get("annotation_modes", {}).get(dimension, "single_label")
    if value not in {"single_label", "multi_label", "free_text", "constant"}:
        raise ValueError(f"Unknown native annotation mode: {value}")
    return value


def canonical(value: Any, taxonomy: dict, dimension: str) -> Any:
    """Validate native labels; preserve raw labels rather than merge variants."""
    kind = mode(taxonomy, dimension)
    allowed = taxonomy.get(dimension, [])
    if kind == "free_text":
        if isinstance(value, str) and value.strip():
            return value.strip()
        raise ValueError("Empty free-text description")
    if kind == "multi_label":
        parts = value.split("||") if isinstance(value, str) else value
        if not isinstance(parts, (list, tuple)) or not parts or not all(isinstance(item, str) for item in parts):
            raise ValueError("Not an atomic label set")
        parts = [item.strip() for item in parts]
        if len(parts) != len(set(parts)) or any(not item or item not in allowed for item in parts):
            raise ValueError("Duplicate, missing, or unknown atomic label")
        return tuple(sorted(parts))
    if kind == "constant" and len(allowed) != 1:
        raise ValueError("Constant taxonomy is not singleton")
    if not isinstance(value, str) or value not in allowed:
        raise ValueError("Unknown categorical label")
    return value


def prediction_parts(prediction: dict, engine: str, stage: str) -> tuple[dict, bool]:
    if engine == "mas":
        return prediction.get("final_prediction") or {}, not prediction.get("invalid", True)
    return {"decision": prediction.get("stage2_prediction"),
            "symptom": prediction.get("symptom_prediction"),
            "root_cause": prediction.get("root_cause_prediction")}, bool(prediction.get(stage + "_valid"))


def score_row(target: dict, predicted: dict, saved_valid: bool, stage: str,
              taxonomy: dict) -> dict:
    scores = {key: "" for key in ("native_schema_valid", "decision_correct", "symptom_correct", "root_cause_correct", "joint_correct")}
    if stage == "stage2":
        schema_valid = predicted.get("decision") in {"accepted_fault", "rejected_candidate"}
        scores["native_schema_valid"] = schema_valid
        scores["decision_correct"] = bool(saved_valid and schema_valid and predicted.get("decision") == target.get("decision")) if target else ""
        return scores
    gold = {dimension: canonical(target.get(dimension), taxonomy, dimension)
            for dimension in ("symptom", "root_cause")} if target else {}
    try:
        actual = {dimension: canonical(predicted.get(dimension), taxonomy, dimension)
                  for dimension in ("symptom", "root_cause")}
        scores["native_schema_valid"] = True
    except (ValueError, TypeError):
        actual = {}
        scores["native_schema_valid"] = False
    jointly_scored = True
    for dimension in ("symptom", "root_cause"):
        scored = mode(taxonomy, dimension) in {"single_label", "multi_label"}
        jointly_scored &= scored
        if target and scored:
            scores[dimension + "_correct"] = bool(saved_valid and actual and actual[dimension] == gold[dimension])
    if target and jointly_scored:
        scores["joint_correct"] = bool(scores["symptom_correct"] and scores["root_cause_correct"])
    return scores


def index_row(*, domain: str, engine: str, stage: str, record_id: str,
              target: dict, prediction: dict | None, taxonomy: dict,
              path: Path | None, line: int | None, duplicate: bool,
              cohort_path: Path, cohort_record_number: int | None) -> dict:
    exists = prediction is not None
    prediction = prediction or {}
    labels, valid = prediction_parts(prediction, engine, stage)
    row = {column: "" for column in COLUMNS}
    row.update(domain=domain, engine=engine, stage=stage, record_id=record_id,
               issue_url=target.get("issue_url", prediction.get("issue_url", "")),
               record_status="missing" if not exists else "unexpected_id" if not target else "saved",
               duplicate_prediction_id=duplicate, saved_valid=valid if exists else False,
               gt_decision=target.get("decision", ""), pred_decision=display(labels.get("decision")),
               gt_symptom=target.get("symptom", ""), pred_symptom=display(labels.get("symptom")),
               gt_root_cause=target.get("root_cause", ""), pred_root_cause=display(labels.get("root_cause")),
               cohort_path=str(cohort_path.resolve()), cohort_record_number=cohort_record_number or "",
               filter_metric_target="agreement_with_paper_selected_cohort" if stage == "stage2" else "")
    for dimension in ("symptom", "root_cause"):
        row[dimension + "_mode"] = mode(taxonomy, dimension)
        row[dimension + "_scored"] = stage == "stage3" and mode(taxonomy, dimension) in {"single_label", "multi_label"}
    row.update(score_row(target, labels, valid and exists, stage, taxonomy))
    if not exists:
        row["recorded_error_or_unresolved"] = "No saved prediction for this fixed-cohort ID at index creation."
        return row
    row.update(prediction_path=str(path.resolve()), prediction_jsonl_line=line,
               prediction_location=f"{path.resolve()}:{line}")
    if engine == "mas":
        audit = prediction.get("explanation_audit") or {}
        rationales = audit.get("rationales") or {}
        for dimension in ("decision", "symptom", "root_cause"):
            row[dimension + "_reason"] = concise((rationales.get(dimension) or {}).get("reason"))
        row["mas_citation_count"] = len(audit.get("citations") or [])
        row["audit_json_pointer"] = "/explanation_audit; /society/turns; /society/forced_finalizer"
        row["recorded_error_or_unresolved"] = concise(prediction.get("error"))
        row["stop_reason"] = (prediction.get("society") or {}).get("stop_reason", "")
    else:
        audit = prediction.get("audit") or {}
        stage_audit = audit.get(stage) or {}
        row["self_ledger_count"] = len(audit.get("evidence_items") or [])
        row["audit_json_pointer"] = f"/audit/{stage}; /audit/evidence_items"
        row["final_reason"] = concise((stage_audit.get("final_decision") or {}).get("rationale"))
        if stage == "stage2":
            assessment = (stage_audit.get("role_assessments") or {}).get("fault_evidence") or {}
            row["decision_reason"] = concise(assessment.get("claim"))
        failure = stage_audit.get("unresolved") or (stage_audit.get("verification") or {}).get("errors") or stage_audit.get("component_failures")
        row["recorded_error_or_unresolved"] = concise(failure)
        row["stop_reason"] = prediction.get("stop_reason", "")
    return row


def build(run_root: Path, cohort_root: Path, output_dir: Path) -> dict:
    output_files = [output_dir / name for name in ("audit_index.csv", "audit_index_manifest.json", "INDEX.md")]
    if any(path.exists() for path in output_files):
        raise FileExistsError("Index output already exists; choose a fresh --output-dir to preserve the earlier index.")
    plan_path = run_root / "run_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8")) if plan_path.exists() else {}
    jobs = {(entry["domain"], entry["engine"], entry["stage"]) for entry in plan.get("jobs", [])
            if entry.get("engine") in {"mas", "self"}}
    if not jobs:
        jobs = {(folder.parent.name, folder.name.split("_")[0], folder.name.split("_")[1])
                for folder in run_root.glob("*/*") if folder.is_dir() and folder.name in
                {"mas_stage2", "mas_stage3", "self_stage2", "self_stage3"}}
    if not jobs:
        raise ValueError("No planned MAS/Self stage jobs found")
    index, inventory, snapshots, malformed = [], [], [], []
    for domain, engine, stage in sorted(jobs):
        cohort_path = cohort_root / domain / ("cohort.csv" if stage == "stage2" else "stage3_sample.csv")
        cohort_bytes = cohort_path.read_bytes()
        targets = read_cohort(cohort_path)
        target_by_id = {row["record_id"]: row for row in targets}
        if len(target_by_id) != len(targets):
            raise ValueError(f"Duplicate frozen cohort ID: {cohort_path}")
        taxonomy = json.loads((cohort_root / domain / "taxonomy.json").read_text(encoding="utf-8"))
        files = sorted((run_root / domain / (engine + "_" + stage)).rglob("*predictions*.jsonl"))
        saved = []
        for path in files:
            payload = path.read_bytes()
            snapshots.append({"path": str(path.resolve()), "byte_count": len(payload),
                              "sha256": hashlib.sha256(payload).hexdigest(),
                              "hash_scope": "exact bytes observed; verify prefix if the file later grows"})
            try:
                decoded = payload.decode("utf-8")
            except UnicodeDecodeError as exc:
                malformed.append({"path": str(path.resolve()), "byte_offset": exc.start, "error": str(exc)})
                decoded = payload[:exc.start].decode("utf-8")
            for line, text in enumerate(decoded.splitlines(), 1):
                if not text.strip():
                    continue
                try:
                    prediction = json.loads(text)
                    if not isinstance(prediction, dict) or not prediction.get("record_id"):
                        raise ValueError("Prediction must be an object with record_id")
                except (ValueError, json.JSONDecodeError) as exc:
                    malformed.append({"path": str(path.resolve()), "jsonl_line": line, "error": str(exc)})
                    continue
                saved.append((prediction, path, line))
        seen = Counter(prediction["record_id"] for prediction, _, _ in saved)
        row_numbers = {target["record_id"]: number for number, target in enumerate(targets, 1)}
        local_rows = []
        for prediction, path, line in saved:
            record_id = prediction["record_id"]
            local_rows.append(index_row(domain=domain, engine=engine, stage=stage, record_id=record_id,
                target=target_by_id.get(record_id, {}), prediction=prediction, taxonomy=taxonomy,
                path=path, line=line, duplicate=seen[record_id] > 1, cohort_path=cohort_path,
                cohort_record_number=row_numbers.get(record_id)))
        for target in targets:
            if target["record_id"] not in seen:
                local_rows.append(index_row(domain=domain, engine=engine, stage=stage, record_id=target["record_id"],
                    target=target, prediction=None, taxonomy=taxonomy, path=None, line=None, duplicate=False,
                    cohort_path=cohort_path, cohort_record_number=row_numbers[target["record_id"]]))
        index.extend(local_rows)
        inventory.append({"domain": domain, "engine": engine, "stage": stage, "expected": len(targets),
                          "saved": len(saved), "saved_valid": sum(bool(row["saved_valid"]) for row in local_rows),
                          "missing": sum(row["record_status"] == "missing" for row in local_rows),
                          "duplicate_ids": [key for key, count in seen.items() if count > 1],
                          "unexpected_ids": sorted(set(seen) - set(target_by_id)),
                          "cohort_path": str(cohort_path.resolve()),
                          "cohort_sha256": hashlib.sha256(cohort_bytes).hexdigest()})
    expected = sum(job["expected"] for job in inventory)
    if plan.get("expected_predictions") == 1500 and expected != 1500:
        raise ValueError(f"Full plan requires 1500 frozen IDs, got {expected}")
    manifest = {"scope": "Human navigation index for MAS/Self; Single LLM excluded. Not a semantic error-cause audit.",
                "run_root": str(run_root.resolve()), "cohort_root": str(cohort_root.resolve()),
                "evidence_projection_version": plan.get("evidence_projection_version"),
                "expected_predictions": expected, "saved_predictions": sum(job["saved"] for job in inventory),
                "saved_valid": sum(job["saved_valid"] for job in inventory),
                "missing_predictions": sum(job["missing"] for job in inventory),
                "index_rows_including_missing": len(index),
                "complete": not malformed and all(not job["missing"] and not job["duplicate_ids"] and not job["unexpected_ids"] for job in inventory),
                "malformed_lines": malformed, "prediction_snapshots": snapshots, "jobs": inventory}
    output_dir.mkdir(parents=True, exist_ok=True)
    with output_files[0].open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(index)
    manifest["audit_index_sha256"] = hashlib.sha256(output_files[0].read_bytes()).hexdigest()
    output_files[1].write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# 实验人工复核索引", "",
             f"[打开 audit_index.csv]({output_files[0].resolve().as_posix()})。每行对应一条保存预测；尚未保存的冻结 cohort ID 也有独立行。", "",
             f"MAS/Self 预期 {expected} 条；已保存 {manifest['saved_predictions']} 条，其中运行记录标记 valid {manifest['saved_valid']} 条；缺失 {manifest['missing_predictions']} 条。完整性：{'完整' if manifest['complete'] else '尚不完整或存在结构异常'}。Single LLM 不在本索引范围。", "",
             "建议先按 domain / engine / stage 筛选，再查看 correctness 为 False 的行。`record_status=missing` 表示该轮没有保存结果，不是模型已给出分类。`saved_valid=False` 的预测仍保留在固定分母中。", "",
             "`prediction_path` 与 `prediction_jsonl_line` 给出原始 JSONL 的绝对路径及从 1 开始的物理行号；`audit_json_pointer` 指向该行内的详细证据账本、角色报告或 MAS 请求/回复。`cohort_record_number` 是 CSV 数据记录序号，不是物理行号。", "",
             "理由列仅摘录该预测已保存的显式 reason / claim / rationale。未保存理由时留空，并提供完整 audit 的位置。此索引没有推断模型出错原因，也没有断言引文在语义上证明分类。", "",
             "IoT 症状按原子标签集合比较，顺序不同不算错；重复或未知标签无效。UAV 自由文本症状与 DL 固定症状均不评分，symptom_correct / joint_correct 留空；根因仍按固定原类别评分。分类标签的拼写和大小写差异不自行合并。", "",
             "`saved_valid` 是运行时保存的状态；`native_schema_valid` 是索引对原任务标签契约的独立检查。只有运行状态有效且所有输出维度通过原任务契约，已评分维度才可能为 True。缺失或无效输出的已评分维度为 False。", "",
             "filter 的 correctness 表示与论文入选集一致性。尤其 PyTorch 的未入选不能解释为不是 bug；不要将该列混入故障识别准确率。", "",
             "| 论文 | 引擎 | 阶段 | 预期 | 保存 | valid | 缺失 |", "|---|---|---|---:|---:|---:|---:|"]
    lines.extend(f"| {job['domain']} | {job['engine']} | {job['stage']} | {job['expected']} | {job['saved']} | {job['saved_valid']} | {job['missing']} |" for job in inventory)
    invalid = [row for row in index if row["record_status"] == "saved" and not row["saved_valid"]][:12]
    if invalid:
        lines.extend(["", "## 已保存的无效或未决记录入口", ""])
        lines.extend(f"- {row['domain']} / {row['engine']} / {row['stage']}：[{row['record_id']}](<{Path(row['prediction_path']).as_posix()}:{row['prediction_jsonl_line']}>)" for row in invalid)
    lines.extend(["", "输入文件的已读字节数与 SHA-256 位于 audit_index_manifest.json。对于仍在增长的 JSONL，它们绑定本次实际读取的字节前缀；生成新索引时请使用新的输出目录，保留旧索引。", ""])
    output_files[2].write_text("\n".join(lines), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--cohort-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or args.run_root / "review_index"
    result = build(args.run_root, args.cohort_root, output_dir)
    print(json.dumps({key: result[key] for key in ("expected_predictions", "saved_predictions", "saved_valid", "missing_predictions", "index_rows_including_missing", "complete")}, ensure_ascii=True))
    print(json.dumps({"index": str((output_dir / "audit_index.csv").resolve()), "readme": str((output_dir / "INDEX.md").resolve())}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
