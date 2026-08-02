from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ISSTA2024_PAPER_ID = "issta2024_bugs_in_pods_understanding_bugs"
ACCEPTED_FAULT = "accepted_fault"
REJECTED_CANDIDATE = "rejected_candidate"
DEFAULT_SEED = 20260730
MAX_CODE_DIFF_BYTES = 200_000

DEFAULT_STAGE1_PATH = (
    "Dataset/by_paper/issta2024_bugs_in_pods_understanding_bugs/stage1.csv"
)
DEFAULT_STAGE2_PATH = (
    "Dataset/by_paper/issta2024_bugs_in_pods_understanding_bugs/stage2.csv"
)
DEFAULT_STAGE3_PATH = (
    "Dataset/by_paper/issta2024_bugs_in_pods_understanding_bugs/stage3.csv"
)

SAFE_MODEL_FIELDS = (
    "source_project",
    "issue_url",
    "title",
    "body",
    "comments",
    "state",
    "created_at",
    "changed_files",
    "code_diff",
)

SYMPTOM_DEFINITIONS = {
    "Build Failure": "The container runtime cannot be built because compilation, package, API, architecture, or operating-system dependencies fail.",
    "Plugin Management Error": "The runtime terminates or fails while loading, calling, registering, or managing a plugin.",
    "Runtime Daemon Crash": "The resident container-runtime daemon crashes or terminates unexpectedly.",
    "System Communication Error": "The runtime terminates while communicating with host-system facilities such as cgroups, namespaces, networking, or other system APIs.",
    "Preset Exit Code": "The runtime terminates with a predetermined abnormal exit code rather than crashing.",
    "Logging Error": "Logging is missing or incorrect, reports inaccurate information, or produces abnormal log volume.",
    "Incorrect Return Value": "A runtime function completes but returns an incorrect or improperly formatted value.",
    "Wrong Container Behavior": "The resulting container lifecycle or runtime behavior is incorrect.",
    "Incorrect Configuration Effect": "A valid or supplied configuration has an incorrect effect on runtime behavior.",
    "Access Permission Denied": "A legitimate runtime operation is denied because the required access permission is unavailable.",
    "Escalized Privilege": "The bug grants privileges beyond those intended, including privilege escalation or container escape.",
    "Out of Memory": "The runtime or a managed container exhausts available memory.",
    "Host Memory Leak": "Runtime execution leaks memory in the host environment.",
    "High Disk Resource Occupancy": "Runtime behavior consumes excessive host disk space.",
    "Excessive Execution Time": "A runtime operation takes abnormally long, hangs, or causes severe performance degradation.",
    "Others": "The observed symptom cannot be assigned to another symptom category in the paper taxonomy.",
}

ROOT_CAUSE_DEFINITIONS = {
    "Build Configuration Error": "Incorrect build flags, dependencies, platform settings, or build-system configuration cause the bug.",
    "Improper Exception Handling": "Errors or exceptional states are missing, mishandled, ignored, or propagated incorrectly.",
    "Inaccurate Permission Assignment": "The implementation assigns an incorrect user, group, capability, or access permission.",
    "Inappropriate Lifecycle Organization": "Container, process, resource, or runtime lifecycle steps are ordered or coordinated incorrectly.",
    "Incorrect 3rd Party Library Usage": "The implementation incorrectly invokes or integrates a third-party library.",
    "Incorrect Image or Config Parsing": "Container image metadata or runtime configuration is parsed, validated, or interpreted incorrectly.",
    "Incorrect Mount Options": "The runtime supplies or handles incorrect filesystem mount options.",
    "Incorrect System Config": "Host, kernel, namespace, cgroup, security, or other system configuration is incorrect.",
    "Plugin Config Error": "A runtime plugin is registered, selected, initialized, or configured incorrectly.",
    "Runtime Shim Config Error": "The runtime shim is created or configured incorrectly.",
    "Unsafe API Usage": "The implementation uses an API in a way that violates its safety or security requirements.",
    "Wrong Code Logic": "The implementation contains incorrect control flow, state updates, conditions, calculations, or other program logic.",
    "Others": "The root cause cannot be assigned to another root-cause category in the paper taxonomy.",
}

SYMPTOM_CODE_TO_LABEL = {
    "A": "Build Failure",
    "B.1.1": "Plugin Management Error",
    "B.1.2": "Runtime Daemon Crash",
    "B.1.3": "System Communication Error",
    "B.2": "Preset Exit Code",
    "C.1": "Logging Error",
    "C.2.1": "Incorrect Return Value",
    "C.2.2": "Wrong Container Behavior",
    "C.2.3": "Incorrect Configuration Effect",
    "C.2.4": "Others",
    "C.3.1": "Access Permission Denied",
    "C.3.2": "Escalized Privilege",
    "D.1": "High Disk Resource Occupancy",
    "D.2": "Excessive Execution Time",
    "D.3.1": "Out of Memory",
    "D.3.2": "Host Memory Leak",
}

ROOT_CAUSE_CODE_TO_LABEL = {
    "A.1": "Improper Exception Handling",
    "A.2": "Incorrect 3rd Party Library Usage",
    "A.3.1": "Unsafe API Usage",
    "A.3.2": "Inappropriate Lifecycle Organization",
    "A.4.1": "Wrong Code Logic",
    "A.4.2": "Incorrect Image or Config Parsing",
    "B.1.1": "Incorrect Mount Options",
    "B.1.2": "Inaccurate Permission Assignment",
    "B.2.1": "Plugin Config Error",
    "B.2.2": "Incorrect System Config",
    "B.3": "Build Configuration Error",
    "B.4": "Runtime Shim Config Error",
    "C": "Others",
}

PAPER_REPORTED_COUNTS = {"stage1": 8271, "stage2": 429, "stage3": 429}


def _allow_large_csv_fields() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


_allow_large_csv_fields()


def _clean(value: object) -> str:
    return "" if value is None else str(value).strip()


def _read_stage(path: str | Path, stage_name: str) -> dict[str, dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"record_id", "paper_id"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{stage_name} is missing columns: {sorted(missing)}")
        rows: dict[str, dict[str, str]] = {}
        for line_number, raw in enumerate(reader, start=2):
            if _clean(raw.get("paper_id")) != ISSTA2024_PAPER_ID:
                continue
            record_id = _clean(raw.get("record_id"))
            if not record_id:
                raise ValueError(f"{stage_name} contains missing record_id at line {line_number}")
            if record_id in rows:
                raise ValueError(f"{stage_name} contains duplicate record_id: {record_id}")
            rows[record_id] = {key: _clean(value) for key, value in raw.items()}
    return rows


def _changed_files(stage1_row: dict[str, str]) -> str:
    raw = stage1_row.get("original_label_json", "")
    try:
        metadata = json.loads(raw) if raw else {}
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"invalid Stage 1 original_label_json for {stage1_row.get('record_id', '')}: {exc}"
        ) from exc
    files = metadata.get("changed_files", [])
    if files is None:
        files = []
    if not isinstance(files, list) or any(not isinstance(item, str) for item in files):
        raise ValueError(
            f"Stage 1 changed_files must be a list of strings for {stage1_row.get('record_id', '')}"
        )
    return json.dumps(files, ensure_ascii=False)


def _validate_taxonomy_fields(
    rows: dict[str, dict[str, str]], stage_name: str
) -> None:
    mappings = (
        ("symptom", "symptom_code", SYMPTOM_CODE_TO_LABEL),
        ("root_cause", "root_cause_code", ROOT_CAUSE_CODE_TO_LABEL),
    )
    for record_id, row in rows.items():
        raw = row.get("original_label_json", "")
        try:
            metadata = json.loads(raw) if raw else {}
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{stage_name} has invalid original_label_json for {record_id}: {exc}"
            ) from exc
        for label_field, code_field, code_to_label in mappings:
            code = _clean(metadata.get(code_field))
            observed_label = _clean(row.get(label_field))
            expected_label = code_to_label.get(code)
            if expected_label is None:
                raise ValueError(
                    f"{stage_name} has unknown {code_field} {code or '<missing>'} "
                    f"for {record_id}"
                )
            if observed_label != expected_label:
                raise ValueError(
                    f"{stage_name} taxonomy fields disagree for {record_id}: "
                    f"{code_field} {code} maps to {expected_label}, "
                    f"not {observed_label or '<missing>'}"
                )


def build_issta2024_taxonomy() -> dict[str, list[str]]:
    return {
        "symptom": list(SYMPTOM_DEFINITIONS),
        "root_cause": list(ROOT_CAUSE_DEFINITIONS),
    }


def _has_complete_sampling_evidence(row: dict[str, str]) -> bool:
    unavailable = {"", "not_available_in_source"}
    required_fields = (
        "record_id",
        "source_project",
        "issue_url",
        "title",
        "body",
        "state",
        "created_at",
        "code_diff",
    )
    if any(_clean(row.get(field)) in unavailable for field in required_fields):
        return False
    try:
        changed_files = json.loads(row.get("changed_files", "[]"))
    except json.JSONDecodeError:
        return False
    code_diff = row.get("code_diff", "")
    return (
        isinstance(changed_files, list)
        and bool(changed_files)
        and all(isinstance(path, str) and path.strip() for path in changed_files)
        and 0 < len(code_diff.encode("utf-8")) <= MAX_CODE_DIFF_BYTES
    )


def _diff_eligibility_audit(
    examples: list[dict[str, str]],
) -> dict[str, Any]:
    sizes = [
        len(row.get("code_diff", "").encode("utf-8"))
        for row in examples
    ]
    oversized = [
        row
        for row, size in zip(examples, sizes)
        if size > MAX_CODE_DIFF_BYTES
    ]
    return {
        "nonempty_count": sum(size > 0 for size in sizes),
        "missing_count": sum(size == 0 for size in sizes),
        "eligible_count": sum(
            0 < size <= MAX_CODE_DIFF_BYTES for size in sizes
        ),
        "oversized_count": len(oversized),
        "oversized_by_decision": dict(
            sorted(Counter(row.get("decision", "") for row in oversized).items())
        ),
        "max_code_diff_bytes_observed": max(sizes, default=0),
    }


def load_issta2024_examples(
    stage1_path: str | Path = DEFAULT_STAGE1_PATH,
    stage2_path: str | Path = DEFAULT_STAGE2_PATH,
    stage3_path: str | Path = DEFAULT_STAGE3_PATH,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    stage1 = _read_stage(stage1_path, "Stage 1")
    stage2 = _read_stage(stage2_path, "Stage 2")
    stage3 = _read_stage(stage3_path, "Stage 3")
    _validate_taxonomy_fields(stage2, "Stage 2")
    _validate_taxonomy_fields(stage3, "Stage 3")

    absent_from_stage1 = sorted(set(stage2) - set(stage1))
    if absent_from_stage1:
        raise ValueError(
            f"Stage 2 record_ids absent from Stage 1: {absent_from_stage1[:5]}"
        )
    if set(stage2) != set(stage3):
        only_stage2 = sorted(set(stage2) - set(stage3))
        only_stage3 = sorted(set(stage3) - set(stage2))
        raise ValueError(
            "Stage 2 and Stage 3 record_ids differ: "
            f"only_stage2={only_stage2[:5]}, only_stage3={only_stage3[:5]}"
        )
    for record_id in sorted(stage2):
        stage2_labels = (
            stage2[record_id].get("symptom", ""),
            stage2[record_id].get("root_cause", ""),
        )
        stage3_labels = (
            stage3[record_id].get("symptom", ""),
            stage3[record_id].get("root_cause", ""),
        )
        if stage2_labels != stage3_labels:
            raise ValueError(
                "Stage 2 and Stage 3 taxonomy fields differ for "
                f"{record_id}: stage2={stage2_labels}, stage3={stage3_labels}"
            )

    examples: list[dict[str, str]] = []
    for record_id, source in sorted(stage1.items()):
        accepted = record_id in stage2
        gold = stage3.get(record_id, {})
        examples.append(
            {
                "record_id": record_id,
                "paper_id": ISSTA2024_PAPER_ID,
                **{
                    field: source.get(field, "")
                    for field in SAFE_MODEL_FIELDS
                    if field != "changed_files"
                },
                "changed_files": _changed_files(source),
                "decision": ACCEPTED_FAULT if accepted else REJECTED_CANDIDATE,
                "symptom": gold.get("symptom", "") if accepted else "",
                "root_cause": gold.get("root_cause", "") if accepted else "",
            }
        )

    local_counts = {
        "stage1": len(stage1),
        "stage2": len(stage2),
        "stage3": len(stage3),
    }
    audit: dict[str, Any] = {
        "local_counts": local_counts,
        "paper_reported_counts": dict(PAPER_REPORTED_COUNTS),
        "count_deltas": {
            stage: local_counts[stage] - PAPER_REPORTED_COUNTS[stage]
            for stage in PAPER_REPORTED_COUNTS
        },
        "counts_match_paper": local_counts == PAPER_REPORTED_COUNTS,
        "stage2_subset_of_stage1": True,
        "stage2_stage3_ids_equal": True,
    }
    return examples, audit


def select_stage3_sample(
    examples: list[dict[str, str]],
    sample_size: int = 50,
    seed: int = DEFAULT_SEED,
) -> list[dict[str, str]]:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    taxonomy = build_issta2024_taxonomy()
    expected_symptoms = set(taxonomy["symptom"])
    expected_root_causes = set(taxonomy["root_cause"])
    pool = [
        row
        for row in examples
        if row.get("decision") == ACCEPTED_FAULT
        and row.get("symptom") in expected_symptoms
        and row.get("root_cause") in expected_root_causes
        and _has_complete_sampling_evidence(row)
    ]
    if len(pool) < sample_size:
        raise ValueError(
            f"requested {sample_size} Stage 3 records but only {len(pool)} are eligible"
        )

    rng = random.Random(seed)
    ranked = sorted(pool, key=lambda row: row["record_id"])
    rng.shuffle(ranked)
    rank = {row["record_id"]: index for index, row in enumerate(ranked)}
    selected: list[dict[str, str]] = []
    selected_ids: set[str] = set()
    uncovered_symptoms = set(expected_symptoms)
    uncovered_root_causes = set(expected_root_causes)

    while uncovered_symptoms or uncovered_root_causes:
        candidates = [row for row in ranked if row["record_id"] not in selected_ids]
        if not candidates:
            break
        best = max(
            candidates,
            key=lambda row: (
                int(row["symptom"] in uncovered_symptoms)
                + int(row["root_cause"] in uncovered_root_causes),
                int(row["symptom"] in uncovered_symptoms),
                -rank[row["record_id"]],
            ),
        )
        gain = int(best["symptom"] in uncovered_symptoms) + int(
            best["root_cause"] in uncovered_root_causes
        )
        if gain == 0:
            break
        selected.append(best)
        selected_ids.add(best["record_id"])
        uncovered_symptoms.discard(best["symptom"])
        uncovered_root_causes.discard(best["root_cause"])

    if uncovered_symptoms or uncovered_root_causes:
        raise ValueError(
            "eligible Stage 3 records do not cover the complete taxonomy: "
            f"missing_symptoms={sorted(uncovered_symptoms)}, "
            f"missing_root_causes={sorted(uncovered_root_causes)}"
        )
    if len(selected) > sample_size:
        raise ValueError(
            f"sample_size {sample_size} is too small to cover the complete taxonomy"
        )

    remaining = [row for row in ranked if row["record_id"] not in selected_ids]
    selected.extend(remaining[: sample_size - len(selected)])
    return sorted((dict(row) for row in selected), key=lambda row: row["record_id"])


def _proportional_project_quotas(
    pool: list[dict[str, str]], sample_size: int
) -> dict[str, int]:
    counts = Counter(row["source_project"] for row in pool)
    if not counts:
        raise ValueError("negative pool is empty")
    total = sum(counts.values())
    exact = {
        project: sample_size * count / total for project, count in counts.items()
    }
    quotas = {project: math.floor(value) for project, value in exact.items()}
    remaining = sample_size - sum(quotas.values())
    order = sorted(
        counts,
        key=lambda project: (-(exact[project] - quotas[project]), project),
    )
    for project in order[:remaining]:
        quotas[project] += 1
    return quotas


def select_stage2_cohort(
    examples: list[dict[str, str]],
    positives: list[dict[str, str]],
    negative_count: int = 50,
    seed: int = DEFAULT_SEED,
) -> list[dict[str, str]]:
    if negative_count <= 0:
        raise ValueError("negative_count must be positive")
    example_by_id = {row["record_id"]: row for row in examples}
    if len(example_by_id) != len(examples):
        raise ValueError("examples contain duplicate record_id")
    positive_ids = [row["record_id"] for row in positives]
    if len(set(positive_ids)) != len(positive_ids):
        raise ValueError("positives contain duplicate record_id")
    for record_id in positive_ids:
        source = example_by_id.get(record_id)
        if (
            source is None
            or source.get("decision") != ACCEPTED_FAULT
            or not _has_complete_sampling_evidence(source)
        ):
            raise ValueError(f"positive record is not an accepted example: {record_id}")

    negative_pool = [
        row
        for row in examples
        if row.get("decision") == REJECTED_CANDIDATE
        and _has_complete_sampling_evidence(row)
    ]
    if len(negative_pool) < negative_count:
        raise ValueError(
            f"requested {negative_count} negatives but only {len(negative_pool)} are eligible"
        )
    quotas = _proportional_project_quotas(negative_pool, negative_count)
    rng = random.Random(seed + 1)
    selected_negatives: list[dict[str, str]] = []
    for project in sorted(quotas):
        project_pool = sorted(
            (
                row
                for row in negative_pool
                if row.get("source_project") == project
            ),
            key=lambda row: row["record_id"],
        )
        selected_negatives.extend(rng.sample(project_pool, quotas[project]))

    cohort = [*(dict(row) for row in positives), *(dict(row) for row in selected_negatives)]
    return sorted(cohort, key=lambda row: row["record_id"])


def _changed_files_for_prompt(record: dict[str, str]) -> list[str]:
    try:
        files = json.loads(record.get("changed_files", "[]"))
    except json.JSONDecodeError:
        files = []
    return files if isinstance(files, list) else []


def _evidence_prompt(record: dict[str, str]) -> str:
    files = _changed_files_for_prompt(record)
    changed_files = "\n".join(f"- {path}" for path in files) or "- not available"
    comments = record.get("comments", "").strip()
    if comments in {"", "no_comments_in_source", "not_available_in_source"}:
        comments = "not available"
    return f"""Project:
{record.get("source_project", "")}

Commit URL:
{record.get("issue_url", "")}

Commit Title:
{record.get("title", "")}

Commit State:
{record.get("state", "")}

Committed At:
{record.get("created_at", "")}

Commit Message:
{record.get("body", "")}

Available Discussion:
{comments}

Changed Files:
{changed_files}

Code Diff (unified diff from GitHub API):
{record.get("code_diff", "")}
"""


def build_stage2_system_prompt() -> str:
    return """You are reproducing the ISSTA2024 empirical study “Bugs in Pods” on container runtime systems.

Decide whether the supplied commit explicitly repairs a pre-existing software fault in runc, gVisor, containerd, or CRI-O. Accept commits that repair observable incorrect behavior, crashes, build failures, security or permission defects, configuration effects, resource/performance problems, or other identifiable implementation faults.

Reject feature additions without repair evidence, refactoring or cleanup, merge/release/documentation commits, test-only maintenance, dependency updates without a concrete fault, and commits that merely contain words such as “fix” without evidence of a pre-existing fault.

Use only the supplied commit evidence. Do not infer dataset membership or request a gold label.

Return ONLY strict JSON with exactly one key named decision. Its value must be accepted_fault or rejected_candidate. Do not add explanations, Markdown, or other keys.
"""


def build_stage2_user_prompt(record: dict[str, str]) -> str:
    return "Classify this container-runtime commit.\n\n" + _evidence_prompt(record)


def _taxonomy_prompt(
    labels: list[str], definitions: dict[str, str]
) -> str:
    lines: list[str] = []
    for label in labels:
        lines.extend((label, f"  Definition: {definitions[label]}"))
    return "\n".join(lines)


def build_stage3_system_prompt(taxonomy: dict[str, list[str]]) -> str:
    expected = build_issta2024_taxonomy()
    if taxonomy != expected:
        raise ValueError("taxonomy must match the fixed ISSTA2024 leaf labels")
    symptom_section = _taxonomy_prompt(
        taxonomy["symptom"], SYMPTOM_DEFINITIONS
    )
    root_cause_section = _taxonomy_prompt(
        taxonomy["root_cause"], ROOT_CAUSE_DEFINITIONS
    )
    return f"""You are reproducing the ISSTA2024 “Bugs in Pods” taxonomy analysis of container runtime bugs.

Classify one confirmed bug-fixing commit using exactly one symptom and exactly one root cause from the fixed leaf labels below. A symptom is the externally observable failure. A root cause is the implementation, API, lifecycle, parsing, permission, mount, plugin, shim, system, or build defect that produces it. Use Others only when the evidence genuinely fits no named category. Do not invent labels.

Allowed symptom labels:
{symptom_section}

Allowed root-cause labels:
{root_cause_section}

Use only the supplied commit evidence. Return ONLY strict JSON with exactly the keys symptom and root_cause. Do not add explanations, Markdown, or other keys.
"""


def build_stage3_user_prompt(record: dict[str, str]) -> str:
    return "Classify this confirmed container-runtime bug fix.\n\n" + _evidence_prompt(
        record
    )


def build_society_task(
    record: dict[str, str],
    stage: str,
    taxonomy: dict[str, list[str]],
) -> str:
    if stage == "stage2":
        return build_stage2_system_prompt() + "\n" + build_stage2_user_prompt(record)
    if stage == "stage3":
        return build_stage3_system_prompt(taxonomy) + "\n" + build_stage3_user_prompt(
            record
        )
    raise ValueError("stage must be stage2 or stage3")


SAMPLE_FIELDS = [
    "record_id",
    "paper_id",
    *SAFE_MODEL_FIELDS,
    "decision",
    "symptom",
    "root_cause",
]


def _write_sample(rows: list[dict[str, str]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SAMPLE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_prompts(
    rows: list[dict[str, str]],
    path: Path,
    *,
    stage: str,
    taxonomy: dict[str, list[str]],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    system_prompt = (
        build_stage2_system_prompt()
        if stage == "stage2"
        else build_stage3_system_prompt(taxonomy)
    )
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            prompt = {
                "record_id": row["record_id"],
                "paper_id": row["paper_id"],
                "issue_url": row["issue_url"],
                "ground_truth": (
                    row["decision"]
                    if stage == "stage2"
                    else {
                        "symptom": row["symptom"],
                        "root_cause": row["root_cause"],
                    }
                ),
                "system_prompt": system_prompt,
                "user_prompt": (
                    build_stage2_user_prompt(row)
                    if stage == "stage2"
                    else build_stage3_user_prompt(row)
                ),
            }
            handle.write(json.dumps(prompt, ensure_ascii=False) + "\n")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distribution(rows: list[dict[str, str]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(row[field] for row in rows if row.get(field)).items()))


def prepare_issta2024_artifacts(
    *,
    stage1_path: str | Path = DEFAULT_STAGE1_PATH,
    stage2_path: str | Path = DEFAULT_STAGE2_PATH,
    stage3_path: str | Path = DEFAULT_STAGE3_PATH,
    output_dir: str | Path = "Benchmark/results/issta2024_bugs_in_pods_baseline",
    positive_count: int = 50,
    negative_count: int = 50,
    seed: int = DEFAULT_SEED,
) -> dict[str, Path]:
    examples, audit = load_issta2024_examples(
        stage1_path=stage1_path,
        stage2_path=stage2_path,
        stage3_path=stage3_path,
    )
    taxonomy = build_issta2024_taxonomy()
    observed_symptoms = {
        row["symptom"] for row in examples if row["decision"] == ACCEPTED_FAULT
    }
    observed_root_causes = {
        row["root_cause"] for row in examples if row["decision"] == ACCEPTED_FAULT
    }
    if observed_symptoms != set(taxonomy["symptom"]):
        raise ValueError(
            "dataset symptom labels differ from the ISSTA2024 taxonomy: "
            f"observed={sorted(observed_symptoms)}"
        )
    if observed_root_causes != set(taxonomy["root_cause"]):
        raise ValueError(
            "dataset root-cause labels differ from the ISSTA2024 taxonomy: "
            f"observed={sorted(observed_root_causes)}"
        )

    stage3_sample = select_stage3_sample(
        examples, sample_size=positive_count, seed=seed
    )
    stage2_cohort = select_stage2_cohort(
        examples,
        stage3_sample,
        negative_count=negative_count,
        seed=seed,
    )

    output = Path(output_dir)
    paths = {
        "stage2_sample": output / "issta2024_stage2_filter_sample.csv",
        "stage2_prompts": output / "issta2024_stage2_filter_prompts.jsonl",
        "stage3_sample": output / "issta2024_stage3_llm_sample.csv",
        "stage3_prompts": output / "issta2024_stage3_llm_prompts.jsonl",
        "taxonomy": output / "issta2024_taxonomy.json",
        "manifest": output / "issta2024_manifest.json",
    }
    _write_sample(stage2_cohort, paths["stage2_sample"])
    _write_prompts(
        stage2_cohort,
        paths["stage2_prompts"],
        stage="stage2",
        taxonomy=taxonomy,
    )
    _write_sample(stage3_sample, paths["stage3_sample"])
    _write_prompts(
        stage3_sample,
        paths["stage3_prompts"],
        stage="stage3",
        taxonomy=taxonomy,
    )
    paths["taxonomy"].write_text(
        json.dumps(taxonomy, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    counts = {
        "stage2_total": len(stage2_cohort),
        "stage2_positive": sum(
            row["decision"] == ACCEPTED_FAULT for row in stage2_cohort
        ),
        "stage2_negative": sum(
            row["decision"] == REJECTED_CANDIDATE for row in stage2_cohort
        ),
        "stage3_total": len(stage3_sample),
    }
    manifest = {
        "task": "issta2024_bugs_in_pods_baseline",
        "paper_id": ISSTA2024_PAPER_ID,
        "seed": seed,
        "source_paths": {
            "stage1": str(stage1_path),
            "stage2": str(stage2_path),
            "stage3": str(stage3_path),
        },
        "counts": counts,
        "stage3_coverage": {
            "symptom": len({row["symptom"] for row in stage3_sample}),
            "root_cause": len({row["root_cause"] for row in stage3_sample}),
        },
        "stage3_distribution": {
            "symptom": _distribution(stage3_sample, "symptom"),
            "root_cause": _distribution(stage3_sample, "root_cause"),
            "source_project": _distribution(stage3_sample, "source_project"),
        },
        "stage2_distribution": {
            "decision": _distribution(stage2_cohort, "decision"),
            "source_project": _distribution(stage2_cohort, "source_project"),
        },
        "model_input_fields": list(SAFE_MODEL_FIELDS),
        "sampling_eligibility": {
            "requires_nonempty_code_diff": True,
            "max_code_diff_bytes": MAX_CODE_DIFF_BYTES,
            "code_diff_truncated_in_prompts": False,
            "audit": _diff_eligibility_audit(examples),
        },
        "gold_fields_excluded_from_prompts": [
            "decision",
            "symptom",
            "root_cause",
            "symptom_code",
            "root_cause_code",
            "original_label_json",
            "Lack of test",
        ],
        "provenance_audit": audit,
        "artifact_sha256": {
            key: _sha256(path)
            for key, path in paths.items()
            if key != "manifest"
        },
    }
    paths["manifest"].write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return paths
