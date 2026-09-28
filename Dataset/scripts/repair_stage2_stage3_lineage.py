from __future__ import annotations

from dataclasses import dataclass
import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable

import pandas as pd


KEY_COLUMNS = ["paper_id", "issue_url"]
FINAL_LABEL_COLUMNS = [
    "symptom",
    "root_cause",
    "bug_type",
    "component",
    "sub_component",
    "trigger_condition",
    "consequence",
    "fix_type",
    "severity_or_impact",
]
AUDIT_COLUMNS = [
    "repair_type",
    "paper_id",
    "issue_url",
    "old_stage3_record_id",
    "new_record_id",
    "notes",
]


@dataclass(frozen=True)
class Alignment:
    stage3_index: int
    old_record_id: str
    new_record_id: str


@dataclass(frozen=True)
class RepairPlan:
    alignments: tuple[Alignment, ...]
    missing_indices: tuple[int, ...]
    missing_keys: list[tuple[str, str]]

    @property
    def alignment_count(self) -> int:
        return len(self.alignments)

    @property
    def missing_count(self) -> int:
        return len(self.missing_indices)


def _validate_columns(frame: pd.DataFrame, name: str) -> None:
    required = {"record_id", *KEY_COLUMNS, *FINAL_LABEL_COLUMNS, "original_label_json"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _stage2_ids_by_key(stage2: pd.DataFrame) -> dict[tuple[str, str], set[str]]:
    result: dict[tuple[str, str], set[str]] = {}
    for key, group in stage2.groupby(KEY_COLUMNS, sort=False, dropna=False):
        result[tuple(key)] = set(group["record_id"])
    return result


def build_repair_plan(stage2: pd.DataFrame, stage3: pd.DataFrame) -> RepairPlan:
    _validate_columns(stage2, "Stage 2")
    _validate_columns(stage3, "Stage 3")
    stage2_ids = _stage2_ids_by_key(stage2)
    alignments: list[Alignment] = []
    missing_indices: list[int] = []
    missing_keys: list[tuple[str, str]] = []

    for index, row in stage3.iterrows():
        key = (row["paper_id"], row["issue_url"])
        candidate_ids = stage2_ids.get(key)
        if not candidate_ids:
            missing_indices.append(index)
            missing_keys.append(key)
            continue
        if row["record_id"] in candidate_ids:
            continue
        if len(candidate_ids) > 1:
            raise ValueError(
                "ambiguous Stage 2 key "
                f"{key!r}: candidate record_ids={sorted(candidate_ids)!r}"
            )
        new_record_id = next(iter(candidate_ids))
        alignments.append(
            Alignment(
                stage3_index=index,
                old_record_id=row["record_id"],
                new_record_id=new_record_id,
            )
        )

    return RepairPlan(
        alignments=tuple(alignments),
        missing_indices=tuple(missing_indices),
        missing_keys=missing_keys,
    )


def _stage2_lineage_json(stage3_row: pd.Series) -> str:
    return json.dumps(
        {
            "stage": "stage2",
            "lineage_repair": {
                "reason": "present_in_stage3_but_missing_from_stage2",
                "source_stage": "stage3",
                "source_record_id": stage3_row["record_id"],
            },
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def apply_lineage_repair(
    stage2: pd.DataFrame, stage3: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    plan = build_repair_plan(stage2, stage3)
    repaired_stage2 = stage2.copy(deep=True)
    repaired_stage3 = stage3.copy(deep=True)
    audit_rows: list[dict[str, str]] = []

    for alignment in plan.alignments:
        row = repaired_stage3.loc[alignment.stage3_index]
        repaired_stage3.loc[alignment.stage3_index, "record_id"] = alignment.new_record_id
        audit_rows.append(
            {
                "repair_type": "record_id_aligned",
                "paper_id": row["paper_id"],
                "issue_url": row["issue_url"],
                "old_stage3_record_id": alignment.old_record_id,
                "new_record_id": alignment.new_record_id,
                "notes": "Stage 3 record_id aligned to the matching Stage 2 row.",
            }
        )

    inserted_rows = []
    for index in plan.missing_indices:
        source = repaired_stage3.loc[index].copy()
        for column in FINAL_LABEL_COLUMNS:
            source[column] = ""
        source["original_label_json"] = _stage2_lineage_json(repaired_stage3.loc[index])
        inserted_rows.append(source)
        audit_rows.append(
            {
                "repair_type": "stage2_row_inserted",
                "paper_id": source["paper_id"],
                "issue_url": source["issue_url"],
                "old_stage3_record_id": repaired_stage3.loc[index, "record_id"],
                "new_record_id": source["record_id"],
                "notes": "Stage 3 source fields copied to Stage 2; final labels cleared.",
            }
        )

    if inserted_rows:
        repaired_stage2 = pd.concat(
            [repaired_stage2, pd.DataFrame(inserted_rows, columns=stage2.columns)],
            ignore_index=True,
        )

    audit = pd.DataFrame(audit_rows, columns=AUDIT_COLUMNS)
    return repaired_stage2, repaired_stage3, audit


def validate_repair_result(
    original_stage2: pd.DataFrame,
    original_stage3: pd.DataFrame,
    repaired_stage2: pd.DataFrame,
    repaired_stage3: pd.DataFrame,
) -> None:
    if len(repaired_stage3) != len(original_stage3):
        raise ValueError("Stage 3 row count changed")
    try:
        pd.testing.assert_frame_equal(
            repaired_stage3[FINAL_LABEL_COLUMNS].reset_index(drop=True),
            original_stage3[FINAL_LABEL_COLUMNS].reset_index(drop=True),
            obj="Stage 3 labels",
        )
    except AssertionError as error:
        raise ValueError(f"Stage 3 labels changed: {error}") from error
    non_id_columns = [column for column in original_stage3.columns if column != "record_id"]
    pd.testing.assert_frame_equal(
        repaired_stage3[non_id_columns].reset_index(drop=True),
        original_stage3[non_id_columns].reset_index(drop=True),
        obj="Stage 3 non-ID fields",
    )
    if len(repaired_stage2) < len(original_stage2):
        raise ValueError("Stage 2 rows were removed")
    pd.testing.assert_frame_equal(
        repaired_stage2.iloc[: len(original_stage2)].reset_index(drop=True),
        original_stage2.reset_index(drop=True),
        obj="Original Stage 2 rows",
    )
    inserted = repaired_stage2.iloc[len(original_stage2) :]
    if not inserted.empty and (inserted[FINAL_LABEL_COLUMNS] != "").any().any():
        raise ValueError("Inserted Stage 2 rows contain final labels")

    stage2_ids = _stage2_ids_by_key(repaired_stage2)
    for _, row in repaired_stage3.iterrows():
        key = (row["paper_id"], row["issue_url"])
        if row["record_id"] not in stage2_ids.get(key, set()):
            raise ValueError(f"Stage 3 row is not aligned to Stage 2: {key!r}")


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def _percentage(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0.0%"
    return f"{numerator / denominator * 100:.1f}%"


def _update_metadata_tables(
    root: Path, stage1: pd.DataFrame, stage2: pd.DataFrame, stage3: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = _read_csv(root / "metadata" / "dataset_metadata.csv")
    summary = _read_csv(root / "metadata" / "paper_dataset_summary.csv")
    stage_frames = {"stage1": stage1, "stage2": stage2, "stage3": stage3}
    counts = {
        stage: frame.groupby("paper_id").size().to_dict()
        for stage, frame in stage_frames.items()
    }

    for index, row in metadata.iterrows():
        paper_id = row["paper_id"]
        if paper_id == "TOTAL":
            s1, s2, s3 = len(stage1), len(stage2), len(stage3)
        else:
            s1 = counts["stage1"].get(paper_id, 0)
            s2 = counts["stage2"].get(paper_id, 0)
            s3 = counts["stage3"].get(paper_id, 0)
        metadata.loc[index, "stage1_raw_count"] = str(s1)
        metadata.loc[index, "stage2_filtered_count"] = str(s2)
        metadata.loc[index, "stage3_annotated_count"] = str(s3)
        metadata.loc[index, "stage1_to_stage2_removed"] = str(s1 - s2)
        metadata.loc[index, "stage2_to_stage3_removed"] = str(s2 - s3)
        metadata.loc[index, "stage1_to_stage2_filter_rate"] = _percentage(s1 - s2, s1)
        metadata.loc[index, "stage2_to_stage3_filter_rate"] = _percentage(s2 - s3, s2)

    for index, row in summary.iterrows():
        paper_id = row["paper_id"]
        s1 = counts["stage1"].get(paper_id, 0)
        s2 = counts["stage2"].get(paper_id, 0)
        s3 = counts["stage3"].get(paper_id, 0)
        summary.loc[index, "stage1_raw_count"] = str(s1)
        summary.loc[index, "stage2_filtered_count"] = str(s2)
        summary.loc[index, "stage3_annotated_count"] = str(s3)
        summary.loc[index, "stage1_to_stage2_filter_rate"] = _percentage(s1 - s2, s1)

    return metadata, summary


def _quality_metrics(stages: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    expected_columns = list(stages["stage1"].columns)
    for stage, frame in stages.items():
        rows.append(
            {
                "stage": stage,
                "rows": len(frame),
                "columns": len(frame.columns),
                "schema_matches_expected": list(frame.columns) == expected_columns,
                "record_id_non_null": int((frame["record_id"] != "").sum()),
                "record_id_unique": frame["record_id"].nunique(),
                "record_id_duplicate_rows": int(frame["record_id"].duplicated().sum()),
                "issue_url_non_null": int((frame["issue_url"] != "").sum()),
                "issue_url_unique": frame["issue_url"].nunique(),
                "issue_url_duplicate_rows": int(frame["issue_url"].duplicated().sum()),
                "title_non_empty": int((frame["title"] != "").sum()),
                "body_non_empty": int((frame["body"] != "").sum()),
                "comments_non_empty": int((frame["comments"] != "").sum()),
                "symptom_non_empty": int((frame["symptom"] != "").sum()),
                "root_cause_non_empty": int((frame["root_cause"] != "").sum()),
                "bug_type_non_empty": int((frame["bug_type"] != "").sum()),
                "component_non_empty": int((frame["component"] != "").sum()),
                "fix_type_non_empty": int((frame["fix_type"] != "").sum()),
                "paper_count": frame["paper_id"].nunique(),
            }
        )
    return pd.DataFrame(rows)


def _duplicate_key_rows(stages: dict[str, pd.DataFrame]) -> pd.DataFrame:
    columns = [
        "stage",
        "duplicate_key_column",
        "record_id",
        "paper_id",
        "source_project",
        "issue_url",
        "title",
    ]
    parts = []
    for stage, frame in stages.items():
        for key in ("record_id", "issue_url"):
            duplicate_rows = frame.loc[
                frame[key].duplicated(keep=False),
                ["record_id", "paper_id", "source_project", "issue_url", "title"],
            ].copy()
            duplicate_rows.insert(0, "duplicate_key_column", key)
            duplicate_rows.insert(0, "stage", stage)
            parts.append(duplicate_rows)
    if not parts:
        return pd.DataFrame(columns=columns)
    return pd.concat(parts, ignore_index=True)[columns]


def _replace_count_text(text: str, stage2_count: int) -> str:
    text = re.sub(
        r"(\| Stage 2 Filtered\s+\| `(?:Dataset/)?stage2\.csv` \|[^|]+\| )[\d,]+( \|)",
        lambda match: f"{match.group(1)}{stage2_count:,}{match.group(2)}",
        text,
    )
    text = re.sub(
        r"(`Dataset/stage2\.csv` \| Human-filtered bug-relevant records \| )[\d,]+",
        lambda match: f"{match.group(1)}{stage2_count:,}",
        text,
    )
    return text


def _update_dataset_readme_text(
    text: str, metadata: pd.DataFrame, metrics: pd.DataFrame
) -> str:
    text = _replace_count_text(
        text,
        int(metadata.loc[metadata["paper_id"] == "TOTAL", "stage2_filtered_count"].iloc[0]),
    )
    for row in metadata.loc[metadata["paper_id"] != "TOTAL"].to_dict("records"):
        pattern = (
            rf"(\| `{re.escape(row['paper_id'])}` \| [^|]+ \| )"
            rf"[\d,]+( \| )[\d,]+( \| )[\d,]+( \|)"
        )
        replacement = (
            rf"\g<1>{int(row['stage1_raw_count']):,}"
            rf"\g<2>{int(row['stage2_filtered_count']):,}"
            rf"\g<3>{int(row['stage3_annotated_count']):,}\g<4>"
        )
        text = re.sub(pattern, replacement, text)

    metric_by_stage = metrics.set_index("stage").to_dict("index")
    for stage, label in (("stage1", "Stage 1"), ("stage2", "Stage 2"), ("stage3", "Stage 3")):
        row = metric_by_stage[stage]
        coverage = "100%" if stage == "stage3" else "partial"
        replacement = (
            f"| {label} | yes | {row['paper_count']} | "
            f"{row['record_id_unique']:,} / {row['rows']:,} | "
            f"{row['record_id_duplicate_rows']:,} | "
            f"{row['issue_url_unique']:,} / {row['rows']:,} | "
            f"{row['issue_url_duplicate_rows']:,} | {coverage} |"
        )
        text = re.sub(rf"^\| {re.escape(label)} \|.*$", replacement, text, flags=re.MULTILINE)
    text = re.sub(
        r"The latest health check was run on \d{4}-\d{2}-\d{2}\.",
        "The latest health check was run on 2026-06-22.",
        text,
    )
    return text


def _render_health_report(stages: dict[str, pd.DataFrame], metrics: pd.DataFrame) -> str:
    metric_rows = []
    for row in metrics.to_dict("records"):
        metric_rows.append(
            "| {stage} | {rows:,} | {columns} | {record_id_unique:,} | "
            "{record_id_duplicate_rows:,} | {issue_url_unique:,} | "
            "{issue_url_duplicate_rows:,} |".format(**row)
        )
    return "\n".join(
        [
            "# AutoEmpirical Dataset Health Report",
            "",
            "Generated on 2026-06-22 after the Stage 2 → Stage 3 lineage repair.",
            "",
            "## Dataset And Grain Summary",
            "",
            "| Stage | Rows | Columns | Unique `record_id` | Duplicate `record_id` excess | Unique `issue_url` | Duplicate `issue_url` excess |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            *metric_rows,
            "",
            "## Stage 2 → Stage 3 Lineage",
            "",
            "- Every Stage 3 row now has a matching Stage 2 row with the same `paper_id`, `issue_url`, and `record_id`.",
            "- 21 previously missing Stage 2 rows were restored from Stage 3 source fields.",
            "- Final annotation fields were cleared from the 21 inserted Stage 2 rows.",
            "- Existing Stage 3 labels and its 2,050-row count were preserved.",
            "- Repair details are recorded in `reports/stage2_stage3_lineage_repairs.csv`.",
            "",
            "## Remaining Known Issues",
            "",
            "- `record_id` and `issue_url` are not strict row-level primary keys because some source artifacts represent multiple bugs or repeated records.",
            "- Cross-paper duplicate URLs still require grouped-URL or paper-level evaluation splits.",
            "- Stage 1 → Stage 2 lineage and Stage 1 count-only placeholders remain outside this repair.",
            "",
        ]
    )


def _render_metadata_markdown(metadata: pd.DataFrame, root: Path) -> str:
    rows = metadata[
        [
            "paper_id",
            "project_name",
            "venue",
            "stage1_raw_count",
            "stage2_filtered_count",
            "stage3_annotated_count",
            "stage1_to_stage2_filter_rate",
            "stage2_to_stage3_filter_rate",
        ]
    ]
    table = rows.to_markdown(index=False)
    sizes = []
    for relative in (
        "Dataset/stage1.csv",
        "Dataset/stage2.csv",
        "Dataset/stage3.csv",
        "metadata/dataset_metadata.csv",
    ):
        size_mb = (root / relative).stat().st_size / (1024 * 1024)
        sizes.append(f"| {relative} | {size_mb:.3f} |")
    return "\n".join(
        [
            "# Dataset Metadata",
            "",
            "This metadata is computed from the three unified stage files.",
            "",
            "## File Sizes",
            "",
            "| File | Size (MiB) |",
            "| --- | ---: |",
            *sizes,
            "",
            "## Paper-Level Metadata",
            "",
            table,
            "",
        ]
    )


def _render_paper_overview(
    metadata: pd.DataFrame, summary: pd.DataFrame, stages: dict[str, pd.DataFrame]
) -> str:
    included = summary[
        [
            "venue",
            "paper_name",
            "stage1_raw_count",
            "stage2_filtered_count",
            "stage3_annotated_count",
            "stage1_to_stage2_filter_rate",
            "symptom_coverage",
            "root_cause_coverage",
        ]
    ]
    return "\n".join(
        [
            "# Paper Dataset Overview",
            "",
            "## Overall Counts",
            "",
            "| Stage | Records |",
            "| --- | ---: |",
            f"| Stage 1 Raw | {len(stages['stage1'])} |",
            f"| Stage 2 Filtered | {len(stages['stage2'])} |",
            f"| Stage 3 Annotated | {len(stages['stage3'])} |",
            "",
            "## Included Papers",
            "",
            included.to_markdown(index=False),
            "",
            "Stage 2 counts include the 21 records restored by the Stage 2 → Stage 3 lineage repair.",
            "",
        ]
    )


def _managed_paths(root: Path) -> list[Path]:
    paths = []
    for relative in (
        "Dataset",
        "Benchmark",
        "metadata",
        "reports",
        "research",
        "scripts",
        "tests",
    ):
        directory = root / relative
        if directory.exists():
            paths.extend(
                path
                for path in directory.rglob("*")
                if path.is_file()
                and path.name != "SHA256SUMS.txt"
                and "__pycache__" not in path.parts
                and path.suffix != ".pyc"
            )
    paths.append(root / "README.md")
    return sorted(set(paths), key=lambda path: path.relative_to(root).as_posix())


def _render_hash_manifest(root: Path) -> str:
    lines = []
    for path in _managed_paths(root):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(root).as_posix()}")
    return "\n".join(lines) + "\n"


def repair_repository(root: Path, write: bool = True) -> dict[str, int]:
    root = root.resolve()
    stage1 = _read_csv(root / "Dataset" / "stage1.csv")
    stage2 = _read_csv(root / "Dataset" / "stage2.csv")
    stage3 = _read_csv(root / "Dataset" / "stage3.csv")
    plan = build_repair_plan(stage2, stage3)

    if plan.alignment_count == 0 and plan.missing_count == 0:
        return {"aligned_rows": 0, "inserted_rows": 0, "changed_keys": 0}

    changed_keys = len(
        {
            (stage3.loc[item.stage3_index, "paper_id"], stage3.loc[item.stage3_index, "issue_url"])
            for item in plan.alignments
        }
    )
    if (plan.alignment_count, plan.missing_count, changed_keys) != (749, 21, 747):
        raise ValueError(
            "Unexpected repair scope: "
            f"aligned_rows={plan.alignment_count}, inserted_rows={plan.missing_count}, "
            f"changed_keys={changed_keys}"
        )

    repaired_stage2, repaired_stage3, audit = apply_lineage_repair(stage2, stage3)
    validate_repair_result(stage2, stage3, repaired_stage2, repaired_stage3)
    post_plan = build_repair_plan(repaired_stage2, repaired_stage3)
    if post_plan.alignment_count or post_plan.missing_count:
        raise ValueError("Repair result is not fully aligned")

    if not write:
        return {
            "aligned_rows": plan.alignment_count,
            "inserted_rows": plan.missing_count,
            "changed_keys": changed_keys,
        }

    _write_csv(repaired_stage2, root / "Dataset" / "stage2.csv")
    _write_csv(repaired_stage3, root / "Dataset" / "stage3.csv")
    _write_csv(audit, root / "reports" / "stage2_stage3_lineage_repairs.csv")

    affected_papers = sorted(set(audit["paper_id"]))
    for paper_id in affected_papers:
        paper_dir = root / "Dataset" / "by_paper" / paper_id
        _write_csv(
            repaired_stage2.loc[repaired_stage2["paper_id"] == paper_id],
            paper_dir / "stage2.csv",
        )
        _write_csv(
            repaired_stage3.loc[repaired_stage3["paper_id"] == paper_id],
            paper_dir / "stage3.csv",
        )

    metadata, summary = _update_metadata_tables(
        root, stage1, repaired_stage2, repaired_stage3
    )
    _write_csv(metadata, root / "metadata" / "dataset_metadata.csv")
    _write_csv(summary, root / "metadata" / "paper_dataset_summary.csv")

    stages = {"stage1": stage1, "stage2": repaired_stage2, "stage3": repaired_stage3}
    metrics = _quality_metrics(stages)
    duplicates = _duplicate_key_rows(stages)
    _write_csv(metrics, root / "reports" / "data_quality_metrics.csv")
    _write_csv(duplicates, root / "reports" / "duplicate_key_rows.csv")

    root_readme_path = root / "README.md"
    root_readme_path.write_text(
        _replace_count_text(root_readme_path.read_text(encoding="utf-8"), len(repaired_stage2)),
        encoding="utf-8",
        newline="\n",
    )
    dataset_readme_path = root / "Dataset" / "README.md"
    dataset_readme_path.write_text(
        _update_dataset_readme_text(
            dataset_readme_path.read_text(encoding="utf-8"), metadata, metrics
        ),
        encoding="utf-8",
        newline="\n",
    )
    (root / "reports" / "dataset_health_report.md").write_text(
        _render_health_report(stages, metrics), encoding="utf-8", newline="\n"
    )
    (root / "metadata" / "dataset_metadata.md").write_text(
        _render_metadata_markdown(metadata, root), encoding="utf-8", newline="\n"
    )
    (root / "metadata" / "paper_dataset_overview.md").write_text(
        _render_paper_overview(metadata, summary, stages),
        encoding="utf-8",
        newline="\n",
    )
    (root / "reports" / "SHA256SUMS.txt").write_text(
        _render_hash_manifest(root), encoding="utf-8", newline="\n"
    )

    return {
        "aligned_rows": plan.alignment_count,
        "inserted_rows": plan.missing_count,
        "changed_keys": changed_keys,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="AutoEmpirical repository root",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate and report the repair scope without writing files",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = repair_repository(args.root, write=not args.check)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    main()
