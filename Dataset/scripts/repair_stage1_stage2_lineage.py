from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Iterable

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Dataset.scripts.repair_stage2_stage3_lineage import (
    FINAL_LABEL_COLUMNS,
    _duplicate_key_rows,
    _quality_metrics,
    _read_csv,
    _render_hash_manifest,
    _render_metadata_markdown,
    _render_paper_overview,
    _update_dataset_readme_text,
    _update_metadata_tables,
    _write_csv,
)


KEY_COLUMNS = ["paper_id", "issue_url"]
EXPECTED_INSERTED_ROWS = 1597
EXPECTED_INSERTED_KEYS = 1590
AUDIT_COLUMNS = [
    "paper_id",
    "issue_url",
    "record_id",
    "source_stage2_index",
    "notes",
]


@dataclass(frozen=True)
class Stage1Insertion:
    stage2_index: int
    paper_id: str
    issue_url: str
    record_id: str


@dataclass(frozen=True)
class Stage1LineageRepairPlan:
    insertions: tuple[Stage1Insertion, ...]

    @property
    def inserted_rows(self) -> int:
        return len(self.insertions)

    @property
    def inserted_keys(self) -> int:
        return len({(item.paper_id, item.issue_url) for item in self.insertions})


def _stage1_key_set(stage1: pd.DataFrame) -> set[tuple[str, str]]:
    return set(zip(stage1["paper_id"], stage1["issue_url"]))


def build_stage1_lineage_repair_plan(
    stage1: pd.DataFrame, stage2: pd.DataFrame
) -> Stage1LineageRepairPlan:
    stage1_keys = _stage1_key_set(stage1)
    insertions = []
    for index, row in stage2.iterrows():
        key = (row["paper_id"], row["issue_url"])
        if key in stage1_keys:
            continue
        insertions.append(
            Stage1Insertion(
                stage2_index=index,
                paper_id=row["paper_id"],
                issue_url=row["issue_url"],
                record_id=row["record_id"],
            )
        )
    return Stage1LineageRepairPlan(insertions=tuple(insertions))


def _stage1_lineage_json(stage2_row: pd.Series) -> str:
    return json.dumps(
        {
            "stage": "stage1",
            "lineage_repair": {
                "reason": "present_in_stage2_but_missing_from_stage1",
                "source_stage": "stage2",
                "source_record_id": stage2_row["record_id"],
                "source_original_label_json": stage2_row["original_label_json"],
            },
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def apply_stage1_lineage_repair(
    stage1: pd.DataFrame, stage2: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    plan = build_stage1_lineage_repair_plan(stage1, stage2)
    inserted_rows = []
    audit_rows = []

    for insertion in plan.insertions:
        source = stage2.loc[insertion.stage2_index].copy()
        for column in FINAL_LABEL_COLUMNS:
            source[column] = ""
        source["original_label_json"] = _stage1_lineage_json(stage2.loc[insertion.stage2_index])
        inserted_rows.append(source)
        audit_rows.append(
            {
                "paper_id": insertion.paper_id,
                "issue_url": insertion.issue_url,
                "record_id": insertion.record_id,
                "source_stage2_index": str(insertion.stage2_index),
                "notes": "Stage 2 source fields copied to Stage 1; downstream labels cleared.",
            }
        )

    repaired_stage1 = stage1.copy(deep=True)
    if inserted_rows:
        repaired_stage1 = pd.concat(
            [repaired_stage1, pd.DataFrame(inserted_rows, columns=stage1.columns)],
            ignore_index=True,
        )
    audit = pd.DataFrame(audit_rows, columns=AUDIT_COLUMNS)
    return repaired_stage1, audit


def validate_stage1_lineage_repair_result(
    original_stage1: pd.DataFrame,
    stage2: pd.DataFrame,
    repaired_stage1: pd.DataFrame,
) -> None:
    if len(repaired_stage1) < len(original_stage1):
        raise ValueError("Stage 1 rows were removed")
    pd.testing.assert_frame_equal(
        repaired_stage1.iloc[: len(original_stage1)].reset_index(drop=True),
        original_stage1.reset_index(drop=True),
        obj="Original Stage 1 rows",
    )
    inserted = repaired_stage1.iloc[len(original_stage1) :]
    if not inserted.empty and (inserted[FINAL_LABEL_COLUMNS] != "").any().any():
        raise ValueError("Inserted Stage 1 rows contain downstream labels")

    stage1_keys = _stage1_key_set(repaired_stage1)
    for row in stage2.itertuples():
        if (row.paper_id, row.issue_url) not in stage1_keys:
            raise ValueError(f"Stage 2 row is not aligned to Stage 1: {row.issue_url!r}")


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
            "Generated on 2026-06-22 after Stage 1 \u2192 Stage 2 and Stage 2 \u2192 Stage 3 lineage repairs.",
            "",
            "## Dataset And Grain Summary",
            "",
            "| Stage | Rows | Columns | Unique `record_id` | Duplicate `record_id` excess | Unique `issue_url` | Duplicate `issue_url` excess |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            *metric_rows,
            "",
            "## Stage 1 \u2192 Stage 2 Lineage",
            "",
            "- Every Stage 2 row now has a matching Stage 1 row with the same `paper_id` and `issue_url`.",
            "- 1,597 previously missing Stage 1 rows were restored from Stage 2 source fields.",
            "- Downstream label fields were cleared from the 1,597 inserted Stage 1 rows.",
            "- Repair details are recorded in `reports/stage1_stage2_lineage_repairs.csv`.",
            "",
            "## Stage 2 \u2192 Stage 3 Lineage",
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
            "- Stage 1 count-only placeholders remain outside this repair and should be excluded from text-model training and evaluation.",
            "",
        ]
    )


def _replace_stage1_count_text(text: str, stage1_count: int) -> str:
    return text.replace("Stage 1 Raw | `Dataset/stage1.csv` | Raw candidate records before human filtering | 33,822", f"Stage 1 Raw | `Dataset/stage1.csv` | Raw candidate records before human filtering | {stage1_count:,}")


def repair_repository(root: Path, write: bool = True) -> dict[str, int]:
    root = root.resolve()
    stage1 = _read_csv(root / "Dataset" / "stage1.csv")
    stage2 = _read_csv(root / "Dataset" / "stage2.csv")
    stage3 = _read_csv(root / "Dataset" / "stage3.csv")
    plan = build_stage1_lineage_repair_plan(stage1, stage2)

    if plan.inserted_rows == 0:
        return {"inserted_rows": 0, "inserted_keys": 0}
    if (plan.inserted_rows, plan.inserted_keys) != (
        EXPECTED_INSERTED_ROWS,
        EXPECTED_INSERTED_KEYS,
    ):
        raise ValueError(
            f"Unexpected repair scope: rows={plan.inserted_rows}, "
            f"keys={plan.inserted_keys}"
        )

    repaired_stage1, audit = apply_stage1_lineage_repair(stage1, stage2)
    validate_stage1_lineage_repair_result(stage1, stage2, repaired_stage1)
    post_plan = build_stage1_lineage_repair_plan(repaired_stage1, stage2)
    if post_plan.inserted_rows:
        raise ValueError("Stage 1 \u2192 Stage 2 lineage remains incomplete")

    if not write:
        return {"inserted_rows": plan.inserted_rows, "inserted_keys": plan.inserted_keys}

    _write_csv(repaired_stage1, root / "Dataset" / "stage1.csv")
    _write_csv(audit, root / "reports" / "stage1_stage2_lineage_repairs.csv")

    affected_papers = sorted(set(audit["paper_id"]))
    for paper_id in affected_papers:
        paper_dir = root / "Dataset" / "by_paper" / paper_id
        _write_csv(
            repaired_stage1.loc[repaired_stage1["paper_id"] == paper_id],
            paper_dir / "stage1.csv",
        )

    metadata, summary = _update_metadata_tables(root, repaired_stage1, stage2, stage3)
    _write_csv(metadata, root / "metadata" / "dataset_metadata.csv")
    _write_csv(summary, root / "metadata" / "paper_dataset_summary.csv")

    stages = {"stage1": repaired_stage1, "stage2": stage2, "stage3": stage3}
    metrics = _quality_metrics(stages)
    duplicates = _duplicate_key_rows(stages)
    _write_csv(metrics, root / "reports" / "data_quality_metrics.csv")
    _write_csv(duplicates, root / "reports" / "duplicate_key_rows.csv")

    root_readme_path = root / "README.md"
    root_readme_text = root_readme_path.read_text(encoding="utf-8")
    root_readme_path.write_text(
        _replace_stage1_count_text(root_readme_text, len(repaired_stage1)).replace(
            "stage1 (33822, 23) 7", f"stage1 ({len(repaired_stage1)}, 23) 7"
        ),
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
        _render_paper_overview(metadata, summary, stages), encoding="utf-8", newline="\n"
    )
    (root / "reports" / "SHA256SUMS.txt").write_text(
        _render_hash_manifest(root), encoding="utf-8", newline="\n"
    )

    return {"inserted_rows": plan.inserted_rows, "inserted_keys": plan.inserted_keys}


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = repair_repository(args.root, write=not args.check)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    main()
