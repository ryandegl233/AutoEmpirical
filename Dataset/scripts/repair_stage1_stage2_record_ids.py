from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from Dataset.scripts.repair_stage2_stage3_lineage import (
    _duplicate_key_rows,
    _quality_metrics,
    _render_hash_manifest,
    _write_csv,
)


TARGET_PAPER = "issta2024_bugs_in_pods_understanding_bugs"
KEY_COLUMNS = ["paper_id", "issue_url"]
AUDIT_COLUMNS = [
    "paper_id",
    "issue_url",
    "old_stage2_record_id",
    "new_record_id",
    "stage2_rows_changed",
    "stage3_rows_changed",
]


@dataclass(frozen=True)
class RecordIdChange:
    issue_url: str
    old_record_id: str
    new_record_id: str
    stage2_indices: tuple[int, ...]


@dataclass(frozen=True)
class RecordIdRepairPlan:
    changes: tuple[RecordIdChange, ...]

    @property
    def change_count(self) -> int:
        return sum(len(change.stage2_indices) for change in self.changes)

    @property
    def unique_url_count(self) -> int:
        return len(self.changes)


def build_record_id_repair_plan(
    stage1: pd.DataFrame, stage2: pd.DataFrame, paper_id: str
) -> RecordIdRepairPlan:
    stage1_paper = stage1.loc[stage1["paper_id"] == paper_id]
    stage2_paper = stage2.loc[stage2["paper_id"] == paper_id]

    stage1_ids = stage1_paper.groupby("issue_url")["record_id"].apply(set)
    ambiguous = stage1_ids.loc[stage1_ids.map(len) > 1]
    if not ambiguous.empty:
        raise ValueError(
            "ambiguous Stage 1 issue_url values: "
            + ", ".join(sorted(ambiguous.index.astype(str)))
        )

    canonical_ids = {
        issue_url: next(iter(record_ids))
        for issue_url, record_ids in stage1_ids.items()
    }
    changes = []
    for issue_url, group in stage2_paper.groupby("issue_url", sort=False):
        new_record_id = canonical_ids.get(issue_url)
        if new_record_id is None:
            continue
        old_ids = set(group["record_id"])
        if old_ids == {new_record_id}:
            continue
        if len(old_ids) != 1:
            raise ValueError(
                f"ambiguous Stage 2 issue_url {issue_url!r}: {sorted(old_ids)!r}"
            )
        changes.append(
            RecordIdChange(
                issue_url=issue_url,
                old_record_id=next(iter(old_ids)),
                new_record_id=new_record_id,
                stage2_indices=tuple(group.index),
            )
        )
    return RecordIdRepairPlan(changes=tuple(changes))


def apply_record_id_repair(
    stage1: pd.DataFrame,
    stage2: pd.DataFrame,
    stage3: pd.DataFrame,
    paper_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    plan = build_record_id_repair_plan(stage1, stage2, paper_id)
    repaired_stage2 = stage2.copy(deep=True)
    repaired_stage3 = stage3.copy(deep=True)
    audit_rows = []

    for change in plan.changes:
        repaired_stage2.loc[list(change.stage2_indices), "record_id"] = change.new_record_id
        stage3_mask = (
            (repaired_stage3["paper_id"] == paper_id)
            & (repaired_stage3["issue_url"] == change.issue_url)
        )
        stage3_rows_changed = int(stage3_mask.sum())
        repaired_stage3.loc[stage3_mask, "record_id"] = change.new_record_id
        audit_rows.append(
            {
                "paper_id": paper_id,
                "issue_url": change.issue_url,
                "old_stage2_record_id": change.old_record_id,
                "new_record_id": change.new_record_id,
                "stage2_rows_changed": len(change.stage2_indices),
                "stage3_rows_changed": stage3_rows_changed,
            }
        )

    return (
        repaired_stage2,
        repaired_stage3,
        pd.DataFrame(audit_rows, columns=AUDIT_COLUMNS),
    )


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _validate_result(
    original_stage2: pd.DataFrame,
    original_stage3: pd.DataFrame,
    repaired_stage2: pd.DataFrame,
    repaired_stage3: pd.DataFrame,
) -> None:
    if len(repaired_stage2) != len(original_stage2):
        raise ValueError("Stage 2 row count changed")
    if len(repaired_stage3) != len(original_stage3):
        raise ValueError("Stage 3 row count changed")
    stage2_non_id = [column for column in stage2_columns(original_stage2) if column != "record_id"]
    stage3_non_id = [column for column in stage2_columns(original_stage3) if column != "record_id"]
    pd.testing.assert_frame_equal(
        repaired_stage2[stage2_non_id], original_stage2[stage2_non_id]
    )
    pd.testing.assert_frame_equal(
        repaired_stage3[stage3_non_id], original_stage3[stage3_non_id]
    )
    stage2_ids = repaired_stage2.groupby(KEY_COLUMNS)["record_id"].apply(set).to_dict()
    for row in repaired_stage3.itertuples():
        if row.record_id not in stage2_ids[(row.paper_id, row.issue_url)]:
            raise ValueError(
                f"Stage 2 → Stage 3 lineage broken for {(row.paper_id, row.issue_url)!r}"
            )


def stage2_columns(frame: pd.DataFrame) -> list[str]:
    return list(frame.columns)


def repair_repository(root: Path, write: bool = True) -> dict[str, int]:
    root = root.resolve()
    stage1 = _read_csv(root / "Dataset" / "stage1.csv")
    stage2 = _read_csv(root / "Dataset" / "stage2.csv")
    stage3 = _read_csv(root / "Dataset" / "stage3.csv")
    plan = build_record_id_repair_plan(stage1, stage2, TARGET_PAPER)
    if plan.change_count == 0:
        return {"changed_rows": 0, "changed_urls": 0}
    if (plan.change_count, plan.unique_url_count) != (425, 423):
        raise ValueError(
            f"Unexpected repair scope: rows={plan.change_count}, "
            f"urls={plan.unique_url_count}"
        )

    repaired_stage2, repaired_stage3, audit = apply_record_id_repair(
        stage1, stage2, stage3, TARGET_PAPER
    )
    _validate_result(stage2, stage3, repaired_stage2, repaired_stage3)
    post_plan = build_record_id_repair_plan(stage1, repaired_stage2, TARGET_PAPER)
    if post_plan.change_count:
        raise ValueError("Stage 1 → Stage 2 record IDs remain inconsistent")

    if not write:
        return {
            "changed_rows": plan.change_count,
            "changed_urls": plan.unique_url_count,
        }

    _write_csv(repaired_stage2, root / "Dataset" / "stage2.csv")
    _write_csv(repaired_stage3, root / "Dataset" / "stage3.csv")
    paper_dir = root / "Dataset" / "by_paper" / TARGET_PAPER
    _write_csv(
        repaired_stage2.loc[repaired_stage2["paper_id"] == TARGET_PAPER],
        paper_dir / "stage2.csv",
    )
    _write_csv(
        repaired_stage3.loc[repaired_stage3["paper_id"] == TARGET_PAPER],
        paper_dir / "stage3.csv",
    )
    _write_csv(audit, root / "reports" / "stage1_stage2_record_id_repairs.csv")

    stages = {"stage1": stage1, "stage2": repaired_stage2, "stage3": repaired_stage3}
    _write_csv(_quality_metrics(stages), root / "reports" / "data_quality_metrics.csv")
    _write_csv(
        _duplicate_key_rows(stages), root / "reports" / "duplicate_key_rows.csv"
    )

    health_path = root / "reports" / "dataset_health_report.md"
    health = health_path.read_text(encoding="utf-8")
    note = (
        "- ISSTA 2024 now uses the Stage 1 `record_id` for 425 Stage 2 rows "
        "covering 423 URLs; matching Stage 3 rows were updated at the same time."
    )
    if note not in health:
        marker = "## Remaining Known Issues"
        health = health.replace(marker, f"{note}\n\n{marker}")
        health_path.write_text(health, encoding="utf-8", newline="\n")

    (root / "reports" / "SHA256SUMS.txt").write_text(
        _render_hash_manifest(root), encoding="utf-8", newline="\n"
    )
    return {
        "changed_rows": plan.change_count,
        "changed_urls": plan.unique_url_count,
    }


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
