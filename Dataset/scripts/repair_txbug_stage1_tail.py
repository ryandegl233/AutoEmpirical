from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Dataset.scripts.repair_stage2_stage3_lineage import _read_csv, _write_csv


PAPER_ID = "icse2024_understanding_transaction_bugs_in_database"


def _stage1_tail_marker(row: pd.Series) -> bool:
    return "stage1 placeholder" in (row.get("title") or "")


def _stage1_lineage_json(stage2_row: pd.Series) -> str:
    return json.dumps(
        {
            "stage": "stage1",
            "lineage_repair": {
                "reason": "stage2_final_sample_restored_into_stage1_tail",
                "source_stage": "stage2",
                "source_record_id": stage2_row["record_id"],
                "source_issue_url": stage2_row["issue_url"],
            },
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def repair_txbug_tail(
    root: Path,
    write: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = root.resolve()
    stage1 = _read_csv(root / "Dataset" / "stage1.csv")
    stage2 = _read_csv(root / "Dataset" / "stage2.csv")

    paper_stage1 = stage1.loc[stage1["paper_id"] == PAPER_ID].reset_index(drop=True)
    paper_stage2 = stage2.loc[
        (stage2["paper_id"] == PAPER_ID)
        & (
            stage2["issue_url"].str.contains("postgresql.org/message-id", regex=False)
            | stage2["issue_url"].str.contains("postgr.es/m/", regex=False)
            | stage2["issue_url"].str.contains("sqlite.org/src/tktview", regex=False)
        )
    ].reset_index(drop=True)

    placeholder_idx = [
        idx for idx, row in paper_stage1.iterrows() if _stage1_tail_marker(row)
    ]
    if len(placeholder_idx) < len(paper_stage2):
        raise ValueError(
            f"Not enough placeholder rows to restore tail: placeholders={len(placeholder_idx)}, "
            f"needed={len(paper_stage2)}"
        )

    repaired = paper_stage1.copy(deep=True)
    audit_rows = []
    for stage1_idx, (_, source) in zip(placeholder_idx[: len(paper_stage2)], paper_stage2.iterrows()):
        repaired.loc[stage1_idx] = source
        repaired.at[stage1_idx, "original_label_json"] = _stage1_lineage_json(source)
        audit_rows.append(
            {
                "paper_id": PAPER_ID,
                "issue_url": source["issue_url"],
                "record_id": source["record_id"],
                "source_stage2_index": str(source.name),
                "notes": "Restored Stage 2 final sample into Stage 1 tail placeholder.",
            }
        )

    audit = pd.DataFrame(
        audit_rows,
        columns=["paper_id", "issue_url", "record_id", "source_stage2_index", "notes"],
    )

    if write:
        stage1_after = pd.concat(
            [stage1.loc[stage1["paper_id"] != PAPER_ID], repaired],
            ignore_index=True,
        )
        _write_csv(stage1_after, root / "Dataset" / "stage1.csv")
        _write_csv(repaired, root / "Dataset" / "by_paper" / PAPER_ID / "stage1.csv")
        _write_csv(audit, root / "reports" / "txbug_stage1_tail_repairs.csv")

    return repaired, audit


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    repaired, audit = repair_txbug_tail(args.root, write=not args.check)
    print(json.dumps({"repaired_rows": len(audit), "paper_rows": len(repaired)}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    main()
