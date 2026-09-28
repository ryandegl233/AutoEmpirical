from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import re
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
from Dataset.scripts.repair_stage1_stage2_lineage import _render_health_report


PAPER_ID = "icse2024_understanding_transaction_bugs_in_database"
TARGET_STAGE1_ROWS = 7775
RECONSTRUCTION_FILES = [
    ("github", Path("reports/txbug_stage1_reconstruction/github/core_plus_tidb_label/txbug_github_candidates.csv")),
    ("mysql", Path("reports/txbug_stage1_reconstruction/mysql/txbug_mysql_candidates_with_text.csv")),
    ("mariadb", Path("reports/txbug_stage1_reconstruction/mariadb/txbug_mariadb_candidates.csv")),
    ("postgresql", Path("reports/txbug_stage1_reconstruction/postgresql/txbug_postgresql_candidates.csv")),
]
OUTPUT_AUDIT = Path("reports/txbug_stage1_reconstruction/integration_audit.csv")


def _parse_keywords(value: str) -> int:
    if value in {"", "nan", "None"}:
        return 0
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return len(parsed)
    except Exception:
        pass
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return len(parsed)
    except Exception:
        pass
    return max(1, value.count(",") + 1)


def _stage1_lineage_json(source_row: pd.Series, reason: str) -> str:
    payload = {
        "stage": "stage1",
        "lineage_repair": {
            "reason": reason,
            "source_stage": "stage2",
            "source_record_id": source_row["record_id"],
            "source_issue_url": source_row["issue_url"],
        },
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _candidate_original_label_json(row: pd.Series) -> str:
    try:
        source_json = json.loads(row["original_label_json"])
    except Exception:
        source_json = {}
    source_json.update(
        {
            "stage": "stage1",
            "integrated_into_dataset": True,
            "integration_note": "TXBug Stage 1 reconstructed from source-specific candidate sets.",
        }
    )
    return json.dumps(source_json, ensure_ascii=False, sort_keys=True)


def _normalize_candidate_frame(source: str, frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    df["source_reconstruction"] = source
    df["kw_count"] = df["matched_keywords"].fillna("").map(_parse_keywords)
    priority = {"github": 0, "mysql": 1, "mariadb": 2, "postgresql": 3}
    df["source_rank"] = df["source_reconstruction"].map(priority)
    return df


def _build_reconstruction(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for source, rel in RECONSTRUCTION_FILES:
        frame = _read_csv(root / rel)
        frames.append(_normalize_candidate_frame(source, frame))
    candidates = pd.concat(frames, ignore_index=True)
    candidates = candidates.drop_duplicates("issue_url", keep="first")
    candidates = candidates.sort_values(
        by=["source_rank", "kw_count", "issue_url"],
        ascending=[True, False, True],
    ).reset_index(drop=True)

    stage2 = _read_csv(root / "Dataset" / "stage2.csv")
    final = stage2.loc[stage2["paper_id"] == PAPER_ID].copy()
    final_urls = set(final["issue_url"])

    # Keep the paper's final 140 rows later via lineage supplement, not by source search.
    reconstruction = candidates.loc[~candidates["issue_url"].isin(final_urls)].copy()
    if len(reconstruction) > TARGET_STAGE1_ROWS - len(final):
        reconstruction = reconstruction.head(TARGET_STAGE1_ROWS - len(final)).copy()

    return reconstruction, final


def _candidate_to_stage1_row(candidate: pd.Series, columns: list[str]) -> dict[str, str]:
    values = {column: "" for column in columns}
    values.update(
        {
            "record_id": candidate["record_id"],
            "paper_id": PAPER_ID,
            "source_project": candidate["source_project"],
            "issue_url": candidate["issue_url"],
            "title": candidate["title"],
            "body": candidate["body"],
            "comments": candidate.get("comments", ""),
            "created_at": candidate.get("created_at", ""),
            "updated_at": candidate.get("updated_at", ""),
            "state": candidate.get("state", ""),
            "original_label_json": _candidate_original_label_json(candidate),
            "source_file": str(candidate.get("source_file", "")),
            "source_sheet": str(candidate.get("source_sheet", "")),
            "source_row_index": str(candidate.get("source_row_index", "")),
        }
    )
    values["source_reconstruction"] = candidate["source_reconstruction"]
    return values


def build_integrated_txbug_stage1(
    original_paper_stage1: pd.DataFrame,
    stage2_paper: pd.DataFrame,
    reconstruction: pd.DataFrame,
    final_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = list(original_paper_stage1.columns)
    rows = [
        _candidate_to_stage1_row(candidate, columns)
        for _, candidate in reconstruction.iterrows()
    ]
    audit_rows = [
        {
            "integration_type": "reconstruction_candidate",
            "paper_id": PAPER_ID,
            "issue_url": row["issue_url"],
            "record_id": row["record_id"],
            "notes": f"Reconstructed from {row['source_reconstruction']} candidate set.",
        }
        for row in rows
    ]

    final_urls = set(final_rows["issue_url"])
    reconstructed_urls = set(reconstruction["issue_url"])
    missing_final = final_rows.loc[~final_rows["issue_url"].isin(reconstructed_urls)].copy()
    for _, source in missing_final.iterrows():
        row = source.copy()
        for column in FINAL_LABEL_COLUMNS:
            row[column] = ""
        row["original_label_json"] = _stage1_lineage_json(
            source, "stage2_final_sample_restored_into_stage1"
        )
        row["source_file"] = "Dataset/stage2.csv"
        row["source_sheet"] = "stage2_lineage_repair"
        rows.append(row.to_dict())
        audit_rows.append(
            {
                "integration_type": "stage2_lineage_repair",
                "paper_id": PAPER_ID,
                "issue_url": source["issue_url"],
                "record_id": source["record_id"],
                "notes": "Stage 2 final sample retained in Stage 1 because it is not present in reconstruction.",
            }
        )

    integrated = pd.DataFrame(rows, columns=columns)
    audit = pd.DataFrame(
        audit_rows,
        columns=["integration_type", "paper_id", "issue_url", "record_id", "notes"],
    )
    if len(integrated) != TARGET_STAGE1_ROWS:
        raise ValueError(f"Expected {TARGET_STAGE1_ROWS} integrated rows, found {len(integrated)}")
    if integrated["issue_url"].duplicated().any():
        raise ValueError("Integrated Stage 1 contains duplicate issue_url rows")
    if not final_urls.issubset(set(integrated["issue_url"])):
        missing = sorted(final_urls - set(integrated["issue_url"]))
        raise ValueError(f"Missing final TXBug rows after integration: {missing}")
    return integrated, audit


def repair_repository(root: Path, write: bool = True) -> dict[str, int]:
    root = root.resolve()
    stage1 = _read_csv(root / "Dataset" / "stage1.csv")
    stage2 = _read_csv(root / "Dataset" / "stage2.csv")
    stage3 = _read_csv(root / "Dataset" / "stage3.csv")
    original_paper_stage1 = stage1.loc[stage1["paper_id"] == PAPER_ID].reset_index(drop=True)
    stage2_paper = stage2.loc[stage2["paper_id"] == PAPER_ID].reset_index(drop=True)
    reconstruction, final_rows = _build_reconstruction(root)
    integrated_paper_stage1, audit = build_integrated_txbug_stage1(
        original_paper_stage1, stage2_paper, reconstruction, final_rows
    )

    if not write:
        return {
            "reconstruction_rows": len(reconstruction),
            "txbug_stage1_rows": len(integrated_paper_stage1),
            "lineage_repair_rows": int((audit["integration_type"] == "stage2_lineage_repair").sum()),
        }

    stage1_after = pd.concat(
        [stage1.loc[stage1["paper_id"] != PAPER_ID], integrated_paper_stage1],
        ignore_index=True,
    )
    _write_csv(stage1_after, root / "Dataset" / "stage1.csv")
    _write_csv(integrated_paper_stage1, root / "Dataset" / "by_paper" / PAPER_ID / "stage1.csv")
    _write_csv(audit, root / OUTPUT_AUDIT)

    metadata, summary = _update_metadata_tables(root, stage1_after, stage2, stage3)
    _write_csv(metadata, root / "metadata" / "dataset_metadata.csv")
    _write_csv(summary, root / "metadata" / "paper_dataset_summary.csv")

    stages = {"stage1": stage1_after, "stage2": stage2, "stage3": stage3}
    metrics = _quality_metrics(stages)
    duplicates = _duplicate_key_rows(stages)
    _write_csv(metrics, root / "reports" / "data_quality_metrics.csv")
    _write_csv(duplicates, root / "reports" / "duplicate_key_rows.csv")
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
    return {
        "reconstruction_rows": len(reconstruction),
        "txbug_stage1_rows": len(integrated_paper_stage1),
        "lineage_repair_rows": int((audit["integration_type"] == "stage2_lineage_repair").sum()),
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = repair_repository(args.root, write=not args.check)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    main()
