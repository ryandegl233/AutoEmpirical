from __future__ import annotations

import argparse
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


PAPER_ID = "icse2023_an_empirical_study_on_bugs"
RECONSTRUCTION_RELATIVE_PATH = (
    "reports/pytorch_stage1_reconstruction/convergence/"
    "closed_cutoff_2022_03_09_exact_2205.csv"
)
AUDIT_COLUMNS = [
    "integration_type",
    "paper_id",
    "issue_url",
    "record_id",
    "notes",
]


def _stage1_lineage_json(stage2_row: pd.Series) -> str:
    return json.dumps(
        {
            "stage": "stage1",
            "lineage_repair": {
                "reason": "stage2_record_not_found_in_pytorch_reconstruction",
                "source_stage": "stage2",
                "source_record_id": stage2_row["record_id"],
                "reconstruction_rule": "GitHub Search API convergence set closed_at <= 2022-03-09",
            },
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _candidate_original_label_json(row: pd.Series) -> str:
    source_json = json.loads(row["original_label_json"])
    source_json.update(
        {
            "stage": "stage1",
            "integrated_into_dataset": True,
            "integration_note": (
                "PyTorch Stage 1 candidate reconstructed from GitHub Search API "
                "using convergence cutoff closed_at <= 2022-03-09."
            ),
        }
    )
    return json.dumps(source_json, ensure_ascii=False, sort_keys=True)


def _candidate_to_stage1_row(
    candidate: pd.Series,
    columns: list[str],
    canonical_record_ids: dict[str, str],
) -> dict[str, str]:
    issue_url = candidate["issue_url"]
    values = {column: "" for column in columns}
    values.update(
        {
            "record_id": canonical_record_ids.get(issue_url, candidate["record_id"]),
            "paper_id": PAPER_ID,
            "source_project": "pytorch",
            "issue_url": issue_url,
            "title": candidate["title"],
            "body": candidate["body"],
            "comments": "not_fetched",
            "created_at": candidate["created_at"],
            "updated_at": candidate["updated_at"],
            "state": candidate["state"],
            "original_label_json": _candidate_original_label_json(candidate),
            "source_file": RECONSTRUCTION_RELATIVE_PATH,
            "source_sheet": "github_search_api_closed_cutoff_2022_03_09",
            "source_row_index": candidate["number"],
        }
    )
    return values


def build_integrated_pytorch_stage1(
    original_paper_stage1: pd.DataFrame,
    stage2_paper: pd.DataFrame,
    reconstruction: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    canonical_record_ids = {
        row.issue_url: row.record_id for row in stage2_paper.itertuples()
    }
    rows = [
        _candidate_to_stage1_row(candidate, list(original_paper_stage1.columns), canonical_record_ids)
        for _, candidate in reconstruction.iterrows()
    ]
    audit_rows = [
        {
            "integration_type": "github_reconstruction",
            "paper_id": PAPER_ID,
            "issue_url": row["issue_url"],
            "record_id": row["record_id"],
            "notes": "PyTorch Stage 1 row reconstructed from GitHub Search API convergence set.",
        }
        for row in rows
    ]

    reconstructed_urls = set(reconstruction["issue_url"])
    missing_stage2 = stage2_paper.loc[~stage2_paper["issue_url"].isin(reconstructed_urls)]
    for _, source in missing_stage2.iterrows():
        row = source.copy()
        for column in FINAL_LABEL_COLUMNS:
            row[column] = ""
        row["original_label_json"] = _stage1_lineage_json(source)
        row["source_file"] = "Dataset/stage2.csv"
        row["source_sheet"] = "stage2_lineage_repair"
        rows.append(row.to_dict())
        audit_rows.append(
            {
                "integration_type": "stage2_lineage_repair",
                "paper_id": PAPER_ID,
                "issue_url": source["issue_url"],
                "record_id": source["record_id"],
                "notes": (
                    "Stage 2 row retained in Stage 1 because it is not present "
                    "in the PyTorch reconstruction; downstream labels cleared."
                ),
            }
        )

    integrated = pd.DataFrame(rows, columns=original_paper_stage1.columns)
    audit = pd.DataFrame(audit_rows, columns=AUDIT_COLUMNS)
    return integrated, audit


def validate_integration(
    stage1_before: pd.DataFrame,
    stage1_after: pd.DataFrame,
    stage2: pd.DataFrame,
    integrated_paper_stage1: pd.DataFrame,
    reconstruction: pd.DataFrame,
) -> None:
    other_before = stage1_before.loc[stage1_before["paper_id"] != PAPER_ID].reset_index(
        drop=True
    )
    other_after = stage1_after.loc[stage1_after["paper_id"] != PAPER_ID].reset_index(
        drop=True
    )
    pd.testing.assert_frame_equal(other_after, other_before, obj="Non-PyTorch Stage 1 rows")

    if len(reconstruction) != 2205:
        raise ValueError(f"Unexpected reconstruction count: {len(reconstruction)}")
    if len(integrated_paper_stage1) != 2207:
        raise ValueError(
            f"Expected 2,207 PyTorch Stage 1 rows after preserving lineage, "
            f"found {len(integrated_paper_stage1)}"
        )
    lineage_rows = integrated_paper_stage1["original_label_json"].str.contains(
        "stage2_record_not_found_in_pytorch_reconstruction", regex=False
    )
    if int(lineage_rows.sum()) != 2:
        raise ValueError("Expected exactly two PyTorch Stage 2 lineage repair rows")
    if (
        integrated_paper_stage1.loc[lineage_rows, FINAL_LABEL_COLUMNS] != ""
    ).any().any():
        raise ValueError("PyTorch lineage repair rows contain downstream labels")

    stage1_keys = set(zip(stage1_after["paper_id"], stage1_after["issue_url"]))
    missing = [
        row.issue_url
        for row in stage2.loc[stage2["paper_id"] == PAPER_ID].itertuples()
        if (row.paper_id, row.issue_url) not in stage1_keys
    ]
    if missing:
        raise ValueError(f"Stage 2 rows missing from integrated Stage 1: {missing}")


def _replace_root_readme_counts(text: str, stage1_count: int, stage2_count: int) -> str:
    text = re.sub(
        r"(\| Stage 1 Raw \| `Dataset/stage1\.csv` \| Raw candidate records before human filtering \| )[\d,]+( \|)",
        lambda match: f"{match.group(1)}{stage1_count:,}{match.group(2)}",
        text,
    )
    text = re.sub(
        r"(\| Stage 2 Filtered \| `Dataset/stage2\.csv` \| Human-filtered bug-relevant records \| )[\d,]+( \|)",
        lambda match: f"{match.group(1)}{stage2_count:,}{match.group(2)}",
        text,
    )
    text = re.sub(r"stage1 \(\d+, 23\) 7", f"stage1 ({stage1_count}, 23) 7", text)
    text = re.sub(r"stage2 \(\d+, 23\) 7", f"stage2 ({stage2_count}, 23) 7", text)
    return text


def _append_health_note(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    note = (
        "- PyTorch Stage 1 placeholders were replaced with 2,205 GitHub "
        "reconstruction rows using the convergence cutoff `closed_at <= 2022-03-09`; "
        "2 additional Stage 2 lineage rows were retained because current GitHub "
        "metadata no longer returns them under that rule."
    )
    if note not in text:
        text = text.replace("## Remaining Known Issues", f"{note}\n\n## Remaining Known Issues")
        path.write_text(text, encoding="utf-8", newline="\n")


def repair_repository(root: Path, write: bool = True) -> dict[str, int]:
    root = root.resolve()
    stage1 = _read_csv(root / "Dataset" / "stage1.csv")
    stage2 = _read_csv(root / "Dataset" / "stage2.csv")
    stage3 = _read_csv(root / "Dataset" / "stage3.csv")
    reconstruction = _read_csv(root / RECONSTRUCTION_RELATIVE_PATH)

    original_paper_stage1 = stage1.loc[stage1["paper_id"] == PAPER_ID].reset_index(
        drop=True
    )
    stage2_paper = stage2.loc[stage2["paper_id"] == PAPER_ID].reset_index(drop=True)
    integrated_paper_stage1, audit = build_integrated_pytorch_stage1(
        original_paper_stage1, stage2_paper, reconstruction
    )
    stage1_after = pd.concat(
        [
            stage1.loc[stage1["paper_id"] != PAPER_ID],
            integrated_paper_stage1,
        ],
        ignore_index=True,
    )
    validate_integration(
        stage1, stage1_after, stage2, integrated_paper_stage1, reconstruction
    )

    if not write:
        return {
            "reconstruction_rows": len(reconstruction),
            "pytorch_stage1_rows": len(integrated_paper_stage1),
            "lineage_repair_rows": int(
                (audit["integration_type"] == "stage2_lineage_repair").sum()
            ),
        }

    _write_csv(stage1_after, root / "Dataset" / "stage1.csv")
    _write_csv(
        integrated_paper_stage1,
        root / "Dataset" / "by_paper" / PAPER_ID / "stage1.csv",
    )
    _write_csv(
        audit, root / "reports" / "pytorch_stage1_reconstruction" / "integration_audit.csv"
    )

    metadata, summary = _update_metadata_tables(root, stage1_after, stage2, stage3)
    _write_csv(metadata, root / "metadata" / "dataset_metadata.csv")
    _write_csv(summary, root / "metadata" / "paper_dataset_summary.csv")

    stages = {"stage1": stage1_after, "stage2": stage2, "stage3": stage3}
    metrics = _quality_metrics(stages)
    duplicates = _duplicate_key_rows(stages)
    _write_csv(metrics, root / "reports" / "data_quality_metrics.csv")
    _write_csv(duplicates, root / "reports" / "duplicate_key_rows.csv")

    root_readme_path = root / "README.md"
    root_readme_path.write_text(
        _replace_root_readme_counts(root_readme_path.read_text(encoding="utf-8"), len(stage1_after), len(stage2)),
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
    health_path = root / "reports" / "dataset_health_report.md"
    health_path.write_text(
        _render_health_report(stages, metrics), encoding="utf-8", newline="\n"
    )
    _append_health_note(health_path)
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
        "reconstruction_rows": len(reconstruction),
        "pytorch_stage1_rows": len(integrated_paper_stage1),
        "lineage_repair_rows": int(
            (audit["integration_type"] == "stage2_lineage_repair").sum()
        ),
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
