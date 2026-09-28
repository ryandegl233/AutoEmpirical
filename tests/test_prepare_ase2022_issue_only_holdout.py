from __future__ import annotations

import csv
from pathlib import Path

from Benchmark.scripts.prepare_ase2022_issue_only_holdout import run_cli
from Benchmark.src.ase2022_camel_mas_baseline import ASE2022_PAPER_ID


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _row(record_id: str, issue_number: int) -> dict[str, str]:
    return {
        "record_id": record_id,
        "paper_id": ASE2022_PAPER_ID,
        "source_project": "org/repo",
        "issue_url": f"https://github.com/org/repo/issues/{issue_number}",
        "title": f"title {record_id}",
        "body": f"body {record_id}",
        "comments": "[]",
        "state": "closed",
        "created_at": "2021-01-01T00:00:00Z",
    }


def test_cli_prepares_reproducible_issue_only_holdout(tmp_path: Path) -> None:
    stage1 = tmp_path / "stage1.csv"
    stage2 = tmp_path / "stage2.csv"
    stage3 = tmp_path / "stage3.csv"
    excluded = tmp_path / "excluded.csv"
    output = tmp_path / "output"
    positive = _row("positive", 1)
    negative = _row("negative", 2)
    _write(stage1, [positive, negative])
    _write(stage2, [positive])
    _write(
        stage3,
        [
            {
                **positive,
                "symptom": "Crash",
                "root_cause": "Incorrect Code Logic",
            }
        ],
    )
    _write(excluded, [_row("old", 3)])

    summary = run_cli(
        [
            "--stage1-path",
            str(stage1),
            "--stage2-path",
            str(stage2),
            "--stage3-path",
            str(stage3),
            "--exclude-cohort-path",
            str(excluded),
            "--output-dir",
            str(output),
            "--positives",
            "1",
            "--negatives",
            "1",
            "--seed",
            "20260806",
        ]
    )

    assert Path(summary["cohort"]).exists()
    assert Path(summary["manifest"]).exists()
