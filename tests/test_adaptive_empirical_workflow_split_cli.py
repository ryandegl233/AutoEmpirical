from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path("Benchmark/scripts/prepare_ase2022_uncontaminated_stage3_splits.py")


def _write_source(path: Path) -> None:
    fieldnames = [
        "record_id",
        "paper_id",
        "source_project",
        "issue_url",
        "title",
        "body",
        "comments",
        "created_at",
        "updated_at",
        "state",
        "symptom",
        "root_cause",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(8):
            writer.writerow(
                {
                    "record_id": f"ase2022:cli:{index}",
                    "paper_id": "ase2022",
                    "source_project": "tensorflow/tfjs",
                    "issue_url": f"https://github.com/tensorflow/tfjs/issues/{index}",
                    "title": f"unique report {index}",
                    "body": f"unique reproduction details {index}",
                    "comments": "maintainer reply",
                    "created_at": "2021-01-01",
                    "updated_at": "2021-01-02",
                    "state": "closed",
                    "symptom": "Crash",
                    "root_cause": "API Misuse",
                }
            )


def test_preparation_cli_writes_frozen_artifacts_without_printing_gold(
    tmp_path: Path,
) -> None:
    source = tmp_path / "stage3.csv"
    taxonomy = tmp_path / "taxonomy.json"
    output = tmp_path / "frozen"
    restricted_gold = tmp_path / "private" / "gold.csv"
    _write_source(source)
    taxonomy.write_text(
        json.dumps({"symptom": ["Crash"], "root_cause": ["API Misuse"]}),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--source-csv",
            str(source),
            "--taxonomy-path",
            str(taxonomy),
            "--label-taxonomy-path",
            str(taxonomy),
            "--output-dir",
            str(output),
            "--restricted-gold-output",
            str(restricted_gold),
            "--validation-size",
            "3",
            "--final-size",
            "3",
            "--seed",
            "7",
            "--split-revision",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary == {
        "final_count": 3,
        "manifest": str((output / "split_manifest.json").resolve()),
        "restricted_gold_created": True,
        "validation_count": 3,
    }
    assert "symptom" not in completed.stdout
    assert "root_cause" not in completed.stdout
    assert restricted_gold.is_file()
    assert not (output / ".restricted_stage3_gold.csv").exists()
