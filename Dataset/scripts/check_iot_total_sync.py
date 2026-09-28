"""Read-only consistency check for IoT per-paper and unified CSVs."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


csv.field_size_limit(2_147_483_647)
PAPER_ID = "icse2021_iot_bugs_and_development_challenges"
DATASET = Path(__file__).resolve().parents[2] / "Dataset"


for stage in ("stage1", "stage2", "stage3"):
    per_path = DATASET / "by_paper" / PAPER_ID / f"{stage}.csv"
    total_path = DATASET / f"{stage}.csv"

    with per_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        per_fields = reader.fieldnames
        per_rows = {row["record_id"]: row for row in reader}

    total_rows = {}
    with total_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        total_fields = reader.fieldnames
        for row in reader:
            if row["paper_id"] == PAPER_ID:
                total_rows[row["record_id"]] = row

    missing = set(per_rows) - set(total_rows)
    extra = set(total_rows) - set(per_rows)
    field_mismatches: Counter[str] = Counter()
    row_mismatches = 0
    for record_id in set(per_rows) & set(total_rows):
        changed = [
            field
            for field in per_fields or []
            if per_rows[record_id][field] != total_rows[record_id][field]
        ]
        if changed:
            row_mismatches += 1
            field_mismatches.update(changed)

    print(
        f"{stage}: per_rows={len(per_rows)} "
        f"total_rows={len(total_rows)} "
        f"schema_match={per_fields == total_fields} "
        f"missing={len(missing)} extra={len(extra)} "
        f"row_mismatches={row_mismatches} "
        f"field_mismatches={dict(field_mismatches)}"
    )
