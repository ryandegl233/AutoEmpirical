"""Read-only ISSTA commit-diff coverage check in unified stage tables."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


csv.field_size_limit(2_147_483_647)
ROOT = Path(__file__).resolve().parents[2]
PAPER_ID = "issta2024_bugs_in_pods_understanding_bugs"


for stage in ("stage1", "stage2", "stage3"):
    rows = 0
    commits = 0
    nonempty_diff = 0
    unavailable_diff = 0
    unavailable_values: Counter[str] = Counter()
    path = ROOT / "Dataset" / f"{stage}.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["paper_id"] != PAPER_ID:
                continue
            rows += 1
            commits += "/commit/" in row["issue_url"]
            diff = (row.get("code_diff") or "").strip()
            nonempty_diff += bool(diff)
            is_unavailable = len(diff) < 500 and (
                "unavailable" in diff.lower()
                or "not_available" in diff.lower()
            )
            unavailable_diff += is_unavailable
            if is_unavailable:
                safe_value = diff.encode(
                    "ascii", errors="backslashreplace"
                ).decode("ascii")
                unavailable_values[safe_value] += 1
    print(
        f"{stage}: rows={rows} commit_urls={commits} "
        f"code_diff_nonempty={nonempty_diff} "
        f"code_diff_unavailable_markers={unavailable_diff} "
        f"marker_counts={dict(unavailable_values)}"
    )
