"""Read-only per-paper information-completeness profile."""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path


csv.field_size_limit(2_147_483_647)
ROOT = Path(__file__).resolve().parents[2]
BY_PAPER = ROOT / "Dataset" / "by_paper"
SENTINELS = {
    "no_comments_in_source": re.compile(
        r"\bno_comments_in_source\b", re.IGNORECASE
    ),
    "comments_unavailable_in_source": re.compile(
        r"\bcomments_unavailable_in_source\b", re.IGNORECASE
    ),
    "other_unavailable_marker": re.compile(
        r"(?:comments?|discussion).{0,30}(?:unavailable|not available)",
        re.IGNORECASE,
    ),
}


for paper_dir in sorted(path for path in BY_PAPER.iterdir() if path.is_dir()):
    print(f"PAPER {paper_dir.name}")
    for stage in ("stage1", "stage2", "stage3"):
        path = paper_dir / f"{stage}.csv"
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

        missing = {
            field: sum(not (row.get(field) or "").strip() for row in rows)
            for field in ("issue_url", "title", "body", "comments")
        }
        numeric_comments = sum(
            bool(re.fullmatch(r"\d+", (row.get("comments") or "").strip()))
            for row in rows
        )
        thin_without_comments = sum(
            not (row.get("comments") or "").strip()
            and len((row.get("body") or "").strip()) < 80
            for row in rows
        )
        sentinels = Counter()
        for row in rows:
            comments = row.get("comments") or ""
            for name, pattern in SENTINELS.items():
                if pattern.search(comments):
                    sentinels[name] += 1
                    break

        print(
            f"  {stage}: rows={len(rows)} "
            f"missing={missing} numeric_comments={numeric_comments} "
            f"thin_without_comments={thin_without_comments} "
            f"sentinels={dict(sentinels)}"
        )
