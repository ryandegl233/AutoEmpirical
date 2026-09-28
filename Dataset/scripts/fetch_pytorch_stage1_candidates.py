from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import date, timedelta
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Iterable
from urllib.parse import urlencode
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PAPER_ID = "icse2023_an_empirical_study_on_bugs"
REPO = "pytorch/pytorch"
DEFAULT_CUTOFF = date(2022, 10, 20)
SEARCH_URL = "https://api.github.com/search/issues"
OUTPUT_COLUMNS = [
    "record_id",
    "paper_id",
    "source_project",
    "issue_url",
    "number",
    "title",
    "body",
    "created_at",
    "updated_at",
    "closed_at",
    "state",
    "labels",
    "comments_count",
    "reactions_total_count",
    "github_api_url",
    "pull_request_url",
    "original_label_json",
]


@dataclass(frozen=True)
class DateWindow:
    start: date
    end: date

    def as_qualifier(self) -> str:
        return f"created:{self.start.isoformat()}..{self.end.isoformat()}"


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _github_token() -> str:
    for name in ("GITHUB_TOKEN", "GH_TOKEN", "GITHUB_PAT"):
        value = os.environ.get(name)
        if value:
            return value
    return ""


def _request_json(url: str, token: str, retries: int = 4) -> dict:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "AutoEmpirical-pytorch-stage1-fetch",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        request = Request(url, headers=headers)
        try:
            with urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            last_error = error
            if error.code not in {403, 429, 500, 502, 503, 504} or attempt == retries:
                raise
            retry_after = error.headers.get("Retry-After")
            delay = int(retry_after) if retry_after and retry_after.isdigit() else 2 ** attempt
            time.sleep(delay)
        except URLError as error:
            last_error = error
            if attempt == retries:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError(f"GitHub request failed: {last_error}") from last_error


def _search(query: str, token: str, page: int = 1, per_page: int = 100) -> dict:
    url = SEARCH_URL + "?" + urlencode(
        {
            "q": query,
            "sort": "created",
            "order": "asc",
            "per_page": per_page,
            "page": page,
        }
    )
    return _request_json(url, token)


def _base_query(cutoff: date) -> str:
    return (
        f"repo:{REPO} is:issue is:closed label:triaged linked:pr "
        f"closed:<={cutoff.isoformat()}"
    )


def _count_for_window(window: DateWindow, cutoff: date, token: str) -> int:
    query = f"{_base_query(cutoff)} {window.as_qualifier()}"
    data = _search(query, token, page=1, per_page=1)
    return int(data["total_count"])


def _split_window(window: DateWindow) -> tuple[DateWindow, DateWindow]:
    midpoint = window.start + (window.end - window.start) // 2
    return (
        DateWindow(window.start, midpoint),
        DateWindow(midpoint + timedelta(days=1), window.end),
    )


def _safe_windows(
    start: date, end: date, cutoff: date, token: str, max_results: int = 950
) -> list[tuple[DateWindow, int]]:
    pending = [DateWindow(start, end)]
    safe: list[tuple[DateWindow, int]] = []
    while pending:
        window = pending.pop()
        count = _count_for_window(window, cutoff, token)
        if count <= max_results:
            safe.append((window, count))
            continue
        if window.start == window.end:
            raise ValueError(f"Single-day search window still exceeds API limit: {window}")
        left, right = _split_window(window)
        pending.extend([right, left])
        time.sleep(0.2)
    return safe


def _iter_window_items(window: DateWindow, cutoff: date, token: str) -> Iterable[dict]:
    query = f"{_base_query(cutoff)} {window.as_qualifier()}"
    first = _search(query, token, page=1, per_page=100)
    total = int(first["total_count"])
    pages = (total + 99) // 100
    for item in first["items"]:
        yield item
    for page in range(2, pages + 1):
        time.sleep(0.2)
        data = _search(query, token, page=page, per_page=100)
        for item in data["items"]:
            yield item


def _stable_record_id(issue_url: str) -> str:
    digest = hashlib.sha256(issue_url.encode("utf-8")).hexdigest()[:16]
    return f"{PAPER_ID}:{digest}"


def _row_from_issue(issue: dict) -> dict[str, str]:
    issue_url = issue["html_url"]
    original = {
        "stage": "stage1",
        "source": "github_search_api",
        "repo": REPO,
        "paper_rule": "closed issue, triaged label, linked pull request, created in search window, closed by 2022-10-20",
        "github_id": issue["id"],
        "node_id": issue["node_id"],
        "number": issue["number"],
    }
    return {
        "record_id": _stable_record_id(issue_url),
        "paper_id": PAPER_ID,
        "source_project": "pytorch",
        "issue_url": issue_url,
        "number": str(issue["number"]),
        "title": issue.get("title") or "",
        "body": issue.get("body") or "",
        "created_at": issue.get("created_at") or "",
        "updated_at": issue.get("updated_at") or "",
        "closed_at": issue.get("closed_at") or "",
        "state": issue.get("state") or "",
        "labels": json.dumps(
            [label.get("name", "") for label in issue.get("labels", [])],
            ensure_ascii=False,
        ),
        "comments_count": str(issue.get("comments", "")),
        "reactions_total_count": str(
            (issue.get("reactions") or {}).get("total_count", "")
        ),
        "github_api_url": issue.get("url") or "",
        "pull_request_url": (issue.get("pull_request") or {}).get("url", ""),
        "original_label_json": json.dumps(original, ensure_ascii=False, sort_keys=True),
    }


def fetch_candidates(root: Path, cutoff: date, write: bool = True) -> dict[str, int]:
    root = root.resolve()
    _load_env(root / ".env")
    token = _github_token()
    if not token:
        raise RuntimeError(
            "GITHUB_TOKEN, GH_TOKEN, or GITHUB_PAT was not found in .env/environment"
        )

    output_dir = root / "reports" / "pytorch_stage1_reconstruction"
    if write:
        output_dir.mkdir(parents=True, exist_ok=True)

    windows = _safe_windows(date(2016, 1, 1), cutoff, cutoff, token)
    issues_by_url: dict[str, dict] = {}
    for window, _ in windows:
        for issue in _iter_window_items(window, cutoff, token):
            if "pull_request" in issue:
                continue
            issues_by_url[issue["html_url"]] = issue

    issues = sorted(issues_by_url.values(), key=lambda item: item["number"])
    rows = [_row_from_issue(issue) for issue in issues]

    if write:
        raw_path = output_dir / "pytorch_stage1_candidates.raw.jsonl"
        csv_path = output_dir / "pytorch_stage1_candidates.csv"
        manifest_path = output_dir / "fetch_manifest.json"
        with raw_path.open("w", encoding="utf-8", newline="\n") as handle:
            for issue in issues:
                handle.write(json.dumps(issue, ensure_ascii=False, sort_keys=True) + "\n")
        with csv_path.open("w", encoding="utf-8", newline="\n") as handle:
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        manifest = {
            "paper_id": PAPER_ID,
            "repo": REPO,
            "cutoff": cutoff.isoformat(),
            "base_query": _base_query(cutoff),
            "window_count": len(windows),
            "window_counts": [
                {
                    "start": window.start.isoformat(),
                    "end": window.end.isoformat(),
                    "reported_total_count": count,
                }
                for window, count in windows
            ],
            "unique_issues_written": len(issues),
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    return {"windows": len(windows), "issues": len(issues)}


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--cutoff", default=DEFAULT_CUTOFF.isoformat())
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    cutoff = date.fromisoformat(args.cutoff)
    result = fetch_candidates(args.root, cutoff, write=not args.check)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    main()
