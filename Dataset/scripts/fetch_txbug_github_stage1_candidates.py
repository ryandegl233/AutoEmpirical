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
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


PAPER_ID = "icse2024_understanding_transaction_bugs_in_database"
REPOSITORIES = {
    "tidb": "pingcap/tidb",
    "cockroachdb": "cockroachdb/cockroach",
}
DEFAULT_START = date(2018, 1, 1)
DEFAULT_END = date(2022, 12, 31)
SEARCH_URL = "https://api.github.com/search/issues"
SEARCH_REQUEST_DELAY_SECONDS = 2.2
KEYWORDS_BROAD = [
    "transaction",
    "transactions",
    "abort",
    "aborted",
    "commit",
    "committed",
    "rollback",
    '"roll back"',
    '"isolation level"',
    '"read committed"',
    "serializable",
]
KEYWORDS_NARROW = [
    "transaction",
    "transactions",
    "rollback",
    '"roll back"',
    '"isolation level"',
    "serializable",
    '"read committed"',
    '"repeatable read"',
    '"snapshot isolation"',
    '"write skew"',
    '"dirty read"',
    '"phantom read"',
    '"commit transaction"',
    '"transaction commit"',
    '"abort transaction"',
    '"transaction abort"',
    "deadlock",
    "txn",
    "XA",
]
KEYWORDS_CORE = [
    "transaction",
    "transactions",
    "rollback",
    '"roll back"',
    '"isolation level"',
    "serializable",
    '"commit transaction"',
    '"transaction commit"',
    '"abort transaction"',
    '"transaction abort"',
    "XA",
]
KEYWORD_MODES = {
    "broad": KEYWORDS_BROAD,
    "narrow": KEYWORDS_NARROW,
    "core": KEYWORDS_CORE,
}
OUTPUT_COLUMNS = [
    "record_id",
    "paper_id",
    "source_project",
    "source_repository",
    "matched_keywords",
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
    "original_label_json",
]


@dataclass(frozen=True)
class DateWindow:
    start: date
    end: date

    def qualifier(self) -> str:
        return f"created:{self.start.isoformat()}..{self.end.isoformat()}"


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


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
        "User-Agent": "AutoEmpirical-txbug-github-stage1-fetch",
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
            if retry_after and retry_after.isdigit():
                delay = int(retry_after)
            elif error.code in {403, 429}:
                delay = 65
            else:
                delay = 2 ** attempt
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
    data = _request_json(url, token)
    time.sleep(SEARCH_REQUEST_DELAY_SECONDS)
    return data


def _base_query(repo: str, keyword: str) -> str:
    return f"repo:{repo} is:issue {keyword}"


def _count_window(repo: str, keyword: str, window: DateWindow, token: str) -> int:
    query = f"{_base_query(repo, keyword)} {window.qualifier()}"
    data = _search(query, token, page=1, per_page=1)
    return int(data["total_count"])


def _split_window(window: DateWindow) -> tuple[DateWindow, DateWindow]:
    midpoint = window.start + (window.end - window.start) // 2
    return (
        DateWindow(window.start, midpoint),
        DateWindow(midpoint + timedelta(days=1), window.end),
    )


def _safe_windows(
    repo: str,
    keyword: str,
    start: date,
    end: date,
    token: str,
    max_results: int = 950,
) -> list[tuple[DateWindow, int]]:
    pending = [DateWindow(start, end)]
    safe: list[tuple[DateWindow, int]] = []
    while pending:
        window = pending.pop()
        count = _count_window(repo, keyword, window, token)
        if count <= max_results:
            safe.append((window, count))
        else:
            if window.start == window.end:
                raise ValueError(
                    f"Single-day search window exceeds API limit: repo={repo}, "
                    f"keyword={keyword}, window={window}"
                )
            left, right = _split_window(window)
            pending.extend([right, left])
    return safe


def _iter_window_items(
    repo: str, keyword: str, window: DateWindow, count: int, token: str
) -> Iterable[dict]:
    query = f"{_base_query(repo, keyword)} {window.qualifier()}"
    pages = (count + 99) // 100
    for page in range(1, pages + 1):
        data = _search(query, token, page=page, per_page=100)
        for item in data["items"]:
            yield item


def _stable_record_id(issue_url: str) -> str:
    digest = hashlib.sha256(issue_url.encode("utf-8")).hexdigest()[:16]
    return f"{PAPER_ID}:{digest}"


def _row_from_issue(issue: dict, source_project: str, repo: str, keywords: set[str]) -> dict[str, str]:
    issue_url = issue["html_url"]
    original = {
        "stage": "stage1",
        "source": "github_search_api",
        "repo": repo,
        "paper_rule": (
            "TXBug GitHub candidate reconstruction using 2018-01-01..2022-12-31 "
            "issue creation window and transaction-related keyword search."
        ),
        "matched_keywords": sorted(keywords),
        "github_id": issue["id"],
        "node_id": issue["node_id"],
        "number": issue["number"],
    }
    return {
        "record_id": _stable_record_id(issue_url),
        "paper_id": PAPER_ID,
        "source_project": source_project,
        "source_repository": repo,
        "matched_keywords": json.dumps(sorted(keywords), ensure_ascii=False),
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
        "original_label_json": json.dumps(original, ensure_ascii=False, sort_keys=True),
    }


def fetch_candidates(root: Path, keyword_mode: str = "core", write: bool = True) -> dict[str, int]:
    root = root.resolve()
    _load_env(root / ".env")
    token = _github_token()
    if not token:
        raise RuntimeError(
            "GITHUB_TOKEN, GH_TOKEN, or GITHUB_PAT was not found in .env/environment"
        )

    keywords = KEYWORD_MODES[keyword_mode]
    output_dir = root / "reports" / "txbug_stage1_reconstruction" / "github" / keyword_mode
    if write:
        output_dir.mkdir(parents=True, exist_ok=True)

    issues: dict[str, dict] = {}
    issue_keywords: dict[str, set[str]] = {}
    query_manifest = []
    for source_project, repo in REPOSITORIES.items():
        for keyword in keywords:
            windows = _safe_windows(repo, keyword, DEFAULT_START, DEFAULT_END, token)
            query_manifest.append(
                {
                    "source_project": source_project,
                    "repo": repo,
                    "keyword": keyword,
                    "windows": [
                        {
                            "start": window.start.isoformat(),
                            "end": window.end.isoformat(),
                            "reported_total_count": count,
                        }
                        for window, count in windows
                    ],
                    "reported_total_count": sum(count for _, count in windows),
                }
            )
            for window, count in windows:
                for issue in _iter_window_items(repo, keyword, window, count, token):
                    if "pull_request" in issue:
                        continue
                    issue_url = issue["html_url"]
                    issues[issue_url] = issue
                    issue_keywords.setdefault(issue_url, set()).add(keyword)

    rows = [
        _row_from_issue(
            issue,
            "tidb" if "pingcap/tidb" in issue["repository_url"] else "cockroachdb",
            issue["repository_url"].removeprefix("https://api.github.com/repos/"),
            issue_keywords[issue["html_url"]],
        )
        for issue in issues.values()
    ]
    rows.sort(key=lambda row: (row["source_project"], int(row["number"])))

    if write:
        raw_path = output_dir / "txbug_github_candidates.raw.jsonl"
        csv_path = output_dir / "txbug_github_candidates.csv"
        manifest_path = output_dir / "fetch_manifest.json"
        with raw_path.open("w", encoding="utf-8", newline="\n") as handle:
            for issue in sorted(
                issues.values(),
                key=lambda item: (
                    item["repository_url"],
                    int(item["number"]),
                ),
            ):
                payload = {
                    "matched_keywords": sorted(issue_keywords[issue["html_url"]]),
                    "issue": issue,
                }
                handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        with csv_path.open("w", encoding="utf-8", newline="\n") as handle:
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        manifest = {
            "paper_id": PAPER_ID,
            "date_window": {
                "start": DEFAULT_START.isoformat(),
                "end": DEFAULT_END.isoformat(),
                "field": "created",
            },
            "repositories": REPOSITORIES,
            "keyword_mode": keyword_mode,
            "keywords": keywords,
            "queries": query_manifest,
            "unique_issues_written": len(rows),
            "unique_by_source_project": {
                source_project: sum(row["source_project"] == source_project for row in rows)
                for source_project in REPOSITORIES
            },
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    return {
        "unique_issues": len(rows),
        "tidb": sum(row["source_project"] == "tidb" for row in rows),
        "cockroachdb": sum(row["source_project"] == "cockroachdb" for row in rows),
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument(
        "--keyword-mode",
        choices=sorted(KEYWORD_MODES),
        default="core",
        help="Keyword set to use for TXBug GitHub reconstruction",
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = fetch_candidates(args.root, keyword_mode=args.keyword_mode, write=not args.check)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    main()
