from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


PAPER_ID = "icse2024_understanding_transaction_bugs_in_database"
JIRA_BASE = "https://jira.mariadb.org"
SEARCH_URL = f"{JIRA_BASE}/rest/api/2/search"
COMMENT_URL_TEMPLATE = f"{JIRA_BASE}/rest/api/2/issue/{{key}}/comment"
DEFAULT_START = "2018-01-01"
DEFAULT_END = "2022-12-31"
PROJECTS = {
    "MDEV": "mariadb",
    "MCOL": "mariadb_columnstore",
}
KEYWORDS = [
    "transaction",
    "transactions",
    "rollback",
    "roll back",
    "isolation level",
    "serializable",
    "read committed",
    "repeatable read",
    "XA",
    "deadlock",
    "commit transaction",
    "abort transaction",
]
FIELDS = ",".join(
    [
        "key",
        "summary",
        "description",
        "created",
        "updated",
        "status",
        "resolution",
        "issuetype",
        "project",
        "components",
        "labels",
        "priority",
        "versions",
        "fixVersions",
        "comment",
    ]
)
REQUEST_DELAY_SECONDS = 0.25
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
    "comments",
    "created_at",
    "updated_at",
    "closed_at",
    "state",
    "labels",
    "comments_count",
    "reactions_total_count",
    "jira_key",
    "jira_project",
    "issue_type",
    "resolution",
    "priority",
    "components",
    "versions",
    "fix_versions",
    "jira_api_url",
    "original_label_json",
]


def _request_json(url: str, retries: int = 4) -> dict:
    headers = {
        "Accept": "application/json",
        "User-Agent": "AutoEmpirical-txbug-mariadb-stage1-fetch",
    }
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urlopen(Request(url, headers=headers), timeout=90) as response:
                payload = response.read().decode("utf-8")
            time.sleep(REQUEST_DELAY_SECONDS)
            return json.loads(payload)
        except HTTPError as error:
            last_error = error
            if error.code not in {429, 500, 502, 503, 504} or attempt == retries:
                raise
            retry_after = error.headers.get("Retry-After")
            delay = int(retry_after) if retry_after and retry_after.isdigit() else 2**attempt
            time.sleep(max(delay, 1))
        except URLError as error:
            last_error = error
            if attempt == retries:
                raise
            time.sleep(2**attempt)
    raise RuntimeError(f"MariaDB JIRA request failed: {last_error}") from last_error


def _jql(keyword: str, start: str, end: str) -> str:
    escaped = keyword.replace("\\", "\\\\").replace('"', '\\"')
    return (
        f'project in ({", ".join(PROJECTS)}) '
        f'AND created >= "{start}" '
        f'AND created <= "{end}" '
        f'AND text ~ "{escaped}"'
    )


def _search_page(jql: str, start_at: int, max_results: int) -> dict:
    url = SEARCH_URL + "?" + urlencode(
        {
            "jql": jql,
            "startAt": start_at,
            "maxResults": max_results,
            "fields": FIELDS,
        }
    )
    return _request_json(url)


def _fetch_comments(key: str, known_comments: dict | None) -> list[dict]:
    if not known_comments:
        known_comments = {}
    comments = list(known_comments.get("comments") or [])
    total = int(known_comments.get("total") or len(comments))
    if len(comments) >= total:
        return comments

    fetched: list[dict] = []
    start_at = 0
    max_results = 100
    while start_at < total:
        url = COMMENT_URL_TEMPLATE.format(key=key) + "?" + urlencode(
            {"startAt": start_at, "maxResults": max_results}
        )
        data = _request_json(url)
        batch = data.get("comments") or []
        fetched.extend(batch)
        total = int(data.get("total") or total)
        if not batch:
            break
        start_at += len(batch)
    return fetched or comments


def _stable_record_id(issue_url: str) -> str:
    digest = hashlib.sha256(issue_url.encode("utf-8")).hexdigest()[:16]
    return f"{PAPER_ID}:{digest}"


def _names(values: list[dict] | None) -> list[str]:
    return [str(item.get("name") or "") for item in values or [] if item.get("name")]


def _comment_text(comments: list[dict]) -> str:
    blocks: list[str] = []
    for comment in comments:
        author = ((comment.get("author") or {}).get("displayName") or "").strip()
        created = (comment.get("created") or "").strip()
        body = (comment.get("body") or "").strip()
        prefix_parts = [part for part in [created, author] if part]
        prefix = " | ".join(prefix_parts)
        blocks.append(f"[{prefix}]\n{body}" if prefix else body)
    return "\n\n---\n\n".join(block for block in blocks if block)


def _row_from_issue(issue: dict, keywords: set[str], comments: list[dict]) -> dict[str, str]:
    fields = issue.get("fields") or {}
    key = issue.get("key") or ""
    issue_url = f"{JIRA_BASE}/browse/{key}"
    project_key = ((fields.get("project") or {}).get("key") or key.split("-", 1)[0]).upper()
    labels = list(fields.get("labels") or [])
    components = _names(fields.get("components"))
    versions = _names(fields.get("versions"))
    fix_versions = _names(fields.get("fixVersions"))
    original = {
        "stage": "stage1",
        "source": "mariadb_jira_rest_api",
        "paper_rule": (
            "TXBug MariaDB candidate reconstruction using 2018-01-01..2022-12-31 "
            "issue creation window and transaction-related JIRA text keyword search."
        ),
        "matched_keywords": sorted(keywords),
        "jira_id": issue.get("id") or "",
        "jira_key": key,
        "jira_project": project_key,
    }
    return {
        "record_id": _stable_record_id(issue_url),
        "paper_id": PAPER_ID,
        "source_project": PROJECTS.get(project_key, project_key.lower()),
        "source_repository": "jira.mariadb.org",
        "matched_keywords": json.dumps(sorted(keywords), ensure_ascii=False),
        "issue_url": issue_url,
        "number": key,
        "title": fields.get("summary") or "",
        "body": fields.get("description") or "",
        "comments": _comment_text(comments),
        "created_at": fields.get("created") or "",
        "updated_at": fields.get("updated") or "",
        "closed_at": "",
        "state": ((fields.get("status") or {}).get("name") or ""),
        "labels": json.dumps(labels, ensure_ascii=False),
        "comments_count": str(len(comments)),
        "reactions_total_count": "",
        "jira_key": key,
        "jira_project": project_key,
        "issue_type": ((fields.get("issuetype") or {}).get("name") or ""),
        "resolution": ((fields.get("resolution") or {}).get("name") or ""),
        "priority": ((fields.get("priority") or {}).get("name") or ""),
        "components": json.dumps(components, ensure_ascii=False),
        "versions": json.dumps(versions, ensure_ascii=False),
        "fix_versions": json.dumps(fix_versions, ensure_ascii=False),
        "jira_api_url": issue.get("self") or "",
        "original_label_json": json.dumps(original, ensure_ascii=False, sort_keys=True),
    }


def _final_mariadb_urls(root: Path) -> set[str]:
    stage2 = root / "Dataset" / "stage2.csv"
    if not stage2.exists():
        return set()
    with stage2.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return {
            row["issue_url"].strip()
            for row in reader
            if row.get("paper_id") == PAPER_ID
            and "jira.mariadb.org" in row.get("issue_url", "")
        }


def _write_coverage_audit(output_dir: Path, candidate_urls: set[str], final_urls: set[str]) -> None:
    audit_path = output_dir / "mariadb_final_coverage_audit.csv"
    with audit_path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["issue_url", "covered_by_reconstruction"],
            lineterminator="\n",
        )
        writer.writeheader()
        for issue_url in sorted(final_urls):
            writer.writerow(
                {
                    "issue_url": issue_url,
                    "covered_by_reconstruction": str(issue_url in candidate_urls),
                }
            )


def _write_readme(
    output_dir: Path,
    rows: list[dict[str, str]],
    manifest: dict,
    final_urls: set[str],
) -> None:
    by_project = {
        source_project: sum(row["source_project"] == source_project for row in rows)
        for source_project in sorted({row["source_project"] for row in rows})
    }
    candidate_urls = {row["issue_url"] for row in rows}
    missing = sorted(final_urls - candidate_urls)
    covered = len(final_urls) - len(missing)
    lines = [
        "# TXBug MariaDB JIRA Stage 1 reconstruction",
        "",
        "This directory contains a reconstructed Stage 1 candidate set for the MariaDB JIRA portion of TXBug.",
        "",
        "## Method",
        "",
        f"- Source: `{JIRA_BASE}` REST API.",
        f"- Projects: `{', '.join(PROJECTS)}`.",
        f"- Created window: `{manifest['date_window']['start']}` to `{manifest['date_window']['end']}`.",
        f"- JIRA text keywords: {', '.join('`' + keyword + '`' for keyword in manifest['keywords'])}.",
        "- Candidate rows are deduplicated by JIRA issue URL; `matched_keywords` records all keywords that matched each issue.",
        "- `body` is the JIRA description. `comments` concatenates public JIRA comments with timestamp/author markers.",
        "",
        "## Output",
        "",
        f"- `txbug_mariadb_candidates.csv`: {len(rows)} unique candidate rows.",
        "- `txbug_mariadb_candidates.raw.jsonl`: raw REST payloads plus matched keywords.",
        "- `fetch_manifest.json`: query totals and fetch parameters.",
        "- `mariadb_final_coverage_audit.csv`: coverage of the local final TXBug MariaDB rows.",
        "",
        "## Counts",
        "",
    ]
    for source_project, count in by_project.items():
        lines.append(f"- {source_project}: {count}")
    lines.extend(
        [
            "",
            "## Coverage against local final TXBug MariaDB rows",
            "",
            f"- Covered: {covered}/{len(final_urls)}",
            f"- Missing: {len(missing)}",
        ]
    )
    for issue_url in missing:
        lines.append(f"  - {issue_url}")
    output_dir.joinpath("README.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def fetch_candidates(
    root: Path,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    write: bool = True,
) -> dict[str, int]:
    root = root.resolve()
    output_dir = root / "reports" / "txbug_stage1_reconstruction" / "mariadb"
    if write:
        output_dir.mkdir(parents=True, exist_ok=True)

    issues: dict[str, dict] = {}
    issue_keywords: dict[str, set[str]] = {}
    query_manifest: list[dict] = []
    max_results = 100
    for keyword in KEYWORDS:
        jql = _jql(keyword, start, end)
        start_at = 0
        total = None
        keyword_keys: set[str] = set()
        while total is None or start_at < total:
            page = _search_page(jql, start_at=start_at, max_results=max_results)
            total = int(page.get("total") or 0)
            batch = page.get("issues") or []
            for issue in batch:
                key = issue.get("key") or ""
                if not key:
                    continue
                issues[key] = issue
                issue_keywords.setdefault(key, set()).add(keyword)
                keyword_keys.add(key)
            if not batch:
                break
            start_at += len(batch)
        query_manifest.append(
            {
                "keyword": keyword,
                "jql": jql,
                "reported_total": total or 0,
                "unique_keys_seen": len(keyword_keys),
            }
        )

    comments_by_key: dict[str, list[dict]] = {}
    for key, issue in sorted(issues.items()):
        known = ((issue.get("fields") or {}).get("comment") or {})
        comments_by_key[key] = _fetch_comments(key, known)

    rows = [
        _row_from_issue(issue, issue_keywords[key], comments_by_key.get(key, []))
        for key, issue in issues.items()
    ]
    rows.sort(key=lambda row: (row["source_project"], row["jira_key"]))

    result = {
        "unique_issues": len(rows),
        "mariadb": sum(row["source_project"] == "mariadb" for row in rows),
        "mariadb_columnstore": sum(
            row["source_project"] == "mariadb_columnstore" for row in rows
        ),
        "with_body": sum(bool(row["body"].strip()) for row in rows),
        "with_comments": sum(bool(row["comments"].strip()) for row in rows),
    }

    if write:
        raw_path = output_dir / "txbug_mariadb_candidates.raw.jsonl"
        csv_path = output_dir / "txbug_mariadb_candidates.csv"
        manifest_path = output_dir / "fetch_manifest.json"
        with raw_path.open("w", encoding="utf-8", newline="\n") as handle:
            for key, issue in sorted(issues.items()):
                payload = {
                    "matched_keywords": sorted(issue_keywords[key]),
                    "comments": comments_by_key.get(key, []),
                    "issue": issue,
                }
                handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        with csv_path.open("w", encoding="utf-8", newline="\n") as handle:
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        manifest = {
            "paper_id": PAPER_ID,
            "source": JIRA_BASE,
            "date_window": {"start": start, "end": end, "field": "created"},
            "projects": PROJECTS,
            "keywords": KEYWORDS,
            "queries": query_manifest,
            **result,
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        final_urls = _final_mariadb_urls(root)
        _write_coverage_audit(output_dir, {row["issue_url"] for row in rows}, final_urls)
        _write_readme(output_dir, rows, manifest, final_urls)

    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = fetch_candidates(args.root, start=args.start, end=args.end, write=not args.check)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    main()
