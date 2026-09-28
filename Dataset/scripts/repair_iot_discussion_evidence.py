"""Repair ICSE 2021 IoT issue and pull-request discussion evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from threading import local
from typing import Any, Callable, Iterable

import requests


ROOT = Path(__file__).resolve().parents[2]
PAPER_ID = "icse2021_iot_bugs_and_development_challenges"
EVIDENCE_VERSION = "current_unversioned"
PER_PAPER_DIR = ROOT / "Dataset/by_paper" / PAPER_ID
PER_PAPER_PATHS = [
    PER_PAPER_DIR / f"stage{stage}.csv" for stage in (1, 2, 3)
]
UNIFIED_PATHS = [ROOT / "Dataset" / f"stage{stage}.csv" for stage in (1, 2, 3)]
CACHE_PATH = ROOT / "Dataset/cache/iot_discussion_evidence.jsonl"
EVIDENCE_PATH = (
    ROOT / "Dataset/evidence/icse2021_iot_discussion_evidence.jsonl"
)
REPORT_DIR = ROOT / "reports/iot_information_reconstruction"
AUDIT_PATH = REPORT_DIR / "audit.json"
csv.field_size_limit(min(sys.maxsize, 2_147_483_647))
_HTTP_LOCAL = local()


GITHUB_RECORD_URL = re.compile(
    r"^https://github\.com/([^/]+)/([^/]+)/(issues|pull)/(\d+)/?$"
)


COMMENT_FIELDS = """
id
body
createdAt
author { login }
"""

PAGE_INFO_FIELDS = """
pageInfo {
  hasNextPage
  endCursor
}
"""


def parse_github_record_url(url: str) -> tuple[str, str, int, str]:
    match = GITHUB_RECORD_URL.fullmatch(url.strip())
    if not match:
        raise ValueError(f"unsupported GitHub record URL: {url}")
    owner, repository, kind, number = match.groups()
    return (
        owner,
        repository,
        int(number),
        "issue" if kind == "issues" else "pull_request",
    )


def build_initial_discussion_query(
    owner: str,
    repository: str,
    records: list[tuple[int, str]],
) -> str:
    fields = []
    for number, _expected_kind in records:
        fields.append(
            f"""
    record_{int(number)}: issueOrPullRequest(number: {int(number)}) {{
      __typename
      ... on Issue {{
        id
        url
        comments(first: 100) {{
          totalCount
          nodes {{ {COMMENT_FIELDS} }}
          {PAGE_INFO_FIELDS}
        }}
      }}
      ... on PullRequest {{
        id
        url
        comments(first: 100) {{
          totalCount
          nodes {{ {COMMENT_FIELDS} }}
          {PAGE_INFO_FIELDS}
        }}
        reviews(first: 100) {{
          totalCount
          nodes {{
            id
            body
            submittedAt
            state
            author {{ login }}
          }}
          {PAGE_INFO_FIELDS}
        }}
        reviewThreads(first: 100) {{
          totalCount
          nodes {{
            id
            comments(first: 100) {{
              totalCount
              nodes {{ {COMMENT_FIELDS} }}
              {PAGE_INFO_FIELDS}
            }}
          }}
          {PAGE_INFO_FIELDS}
        }}
      }}
    }}
"""
        )
    return f"""
query {{
  repository(owner: {json.dumps(owner)}, name: {json.dumps(repository)}) {{
{''.join(fields)}
  }}
}}
"""


def build_connection_page_query(task: dict[str, str]) -> str:
    node_id = json.dumps(task["node_id"])
    cursor = json.dumps(task["cursor"])
    connection = task["connection"]
    if connection == "comments":
        page = f"""
        page: comments(first: 100, after: {cursor}) {{
          totalCount
          nodes {{ {COMMENT_FIELDS} }}
          {PAGE_INFO_FIELDS}
        }}
"""
        fragments = "\n".join(
            f"... on {kind} {{ {page} }}"
            for kind in ("Issue", "PullRequest", "PullRequestReviewThread")
        )
    elif connection == "reviews":
        fragments = f"""
        ... on PullRequest {{
          page: reviews(first: 100, after: {cursor}) {{
            totalCount
            nodes {{
              id
              body
              submittedAt
              state
              author {{ login }}
            }}
            {PAGE_INFO_FIELDS}
          }}
        }}
"""
    elif connection == "reviewThreads":
        fragments = f"""
        ... on PullRequest {{
          page: reviewThreads(first: 100, after: {cursor}) {{
            totalCount
            nodes {{
              id
              comments(first: 100) {{
                totalCount
                nodes {{ {COMMENT_FIELDS} }}
                {PAGE_INFO_FIELDS}
              }}
            }}
            {PAGE_INFO_FIELDS}
          }}
        }}
"""
    else:
        raise ValueError(f"unsupported pagination connection: {connection}")
    return f"""
query {{
  node(id: {node_id}) {{
    {fragments}
  }}
}}
"""


def _next_page_task(
    *,
    node_id: str,
    connection: str,
    value: dict[str, Any] | None,
) -> dict[str, str] | None:
    page_info = (value or {}).get("pageInfo") or {}
    if not page_info.get("hasNextPage"):
        return None
    cursor = page_info.get("endCursor")
    if not cursor:
        raise ValueError(
            f"missing cursor for {node_id} {connection} pagination"
        )
    return {
        "node_id": node_id,
        "connection": connection,
        "cursor": cursor,
    }


def find_pagination_tasks(payload: dict[str, Any]) -> list[dict[str, str]]:
    node_id = payload.get("id")
    if not node_id:
        raise ValueError("discussion payload is missing node id")
    tasks = []
    for connection in ("comments", "reviews", "reviewThreads"):
        if connection == "reviews" and payload.get("__typename") != "PullRequest":
            continue
        if (
            connection == "reviewThreads"
            and payload.get("__typename") != "PullRequest"
        ):
            continue
        task = _next_page_task(
            node_id=node_id,
            connection=connection,
            value=payload.get(connection),
        )
        if task:
            tasks.append(task)
    for thread in (payload.get("reviewThreads") or {}).get("nodes") or []:
        task = _next_page_task(
            node_id=thread.get("id") or "",
            connection="comments",
            value=thread.get("comments"),
        )
        if task:
            tasks.append(task)
    return tasks


def _find_node(value: Any, node_id: str) -> dict[str, Any] | None:
    if isinstance(value, dict):
        if value.get("id") == node_id:
            return value
        for child in value.values():
            found = _find_node(child, node_id)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_node(child, node_id)
            if found is not None:
                return found
    return None


def merge_connection_page(
    payload: dict[str, Any],
    task: dict[str, str],
    page: dict[str, Any],
) -> None:
    target = _find_node(payload, task["node_id"])
    if target is None:
        raise ValueError(f"pagination node not found: {task['node_id']}")
    connection = target.get(task["connection"])
    if not isinstance(connection, dict):
        raise ValueError(
            f"pagination connection not found: {task['connection']}"
        )
    connection.setdefault("nodes", []).extend(page.get("nodes") or [])
    connection["pageInfo"] = page.get("pageInfo") or {
        "hasNextPage": False,
        "endCursor": None,
    }
    if "totalCount" in page:
        connection["totalCount"] = page["totalCount"]


def _discussion_item(
    node: dict[str, Any],
    *,
    timestamp_field: str,
    kind: str,
) -> dict[str, str] | None:
    body = (node.get("body") or "").strip()
    if not body:
        return None
    author = (node.get("author") or {}).get("login") or "unknown_author"
    timestamp = node.get(timestamp_field) or "timestamp_unavailable"
    return {
        "id": str(node.get("id") or f"{kind}:{timestamp}:{author}:{body}"),
        "timestamp": timestamp,
        "author": author,
        "kind": kind,
        "body": body,
    }


def compose_discussion(payload: dict[str, Any]) -> tuple[str, int]:
    items: dict[str, dict[str, str]] = {}

    for node in (payload.get("comments") or {}).get("nodes") or []:
        item = _discussion_item(
            node,
            timestamp_field="createdAt",
            kind="comment",
        )
        if item:
            items.setdefault(item["id"], item)

    if payload.get("__typename") == "PullRequest":
        for node in (payload.get("reviews") or {}).get("nodes") or []:
            state = node.get("state") or "UNKNOWN"
            item = _discussion_item(
                node,
                timestamp_field="submittedAt",
                kind=f"review:{state}",
            )
            if item:
                items.setdefault(item["id"], item)
        for thread in (payload.get("reviewThreads") or {}).get("nodes") or []:
            for node in (thread.get("comments") or {}).get("nodes") or []:
                item = _discussion_item(
                    node,
                    timestamp_field="createdAt",
                    kind="review_comment",
                )
                if item:
                    items.setdefault(item["id"], item)

    ordered = sorted(
        items.values(),
        key=lambda item: (
            item["timestamp"],
            item["kind"],
            item["id"],
        ),
    )
    comments = "\n\n".join(
        f"[{item['timestamp']} | {item['author']} | {item['kind']}]\n"
        f"{item['body']}"
        for item in ordered
    )
    return comments, len(ordered)


def _rest_comment_node(node: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(node.get("node_id") or node.get("id") or ""),
        "createdAt": node.get("created_at"),
        "author": {"login": (node.get("user") or {}).get("login")},
        "body": node.get("body") or "",
    }


def build_payload_from_rest(
    metadata: dict[str, Any],
    comments: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
    review_comments: list[dict[str, Any]],
) -> dict[str, Any]:
    is_pull_request = bool(metadata.get("pull_request"))
    payload = {
        "__typename": "PullRequest" if is_pull_request else "Issue",
        "id": str(metadata.get("node_id") or metadata.get("id") or ""),
        "comments": {
            "nodes": [_rest_comment_node(node) for node in comments],
            "pageInfo": {"hasNextPage": False, "endCursor": None},
        },
    }
    if is_pull_request:
        payload["reviews"] = {
            "nodes": [
                {
                    "id": str(node.get("node_id") or node.get("id") or ""),
                    "submittedAt": node.get("submitted_at"),
                    "author": {
                        "login": (node.get("user") or {}).get("login")
                    },
                    "body": node.get("body") or "",
                    "state": node.get("state") or "UNKNOWN",
                }
                for node in reviews
            ],
            "pageInfo": {"hasNextPage": False, "endCursor": None},
        }
        payload["reviewThreads"] = {
            "nodes": [
                {
                    "id": "rest_flat_review_comments",
                    "comments": {
                        "nodes": [
                            _rest_comment_node(node)
                            for node in review_comments
                        ],
                        "pageInfo": {
                            "hasNextPage": False,
                            "endCursor": None,
                        },
                    },
                }
            ],
            "pageInfo": {"hasNextPage": False, "endCursor": None},
        }
    return payload


def normalize_evidence_comment(record: dict[str, Any]) -> str:
    status = record.get("retrieval_status")
    if status == "ok":
        comments = (record.get("comments") or "").strip()
        if not comments:
            raise ValueError("ok evidence has no comments")
        return comments
    if status == "ok_zero_comments":
        return "no_comments_in_source"
    if status == "source_unavailable":
        return "comments_unavailable_in_source"
    raise ValueError(f"refusing evidence status: {status}")


def apply_comments_to_rows(
    rows: list[dict[str, str]],
    evidence: dict[str, dict[str, Any]],
    paper_id: str,
) -> tuple[list[dict[str, str]], int]:
    target_ids = {
        row["record_id"] for row in rows if row.get("paper_id") == paper_id
    }
    missing = sorted(target_ids - set(evidence))
    if missing:
        raise ValueError(f"missing evidence for {len(missing)} records")

    updated: list[dict[str, str]] = []
    changed = 0
    for row in rows:
        copied = dict(row)
        if row.get("paper_id") == paper_id:
            comments = normalize_evidence_comment(evidence[row["record_id"]])
            if copied.get("comments") != comments:
                copied["comments"] = comments
                changed += 1
        updated.append(copied)
    return updated, changed


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = "\n".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows
    )
    path.write_text(f"{value}\n" if value else "", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").split("\n")
        if line.strip()
    ]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def graphql_request(query: str, retries: int = 4) -> dict[str, Any]:
    request = json.dumps({"query": query}, ensure_ascii=False)
    for attempt in range(retries):
        process = subprocess.run(
            ["gh", "api", "graphql", "--input", "-"],
            cwd=ROOT,
            input=request,
            text=True,
            encoding="utf-8",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if process.returncode == 0:
            payload = json.loads(process.stdout)
            if payload.get("errors"):
                raise RuntimeError(
                    "GitHub GraphQL errors: "
                    + json.dumps(payload["errors"], ensure_ascii=False)
                )
            return payload
        if "Could not resolve to" in process.stderr:
            raise RuntimeError(
                f"GitHub GraphQL failed: {process.stderr.strip()}"
            )
        if attempt + 1 == retries:
            raise RuntimeError(
                f"GitHub GraphQL failed: {process.stderr.strip()}"
            )
        time.sleep(2**attempt)
    raise AssertionError("unreachable")


def _connection_counts(payload: dict[str, Any]) -> dict[str, int]:
    counts = {
        "conversation_comments": len(
            (payload.get("comments") or {}).get("nodes") or []
        ),
        "reviews": sum(
            1
            for node in (payload.get("reviews") or {}).get("nodes") or []
            if (node.get("body") or "").strip()
        ),
        "review_comments": sum(
            len((thread.get("comments") or {}).get("nodes") or [])
            for thread in (
                (payload.get("reviewThreads") or {}).get("nodes") or []
            )
        ),
    }
    return counts


def evidence_record(
    row: dict[str, str],
    *,
    retrieval_status: str,
    comments: str,
    source_comment_count: int | None,
    source_payload: dict[str, Any],
    retrieved_at: str | None = None,
) -> dict[str, Any]:
    _owner, _repository, _number, expected_kind = parse_github_record_url(
        row["issue_url"]
    )
    return {
        "record_id": row["record_id"],
        "paper_id": PAPER_ID,
        "issue_url": row["issue_url"],
        "source_project": row["source_project"],
        "source_kind": f"github_{expected_kind}",
        "evidence_version": EVIDENCE_VERSION,
        "retrieved_at": retrieved_at or now_utc(),
        "retrieval_status": retrieval_status,
        "source_comment_count": source_comment_count,
        "comments": comments,
        "source_payload": source_payload,
    }


def complete_pagination(payload: dict[str, Any]) -> dict[str, Any]:
    while True:
        tasks = find_pagination_tasks(payload)
        if not tasks:
            return payload
        task = tasks[0]
        response = graphql_request(build_connection_page_query(task))
        page = ((response.get("data") or {}).get("node") or {}).get("page")
        if page is None:
            raise ValueError(
                f"missing page for {task['node_id']} {task['connection']}"
            )
        merge_connection_page(payload, task, page)


def collect_batch(
    owner: str,
    repository: str,
    rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    specs = []
    by_number = {}
    for row in rows:
        parsed_owner, parsed_repository, number, expected_kind = (
            parse_github_record_url(row["issue_url"])
        )
        if (parsed_owner, parsed_repository) != (owner, repository):
            raise ValueError("mixed repositories in collection batch")
        specs.append((number, expected_kind))
        by_number[number] = row

    response = graphql_request(
        build_initial_discussion_query(owner, repository, specs)
    )
    repository_payload = (response.get("data") or {}).get("repository")
    if repository_payload is None:
        return [
            evidence_record(
                row,
                retrieval_status="source_unavailable",
                comments="",
                source_comment_count=None,
                source_payload={
                    "reason": "repository unavailable",
                    "repository": f"{owner}/{repository}",
                },
            )
            for row in rows
        ]

    result = []
    for number, expected_kind in specs:
        row = by_number[number]
        payload = repository_payload.get(f"record_{number}")
        if payload is None:
            result.append(
                evidence_record(
                    row,
                    retrieval_status="source_unavailable",
                    comments="",
                    source_comment_count=None,
                    source_payload={
                        "reason": "issue or pull request unavailable",
                        "repository": f"{owner}/{repository}",
                        "number": number,
                    },
                )
            )
            continue
        complete_pagination(payload)
        comments, count = compose_discussion(payload)
        actual_kind = (
            "issue"
            if payload.get("__typename") == "Issue"
            else "pull_request"
            if payload.get("__typename") == "PullRequest"
            else "unknown"
        )
        if actual_kind == "unknown":
            raise ValueError(
                f"unexpected GraphQL type for {owner}/{repository}#{number}: "
                f"{payload.get('__typename')}"
            )
        status = "ok" if count else "ok_zero_comments"
        result.append(
            evidence_record(
                row,
                retrieval_status=status,
                comments=comments,
                source_comment_count=count,
                source_payload={
                    "source_reconstruction": "github_graphql",
                    "repository": f"{owner}/{repository}",
                    "number": number,
                    "expected_kind": expected_kind,
                    "actual_kind": actual_kind,
                    "kind_matches_url": expected_kind == actual_kind,
                    "channel_counts": _connection_counts(payload),
                },
            )
        )
    return result


def collect_resilient_batch(
    owner: str,
    repository: str,
    rows: list[dict[str, str]],
    *,
    batch_collector: Callable[
        [str, str, list[dict[str, str]]],
        list[dict[str, Any]],
    ] = collect_batch,
) -> list[dict[str, Any]]:
    try:
        return batch_collector(owner, repository, rows)
    except Exception as error:
        message = str(error)
        if "Could not resolve to a Repository" in message:
            return [
                evidence_record(
                    row,
                    retrieval_status="source_unavailable",
                    comments="",
                    source_comment_count=None,
                    source_payload={
                        "reason": "repository unavailable",
                        "repository": f"{owner}/{repository}",
                    },
                )
                for row in rows
            ]
        if "Could not resolve to an issue or pull request" not in message:
            raise
        unresolved_numbers = {
            int(value)
            for value in re.findall(
                r"issue or pull request with the number of (\d+)",
                message,
            )
        }
        if unresolved_numbers:
            unavailable = []
            survivors = []
            for row in rows:
                _owner, _repository, number, _kind = (
                    parse_github_record_url(row["issue_url"])
                )
                if number in unresolved_numbers:
                    unavailable.append(
                        evidence_record(
                            row,
                            retrieval_status="source_unavailable",
                            comments="",
                            source_comment_count=None,
                            source_payload={
                                "reason": (
                                    "issue or pull request unavailable"
                                ),
                                "repository": f"{owner}/{repository}",
                                "number": number,
                            },
                        )
                    )
                else:
                    survivors.append(row)
            if unavailable:
                return [
                    *unavailable,
                    *(
                        collect_resilient_batch(
                            owner,
                            repository,
                            survivors,
                            batch_collector=batch_collector,
                        )
                        if survivors
                        else []
                    ),
                ]
        if len(rows) == 1:
            row = rows[0]
            _owner, _repository, number, _kind = parse_github_record_url(
                row["issue_url"]
            )
            return [
                evidence_record(
                    row,
                    retrieval_status="source_unavailable",
                    comments="",
                    source_comment_count=None,
                    source_payload={
                        "reason": "issue or pull request unavailable",
                        "repository": f"{owner}/{repository}",
                        "number": number,
                    },
                )
            ]
        middle = len(rows) // 2
        return [
            *collect_resilient_batch(
                owner,
                repository,
                rows[:middle],
                batch_collector=batch_collector,
            ),
            *collect_resilient_batch(
                owner,
                repository,
                rows[middle:],
                batch_collector=batch_collector,
            ),
        ]


def collect(
    rows: list[dict[str, str]],
    *,
    batch_size: int = 10,
) -> list[dict[str, Any]]:
    cached = {
        row["record_id"]: row
        for row in read_jsonl(CACHE_PATH)
        if row.get("retrieval_status") != "collection_failed"
    }
    pending = [row for row in rows if row["record_id"] not in cached]
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in pending:
        owner, repository, _number, _kind = parse_github_record_url(
            row["issue_url"]
        )
        grouped.setdefault((owner, repository), []).append(row)

    completed = 0
    for (owner, repository), repository_rows in grouped.items():
        for start in range(0, len(repository_rows), batch_size):
            batch = repository_rows[start : start + batch_size]
            try:
                records = collect_resilient_batch(
                    owner,
                    repository,
                    batch,
                )
            except Exception as error:
                records = [
                    evidence_record(
                        row,
                        retrieval_status="collection_failed",
                        comments="",
                        source_comment_count=None,
                        source_payload={
                            "reason": str(error),
                            "repository": f"{owner}/{repository}",
                        },
                    )
                    for row in batch
                ]
            for record in records:
                cached[record["record_id"]] = record
            write_jsonl(
                CACHE_PATH,
                sorted(cached.values(), key=lambda item: item["record_id"]),
            )
            completed += len(batch)
            print(
                f"collected {completed}/{len(pending)} pending; "
                f"cache={len(cached)}; repository={owner}/{repository}",
                flush=True,
            )
    return sorted(cached.values(), key=lambda item: item["record_id"])


class SourceUnavailableError(RuntimeError):
    """The GitHub source returns a stable 404/410 response."""


def _http_session(token: str) -> requests.Session:
    session = getattr(_HTTP_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        session.headers.update(
            {
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {token}",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "AutoEmpirical-evidence-reconstruction",
            }
        )
        _HTTP_LOCAL.session = session
    return session


def rest_get(
    path: str,
    token: str,
    *,
    paginate: bool = False,
) -> Any:
    url = f"https://api.github.com/{path.lstrip('/')}"
    session = _http_session(token)
    collected = []
    while url:
        response = session.get(
            url,
            params={"per_page": 100} if paginate and not collected else None,
            timeout=45,
        )
        if response.status_code in {404, 410}:
            raise SourceUnavailableError(
                f"GitHub REST source unavailable: HTTP {response.status_code}"
            )
        response.raise_for_status()
        value = response.json()
        if not paginate:
            return value
        if not isinstance(value, list):
            raise ValueError(f"expected REST list response: {path}")
        collected.extend(value)
        url = response.links.get("next", {}).get("url")
    return collected


def collect_rest_record(
    row: dict[str, str],
    token: str,
) -> dict[str, Any]:
    owner, repository, number, expected_kind = parse_github_record_url(
        row["issue_url"]
    )
    base = f"repos/{owner}/{repository}"
    try:
        metadata = rest_get(f"{base}/issues/{number}", token)
        comments = rest_get(
            f"{base}/issues/{number}/comments",
            token,
            paginate=True,
        )
        is_pull_request = bool(metadata.get("pull_request"))
        reviews = (
            rest_get(
                f"{base}/pulls/{number}/reviews",
                token,
                paginate=True,
            )
            if is_pull_request
            else []
        )
        review_comments = (
            rest_get(
                f"{base}/pulls/{number}/comments",
                token,
                paginate=True,
            )
            if is_pull_request
            else []
        )
        payload = build_payload_from_rest(
            metadata,
            comments,
            reviews,
            review_comments,
        )
        rendered, count = compose_discussion(payload)
        actual_kind = "pull_request" if is_pull_request else "issue"
        return evidence_record(
            row,
            retrieval_status="ok" if count else "ok_zero_comments",
            comments=rendered,
            source_comment_count=count,
            source_payload={
                "source_reconstruction": "github_rest_fallback",
                "repository": f"{owner}/{repository}",
                "number": number,
                "expected_kind": expected_kind,
                "actual_kind": actual_kind,
                "kind_matches_url": expected_kind == actual_kind,
                "channel_counts": _connection_counts(payload),
            },
        )
    except SourceUnavailableError as error:
        return evidence_record(
            row,
            retrieval_status="source_unavailable",
            comments="",
            source_comment_count=None,
            source_payload={
                "reason": str(error),
                "repository": f"{owner}/{repository}",
                "number": number,
            },
        )
    except Exception as error:
        return evidence_record(
            row,
            retrieval_status="collection_failed",
            comments="",
            source_comment_count=None,
            source_payload={
                "reason": str(error),
                "repository": f"{owner}/{repository}",
                "number": number,
            },
        )


def collect_rest_fallback(
    rows: list[dict[str, str]],
    *,
    workers: int = 8,
) -> list[dict[str, Any]]:
    token_process = subprocess.run(
        ["gh", "auth", "token"],
        cwd=ROOT,
        check=True,
        text=True,
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    token = token_process.stdout.strip()
    if not token:
        raise ValueError("GitHub token is unavailable")

    cached = {row["record_id"]: row for row in read_jsonl(CACHE_PATH)}
    pending = [
        row
        for row in rows
        if (cached.get(row["record_id"]) or {}).get("retrieval_status")
        == "collection_failed"
    ]
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(collect_rest_record, row, token): row
            for row in pending
        }
        for future in as_completed(futures):
            record = future.result()
            cached[record["record_id"]] = record
            completed += 1
            if completed % 25 == 0 or completed == len(pending):
                write_jsonl(
                    CACHE_PATH,
                    sorted(
                        cached.values(),
                        key=lambda item: item["record_id"],
                    ),
                )
                print(
                    f"REST fallback {completed}/{len(pending)}; "
                    f"status={record['retrieval_status']}",
                    flush=True,
                )
    return sorted(cached.values(), key=lambda item: item["record_id"])


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"missing CSV header: {path}")
        return list(reader.fieldnames), list(reader)


def write_csv(
    path: Path,
    fields: list[str],
    rows: Iterable[dict[str, str]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def immutable_projection(rows: list[dict[str, str]]) -> str:
    payload = [
        {field: value for field, value in row.items() if field != "comments"}
        for row in rows
    ]
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


SECRET_PATTERNS = {
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "azure_iot_shared_access_key": re.compile(
        r"(?i)(?<=SharedAccessKey=)"
        r"(?!(?:([A-Za-z0-9+/])\1{29,}={0,2})(?=[;\s\"',]|\Z))"
        r"[^;\s\"',\[]+"
    ),
    "private_key": re.compile(
        r"-----BEGIN ([A-Z0-9 ]*PRIVATE KEY)-----.*?"
        r"(?:-----END \1-----|\Z)",
        re.DOTALL,
    ),
}


def redact_high_confidence_secrets(
    text: str,
) -> tuple[str, dict[str, int]]:
    cleaned = text
    findings = {}
    for name, pattern in SECRET_PATTERNS.items():
        cleaned, count = pattern.subn(
            f"[REDACTED_HIGH_CONFIDENCE_SECRET:{name}]",
            cleaned,
        )
        if count:
            findings[name] = count
    return cleaned, findings


def redact_secrets_in_rows(
    rows: list[dict[str, str]],
    paper_id: str,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    cleaned_rows = []
    redactions = []
    for row in rows:
        copied = dict(row)
        if row.get("paper_id") == paper_id:
            for field, value in row.items():
                cleaned, findings = redact_high_confidence_secrets(value)
                if findings:
                    copied[field] = cleaned
                    redactions.append(
                        {
                            "record_id": row["record_id"],
                            "field": field,
                            "patterns": findings,
                        }
                    )
        cleaned_rows.append(copied)
    return cleaned_rows, redactions


def secret_findings(
    evidence_rows: Iterable[dict[str, Any]],
) -> list[dict[str, str]]:
    findings = []
    for row in evidence_rows:
        comments = str(row.get("comments") or "")
        for name, pattern in SECRET_PATTERNS.items():
            if pattern.search(comments):
                findings.append(
                    {"record_id": row["record_id"], "pattern": name}
                )
    return findings


def coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "records": len(rows),
        "unique_record_ids": len({row["record_id"] for row in rows}),
        "unique_issue_urls": len({row["issue_url"] for row in rows}),
        "status_counts": dict(
            Counter(row["retrieval_status"] for row in rows)
        ),
        "kind_counts": dict(Counter(row["source_kind"] for row in rows)),
        "kind_mismatches": sum(
            not (row.get("source_payload") or {}).get(
                "kind_matches_url",
                True,
            )
            for row in rows
        ),
    }


def apply() -> dict[str, Any]:
    evidence_rows = read_jsonl(CACHE_PATH)
    if len(evidence_rows) != 5548:
        raise ValueError(
            f"expected 5548 evidence records, found {len(evidence_rows)}"
        )
    evidence = {row["record_id"]: row for row in evidence_rows}
    if len(evidence) != 5548:
        raise ValueError("duplicate evidence record IDs")
    failed = [
        row
        for row in evidence_rows
        if row["retrieval_status"] == "collection_failed"
    ]
    if failed:
        raise ValueError(f"refusing {len(failed)} collection failures")
    credential_redactions = []
    sanitized_evidence_rows = []
    for row in evidence_rows:
        copied = dict(row)
        copied["comments"], redactions = redact_high_confidence_secrets(
            str(row.get("comments") or "")
        )
        if redactions:
            credential_redactions.append(
                {
                    "record_id": row["record_id"],
                    "patterns": redactions,
                }
            )
        sanitized_evidence_rows.append(copied)
    evidence_rows = sanitized_evidence_rows
    evidence = {row["record_id"]: row for row in evidence_rows}
    findings = secret_findings(evidence_rows)
    if findings:
        raise ValueError("secret redaction failed")

    file_results = []
    for path in [*PER_PAPER_PATHS, *UNIFIED_PATHS]:
        fields, rows = read_csv(path)
        sanitized_rows, row_redactions = redact_secrets_in_rows(
            rows,
            PAPER_ID,
        )
        before = immutable_projection(sanitized_rows)
        updated, changed = apply_comments_to_rows(
            sanitized_rows,
            evidence,
            PAPER_ID,
        )
        after = immutable_projection(updated)
        if before != after:
            raise ValueError(f"immutable fields changed in memory: {path}")
        write_csv(path, fields, updated)
        file_results.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "rows": len(rows),
                "target_rows": sum(
                    row["paper_id"] == PAPER_ID for row in rows
                ),
                "updated_rows": changed,
                "security_redactions": row_redactions,
                "immutable_projection_sha256_before": before,
                "immutable_projection_sha256_after": after,
            }
        )

    write_jsonl(EVIDENCE_PATH, evidence_rows)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    audit = {
        "task": "icse2021_iot_discussion_reconstruction",
        "generated_at": now_utc(),
        "paper_id": PAPER_ID,
        "evidence_version": EVIDENCE_VERSION,
        **coverage(evidence_rows),
        "high_confidence_secret_findings": findings,
        "credential_redactions": credential_redactions,
        "files": file_results,
    }
    AUDIT_PATH.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--rest-fallback", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    _fields, stage1_rows = read_csv(PER_PAPER_PATHS[0])
    evidence_rows = read_jsonl(CACHE_PATH)
    if args.collect:
        evidence_rows = collect(stage1_rows, batch_size=args.batch_size)
    if args.rest_fallback:
        evidence_rows = collect_rest_fallback(
            stage1_rows,
            workers=args.workers,
        )
    result: dict[str, Any] = {"coverage": coverage(evidence_rows)}
    if args.apply:
        result["apply"] = apply()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
