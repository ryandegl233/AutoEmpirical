"""Full Stage 1 discussion reconstruction for ICSE 2024 Transaction Bugs.

Source reconstruction utility. Raw download caches stay local under Dataset/cache/.
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Iterable
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any
from urllib.parse import urlparse


NO_COMMENTS = "no_comments_in_source"
COMMENTS_UNAVAILABLE = "comments_unavailable_in_source"
PAPER_ID = "icse2024_understanding_transaction_bugs_in_database"
EVIDENCE_VERSION = "current_unversioned"
ROOT = Path(__file__).resolve().parents[2]
PER_PAPER_STAGE1 = (
    ROOT / "Dataset/by_paper" / PAPER_ID / "stage1.csv"
)
UNIFIED_STAGE1 = ROOT / "Dataset/stage1.csv"
FINAL_EVIDENCE = ROOT / "Dataset/evidence/icse2024_transaction_bugs_evidence.jsonl"
STAGE1_EVIDENCE = (
    ROOT
    / "Dataset/evidence/icse2024_transaction_bugs_stage1_evidence.jsonl"
)
GITHUB_CACHE = (
    ROOT / "Dataset/cache/txbug_stage1_github_discussions.jsonl"
)
GITHUB_CANDIDATES = (
    ROOT
    / "reports/txbug_stage1_reconstruction/github/core_plus_tidb_label/"
    "txbug_github_candidates.csv"
)
MARIADB_CANDIDATES = (
    ROOT
    / "reports/txbug_stage1_reconstruction/mariadb/"
    "txbug_mariadb_candidates.csv"
)
MYSQL_CANDIDATES = (
    ROOT
    / "reports/txbug_stage1_reconstruction/mysql/"
    "txbug_mysql_candidates_with_text.csv"
)
AUDIT_PATH = (
    ROOT / "reports/txbug_stage1_information_reconstruction/audit.json"
)
README_PATH = (
    ROOT / "reports/txbug_stage1_information_reconstruction/README.md"
)
csv.field_size_limit(min(sys.maxsize, 2_147_483_647))

SECRET_PATTERNS = {
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    "openai_key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "google_api_key": re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b"),
    "private_key": re.compile(
        r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
    ),
    "bearer_token": re.compile(
        r"\bBearer\s+[A-Za-z0-9._~+/=-]{20,}",
        re.IGNORECASE,
    ),
}


def normalize_evidence_comment(retrieval_status: str, comments: str) -> str:
    value = str(comments or "").strip()
    if retrieval_status == "ok":
        if not value:
            raise ValueError("ok evidence must contain comments")
        return value
    if retrieval_status == "ok_zero_comments":
        if value and value != NO_COMMENTS:
            raise ValueError("zero-comment evidence contains unexpected text")
        return NO_COMMENTS
    if retrieval_status == "source_unavailable":
        if value and value != COMMENTS_UNAVAILABLE:
            raise ValueError("unavailable evidence contains unexpected text")
        return COMMENTS_UNAVAILABLE
    raise ValueError(f"refusing evidence status: {retrieval_status}")


def apply_comments_to_rows(
    rows: list[dict[str, str]],
    evidence: dict[str, dict[str, Any]],
) -> dict[str, int]:
    row_ids = {row["record_id"] for row in rows}
    missing = row_ids - set(evidence)
    if missing:
        raise ValueError(f"missing evidence for {len(missing)} records")
    updated = 0
    for row in rows:
        item = evidence[row["record_id"]]
        replacement = normalize_evidence_comment(
            item["retrieval_status"],
            item.get("comments", ""),
        )
        if row.get("comments") != replacement:
            row["comments"] = replacement
            updated += 1
    return {"target_rows": len(rows), "updated_rows": updated}


def build_github_comments_query(
    owner: str,
    repository: str,
    issue_numbers: Iterable[int],
    *,
    comments_first: int = 100,
) -> str:
    if not 1 <= comments_first <= 100:
        raise ValueError("comments_first must be between 1 and 100")
    issue_fields = []
    for number in issue_numbers:
        issue_fields.append(
            f"""
      issue_{int(number)}: issue(number: {int(number)}) {{
        url
        comments(first: {comments_first}) {{
          totalCount
          nodes {{
            author {{ login }}
            body
            createdAt
            updatedAt
            url
          }}
          pageInfo {{ hasNextPage endCursor }}
        }}
      }}"""
        )
    return (
        "query {\n"
        f'  repository(owner: "{owner}", name: "{repository}") {{'
        + "".join(issue_fields)
        + "\n  }\n}"
    )


def _format_comment(node: dict[str, Any]) -> str:
    author = (node.get("author") or {}).get("login") or "unknown"
    created = node.get("createdAt") or "not_available_in_source"
    updated = node.get("updatedAt") or "not_available_in_source"
    url = node.get("url") or ""
    body = str(node.get("body") or "").strip()
    return (
        f"[issue_comment] {author} | created={created} | "
        f"updated={updated} | {url}\n{body}"
    ).strip()


def parse_github_comments_response(
    payload: dict[str, Any],
    issue_numbers: Iterable[int],
) -> dict[int, dict[str, Any]]:
    repository = (payload.get("data") or {}).get("repository") or {}
    parsed: dict[int, dict[str, Any]] = {}
    for number in issue_numbers:
        issue = repository.get(f"issue_{int(number)}")
        if issue is None:
            parsed[int(number)] = {
                "retrieval_status": "source_unavailable",
                "comments": "",
                "source_comment_count": None,
            }
            continue
        comments = issue["comments"]
        nodes = comments.get("nodes") or []
        rendered = "\n\n---COMMENT---\n\n".join(
            _format_comment(node) for node in nodes
        )
        total = int(comments.get("totalCount") or 0)
        parsed[int(number)] = {
            "retrieval_status": "ok" if total else "ok_zero_comments",
            "comments": rendered,
            "source_comment_count": total,
            "retrieved_comment_count": len(nodes),
            "page_info": comments.get("pageInfo") or {},
            "issue_url": issue.get("url"),
        }
    return parsed


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"missing CSV header: {path}")
        return list(reader.fieldnames), list(reader)


def write_csv(
    path: Path,
    fields: list[str],
    rows: list[dict[str, str]],
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        # JSONL records are delimited only by LF. str.splitlines() also
        # splits U+2028/U+2029 embedded in valid JSON string values.
        for line in path.read_text(encoding="utf-8").split("\n")
        if line.strip()
    ]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in rows
    )
    path.write_text(value, encoding="utf-8", newline="\n")


def url_key(value: str) -> str:
    return value.strip().rstrip("/")


def source_kind(value: str) -> str:
    host = urlparse(value).netloc.lower()
    if host == "github.com":
        return "github_issue"
    if host == "jira.mariadb.org":
        return "mariadb_jira"
    if host == "bugs.mysql.com":
        return "mysql_bugs"
    if host in {"postgresql.org", "www.postgresql.org", "postgr.es"}:
        return "postgresql_mail_thread"
    if "sqlite.org" in host:
        return "sqlite_fossil_ticket"
    return "unknown"


def read_candidate_map(path: Path) -> dict[str, dict[str, str]]:
    _, rows = read_csv(path)
    return {url_key(row["issue_url"]): row for row in rows}


def evidence_record(
    row: dict[str, str],
    *,
    retrieval_status: str,
    comments: str,
    source_comment_count: int | None,
    source_payload: dict[str, Any],
    retrieved_at: str | None = None,
) -> dict[str, Any]:
    return {
        "record_id": row["record_id"],
        "paper_id": PAPER_ID,
        "issue_url": row["issue_url"],
        "source_project": row["source_project"],
        "source_kind": source_kind(row["issue_url"]),
        "evidence_version": EVIDENCE_VERSION,
        "retrieved_at": retrieved_at or now_utc(),
        "retrieval_status": retrieval_status,
        "source_comment_count": source_comment_count,
        "comments": normalize_evidence_comment(
            retrieval_status,
            comments,
        ),
        "source_payload": source_payload,
    }


def parse_github_url(value: str) -> tuple[str, str, int]:
    parts = [part for part in urlparse(value).path.split("/") if part]
    if len(parts) < 4 or parts[2] != "issues":
        raise ValueError(f"unsupported GitHub issue URL: {value}")
    return parts[0], parts[1], int(parts[3])


def graphql_request(query: str, retries: int = 4) -> dict[str, Any]:
    for attempt in range(retries):
        process = subprocess.run(
            ["gh", "api", "graphql", "-f", f"query={query}"],
            cwd=ROOT,
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
        if attempt + 1 == retries:
            raise RuntimeError(
                f"GitHub GraphQL failed: {process.stderr.strip()}"
            )
        time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def build_github_comments_page_query(
    owner: str,
    repository: str,
    issue_number: int,
    cursor: str,
) -> str:
    encoded_cursor = json.dumps(cursor)
    return f"""
query {{
  repository(owner: {json.dumps(owner)}, name: {json.dumps(repository)}) {{
    issue(number: {int(issue_number)}) {{
      url
      comments(first: 100, after: {encoded_cursor}) {{
        totalCount
        nodes {{
          author {{ login }}
          body
          createdAt
          updatedAt
          url
        }}
        pageInfo {{ hasNextPage endCursor }}
      }}
    }}
  }}
}}"""


def fetch_remaining_comment_pages(
    owner: str,
    repository: str,
    issue_number: int,
    item: dict[str, Any],
) -> dict[str, Any]:
    rendered_pages = [item["comments"]] if item["comments"] else []
    retrieved = int(item.get("retrieved_comment_count") or 0)
    page_info = item.get("page_info") or {}
    while page_info.get("hasNextPage"):
        cursor = page_info.get("endCursor")
        if not cursor:
            raise ValueError(
                f"missing pagination cursor for {owner}/{repository}#{issue_number}"
            )
        payload = graphql_request(
            build_github_comments_page_query(
                owner,
                repository,
                issue_number,
                cursor,
            )
        )
        issue = (
            ((payload.get("data") or {}).get("repository") or {}).get("issue")
        )
        if issue is None:
            raise ValueError(
                f"issue disappeared during pagination: "
                f"{owner}/{repository}#{issue_number}"
            )
        comments = issue["comments"]
        nodes = comments.get("nodes") or []
        page_text = "\n\n---COMMENT---\n\n".join(
            _format_comment(node) for node in nodes
        )
        if page_text:
            rendered_pages.append(page_text)
        retrieved += len(nodes)
        page_info = comments.get("pageInfo") or {}
    total = int(item.get("source_comment_count") or 0)
    if retrieved != total:
        raise ValueError(
            f"incomplete comments for {owner}/{repository}#{issue_number}: "
            f"{retrieved}/{total}"
        )
    item["comments"] = "\n\n---COMMENT---\n\n".join(rendered_pages)
    item["retrieved_comment_count"] = retrieved
    item["page_info"] = page_info
    return item


def collect_github(
    pending: list[dict[str, Any]],
    *,
    batch_size: int = 20,
) -> dict[str, dict[str, Any]]:
    cached = {
        row["record_id"]: row
        for row in read_jsonl(GITHUB_CACHE)
        if row.get("retrieval_status") != "collection_failed"
    }
    remaining = [row for row in pending if row["record_id"] not in cached]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in remaining:
        owner, repository, number = parse_github_url(row["issue_url"])
        item = dict(row)
        item["_owner"] = owner
        item["_repository"] = repository
        item["_number"] = number
        grouped.setdefault((owner, repository), []).append(item)

    for (owner, repository), rows in grouped.items():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            numbers = [row["_number"] for row in batch]
            payload = graphql_request(
                build_github_comments_query(
                    owner,
                    repository,
                    numbers,
                )
            )
            parsed = parse_github_comments_response(payload, numbers)
            for row in batch:
                number = row["_number"]
                item = parsed[number]
                if (
                    item["retrieval_status"] == "ok"
                    and (item.get("page_info") or {}).get("hasNextPage")
                ):
                    item = fetch_remaining_comment_pages(
                        owner,
                        repository,
                        number,
                        item,
                    )
                cached[row["record_id"]] = evidence_record(
                    row,
                    retrieval_status=item["retrieval_status"],
                    comments=item.get("comments", ""),
                    source_comment_count=item.get("source_comment_count"),
                    source_payload={
                        "source_reconstruction": "github_graphql",
                        "repository": f"{owner}/{repository}",
                        "issue_number": number,
                        "candidate_snapshot_comment_count": row[
                            "candidate_snapshot_comment_count"
                        ],
                        "retrieved_comment_count": item.get(
                            "retrieved_comment_count"
                        ),
                    },
                )
            write_jsonl(
                GITHUB_CACHE,
                sorted(cached.values(), key=lambda item: item["record_id"]),
            )
            print(
                f"github {owner}/{repository}: "
                f"{min(start + len(batch), len(rows))}/{len(rows)} "
                f"(cache total {len(cached)})",
                flush=True,
            )
    return cached


def retained_evidence(
    row: dict[str, str],
    source: dict[str, str],
    *,
    source_name: str,
    count_field: str,
) -> dict[str, Any]:
    comments = str(source.get("comments") or "").strip()
    count_value = str(source.get(count_field) or "").strip()
    count = int(count_value) if count_value else 0
    status = "ok" if comments else "ok_zero_comments" if count == 0 else "source_unavailable"
    return evidence_record(
        row,
        retrieval_status=status,
        comments=comments,
        source_comment_count=count if status != "source_unavailable" else None,
        source_payload={
            "source_reconstruction": source_name,
            "candidate_snapshot_comment_count": count,
        },
    )


def build_full_evidence(
    stage1_rows: list[dict[str, str]],
    *,
    collect: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    final_evidence = {
        row["record_id"]: row for row in read_jsonl(FINAL_EVIDENCE)
    }
    github = read_candidate_map(GITHUB_CANDIDATES)
    mariadb = read_candidate_map(MARIADB_CANDIDATES)
    mysql = read_candidate_map(MYSQL_CANDIDATES)
    cache = {
        row["record_id"]: row for row in read_jsonl(GITHUB_CACHE)
    }

    evidence: dict[str, dict[str, Any]] = {}
    pending: list[dict[str, Any]] = []
    for row in stage1_rows:
        record_id = row["record_id"]
        if record_id in final_evidence:
            evidence[record_id] = final_evidence[record_id]
            continue
        kind = source_kind(row["issue_url"])
        source = {
            "github_issue": github,
            "mariadb_jira": mariadb,
            "mysql_bugs": mysql,
        }.get(kind, {}).get(url_key(row["issue_url"]))
        if source is None:
            raise ValueError(
                f"no retained source or final evidence for {row['issue_url']}"
            )
        if kind == "github_issue":
            count = int(source["comments_count"])
            if count == 0:
                evidence[record_id] = evidence_record(
                    row,
                    retrieval_status="ok_zero_comments",
                    comments="",
                    source_comment_count=0,
                    source_payload={
                        "source_reconstruction": "github_candidate_metadata",
                        "candidate_snapshot_comment_count": 0,
                    },
                )
            elif record_id in cache:
                evidence[record_id] = cache[record_id]
            else:
                item = dict(row)
                item["candidate_snapshot_comment_count"] = count
                pending.append(item)
        elif kind == "mariadb_jira":
            evidence[record_id] = retained_evidence(
                row,
                source,
                source_name="mariadb_jira_retained",
                count_field="comments_count",
            )
        elif kind == "mysql_bugs":
            evidence[record_id] = retained_evidence(
                row,
                source,
                source_name="mysql_bugs_retained",
                count_field="comment_count_extracted",
            )
        else:
            raise ValueError(f"unsupported source kind: {kind}")

    if collect and pending:
        collected = collect_github(pending)
        for row in pending:
            evidence[row["record_id"]] = collected[row["record_id"]]
        pending = []

    ordered = [evidence[row["record_id"]] for row in stage1_rows if row["record_id"] in evidence]
    audit = {
        "stage1_rows": len(stage1_rows),
        "evidence_rows_ready": len(ordered),
        "pending_github_rows": len(pending),
        "source_counts": {},
        "status_counts": {},
    }
    for item in ordered:
        kind = item["source_kind"]
        status = item["retrieval_status"]
        audit["source_counts"][kind] = audit["source_counts"].get(kind, 0) + 1
        audit["status_counts"][status] = audit["status_counts"].get(status, 0) + 1
    return ordered, audit


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


def secret_findings(rows: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    findings = []
    for row in rows:
        comments = str(row.get("comments") or "")
        for name, pattern in SECRET_PATTERNS.items():
            if pattern.search(comments):
                findings.append(
                    {"record_id": row["record_id"], "pattern": name}
                )
    return findings


def apply(root: Path = ROOT) -> dict[str, Any]:
    _, per_paper_rows = read_csv(PER_PAPER_STAGE1)
    evidence_rows, coverage = build_full_evidence(
        per_paper_rows,
        collect=False,
    )
    if coverage["pending_github_rows"]:
        raise ValueError(
            f"cannot apply with {coverage['pending_github_rows']} pending GitHub rows"
        )
    if len(evidence_rows) != 7775:
        raise ValueError(f"expected 7775 evidence rows, found {len(evidence_rows)}")
    evidence = {row["record_id"]: row for row in evidence_rows}
    if len(evidence) != len(evidence_rows):
        raise ValueError("duplicate evidence record IDs")
    findings = secret_findings(evidence_rows)
    if findings:
        raise ValueError(
            "high-confidence secret findings block apply: "
            + json.dumps(findings[:10], ensure_ascii=False)
        )

    file_audits = []
    for path, unified in (
        (PER_PAPER_STAGE1, False),
        (UNIFIED_STAGE1, True),
    ):
        fields, rows = read_csv(path)
        before_hash = immutable_projection(rows)
        target_rows = (
            [row for row in rows if row["paper_id"] == PAPER_ID]
            if unified
            else rows
        )
        result = apply_comments_to_rows(target_rows, evidence)
        after_hash = immutable_projection(rows)
        if before_hash != after_hash:
            raise ValueError(f"non-comment fields changed in {path}")
        write_csv(path, fields, rows)
        file_audits.append(
            {
                "path": str(path).replace("\\", "/"),
                **result,
                "immutable_projection_sha256_before": before_hash,
                "immutable_projection_sha256_after": after_hash,
            }
        )

    write_jsonl(STAGE1_EVIDENCE, evidence_rows)
    audit = {
        "task": "icse2024_transaction_bugs_full_stage1_discussion_reconstruction",
        "generated_at": now_utc(),
        "paper_id": PAPER_ID,
        "evidence_version": EVIDENCE_VERSION,
        **coverage,
        "record_id_unique": len(evidence) == 7775,
        "issue_url_unique": len(
            {row["issue_url"] for row in evidence_rows}
        )
        == 7775,
        "high_confidence_secret_findings": findings,
        "files": file_audits,
    }
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--batch-size", type=int, default=20)
    args = parser.parse_args()
    _, rows = read_csv(PER_PAPER_STAGE1)
    evidence_rows, audit = build_full_evidence(rows, collect=False)
    if args.collect and audit["pending_github_rows"]:
        pending_ids = {
            row["record_id"] for row in rows
        } - {row["record_id"] for row in evidence_rows}
        github_map = read_candidate_map(GITHUB_CANDIDATES)
        pending = []
        for row in rows:
            if row["record_id"] not in pending_ids:
                continue
            source = github_map[url_key(row["issue_url"])]
            item = dict(row)
            item["candidate_snapshot_comment_count"] = int(
                source["comments_count"]
            )
            pending.append(item)
        collect_github(pending, batch_size=args.batch_size)
        evidence_rows, audit = build_full_evidence(rows, collect=False)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if args.apply:
        print(json.dumps(apply(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
