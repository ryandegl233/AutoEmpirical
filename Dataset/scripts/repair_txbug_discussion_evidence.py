from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote, unquote, urlparse
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

from bs4 import BeautifulSoup

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Dataset.scripts.fetch_txbug_postgresql_stage1_candidates import (
    _canonical_issue_url as canonical_postgresql_url,
    _text_from_html,
)
csv.field_size_limit(sys.maxsize)

PAPER_ID = "icse2024_understanding_transaction_bugs_in_database"
STAGES = ("stage1", "stage2", "stage3")
PLACEHOLDER_NO_COMMENTS = "no_comments_in_source"
PLACEHOLDER_UNAVAILABLE = "comments_unavailable_in_source"
CACHE_PATH = Path("Dataset/cache/txbug_discussion_evidence.jsonl")
SIDECAR_PATH = Path("Dataset/evidence/icse2024_transaction_bugs_evidence.jsonl")
AUDIT_PATH = Path("reports/txbug_information_reconstruction/audit.json")
README_PATH = Path("reports/txbug_information_reconstruction/README.md")
DATASET_README = Path("Dataset/README.md")
DATA_DICTIONARY = Path("metadata/data_dictionary.md")
METADATA_CSV = Path("metadata/dataset_metadata.csv")
METADATA_MD = Path("metadata/dataset_metadata.md")
OVERVIEW_MD = Path("metadata/paper_dataset_overview.md")
HASH_MANIFEST = Path("reports/SHA256SUMS.txt")
EXISTING_MYSQL = Path(
    "reports/txbug_stage1_reconstruction/mysql/"
    "txbug_mysql_candidates_with_text.csv"
)
EXISTING_MARIADB = Path(
    "reports/txbug_stage1_reconstruction/mariadb/"
    "txbug_mariadb_candidates.csv"
)
RETRIEVAL_VERSION = "current_unversioned"
USER_AGENT = "AutoEmpirical-TXBug-discussion-reconstruction/1.0"
RETRY_CODES = {429, 500, 502, 503, 504}
HIGH_CONFIDENCE_SECRET_PATTERNS = {
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    "openai_key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "google_api_key": re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b"),
    "private_key": re.compile(
        r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
    ),
    "bearer_token": re.compile(
        r"\bBearer\s+[A-Za-z0-9._~+/=-]{20,}", re.IGNORECASE
    ),
}
CHECKSUM_TARGETS = (
    "Dataset/README.md",
    f"Dataset/by_paper/{PAPER_ID}/stage1.csv",
    f"Dataset/by_paper/{PAPER_ID}/stage2.csv",
    f"Dataset/by_paper/{PAPER_ID}/stage3.csv",
    "Dataset/stage1.csv",
    "Dataset/stage2.csv",
    "Dataset/stage3.csv",
    str(SIDECAR_PATH).replace("\\", "/"),
    "metadata/data_dictionary.md",
    "metadata/dataset_metadata.csv",
    str(AUDIT_PATH).replace("\\", "/"),
    str(README_PATH).replace("\\", "/"),
)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    # Universal newline translation prevents a Windows checkout's CRLF bytes
    # from leaking into quoted multi-line field values during a rewrite.
    with path.open("r", encoding="utf-8-sig", newline=None) as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header: {path}")
        return list(reader.fieldnames), list(reader)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
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
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in rows
    )
    path.write_text(text, encoding="utf-8", newline="\n")


def request_bytes(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    retries: int = 4,
) -> tuple[bytes, dict[str, str]]:
    request_headers = {"User-Agent": USER_AGENT, **(headers or {})}
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urlopen(
                Request(url, headers=request_headers), timeout=90
            ) as response:
                return response.read(), dict(response.headers.items())
        except HTTPError as error:
            last_error = error
            if error.code not in RETRY_CODES or attempt == retries:
                raise
        except URLError as error:
            last_error = error
            if attempt == retries:
                raise
        time.sleep(min(2**attempt, 16))
    raise RuntimeError(f"Request failed: {url}: {last_error}") from last_error


def request_json(
    url: str,
    *,
    headers: dict[str, str] | None = None,
) -> tuple[Any, dict[str, str]]:
    payload, response_headers = request_bytes(url, headers=headers)
    return json.loads(payload.decode("utf-8")), response_headers


def github_token() -> str:
    completed = subprocess.run(
        ["gh", "auth", "token"],
        check=True,
        capture_output=True,
        text=True,
    )
    token = completed.stdout.strip()
    if not token:
        raise RuntimeError("gh auth token returned an empty token")
    return token


def github_issue_parts(issue_url: str) -> tuple[str, str, int]:
    match = re.fullmatch(
        r"https://github\.com/([^/]+)/([^/]+)/issues/(\d+)/?", issue_url
    )
    if not match:
        raise ValueError(f"Unsupported GitHub issue URL: {issue_url}")
    return match.group(1), match.group(2), int(match.group(3))


def github_paginated(
    url: str,
    *,
    headers: dict[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    page = 1
    while True:
        separator = "&" if "?" in url else "?"
        payload, _ = request_json(
            f"{url}{separator}per_page=100&page={page}", headers=headers
        )
        if not isinstance(payload, list):
            raise ValueError(f"Expected GitHub list response: {url}")
        rows.extend(payload)
        if len(payload) < 100:
            break
        page += 1
    return rows


def formatted_comment(
    author: str,
    created_at: str,
    body: str,
    *,
    kind: str = "comment",
) -> str:
    header = " | ".join(
        part for part in (kind, author.strip(), created_at.strip()) if part
    )
    return f"[{header}]\n{body.strip()}" if header else body.strip()


def joined_comments(comments: Iterable[str]) -> str:
    return "\n\n---COMMENT---\n\n".join(
        comment.strip() for comment in comments if comment.strip()
    )


def collect_github(record: dict[str, str], token: str) -> dict[str, Any]:
    owner, repo, number = github_issue_parts(record["issue_url"])
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    api = f"https://api.github.com/repos/{owner}/{repo}/issues/{number}"
    issue, _ = request_json(api, headers=headers)
    comments = github_paginated(f"{api}/comments", headers=headers)
    normalized = [
        {
            "author": str((item.get("user") or {}).get("login") or ""),
            "created_at": str(item.get("created_at") or ""),
            "updated_at": str(item.get("updated_at") or ""),
            "body": str(item.get("body") or ""),
            "html_url": str(item.get("html_url") or ""),
        }
        for item in comments
    ]
    text = joined_comments(
        formatted_comment(
            item["author"],
            item["created_at"],
            item["body"],
            kind="issue_comment",
        )
        for item in normalized
    )
    return evidence_record(
        record,
        source_kind="github_issue",
        retrieval_status="ok" if text else "ok_zero_comments",
        comments=text,
        source_comment_count=len(normalized),
        source_payload={
            "issue": {
                "api_url": api,
                "html_url": str(issue.get("html_url") or record["issue_url"]),
                "title": str(issue.get("title") or ""),
                "state": str(issue.get("state") or ""),
                "created_at": str(issue.get("created_at") or ""),
                "updated_at": str(issue.get("updated_at") or ""),
                "comments": int(issue.get("comments") or 0),
            },
            "comments": normalized,
        },
    )


def source_csv_index(path: Path) -> dict[str, dict[str, str]]:
    _, rows = read_csv(path)
    return {row["issue_url"]: row for row in rows}


def parse_int(value: str) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def collect_existing_tracker(
    record: dict[str, str],
    *,
    source_kind: str,
    source_row: dict[str, str] | None,
) -> dict[str, Any]:
    if source_row is None:
        return evidence_record(
            record,
            source_kind=source_kind,
            retrieval_status="source_unavailable",
            comments="",
            source_comment_count=None,
            source_payload={
                "reason": "final record not covered by the retained source reconstruction"
            },
        )
    comments = (source_row.get("comments") or "").strip()
    count = parse_int(
        source_row.get("comment_count_extracted")
        or source_row.get("comments_count")
        or ""
    )
    if comments and count == 0:
        count = comments.count("---COMMENT---") + 1
    return evidence_record(
        record,
        source_kind=source_kind,
        retrieval_status="ok" if comments else "ok_zero_comments",
        comments=comments,
        source_comment_count=count,
        source_payload={
            "source_reconstruction": source_row.get(
                "source_reconstruction", source_kind
            ),
            "detail_fetch_status": source_row.get("detail_fetch_status", ""),
            "source_comments_count": source_row.get("comments_count", ""),
            "comment_count_extracted": source_row.get(
                "comment_count_extracted", ""
            ),
        },
    )


def canonical_pg_url(url: str) -> str:
    return canonical_postgresql_url(url).replace("@", "%40")


def collect_postgresql(record: dict[str, str]) -> dict[str, Any]:
    root_url = canonical_pg_url(record["issue_url"])
    message_id = root_url.rsplit("/message-id/", 1)[-1]
    flat_url = f"https://www.postgresql.org/message-id/flat/{message_id}"
    payload, _ = request_bytes(flat_url)
    soup = BeautifulSoup(payload.decode("utf-8", errors="replace"), "html.parser")
    contents = soup.select(".message-content")
    responses: list[dict[str, str]] = []
    root_title = ""
    for index, content in enumerate(contents):
        header = content.find_previous("table", class_="message-header")
        metadata: dict[str, str] = {}
        if header is not None:
            for row in header.select("tr"):
                heading = row.find("th")
                cell = row.find("td")
                if heading and cell:
                    metadata[
                        heading.get_text(" ", strip=True).rstrip(":")
                    ] = cell.get_text(" ", strip=True)
        title = metadata.get("Subject", "")
        if index == 0:
            root_title = title
            continue
        responses.append(
            {
                "url": metadata.get("Message-ID", ""),
                "author": metadata.get("From", ""),
                "created_at": metadata.get("Date", ""),
                "title": title,
                "body": _text_from_html(content),
            }
        )
    text = joined_comments(
        formatted_comment(
            item["author"],
            item["created_at"],
            item["body"],
            kind="mail_reply",
        )
        for item in responses
    )
    return evidence_record(
        record,
        source_kind="postgresql_mail_thread",
        retrieval_status="ok" if text else "ok_zero_comments",
        comments=text,
        source_comment_count=len(responses),
        source_payload={
            "canonical_url": root_url,
            "flat_thread_url": flat_url,
            "root_title": root_title,
            "responses": responses,
        },
    )


def fossil_unescape(value: str) -> str:
    output: list[str] = []
    index = 0
    escapes = {
        "s": " ",
        "n": "\n",
        "r": "\r",
        "t": "\t",
        "\\": "\\",
    }
    while index < len(value):
        if value[index] == "\\" and index + 1 < len(value):
            output.append(escapes.get(value[index + 1], value[index + 1]))
            index += 2
            continue
        output.append(value[index])
        index += 1
    return "".join(output)


def sqlite_ticket_id(issue_url: str) -> str:
    parsed = urlparse(issue_url)
    query_name = parse_qs(parsed.query).get("name", [""])[0]
    path_name = parsed.path.rstrip("/").rsplit("/", 1)[-1]
    ticket = query_name or path_name
    if not re.fullmatch(r"[0-9a-fA-F]{10,64}", ticket):
        raise ValueError(f"Unsupported SQLite ticket URL: {issue_url}")
    return ticket.lower()


def sqlite_rss_items(ticket: str) -> list[dict[str, str]]:
    url = f"https://www2.sqlite.org/src/timeline.rss?tkt={ticket}"
    payload, _ = request_bytes(url)
    root = ET.fromstring(payload)
    items = []
    for item in root.findall("./channel/item"):
        link = (item.findtext("link") or "").strip()
        artifact = link.rstrip("/").rsplit("/", 1)[-1]
        items.append(
            {
                "title": (item.findtext("title") or "").strip(),
                "link": link,
                "artifact": artifact,
                "created_at": (item.findtext("pubDate") or "").strip(),
                "author": (
                    item.findtext("{http://purl.org/dc/elements/1.1/}creator")
                    or ""
                ).strip(),
            }
        )
    return items


def sqlite_raw_artifact(artifact: str) -> dict[str, Any]:
    url = (
        f"https://www2.sqlite.org/src/raw/{artifact}"
        f"?at={artifact}"
    )
    payload, _ = request_bytes(url)
    text = payload.decode("utf-8", errors="replace")
    fields: list[tuple[str, str]] = []
    timestamp = ""
    for line in text.splitlines():
        if line.startswith("D "):
            timestamp = line[2:].strip()
        elif line.startswith("J "):
            _, name, value = line.split(" ", 2)
            fields.append((name, fossil_unescape(value)))
    return {
        "artifact": artifact,
        "raw_url": url,
        "timestamp": timestamp,
        "fields": fields,
    }


def collect_sqlite(record: dict[str, str]) -> dict[str, Any]:
    ticket = sqlite_ticket_id(record["issue_url"])
    rss_items = sqlite_rss_items(ticket)
    artifacts = [sqlite_raw_artifact(item["artifact"]) for item in rss_items]
    artifacts.sort(key=lambda item: item["timestamp"])
    all_comments: list[dict[str, str]] = []
    for artifact in artifacts:
        author = ""
        for name, value in artifact["fields"]:
            if name == "login":
                author = value
        for name, value in artifact["fields"]:
            if name.lstrip("+") not in {"icomment", "comment"} or not value.strip():
                continue
            all_comments.append(
                {
                    "artifact": artifact["artifact"],
                    "author": author,
                    "created_at": artifact["timestamp"],
                    "body": value.strip(),
                }
            )
    # The first ticket comment is the original description already represented
    # in the dataset body. Later comment-bearing ticket changes are discussion.
    discussion = all_comments[1:] if all_comments else []
    text = joined_comments(
        formatted_comment(
            item["author"],
            item["created_at"],
            item["body"],
            kind="ticket_comment",
        )
        for item in discussion
    )
    return evidence_record(
        record,
        source_kind="sqlite_fossil_ticket",
        retrieval_status="ok" if text else "ok_zero_comments",
        comments=text,
        source_comment_count=len(discussion),
        source_payload={
            "ticket": ticket,
            "rss_url": (
                f"https://www2.sqlite.org/src/timeline.rss?tkt={ticket}"
            ),
            "artifacts": artifacts,
            "ticket_comments": all_comments,
        },
    )


def evidence_record(
    record: dict[str, str],
    *,
    source_kind: str,
    retrieval_status: str,
    comments: str,
    source_comment_count: int | None,
    source_payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "record_id": record["record_id"],
        "paper_id": PAPER_ID,
        "issue_url": record["issue_url"],
        "source_project": record["source_project"],
        "source_kind": source_kind,
        "evidence_version": RETRIEVAL_VERSION,
        "retrieved_at": now_utc(),
        "retrieval_status": retrieval_status,
        "source_comment_count": source_comment_count,
        "comments": comments,
        "source_payload": source_payload,
    }


def normalize_evidence_comments(
    evidence_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    normalized = []
    for original in evidence_rows:
        row = dict(original)
        if not str(row.get("comments") or "").strip():
            if row["retrieval_status"] == "ok_zero_comments":
                row["comments"] = PLACEHOLDER_NO_COMMENTS
            elif row["retrieval_status"] == "source_unavailable":
                row["comments"] = PLACEHOLDER_UNAVAILABLE
        normalized.append(row)
    return normalized


def source_kind(issue_url: str) -> str:
    host = urlparse(issue_url).netloc.lower()
    if host == "github.com":
        return "github"
    if host == "bugs.mysql.com":
        return "mysql"
    if host == "jira.mariadb.org":
        return "mariadb"
    if "postgresql.org" in host or host == "postgr.es":
        return "postgresql"
    if host in {"sqlite.org", "www.sqlite.org"}:
        return "sqlite"
    raise ValueError(f"Unknown TXBug source: {issue_url}")


def collect(root: Path, resume: bool = True) -> list[dict[str, Any]]:
    _, stage2 = read_csv(
        root / "Dataset/by_paper" / PAPER_ID / "stage2.csv"
    )
    existing = {
        row["record_id"]: row
        for row in read_jsonl(root / CACHE_PATH)
        if resume
    }
    mysql = source_csv_index(root / EXISTING_MYSQL)
    mariadb = source_csv_index(root / EXISTING_MARIADB)
    github_records = [row for row in stage2 if source_kind(row["issue_url"]) == "github"]
    token = github_token() if any(row["record_id"] not in existing for row in github_records) else ""
    results: list[dict[str, Any]] = []
    for index, record in enumerate(stage2, start=1):
        cached = existing.get(record["record_id"])
        if cached is not None:
            results.append(cached)
            continue
        kind = source_kind(record["issue_url"])
        try:
            if kind == "github":
                item = collect_github(record, token)
            elif kind == "mysql":
                item = collect_existing_tracker(
                    record,
                    source_kind="mysql_bugs",
                    source_row=mysql.get(record["issue_url"]),
                )
            elif kind == "mariadb":
                item = collect_existing_tracker(
                    record,
                    source_kind="mariadb_jira",
                    source_row=mariadb.get(record["issue_url"]),
                )
            elif kind == "postgresql":
                item = collect_postgresql(record)
            elif kind == "sqlite":
                item = collect_sqlite(record)
            else:
                raise AssertionError(kind)
        except Exception as error:
            item = evidence_record(
                record,
                source_kind=kind,
                retrieval_status="collection_failed",
                comments="",
                source_comment_count=None,
                source_payload={
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
        results.append(item)
        write_jsonl(root / CACHE_PATH, results)
        print(
            f"[{index:03d}/{len(stage2)}] {kind}: "
            f"{item['retrieval_status']}: {record['issue_url']}",
            flush=True,
        )
    write_jsonl(root / CACHE_PATH, results)
    return results


def secret_matches(value: Any, path: str = "$") -> list[dict[str, str]]:
    matches: list[dict[str, str]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            matches.extend(secret_matches(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            matches.extend(secret_matches(child, f"{path}[{index}]"))
    elif isinstance(value, str):
        for name, pattern in HIGH_CONFIDENCE_SECRET_PATTERNS.items():
            if pattern.search(value):
                matches.append({"pattern": name, "path": path})
    return matches


def immutable_projection(rows: list[dict[str, str]]) -> str:
    mutable = {"title", "body", "comments", "created_at", "updated_at", "state"}
    payload = [
        {key: value for key, value in row.items() if key not in mutable}
        for row in rows
    ]
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def update_dataset_stage(
    path: Path,
    evidence: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    fields, rows = read_csv(path)
    before_hash = immutable_projection(rows)
    target_rows = 0
    updated_rows = 0
    status_counts: dict[str, int] = {}
    for row in rows:
        if row["paper_id"] != PAPER_ID:
            continue
        item = evidence.get(row["record_id"])
        if item is None:
            continue
        target_rows += 1
        status = item["retrieval_status"]
        status_counts[status] = status_counts.get(status, 0) + 1
        comments = str(item.get("comments") or "").strip()
        replacement = (
            comments
            if comments
            else PLACEHOLDER_NO_COMMENTS
            if status == "ok_zero_comments"
            else PLACEHOLDER_UNAVAILABLE
        )
        if row.get("comments") != replacement:
            row["comments"] = replacement
            updated_rows += 1
    after_hash = immutable_projection(rows)
    if before_hash != after_hash:
        raise ValueError(f"Immutable columns changed while updating {path}")
    write_csv(path, fields, rows)
    return {
        "path": str(path).replace("\\", "/"),
        "row_count": len(rows),
        "target_rows": target_rows,
        "updated_rows": updated_rows,
        "immutable_projection_sha256_before": before_hash,
        "immutable_projection_sha256_after": after_hash,
        "status_counts": status_counts,
    }


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        if new in text:
            return
        raise ValueError(f"Expected text not found in {path}: {old!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8", newline="\n")


def update_metadata_csv(path: Path) -> None:
    fields, rows = read_csv(path)
    for row in rows:
        if row.get("paper_id") == PAPER_ID:
            row["source_platform"] = "mixed_issue_trackers"
            note = (
                "Discussion evidence reconstructed from GitHub Issues, MySQL "
                "Bugs, MariaDB JIRA, PostgreSQL mailing-list threads, and "
                "SQLite Fossil tickets; current_unversioned retrieval."
            )
            if note not in row.get("notes", ""):
                row["notes"] = (row.get("notes", "").rstrip(". ") + ". " + note).strip()
    write_csv(path, fields, rows)


def update_documentation(root: Path) -> None:
    replace_once(
        root / DATASET_README,
        "| `evidence/icse2022_dl_performance_evidence.jsonl` | Structured issue, retained-comment, and fixing-commit evidence for the ICSME 2022 DL-performance cohort |",
        "| `evidence/icse2022_dl_performance_evidence.jsonl` | Structured issue, retained-comment, and fixing-commit evidence for the ICSME 2022 DL-performance cohort |\n"
        "| `evidence/icse2024_transaction_bugs_evidence.jsonl` | Structured discussion evidence and retrieval status for the ICSE 2024 Transaction Bugs cohort |",
    )
    replace_once(
        root / DATA_DICTIONARY,
        "placeholder such as `no_comments_in_source` when comments were unavailable.",
        "placeholder `no_comments_in_source` when the source exposes zero comments, "
        "or `comments_unavailable_in_source` when authoritative discussion evidence "
        "could not be recovered.",
    )


def update_hash_manifest(root: Path) -> None:
    completed = subprocess.run(
        ["git", "show", f"origin/main:{HASH_MANIFEST.as_posix()}"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    entries: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        digest, path = line.split("  ", 1)
        entries[path] = digest
    for relative in CHECKSUM_TARGETS:
        path = root / relative
        if not path.exists():
            raise FileNotFoundError(path)
        entries[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    rendered = "".join(
        f"{entries[path]}  {path}\n" for path in sorted(entries)
    )
    (root / HASH_MANIFEST).write_text(
        rendered, encoding="utf-8", newline="\n"
    )


def build_audit(
    evidence_rows: list[dict[str, Any]],
    file_audits: list[dict[str, Any]],
) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    nonempty_counts: dict[str, int] = {}
    secret_findings: list[dict[str, str]] = []
    for row in evidence_rows:
        status = row["retrieval_status"]
        source = row["source_kind"]
        status_counts[status] = status_counts.get(status, 0) + 1
        source_counts[source] = source_counts.get(source, 0) + 1
        if str(row.get("comments") or "").strip():
            nonempty_counts[source] = nonempty_counts.get(source, 0) + 1
        for match in secret_matches(row):
            secret_findings.append(
                {"record_id": row["record_id"], **match}
            )
    return {
        "task": "icse2024_transaction_bugs_discussion_reconstruction",
        "generated_at": now_utc(),
        "paper_id": PAPER_ID,
        "evidence_version": RETRIEVAL_VERSION,
        "record_count": len(evidence_rows),
        "record_id_unique": len({row["record_id"] for row in evidence_rows})
        == len(evidence_rows),
        "issue_url_unique": len({row["issue_url"] for row in evidence_rows})
        == len(evidence_rows),
        "source_counts": source_counts,
        "status_counts": status_counts,
        "nonempty_discussion_counts": nonempty_counts,
        "nonempty_discussion_total": sum(nonempty_counts.values()),
        "high_confidence_secret_findings": secret_findings,
        "files": file_audits,
    }


def audit_markdown(audit: dict[str, Any]) -> str:
    sources = "\n".join(
        f"- `{source}`: {count} records, "
        f"{audit['nonempty_discussion_counts'].get(source, 0)} with discussion"
        for source, count in sorted(audit["source_counts"].items())
    )
    statuses = "\n".join(
        f"- `{status}`: {count}"
        for status, count in sorted(audit["status_counts"].items())
    )
    return f"""# ICSE 2024 Transaction Bugs discussion reconstruction

This reconstruction restores public discussion evidence for the 140-record
Stage 2/3 cohort while preserving record identity, stage membership, and all
gold labels.

## Version semantics

The author artifact does not contain a frozen copy of every source discussion.
Recovered source text is therefore classified as `{RETRIEVAL_VERSION}`. It is
currently visible public evidence, not a claim that every comment existed at
the paper's original collection cutoff.

## Source coverage

{sources}

## Retrieval status

{statuses}

`no_comments_in_source` means the authoritative source or retained source
reconstruction exposes zero comments. `comments_unavailable_in_source` means
the discussion could not be recovered and must not be interpreted as zero.

## Integrity

- Evidence records: {audit['record_count']}
- Unique record IDs: {str(audit['record_id_unique']).lower()}
- Unique issue URLs: {str(audit['issue_url_unique']).lower()}
- Non-empty reconstructed discussions: {audit['nonempty_discussion_total']}
- High-confidence secret findings: {len(audit['high_confidence_secret_findings'])}
- Immutable identity and label projections preserved for every rewritten file
"""


def apply(root: Path, evidence_rows: list[dict[str, Any]]) -> dict[str, Any]:
    evidence_rows = normalize_evidence_comments(evidence_rows)
    if len(evidence_rows) != 140:
        raise ValueError(f"Expected 140 evidence rows, found {len(evidence_rows)}")
    evidence = {row["record_id"]: row for row in evidence_rows}
    if len(evidence) != 140:
        raise ValueError("Evidence record IDs are not unique")
    secret_findings = [
        finding
        for row in evidence_rows
        for finding in secret_matches(row)
    ]
    if secret_findings:
        raise ValueError(
            "Refusing to apply evidence with high-confidence secret matches: "
            f"{secret_findings[:5]}"
        )

    file_audits = []
    for stage in STAGES:
        file_audits.append(
            update_dataset_stage(
                root / "Dataset/by_paper" / PAPER_ID / f"{stage}.csv",
                evidence,
            )
        )
        file_audits.append(
            update_dataset_stage(root / "Dataset" / f"{stage}.csv", evidence)
        )

    write_jsonl(root / SIDECAR_PATH, evidence_rows)
    update_metadata_csv(root / METADATA_CSV)
    update_documentation(root)

    audit = build_audit(evidence_rows, file_audits)
    (root / AUDIT_PATH).parent.mkdir(parents=True, exist_ok=True)
    (root / AUDIT_PATH).write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (root / README_PATH).write_text(
        audit_markdown(audit), encoding="utf-8", newline="\n"
    )
    update_hash_manifest(root)
    return audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reconstruct ICSE 2024 Transaction Bugs discussions."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--collect",
        action="store_true",
        help="Collect or resume source evidence into the ignored cache.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore the existing cache during collection.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the complete cached evidence to tracked datasets.",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if not args.collect and not args.apply:
        parser.error("select --collect and/or --apply")
    evidence_rows = (
        collect(root, resume=not args.no_resume)
        if args.collect
        else read_jsonl(root / CACHE_PATH)
    )
    if args.apply:
        audit = apply(root, evidence_rows)
        print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
