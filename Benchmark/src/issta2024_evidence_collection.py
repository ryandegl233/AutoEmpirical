from __future__ import annotations

import csv
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

from Benchmark.src.issta2024_bugs_in_pods_baseline import (
    ACCEPTED_FAULT,
    REJECTED_CANDIDATE,
    SAMPLE_FIELDS,
    build_issta2024_taxonomy,
    build_stage2_system_prompt,
    build_stage2_user_prompt,
    build_stage3_system_prompt,
    build_stage3_user_prompt,
)


NOT_AVAILABLE = "not_available_in_source"
GITHUB_API_ROOT = "https://api.github.com"
NO_COMMENTS = "no_comments_in_source"
ASE_DATASET_FIELDS = [
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
    "bug_type",
    "component",
    "sub_component",
    "trigger_condition",
    "consequence",
    "fix_type",
    "severity_or_impact",
    "original_label_json",
    "source_file",
    "source_sheet",
    "source_row_index",
]
AMBIGUOUS_RECORD_IDS = {
    "issta2024_bugs_in_pods_understanding_bugs:524debb095ec57a2",
    "issta2024_bugs_in_pods_understanding_bugs:5d46002234cb26af",
}
Transport = Callable[
    [str, dict[str, str], float],
    tuple[int, dict[str, str], bytes],
]


def parse_github_commit_url(url: str) -> tuple[str, str, str]:
    parsed = urlparse(url.strip())
    parts = [part for part in parsed.path.split("/") if part]
    if (
        parsed.scheme != "https"
        or parsed.netloc.lower() != "github.com"
        or len(parts) != 4
        or parts[2] != "commit"
        or not parts[3]
    ):
        raise ValueError(f"not a canonical GitHub commit URL: {url}")
    return parts[0], parts[1], parts[3]


def split_commit_message(message: str) -> tuple[str, str]:
    lines = [line.strip() for line in message.splitlines()]
    nonempty = [line for line in lines if line]
    if not nonempty:
        return NOT_AVAILABLE, NOT_AVAILABLE
    title = nonempty[0]
    body = "\n".join(nonempty[1:]).strip() or NOT_AVAILABLE
    return title, body


def normalize_discussion(items: list[dict[str, object]]) -> list[str]:
    ordered = sorted(
        items,
        key=lambda item: (
            str(item.get("created_at") or ""),
            str(item.get("html_url") or ""),
        ),
    )
    return [
        body
        for item in ordered
        if (body := str(item.get("body") or "").strip())
    ]


def _urllib_transport(
    url: str,
    headers: dict[str, str],
    timeout_seconds: float,
) -> tuple[int, dict[str, str], bytes]:
    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return (
                response.status,
                {key.lower(): value for key, value in response.headers.items()},
                response.read(),
            )
    except urllib.error.HTTPError as exc:
        return (
            exc.code,
            {key.lower(): value for key, value in exc.headers.items()},
            exc.read(),
        )


class GitHubClient:
    def __init__(
        self,
        *,
        token: str,
        transport: Transport = _urllib_transport,
        timeout_seconds: float = 30,
        max_retries: int = 3,
        retry_delay_seconds: float = 1,
    ) -> None:
        self.token = token
        self.transport = transport
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds

    def _request_json(
        self, url: str
    ) -> tuple[object | None, str, dict[str, str]]:
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self.token}",
            "User-Agent": "AutoEmpirical-ISSTA2024-Evidence-Collector",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        for attempt in range(self.max_retries + 1):
            try:
                status, response_headers, raw = self.transport(
                    url,
                    headers,
                    self.timeout_seconds,
                )
            except (OSError, TimeoutError, urllib.error.URLError):
                if attempt < self.max_retries:
                    if self.retry_delay_seconds:
                        time.sleep(self.retry_delay_seconds)
                    continue
                return None, "fetch_failed", {}

            normalized_headers = {
                str(key).lower(): str(value)
                for key, value in response_headers.items()
            }
            if status == 200:
                try:
                    return (
                        json.loads(raw.decode("utf-8")),
                        "available",
                        normalized_headers,
                    )
                except (UnicodeDecodeError, json.JSONDecodeError):
                    return None, "fetch_failed", normalized_headers
            if status == 404:
                return None, "deleted", normalized_headers
            if status == 429 or (
                status == 403
                and normalized_headers.get("x-ratelimit-remaining") == "0"
            ):
                return None, "rate_limited", normalized_headers
            if status in {401, 403}:
                return None, "permission_denied", normalized_headers
            if status >= 500 and attempt < self.max_retries:
                if self.retry_delay_seconds:
                    time.sleep(self.retry_delay_seconds)
                continue
            return None, "fetch_failed", normalized_headers
        return None, "fetch_failed", {}

    def get_json(self, url: str) -> tuple[object | None, str]:
        payload, status, _ = self._request_json(url)
        return payload, status

    def get_paginated_json(self, url: str) -> tuple[object | None, str]:
        rows: list[object] = []
        next_url: str | None = url
        while next_url:
            payload, status, headers = self._request_json(next_url)
            if status != "available":
                return None, status
            if not isinstance(payload, list):
                return payload, status
            rows.extend(payload)
            next_url = None
            for part in headers.get("link", "").split(","):
                match = re.match(r'\s*<([^>]+)>;\s*rel="([^"]+)"', part)
                if match and match.group(2) == "next":
                    next_url = match.group(1)
                    break
        return rows, "available"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _endpoint(path: str) -> str:
    return GITHUB_API_ROOT + path


def _fetch_and_cache(
    client: GitHubClient,
    url: str,
    path: Path,
    *,
    paginated: bool = False,
) -> tuple[object | None, str]:
    payload, status = (
        client.get_paginated_json(url)
        if paginated
        else client.get_json(url)
    )
    _write_json(
        path,
        {
            "url": url,
            "status": status,
            "payload": payload,
        },
    )
    return payload, status


def _issue_references(texts: list[str]) -> set[int]:
    references: set[int] = set()
    for text in texts:
        references.update(
            int(value)
            for value in re.findall(r"(?<![\w/])#(\d+)\b", text or "")
        )
    return references


def _slim_pull_request(payload: dict[str, object]) -> dict[str, object]:
    return {
        "number": payload.get("number"),
        "html_url": payload.get("html_url"),
        "title": payload.get("title"),
        "body": payload.get("body"),
        "created_at": payload.get("created_at"),
        "updated_at": payload.get("updated_at"),
    }


def _slim_issue(payload: dict[str, object]) -> dict[str, object]:
    return {
        "number": payload.get("number"),
        "html_url": payload.get("html_url"),
        "title": payload.get("title"),
        "body": payload.get("body"),
        "created_at": payload.get("created_at"),
        "updated_at": payload.get("updated_at"),
    }


def collect_github_record(
    record: dict[str, str],
    client: GitHubClient,
    *,
    cache_dir: Path,
    resume: bool,
) -> dict[str, object]:
    owner, repo, sha = parse_github_commit_url(record["issue_url"])
    record_cache = cache_dir / sha
    aggregate_path = record_cache / "evidence.json"
    if resume and aggregate_path.exists():
        cached = json.loads(aggregate_path.read_text(encoding="utf-8"))
        if cached.get("complete") is True:
            return cached

    statuses: dict[str, str] = {}
    commit_url = _endpoint(f"/repos/{owner}/{repo}/commits/{sha}")
    commit_payload, commit_status = _fetch_and_cache(
        client,
        commit_url,
        record_cache / "commit.json",
    )
    statuses["commit"] = commit_status
    if not isinstance(commit_payload, dict):
        evidence: dict[str, object] = {
            "record_id": record["record_id"],
            "repository": f"{owner}/{repo}",
            "sha": sha,
            "complete": False,
            "commit": {},
            "changed_files": [],
            "patches": [],
            "pull_requests": [],
            "issues": [],
            "discussion": [],
            "statuses": statuses,
        }
        _write_json(aggregate_path, evidence)
        return evidence

    commit_data = commit_payload.get("commit")
    commit_data = commit_data if isinstance(commit_data, dict) else {}
    files = commit_payload.get("files")
    file_rows = files if isinstance(files, list) else []
    changed_files = [
        str(item.get("filename"))
        for item in file_rows
        if isinstance(item, dict) and item.get("filename")
    ]
    patches = [
        {
            "filename": item.get("filename"),
            "patch": item.get("patch"),
        }
        for item in file_rows
        if isinstance(item, dict) and item.get("patch")
    ]
    commit = {
        "sha": commit_payload.get("sha") or sha,
        "html_url": commit_payload.get("html_url") or record["issue_url"],
        "message": commit_data.get("message") or "",
        "author_date": (
            commit_data.get("author", {}).get("date")
            if isinstance(commit_data.get("author"), dict)
            else None
        ),
        "committer_date": (
            commit_data.get("committer", {}).get("date")
            if isinstance(commit_data.get("committer"), dict)
            else None
        ),
    }

    pulls_url = _endpoint(
        f"/repos/{owner}/{repo}/commits/{sha}/pulls?per_page=100"
    )
    pulls_payload, pulls_status = _fetch_and_cache(
        client,
        pulls_url,
        record_cache / "pull_requests.json",
        paginated=True,
    )
    pulls = (
        [_slim_pull_request(item) for item in pulls_payload if isinstance(item, dict)]
        if isinstance(pulls_payload, list)
        else []
    )
    statuses["pull_requests"] = (
        "not_linked" if pulls_status == "available" and not pulls else pulls_status
    )

    discussion_items: list[dict[str, object]] = []
    reference_texts = [str(commit.get("message") or "")]
    pr_numbers: set[int] = set()
    for pull in pulls:
        number = pull.get("number")
        if not isinstance(number, int):
            continue
        pr_numbers.add(number)
        reference_texts.append(str(pull.get("body") or ""))
        if pull.get("body"):
            discussion_items.append(
                {
                    "body": pull["body"],
                    "created_at": pull.get("created_at"),
                    "html_url": pull.get("html_url"),
                }
            )
        for kind, url, filename in (
            (
                "issue_comments",
                _endpoint(
                    f"/repos/{owner}/{repo}/issues/{number}/comments?per_page=100"
                ),
                f"pr_{number}_issue_comments.json",
            ),
            (
                "review_comments",
                _endpoint(
                    f"/repos/{owner}/{repo}/pulls/{number}/comments?per_page=100"
                ),
                f"pr_{number}_review_comments.json",
            ),
        ):
            payload, status = _fetch_and_cache(
                client,
                url,
                record_cache / filename,
                paginated=True,
            )
            statuses[f"pr_{number}_{kind}"] = (
                "empty_in_source"
                if status == "available" and payload == []
                else status
            )
            if isinstance(payload, list):
                discussion_items.extend(
                    item for item in payload if isinstance(item, dict)
                )

    issues: list[dict[str, object]] = []
    issue_numbers = _issue_references(reference_texts) - pr_numbers
    issue_fetch_statuses: list[str] = []
    for number in sorted(issue_numbers):
        issue_url = _endpoint(f"/repos/{owner}/{repo}/issues/{number}")
        issue_payload, issue_status = _fetch_and_cache(
            client,
            issue_url,
            record_cache / f"issue_{number}.json",
        )
        issue_fetch_statuses.append(issue_status)
        if not isinstance(issue_payload, dict) or "pull_request" in issue_payload:
            continue
        issue = _slim_issue(issue_payload)
        issues.append(issue)
        if issue.get("body"):
            discussion_items.append(
                {
                    "body": issue["body"],
                    "created_at": issue.get("created_at"),
                    "html_url": issue.get("html_url"),
                }
            )
        comments_url = _endpoint(
            f"/repos/{owner}/{repo}/issues/{number}/comments?per_page=100"
        )
        comments_payload, comments_status = _fetch_and_cache(
            client,
            comments_url,
            record_cache / f"issue_{number}_comments.json",
            paginated=True,
        )
        statuses[f"issue_{number}_comments"] = (
            "empty_in_source"
            if comments_status == "available" and comments_payload == []
            else comments_status
        )
        if isinstance(comments_payload, list):
            discussion_items.extend(
                item for item in comments_payload if isinstance(item, dict)
            )

    if not issue_numbers:
        statuses["issues"] = "not_linked"
    elif issues:
        statuses["issues"] = "available"
    else:
        statuses["issues"] = (
            issue_fetch_statuses[0] if issue_fetch_statuses else "fetch_failed"
        )
    discussion = normalize_discussion(discussion_items)
    statuses["discussion"] = "available" if discussion else "empty_in_source"
    complete = not any(
        status in {"rate_limited", "fetch_failed"}
        for status in statuses.values()
    )
    evidence = {
        "record_id": record["record_id"],
        "repository": f"{owner}/{repo}",
        "sha": sha,
        "complete": complete,
        "commit": commit,
        "changed_files": changed_files,
        "patches": patches,
        "pull_requests": pulls,
        "issues": issues,
        "discussion": discussion,
        "statuses": statuses,
    }
    _write_json(aggregate_path, evidence)
    return evidence


def _parse_json_object(raw: str) -> dict[str, object]:
    try:
        value = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _latest_timestamp(evidence: dict[str, object], fallback: str) -> str:
    timestamps = [fallback] if fallback else []
    for field in ("pull_requests", "issues"):
        values = evidence.get(field)
        if not isinstance(values, list):
            continue
        for value in values:
            if isinstance(value, dict) and value.get("updated_at"):
                timestamps.append(str(value["updated_at"]))
    commit = evidence.get("commit")
    if isinstance(commit, dict):
        for field in ("committer_date", "author_date"):
            if commit.get(field):
                timestamps.append(str(commit[field]))
    return max(timestamps) if timestamps else NOT_AVAILABLE


def build_enriched_row(
    cohort_row: dict[str, str],
    stage1_row: dict[str, str],
    evidence: dict[str, object],
) -> dict[str, str]:
    row = {
        field: str(stage1_row.get(field, "") or "")
        for field in ASE_DATASET_FIELDS
    }
    commit = evidence.get("commit")
    commit = commit if isinstance(commit, dict) else {}
    source_message = str(
        commit.get("message")
        or stage1_row.get("body")
        or stage1_row.get("title")
        or ""
    )
    title, body = split_commit_message(source_message)
    discussion = evidence.get("discussion")
    comments = (
        [str(value) for value in discussion if str(value).strip()]
        if isinstance(discussion, list)
        else []
    )
    commit_date = str(
        commit.get("committer_date")
        or commit.get("author_date")
        or stage1_row.get("created_at")
        or NOT_AVAILABLE
    )
    row.update(
        {
            "title": title,
            "body": body,
            "comments": (
                json.dumps(comments, ensure_ascii=False)
                if comments
                else NO_COMMENTS
            ),
            "created_at": commit_date,
            "updated_at": _latest_timestamp(
                evidence,
                str(stage1_row.get("updated_at") or commit_date),
            ),
            "state": "committed",
            "symptom": (
                str(cohort_row.get("symptom") or "")
                if cohort_row.get("decision") == ACCEPTED_FAULT
                else ""
            ),
            "root_cause": (
                str(cohort_row.get("root_cause") or "")
                if cohort_row.get("decision") == ACCEPTED_FAULT
                else ""
            ),
        }
    )
    provenance = _parse_json_object(stage1_row.get("original_label_json", ""))
    changed_files = evidence.get("changed_files")
    if not isinstance(changed_files, list):
        changed_files = provenance.get("changed_files", [])
    pull_requests = evidence.get("pull_requests")
    issues = evidence.get("issues")
    provenance.update(
        {
            "changed_files": changed_files,
            "sha": evidence.get("sha"),
            "evidence_repository": evidence.get("repository"),
            "linked_pull_requests": [
                value["html_url"]
                for value in (pull_requests if isinstance(pull_requests, list) else [])
                if isinstance(value, dict) and value.get("html_url")
            ],
            "linked_issues": [
                value["html_url"]
                for value in (issues if isinstance(issues, list) else [])
                if isinstance(value, dict) and value.get("html_url")
            ],
            "evidence_statuses": evidence.get("statuses", {}),
            "evidence_complete": evidence.get("complete") is True,
        }
    )
    row["original_label_json"] = json.dumps(provenance, ensure_ascii=False)
    return row


def _read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return [
            {
                str(key): "" if value is None else str(value)
                for key, value in row.items()
            }
            for row in csv.DictReader(handle)
        ]


def _write_csv_rows(
    path: Path,
    rows: list[dict[str, str]],
    fieldnames: list[str],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def _sample_row(
    cohort_row: dict[str, str],
    enriched_row: dict[str, str],
) -> dict[str, str]:
    provenance = _parse_json_object(enriched_row["original_label_json"])
    return {
        "record_id": enriched_row["record_id"],
        "paper_id": enriched_row["paper_id"],
        "source_project": enriched_row["source_project"],
        "issue_url": enriched_row["issue_url"],
        "title": enriched_row["title"],
        "body": enriched_row["body"],
        "comments": enriched_row["comments"],
        "state": enriched_row["state"],
        "created_at": enriched_row["created_at"],
        "changed_files": json.dumps(
            provenance.get("changed_files", []),
            ensure_ascii=False,
        ),
        "decision": cohort_row["decision"],
        "symptom": enriched_row["symptom"],
        "root_cause": enriched_row["root_cause"],
    }


def _write_prompt_rows(
    path: Path,
    rows: list[dict[str, str]],
    *,
    stage: str,
) -> Path:
    taxonomy = build_issta2024_taxonomy()
    system_prompt = (
        build_stage2_system_prompt()
        if stage == "stage2"
        else build_stage3_system_prompt(taxonomy)
    )
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            payload = {
                "record_id": row["record_id"],
                "paper_id": row["paper_id"],
                "issue_url": row["issue_url"],
                "ground_truth": (
                    row["decision"]
                    if stage == "stage2"
                    else {
                        "symptom": row["symptom"],
                        "root_cause": row["root_cause"],
                    }
                ),
                "system_prompt": system_prompt,
                "user_prompt": (
                    build_stage2_user_prompt(row)
                    if stage == "stage2"
                    else build_stage3_user_prompt(row)
                ),
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return path


def collect_issta2024_evidence(
    *,
    cohort_path: str | Path,
    stage1_path: str | Path,
    output_dir: str | Path,
    client: GitHubClient,
    resume: bool,
    limit: int | None,
) -> dict[str, Path]:
    cohort = _read_csv_rows(cohort_path)
    stage1_rows = {
        row["record_id"]: row
        for row in _read_csv_rows(stage1_path)
    }
    excluded = [
        row["record_id"]
        for row in cohort
        if row["record_id"] in AMBIGUOUS_RECORD_IDS
    ]
    selected = [
        row
        for row in cohort
        if row["record_id"] not in AMBIGUOUS_RECORD_IDS
    ]
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        selected = selected[:limit]

    output = Path(output_dir)
    raw_dir = output / "raw"
    enriched_rows: list[dict[str, str]] = []
    stage2_rows: list[dict[str, str]] = []
    failures: list[dict[str, object]] = []
    status_counts: dict[str, dict[str, int]] = {}
    coverage = {
        "commit": 0,
        "pull_request": 0,
        "issue": 0,
        "discussion": 0,
        "body": 0,
        "changed_files": 0,
        "title_body_distinct": 0,
    }
    for cohort_row in selected:
        record_id = cohort_row["record_id"]
        stage1_row = stage1_rows.get(record_id)
        if stage1_row is None:
            raise ValueError(f"cohort record absent from Stage 1: {record_id}")
        evidence = collect_github_record(
            stage1_row,
            client,
            cache_dir=raw_dir,
            resume=resume,
        )
        enriched = build_enriched_row(cohort_row, stage1_row, evidence)
        enriched_rows.append(enriched)
        stage2_rows.append(_sample_row(cohort_row, enriched))
        statuses = evidence.get("statuses")
        if isinstance(statuses, dict) and statuses.get("commit") == "available":
            coverage["commit"] += 1
        if evidence.get("pull_requests"):
            coverage["pull_request"] += 1
        if evidence.get("issues"):
            coverage["issue"] += 1
        if evidence.get("discussion"):
            coverage["discussion"] += 1
        if enriched["body"] != NOT_AVAILABLE:
            coverage["body"] += 1
        if evidence.get("changed_files"):
            coverage["changed_files"] += 1
        if enriched["title"] != enriched["body"]:
            coverage["title_body_distinct"] += 1
        if isinstance(statuses, dict):
            for field, status in statuses.items():
                field_counts = status_counts.setdefault(str(field), {})
                key = str(status)
                field_counts[key] = field_counts.get(key, 0) + 1
        if evidence.get("complete") is not True:
            failures.append(
                {
                    "record_id": record_id,
                    "issue_url": stage1_row["issue_url"],
                    "statuses": evidence.get("statuses", {}),
                }
            )

    stage3_rows = [
        row for row in stage2_rows if row["decision"] == ACCEPTED_FAULT
    ]
    paths = {
        "records": output / "issta2024_evidence_enriched_records.csv",
        "stage2_sample": output
        / "issta2024_stage2_evidence_enriched_sample.csv",
        "stage2_prompts": output
        / "issta2024_stage2_evidence_enriched_prompts.jsonl",
        "stage3_sample": output
        / "issta2024_stage3_evidence_enriched_sample.csv",
        "stage3_prompts": output
        / "issta2024_stage3_evidence_enriched_prompts.jsonl",
        "manifest": output / "issta2024_evidence_collection_manifest.json",
        "failures": output / "issta2024_evidence_collection_failures.jsonl",
    }
    _write_csv_rows(paths["records"], enriched_rows, ASE_DATASET_FIELDS)
    _write_csv_rows(paths["stage2_sample"], stage2_rows, SAMPLE_FIELDS)
    _write_prompt_rows(paths["stage2_prompts"], stage2_rows, stage="stage2")
    _write_csv_rows(paths["stage3_sample"], stage3_rows, SAMPLE_FIELDS)
    _write_prompt_rows(paths["stage3_prompts"], stage3_rows, stage="stage3")
    with paths["failures"].open("w", encoding="utf-8") as handle:
        for failure in failures:
            handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
    counts = {
        "input": len(cohort),
        "excluded_ambiguous": len(excluded),
        "processed": len(stage2_rows),
        "stage2_positive": sum(
            row["decision"] == ACCEPTED_FAULT for row in stage2_rows
        ),
        "stage2_negative": sum(
            row["decision"] == REJECTED_CANDIDATE for row in stage2_rows
        ),
        "stage3": len(stage3_rows),
        "incomplete": len(failures),
    }
    _write_json(
        paths["manifest"],
        {
            "task": "issta2024_ase_format_evidence_collection",
            "cohort_path": str(cohort_path),
            "stage1_path": str(stage1_path),
            "counts": counts,
            "excluded_record_ids": excluded,
            "configured_excluded_record_ids": sorted(AMBIGUOUS_RECORD_IDS),
            "coverage": coverage,
            "status_counts": status_counts,
            "model_input_fields": [
                "source_project",
                "issue_url",
                "title",
                "body",
                "comments",
                "state",
                "created_at",
                "changed_files",
            ],
        },
    )
    return paths
