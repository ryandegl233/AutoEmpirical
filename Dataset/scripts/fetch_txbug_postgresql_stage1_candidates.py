from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime
import hashlib
import html
import json
from pathlib import Path
import re
import time
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urljoin
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup


PAPER_ID = "icse2024_understanding_transaction_bugs_in_database"
BASE_URL = "https://www.postgresql.org"
LIST_BASE = f"{BASE_URL}/list/pgsql-bugs"
START_YEAR = 2018
END_YEAR = 2022
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
REQUEST_DELAY_SECONDS = 0.02
MAX_WORKERS = 8
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
    "message_id",
    "author",
    "list_name",
    "original_label_json",
]


def _request_text(url: str, retries: int = 4) -> str:
    headers = {"User-Agent": "AutoEmpirical-txbug-postgresql-stage1-fetch"}
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urlopen(Request(url, headers=headers), timeout=90) as response:
                payload = response.read().decode("utf-8", errors="replace")
            time.sleep(REQUEST_DELAY_SECONDS)
            return payload
        except HTTPError as error:
            last_error = error
            if error.code not in {429, 500, 502, 503, 504} or attempt == retries:
                raise
            time.sleep(2**attempt)
        except URLError as error:
            last_error = error
            if attempt == retries:
                raise
            time.sleep(2**attempt)
    raise RuntimeError(f"PostgreSQL request failed: {last_error}") from last_error


def _stable_record_id(issue_url: str) -> str:
    digest = hashlib.sha256(issue_url.encode("utf-8")).hexdigest()[:16]
    return f"{PAPER_ID}:{digest}"


def _canonical_issue_url(issue_url: str) -> str:
    issue_url = unquote(issue_url)
    return issue_url.replace("https://postgr.es/m/", f"{BASE_URL}/message-id/")


def _normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _text_from_html(fragment) -> str:
    if fragment is None:
        return ""
    text = fragment.get_text("\n", strip=True)
    return html.unescape(re.sub(r"\n{3,}", "\n\n", text)).strip()


def _matched_keywords(title: str, body: str) -> set[str]:
    haystack = f"{title}\n{body}".lower()
    return {keyword for keyword in KEYWORDS if keyword.lower() in haystack}


def _iter_month_archive_entries(year: int, month: int) -> Iterable[dict[str, str]]:
    url = f"{LIST_BASE}/{year}-{month:02d}/"
    soup = BeautifulSoup(_request_text(url), "html.parser")
    current_date = ""
    for element in soup.select("h2, table.thread-list tr"):
        if element.name == "h2":
            current_date = _normalize_space(element.get_text(" ", strip=True))
            continue
        link = element.find("a", href=re.compile(r"/message-id/"))
        if not link:
            continue
        title = _normalize_space(link.get_text(" ", strip=True))
        if not re.match(r"^BUG #\d+:", title):
            continue
        cells = element.find_all("td")
        author = _normalize_space(cells[0].get_text(" ", strip=True)) if cells else ""
        time_text = _normalize_space(cells[1].get_text(" ", strip=True)) if len(cells) > 1 else ""
        created_at = ""
        if current_date and time_text:
            try:
                parsed = datetime.strptime(
                    f"{current_date} {time_text}", "%B %d, %Y %H:%M"
                )
                created_at = parsed.isoformat()
            except ValueError:
                created_at = f"{current_date} {time_text}"
        yield {
            "title": title,
            "author": author,
            "created_at": created_at,
            "issue_url": _canonical_issue_url(urljoin(BASE_URL, link["href"])),
        }


def _message_id_from_url(url: str) -> str:
    return _canonical_issue_url(url).rsplit("/message-id/", 1)[-1]


def _parse_message_page(url: str) -> dict:
    soup = BeautifulSoup(_request_text(url), "html.parser")
    title = _normalize_space((soup.find("h1") or soup.find("title")).get_text(" ", strip=True))
    title = re.sub(r"^PostgreSQL:\s*", "", title)
    content = _text_from_html(soup.select_one(".message-content"))
    metadata: dict[str, str] = {}
    for row in soup.select("table.message-header tr"):
        heading = row.find("th")
        cell = row.find("td")
        if heading and cell:
            metadata[_normalize_space(heading.get_text(" ", strip=True)).rstrip(":")] = (
                _normalize_space(cell.get_text(" ", strip=True))
            )
    response_links = []
    for link in soup.select("li.message-responses a[href*='/message-id/']"):
        href = urljoin(BASE_URL, link.get("href") or "")
        if href != url:
            response_links.append(href)
    return {
        "title": title,
        "body": content,
        "metadata": metadata,
        "response_links": sorted(set(response_links)),
    }


def _response_comment(url: str) -> str:
    parsed = _parse_message_page(url)
    metadata = parsed["metadata"]
    prefix = " | ".join(
        part
        for part in [
            metadata.get("Date", ""),
            metadata.get("From", ""),
            parsed["title"],
        ]
        if part
    )
    body = parsed["body"]
    return f"[{prefix}]\n{body}" if prefix else body


def _parse_message_worker(item: tuple[str, dict[str, str]]) -> tuple[str, dict[str, str], dict]:
    issue_url, entry = item
    return issue_url, entry, _parse_message_page(issue_url)


def _response_worker(url: str) -> tuple[str, str]:
    return url, _response_comment(url)


def _final_postgresql_urls(root: Path) -> set[str]:
    stage2 = root / "Dataset" / "stage2.csv"
    if not stage2.exists():
        return set()
    urls: set[str] = set()
    with stage2.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("paper_id") != PAPER_ID:
                continue
            issue_url = row.get("issue_url", "")
            if "postgresql.org/message-id" in issue_url or "postgr.es/m/" in issue_url:
                if "postgr.es/m/" in issue_url:
                    issue_url = issue_url.replace("https://postgr.es/m/", f"{BASE_URL}/message-id/")
                urls.add(issue_url)
    return urls


def _write_coverage_audit(output_dir: Path, candidate_urls: set[str], final_urls: set[str]) -> None:
    with (output_dir / "postgresql_final_coverage_audit.csv").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
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


def _write_readme(output_dir: Path, rows: list[dict[str, str]], final_urls: set[str]) -> None:
    candidate_urls = {row["issue_url"] for row in rows}
    missing = sorted(final_urls - candidate_urls)
    lines = [
        "# TXBug PostgreSQL Stage 1 reconstruction",
        "",
        "This directory contains a reconstructed Stage 1 candidate set for PostgreSQL bug-report emails.",
        "",
        "## Method",
        "",
        f"- Source: `{LIST_BASE}` monthly `pgsql-bugs` archives.",
        f"- Archive window: `{START_YEAR}-01` through `{END_YEAR}-12`.",
        "- Initial candidate universe: original `BUG #...` reports, excluding `Re:` replies.",
        f"- Keyword filter over title + message body: {', '.join('`' + k + '`' for k in KEYWORDS)}.",
        "- `body` is the original bug-report email content. `comments` concatenates response emails linked by the archive page.",
        "",
        "## Output",
        "",
        f"- `txbug_postgresql_candidates.csv`: {len(rows)} unique candidate rows.",
        "- `txbug_postgresql_candidates.raw.jsonl`: raw parsed message payloads.",
        "- `fetch_manifest.json`: fetch parameters and counts.",
        "- `postgresql_final_coverage_audit.csv`: coverage of local final TXBug PostgreSQL rows.",
        "",
        "## Coverage against local final TXBug PostgreSQL rows",
        "",
        f"- Covered: {len(final_urls) - len(missing)}/{len(final_urls)}",
        f"- Missing: {len(missing)}",
    ]
    for issue_url in missing:
        lines.append(f"  - {issue_url}")
    (output_dir / "README.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )


def fetch_candidates(root: Path, write: bool = True) -> dict[str, int]:
    root = root.resolve()
    output_dir = root / "reports" / "txbug_stage1_reconstruction" / "postgresql"
    if write:
        output_dir.mkdir(parents=True, exist_ok=True)

    archive_entries: dict[str, dict[str, str]] = {}
    for year in range(START_YEAR, END_YEAR + 1):
        for month in range(1, 13):
            for entry in _iter_month_archive_entries(year, month):
                archive_entries[entry["issue_url"]] = entry

    parsed_by_url: dict[str, tuple[dict[str, str], dict]] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(_parse_message_worker, item)
            for item in sorted(archive_entries.items())
        ]
        for future in as_completed(futures):
            issue_url, entry, parsed = future.result()
            parsed_by_url[issue_url] = (entry, parsed)

    candidate_payloads: dict[str, dict] = {}
    response_urls: set[str] = set()
    for issue_url, (entry, parsed) in sorted(parsed_by_url.items()):
        title = entry["title"] or parsed["title"]
        body = parsed["body"]
        keywords = _matched_keywords(title, body)
        if not keywords:
            continue
        candidate_payloads[issue_url] = {
            "entry": entry,
            "parsed": parsed,
            "keywords": keywords,
        }
        response_urls.update(parsed["response_links"])

    comments_by_response_url: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(_response_worker, url) for url in sorted(response_urls)]
        for future in as_completed(futures):
            url, comment = future.result()
            comments_by_response_url[url] = comment

    parsed_messages: dict[str, dict] = {}
    rows: list[dict[str, str]] = []
    for issue_url, payload in sorted(candidate_payloads.items()):
        entry = payload["entry"]
        parsed = payload["parsed"]
        keywords = payload["keywords"]
        title = entry["title"] or parsed["title"]
        body = parsed["body"]
        response_comments = [
            comments_by_response_url[url]
            for url in parsed["response_links"]
            if comments_by_response_url.get(url, "").strip()
        ]
        comments = "\n\n---\n\n".join(comment for comment in response_comments if comment.strip())
        message_id = _message_id_from_url(issue_url)
        number_match = re.search(r"BUG #(\d+):", title)
        original = {
            "stage": "stage1",
            "source": "postgresql_pgsql_bugs_archive",
            "paper_rule": (
                "TXBug PostgreSQL candidate reconstruction using 2018-01..2022-12 "
                "pgsql-bugs original BUG reports and transaction-related keyword filtering."
            ),
            "matched_keywords": sorted(keywords),
            "message_id": message_id,
        }
        row = {
            "record_id": _stable_record_id(issue_url),
            "paper_id": PAPER_ID,
            "source_project": "postgresql",
            "source_repository": "postgresql.org/pgsql-bugs",
            "matched_keywords": json.dumps(sorted(keywords), ensure_ascii=False),
            "issue_url": issue_url,
            "number": number_match.group(1) if number_match else "",
            "title": title,
            "body": body,
            "comments": comments,
            "created_at": entry.get("created_at") or parsed["metadata"].get("Date", ""),
            "updated_at": "",
            "closed_at": "",
            "state": "",
            "labels": json.dumps(["pgsql-bugs"], ensure_ascii=False),
            "comments_count": str(len(response_comments)),
            "reactions_total_count": "",
            "message_id": message_id,
            "author": entry.get("author") or parsed["metadata"].get("From", ""),
            "list_name": "pgsql-bugs",
            "original_label_json": json.dumps(original, ensure_ascii=False, sort_keys=True),
        }
        rows.append(row)
        parsed_messages[issue_url] = {
            "archive_entry": entry,
            "parsed": parsed,
            "matched_keywords": sorted(keywords),
            "response_comments": response_comments,
        }

    rows.sort(key=lambda row: int(row["number"] or 0))
    result = {
        "archive_bug_reports": len(archive_entries),
        "unique_issues": len(rows),
        "with_body": sum(bool(row["body"].strip()) for row in rows),
        "with_comments": sum(bool(row["comments"].strip()) for row in rows),
    }

    if write:
        with (output_dir / "txbug_postgresql_candidates.csv").open(
            "w", encoding="utf-8", newline="\n"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        with (output_dir / "txbug_postgresql_candidates.raw.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        ) as handle:
            for issue_url, payload in sorted(parsed_messages.items()):
                handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        manifest = {
            "paper_id": PAPER_ID,
            "source": LIST_BASE,
            "archive_window": {"start": f"{START_YEAR}-01", "end": f"{END_YEAR}-12"},
            "keywords": KEYWORDS,
            **result,
        }
        (output_dir / "fetch_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        final_urls = {_canonical_issue_url(url) for url in _final_postgresql_urls(root)}
        candidate_urls = {_canonical_issue_url(row["issue_url"]) for row in rows}
        _write_coverage_audit(output_dir, candidate_urls, final_urls)
        _write_readme(output_dir, rows, final_urls)
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = fetch_candidates(args.root, write=not args.check)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    main()
