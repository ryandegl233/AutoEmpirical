"""Independent logical audit for the ICSE 2021 IoT information repair."""

from __future__ import annotations

import csv
import io
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PAPER_ID = "icse2021_iot_bugs_and_development_challenges"
EVIDENCE_PATH = (
    ROOT / "Dataset/evidence/icse2021_iot_discussion_evidence.jsonl"
)
REPORT_PATH = ROOT / "reports/iot_information_reconstruction/audit.json"
CSV_PATHS = [
    f"Dataset/by_paper/{PAPER_ID}/stage1.csv",
    f"Dataset/by_paper/{PAPER_ID}/stage2.csv",
    f"Dataset/by_paper/{PAPER_ID}/stage3.csv",
    "Dataset/stage1.csv",
    "Dataset/stage2.csv",
    "Dataset/stage3.csv",
]
FORBIDDEN_EVIDENCE_KEYS = {
    "label",
    "labels",
    "root_cause",
    "root_cause_label",
    "symptom",
    "symptom_label",
    "bug_type",
    "bug_type_label",
    "stage2_label",
    "stage3_label",
}
RAW_SECRET_PATTERNS = {
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "azure_iot_shared_access_key": re.compile(
        r"(?i)(?<=SharedAccessKey=)"
        r"(?!(?:([A-Za-z0-9+/])\1{29,}={0,2})(?=[;\s\"',]|\Z))"
        r"[^;\s\"',\[]+"
    ),
    "private_key_header": re.compile(
        r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----"
    ),
}
csv.field_size_limit(min(sys.maxsize, 2_147_483_647))


def raw_secret_counts(text: str) -> dict[str, int]:
    return {
        name: len(pattern.findall(text))
        for name, pattern in RAW_SECRET_PATTERNS.items()
    }


def redact_publication_secrets(text: str) -> str:
    cleaned = text
    for name, pattern in RAW_SECRET_PATTERNS.items():
        marker_name = (
            "private_key"
            if name == "private_key_header"
            else name
        )
        cleaned = pattern.sub(
            f"[REDACTED_HIGH_CONFIDENCE_SECRET:{marker_name}]",
            cleaned,
        )
    return cleaned


def git_bytes(revision: str, path: str) -> bytes:
    value = subprocess.run(
        ["git", "show", f"{revision}:{path}"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    if not value.startswith(b"version https://git-lfs.github.com/spec/v1\n"):
        return value
    pointer = value.decode("ascii")
    oid = next(
        line.removeprefix("oid sha256:")
        for line in pointer.splitlines()
        if line.startswith("oid sha256:")
    )
    common = subprocess.run(
        ["git", "rev-parse", "--git-common-dir"],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    common_path = Path(common)
    if not common_path.is_absolute():
        common_path = ROOT / common_path
    object_path = common_path / "lfs/objects" / oid[:2] / oid[2:4] / oid
    if not object_path.exists():
        raise FileNotFoundError(object_path)
    return object_path.read_bytes()


def read_csv_bytes(value: bytes) -> tuple[list[str], list[dict[str, str]]]:
    reader = csv.DictReader(
        io.StringIO(value.decode("utf-8-sig"), newline=None)
    )
    return list(reader.fieldnames or []), list(reader)


def read_current(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline=None) as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def nested_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        result = set(value)
        for child in value.values():
            result.update(nested_keys(child))
        return result
    if isinstance(value, list):
        result = set()
        for child in value:
            result.update(nested_keys(child))
        return result
    return set()


def normalize_newlines(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n")


def expected_comment(row: dict[str, object]) -> str:
    status = row["retrieval_status"]
    if status == "ok":
        value = str(row.get("comments") or "")
        assert value.strip(), row["record_id"]
        return normalize_newlines(value)
    if status == "ok_zero_comments":
        return "no_comments_in_source"
    if status == "source_unavailable":
        return "comments_unavailable_in_source"
    raise AssertionError((row["record_id"], status))


def main() -> None:
    evidence_rows = [
        json.loads(line)
        for line in EVIDENCE_PATH.read_text(encoding="utf-8").split("\n")
        if line.strip()
    ]
    evidence = {row["record_id"]: row for row in evidence_rows}
    assert len(evidence_rows) == 5548
    assert len(evidence) == 5548
    assert len({row["issue_url"] for row in evidence_rows}) == 5548
    assert {row["paper_id"] for row in evidence_rows} == {PAPER_ID}
    assert Counter(row["retrieval_status"] for row in evidence_rows) == {
        "ok": 3728,
        "ok_zero_comments": 1110,
        "source_unavailable": 710,
    }
    assert Counter(row["source_kind"] for row in evidence_rows) == {
        "github_issue": 4697,
        "github_pull_request": 851,
    }
    assert not any(
        nested_keys(row) & FORBIDDEN_EVIDENCE_KEYS for row in evidence_rows
    )

    evidence_text = "\n".join(
        str(row.get("comments") or "") for row in evidence_rows
    )
    raw_secret_findings = raw_secret_counts(evidence_text)
    assert not any(raw_secret_findings.values())
    credential_redactions = {
        "aws_access_key": evidence_text.count(
            "[REDACTED_HIGH_CONFIDENCE_SECRET:aws_access_key]"
        ),
        "azure_iot_shared_access_key": evidence_text.count(
            "[REDACTED_HIGH_CONFIDENCE_SECRET:"
            "azure_iot_shared_access_key]"
        ),
        "private_key": evidence_text.count(
            "[REDACTED_HIGH_CONFIDENCE_SECRET:private_key]"
        ),
    }
    assert credential_redactions == {
        "aws_access_key": 1,
        "azure_iot_shared_access_key": 2,
        "private_key": 1,
    }

    file_results = []
    body_credential_redactions = Counter()
    published_raw_secret_findings = Counter(
        {name: 0 for name in RAW_SECRET_PATTERNS}
    )
    current_by_stage: dict[
        tuple[str, str], dict[str, dict[str, str]]
    ] = {}
    for relative_path in CSV_PATHS:
        baseline_fields, baseline_rows = read_csv_bytes(
            git_bytes("origin/main", relative_path)
        )
        current_fields, current_rows = read_current(ROOT / relative_path)
        assert current_fields == baseline_fields, relative_path
        assert len(current_rows) == len(baseline_rows), relative_path
        baseline = {row["record_id"]: row for row in baseline_rows}
        current = {row["record_id"]: row for row in current_rows}
        assert set(current) == set(baseline), relative_path

        changed_ids = set()
        file_changed_fields = set()
        for record_id, current_row in current.items():
            baseline_row = baseline[record_id]
            changed_fields = {
                field
                for field in current_fields
                if current_row[field] != baseline_row[field]
            }
            if changed_fields:
                changed_ids.add(record_id)
                file_changed_fields.update(changed_fields)
                assert changed_fields <= {"comments", "body"}, (
                    relative_path,
                    record_id,
                    changed_fields,
                )
                assert current_row["paper_id"] == PAPER_ID, (
                    relative_path,
                    record_id,
                )
                if "body" in changed_fields:
                    assert current_row["body"] == (
                        redact_publication_secrets(baseline_row["body"])
                    ), (
                        relative_path,
                        record_id,
                        "body change is not an exact security redaction",
                    )

        targets = {
            record_id: row
            for record_id, row in current.items()
            if row["paper_id"] == PAPER_ID
        }
        expected_target_count = (
            5548 if relative_path.endswith("stage1.csv") else 320
        )
        assert len(targets) == expected_target_count, relative_path
        for record_id, row in targets.items():
            assert normalize_newlines(row["comments"]) == expected_comment(
                evidence[record_id]
            ), (
                relative_path,
                record_id,
            )
            for value in row.values():
                published_raw_secret_findings.update(
                    raw_secret_counts(value)
                )
        if relative_path == CSV_PATHS[0]:
            for row in targets.values():
                body = row["body"]
                for name in (
                    "aws_access_key",
                    "azure_iot_shared_access_key",
                    "private_key",
                ):
                    body_credential_redactions[name] += body.count(
                        f"[REDACTED_HIGH_CONFIDENCE_SECRET:{name}]"
                    )
        stage = Path(relative_path).stem
        scope = "per_paper" if "by_paper" in relative_path else "unified"
        current_by_stage[(scope, stage)] = targets
        file_results.append(
            {
                "path": relative_path,
                "rows": len(current_rows),
                "target_rows": len(targets),
                "changed_rows": len(changed_ids),
                "changed_fields": sorted(file_changed_fields),
                "unrelated_changed_rows": sum(
                    current[record_id]["paper_id"] != PAPER_ID
                    for record_id in changed_ids
                ),
            }
        )

    assert not any(published_raw_secret_findings.values())

    for stage in ("stage1", "stage2", "stage3"):
        per_paper = current_by_stage[("per_paper", stage)]
        unified = current_by_stage[("unified", stage)]
        assert per_paper.keys() == unified.keys(), stage
        assert all(
            per_paper[record_id]["comments"]
            == unified[record_id]["comments"]
            for record_id in per_paper
        ), stage

    result = {
        "result": "PASS",
        "baseline": "origin/main",
        "paper_id": PAPER_ID,
        "evidence_version": "current_unversioned",
        "evidence_records": len(evidence_rows),
        "status_counts": dict(
            Counter(row["retrieval_status"] for row in evidence_rows)
        ),
        "kind_counts": dict(
            Counter(row["source_kind"] for row in evidence_rows)
        ),
        "evidence_contains_gold_label_keys": False,
        "raw_secret_findings": raw_secret_findings,
        "credential_redactions": credential_redactions,
        "body_credential_redactions": dict(
            body_credential_redactions
        ),
        "published_raw_secret_findings": dict(
            published_raw_secret_findings
        ),
        "per_paper_unified_comments_match": True,
        "files": file_results,
    }
    REPORT_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
