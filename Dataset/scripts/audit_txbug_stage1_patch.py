"""Independent logical audit for the full TXBug Stage 1 repair."""

from __future__ import annotations

import csv
import io
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PAPER_ID = "icse2024_understanding_transaction_bugs_in_database"
EVIDENCE_PATH = (
    ROOT
    / "Dataset/evidence/icse2024_transaction_bugs_stage1_evidence.jsonl"
)
CSV_PATHS = [
    "Dataset/by_paper/icse2024_understanding_transaction_bugs_in_database/stage1.csv",
    "Dataset/stage1.csv",
]
UNTOUCHED_PATHS = [
    "Dataset/by_paper/icse2024_understanding_transaction_bugs_in_database/stage2.csv",
    "Dataset/by_paper/icse2024_understanding_transaction_bugs_in_database/stage3.csv",
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
csv.field_size_limit(min(sys.maxsize, 2_147_483_647))


def normalize_newlines(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n")


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
        result: set[str] = set()
        for child in value:
            result.update(nested_keys(child))
        return result
    return set()


def main() -> None:
    evidence_rows = [
        json.loads(line)
        for line in EVIDENCE_PATH.read_text(encoding="utf-8").split("\n")
        if line.strip()
    ]
    for row in evidence_rows:
        row["comments"] = normalize_newlines(row["comments"])
    evidence = {row["record_id"]: row for row in evidence_rows}
    assert len(evidence_rows) == 7775
    assert len(evidence) == 7775
    assert len({row["issue_url"] for row in evidence_rows}) == 7775
    assert {row["paper_id"] for row in evidence_rows} == {PAPER_ID}
    assert Counter(row["retrieval_status"] for row in evidence_rows) == {
        "ok": 6825,
        "ok_zero_comments": 948,
        "source_unavailable": 2,
    }
    assert not any(
        nested_keys(row) & FORBIDDEN_EVIDENCE_KEYS
        for row in evidence_rows
    )

    file_results = []
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

        changed_ids: set[str] = set()
        for record_id, current_row in current.items():
            baseline_row = baseline[record_id]
            changed_fields = {
                field
                for field in current_fields
                if current_row[field] != baseline_row[field]
            }
            if changed_fields:
                changed_ids.add(record_id)
                assert changed_fields == {"comments"}, (
                    relative_path,
                    record_id,
                    changed_fields,
                )
                assert record_id in evidence, (relative_path, record_id)

        targets = {
            record_id: row
            for record_id, row in current.items()
            if record_id in evidence
        }
        assert len(targets) == 7775, relative_path
        for record_id, row in targets.items():
            assert normalize_newlines(row["comments"]) == evidence[record_id][
                "comments"
            ], (relative_path, record_id)
        assert all(row["comments"].strip() for row in targets.values())

        file_results.append(
            {
                "path": relative_path,
                "rows": len(current_rows),
                "target_rows": len(targets),
                "changed_rows": len(changed_ids),
                "changed_fields": ["comments"],
                "unrelated_changed_rows": len(changed_ids - set(evidence)),
            }
        )

    untouched = {}
    for relative_path in UNTOUCHED_PATHS:
        process = subprocess.run(
            ["git", "diff", "--quiet", "origin/main", "--", relative_path],
            cwd=ROOT,
        )
        untouched[relative_path] = process.returncode == 0
        assert process.returncode == 0, relative_path

    result = {
        "result": "PASS",
        "baseline": "origin/main",
        "evidence_records": len(evidence_rows),
        "status_counts": dict(
            Counter(row["retrieval_status"] for row in evidence_rows)
        ),
        "evidence_contains_gold_label_keys": False,
        "stage2_stage3_untouched": all(untouched.values()),
        "files": file_results,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
