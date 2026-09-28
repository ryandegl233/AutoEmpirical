"""Independent audit for the Transaction Bugs discussion-only patch.

Dataset reconstruction audit utility.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
csv.field_size_limit(min(sys.maxsize, 2_147_483_647))
PAPER_ID = "icse2024_understanding_transaction_bugs_in_database"
EVIDENCE = ROOT / "Dataset/evidence/icse2024_transaction_bugs_evidence.jsonl"
CSV_PATHS = [
    "Dataset/by_paper/icse2024_understanding_transaction_bugs_in_database/stage1.csv",
    "Dataset/by_paper/icse2024_understanding_transaction_bugs_in_database/stage2.csv",
    "Dataset/by_paper/icse2024_understanding_transaction_bugs_in_database/stage3.csv",
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
    oid_line = next(line for line in pointer.splitlines() if line.startswith("oid sha256:"))
    oid = oid_line.removeprefix("oid sha256:")
    common_dir = subprocess.run(
        ["git", "rev-parse", "--git-common-dir"],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    common_path = Path(common_dir)
    if not common_path.is_absolute():
        common_path = ROOT / common_path
    object_path = common_path / "lfs/objects" / oid[:2] / oid[2:4] / oid
    if not object_path.exists():
        raise FileNotFoundError(f"Missing local Git LFS baseline object: {object_path}")
    return object_path.read_bytes()


def read_csv_bytes(value: bytes) -> tuple[list[str], list[dict[str, str]]]:
    stream = io.StringIO(value.decode("utf-8-sig"), newline=None)
    reader = csv.DictReader(stream)
    return list(reader.fieldnames or []), list(reader)


def read_current_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
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


def normalize_newlines(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n")


def restore_baseline_non_comment_fields() -> None:
    """Rebuild the six files from origin/main, changing only target comments."""
    evidence_rows = [
        json.loads(line)
        for line in EVIDENCE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in evidence_rows:
        row["comments"] = normalize_newlines(row["comments"])
    evidence_by_id = {row["record_id"]: row for row in evidence_rows}
    for relative_path in CSV_PATHS:
        fields, rows = read_csv_bytes(git_bytes("origin/main", relative_path))
        for row in rows:
            item = evidence_by_id.get(row["record_id"])
            if item is not None:
                row["comments"] = item["comments"]
        path = ROOT / relative_path
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=fields,
                extrasaction="ignore",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    evidence_rows = [
        json.loads(line)
        for line in EVIDENCE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in evidence_rows:
        row["comments"] = normalize_newlines(row["comments"])
    evidence_by_id = {row["record_id"]: row for row in evidence_rows}
    assert len(evidence_rows) == 140
    assert len(evidence_by_id) == 140
    assert len({row["issue_url"] for row in evidence_rows}) == 140
    assert {row["paper_id"] for row in evidence_rows} == {PAPER_ID}

    statuses = Counter(row["retrieval_status"] for row in evidence_rows)
    assert statuses == {"ok": 127, "ok_zero_comments": 11, "source_unavailable": 2}
    for row in evidence_rows:
        forbidden = nested_keys(row) & FORBIDDEN_EVIDENCE_KEYS
        assert not forbidden, (row["record_id"], forbidden)
        if row["retrieval_status"] == "ok":
            assert row["comments"] not in {
                "",
                "no_comments_in_source",
                "comments_unavailable_in_source",
            }
            assert row["source_comment_count"] > 0
        elif row["retrieval_status"] == "ok_zero_comments":
            assert row["comments"] == "no_comments_in_source"
            assert row["source_comment_count"] == 0
        else:
            assert row["comments"] == "comments_unavailable_in_source"

    file_results = []
    comments_by_file: dict[str, dict[str, str]] = {}
    for relative_path in CSV_PATHS:
        baseline_fields, baseline_rows = read_csv_bytes(git_bytes("origin/main", relative_path))
        current_fields, current_rows = read_current_csv(ROOT / relative_path)
        assert current_fields == baseline_fields, relative_path
        assert len(current_rows) == len(baseline_rows), relative_path

        baseline_by_id = {row["record_id"]: row for row in baseline_rows}
        current_by_id = {row["record_id"]: row for row in current_rows}
        assert len(baseline_by_id) == len(baseline_rows), relative_path
        assert len(current_by_id) == len(current_rows), relative_path
        assert set(current_by_id) == set(baseline_by_id), relative_path

        changed_ids: set[str] = set()
        changed_fields: set[str] = set()
        for record_id, current_row in current_by_id.items():
            baseline_row = baseline_by_id[record_id]
            fields = {
                field
                for field in current_fields
                if current_row[field] != baseline_row[field]
            }
            if fields:
                changed_ids.add(record_id)
                changed_fields.update(fields)
                assert fields == {"comments"}, (relative_path, record_id, fields)

        target_rows = {
            record_id: row
            for record_id, row in current_by_id.items()
            if record_id in evidence_by_id
        }
        assert len(target_rows) == 140, relative_path
        assert changed_ids <= set(evidence_by_id), relative_path
        assert len(changed_ids) == 129, relative_path
        for record_id, row in target_rows.items():
            assert row["comments"] == evidence_by_id[record_id]["comments"], (
                relative_path,
                record_id,
            )

        comments_by_file[relative_path] = {
            record_id: row["comments"] for record_id, row in target_rows.items()
        }
        file_results.append(
            {
                "path": relative_path,
                "rows": len(current_rows),
                "target_rows": len(target_rows),
                "changed_rows": len(changed_ids),
                "changed_fields": sorted(changed_fields),
                "unrelated_changed_rows": len(changed_ids - set(evidence_by_id)),
            }
        )

    reference_comments = next(iter(comments_by_file.values()))
    for relative_path, comments in comments_by_file.items():
        assert comments == reference_comments, relative_path

    result = {
        "result": "PASS",
        "baseline": "origin/main",
        "evidence_records": len(evidence_rows),
        "evidence_status_counts": dict(statuses),
        "evidence_contains_gold_label_keys": False,
        "all_six_files_have_identical_target_comments": True,
        "files": file_results,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--restore-baseline",
        action="store_true",
        help="Rebuild CSVs from origin/main while applying only evidence comments.",
    )
    arguments = parser.parse_args()
    if arguments.restore_baseline:
        restore_baseline_non_comment_fields()
    main()
