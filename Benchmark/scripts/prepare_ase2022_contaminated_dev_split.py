"""Prepare the frozen evidence-only ASE2022 contaminated development split."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE_COHORT = ROOT / (
    "Benchmark/inputs/ase2022_issue_only_holdout_seed20260806/"
    "ase2022_issue_only_holdout_cohort.csv"
)
CATEGORY_MANIFEST = ROOT / (
    "Benchmark/configs/ase2022_issue_only_holdout_mas_baseline_error_categories.json"
)
TAXONOMY = ROOT / (
    "Benchmark/inputs/ase2022_issue_only_holdout_seed20260806/"
    "ase2022_issue_only_holdout_taxonomy.json"
)
OUTPUT = ROOT / "Benchmark/configs/splits/ase2022_stage3_contaminated_dev50"
FIELDS = (
    "record_id",
    "paper_id",
    "issue_url",
    "title",
    "body",
    "comments",
    "state",
    "created_at",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _text_sha(path: Path) -> str:
    text = (
        path.read_text(encoding="utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_sha(value: object) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    categories = json.loads(CATEGORY_MANIFEST.read_text(encoding="utf-8"))
    record_ids = tuple(categories["record_ids"])
    with SOURCE_COHORT.open(encoding="utf-8-sig", newline="") as handle:
        source = {row["record_id"]: row for row in csv.DictReader(handle)}
    if len(record_ids) != 50 or len(set(record_ids)) != 50:
        raise ValueError("development category manifest must freeze 50 unique IDs")
    if set(record_ids) - set(source):
        raise ValueError("development IDs are missing from source cohort")
    rows = [
        {field: source[record_id].get(field, "") for field in FIELDS}
        for record_id in record_ids
    ]
    if any(
        set(row).intersection({"decision", "symptom", "root_cause", "gold"})
        for row in rows
    ):
        raise ValueError("runner cohort contains forbidden label fields")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cohort = OUTPUT / "development_runner_cohort.csv"
    with cohort.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "schema_version": 1,
        "status": "active",
        "split_revision": 1,
        "split_id": "ase2022-stage3-contaminated-development-seed20260806-revision1",
        "split_kind": "contaminated_development",
        "domain": "ase2022",
        "seed": 20260806,
        "source": {
            "category_manifest": CATEGORY_MANIFEST.relative_to(ROOT).as_posix(),
            "category_manifest_sha256": _sha(CATEGORY_MANIFEST),
            "source_cohort_sha256": _sha(SOURCE_COHORT),
            "taxonomy_structure_sha256": _text_sha(TAXONOMY),
            "text_hash_algorithm": "canonical_utf8_lf_sha256_v1",
        },
        "splits": {
            "development": {
                "count": len(record_ids),
                "record_ids": list(record_ids),
                "record_ids_sha256": _canonical_sha(list(record_ids)),
                "runner_cohort_file": cohort.name,
                "runner_cohort_sha256": _text_sha(cohort),
            }
        },
        "label_access_policy": {
            "runner_may_load_labels": False,
            "runner_cohorts_are_evidence_only": True,
            "development_is_contaminated": True,
        },
    }
    (OUTPUT / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
