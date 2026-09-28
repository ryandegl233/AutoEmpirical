"""Prepare the frozen evidence-only ASE2022 contaminated development split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE_COHORT = ROOT / (
    "Benchmark/inputs/ase2022_issue_only_holdout_seed20260806/"
    "ase2022_issue_only_holdout_cohort.csv"
)
SELECTION_MANIFEST = ROOT / "Benchmark/configs/splits/ase2022_stage3_contaminated_dev50/split_manifest.json"

TAXONOMY = ROOT / (
    "Benchmark/inputs/ase2022_issue_only_holdout_seed20260806/"
    "ase2022_issue_only_holdout_taxonomy.json"
)
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


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection-manifest', type=Path, default=SELECTION_MANIFEST)
    parser.add_argument('--source-cohort', type=Path, default=SOURCE_COHORT)
    parser.add_argument('--taxonomy', type=Path, default=TAXONOMY)
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='New output directory; the published split is never overwritten.')
    args = parser.parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError('Use a new output directory to preserve frozen inputs')
    selection = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
    record_ids = tuple(selection['splits']['development']['record_ids'])
    with args.source_cohort.open(encoding="utf-8-sig", newline="") as handle:
        source = {row["record_id"]: row for row in csv.DictReader(handle)}
    if len(record_ids) != 50 or len(set(record_ids)) != 50:
        raise ValueError("development selection manifest must freeze 50 unique IDs")
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cohort = args.output_dir / "development_runner_cohort.csv"
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
            "selection_manifest": str(args.selection_manifest),
            "selection_manifest_sha256": _sha(args.selection_manifest),
            "source_cohort_sha256": _sha(args.source_cohort),
            "taxonomy_structure_sha256": _text_sha(args.taxonomy),
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
    (args.output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
