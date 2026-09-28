"""Freeze leakage-isolated ASE2022 Stage 3 validation and final splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Benchmark.src.adaptive_empirical_workflow.splits import (  # noqa: E402
    SplitPreparationConfig,
    prepare_frozen_stage3_splits,
)


DEFAULT_SOURCE = (
    REPO_ROOT
    / "Dataset"
    / "by_paper"
    / "ase2022_towards_understanding_the_faults_of"
    / "stage3.csv"
)
DEFAULT_TAXONOMY = (
    REPO_ROOT / "Benchmark" / "configs" / "ase2022_taxonomy_structure_v1.json"
)
DEFAULT_LABEL_TAXONOMY = (
    REPO_ROOT
    / "Benchmark"
    / "inputs"
    / "ase2022_issue_only_holdout_seed20260806"
    / "ase2022_issue_only_holdout_taxonomy.json"
)
DEFAULT_CONTAMINATION_ROOTS = tuple(
    REPO_ROOT / relative for relative in (
        "Benchmark/inputs", "Benchmark/configs/splits", "Benchmark/results", "Benchmark/runs"
    )
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare deterministic ASE2022 Stage 3 validation/final cohorts while excluding "
            "every record ID found in prior result artifacts."
        )
    )
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--taxonomy-path", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument(
        "--label-taxonomy-path", type=Path, default=DEFAULT_LABEL_TAXONOMY
    )
    parser.add_argument(
        "--contamination-root",
        action="append",
        type=Path,
        help="File/directory containing prior cohorts, manifests or predictions; repeatable.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--restricted-gold-output",
        dest="restricted_gold_output",
        type=Path,
        required=True,
        help="Evaluator-only gold file outside the runner artifact directory.",
    )
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--validation-size", type=int, default=8)
    parser.add_argument("--final-size", type=int, default=60)
    parser.add_argument("--split-revision", type=int, required=True)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.82)
    parser.add_argument("--supersedes-manifest", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    artifacts = prepare_frozen_stage3_splits(
        SplitPreparationConfig(
            source_csv=args.source_csv,
            taxonomy_path=args.taxonomy_path,
            label_taxonomy_path=args.label_taxonomy_path,
            contamination_roots=tuple(args.contamination_root or DEFAULT_CONTAMINATION_ROOTS),
            output_dir=args.output_dir,
            restricted_gold_path=args.restricted_gold_output,
            split_revision=args.split_revision,
            seed=args.seed,
            validation_min_size=args.validation_size,
            final_min_size=args.final_size,
            near_duplicate_threshold=args.near_duplicate_threshold,
            supersedes_manifest=args.supersedes_manifest,
        )
    )
    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "final_count": manifest["splits"]["final"]["count"],
                "manifest": str(artifacts.manifest_path.resolve()),
                "restricted_gold_created": artifacts.restricted_gold_path.is_file(),
                "validation_count": manifest["splits"]["validation"]["count"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
