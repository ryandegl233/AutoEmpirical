from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Benchmark.src.ase2022_camel_mas_baseline import (  # noqa: E402
    prepare_issue_only_holdout_artifacts,
)


DEFAULT_DATA_DIR = (
    "Dataset/by_paper/"
    "ase2022_towards_understanding_the_faults_of"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a balanced ASE2022 issue-only holdout that excludes "
            "a previously analyzed cohort."
        )
    )
    parser.add_argument(
        "--stage1-path",
        default=f"{DEFAULT_DATA_DIR}/stage1.csv",
    )
    parser.add_argument(
        "--stage2-path",
        default=f"{DEFAULT_DATA_DIR}/stage2.csv",
    )
    parser.add_argument(
        "--stage3-path",
        default=f"{DEFAULT_DATA_DIR}/stage3.csv",
    )
    parser.add_argument(
        "--exclude-cohort-path",
        default=(
            "Benchmark/results/ase2022_camel_mas_baseline/"
            "ase2022_camel_mas_cohort.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "Benchmark/results/"
            "ase2022_issue_only_holdout_seed20260806"
        ),
    )
    parser.add_argument("--positives", type=int, default=50)
    parser.add_argument("--negatives", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260806)
    return parser


def run_cli(argv: Sequence[str] | None = None) -> dict[str, str]:
    args = _parser().parse_args(argv)
    paths = prepare_issue_only_holdout_artifacts(
        stage1_path=args.stage1_path,
        stage2_path=args.stage2_path,
        stage3_path=args.stage3_path,
        excluded_cohort_path=args.exclude_cohort_path,
        output_dir=args.output_dir,
        positives=args.positives,
        negatives=args.negatives,
        seed=args.seed,
    )
    summary = {name: str(path) for name, path in paths.items()}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    run_cli()


if __name__ == "__main__":
    main()
