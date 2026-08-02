from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Benchmark.src.issta2024_bugs_in_pods_baseline import (  # noqa: E402
    DEFAULT_SEED,
    DEFAULT_STAGE1_PATH,
    DEFAULT_STAGE2_PATH,
    DEFAULT_STAGE3_PATH,
    prepare_issta2024_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare aligned ISSTA2024 Bugs in Pods Stage 2 and Stage 3 "
            "baseline inputs."
        )
    )
    parser.add_argument("--stage1-path", default=DEFAULT_STAGE1_PATH)
    parser.add_argument("--stage2-path", default=DEFAULT_STAGE2_PATH)
    parser.add_argument("--stage3-path", default=DEFAULT_STAGE3_PATH)
    parser.add_argument(
        "--output-dir",
        default="Benchmark/results/issta2024_bugs_in_pods_baseline",
    )
    parser.add_argument("--positives", type=int, default=50)
    parser.add_argument("--negatives", type=int, default=50)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    paths = prepare_issta2024_artifacts(
        stage1_path=args.stage1_path,
        stage2_path=args.stage2_path,
        stage3_path=args.stage3_path,
        output_dir=args.output_dir,
        positive_count=args.positives,
        negative_count=args.negatives,
        seed=args.seed,
    )
    print(
        json.dumps(
            {key: str(path.resolve()) for key, path in paths.items()},
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
