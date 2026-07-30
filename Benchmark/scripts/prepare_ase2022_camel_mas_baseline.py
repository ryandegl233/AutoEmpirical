from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Benchmark.src.ase2022_camel_mas_baseline import prepare_artifacts  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the unified ASE2022 Stage 1→2→3 CAMEL-AI MAS cohort."
    )
    parser.add_argument(
        "--stage2-sample-path",
        default=(
            "Benchmark/results/ase2022_stage2_filter_baseline/"
            "ase2022_stage2_filter_sample.csv"
        ),
    )
    parser.add_argument(
        "--stage3-sample-path",
        default=(
            "Benchmark/results/ase2022_llm_baseline/"
            "ase2022_stage3_llm_sample.csv"
        ),
    )
    parser.add_argument(
        "--output-dir", default="Benchmark/results/ase2022_camel_mas_baseline"
    )
    parser.add_argument("--positives", type=int, default=50)
    parser.add_argument("--negatives", type=int, default=50)
    args = parser.parse_args()

    paths = prepare_artifacts(
        stage2_sample_path=args.stage2_sample_path,
        stage3_sample_path=args.stage3_sample_path,
        output_dir=args.output_dir,
        positives=args.positives,
        negatives=args.negatives,
    )
    print(json.dumps({key: str(path) for key, path in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
