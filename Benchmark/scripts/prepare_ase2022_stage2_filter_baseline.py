from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Benchmark.src.ase2022_stage2_filter_baseline import (  # noqa: E402
    load_ase2022_stage2_filter_examples,
    select_balanced_sample,
    write_manifest,
    write_prompts_jsonl,
    write_sample_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the ASE2022 Stage 2 filtering baseline.")
    parser.add_argument("--stage1-path", default="Dataset/stage1.csv")
    parser.add_argument("--stage2-path", default="Dataset/stage2.csv")
    parser.add_argument(
        "--output-dir", default="Benchmark/results/ase2022_stage2_filter_baseline"
    )
    parser.add_argument("--per-class", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    all_examples = load_ase2022_stage2_filter_examples(args.stage1_path, args.stage2_path)
    selected = (
        all_examples
        if args.full
        else select_balanced_sample(all_examples, per_class=args.per_class, seed=args.seed)
    )
    output_dir = Path(args.output_dir)
    sample_path = write_sample_csv(selected, output_dir / "ase2022_stage2_filter_sample.csv")
    prompts_path = write_prompts_jsonl(
        selected, output_dir / "ase2022_stage2_filter_anchored_prompts.jsonl"
    )
    manifest_path = write_manifest(
        all_examples,
        selected,
        output_dir / "ase2022_stage2_filter_manifest.json",
        seed=args.seed,
        per_class=args.per_class,
        stage1_path=args.stage1_path,
        stage2_path=args.stage2_path,
        mode="full" if args.full else "balanced_pilot",
    )
    print(f"sample={sample_path}")
    print(f"prompts={prompts_path}")
    print(f"manifest={manifest_path}")
    print(f"selected={len(selected)}")


if __name__ == "__main__":
    main()
