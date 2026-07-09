from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Benchmark.src.ase2022_llm_baseline import (
    build_taxonomy,
    load_ase2022_examples,
    select_fixed_sample,
    write_prompt_jsonl,
    write_sample_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare ASE2022 anchored-label single-LLM Stage 3 baseline inputs."
    )
    parser.add_argument("--stage2-path", default="Dataset/stage2.csv")
    parser.add_argument("--stage3-path", default="Dataset/stage3.csv")
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--output-dir", default="Benchmark/results/ase2022_llm_baseline")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    examples = load_ase2022_examples(args.stage2_path, args.stage3_path)
    taxonomy = build_taxonomy(examples)
    sample = select_fixed_sample(examples, args.sample_size, args.seed)

    sample_path = write_sample_csv(sample, output_dir / "ase2022_stage3_llm_sample.csv")
    prompt_path = write_prompt_jsonl(
        sample,
        taxonomy,
        output_dir / "ase2022_stage3_anchored_prompts.jsonl",
    )
    taxonomy_path = output_dir / "ase2022_stage3_taxonomy.json"
    taxonomy_path.write_text(json.dumps(taxonomy, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        json.dumps(
            {
                "examples_total": len(examples),
                "sample_size": len(sample),
                "symptom_labels": len(taxonomy["symptom"]),
                "root_cause_labels": len(taxonomy["root_cause"]),
                "sample_path": str(sample_path),
                "prompt_path": str(prompt_path),
                "taxonomy_path": str(taxonomy_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
