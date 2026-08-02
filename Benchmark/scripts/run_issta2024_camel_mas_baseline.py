from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Benchmark.scripts.run_ase2022_camel_mas_baseline import (  # noqa: E402
    CamelMasCliProfile,
    run_profile,
)
from Benchmark.src.issta2024_bugs_in_pods_baseline import (  # noqa: E402
    build_society_task,
)


PROFILE = CamelMasCliProfile(
    study_slug="issta2024",
    description=(
        "Run the ISSTA2024 Bugs in Pods CAMEL-AI RolePlaying Society baseline."
    ),
    default_provider="deepseek",
    default_cohort_path=(
        "Benchmark/results/issta2024_bugs_in_pods_baseline/"
        "issta2024_stage2_filter_sample.csv"
    ),
    default_taxonomy_path=(
        "Benchmark/results/issta2024_bugs_in_pods_baseline/"
        "issta2024_taxonomy.json"
    ),
    default_output_dir=(
        "Benchmark/results/issta2024_bugs_in_pods_baseline/mas"
    ),
    default_single_stage2_metrics=(
        "Benchmark/results/issta2024_bugs_in_pods_baseline/"
        "single_llm_control/issta2024_stage2_filter_metrics_{slug}.json"
    ),
    default_single_stage3_metrics=(
        "Benchmark/results/issta2024_bugs_in_pods_baseline/"
        "single_llm_control/issta2024_stage3_llm_metrics_{slug}.json"
    ),
    task_builder=build_society_task,
    default_require_valid_json=True,
)


def main(argv: list[str] | None = None) -> None:
    run_profile(PROFILE, argv)


if __name__ == "__main__":
    main()
