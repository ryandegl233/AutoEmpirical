"""Run the existing CAMEL society with a prepared native paper contract."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Benchmark.scripts.run_ase2022_camel_mas_baseline import CamelMasCliProfile, build_parser, run_profile
from Benchmark.src.paper_benchmark import DEFAULT_OUTPUT_ROOT

DEFAULT_PREPARED_ROOT = str(DEFAULT_OUTPUT_ROOT)
DEFAULT_RUN_ROOT = Path("Benchmark/runs/seven_papers")


def paper_cli_profile(domain: str, prepared_root: str | Path = DEFAULT_PREPARED_ROOT) -> CamelMasCliProfile:
    from Benchmark.src.paper_benchmark import (
        build_society_task, get_paper_profile, model_evidence_fields,
    )

    paper = get_paper_profile(domain)
    base = Path(prepared_root) / paper.domain
    runs = DEFAULT_RUN_ROOT / paper.domain
    return CamelMasCliProfile(
        study_slug=paper.domain,
        description=f"Run {paper.title} through the CAMEL society with its native annotation contract.",
        default_provider="gemini",
        default_cohort_path=str(base / "cohort.csv"),
        default_taxonomy_path=str(base / "taxonomy.json"),
        default_output_dir=str(runs / "mas"),
        default_single_stage2_metrics=str(runs / "single_llm" / f"{paper.domain}_stage2_filter_metrics_{{slug}}.json"),
        default_single_stage3_metrics=str(runs / "single_llm" / f"{paper.domain}_stage3_llm_metrics_{{slug}}.json"),
        task_builder=build_society_task,
        evidence_builder=model_evidence_fields,
        default_require_valid_json=True,
        default_explanation_mode="evidence_rationale",
        expected_paper_id=paper.paper_id,
    )


def main(argv: list[str] | None = None) -> None:
    from Benchmark.src.paper_benchmark import PAPER_PROFILES

    values = list(sys.argv[1:] if argv is None else argv)
    help_requested = "--help" in values or "-h" in values
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--domain", choices=sorted(PAPER_PROFILES), required=not help_requested)
    selector.add_argument("--prepared-root", default=DEFAULT_PREPARED_ROOT)
    args, remaining = selector.parse_known_args(values)
    if help_requested:
        parser = build_parser(paper_cli_profile(args.domain or "ase2022", args.prepared_root))
        parser.description = __doc__
        parser.add_argument("--domain", choices=sorted(PAPER_PROFILES), required=True)
        parser.add_argument("--prepared-root", default=DEFAULT_PREPARED_ROOT)
        parser.parse_args(remaining)
    run_profile(paper_cli_profile(args.domain, args.prepared_root), remaining)


if __name__ == "__main__":
    main()
