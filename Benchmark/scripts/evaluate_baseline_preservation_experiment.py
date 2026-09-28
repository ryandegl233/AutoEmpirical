"""Evaluate a completed evidence-only preservation run against trusted labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Benchmark.src.adaptive_empirical_workflow.offline_evaluation import (
    evaluate_registered_baseline_preservation_experiment,
    evaluate_registered_restricted_holdout,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline aggregate evaluation for Baseline preservation runs."
    )
    parser.add_argument("--predictions-path", required=True)
    parser.add_argument("--evaluation-manifest")
    parser.add_argument("--split-manifest")
    parser.add_argument("--restricted-gold-path")
    parser.add_argument("--split-name", choices=("validation", "final"))
    parser.add_argument("--output-path")
    return parser


def run_cli(argv: list[str] | None = None) -> dict[str, object]:
    args = build_parser().parse_args(argv)
    development = bool(args.evaluation_manifest)
    restricted = all((args.split_manifest, args.restricted_gold_path, args.split_name))
    if development == restricted:
        raise ValueError(
            "supply either --evaluation-manifest or the complete restricted "
            "--split-manifest/--restricted-gold-path/--split-name surface"
        )
    if development:
        metrics = evaluate_registered_baseline_preservation_experiment(
            args.predictions_path,
            evaluation_manifest_path=args.evaluation_manifest,
        )
    else:
        metrics = evaluate_registered_restricted_holdout(
            args.predictions_path,
            split_manifest_path=args.split_manifest,
            restricted_gold_path=args.restricted_gold_path,
            split_name=args.split_name,
        )
    if args.output_path:
        output = Path(args.output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))
    return metrics


def main() -> None:
    run_cli()


if __name__ == "__main__":
    main()
