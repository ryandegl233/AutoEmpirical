"""Content-bound offline evaluation for the fixed targeted-SLA experiment."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Benchmark.src.adaptive_empirical_workflow.offline_evaluation import (  # noqa: E402
    evaluate_bound_targeted_sla_experiment,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the exact 14-record targeted-SLA predictions against "
            "content-bound Baseline and registered offline gold artifacts."
        )
    )
    parser.add_argument("--predictions-path", required=True)
    parser.add_argument("--predictions-sha256", required=True)
    parser.add_argument("--run-manifest-path", required=True)
    parser.add_argument("--run-manifest-sha256", required=True)
    parser.add_argument("--baseline-anchor-path", required=True)
    parser.add_argument("--baseline-sha256", required=True)
    parser.add_argument("--baseline-config-hash", required=True)
    parser.add_argument("--evaluation-manifest", required=True)
    parser.add_argument("--output-path", required=True)
    return parser


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".targeted-sla-evaluation-",
        suffix=".json.tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run_cli(argv: list[str] | None = None) -> dict[str, object]:
    args = build_parser().parse_args(argv)
    artifact = evaluate_bound_targeted_sla_experiment(
        args.predictions_path,
        expected_predictions_sha256=args.predictions_sha256,
        run_manifest_path=args.run_manifest_path,
        expected_run_manifest_sha256=args.run_manifest_sha256,
        baseline_anchor_path=args.baseline_anchor_path,
        expected_baseline_sha256=args.baseline_sha256,
        expected_baseline_config_hash=args.baseline_config_hash,
        evaluation_manifest_path=args.evaluation_manifest,
    )
    _write_json_atomic(Path(args.output_path), artifact)
    print(json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True))
    return artifact


def main() -> None:
    run_cli()


if __name__ == "__main__":
    main()
