from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Benchmark.scripts.run_adaptive_empirical_workflow import run_cli  # noqa: E402
from Benchmark.src.adaptive_empirical_workflow.concurrency_profile import (  # noqa: E402
    choose_concurrency,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Explicitly benchmark adaptive-workflow record concurrency and "
            "select the fastest accuracy-preserving candidate."
        )
    )
    parser.add_argument(
        "--candidates", nargs="+", type=int, default=[3, 4, 5]
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Explicitly disable resume and repeat all paid candidate calls.",
    )
    parser.add_argument(
        "workflow_args",
        nargs=argparse.REMAINDER,
        help="Arguments for run_adaptive_empirical_workflow.py after --.",
    )
    return parser


def candidate_arguments(
    workflow_args: list[str],
    *,
    concurrency: int,
    output_root: Path,
    force_rerun: bool,
) -> list[str]:
    args = workflow_args + [
        "--concurrency",
        str(concurrency),
        "--output-dir",
        str(output_root / f"concurrency-{concurrency}"),
    ]
    del force_rerun
    args.append("--no-resume")
    return args


def profile_request_fingerprint(
    candidates: list[int],
    workflow_args: list[str],
) -> str:
    payload = {
        "candidates": list(dict.fromkeys(candidates)),
        "workflow_args": workflow_args,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if any(candidate < 1 for candidate in args.candidates):
        raise ValueError("concurrency candidates must be positive")
    workflow_args = list(args.workflow_args)
    if workflow_args[:1] == ["--"]:
        workflow_args = workflow_args[1:]
    forbidden = {
        "--concurrency",
        "--output-dir",
        "--dry-run",
        "--no-resume",
    }
    if forbidden.intersection(workflow_args):
        raise ValueError(
            "workflow_args must not override --concurrency, --output-dir, "
            "or --dry-run"
        )
    request_fingerprint = profile_request_fingerprint(
        args.candidates,
        workflow_args,
    )

    output_root = Path(args.output_root)
    profile_path = output_root / "concurrency_profile.json"
    if profile_path.exists() and not args.force_rerun:
        existing_profile = json.loads(
            profile_path.read_text(encoding="utf-8")
        )
        if existing_profile.get("status") == "completed":
            if (
                existing_profile.get("request_fingerprint")
                != request_fingerprint
            ):
                raise ValueError(
                    "completed concurrency profile belongs to a different "
                    "request; use another output root or --force-rerun"
                )
            print(json.dumps(existing_profile, ensure_ascii=False, indent=2))
            return
        raise ValueError(
            "incomplete concurrency profile exists; inspect it, then use "
            "--force-rerun to authorize fresh paid candidate runs"
        )
    if (
        output_root.exists()
        and any(output_root.iterdir())
        and not args.force_rerun
    ):
        raise ValueError(
            "incomplete concurrency profile exists; inspect it, then use "
            "--force-rerun to authorize fresh paid candidate runs"
        )

    output_root.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(
        json.dumps(
            {
                "status": "in_progress",
                "request_fingerprint": request_fingerprint,
                "candidates_requested": list(
                    dict.fromkeys(args.candidates)
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    results: list[dict[str, object]] = []
    for candidate in dict.fromkeys(args.candidates):
        summary = run_cli(
            candidate_arguments(
                workflow_args,
                concurrency=candidate,
                output_root=output_root,
                force_rerun=args.force_rerun,
            )
        )
        results.append(
            {
                "concurrency": candidate,
                "elapsed_seconds": summary["metrics"][
                    "benchmark_wall_time_seconds"
                ],
                "metrics": summary["metrics"],
            }
        )

    winner = choose_concurrency(results)
    profile = {
        "status": "completed",
        "request_fingerprint": request_fingerprint,
        "workflow_args": workflow_args,
        "winner": winner,
        "candidates": results,
    }
    profile_path.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(profile, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
