from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Benchmark.scripts.run_ase2022_llm_baseline import (  # noqa: E402
    _load_env_file,
)
from Benchmark.src.issta2024_bugs_in_pods_baseline import (  # noqa: E402
    DEFAULT_STAGE1_PATH,
)
from Benchmark.src.issta2024_evidence_collection import (  # noqa: E402
    GitHubClient,
    collect_issta2024_evidence,
)


DEFAULT_COHORT_PATH = (
    "Benchmark/results/issta2024_bugs_in_pods_baseline/"
    "issta2024_stage2_filter_sample.csv"
)
DEFAULT_OUTPUT_DIR = (
    "Benchmark/results/issta2024_bugs_in_pods_baseline/evidence_enriched"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect ASE2022-format GitHub evidence for the fixed ISSTA2024 cohort."
        )
    )
    parser.add_argument("--cohort-path", default=DEFAULT_COHORT_PATH)
    parser.add_argument("--stage1-path", default=DEFAULT_STAGE1_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-delay-seconds", type=float, default=1)
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    _load_env_file(REPO_ROOT / ".env")
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        raise SystemExit("Missing GITHUB_TOKEN in .env or environment")
    client = GitHubClient(
        token=token,
        timeout_seconds=args.timeout,
        max_retries=args.max_retries,
        retry_delay_seconds=args.retry_delay_seconds,
    )
    paths = collect_issta2024_evidence(
        cohort_path=args.cohort_path,
        stage1_path=args.stage1_path,
        output_dir=args.output_dir,
        client=client,
        resume=not args.no_resume,
        limit=args.limit,
    )
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "paths": {
                    key: str(path.resolve())
                    for key, path in paths.items()
                },
                "counts": manifest["counts"],
                "status_counts": manifest["status_counts"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
