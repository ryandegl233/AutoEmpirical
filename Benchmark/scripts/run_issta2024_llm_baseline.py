from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Benchmark.scripts.run_ase2022_camel_mas_baseline import (  # noqa: E402
    resolve_model,
    resolve_run_config,
)
from Benchmark.scripts.run_ase2022_llm_baseline import (  # noqa: E402
    _load_env_file,
    model_slug,
)
from Benchmark.src.ase2022_llm_baseline import run_llm_prompts  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the ISSTA2024 Bugs in Pods Stage 3 Single-LLM baseline."
    )
    parser.add_argument("--provider", choices=("proxy", "deepseek"), default="deepseek")
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument(
        "--prompts-path",
        default=(
            "Benchmark/results/issta2024_bugs_in_pods_baseline/"
            "issta2024_stage3_llm_prompts.jsonl"
        ),
    )
    parser.add_argument(
        "--sample-path",
        default=(
            "Benchmark/results/issta2024_bugs_in_pods_baseline/"
            "issta2024_stage3_llm_sample.csv"
        ),
    )
    parser.add_argument(
        "--taxonomy-path",
        default=(
            "Benchmark/results/issta2024_bugs_in_pods_baseline/"
            "issta2024_taxonomy.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "Benchmark/results/issta2024_bugs_in_pods_baseline/"
            "single_llm_control"
        ),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-delay-seconds", type=float, default=2.0)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--http-client", choices=("urllib", "powershell"), default="urllib"
    )
    return parser


def _load_sample(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return [
            {key: (value or "").strip() for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def _load_taxonomy(path: str | Path) -> dict[str, list[str]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("symptom"), list)
        or not isinstance(payload.get("root_cause"), list)
    ):
        raise ValueError("taxonomy must contain symptom and root_cause lists")
    return {
        "symptom": [str(value) for value in payload["symptom"]],
        "root_cause": [str(value) for value in payload["root_cause"]],
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    _load_env_file(REPO_ROOT / ".env")
    model = resolve_model(args.provider, args.model)
    config = resolve_run_config(
        os.environ,
        base_url_override=args.base_url,
        provider=args.provider,
    )
    examples = _load_sample(args.sample_path)
    taxonomy = _load_taxonomy(args.taxonomy_path)
    output_dir = Path(args.output_dir)
    slug = model_slug(model)
    metrics = run_llm_prompts(
        prompts_path=args.prompts_path,
        examples=examples,
        taxonomy=taxonomy,
        predictions_path=output_dir
        / f"issta2024_stage3_llm_predictions_{slug}.jsonl",
        metrics_path=output_dir / f"issta2024_stage3_llm_metrics_{slug}.json",
        model=model,
        api_key=config["api_key"],
        base_url=config["base_url"],
        wire_api="chat_completions",
        http_client=args.http_client,
        limit=args.limit,
        sleep_seconds=args.sleep_seconds,
        resume=not args.no_resume,
        max_retries=args.max_retries,
        retry_delay_seconds=args.retry_delay_seconds,
        task="issta2024_stage3_classification",
    )
    print(metrics)


if __name__ == "__main__":
    main()
