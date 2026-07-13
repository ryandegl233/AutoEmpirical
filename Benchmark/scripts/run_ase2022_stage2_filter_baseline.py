from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Benchmark.scripts.run_ase2022_llm_baseline import (  # noqa: E402
    DEFAULT_MODELS,
    _load_env_file,
    model_slug,
)
from Benchmark.src.ase2022_stage2_filter_baseline import run_filter_prompts  # noqa: E402


def resolve_run_config(
    env: dict[str, str], base_url_override: str | None = None
) -> dict[str, object]:
    base_url = (
        base_url_override
        or env.get("SELF_BASE_URL")
        or env.get("BASE_URL")
        or env.get("LLM_BASE_URL")
    )
    if not base_url:
        raise SystemExit("Missing SELF_BASE_URL or BASE_URL in .env or --base-url")
    api_key = env.get("SELF_API") or env.get("OPENAI_API_KEY") or env.get("API_KEY")
    if not api_key:
        raise SystemExit("Missing SELF_API, OPENAI_API_KEY, or API_KEY in .env or environment")
    return {
        "models": DEFAULT_MODELS,
        "base_url": base_url,
        "api_key": api_key,
        "wire_api": env.get("WIRE_API", "chat_completions"),
        "responses_path": env.get("RESPONSES_PATH", "/responses"),
        "http_client": env.get("HTTP_CLIENT", "urllib"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ASE2022 Stage 2 filtering baseline.")
    parser.add_argument("--base-url", default=None)
    parser.add_argument(
        "--prompts-path",
        default=(
            "Benchmark/results/ase2022_stage2_filter_baseline/"
            "ase2022_stage2_filter_anchored_prompts.jsonl"
        ),
    )
    parser.add_argument(
        "--output-dir", default="Benchmark/results/ase2022_stage2_filter_baseline"
    )
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-delay-seconds", type=float, default=2.0)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--responses-path", default=None)
    parser.add_argument("--http-client", choices=("urllib", "powershell"), default=None)
    args = parser.parse_args()

    _load_env_file(REPO_ROOT / ".env")
    config = resolve_run_config(os.environ, args.base_url)
    models = args.models or config["models"]
    output_dir = Path(args.output_dir)
    for model in models:
        slug = model_slug(str(model))
        metrics = run_filter_prompts(
            prompts_path=args.prompts_path,
            predictions_path=output_dir / f"ase2022_stage2_filter_predictions_{slug}.jsonl",
            metrics_path=output_dir / f"ase2022_stage2_filter_metrics_{slug}.json",
            model=str(model),
            api_key=str(config["api_key"]),
            base_url=str(config["base_url"]),
            wire_api=str(config["wire_api"]),
            responses_path=args.responses_path or str(config["responses_path"]),
            http_client=args.http_client or str(config["http_client"]),
            limit=args.limit,
            sleep_seconds=args.sleep_seconds,
            resume=not args.no_resume,
            max_retries=args.max_retries,
            retry_delay_seconds=args.retry_delay_seconds,
        )
        print(metrics)


if __name__ == "__main__":
    main()
