from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Benchmark.src.ase2022_llm_baseline import (  # noqa: E402
    build_taxonomy,
    load_ase2022_examples,
    run_llm_prompts,
)

DEFAULT_MODELS = [
    "gemini-2.5-flash",
    "claude-3-5-sonnet-20241022",
    "o3",
    "deepseek-chat",
    "gpt-4o-mini",
]


def model_slug(model: str) -> str:
    return "".join(char.lower() if char.isalnum() else "-" for char in model).strip("-")


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def resolve_run_config(
    env: dict[str, str],
    base_url_override: str | None = None,
) -> dict[str, str]:
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

    wire_api = env.get("WIRE_API", "chat_completions")
    responses_path = env.get("RESPONSES_PATH", "/responses")
    http_client = env.get("HTTP_CLIENT", "urllib")

    return {
        "models": DEFAULT_MODELS,
        "base_url": base_url,
        "api_key": api_key,
        "wire_api": wire_api,
        "responses_path": responses_path,
        "http_client": http_client,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ASE2022 anchored-label single-LLM Stage 3 baseline."
    )
    parser.add_argument("--base-url", default=None)
    parser.add_argument(
        "--prompts-path",
        default="Benchmark/results/ase2022_llm_baseline/ase2022_stage3_anchored_prompts.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="Benchmark/results/ase2022_llm_baseline",
    )
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-delay-seconds", type=float, default=2.0)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing predictions and restart each selected model.",
    )
    parser.add_argument("--responses-path", default=None)
    parser.add_argument(
        "--http-client",
        choices=("urllib", "powershell"),
        default=None,
    )
    args = parser.parse_args()

    _load_env_file(REPO_ROOT / ".env")
    config = resolve_run_config(
        os.environ,
        base_url_override=args.base_url,
    )

    examples = load_ase2022_examples()
    taxonomy = build_taxonomy(examples)
    output_dir = Path(args.output_dir)
    models = args.models if args.models else config["models"]
    for model in models:
        slug = model_slug(model)
        predictions_path = output_dir / f"ase2022_stage3_llm_predictions_{slug}.jsonl"
        metrics_path = output_dir / f"ase2022_stage3_llm_metrics_{slug}.json"
        metrics = run_llm_prompts(
            prompts_path=args.prompts_path,
            examples=examples,
            taxonomy=taxonomy,
            predictions_path=predictions_path,
            metrics_path=metrics_path,
            model=model,
            api_key=config["api_key"],
            base_url=config["base_url"],
            wire_api=config["wire_api"],
            responses_path=args.responses_path or config["responses_path"],
            http_client=args.http_client or config["http_client"],
            limit=args.limit,
            sleep_seconds=args.sleep_seconds,
            resume=not args.no_resume,
            max_retries=args.max_retries,
            retry_delay_seconds=args.retry_delay_seconds,
        )
        print(metrics)


if __name__ == "__main__":
    main()
