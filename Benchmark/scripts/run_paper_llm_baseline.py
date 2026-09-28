"""Run Stage 2 filtering and native Stage 3 annotation for a prepared paper."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Benchmark.scripts.run_ase2022_camel_mas_baseline import _load_taxonomy, resolve_model, resolve_run_config
from Benchmark.scripts.run_ase2022_llm_baseline import _load_env_file, model_slug
from Benchmark.scripts.run_paper_camel_mas_baseline import DEFAULT_PREPARED_ROOT, DEFAULT_RUN_ROOT
from Benchmark.src.ase2022_llm_baseline import run_llm_prompts
from Benchmark.src.ase2022_stage2_filter_baseline import run_filter_prompts
from Benchmark.src.llm_provider_config import GEMINI_PROVIDER_ALIASES, canonical_provider


def build_parser() -> argparse.ArgumentParser:
    from Benchmark.src.paper_benchmark import PAPER_PROFILES

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=sorted(PAPER_PROFILES), required=True)
    parser.add_argument("--prepared-root", default=DEFAULT_PREPARED_ROOT)
    parser.add_argument("--stage", choices=("stage2", "stage3", "all"), default="all")
    parser.add_argument("--provider", choices=("proxy", "deepseek", *sorted(GEMINI_PROVIDER_ALIASES)), default="gemini")
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-delay-seconds", type=float, default=2.0)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--http-client", choices=("urllib", "powershell"), default="urllib")
    parser.add_argument("--no-resume", action="store_true")
    return parser


def _load_rows(path: Path, paper_id: str) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    ids = [row.get("record_id") for row in rows]
    if not rows or not all(ids) or len(ids) != len(set(ids)):
        raise ValueError("prepared sample requires unique nonempty record IDs")
    if any(row.get("paper_id") != paper_id for row in rows):
        raise ValueError("prepared sample paper_id does not match the selected paper")
    return rows


def _prepare_manifest(path: Path, inputs: list[Path], outputs: list[Path], config: dict) -> dict:
    manifest = {
        "manifest_version": 1,
        "config": config,
        "inputs": {str(item.resolve()): hashlib.sha256(item.read_bytes()).hexdigest() for item in inputs},
        "code_sha256": {
            name: hashlib.sha256((REPO_ROOT / name).read_bytes()).hexdigest()
            for name in (
                "Benchmark/scripts/run_paper_llm_baseline.py",
                "Benchmark/src/ase2022_llm_baseline.py",
                "Benchmark/src/ase2022_stage2_filter_baseline.py",
                "Benchmark/src/paper_benchmark.py",
                "Benchmark/src/annotation_contracts.py",
                "Benchmark/configs/paper_codebooks_v1.json",
            )
        },
    }
    manifest["run_id"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode("utf-8")).hexdigest()
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("run_id") != manifest["run_id"]:
            raise ValueError("Existing SingleLLM run has different inputs or configuration; use a new --output-dir")
    elif any(item.exists() for item in outputs):
        raise ValueError("Existing SingleLLM artifacts have no matching manifest; use a new --output-dir")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> None:
    from Benchmark.src.paper_benchmark import build_user_prompt, get_paper_profile

    args = build_parser().parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")
    paper = get_paper_profile(args.domain)
    base = Path(args.prepared_root) / paper.domain
    output = Path(args.output_dir) if args.output_dir else DEFAULT_RUN_ROOT / args.domain / "single_llm"
    cohort = _load_rows(base / "cohort.csv", paper.paper_id)
    taxonomy = _load_taxonomy(base / "taxonomy.json")
    sample = _load_rows(base / "stage3_sample.csv", paper.paper_id)
    if set(row["record_id"] for row in sample) != {row["record_id"] for row in cohort if row["decision"] == "accepted_fault"}:
        raise ValueError("Stage 3 sample must equal the accepted subset of the frozen cohort")
    _load_env_file(REPO_ROOT / ".env")
    provider = canonical_provider(args.provider)
    model = resolve_model(provider, args.model)
    config = resolve_run_config(os.environ, args.base_url, provider=provider)
    slug = model_slug(model)
    for stage in (("stage2", "stage3") if args.stage == "all" else (args.stage,)):
        task = f"{paper.domain}_{stage}_{'filter' if stage == 'stage2' else 'llm'}"
        predictions = output / f"{task}_predictions_{slug}.jsonl"
        metrics_path = output / f"{task}_metrics_{slug}.json"
        if args.no_resume and (predictions.exists() or metrics_path.exists()):
            raise ValueError("A fresh SingleLLM run requires a new --output-dir to preserve existing evidence")
        prompts_path = base / f"{stage}_prompts.jsonl"
        prompts = [json.loads(line) for line in prompts_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        expected = cohort if stage == "stage2" else sample
        if [row.get("record_id") for row in prompts] != [row["record_id"] for row in expected]:
            raise ValueError(f"{stage} prompt IDs/order must match the prepared sample")
        if any(row.get("paper_id") != paper.paper_id for row in prompts):
            raise ValueError(f"{stage} prompt paper_id does not match the selected paper")
        if any(prompt.get("user_prompt") != build_user_prompt(row) for prompt, row in zip(prompts, expected)):
            raise ValueError(f"{stage} prompt evidence does not match the current source projection; prepare a new input version")
        if stage == "stage2" and any(prompt.get("ground_truth") != row["decision"] for prompt, row in zip(prompts, cohort)):
            raise ValueError("Stage 2 prompt metadata ground_truth disagrees with the cohort")
        manifest_path = output / f"{task}_run_manifest_{slug}.json"
        manifest = _prepare_manifest(
            manifest_path, [base / "cohort.csv", base / "stage3_sample.csv", base / "taxonomy.json", prompts_path],
            [predictions, metrics_path],
            {"domain": paper.domain, "paper_id": paper.paper_id, "stage": stage, "model": model,
             "provider": provider, "base_url": config["base_url"], "limit": args.limit,
             "max_retries": args.max_retries, "http_client": args.http_client,
             "retry_delay_seconds": args.retry_delay_seconds, "annotation_modes": taxonomy.get("annotation_modes", {})},
        )
        shared = dict(prompts_path=prompts_path, predictions_path=predictions,
                      metrics_path=metrics_path, model=model, api_key=config["api_key"],
                      base_url=config["base_url"], limit=args.limit, resume=not args.no_resume,
                      max_retries=args.max_retries, retry_delay_seconds=args.retry_delay_seconds,
                      sleep_seconds=args.sleep_seconds, http_client=args.http_client, task=task)
        if stage == "stage2":
            metrics = run_filter_prompts(**shared)
        else:
            metrics = run_llm_prompts(examples=sample, taxonomy=taxonomy, **shared)
        metrics.update(run_id=manifest["run_id"], run_manifest_path=str(manifest_path.resolve()),
                       paper_id=paper.paper_id, domain=paper.domain,
                       predictions_sha256=hashlib.sha256(predictions.read_bytes()).hexdigest())
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
