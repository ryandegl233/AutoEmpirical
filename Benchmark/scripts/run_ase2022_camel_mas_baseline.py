from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TextIO


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Benchmark.scripts.run_ase2022_llm_baseline import (  # noqa: E402
    _load_env_file,
    model_slug,
)
from Benchmark.src.ase2022_camel_mas_baseline import (  # noqa: E402
    DEFAULT_MAX_TURNS,
    DEFAULT_MODEL,
    DEFAULT_SOCIETY_MODE,
    SocietyMode,
    TaskBuilder,
    build_config_hash,
    build_society_task,
    evaluate_end_to_end,
    evaluate_society_diagnostics,
    evaluate_stage2,
    evaluate_stage3,
    load_unified_cohort,
    make_camel_agent_factory,
    make_camel_society_factory,
    run_roleplaying_society_record,
    run_stage_records,
    select_requested_records,
    select_stage3_records,
    society_architecture,
)


@dataclass(frozen=True)
class CamelMasCliProfile:
    study_slug: str
    description: str
    default_provider: str
    default_cohort_path: str
    default_taxonomy_path: str
    default_output_dir: str
    default_single_stage2_metrics: str
    default_single_stage3_metrics: str
    task_builder: TaskBuilder
    default_require_valid_json: bool = False


ASE2022_PROFILE = CamelMasCliProfile(
    study_slug="ase2022",
    description="Run the ASE2022 CAMEL-AI RolePlaying Society baseline.",
    default_provider="proxy",
    default_cohort_path=(
        "Benchmark/results/ase2022_camel_mas_baseline/"
        "ase2022_camel_mas_cohort.csv"
    ),
    default_taxonomy_path=(
        "Benchmark/results/ase2022_camel_mas_baseline/"
        "ase2022_camel_mas_taxonomy.json"
    ),
    default_output_dir="Benchmark/results/ase2022_camel_mas_baseline",
    default_single_stage2_metrics=(
        "Benchmark/results/ase2022_camel_mas_baseline/single_llm_control/"
        "ase2022_stage2_filter_metrics_{slug}.json"
    ),
    default_single_stage3_metrics=(
        "Benchmark/results/ase2022_llm_baseline/paper_models_50/"
        "ase2022_stage3_llm_metrics_{slug}.json"
    ),
    task_builder=build_society_task,
)


def resolve_run_config(
    env: dict[str, str],
    base_url_override: str | None = None,
    provider: str = "proxy",
) -> dict[str, str]:
    if provider == "deepseek":
        api_key = env.get("DEEPSEEK_API_KEY") or env.get("DEEPSEEK_API")
        if not api_key:
            raise SystemExit("Missing DEEPSEEK_API_KEY in .env or environment")
        return {
            "base_url": (
                base_url_override
                or env.get("DEEPSEEK_BASE_URL")
                or "https://api.deepseek.com"
            ),
            "api_key": api_key,
        }
    if provider != "proxy":
        raise ValueError("provider must be proxy or deepseek")
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
    return {"base_url": base_url, "api_key": api_key}


def resolve_model(provider: str, model_override: str | None) -> str:
    if model_override:
        return model_override
    if provider == "deepseek":
        return "deepseek-v4-flash"
    if provider == "proxy":
        return DEFAULT_MODEL
    raise ValueError("provider must be proxy or deepseek")


def validate_max_turns(value: int) -> int:
    if value <= 0:
        raise ValueError("max_turns must be positive")
    return value


def society_artifact_prefix(
    society_mode: SocietyMode,
    *,
    study_slug: str = "ase2022",
) -> str:
    society_architecture(society_mode)
    if society_mode == "native":
        return f"{study_slug}_camel_society"
    return f"{study_slug}_camel_evidence_anchored"


def make_runner_factories(
    model: str,
    api_key: str,
    base_url: str,
    *,
    temperature: float | None,
    max_retries: int,
    timeout: float | None,
    society_mode: SocietyMode = DEFAULT_SOCIETY_MODE,
):
    options = {
        "temperature": temperature,
        "max_retries": max_retries,
        "timeout": timeout,
    }
    return (
        make_camel_society_factory(
            model,
            api_key,
            base_url,
            society_mode=society_mode,
            **options,
        ),
        make_camel_agent_factory(
            model,
            api_key,
            base_url,
            **options,
        ),
    )


class CamelNoiseFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno != logging.WARNING:
            return True
        message = record.getMessage()
        unknown_context = (
            message.startswith("Unknown model '")
            and "context window size not defined" in message
            and "Defaulting to 999_999_999" in message
        )
        recoverable_format = (
            message.startswith("Format validation error:")
            and "Attempting fallback with JSON format" in message
        )
        return not (unknown_context or recoverable_format)


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    if seconds < 60:
        return f"{seconds}s"
    minutes, remaining_seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m{remaining_seconds:02d}s"
    hours, remaining_minutes = divmod(minutes, 60)
    return f"{hours}h{remaining_minutes:02d}m"


class ConsoleProgress:
    def __init__(
        self,
        label: str,
        *,
        stream: TextIO = sys.stderr,
        clock: Callable[[], float] = time.monotonic,
        width: int = 30,
    ) -> None:
        self.label = label
        self.stream = stream
        self.clock = clock
        self.width = width
        self.started_at = clock()
        self.live_completed = 0
        self.last_length = 0

    def start(self, total: int) -> None:
        self(0, total, "", "starting")

    def __call__(
        self,
        current: int,
        total: int,
        record_id: str,
        status: str,
    ) -> None:
        if status == "completed":
            self.live_completed += 1
        fraction = current / total if total else 1.0
        filled = min(self.width, int(round(self.width * fraction)))
        bar = "█" * filled + "░" * (self.width - filled)
        elapsed = self.clock() - self.started_at
        if current >= total:
            eta_text = "ETA 0s"
        elif self.live_completed:
            remaining = total - current
            eta_text = f"ETA {_format_duration(elapsed / self.live_completed * remaining)}"
        else:
            eta_text = "ETA --"
        short_id = record_id.rsplit(":", 1)[-1] if record_id else ""
        line = (
            f"{self.label} [{bar}] {current}/{total} "
            f"{fraction * 100:5.1f}%  {eta_text}  {status} {short_id}"
        ).rstrip()
        padded = line.ljust(self.last_length)
        self.last_length = len(line)
        ending = "\n" if current >= total else ""
        self.stream.write("\r" + padded + ending)
        self.stream.flush()


def configure_camel_logging(show_camel_warnings: bool = False) -> None:
    if show_camel_warnings:
        return
    noise_filter = CamelNoiseFilter()
    for handler in logging.getLogger().handlers:
        if not any(isinstance(item, CamelNoiseFilter) for item in handler.filters):
            handler.addFilter(noise_filter)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _load_taxonomy(path: str | Path) -> dict[str, list[str]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not all(
        isinstance(payload.get(key), list) and payload[key]
        for key in ("symptom", "root_cause")
    ):
        raise ValueError("taxonomy must contain non-empty symptom and root_cause lists")
    return payload


def _load_optional_json(path: str | Path) -> dict[str, object] | None:
    source = Path(path)
    if not source.exists():
        return None
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {source}")
    return payload


def build_parser(
    profile: CamelMasCliProfile = ASE2022_PROFILE,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=profile.description)
    parser.add_argument("--stage", choices=("stage2", "stage3", "all"), default="all")
    parser.add_argument(
        "--provider",
        choices=("proxy", "deepseek"),
        default=profile.default_provider,
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument(
        "--cohort-path",
        default=profile.default_cohort_path,
    )
    parser.add_argument(
        "--taxonomy-path",
        default=profile.default_taxonomy_path,
    )
    parser.add_argument("--output-dir", default=profile.default_output_dir)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--record-ids",
        nargs="*",
        default=None,
        help="Run an exact record subset, primarily for reproducible smoke tests.",
    )
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    parser.add_argument(
        "--society-mode",
        choices=("native", "evidence_anchored"),
        default=DEFAULT_SOCIETY_MODE,
        help=(
            "Use evidence_anchored by default, or native to reproduce the "
            "unmodified CAMEL RolePlaying Society baseline."
        ),
    )
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--no-temperature", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--show-camel-warnings", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    validity = parser.add_mutually_exclusive_group()
    validity.add_argument(
        "--require-valid-json",
        dest="require_valid_json",
        action="store_true",
        help=(
            "Abort before writing any record whose final prediction does not "
            "pass the stage schema and taxonomy."
        ),
    )
    validity.add_argument(
        "--allow-invalid",
        dest="require_valid_json",
        action="store_false",
        help="Allow invalid prediction rows for diagnostic runs.",
    )
    parser.set_defaults(
        require_valid_json=profile.default_require_valid_json
    )
    parser.add_argument("--single-llm-stage2-metrics", default=None)
    parser.add_argument("--single-llm-stage3-metrics", default=None)
    return parser


def run_profile(
    profile: CamelMasCliProfile,
    argv: list[str] | None = None,
) -> None:
    args = build_parser(profile).parse_args(argv)
    args.max_turns = validate_max_turns(args.max_turns)

    _load_env_file(REPO_ROOT / ".env")
    args.model = resolve_model(args.provider, args.model)
    config = resolve_run_config(os.environ, args.base_url, provider=args.provider)
    cohort = load_unified_cohort(args.cohort_path)
    taxonomy = _load_taxonomy(args.taxonomy_path)
    temperature = None if args.no_temperature else args.temperature
    society_factory, finalizer_factory = make_runner_factories(
        args.model,
        config["api_key"],
        config["base_url"],
        temperature=temperature,
        max_retries=args.max_retries,
        timeout=args.timeout,
        society_mode=args.society_mode,
    )
    configure_camel_logging(args.show_camel_warnings)
    output_dir = Path(args.output_dir)
    slug = model_slug(args.model)
    artifact_prefix = society_artifact_prefix(
        args.society_mode,
        study_slug=profile.study_slug,
    )
    architecture = society_architecture(args.society_mode)
    single_stage2_path = Path(
        args.single_llm_stage2_metrics
        or profile.default_single_stage2_metrics.format(slug=slug)
    )
    single_stage3_path = Path(
        args.single_llm_stage3_metrics
        or profile.default_single_stage3_metrics.format(slug=slug)
    )

    stage2_path = output_dir / f"{artifact_prefix}_stage2_predictions_{slug}.jsonl"
    stage3_path = output_dir / f"{artifact_prefix}_stage3_predictions_{slug}.jsonl"
    stage2_rows: list[dict[str, object]] = []
    stage3_rows: list[dict[str, object]] = []
    backend_id = f"{args.provider}:{config['base_url']}"

    if args.stage in {"stage2", "all"}:
        stage2_records = select_requested_records(cohort, args.record_ids, args.limit)
        stage2_hash = build_config_hash(
            args.model,
            "stage2",
            stage2_records,
            taxonomy,
            temperature,
            backend_id=backend_id,
            max_turns=args.max_turns,
            society_mode=args.society_mode,
            require_valid_json=args.require_valid_json,
        )

        def run_stage2(record: dict[str, str]) -> dict[str, object]:
            return run_roleplaying_society_record(
                record,
                stage="stage2",
                taxonomy=taxonomy,
                model=args.model,
                society_factory=society_factory,
                finalizer_factory=finalizer_factory,
                finalizer_max_retries=args.max_retries,
                max_turns=args.max_turns,
                config_hash=stage2_hash,
                backend_id=backend_id,
                society_mode=args.society_mode,
                task_builder=profile.task_builder,
            )

        stage2_progress = None if args.no_progress else ConsoleProgress("Stage 2")
        if stage2_progress is not None:
            stage2_progress.start(len(stage2_records))
        stage2_rows = run_stage_records(
            stage2_records,
            stage2_path,
            stage="stage2",
            model=args.model,
            config_hash=stage2_hash,
            record_runner=run_stage2,
            resume=not args.no_resume,
            progress_callback=stage2_progress,
            require_valid_json=args.require_valid_json,
            taxonomy=taxonomy,
        )
        _write_json(
            output_dir / f"{artifact_prefix}_stage2_metrics_{slug}.json",
            {
                "task": f"{artifact_prefix}_stage2",
                "architecture": architecture,
                "society_mode": args.society_mode,
                "model": args.model,
                "provider": args.provider,
                "base_url": config["base_url"],
                "config_hash": stage2_hash,
                "max_turns": args.max_turns,
                "require_valid_json": args.require_valid_json,
                "final": evaluate_stage2(stage2_records, stage2_rows, source="final"),
                "http_single_llm_control": {
                    "metrics_path": str(single_stage2_path),
                    "metrics": _load_optional_json(single_stage2_path),
                },
                "society": evaluate_society_diagnostics(stage2_rows),
            },
        )

    if args.stage in {"stage3", "all"}:
        if not stage2_rows and stage2_path.exists():
            stage2_rows = [
                json.loads(line)
                for line in stage2_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        stage3_source_records = stage2_records if args.stage == "all" else cohort
        stage3_records = select_stage3_records(stage3_source_records, stage2_rows)
        if args.stage == "stage3":
            stage3_records = select_requested_records(
                stage3_records, args.record_ids, args.limit
            )
        stage3_hash = build_config_hash(
            args.model,
            "stage3",
            stage3_records,
            taxonomy,
            temperature,
            backend_id=backend_id,
            max_turns=args.max_turns,
            society_mode=args.society_mode,
            require_valid_json=args.require_valid_json,
        )

        def run_stage3(record: dict[str, str]) -> dict[str, object]:
            return run_roleplaying_society_record(
                record,
                stage="stage3",
                taxonomy=taxonomy,
                model=args.model,
                society_factory=society_factory,
                finalizer_factory=finalizer_factory,
                finalizer_max_retries=args.max_retries,
                max_turns=args.max_turns,
                config_hash=stage3_hash,
                backend_id=backend_id,
                society_mode=args.society_mode,
                task_builder=profile.task_builder,
            )

        stage3_progress = None if args.no_progress else ConsoleProgress("Stage 3")
        if stage3_progress is not None:
            stage3_progress.start(len(stage3_records))
        stage3_rows = run_stage_records(
            stage3_records,
            stage3_path,
            stage="stage3",
            model=args.model,
            config_hash=stage3_hash,
            record_runner=run_stage3,
            resume=not args.no_resume,
            progress_callback=stage3_progress,
            require_valid_json=args.require_valid_json,
            taxonomy=taxonomy,
        )
        evaluated_ids = {row["record_id"] for row in stage3_records}
        positive_records = [
            row
            for row in cohort
            if row.get("decision") == "accepted_fault"
            and row["record_id"] in evaluated_ids
        ]
        positive_ids = {row["record_id"] for row in positive_records}
        positive_stage3_rows = [
            row for row in stage3_rows if row.get("record_id") in positive_ids
        ]
        _write_json(
            output_dir / f"{artifact_prefix}_stage3_metrics_{slug}.json",
            {
                "task": f"{artifact_prefix}_stage3",
                "architecture": architecture,
                "society_mode": args.society_mode,
                "model": args.model,
                "provider": args.provider,
                "base_url": config["base_url"],
                "config_hash": stage3_hash,
                "max_turns": args.max_turns,
                "require_valid_json": args.require_valid_json,
                "final": evaluate_stage3(
                    positive_records, positive_stage3_rows, source="final"
                ),
                "http_single_llm_control": {
                    "metrics_path": str(single_stage3_path),
                    "metrics": _load_optional_json(single_stage3_path),
                },
                "society": evaluate_society_diagnostics(stage3_rows),
            },
        )

    if args.stage == "all":
        _write_json(
            output_dir / f"{artifact_prefix}_end_to_end_metrics_{slug}.json",
            {
                "task": f"{artifact_prefix}_end_to_end",
                "architecture": architecture,
                "society_mode": args.society_mode,
                "model": args.model,
                "provider": args.provider,
                "base_url": config["base_url"],
                "max_turns": args.max_turns,
                "require_valid_json": args.require_valid_json,
                **evaluate_end_to_end(stage2_records, stage2_rows, stage3_rows),
            },
        )


def main() -> None:
    run_profile(ASE2022_PROFILE)


if __name__ == "__main__":
    main()
