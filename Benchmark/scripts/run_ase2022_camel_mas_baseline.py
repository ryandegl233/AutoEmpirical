from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
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
from Benchmark.src.llm_provider_config import (  # noqa: E402
    GEMINI_PROVIDER_ALIASES,
    canonical_provider,
    resolve_gemini_config,
)
from Benchmark.src.mas_rationale import (  # noqa: E402
    CONTRACT_VERSION,
    ExplanationMode,
    validate_mode,
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
    evidence_builder: Callable[[dict[str, str]], dict[str, str]] | None = None
    default_explanation_mode: ExplanationMode = "label_only"
    expected_paper_id: str | None = None


@dataclass(frozen=True)
class ExperimentPaths:
    cohort: Path
    taxonomy: Path
    output_dir: Path
    single_stage2_metrics: Path
    single_stage3_metrics: Path


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
    provider = canonical_provider(provider)
    if provider == "gemini":
        return resolve_gemini_config(env, base_url_override=base_url_override)
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
        raise ValueError("provider must be proxy, deepseek, or gemini")
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
    provider = canonical_provider(provider)
    if model_override:
        return model_override
    if provider == "gemini":
        raise SystemExit("--provider gemini requires an explicit --model")
    if provider == "deepseek":
        return "deepseek-v4-flash"
    if provider == "proxy":
        return DEFAULT_MODEL
    raise ValueError("provider must be proxy, deepseek, or gemini")


def validate_max_turns(value: int) -> int:
    if value <= 0:
        raise ValueError("max_turns must be positive")
    return value


def society_artifact_prefix(
    society_mode: SocietyMode,
    *,
    study_slug: str = "ase2022",
    explanation_mode: ExplanationMode = "label_only",
) -> str:
    society_architecture(society_mode)
    validate_mode(explanation_mode)
    if society_mode == "native":
        prefix = f"{study_slug}_camel_society"
    else:
        prefix = f"{study_slug}_camel_evidence_anchored"
    if explanation_mode == "evidence_rationale":
        prefix += f"_rationale_v{CONTRACT_VERSION}"
    return prefix


def make_runner_factories(
    model: str,
    api_key: str,
    base_url: str,
    *,
    temperature: float | None,
    max_retries: int,
    timeout: float | None,
    society_mode: SocietyMode = DEFAULT_SOCIETY_MODE,
    capture_trace: bool = False,
):
    options = {
        "temperature": temperature,
        "max_retries": max_retries,
        "timeout": timeout,
    }
    if capture_trace:
        options["capture_trace"] = True
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
    from Benchmark.src.annotation_contracts import annotation_mode

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not all(
        isinstance(payload.get(key), list)
        and (payload[key] or annotation_mode(payload, key) == "free_text")
        for key in ("symptom", "root_cause")
    ):
        raise ValueError("taxonomy must contain non-empty symptom and root_cause lists")
    if "annotation_modes" in payload:
        for dimension in ("symptom", "root_cause"):
            mode = annotation_mode(payload, dimension)
            labels = payload[dimension]
            if not all(isinstance(label, str) and label.strip() for label in labels):
                raise ValueError(f"taxonomy {dimension} labels must be nonblank strings")
            if len(labels) != len(set(labels)):
                raise ValueError(f"taxonomy {dimension} contains duplicate labels")
            if mode == "constant" and len(labels) != 1:
                raise ValueError(f"taxonomy {dimension} constant mode requires one label")
            if mode == "free_text" and labels:
                raise ValueError(f"taxonomy {dimension} free_text mode must not enumerate descriptions")
    return payload


def _load_optional_json(path: str | Path) -> dict[str, object] | None:
    source = Path(path)
    if not source.exists():
        return None
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {source}")
    return payload


def _preflight_rationale_outputs(
    output_dir: Path,
    *,
    artifact_prefix: str,
    slug: str,
    stage: str,
    run_id: str,
) -> None:
    stages = ("stage2", "stage3") if stage == "all" else (stage,)
    outputs = [
        output_dir / f"{artifact_prefix}_{current}_{kind}_{slug}.{extension}"
        for current in stages
        for kind, extension in (("predictions", "jsonl"), ("metrics", "json"))
    ]
    if stage == "all":
        outputs.append(output_dir / f"{artifact_prefix}_end_to_end_metrics_{slug}.json")
    for path in outputs:
        if not path.exists():
            continue
        message = (
            f"Existing rationale artifact {path} belongs to a different or unknown "
            "run identity; use a new --output-dir to preserve the earlier run."
        )
        try:
            content = path.read_text(encoding="utf-8")
            payloads = (
                [json.loads(line) for line in content.splitlines() if line.strip()]
                if path.suffix == ".jsonl" else [json.loads(content)]
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(message) from error
        if any(not isinstance(row, dict) or row.get("run_id") != run_id for row in payloads):
            raise ValueError(message)


def _write_rationale_manifest(
    args: argparse.Namespace,
    paths: ExperimentPaths,
    *,
    artifact_prefix: str,
    slug: str,
    backend_id: str,
    temperature: float | None,
    stage2_path: Path,
) -> tuple[Path, str]:
    input_paths = {
        "cohort": paths.cohort,
        "taxonomy": paths.taxonomy,
        "single_llm_stage2_metrics": paths.single_stage2_metrics,
        "single_llm_stage3_metrics": paths.single_stage3_metrics,
    }
    if args.stage == "stage3":
        input_paths["stage2_predictions"] = stage2_path
    inputs = {
        name: {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()
            if path.is_file() else None,
        }
        for name, path in input_paths.items()
    }
    source_paths = [
        "Benchmark/scripts/run_ase2022_camel_mas_baseline.py",
        "Benchmark/scripts/run_issta2024_camel_mas_baseline.py",
        "Benchmark/scripts/run_ase2022_llm_baseline.py",
        "Benchmark/src/ase2022_camel_mas_baseline.py",
        "Benchmark/src/ase2022_llm_baseline.py",
        "Benchmark/src/ase2022_stage2_filter_baseline.py",
        "Benchmark/src/issta2024_bugs_in_pods_baseline.py",
        "Benchmark/src/llm_provider_config.py",
        "Benchmark/src/mas_rationale.py",
        "Benchmark/src/mas_request_trace.py",
        "Benchmark/scripts/run_paper_camel_mas_baseline.py",
        "Benchmark/src/paper_benchmark.py",
        "Benchmark/src/annotation_contracts.py",
        "Benchmark/configs/paper_codebooks_v1.json",
    ]
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            capture_output=True, text=True, check=False, timeout=10,
        )
        git_revision = revision.stdout.strip() if revision.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        git_revision = None
    # Persist an explicit allowlist, never the provider credential dictionary.
    manifest: dict[str, object] = {
        "manifest_version": 1,
        "artifact_prefix": artifact_prefix,
        "config": {
            "stage": args.stage,
            "model": args.model,
            "provider": args.provider,
            "backend_id_sha256": hashlib.sha256(backend_id.encode("utf-8")).hexdigest(),
            "society_mode": args.society_mode,
            "explanation_mode": args.explanation_mode,
            "explanation_contract_version": CONTRACT_VERSION,
            "temperature": temperature,
            "max_turns": args.max_turns,
            "max_retries": args.max_retries,
            "timeout": args.timeout,
            "require_valid_json": args.require_valid_json,
            "record_ids": args.record_ids,
            "limit": args.limit,
        },
        "inputs": inputs,
        "code": {
            "git_revision": git_revision,
            "source_sha256": {
                source: hashlib.sha256((REPO_ROOT / source).read_bytes()).hexdigest()
                for source in source_paths if (REPO_ROOT / source).is_file()
            },
        },
    }
    run_id = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    manifest["run_id"] = run_id
    _preflight_rationale_outputs(
        paths.output_dir, artifact_prefix=artifact_prefix, slug=slug,
        stage=args.stage, run_id=run_id,
    )
    path = paths.output_dir / f"{artifact_prefix}_run_manifest_{slug}_{run_id[:16]}.json"
    _write_json(path, manifest)
    return path, run_id


def _bind_run_identity(config_hash: str, run_id: str | None) -> str:
    if run_id is None:
        return config_hash
    payload = {"stage_config_hash": config_hash, "run_id": run_id}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def resolve_experiment_paths(
    args: argparse.Namespace,
    profile: CamelMasCliProfile,
    *,
    model: str,
) -> ExperimentPaths:
    comparison_dir_value = getattr(args, "comparison_dir", None)
    if not comparison_dir_value:
        slug = model_slug(model)
        return ExperimentPaths(
            cohort=Path(args.cohort_path or profile.default_cohort_path),
            taxonomy=Path(args.taxonomy_path or profile.default_taxonomy_path),
            output_dir=Path(args.output_dir or profile.default_output_dir),
            single_stage2_metrics=Path(
                args.single_llm_stage2_metrics
                or profile.default_single_stage2_metrics.format(slug=slug)
            ),
            single_stage3_metrics=Path(
                args.single_llm_stage3_metrics
                or profile.default_single_stage3_metrics.format(slug=slug)
            ),
        )

    conflicting = [
        option
        for option, value in (
            ("--cohort-path", args.cohort_path),
            ("--taxonomy-path", args.taxonomy_path),
            ("--output-dir", args.output_dir),
            ("--single-llm-stage2-metrics", args.single_llm_stage2_metrics),
            ("--single-llm-stage3-metrics", args.single_llm_stage3_metrics),
        )
        if value is not None
    ]
    if conflicting:
        raise ValueError(
            "--comparison-dir cannot be combined with " + ", ".join(conflicting)
        )

    comparison_dir = Path(comparison_dir_value)
    slug = model_slug(model)
    output_name = (
        "mas_evidence_anchored"
        if args.society_mode == "evidence_anchored"
        else "mas_native"
    )
    paths = ExperimentPaths(
        cohort=comparison_dir / "ase2022_issue_only_holdout_cohort.csv",
        taxonomy=comparison_dir / "ase2022_issue_only_holdout_taxonomy.json",
        output_dir=comparison_dir / output_name,
        single_stage2_metrics=(
            comparison_dir
            / "single_llm"
            / f"ase2022_stage2_filter_metrics_{slug}.json"
        ),
        single_stage3_metrics=(
            comparison_dir
            / "single_llm"
            / f"ase2022_stage3_llm_metrics_{slug}.json"
        ),
    )
    for label, path in (
        ("cohort", paths.cohort),
        ("taxonomy", paths.taxonomy),
        ("Single LLM Stage 2 metrics", paths.single_stage2_metrics),
        ("Single LLM Stage 3 metrics", paths.single_stage3_metrics),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"missing {label}: {path}")

    cohort = load_unified_cohort(paths.cohort)
    record_ids = [str(row.get("record_id", "")) for row in cohort]
    if len(cohort) != 100:
        raise ValueError(f"comparison cohort must contain 100 rows, got {len(cohort)}")
    if not all(record_ids) or len(set(record_ids)) != 100:
        raise ValueError("comparison cohort must contain 100 unique non-empty record IDs")

    for label, path, expected_n in (
        ("Stage 2", paths.single_stage2_metrics, 100),
        ("Stage 3", paths.single_stage3_metrics, 50),
    ):
        metrics = _load_optional_json(path)
        if metrics is None:
            raise FileNotFoundError(f"missing Single LLM {label} metrics: {path}")
        if metrics.get("n") != expected_n:
            raise ValueError(
                f"Single LLM {label} metrics n={expected_n} required, "
                f"got {metrics.get('n')!r}"
            )
        if metrics.get("model") != model:
            raise ValueError(
                f"Single LLM {label} metrics model must be {model!r}, "
                f"got {metrics.get('model')!r}"
            )
    return paths


def build_parser(
    profile: CamelMasCliProfile = ASE2022_PROFILE,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=profile.description)
    parser.add_argument("--stage", choices=("stage2", "stage3", "all"), default="all")
    parser.add_argument(
        "--provider",
        choices=("proxy", "deepseek", *sorted(GEMINI_PROVIDER_ALIASES)),
        default=profile.default_provider,
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    if profile.study_slug == "ase2022":
        parser.add_argument(
            "--comparison-dir",
            default=None,
            help=(
                "Resolve an ASE issue-only cohort, taxonomy, matching Single "
                "LLM controls, and isolated MAS output from one directory."
            ),
        )
    parser.add_argument(
        "--cohort-path",
        default=None,
    )
    parser.add_argument(
        "--taxonomy-path",
        default=None,
    )
    parser.add_argument("--output-dir", default=None)
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
    parser.add_argument(
        "--explanation-mode",
        choices=("label_only", "evidence_rationale"),
        default=profile.default_explanation_mode,
        help=(
            "Opt in to brief label rationales and cited source evidence. "
            "Uses separate rationale_v1 artifacts; label_only preserves the baseline."
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
            "Require final predictions to pass the stage schema, taxonomy, and "
            "selected rationale contract. Rationale mode saves a failed record "
            "before aborting."
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
    args.provider = canonical_provider(args.provider)
    args.max_turns = validate_max_turns(args.max_turns)

    _load_env_file(REPO_ROOT / ".env")
    args.model = resolve_model(args.provider, args.model)
    config = resolve_run_config(os.environ, args.base_url, provider=args.provider)
    paths = resolve_experiment_paths(args, profile, model=args.model)
    cohort = load_unified_cohort(paths.cohort)
    if profile.expected_paper_id is not None:
        if not cohort or any(row.get("paper_id") != profile.expected_paper_id for row in cohort):
            raise ValueError("cohort paper_id does not match the selected paper profile")
    taxonomy = _load_taxonomy(paths.taxonomy)
    temperature = None if args.no_temperature else args.temperature
    trace_options = (
        {"capture_trace": True} if args.explanation_mode == "evidence_rationale" else {}
    )
    society_factory, finalizer_factory = make_runner_factories(
        args.model,
        config["api_key"],
        config["base_url"],
        temperature=temperature,
        max_retries=args.max_retries,
        timeout=args.timeout,
        society_mode=args.society_mode,
        **trace_options,
    )
    configure_camel_logging(args.show_camel_warnings)
    output_dir = paths.output_dir
    slug = model_slug(args.model)
    artifact_prefix = society_artifact_prefix(
        args.society_mode,
        study_slug=profile.study_slug,
        explanation_mode=args.explanation_mode,
    )
    architecture = society_architecture(args.society_mode)
    single_stage2_path = paths.single_stage2_metrics
    single_stage3_path = paths.single_stage3_metrics
    experiment_inputs: dict[str, object] = {
        "cohort_path": str(paths.cohort),
        "taxonomy_path": str(paths.taxonomy),
        "single_llm_stage2_metrics_path": str(single_stage2_path),
        "single_llm_stage3_metrics_path": str(single_stage3_path),
    }

    stage2_path = output_dir / f"{artifact_prefix}_stage2_predictions_{slug}.jsonl"
    stage3_path = output_dir / f"{artifact_prefix}_stage3_predictions_{slug}.jsonl"
    stage2_rows: list[dict[str, object]] = []
    stage3_rows: list[dict[str, object]] = []
    backend_id = f"{args.provider}:{config['base_url']}"
    run_id = None
    if args.explanation_mode == "evidence_rationale":
        manifest_path, run_id = _write_rationale_manifest(
            args,
            paths,
            artifact_prefix=artifact_prefix,
            slug=slug,
            backend_id=backend_id,
            temperature=temperature,
            stage2_path=stage2_path,
        )
        experiment_inputs.update({
            "explanation_mode": args.explanation_mode,
            "explanation_contract_version": CONTRACT_VERSION,
            "run_manifest_path": str(manifest_path),
            "run_id": run_id,
        })

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
            explanation_mode=args.explanation_mode,
        )
        stage2_hash = _bind_run_identity(stage2_hash, run_id)

        def run_stage2(record: dict[str, str]) -> dict[str, object]:
            result = run_roleplaying_society_record(
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
                explanation_mode=args.explanation_mode,
                model_evidence_builder=profile.evidence_builder,
            )
            if run_id is not None:
                result["run_id"] = run_id
            return result

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
            explanation_mode=args.explanation_mode,
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
                **experiment_inputs,
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
            explanation_mode=args.explanation_mode,
        )

        stage3_hash = _bind_run_identity(stage3_hash, run_id)

        def run_stage3(record: dict[str, str]) -> dict[str, object]:
            result = run_roleplaying_society_record(
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
                explanation_mode=args.explanation_mode,
                model_evidence_builder=profile.evidence_builder,
            )
            if run_id is not None:
                result["run_id"] = run_id
            return result

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
            explanation_mode=args.explanation_mode,
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
                **experiment_inputs,
                "final": evaluate_stage3(
                    positive_records, positive_stage3_rows, source="final", taxonomy=taxonomy
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
                **experiment_inputs,
                **evaluate_end_to_end(stage2_records, stage2_rows, stage3_rows, taxonomy=taxonomy),
            },
        )


def main() -> None:
    run_profile(ASE2022_PROFILE)


if __name__ == "__main__":
    main()
