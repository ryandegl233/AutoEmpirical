from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, TextIO
from urllib.parse import urlsplit


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
from Benchmark.src.adaptive_empirical_workflow.agents import (  # noqa: E402
    ModelCallOptions,
    ModelTransportError,
    ModelTransportResponse,
    WORKFLOW_PROMPT_VERSION,
    StructuredModelClient,
    StructuredOutputError,
    StructuredRoleAgents,
    baseline_revision_prompt_hashes,
    baseline_revision_schema_hashes,
    sla_prompt_hashes,
)
from Benchmark.src.adaptive_empirical_workflow.capabilities import (  # noqa: E402
    AnalystRole,
    DEFAULT_STAGE3_TEAM_PERSPECTIVES,
    render_system_prompt,
)
from Benchmark.src.adaptive_empirical_workflow.explainable_tools import (  # noqa: E402
    ExplainableTools,
    read_allowed_source,
)
from Benchmark.src.adaptive_empirical_workflow.baseline_anchor import (  # noqa: E402
    baseline_anchor_hash,
    load_baseline_anchor_bundle,
    load_baseline_anchors,
)
from Benchmark.src.adaptive_empirical_workflow.baseline_preservation import (  # noqa: E402
    BASELINE_PRESERVATION_POLICY_VERSION,
)
from Benchmark.src.adaptive_empirical_workflow.contracts import (  # noqa: E402
    HETEROGENEOUS_REVISION_ASSESSMENT_POLICY,
    RoleModelPolicy,
    Stage3AgentPolicy,
    TASK_SPECIFIC_REVISION_CROSS_POLICY,
    ThinkingMode,
)
from Benchmark.src.adaptive_empirical_workflow.controller import (  # noqa: E402
    Stage2WorkflowConfig,
    Stage3WorkflowConfig,
)
from Benchmark.src.adaptive_empirical_workflow.domains import (  # noqa: E402
    DOMAIN_PROFILES,
    GOLD_FIELDS,
    load_domain_inputs,
)
from Benchmark.src.adaptive_empirical_workflow.experiment import (  # noqa: E402
    ProgressSnapshot,
    evaluate_experiment,
    evaluate_evidence_only_execution,
    run_adaptive_record,
    run_records,
    run_targeted_sla_record,
    select_records,
)
from Benchmark.src.adaptive_empirical_workflow.evaluation import (  # noqa: E402
    stage3_team_diagnostics,
    targeted_sla_diagnostics,
)
from Benchmark.src.adaptive_empirical_workflow.frozen_evidence_runtime import (  # noqa: E402
    FROZEN_EVIDENCE_PROJECTION_POLICY_HASH,
    FROZEN_EVIDENCE_PROJECTION_POLICY_VERSION,
    frozen_evidence_projection_audit,
    load_registered_frozen_evidence_runtime,
    project_frozen_evidence_for_record,
)
from Benchmark.src.adaptive_empirical_workflow.experiment_manifest import (  # noqa: E402
    build_run_identity,
    config_hash as experiment_config_hash,
    prepare_run_directory,
    write_run_manifest,
)
from Benchmark.src.adaptive_empirical_workflow.taxonomy_runtime import (  # noqa: E402
    bound_taxonomy_artifact_manifest_for_domain,
    load_bound_taxonomy_structure_for_domain,
)
from Benchmark.src.adaptive_empirical_workflow.taxonomy_structure import (  # noqa: E402
    official_boundary_card_manifest,
    taxonomy_structure_hash,
)
from Benchmark.src.adaptive_empirical_workflow.stage3_composition import (  # noqa: E402
    REVISION_CANDIDATE_GRAPH_POLICY_VERSION,
    REVISION_CROSS_CHECK_GRAPH_POLICY_VERSION,
)
from Benchmark.src.ase2022_llm_baseline import (  # noqa: E402
    PooledChatCompletionClient,
    RequestCompletionLease,
    call_fresh_chat_completion,
    call_model_with_retries,
    model_transport_error_details,
)
from Benchmark.src.adaptive_empirical_workflow.sla_budget import (  # noqa: E402
    GlobalSlaBudget,
    RecordSlaBudget,
    RoleNetworkRetryBudget,
    SlaBudgetConfig,
    SlaBudgetExhausted,
    SlaRequestDeadline,
)
from Benchmark.src.adaptive_empirical_workflow.sla_controller import (  # noqa: E402
    TargetedSlaController,
)
from Benchmark.src.llm_provider_config import (  # noqa: E402
    GEMINI_DEFAULT_BASE_URL,
    GEMINI_PROVIDER_ALIASES,
    canonical_provider as _canonical_provider,
    normalize_gemini_base_url,
    resolve_gemini_config,
)


MICU_DEFAULT_BASE_URL = "https://www.micuapi.ai/v1"
MICU_BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:149.0) "
    "Gecko/20100101 Firefox/149.0"
)
MICU_OFFICIAL_HOSTS = frozenset({"www.micuapi.ai", "api-slb.micuapi.ai"})
TARGETED_SLA_PROFILE = "targeted-sla30"
TARGETED_SLA_POLICY_VERSION = "targeted-sla30-v1"


def _format_duration(seconds: float) -> str:
    rounded = max(0, int(round(seconds)))
    if rounded < 60:
        return f"{rounded}s"
    minutes, remaining_seconds = divmod(rounded, 60)
    if minutes < 60:
        return f"{minutes}m{remaining_seconds:02d}s"
    hours, remaining_minutes = divmod(minutes, 60)
    return f"{hours}h{remaining_minutes:02d}m"


class ConsoleProgress:
    """Single-line progress bar with elapsed time and rolling ETA."""

    def __init__(
        self,
        *,
        stream: TextIO = sys.stderr,
        width: int = 30,
    ) -> None:
        self._stream = stream
        self._width = width
        self._last_length = 0

    def __call__(self, snapshot: ProgressSnapshot) -> None:
        fraction = snapshot.completed / snapshot.total if snapshot.total else 1.0
        filled = min(self._width, int(round(self._width * fraction)))
        bar = "#" * filled + "-" * (self._width - filled)
        eta = (
            _format_duration(snapshot.eta_seconds)
            if snapshot.eta_seconds is not None
            else "--"
        )
        short_id = snapshot.record_id.rsplit(":", 1)[-1] if snapshot.record_id else ""
        resumed = (
            f" {snapshot.resumed_count} resumed"
            if snapshot.resumed_count and snapshot.status == "starting"
            else ""
        )
        line = (
            f"Adaptive [{bar}] {snapshot.completed}/{snapshot.total} "
            f"{fraction * 100:5.1f}% "
            f"elapsed {_format_duration(snapshot.elapsed_seconds)} "
            f"ETA {eta} {snapshot.status} {short_id}{resumed}"
        ).rstrip()
        self._stream.write("\r" + line.ljust(self._last_length))
        self._last_length = max(self._last_length, len(line))
        if snapshot.status == "completed":
            self._stream.write("\n")
        self._stream.flush()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the evidence-ledger adaptive workflow on the seven supported papers."
        )
    )
    parser.add_argument("--domain", choices=tuple(DOMAIN_PROFILES), required=True)
    parser.add_argument("--stage", choices=("stage2", "stage3", "all"), default="all")
    parser.add_argument(
        "--provider",
        choices=("proxy", "deepseek", "micu", *sorted(GEMINI_PROVIDER_ALIASES)),
        default="deepseek",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--cohort-path", default=None)
    parser.add_argument("--taxonomy-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--split-id", default=None)
    parser.add_argument("--arm-id", default=None)
    parser.add_argument("--run-id", default="run-1")
    parser.add_argument("--split-manifest", default=None)
    parser.add_argument("--module-a", action="store_true", help="Enable source-backed few-shot and decision-table assistance.")
    parser.add_argument("--a-protocol", choices=("v1", "formal-v2", "rules-v4", "tools-v4", "rules-v5", "proof-v6"), default="formal-v2",
                        help="Formal v2 examples by default; use v1 to reproduce the original Module A.")
    parser.add_argument("--global-supervisor", action="store_true", help="v7 case-wide causal controller over v4 teams and challenger (temperature 0.3).")
    parser.add_argument("--module-b", action="store_true", help="Replace both dimension verifier slots with evidence-chain checkers.")
    parser.add_argument("--experiment-two-arm", choices=['E00','E10','E01','E11'], default=None,
                        help="Approved rule/source-graph experiment; shared rule-delivery fix and matched role budgets.")
    parser.add_argument("--source-graph-mode", choices=['graph','flat'], default=None,
                        help="Optional content-matched graph serialization control for E01/E11.")
    parser.add_argument("--supplemental-evidence", default=None,
                        help="Frozen per-case supplementary text and original images; Gemini Stage 3 only.")
    parser.add_argument("--a-examples", default=None, help="Strict method-example JSON bundle; requires --a-example-source.")
    parser.add_argument("--a-example-source", default=None, help="Explicit evidence-only CSV containing just the example records.")
    parser.add_argument(
        "--baseline-preservation",
        action="store_true",
        help="Enable the trusted Baseline-preservation gate for Stage 3.",
    )
    parser.add_argument(
        "--frozen-evidence",
        action="store_true",
        help=(
            "Inject only the module-registered, content-addressed frozen evidence "
            "bundle. Runtime network access remains forbidden."
        ),
    )
    parser.add_argument(
        "--baseline-anchor-path",
        "--baseline-anchor-predictions",
        dest="baseline_anchor_path",
        default=None,
        help="Complete frozen Baseline predictions JSONL artifact.",
    )
    parser.add_argument(
        "--baseline-trust-manifest",
        default=None,
        help="Pre-registered external trust root for the Baseline artifact.",
    )
    parser.add_argument("--ablation-profile", default="full-dual-team")
    parser.add_argument(
        "--execution-profile",
        choices=("full-dual-team", TARGETED_SLA_PROFILE),
        default="full-dual-team",
    )
    parser.add_argument("--budget-class", default="accuracy-priority-2h")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--record-ids", nargs="*", default=None)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--global-deadline-seconds", type=float, default=1680.0)
    parser.add_argument("--record-budget-seconds", type=float, default=240.0)
    parser.add_argument("--request-timeout-seconds", type=float, default=75.0)
    parser.add_argument("--max-schema-retries", type=int, default=2)
    parser.add_argument("--role-max-tokens", type=int, default=1600)
    parser.add_argument(
        "--thinking-profile",
        choices=("off", "causal", "label-thinking", "all-stage3"),
        default="label-thinking",
    )
    parser.add_argument(
        "--thinking-override",
        action="append",
        default=[],
        metavar="ROLE[@TEAM]=MODE",
    )
    parser.add_argument(
        "--max-tokens-override",
        action="append",
        default=[],
        metavar="ROLE[@TEAM]=TOKENS",
    )
    parser.add_argument("--max-network-retries", type=int, default=5)
    parser.add_argument(
        "--provider-max-inflight",
        type=int,
        default=2,
        help="Maximum concurrent HTTP requests to a third-party model provider.",
    )
    parser.add_argument("--retry-delay-seconds", type=float, default=2.0)
    parser.add_argument("--max-retrieval-rounds", type=int, default=2)
    parser.add_argument("--max-specialist-calls", type=int, default=6)
    parser.add_argument("--consensus-confidence", type=float, default=0.8)
    parser.add_argument("--retrieved-at", default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and write the manifest without calling a model.",
    )
    return parser


def _targeted_sla_config(
    args: argparse.Namespace,
    *,
    concurrency_explicit: bool,
    provider_max_inflight_explicit: bool,
) -> SlaBudgetConfig | None:
    """Validate the narrow development-only SLA profile before any model setup."""

    if args.execution_profile != TARGETED_SLA_PROFILE:
        return None
    if args.stage != "stage3":
        raise ValueError("targeted-sla30 requires --stage stage3")
    if not args.record_ids:
        raise ValueError("targeted-sla30 requires explicit --record-ids")
    if not args.baseline_anchor_path:
        raise ValueError("targeted-sla30 requires --baseline-anchor-path")
    if args.baseline_preservation:
        raise ValueError(
            "targeted-sla30 cannot be combined with --baseline-preservation"
        )
    if args.provider not in {"micu", "gemini"}:
        raise ValueError("targeted-sla30 requires --provider micu or gemini")
    if not concurrency_explicit:
        args.concurrency = 4
    if not provider_max_inflight_explicit:
        args.provider_max_inflight = 4
    config = SlaBudgetConfig(
        global_seconds=args.global_deadline_seconds,
        record_seconds=args.record_budget_seconds,
        request_seconds=args.request_timeout_seconds,
    )
    if args.concurrency != 4:
        raise ValueError("targeted-sla30 requires concurrency=4")
    if not 1 <= args.provider_max_inflight <= 4:
        raise ValueError("targeted-sla30 provider_max_inflight must be between 1 and 4")
    return config


def _option_was_supplied(arguments: list[str], option: str) -> bool:
    return any(
        argument == option or argument.startswith(f"{option}=")
        for argument in arguments
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".aew-metrics-",
        suffix=".tmp",
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


def _config_hash(payload: dict[str, Any]) -> str:
    return experiment_config_hash(payload)


def _code_state(repo_root: Path = REPO_ROOT) -> dict[str, str]:
    def run_git(*arguments: str, text: bool = True) -> str | bytes:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=text,
        )
        return completed.stdout.strip()

    revision = run_git("rev-parse", "HEAD")
    assert isinstance(revision, str)
    tracked_diff = run_git("diff", "--binary", "HEAD", text=False)
    assert isinstance(tracked_diff, bytes)
    untracked_output = run_git(
        "ls-files",
        "--others",
        "--exclude-standard",
        "--",
        "Benchmark/src",
        "Benchmark/scripts",
        "Benchmark/configs/paper_codebooks_v1.json",
    )
    assert isinstance(untracked_output, str)
    source_state = hashlib.sha256(tracked_diff)
    for relative in sorted(line for line in untracked_output.splitlines() if line):
        path = repo_root / relative
        source_state.update(relative.replace("\\", "/").encode("utf-8"))
        source_state.update(b"\0")
        source_state.update(path.read_bytes())
        source_state.update(b"\0")
    return {
        "code_revision": revision,
        "dirty_source_state_sha256": source_state.hexdigest(),
    }


def _parse_policy_overrides(
    values: list[str],
    *,
    option: str,
) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        target, separator, setting = value.partition("=")
        if not separator or not target or not setting:
            raise ValueError(f"{option} must use ROLE[@TEAM]=VALUE syntax")
        if target in parsed:
            raise ValueError(f"duplicate {option} target: {target}")
        parsed[target] = setting
    return parsed


def _resolve_agent_policy(args: argparse.Namespace) -> Stage3AgentPolicy:
    policy = Stage3AgentPolicy.from_profile(
        args.thinking_profile,
        default_max_tokens=args.role_max_tokens,
    )
    thinking_values = _parse_policy_overrides(
        args.thinking_override,
        option="--thinking-override",
    )
    token_values = _parse_policy_overrides(
        args.max_tokens_override,
        option="--max-tokens-override",
    )
    targets = set(thinking_values) | set(token_values)
    known_roles = set(policy.profile_policies)
    for target in targets:
        role, marker, team_id = target.partition("@")
        if role not in known_roles:
            raise ValueError(f"unknown Stage 3 policy role: {role}")
        if marker and team_id not in {"A", "B"}:
            raise ValueError(f"unknown Stage 3 policy team: {team_id}")
        if marker and role not in {
            "symptom_analyst",
            "root_cause_analyst",
            "joint_anchor",
            "symptom_verifier",
            "root_cause_verifier",
            "causal_consistency_checker",
            "baseline_revision_assessment",
            "baseline_revision_consistency",
        }:
            raise ValueError(f"role does not accept a team override: {role}")

    role_overrides: dict[str, RoleModelPolicy] = {}
    for target in sorted(item for item in targets if "@" not in item):
        current = policy.resolve(target, None)
        try:
            thinking = ThinkingMode(thinking_values.get(target, current.thinking.value))
        except ValueError as error:
            raise ValueError(
                f"invalid thinking mode for {target}: {thinking_values[target]}"
            ) from error
        try:
            max_tokens = int(token_values.get(target, current.max_tokens))
        except ValueError as error:
            raise ValueError(
                f"invalid max token ceiling for {target}: {token_values[target]}"
            ) from error
        role_overrides[target] = RoleModelPolicy(
            thinking=thinking,
            max_tokens=max_tokens,
        )
    policy = policy.model_copy(update={"role_overrides": role_overrides})

    role_team_overrides: dict[str, RoleModelPolicy] = {}
    for target in sorted(item for item in targets if "@" in item):
        role, team_id = target.split("@", 1)
        current = policy.resolve(role, team_id)
        try:
            thinking = ThinkingMode(thinking_values.get(target, current.thinking.value))
        except ValueError as error:
            raise ValueError(
                f"invalid thinking mode for {target}: {thinking_values[target]}"
            ) from error
        try:
            max_tokens = int(token_values.get(target, current.max_tokens))
        except ValueError as error:
            raise ValueError(
                f"invalid max token ceiling for {target}: {token_values[target]}"
            ) from error
        role_team_overrides[target] = RoleModelPolicy(
            thinking=thinking,
            max_tokens=max_tokens,
        )
    return policy.model_copy(update={"role_team_overrides": role_team_overrides})


def _agent_policy_manifest(
    policy: Stage3AgentPolicy,
    *,
    provider_supports_thinking: bool,
) -> dict[str, Any]:
    payload = policy.model_dump(mode="json")
    payload["provider_supports_thinking"] = provider_supports_thinking
    payload["team_perspectives"] = {
        team_id: perspective.value
        for team_id, perspective in DEFAULT_STAGE3_TEAM_PERSPECTIVES.items()
    }
    resolved: dict[str, dict[str, Any]] = {}
    invocation_teams: dict[str, tuple[str | None, ...]] = {
        "stage2_fault_verifier": ("A", "B"),
        "fault_evidence_analyst": ("fault_evidence",),
        "scope_boundary_analyst": ("scope_boundary",),
        "repair_causality_analyst": ("repair_causality",),
        "evidence_readiness": ("evidence_readiness",),
        "stage2_arbitrator": (None,),
        "symptom_analyst": ("A", "B"),
        "root_cause_analyst": ("A", "B"),
        "joint_anchor": ("A", "B"),
        "symptom_verifier": ("A", "B"),
        "root_cause_verifier": ("A", "B"),
        "causal_consistency_checker": ("A", "B"),
        "baseline_revision_assessment": ("A", "B"),
        "baseline_revision_consistency": ("A", "B"),
        "boundary_challenger": (None,),
        "stage3_arbitrator": (None,),
    }
    for role, team_ids in invocation_teams.items():
        for team_id in team_ids:
            role_policy = policy.resolve(role, team_id)
            effective = (
                None
                if not provider_supports_thinking
                or role_policy.thinking is ThinkingMode.PROVIDER_DEFAULT
                else role_policy.thinking is ThinkingMode.ENABLED
            )
            key = role if team_id is None else f"{role}@{team_id}"
            resolved[key] = {
                "thinking": role_policy.thinking.value,
                "effective_thinking_enabled": effective,
                "max_tokens": role_policy.max_tokens,
            }
    payload["resolved"] = resolved
    return payload


def _merge_llm_telemetry(
    previous: dict[str, Any] | None,
    current: dict[str, object],
) -> dict[str, object]:
    if not previous:
        return current
    calls = list(previous.get("calls", [])) + list(current["calls"])
    first_pass_success_count = sum(
        call.get("status") == "valid" and int(call.get("attempts", 0)) == 1
        for call in calls
    )
    schema_validation_failure_count = sum(
        int(
            call.get(
                "schema_validation_failure_count",
                len(call.get("validation_errors", [])),
            )
        )
        for call in calls
    )
    transport_retry_count = sum(
        int(call.get("transport_retry_count", 0)) for call in calls
    )
    transport_error_count = sum(
        int(call.get("transport_error_count", 0)) for call in calls
    )
    provider_retry_count = sum(
        event.get("status") == "retryable_error"
        for call in calls
        for event in call.get("network_attempt_details", [])
    )
    merged: dict[str, object] = {
        "total_calls": int(previous.get("total_calls", 0))
        + int(current["total_calls"]),
        "total_attempts": int(previous.get("total_attempts", 0))
        + int(current["total_attempts"]),
        "total_latency_seconds": float(previous.get("total_latency_seconds", 0))
        + float(current["total_latency_seconds"]),
        "estimated_prompt_tokens": int(previous.get("estimated_prompt_tokens", 0))
        + int(current["estimated_prompt_tokens"]),
        "estimated_completion_tokens": int(
            previous.get("estimated_completion_tokens", 0)
        )
        + int(current["estimated_completion_tokens"]),
        "reasoning_characters": int(previous.get("reasoning_characters", 0))
        + int(current.get("reasoning_characters", 0)),
        "first_pass_success_count": first_pass_success_count,
        "first_pass_success_rate": (
            first_pass_success_count / len(calls) if calls else 0.0
        ),
        "schema_retry_calls": sum(
            int(
                call.get(
                    "schema_validation_failure_count",
                    len(call.get("validation_errors", [])),
                )
            )
            > 0
            for call in calls
        ),
        "schema_validation_failure_count": schema_validation_failure_count,
        "transport_retry_calls": sum(
            int(call.get("transport_retry_count", 0)) > 0 for call in calls
        ),
        "transport_retry_count": transport_retry_count,
        "transport_error_count": transport_error_count,
        "provider_retry_calls": sum(
            any(
                event.get("status") == "retryable_error"
                for event in call.get("network_attempt_details", [])
            )
            for call in calls
        ),
        "provider_retry_count": provider_retry_count,
        "calls": calls,
    }
    for field in ("actual_prompt_tokens", "actual_completion_tokens"):
        values = [
            source.get(field)
            for source in (previous, current)
            if source.get(field) is not None
        ]
        merged[field] = sum(int(value) for value in values) if values else None
    return merged


def _preservation_compute_metrics(
    telemetry: dict[str, Any],
) -> dict[str, int | float | None]:
    roles = {
        "baseline_revision_assessment",
        "baseline_revision_consistency",
    }
    calls = [
        call
        for call in telemetry.get("calls", [])
        if isinstance(call, dict) and call.get("role") in roles
    ]

    def sum_optional(field: str) -> int | None:
        values = [call.get(field) for call in calls if call.get(field) is not None]
        return sum(int(value) for value in values) if values else None

    return {
        "assessment_calls": sum(
            call.get("role") == "baseline_revision_assessment" for call in calls
        ),
        "consistency_calls": sum(
            call.get("role") == "baseline_revision_consistency" for call in calls
        ),
        "total_calls": len(calls),
        "total_attempts": sum(int(call.get("attempts", 0)) for call in calls),
        "total_latency_seconds": sum(
            float(call.get("latency_seconds", 0.0)) for call in calls
        ),
        "estimated_prompt_tokens": sum(
            int(call.get("estimated_prompt_tokens", 0)) for call in calls
        ),
        "estimated_completion_tokens": sum(
            int(call.get("estimated_completion_tokens", 0)) for call in calls
        ),
        "actual_prompt_tokens": sum_optional("actual_prompt_tokens"),
        "actual_completion_tokens": sum_optional("actual_completion_tokens"),
        "schema_validation_failure_count": sum(
            int(call.get("schema_validation_failure_count", 0)) for call in calls
        ),
        "transport_error_count": sum(
            int(call.get("transport_error_count", 0)) for call in calls
        ),
        "transport_retry_count": sum(
            int(call.get("transport_retry_count", 0)) for call in calls
        ),
    }


def _frozen_evidence_metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    node_types: dict[str, int] = {}
    authorities: dict[str, int] = {}
    available = 0
    cited_nodes = 0
    for row in rows:
        audit = row.get("audit")
        frozen = audit.get("frozen_evidence") if isinstance(audit, dict) else None
        if not isinstance(frozen, dict):
            continue
        available += int(frozen.get("available") is True)
        for node in frozen.get("nodes", []):
            if not isinstance(node, dict):
                continue
            node_type = str(node.get("node_type", "unknown"))
            authority = str(node.get("authority", "unknown"))
            node_types[node_type] = node_types.get(node_type, 0) + 1
            authorities[authority] = authorities.get(authority, 0) + 1
            cited_nodes += int(node.get("cited") is True)
    return {
        "record_count": len(rows),
        "available_record_count": available,
        "node_type_counts": dict(sorted(node_types.items())),
        "authority_counts": dict(sorted(authorities.items())),
        "cited_node_count": cited_nodes,
    }


def _cited_frozen_ids(stage3_audit: object, known_ids: set[str]) -> tuple[str, ...]:
    found: set[str] = set()

    def visit(value: object) -> None:
        if isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        elif isinstance(value, str) and value in known_ids:
            found.add(value)

    visit(stage3_audit)
    return tuple(sorted(found))


def _model_transport_response(
    raw: str,
    *,
    attempts: int,
    network_attempt_details: tuple[dict[str, object], ...] = (),
) -> ModelTransportResponse:
    return ModelTransportResponse(
        raw,
        network_attempts=attempts,
        prompt_tokens=getattr(raw, "prompt_tokens", None),
        completion_tokens=getattr(raw, "completion_tokens", None),
        reasoning_characters=int(getattr(raw, "reasoning_characters", 0)),
        network_attempt_details=network_attempt_details,
    )


def _targeted_provider_request(
    *,
    record_budget: RecordSlaBudget,
    admission: threading.BoundedSemaphore,
    call_once: Callable[[SlaRequestDeadline, RequestCompletionLease], str],
    retry_budget: RoleNetworkRetryBudget | None = None,
) -> ModelTransportResponse:
    """Make at most two fresh requests under one caller-owned SLA budget."""

    role_retry_budget = retry_budget or RoleNetworkRetryBudget(max_retries=1)
    attempt_events: list[dict[str, object]] = []

    def require_request_deadline() -> SlaRequestDeadline:
        deadline_method = getattr(record_budget, "require_request_deadline", None)
        if callable(deadline_method):
            return deadline_method()
        timeout_seconds = record_budget.require_request_budget()
        return SlaRequestDeadline(
            timeout_seconds=timeout_seconds,
            deadline_monotonic=time.monotonic() + timeout_seconds,
        )

    def annotate_budget_error(error: SlaBudgetExhausted) -> None:
        error.attempts = len(attempt_events)
        error.network_attempt_details = tuple(attempt_events)

    while True:
        attempt = len(attempt_events) + 1
        admission_deadline = time.monotonic() + record_budget.remaining_seconds()
        remaining = max(0.0, admission_deadline - time.monotonic())
        if remaining <= 0.0 or not admission.acquire(timeout=remaining):
            error = SlaBudgetExhausted("provider admission SLA deadline exceeded")
            annotate_budget_error(error)
            raise error
        completion_lease = RequestCompletionLease(admission.release)
        try:
            try:
                request_deadline = require_request_deadline()
            except SlaBudgetExhausted as error:
                annotate_budget_error(error)
                raise
            started = time.perf_counter()
            try:
                raw = call_once(request_deadline, completion_lease)
            except SlaBudgetExhausted as error:
                annotate_budget_error(error)
                raise
            except Exception as error:
                details = (
                    {
                        "retryable": error.retryable,
                        "cause_type": error.cause_type,
                        "http_status": error.http_status,
                        "retry_after_seconds": error.retry_after_seconds,
                    }
                    if isinstance(error, ModelTransportError)
                    else model_transport_error_details(error)
                )
                attempt_events.append(
                    {
                        "attempt": attempt,
                        "status": (
                            "retryable_error" if details["retryable"] else "error"
                        ),
                        "latency_seconds": max(0.0, time.perf_counter() - started),
                        **details,
                    }
                )
                if bool(details["retryable"]) and role_retry_budget.claim_retry():
                    continue
                transport_error = ModelTransportError(
                    "Model provider transport failed.",
                    attempts=attempt,
                    **details,
                )
                transport_error.network_attempt_details = tuple(attempt_events)
                raise transport_error from None
        finally:
            completion_lease.release_if_caller_owned()
        attempt_events.append(
            {
                "attempt": attempt,
                "status": "success",
                "latency_seconds": max(0.0, time.perf_counter() - started),
            }
        )
        return _model_transport_response(
            raw,
            attempts=attempt,
            network_attempt_details=tuple(attempt_events),
        )


def _network_retry_limit(
    *,
    options: ModelCallOptions,
    configured_max_retries: int,
) -> int:
    """Prevent optional certificate schema repairs from nesting network retries."""

    if options.share_transport_schema_budget:
        return 0
    return configured_max_retries


def _uses_pooled_provider_client(provider: str, *, targeted_sla: bool) -> bool:
    """Use connection pooling for compatible non-SLA provider traffic."""

    return _canonical_provider(provider) in {"micu", "gemini"} and not targeted_sla


def _provider_thinking_option(
    provider: str,
    thinking_enabled: bool,
) -> bool | None:
    """Avoid sending DeepSeek-specific thinking controls to Gemini."""

    if _canonical_provider(provider) == "gemini":
        return None
    return thinking_enabled


def _default_backend(args: argparse.Namespace) -> str:
    if args.base_url:
        return (
            _normalize_micu_base_url(str(args.base_url))
            if args.provider == "micu"
            else normalize_gemini_base_url(str(args.base_url))
            if args.provider == "gemini"
            else str(args.base_url)
        )
    if args.provider == "deepseek":
        return os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    if args.provider == "micu":
        return _normalize_micu_base_url(
            os.environ.get("MICU_BASE_URL", MICU_DEFAULT_BASE_URL)
        )
    if args.provider == "gemini":
        return normalize_gemini_base_url(
            os.environ.get("GEMINI_BASE_URL", GEMINI_DEFAULT_BASE_URL)
        )
    return (
        os.environ.get("SELF_BASE_URL")
        or os.environ.get("BASE_URL")
        or os.environ.get("LLM_BASE_URL")
        or "proxy:not-configured"
    )


def _normalize_micu_base_url(value: str) -> str:
    normalized = value.strip().rstrip("/")
    if not normalized:
        raise SystemExit("Missing MICU_BASE_URL or --base-url")
    parsed = urlsplit(normalized)
    if parsed.scheme.lower() != "https":
        raise ValueError("MICU provider requires an HTTPS base URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("MICU provider base URL cannot contain userinfo")
    host = (parsed.hostname or "").lower()
    if host not in MICU_OFFICIAL_HOSTS:
        raise ValueError("MICU provider requires an official MICU host")
    if parsed.port not in {None, 443}:
        raise ValueError("MICU provider base URL cannot use a custom port")
    if parsed.query or parsed.fragment:
        raise ValueError("MICU provider base URL cannot contain a query or fragment")
    if parsed.path not in {"", "/", "/v1"}:
        raise ValueError("MICU provider base path must be empty or /v1")
    return f"https://{host}/v1"


def _resolve_adaptive_model(provider: str, model_override: str | None) -> str:
    provider = _canonical_provider(provider)
    if provider in {"micu", "gemini"}:
        if not model_override:
            raise SystemExit(f"--provider {provider} requires an explicit --model")
        return model_override
    return resolve_model(provider, model_override)


def _resolve_adaptive_provider_config(
    env: Mapping[str, str],
    *,
    provider: str,
    base_url_override: str | None,
) -> dict[str, str]:
    provider = _canonical_provider(provider)
    if provider == "gemini":
        return resolve_gemini_config(env, base_url_override=base_url_override)
    if provider != "micu":
        return resolve_run_config(
            env,
            base_url_override=base_url_override,
            provider=provider,
        )
    api_key = env.get("MICU_API_KEY")
    if not api_key:
        raise SystemExit("Missing MICU_API_KEY in environment")
    return {
        "base_url": _normalize_micu_base_url(
            base_url_override or env.get("MICU_BASE_URL") or MICU_DEFAULT_BASE_URL
        ),
        "api_key": api_key,
        "user_agent": MICU_BROWSER_USER_AGENT,
    }


def run_cli(
    argv: list[str] | None = None,
    *,
    transport_override: Callable[..., str] | None = None,
) -> dict[str, Any]:
    parsed_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(parsed_argv)
    if args.supplemental_evidence and (
        _canonical_provider(args.provider) != "gemini" or args.domain != "ase2022"
        or args.stage != "stage3" or args.execution_profile != "full-dual-team"
        or args.frozen_evidence or args.baseline_preservation
    ):
        raise ValueError("supplemental evidence requires Gemini ASE2022 Stage 3 full-dual-team without frozen evidence or baseline preservation")
    module_a = None
    if args.experiment_two_arm:
        if (not args.global_supervisor or not args.module_a or args.a_protocol != 'rules-v4'
                or args.module_b or not args.supplemental_evidence):
            raise ValueError('Experiment two requires rules-v4, global supervisor, frozen supplemental evidence; excludes legacy Module B')
        if args.arm_id != args.experiment_two_arm:
            raise ValueError('Experiment-two arm must match --arm-id')
    if args.source_graph_mode and args.experiment_two_arm not in {'E01','E11'}:
        raise ValueError('Graph serialization option requires E01/E11')
    if args.global_supervisor and (not args.module_a or args.a_protocol != "rules-v4" or args.module_b):
        raise ValueError("--global-supervisor requires --module-a --a-protocol rules-v4 and excludes --module-b")
    if (args.a_examples or args.a_example_source) and not args.module_a:
        raise ValueError("Module A example flags require --module-a")
    if args.module_a or args.module_b:
        if (args.domain != "ase2022" or args.stage != "stage3"
                or args.execution_profile != "full-dual-team" or args.ablation_profile != "full-dual-team"
                or args.baseline_preservation or args.baseline_anchor_path or args.baseline_trust_manifest):
            raise ValueError("Module A/Module B support ASE2022 Stage 3 full-dual-team without baseline preservation")
        if not args.cohort_path:
            raise ValueError("Module A requires an explicit evidence-only --cohort-path")
        # Validate the input boundary BEFORE the general loader can read answer-bearing rows.
        _, allowed_records = read_allowed_source(args.cohort_path)
    if args.module_a:
        if bool(args.a_examples) != bool(args.a_example_source):
            raise ValueError("Module A requires both --a-examples and --a-example-source")
        from Benchmark.src.adaptive_empirical_workflow.formal_reasoning import FormalExplainableTools
        if args.a_protocol in ('rules-v4', 'tools-v4', 'rules-v5', 'proof-v6'):
            if args.module_b:
                raise ValueError('A v4/v5/v6 experiments exclude Module B')
            from Benchmark.src.adaptive_empirical_workflow.a_decision import RuleDecisionTools, MinedDecisionTools
            if args.a_protocol == 'proof-v6':
                from Benchmark.src.adaptive_empirical_workflow.rule_proof import ProofDecisionTools
                tool_class = ProofDecisionTools
            elif args.a_protocol == 'rules-v5':
                from Benchmark.src.adaptive_empirical_workflow.a_answer_policy import AnswerDecisionTools
                tool_class = AnswerDecisionTools
            else:
                tool_class = RuleDecisionTools if args.a_protocol == 'rules-v4' else MinedDecisionTools
        else:
            tool_class = FormalExplainableTools if args.a_protocol == "formal-v2" else ExplainableTools
        if args.experiment_two_arm in {'E10','E11'}:
            from Benchmark.src.adaptive_empirical_workflow.experiment_two import GroundedDecisionTools
            tool_class = GroundedDecisionTools
        module_a = (tool_class.from_files(args.a_examples, args.a_example_source)
                    if args.a_examples else tool_class())
        module_a.check_evaluation_records(allowed_records)
        if args.experiment_two_arm:
            module_a.query_max_tokens = 8192
    args.provider = _canonical_provider(args.provider)
    targeted_sla_config = _targeted_sla_config(
        args,
        concurrency_explicit=_option_was_supplied(parsed_argv, "--concurrency"),
        provider_max_inflight_explicit=_option_was_supplied(
            parsed_argv, "--provider-max-inflight"
        ),
    )
    targeted_sla = targeted_sla_config is not None
    deadline_started_at = (
        datetime.now(timezone.utc).isoformat() if targeted_sla else None
    )
    deadline_monotonic_started = time.perf_counter() if targeted_sla else None
    global_sla_budget = (
        GlobalSlaBudget(targeted_sla_config)
        if targeted_sla_config is not None
        else None
    )
    _load_env_file(REPO_ROOT / ".env")
    model = _resolve_adaptive_model(args.provider, args.model)
    agent_policy = _resolve_agent_policy(args)
    provider_supports_thinking = args.provider in {"deepseek", "micu"}
    agent_policy_manifest = _agent_policy_manifest(
        agent_policy,
        provider_supports_thinking=provider_supports_thinking,
    )
    split_manifest_path = Path(args.split_manifest) if args.split_manifest else None
    if split_manifest_path is None and args.cohort_path:
        sidecar = Path(args.cohort_path).parent / "split_manifest.json"
        if sidecar.exists():
            split_manifest_path = sidecar
    if not targeted_sla:
        preservation_surface = (
            args.baseline_preservation,
            bool(args.baseline_anchor_path),
            bool(args.baseline_trust_manifest),
        )
        if len(set(preservation_surface)) != 1:
            raise ValueError(
                "--baseline-preservation, --baseline-anchor-path, and "
                "--baseline-trust-manifest must be supplied together"
            )
    if args.baseline_preservation and split_manifest_path is None:
        raise ValueError(
            "Baseline preservation Stage 3 requires an active split manifest"
        )
    inputs = load_domain_inputs(
        args.domain,
        cohort_path=args.cohort_path,
        taxonomy_path=args.taxonomy_path,
        split_manifest_path=split_manifest_path,
    )
    supplemental_bundle = None
    if args.supplemental_evidence:
        from Benchmark.src.adaptive_empirical_workflow.supplemental_evidence import SupplementalEvidenceBundle
        supplemental_bundle = SupplementalEvidenceBundle.load(
            args.supplemental_evidence,
            allowed_record_ids={record["record_id"] for record in inputs.records},
        )
    frozen_evidence_bundle = None
    if args.frozen_evidence:
        if args.domain != "ase2022" or args.stage not in {"stage3", "all"}:
            raise ValueError(
                "registered frozen evidence is available only for ASE2022 Stage 3"
            )
        frozen_evidence_bundle = load_registered_frozen_evidence_runtime(args.domain)
    records = select_records(
        inputs.records,
        record_ids=args.record_ids,
        limit=None,
    )
    if (args.baseline_preservation or args.frozen_evidence) and any(
        set(record).intersection(GOLD_FIELDS) for record in records
    ):
        raise ValueError(
            "Baseline preservation and frozen evidence require an evidence-only "
            "runner cohort"
        )
    if args.stage == "stage3" and split_manifest_path is None:
        records = [
            record for record in records if record.get("decision") == "accepted_fault"
        ]
    records = select_records(records, record_ids=None, limit=args.limit)
    preservation_bundle = None
    targeted_baseline_by_id: dict[str, object] | None = None
    targeted_baseline_manifest: dict[str, object] | None = None
    taxonomy_structure = None
    preservation_manifest: dict[str, Any] | None = None
    frozen_evidence_manifest: dict[str, Any] | None = None
    if args.baseline_preservation:
        if args.domain != "ase2022" or args.stage not in {"stage3", "all"}:
            raise ValueError(
                "Baseline preservation is registered only for ASE2022 Stage 3"
            )
        selected_ids = tuple(record["record_id"] for record in records)
        preservation_bundle = load_baseline_anchor_bundle(
            args.baseline_anchor_path,
            trust_manifest_path=args.baseline_trust_manifest,
            expected_domain=args.domain,
            expected_record_ids=selected_ids,
            taxonomy=inputs.taxonomy,
        )
        taxonomy_structure = load_bound_taxonomy_structure_for_domain(
            args.domain,
            inputs.taxonomy,
        )
        trust_manifest = bound_taxonomy_artifact_manifest_for_domain(args.domain)
        canonical_structure_hash = taxonomy_structure_hash(taxonomy_structure)
        boundary_manifest = official_boundary_card_manifest(taxonomy_structure)
        preservation_manifest = {
            "enabled": True,
            "source_config_hash": preservation_bundle.source_config_hash,
            "source_predictions_sha256": (
                preservation_bundle.source_predictions_sha256
            ),
            "selected_anchor_set_sha256": (
                preservation_bundle.selected_anchor_set_sha256
            ),
            "anchor_digest_map_sha256": (preservation_bundle.anchor_digest_map_sha256),
            "record_anchor_digests": dict(preservation_bundle.anchor_digests),
            "baseline_trust_manifest_sha256": (
                preservation_bundle.trust_manifest_sha256
            ),
            "baseline_trust_artifact_id": preservation_bundle.trust_artifact_id,
            "baseline_trust_manifest_id": (
                preservation_bundle.trust_manifest_relative_path
            ),
            "taxonomy_structure_sha256": canonical_structure_hash,
            "taxonomy_structure_artifact_sha256": trust_manifest[
                "taxonomy_structure_sha256"
            ],
            "bound_corpus_sidecar_sha256": trust_manifest[
                "bound_corpus_sidecar_sha256"
            ],
            "policy_version": BASELINE_PRESERVATION_POLICY_VERSION,
            "official_boundary_generator_version": boundary_manifest[
                "generator_version"
            ],
            "materialized_boundary_cards_sha256": boundary_manifest[
                "materialized_boundary_cards_sha256"
            ],
            "candidate_graph_policy": REVISION_CANDIDATE_GRAPH_POLICY_VERSION,
            "cross_check_graph_policy": REVISION_CROSS_CHECK_GRAPH_POLICY_VERSION,
            "revision_assessment_policy": (HETEROGENEOUS_REVISION_ASSESSMENT_POLICY),
            "revision_cross_policy": TASK_SPECIFIC_REVISION_CROSS_POLICY,
            "revision_schema_hashes": baseline_revision_schema_hashes(),
            "revision_role_model_options": {
                f"{role}@{team_id}": agent_policy.resolve(role, team_id).model_dump(
                    mode="json"
                )
                for role in (
                    "baseline_revision_assessment",
                    "baseline_revision_consistency",
                )
                for team_id in ("A", "B")
            },
        }
    if targeted_sla:
        selected_ids = tuple(record["record_id"] for record in records)
        targeted_baseline_by_id = dict(
            load_baseline_anchors(
                args.baseline_anchor_path,
                expected_record_ids=selected_ids,
                taxonomy=inputs.taxonomy,
            )
        )
        if any(
            not anchor.valid
            or anchor.symptom_label is None
            or anchor.root_cause_label is None
            for anchor in targeted_baseline_by_id.values()
        ):
            raise ValueError("targeted-sla30 requires valid Baseline anchors")
        anchor_digests = {
            record_id: baseline_anchor_hash(targeted_baseline_by_id[record_id])
            for record_id in selected_ids
        }
        first_anchor = targeted_baseline_by_id[selected_ids[0]]
        targeted_baseline_manifest = {
            "source_config_hash": first_anchor.source_config_hash,
            "source_predictions_sha256": first_anchor.source_predictions_sha256,
            "selected_anchor_set_sha256": _config_hash(
                [
                    targeted_baseline_by_id[record_id].model_dump(mode="json")
                    for record_id in selected_ids
                ]
            ),
            "anchor_digest_map_sha256": _config_hash(anchor_digests),
            "record_anchor_digests": anchor_digests,
        }
    cohort_path = Path(args.cohort_path or inputs.profile.default_cohort_path)
    taxonomy_path = Path(args.taxonomy_path or inputs.profile.default_taxonomy_path)
    if frozen_evidence_bundle is not None:
        if split_manifest_path is None:
            raise ValueError(
                "registered frozen evidence requires an active split manifest"
            )
        cohort_sha256 = _sha256_file(cohort_path)
        split_manifest_sha256 = _sha256_file(split_manifest_path)
        if frozen_evidence_bundle.domain != args.domain:
            raise ValueError("frozen evidence domain binding mismatch")
        if frozen_evidence_bundle.manifest.runner_cohort_sha256 != cohort_sha256:
            raise ValueError("frozen evidence runner cohort hash mismatch")
        if (
            frozen_evidence_bundle.manifest.split_manifest_sha256
            != split_manifest_sha256
        ):
            raise ValueError("frozen evidence split manifest hash mismatch")
        selected_ids = tuple(record["record_id"] for record in records)
        unknown_records = set(selected_ids) - set(frozen_evidence_bundle.record_ids)
        if unknown_records:
            raise ValueError(
                "frozen evidence bundle is missing selected records: "
                + ", ".join(sorted(unknown_records))
            )
        if (
            frozen_evidence_bundle.policy.runtime_network_forbidden is not True
            or frozen_evidence_bundle.manifest.network_policy
            != "offline_capture_only_runtime_network_forbidden"
            or frozen_evidence_bundle.manifest.gold_accessed is not False
        ):
            raise ValueError("frozen evidence runtime safety policy mismatch")
        retrieval_tool = frozen_evidence_bundle.manifest.retrieval_tool
        frozen_evidence_manifest = {
            "enabled": True,
            "bundle_id": frozen_evidence_bundle.bundle_id,
            "bundle_split_id": frozen_evidence_bundle.split_id,
            "trust_manifest_id": (frozen_evidence_bundle.trust_manifest_relative_path),
            "trust_manifest_sha256": (frozen_evidence_bundle.trust_manifest_sha256),
            "bundle_merkle_root": (frozen_evidence_bundle.manifest.bundle_merkle_root),
            "capture_policy_schema_version": (
                frozen_evidence_bundle.policy.schema_version
            ),
            "capture_policy_id": frozen_evidence_bundle.policy.policy_id,
            "capture_policy_sha256": (
                frozen_evidence_bundle.manifest.capture_policy.sha256
            ),
            "retrieval_tool": {
                "name": retrieval_tool.name,
                "version": retrieval_tool.version,
                "module_sha256": retrieval_tool.module_sha256,
            },
            "projection_policy_version": (FROZEN_EVIDENCE_PROJECTION_POLICY_VERSION),
            "projection_policy_sha256": FROZEN_EVIDENCE_PROJECTION_POLICY_HASH,
            "runner_cohort_sha256": cohort_sha256,
            "split_manifest_sha256": split_manifest_sha256,
            "runtime_network_forbidden": True,
            "gold_accessed": False,
        }
    output_root = Path(
        args.output_dir
        or f"Benchmark/results/adaptive_empirical_workflow/{args.domain}"
    )
    prompt_hashes = {
        "workflow_prompt_version": hashlib.sha256(
            WORKFLOW_PROMPT_VERSION.encode("utf-8")
        ).hexdigest(),
        "team_perspective_policy": hashlib.sha256(
            json.dumps(
                agent_policy_manifest.get("team_perspectives", {}),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        **{
            role.value: hashlib.sha256(
                render_system_prompt(
                    role,
                    include_dataset_guidance=False,
                ).encode("utf-8")
            ).hexdigest()
            for role in (
                AnalystRole.JOINT_ANCHOR,
                AnalystRole.SYMPTOM_VERIFIER,
                AnalystRole.ROOT_CAUSE_VERIFIER,
            )
        },
    }
    if preservation_manifest is not None:
        prompt_hashes.update(baseline_revision_prompt_hashes())
    if targeted_sla:
        prompt_hashes.update(sla_prompt_hashes())
    config_payload = {
        "architecture": "adaptive_empirical_expert_workflow",
        "domain": args.domain,
        "stage": args.stage,
        "provider": args.provider,
        "model": model,
        "backend": _default_backend(args),
        "cohort_sha256": _sha256_file(cohort_path),
        "taxonomy_sha256": _sha256_file(taxonomy_path),
        "record_ids": [record["record_id"] for record in records],
        "concurrency": args.concurrency,
        "max_schema_retries": args.max_schema_retries,
        "max_network_retries": args.max_network_retries,
        "provider_max_inflight": args.provider_max_inflight,
        "max_retrieval_rounds": args.max_retrieval_rounds,
        "max_specialist_calls": args.max_specialist_calls,
        "consensus_confidence": args.consensus_confidence,
        "prompt_version": WORKFLOW_PROMPT_VERSION,
        "role_max_tokens": args.role_max_tokens,
        "agent_policy": agent_policy_manifest,
        "prompt_hashes": prompt_hashes,
        "split_manifest_sha256": (
            _sha256_file(split_manifest_path) if split_manifest_path else None
        ),
        "ablation_profile": args.ablation_profile,
        "budget_class": args.budget_class,
        **_code_state(),
    }
    if targeted_sla:
        assert targeted_sla_config is not None
        config_payload.update(
            {
                "execution_profile": TARGETED_SLA_PROFILE,
                "development_only": True,
                "baseline_preservation_protocol": False,
                "sla_policy_version": TARGETED_SLA_POLICY_VERSION,
                "max_network_retries": 1,
                "max_schema_retries": 1,
                "global_deadline_seconds": targeted_sla_config.global_seconds,
                "record_budget_seconds": targeted_sla_config.record_seconds,
                "request_timeout_seconds": targeted_sla_config.request_seconds,
                "targeted_sla_role_options": {
                    "sla_joint_diagnosis": {
                        "thinking_enabled": False,
                        "max_tokens": 2400,
                        "max_schema_retries": 1,
                    },
                    "sla_joint_verifier": {
                        "thinking_enabled": False,
                        "max_tokens": 1800,
                        "max_schema_retries": 1,
                    },
                    "sla_conditional_arbitrator": {
                        "thinking_enabled": False,
                        "max_tokens": 1800,
                        "max_schema_retries": 1,
                    },
                },
                "targeted_sla_baseline": targeted_baseline_manifest,
            }
        )
    if preservation_manifest is not None:
        config_payload["baseline_preservation"] = preservation_manifest
    if frozen_evidence_manifest is not None:
        config_payload["frozen_evidence"] = frozen_evidence_manifest
    if supplemental_bundle is not None:
        config_payload["supplemental_evidence"] = supplemental_bundle.manifest()
    if module_a is not None:
        config_payload["module_a"] = module_a.manifest()
    if args.experiment_two_arm:
        from Benchmark.src.adaptive_empirical_workflow.experiment_two import manifest as experiment_two_manifest
        config_payload['experiment_two'] = experiment_two_manifest(args.experiment_two_arm,args.source_graph_mode)
    if args.module_b:
        from Benchmark.src.adaptive_empirical_workflow.evidence_chain import chain_manifest
        config_payload["module_b"] = chain_manifest()
    if args.global_supervisor:
        from Benchmark.src.adaptive_empirical_workflow.global_supervisor import manifest as supervisor_manifest
        config_payload["global_supervisor"] = supervisor_manifest()
    config_hash = _config_hash(config_payload)
    identity = build_run_identity(
        output_root=output_root,
        experiment_id=args.experiment_id or "adaptive-empirical-workflow",
        split_id=args.split_id or cohort_path.stem,
        arm_id=args.arm_id or f"config-{config_hash[:12]}",
        run_id=args.run_id,
        resolved_config=config_payload,
    )
    if targeted_sla and not args.no_resume and identity.manifest_path.exists():
        raise ValueError(
            "targeted-sla30 does not support resume because the persisted SLA "
            "deadline cannot be reset; rerun with --no-resume"
        )
    output_dir = prepare_run_directory(identity)
    slug = model_slug(model)
    predictions_path = output_dir / f"predictions_{slug}.jsonl"
    metrics_path = output_dir / f"metrics_{slug}.json"
    manifest_path = identity.manifest_path
    previous_metrics: dict[str, Any] = {}
    if not args.no_resume and metrics_path.exists():
        candidate_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if candidate_metrics.get("config_hash") == config_hash:
            previous_metrics = candidate_metrics
    retrieved_at = args.retrieved_at
    if retrieved_at is None and not args.no_resume and manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        retrieved_at = existing_manifest.get("retrieved_at")
    retrieved_at = retrieved_at or datetime.now(timezone.utc).isoformat()
    manifest = {
        **config_payload,
        "config_hash": config_hash,
        "record_count": len(records),
        "cohort_artifact_id": (
            cohort_path.resolve().relative_to(REPO_ROOT).as_posix()
            if cohort_path.resolve().is_relative_to(REPO_ROOT)
            else cohort_path.name
        ),
        "taxonomy_artifact_id": (
            taxonomy_path.resolve().relative_to(REPO_ROOT).as_posix()
            if taxonomy_path.resolve().is_relative_to(REPO_ROOT)
            else taxonomy_path.name
        ),
        "retrieved_at": retrieved_at,
        "dry_run": args.dry_run,
    }
    if targeted_sla:
        assert deadline_started_at is not None
        manifest.update(
            {
                "deadline_started_at": deadline_started_at,
                "deadline_clock_policy": "process_monotonic-v1",
            }
        )
    if preservation_manifest is not None:
        manifest["baseline_preservation"] = preservation_manifest
    else:
        manifest["gold_fields_excluded_from_agent_prompts"] = sorted(GOLD_FIELDS)
    if frozen_evidence_manifest is not None:
        manifest["frozen_evidence"] = frozen_evidence_manifest
    write_run_manifest(identity, manifest)
    summary: dict[str, Any] = {
        "config_hash": config_hash,
        "record_count": len(records),
        "manifest_path": str(manifest_path),
        "predictions_path": str(predictions_path),
        "metrics_path": str(metrics_path),
        "dry_run": args.dry_run,
        "agent_policy": agent_policy_manifest,
    }
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary

    consistency_admission = threading.BoundedSemaphore(1)
    targeted_provider_admission = (
        threading.BoundedSemaphore(args.provider_max_inflight) if targeted_sla else None
    )
    targeted_budget_local = threading.local()
    pooled_provider_client: PooledChatCompletionClient | None = None
    if transport_override is None:
        provider = _resolve_adaptive_provider_config(
            os.environ,
            base_url_override=args.base_url,
            provider=args.provider,
        )
        if args.provider_max_inflight < 1:
            raise ValueError("--provider-max-inflight must be positive")
        if _uses_pooled_provider_client(args.provider, targeted_sla=targeted_sla):
            pooled_provider_client = PooledChatCompletionClient(
                base_url=provider["base_url"],
                max_connections=args.provider_max_inflight,
            )

        def provider_transport(
            system_prompt: str,
            user_prompt: str,
            *,
            options: ModelCallOptions,
            timeout_seconds: float | None = None,
            deadline_monotonic: float | None = None,
            completion_lease: RequestCompletionLease | None = None,
        ) -> str:
            if targeted_sla:
                if timeout_seconds is None or deadline_monotonic is None:
                    raise RuntimeError("targeted SLA transport has no request deadline")
                return call_fresh_chat_completion(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model=model,
                    api_key=provider["api_key"],
                    base_url=provider["base_url"],
                    timeout_seconds=timeout_seconds,
                    deadline_monotonic=deadline_monotonic,
                    max_tokens=options.max_tokens,
                    **({"temperature": options.temperature} if options.temperature else {}),
                    json_mode=True,
                    thinking_enabled=_provider_thinking_option(
                        args.provider, options.thinking_enabled
                    ),
                    user_agent=provider.get("user_agent"),
                    stream=True,
                    completion_lease=completion_lease,
                )
            attempt_events: list[dict[str, object]] = []
            try:
                raw, attempts = call_model_with_retries(
                    max_retries=_network_retry_limit(
                        options=options,
                        configured_max_retries=args.max_network_retries,
                    ),
                    retry_delay_seconds=args.retry_delay_seconds,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model=model,
                    api_key=provider["api_key"],
                    base_url=provider["base_url"],
                    wire_api="chat_completions",
                    max_tokens=options.max_tokens,
                    **({"temperature": options.temperature} if options.temperature else {}),
                    json_mode=True,
                    thinking_enabled=_provider_thinking_option(
                        args.provider, options.thinking_enabled
                    ),
                    user_agent=provider.get("user_agent"),
                    attempt_observer=attempt_events.append,
                    model_call=(
                        pooled_provider_client.call_model
                        if pooled_provider_client is not None
                        else None
                    ),
                )
            except Exception as error:
                transport_error = ModelTransportError(
                    "Model provider transport failed.",
                    attempts=int(getattr(error, "attempts", 1)),
                    **model_transport_error_details(error),
                )
                transport_error.network_attempt_details = tuple(attempt_events)
            else:
                return _model_transport_response(
                    raw,
                    attempts=attempts,
                    network_attempt_details=tuple(attempt_events),
                )
            raise transport_error from None

    else:
        override_parameters = inspect.signature(transport_override).parameters
        override_options = override_parameters.get("options")
        override_max_tokens = override_parameters.get("max_tokens")
        override_thinking = override_parameters.get("thinking_enabled")
        explicit_option_kinds = {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
        override_accepts_options = (
            override_options is not None
            and override_options.kind in explicit_option_kinds
        )
        override_accepts_expanded_options = (
            override_max_tokens is not None
            and override_max_tokens.kind in explicit_option_kinds
            and override_thinking is not None
            and override_thinking.kind in explicit_option_kinds
        )

        if args.global_supervisor and not override_accepts_options:
            raise ValueError("Global supervisor transport override must accept ModelCallOptions to carry temperature")

        def provider_transport(
            system_prompt: str,
            user_prompt: str,
            *,
            options: ModelCallOptions,
            timeout_seconds: float | None = None,
            deadline_monotonic: float | None = None,
            completion_lease: RequestCompletionLease | None = None,
        ) -> str:
            del timeout_seconds, deadline_monotonic, completion_lease
            if override_accepts_options:
                return transport_override(system_prompt, user_prompt, options=options)
            if override_accepts_expanded_options:
                return transport_override(
                    system_prompt,
                    user_prompt,
                    max_tokens=options.max_tokens,
                    thinking_enabled=options.thinking_enabled,
                )
            return transport_override(system_prompt, user_prompt)

    def transport(
        system_prompt: str,
        user_prompt: str,
        *,
        options: ModelCallOptions,
    ) -> str:
        if targeted_sla:
            record_budget = getattr(targeted_budget_local, "record_budget", None)
            if record_budget is None:
                raise RuntimeError("targeted SLA transport has no record budget")
            role_retry_budgets = getattr(
                targeted_budget_local, "role_retry_budgets", None
            )
            if not isinstance(role_retry_budgets, dict):
                raise RuntimeError("targeted SLA transport has no role retry budget")
            retry_budget = role_retry_budgets.setdefault(
                options.role,
                RoleNetworkRetryBudget(max_retries=1),
            )
            assert targeted_provider_admission is not None
            return _targeted_provider_request(
                record_budget=record_budget,
                admission=targeted_provider_admission,
                retry_budget=retry_budget,
                call_once=lambda request_deadline, completion_lease: provider_transport(
                    system_prompt,
                    user_prompt,
                    options=options,
                    timeout_seconds=request_deadline.timeout_seconds,
                    deadline_monotonic=request_deadline.deadline_monotonic,
                    completion_lease=completion_lease,
                ),
            )
        if options.role == "baseline_revision_consistency":
            with consistency_admission:
                return provider_transport(system_prompt, user_prompt, options=options)
        return provider_transport(system_prompt, user_prompt, options=options)

    if supplemental_bundle is not None:
        text_transport = transport

        def transport(system_prompt, user_prompt, *, options):
            content, images = supplemental_bundle.prepare_message(user_prompt)
            try:
                response = text_transport(system_prompt, content, options=options)
            except Exception:
                supplemental_bundle.record_delivery(images, role=options.role, status="failed")
                raise
            supplemental_bundle.record_delivery(images, role=options.role, status="response_received")
            return response

    agents = StructuredRoleAgents(
        StructuredModelClient(
            transport,
            max_schema_retries=args.max_schema_retries,
            retry_delay_seconds=args.retry_delay_seconds,
        ),
        role_max_tokens=args.role_max_tokens,
        agent_policy=agent_policy,
        provider_supports_thinking=provider_supports_thinking,
        explainable_tools=module_a,
        chain_checker=args.module_b,
        global_supervisor=args.global_supervisor,
        experiment_two_arm=args.experiment_two_arm,
        source_graph_mode=args.source_graph_mode,
    )
    stage2_config = Stage2WorkflowConfig(
        max_retrieval_rounds=args.max_retrieval_rounds,
        max_specialist_calls=args.max_specialist_calls,
        consensus_confidence=args.consensus_confidence,
    )
    stage3_config = Stage3WorkflowConfig(
        max_retrieval_rounds=args.max_retrieval_rounds,
        max_specialist_calls=args.max_specialist_calls,
        consensus_confidence=args.consensus_confidence,
    )
    targeted_controller = TargetedSlaController(agents) if targeted_sla else None

    def record_runner(record: dict[str, str]) -> dict[str, object]:
        frozen_projection = (
            project_frozen_evidence_for_record(
                frozen_evidence_bundle, record["record_id"]
            )
            if frozen_evidence_bundle is not None
            else None
        )
        if targeted_sla:
            assert global_sla_budget is not None
            assert targeted_controller is not None
            assert targeted_baseline_by_id is not None
            record_budget = global_sla_budget.start_record()
            targeted_budget_local.record_budget = record_budget
            targeted_budget_local.role_retry_budgets = {}
            try:
                return run_targeted_sla_record(
                    record,
                    domain=args.domain,
                    taxonomy=inputs.taxonomy,
                    controller=targeted_controller,
                    baseline_anchor=targeted_baseline_by_id[record["record_id"]],
                    record_budget=record_budget,
                    retrieved_at=retrieved_at,
                    frozen_evidence_projection=frozen_projection,
                )
            finally:
                del targeted_budget_local.role_retry_budgets
                del targeted_budget_local.record_budget
        try:
            record_id = record["record_id"]
            anchor = (
                preservation_bundle.anchors[record_id]
                if preservation_bundle is not None
                else None
            )
            row = run_adaptive_record(
                record,
                domain=args.domain,
                taxonomy=inputs.taxonomy,
                agents=agents,
                stage=args.stage,
                retrieved_at=retrieved_at,
                stage2_config=stage2_config,
                stage3_config=stage3_config,
                baseline_anchor=anchor,
                taxonomy_structure=taxonomy_structure,
                expected_baseline_config_hash=(
                    preservation_bundle.source_config_hash
                    if preservation_bundle is not None
                    else None
                ),
                expected_baseline_predictions_sha256=(
                    preservation_bundle.source_predictions_sha256
                    if preservation_bundle is not None
                    else None
                ),
                expected_baseline_anchor_hash=(
                    preservation_bundle.anchor_digests[record_id]
                    if preservation_bundle is not None
                    else None
                ),
                expected_taxonomy_structure_hash=(
                    preservation_manifest["taxonomy_structure_sha256"]
                    if preservation_manifest is not None
                    else None
                ),
                frozen_evidence_projection=frozen_projection,
                **({"supplemental_items": supplemental_bundle.items_for(record_id)}
                   if supplemental_bundle is not None else {}),
            )
        except (StructuredOutputError, ModelTransportError) as error:
            if isinstance(error, ModelTransportError):
                error_payload = {
                    "type": "ModelTransportError",
                    "code": "model_transport_failed",
                    "message": "Model transport failed during record execution.",
                }
            else:
                error_payload = {
                    "type": "StructuredOutputError",
                    "code": "structured_output_invalid",
                    "message": "Structured model output failed validation.",
                }
                if error.schema_name is not None:
                    error_payload["schema_name"] = error.schema_name
            row = {
                "record_id": record["record_id"],
                "stage2_prediction": None,
                "stage2_valid": False,
                "symptom_prediction": None,
                "root_cause_prediction": None,
                "stage3_valid": False,
                "stop_reason": (
                    "transport_error"
                    if isinstance(error, ModelTransportError)
                    else "structured_output_invalid"
                ),
                "audit": None,
                "error": error_payload,
            }
        if supplemental_bundle is not None:
            if not isinstance(row.get("audit"), dict):
                row["audit"] = {}
            row["audit"]["supplemental_evidence"] = supplemental_bundle.audit_for(record["record_id"])
        if frozen_projection is not None:
            audit = row.get("audit")
            if not isinstance(audit, dict):
                audit = {}
                row["audit"] = audit
            audit["frozen_evidence"] = frozen_evidence_projection_audit(
                frozen_projection,
                cited_evidence_ids=_cited_frozen_ids(
                    audit.get("stage3"),
                    {item.evidence_id for item in frozen_projection.items},
                ),
            )
        return row

    def targeted_sla_failure_row(
        record: dict[str, str], error: SlaBudgetExhausted
    ) -> dict[str, object]:
        del error
        assert targeted_baseline_by_id is not None
        anchor = targeted_baseline_by_id[record["record_id"]]
        decision_status = {
            "symptom": "budget_fallback",
            "root_cause": "budget_fallback",
        }
        return {
            "record_id": record["record_id"],
            "stage2_prediction": None,
            "stage2_valid": False,
            "symptom_prediction": anchor.symptom_label,
            "root_cause_prediction": anchor.root_cause_label,
            "stage3_valid": True,
            "stop_reason": "targeted_sla30",
            "decision_status": decision_status,
            "call_count": 0,
            "role_timings": {},
            "call_audit": [],
            "fallback": True,
            "budget_remaining_seconds": 0.0,
            "audit": {
                "targeted_sla": {
                    "decision_status": decision_status,
                    "call_count": 0,
                    "role_timings": {},
                    "call_audit": [],
                    "fallback": True,
                    "budget_remaining_seconds": 0.0,
                    "arbitrated": False,
                }
            },
        }

    progress = None if args.no_progress else ConsoleProgress()
    run_started = (
        deadline_monotonic_started
        if deadline_monotonic_started is not None
        else time.perf_counter()
    )
    try:
        rows = run_records(
            records,
            predictions_path=predictions_path,
            config_hash=config_hash,
            record_runner=record_runner,
            resume=not args.no_resume,
            concurrency=args.concurrency,
            progress_callback=progress,
            failure_row_factory=(targeted_sla_failure_row if targeted_sla else None),
        )
    finally:
        if pooled_provider_client is not None:
            pooled_provider_client.close()
    run_wall_time = time.perf_counter() - run_started
    telemetry = _merge_llm_telemetry(
        previous_metrics.get("llm_telemetry"),
        agents.telemetry(),
    )
    evidence_only_cohort = args.stage == "stage3" and all(
        not set(record).intersection(GOLD_FIELDS) for record in records
    )
    outcome_metrics = (
        {"stage3": targeted_sla_diagnostics(records, rows, targeted_baseline_by_id)}
        if targeted_sla
        else (
            evaluate_evidence_only_execution(records, rows, stage=args.stage)
            if preservation_manifest is not None or evidence_only_cohort
            else evaluate_experiment(records, rows, stage=args.stage,
                                     **({"taxonomy": inputs.taxonomy} if inputs.taxonomy.get("annotation_modes") else {}))
        )
    )
    metrics = {
        "task": f"{args.domain}_adaptive_empirical_workflow",
        "domain": args.domain,
        "model": model,
        "provider": args.provider,
        "config_hash": config_hash,
        "llm_telemetry": telemetry,
        "run_wall_time_seconds": run_wall_time,
        "benchmark_wall_time_seconds": previous_metrics.get(
            "benchmark_wall_time_seconds",
            run_wall_time,
        ),
        "cumulative_wall_time_seconds": float(
            previous_metrics.get("cumulative_wall_time_seconds", 0)
        )
        + run_wall_time,
        **outcome_metrics,
    }
    if (
        args.stage in {"stage3", "all"}
        and preservation_manifest is None
        and not targeted_sla
        and not evidence_only_cohort
    ):
        accepted_records = records if evidence_only_cohort else [
            record for record in records if record.get("decision") == "accepted_fault"
        ]
        accepted_ids = {record["record_id"] for record in accepted_records}
        accepted_rows = [
            row for row in rows if str(row.get("record_id")) in accepted_ids
        ]
        metrics["stage3"]["dual_team"] = stage3_team_diagnostics(
            accepted_records,
            accepted_rows,
        )
    if preservation_manifest is not None:
        metrics["stage3"]["baseline_preservation"]["compute"] = (
            _preservation_compute_metrics(telemetry)
        )
    if targeted_sla:
        metrics["stage3"]["run_wall_time_seconds"] = run_wall_time
    if frozen_evidence_manifest is not None:
        metrics["frozen_evidence"] = _frozen_evidence_metrics(rows)
    if targeted_sla:
        run_wall_time = time.perf_counter() - run_started
        metrics["run_wall_time_seconds"] = run_wall_time
        metrics["benchmark_wall_time_seconds"] = run_wall_time
        metrics["cumulative_wall_time_seconds"] = run_wall_time
        metrics["stage3"]["run_wall_time_seconds"] = run_wall_time
    _write_json(metrics_path, metrics)
    summary["metrics"] = metrics
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    run_cli()


if __name__ == "__main__":
    main()
