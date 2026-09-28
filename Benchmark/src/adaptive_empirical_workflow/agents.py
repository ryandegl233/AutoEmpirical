"""Provider-neutral structured model adapters for workflow roles."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import re
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError
from .explainable_tools import CONDITIONS, GUIDANCE, ExplainableTools, ToolQuery
from .evidence_chain import (ChainAudit, CHECKER_PROMPT, ROOT_CAUSE_DECISION_POLICY, check_targets,
                             validate_chain_audit, verification_from_audit, checker_output_schema)
from Benchmark.src.annotation_contracts import (
    NEW_PAPER_DOMAINS, annotation_guidance, annotation_mode,
    canonical_label, label_valid, labels_equal,
)

from .capabilities import (
    AnalystRole,
    render_system_prompt,
    render_stage3_perspective_instructions,
    required_readiness_dimensions,
    stage3_team_policy,
)
from .contracts import (
    AnonymousStage3TeamReport,
    BaselineAnchor,
    BaselineRevisionAssessment,
    baseline_revision_assessment_digest,
    BoundaryCard,
    canonical_evidence_view_hash,
    BoundaryChallenge,
    CausalConsistencyReport,
    DimensionVerificationReport,
    DisagreementMap,
    EvidenceDimension,
    EvidenceView,
    EvidenceReadinessReport,
    FaultEvidenceAssessment,
    IsstaStage2AnalysisReport,
    JointAnchorReport,
    RootCauseReport,
    Stage3AgentPolicy,
    RepairCausalityAssessment,
    ScopeBoundaryAssessment,
    SlaConditionalArbitration,
    SlaJointDiagnosis,
    SlaJointVerification,
    SlaVerifierVerdict,
    Stage2AnalysisReport,
    Stage2ArbitrationDecision,
    Stage2ArbitrationPacket,
    Stage3ArbitrationDecision,
    Stage3ArbitrationPacket,
    Stage3TeamReport,
    SpecialistType,
    SymptomReport,
    ThinkingMode,
    TaxonomyStructure,
    RevisionAssessmentVerdict,
    RevisionEntailmentResult,
    RevisionFalsificationResult,
    EntailmentMappingCrossResult,
    FalsificationCoverageCrossResult,
    normalize_revision_entailment_result,
    normalize_revision_falsification_result,
    RevisionConsistencyReport,
    RevisionProposalEnvelope,
    revision_proposal_digest,
    sanitize_schema_name,
    sanitize_validation_summary,
    validate_sla_conditional_arbitration,
    validate_sla_joint_diagnosis,
    validate_sla_joint_verification,
)
from .domains import (
    allowed_frozen_evidence_targets,
    taxonomy_guidance,
    validate_frozen_evidence_target,
)
from .evidence_capabilities import (
    assess_stage3_evidence_validity,
    supports_readiness_dimension,
)
from .frozen_evidence_runtime import frozen_evidence_may_be_support
from .sla_budget import SlaBudgetExhausted
from .stage2_policy import stage2_policy_guidance, stage2_test_policy_guidance
from .taxonomy_structure import render_taxonomy_structure, taxonomy_structure_hash

SchemaT = TypeVar("SchemaT", bound=BaseModel)
ModelTransport = Callable[..., str]


_TELEMETRY_RECORD_ID: ContextVar[str | None] = ContextVar(
    "structured_model_telemetry_record_id", default=None
)


@contextmanager
def record_telemetry_scope(record_id: str):
    """Attach one record identity to synchronous structured-model calls."""

    token = _TELEMETRY_RECORD_ID.set(record_id)
    try:
        yield
    finally:
        _TELEMETRY_RECORD_ID.reset(token)


WORKFLOW_PROMPT_VERSION = "causal-responsibility-revision-v25"

ASE2022_CAUSAL_RESPONSIBILITY_INSTRUCTIONS = (
    "\nASE2022 CAUSAL RESPONSIBILITY: Classify the pre-fault mechanism by "
    "who owns the violated responsibility, not by the final exception or the "
    "file changed by a repair. API Misuse applies when the caller violates an "
    "external API contract, including a documented precondition, nullable or "
    "optional return, or required call sequence. Improper Exception Handling "
    "applies only when error catching or propagation is itself causal rather "
    "than merely the downstream manifestation. Incorrect Code Logic applies "
    "when the faulty mechanism is the component's own algorithm or control flow "
    "independent of an external API contract. State the closest alternative and "
    "use the cited causal chain to identify the owning boundary."
)

REVISION_ENTAILMENT_SYSTEM_INSTRUCTIONS = (
    "ROLE: revision_entailment\n"
    "Perform only a positive entailment review of the exact candidate under the "
    "single supplied Boundary Card. Cover every proposed positive condition and "
    "one exact Baseline exclusion condition. A condition is supported only by "
    "proposal-owned positive evidence; counter-only evidence cannot establish it. "
    "Copy the exact semantic condition strings and proposal-owned citation IDs "
    "from context.required_findings; preserve their supplied order and do not "
    "add citation_ids outside the task-specific output fields. "
    "Return entailed only when every fixed condition is supported, not_entailed "
    "when one is refuted, and insufficient otherwise. Do not generate or copy "
    "opaque hashes, digests, team identities, or provenance bindings. A direct "
    "proposal-owned repair mechanism may support an exact card condition even "
    "when the evidence does not need to repeat the proposed taxonomy label "
    "verbatim; require the causal mechanism and direction, not a word match.\n"
)
REVISION_FALSIFICATION_SYSTEM_INSTRUCTIONS = (
    "ROLE: revision_falsification\n"
    "Perform only an adversarial falsification review of the exact candidate under "
    "the single supplied Boundary Card. Check whether the Baseline explanation "
    "survives, inspect every fixed proposed-label defeater in the supplied order, "
    "and cite the strongest competing Baseline reading. Copy the exact semantic "
    "condition strings, competing label, and proposal-owned citation IDs from "
    "context.required_findings; do not rename keys or add citation_ids outside "
    "the task-specific output fields. revision_survives means the proposed revision "
    "survived falsification; it never means the Baseline survived. Return "
    "revision_falsified when the Baseline survives or any fixed defeater is "
    "supported, and insufficient when the evidence cannot decide. Do not generate "
    "or copy opaque hashes, digests, team identities, or provenance bindings. "
    "Judge the pre-fault causal mechanism separately from the files, tools, or "
    "configuration surfaces touched by the repair. A repair touching a Baseline-"
    "named surface does not show that the Baseline survived; require evidence "
    "that the Baseline mechanism caused the fault before the repair.\n"
)
BASELINE_REVISION_CONSISTENCY_SYSTEM_INSTRUCTIONS = (
    "ROLE: baseline_revision_consistency\n"
    "Cross-check the other team's candidate-bound assessment only. Never "
    "check your own assessment. Return consistent only when the exact pair, "
    "Boundary Card, conditions, and assessment-owned citations support it. "
    "Return only the task-specific semantic findings required by the output "
    "contract. Do not output opaque identity, provenance, digest, proposal, "
    "card, report, view, trust, owner, raw-result, or assessment bindings; "
    "the program adapter supplies all such bindings.\n"
)
SLA_JOINT_DIAGNOSIS_SYSTEM_INSTRUCTIONS = (
    "SLA FAST PATH: Produce both classification dimensions from the exact frozen "
    "evidence view. For each dimension, select only an owned taxonomy label and "
    "cite only dimension-capable evidence. Compare each result with the supplied "
    "Baseline label and persist each exact label-equality result in the typed "
    "comparison fields, but do not infer any answer from dataset membership. State an "
    "evidence-bound pre-fix mechanism and a concise causal chain. Use only the "
    "exact labels in context.taxonomy_labels; do not invent another label.\n"
)
SLA_JOINT_VERIFIER_SYSTEM_INSTRUCTIONS = (
    "SLA FAST PATH: Independently check the canonical candidate label, causal "
    "chain, and citation projections against the exact frozen evidence and supplied "
    "Baseline pair. Do not use, request, or reconstruct free-text diagnosis "
    "rationales. For each dimension return only accept_candidate, preserve_baseline, "
    "or unresolved, with exact dimension-capable citations: candidate acceptance "
    "requires supporting evidence, Baseline preservation requires counter-evidence "
    "against the candidate, and unresolved requires cited ambiguity evidence.\n"
)
SLA_CONDITIONAL_ARBITRATOR_SYSTEM_INSTRUCTIONS = (
    "SLA FAST PATH: Resolve only the supplied conflicting candidate-versus-Baseline "
    "pairs. Emit exactly those conflicting dimensions. For each choice select exactly "
    "the candidate or Baseline value; never invent a third label or reopen taxonomy-wide "
    "classification. Cite at least one exact, dimension-capable supporting item for "
    "every emitted choice.\n"
)
_SLA_ROLE_MAX_TOKENS = {
    AnalystRole.SLA_JOINT_DIAGNOSIS: 2400,
    AnalystRole.SLA_JOINT_VERIFIER: 1800,
    AnalystRole.SLA_CONDITIONAL_ARBITRATOR: 1800,
}
_SLA_SCHEMA_REPAIRS = 1

_SPECIALIST_FROZEN_SOURCE_TYPES = {
    SpecialistType.ISSUE_PR: frozenset({"issue_body", "issue_comments"}),
    SpecialistType.COMMIT_HISTORY: frozenset({"commit_history"}),
    SpecialistType.CODE_CONTEXT: frozenset({"changed_files", "code_diff"}),
    SpecialistType.TEST_EVIDENCE: frozenset({"code_diff"}),
    SpecialistType.TAXONOMY_KNOWLEDGE: frozenset({"taxonomy"}),
}


class StructuredOutputError(ValueError):
    """Raised after bounded schema-repair attempts are exhausted."""

    def __init__(
        self,
        message: str,
        *,
        schema_name: str | None = None,
        validation_summary: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.schema_name = sanitize_schema_name(schema_name)
        self.validation_summary = sanitize_validation_summary(validation_summary)


_TRANSPORT_CAUSE_TYPES = frozenset(
    {
        "http_error",
        "incomplete_read",
        "remote_disconnected",
        "url_error",
        "timeout_error",
        "connection_error",
        "unknown_error",
    }
)


class ModelTransportError(RuntimeError):
    """A sanitized provider transport failure with retry-safe metadata only."""

    def __init__(
        self,
        message: str,
        *,
        attempts: int = 1,
        retryable: bool = False,
        cause_type: str = "unknown_error",
        http_status: int | None = None,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        if attempts < 1:
            raise ValueError("attempts must be positive")
        normalized_cause_type = (
            cause_type if cause_type in _TRANSPORT_CAUSE_TYPES else "unknown_error"
        )
        self.attempts = attempts
        self.retryable = bool(retryable) and normalized_cause_type != "unknown_error"
        self.cause_type = normalized_cause_type
        self.http_status = http_status if isinstance(http_status, int) else None
        candidate_retry_after = (
            float(retry_after_seconds)
            if isinstance(retry_after_seconds, (int, float))
            else None
        )
        self.retry_after_seconds = (
            candidate_retry_after
            if candidate_retry_after is not None
            and math.isfinite(candidate_retry_after)
            and candidate_retry_after >= 0
            else None
        )


class ModelTransportResponse(str):
    """String response carrying provider retry metadata."""

    def __new__(
        cls,
        value: str,
        *,
        network_attempts: int = 1,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        reasoning_characters: int = 0,
        network_attempt_details: Sequence[Mapping[str, object]] = (),
    ) -> "ModelTransportResponse":
        if network_attempts < 1:
            raise ValueError("network_attempts must be positive")
        if prompt_tokens is not None and prompt_tokens < 0:
            raise ValueError("prompt_tokens must be non-negative")
        if completion_tokens is not None and completion_tokens < 0:
            raise ValueError("completion_tokens must be non-negative")
        if reasoning_characters < 0:
            raise ValueError("reasoning_characters must be non-negative")
        details = tuple(dict(event) for event in network_attempt_details)
        if details and len(details) != network_attempts:
            raise ValueError(
                "network_attempt_details must describe every network attempt"
            )
        for index, event in enumerate(details, start=1):
            if event.get("attempt") != index:
                raise ValueError("network attempt numbers must be contiguous")
            if event.get("status") not in {
                "success",
                "retryable_error",
                "error",
            }:
                raise ValueError("invalid network attempt status")
            latency = event.get("latency_seconds")
            if (
                not isinstance(latency, (int, float))
                or not math.isfinite(float(latency))
                or float(latency) < 0
            ):
                raise ValueError(
                    "network attempt latency must be finite and non-negative"
                )
        instance = super().__new__(cls, value)
        instance.network_attempts = network_attempts
        instance.prompt_tokens = prompt_tokens
        instance.completion_tokens = completion_tokens
        instance.reasoning_characters = reasoning_characters
        instance.network_attempt_details = details
        return instance


@dataclass(frozen=True)
class ModelCallOptions:
    """Complete resolved transport settings for one model invocation.

    ``share_transport_schema_budget`` is reserved for optional calls whose
    transport failures may consume the same bounded loop as schema repairs.
    """

    role: str
    team_id: str | None
    perspective: str | None
    max_tokens: int
    thinking_enabled: bool | None
    schema_attempt: int = 1
    share_transport_schema_budget: bool = False
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.temperature) or not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be finite and between 0 and 2")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if self.schema_attempt < 1:
            raise ValueError("schema_attempt must be positive")


def _unwrap_json(raw: str) -> str:
    stripped = raw.strip()
    fenced = re.fullmatch(
        r"```(?:json)?[ \t]*\r?\n(?P<body>[\s\S]*?)\r?\n```",
        stripped,
        flags=re.IGNORECASE,
    )
    return fenced.group("body").strip() if fenced else stripped


class StructuredModelClient:
    """Validates model output and retries only schema/JSON failures."""

    def __init__(
        self,
        transport: ModelTransport,
        *,
        max_schema_retries: int = 2,
        retry_delay_seconds: float = 2.0,
        waiter: Callable[[float], None] | None = None,
    ) -> None:
        if max_schema_retries < 0:
            raise ValueError("max_schema_retries must be non-negative")
        if retry_delay_seconds < 0:
            raise ValueError("retry_delay_seconds must be non-negative")
        self._transport = transport
        self._max_schema_retries = max_schema_retries
        self._retry_delay_seconds = retry_delay_seconds
        self._waiter = waiter or time.sleep
        parameters = inspect.signature(transport).parameters
        explicit_option_kinds = {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
        is_two_positional_legacy = len(parameters) == 2 and all(
            parameter.kind
            in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            }
            for parameter in parameters.values()
        )
        options_parameter = parameters.get("options")
        max_tokens_parameter = parameters.get("max_tokens")
        thinking_parameter = parameters.get("thinking_enabled")
        if is_two_positional_legacy:
            self._transport_mode = "legacy"
        elif (
            options_parameter is not None
            and options_parameter.kind in explicit_option_kinds
        ):
            self._transport_mode = "options"
        elif (
            max_tokens_parameter is not None
            and max_tokens_parameter.kind in explicit_option_kinds
            and thinking_parameter is not None
            and thinking_parameter.kind in explicit_option_kinds
        ):
            self._transport_mode = "expanded_options"
        elif any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        ):
            self._transport_mode = "generic_kwargs"
        elif any(
            name in parameters for name in ("options", "max_tokens", "thinking_enabled")
        ):
            self._transport_mode = "partial_options"
        else:
            self._transport_mode = "invalid_signature"
        self._telemetry_lock = threading.Lock()
        self._calls: list[dict[str, object]] = []

    def telemetry(self, *, record_id: str | None = None) -> dict[str, object]:
        with self._telemetry_lock:
            calls = [
                dict(call)
                for call in self._calls
                if record_id is None or call.get("record_id") == record_id
            ]
        first_pass_success_count = sum(
            call["status"] == "valid" and int(call["attempts"]) == 1 for call in calls
        )
        schema_retry_calls = sum(
            int(call.get("schema_validation_failure_count", 0)) > 0 for call in calls
        )
        transport_retry_calls = sum(
            int(call.get("transport_retry_count", 0)) > 0 for call in calls
        )
        provider_retry_calls = sum(
            any(
                event.get("status") == "retryable_error"
                for event in call.get("network_attempt_details", [])
            )
            for call in calls
        )
        return {
            "total_calls": len(calls),
            "total_attempts": sum(int(call["network_attempts"]) for call in calls),
            "total_latency_seconds": sum(
                float(call["latency_seconds"]) for call in calls
            ),
            "estimated_prompt_tokens": sum(
                int(call["prompt_characters"]) for call in calls
            )
            // 4,
            "estimated_completion_tokens": sum(
                int(call["completion_characters"]) for call in calls
            )
            // 4,
            "actual_prompt_tokens": _sum_available_usage(calls, "actual_prompt_tokens"),
            "actual_completion_tokens": _sum_available_usage(
                calls, "actual_completion_tokens"
            ),
            "reasoning_characters": sum(
                int(call.get("reasoning_characters", 0)) for call in calls
            ),
            "first_pass_success_count": first_pass_success_count,
            "first_pass_success_rate": (
                first_pass_success_count / len(calls) if calls else 0.0
            ),
            "schema_retry_calls": schema_retry_calls,
            "schema_validation_failure_count": sum(
                int(call.get("schema_validation_failure_count", 0)) for call in calls
            ),
            "transport_retry_calls": transport_retry_calls,
            "transport_retry_count": sum(
                int(call.get("transport_retry_count", 0)) for call in calls
            ),
            "provider_retry_calls": provider_retry_calls,
            "provider_retry_count": sum(
                event.get("status") == "retryable_error"
                for call in calls
                for event in call.get("network_attempt_details", [])
            ),
            "transport_error_count": sum(
                int(call.get("transport_error_count", 0)) for call in calls
            ),
            "calls": calls,
        }

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: type[SchemaT],
        role: str = "unspecified",
        max_tokens: int | None = None,
        options: ModelCallOptions | None = None,
        result_validator: Callable[[SchemaT], None] | None = None,
        max_schema_retries: int | None = None,
        validation_observer: Callable[[str, Exception | None], None] | None = None,
    ) -> SchemaT:
        if max_schema_retries is not None and max_schema_retries < 0:
            raise ValueError("max_schema_retries must be non-negative")
        call_options = options or ModelCallOptions(
            role=role,
            team_id=None,
            perspective=None,
            max_tokens=max_tokens or 1600,
            thinking_enabled=None,
        )
        current_prompt = user_prompt
        started = time.perf_counter()
        attempts = 0
        prompt_characters = 0
        completion_characters = 0
        network_attempts = 0
        actual_prompt_tokens = 0
        actual_completion_tokens = 0
        has_actual_prompt_tokens = False
        has_actual_completion_tokens = False
        reasoning_characters = 0
        validation_errors: list[dict[str, object]] = []
        transport_failures: list[dict[str, object]] = []
        transport_error_count = 0
        transport_retry_count = 0
        network_attempt_details: list[dict[str, object]] = []
        attempt_options = call_options
        attempt_policies: list[dict[str, object]] = []
        schema_retry_limit = (
            self._max_schema_retries
            if max_schema_retries is None
            else max_schema_retries
        )
        for attempt in range(schema_retry_limit + 1):
            attempts += 1
            attempt_options = replace(attempt_options, schema_attempt=attempts)
            attempt_policies.append(
                {
                    "thinking_enabled": attempt_options.thinking_enabled,
                    **({"temperature": attempt_options.temperature} if attempt_options.temperature else {}),
                    "max_tokens": attempt_options.max_tokens,
                }
            )
            attempt_prompt_characters = len(system_prompt) + len(current_prompt)
            try:
                raw = self._call_transport(
                    system_prompt,
                    current_prompt,
                    attempt_options,
                )
            except SlaBudgetExhausted as error:
                failed_attempts = int(getattr(error, "attempts", 0))
                network_attempts += failed_attempts
                prompt_characters += attempt_prompt_characters * failed_attempts
                network_attempt_details.extend(
                    dict(event)
                    for event in getattr(error, "network_attempt_details", ())
                )
                self._record_call(
                    options=call_options,
                    attempt_policies=attempt_policies,
                    attempts=attempts,
                    network_attempts=network_attempts,
                    started=started,
                    status="budget_exhausted",
                    prompt_characters=prompt_characters,
                    completion_characters=completion_characters,
                    validation_errors=validation_errors,
                    actual_prompt_tokens=(
                        actual_prompt_tokens if has_actual_prompt_tokens else None
                    ),
                    actual_completion_tokens=(
                        actual_completion_tokens
                        if has_actual_completion_tokens
                        else None
                    ),
                    reasoning_characters=reasoning_characters,
                    transport_error_count=transport_error_count,
                    transport_retry_count=transport_retry_count,
                    transport_failures=transport_failures,
                    network_attempt_details=network_attempt_details,
                )
                raise
            except ModelTransportError as error:
                failed_attempts = int(getattr(error, "attempts", 1))
                network_attempts += failed_attempts
                prompt_characters += attempt_prompt_characters * failed_attempts
                transport_error_count += 1
                network_attempt_details.extend(
                    dict(event)
                    for event in getattr(error, "network_attempt_details", ())
                )
                transport_failures.append(
                    {
                        "retryable": error.retryable,
                        "cause_type": error.cause_type,
                        "http_status": error.http_status,
                        "retry_after_seconds": error.retry_after_seconds,
                    }
                )
                if (
                    call_options.share_transport_schema_budget
                    and error.retryable
                    and attempt < schema_retry_limit
                    and network_attempts < schema_retry_limit + 1
                ):
                    delay_seconds = (
                        error.retry_after_seconds
                        if error.retry_after_seconds is not None
                        else self._retry_delay_seconds * (2**transport_retry_count)
                    )
                    transport_retry_count += 1
                    if delay_seconds:
                        self._waiter(delay_seconds)
                    continue
                self._record_call(
                    options=call_options,
                    attempt_policies=attempt_policies,
                    attempts=attempts,
                    network_attempts=network_attempts,
                    started=started,
                    status="transport_error",
                    prompt_characters=prompt_characters,
                    completion_characters=completion_characters,
                    validation_errors=validation_errors,
                    actual_prompt_tokens=(
                        actual_prompt_tokens if has_actual_prompt_tokens else None
                    ),
                    actual_completion_tokens=(
                        actual_completion_tokens
                        if has_actual_completion_tokens
                        else None
                    ),
                    reasoning_characters=reasoning_characters,
                    transport_error_count=transport_error_count,
                    transport_retry_count=transport_retry_count,
                    transport_failures=transport_failures,
                    network_attempt_details=network_attempt_details,
                )
                raise
            except Exception as error:
                failed_attempts = int(getattr(error, "attempts", 1))
                network_attempts += failed_attempts
                prompt_characters += attempt_prompt_characters * failed_attempts
                self._record_call(
                    options=call_options,
                    attempt_policies=attempt_policies,
                    attempts=attempts,
                    network_attempts=network_attempts,
                    started=started,
                    status="transport_error",
                    prompt_characters=prompt_characters,
                    completion_characters=completion_characters,
                    validation_errors=validation_errors,
                    actual_prompt_tokens=(
                        actual_prompt_tokens if has_actual_prompt_tokens else None
                    ),
                    actual_completion_tokens=(
                        actual_completion_tokens
                        if has_actual_completion_tokens
                        else None
                    ),
                    reasoning_characters=reasoning_characters,
                    transport_error_count=transport_error_count,
                    transport_retry_count=transport_retry_count,
                    transport_failures=transport_failures,
                    network_attempt_details=network_attempt_details,
                )
                raise
            response_attempts = int(getattr(raw, "network_attempts", 1))
            network_attempts += response_attempts
            prompt_characters += attempt_prompt_characters * response_attempts
            completion_characters += len(raw)
            response_prompt_tokens = getattr(raw, "prompt_tokens", None)
            response_completion_tokens = getattr(raw, "completion_tokens", None)
            if response_prompt_tokens is not None:
                actual_prompt_tokens += int(response_prompt_tokens)
                has_actual_prompt_tokens = True
            if response_completion_tokens is not None:
                actual_completion_tokens += int(response_completion_tokens)
                has_actual_completion_tokens = True
            reasoning_characters += int(getattr(raw, "reasoning_characters", 0))
            network_attempt_details.extend(
                dict(event) for event in getattr(raw, "network_attempt_details", ())
            )
            try:
                result = schema.model_validate_json(_unwrap_json(raw))
                if result_validator is not None:
                    result_validator(result)
                if validation_observer is not None:
                    validation_observer(str(raw), None)
                self._record_call(
                    options=call_options,
                    attempt_policies=attempt_policies,
                    attempts=attempts,
                    network_attempts=network_attempts,
                    started=started,
                    status="valid",
                    prompt_characters=prompt_characters,
                    completion_characters=completion_characters,
                    validation_errors=validation_errors,
                    actual_prompt_tokens=(
                        actual_prompt_tokens if has_actual_prompt_tokens else None
                    ),
                    actual_completion_tokens=(
                        actual_completion_tokens
                        if has_actual_completion_tokens
                        else None
                    ),
                    reasoning_characters=reasoning_characters,
                    transport_error_count=transport_error_count,
                    transport_retry_count=transport_retry_count,
                    transport_failures=transport_failures,
                    network_attempt_details=network_attempt_details,
                )
                return result
            except (ValidationError, ValueError) as error:
                if validation_observer is not None:
                    validation_observer(str(raw), error)
                validation_errors.append(_validation_error_summary(error))
                if attempt >= schema_retry_limit:
                    self._record_call(
                        options=call_options,
                        attempt_policies=attempt_policies,
                        attempts=attempts,
                        network_attempts=network_attempts,
                        started=started,
                        status="invalid",
                        prompt_characters=prompt_characters,
                        completion_characters=completion_characters,
                        validation_errors=validation_errors,
                        actual_prompt_tokens=(
                            actual_prompt_tokens if has_actual_prompt_tokens else None
                        ),
                        actual_completion_tokens=(
                            actual_completion_tokens
                            if has_actual_completion_tokens
                            else None
                        ),
                        reasoning_characters=reasoning_characters,
                        transport_error_count=transport_error_count,
                        transport_retry_count=transport_retry_count,
                        transport_failures=transport_failures,
                        network_attempt_details=network_attempt_details,
                    )
                    raise StructuredOutputError(
                        f"model output failed {schema.__name__} validation "
                        f"after {attempts} attempt(s)",
                        schema_name=schema.__name__,
                        validation_summary=_validation_error_summary(error),
                    ) from error
                if attempt_options.thinking_enabled is True:
                    attempt_options = replace(
                        attempt_options,
                        thinking_enabled=False,
                    )
                current_prompt = (
                    f"{user_prompt}\n\n"
                    "Validation failed. Return a corrected JSON object only.\n"
                    f"VALIDATION ERROR:\n{error}\n"
                    f"INVALID OUTPUT:\n{raw}"
                )
        raise RuntimeError("unreachable structured completion state")

    def _call_transport(
        self,
        system_prompt: str,
        user_prompt: str,
        options: ModelCallOptions,
    ) -> str:
        if self._transport_mode == "options":
            return self._transport(system_prompt, user_prompt, options=options)
        if options.temperature:
            raise TypeError("Nonzero role temperature requires a transport accepting ModelCallOptions")
        if self._transport_mode == "expanded_options":
            return self._transport(
                system_prompt,
                user_prompt,
                max_tokens=options.max_tokens,
                thinking_enabled=options.thinking_enabled,
            )
        if self._transport_mode == "partial_options":
            raise TypeError(
                "partial option-aware transport must declare either explicit "
                "options or both max_tokens and thinking_enabled"
            )
        if self._transport_mode == "generic_kwargs":
            raise TypeError(
                "generic kwargs transport must declare explicit options or both "
                "max_tokens and thinking_enabled"
            )
        if self._transport_mode == "legacy":
            return self._transport(system_prompt, user_prompt)
        raise TypeError(
            "transport must accept exactly two positional arguments, explicit "
            "options, or both max_tokens and thinking_enabled"
        )

    def _record_call(
        self,
        *,
        options: ModelCallOptions,
        attempt_policies: list[dict[str, object]],
        attempts: int,
        network_attempts: int,
        started: float,
        status: str,
        prompt_characters: int,
        completion_characters: int,
        validation_errors: list[dict[str, object]],
        actual_prompt_tokens: int | None,
        actual_completion_tokens: int | None,
        reasoning_characters: int,
        transport_error_count: int,
        transport_retry_count: int,
        transport_failures: list[dict[str, object]],
        network_attempt_details: list[dict[str, object]],
    ) -> None:
        event = {
            "role": options.role,
            "team_id": options.team_id,
            "perspective": options.perspective,
            "thinking_enabled": options.thinking_enabled,
            "temperature": options.temperature,
            "attempts": attempts,
            "network_attempts": network_attempts,
            "latency_seconds": time.perf_counter() - started,
            "max_tokens": options.max_tokens,
            "status": status,
            "prompt_characters": prompt_characters,
            "completion_characters": completion_characters,
            "estimated_prompt_tokens": prompt_characters // 4,
            "estimated_completion_tokens": completion_characters // 4,
            "actual_prompt_tokens": actual_prompt_tokens,
            "actual_completion_tokens": actual_completion_tokens,
            "reasoning_characters": reasoning_characters,
            "validation_errors": [dict(error) for error in validation_errors],
            "schema_validation_failure_count": len(validation_errors),
            "transport_error_count": transport_error_count,
            "transport_retry_count": transport_retry_count,
            "transport_failures": [dict(failure) for failure in transport_failures],
            "network_attempt_details": [
                dict(event) for event in network_attempt_details
            ],
            "network_budget_limit": (
                self._max_schema_retries + 1
                if options.share_transport_schema_budget
                else None
            ),
            "network_budget_overspent": (
                max(0, network_attempts - (self._max_schema_retries + 1))
                if options.share_transport_schema_budget
                else 0
            ),
            "attempt_policies": [dict(policy) for policy in attempt_policies],
        }
        record_id = _TELEMETRY_RECORD_ID.get()
        if record_id is not None:
            event["record_id"] = record_id
        with self._telemetry_lock:
            self._calls.append(event)


def _sum_available_usage(
    calls: list[dict[str, object]],
    field: str,
) -> int | None:
    available = [call[field] for call in calls if call.get(field) is not None]
    return sum(int(value) for value in available) if available else None


def _compact_schema(schema: type[BaseModel]) -> dict[str, Any]:
    """Remove descriptive JSON Schema metadata while retaining constraints."""

    def compact(value: object) -> object:
        if isinstance(value, dict):
            return {
                key: compact(item)
                for key, item in value.items()
                if key not in {"title", "description", "default"}
            }
        if isinstance(value, list):
            return [compact(item) for item in value]
        return value

    return compact(schema.model_json_schema())  # type: ignore[return-value]


def baseline_revision_prompt_hashes() -> dict[str, str]:
    """Hash the exact fixed instructions and structured output contracts."""

    prompts = {
        "revision_entailment": {
            "instructions": REVISION_ENTAILMENT_SYSTEM_INSTRUCTIONS,
            "schema": _compact_schema(RevisionEntailmentResult),
        },
        "revision_falsification": {
            "instructions": REVISION_FALSIFICATION_SYSTEM_INSTRUCTIONS,
            "schema": _compact_schema(RevisionFalsificationResult),
        },
        "falsification_coverage_cross": {
            "instructions": BASELINE_REVISION_CONSISTENCY_SYSTEM_INSTRUCTIONS
            + "AUDIT TARGET: falsification coverage.",
            "schema": _compact_schema(FalsificationCoverageCrossResult),
        },
        "entailment_mapping_cross": {
            "instructions": BASELINE_REVISION_CONSISTENCY_SYSTEM_INSTRUCTIONS
            + "AUDIT TARGET: entailment mapping.",
            "schema": _compact_schema(EntailmentMappingCrossResult),
        },
    }
    return {
        role: hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        for role, payload in prompts.items()
    }


def baseline_revision_schema_hashes() -> dict[str, str]:
    """Hash revision output schemas separately for resume compatibility checks."""

    schemas = {
        "revision_entailment": _compact_schema(RevisionEntailmentResult),
        "revision_falsification": _compact_schema(RevisionFalsificationResult),
        "falsification_coverage_cross": _compact_schema(
            FalsificationCoverageCrossResult
        ),
        "entailment_mapping_cross": _compact_schema(EntailmentMappingCrossResult),
    }
    return {
        role: hashlib.sha256(
            json.dumps(
                schema,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        for role, schema in schemas.items()
    }


def sla_prompt_hashes() -> dict[str, str]:
    """Hash the fixed bounded SLA prompts and their output contracts."""

    prompts = {
        AnalystRole.SLA_JOINT_DIAGNOSIS.value: {
            "instructions": SLA_JOINT_DIAGNOSIS_SYSTEM_INSTRUCTIONS,
            "schema": _compact_schema(SlaJointDiagnosis),
        },
        AnalystRole.SLA_JOINT_VERIFIER.value: {
            "instructions": SLA_JOINT_VERIFIER_SYSTEM_INSTRUCTIONS,
            "schema": _compact_schema(SlaJointVerification),
        },
        AnalystRole.SLA_CONDITIONAL_ARBITRATOR.value: {
            "instructions": SLA_CONDITIONAL_ARBITRATOR_SYSTEM_INSTRUCTIONS,
            "schema": _compact_schema(SlaConditionalArbitration),
        },
    }
    return {
        role: hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        for role, payload in prompts.items()
    }


def _validation_error_summary(error: Exception) -> dict[str, object]:
    if isinstance(error, ValidationError):
        details = error.errors(
            include_url=False,
            include_input=False,
        )
        return {
            "error_type": "ValidationError",
            "fields": [
                ".".join(str(part) for part in detail.get("loc", ())) or "__root__"
                for detail in details
            ],
            "codes": [str(detail.get("type", "")) for detail in details],
        }
    return {
        "error_type": type(error).__name__,
        "fields": ["__root__"],
        "codes": ["value_error"],
    }


def _prompt_payload(
    *,
    team_id: str,
    view: EvidenceView,
    context: dict[str, object] | None = None,
) -> str:
    def contains_forbidden_metadata(value: object) -> bool:
        def canonical_key(key: object) -> str:
            return re.sub(r"[^a-z0-9]", "", str(key).lower())

        forbidden = {
            "groundtruth",
            "gold",
            "goldlabel",
            "expectedanswer",
            "labelanswer",
            "answerlabel",
            "targetlabel",
            "referencelabel",
            "symptomlabel",
            "rootcauselabel",
        }
        if isinstance(value, Mapping):
            return any(
                canonical_key(key) in forbidden or contains_forbidden_metadata(item)
                for key, item in value.items()
            )
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            return any(contains_forbidden_metadata(item) for item in value)
        return False

    if any(contains_forbidden_metadata(item.metadata) for item in view.items):
        raise ValueError("evidence metadata contains forbidden answer-bearing fields")
    payload: dict[str, object] = {
        "team_id": team_id,
        "evidence_view": view.model_dump(
            mode="json",
            exclude={"taxonomy", "domain_profile"},
        ),
    }
    if context:
        payload["context"] = context
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _sla_prompt_view(view: EvidenceView) -> EvidenceView:
    """Remove non-evidence metadata before a bounded SLA role sees the view."""

    return view.model_copy(
        update={
            "items": tuple(
                item.model_copy(update={"metadata": {}}) for item in view.items
            )
        }
    )


def _candidate_prompt_view(
    view: EvidenceView, proposal: RevisionProposalEnvelope
) -> EvidenceView:
    """Project a full trusted view onto only evidence owned by one proposal."""

    candidate_ids = {
        *proposal.supporting_evidence_ids,
        *proposal.counter_evidence_ids,
    }
    return view.model_copy(
        update={
            "items": tuple(
                item for item in view.items if item.evidence_id in candidate_ids
            )
        }
    )


def _validate_citation_fields(
    report: BaseModel,
    view: EvidenceView,
    *fields: str,
) -> None:
    """Reject role output citations absent from its exact canonical view."""

    known_ids = {item.evidence_id for item in view.items}
    item_by_id = {item.evidence_id: item for item in view.items}
    for field in fields:
        cited_ids = getattr(report, field)
        unknown_ids = tuple(
            dict.fromkeys(
                evidence_id for evidence_id in cited_ids if evidence_id not in known_ids
            )
        )
        if unknown_ids:
            raise ValueError(
                f"{type(report).__name__}.{field} cites evidence IDs not present "
                f"in the exact evidence view: {list(unknown_ids)!r}"
            )
        if "supporting" in field or "boundary" in field:
            counter_only_ids = tuple(
                evidence_id
                for evidence_id in cited_ids
                if not frozen_evidence_may_be_support(item_by_id[evidence_id])
            )
            if counter_only_ids:
                raise ValueError(
                    f"{type(report).__name__}.{field} cannot use counter-only "
                    f"frozen evidence as positive support: {list(counter_only_ids)!r}"
                )


def _validate_classification_labels(
    report: SymptomReport | RootCauseReport,
    view: EvidenceView,
    dimension: EvidenceDimension,
) -> None:
    allowed = set(view.taxonomy.get(dimension.value, ()))
    if not label_valid(view.taxonomy, dimension.value, report.label) or not label_valid(view.taxonomy, dimension.value, report.alternative_label):
        raise ValueError(
            f"{dimension.value} label and alternative_label must belong to the "
            f"supplied {dimension.value} taxonomy"
        )
    if len(allowed) > 1 and labels_equal(view.taxonomy, dimension.value, report.label, report.alternative_label):
        raise ValueError(f"{dimension.value} alternative_label must be distinct")


def _validate_joint_anchor(report: JointAnchorReport, view: EvidenceView) -> None:
    _validate_classification_labels(report.symptom, view, EvidenceDimension.SYMPTOM)
    _validate_classification_labels(
        report.root_cause,
        view,
        EvidenceDimension.ROOT_CAUSE,
    )
    _validate_citation_fields(
        report.symptom,
        view,
        "supporting_evidence_ids",
        "counter_evidence_ids",
        "boundary_evidence_ids",
    )
    _validate_citation_fields(
        report.root_cause,
        view,
        "supporting_evidence_ids",
        "counter_evidence_ids",
        "boundary_evidence_ids",
    )
    _validate_citation_fields(
        report,
        view,
        "shared_supporting_evidence_ids",
        "shared_counter_evidence_ids",
        "shared_boundary_evidence_ids",
    )
    items_by_id = {item.evidence_id: item for item in view.items}
    for dimension, classification in (
        (EvidenceDimension.SYMPTOM, report.symptom),
        (EvidenceDimension.ROOT_CAUSE, report.root_cause),
    ):
        for field_name in ("supporting_evidence_ids", "boundary_evidence_ids"):
            for evidence_id in getattr(classification, field_name):
                if not supports_readiness_dimension(
                    items_by_id[evidence_id],
                    dimension,
                    domain=view.domain_profile,
                ):
                    raise ValueError(
                        f"{field_name} evidence_id {evidence_id!r} does not support "
                        f"the {dimension.value} dimension"
                    )


def _joint_anchor_evidence_eligibility(view: EvidenceView) -> dict[str, list[str]]:
    return {
        item.evidence_id: sorted(
            dimension.value
            for dimension in (
                EvidenceDimension.SYMPTOM,
                EvidenceDimension.ROOT_CAUSE,
            )
            if supports_readiness_dimension(
                item,
                dimension,
                domain=view.domain_profile,
            )
        )
        for item in view.items
    }


def _joint_label_ownership_instruction(
    subject: str,
    set_name: str,
    labels: list[str],
) -> str:
    label_set = json.dumps(labels, ensure_ascii=False)
    if len(labels) == 1:
        return (
            f"The {subject} label and alternative_label must both use the only member "
            f"of this exact {set_name} LABEL SET: {label_set}; they may therefore be "
            "identical."
        )
    if len(labels) > 1:
        return (
            f"The {subject} label and alternative_label must be distinct members of "
            f"this exact {set_name} LABEL SET: {label_set}."
        )
    return (
        f"The {subject} label and alternative_label must both belong to this exact "
        f"{set_name} LABEL SET: {label_set}."
    )


class StructuredRoleAgents:
    """Concrete callable role set accepted by both workflow controllers."""

    def __init__(
        self,
        client: StructuredModelClient,
        *,
        role_max_tokens: int = 1600,
        agent_policy: Stage3AgentPolicy | None = None,
        provider_supports_thinking: bool = True,
        explainable_tools: ExplainableTools | None = None,
        chain_checker: bool = False,
        global_supervisor: bool = False,
        experiment_two_arm: str | None = None,
        source_graph_mode: str | None = None,
    ) -> None:
        if role_max_tokens < 1:
            raise ValueError("role_max_tokens must be positive")
        self._client = client
        self._agent_policy = agent_policy or Stage3AgentPolicy.from_profile(
            "all-stage3",
            default_max_tokens=role_max_tokens,
        )
        self._provider_supports_thinking = provider_supports_thinking
        self._explainable_tools = explainable_tools
        from .experiment_two import arm_options
        experiment_options = arm_options(experiment_two_arm) if experiment_two_arm else {}
        self._experiment_two_arm = experiment_two_arm
        self._rule_checks = experiment_options.get('rule_checks', False)
        self._source_graph_mode = source_graph_mode or experiment_options.get('source_graph_mode', 'off')
        self._chain_checker_enabled = chain_checker or self._rule_checks
        self.global_supervisor_enabled = global_supervisor
        self._supervisor_feedback = ContextVar("supervisor_feedback", default=None)
        self._module_b_calls: list[dict[str, object]] = []
        self._module_b_attempts: list[dict[str, object]] = []
        self._module_a_calls: list[dict[str, object]] = []
        self._module_a_attempts: list[dict[str, object]] = []
        self._final_claim_audits: list[dict[str, object]] = []
        self._module_a_lock = threading.Lock()

    def _payload(self, *, team_id, view, context=None):
        if self._source_graph_mode != 'off':
            from .source_graph import source_context, case_query
            context = {**(context or {}), 'source_navigation': source_context(
                view, case_query(view), mode=self._source_graph_mode)}
        return _prompt_payload(team_id=team_id, view=view, context=context)

    def telemetry(self, *, record_id: str | None = None) -> dict[str, object]:
        result = self._client.telemetry(record_id=record_id)
        if self._rule_checks:
            with self._module_a_lock:
                result['final_claim_audits'] = [x for x in self._final_claim_audits
                    if record_id is None or x['record_id'] == record_id]
        if self._explainable_tools is not None:
            with self._module_a_lock:
                result["module_a_calls"] = [entry for entry in self._module_a_calls
                                            if record_id is None or entry["record_id"] == record_id]
                if hasattr(self._explainable_tools, 'decision_policy'):
                    result['module_a_attempts'] = [entry for entry in self._module_a_attempts
                                                  if record_id is None or entry['record_id'] == record_id]
        if self._chain_checker_enabled:
            with self._module_a_lock:
                result["module_b_calls"] = [entry for entry in self._module_b_calls
                                            if record_id is None or entry["record_id"] == record_id]
                result["module_b_attempts"] = [entry for entry in self._module_b_attempts
                                               if record_id is None or entry["record_id"] == record_id]
        return result

    def _model_call_options(
        self,
        role: str,
        *,
        team_id: str | None,
        perspective: str | None,
    ) -> ModelCallOptions:
        resolved = self._agent_policy.resolve(role, team_id)
        thinking_enabled = (
            None
            if not self._provider_supports_thinking
            or resolved.thinking is ThinkingMode.PROVIDER_DEFAULT
            else resolved.thinking is ThinkingMode.ENABLED
        )
        return ModelCallOptions(
            role=role,
            team_id=team_id,
            perspective=perspective,
            max_tokens=({'symptom_verifier': 8192, 'root_cause_verifier': 8192,
                         'stage3_arbitrator': 8192}.get(role, resolved.max_tokens)
                        if self._experiment_two_arm else resolved.max_tokens),
            thinking_enabled=thinking_enabled,
            share_transport_schema_budget=role
            in {
                "baseline_revision_assessment",
                "baseline_revision_consistency",
            },
        )

    def _root_cause_decision_policy(self, domain: str, task: str) -> str:
        if (domain == "ase2022" and task == "stage3"
                and (self._chain_checker_enabled or hasattr(self._explainable_tools, "query_prompt"))):
            return "\n" + getattr(self._explainable_tools, "decision_policy", ROOT_CAUSE_DECISION_POLICY) + "\n"
        return ""

    def _a_root_gate(self, record_id: str, ledger_version: int, team_id: str):
        if not hasattr(self._explainable_tools, "decision_policy"):
            return None
        with self._module_a_lock:
            return next((entry['result']['root_gate'] for entry in reversed(self._module_a_calls)
                         if entry['record_id'] == record_id and entry['team_id'] == team_id
                         and entry['ledger_version'] == ledger_version and 'root_gate' in entry['result']), None)

    def _role(
        self,
        role: AnalystRole,
        *,
        team_id: str,
        view: EvidenceView,
        schema: type[SchemaT],
        context: dict[str, object] | None = None,
        result_validator: Callable[[SchemaT], None] | None = None,
    ) -> SchemaT:
        feedback = self._supervisor_feedback.get()
        if feedback is not None:
            context = {**(context or {}), "supervisor_feedback": feedback}
        if self._explainable_tools is not None and role in {
            AnalystRole.JOINT_ANCHOR, AnalystRole.SYMPTOM_ANALYST, AnalystRole.ROOT_CAUSE_ANALYST,
        }:
            tool = self._explainable_tools
            query_schema = tool.query_schema
            preparation = tool.run(view, query_schema())  # Reject overlap before any model call.
            query_prompt = (tool.query_prompt + "\n" + taxonomy_guidance(view.domain_profile, view.taxonomy)
                            + "\n" + json.dumps(_compact_schema(query_schema), ensure_ascii=False)
                            if hasattr(tool, "query_prompt") else
                            GUIDANCE + "\nSubmit relevant conditions with exact source quotes. "
                            "Omit irrelevant conditions; unknown may have no quotation. Return one JSON object.\n"
                            + json.dumps({"conditions": CONDITIONS, "output_contract": _compact_schema(ToolQuery)}, ensure_ascii=False))
            def retain_a_attempt(raw, error):
                with self._module_a_lock:
                    self._module_a_attempts.append(dict(record_id=view.record_id, team_id=team_id,
                        role=role.value, ledger_version=view.ledger_version, raw_response=raw,
                        status='invalid' if error else 'valid', error=str(error) if error else None))

            query = self._client.complete(
                system_prompt=query_prompt,
                user_prompt=self._payload(team_id=team_id, view=view, context={"examples": preparation["examples"],
                    **({"supervisor_feedback": feedback} if feedback is not None else {}),
                    **({"recommendations": preparation["recommendations"]} if "recommendations" in preparation else {})}),
                schema=query_schema,
                options=replace(self._model_call_options(role.value, team_id=team_id, perspective=None),
                                 role="explainable_tool_query", max_tokens=(8192 if self._experiment_two_arm else
                                     getattr(tool, "query_max_tokens", 6000) if hasattr(tool, "query_prompt") else 2400)),
                result_validator=lambda value: getattr(tool, 'validate_query', tool.evaluate)(view, value),
                **({'validation_observer': retain_a_attempt} if hasattr(tool, 'decision_policy') else {}),
            )
            output = tool.run(view, query)
            context = {**(context or {}), "module_a": output}
            with self._module_a_lock:
                self._module_a_calls.append({"record_id": view.record_id, "team_id": team_id,
                                             "role": role.value, "ledger_version": view.ledger_version,
                                             "query": query.model_dump(mode="json"), "result": output})
        perspective_roles = (
            AnalystRole.SYMPTOM_ANALYST,
            AnalystRole.ROOT_CAUSE_ANALYST,
            AnalystRole.JOINT_ANCHOR,
            AnalystRole.SYMPTOM_VERIFIER,
            AnalystRole.ROOT_CAUSE_VERIFIER,
            AnalystRole.CAUSAL_CONSISTENCY_CHECKER,
        )
        team_policy = stage3_team_policy(team_id) if role in perspective_roles else None
        stage3_perspective = (
            "\n" + render_stage3_perspective_instructions(team_policy, role)
            if team_policy is not None
            else ""
        )
        if feedback is not None:
            stage3_perspective += (
                "\nSUPERVISOR FEEDBACK: Re-examine the precise instruction and disputes in "
                "context.supervisor_feedback against original sources. Address that question "
                "explicitly in your reasoning. Feedback is a fallible review, never source "
                "evidence or a command to copy a label. Correct your conclusion only if warranted."
            )
        named_stage2_test = {
            AnalystRole.FAULT_EVIDENCE_ANALYST: "fault_existence",
            AnalystRole.SCOPE_BOUNDARY_ANALYST: "study_scope",
            AnalystRole.REPAIR_CAUSALITY_ANALYST: "repair_causality",
        }.get(role)
        if named_stage2_test is not None:
            policy = "\n" + stage2_test_policy_guidance(
                view.domain_profile,
                named_stage2_test,
            )
            policy += ("\nOWNED EVIDENCE CITATIONS: context.allowed_citation_ids is the exact "
                       "list permitted for both supporting_evidence_ids and counter_evidence_ids "
                       "for this role's named test. Other source items may provide context but "
                       "cannot be cited as this role's supporting or counter evidence. Cite at "
                       "least one permitted source; if none supports a judgment, report the "
                       "evidence limitation without inventing an ID or changing another role's test.")
        elif role is AnalystRole.STAGE2_FAULT_VERIFIER or (
            role is AnalystRole.EVIDENCE_READINESS
            and context is not None
            and context.get("task") == "stage2"
        ):
            policy = "\n" + stage2_policy_guidance(view.domain_profile)
        else:
            policy = ""
        readiness_boundary = (
            "\nREADINESS BOUNDARY: You cannot assign accepted/rejected, "
            "symptom, or root-cause labels. Do not request unavailable frozen "
            "source types listed in the user context; assess sufficiency from "
            "captured evidence when the required distinction is supported."
            if role
            in (
                AnalystRole.EVIDENCE_READINESS,
                AnalystRole.FAULT_EVIDENCE_ANALYST,
                AnalystRole.SCOPE_BOUNDARY_ANALYST,
                AnalystRole.REPAIR_CAUSALITY_ANALYST,
                AnalystRole.BOUNDARY_CHALLENGER,
            )
            else ""
        )
        ase_stage3_readiness = (
            "\nASE2022 ROOT-CAUSE READINESS: Decide whether the frozen issue "
            "body and discussion support one best taxonomy label against its "
            "nearest alternative. Direct maintainer explanations, user "
            "reproductions, and documented platform limitations may be "
            "sufficient without code, tests, patches, or commit history. Do "
            "not require an unavailable implementation artifact once the "
            "issue evidence supports the classification boundary."
            if role is AnalystRole.EVIDENCE_READINESS
            and context is not None
            and context.get("task") == "stage3"
            and view.domain_profile == "ase2022"
            else ""
        )
        ase_root_cause_evidence = (
            "\nASE2022 ROOT-CAUSE EVIDENCE: The frozen issue body and "
            "maintainer discussion are legitimate root-cause evidence. Use "
            "their direct explanations and record-local causal statements at "
            "the strength they support. Do not require unavailable code, patch, "
            "commit history, or regression tests as a prerequisite. When no "
            "change artifact is present, do not treat invert_patch as mandatory."
            if role is AnalystRole.ROOT_CAUSE_ANALYST
            and view.domain_profile == "ase2022"
            else ""
        )
        ase_causal_responsibility = (
            ASE2022_CAUSAL_RESPONSIBILITY_INSTRUCTIONS
            if view.domain_profile == "ase2022"
            and role
            in (
                AnalystRole.ROOT_CAUSE_ANALYST,
                AnalystRole.ROOT_CAUSE_VERIFIER,
                AnalystRole.JOINT_ANCHOR,
            )
            else ""
        )
        boundary_evidence_rule = (
            "\nBOUNDARY EVIDENCE: boundary_evidence_ids must cite direct, "
            "dimension-relevant evidence from supporting_evidence_ids or "
            "counter_evidence_ids. The alternative_label must be a distinct "
            "member of the supplied frozen taxonomy when a near neighbor exists."
            if role
            in (
                AnalystRole.SYMPTOM_ANALYST,
                AnalystRole.ROOT_CAUSE_ANALYST,
            )
            else ""
        )
        owned_dimension = {
            AnalystRole.SYMPTOM_ANALYST: "symptom",
            AnalystRole.ROOT_CAUSE_ANALYST: "root_cause",
            AnalystRole.SYMPTOM_VERIFIER: "symptom",
            AnalystRole.ROOT_CAUSE_VERIFIER: "root_cause",
        }.get(role)
        label_ownership = (
            "\nLABEL OWNERSHIP: Both label and alternative_label must be "
            f"distinct members of this exact OWNED LABEL SET: "
            f"{json.dumps(list(view.taxonomy.get(owned_dimension, ())), ensure_ascii=False)}. "
            "Never use a label from the other taxonomy dimension."
            if owned_dimension is not None
            else ""
        )
        verifier_contract_guidance = (
            "\nVERIFIER VERDICT SHAPES: "
            'For verdict "accept" or "insufficient_to_reject", output '
            '"alternative_label": null, "corrected_claim": null, and '
            '"corrected_causal_chain": []. '
            'For verdict "reject", alternative_label must be a supplied '
            "owned taxonomy label distinct from anchor_label, corrected_claim "
            "must be a non-null string, and supporting_evidence_ids must be a "
            "non-empty JSON array of exact evidence IDs. corrected_causal_chain "
            "must always be a JSON array of strings, never a string or object. "
            + (
                "For root-cause reject, corrected_causal_chain must contain "
                "at least 3 JSON strings."
                if role is AnalystRole.ROOT_CAUSE_VERIFIER
                else ""
            )
            if role
            in (
                AnalystRole.SYMPTOM_VERIFIER,
                AnalystRole.ROOT_CAUSE_VERIFIER,
            )
            else ""
        )
        joint_label_ownership = (
            "\nJOINT LABEL OWNERSHIP: "
            + _joint_label_ownership_instruction(
                "symptom",
                "SYMPTOM",
                list(view.taxonomy.get("symptom", ())),
            )
            + " "
            + _joint_label_ownership_instruction(
                "root-cause",
                "ROOT-CAUSE",
                list(view.taxonomy.get("root_cause", ())),
            )
            + " "
            "Never exchange labels between dimensions. POSITIVE EVIDENCE "
            "ELIGIBILITY: context.evidence_eligibility lists the exact dimensions "
            "each evidence ID may positively support. Every dimension-owned "
            "supporting_evidence_id and boundary_evidence_id must list that "
            "dimension; counter evidence does not make a positive claim."
            if role is AnalystRole.JOINT_ANCHOR
            else ""
        )
        if view.taxonomy.get("annotation_modes"):
            if owned_dimension is not None and annotation_mode(view.taxonomy, owned_dimension) != "single_label":
                label_ownership = "\nNATIVE ANNOTATION OWNERSHIP: " + annotation_guidance(view.taxonomy, owned_dimension)
                boundary_evidence_rule = "\nBOUNDARY EVIDENCE: Cite direct dimension-relevant evidence for both the proposed annotation and any alternative."
                verifier_contract_guidance = verifier_contract_guidance.replace(
                    "a supplied owned taxonomy label distinct from anchor_label",
                    "a value permitted by the native annotation contract, distinct from anchor_label",
                )
            if role is AnalystRole.JOINT_ANCHOR:
                joint_label_ownership = ("\nJOINT NATIVE ANNOTATION CONTRACT: " +
                    " ".join(annotation_guidance(view.taxonomy, dimension) for dimension in ("symptom", "root_cause")) +
                    " Every supporting evidence ID must be eligible for its owned dimension.")
        taxonomy = (
            ""
            if named_stage2_test is not None
            else taxonomy_guidance(
                view.domain_profile,
                view.taxonomy,
                include_stage3_boundaries=(
                    role
                    not in (
                        AnalystRole.STAGE2_FAULT_VERIFIER,
                        AnalystRole.EVIDENCE_READINESS,
                    )
                ),
            )
            + "\n"
        )
        root_gate = self._a_root_gate(view.record_id, view.ledger_version, team_id)
        if root_gate is not None:
            context = {**(context or {}), 'module_a_root_gate': root_gate}

        def validate_with_root_gate(value):
            if result_validator is not None:
                result_validator(value)
            if not getattr(self._explainable_tools, 'enforce_proof_labels', False) and (root_gate is None or root_gate['unknown_allowed']):
                return
            root_label = (value.root_cause.label if role is AnalystRole.JOINT_ANCHOR else
                          value.label if role is AnalystRole.ROOT_CAUSE_ANALYST else None)
            if role is AnalystRole.ROOT_CAUSE_VERIFIER:
                root_label = value.alternative_label if value.verdict.value == 'reject' else value.anchor_label
            if getattr(self._explainable_tools, 'enforce_proof_labels', False) and root_label is not None:
                if root_gate is None or root_label not in root_gate.get('allowed_labels', ()):
                    raise ValueError('Root label lacks a complete rule proof in the current team/evidence gate')
            if root_gate is None or root_gate['unknown_allowed']:
                return
            if root_label == 'Unknown':
                raise ValueError('Unknown contradicts the pre-decision gate: a specific category has support without unresolved conflict')

        result = self._client.complete(
            system_prompt=(
                taxonomy
                + policy
                + "\n"
                + render_system_prompt(
                    role,
                    include_dataset_guidance=role
                    not in (
                        AnalystRole.JOINT_ANCHOR,
                        AnalystRole.SYMPTOM_VERIFIER,
                        AnalystRole.ROOT_CAUSE_VERIFIER,
                    ),
                )
                + stage3_perspective
                + readiness_boundary
                + ase_stage3_readiness
                + ase_root_cause_evidence
                + ase_causal_responsibility
                + boundary_evidence_rule
                + label_ownership
                + verifier_contract_guidance
                + joint_label_ownership
                + self._root_cause_decision_policy(view.domain_profile,
                    "stage3" if role in {AnalystRole.JOINT_ANCHOR, AnalystRole.SYMPTOM_ANALYST,
                        AnalystRole.ROOT_CAUSE_ANALYST, AnalystRole.SYMPTOM_VERIFIER,
                        AnalystRole.ROOT_CAUSE_VERIFIER, AnalystRole.CAUSAL_CONSISTENCY_CHECKER,
                        AnalystRole.BOUNDARY_CHALLENGER} and "root_cause" in view.taxonomy else "stage2")
                + "\nOUTPUT CONTRACT:\n"
                + json.dumps(_compact_schema(schema), ensure_ascii=False)
                + "\nReturn one JSON object matching this contract exactly."
            ),
            user_prompt=self._payload(
                team_id=team_id,
                view=view,
                context=context,
            ),
            schema=schema,
            options=self._model_call_options(
                role.value,
                team_id=(None if role is AnalystRole.BOUNDARY_CHALLENGER else team_id),
                perspective=(
                    team_policy.perspective.value if team_policy is not None else None
                ),
            ),
            result_validator=validate_with_root_gate,
        )
        if view.taxonomy.get("annotation_modes"):
            def normalize_report(report, dimension):
                return report.model_copy(update={key: canonical_label(view.taxonomy, dimension, getattr(report, key))
                                                for key in ("label", "alternative_label")})
            if isinstance(result, SymptomReport):
                result = normalize_report(result, "symptom")
            elif isinstance(result, RootCauseReport):
                result = normalize_report(result, "root_cause")
            elif isinstance(result, JointAnchorReport):
                result = result.model_copy(update={"symptom": normalize_report(result.symptom, "symptom"),
                                                   "root_cause": normalize_report(result.root_cause, "root_cause")})
            elif isinstance(result, DimensionVerificationReport):
                update = {"anchor_label": canonical_label(view.taxonomy, result.dimension.value, result.anchor_label)}
                if result.alternative_label is not None:
                    update["alternative_label"] = canonical_label(view.taxonomy, result.dimension.value, result.alternative_label)
                result = result.model_copy(update=update)
        return result

    def stage2_analyst(self, team_id: str, view: EvidenceView) -> Stage2AnalysisReport:
        schema: type[Stage2AnalysisReport] = (
            IsstaStage2AnalysisReport
            if view.domain_profile == "issta2024"
            else Stage2AnalysisReport
        )
        return self._role(
            AnalystRole.STAGE2_FAULT_VERIFIER,
            team_id=team_id,
            view=view,
            schema=schema,
        )

    def _sla_role(
        self,
        role: AnalystRole,
        *,
        view: EvidenceView,
        schema: type[SchemaT],
        context: dict[str, object],
        instructions: str,
        result_validator: Callable[[SchemaT], None],
    ) -> SchemaT:
        """Invoke one isolated, non-thinking fast-path role with one repair."""

        try:
            max_tokens = _SLA_ROLE_MAX_TOKENS[role]
        except KeyError as error:
            raise ValueError(f"unsupported SLA role {role.value!r}") from error
        return self._client.complete(
            system_prompt=(
                f"ROLE: {role.value}\n"
                + instructions
                + "OUTPUT CONTRACT:\n"
                + json.dumps(_compact_schema(schema), ensure_ascii=False)
                + "\nReturn one JSON object matching this contract exactly."
            ),
            user_prompt=self._payload(
                team_id="targeted_sla",
                view=_sla_prompt_view(view),
                context=context,
            ),
            schema=schema,
            options=ModelCallOptions(
                role=role.value,
                team_id=None,
                perspective=None,
                max_tokens=max_tokens,
                thinking_enabled=False,
            ),
            result_validator=result_validator,
            max_schema_retries=_SLA_SCHEMA_REPAIRS,
        )

    @staticmethod
    def _sla_baseline_labels(anchor: BaselineAnchor) -> dict[str, str]:
        if (
            not anchor.valid
            or anchor.symptom_label is None
            or anchor.root_cause_label is None
        ):
            raise ValueError("SLA roles require a valid Baseline anchor")
        return {
            "symptom_label": anchor.symptom_label,
            "root_cause_label": anchor.root_cause_label,
        }

    @staticmethod
    def _sla_diagnosis_taxonomy_labels(view: EvidenceView) -> dict[str, list[str]]:
        """Project only the two exact label sets needed by SLA diagnosis."""

        return {
            "symptom": list(view.taxonomy.get("symptom", ())),
            "root_cause": list(view.taxonomy.get("root_cause", ())),
        }

    def sla_joint_diagnosis(
        self, view: EvidenceView, anchor: BaselineAnchor
    ) -> SlaJointDiagnosis:
        """Produce the one bounded, evidence-validated SLA diagnosis."""

        def validate_diagnosis(report: SlaJointDiagnosis) -> None:
            _validate_classification_labels(
                report.symptom, view, EvidenceDimension.SYMPTOM
            )
            _validate_classification_labels(
                report.root_cause, view, EvidenceDimension.ROOT_CAUSE
            )
            for part in (report.symptom, report.root_cause):
                _validate_citation_fields(
                    part,
                    view,
                    "supporting_evidence_ids",
                    "counter_evidence_ids",
                    "boundary_evidence_ids",
                )
            validate_sla_joint_diagnosis(report, view, anchor)

        return self._sla_role(
            AnalystRole.SLA_JOINT_DIAGNOSIS,
            view=view,
            schema=SlaJointDiagnosis,
            context={
                "baseline_labels": self._sla_baseline_labels(anchor),
                "taxonomy_labels": self._sla_diagnosis_taxonomy_labels(view),
                "evidence_eligibility": _joint_anchor_evidence_eligibility(view),
            },
            instructions=SLA_JOINT_DIAGNOSIS_SYSTEM_INSTRUCTIONS,
            result_validator=validate_diagnosis,
        )

    def sla_joint_verifier(
        self,
        view: EvidenceView,
        anchor: BaselineAnchor,
        diagnosis: SlaJointDiagnosis,
    ) -> SlaJointVerification:
        """Verify only canonical labels, chains, and citations from diagnosis."""

        validate_sla_joint_diagnosis(diagnosis, view, anchor)

        def validate_verification(report: SlaJointVerification) -> None:
            for part in (report.symptom, report.root_cause):
                _validate_citation_fields(
                    part,
                    view,
                    "supporting_evidence_ids",
                    "counter_evidence_ids",
                )
            validate_sla_joint_verification(report, view)

        candidate_projection = {
            "symptom": {
                "label": diagnosis.symptom.label,
                "supporting_evidence_ids": list(
                    diagnosis.symptom.supporting_evidence_ids
                ),
                "counter_evidence_ids": list(diagnosis.symptom.counter_evidence_ids),
                "boundary_evidence_ids": list(diagnosis.symptom.boundary_evidence_ids),
            },
            "root_cause": {
                "label": diagnosis.root_cause.label,
                "causal_chain": list(diagnosis.root_cause.causal_chain),
                "supporting_evidence_ids": list(
                    diagnosis.root_cause.supporting_evidence_ids
                ),
                "counter_evidence_ids": list(diagnosis.root_cause.counter_evidence_ids),
                "boundary_evidence_ids": list(
                    diagnosis.root_cause.boundary_evidence_ids
                ),
            },
        }
        return self._sla_role(
            AnalystRole.SLA_JOINT_VERIFIER,
            view=view,
            schema=SlaJointVerification,
            context={
                "baseline_labels": self._sla_baseline_labels(anchor),
                "candidate_projection": candidate_projection,
            },
            instructions=SLA_JOINT_VERIFIER_SYSTEM_INSTRUCTIONS,
            result_validator=validate_verification,
        )

    def sla_conditional_arbitrator(
        self,
        view: EvidenceView,
        anchor: BaselineAnchor,
        diagnosis: SlaJointDiagnosis,
        verification: SlaJointVerification,
    ) -> SlaConditionalArbitration:
        """Select from the controller's fixed candidate-versus-Baseline pairs."""

        validate_sla_joint_diagnosis(diagnosis, view, anchor)
        validate_sla_joint_verification(verification, view)
        baseline_labels = self._sla_baseline_labels(anchor)
        pairs: dict[str, dict[str, str]] = {}
        for dimension, candidate, baseline, verdict in (
            (
                "symptom",
                diagnosis.symptom.label,
                baseline_labels["symptom_label"],
                verification.symptom.verdict,
            ),
            (
                "root_cause",
                diagnosis.root_cause.label,
                baseline_labels["root_cause_label"],
                verification.root_cause.verdict,
            ),
        ):
            if (
                verdict is SlaVerifierVerdict.PRESERVE_BASELINE
                and candidate != baseline
            ):
                pairs[dimension] = {
                    "candidate_label": candidate,
                    "baseline_label": baseline,
                }

        def validate_arbitration(report: SlaConditionalArbitration) -> None:
            validate_sla_conditional_arbitration(
                report,
                view=view,
                diagnosis=diagnosis,
                verification=verification,
                baseline=anchor,
            )

        return self._sla_role(
            AnalystRole.SLA_CONDITIONAL_ARBITRATOR,
            view=view,
            schema=SlaConditionalArbitration,
            context={
                "conflicting_dimensions": list(pairs),
                "conflicting_candidate_baseline_pairs": pairs,
            },
            instructions=SLA_CONDITIONAL_ARBITRATOR_SYSTEM_INSTRUCTIONS,
            result_validator=validate_arbitration,
        )

    def fault_evidence_analyst(
        self,
        view: EvidenceView,
    ) -> FaultEvidenceAssessment:
        return self._stage2_owned_assessment(
            AnalystRole.FAULT_EVIDENCE_ANALYST,
            dimension=EvidenceDimension.FAULT_EXISTENCE,
            team_id="fault_evidence",
            view=view,
            schema=FaultEvidenceAssessment,
        )

    def scope_boundary_analyst(
        self,
        view: EvidenceView,
    ) -> ScopeBoundaryAssessment:
        return self._stage2_owned_assessment(
            AnalystRole.SCOPE_BOUNDARY_ANALYST,
            dimension=EvidenceDimension.STUDY_SCOPE,
            team_id="scope_boundary",
            view=view,
            schema=ScopeBoundaryAssessment,
        )

    def repair_causality_analyst(
        self,
        view: EvidenceView,
    ) -> RepairCausalityAssessment:
        return self._stage2_owned_assessment(
            AnalystRole.REPAIR_CAUSALITY_ANALYST,
            dimension=EvidenceDimension.REPAIR_CAUSALITY,
            team_id="repair_causality",
            view=view,
            schema=RepairCausalityAssessment,
        )

    def _stage2_owned_assessment(self, role, *, dimension, team_id, view, schema):
        allowed_ids = [item.evidence_id for item in view.items
                       if supports_readiness_dimension(item, dimension, domain=view.domain_profile)]
        allowed = frozenset(allowed_ids)
        def validate_assessment(report):
            cited = set(report.supporting_evidence_ids) | set(report.counter_evidence_ids)
            if not cited:
                raise ValueError(f"{dimension.value} assessment cites no evidence")
            if invalid := cited - allowed:
                raise ValueError(f"{dimension.value} assessment cites capability-incompatible or unknown "
                                 f"evidence IDs {sorted(invalid)!r}; use only allowed_citation_ids {allowed_ids!r}")
        return self._role(role, team_id=team_id, view=view, schema=schema,
                          context={"owned_dimension": dimension.value, "allowed_citation_ids": allowed_ids},
                          result_validator=validate_assessment)

    def evidence_readiness(
        self, task: str, view: EvidenceView
    ) -> EvidenceReadinessReport:
        required_dimensions = required_readiness_dimensions(
            view.domain_profile,
            task,
        )

        def validate_readiness_result(report: EvidenceReadinessReport) -> None:
            reported_dimensions = tuple(
                readiness.dimension for readiness in report.dimensions
            )
            if len(reported_dimensions) != len(required_dimensions) or set(
                reported_dimensions
            ) != set(required_dimensions):
                raise ValueError(
                    "readiness dimensions must exactly match required dimensions"
                )
            if report.task != task:
                raise ValueError("readiness report task must match requested task")
            validity = assess_stage3_evidence_validity(view)
            valid_ids = set(validity.valid_evidence_ids)
            items_by_id = {item.evidence_id: item for item in view.items}
            for readiness in report.dimensions:
                for evidence_id in readiness.confirmed_evidence_ids:
                    item = items_by_id.get(evidence_id)
                    if (
                        item is None
                        or evidence_id not in valid_ids
                        or not supports_readiness_dimension(
                            item,
                            readiness.dimension,
                            domain=view.domain_profile,
                        )
                    ):
                        raise ValueError(
                            f"required readiness dimension "
                            f"{readiness.dimension.value} cites capability-"
                            f"incompatible confirmed evidence_id {evidence_id!r}"
                        )
                for request in readiness.evidence_requests:
                    validate_frozen_evidence_target(
                        request,
                        domain=view.domain_profile,
                    )
                    unavailable = frozenset(
                        availability_metadata.get("unavailable_frozen_source_types", ())
                    )
                    specialist_sources = _SPECIALIST_FROZEN_SOURCE_TYPES[
                        request.target_specialist
                    ]
                    if specialist_sources and specialist_sources <= unavailable:
                        raise ValueError(
                            f"request {request.request_id!r} targets only "
                            "uncaptured frozen sources"
                        )

        availability_metadata = next(
            (
                item.metadata
                for item in view.items
                if "captured_frozen_source_types" in item.metadata
            ),
            {},
        )
        validity = assess_stage3_evidence_validity(view)
        valid_ids = set(validity.valid_evidence_ids)
        confirmed_evidence_eligibility = {
            item.evidence_id: [
                dimension.value
                for dimension in required_dimensions
                if item.evidence_id in valid_ids
                and supports_readiness_dimension(
                    item,
                    dimension,
                    domain=view.domain_profile,
                )
            ]
            for item in view.items
        }
        return self._role(
            AnalystRole.EVIDENCE_READINESS,
            team_id="evidence_readiness",
            view=view.model_copy(update={"task": task}),
            schema=EvidenceReadinessReport,
            context={
                "task": task,
                "required_dimensions": [
                    dimension.value for dimension in required_dimensions
                ],
                "confirmed_evidence_eligibility": confirmed_evidence_eligibility,
                "allowed_evidence_targets": allowed_frozen_evidence_targets(
                    view.domain_profile
                ),
                "captured_frozen_source_types": list(
                    availability_metadata.get("captured_frozen_source_types", ())
                ),
                "unavailable_frozen_source_types": list(
                    availability_metadata.get("unavailable_frozen_source_types", ())
                ),
            },
            result_validator=validate_readiness_result,
        )

    def symptom_analyst(self, team_id: str, view: EvidenceView) -> SymptomReport:
        def validate_symptom(report: SymptomReport) -> None:
            allowed = set(view.taxonomy.get("symptom", ()))
            if not label_valid(view.taxonomy, "symptom", report.label) or not label_valid(view.taxonomy, "symptom", report.alternative_label):
                raise ValueError(
                    "symptom label and alternative_label must belong to the "
                    "supplied symptom taxonomy"
                )
            if len(allowed) > 1 and labels_equal(view.taxonomy, "symptom", report.label, report.alternative_label):
                raise ValueError("symptom alternative_label must be distinct")
            _validate_citation_fields(
                report,
                view,
                "supporting_evidence_ids",
                "counter_evidence_ids",
                "boundary_evidence_ids",
            )

        return self._role(
            AnalystRole.SYMPTOM_ANALYST,
            team_id=team_id,
            view=view,
            schema=SymptomReport,
            result_validator=validate_symptom,
        )

    def joint_anchor(self, team_id: str, view: EvidenceView) -> JointAnchorReport:
        return self._role(
            AnalystRole.JOINT_ANCHOR,
            team_id=team_id,
            view=view,
            schema=JointAnchorReport,
            context={
                "evidence_eligibility": _joint_anchor_evidence_eligibility(view),
            },
            result_validator=lambda report: _validate_joint_anchor(report, view),
        )

    def symptom_verifier(
        self,
        team_id: str,
        anchor: JointAnchorReport,
        view: EvidenceView,
    ) -> DimensionVerificationReport:
        return self._dimension_verifier(
            AnalystRole.SYMPTOM_VERIFIER,
            EvidenceDimension.SYMPTOM,
            team_id,
            anchor,
            view,
        )

    def root_cause_verifier(
        self,
        team_id: str,
        anchor: JointAnchorReport,
        view: EvidenceView,
    ) -> DimensionVerificationReport:
        return self._dimension_verifier(
            AnalystRole.ROOT_CAUSE_VERIFIER,
            EvidenceDimension.ROOT_CAUSE,
            team_id,
            anchor,
            view,
        )

    def _dimension_verifier(
        self,
        role: AnalystRole,
        dimension: EvidenceDimension,
        team_id: str,
        anchor: JointAnchorReport,
        view: EvidenceView,
    ) -> DimensionVerificationReport:
        _validate_joint_anchor(anchor, view)
        if self._chain_checker_enabled:
            part = getattr(anchor, dimension.value)
            schema = checker_output_schema(view, dimension.value)
            def retain_attempt(raw: str, error: Exception | None) -> None:
                with self._module_a_lock:
                    self._module_b_attempts.append({"record_id": view.record_id, "team_id": team_id,
                        "dimension": dimension.value, "ledger_version": view.ledger_version,
                        "anchor": anchor.model_dump(mode="json"),
                        "status": "invalid" if error else "valid", "raw_response": raw,
                        "error_type": type(error).__name__ if error else None,
                        "error": str(error) if error else None})
            audit = self._client.complete(
                system_prompt=CHECKER_PROMPT + "\n" + taxonomy_guidance(view.domain_profile, view.taxonomy)
                              + "\n" + json.dumps(_compact_schema(schema), ensure_ascii=False),
                user_prompt=self._payload(team_id=team_id, view=view, context={
                    "owned_dimension": dimension.value,
                    "targets": check_targets(part, dimension.value, anchor.causal_account),
                    "required_target_ids": list(check_targets(part, dimension.value, anchor.causal_account)),
                    "anchor_label": part.label,
                    "specific_candidate_labels": [label for label in view.taxonomy.get(dimension.value, ()) if label != "Unknown"],
                }),
                schema=schema,
                options=replace(self._model_call_options(role.value, team_id=team_id, perspective=None),
                                role=f"{dimension.value}_chain_checker", max_tokens=8192),
                result_validator=lambda value: validate_chain_audit(value, anchor, dimension.value, view),
                validation_observer=retain_attempt,
            )
            with self._module_a_lock:
                self._module_b_calls.append({"record_id": view.record_id, "team_id": team_id,
                                             "dimension": dimension.value, "ledger_version": view.ledger_version,
                                             "targets": check_targets(part, dimension.value, anchor.causal_account),
                                             "audit": audit.model_dump(mode="json")})
            return verification_from_audit(audit, anchor, dimension.value)
        anchor_label = (
            anchor.symptom.label
            if dimension is EvidenceDimension.SYMPTOM
            else anchor.root_cause.label
        )
        allowed = set(view.taxonomy.get(dimension.value, ()))

        def validate_verification(report: DimensionVerificationReport) -> None:
            if report.dimension is not dimension:
                raise ValueError(f"verification dimension must be {dimension.value!r}")
            if not labels_equal(view.taxonomy, dimension.value, report.anchor_label, anchor_label):
                raise ValueError(
                    "verification anchor_label must match the current anchor label"
                )
            if report.alternative_label is not None and (
                not label_valid(view.taxonomy, dimension.value, report.alternative_label)
            ):
                raise ValueError(
                    f"verification alternative_label must belong to the supplied "
                    f"{dimension.value} taxonomy"
                )
            _validate_citation_fields(
                report,
                view,
                "supporting_evidence_ids",
                "counter_evidence_ids",
            )
            evidence_by_id = {item.evidence_id: item for item in view.items}
            incapable_ids = tuple(
                evidence_id
                for evidence_id in report.supporting_evidence_ids
                if not supports_readiness_dimension(
                    evidence_by_id[evidence_id],
                    dimension,
                    domain=view.domain_profile,
                )
            )
            if incapable_ids:
                raise ValueError(
                    "verification supporting_evidence_ids does not support the "
                    f"{dimension.value} dimension: {list(incapable_ids)!r}"
                )

        return self._role(
            role,
            team_id=team_id,
            view=view,
            schema=DimensionVerificationReport,
            context={
                "anchor": anchor.model_dump(mode="json"),
                "owned_dimension": dimension.value,
            },
            result_validator=validate_verification,
        )

    def root_cause_analyst(self, team_id: str, view: EvidenceView) -> RootCauseReport:
        def validate_root_cause(report: RootCauseReport) -> None:
            allowed = set(view.taxonomy.get("root_cause", ()))
            if not label_valid(view.taxonomy, "root_cause", report.label) or not label_valid(view.taxonomy, "root_cause", report.alternative_label):
                raise ValueError(
                    "root-cause label and alternative_label must belong to the "
                    "supplied root_cause taxonomy"
                )
            if len(allowed) > 1 and report.label == report.alternative_label:
                raise ValueError("root-cause alternative_label must be distinct")
            _validate_citation_fields(
                report,
                view,
                "supporting_evidence_ids",
                "counter_evidence_ids",
                "boundary_evidence_ids",
            )

        return self._role(
            AnalystRole.ROOT_CAUSE_ANALYST,
            team_id=team_id,
            view=view,
            schema=RootCauseReport,
            result_validator=validate_root_cause,
        )

    def consistency_checker(
        self,
        team_id: str,
        symptom: SymptomReport,
        root_cause: RootCauseReport,
        view: EvidenceView,
    ) -> CausalConsistencyReport:
        def validate_consistency(report: CausalConsistencyReport) -> None:
            _validate_citation_fields(
                report,
                view,
                "supporting_evidence_ids",
            )

        return self._role(
            AnalystRole.CAUSAL_CONSISTENCY_CHECKER,
            team_id=team_id,
            view=view,
            schema=CausalConsistencyReport,
            context={
                "symptom_report": symptom.model_dump(mode="json"),
                "root_cause_report": root_cause.model_dump(mode="json"),
            },
            result_validator=validate_consistency,
        )

    def baseline_revision_assessment(
        self,
        team_id: str,
        proposal: RevisionProposalEnvelope,
        anchor: BaselineAnchor,
        routed_card: BoundaryCard,
        view: EvidenceView,
    ) -> BaselineRevisionAssessment:
        """Ask one isolated team to assess one exact Baseline challenger."""

        if team_id not in {"A", "B"}:
            raise ValueError("revision assessor team must be A or B")
        if not anchor.valid:
            raise ValueError("cannot assess an unavailable Baseline")
        if anchor.record_id != view.record_id:
            raise ValueError("Baseline anchor record_id must match evidence view")
        dimension = proposal.dimension
        baseline_label = (
            anchor.symptom_label
            if dimension is EvidenceDimension.SYMPTOM
            else anchor.root_cause_label
        )
        assert baseline_label is not None
        if (
            proposal.baseline_label != baseline_label
            or proposal.baseline_source_config_hash != anchor.source_config_hash
            or proposal.baseline_source_predictions_sha256
            != anchor.source_predictions_sha256
        ):
            raise ValueError("proposal must bind the frozen Baseline anchor")
        if proposal.evidence_view_hash != canonical_evidence_view_hash(view):
            raise ValueError("proposal must bind the exact evidence view")
        owned_labels = set(view.taxonomy.get(dimension.value, ()))
        if {proposal.baseline_label, proposal.proposed_label}.difference(owned_labels):
            raise ValueError("proposal pair must belong to the owned taxonomy")
        if (
            routed_card.card_id != proposal.boundary_card_id
            or routed_card.dimension != dimension.value
            or set(routed_card.labels)
            != {proposal.baseline_label, proposal.proposed_label}
        ):
            raise ValueError("routed Boundary Card must match the exact proposal pair")
        validity = assess_stage3_evidence_validity(view)
        valid_ids = set(validity.valid_evidence_ids)
        items_by_id = {item.evidence_id: item for item in view.items}
        proposal_citations = {
            *proposal.supporting_evidence_ids,
            *proposal.counter_evidence_ids,
        }
        if not proposal_citations.issubset(valid_ids):
            raise ValueError("proposal citations must be valid exact-view evidence")
        if any(
            not supports_readiness_dimension(
                items_by_id[evidence_id], dimension, domain=view.domain_profile
            )
            for evidence_id in proposal_citations
        ):
            raise ValueError("proposal citations must match dimension capability")
        system_instructions = (
            REVISION_ENTAILMENT_SYSTEM_INSTRUCTIONS
            if team_id == "A"
            else REVISION_FALSIFICATION_SYSTEM_INSTRUCTIONS
        )
        raw_schema = (
            RevisionEntailmentResult if team_id == "A" else RevisionFalsificationResult
        )
        criteria = {criterion.label: criterion for criterion in routed_card.criteria}
        baseline_criterion = criteria[proposal.baseline_label]
        proposed_criterion = criteria[proposal.proposed_label]
        if team_id == "A":
            required_findings = {
                "proposed_positive_conditions": list(
                    proposed_criterion.positive_conditions
                ),
                "baseline_exclusion_condition": (
                    baseline_criterion.exclusion_conditions[0]
                ),
                "allowed_positive_citation_ids": sorted(
                    proposal.supporting_evidence_ids
                ),
                "allowed_exclusion_citation_ids": sorted(proposal_citations),
            }
        else:
            required_findings = {
                "baseline_survival_condition": (
                    baseline_criterion.positive_conditions[0]
                ),
                "proposed_defeater_conditions": list(
                    proposed_criterion.exclusion_conditions
                ),
                "strongest_competing_reading_label": proposal.baseline_label,
                "allowed_citation_ids": sorted(proposal_citations),
            }

        def validate_raw_result(
            result: RevisionEntailmentResult | RevisionFalsificationResult,
        ) -> None:
            if isinstance(result, RevisionEntailmentResult):
                normalize_revision_entailment_result(
                    result=result, proposal=proposal, routed_card=routed_card
                )
            else:
                normalize_revision_falsification_result(
                    result=result, proposal=proposal, routed_card=routed_card
                )

        raw_result = self._client.complete(
            system_prompt=(
                system_instructions
                + "\nOUTPUT CONTRACT:\n"
                + json.dumps(_compact_schema(raw_schema), ensure_ascii=False)
                + "\nReturn one JSON object matching this contract exactly."
            ),
            user_prompt=self._payload(
                team_id=team_id,
                view=_candidate_prompt_view(view, proposal),
                context={
                    "proposal": {
                        "dimension": proposal.dimension.value,
                        "baseline_label": proposal.baseline_label,
                        "proposed_label": proposal.proposed_label,
                        "boundary_card_id": proposal.boundary_card_id,
                        "supporting_evidence_ids": list(
                            proposal.supporting_evidence_ids
                        ),
                        "counter_evidence_ids": list(proposal.counter_evidence_ids),
                    },
                    "boundary_card": routed_card.model_dump(mode="json"),
                    "required_findings": required_findings,
                },
            ),
            schema=raw_schema,
            options=self._model_call_options(
                "baseline_revision_assessment",
                team_id=team_id,
                perspective=("entailment" if team_id == "A" else "falsification"),
            ),
            result_validator=validate_raw_result,
        )
        if isinstance(raw_result, RevisionEntailmentResult):
            return normalize_revision_entailment_result(
                result=raw_result, proposal=proposal, routed_card=routed_card
            )
        return normalize_revision_falsification_result(
            result=raw_result, proposal=proposal, routed_card=routed_card
        )

    def baseline_revision_consistency(
        self,
        checker_team_id: str,
        assessment_owner_team_id: str,
        proposal: RevisionProposalEnvelope,
        assessment: BaselineRevisionAssessment,
        routed_card: BoundaryCard,
        view: EvidenceView,
    ) -> RevisionConsistencyReport:
        """Have the opposite team cross-check one candidate-bound assessment."""

        if checker_team_id not in {"A", "B"} or assessment_owner_team_id not in {
            "A",
            "B",
        }:
            raise ValueError("cross-check team identities must be A or B")
        if checker_team_id == assessment_owner_team_id:
            raise ValueError("cross-check must be performed by the opposite team")
        if assessment.verdict is not RevisionAssessmentVerdict.REVISE:
            raise ValueError("only revise assessments require revision consistency")
        proposal_digest = revision_proposal_digest(proposal)
        if (
            assessment.assessor_team_id,
            assessment.proposal_digest,
        ) != (assessment_owner_team_id, proposal_digest):
            raise ValueError("cross-check assessment owner or proposal mismatch")
        dimension = proposal.dimension
        if (
            assessment.dimension,
            assessment.baseline_label,
            assessment.proposed_label,
            assessment.boundary_card_id,
        ) != (
            dimension,
            proposal.baseline_label,
            proposal.proposed_label,
            proposal.boundary_card_id,
        ):
            raise ValueError("cross-check assessment must bind the exact proposal")
        if (
            routed_card.card_id != proposal.boundary_card_id
            or routed_card.dimension != dimension.value
            or set(routed_card.labels)
            != {proposal.baseline_label, proposal.proposed_label}
        ):
            raise ValueError("cross-check Boundary Card must match the exact proposal")
        criteria = {criterion.label: criterion for criterion in routed_card.criteria}
        if assessment.contradicted_baseline_condition not in criteria[
            proposal.baseline_label
        ].exclusion_conditions or not set(
            assessment.satisfied_proposed_conditions
        ).issubset(
            criteria[proposal.proposed_label].positive_conditions
        ):
            raise ValueError(
                "cross-check assessment conditions must belong to the routed card"
            )
        owned_candidate_ids = {
            *proposal.supporting_evidence_ids,
            *proposal.counter_evidence_ids,
        }
        if not {
            *assessment.supporting_evidence_ids,
            *assessment.counter_evidence_ids,
        }.issubset(owned_candidate_ids):
            raise ValueError(
                "cross-check assessment citations must belong to the proposal"
            )
        valid = set(assess_stage3_evidence_validity(view).valid_evidence_ids)
        by_id = {item.evidence_id: item for item in view.items}
        if any(
            evidence_id not in valid
            or not supports_readiness_dimension(
                by_id[evidence_id], dimension, domain=view.domain_profile
            )
            for evidence_id in owned_candidate_ids
        ):
            raise ValueError(
                "cross-check proposal citations must be valid and owned-capable"
            )
        expected = (
            proposal.baseline_source_config_hash,
            proposal.baseline_source_predictions_sha256,
            proposal.taxonomy_structure_hash,
            proposal.evidence_view_hash,
        )
        role = (
            AnalystRole.SYMPTOM_ANALYST
            if dimension is EvidenceDimension.SYMPTOM
            else AnalystRole.ROOT_CAUSE_ANALYST
        )
        perspective = render_stage3_perspective_instructions(
            stage3_team_policy(checker_team_id), role
        )
        if assessment.raw_falsification_result is not None:
            if checker_team_id != "A" or assessment_owner_team_id != "B":
                raise ValueError(
                    "falsification coverage must be checked by A against B"
                )
            cross_schema = FalsificationCoverageCrossResult
            owner_raw = assessment.raw_falsification_result
            cross_task_instruction = (
                "AUDIT TARGET: falsification coverage. Verify the owner checked Baseline "
                "survival, every fixed proposed defeater, and the cited strongest "
                "competitor.\n"
            )
        elif assessment.raw_entailment_result is not None:
            if checker_team_id != "B" or assessment_owner_team_id != "A":
                raise ValueError("entailment mapping must be checked by B against A")
            cross_schema = EntailmentMappingCrossResult
            owner_raw = assessment.raw_entailment_result
            cross_task_instruction = (
                "AUDIT TARGET: entailment mapping. Verify every proposed condition "
                "and the exact Baseline exclusion are mapped to owned evidence.\n"
            )
        else:
            raise ValueError(
                "active cross-check requires a task-owned heterogeneous raw assessment"
            )
        if (
            assessment.baseline_source_config_hash,
            assessment.baseline_source_predictions_sha256,
            assessment.taxonomy_structure_hash,
            assessment.evidence_view_hash,
        ) != expected or proposal.evidence_view_hash != canonical_evidence_view_hash(
            view
        ):
            raise ValueError("assessment provenance does not match revision inputs")

        def validate(
            report: EntailmentMappingCrossResult | FalsificationCoverageCrossResult,
        ) -> None:
            if isinstance(report, EntailmentMappingCrossResult):
                if not isinstance(owner_raw, RevisionEntailmentResult):
                    raise ValueError("cross raw task does not match entailment owner")
                if tuple(
                    (finding.condition, finding.citation_ids)
                    for finding in report.condition_findings
                ) != tuple(
                    (finding.condition, finding.citation_ids)
                    for finding in owner_raw.condition_findings
                ) or (
                    report.baseline_exclusion_finding.condition,
                    report.baseline_exclusion_finding.citation_ids,
                ) != (
                    owner_raw.baseline_exclusion_finding.condition,
                    owner_raw.baseline_exclusion_finding.citation_ids,
                ):
                    raise ValueError(
                        "entailment cross must cover every exact owner finding and citation"
                    )
                supporting_evidence_ids = tuple(
                    dict.fromkeys(
                        citation
                        for finding in report.condition_findings
                        for citation in finding.citation_ids
                    )
                )
                counter_evidence_ids = report.baseline_exclusion_finding.citation_ids
            else:
                if not isinstance(owner_raw, RevisionFalsificationResult):
                    raise ValueError(
                        "cross raw task does not match falsification owner"
                    )
                if (
                    (
                        report.baseline_survival_finding.condition,
                        report.baseline_survival_finding.citation_ids,
                    )
                    != (
                        owner_raw.baseline_survival_finding.condition,
                        owner_raw.baseline_survival_finding.citation_ids,
                    )
                    or tuple(
                        (finding.condition, finding.citation_ids)
                        for finding in report.proposed_defeater_findings
                    )
                    != tuple(
                        (finding.condition, finding.citation_ids)
                        for finding in owner_raw.proposed_defeater_findings
                    )
                    or (
                        report.strongest_competing_reading_finding.label,
                        report.strongest_competing_reading_finding.citation_ids,
                    )
                    != (
                        owner_raw.strongest_competing_reading.label,
                        owner_raw.strongest_competing_reading.citation_ids,
                    )
                ):
                    raise ValueError(
                        "falsification cross must cover every exact owner finding and citation"
                    )
                supporting_evidence_ids = tuple(
                    dict.fromkeys(
                        (
                            *report.strongest_competing_reading_finding.citation_ids,
                            *(
                                citation
                                for finding in report.proposed_defeater_findings
                                for citation in finding.citation_ids
                            ),
                        )
                    )
                )
                counter_evidence_ids = report.baseline_survival_finding.citation_ids
            if (
                supporting_evidence_ids != assessment.supporting_evidence_ids
                or counter_evidence_ids != assessment.counter_evidence_ids
            ):
                raise ValueError("revision consistency citations must match assessment")
            if any(
                evidence_id not in valid
                or not supports_readiness_dimension(
                    by_id[evidence_id], dimension, domain=view.domain_profile
                )
                for evidence_id in {
                    *supporting_evidence_ids,
                    *counter_evidence_ids,
                }
            ):
                raise ValueError(
                    "revision consistency citations must be valid and owned-capable"
                )

        raw_cross = self._client.complete(
            system_prompt=BASELINE_REVISION_CONSISTENCY_SYSTEM_INSTRUCTIONS
            + perspective
            + cross_task_instruction
            + "OUTPUT CONTRACT:\n"
            + json.dumps(_compact_schema(cross_schema), ensure_ascii=False),
            user_prompt=self._payload(
                team_id=checker_team_id,
                view=_candidate_prompt_view(view, proposal),
                context={
                    "cross_check_input": {
                        "assessment_owner_task": assessment.assessor_task.value,
                        "raw_assessment": (
                            assessment.raw_entailment_result.model_dump(mode="json")
                            if assessment.raw_entailment_result is not None
                            else (
                                assessment.raw_falsification_result.model_dump(
                                    mode="json"
                                )
                                if assessment.raw_falsification_result is not None
                                else None
                            )
                        ),
                        "dimension": proposal.dimension.value,
                        "baseline_label": proposal.baseline_label,
                        "proposed_label": proposal.proposed_label,
                        "boundary_card_id": proposal.boundary_card_id,
                        "supporting_evidence_ids": list(
                            assessment.supporting_evidence_ids
                        ),
                        "counter_evidence_ids": list(assessment.counter_evidence_ids),
                        "normalized_conditions": list(
                            assessment.satisfied_proposed_conditions
                        ),
                    },
                    "boundary_card": routed_card.model_dump(mode="json"),
                },
            ),
            schema=cross_schema,
            options=self._model_call_options(
                "baseline_revision_consistency",
                team_id=checker_team_id,
                perspective=stage3_team_policy(checker_team_id).perspective.value,
            ),
            result_validator=validate,
        )
        return RevisionConsistencyReport(
            checker_team_id=checker_team_id,
            assessment_owner_team_id=assessment_owner_team_id,
            proposal_digest=proposal_digest,
            dimension=dimension,
            baseline_label=proposal.baseline_label,
            proposed_label=proposal.proposed_label,
            status=raw_cross.status,
            rationale=raw_cross.rationale,
            supporting_evidence_ids=assessment.supporting_evidence_ids,
            counter_evidence_ids=assessment.counter_evidence_ids,
            assessment_digest=baseline_revision_assessment_digest(assessment),
            assessment_owner_task=assessment.assessor_task,
            assessment_raw_result_digest=assessment.raw_result_digest,
            boundary_card_id=proposal.boundary_card_id,
            baseline_source_config_hash=proposal.baseline_source_config_hash,
            baseline_source_predictions_sha256=(
                proposal.baseline_source_predictions_sha256
            ),
            taxonomy_structure_hash=proposal.taxonomy_structure_hash,
            evidence_view_hash=proposal.evidence_view_hash,
        )

    def supervise_framework(self, reports, view):
        from .global_supervisor import (
            run_supervision, SupervisorDecision, PROMPT, TEMPERATURE,
            validate_decision, pending_consistency,
        )
        from .stage3_composition import compose_stage3_team_candidate

        def review(current_view, state, current_reports, actions):
            from .experiment_two import RULES, supervisor_schema, supervisor_targets, validate_supervisor_claims
            schema = supervisor_schema() if self._rule_checks else SupervisorDecision
            if self._rule_checks:
                state = {**state, 'claim_targets': supervisor_targets(current_reports)}
            def validate(value):
                validate_decision(value, current_view, current_reports, actions)
                if self._rule_checks:
                    validate_supervisor_claims(value, current_view, current_reports)
                if any(not state["remaining_team_reworks"][t] for t in value.targets):
                    raise ValueError("Requested team has exhausted its rework budget")
            return self._client.complete(
                system_prompt=PROMPT + "\n" + taxonomy_guidance(current_view.domain_profile, current_view.taxonomy)
                    + ("\n" + RULES + "\nReturn claim_findings for EVERY context.claim_targets key exactly once. "
                       "Audit each mechanism, repair claim and dependency against original quotations. "
                       "Use entailed/contradicted/unknown, applicable rule, rationale and missing_evidence. "
                       "Source quote existence does not prove the inferred mechanism. Rework unsupported "
                       "explanations, even with unchanged labels; otherwise record evidence_gap."
                       if self._rule_checks else "")
                    + "\nOUTPUT CONTRACT:\n" + json.dumps(_compact_schema(schema), ensure_ascii=False),
                user_prompt=self._payload(team_id="global_supervisor", view=current_view, context=state),
                schema=schema,
                options=ModelCallOptions(role="global_supervisor", team_id="controller", perspective=None,
                    max_tokens=12288 if self._experiment_two_arm else 6000, thinking_enabled=False if self._provider_supports_thinking else None, temperature=TEMPERATURE),
                result_validator=validate,
            )

        def rebuild(team, current_view, feedback):
            token = self._supervisor_feedback.set(feedback)
            try:
                anchor = self.joint_anchor(team, current_view)
                symptom = self.symptom_verifier(team, anchor, current_view)
                cause = self.root_cause_verifier(team, anchor, current_view)
                candidate = compose_stage3_team_candidate(team, anchor, symptom, cause, current_view)
                return Stage3TeamReport(team_id=team, symptom=candidate.symptom,
                    root_cause=candidate.root_cause, anchor=candidate.anchor,
                    verifications=candidate.verifications, correction_audit=candidate.correction_audit,
                    consistency=pending_consistency(team, candidate.symptom, candidate.root_cause, current_view))
            finally:
                self._supervisor_feedback.reset(token)

        def challenge(current_reports, current_view, feedback):
            token = self._supervisor_feedback.set(feedback)
            try:
                return self.boundary_challenger(current_reports, current_view)
            finally:
                self._supervisor_feedback.reset(token)

        return run_supervision(view, reports, review=review, rebuild_team=rebuild, challenge=challenge)

    def boundary_challenger(
        self,
        reports: tuple[Stage3TeamReport, Stage3TeamReport],
        view: EvidenceView,
    ) -> BoundaryChallenge:
        from .stage3_composition import boundary_challenge_semantic_errors

        def validate_challenge(result):
            errors = boundary_challenge_semantic_errors(result, reports, view)
            if errors:
                raise ValueError("; ".join(errors))

        anonymized = [
            AnonymousStage3TeamReport(
                symptom=report.symptom,
                root_cause=report.root_cause,
                consistency=report.consistency,
            ).model_dump(mode="json")
            for report in reports
        ]
        return self._role(
            AnalystRole.BOUNDARY_CHALLENGER,
            team_id="boundary_challenger",
            view=view,
            schema=BoundaryChallenge,
            context={"proposed_anonymized_reports": anonymized},
            result_validator=validate_challenge,
        )

    def stage2_arbitrator(
        self, packet: Stage2ArbitrationPacket
    ) -> Stage2ArbitrationDecision:
        return self._arbitrate(
            disagreement=packet.disagreement,
            packet=packet,
            schema=Stage2ArbitrationDecision,
        )

    def stage3_arbitrator(
        self, packet: Stage3ArbitrationPacket
    ) -> Stage3ArbitrationDecision:
        return self._arbitrate(
            disagreement=packet.disagreement,
            packet=packet,
            schema=Stage3ArbitrationDecision,
        )

    def _arbitrate(
        self,
        *,
        disagreement: DisagreementMap,
        packet: BaseModel,
        schema: type[SchemaT],
    ) -> SchemaT:
        candidates = _focused_arbitration_candidates(packet)
        canonical_schema = schema
        role = (
            "stage2_arbitrator"
            if isinstance(packet, Stage2ArbitrationPacket)
            else "stage3_arbitrator"
        )
        policy = (
            "\n" + stage2_policy_guidance(packet.domain_profile)
            if isinstance(packet, Stage2ArbitrationPacket)
            else ""
        )
        stage3_authority = (
            "\nLABEL AUTHORITY: Preserve each unanimous candidate label unless "
            "the exact dimension symptom_label or root_cause_label is listed "
            "in disagreement.dimensions. In particular, symptom_review does "
            "not authorize changing symptom_label, and cause_review does not "
            "authorize changing root_cause_label. Unsupported-boundary or "
            "specificity reviews may confirm support or return UNRESOLVED, but "
            "must copy labels unchanged."
            if isinstance(packet, Stage3ArbitrationPacket)
            else ""
        )
        stage3_causal_responsibility = (
            ASE2022_CAUSAL_RESPONSIBILITY_INSTRUCTIONS
            if isinstance(packet, Stage3ArbitrationPacket)
            and packet.domain_profile == "ase2022"
            else ""
        )
        native_arbitration = isinstance(packet, Stage3ArbitrationPacket) and bool(packet.taxonomy.get("annotation_modes"))
        explicit_claims = self._rule_checks and isinstance(packet, Stage3ArbitrationPacket)
        if explicit_claims:
            from .experiment_two import arbitration_schema
            schema = arbitration_schema(schema)
        root_gates = []
        if isinstance(packet, Stage3ArbitrationPacket) and packet.relevant_evidence:
            record_ids = {item.record_id for item in packet.relevant_evidence}
            if len(record_ids) == 1:
                record_id = next(iter(record_ids))
                root_gates = [gate for team in ('A', 'B')
                              if (gate := self._a_root_gate(record_id, packet.classification_ledger_version, team)) is not None]
        def validate_native_arbitration(result):
            if getattr(self._explainable_tools, 'enforce_proof_labels', False):
                label = getattr(result, 'root_cause_label', None)
                allowed = {label for gate in root_gates for label in gate.get('allowed_labels', ())}
                if label is not None and label not in allowed:
                    raise ValueError('Arbitrated root label lacks a complete current rule proof; use supported labels or return UNRESOLVED')
            if native_arbitration:
                for dimension in ("symptom", "root_cause"):
                    value = getattr(result, f"{dimension}_label", None)
                    if value is not None:
                        canonical_label(packet.taxonomy, dimension, value)
            if (not explicit_claims and len(root_gates) == 2 and not any(g['unknown_allowed'] for g in root_gates)
                    and getattr(result, 'root_cause_label', None) == 'Unknown'):
                raise ValueError('Unknown contradicts both pre-decision gates; resolve the listed labels or return UNRESOLVED')
            if explicit_claims:
                from .experiment_two import validate_final_claims
                validate_final_claims(result, arbitration_view)
        arbitration_view = EvidenceView(record_id=(packet.relevant_evidence[0].record_id if packet.relevant_evidence else 'empty'),
            task='stage3',domain_profile=packet.domain_profile,taxonomy=packet.taxonomy,
            ledger_version=packet.classification_ledger_version,
            items=tuple(packet.relevant_evidence)) if isinstance(packet, Stage3ArbitrationPacket) else None
        navigation = {}
        if arbitration_view is not None and self._source_graph_mode != 'off':
            from .source_graph import source_context, case_query
            navigation['source_navigation'] = source_context(arbitration_view,case_query(arbitration_view),mode=self._source_graph_mode)
        result = self._client.complete(
            system_prompt=(
                taxonomy_guidance(
                    packet.domain_profile,
                    packet.taxonomy,
                    include_stage3_boundaries=not isinstance(
                        packet,
                        Stage2ArbitrationPacket,
                    ),
                )
                + policy
                + stage3_authority
                + ("\nGLOBAL SUPERVISION: Use the latest supervisory review as a fallible agenda. "
                   "Check its warrants against original evidence; prior model agreement is not evidence. "
                   "Resolve the actual current candidate dispute, address recorded causal gaps, and "
                   "do not automatically choose Unknown because the supervisor reports evidence_gap."
                   if isinstance(packet, Stage3ArbitrationPacket) and packet.supervision is not None else "")
                + stage3_causal_responsibility
                + self._root_cause_decision_policy(
                    packet.domain_profile, "stage3" if isinstance(packet, Stage3ArbitrationPacket) else "stage2"
                )
                + "\nYou are a targeted anonymous arbitrator. Resolve only the "
                "listed disagreement dimensions using the exact cited evidence "
                "content in this packet. Do not perform a fresh unconstrained "
                "analysis. If that bounded evidence is insufficient, return "
                "UNRESOLVED without a decision or labels.\n"
                + ("FINAL CLAIM AUDIT: Return rationale_sentences and rationale_findings. "
                   "rationale must equal those sentences joined by one space. For each sentence "
                   "use target_id rationale:0, rationale:1, etc., an exact source quotation, "
                   "applicable rule, rationale, status and missing_evidence. A resolved answer "
                   "may retain only entailed, appropriately qualified sentences. Do not reintroduce "
                   "mechanisms withdrawn by the current team checkers. Preliminary module_a_root_gates "
                   "are not evidence and may be superseded by the current checked candidates. "
                   "A supported reported cause is not an independently demonstrated mechanism.\n"
                   if explicit_claims else "")
                + "\nOUTPUT CONTRACT:\n"
                + json.dumps(_compact_schema(schema), ensure_ascii=False)
                + "\nReturn JSON only."
            ),
            user_prompt=json.dumps(
                {
                    "disagreement": disagreement.model_dump(mode="json"),
                    **navigation,
                    "candidates": candidates,
                    **({"supervision": packet.supervision} if isinstance(packet, Stage3ArbitrationPacket) and packet.supervision is not None else {}),
                    **({"module_a_root_gates": root_gates} if root_gates else {}),
                    "evidence_snapshot": {
                        "ledger_version": packet.classification_ledger_version,
                        "items": [
                            item.model_dump(mode="json")
                            for item in packet.relevant_evidence
                        ],
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            schema=schema,
            options=self._model_call_options(
                role,
                team_id=None,
                perspective=None,
            ),
            **({"result_validator": validate_native_arbitration} if explicit_claims or native_arbitration or root_gates
               or getattr(self._explainable_tools, 'enforce_proof_labels', False) else {}),
        )
        if explicit_claims:
            with self._module_a_lock:
                self._final_claim_audits.append({'record_id': arbitration_view.record_id,
                    'audit': result.model_dump(mode='json')})
            # Audit extensions have been validated and are retained above. The
            # controller/final verifier consumes the strict canonical contract.
            result = canonical_schema.model_validate(result.model_dump(
                mode='python', exclude={'rationale_sentences', 'rationale_findings'}))
        if native_arbitration:
            result = result.model_copy(update={f"{dimension}_label": canonical_label(packet.taxonomy, dimension, value)
                for dimension in ("symptom", "root_cause")
                if (value := getattr(result, f"{dimension}_label", None)) is not None})
        return result


def _focused_arbitration_candidates(packet: BaseModel) -> dict[str, object]:
    """Keep only decision-boundary fields needed for targeted arbitration."""

    if isinstance(packet, Stage2ArbitrationPacket):

        def stage2(report: BaseModel) -> dict[str, object]:
            data = report.model_dump(mode="json")
            return {
                key: data[key]
                for key in (
                    "decision",
                    "confidence",
                    "fault_claim",
                    "repair_claim",
                    "evidence_tests",
                    "alternative_hypothesis",
                    "decision_boundary",
                    "evidence_sufficiency",
                    "unresolved_evidence_gaps",
                    "supporting_evidence_ids",
                    "counter_evidence_ids",
                )
            }

        return {
            "team_a": stage2(packet.report_a),
            "team_b": stage2(packet.report_b),
        }

    if isinstance(packet, Stage3ArbitrationPacket):

        def stage3(team: BaseModel) -> dict[str, object]:
            data = team.model_dump(mode="json")
            return {
                "symptom": {
                    key: data["symptom"][key]
                    for key in (
                        "label",
                        "behavior_claim",
                        "alternative_label",
                        "boundary_reason",
                        "boundary_evidence_ids",
                        "confidence",
                        "evidence_sufficiency",
                        "unresolved_evidence_gaps",
                        "supporting_evidence_ids",
                        "counter_evidence_ids",
                    )
                },
                "root_cause": {
                    key: data["root_cause"][key]
                    for key in (
                        "label",
                        "defect_mechanism",
                        "causal_chain",
                        "alternative_label",
                        "boundary_reason",
                        "boundary_evidence_ids",
                        "confidence",
                        "evidence_sufficiency",
                        "unresolved_evidence_gaps",
                        "supporting_evidence_ids",
                        "counter_evidence_ids",
                    )
                },
                "consistency": data["consistency"],
            }

        return {
            "team_a": stage3(packet.team_a),
            "team_b": stage3(packet.team_b),
        }
    raise TypeError(f"unsupported arbitration packet: {type(packet).__name__}")
