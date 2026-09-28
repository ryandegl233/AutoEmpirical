from __future__ import annotations

import copy
import csv
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Callable, Literal, Protocol, TypeVar
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from Benchmark.src import ase2022_llm_baseline as stage3_baseline
from Benchmark.src import ase2022_stage2_filter_baseline as stage2_baseline
from Benchmark.src import mas_rationale
from Benchmark.src.annotation_contracts import (
    annotation_guidance, annotation_metrics, annotation_mode, label_valid, labels_equal,
)
from Benchmark.src.mas_request_trace import capture_request, capture_response


ASE2022_PAPER_ID = stage2_baseline.ASE2022_PAPER_ID
ACCEPTED_FAULT = stage2_baseline.ACCEPTED_FAULT
REJECTED_CANDIDATE = stage2_baseline.REJECTED_CANDIDATE
ALLOWED_DECISIONS = stage2_baseline.ALLOWED_DECISIONS
DEFAULT_MODEL = "claude-3-5-sonnet-20241022"
DEFAULT_MAX_TURNS = 10
DEFAULT_SOCIETY_MODE = "evidence_anchored"
SAFE_TEXT_FIELDS = ("title", "body", "comments", "state", "created_at")


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Stage2ProposerOutput(_StrictModel):
    decision: Literal["accepted_fault", "rejected_candidate"]


class Stage2CriticOutput(_StrictModel):
    verdict: Literal["uphold", "revise"]
    suggested_decision: Literal["accepted_fault", "rejected_candidate"]
    evidence: list[str] = Field(min_length=1)
    reason: str = Field(min_length=1)


class Stage2JudgeOutput(_StrictModel):
    decision: Literal["accepted_fault", "rejected_candidate"]
    evidence: list[str] = Field(min_length=1)
    reason: str = Field(min_length=1)


class Stage3ProposerOutput(_StrictModel):
    symptom: str = Field(min_length=1)
    root_cause: str = Field(min_length=1)


class Stage3CriticOutput(_StrictModel):
    verdict: Literal["uphold", "revise"]
    suggested_symptom: str = Field(min_length=1)
    suggested_root_cause: str = Field(min_length=1)
    evidence: list[str] = Field(min_length=1)
    reason: str = Field(min_length=1)


class Stage3JudgeOutput(_StrictModel):
    symptom: str = Field(min_length=1)
    root_cause: str = Field(min_length=1)
    evidence: list[str] = Field(min_length=1)
    reason: str = Field(min_length=1)


class Stage2SocietyOutput(_StrictModel):
    decision: Literal["accepted_fault", "rejected_candidate"]


class Stage3SocietyOutput(_StrictModel):
    symptom: str = Field(min_length=1)
    root_cause: str = Field(min_length=1)


class AgentLike(Protocol):
    def step(self, prompt: str, response_format: type[BaseModel] | None = None) -> Any: ...


AgentFactory = Callable[[str, str], AgentLike]


class SocietyLike(Protocol):
    specified_task_prompt: Any
    user_agent: AgentLike
    assistant_agent: AgentLike

    def init_chat(self, init_msg_content: str | None = None) -> Any: ...

    def step(self, input_msg: Any) -> tuple[Any, Any]: ...


SocietyFactory = Callable[[str], SocietyLike]
SchemaT = TypeVar("SchemaT", bound=BaseModel)
TaskBuilder = Callable[
    [dict[str, str], Literal["stage2", "stage3"], dict[str, list[str]]],
    str,
]
CONFIG_SCHEMA_VERSION = 7
SocietyMode = Literal["native", "evidence_anchored"]
ExplanationMode = mas_rationale.ExplanationMode
EvidenceBuilder = Callable[[dict[str, str]], dict[str, str]]


class RoleExecutionError(ValueError):
    def __init__(self, message: str, metadata: dict[str, Any]) -> None:
        super().__init__(message)
        self.metadata = metadata


class SocietyInitializationError(RuntimeError):
    def __init__(self, cause: Exception, role_stats: dict[str, dict[str, Any]]) -> None:
        super().__init__(f"{type(cause).__name__}: {cause}")
        self.role_stats = role_stats


class _TrackedCallable:
    def __init__(self, target: Callable[..., Any], stats: dict[str, Any]) -> None:
        self._target = target
        self._stats = stats

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self._stats["api_request_count"] += 1
        started = time.perf_counter()
        trace = None
        http_start = len(self._stats.get("http_exchange_trace", []))
        if "request_trace" in self._stats:
            trace = {"attempt": self._stats["api_request_count"],
                     "request": capture_request(kwargs), "response": None, "error_type": None}
        try:
            response = self._target(*args, **kwargs)
            if trace is not None:
                trace["response"] = capture_response(response)
        except Exception as error:
            if trace is not None:
                trace["error_type"] = type(error).__name__
            raise
        finally:
            self._stats["latency_seconds"] += time.perf_counter() - started
            if trace is not None:
                trace["latency_seconds"] = time.perf_counter() - started
                trace["http_exchanges"] = copy.deepcopy(
                    self._stats.get("http_exchange_trace", [])[http_start:]
                )
                self._stats["request_trace"].append(trace)
        self._stats["usage_observed_request_count"] += 1
        usage = getattr(response, "usage", None)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = (
                usage.get(key, 0)
                if isinstance(usage, dict)
                else getattr(usage, key, 0) if usage is not None else 0
            )
            self._stats[key] += int(value or 0)
        return response


class _CompletionsProxy:
    def __init__(self, target: Any, stats: dict[str, Any]) -> None:
        self._target = target
        if hasattr(target, "create"):
            self.create = _TrackedCallable(target.create, stats)
        if hasattr(target, "parse"):
            self.parse = _TrackedCallable(target.parse, stats)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)


class _ChatProxy:
    def __init__(self, target: Any, stats: dict[str, Any]) -> None:
        self._target = target
        self.completions = _CompletionsProxy(target.completions, stats)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)


class _BetaProxy:
    def __init__(self, target: Any, stats: dict[str, Any]) -> None:
        self._target = target
        self.chat = _ChatProxy(target.chat, stats)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)


class _CountingOpenAIClient:
    def __init__(self, client: Any, *, capture_trace: bool = False) -> None:
        self._client = client
        self.request_stats = {
            "api_request_count": 0,
            "usage_observed_request_count": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "latency_seconds": 0.0,
        }
        if capture_trace:
            self.request_stats["request_trace"] = []
            self.request_stats["http_exchange_trace"] = []
            transport = getattr(client, "_client", None)
            hooks = getattr(transport, "event_hooks", None)
            if isinstance(hooks, dict):
                hooks.setdefault("response", []).append(self._capture_http_response)
        self.chat = _ChatProxy(client.chat, self.request_stats)
        self.beta = _BetaProxy(client.beta, self.request_stats)

    def _capture_http_response(self, response: Any) -> None:
        """Keep visible answers before SDK structured parsing can discard them."""
        exchange = {
            "sdk_attempt": self.request_stats["api_request_count"],
            "status_code": response.status_code,
            "request": {}, "response": None, "read_error_type": None,
        }
        try:
            request_body = json.loads(response.request.content)
            exchange["request"] = capture_request(request_body)
            # httpx caches read bodies, so the SDK still consumes the same bytes.
            response.read()
            exchange["response"] = capture_response(response.json())
        except Exception as error:
            exchange["read_error_type"] = type(error).__name__
        self.request_stats["http_exchange_trace"].append(exchange)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def build_config_hash(
    model: str,
    stage: str,
    records: list[dict[str, str]],
    taxonomy: dict[str, list[str]],
    temperature: float | None,
    backend_id: str = "",
    max_turns: int = DEFAULT_MAX_TURNS,
    society_mode: SocietyMode = DEFAULT_SOCIETY_MODE,
    require_valid_json: bool = False,
    explanation_mode: ExplanationMode = "label_only",
) -> str:
    mas_rationale.validate_mode(explanation_mode)
    if max_turns <= 0:
        raise ValueError("max_turns must be positive")
    architecture = society_architecture(society_mode)
    payload = {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "architecture": architecture,
        "society_mode": society_mode,
        "model": model,
        "stage": stage,
        "record_ids": [row["record_id"] for row in records],
        "taxonomy": taxonomy,
        "temperature": temperature,
        "backend_id": backend_id,
        "max_turns": max_turns,
        "require_valid_json": require_valid_json,
    }
    if explanation_mode == "evidence_rationale":
        payload.update({
            "explanation_mode": explanation_mode,
            "explanation_contract_version": mas_rationale.CONTRACT_VERSION,
            "evidence_sha256": hashlib.sha256(json.dumps(
                [{key: row.get(key, "") for key in sorted(mas_rationale.CITABLE_FIELDS)} for row in records],
                ensure_ascii=False, sort_keys=True,
            ).encode("utf-8")).hexdigest(),
        })
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def society_architecture(society_mode: SocietyMode) -> str:
    if society_mode == "native":
        return "camel_roleplaying_society"
    if society_mode == "evidence_anchored":
        return "camel_roleplaying_evidence_anchored"
    raise ValueError("society_mode must be native or evidence_anchored")


def make_camel_agent_factory(
    model: str,
    api_key: str,
    base_url: str,
    *,
    temperature: float | None = 0.0,
    max_retries: int = 3,
    timeout: float | None = None,
    capture_trace: bool = False,
) -> AgentFactory:
    from camel.agents import ChatAgent
    from camel.models import ModelFactory
    from camel.types import ModelPlatformType
    from openai import OpenAI

    model_config = {} if temperature is None else {"temperature": temperature}

    def create(_role: str, system_message: str) -> AgentLike:
        client_options: dict[str, Any] = {
            "api_key": api_key,
            "base_url": base_url,
            "max_retries": max_retries,
        }
        if timeout is not None:
            client_options["timeout"] = timeout
        counting_client = _CountingOpenAIClient(OpenAI(**client_options), capture_trace=capture_trace)
        backend = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            model_type=model,
            url=base_url,
            api_key=api_key,
            model_config_dict=model_config,
            max_retries=max_retries,
            timeout=timeout,
            client=counting_client,
        )
        agent = ChatAgent(system_message=system_message, model=backend)
        agent._mas_request_stats = counting_client.request_stats
        return agent

    return create


def make_camel_society_factory(
    model: str,
    api_key: str,
    base_url: str,
    *,
    temperature: float | None = 0.0,
    max_retries: int = 3,
    timeout: float | None = None,
    society_mode: SocietyMode = DEFAULT_SOCIETY_MODE,
    capture_trace: bool = False,
) -> SocietyFactory:
    from camel.models import ModelFactory
    from camel.societies import RolePlaying
    from camel.types import ModelPlatformType
    from openai import OpenAI

    model_config = {} if temperature is None else {"temperature": temperature}
    society_architecture(society_mode)

    def create_backend() -> tuple[Any, dict[str, Any]]:
        client_options: dict[str, Any] = {
            "api_key": api_key,
            "base_url": base_url,
            "max_retries": max_retries,
        }
        if timeout is not None:
            client_options["timeout"] = timeout
        counting_client = _CountingOpenAIClient(OpenAI(**client_options), capture_trace=capture_trace)
        backend = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            model_type=model,
            url=base_url,
            api_key=api_key,
            model_config_dict=model_config,
            max_retries=max_retries,
            timeout=timeout,
            client=counting_client,
        )
        return backend, counting_client.request_stats

    def create(task_prompt: str) -> SocietyLike:
        user_backend, user_stats = create_backend()
        assistant_backend, assistant_stats = create_backend()
        roleplaying_options: dict[str, Any] = {
            "assistant_role_name": "Software Fault Analyst",
            "user_role_name": "Empirical Software Researcher",
            "task_prompt": task_prompt,
            "assistant_agent_kwargs": {"model": assistant_backend},
            "user_agent_kwargs": {"model": user_backend},
        }
        role_stats = {
            "ai_user": user_stats,
            "ai_assistant": assistant_stats,
        }
        if society_mode == "native":
            task_backend, task_stats = create_backend()
            roleplaying_options["task_prompt"] = _escape_task_for_camel_specifier(
                task_prompt
            )
            roleplaying_options["task_specify_agent_kwargs"] = {
                "model": task_backend
            }
            role_stats = {"task_specifier": task_stats, **role_stats}
        else:
            roleplaying_options["with_task_specify"] = False
        try:
            society = RolePlaying(**roleplaying_options)
        except Exception as error:
            if capture_trace:
                raise SocietyInitializationError(error, role_stats) from error
            raise
        society._mas_role_request_stats = role_stats
        return society

    return create


def _escape_task_for_camel_specifier(task_prompt: str) -> str:
    return task_prompt.replace("{", "{{").replace("}", "}}")


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return [{key: (value or "").strip() for key, value in row.items()} for row in csv.DictReader(handle)]


def _cohort_row(row: dict[str, str], decision: str) -> dict[str, str]:
    return {
        "record_id": row.get("record_id", ""),
        "paper_id": row.get("paper_id", ""),
        "issue_url": row.get("issue_url", ""),
        **{field: row.get(field, "") for field in SAFE_TEXT_FIELDS},
        "decision": decision,
        "symptom": row.get("symptom", "") if decision == ACCEPTED_FAULT else "",
        "root_cause": row.get("root_cause", "") if decision == ACCEPTED_FAULT else "",
    }


def _is_github_issue_url(source_uri: str) -> bool:
    parsed = urlparse(source_uri.strip())
    segments = [segment for segment in parsed.path.split("/") if segment]
    return (
        parsed.scheme.casefold() == "https"
        and parsed.netloc.casefold() == "github.com"
        and len(segments) == 4
        and segments[2].casefold() == "issues"
        and segments[3].isdigit()
        and int(segments[3]) > 0
    )


def _issue_only_candidate_pools(
    examples: list[dict[str, str]],
    stage3_rows: list[dict[str, str]],
    excluded_record_ids: set[str],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    stage3_by_id = {
        row.get("record_id", ""): row
        for row in stage3_rows
    }
    if len(stage3_by_id) != len(stage3_rows):
        raise ValueError("Stage 3 contains duplicate record_id")
    eligible = [
        row
        for row in examples
        if row.get("record_id") not in excluded_record_ids
        and _is_github_issue_url(row.get("issue_url", ""))
    ]
    positive_pool: list[dict[str, str]] = []
    for row in eligible:
        if row.get("decision") != ACCEPTED_FAULT:
            continue
        labels = stage3_by_id.get(row["record_id"])
        if labels is None:
            continue
        if not labels.get("symptom", "").strip() or not labels.get(
            "root_cause", ""
        ).strip():
            raise ValueError(
                f"positive {row['record_id']} is missing Stage 3 labels"
            )
        positive_pool.append(
            {
                **row,
                "symptom": labels["symptom"],
                "root_cause": labels["root_cause"],
            }
        )
    negative_pool = [
        row
        for row in eligible
        if row.get("decision") == REJECTED_CANDIDATE
    ]
    return positive_pool, negative_pool


def _sha256_ranked_sample(
    pool: list[dict[str, str]],
    *,
    count: int,
    seed: int,
    decision: str,
) -> list[dict[str, str]]:
    def rank(row: dict[str, str]) -> tuple[str, str]:
        record_id = row["record_id"]
        payload = (
            f"sha256_rank_v1:{seed}:{decision}:{record_id}"
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest(), record_id

    return sorted(pool, key=rank)[:count]


def select_issue_only_holdout(
    examples: list[dict[str, str]],
    stage3_rows: list[dict[str, str]],
    *,
    excluded_record_ids: set[str],
    positives: int = 50,
    negatives: int = 50,
    seed: int = 20260806,
) -> list[dict[str, str]]:
    """Select a reproducible balanced holdout where the PR gate is inert."""

    if positives <= 0 or negatives <= 0:
        raise ValueError("positives and negatives must be positive")
    positive_pool, negative_pool = _issue_only_candidate_pools(
        examples,
        stage3_rows,
        excluded_record_ids,
    )
    if len(positive_pool) < positives:
        raise ValueError(
            f"requested {positives} positive issues but found "
            f"{len(positive_pool)}"
        )
    if len(negative_pool) < negatives:
        raise ValueError(
            f"requested {negatives} negative issues but found "
            f"{len(negative_pool)}"
        )

    selected_positives = _sha256_ranked_sample(
        positive_pool,
        count=positives,
        seed=seed,
        decision=ACCEPTED_FAULT,
    )
    selected_negatives = _sha256_ranked_sample(
        negative_pool,
        count=negatives,
        seed=seed,
        decision=REJECTED_CANDIDATE,
    )
    return [
        *(
            _cohort_row(row, ACCEPTED_FAULT)
            for row in sorted(
                selected_positives,
                key=lambda row: row["record_id"],
            )
        ),
        *(
            _cohort_row(row, REJECTED_CANDIDATE)
            for row in sorted(
                selected_negatives,
                key=lambda row: row["record_id"],
            )
        ),
    ]


def build_unified_cohort(
    stage2_sample_path: str | Path,
    stage3_sample_path: str | Path,
    positives: int = 50,
    negatives: int = 50,
) -> list[dict[str, str]]:
    if positives <= 0 or negatives <= 0:
        raise ValueError("positives and negatives must be positive")
    stage2_rows = _read_csv(stage2_sample_path)
    stage3_rows = _read_csv(stage3_sample_path)
    positive_rows = stage3_rows[:positives]
    negative_rows = [
        row for row in stage2_rows if row.get("decision") == REJECTED_CANDIDATE
    ][:negatives]
    if len(positive_rows) != positives:
        raise ValueError(f"requested {positives} positives but found {len(positive_rows)}")
    if len(negative_rows) != negatives:
        raise ValueError(f"requested {negatives} negatives but found {len(negative_rows)}")
    for row in positive_rows:
        if not row.get("symptom") or not row.get("root_cause"):
            raise ValueError(f"positive {row.get('record_id', '')} is missing Stage 3 labels")
    cohort = [
        *(_cohort_row(row, ACCEPTED_FAULT) for row in positive_rows),
        *(_cohort_row(row, REJECTED_CANDIDATE) for row in negative_rows),
    ]
    ids = [row["record_id"] for row in cohort]
    if any(not record_id for record_id in ids):
        raise ValueError("cohort contains missing record_id")
    if len(ids) != len(set(ids)):
        raise ValueError("cohort contains duplicate record_id")
    if any(row["paper_id"] != ASE2022_PAPER_ID for row in cohort):
        raise ValueError("cohort contains records outside ASE2022")
    return cohort


def write_unified_cohort(cohort: list[dict[str, str]], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "record_id",
        "paper_id",
        "issue_url",
        *SAFE_TEXT_FIELDS,
        "decision",
        "symptom",
        "root_cause",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cohort)
    return output


def load_unified_cohort(path: str | Path) -> list[dict[str, str]]:
    return _read_csv(path)


def _write_stage3_prompts(
    cohort: list[dict[str, str]],
    taxonomy: dict[str, list[str]],
    path: str | Path,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    system_prompt = stage3_baseline.build_system_prompt(taxonomy)
    with output.open("w", encoding="utf-8") as handle:
        for record in cohort:
            if record["decision"] != ACCEPTED_FAULT:
                continue
            row = {
                "record_id": record["record_id"],
                "paper_id": record["paper_id"],
                "issue_url": record["issue_url"],
                "system_prompt": system_prompt,
                "user_prompt": stage3_baseline.build_user_prompt(record),
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return output


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_text_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8-sig")
    canonical = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def prepare_issue_only_holdout_artifacts(
    *,
    stage1_path: str | Path,
    stage2_path: str | Path,
    stage3_path: str | Path,
    excluded_cohort_path: str | Path,
    output_dir: str | Path,
    positives: int = 50,
    negatives: int = 50,
    seed: int = 20260806,
) -> dict[str, Path]:
    """Prepare a frozen issue-only holdout and shared baseline inputs."""

    sources = {
        "stage1": Path(stage1_path),
        "stage2": Path(stage2_path),
        "stage3": Path(stage3_path),
        "excluded_cohort": Path(excluded_cohort_path),
    }
    examples = stage2_baseline.load_ase2022_stage2_filter_examples(
        stage1_path=sources["stage1"],
        stage2_path=sources["stage2"],
    )
    stage3_rows = _read_csv(sources["stage3"])
    excluded_rows = _read_csv(sources["excluded_cohort"])
    excluded_ids = {
        row.get("record_id", "")
        for row in excluded_rows
        if row.get("record_id", "")
    }
    cohort = select_issue_only_holdout(
        examples,
        stage3_rows,
        excluded_record_ids=excluded_ids,
        positives=positives,
        negatives=negatives,
        seed=seed,
    )
    selected_ids = {row["record_id"] for row in cohort}
    overlap = selected_ids & excluded_ids
    if overlap:
        raise ValueError(
            f"holdout overlaps excluded cohort: {sorted(overlap)[:5]}"
        )
    positive_pool, negative_pool = _issue_only_candidate_pools(
        examples,
        stage3_rows,
        excluded_ids,
    )

    output = Path(output_dir)
    taxonomy = {
        "symptom": sorted(stage3_baseline.SYMPTOM_DEFINITIONS),
        "root_cause": sorted(stage3_baseline.ROOT_CAUSE_DEFINITIONS),
    }
    cohort_path = write_unified_cohort(
        cohort,
        output / "ase2022_issue_only_holdout_cohort.csv",
    )
    stage2_prompts = stage2_baseline.write_prompts_jsonl(
        cohort,
        output / "ase2022_issue_only_holdout_stage2_prompts.jsonl",
    )
    stage3_prompts = _write_stage3_prompts(
        cohort,
        taxonomy,
        output / "ase2022_issue_only_holdout_stage3_prompts.jsonl",
    )
    taxonomy_path = output / "ase2022_issue_only_holdout_taxonomy.json"
    taxonomy_path.write_text(
        json.dumps(taxonomy, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    manifest_path = output / "ase2022_issue_only_holdout_manifest.json"
    manifest = {
        "task": "ase2022_issue_only_holdout",
        "paper_id": ASE2022_PAPER_ID,
        "sampling": {
            "seed": seed,
            "algorithm": "sha256_rank_v1",
            "positive_count": positives,
            "negative_count": negatives,
            "issue_only": True,
            "without_replacement": True,
            "excluded_record_count": len(excluded_ids),
            "overlap_with_excluded_count": len(overlap),
            "candidate_pool_counts": {
                ACCEPTED_FAULT: len(positive_pool),
                REJECTED_CANDIDATE: len(negative_pool),
            },
        },
        "record_ids": [row["record_id"] for row in cohort],
        "source_paths": {
            name: str(path) for name, path in sources.items()
        },
        "source_sha256": {
            name: _canonical_text_sha256(path)
            for name, path in sources.items()
        },
        "cohort_sha256": _canonical_text_sha256(cohort_path),
        "artifact_sha256": {
            "cohort": _canonical_text_sha256(cohort_path),
            "stage2_prompts": _canonical_text_sha256(stage2_prompts),
            "stage3_prompts": _canonical_text_sha256(stage3_prompts),
            "taxonomy": _canonical_text_sha256(taxonomy_path),
        },
        "model_input_fields": list(SAFE_TEXT_FIELDS),
        "gold_fields_excluded_from_agent_prompts": [
            "decision",
            "symptom",
            "root_cause",
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "cohort": cohort_path,
        "stage2_prompts": stage2_prompts,
        "stage3_prompts": stage3_prompts,
        "taxonomy": taxonomy_path,
        "manifest": manifest_path,
    }


def prepare_artifacts(
    stage2_sample_path: str | Path,
    stage3_sample_path: str | Path,
    output_dir: str | Path,
    positives: int = 50,
    negatives: int = 50,
) -> dict[str, Path]:
    output = Path(output_dir)
    cohort = build_unified_cohort(
        stage2_sample_path,
        stage3_sample_path,
        positives=positives,
        negatives=negatives,
    )
    taxonomy = {
        "symptom": sorted(stage3_baseline.SYMPTOM_DEFINITIONS),
        "root_cause": sorted(stage3_baseline.ROOT_CAUSE_DEFINITIONS),
    }
    cohort_path = write_unified_cohort(cohort, output / "ase2022_camel_mas_cohort.csv")
    stage2_prompts = stage2_baseline.write_prompts_jsonl(
        cohort, output / "ase2022_camel_mas_stage2_single_llm_prompts.jsonl"
    )
    stage3_prompts = _write_stage3_prompts(
        cohort, taxonomy, output / "ase2022_camel_mas_stage3_prompts.jsonl"
    )
    taxonomy_path = output / "ase2022_camel_mas_taxonomy.json"
    taxonomy_path.write_text(
        json.dumps(taxonomy, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    manifest_path = output / "ase2022_camel_mas_manifest.json"
    manifest = {
        "task": "ase2022_camel_mas_baseline",
        "paper_id": ASE2022_PAPER_ID,
        "stage2_sample_path": str(stage2_sample_path),
        "stage3_sample_path": str(stage3_sample_path),
        "counts": {ACCEPTED_FAULT: positives, REJECTED_CANDIDATE: negatives},
        "cohort_sha256": _sha256_file(cohort_path),
        "model_input_fields": list(SAFE_TEXT_FIELDS),
        "gold_fields_excluded_from_agent_prompts": ["decision", "symptom", "root_cause"],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return {
        "cohort": cohort_path,
        "stage2_prompts": stage2_prompts,
        "stage3_prompts": stage3_prompts,
        "taxonomy": taxonomy_path,
        "manifest": manifest_path,
    }


def parse_role_output(raw_text: str, schema: type[SchemaT]) -> dict[str, Any]:
    raw = raw_text.strip() if isinstance(raw_text, str) else ""
    normalized = raw
    format_normalized = False
    output_format = "strict_json"
    try:
        payload = json.loads(normalized)
    except (json.JSONDecodeError, TypeError) as strict_error:
        fenced = re.fullmatch(
            r"```(?:json)?[ \t]*\r?\n(?P<body>[\s\S]*?)\r?\n```",
            raw,
            flags=re.IGNORECASE,
        )
        if not fenced:
            raise ValueError(f"invalid JSON: {strict_error}") from strict_error
        normalized = fenced.group("body").strip()
        format_normalized = True
        output_format = "markdown_fenced_json"
        try:
            payload = json.loads(normalized)
        except json.JSONDecodeError as fenced_error:
            raise ValueError(f"invalid JSON inside Markdown fence: {fenced_error}") from fenced_error
    try:
        value = schema.model_validate(payload)
    except ValidationError as error:
        raise ValueError(str(error)) from error
    return {
        "value": value,
        "normalized_output": normalized,
        "format_normalized": format_normalized,
        "output_format": output_format,
    }


def parse_society_prediction(
    raw_text: str,
    schema: type[SchemaT],
) -> dict[str, Any]:
    raw = raw_text.strip() if isinstance(raw_text, str) else ""
    try:
        return parse_role_output(raw, schema)
    except ValueError as strict_error:
        decoder = json.JSONDecoder()
        for index, character in enumerate(raw):
            if character != "{":
                continue
            try:
                payload, _ = decoder.raw_decode(raw[index:])
                value = schema.model_validate(payload)
            except (json.JSONDecodeError, ValidationError):
                continue
            return {
                "value": value,
                "normalized_output": json.dumps(payload, ensure_ascii=False),
                "format_normalized": True,
                "output_format": "camel_wrapped_json",
            }
        raise ValueError(
            f"no valid Society prediction: {strict_error}"
        ) from strict_error


def build_society_task(
    record: dict[str, str],
    stage: Literal["stage2", "stage3"],
    taxonomy: dict[str, list[str]],
    *,
    output_contract: str | None = None,
) -> str:
    if stage == "stage2":
        return (
            f"{stage2_baseline.build_system_prompt(output_contract=output_contract)}\n\n"
            f"{stage2_baseline.build_user_prompt(record)}"
        )
    if stage == "stage3":
        return (
            f"{stage3_baseline.build_system_prompt(taxonomy, output_contract=output_contract)}\n\n"
            f"{stage3_baseline.build_user_prompt(record)}"
        )
    raise ValueError("stage must be stage2 or stage3")


def _response_content(response: Any) -> str:
    messages = getattr(response, "msgs", None)
    if messages:
        return str(messages[0].content)
    message = getattr(response, "msg", None)
    if message is not None:
        return str(message.content)
    if isinstance(response, str):
        return response
    raise ValueError("CAMEL response contains no message content")


def _response_message(response: Any) -> Any:
    message = getattr(response, "msg", None)
    if message is not None:
        return message
    messages = getattr(response, "msgs", None)
    if messages:
        return messages[0]
    raise ValueError("CAMEL response contains no message")


def _token_usage(response: Any) -> dict[str, int]:
    info = getattr(response, "info", None) or {}
    usage = info.get("usage") if isinstance(info, dict) else None
    if not isinstance(usage, dict):
        return {}
    return {
        key: int(value)
        for key, value in usage.items()
        if key in {"prompt_tokens", "completion_tokens", "total_tokens"}
        and isinstance(value, (int, float))
    }


def _run_role(
    agent: AgentLike,
    prompt: str,
    schema: type[SchemaT],
    max_retries: int,
    validator: Callable[[SchemaT], None] | None = None,
    *,
    capture_attempts: bool = False,
) -> tuple[SchemaT, dict[str, Any]]:
    errors: list[str] = []
    total_latency = 0.0
    accumulated_usage: dict[str, int] = {}
    last_raw = ""
    request_stats = getattr(agent, "_mas_request_stats", None)
    initial_stats = dict(request_stats) if isinstance(request_stats, dict) else None
    attempt_trace: list[dict[str, Any]] = []
    trace_start = len(request_stats.get("request_trace", [])) if isinstance(request_stats, dict) else 0
    for attempt in range(1, max_retries + 2):
        started = time.perf_counter()
        raw = ""
        try:
            response = agent.step(prompt, response_format=schema)
            raw = _response_content(response)
            last_raw = raw
            if initial_stats is None:
                for key, value in _token_usage(response).items():
                    accumulated_usage[key] = accumulated_usage.get(key, 0) + value
            parsed = parse_role_output(raw, schema)
            value = parsed["value"]
            if validator is not None:
                validator(value)
            total_latency += time.perf_counter() - started
            if initial_stats is not None:
                accumulated_usage = {
                    key: int(request_stats.get(key, 0) - initial_stats.get(key, 0))
                    for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                }
            metadata = {
                "raw_output": raw,
                "parsed_output": value.model_dump(),
                "attempts": attempt,
                "latency_seconds": total_latency,
                "token_usage": accumulated_usage,
                "api_request_count": (
                    int(
                        request_stats.get("api_request_count", 0)
                        - initial_stats.get("api_request_count", 0)
                    )
                    if initial_stats is not None
                    else attempt
                ),
                "token_usage_complete": (
                    request_stats.get("usage_observed_request_count", 0)
                    - initial_stats.get("usage_observed_request_count", 0)
                    == request_stats.get("api_request_count", 0)
                    - initial_stats.get("api_request_count", 0)
                    if initial_stats is not None
                    else True
                ),
                "format_normalized": parsed["format_normalized"],
                "output_format": parsed["output_format"],
                "errors": errors,
            }
            if capture_attempts:
                attempt_trace.append({"attempt": attempt, "prompt": prompt,
                    "raw_output": raw, "parsed_output": value.model_dump(), "error": ""})
                metadata["attempt_trace"] = attempt_trace
                metadata["request_trace"] = copy.deepcopy(request_stats.get("request_trace", [])[trace_start:]) if isinstance(request_stats, dict) else []
            return value, metadata
        except Exception as error:  # role errors are recorded and retried uniformly
            total_latency += time.perf_counter() - started
            errors.append(f"{type(error).__name__}: {error}")
            if capture_attempts:
                attempt_trace.append({"attempt": attempt, "prompt": prompt,
                    "raw_output": raw, "parsed_output": {}, "error": errors[-1]})
    if initial_stats is not None:
        accumulated_usage = {
            key: int(request_stats.get(key, 0) - initial_stats.get(key, 0))
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        }
    metadata = {
        "raw_output": last_raw,
        "parsed_output": {},
        "attempts": max_retries + 1,
        "latency_seconds": total_latency,
        "token_usage": accumulated_usage,
        "api_request_count": (
            int(
                request_stats.get("api_request_count", 0)
                - initial_stats.get("api_request_count", 0)
            )
            if initial_stats is not None
            else max_retries + 1
        ),
        "token_usage_complete": (
            request_stats.get("usage_observed_request_count", 0)
            - initial_stats.get("usage_observed_request_count", 0)
            == request_stats.get("api_request_count", 0)
            - initial_stats.get("api_request_count", 0)
            if initial_stats is not None
            else False
        ),
        "format_normalized": False,
        "output_format": "invalid",
        "errors": errors,
    }
    if capture_attempts:
        metadata["attempt_trace"] = attempt_trace
        metadata["request_trace"] = copy.deepcopy(request_stats.get("request_trace", [])[trace_start:]) if isinstance(request_stats, dict) else []
    raise RoleExecutionError("; ".join(errors), metadata)


def _taxonomy_validator(taxonomy: dict[str, list[str]]) -> Callable[[BaseModel], None]:
    def validate(value: BaseModel) -> None:
        payload = value.model_dump()
        symptom = payload.get("symptom", payload.get("suggested_symptom"))
        root_cause = payload.get("root_cause", payload.get("suggested_root_cause"))
        if not label_valid(taxonomy, "symptom", symptom):
            raise ValueError(f"unsupported symptom label: {symptom}")
        if not label_valid(taxonomy, "root_cause", root_cause):
            raise ValueError(f"unsupported root_cause label: {root_cause}")

    return validate


def _stage_specs(stage: str, taxonomy: dict[str, list[str]]) -> dict[str, Any]:
    if stage == "stage2":
        return {
            "schemas": (Stage2ProposerOutput, Stage2CriticOutput, Stage2JudgeOutput),
            "systems": (
                stage2_baseline.build_system_prompt(),
                "You are a fault-screening critic. Check the proposed decision against only the issue evidence. Return concise structured evidence; do not infer or request a gold label.",
                "You are the final fault-screening judge. Decide from the issue evidence, proposal, and critique. Never use or ask for a gold label.",
            ),
            "evidence": stage2_baseline.build_user_prompt,
            "validator": None,
        }
    if stage == "stage3":
        taxonomy_reference = (
            "\n\nAllowed symptom labels (exact strings):\n"
            f"{json.dumps(taxonomy['symptom'], ensure_ascii=False)}\n\n"
            "Allowed root_cause labels (exact strings):\n"
            f"{json.dumps(taxonomy['root_cause'], ensure_ascii=False)}"
        )
        return {
            "schemas": (Stage3ProposerOutput, Stage3CriticOutput, Stage3JudgeOutput),
            "systems": (
                stage3_baseline.build_system_prompt(taxonomy),
                "You are an ASE2022 taxonomy critic. Check both proposed labels against only the issue evidence and the allowed taxonomy. Return exact taxonomy labels and concise evidence."
                + taxonomy_reference,
                "You are the final ASE2022 taxonomy judge. Decide both labels from the issue evidence, proposal, critique, and allowed taxonomy. Never invent a label."
                + taxonomy_reference,
            ),
            "evidence": stage3_baseline.build_user_prompt,
            "validator": _taxonomy_validator(taxonomy),
        }
    raise ValueError("stage must be stage2 or stage3")


def run_three_role_record(
    record: dict[str, str],
    stage: Literal["stage2", "stage3"],
    taxonomy: dict[str, list[str]],
    model: str,
    agent_factory: AgentFactory,
    max_retries: int = 3,
    config_hash: str = "",
    backend_id: str = "",
) -> dict[str, Any]:
    specs = _stage_specs(stage, taxonomy)
    schemas = specs["schemas"]
    evidence_prompt = specs["evidence"](record)
    roles: dict[str, dict[str, Any]] = {}
    total_started = time.perf_counter()
    try:
        proposer_agent = agent_factory("proposer", specs["systems"][0])
        try:
            proposer, roles["proposer"] = _run_role(
                proposer_agent,
                evidence_prompt,
                schemas[0],
                max_retries,
                specs["validator"],
            )
        except RoleExecutionError as role_error:
            roles["proposer"] = role_error.metadata
            raise
        critic_agent = agent_factory("critic", specs["systems"][1])
        critic_prompt = (
            f"{evidence_prompt}\n\nPROPOSER OUTPUT:\n"
            f"{json.dumps(proposer.model_dump(), ensure_ascii=False)}"
        )
        try:
            critic, roles["critic"] = _run_role(
                critic_agent,
                critic_prompt,
                schemas[1],
                max_retries,
                specs["validator"],
            )
        except RoleExecutionError as role_error:
            roles["critic"] = role_error.metadata
            raise
        judge_agent = agent_factory("judge", specs["systems"][2])
        judge_prompt = (
            f"{evidence_prompt}\n\nPROPOSER OUTPUT:\n"
            f"{json.dumps(proposer.model_dump(), ensure_ascii=False)}\n\n"
            f"CRITIC OUTPUT:\n{json.dumps(critic.model_dump(), ensure_ascii=False)}"
        )
        try:
            judge, roles["judge"] = _run_role(
                judge_agent,
                judge_prompt,
                schemas[2],
                max_retries,
                specs["validator"],
            )
        except RoleExecutionError as role_error:
            roles["judge"] = role_error.metadata
            raise
        final = (
            {"decision": judge.decision}
            if stage == "stage2"
            else {"symptom": judge.symptom, "root_cause": judge.root_cause}
        )
        invalid = False
        error = ""
    except Exception as run_error:
        final = {}
        invalid = True
        error = f"{type(run_error).__name__}: {run_error}"
    return {
        "record_id": record["record_id"],
        "paper_id": record.get("paper_id", ""),
        "issue_url": record.get("issue_url", ""),
        "stage": stage,
        "model": model,
        "backend_id": backend_id,
        "config_hash": config_hash,
        "roles": roles,
        "final_prediction": final,
        "invalid": invalid,
        "error": error,
        "latency_seconds": time.perf_counter() - total_started,
    }


def _society_response_metadata(response: Any) -> dict[str, Any]:
    info = getattr(response, "info", None)
    return {
        "content": _response_content(response),
        "terminated": bool(getattr(response, "terminated", False)),
        "termination_reasons": (
            info.get("termination_reasons", []) if isinstance(info, dict) else []
        ),
    }


def _copy_society_role_stats(
    society: SocietyLike | None,
    initialization_stats: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    stats = getattr(society, "_mas_role_request_stats", initialization_stats)
    if not isinstance(stats, dict):
        return {}
    return {
        str(role): {
            key: value
            for key, value in role_stats.items()
            if key
            in {
                "api_request_count",
                "usage_observed_request_count",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "latency_seconds",
            }
        }
        for role, role_stats in stats.items()
        if isinstance(role_stats, dict)
    }


def _build_society_repair_feedback(
    stage: Literal["stage2", "stage3"],
    taxonomy: dict[str, list[str]],
    parse_error: str,
    explanation_mode: ExplanationMode = "label_only",
) -> str:
    if explanation_mode == "evidence_rationale":
        schema_instruction = mas_rationale.output_contract(stage)
    elif stage == "stage2":
        schema_instruction = (
            'Return exactly one of these strict JSON objects:\n'
            '{"decision":"accepted_fault"}\n'
            '{"decision":"rejected_candidate"}'
        )
    else:
        symptoms = json.dumps(taxonomy["symptom"], ensure_ascii=False)
        root_causes = json.dumps(taxonomy["root_cause"], ensure_ascii=False)
        schema_instruction = (
            'Return one strict JSON object with exactly these keys:\n'
            '{"symptom":"<allowed symptom>",'
            '"root_cause":"<allowed root cause>"}\n'
            f"Allowed symptom labels: {symptoms}\n"
            f"Allowed root_cause labels: {root_causes}"
        )
    if stage == "stage3" and "annotation_modes" in taxonomy:
        schema_instruction += "\n" + "\n".join(
            annotation_guidance(taxonomy, dimension)
            for dimension in ("symptom", "root_cause")
        )
    return (
        "You are the AI User in a RolePlaying Society. The task is not "
        "complete because the previous AI Assistant response did not contain "
        "a valid final prediction; please issue a new Instruction to the AI Assistant "
        "that requires the exact output below. Do not answer the classification "
        "yourself.\n"
        f"Parser error: {parse_error}\n"
        f"{schema_instruction}\n"
        "Preserve the classification already supported by the issue evidence. "
        "Do not invent or map labels. "
        + (
            "Return only the required JSON object with its rationale fields; no Markdown fences or CAMEL_TASK_DONE."
            if explanation_mode == "evidence_rationale"
            else "Return only the JSON object, without explanation, Markdown fences, or CAMEL_TASK_DONE."
        )
    )


def _split_user_instruction(raw: str) -> tuple[str, str]:
    """Separate the AI User's advisory instruction from its generated input."""
    input_match = re.search(r"(?im)^\s*Input\s*:\s*", raw)
    before_input = raw[: input_match.start()] if input_match else raw
    discarded_input = raw[input_match.end() :] if input_match else ""
    instruction_match = re.search(
        r"(?im)^\s*Instruction\s*:\s*", before_input
    )
    instruction = (
        before_input[instruction_match.end() :]
        if instruction_match
        else before_input
    )
    return instruction.strip(), discarded_input.strip()


def _build_evidence_anchored_assistant_input(
    task_prompt: str,
    record_id: str,
    evidence_sha256: str,
    instruction: str,
) -> str:
    advisory = instruction or "Analyze the immutable evidence and produce the required result."
    return (
        "EVIDENCE-ANCHORED SOCIETY TURN\n"
        f"Record ID: {record_id}\n"
        f"Immutable evidence SHA-256: {evidence_sha256}\n\n"
        "ORIGINAL IMMUTABLE TASK AND EVIDENCE:\n"
        f"{task_prompt}\n\n"
        "AI USER ADVISORY INSTRUCTION:\n"
        f"{advisory}\n\n"
        "Use only the ORIGINAL IMMUTABLE TASK AND EVIDENCE above. The AI "
        "User instruction is advisory and may not add facts. Ignore or reject "
        "any invented issue, code, symptoms, causes, labels, or other evidence. "
        "Return the prediction required by the original task."
    )


def _step_evidence_anchored_society(
    society: SocietyLike,
    input_msg: Any,
    task_prompt: str,
    record_id: str,
    evidence_sha256: str,
) -> tuple[Any, Any, dict[str, str]]:
    user_response = society.user_agent.step(input_msg)
    raw_user_output = _response_content(user_response)
    instruction, discarded_input = _split_user_instruction(raw_user_output)
    assistant_input = _build_evidence_anchored_assistant_input(
        task_prompt,
        record_id,
        evidence_sha256,
        instruction,
    )
    assistant_response = society.assistant_agent.step(assistant_input)
    return assistant_response, user_response, {
        "user_instruction": instruction,
        "discarded_user_input": discarded_input,
        "assistant_input": assistant_input,
    }


def _message_with_content(message: Any, content: str) -> Any:
    if not hasattr(message, "content"):
        raise ValueError("CAMEL message contains no content field")
    updated = copy.copy(message)
    updated.content = content
    return updated


def _build_forced_finalizer_prompt(
    task_prompt: str,
    stage: Literal["stage2", "stage3"],
    taxonomy: dict[str, list[str]],
    turns: list[dict[str, Any]],
    explanation_mode: ExplanationMode = "label_only",
) -> str:
    last_turn = turns[-1] if turns else {}
    last_user = last_turn.get("user", {}).get("content", "")
    last_assistant = last_turn.get("assistant", {}).get("content", "")
    output_rule = _build_society_repair_feedback(
        stage,
        taxonomy,
        "The Society reached its turn limit without valid JSON.",
        explanation_mode=explanation_mode,
    )
    return (
        "Finalize the following empirical software fault analysis. Base the "
        "decision on the original issue evidence and fixed taxonomy, not on "
        "any invented example from the discussion.\n\n"
        f"ORIGINAL TASK:\n{task_prompt}\n\n"
        f"LAST AI USER MESSAGE:\n{last_user}\n\n"
        f"LAST AI ASSISTANT MESSAGE:\n{last_assistant}\n\n"
        f"FINAL OUTPUT REQUIREMENT:\n{output_rule}"
    )


def run_roleplaying_society_record(
    record: dict[str, str],
    stage: Literal["stage2", "stage3"],
    taxonomy: dict[str, list[str]],
    model: str,
    society_factory: SocietyFactory,
    finalizer_factory: AgentFactory | None = None,
    finalizer_max_retries: int = 3,
    max_turns: int = DEFAULT_MAX_TURNS,
    config_hash: str = "",
    backend_id: str = "",
    society_mode: SocietyMode = DEFAULT_SOCIETY_MODE,
    task_builder: TaskBuilder | None = None,
    explanation_mode: ExplanationMode = "label_only",
    model_evidence_builder: EvidenceBuilder | None = None,
) -> dict[str, Any]:
    mas_rationale.validate_mode(explanation_mode)
    if max_turns <= 0:
        raise ValueError("max_turns must be positive")
    if finalizer_max_retries < 0:
        raise ValueError("finalizer_max_retries must be non-negative")
    architecture = society_architecture(society_mode)
    builder = task_builder or build_society_task
    evidence_fields: dict[str, str] = {}
    if explanation_mode == "evidence_rationale":
        task_prompt = builder(record, stage, taxonomy, output_contract=mas_rationale.output_contract(stage))
        evidence_builder = model_evidence_builder or stage3_baseline.model_evidence_fields
        evidence_fields = mas_rationale.prepare_evidence_fields(evidence_builder(record), task_prompt)
        citable_fields = [field for field, text in evidence_fields.items()
                          if text.strip() and not mas_rationale.is_source_availability_marker(field, text)]
        task_prompt += "\n\nCITABLE EVIDENCE FIELDS: " + json.dumps(citable_fields, ensure_ascii=False)
    else:
        task_prompt = builder(record, stage, taxonomy)
    immutable_evidence_sha256 = hashlib.sha256(
        task_prompt.encode("utf-8")
    ).hexdigest()
    schema: type[BaseModel] = (
        Stage2SocietyOutput if stage == "stage2" else Stage3SocietyOutput
    )
    validator = _taxonomy_validator(taxonomy) if stage == "stage3" else None
    if explanation_mode == "evidence_rationale":
        schema = mas_rationale.output_schema(stage)
        label_validator = validator
        def validate_with_rationale(value: BaseModel) -> None:
            if label_validator is not None:
                label_validator(value)
            mas_rationale.audit_citations(value, evidence_fields, stage)
        validator = validate_with_rationale
    started = time.perf_counter()
    society: SocietyLike | None = None
    initialization_stats: dict[str, dict[str, Any]] = {}
    turns: list[dict[str, Any]] = []
    latest_prediction: BaseModel | None = None
    latest_parse: dict[str, Any] | None = None
    final_answer_turn: int | None = None
    pending_repair_feedback = ""
    ignored_completion_signal_count = 0
    final_parse_error = ""
    forced_finalization_attempted = False
    forced_finalizer: dict[str, Any] = {}
    output_source = ""
    stop_reason = "error"
    error = ""
    try:
        society = society_factory(task_prompt)
        if society_mode == "evidence_anchored":
            input_msg = society.init_chat(
                init_msg_content=(
                    "Using only the immutable shared task evidence, issue one "
                    "advisory instruction to the Software Fault Analyst. Reply "
                    "using `Instruction: ...` and `Input: "
                    "USE_IMMUTABLE_EVIDENCE`. Do not invent or substitute an "
                    "issue instance."
                )
            )
        else:
            input_msg = society.init_chat()
        for turn_number in range(1, max_turns + 1):
            turn_audit = {
                "user_instruction": "",
                "discarded_user_input": "",
                "assistant_input": "",
            }
            if society_mode == "evidence_anchored":
                assistant_response, user_response, turn_audit = (
                    _step_evidence_anchored_society(
                        society,
                        input_msg,
                        task_prompt,
                        record["record_id"],
                        immutable_evidence_sha256,
                    )
                )
            else:
                assistant_response, user_response = society.step(input_msg)
            assistant_meta = _society_response_metadata(assistant_response)
            user_meta = _society_response_metadata(user_response)
            assistant_meta["parse_error"] = ""
            try:
                parsed = parse_society_prediction(assistant_meta["content"], schema)
                prediction = parsed["value"]
                if validator is not None:
                    validator(prediction)
                latest_prediction = prediction
                latest_parse = parsed
                final_answer_turn = turn_number
                assistant_meta["parsed_output"] = prediction.model_dump()
                assistant_meta["output_format"] = parsed["output_format"]
                assistant_meta["format_normalized"] = parsed["format_normalized"]
            except Exception as parse_error:
                assistant_meta["parsed_output"] = {}
                assistant_meta["output_format"] = "invalid"
                assistant_meta["format_normalized"] = False
                assistant_meta["parse_error"] = (
                    f"{type(parse_error).__name__}: {parse_error}"
                )
                final_parse_error = assistant_meta["parse_error"]
            turns.append(
                {
                    "turn": turn_number,
                    "is_format_repair": bool(pending_repair_feedback),
                    "repair_feedback": pending_repair_feedback,
                    **turn_audit,
                    "assistant": assistant_meta,
                    "user": user_meta,
                }
            )
            if latest_prediction is not None:
                stop_reason = "valid_json"
                output_source = "society_assistant"
                break
            if assistant_meta["terminated"]:
                stop_reason = "assistant_terminated"
                error = "CAMEL Assistant terminated before producing valid JSON"
                break
            if user_meta["terminated"]:
                stop_reason = "user_terminated"
                error = "CAMEL AI User terminated before valid JSON was produced"
                break
            if "CAMEL_TASK_DONE" in user_meta["content"]:
                ignored_completion_signal_count += 1
            if turn_number == max_turns:
                stop_reason = "max_turns"
                break
            pending_repair_feedback = _build_society_repair_feedback(
                stage,
                taxonomy,
                assistant_meta["parse_error"],
                explanation_mode=explanation_mode,
            )
            input_msg = _message_with_content(
                _response_message(assistant_response),
                pending_repair_feedback,
            )
        if (
            latest_prediction is None
            and stop_reason == "max_turns"
            and finalizer_factory is not None
        ):
            forced_finalization_attempted = True
            finalizer_agent = finalizer_factory(
                "forced_finalizer",
                (
                    "You are the final decision maker for an empirical software "
                    "fault classification task. Follow the supplied original "
                    "evidence, schema, and taxonomy exactly."
                ),
            )
            finalizer_prompt = _build_forced_finalizer_prompt(
                task_prompt,
                stage,
                taxonomy,
                turns,
                explanation_mode=explanation_mode,
            )
            try:
                final_value, forced_finalizer = _run_role(
                    finalizer_agent,
                    finalizer_prompt,
                    schema,
                    finalizer_max_retries,
                    validator,
                    capture_attempts=explanation_mode == "evidence_rationale",
                )
                latest_prediction = final_value
                latest_parse = {
                    "output_format": forced_finalizer["output_format"],
                    "format_normalized": forced_finalizer["format_normalized"],
                }
                stop_reason = "forced_finalizer_json"
                output_source = "forced_finalizer"
            except RoleExecutionError as finalizer_error:
                forced_finalizer = finalizer_error.metadata
                error = str(finalizer_error)
                stop_reason = "forced_finalizer_failed"
    except Exception as run_error:
        if isinstance(run_error, SocietyInitializationError):
            initialization_stats = run_error.role_stats
        error = f"{type(run_error).__name__}: {run_error}"
        stop_reason = "error"

    if latest_prediction is None:
        if not error:
            error = f"no valid Assistant prediction after {len(turns)} turns"
        final_prediction: dict[str, Any] = {}
        invalid = True
    else:
        payload = latest_prediction.model_dump()
        final_prediction = (
            {"decision": payload["decision"]}
            if stage == "stage2"
            else {
                "symptom": payload["symptom"],
                "root_cause": payload["root_cause"],
            }
        )
        invalid = False

    role_stats = _copy_society_role_stats(society, initialization_stats)
    if forced_finalization_attempted and forced_finalizer:
        finalizer_requests = int(forced_finalizer.get("api_request_count", 0))
        finalizer_usage = forced_finalizer.get("token_usage", {})
        role_stats["forced_finalizer"] = {
            "api_request_count": finalizer_requests,
            "usage_observed_request_count": (
                finalizer_requests
                if forced_finalizer.get("token_usage_complete", False)
                else 0
            ),
            "prompt_tokens": int(finalizer_usage.get("prompt_tokens", 0)),
            "completion_tokens": int(
                finalizer_usage.get("completion_tokens", 0)
            ),
            "total_tokens": int(finalizer_usage.get("total_tokens", 0)),
            "latency_seconds": float(
                forced_finalizer.get("latency_seconds", 0.0)
            ),
        }
    api_request_count = sum(
        int(stats.get("api_request_count", 0)) for stats in role_stats.values()
    )
    token_usage = {
        key: sum(int(stats.get(key, 0)) for stats in role_stats.values())
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    latency = time.perf_counter() - started
    specified_task_prompt = (
        getattr(society, "specified_task_prompt", "") if society else ""
    )
    if specified_task_prompt is None:
        specified_task_prompt = ""
    elif not isinstance(specified_task_prompt, str):
        specified_task_prompt = str(specified_task_prompt)
    result = {
        "record_id": record["record_id"],
        "paper_id": record.get("paper_id", ""),
        "issue_url": record.get("issue_url", ""),
        "stage": stage,
        "model": model,
        "backend_id": backend_id,
        "config_hash": config_hash,
        "architecture": architecture,
        "society_mode": society_mode,
        "output_source": output_source,
        "society": {
            "task_prompt": task_prompt,
            "specified_task_prompt": specified_task_prompt,
            "immutable_evidence_sha256": immutable_evidence_sha256,
            "max_turns": max_turns,
            "turn_count": len(turns),
            "turns": turns,
            "stop_reason": stop_reason,
            "final_answer_turn": final_answer_turn,
            "ignored_completion_signal_count": ignored_completion_signal_count,
            "final_parse_error": final_parse_error,
            "forced_finalization_attempted": forced_finalization_attempted,
            "forced_finalizer": forced_finalizer,
            "raw_final_output": (
                forced_finalizer.get("raw_output", "")
                if output_source == "forced_finalizer"
                else (
                    turns[final_answer_turn - 1]["assistant"]["content"]
                    if final_answer_turn is not None
                    else ""
                )
            ),
            "parsed_output": (
                latest_prediction.model_dump() if latest_prediction is not None else {}
            ),
            "output_format": (
                latest_parse["output_format"] if latest_parse is not None else "invalid"
            ),
            "format_normalized": (
                latest_parse["format_normalized"]
                if latest_parse is not None
                else False
            ),
            "role_request_stats": role_stats,
            "api_request_count": api_request_count,
            "token_usage": token_usage,
            "latency_seconds": latency,
        },
        "final_prediction": final_prediction,
        "invalid": invalid,
        "error": error,
        "latency_seconds": latency,
    }
    if explanation_mode == "evidence_rationale":
        rationales, citations = ({}, []) if latest_prediction is None else mas_rationale.audit_citations(latest_prediction, evidence_fields, stage)
        request_trace = []
        raw_role_stats = getattr(society, "_mas_role_request_stats", initialization_stats)
        for role, stats in raw_role_stats.items():
            for entry in stats.get("request_trace", []):
                request_trace.append({**copy.deepcopy(entry), "record_id": record["record_id"], "role": role,
                    "call_id": f"{record['record_id']}:{role}:{entry['attempt']}"})
        for entry in forced_finalizer.get("request_trace", []):
            request_trace.append({**copy.deepcopy(entry), "record_id": record["record_id"], "role": "forced_finalizer",
                "call_id": f"{record['record_id']}:forced_finalizer:{entry['attempt']}"})
        result.update({
            "explanation_mode": explanation_mode,
            "explanation_contract_version": mas_rationale.CONTRACT_VERSION,
            "explanation_audit": {
                "status": "invalid" if invalid else "schema_and_citations_valid",
                "record_id": record["record_id"], "source": output_source,
                "task_prompt_sha256": immutable_evidence_sha256,
                "evidence_fields": evidence_fields, "rationales": rationales, "citations": citations,
                "semantic_support": "not_verified",
                "request_trace": request_trace,
                "request_trace_scope": "Observed SDK create/parse calls and visible responses. Nested http_exchanges preserve received HTTP responses before SDK parsing, including received retry responses when the transport hook is available. Transport attempts with no response may only appear as an SDK error.",
                "request_trace_available": bool(request_trace),
            },
        })
    return result


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate_stage2(
    cohort: list[dict[str, str]],
    predictions: list[dict[str, Any]],
    source: Literal["final", "proposer"] = "final",
) -> dict[str, int | float]:
    rows = []
    for row in predictions:
        if source == "final":
            decision = row.get("final_prediction", {}).get("decision", "")
        else:
            decision = row.get("roles", {}).get("proposer", {}).get("parsed_output", {}).get("decision", "")
        rows.append({"record_id": row["record_id"], "decision": decision, "invalid": row.get("invalid", False) or not decision})
    return stage2_baseline.evaluate_filter_predictions(cohort, rows)


def evaluate_stage3(
    cohort: list[dict[str, str]],
    predictions: list[dict[str, Any]],
    source: Literal["final", "proposer"] = "final",
    *,
    taxonomy: dict[str, Any] | None = None,
) -> dict[str, int | float]:
    gold = {row["record_id"]: row for row in cohort if row.get("decision") == ACCEPTED_FAULT}
    if taxonomy is not None and "annotation_modes" in taxonomy:
        rows = []
        for row in predictions:
            prediction = (
                row.get("final_prediction", {}) if source == "final"
                else row.get("roles", {}).get("proposer", {}).get("parsed_output", {})
            )
            rows.append({**prediction, "record_id": row["record_id"], "invalid": row.get("invalid", False)})
        return annotation_metrics(list(gold.values()), rows, taxonomy)
    prediction_by_id = {row["record_id"]: row for row in predictions}
    symptom_correct = root_correct = joint_correct = invalid = 0
    for record_id, truth in gold.items():
        row = prediction_by_id.get(record_id)
        if not row or row.get("invalid", False):
            invalid += 1
            continue
        prediction = (
            row.get("final_prediction", {})
            if source == "final"
            else row.get("roles", {}).get("proposer", {}).get("parsed_output", {})
        )
        symptom_match = prediction.get("symptom") == truth["symptom"]
        root_match = prediction.get("root_cause") == truth["root_cause"]
        symptom_correct += int(symptom_match)
        root_correct += int(root_match)
        joint_correct += int(symptom_match and root_match)
    n = len(gold)
    valid = n - invalid
    return {
        "n": n,
        "valid_count": valid,
        "invalid_count": invalid,
        "invalid_rate": _safe_ratio(invalid, n),
        "symptom_correct": symptom_correct,
        "root_cause_correct": root_correct,
        "joint_correct": joint_correct,
        "symptom_accuracy": _safe_ratio(symptom_correct, valid),
        "root_cause_accuracy": _safe_ratio(root_correct, valid),
        "joint_accuracy": _safe_ratio(joint_correct, valid),
    }


def evaluate_end_to_end(
    cohort: list[dict[str, str]],
    stage2_predictions: list[dict[str, Any]],
    stage3_predictions: list[dict[str, Any]],
    *,
    taxonomy: dict[str, Any] | None = None,
) -> dict[str, int | float]:
    native_contract = taxonomy is not None and "annotation_modes" in taxonomy
    scored_dimensions = [
        dimension for dimension in ("symptom", "root_cause")
        if not native_contract or annotation_mode(taxonomy, dimension) in {"single_label", "multi_label"}
    ]
    stage2_by_id = {row["record_id"]: row for row in stage2_predictions}
    stage3_by_id = {row["record_id"]: row for row in stage3_predictions}
    positives = [row for row in cohort if row.get("decision") == ACCEPTED_FAULT]
    entered_ids: set[str] = set()
    missed_by_stage2 = 0
    complete_correct = 0
    stage2_invalid = 0
    stage3_invalid_after_entry = 0
    for truth in cohort:
        s2 = stage2_by_id.get(truth["record_id"], {})
        stage2_invalid += int(not s2 or s2.get("invalid", True))
        accepted = (
            not s2.get("invalid", True)
            and s2.get("final_prediction", {}).get("decision") == ACCEPTED_FAULT
        )
        if accepted:
            entered_ids.add(truth["record_id"])
        if truth.get("decision") != ACCEPTED_FAULT:
            continue
        if not accepted:
            missed_by_stage2 += 1
            continue
        s3 = stage3_by_id.get(truth["record_id"], {})
        stage3_invalid_after_entry += int(not s3 or s3.get("invalid", True))
        prediction = s3.get("final_prediction", {})
        if native_contract:
            complete_correct += int(
                not s3.get("invalid", True)
                and all(label_valid(taxonomy, dimension, prediction.get(dimension)) for dimension in ("symptom", "root_cause"))
                and all(labels_equal(taxonomy, dimension, prediction.get(dimension), truth.get(dimension)) for dimension in scored_dimensions)
            )
        else:
            complete_correct += int(
                not s3.get("invalid", True)
                and prediction.get("symptom") == truth.get("symptom")
                and prediction.get("root_cause") == truth.get("root_cause")
            )
    precision = _safe_ratio(complete_correct, len(entered_ids))
    recall = _safe_ratio(complete_correct, len(positives))
    execution_rows = [*stage2_predictions, *stage3_predictions]
    total_role_attempts = total_api_requests = total_tokens = 0
    total_role_latency = 0.0
    incomplete_token_roles = 0
    for row in execution_rows:
        society = row.get("society")
        if isinstance(society, dict):
            requests = int(society.get("api_request_count", 0))
            total_role_attempts += requests
            total_api_requests += requests
            total_tokens += int(
                society.get("token_usage", {}).get("total_tokens", 0)
            )
            total_role_latency += float(society.get("latency_seconds", 0.0))
            for stats in society.get("role_request_stats", {}).values():
                incomplete_token_roles += int(
                    int(stats.get("usage_observed_request_count", 0))
                    != int(stats.get("api_request_count", 0))
                )
            continue
        for metadata in row.get("roles", {}).values():
            attempts = int(metadata.get("attempts", 0))
            total_role_attempts += attempts
            total_api_requests += int(metadata.get("api_request_count", attempts))
            total_tokens += int(metadata.get("token_usage", {}).get("total_tokens", 0))
            total_role_latency += float(metadata.get("latency_seconds", 0.0))
            incomplete_token_roles += int(
                not metadata.get("token_usage_complete", True)
            )
    return {
        "cohort_count": len(cohort),
        "gold_positive_count": len(positives),
        "entered_stage3_count": len(entered_ids),
        "complete_correct_count": complete_correct,
        "missed_by_stage2_count": missed_by_stage2,
        "stage2_invalid_count": stage2_invalid,
        "stage3_invalid_after_entry_count": stage3_invalid_after_entry,
        "precision": precision,
        "recall": recall,
        "f1": _safe_ratio(2 * precision * recall, precision + recall),
        "total_role_attempts": total_role_attempts,
        "total_api_requests": total_api_requests,
        "total_role_latency_seconds": total_role_latency,
        "total_tokens": total_tokens,
        "token_usage_incomplete_role_count": incomplete_token_roles,
        **({"scored_dimensions": scored_dimensions, "annotation_modes": taxonomy["annotation_modes"]} if native_contract else {}),
    }


def select_stage3_records(
    cohort: list[dict[str, str]],
    stage2_predictions: list[dict[str, Any]],
) -> list[dict[str, str]]:
    stage2_by_id = {row["record_id"]: row for row in stage2_predictions}
    selected = []
    for record in cohort:
        if record.get("decision") == ACCEPTED_FAULT:
            selected.append(record)
            continue
        prediction = stage2_by_id.get(record["record_id"], {})
        if (
            not prediction.get("invalid", True)
            and prediction.get("final_prediction", {}).get("decision") == ACCEPTED_FAULT
        ):
            selected.append(record)
    return selected


def select_requested_records(
    records: list[dict[str, str]],
    record_ids: list[str] | None,
    limit: int | None,
) -> list[dict[str, str]]:
    selected = records
    if record_ids:
        requested = set(record_ids)
        known = {row["record_id"] for row in records}
        unknown = sorted(requested - known)
        if unknown:
            raise ValueError(f"unknown record_ids: {unknown}")
        selected = [row for row in records if row["record_id"] in requested]
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        selected = selected[:limit]
    return selected


def evaluate_society_diagnostics(
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    stop_reason_counts: dict[str, int] = {}
    final_answer_turn_counts: dict[str, int] = {}
    total_turns = 0
    early_stops = 0
    total_api_requests = 0
    total_tokens = 0
    total_latency = 0.0
    normalized = 0
    incomplete_token_roles = 0
    valid = 0
    for row in predictions:
        valid += int(not row.get("invalid", True))
        society = row.get("society", {})
        turn_count = int(society.get("turn_count", 0))
        max_turns = int(society.get("max_turns", 0))
        stop_reason = str(society.get("stop_reason", "unknown") or "unknown")
        answer_turn = society.get("final_answer_turn")
        answer_key = str(answer_turn) if answer_turn is not None else "none"
        total_turns += turn_count
        early_stops += int(
            turn_count < max_turns
            and stop_reason
            in {
                "valid_json",
                "camel_task_done",
                "assistant_terminated",
                "user_terminated",
            }
        )
        stop_reason_counts[stop_reason] = stop_reason_counts.get(stop_reason, 0) + 1
        final_answer_turn_counts[answer_key] = (
            final_answer_turn_counts.get(answer_key, 0) + 1
        )
        total_api_requests += int(society.get("api_request_count", 0))
        total_tokens += int(society.get("token_usage", {}).get("total_tokens", 0))
        total_latency += float(society.get("latency_seconds", 0.0))
        normalized += int(society.get("format_normalized", False))
        for stats in society.get("role_request_stats", {}).values():
            requests = int(stats.get("api_request_count", 0))
            observed = int(stats.get("usage_observed_request_count", 0))
            incomplete_token_roles += int(observed != requests)
    n = len(predictions)
    return {
        "n": n,
        "valid_count": valid,
        "invalid_count": n - valid,
        "total_turn_count": total_turns,
        "average_turn_count": _safe_ratio(total_turns, n),
        "early_stop_count": early_stops,
        "early_stop_rate": _safe_ratio(early_stops, n),
        "stop_reason_counts": stop_reason_counts,
        "final_answer_turn_counts": final_answer_turn_counts,
        "total_api_requests": total_api_requests,
        "total_tokens": total_tokens,
        "total_latency_seconds": total_latency,
        "format_normalized_count": normalized,
        "token_usage_incomplete_role_count": incomplete_token_roles,
    }


def _stage_label(payload: dict[str, Any], stage: str, critic: bool = False) -> Any:
    if stage == "stage2":
        return payload.get("suggested_decision" if critic else "decision")
    if stage == "stage3":
        if critic:
            return payload.get("suggested_symptom"), payload.get("suggested_root_cause")
        return payload.get("symptom"), payload.get("root_cause")
    raise ValueError("stage must be stage2 or stage3")


def _truth_label(record: dict[str, str], stage: str) -> Any:
    if stage == "stage2":
        return record.get("decision")
    return record.get("symptom"), record.get("root_cause")


def evaluate_collaboration(
    cohort: list[dict[str, str]],
    predictions: list[dict[str, Any]],
    stage: Literal["stage2", "stage3"],
) -> dict[str, int | float]:
    truth_by_id = {row["record_id"]: row for row in cohort}
    change = disagreement = improvement = degradation = 0
    attempts = 0
    api_requests = 0
    latency = 0.0
    total_tokens = 0
    normalized = 0
    incomplete_token_roles = 0
    complete_count = 0
    for row in predictions:
        roles = row.get("roles", {})
        if not {"proposer", "critic", "judge"}.issubset(roles):
            continue
        complete_count += 1
        proposer = roles["proposer"].get("parsed_output", {})
        critic = roles["critic"].get("parsed_output", {})
        judge = roles["judge"].get("parsed_output", {})
        proposer_label = _stage_label(proposer, stage)
        critic_label = _stage_label(critic, stage, critic=True)
        judge_label = _stage_label(judge, stage)
        change += int(proposer_label != judge_label)
        disagreement += int(proposer_label != critic_label)
        truth = _truth_label(truth_by_id.get(row["record_id"], {}), stage)
        proposer_correct = proposer_label == truth
        judge_correct = judge_label == truth
        improvement += int(not proposer_correct and judge_correct)
        degradation += int(proposer_correct and not judge_correct)
        for metadata in roles.values():
            attempts += int(metadata.get("attempts", 0))
            api_requests += int(
                metadata.get("api_request_count", metadata.get("attempts", 0))
            )
            latency += float(metadata.get("latency_seconds", 0.0))
            total_tokens += int(metadata.get("token_usage", {}).get("total_tokens", 0))
            normalized += int(metadata.get("format_normalized", False))
            incomplete_token_roles += int(
                not metadata.get("token_usage_complete", True)
            )
    n = len(predictions)
    return {
        "n": n,
        "complete_role_set_count": complete_count,
        "proposer_judge_change_count": change,
        "proposer_judge_change_rate": _safe_ratio(change, complete_count),
        "critic_disagreement_count": disagreement,
        "critic_disagreement_rate": _safe_ratio(disagreement, complete_count),
        "judge_improvement_count": improvement,
        "judge_degradation_count": degradation,
        "total_role_attempts": attempts,
        "total_api_requests": api_requests,
        "total_role_latency_seconds": latency,
        "total_tokens": total_tokens,
        "token_usage_incomplete_role_count": incomplete_token_roles,
        "format_normalized_count": normalized,
    }


def _complete_result(
    row: dict[str, Any], stage: str, model: str, config_hash: str,
    explanation_mode: ExplanationMode = "label_only",
) -> bool:
    if row.get("stage") != stage or row.get("model") != model:
        return False
    if row.get("config_hash") != config_hash or row.get("invalid", True):
        return False
    if row.get("explanation_mode", "label_only") != explanation_mode:
        return False
    if explanation_mode == "evidence_rationale":
        try:
            mas_rationale.validate_saved_audit(row, stage, parse_society_prediction)
        except (ValueError, TypeError, AttributeError):
            return False
    architecture = row.get("architecture")
    society_architectures = {
        "camel_roleplaying_society",
        "camel_roleplaying_evidence_anchored",
    }
    if architecture in society_architectures:
        society = row.get("society", {})
        if not isinstance(society, dict):
            return False
        required_society = {
            "task_prompt",
            "specified_task_prompt",
            "turn_count",
            "turns",
            "stop_reason",
        }
        if not required_society.issubset(society):
            return False
        if architecture == "camel_roleplaying_evidence_anchored":
            if row.get("society_mode") != "evidence_anchored":
                return False
            if "immutable_evidence_sha256" not in society:
                return False
            required_turn_audit = {
                "user_instruction",
                "discarded_user_input",
                "assistant_input",
            }
            turns = society.get("turns", [])
            if not isinstance(turns, list) or any(
                not isinstance(turn, dict)
                or not required_turn_audit.issubset(turn)
                for turn in turns
            ):
                return False
    elif not {"proposer", "critic", "judge"}.issubset(row.get("roles", {})):
        return False
    final = row.get("final_prediction", {})
    required = {"decision"} if stage == "stage2" else {"symptom", "root_cause"}
    return required.issubset(final)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def validate_final_prediction(
    row: dict[str, Any],
    stage: Literal["stage2", "stage3"],
    taxonomy: dict[str, list[str]],
) -> None:
    record_id = str(row.get("record_id") or "<missing>")
    if row.get("invalid", True):
        raise ValueError(
            f"record {record_id} has no valid final prediction"
        )
    prediction = row.get("final_prediction")
    if not isinstance(prediction, dict):
        raise ValueError(
            f"record {record_id} final_prediction must be a JSON object"
        )
    if stage == "stage2":
        if set(prediction) != {"decision"} or prediction.get("decision") not in {
            ACCEPTED_FAULT,
            REJECTED_CANDIDATE,
        }:
            raise ValueError(
                f"record {record_id} has invalid Stage 2 final_prediction"
            )
        return
    if stage != "stage3":
        raise ValueError("stage must be stage2 or stage3")
    if set(prediction) != {"symptom", "root_cause"}:
        raise ValueError(
            f"record {record_id} has invalid Stage 3 final_prediction keys"
        )
    if not isinstance(prediction.get("symptom"), str) or not label_valid(taxonomy, "symptom", prediction.get("symptom")):
        raise ValueError(
            f"record {record_id} has symptom outside the taxonomy"
        )
    if not isinstance(prediction.get("root_cause"), str) or not label_valid(taxonomy, "root_cause", prediction.get("root_cause")):
        raise ValueError(
            f"record {record_id} has root_cause outside the taxonomy"
        )


def run_stage_records(
    records: list[dict[str, str]],
    predictions_path: str | Path,
    stage: Literal["stage2", "stage3"],
    model: str,
    config_hash: str,
    record_runner: Callable[[dict[str, str]], dict[str, Any]],
    resume: bool = True,
    progress_callback: Callable[[int, int, str, str], None] | None = None,
    require_valid_json: bool = False,
    taxonomy: dict[str, list[str]] | None = None,
    explanation_mode: ExplanationMode = "label_only",
) -> list[dict[str, Any]]:
    mas_rationale.validate_mode(explanation_mode)
    output = Path(predictions_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if require_valid_json and taxonomy is None:
        raise ValueError("taxonomy is required when require_valid_json is enabled")
    existing: dict[str, dict[str, Any]] = {}
    for row in (_load_jsonl(output) if resume else []):
        if not _complete_result(row, stage, model, config_hash, explanation_mode):
            continue
        if require_valid_json:
            try:
                validate_final_prediction(row, stage, taxonomy or {})
            except ValueError:
                continue
        existing[row["record_id"]] = row
    results: list[dict[str, Any]] = []
    with output.open("w", encoding="utf-8") as handle:
        total = len(records)
        for current, record in enumerate(records, start=1):
            row = existing.get(record["record_id"])
            status = "resumed"
            if row is None:
                row = record_runner(record)
                row["stage"] = stage
                row["model"] = model
                row["config_hash"] = config_hash
                status = "completed"
            if explanation_mode == "evidence_rationale" and not row.get("invalid", True):
                try:
                    mas_rationale.validate_saved_audit(row, stage, parse_society_prediction)
                except (ValueError, TypeError, AttributeError) as exc:
                    row["invalid"] = True
                    row["error"] = f"Invalid saved rationale audit: {exc}"
            if require_valid_json and explanation_mode == "label_only":
                validate_final_prediction(row, stage, taxonomy or {})
            results.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            # Preserve the observable failure before strict mode stops the run.
            if require_valid_json and explanation_mode == "evidence_rationale":
                validate_final_prediction(row, stage, taxonomy or {})
            if progress_callback is not None:
                progress_callback(current, total, record["record_id"], status)
    return results
