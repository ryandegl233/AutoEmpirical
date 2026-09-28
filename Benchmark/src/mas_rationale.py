"""Versioned, observable classification rationales and exact evidence references."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


ExplanationMode = Literal["label_only", "evidence_rationale"]
CONTRACT_VERSION = 2
CITABLE_FIELDS = frozenset({
    "title", "body", "comments", "state", "created_at", "source_project",
    "issue_url", "changed_files", "code_diff",
})
MISSING_SOURCE_MARKERS = frozenset({
    "not_fetched", "not_available_in_source", "comments_unavailable_in_source",
})
KNOWN_EMPTY_COMMENT_MARKERS = frozenset({"no_comments_in_source", "[]"})


def is_source_availability_marker(field: str, content: str) -> bool:
    """Distinguish whole-field capture status from actual source text."""
    normalized = content.strip().casefold()
    return normalized in MISSING_SOURCE_MARKERS or (
        field == "comments" and normalized in KNOWN_EMPTY_COMMENT_MARKERS
    )


class StrictOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvidenceReference(StrictOutput):
    field: str = Field(min_length=1)
    quote: str = Field(min_length=1, max_length=500)

    @field_validator("field", "quote")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Evidence field and quote must be nonblank")
        return value


class LabelRationale(StrictOutput):
    reason: str = Field(min_length=1, max_length=2000)
    evidence_refs: list[EvidenceReference] = Field(max_length=8)
    evidence_limitations: str = Field(max_length=4000)

    @field_validator("reason")
    @classmethod
    def nonblank_reason(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("The model must supply a nonblank decision reason")
        return value

    @model_validator(mode="after")
    def require_limitations_without_citations(self) -> LabelRationale:
        if not self.evidence_refs and not self.evidence_limitations.strip():
            raise ValueError("Empty evidence_refs require explicit evidence_limitations")
        return self


class Stage2RationaleOutput(StrictOutput):
    decision: Literal["accepted_fault", "rejected_candidate"]
    decision_rationale: LabelRationale


class Stage3RationaleOutput(StrictOutput):
    symptom: str = Field(min_length=1)
    root_cause: str = Field(min_length=1)
    symptom_rationale: LabelRationale
    root_cause_rationale: LabelRationale


def validate_mode(mode: str) -> None:
    if mode not in {"label_only", "evidence_rationale"}:
        raise ValueError("explanation_mode must be label_only or evidence_rationale")


def output_schema(stage: str) -> type[BaseModel]:
    if stage == "stage2":
        return Stage2RationaleOutput
    if stage == "stage3":
        return Stage3RationaleOutput
    raise ValueError("stage must be stage2 or stage3")


def output_contract(stage: str) -> str:
    return (
        "Return ONLY one strict JSON object conforming to the schema below. "
        "For each label, give a brief decision rationale (one to three sentences) "
        "and short exact quotations from the supplied evidence fields. These are "
        "observable explanations, not a request for private internal deliberation. "
        "Each evidence_refs entry names the original input field and copies its "
        "quote exactly, including punctuation and whitespace. Cite only fields "
        "listed in CITABLE EVIDENCE FIELDS; never cite taxonomy definitions, "
        "instructions, gold labels or an AI User's invented input as evidence. "
        "Source availability markers and known-empty comment fields are capture "
        "status, not technical evidence, and cannot be quoted as support. "
        "Use evidence_limitations to state uncertainty or missing support. "
        "If no quotation is available, return evidence_refs=[] and explain the "
        "missing evidence in evidence_limitations; do not invent a quotation. "
        "Keep all explanations inside the JSON. No Markdown or additional keys.\n"
        + json.dumps(output_schema(stage).model_json_schema(), ensure_ascii=False)
    )


def prepare_evidence_fields(fields: dict[str, str], task_prompt: str) -> dict[str, str]:
    if not isinstance(fields, dict):
        raise ValueError("model_evidence_builder must return a field-to-text mapping")
    safe = {}
    for field, content in fields.items():
        if field not in CITABLE_FIELDS or not isinstance(content, str):
            raise ValueError(f"Ineligible model evidence field: {field}")
        if content and content not in task_prompt:
            raise ValueError(f"Evidence field {field} is not present in the built task")
        safe[field] = content
    return safe


def audit_citations(value: BaseModel, fields: dict[str, str], stage: str) -> tuple[dict, list]:
    dimensions = ("decision",) if stage == "stage2" else ("symptom", "root_cause")
    rationales, citations = {}, []
    for dimension in dimensions:
        rationale = getattr(value, dimension + "_rationale")
        rationales[dimension] = rationale.model_dump()
        for reference in rationale.evidence_refs:
            if reference.field not in CITABLE_FIELDS or reference.field not in fields:
                raise ValueError(f"Citation field {reference.field} is not model-visible evidence")
            content = fields[reference.field]
            if is_source_availability_marker(reference.field, content):
                raise ValueError(
                    f"Citation field {reference.field} is a source availability marker, not technical evidence"
                )
            start = content.find(reference.quote)
            if start < 0:
                raise ValueError(f"Citation quote is absent from field {reference.field}")
            citations.append({
                "dimension": dimension, "field": reference.field, "quote": reference.quote,
                "start_char": start, "end_char": start + len(reference.quote),
                "offset_convention": "zero-based Unicode code points; end exclusive; first exact occurrence",
                "source_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            })
    return rationales, citations


def validate_saved_audit(
    row: dict[str, Any], stage: str,
    parse_output: Callable[[str, type[BaseModel]], dict[str, Any]],
) -> None:
    if row.get("explanation_mode") != "evidence_rationale" or row.get("explanation_contract_version") != CONTRACT_VERSION:
        raise ValueError("Missing or incompatible rationale contract")
    society, audit = row.get("society", {}), row.get("explanation_audit", {})
    if audit.get("status") != "schema_and_citations_valid":
        raise ValueError("Missing successful rationale audit")
    task = society.get("task_prompt", "")
    if not isinstance(task, str) or not task:
        raise ValueError("Missing task prompt")
    if audit.get("task_prompt_sha256") != hashlib.sha256(task.encode("utf-8")).hexdigest():
        raise ValueError("Rationale task hash mismatch")
    fields = prepare_evidence_fields(audit.get("evidence_fields"), task)
    value = output_schema(stage).model_validate(society.get("parsed_output"))
    raw = society.get("raw_final_output")
    parsed_raw = parse_output(raw, output_schema(stage))["value"].model_dump()
    if parsed_raw != value.model_dump():
        raise ValueError("Saved explanation differs from the raw model response")
    source = row.get("output_source")
    if source == "society_assistant":
        turn_number = society.get("final_answer_turn")
        turns = society.get("turns", [])
        if type(turn_number) is not int or not 1 <= turn_number <= len(turns):
            raise ValueError("Missing selected assistant turn")
        selected = turns[turn_number - 1].get("assistant", {})
        if selected.get("content") != raw or selected.get("parsed_output") != parsed_raw:
            raise ValueError("Selected assistant turn differs from final output")
    elif source == "forced_finalizer":
        finalizer = society.get("forced_finalizer", {})
        attempts = finalizer.get("attempt_trace", [])
        if not attempts or attempts[-1].get("error") or attempts[-1].get("raw_output") != raw:
            raise ValueError("Missing successful finalizer attempt")
        if finalizer.get("raw_output") != raw or finalizer.get("parsed_output") != parsed_raw:
            raise ValueError("Selected finalizer differs from final output")
    else:
        raise ValueError("Unknown explanation output source")
    rationales, citations = audit_citations(value, fields, stage)
    if audit.get("rationales") != rationales or audit.get("citations") != citations:
        raise ValueError("Saved rationale or citation provenance mismatch")
    if audit.get("record_id") != row.get("record_id") or audit.get("source") != row.get("output_source"):
        raise ValueError("Rationale output provenance mismatch")
    labels = ("decision",) if stage == "stage2" else ("symptom", "root_cause")
    if row.get("final_prediction") != {field: getattr(value, field) for field in labels}:
        raise ValueError("Rationale labels differ from final_prediction")
