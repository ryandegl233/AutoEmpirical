"""JSON audit serialization for reproducible workflow inspection."""

from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any

from pydantic import BaseModel

from .ledger import EvidenceLedger
from .workflow import AdaptiveWorkflowResult


def _json_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def workflow_audit_record(
    ledger: EvidenceLedger,
    result: AdaptiveWorkflowResult,
) -> dict[str, Any]:
    """Build a lossless record, including Stage 3 validity quarantine state."""

    view = ledger.view()
    serialized_view = view.model_dump(mode="json")
    preservation_snapshot = getattr(
        result.stage3, "preservation_audit_snapshot_json", None
    )
    serialized_stage3 = (
        json.loads(preservation_snapshot)
        if isinstance(preservation_snapshot, str)
        else _json_value(result.stage3)
    )
    if (
        isinstance(serialized_stage3, dict)
        and serialized_stage3.get("baseline_anchor") is None
    ):
        for preservation_field in (
            "pre_gate_candidate",
            "pre_gate_verification",
            "pre_gate_disagreement",
            "pre_gate_arbitration",
            "pre_gate_arbitration_evidence_ids",
            "pre_gate_consensus_confidence",
            "baseline_anchor",
            "revision_certificates",
            "preservation_result",
            "expected_baseline_anchor_hash",
            "expected_taxonomy_structure_hash",
            "revision_audit",
            "preservation_audit_snapshot_json",
        ):
            serialized_stage3.pop(preservation_field, None)
    return {
        "record_id": ledger.record_id,
        "ledger_version": ledger.version,
        "task": view.task,
        "domain_profile": view.domain_profile,
        "taxonomy": serialized_view["taxonomy"],
        "evidence_items": [item.model_dump(mode="json") for item in view.items],
        "stage2": _json_value(result.stage2),
        "stage3": serialized_stage3,
        "stop_reason": result.stop_reason,
    }


def workflow_audit_json(
    ledger: EvidenceLedger,
    result: AdaptiveWorkflowResult,
) -> str:
    """Serialize an audit record deterministically for JSONL persistence."""

    return json.dumps(
        workflow_audit_record(ledger, result),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
