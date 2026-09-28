"""Native paper annotation contracts shared by all experiment runners.

Multi-label values use a canonical string inside legacy message contracts;
this is a lossless set serialization, never a powerset of synthetic classes.
Free-text and constant symptoms are explicitly excluded from label accuracy.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

ANNOTATION_MODES = frozenset({"single_label", "multi_label", "free_text", "constant"})
MULTI_LABEL_SEPARATOR = " || "
NEW_PAPER_DOMAINS = frozenset({"fse2021", "icse2021", "icse2022", "icse2023", "icse2024"})


def annotation_mode(taxonomy: Mapping[str, Any], dimension: str) -> str:
    modes = taxonomy.get("annotation_modes", {})
    if not isinstance(modes, Mapping):
        raise ValueError("annotation_modes must be a mapping")
    mode = modes.get(dimension, "single_label")
    if mode not in ANNOTATION_MODES:
        raise ValueError(f"unsupported annotation mode {mode!r}")
    return str(mode)


def canonical_label(taxonomy: Mapping[str, Any], dimension: str, value: Any) -> str:
    """Validate and return an unambiguous value without silently dropping labels."""
    mode = annotation_mode(taxonomy, dimension)
    allowed = taxonomy.get(dimension, ())
    if mode == "free_text":
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{dimension} requires nonempty free text")
        return value.strip()
    if mode == "multi_label":
        if isinstance(value, str):
            parts = [part.strip() for part in value.split("||")]
        elif isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
            parts = [item.strip() for item in value]
        else:
            raise ValueError(f"{dimension} requires an atomic label set")
        if not parts or any(not part or part not in allowed for part in parts):
            raise ValueError(f"{dimension} contains an empty or unknown atomic label")
        if len(parts) != len(set(parts)):
            raise ValueError(f"{dimension} contains duplicate labels")
        return MULTI_LABEL_SEPARATOR.join(sorted(parts))
    if mode == "constant" and len(allowed) != 1:
        raise ValueError(f"{dimension} constant mode requires exactly one label")
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"{dimension} requires exactly one supplied label")
    return value


def label_valid(taxonomy: Mapping[str, Any], dimension: str, value: Any) -> bool:
    try:
        canonical_label(taxonomy, dimension, value)
    except (ValueError, TypeError):
        return False
    return True


def labels_equal(taxonomy: Mapping[str, Any], dimension: str, left: Any, right: Any) -> bool:
    try:
        return canonical_label(taxonomy, dimension, left) == canonical_label(taxonomy, dimension, right)
    except (ValueError, TypeError):
        return False


def annotation_guidance(taxonomy: Mapping[str, Any], dimension: str) -> str:
    mode = annotation_mode(taxonomy, dimension)
    if mode == "multi_label":
        return (f"{dimension}: MULTI-LABEL annotation. Select every supported atomic label; "
                "serialize the set as a string with labels sorted lexicographically and joined "
                "by ' || '. One label is allowed. Never invent a combined category or repeat "
                "a label. An alternative is a different supported label set.")
    if mode == "free_text":
        return (f"{dimension}: FREE-TEXT annotation. Describe the observed pre-fix behavior "
                "in a concise nonempty sentence. There is no symptom label enum. Do not turn "
                "example descriptions into classes. If an alternative is requested, describe "
                "a plausible different observed behavior; no exact-match symptom score is computed.")
    if mode == "constant":
        return (f"{dimension}: CONSTANT study descriptor. Use {taxonomy.get(dimension, [])!r}. "
                "The alternative may equal this sole descriptor. This dimension has no "
                "discriminative classification score.")
    return f"{dimension}: SINGLE-LABEL annotation. Return exactly one supplied label."


def annotation_metrics(
    gold: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    taxonomy: Mapping[str, Any],
) -> dict[str, Any]:
    """Fixed-cohort metrics on record_id/symptom/root_cause rows.

Missing or invalid predictions count as failures; unexpected and duplicate IDs
are errors. An ``invalid`` flag or ``stage3_valid=False`` invalidates a prediction.
"""
    def by_id(rows: Sequence[Mapping[str, Any]], owner: str) -> dict[str, Mapping[str, Any]]:
        result = {}
        for row in rows:
            record_id = str(row.get("record_id") or "")
            if not record_id:
                raise ValueError(f"{owner} is missing record_id")
            if record_id in result:
                raise ValueError(f"duplicate {owner} record_id: {record_id}")
            result[record_id] = row
        return result

    truth = by_id(gold, "gold")
    predicted = by_id(predictions, "prediction")
    if extra := set(predicted) - set(truth):
        raise ValueError(f"unexpected prediction IDs: {sorted(extra)}")
    modes = {dimension: annotation_mode(taxonomy, dimension) for dimension in ("symptom", "root_cause")}
    counts = {dimension: 0 for dimension in modes}
    valid_count = joint_count = 0
    true_positive = predicted_positive = gold_positive = 0
    for record_id, row in truth.items():
        target = {dimension: canonical_label(taxonomy, dimension, row.get(dimension)) for dimension in modes}
        candidate = predicted.get(record_id, {})
        valid = bool(candidate) and not candidate.get("invalid", False) and candidate.get("stage3_valid") is not False
        valid = valid and all(label_valid(taxonomy, dimension, candidate.get(dimension)) for dimension in modes)
        valid_count += int(valid)
        correct = {}
        for dimension, mode in modes.items():
            correct[dimension] = valid and labels_equal(taxonomy, dimension, candidate.get(dimension), target[dimension])
            counts[dimension] += int(correct[dimension])
        joint_count += int(all(correct.values()))
        if modes["symptom"] == "multi_label":
            expected_set = set(target["symptom"].split(MULTI_LABEL_SEPARATOR))
            actual_set = set(canonical_label(taxonomy, "symptom", candidate["symptom"]).split(MULTI_LABEL_SEPARATOR)) if valid else set()
            true_positive += len(expected_set & actual_set)
            predicted_positive += len(actual_set)
            gold_positive += len(expected_set)
    n = len(truth)
    scored = {dimension: mode in {"single_label", "multi_label"} for dimension, mode in modes.items()}
    result = {
        "n": n, "valid_count": valid_count, "invalid_count": n - valid_count,
        "symptom_mode": modes["symptom"], "root_cause_mode": modes["root_cause"],
        "symptom_accuracy": counts["symptom"] / n if n and scored["symptom"] else (0.0 if scored["symptom"] else None),
        "root_cause_accuracy": counts["root_cause"] / n if n and scored["root_cause"] else (0.0 if scored["root_cause"] else None),
        "joint_accuracy": joint_count / n if n and all(scored.values()) else (0.0 if all(scored.values()) else None),
        "symptom_scoring": "set_exact_match" if modes["symptom"] == "multi_label" else "categorical_accuracy" if scored["symptom"] else "not_scored",
        "root_cause_correct_count": counts["root_cause"],
    }
    if modes["symptom"] == "multi_label":
        precision = true_positive / predicted_positive if predicted_positive else 0.0
        recall = true_positive / gold_positive if gold_positive else 0.0
        result.update(symptom_set_exact_match=result["symptom_accuracy"],
                      symptom_micro_precision=precision, symptom_micro_recall=recall,
                      symptom_micro_f1=2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    if modes["symptom"] in {"free_text", "constant"}:
        result["symptom_scoring_reason"] = "Human review required for free text." if modes["symptom"] == "free_text" else "Study descriptor is constant and is not discriminative."
    return result
