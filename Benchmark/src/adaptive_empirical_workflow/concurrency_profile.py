"""Accuracy-gated selection for explicit concurrency profiling runs."""

from __future__ import annotations

from collections.abc import Sequence


def _score(
    candidate: dict[str, object],
) -> tuple[float, float, float, int, int, float]:
    metrics = candidate["metrics"]
    assert isinstance(metrics, dict)
    stage2 = metrics["stage2"]
    stage3 = metrics["stage3"]
    end_to_end = metrics["end_to_end"]
    assert isinstance(stage2, dict)
    assert isinstance(stage3, dict)
    assert isinstance(end_to_end, dict)
    return (
        float(end_to_end["exact_match_accuracy"]),
        float(stage3["joint_accuracy"]),
        float(stage2["accuracy"]),
        -int(stage2["invalid_count"]),
        -int(stage3["invalid_count"]),
        -float(candidate["elapsed_seconds"]),
    )


def choose_concurrency(
    candidates: Sequence[dict[str, object]],
) -> dict[str, object]:
    """Choose accuracy and validity first, then the lowest wall time."""

    if not candidates:
        raise ValueError("at least one concurrency candidate is required")
    return max(candidates, key=_score)
