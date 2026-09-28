from __future__ import annotations

import math

import pytest

from Benchmark.src.adaptive_empirical_workflow.sla_budget import (
    GlobalSlaBudget,
    RecordSlaBudget,
    SlaBudgetConfig,
    SlaBudgetExhausted,
)


class FakeClock:
    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def test_record_request_timeout_is_bounded_by_all_three_limits() -> None:
    clock = FakeClock(100.0)
    global_budget = GlobalSlaBudget(
        SlaBudgetConfig(1680.0, 240.0, 75.0),
        clock=clock,
    )
    record = global_budget.start_record()
    clock.advance(210.0)
    assert record.request_timeout() == pytest.approx(30.0)


def test_record_budget_never_resets_between_retries() -> None:
    clock = FakeClock(0.0)
    record = GlobalSlaBudget(
        SlaBudgetConfig(1680.0, 240.0, 75.0),
        clock=clock,
    ).start_record()
    assert record.request_timeout() == 75.0
    clock.advance(70.0)
    assert record.request_timeout() == 75.0
    clock.advance(120.0)
    assert record.request_timeout() == 50.0


def test_request_deadline_is_absolute_and_bounded_by_active_global_expiry() -> None:
    clock = FakeClock(100.0)
    record = GlobalSlaBudget(
        SlaBudgetConfig(global_seconds=20.0, record_seconds=30.0, request_seconds=75.0),
        clock=clock,
    ).start_record()

    clock.advance(5.0)
    request = record.require_request_deadline()

    assert request.timeout_seconds == pytest.approx(15.0)
    assert request.deadline_monotonic == pytest.approx(120.0)
    clock.advance(10.0)
    assert request.deadline_monotonic == pytest.approx(120.0)


@pytest.mark.parametrize("values", [(0.0, 240.0, 75.0), (1680.0, -1.0, 75.0)])
def test_budget_config_requires_positive_finite_values(
    values: tuple[float, float, float]
) -> None:
    with pytest.raises(ValueError):
        SlaBudgetConfig(*values)


def test_budget_config_rejects_bool_durations() -> None:
    with pytest.raises(ValueError):
        SlaBudgetConfig(True, 240.0, 75.0)


def test_budget_config_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError):
        SlaBudgetConfig(math.inf, 240.0, 75.0)


def test_global_admission_exhaustion_is_monotonic_and_clamped() -> None:
    clock = FakeClock(10.0)
    global_budget = GlobalSlaBudget(SlaBudgetConfig(1680.0, 240.0, 75.0), clock=clock)
    record = global_budget.start_record()
    clock.advance(1681.0)
    assert global_budget.remaining_seconds() == 0.0
    assert record.remaining_seconds() == 0.0
    with pytest.raises(SlaBudgetExhausted):
        global_budget.start_record()
    with pytest.raises(SlaBudgetExhausted):
        record.require_request_budget()


def test_require_request_budget_rejects_insufficient_remaining_time() -> None:
    clock = FakeClock(0.0)
    record = GlobalSlaBudget(
        SlaBudgetConfig(1680.0, 240.0, 75.0), clock=clock
    ).start_record()
    clock.advance(239.5)
    with pytest.raises(SlaBudgetExhausted):
        record.require_request_budget(minimum_seconds=1.0)


def test_require_request_budget_rejects_bool_minimum() -> None:
    record = GlobalSlaBudget(SlaBudgetConfig()).start_record()
    with pytest.raises(ValueError):
        record.require_request_budget(minimum_seconds=True)


def test_backward_clock_does_not_create_negative_elapsed_or_remaining_time() -> None:
    clock = FakeClock(100.0)
    global_budget = GlobalSlaBudget(SlaBudgetConfig(), clock=clock)
    record = global_budget.start_record()
    clock.value = 90.0
    assert global_budget.elapsed_seconds() == 0.0
    assert global_budget.remaining_seconds() == 1680.0
    assert record.elapsed_seconds() == 0.0
    assert record.remaining_seconds() == 240.0


def test_public_types_are_constructible() -> None:
    assert isinstance(GlobalSlaBudget(SlaBudgetConfig()), GlobalSlaBudget)
    assert isinstance(RecordSlaBudget, type)
