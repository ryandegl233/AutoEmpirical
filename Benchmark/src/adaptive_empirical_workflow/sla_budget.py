"""Monotonic global, record, and request SLA budgets."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from time import monotonic
from typing import Callable


class SlaBudgetExhausted(RuntimeError):
    """Raised when a global, record, or request budget has no time left."""


@dataclass(frozen=True)
class SlaRequestDeadline:
    """One request's duration and absolute deadline in the budget clock domain."""

    timeout_seconds: float
    deadline_monotonic: float


class RoleNetworkRetryBudget:
    """A role-scoped retry allowance shared by every schema attempt."""

    def __init__(self, *, max_retries: int) -> None:
        if (
            isinstance(max_retries, bool)
            or not isinstance(max_retries, int)
            or max_retries < 0
        ):
            raise ValueError("max_retries must be a non-negative integer")
        self.max_retries = max_retries
        self._used_retries = 0
        self._lock = threading.Lock()

    @property
    def used_retries(self) -> int:
        with self._lock:
            return self._used_retries

    def claim_retry(self) -> bool:
        with self._lock:
            if self._used_retries >= self.max_retries:
                return False
            self._used_retries += 1
            return True


@dataclass(frozen=True)
class SlaBudgetConfig:
    global_seconds: float = 1680.0
    record_seconds: float = 240.0
    request_seconds: float = 75.0

    def __post_init__(self) -> None:
        for name in ("global_seconds", "record_seconds", "request_seconds"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")


class GlobalSlaBudget:
    def __init__(
        self, config: SlaBudgetConfig, *, clock: Callable[[], float] = monotonic
    ) -> None:
        self.config = config
        self._clock = clock
        self._started_at = clock()
        self._deadline_at = self._started_at + config.global_seconds

    def elapsed_seconds(self) -> float:
        return max(0.0, self._clock() - self._started_at)

    def remaining_seconds(self) -> float:
        return max(0.0, self.config.global_seconds - self.elapsed_seconds())

    def start_record(self) -> "RecordSlaBudget":
        if self.remaining_seconds() <= 0.0:
            raise SlaBudgetExhausted("global SLA budget exhausted")
        return RecordSlaBudget(self, started_at=self._clock())


class RecordSlaBudget:
    def __init__(self, global_budget: GlobalSlaBudget, *, started_at: float) -> None:
        self._global_budget = global_budget
        self._started_at = started_at
        self._deadline_at = min(
            global_budget._deadline_at,
            started_at + global_budget.config.record_seconds,
        )

    @property
    def config(self) -> SlaBudgetConfig:
        return self._global_budget.config

    def elapsed_seconds(self) -> float:
        return max(0.0, self._global_budget._clock() - self._started_at)

    def remaining_seconds(self) -> float:
        record_remaining = max(0.0, self.config.record_seconds - self.elapsed_seconds())
        return min(record_remaining, self._global_budget.remaining_seconds())

    def request_timeout(self) -> float:
        timeout = min(self.config.request_seconds, self.remaining_seconds())
        if timeout <= 0.0:
            raise SlaBudgetExhausted("SLA budget exhausted")
        return timeout

    def require_request_budget(self, *, minimum_seconds: float = 1.0) -> float:
        return self.require_request_deadline(
            minimum_seconds=minimum_seconds
        ).timeout_seconds

    def require_request_deadline(
        self, *, minimum_seconds: float = 1.0
    ) -> SlaRequestDeadline:
        if (
            isinstance(minimum_seconds, bool)
            or not isinstance(minimum_seconds, (int, float))
            or not math.isfinite(minimum_seconds)
            or minimum_seconds <= 0
        ):
            raise ValueError("minimum_seconds must be finite and positive")
        now = self._global_budget._clock()
        absolute_deadline = min(
            now + self.config.request_seconds,
            self._deadline_at,
            self._global_budget._deadline_at,
        )
        timeout = max(0.0, absolute_deadline - now)
        if timeout < minimum_seconds:
            raise SlaBudgetExhausted("insufficient SLA budget for request")
        return SlaRequestDeadline(
            timeout_seconds=timeout,
            deadline_monotonic=absolute_deadline,
        )
