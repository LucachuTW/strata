"""Lightweight runtime metrics for retrieval and agent workflows."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter

LATENCY_BUDGETS_MS = {
    "rewrite": (200, 400),
    "hybrid_retrieval": (50, 150),
    "rerank": (50, 150),
    "llm_first_token": (200, 500),
    "generation": (500, 1500),
    "critic": (1000, 2000),
}


@dataclass
class RuntimeMetrics:
    """Accumulates millisecond timings and numeric counters."""

    timings_ms: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        start = perf_counter()
        try:
            yield
        finally:
            self.timings_ms[name] = self.timings_ms.get(name, 0.0) + (perf_counter() - start) * 1000

    def count(self, name: str, value: int) -> None:
        self.counts[name] = value

    def as_dict(self) -> dict:
        budget_status = {}
        for name, elapsed in self.timings_ms.items():
            budget = LATENCY_BUDGETS_MS.get(name)
            if not budget:
                continue
            target_ms, max_ms = budget
            budget_status[name] = {
                "elapsed_ms": round(elapsed, 2),
                "target_ms": target_ms,
                "max_ms": max_ms,
                "within_target": elapsed <= target_ms,
                "within_max": elapsed <= max_ms,
            }
        return {
            "timings_ms": {k: round(v, 2) for k, v in self.timings_ms.items()},
            "counts": dict(self.counts),
            "budget_status": budget_status,
        }
