"""Lightweight in-memory metrics collection for Phase 1 instrumentation.

Avoids external dependencies; designed for optional usage. All methods are
no-ops if the global singleton isn't initialized. Thread safety is coarse
but sufficient (GIL-protected increments)."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import time
import math


@dataclass
class _LatencyStats:
    count: int = 0
    total: float = 0.0
    min: float = math.inf
    max: float = 0.0

    def record(self, value: float):
        self.count += 1
        self.total += value
        if value < self.min:
            self.min = value
        if value > self.max:
            self.max = value

    def snapshot(self) -> Dict[str, Any]:
        if self.count == 0:
            return {"count": 0, "avg_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}
        return {
            "count": self.count,
            "avg_ms": (self.total / self.count) * 1000.0,
            "min_ms": self.min * 1000.0 if self.min is not math.inf else 0.0,
            "max_ms": self.max * 1000.0,
        }


@dataclass
class MetricsCollector:
    retrieval_calls: int = 0
    retrieval_hits: int = 0
    retrieval_semantic: int = 0
    retrieval_latency: _LatencyStats = field(default_factory=_LatencyStats)

    # Preference learning metrics (Phase 2)
    preference_variant_batches: int = 0
    preference_variants_total: int = 0
    preference_pairs_created: int = 0
    preference_latency: _LatencyStats = field(default_factory=_LatencyStats)

    score_components_sum: Dict[str, float] = field(default_factory=lambda: {"coherence": 0.0, "novelty": 0.0, "memory_alignment": 0.0, "relevance": 0.0, "semantic_relevance": 0.0})
    score_components_count: int = 0

    def record_retrieval(self, latency_s: float, hits: int, semantic: bool):  # pragma: no cover - simple arithmetic
        self.retrieval_calls += 1
        if hits > 0:
            self.retrieval_hits += 1
        if semantic:
            self.retrieval_semantic += 1
        self.retrieval_latency.record(latency_s)

    def record_scores(self, scores: Dict[str, float]):  # pragma: no cover
        self.score_components_count += 1
        for k, v in scores.items():
            if k in self.score_components_sum:
                self.score_components_sum[k] += v

    def record_preference_generation(self, variants: int, pairs: int, latency_s: float):  # pragma: no cover
        """Record a preference generation event."""
        self.preference_variant_batches += 1
        self.preference_variants_total += variants
        self.preference_pairs_created += pairs
        self.preference_latency.record(latency_s)

    def snapshot(self) -> Dict[str, Any]:
        avg_scores = {}
        for k, total in self.score_components_sum.items():
            avg_scores[k] = total / self.score_components_count if self.score_components_count else 0.0
        return {
            "retrieval": {
                "calls": self.retrieval_calls,
                "hit_rate": (self.retrieval_hits / self.retrieval_calls) if self.retrieval_calls else 0.0,
                "semantic_ratio": (self.retrieval_semantic / self.retrieval_calls) if self.retrieval_calls else 0.0,
                "latency": self.retrieval_latency.snapshot(),
            },
            "scoring": {
                "avg_components": avg_scores,
                "samples": self.score_components_count,
            },
            "preference": {
                "batches": self.preference_variant_batches,
                "total_variants": self.preference_variants_total,
                "pairs_created": self.preference_pairs_created,
                "avg_variants_per_batch": (self.preference_variants_total / self.preference_variant_batches) if self.preference_variant_batches else 0.0,
                "latency": self.preference_latency.snapshot(),
            }
        }


# Global singleton (imported where needed)
metrics: Optional[MetricsCollector] = MetricsCollector()

def time_block():  # pragma: no cover - utility
    start = time.perf_counter()
    def end():
        return time.perf_counter() - start
    return end
