"""Reciprocal Rank Fusion.  score(d) = Σ_m 1 / (k + rank_m(d)),  k≈60 (README)."""

from __future__ import annotations


def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1)  # rank is 1-based
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
