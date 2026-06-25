"""Local CPU cross-encoder reranker (BGE-reranker-v2-m3)."""

from __future__ import annotations

from functools import lru_cache

from sentence_transformers import CrossEncoder

from .config import get_settings


@lru_cache
def _model() -> CrossEncoder:
    s = get_settings()
    return CrossEncoder(s.reranker_model, device=s.reranker_device)


def rerank(query: str, documents: list[str], top_n: int | None = None) -> list[tuple[int, float]]:
    """Score (query, doc) pairs; return (original_index, score) sorted desc, truncated."""
    if not documents:
        return []
    scores = _model().predict([(query, d) for d in documents])
    ranked = sorted(
        ((i, float(score)) for i, score in enumerate(scores)),
        key=lambda x: x[1],
        reverse=True,
    )
    return ranked[: top_n or len(ranked)]
