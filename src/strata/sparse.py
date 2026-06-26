"""Local BM25 sparse embeddings (fastembed, CPU) for lexical exact-token matching.

Dense retrieval (BGE-M3) matches meaning but misses exact identifiers — among 150
near-identical CVE records it cannot surface "CVE-2026-35273" reliably. BM25 sparse
vectors match the rare token directly; Qdrant fuses dense + sparse with RRF. No torch.
"""

from __future__ import annotations

from functools import lru_cache

from fastembed import SparseTextEmbedding

from .config import get_settings

SparseVec = tuple[list[int], list[float]]  # (indices, values)


@lru_cache
def _model() -> SparseTextEmbedding:
    return SparseTextEmbedding(model_name=get_settings().sparse_model)


def embed_documents(texts: list[str]) -> list[SparseVec]:
    return [(e.indices.tolist(), e.values.tolist()) for e in _model().embed(texts)]


def embed_query(text: str) -> SparseVec:
    e = next(iter(_model().query_embed(text)))
    return e.indices.tolist(), e.values.tolist()
