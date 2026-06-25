"""Local CPU embeddings (BGE-M3), exposed via the neo4j-graphrag Embedder interface."""

from __future__ import annotations

from functools import lru_cache

from neo4j_graphrag.embeddings.base import Embedder
from sentence_transformers import SentenceTransformer

from .config import get_settings


@lru_cache
def _model() -> SentenceTransformer:
    s = get_settings()
    return SentenceTransformer(s.embedding_model, device=s.embedding_device)


class LocalEmbeddings(Embedder):
    """BGE-M3 on CPU. Normalized vectors so cosine == dot product in Qdrant."""

    def embed_query(self, text: str) -> list[float]:
        return _model().encode(text, normalize_embeddings=True).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return _model().encode(texts, normalize_embeddings=True, batch_size=16).tolist()
