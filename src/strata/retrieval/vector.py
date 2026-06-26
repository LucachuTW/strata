"""Vector branch: ACL-filtered hybrid dense+sparse search in Qdrant.

Dense (BGE-M3) carries meaning; sparse (BM25) carries exact tokens — identifiers like
CVE IDs that dense embeddings blur across near-duplicate documents. Qdrant runs both
prefetches (each ACL-filtered) and fuses them server-side with RRF.
"""

from __future__ import annotations

from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client import models as qmodels

from ..config import get_settings
from ..embeddings import LocalEmbeddings
from ..sparse import embed_query as sparse_query
from .acl import AclContext


@lru_cache
def _client() -> QdrantClient:
    """Shared Qdrant client — the agent loop retrieves up to 3× per question."""
    return QdrantClient(url=get_settings().qdrant_url)


class VectorRetriever:
    def __init__(self) -> None:
        self._collection = get_settings().qdrant_collection
        self._embedder = LocalEmbeddings()

    def retrieve(self, query: str, acl: AclContext, top_k: int) -> list[dict]:
        dense = self._embedder.embed_query(query)
        s_idx, s_val = sparse_query(query)
        acl_filter = acl.qdrant_filter()
        result = _client().query_points(
            self._collection,
            prefetch=[
                qmodels.Prefetch(query=dense, using="dense", filter=acl_filter, limit=top_k),
                qmodels.Prefetch(
                    query=qmodels.SparseVector(indices=s_idx, values=s_val),
                    using="bm25",
                    filter=acl_filter,
                    limit=top_k,
                ),
            ],
            query=qmodels.FusionQuery(fusion=qmodels.Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )
        out = []
        for p in result.points:
            pl = p.payload or {}  # with_payload=True, but the client types it Optional
            out.append(
                {
                    "id": pl["neo4j_id"],
                    "text": pl["text"],
                    "heading_path": pl.get("heading_path", []),
                    "source": pl.get("source"),
                    "vector_score": p.score,
                }
            )
        return out
