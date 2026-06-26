"""Hybrid retrieve: vector ∥ graph -> RRF fuse -> cross-encoder rerank -> top-n."""

from __future__ import annotations

from ..config import get_settings
from ..log import get_logger
from ..metrics import RuntimeMetrics
from ..rerank import rerank
from .acl import AclContext
from .graph import GraphRetriever
from .rrf import reciprocal_rank_fusion
from .vector import VectorRetriever

log = get_logger(__name__)

_GRAPH_SEEDS = 5  # expand the graph around the top-N vector hits


def retrieve(
    query: str,
    acl: AclContext,
    top_k: int | None = None,
    rerank_top_n: int | None = None,
    rerank_candidate_k: int | None = None,
) -> list[dict]:
    return retrieve_with_metrics(query, acl, top_k, rerank_top_n, rerank_candidate_k)["chunks"]


def retrieve_with_metrics(
    query: str,
    acl: AclContext,
    top_k: int | None = None,
    rerank_top_n: int | None = None,
    rerank_candidate_k: int | None = None,
) -> dict:
    s = get_settings()
    top_k = top_k or s.retriever_top_k
    rerank_top_n = rerank_top_n or s.rerank_top_n
    rerank_candidate_k = max(rerank_candidate_k or s.rerank_candidate_k, rerank_top_n)
    floor = s.rerank_score_floor
    metrics = RuntimeMetrics()
    chunks: list[dict] = []

    # One outer phase for the whole hybrid pipeline; vector/graph/rrf/rerank are its
    # children. (Earlier this phase was opened twice and silently summed only
    # vector+graph time.) as_dict() runs after the block closes so the total is recorded.
    with metrics.phase("hybrid_retrieval"):
        with metrics.phase("vector_retrieval"):
            vector_hits = VectorRetriever().retrieve(query, acl, top_k)
        seed_ids = [h["id"] for h in vector_hits[:_GRAPH_SEEDS]]
        metrics.count("vector_hits", len(vector_hits))
        metrics.count("graph_seed_chunks", len(seed_ids))

        with metrics.phase("graph_retrieval"):
            graph_hits = GraphRetriever().retrieve(seed_ids, acl, hops=2, limit=top_k)
        metrics.count("graph_hits", len(graph_hits))

        # Merge records by id (vector text wins; include graph-only chunks).
        records: dict[str, dict] = {}
        for hit in vector_hits:
            records[hit["id"]] = hit
        for hit in graph_hits:
            records.setdefault(hit["id"], hit)

        with metrics.phase("rrf"):
            fused = reciprocal_rank_fusion(
                [[h["id"] for h in vector_hits], [h["id"] for h in graph_hits]], k=s.rrf_k
            )
        fused_ids = [item for item, _ in fused]
        metrics.count("fused_candidates", len(fused_ids))

        if fused_ids:
            # Cross-encoder rerank a wider candidate pool, then return only the top N.
            # Keeping the candidate pool wider protects compound questions where one
            # relevant chunk is not in the first few fused positions but is obvious to
            # the cross-encoder once scored.
            rerank_ids = fused_ids[: max(1, rerank_candidate_k)]
            metrics.count("rerank_candidates", len(rerank_ids))
            with metrics.phase("rerank"):
                order = rerank(query, [records[i]["text"] for i in rerank_ids], top_n=rerank_top_n)
            # Relevance gate: drop chunks the cross-encoder scores below the floor so a
            # query with no relevant context returns nothing (honest refusal) instead of
            # the least-bad chunks. Off by default (floor=None) to preserve recall.
            if floor is not None:
                kept = [(idx, score) for idx, score in order if score >= floor]
                metrics.count("gated_out", len(order) - len(kept))
                order = kept
            chunks = [{**records[rerank_ids[idx]], "rerank_score": score} for idx, score in order]
        metrics.count("returned_chunks", len(chunks))

    c = metrics.counts
    log.info(
        "hybrid: vector=%d graph=%d fused=%d returned=%d (%.0f ms)",
        c.get("vector_hits", 0),
        c.get("graph_hits", 0),
        c.get("fused_candidates", 0),
        c.get("returned_chunks", 0),
        metrics.timings_ms.get("hybrid_retrieval", 0.0),
    )
    return {"chunks": chunks, "metrics": metrics.as_dict()}
