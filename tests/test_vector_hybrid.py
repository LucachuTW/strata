"""Offline check: the vector branch issues a dense+sparse RRF-fused, ACL-filtered query."""

from __future__ import annotations

from types import SimpleNamespace

from qdrant_client import models as qmodels

from strata.retrieval import vector
from strata.retrieval.acl import AclContext
from strata.schema import Confidentiality


def test_vector_retriever_fuses_dense_and_sparse(monkeypatch):
    captured = {}

    class FakeClient:
        def query_points(self, collection, **kw):
            captured["collection"] = collection
            captured.update(kw)
            return SimpleNamespace(points=[])

    monkeypatch.setattr(vector, "_client", lambda: FakeClient())
    monkeypatch.setattr(vector, "sparse_query", lambda q: ([7, 42], [0.5, 0.9]))

    vr = vector.VectorRetriever()
    monkeypatch.setattr(vr._embedder, "embed_query", lambda q: [0.0] * 4)

    vr.retrieve("CVE-2026-35273", AclContext(clearance=Confidentiality.public), top_k=10)

    prefetch = captured["prefetch"]
    assert {p.using for p in prefetch} == {"dense", "bm25"}
    # both branches are ACL-filtered, and results are RRF-fused
    assert all(p.filter is not None for p in prefetch)
    assert captured["query"].fusion == qmodels.Fusion.RRF
