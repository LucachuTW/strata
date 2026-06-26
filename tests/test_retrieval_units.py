"""Offline units for RRF fusion, ACL logic, and hybrid candidate handling."""

from strata.retrieval import hybrid
from strata.retrieval.acl import AclContext
from strata.retrieval.rrf import reciprocal_rank_fusion
from strata.schema import Confidentiality


def test_rrf_combines_and_orders():
    a = ["x", "y", "z"]
    b = ["y", "x", "w"]
    fused = dict(reciprocal_rank_fusion([a, b], k=60))
    assert set(fused) == {"x", "y", "z", "w"}
    # items appearing high in both lists beat items appearing once, low.
    assert fused["x"] > fused["z"]
    assert fused["y"] > fused["w"]


def test_acl_clearance_is_a_ceiling():
    acl = AclContext(tenant="t", clearance=Confidentiality.internal)
    allowed = acl.allowed_confidentialities()
    assert "public" in allowed and "internal" in allowed
    assert "confidential" not in allowed and "restricted" not in allowed


def test_acl_qdrant_filter_shape():
    f = AclContext(tenant="acme", clearance=Confidentiality.restricted).qdrant_filter()
    keys = {c.key for c in f.must}
    assert keys == {"tenant", "confidentiality"}


def test_acl_cypher_where_clause_and_params():
    clause, params = AclContext(
        tenant="acme", clearance=Confidentiality.internal
    ).cypher_where("other")
    assert clause == "other.tenant = $acl_tenant AND other.confidentiality IN $acl_conf"
    assert params == {"acl_tenant": "acme", "acl_conf": ["public", "internal"]}


def test_hybrid_retrieve_with_metrics(monkeypatch):
    class FakeVectorRetriever:
        def retrieve(self, _query, _acl, _top_k):
            return [
                {"id": "a", "text": "alpha", "source": "one.md"},
                {"id": "b", "text": "beta", "source": "two.md"},
            ]

    class FakeGraphRetriever:
        def retrieve(self, _seed_ids, _acl, hops=2, limit=20):
            return [{"id": "c", "text": "gamma", "source": "three.md"}]

        def close(self):
            pass

    monkeypatch.setattr(hybrid, "VectorRetriever", FakeVectorRetriever)
    monkeypatch.setattr(hybrid, "GraphRetriever", FakeGraphRetriever)

    def fake_rerank(_query, documents, top_n=None):
        return [(i, 1.0 - i / 10) for i, _document in enumerate(documents)][:top_n]

    monkeypatch.setattr(hybrid, "rerank", fake_rerank)

    result = hybrid.retrieve_with_metrics("question", AclContext(), top_k=3, rerank_top_n=2)

    assert len(result["chunks"]) == 2
    assert result["metrics"]["counts"]["vector_hits"] == 2
    assert result["metrics"]["counts"]["graph_hits"] == 1
    assert result["metrics"]["counts"]["fused_candidates"] == 3
    assert result["metrics"]["counts"]["rerank_candidates"] == 3
    assert result["metrics"]["counts"]["returned_chunks"] == 2
    assert "hybrid_retrieval" in result["metrics"]["timings_ms"]
    assert "rerank" in result["metrics"]["budget_status"]


def test_hybrid_reranks_wider_candidate_pool_for_compound_questions(monkeypatch):
    """Regression: the risk chunk must not be cut before cross-encoder scoring."""

    class FakeVectorRetriever:
        def retrieve(self, _query, _acl, _top_k):
            return [
                {"id": "subsidiaries", "text": "Jane Smith is CFO of Acme Robotics."},
                {"id": "products", "text": "Acme offers industrial automation systems."},
                {"id": "overview", "text": "Acme Corporation is a manufacturer."},
                {"id": "retention", "text": "Financial records are retained for seven years."},
                {"id": "risk", "text": "Acme faces supply chain risk and currency risk."},
            ]

    class FakeGraphRetriever:
        def retrieve(self, _seed_ids, _acl, hops=2, limit=20):
            return []

        def close(self):
            pass

    def fake_rerank(_query, documents, top_n=None):
        assert any("supply chain risk" in doc for doc in documents)
        scores = []
        for i, doc in enumerate(documents):
            score = 10.0 if "supply chain risk" in doc else 1.0 - (i / 10)
            scores.append((i, score))
        return sorted(scores, key=lambda item: item[1], reverse=True)[:top_n]

    monkeypatch.setattr(hybrid, "VectorRetriever", FakeVectorRetriever)
    monkeypatch.setattr(hybrid, "GraphRetriever", FakeGraphRetriever)
    monkeypatch.setattr(hybrid, "rerank", fake_rerank)

    result = hybrid.retrieve_with_metrics(
        "Who is the CFO and which risks affect Acme?",
        AclContext(),
        top_k=5,
        rerank_top_n=2,
        rerank_candidate_k=5,
    )

    assert result["chunks"][0]["id"] == "risk"
    assert result["metrics"]["counts"]["rerank_candidates"] == 5
    assert result["metrics"]["counts"]["returned_chunks"] == 2


def test_rerank_score_floor_gates_irrelevant_chunks(monkeypatch):
    """With a floor set, sub-threshold chunks are dropped -> honest empty result."""

    class FakeVectorRetriever:
        def retrieve(self, _query, _acl, _top_k):
            return [{"id": "a", "text": "unrelated"}, {"id": "b", "text": "also unrelated"}]

    class FakeGraphRetriever:
        def retrieve(self, _seed_ids, _acl, hops=2, limit=20):
            return []

    monkeypatch.setattr(hybrid, "VectorRetriever", FakeVectorRetriever)
    monkeypatch.setattr(hybrid, "GraphRetriever", FakeGraphRetriever)
    monkeypatch.setattr(hybrid, "rerank", lambda _q, docs, top_n=None: [(0, -3.0), (1, -4.0)])
    monkeypatch.setattr(hybrid.get_settings(), "rerank_score_floor", 0.0)

    result = hybrid.retrieve_with_metrics("anything", AclContext(), top_k=2, rerank_top_n=2)

    assert result["chunks"] == []
    assert result["metrics"]["counts"]["gated_out"] == 2
    assert result["metrics"]["counts"]["returned_chunks"] == 0
