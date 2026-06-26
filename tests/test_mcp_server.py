"""Offline tests for the MCP tool boundary."""

from __future__ import annotations

import pytest

from strata import mcp_server
from strata.retrieval import AclContext
from strata.schema import Confidentiality


def test_search_corpus_applies_acl_and_shapes_results(monkeypatch):
    seen = {}

    def fake_retrieve(question, acl):
        seen["question"] = question
        seen["acl"] = acl
        return [
            {
                "source": "policy.md",
                "heading_path": ["Policy"],
                "text": "Only finance can see this.",
                "rerank_score": 0.91,
                "internal_field": "not exported",
            }
        ]

    monkeypatch.setattr(mcp_server, "retrieve", fake_retrieve)

    result = mcp_server.search_corpus(
        "who can approve spend?", tenant="acme", clearance="confidential"
    )

    assert seen["question"] == "who can approve spend?"
    assert seen["acl"].tenant == "acme"
    assert seen["acl"].clearance is Confidentiality.confidential
    assert result == [
        {
            "source": "policy.md",
            "heading_path": ["Policy"],
            "text": "Only finance can see this.",
            "score": 0.91,
        }
    ]


def test_acl_cypher_injects_chunk_acl_and_limit():
    query, params = mcp_server._acl_cypher(
        "MATCH (c:Chunk)-[:MENTIONS]->(e:Entity) RETURN e.name",
        AclContext(tenant="acme", clearance=Confidentiality.internal),
        limit=25,
    )

    assert "c.tenant = $_acl_tenant" in query
    assert "c.confidentiality IN $_acl_conf" in query
    assert "LIMIT $_acl_limit" in query
    assert params == {
        "_acl_tenant": "acme",
        "_acl_conf": ["public", "internal"],
        "_acl_limit": 25,
    }


def test_acl_cypher_preserves_existing_where():
    query, _params = mcp_server._acl_cypher(
        "MATCH (c:Chunk)-[:MENTIONS]->(e:Entity) WHERE e.type = 'Person' RETURN e.name",
        AclContext(),
        limit=10,
    )

    assert "WHERE (c.tenant = $_acl_tenant AND c.confidentiality IN $_acl_conf) AND e.type" in query


@pytest.mark.parametrize(
    "query",
    [
        "MATCH (e:Entity) RETURN e",
        "MATCH (c:Chunk)-[:MENTIONS]->(e:Entity), (leak:Entity) RETURN leak",
        "MATCH (c:Chunk)-[:MENTIONS]->(e:Entity) WITH e MATCH (leak:Entity) RETURN leak",
        "MATCH (c:Chunk) SET c.text = 'changed' RETURN c",
        "MATCH (c:Chunk) RETURN c; MATCH (n) RETURN n",
    ],
)
def test_acl_cypher_rejects_unsafe_queries(query):
    with pytest.raises(ValueError):
        mcp_server._acl_cypher(query, AclContext(), limit=10)


def test_search_corpus_rejects_invalid_clearance():
    with pytest.raises(ValueError):
        mcp_server.search_corpus("anything", clearance="secret")
