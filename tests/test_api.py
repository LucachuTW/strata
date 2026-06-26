"""Offline tests for the FastAPI interface."""

from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from strata import api


def test_health():
    client = TestClient(api.app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_streams_sources_and_tokens(monkeypatch):
    def fake_retrieve(question, acl):
        assert question == "Who approves spend?"
        assert acl.tenant == "acme"
        return [
            {
                "source": "policy.md",
                "heading_path": ["Policy"],
                "text": "The CFO approves spend.",
            }
        ]

    class FakeLlm:
        def stream(self, prompt):
            assert "The CFO approves spend." in prompt[1][1]
            yield SimpleNamespace(content="The CFO")
            yield SimpleNamespace(content=" approves spend.")

    monkeypatch.setattr(api, "retrieve", fake_retrieve)
    monkeypatch.setattr(api, "generation_llm", lambda: FakeLlm())

    client = TestClient(api.app)
    response = client.post(
        "/query",
        json={"question": "Who approves spend?", "tenant": "acme", "clearance": "internal"},
    )

    assert response.status_code == 200
    body = response.text
    assert "event: sources" in body
    assert '"source": "policy.md"' in body
    assert "event: token" in body
    assert "The CFO" in body
    assert "event: done" in body


def test_query_empty_accessible_context_does_not_call_llm(monkeypatch):
    monkeypatch.setattr(api, "retrieve", lambda _question, _acl: [])

    def fail_llm():
        raise AssertionError("LLM should not be called when retrieval is empty")

    monkeypatch.setattr(api, "generation_llm", fail_llm)

    client = TestClient(api.app)
    response = client.post("/query", json={"question": "anything"})

    assert response.status_code == 200
    assert "No accessible documents answer this question." in response.text
    assert "event: done" in response.text


def test_query_retrieval_failure_is_503(monkeypatch):
    def fail_retrieve(_question, _acl):
        raise RuntimeError("qdrant unavailable")

    monkeypatch.setattr(api, "retrieve", fail_retrieve)

    client = TestClient(api.app)
    response = client.post("/query", json={"question": "anything"})

    assert response.status_code == 503
    assert response.json()["detail"] == "Retrieval backend unavailable."


def test_ask_response_shape(monkeypatch):
    def fake_run_agent(question, acl):
        assert question == "What changed?"
        assert acl.tenant == "acme"
        return {
            "answer": "The policy changed.",
            "iteration": 1,
            "elapsed_ms": 1234.5,
            "faithfulness": 0.95,
            "sufficient": True,
            "chunks": [{"source": "policy.md", "heading_path": ["Changes"], "text": "..."}],
        }

    monkeypatch.setattr(api, "run_agent", fake_run_agent)

    client = TestClient(api.app)
    response = client.post(
        "/ask",
        json={"question": "What changed?", "tenant": "acme", "clearance": "restricted"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "The policy changed.",
        "iterations": 1,
        "elapsed_ms": 1234.5,
        "faithfulness": 0.95,
        "sufficient": True,
        "sources": [{"source": "policy.md", "heading_path": ["Changes"]}],
    }


def test_invalid_clearance_is_422():
    client = TestClient(api.app)
    response = client.post("/query", json={"question": "anything", "clearance": "secret"})

    assert response.status_code == 422
