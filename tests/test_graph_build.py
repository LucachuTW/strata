"""Offline: the agent graph wires up and compiles (no LLM/service calls)."""

from strata.graph.build import build_agent


def test_agent_compiles_with_expected_nodes():
    app = build_agent()
    nodes = set(app.get_graph().nodes)
    assert {"rewrite", "retrieve", "generate", "critic"} <= nodes
