"""Assemble the LangGraph state machine."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .nodes import critic, generate, retrieve_node, rewrite, route_after_critic
from .state import AgentState


def build_agent():
    g = StateGraph(AgentState)
    g.add_node("rewrite", rewrite)
    g.add_node("retrieve", retrieve_node)
    g.add_node("generate", generate)
    g.add_node("critic", critic)

    g.add_edge(START, "rewrite")
    g.add_edge("rewrite", "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", "critic")
    g.add_conditional_edges("critic", route_after_critic, {"retry": "rewrite", "end": END})

    return g.compile()
