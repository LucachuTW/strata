"""Agentic GraphRAG loop built on LangGraph."""

from __future__ import annotations

from functools import lru_cache
from time import perf_counter

from ..retrieval.acl import AclContext
from .build import build_agent
from .state import AgentState


@lru_cache
def _agent():
    return build_agent()


def run_agent(question: str, acl: AclContext) -> AgentState:
    final = _agent().invoke({"question": question, "acl": acl, "iteration": 0})
    if final.get("started_at"):
        final["elapsed_ms"] = round((perf_counter() - final["started_at"]) * 1000, 1)
    return final


__all__ = ["build_agent", "run_agent", "AgentState"]
