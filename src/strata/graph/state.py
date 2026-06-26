"""Shared state for the LangGraph agent."""

from __future__ import annotations

from typing import Any, TypedDict

from ..retrieval.acl import AclContext


class AgentState(TypedDict, total=False):
    question: str  # original user question (used for generation + grading)
    acl: AclContext  # tenant + clearance, threaded through retrieval
    query: str  # current rewritten/expanded search query
    chunks: list[dict[str, Any]]
    answer: str
    faithfulness: float
    sufficient: bool
    feedback: str  # critic's note on what was missing (fed back into rewrite)
    iteration: int
    started_at: float  # perf_counter() at loop start; drives the total time budget
    elapsed_ms: float  # total wall-clock the loop took (set by run_agent)
