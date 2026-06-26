"""LangGraph nodes: rewrite (planner) -> retrieve -> generate -> critic, with a retry loop.

Latency budgets (README): the loop is bounded two ways — an iteration cap
(`max_iterations`) and a total wall-clock budget (`agent_time_budget_s`). On budget
overrun `route_after_critic` stops and returns the best answer so far, and `rewrite`
applies the README's "skip rewrite" mitigation (reuse the question, skip the LLM call).
Finer per-phase cancellation is intentionally out of scope for a local single-user MVP.
"""

from __future__ import annotations

from time import perf_counter
from typing import cast

from pydantic import BaseModel, Field

from ..config import get_settings
from ..llm import generation_llm, rewrite_llm
from ..log import get_logger
from ..retrieval import retrieve
from ..retrieval.answer import answer, build_context
from .state import AgentState

log = get_logger(__name__)

_REWRITE_SYS = (
    "/no_think\n"
    "You turn a user's question into a single, information-dense search query for hybrid "
    "vector+graph retrieval over corporate documents. Expand key entities and intent. If "
    "feedback about previously missing information is provided, fold it in. Return ONLY the query."
)


def _elapsed_s(state: AgentState) -> float:
    started = state.get("started_at")
    return perf_counter() - started if started else 0.0


def _over_budget(state: AgentState) -> bool:
    budget = get_settings().agent_time_budget_s
    return budget > 0 and _elapsed_s(state) >= budget


def rewrite(state: AgentState) -> dict:
    question = state["question"]
    started_at = state.get("started_at") or perf_counter()
    out: dict = {"started_at": started_at, "iteration": state.get("iteration", 0) + 1}
    # Mitigation: if we're already over the time budget on a retry, skip the LLM
    # rewrite and reuse the current query (or the raw question on the first pass).
    if _over_budget({**state, "started_at": started_at}):
        out["query"] = state.get("query") or question
        return out
    feedback = state.get("feedback")
    user = f"Question: {question}"
    if feedback:
        user += f"\nThe previous attempt was missing: {feedback}"
    reply = rewrite_llm().invoke([("system", _REWRITE_SYS), ("user", user)])
    out["query"] = (cast(str, reply.content) or "").strip() or question
    log.debug("rewrite[iter %d]: %r", out["iteration"], out["query"])
    return out


def retrieve_node(state: AgentState) -> dict:
    chunks = retrieve(state["query"], state["acl"])
    log.info("retrieved %d chunks (clearance=%s)", len(chunks), state["acl"].clearance.value)
    return {"chunks": chunks}


def generate(state: AgentState) -> dict:
    return {"answer": answer(state["question"], state.get("chunks", []))}


class _Verdict(BaseModel):
    faithfulness: float = Field(
        ge=0.0,
        le=1.0,
        description="fraction of the answer's claims supported by the context",
    )
    sufficient: bool = Field(
        description="true only if every requested part of the question is answered from the context"
    )
    missing: str = Field(
        default="",
        description="what information is missing or unsupported, if any",
    )


_CRITIC_SYS = (
    "/no_think\n"
    "You are a strict grader. Given a question, the retrieved context, and a candidate answer, "
    "judge (1) faithfulness, the share of the answer supported by the context, and "
    "(2) whether it sufficiently answers the question. For multi-part questions, "
    "sufficient is false if any part is unanswered. If the answer says the context "
    "is missing information for a requested part, sufficient is false. If not "
    "sufficient, state briefly what is missing."
)
_MISSING_INFO_MARKERS = (
    "does not contain information",
    "doesn't contain information",
    "do not have enough information",
    "don't have enough information",
    "not have enough information",
    "not enough information",
    "provided context does not contain",
    "provided context doesn't contain",
)


def critic(state: AgentState) -> dict:
    chunks = state.get("chunks", [])
    if not chunks:
        return {"faithfulness": 0.0, "sufficient": False, "feedback": "no documents were retrieved"}
    answer_text = state.get("answer", "").lower()
    if any(marker in answer_text for marker in _MISSING_INFO_MARKERS):
        return {
            "faithfulness": 1.0,
            "sufficient": False,
            "feedback": "the answer says a requested part is missing from context",
        }
    judge = generation_llm().with_structured_output(_Verdict, method="function_calling")
    prompt = (
        f"Question: {state['question']}\n\n"
        f"Context:\n{build_context(chunks)}\n\n"
        f"Answer:\n{state.get('answer', '')}"
    )
    try:
        v = cast(_Verdict, judge.invoke([("system", _CRITIC_SYS), ("user", prompt)]))
        log.info("critic: faithfulness=%.2f sufficient=%s", v.faithfulness, v.sufficient)
        return {"faithfulness": v.faithfulness, "sufficient": v.sufficient, "feedback": v.missing}
    except Exception as exc:  # noqa: BLE001 — keep ACL-safe retrieval, but do not bless weak answers.
        return {
            "faithfulness": 0.0,
            "sufficient": False,
            "feedback": f"critic failed: {exc.__class__.__name__}",
        }


def route_after_critic(state: AgentState) -> str:
    s = get_settings()
    decision = "retry"
    if not state.get("chunks"):
        decision = "end"  # nothing accessible (e.g. ACL) — don't loop
    elif state.get("iteration", 0) >= s.max_iterations:
        decision = "end"
    elif _over_budget(state):
        decision = "end"  # time budget spent — return best answer so far
    elif state.get("faithfulness", 0.0) >= s.faithfulness_threshold and state.get(
        "sufficient", False
    ):
        decision = "end"
    log.debug("route[iter %s]: %s", state.get("iteration"), decision)
    return decision
