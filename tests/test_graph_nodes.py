"""Offline checks for agent node guardrails."""

from time import perf_counter

import pytest

from strata.graph.nodes import critic, route_after_critic
from strata.retrieval import AclContext

_NOW = perf_counter()
_LONG_AGO = _NOW - 10_000  # well past any positive time budget

_CHUNK = [{"text": "x", "source": "s.md"}]


@pytest.mark.parametrize(
    "state, expected",
    [
        # nothing accessible (e.g. ACL filtered everything) — never loop.
        ({"chunks": [], "iteration": 1, "started_at": _NOW}, "end"),
        # iteration cap reached.
        ({"chunks": _CHUNK, "iteration": 3, "started_at": _NOW}, "end"),
        # good answer: faithful AND sufficient.
        (
            {
                "chunks": _CHUNK,
                "iteration": 1,
                "started_at": _NOW,
                "faithfulness": 1.0,
                "sufficient": True,
            },
            "end",
        ),
        # weak answer, budget + iterations left — keep trying.
        (
            {
                "chunks": _CHUNK,
                "iteration": 1,
                "started_at": _NOW,
                "faithfulness": 0.0,
                "sufficient": False,
            },
            "retry",
        ),
        # same weak answer, but the wall-clock budget is spent — stop with best-so-far.
        (
            {
                "chunks": _CHUNK,
                "iteration": 1,
                "started_at": _LONG_AGO,
                "faithfulness": 0.0,
                "sufficient": False,
            },
            "end",
        ),
    ],
)
def test_route_after_critic_branches(state, expected):
    assert route_after_critic({"acl": AclContext(), **state}) == expected


def test_critic_marks_missing_partial_answer_as_insufficient():
    result = critic(
        {
            "question": "Who is the CFO and which risks affect Acme?",
            "query": "Acme CFO risks",
            "acl": AclContext(),
            "chunks": [{"text": "Jane Smith is CFO.", "source": "policy.md"}],
            "answer": (
                "Jane Smith is CFO. The provided context does not contain information about risks."
            ),
            "iteration": 1,
        }
    )

    assert result["faithfulness"] == 1.0
    assert result["sufficient"] is False
    assert "missing" in result["feedback"]
