"""Evaluation harness over a golden set.

Scores each item two ways so the metric itself is auditable:
  - retrieval recall@k         — labelled relevant source present in the retrieved chunks
  - correctness (substring)    — lenient deterministic floor: expected substrings present
  - correctness (LLM-judge)    — strict: a local judge checks the answer states the required
                                 facts and asserts nothing unsupported (catches lenient passes)
  - refusal accuracy           — for `should_refuse` items (absent facts), the answer must decline
  - ACL-safety                 — a higher-clearance secret never leaks at a lower clearance

It also compares **single-pass** (retrieve+answer) against the **agent loop** (run_agent).

A real run needs the live stack (Qdrant/Neo4j/LLM); seed first:
    uv run strata seed-demo --reset
    uv run strata eval
    uv run strata eval --golden datasets/real_corpus/eval/golden_hard.jsonl
The deterministic scoring functions are pure and unit-tested offline.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import cast

from pydantic import BaseModel, Field

from .log import get_logger
from .retrieval import AclContext, retrieve
from .retrieval.answer import answer
from .schema import Confidentiality

log = get_logger(__name__)

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "datasets" / "acme_corpus" / "eval" / "golden.jsonl"


@dataclass
class GoldenItem:
    id: str
    category: str
    question: str
    clearance: str
    expect: list[str]
    relevant_sources: list[str]
    must_not_contain: list[str]
    should_refuse: bool = False  # answer must decline (fact is absent from the corpus)


def load_golden(path: Path = GOLDEN) -> list[GoldenItem]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(GoldenItem(**json.loads(line)))
    return items


# --- pure scoring (unit-tested offline) ---------------------------------------------------


def recall_at_k(retrieved_sources: list[str], relevant: list[str]) -> float | None:
    """Fraction of labelled relevant sources that appear in the retrieved set. None = N/A."""
    if not relevant:
        return None
    hits = sum(1 for r in relevant if r in retrieved_sources)
    return hits / len(relevant)


def answer_correct(answer_text: str, expect: list[str]) -> bool | None:
    """True iff every expected substring is present (case-insensitive). None = N/A."""
    if not expect:
        return None
    low = answer_text.lower()
    return all(s.lower() in low for s in expect)


def acl_safe(answer_text: str, must_not_contain: list[str]) -> bool | None:
    """True iff no forbidden (higher-clearance) substring leaked. None = N/A."""
    if not must_not_contain:
        return None
    low = answer_text.lower()
    return not any(s.lower() in low for s in must_not_contain)


_REFUSAL_MARKERS = (
    "don't have enough information",
    "do not have enough information",
    "not have enough information",
    "not enough information",
    "does not contain",
    "doesn't contain",
    "do not contain",
    "don't contain",
    "does not include",
    "doesn't include",
    "do not include",
    "does not mention",
    "doesn't mention",
    "do not mention",
    "no information",
    "not provided",
    "not specified",
    "not mentioned",
    "cannot find",
    "could not find",
    "unable to",
    "i don't know",
    "no mention",
    "not available",
)


def refused(answer_text: str) -> bool:
    """True iff the answer declines (used to grade `should_refuse` absent-fact items)."""
    low = answer_text.lower()
    return any(m in low for m in _REFUSAL_MARKERS)


class _CorrectVerdict(BaseModel):
    correct: bool = Field(
        description="true only if the answer states every required fact and asserts nothing "
        "unsupported or contradictory"
    )


_JUDGE_SYS = (
    "/no_think\n"
    "You are a strict grader for a question-answering system. Given a question, the list of "
    "facts a correct answer MUST convey, and a candidate answer, decide whether the candidate "
    "is correct. correct=true only if it conveys all required facts AND does not assert extra "
    "specific claims that go beyond them. Be strict: a vague, partial, or padded answer is not "
    "correct."
)


def judge_correct(question: str, expect: list[str], answer_text: str) -> bool | None:
    """Strict LLM-judged correctness against the required facts. None = N/A or judge failure."""
    if not expect:
        return None
    from .llm import generation_llm

    judge = generation_llm().with_structured_output(_CorrectVerdict, method="function_calling")
    prompt = (
        f"Question: {question}\n\n"
        f"Required facts: {expect}\n\n"
        f"Candidate answer:\n{answer_text}"
    )
    try:
        verdict = cast(_CorrectVerdict, judge.invoke([("system", _JUDGE_SYS), ("user", prompt)]))
        return verdict.correct
    except Exception as exc:  # noqa: BLE001 — a judge failure must not abort the eval
        log.warning("judge failed: %s", exc.__class__.__name__)
        return None


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def _pct(flags: list[bool]) -> float | None:
    return round(100.0 * sum(flags) / len(flags), 1) if flags else None


# --- runner (needs the live stack) ---------------------------------------------------------


def _run_item(item: GoldenItem) -> dict:
    from .graph import run_agent  # local import: avoids importing the agent graph offline

    acl = AclContext(tenant="default", clearance=Confidentiality(item.clearance))

    t0 = perf_counter()
    sp_chunks = retrieve(item.question, acl)
    sp_answer = answer(item.question, sp_chunks)
    sp_ms = round((perf_counter() - t0) * 1000, 1)
    sp_sources = [s for c in sp_chunks if (s := c.get("source"))]

    final = run_agent(item.question, acl)
    ag_answer = final.get("answer", "")
    ag_sources = [s for c in final.get("chunks", []) if (s := c.get("source"))]

    row = {
        "id": item.id,
        "category": item.category,
        "clearance": item.clearance,
        # raw answers kept so every score/judge verdict below is auditable in the JSON report
        "sp_answer": sp_answer,
        "ag_answer": ag_answer,
        "recall_single": recall_at_k(sp_sources, item.relevant_sources),
        "recall_agent": recall_at_k(ag_sources, item.relevant_sources),
        "acl_safe_single": acl_safe(sp_answer, item.must_not_contain),
        "acl_safe_agent": acl_safe(ag_answer, item.must_not_contain),
        "faithfulness_agent": final.get("faithfulness"),
        "iterations_agent": final.get("iteration"),
        "ms_single": sp_ms,
        "ms_agent": final.get("elapsed_ms"),
        "correct_sub_single": None,
        "correct_sub_agent": None,
        "correct_judge_single": None,
        "correct_judge_agent": None,
        "refused_single": None,
        "refused_agent": None,
    }
    if item.should_refuse:
        # absent-fact item: the only correct behaviour is to decline
        row["refused_single"] = refused(sp_answer)
        row["refused_agent"] = refused(ag_answer)
    else:
        row["correct_sub_single"] = answer_correct(sp_answer, item.expect)
        row["correct_sub_agent"] = answer_correct(ag_answer, item.expect)
        row["correct_judge_single"] = judge_correct(item.question, item.expect, sp_answer)
        row["correct_judge_agent"] = judge_correct(item.question, item.expect, ag_answer)
    return row


def evaluate(limit: int | None = None, golden: Path = GOLDEN) -> dict:
    golden = Path(golden)
    items = load_golden(golden)[:limit]
    rows = []
    for i, item in enumerate(items, start=1):
        log.info("eval %d/%d: %s", i, len(items), item.id)
        rows.append(_run_item(item))
    try:  # keep the report repo-relative, never an absolute home path
        golden_display = str(golden.resolve().relative_to(ROOT))
    except ValueError:
        golden_display = str(golden)
    return {"golden": golden_display, "count": len(rows), "summary": _aggregate(rows), "rows": rows}


def _aggregate(rows: list[dict]) -> dict:
    def collect(key, keep=lambda v: v is not None):
        return [r[key] for r in rows if keep(r[key])]

    return {
        "retrieval_recall_at_k": _mean(collect("recall_single")),
        "correctness_substring_pct_single": _pct(collect("correct_sub_single")),
        "correctness_substring_pct_agent": _pct(collect("correct_sub_agent")),
        "correctness_judge_pct_single": _pct(collect("correct_judge_single")),
        "correctness_judge_pct_agent": _pct(collect("correct_judge_agent")),
        "refusal_accuracy_pct_single": _pct(collect("refused_single")),
        "refusal_accuracy_pct_agent": _pct(collect("refused_agent")),
        "acl_safety_pct_single": _pct(collect("acl_safe_single")),
        "acl_safety_pct_agent": _pct(collect("acl_safe_agent")),
        "mean_faithfulness_agent": _mean(collect("faithfulness_agent")),
        "mean_iterations_agent": _mean([float(v) for v in collect("iterations_agent")]),
        "mean_ms_single": _mean(collect("ms_single")),
        "mean_ms_agent": _mean(collect("ms_agent")),
    }


# --- reporting -----------------------------------------------------------------------------


def _pct_str(value: float | None) -> str:
    return "n/a" if value is None else f"{value}%"


def write_reports(results: dict, json_path: Path, md_path: Path) -> None:
    json_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    s = results["summary"]
    acl_single, acl_agent = s["acl_safety_pct_single"], s["acl_safety_pct_agent"]
    if acl_single is None and acl_agent is None:
        acl_line = "- ACL-safety: n/a — this corpus has no confidentiality tiers"
    else:
        acl_line = (
            f"- ACL-safety (no leak below clearance) — single-pass: "
            f"**{_pct_str(acl_single)}** | agent: **{_pct_str(acl_agent)}** (target 100%)"
        )
    refusal_single = s["refusal_accuracy_pct_single"]
    refusal_line = (
        f"- Refusal accuracy (absent facts) — single-pass: "
        f"**{_pct_str(refusal_single)}** | agent: **{_pct_str(s['refusal_accuracy_pct_agent'])}**"
        if refusal_single is not None
        else None
    )
    lines = [
        "# Strata — Evaluation Metrics",
        "",
        f"Golden set: `{results.get('golden', '?')}` — {results['count']} items.",
        "",
        "## Headline",
        "",
        f"- Retrieval recall@k: **{s['retrieval_recall_at_k']}**",
        f"- Correctness, substring (lenient) — single-pass: "
        f"**{_pct_str(s['correctness_substring_pct_single'])}** | "
        f"agent: **{_pct_str(s['correctness_substring_pct_agent'])}**",
        f"- Correctness, LLM-judge (strict) — single-pass: "
        f"**{_pct_str(s['correctness_judge_pct_single'])}** | "
        f"agent: **{_pct_str(s['correctness_judge_pct_agent'])}**",
        *([refusal_line] if refusal_line else []),
        acl_line,
        "",
        "## Agent loop vs single-pass",
        "",
        f"- Mean critic faithfulness (agent): {s['mean_faithfulness_agent']}",
        f"- Mean iterations (agent): {s['mean_iterations_agent']}",
        f"- Mean latency: single-pass {s['mean_ms_single']} ms | agent {s['mean_ms_agent']} ms",
        "",
        "## Per-item",
        "",
        "| id | category | recall | sub (sp/ag) | judge (sp/ag) | refuse (sp/ag) | faith | iters |",
        "|----|----------|--------|-------------|---------------|----------------|-------|-------|",
    ]
    for r in results["rows"]:
        lines.append(
            f"| {r['id']} | {r['category']} | {r['recall_single']} | "
            f"{r['correct_sub_single']}/{r['correct_sub_agent']} | "
            f"{r['correct_judge_single']}/{r['correct_judge_agent']} | "
            f"{r['refused_single']}/{r['refused_agent']} | "
            f"{r['faithfulness_agent']} | {r['iterations_agent']} |"
        )
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Strata golden-set evaluation.")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N items")
    parser.add_argument(
        "--golden", default=None, help="Path to a golden JSONL (default: synthetic Acme set)"
    )
    parser.add_argument("--json-output", default="eval_metrics.json")
    parser.add_argument("--markdown-output", default="EVAL_METRICS.md")
    args = parser.parse_args(argv)

    golden = Path(args.golden) if args.golden else GOLDEN
    results = evaluate(limit=args.limit, golden=golden)
    write_reports(results, Path(args.json_output), Path(args.markdown_output))
    print(json.dumps({"count": results["count"], **results["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
