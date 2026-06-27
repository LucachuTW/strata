"""Offline tests for the evaluation scoring + golden-set integrity (no stack needed)."""

from __future__ import annotations

import json

from strata import evaluation
from strata.evaluation import (
    acl_safe,
    answer_correct,
    load_golden,
    recall_at_k,
    refused,
    write_reports,
)
from strata.schema import Confidentiality

SRC = "datasets/acme_corpus/documents/confidential_financial_report_2025.md"


def test_recall_at_k():
    assert recall_at_k([SRC, "other.md"], [SRC]) == 1.0
    assert recall_at_k(["other.md"], [SRC]) == 0.0
    assert recall_at_k(["a", "b"], ["a", "c"]) == 0.5
    assert recall_at_k([], []) is None  # N/A (e.g. ACL-denial item)


def test_answer_correct_is_case_insensitive_and_requires_all():
    assert answer_correct("Revenue was 420 million USD.", ["420"]) is True
    assert answer_correct("The CFO is Jane Smith.", ["jane smith"]) is True
    assert answer_correct("Only Jane Smith.", ["Jane Smith", "supply chain"]) is False
    assert answer_correct("anything", []) is None


def test_acl_safe_detects_leak():
    assert acl_safe("I don't have access to that.", ["140"]) is True
    assert acl_safe("The target is 140 million USD.", ["140"]) is False
    assert acl_safe("anything", []) is None


def test_refused_detects_declines():
    assert refused("The provided context does not contain a CVSS score.") is True
    assert refused("I don't have enough information to answer that.") is True
    # phrasings the models actually used that the first marker list missed (found via audit)
    assert refused("The provided context does not include the CVSS base score.") is True
    assert refused("The context does not mention any vulnerabilities related to Tesla.") is True
    assert refused("The product is TeamCity.") is False


def test_write_reports_is_corpus_agnostic_and_audits_answers(tmp_path):
    # A no-ACL corpus (real KEV) with a refusal item and a missing judge number — exercises the
    # n/a rendering and the corpus-agnostic ACL line, and checks the answers land in the JSON.
    summary = {
        "retrieval_recall_at_k": 1.0,
        "correctness_substring_pct_single": 100.0,
        "correctness_substring_pct_agent": 100.0,
        "correctness_judge_pct_single": None,  # judge unavailable -> must render n/a
        "correctness_judge_pct_agent": 57.0,
        "refusal_accuracy_pct_single": 67.0,
        "refusal_accuracy_pct_agent": 100.0,
        "acl_safety_pct_single": None,  # this corpus has no confidentiality tiers
        "acl_safety_pct_agent": None,
        "mean_faithfulness_agent": 0.9,
        "mean_iterations_agent": 1.5,
        "mean_ms_single": 1000.0,
        "mean_ms_agent": 22000.0,
    }
    rows = [
        {
            "id": "kev-1", "category": "factual", "recall_single": 1.0,
            "sp_answer": "The product is TeamCity.", "ag_answer": "TeamCity is affected.",
            "correct_sub_single": True, "correct_sub_agent": True,
            "correct_judge_single": True, "correct_judge_agent": False,
            "refused_single": None, "refused_agent": None,
            "faithfulness_agent": 1.0, "iterations_agent": 1,
        },
        {
            "id": "kev-refuse", "category": "refusal", "recall_single": None,
            "sp_answer": "The CVSS score is 9.8.", "ag_answer": "The context does not contain it.",
            "correct_sub_single": None, "correct_sub_agent": None,
            "correct_judge_single": None, "correct_judge_agent": None,
            "refused_single": False, "refused_agent": True,
            "faithfulness_agent": 0.8, "iterations_agent": 3,
        },
    ]
    results = {
        "golden": "datasets/real_corpus/eval/golden_hard.jsonl",
        "count": 2, "summary": summary, "rows": rows,
    }
    json_path, md_path = tmp_path / "m.json", tmp_path / "M.md"
    write_reports(results, json_path, md_path)

    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved["rows"][0]["sp_answer"] == "The product is TeamCity."  # answers are auditable
    assert saved["rows"][1]["ag_answer"] == "The context does not contain it."

    md = md_path.read_text(encoding="utf-8")
    assert "this corpus has no confidentiality tiers" in md  # corpus-agnostic ACL line
    assert "n/a" in md  # the missing judge number renders as n/a, not None% or a crash
    assert "Refusal accuracy" in md and "67.0%" in md
    assert "kev-1" in md and "kev-refuse" in md  # per-item table rendered


def test_golden_set_is_well_formed():
    items = load_golden()
    assert len(items) >= 18
    clearances = {c.value for c in Confidentiality}
    denial = 0
    for it in items:
        assert it.question and it.clearance in clearances
        if it.category == "acl-denial":
            denial += 1
            assert it.must_not_contain and not it.expect and not it.relevant_sources
        else:
            assert it.expect and it.relevant_sources
            for src in it.relevant_sources:
                assert (evaluation.ROOT / src).exists(), f"missing corpus doc: {src}"
    assert denial >= 3  # the ACL boundary must be exercised across clearances
