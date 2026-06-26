"""Offline checks for the bundled synthetic sample corpus."""

from __future__ import annotations

import csv
from pathlib import Path

from strata.schema import Confidentiality

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "datasets" / "acme_corpus"


def test_sample_corpus_manifest_points_to_existing_markdown_files():
    rows = list(csv.DictReader((CORPUS / "manifest.csv").open()))

    assert len(rows) >= 6
    for row in rows:
        path = CORPUS / row["path"]
        assert path.exists(), row["path"]
        assert path.suffix == ".md"
        assert row["confidentiality"] in {c.value for c in Confidentiality}
        assert row["tenant"]
        assert row["owner"]


def test_sample_corpus_exercises_all_clearance_levels():
    rows = list(csv.DictReader((CORPUS / "manifest.csv").open()))
    levels = {row["confidentiality"] for row in rows}

    assert levels == {c.value for c in Confidentiality}


def test_sample_corpus_contains_multi_hop_entities():
    text = "\n".join(path.read_text() for path in (CORPUS / "documents").glob("*.md"))

    for expected in [
        "Acme Corporation",
        "Acme Robotics",
        "Jane Smith",
        "Information Governance Policy",
        "SafeGuard compliance platform",
        "supply chain risk",
        "fiscal year 2025",
    ]:
        assert expected in text
