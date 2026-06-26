"""Offline checks for the real public corpus downloader."""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REAL_CORPUS = ROOT / "datasets" / "real_corpus"
DOWNLOADER = ROOT / "scripts" / "download_real_corpus.py"
METRICS = ROOT / "scripts" / "corpus_metrics.py"


def _load_downloader():
    spec = importlib.util.spec_from_file_location("download_real_corpus", DOWNLOADER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_real_corpus_sources_are_official_public_sources():
    rows = list(csv.DictReader((REAL_CORPUS / "sources.csv").open()))

    assert len(rows) >= 6
    official_prefixes = (
        "https://data.sec.gov/",
        "https://www.cisa.gov/",
        "https://nvlpubs.nist.gov/",
    )
    assert all(row["url"].startswith(official_prefixes) for row in rows)
    assert len(rows) < len(_load_downloader().SEC_COMPANIES) + 10


def test_html_to_text_normalizes_sec_html():
    downloader = _load_downloader()

    text = downloader._html_to_text("<html><h1>Risk Factors</h1><p>Supply chain risk.</p></html>")

    assert "Risk Factors" in text
    assert "Supply chain risk." in text


def test_render_kev_markdown_contains_operational_fields():
    downloader = _load_downloader()
    markdown = downloader._render_kev_markdown(
        {
            "catalogVersion": "test",
            "vulnerabilities": [
                {
                    "cveID": "CVE-2099-0001",
                    "vendorProject": "Example",
                    "product": "Gateway",
                    "vulnerabilityName": "Example Gateway Vulnerability",
                    "dateAdded": "2099-01-02",
                    "dueDate": "2099-01-20",
                    "knownRansomwareCampaignUse": "Known",
                    "requiredAction": "Apply updates.",
                    "shortDescription": "A gateway flaw is exploited.",
                }
            ],
        },
        limit=1,
    )

    assert "# CISA Known Exploited Vulnerabilities" in markdown
    assert "CVE-2099-0001" in markdown
    assert "Apply updates." in markdown


def test_write_cisa_kev_creates_one_document_per_entry(tmp_path, monkeypatch):
    downloader = _load_downloader()

    def fake_download_json(_url, _user_agent):
        return {
            "catalogVersion": "test",
            "vulnerabilities": [
                {
                    "cveID": "CVE-2099-0001",
                    "vendorProject": "Example",
                    "product": "Gateway",
                    "vulnerabilityName": "Example Gateway Vulnerability",
                    "dateAdded": "2099-01-02",
                    "dueDate": "2099-01-20",
                    "knownRansomwareCampaignUse": "Known",
                    "requiredAction": "Apply updates.",
                    "shortDescription": "A gateway flaw is exploited.",
                },
                {
                    "cveID": "CVE-2099-0002",
                    "vendorProject": "Example",
                    "product": "Server",
                    "vulnerabilityName": "Example Server Vulnerability",
                    "dateAdded": "2099-01-03",
                    "dueDate": "2099-01-21",
                    "knownRansomwareCampaignUse": "Unknown",
                    "requiredAction": "Apply mitigations.",
                    "shortDescription": "A server flaw is exploited.",
                },
            ],
        }

    monkeypatch.setattr(downloader, "_download_json", fake_download_json)

    rows = downloader._write_cisa_kev(tmp_path, "test-agent", limit=2)

    assert len(rows) == 3
    assert (tmp_path / "documents" / "cisa_kev_recent_index.md").exists()
    assert len(list((tmp_path / "documents" / "cisa_kev").glob("*.md"))) == 2


def test_corpus_metrics_counts_manifest_documents(tmp_path):
    spec = importlib.util.spec_from_file_location("corpus_metrics", METRICS)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "one.md").write_text("# One\n\nFirst document.", encoding="utf-8")
    (docs / "two.md").write_text("# Two\n\nSecond document.", encoding="utf-8")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "\n".join(
            [
                "path,tenant,owner,confidentiality,effective_from,effective_to,source_url,description",
                "documents/one.md,default,test,public,,,,One",
                "documents/two.md,default,test,public,,,,Two",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    metrics = module.compute_metrics(manifest, tmp_path)

    assert metrics["documents"] == 2
    assert metrics["text_documents"] == 2
    assert metrics["estimated_chunks"] >= 2
