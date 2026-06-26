"""Offline tests for the simplified CLI surface."""

from __future__ import annotations

from pathlib import Path

import pytest

from strata import cli


def test_cli_has_friendly_commands():
    parser = cli.build_parser()

    for command in [
        "doctor",
        "examples",
        "download-real",
        "ingest-corpus",
        "demo",
        "reset-stores",
        "seed-demo",
        "metrics",
        "eval",
        "serve",
        "ask",
    ]:
        with pytest.raises(SystemExit) as exc:
            parser.parse_args([command, "--help"])
        assert exc.value.code == 0


def test_examples_print_ready_to_run_queries(capsys):
    assert cli.main(["examples"]) == 0

    out = capsys.readouterr().out
    assert "strata ask" in out
    assert "CISA KEV" in out
    assert "--clearance" in out


def test_metrics_dispatches_expected_scripts(monkeypatch):
    calls = []

    def fake_run_script(script, args):
        calls.append((script, args))
        return 0

    monkeypatch.setattr(cli, "_run_script", fake_run_script)

    assert cli.main(["metrics", "all"]) == 0

    assert [script for script, _args in calls] == [
        "corpus_metrics.py",
        "project_metrics.py",
        "benchmark_retrieval.py",
    ]


def test_ingest_corpus_uses_manifest(monkeypatch):
    seen = {}

    def fake_ingest_manifest(manifest, base_dir, include_pdf=False, limit=None):
        seen["manifest"] = manifest
        seen["base_dir"] = base_dir
        seen["include_pdf"] = include_pdf
        seen["limit"] = limit
        return {"manifest_ingested": 1, "manifest_skipped": 0}

    monkeypatch.setattr(cli, "_ingest_manifest", fake_ingest_manifest)

    assert cli.main(["ingest-corpus", "real", "--include-pdf", "--limit", "3"]) == 0

    assert seen["manifest"].name == "manifest.csv"
    assert Path(seen["base_dir"]).name == "real_corpus"
    assert seen["include_pdf"] is True
    assert seen["limit"] == 3


def test_reset_stores_requires_explicit_confirmation():
    with pytest.raises(SystemExit) as exc:
        cli.main(["reset-stores"])

    assert exc.value.code == 2


def test_seed_demo_can_reset_before_ingesting(monkeypatch):
    calls = []

    monkeypatch.setattr(cli, "_reset_stores", lambda: {"chunks": 0, "qdrant_points": 0})

    def fake_ingest_manifest(manifest, base_dir, include_pdf=False, limit=None):
        calls.append((manifest, base_dir, include_pdf, limit))
        return {"manifest_ingested": 6, "manifest_skipped": 0}

    monkeypatch.setattr(cli, "_ingest_manifest", fake_ingest_manifest)

    assert cli.main(["seed-demo", "--reset", "--limit", "1"]) == 0

    assert len(calls) == 1
    assert calls[0][0].name == "manifest.csv"
    assert calls[0][3] == 1
