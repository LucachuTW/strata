"""Offline test for heading-aware chunking — no services/models."""

from strata.ingest.chunking import chunk_markdown
from strata.ingest.pipeline import _chunk_id, _source_id

_MD = (
    "# Acme Corp Policy\n\nIntro paragraph.\n\n"
    "## Data Retention\n\nKeep records for seven years.\n\n"
    "### GDPR\n\nComply with GDPR requirements.\n"
)


def test_chunks_carry_heading_path():
    chunks = chunk_markdown(_MD)
    assert chunks
    paths = [tuple(c.heading_path) for c in chunks]
    assert any("Data Retention" in p for p in paths)
    # nested heading is captured as a full path
    assert any(len(p) >= 3 for p in paths)


def test_chunks_indexed_and_unique():
    chunks = chunk_markdown(_MD)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert len({c.id for c in chunks}) == len(chunks)


def test_ingest_chunk_ids_are_deterministic_uuid_strings():
    first = _chunk_id("policy.md", 0, "same text")
    second = _chunk_id("policy.md", 0, "same text")
    changed = _chunk_id("policy.md", 1, "same text")

    assert first == second
    assert first != changed
    assert len(first) == 36


def test_source_id_prefers_repo_relative_path():
    source = _source_id("tests/fixtures/sample_policy.md")

    assert source == "tests/fixtures/sample_policy.md"
