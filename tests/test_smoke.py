"""Offline Phase 0 checks — no network, no models. Run: uv run pytest -q"""

from strata.config import get_settings
from strata.schema import NODE_TYPES, PATTERNS, ChunkMetadata, Confidentiality


def test_settings_defaults():
    s = get_settings()
    assert s.llm_base_url.endswith("/v1")
    assert s.embedding_dim == 1024
    assert s.rrf_k == 60


def test_chunk_metadata_validates():
    m = ChunkMetadata(source="acme-10k.pdf", owner="finance", confidentiality="confidential")
    assert m.confidentiality is Confidentiality.confidential
    assert m.tenant == "default"
    assert m.confidentiality.level > Confidentiality.public.level


def test_kg_schema_consistent():
    labels = {n["label"] for n in NODE_TYPES}
    for head, _rel, tail in PATTERNS:
        assert head in labels and tail in labels
