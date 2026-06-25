"""End-to-end ingestion: one document -> Neo4j graph + Qdrant vectors (ACL-tagged)."""

from __future__ import annotations

from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from ..embeddings import LocalEmbeddings
from ..schema import Confidentiality
from .chunking import chunk_markdown
from .extract import extract
from .loaders import load_document
from .stores import Neo4jWriter, QdrantWriter

ROOT = Path(__file__).resolve().parents[3]


def _key(label: str, name: str) -> str:
    """Stable dedup key: same (type, normalized name) -> same node."""
    return f"{label}::{name.strip().lower()}"


def _chunk_id(source: str, index: int, text: str) -> str:
    """Deterministic Qdrant/Neo4j id so re-ingesting a document updates chunks."""
    return str(uuid5(NAMESPACE_URL, f"graphrag-assist:{source}:{index}:{text}"))


def _source_id(path: str | Path) -> str:
    """Stable display/source id; prefer repo-relative paths over collision-prone basenames."""
    p = Path(path).expanduser()
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return p.name


def ingest_document(
    path: str | Path,
    owner: str,
    tenant: str = "default",
    confidentiality: Confidentiality | str = Confidentiality.internal,
    effective_from: str | None = None,
    effective_to: str | None = None,
    verbose: bool = True,
) -> dict:
    text = load_document(path)
    source = _source_id(path)
    chunks = chunk_markdown(text)
    for chunk in chunks:
        chunk.id = _chunk_id(source, chunk.index, chunk.text)
    if verbose:
        print(f"  parsed {len(chunks)} chunks")

    embeddings = LocalEmbeddings().embed_documents([c.text for c in chunks])
    meta = {
        "source": source,
        "doc_key": f"{tenant}::{source}",
        "owner": owner,
        "tenant": tenant,
        "confidentiality": Confidentiality(confidentiality).value,
        "effective_from": effective_from,
        "effective_to": effective_to,
    }

    qw = QdrantWriter()
    qw.ensure_collection()
    qw.delete_document(tenant=tenant, source=source)
    qw.upsert(chunks, embeddings, meta)

    with Neo4jWriter() as nw:
        nw.ensure_constraints()
        nw.delete_document(meta)
        nw.write_document(meta)
        nw.write_chunks(chunks, meta)
        for i, chunk in enumerate(chunks):
            ext = extract(chunk.text)
            names = {e.name.strip().lower(): e for e in ext.entities if e.name.strip()}
            entities = [
                {"key": _key(e.type.value, e.name), "name": e.name.strip(), "type": e.type.value}
                for e in names.values()
            ]
            name_to_key = {n: _key(e.type.value, e.name) for n, e in names.items()}
            relations = []
            for r in ext.relations:
                sk = name_to_key.get(r.source.strip().lower())
                tk = name_to_key.get(r.target.strip().lower())
                if sk and tk and sk != tk:
                    relations.append({"source_key": sk, "target_key": tk, "type": r.type.value})
            nw.write_extraction(chunk.id, entities, relations)
            if verbose:
                print(
                    f"  chunk {i + 1}/{len(chunks)}: "
                    f"{len(entities)} entities, {len(relations)} relations"
                )
        stats = nw.stats()

    stats["qdrant_points"] = qw.count()
    return stats
