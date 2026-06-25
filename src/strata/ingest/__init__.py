"""Ingestion: load -> heading-aware chunk -> embed -> write graph (Neo4j) + vectors (Qdrant)."""

from .pipeline import ingest_document

__all__ = ["ingest_document"]
