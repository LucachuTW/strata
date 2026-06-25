"""Heading-aware chunking: split on markdown headers, then size-cap within each section.

Each chunk keeps its heading_path (e.g. ["10-K", "Risk Factors"]) to preserve the
document's thematic hierarchy, as the README requires.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")]
_HEADER_KEYS = ("h1", "h2", "h3", "h4")


@dataclass
class Chunk:
    text: str
    index: int
    heading_path: list[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: str(uuid4()))


def chunk_markdown(text: str, chunk_size: int = 1200, overlap: int = 150) -> list[Chunk]:
    sections = MarkdownHeaderTextSplitter(_HEADERS, strip_headers=False).split_text(text)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks: list[Chunk] = []
    for section in sections:
        heading_path = [section.metadata[k] for k in _HEADER_KEYS if k in section.metadata]
        for piece in splitter.split_text(section.page_content):
            chunks.append(Chunk(text=piece, index=len(chunks), heading_path=heading_path))
    return chunks or [Chunk(text=text, index=0)]
