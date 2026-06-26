"""Grounded answer generation over retrieved chunks (local LLM, citations)."""

from __future__ import annotations

from typing import cast

from ..llm import generation_llm

_SYSTEM = (
    "/no_think\n"
    "You are a corporate knowledge assistant. Answer the question using ONLY the "
    "provided context. Cite the sources you use as [n]. If the context does not "
    "contain the answer, say you don't have enough information — do not guess."
)


def build_context(chunks: list[dict]) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        heading = " > ".join(c.get("heading_path") or [])
        label = f"[{i}] {c.get('source', '?')}" + (f" :: {heading}" if heading else "")
        blocks.append(f"{label}\n{c['text']}")
    return "\n\n".join(blocks)


def answer(query: str, chunks: list[dict]) -> str:
    if not chunks:
        return "I don't have enough information in the accessible documents to answer that."
    context = build_context(chunks)
    reply = generation_llm().invoke(
        [("system", _SYSTEM), ("user", f"Context:\n{context}\n\nQuestion: {query}")]
    )
    return cast(str, reply.content or "")  # chat content is str for these models
