"""FastAPI surface: streaming /query (single-pass) and /ask (full agent loop).

Run:  uv run uvicorn strata.api:app --port 8000
"""

from __future__ import annotations

import json

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .config import get_settings
from .graph import run_agent
from .llm import generation_llm
from .log import configure_logging
from .retrieval import AclContext, retrieve
from .retrieval.answer import _SYSTEM, build_context
from .schema import Confidentiality

configure_logging(get_settings().log_level)
app = FastAPI(title="Strata")


class Query(BaseModel):
    question: str
    tenant: str = "default"
    clearance: Confidentiality = Confidentiality.restricted


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/query")
def query(q: Query) -> StreamingResponse:
    """Hybrid retrieve (ACL-filtered) + streamed grounded answer (SSE)."""
    acl = AclContext(tenant=q.tenant, clearance=q.clearance)
    try:
        chunks = retrieve(q.question, acl)
    except Exception as exc:  # noqa: BLE001 - surface backend failures consistently
        raise HTTPException(status_code=503, detail="Retrieval backend unavailable.") from exc

    def events():
        sources = [
            {"source": c.get("source"), "heading_path": c.get("heading_path")} for c in chunks
        ]
        yield f"event: sources\ndata: {json.dumps(sources)}\n\n"
        if not chunks:
            msg = "No accessible documents answer this question."
            yield f"event: token\ndata: {json.dumps(msg)}\n\n"
        else:
            prompt = [
                ("system", _SYSTEM),
                ("user", f"Context:\n{build_context(chunks)}\n\nQuestion: {q.question}"),
            ]
            try:
                for part in generation_llm().stream(prompt):
                    if part.content:
                        yield f"event: token\ndata: {json.dumps(part.content)}\n\n"
            except Exception as exc:  # noqa: BLE001 - SSE clients need an event, not a broken socket
                msg = f"Generation backend unavailable: {exc.__class__.__name__}"
                yield f"event: error\ndata: {json.dumps(msg)}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@app.post("/ask")
def ask(q: Query) -> dict:
    """Full agentic loop (planner→retrieve→generate→critic) — JSON response."""
    acl = AclContext(tenant=q.tenant, clearance=q.clearance)
    try:
        final = run_agent(q.question, acl)
    except Exception as exc:  # noqa: BLE001 - surface backend failures consistently
        raise HTTPException(status_code=503, detail="Agent backend unavailable.") from exc
    return {
        "answer": final.get("answer", ""),
        "iterations": final.get("iteration"),
        "elapsed_ms": final.get("elapsed_ms"),
        "faithfulness": final.get("faithfulness"),
        "sufficient": final.get("sufficient"),
        "sources": [
            {"source": c.get("source"), "heading_path": c.get("heading_path")}
            for c in final.get("chunks", [])
        ],
    }
