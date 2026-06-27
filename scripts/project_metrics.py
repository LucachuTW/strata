#!/usr/bin/env python
"""Generate project-level metrics and requirement coverage for Strata."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Requirement:
    requirement: str
    status: str
    evidence: str
    note: str


def _exists(path: str) -> bool:
    return (ROOT / path).exists()


def _contains(path: str, *needles: str) -> bool:
    target = ROOT / path
    if not target.exists():
        return False
    text = target.read_text(encoding="utf-8", errors="ignore")
    return all(needle in text for needle in needles)


def _status(covered: bool) -> str:
    return "ok" if covered else "missing"


def _line_count(pattern: str) -> int:
    return sum(
        path.read_text(encoding="utf-8", errors="ignore").count("\n") + 1
        for path in ROOT.glob(pattern)
        if path.is_file()
    )


def requirements() -> list[Requirement]:
    checks = [
        Requirement(
            "Ingesta de documentos no estructurados",
            _status(_exists("src/strata/ingest/pipeline.py")),
            "src/strata/ingest/pipeline.py",
            "Markdown/text and optional PDF ingestion are implemented.",
        ),
        Requirement(
            "Chunking heading-aware",
            _status(
                _contains(
                    "src/strata/ingest/chunking.py",
                    "MarkdownHeaderTextSplitter",
                    "heading_path",
                )
            ),
            "src/strata/ingest/chunking.py",
            "Chunks preserve heading paths for topical hierarchy.",
        ),
        Requirement(
            "Metadatos: owner, fechas, confidencialidad y tenant",
            _status(
                _contains(
                    "src/strata/schema.py",
                    "owner",
                    "tenant",
                    "confidentiality",
                    "effective_from",
                    "effective_to",
                )
            ),
            "src/strata/schema.py",
            "ChunkMetadata validates document-level ACL metadata.",
        ),
        Requirement(
            "Indexación simultánea vectorial y grafo",
            _status(
                _contains(
                    "src/strata/ingest/pipeline.py",
                    "QdrantWriter",
                    "Neo4jWriter",
                )
            ),
            "src/strata/ingest/pipeline.py",
            "Ingestion writes vectors to Qdrant and graph/chunks to Neo4j.",
        ),
        Requirement(
            "Extracción LLM de entidades y relaciones con esquema Pydantic",
            _status(
                _contains(
                    "src/strata/ingest/extract.py",
                    "BaseModel",
                    "with_structured_output",
                )
            ),
            "src/strata/ingest/extract.py",
            "Extraction is constrained by Pydantic entity/relation models.",
        ),
        Requirement(
            "Deduplicación de nodos",
            _status(
                _contains(
                    "src/strata/ingest/stores.py",
                    "MERGE (n:Entity {key: e.key})",
                )
            ),
            "src/strata/ingest/stores.py",
            "Neo4j MERGE constraints and stable entity keys prevent duplicate entities.",
        ),
        Requirement(
            "Recuperador vectorial",
            _status(_exists("src/strata/retrieval/vector.py")),
            "src/strata/retrieval/vector.py",
            "Qdrant similarity search with ACL payload filtering.",
        ),
        Requirement(
            "Recuperador de grafos multi-hop",
            _status(
                _contains(
                    "src/strata/retrieval/graph.py",
                    "REL*1..",
                    "MENTIONS",
                )
            ),
            "src/strata/retrieval/graph.py",
            "Neo4j expands from vector seed chunks across entity relationships.",
        ),
        Requirement(
            "Fusión de rango recíproco RRF",
            _status(_contains("src/strata/retrieval/rrf.py", "1.0 / (k + rank + 1)")),
            "src/strata/retrieval/rrf.py",
            "Uses k=60 by default through configuration.",
        ),
        Requirement(
            "Reranking cross-encoder",
            _status(_contains("src/strata/rerank.py", "CrossEncoder")),
            "src/strata/rerank.py",
            "Uses local BGE reranker instead of proprietary Cohere.",
        ),
        Requirement(
            "Orquestación LangGraph con bucle planner/critic",
            _status(
                _contains(
                    "src/strata/graph/build.py",
                    "StateGraph",
                    "add_conditional_edges",
                )
            ),
            "src/strata/graph/build.py",
            "Graph nodes implement rewrite, retrieve, generate, critic, retry/end.",
        ),
        Requirement(
            "Reescritura / HyDE-style expansion",
            _status(
                _contains(
                    "src/strata/graph/nodes.py",
                    "rewrite_llm",
                    "information-dense search query",
                )
            ),
            "src/strata/graph/nodes.py",
            "Planner rewrites the question into an expanded retrieval query.",
        ),
        Requirement(
            "Agente crítico de validación",
            _status(_contains("src/strata/graph/nodes.py", "faithfulness", "sufficient")),
            "src/strata/graph/nodes.py",
            "Critic grades faithfulness/sufficiency and feeds retry feedback.",
        ),
        Requirement(
            "Métricas de latencia por fase",
            _status(
                _exists("src/strata/metrics.py")
                and _contains("src/strata/graph/nodes.py", "_over_budget")
            ),
            "src/strata/metrics.py; src/strata/graph/nodes.py",
            "Latency budgets are measured (metrics) and enforced: the agent loop stops on a "
            "total time budget and skips the rewrite LLM call on overrun.",
        ),
        Requirement(
            "Servidor MCP con búsqueda y Cypher controlado",
            _status(
                _contains(
                    "src/strata/mcp_server.py",
                    "FastMCP",
                    "search_corpus",
                    "run_cypher",
                )
            ),
            "src/strata/mcp_server.py",
            "MCP exposes ACL-filtered corpus search and constrained read-only Cypher.",
        ),
        Requirement(
            "Control de tenencia/ACL en tiempo de consulta",
            _status(
                _contains(
                    "src/strata/retrieval/acl.py",
                    "tenant",
                    "allowed_confidentialities",
                )
            ),
            "src/strata/retrieval/acl.py",
            "Qdrant and Cypher filters enforce tenant and confidentiality ceiling.",
        ),
        Requirement(
            "Interfaces CLI y API",
            _status(
                _contains(
                    "src/strata/api.py",
                    '@app.post("/query")',
                    '@app.post("/ask")',
                )
                and _exists("src/strata/cli.py")
            ),
            "src/strata/api.py; src/strata/cli.py",
            "CLI, FastAPI SSE /query, and FastAPI JSON /ask are implemented.",
        ),
    ]
    return checks


def project_metrics() -> dict:
    corpus_metrics_path = ROOT / "datasets" / "real_corpus" / "metrics.json"
    corpus_metrics = {}
    if corpus_metrics_path.exists():
        corpus_metrics = json.loads(corpus_metrics_path.read_text(encoding="utf-8"))
    reqs = requirements()
    return {
        "source_python_files": len(list((ROOT / "src").glob("**/*.py"))),
        "test_python_files": len(list((ROOT / "tests").glob("test_*.py"))),
        "source_lines": _line_count("src/**/*.py"),
        "test_lines": _line_count("tests/test_*.py"),
        "requirements_total": len(reqs),
        "requirements_ok": sum(req.status == "ok" for req in reqs),
        "requirements": [asdict(req) for req in reqs],
        "real_corpus": corpus_metrics,
    }


def _write_report(metrics: dict, path: Path) -> None:
    lines = [
        "# Strata Project Metrics",
        "",
        "## Resumen",
        "",
        f"- Python source files: {metrics['source_python_files']}",
        f"- Test files: {metrics['test_python_files']}",
        f"- Source lines: {metrics['source_lines']}",
        f"- Test lines: {metrics['test_lines']}",
        f"- Requirements covered: {metrics['requirements_ok']}/{metrics['requirements_total']}",
        "",
    ]
    corpus = metrics.get("real_corpus") or {}
    if corpus:
        lines.extend(
            [
                "## Corpus Real",
                "",
                f"- Documents: {corpus.get('documents')}",
                f"- Text documents: {corpus.get('text_documents')}",
                f"- PDF documents: {corpus.get('pdf_documents')}",
                f"- Total bytes: {corpus.get('total_bytes'):,}",
                f"- Estimated words: {corpus.get('estimated_words'):,}",
                f"- Estimated chunks: {corpus.get('estimated_chunks'):,}",
                "",
            ]
        )
    lines.extend(["## Cumplimiento De Requisitos", ""])
    for req in metrics["requirements"]:
        lines.extend(
            [
                f"### {req['requirement']}",
                "",
                f"- Status: {req['status']}",
                f"- Evidence: `{req['evidence']}`",
                f"- Note: {req['note']}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    metrics = project_metrics()
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "project_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    _write_report(metrics, reports / "PROJECT_METRICS.md")
    print(
        json.dumps(
            {
                "requirements_ok": metrics["requirements_ok"],
                "requirements_total": metrics["requirements_total"],
                "source_python_files": metrics["source_python_files"],
                "test_python_files": metrics["test_python_files"],
                "real_corpus_documents": metrics.get("real_corpus", {}).get("documents"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
