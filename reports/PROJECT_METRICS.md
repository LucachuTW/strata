# Strata Project Metrics

## Resumen

- Python source files: 30
- Test files: 12
- Source lines: 2244
- Test lines: 871
- Requirements covered: 17/17

## Corpus Real

- Documents: 173
- Text documents: 171
- PDF documents: 2
- Total bytes: 5,328,206
- Estimated words: 251,613
- Estimated chunks: 2,344

## Cumplimiento De Requisitos

### Ingesta de documentos no estructurados

- Status: ok
- Evidence: `src/strata/ingest/pipeline.py`
- Note: Markdown/text and optional PDF ingestion are implemented.

### Chunking heading-aware

- Status: ok
- Evidence: `src/strata/ingest/chunking.py`
- Note: Chunks preserve heading paths for topical hierarchy.

### Metadatos: owner, fechas, confidencialidad y tenant

- Status: ok
- Evidence: `src/strata/schema.py`
- Note: ChunkMetadata validates document-level ACL metadata.

### Indexación simultánea vectorial y grafo

- Status: ok
- Evidence: `src/strata/ingest/pipeline.py`
- Note: Ingestion writes vectors to Qdrant and graph/chunks to Neo4j.

### Extracción LLM de entidades y relaciones con esquema Pydantic

- Status: ok
- Evidence: `src/strata/ingest/extract.py`
- Note: Extraction is constrained by Pydantic entity/relation models.

### Deduplicación de nodos

- Status: ok
- Evidence: `src/strata/ingest/stores.py`
- Note: Neo4j MERGE constraints and stable entity keys prevent duplicate entities.

### Recuperador vectorial

- Status: ok
- Evidence: `src/strata/retrieval/vector.py`
- Note: Qdrant similarity search with ACL payload filtering.

### Recuperador de grafos multi-hop

- Status: ok
- Evidence: `src/strata/retrieval/graph.py`
- Note: Neo4j expands from vector seed chunks across entity relationships.

### Fusión de rango recíproco RRF

- Status: ok
- Evidence: `src/strata/retrieval/rrf.py`
- Note: Uses k=60 by default through configuration.

### Reranking cross-encoder

- Status: ok
- Evidence: `src/strata/rerank.py`
- Note: Uses local BGE reranker instead of proprietary Cohere.

### Orquestación LangGraph con bucle planner/critic

- Status: ok
- Evidence: `src/strata/graph/build.py`
- Note: Graph nodes implement rewrite, retrieve, generate, critic, retry/end.

### Reescritura / HyDE-style expansion

- Status: ok
- Evidence: `src/strata/graph/nodes.py`
- Note: Planner rewrites the question into an expanded retrieval query.

### Agente crítico de validación

- Status: ok
- Evidence: `src/strata/graph/nodes.py`
- Note: Critic grades faithfulness/sufficiency and feeds retry feedback.

### Métricas de latencia por fase

- Status: ok
- Evidence: `src/strata/metrics.py; src/strata/graph/nodes.py`
- Note: Latency budgets are measured (metrics) and enforced: the agent loop stops on a total time budget and skips the rewrite LLM call on overrun.

### Servidor MCP con búsqueda y Cypher controlado

- Status: ok
- Evidence: `src/strata/mcp_server.py`
- Note: MCP exposes ACL-filtered corpus search and constrained read-only Cypher.

### Control de tenencia/ACL en tiempo de consulta

- Status: ok
- Evidence: `src/strata/retrieval/acl.py`
- Note: Qdrant and Cypher filters enforce tenant and confidentiality ceiling.

### Interfaces CLI y API

- Status: ok
- Evidence: `src/strata/api.py; src/strata/cli.py`
- Note: CLI, FastAPI SSE /query, and FastAPI JSON /ask are implemented.
