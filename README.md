<p align="center">
  <img src="docs/logo.svg" alt="Strata" width="440">
</p>

<p align="center"><strong>Governed graph retrieval — answers only from what you're cleared to see.</strong></p>

<p align="center">
  <img alt="Python 3.12" src="https://img.shields.io/badge/python-3.12-blue">
  <img alt="Tests" src="https://img.shields.io/badge/tests-passing-brightgreen">
  <img alt="Models" src="https://img.shields.io/badge/models-100%25%20local%20%2F%20OSS-success">
  <img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-lightgrey">
</p>

# Strata

Strata is an **agentic GraphRAG** system for local corporate document retrieval. Its
defining feature is **clearance-aware retrieval**: a user's clearance is a ceiling
(`public < internal < confidential < restricted`) enforced *inside* the retrieval path —
the model never sees a chunk the caller isn't allowed to read, so it cannot leak it.

Under the hood it combines Qdrant vector search, Neo4j knowledge-graph expansion,
reciprocal-rank fusion, cross-encoder reranking, and a LangGraph planner/critic loop. All
model calls go to local or self-hosted OpenAI-compatible endpoints; no proprietary cloud
model API is required.

See [`docs/DEMO.md`](docs/DEMO.md) for a 60-second walkthrough.

### What this project demonstrates

- **Security-first retrieval** — ACLs enforced *inside* the Qdrant filter and Neo4j `WHERE`, never as a
  post-generation scrub, so a higher-clearance fact can't reach the model in the first place.
- **Retrieval engineering** — a measured failure (dense retrieval can't pin exact CVE IDs) root-caused
  and fixed with a dense + BM25 hybrid, with before/after numbers.
- **Agentic orchestration** — a LangGraph rewrite→retrieve→generate→critic loop bounded by an iteration
  cap *and* a hard wall-clock budget, with a graceful skip-rewrite mitigation on overrun.
- **Evaluation rigour** — versioned golden sets, lenient + strict scorers, and stored answer texts that
  caught a bug in the metric itself (see [Limitations](#limitations--evaluation-notes)).
- **Fully local / OSS** — Qwen3 (Ollama/vLLM), BGE-M3, BGE reranker, Qdrant, Neo4j. No cloud model API.

## Results

Measured by `uv run strata eval` (Qwen3-8B via Ollama on an RTX 2070) against versioned golden
sets, on two corpora.

### Synthetic Acme corpus — the ACL story

20-item golden set with confidentiality tiers (12 factual, 3 multi-hop, 5 ACL-denial); full breakdown
in [`reports/EVAL_METRICS.md`](reports/EVAL_METRICS.md):

| Metric | Result |
|---|---|
| Retrieval recall@k | **0.97** |
| Answer correctness — substring (single-pass) | **100%** |
| Answer correctness — LLM-judge, strict (single-pass) | **80%** |
| **ACL-safety** — no higher-clearance fact leaks below its level | **100%** (5/5 denial cases) |
| Mean critic faithfulness (agent) | **0.96** |
| Mean latency — single-pass vs agent loop | 7.3 s vs 27.8 s |

The headline is **ACL-safety: 100%**. The ACL-denial cases run the full retry budget and still refuse —
the secret is filtered in Qdrant **and** Neo4j before generation, so it never reaches the model. (Recall
is 1.00 on every item except one multi-hop question that missed a second supporting chunk this run.)

### Real public corpus (CISA KEV) — measuring a gap, then closing it

14-item golden set over a **150-document slice of the live CISA Known Exploited Vulnerabilities
catalog** (450 chunks of near-identical CVE records); breakdown in
[`reports/EVAL_METRICS_real.md`](reports/EVAL_METRICS_real.md).

The first run exposed a real weakness: pure **dense** retrieval (BGE-M3) can't pin exact
identifiers. For a query like *"what product is affected by CVE-2026-35273?"* the right record
didn't even reach the top-20 candidates among 150 look-alikes — so the reranker couldn't rescue
it. Root-caused, then fixed with **dense + BM25 sparse hybrid retrieval** (fused by Qdrant). Recall
on exact-identifier lookups went from partial to perfect:

| Metric (single-pass) | Dense only | + BM25 hybrid |
|---|---|---|
| Retrieval recall@k (right CVE among 150) | 0.79 | **1.00** |
| Answer correctness | 79% | **100%** |

Honest nuance on the **agent loop** for this corpus: it is latency-heavy (median ~22 s, with occasional
multi-minute Ollama model-swap stalls on 8 GB VRAM) and does not beat single-pass on these lookups — the
latest run matches it (judge 100% / 100%), while an earlier run had the query-rewrite step drop the
precise CVE ID and score slightly lower (run-to-run variance; see
[Limitations](#limitations--evaluation-notes)). For an identifier-heavy corpus the **retriever, not the
agent, is the right lever**. Reproduce: `strata download-real` →
`strata eval --golden datasets/real_corpus/eval/golden.jsonl`.

### Hardened benchmark — a metric that discriminates

The runs above hit 100% on a lenient substring metric — which mostly says the questions were easy
(single-fact lookups over labelled fields). So the benchmark was hardened (`golden_hard.jsonl`):
questions phrased **without** the CVE ID (semantic retrieval), disambiguation between two CVEs, and
**absent-fact refusal** cases (a missing CVE, an off-corpus topic, and an attribute the KEV records
don't carry — a CVSS score). Scoring adds a **strict LLM-judge** alongside the lenient substring.
Breakdown in [`reports/EVAL_METRICS_real_hard.md`](reports/EVAL_METRICS_real_hard.md):

| Metric | single-pass | agent loop |
|---|---|---|
| Retrieval recall@k (no CVE ID in the query) | **1.00** | 1.00 |
| Correctness — substring (lenient) | 100% | 100% |
| Correctness — **LLM-judge (strict)** | **71%** | 71% |
| Refusal accuracy (absent facts) | 100% | 100% |

The useful finding here is about **metric design**, not a model ranking. (1) The strict LLM-judge drops
the saturated 100% substring score to **71%** for *both* paths — the lenient metric was hiding
answer-quality differences, so the judge is the discriminating number. It is deliberately strict and
also penalises correct-but-padded answers, so 71% is a conservative floor, not a hallucination rate.
(2) The single-pass-vs-agent comparison on this set is **within noise** (see
[Limitations](#limitations--evaluation-notes)). One run scored the agent loop's refusal accuracy at
100% and a later run at 33% — and storing the answer texts showed *both* were **scoring artifacts**: the
keyword refusal-matcher didn't recognise phrasings like *"does not include"* / *"does not mention"*.
With the matcher fixed, both paths correctly decline on every absent-fact item (100% / 100%). On this
corpus the planner/critic loop adds latency without a measurable accuracy gain — and at these sample
sizes it doesn't clearly beat single-pass on the multi-hop set either (see
[Limitations](#limitations--evaluation-notes)). Its value is architectural — grounded refusal, bounded
retry, a hard time budget — not something these small sets resolve. Storing the answers is what made all
of this auditable.

## Limitations & evaluation notes

These numbers are honest, but small and local — read them with the right caveats:

- **Small sample, single run.** The golden sets are 20 (synthetic) / 14 (real) / 10 (hard) items, run
  once on one machine, not averaged over seeds. The hard set judges only 7 items (3 are refusal), so a
  single-pass-vs-agent gap of one or two items is well within run-to-run noise and **flips direction
  between runs** — do not read those deltas as a model ranking.
- **The judge is a same-model, stochastic grader.** A local Qwen3 judges Qwen3 answers. It is
  deliberately strict (it penalises correct-but-elaborated answers), so the strict score is a
  conservative floor on *concise* correctness, not a hallucination rate.
- **Latency is 8 GB-VRAM-bound.** The agent loop's median ~22 s and the occasional multi-minute spike
  are Ollama swapping the 8B generation/critic model and the 1.7B rewrite model in and out of VRAM, not
  algorithmic cost.
- **Answers are stored for auditability.** Every eval row keeps the raw `sp_answer` / `ag_answer` in the
  JSON report. This is not decoration: it is how the refusal-accuracy artifact above was caught — the
  metric, not the model, was wrong, and that was only visible by reading the answers.

What the evaluation *does* support, repeatably: **ACL-safety** (no higher-clearance fact leaks below its
level — enforced inside retrieval, not filtered post-hoc), the **dense→hybrid recall fix** (0.79→1.00 on
exact identifiers), and that a **strict judge discriminates** where the saturated substring metric does
not. Across all three small golden sets the **agent loop does not measurably beat single-pass** and is
~3× slower — its value is architectural (in-retrieval ACL, grounded refusal, bounded retry + a hard time
budget), not a demonstrated accuracy gain at this sample size.

## What Is Implemented

- Heading-aware ingestion for Markdown/text, with optional PDF support through
  `docling`.
- Chunk metadata and query-time ACL filtering by tenant and confidentiality.
- LLM-driven entity/relation extraction into a constrained Pydantic schema.
- Dual writes to Qdrant and Neo4j, including graph constraints and vector payload
  fields needed for ACL filtering.
- Hybrid retrieval: dense (BGE-M3) + BM25 sparse search fused in Qdrant, graph
  expansion around vector seeds, RRF fusion, and BGE cross-encoder reranking.
- LangGraph agent loop: rewrite, retrieve, generate, critic, and bounded retry.
- CLI, FastAPI, and MCP interfaces.
- A bundled synthetic sample corpus under `datasets/acme_corpus` for demos,
  integration testing, and ACL validation.
- A real public corpus pack under `datasets/real_corpus` that downloads SEC
  10-K filings, CISA KEV vulnerability data, and NIST AI/cybersecurity
  framework documents.

## Architecture

```text
Question
  -> rewrite / HyDE-style expansion
  -> dense + BM25 sparse retrieval in Qdrant (RRF-fused), ACL filtered
  -> graph expansion in Neo4j from vector seeds, ACL filtered
  -> reciprocal-rank fusion (vector + graph)
  -> BGE reranking
  -> grounded answer generation
  -> critic verdict and optional retry
```

The ACL boundary is enforced before context reaches the generator. A user's
clearance is a ceiling: `public < internal < confidential < restricted`.

## Local Stack

Requirements:

- Python 3.12
- `uv`
- Docker, for Qdrant and Neo4j
- Ollama or another OpenAI-compatible local/self-hosted LLM endpoint

Setup:

```bash
uv sync
cp .env.example .env
docker compose up -d
ollama serve
bash scripts/pull_models.sh
```

Check dependencies and services:

```bash
uv run strata doctor
```

Seed a clean bundled sample corpus:

```bash
uv run strata seed-demo --reset
```

The sample corpus contains public, internal, confidential, and restricted
documents about Acme Corporation. It is synthetic and versioned in the repo so
the system is useful immediately after setup.

If you want to add to an existing store instead of resetting it:

```bash
uv run strata ingest-corpus synthetic
```

For a faster smoke seed, limit the number of demo documents:

```bash
uv run strata seed-demo --reset --limit 1
```

Download a real public corpus:

```bash
SEC_USER_AGENT="Your Name your.email@example.com" \
  uv run strata download-real --sec-limit 20 --kev-limit 150
```

The current real-corpus snapshot has 173 manifest documents: 20 SEC 10-K
filings, 151 CISA KEV Markdown documents, and 2 NIST PDFs. It is about 5.3 MB,
251k estimated words, and 2,344 estimated text chunks before PDF parsing.

Ingest the generated real corpus:

```bash
uv run strata ingest-corpus real
```

The real corpus uses official public sources: SEC EDGAR, CISA KEV, NIST AI RMF,
and NIST Cybersecurity Framework. Metrics are written to
`datasets/real_corpus/metrics.json` and `datasets/real_corpus/metrics.md`.
See `datasets/real_corpus/README.md`.

Generate project-level requirement coverage:

```bash
uv run strata metrics project
```

This writes `reports/PROJECT_METRICS.md` and `reports/project_metrics.json`. The current report
shows 17/17 project requirements covered.

Retrieval uses two separate rerank controls:

- `RERANK_CANDIDATE_K` controls how many fused vector/graph candidates the
  cross-encoder scores.
- `RERANK_TOP_N` controls how many reranked chunks are returned to generation.

Keeping the candidate pool wider than the returned context protects compound
questions where the best supporting chunk is not in the first few fused results.

Two more controls bound quality and latency:

- `RERANK_SCORE_FLOOR` (blank = off) drops chunks the cross-encoder scores below the
  floor, so a query with no relevant context returns an honest empty result instead of
  the least-bad chunks.
- `AGENT_TIME_BUDGET_S` (0 = off) caps the agent loop's total wall-clock time. On
  overrun it returns the best answer so far and skips the rewrite LLM call. `/ask` and
  the CLI `ask` report `elapsed_ms`.

For a single-document smoke test, ingest the minimal fixture:

```bash
uv run strata ingest tests/fixtures/sample_policy.md \
  --owner finance \
  --confidentiality confidential
```

Query with the single-pass retriever:

```bash
uv run strata query \
  "Who is the CFO of Acme Robotics and which risks affect Acme Corporation?" \
  --clearance restricted
```

Run the full agent loop:

```bash
uv run strata ask \
  "What restricted board targets exist for Acme Robotics?" \
  --clearance restricted
```

Print ready-to-run query examples:

```bash
uv run strata examples
```

Measure hybrid retrieval latency and candidate counts without LLM answer
generation:

```bash
uv run strata query-metrics \
  "Which CISA KEV entries mention ransomware?" \
  --clearance public
```

Write benchmark metrics for a small query set:

```bash
uv run strata metrics runtime
```

This writes `reports/runtime_metrics.json` and `reports/RUNTIME_METRICS.md`.

## Interfaces

FastAPI:

```bash
uv run strata serve api --port 8000
```

- `GET /health` returns service liveness.
- `POST /query` streams SSE events for sources, tokens, errors, and completion.
- `POST /ask` returns the full LangGraph agent result as JSON.

MCP:

```bash
uv run strata serve mcp
```

Tools:

- `search_corpus(question, tenant, clearance)` runs ACL-filtered hybrid search.
- `run_cypher(query, tenant, clearance, limit)` runs controlled read-only Cypher.
  Queries must start from `MATCH (c:Chunk)` so the server can inject tenant and
  confidentiality filtering before returning graph data.

## Run in a container

The datastores stay in `docker-compose`; the app ships as its own image. Build it and run any CLI
command:

```bash
docker build -t strata .
docker run --rm strata --version
```

To serve the API in a container wired to the datastores and your host LLM (Ollama/vLLM), bring up the
optional, profile-gated `strata` service:

```bash
docker compose --profile app up --build
```

It connects to `neo4j`/`qdrant` over the compose network and reaches the host LLM via
`host.docker.internal`. Override `LLM_BASE_URL` to point elsewhere.

## Testing

Offline unit and interface tests do not require Docker, Ollama, Neo4j, Qdrant, or
model downloads:

```bash
uv run pytest -q
```

CI (GitHub Actions) runs the same gate on every push and pull request — lint, type-check, and tests:

```bash
uv run ruff check .
uv run mypy src
uv run pytest -q
```

Full local verification requires the stack described above and should include:

```bash
uv run strata doctor
uv run strata seed-demo --reset
uv run strata stats
uv run strata query "Which information is visible with public clearance?" --clearance public
uv run strata ask "Who is the CFO of Acme Robotics and which risks affect Acme Corporation?" --clearance restricted
```

To clear the app-owned Neo4j/Qdrant stores explicitly:

```bash
uv run strata reset-stores --yes
```
