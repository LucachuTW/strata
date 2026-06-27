# Strata application image. The datastores (Qdrant/Neo4j) run via docker-compose and the
# LLM runs on the host (Ollama/vLLM) — point LLM_BASE_URL at it. torch is CPU-only (see
# pyproject [tool.uv.sources]); the BGE embedding/reranker weights download on first use.
# ponytail: single-stage, runs as root, no in-image corpus — mount datasets if you want to
# seed from inside the container. Add a non-root user / multi-stage trim only if it earns it.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Use the base image's interpreter instead of downloading a second one.
ENV UV_PYTHON_DOWNLOADS=never UV_PYTHON=python3.12
WORKDIR /app

# Cache the heavy dependency layer (torch et al.) separately from source churn.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Then install the package itself (hatchling reads README.md from pyproject).
COPY README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev

ENTRYPOINT ["uv", "run", "strata"]
CMD ["--help"]
