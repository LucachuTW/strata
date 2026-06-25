"""Central configuration. All values are env-overridable (see .env.example)."""

from __future__ import annotations

import os
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Observability ---
    log_level: str = Field(default="WARNING", validation_alias="STRATA_LOG_LEVEL")

    # --- LLM (OpenAI-compatible: Ollama in dev, vLLM in prod) ---
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "ollama"  # any non-empty string for local servers
    llm_model: str = "qwen3:8b"
    llm_rewrite_model: str = "qwen3:1.7b"
    llm_temperature: float = 0.1
    llm_request_timeout: float = 120.0

    # --- HuggingFace (optional: only raises model-download rate limits) ---
    hf_token: str | None = None

    # --- Embeddings (CPU) ---
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cpu"
    embedding_dim: int = 1024

    # --- Reranker (CPU) ---
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: str = "cpu"

    # --- Sparse / lexical (BM25, fused with dense in Qdrant) ---
    sparse_model: str = "Qdrant/bm25"

    # --- Neo4j (graph) ---
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "graphrag_dev_pw"
    neo4j_database: str = "neo4j"

    # --- Qdrant (vectors) ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "graphrag_chunks"

    # --- Retrieval / fusion ---
    retriever_top_k: int = 20
    rrf_k: int = 60
    rerank_candidate_k: int = 20
    rerank_top_n: int = 6
    # Minimum cross-encoder score to keep a chunk (None = off). BGE-reranker emits
    # logits where >0 is roughly "relevant"; 0.0 is a sane starting floor. Left off
    # by default so it never silently drops borderline-but-useful context.
    rerank_score_floor: float | None = None

    # --- Agent loop ---
    faithfulness_threshold: float = 0.7
    max_iterations: int = 3
    # Total wall-clock budget for the agent loop (0 disables). On overrun the loop
    # stops retrying and returns the best answer so far, and rewrite skips its LLM call.
    agent_time_budget_s: float = 30.0


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    if s.hf_token:  # let huggingface_hub pick it up for downloads
        os.environ.setdefault("HF_TOKEN", s.hf_token)
    return s
