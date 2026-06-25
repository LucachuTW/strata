#!/usr/bin/env bash
# Pull the local open-source LLMs into Ollama. Embedding + reranker models
# (BAAI/bge-m3, BAAI/bge-reranker-v2-m3) are HuggingFace models that
# sentence-transformers downloads automatically on first use.
set -euo pipefail

echo ">> Pulling generation/critic model (Qwen3-8B, ~5GB)…"
ollama pull qwen3:8b

echo ">> Pulling query-rewrite model (Qwen3-1.7B, ~1.4GB)…"
ollama pull qwen3:1.7b

echo ">> Done. Embedding/reranker weights download on first use of the Python code."
