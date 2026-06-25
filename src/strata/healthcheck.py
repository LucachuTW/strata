"""Phase 0 self-check: confirm every local component answers.

Run:  uv run python -m strata.healthcheck
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import cast


def _check(name: str, fn: Callable[[], str]) -> bool:
    try:
        print(f"[ OK ] {name}: {fn()}")
        return True
    except Exception as e:  # noqa: BLE001 — health check reports any failure
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")
        return False


def check_neo4j() -> str:
    import neo4j

    from .config import get_settings

    s = get_settings()
    with neo4j.GraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password)) as driver:
        driver.verify_connectivity()
    return s.neo4j_uri


def check_qdrant() -> str:
    from qdrant_client import QdrantClient

    from .config import get_settings

    client = QdrantClient(url=get_settings().qdrant_url)
    return f"{len(client.get_collections().collections)} collection(s)"


def check_embeddings() -> str:
    from .embeddings import LocalEmbeddings

    return f"dim={len(LocalEmbeddings().embed_query('hello world'))}"


def check_reranker() -> str:
    from .rerank import rerank

    ranked = rerank(
        "capital of France",
        ["Paris is the capital of France.", "Bananas are yellow."],
        top_n=1,
    )
    return f"best doc index={ranked[0][0]}"


def check_llm() -> str:
    from .llm import rewrite_llm

    reply = rewrite_llm().invoke("Reply with exactly one word: pong")
    return (cast(str, reply.content) or "").strip()[:40] or "(empty reply)"


def main() -> int:
    checks: list[tuple[str, Callable[[], str]]] = [
        ("Neo4j", check_neo4j),
        ("Qdrant", check_qdrant),
        ("Embeddings (BGE-M3, CPU)", check_embeddings),
        ("Reranker (BGE-reranker-v2-m3, CPU)", check_reranker),
        ("LLM (OpenAI-compatible)", check_llm),
    ]
    ok = all(_check(name, fn) for name, fn in checks)
    print("\nAll components healthy." if ok else "\nSome components failed — see above.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
