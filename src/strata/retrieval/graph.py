"""Graph branch: multi-hop expansion in Neo4j around the vector seeds.

This is what plain vector RAG can't do — from the semantically-retrieved chunks we
walk the entity graph up to N relationship hops and pull in chunks that mention the
connected entities, surfacing multi-hop evidence. ACL-filtered in the same query.
"""

from __future__ import annotations

from functools import lru_cache

import neo4j

from ..config import get_settings
from .acl import AclContext


@lru_cache
def _driver() -> neo4j.Driver:
    """Shared Neo4j driver (own connection pool) — reused across queries; closes at exit."""
    s = get_settings()
    return neo4j.GraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password))


class GraphRetriever:
    def __init__(self) -> None:
        self._db = get_settings().neo4j_database

    def retrieve(
        self, seed_ids: list[str], acl: AclContext, hops: int = 2, limit: int = 20
    ) -> list[dict]:
        if not seed_ids:
            return []
        where, params = acl.cypher_where("other")
        # hops is an int we control (not user input) — safe to inline for the
        # variable-length pattern, which can't take a parameter.
        query = f"""
            MATCH (seed:Chunk) WHERE seed.id IN $seed_ids
            MATCH (seed)-[:MENTIONS]->(:Entity)-[:REL*1..{int(hops)}]-(n:Entity)
            MATCH (other:Chunk)-[:MENTIONS]->(n)
            WHERE {where} AND NOT other.id IN $seed_ids
            WITH other, count(DISTINCT n) AS overlap
            RETURN other.id AS id, other.text AS text,
                   other.heading_path AS heading_path, other.source AS source,
                   overlap AS graph_overlap
            ORDER BY graph_overlap DESC
            LIMIT $limit
        """
        with _driver().session(database=self._db) as ses:
            rows = ses.run(query, seed_ids=seed_ids, limit=limit, **params)
            return [dict(r) for r in rows]
