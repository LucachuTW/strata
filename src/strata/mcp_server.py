"""MCP server exposing ACL-enforced corpus search and controlled read-only Cypher.

Run:  uv run python -m strata.mcp_server   (stdio transport)
"""

from __future__ import annotations

import re

import neo4j
from mcp.server.fastmcp import FastMCP

from .config import get_settings
from .retrieval import AclContext, retrieve
from .schema import Confidentiality

mcp = FastMCP("strata")

# Word-boundary matching avoids false positives such as "ASSET" containing "SET".
_DISALLOWED = re.compile(
    r"\b("
    r"CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|FOREACH|LOAD\s+CSV|"
    r"CALL|USE|ALTER|GRANT|DENY|REVOKE|START|STOP|TERMINATE|UNION|WITH|UNWIND|"
    r"CREATE\s+CONSTRAINT|DROP\s+CONSTRAINT|CREATE\s+INDEX|DROP\s+INDEX"
    r")\b",
    re.IGNORECASE,
)
_CHUNK_ANCHOR = re.compile(r"\(\s*c\s*:\s*`?Chunk`?\b", re.IGNORECASE)
_FIRST_MATCH = re.compile(
    r"^\s*MATCH\b(?P<body>.*?)(?=\bWHERE\b|\bOPTIONAL\s+MATCH\b|\bMATCH\b|\bRETURN\b|$)",
    re.IGNORECASE | re.DOTALL,
)
_MATCH_CLAUSE = re.compile(
    r"\b(?:OPTIONAL\s+)?MATCH\b(?P<body>.*?)(?=\bWHERE\b|\bOPTIONAL\s+MATCH\b|\bMATCH\b|\bRETURN\b|$)",
    re.IGNORECASE | re.DOTALL,
)
_LEADING_WHERE = re.compile(r"^\s*WHERE\s+", re.IGNORECASE)
_ACL_CYPHER = "c.tenant = $_acl_tenant AND c.confidentiality IN $_acl_conf"


def _acl_cypher(query: str, acl: AclContext, limit: int) -> tuple[str, dict]:
    """Validate a controlled read-only query and inject Chunk ACL predicates.

    The MCP Cypher tool is intentionally narrower than a general Neo4j console:
    every graph exploration must start from the caller-visible ``c:Chunk`` node.
    That keeps entity/relationship reads tied to accessible document chunks.
    """
    clean = query.strip()
    if clean.endswith(";"):
        clean = clean[:-1].strip()
    if not clean:
        raise ValueError("Cypher query cannot be empty.")
    if ";" in clean:
        raise ValueError("Only a single Cypher statement is permitted.")
    if _DISALLOWED.search(clean):
        raise ValueError("Only simple read-only MATCH/RETURN Cypher is permitted.")

    first = _FIRST_MATCH.search(clean)
    if not first or not _CHUNK_ANCHOR.search(first.group("body")):
        raise ValueError("Cypher must begin with a MATCH anchored on (c:Chunk).")

    for clause in _MATCH_CLAUSE.finditer(clean):
        if not re.search(r"\bc\b", clause.group("body")):
            raise ValueError("Every MATCH clause must remain anchored on c:Chunk.")
        if "," in clause.group("body"):
            raise ValueError("Comma-separated MATCH patterns are not permitted.")

    rest = clean[first.end() :]
    where = _LEADING_WHERE.match(rest)
    if where:
        acl_query = (
            clean[: first.end()] + where.group(0) + f"({_ACL_CYPHER}) AND " + rest[where.end() :]
        )
    else:
        acl_query = clean[: first.end()] + f" WHERE {_ACL_CYPHER} " + rest

    result_limit = min(max(1, int(limit)), 500)
    if not re.search(r"\bLIMIT\b", acl_query, re.IGNORECASE):
        acl_query = f"{acl_query}\nLIMIT $_acl_limit"

    return acl_query, {
        "_acl_tenant": acl.tenant,
        "_acl_conf": acl.allowed_confidentialities(),
        "_acl_limit": result_limit,
    }


@mcp.tool()
def search_corpus(question: str, tenant: str = "default", clearance: str = "public") -> list[dict]:
    """Hybrid vector+graph search over the corporate corpus.

    Filtered by the caller's `tenant` and confidentiality `clearance`
    (public|internal|confidential|restricted). Returns the top reranked chunks.
    """
    acl = AclContext(tenant=tenant, clearance=Confidentiality(clearance))
    return [
        {
            "source": h.get("source"),
            "heading_path": h.get("heading_path"),
            "text": h["text"],
            "score": h.get("rerank_score"),
        }
        for h in retrieve(question, acl)
    ]


@mcp.tool()
def run_cypher(
    query: str,
    tenant: str = "default",
    clearance: str = "public",
    limit: int = 50,
) -> list[dict]:
    """Execute a READ-ONLY Cypher query against the knowledge graph.

    Queries must start from the ACL-filtered chunk variable `c:Chunk`, for example:
    `MATCH (c:Chunk)-[:MENTIONS]->(e:Entity) RETURN c.source, e.name`.
    For document content, prefer `search_corpus`.
    """
    acl = AclContext(tenant=tenant, clearance=Confidentiality(clearance))
    safe_query, params = _acl_cypher(query, acl, limit)
    s = get_settings()
    with neo4j.GraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password)) as driver:
        with driver.session(database=s.neo4j_database) as ses:
            return ses.execute_read(
                lambda tx: [r.data() for r in tx.run(safe_query, **params)][: params["_acl_limit"]]
            )


def main() -> None:
    from .log import configure_logging

    configure_logging(get_settings().log_level)
    mcp.run()


if __name__ == "__main__":
    main()
