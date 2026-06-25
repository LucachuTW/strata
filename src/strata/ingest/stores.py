"""Writers for the two stores: Neo4j (graph + chunks) and Qdrant (chunk vectors).

Both carry the ACL fields (tenant/owner/confidentiality) so retrieval can filter
at query time. Qdrant points reference their Neo4j chunk via payload `neo4j_id`,
which QdrantNeo4jRetriever uses to fetch graph context.
"""

from __future__ import annotations

import neo4j
from qdrant_client import QdrantClient
from qdrant_client import models as qmodels

from ..config import get_settings
from .chunking import Chunk

_CONSTRAINTS = [
    "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT doc_key IF NOT EXISTS FOR (d:Document) REQUIRE d.key IS UNIQUE",
    "CREATE CONSTRAINT entity_key IF NOT EXISTS FOR (e:Entity) REQUIRE e.key IS UNIQUE",
]
_MIGRATIONS = [
    # Older versions keyed documents only by basename/source. That collides when
    # two tenants or corpus folders contain the same filename.
    "DROP CONSTRAINT doc_source IF EXISTS",
]


class Neo4jWriter:
    def __init__(self) -> None:
        s = get_settings()
        self._driver = neo4j.GraphDatabase.driver(
            s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password)
        )
        self._db = s.neo4j_database

    def __enter__(self) -> Neo4jWriter:
        return self

    def __exit__(self, *exc) -> None:
        self._driver.close()

    def ensure_constraints(self) -> None:
        with self._driver.session(database=self._db) as ses:
            for stmt in _MIGRATIONS:
                ses.run(stmt)
            for stmt in _CONSTRAINTS:
                ses.run(stmt)

    def write_document(self, meta: dict) -> None:
        with self._driver.session(database=self._db) as ses:
            ses.run(
                """
                MERGE (d:Document {key: $doc_key})
                SET d.source=$source, d.owner=$owner, d.tenant=$tenant,
                    d.confidentiality=$confidentiality,
                    d.effective_from=$effective_from, d.effective_to=$effective_to
                """,
                **meta,
            )

    def write_chunks(self, chunks: list[Chunk], meta: dict) -> None:
        rows = [
            {"id": c.id, "text": c.text, "index": c.index, "heading_path": c.heading_path}
            for c in chunks
        ]
        with self._driver.session(database=self._db) as ses:
            ses.run(
                """
                UNWIND $rows AS ch
                MERGE (c:Chunk {id: ch.id})
                SET c.text=ch.text, c.index=ch.index, c.heading_path=ch.heading_path,
                    c.source=$source, c.owner=$owner, c.tenant=$tenant,
                    c.confidentiality=$confidentiality,
                    c.effective_from=$effective_from, c.effective_to=$effective_to
                WITH c
                MATCH (d:Document {key: $doc_key})
                MERGE (c)-[:PART_OF]->(d)
                """,
                rows=rows,
                **meta,
            )

    def delete_document(self, meta: dict) -> None:
        """Remove previous chunks for this tenant/source before replacing a document."""
        with self._driver.session(database=self._db) as ses:
            ses.run(
                """
                MATCH (d:Document {key: $doc_key})
                DETACH DELETE d
                """,
                **meta,
            )
            ses.run(
                """
                MATCH (c:Chunk {tenant: $tenant, source: $source})
                DETACH DELETE c
                """,
                **meta,
            )
            ses.run(
                """
                MATCH (e:Entity)
                WHERE NOT EXISTS { MATCH (:Chunk)-[:MENTIONS]->(e) }
                DETACH DELETE e
                """
            )

    def write_extraction(self, chunk_id: str, entities: list[dict], relations: list[dict]) -> None:
        with self._driver.session(database=self._db) as ses:
            if entities:
                ses.run(
                    """
                    MATCH (c:Chunk {id: $chunk_id})
                    UNWIND $entities AS e
                    MERGE (n:Entity {key: e.key})
                      ON CREATE SET n.name = e.name, n.type = e.type
                    MERGE (c)-[:MENTIONS]->(n)
                    """,
                    chunk_id=chunk_id,
                    entities=entities,
                )
            if relations:
                ses.run(
                    """
                    UNWIND $relations AS r
                    MATCH (a:Entity {key: r.source_key}), (b:Entity {key: r.target_key})
                    MERGE (a)-[rel:REL {type: r.type}]->(b)
                    """,
                    relations=relations,
                )

    def stats(self) -> dict:
        with self._driver.session(database=self._db) as ses:
            row = ses.run(
                """
                RETURN count{ (c:Chunk) } AS chunks,
                       count{ (e:Entity) } AS entities,
                       count{ (d:Document) } AS documents,
                       count{ ()-[r:REL]->() } AS relations
                """
            ).single()
            return dict(row) if row is not None else {}

    def clear(self) -> None:
        with self._driver.session(database=self._db) as ses:
            ses.run("MATCH (n) DETACH DELETE n")


class QdrantWriter:
    _ACL_FIELDS = ("tenant", "owner", "confidentiality", "source")

    def __init__(self) -> None:
        s = get_settings()
        self._client = QdrantClient(url=s.qdrant_url)
        self._collection = s.qdrant_collection
        self._dim = s.embedding_dim

    def ensure_collection(self) -> None:
        if not self._client.collection_exists(self._collection):
            # Named "dense" (BGE-M3) + "bm25" sparse vectors; Qdrant fuses them at
            # query time. IDF modifier weights rare tokens (exact IDs) in BM25.
            self._client.create_collection(
                self._collection,
                vectors_config={
                    "dense": qmodels.VectorParams(
                        size=self._dim, distance=qmodels.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "bm25": qmodels.SparseVectorParams(modifier=qmodels.Modifier.IDF)
                },
            )
            for field in self._ACL_FIELDS:  # keyword indexes for fast ACL filtering
                self._client.create_payload_index(
                    self._collection, field, qmodels.PayloadSchemaType.KEYWORD
                )

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]], meta: dict) -> None:
        from ..sparse import embed_documents as sparse_embed_documents

        sparse = sparse_embed_documents([c.text for c in chunks])
        points = [
            qmodels.PointStruct(
                id=c.id,
                vector={
                    "dense": emb,
                    "bm25": qmodels.SparseVector(indices=s_idx, values=s_val),
                },
                payload={
                    "neo4j_id": c.id,
                    "text": c.text,
                    "heading_path": c.heading_path,
                    "source": meta["source"],
                    "owner": meta["owner"],
                    "tenant": meta["tenant"],
                    "confidentiality": meta["confidentiality"],
                },
            )
            for c, emb, (s_idx, s_val) in zip(chunks, embeddings, sparse, strict=True)
        ]
        self._client.upsert(self._collection, points=points)

    def delete_document(self, tenant: str, source: str) -> None:
        if not self._client.collection_exists(self._collection):
            return
        self._client.delete(
            collection_name=self._collection,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="tenant", match=qmodels.MatchValue(value=tenant)
                        ),
                        qmodels.FieldCondition(
                            key="source", match=qmodels.MatchValue(value=source)
                        ),
                    ]
                )
            ),
            wait=True,
        )

    def count(self) -> int:
        if not self._client.collection_exists(self._collection):
            return 0
        return self._client.count(self._collection).count

    def clear(self) -> None:
        if self._client.collection_exists(self._collection):
            self._client.delete_collection(self._collection)
