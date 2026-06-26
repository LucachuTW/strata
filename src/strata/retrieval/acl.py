"""Query-time access control. One context yields both a Qdrant filter and a Cypher WHERE.

Clearance is a confidentiality ceiling: a user sees documents at or below their level,
within their tenant. This is the GDPR/RGPD boundary the README requires — enforced in
the retrieval path, never after generation.
"""

from __future__ import annotations

from dataclasses import dataclass

from qdrant_client import models as qmodels

from ..schema import Confidentiality


@dataclass
class AclContext:
    tenant: str = "default"
    clearance: Confidentiality = Confidentiality.public

    def allowed_confidentialities(self) -> list[str]:
        return [c.value for c in Confidentiality if c.level <= self.clearance.level]

    def qdrant_filter(self) -> qmodels.Filter:
        return qmodels.Filter(
            must=[
                qmodels.FieldCondition(key="tenant", match=qmodels.MatchValue(value=self.tenant)),
                qmodels.FieldCondition(
                    key="confidentiality",
                    match=qmodels.MatchAny(any=self.allowed_confidentialities()),
                ),
            ]
        )

    def cypher_where(self, var: str) -> tuple[str, dict]:
        clause = f"{var}.tenant = $acl_tenant AND {var}.confidentiality IN $acl_conf"
        params = {"acl_tenant": self.tenant, "acl_conf": self.allowed_confidentialities()}
        return clause, params
