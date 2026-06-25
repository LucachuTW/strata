"""Pydantic-validated chunk metadata (drives ACL) and the fixed KG extraction schema.

Fixed entity/relation types constrain LLM extraction and, together with MERGE-on-key
writes, prevent node duplication — as the README requires.
"""

from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import BaseModel, Field

_CONF_ORDER = ["public", "internal", "confidential", "restricted"]


class Confidentiality(str, Enum):
    public = "public"
    internal = "internal"
    confidential = "confidential"
    restricted = "restricted"

    @property
    def level(self) -> int:
        return _CONF_ORDER.index(self.value)


class EntityLabel(str, Enum):
    organization = "Organization"
    person = "Person"
    financial_metric = "FinancialMetric"
    policy = "Policy"
    product = "Product"
    risk = "Risk"
    reporting_period = "ReportingPeriod"


class RelationType(str, Enum):
    subsidiary_of = "SUBSIDIARY_OF"
    employed_by = "EMPLOYED_BY"
    reports_metric = "REPORTS_METRIC"
    governed_by = "GOVERNED_BY"
    offers = "OFFERS"
    exposed_to = "EXPOSED_TO"
    covers_period = "COVERS_PERIOD"


class ChunkMetadata(BaseModel):
    """Validated metadata attached to every chunk; the basis for query-time ACL."""

    source: str
    owner: str
    tenant: str = "default"
    confidentiality: Confidentiality = Confidentiality.internal
    effective_from: date | None = None
    effective_to: date | None = None
    heading_path: list[str] = Field(default_factory=list)


_ENTITY_DESCRIPTIONS = {
    EntityLabel.organization: "A company, subsidiary, or corporate entity",
    EntityLabel.person: "A named individual (executive, director, employee)",
    EntityLabel.financial_metric: "A reported financial figure or KPI",
    EntityLabel.policy: "A corporate policy, regulation, or guideline",
    EntityLabel.product: "A product, service, or business segment",
    EntityLabel.risk: "A risk factor or contingency",
    EntityLabel.reporting_period: "A fiscal period or relevant date range",
}

# Consumed by the graph retriever / docs; extraction itself is driven by the
# Pydantic models in ingest/extract.py.
NODE_TYPES = [{"label": e.value, "description": _ENTITY_DESCRIPTIONS[e]} for e in EntityLabel]
RELATIONSHIP_TYPES = [{"label": r.value} for r in RelationType]

# Allowed (head, relation, tail) patterns — keeps the graph clean and queryable.
PATTERNS = [
    ("Organization", "SUBSIDIARY_OF", "Organization"),
    ("Person", "EMPLOYED_BY", "Organization"),
    ("Organization", "REPORTS_METRIC", "FinancialMetric"),
    ("Organization", "GOVERNED_BY", "Policy"),
    ("Organization", "OFFERS", "Product"),
    ("Organization", "EXPOSED_TO", "Risk"),
    ("FinancialMetric", "COVERS_PERIOD", "ReportingPeriod"),
]
