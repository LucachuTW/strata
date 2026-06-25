"""LLM entity/relation extraction into the fixed Pydantic schema (local model)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..llm import generation_llm
from ..schema import EntityLabel, RelationType


class Entity(BaseModel):
    name: str = Field(description="Canonical, deduplicated name of the entity")
    type: EntityLabel


class Relation(BaseModel):
    source: str = Field(description="Name of the source entity")
    target: str = Field(description="Name of the target entity")
    type: RelationType


class Extraction(BaseModel):
    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)


_SYSTEM = (
    "/no_think\n"
    "You extract a knowledge graph from corporate and financial text. "
    "Identify entities and relationships using ONLY the allowed types. "
    "Use canonical, deduplicated names (e.g. 'Acme Corporation', never 'Acme' and "
    "'the company' as two entities). Only emit a relation when BOTH endpoints are "
    "present in your entities list. If nothing relevant appears, return empty lists."
)


def extract(text: str) -> Extraction:
    """Extract entities/relations from one chunk. Returns empty on any model error."""
    llm = generation_llm().with_structured_output(Extraction, method="function_calling")
    try:
        result = llm.invoke([("system", _SYSTEM), ("user", text)])
        return result if isinstance(result, Extraction) else Extraction()
    except Exception:  # noqa: BLE001 — a bad extraction must not abort ingestion
        return Extraction()
