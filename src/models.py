"""Typed data structures for the Tier Deep pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Source:
    """A single evidence source backing a supplier claim."""
    url: str
    source_type: str          # sec_filing | news | corporate_registry | web_search | curated
    authority: float          # 0..1  SEC=1.0, major news=0.8, blog=0.3
    date: str                 # ISO date
    excerpt: str              # verbatim quote or paraphrase from the source
    extraction_mode: str      # "explicit" (stated) | "inferred" (deduced)


@dataclass
class SupplierEdge:
    """One inferred relationship in the supplier graph."""
    id: str
    parent: str               # e.g. "Tesla, Inc."
    supplier: str             # e.g. "Panasonic Holdings"
    tier: int                 # 1, 2, 3
    component: str            # what is supplied ("battery cells")
    location: str             # supplier country/region if known

    # populated by the pipeline
    sources: list[Source] = field(default_factory=list)
    discovery_reasoning: str = ""
    verification_status: str = "unchecked"   # verified | partial | unverified
    verification_reasoning: str = ""
    uncertainty_score: float = 0.0           # 0..1 higher = more confident
    uncertainty_breakdown: dict = field(default_factory=dict)


@dataclass
class ConcentrationRisk:
    """A concrete risk finding surfaced to the analyst."""
    risk_type: str            # shared_tier2 | geographic_cluster | single_source
    severity: str             # high | medium | low
    description: str
    affected_edges: list[str] # SupplierEdge ids
    evidence: str


def edge_to_dict(edge: SupplierEdge) -> dict:
    d = asdict(edge)
    return d
