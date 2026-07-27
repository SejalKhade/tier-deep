"""
Concentration risk analytics.

Given a supplier graph, surface concrete risks a procurement team would
want to know about. Everything here is deterministic graph math.

Risks detected:
  shared_tier2       - multiple Tier 1 suppliers depend on the same Tier 2
  geographic_cluster - multiple suppliers concentrated in one region
  single_source      - only one supplier for a critical component
  low_confidence     - relationships surfaced but uncertainty is low
"""
from __future__ import annotations

from collections import defaultdict

from .models import SupplierEdge, ConcentrationRisk


def find_shared_tier2(edges: list[SupplierEdge]) -> list[ConcentrationRisk]:
    """A Tier 2 supplier that feeds two or more Tier 1s is a chokepoint."""
    tier2_to_parents = defaultdict(list)
    tier2_edges = defaultdict(list)
    for e in edges:
        if e.tier == 2:
            tier2_to_parents[e.supplier].append(e.parent)
            tier2_edges[e.supplier].append(e.id)

    risks = []
    for t2, parents in tier2_to_parents.items():
        if len(set(parents)) >= 2:
            risks.append(ConcentrationRisk(
                risk_type="shared_tier2",
                severity="high" if len(set(parents)) >= 3 else "medium",
                description=(
                    f"{t2} supplies {len(set(parents))} of your Tier 1 suppliers "
                    f"({', '.join(sorted(set(parents)))}). A disruption at {t2} "
                    f"would cascade through multiple direct suppliers simultaneously."
                ),
                affected_edges=tier2_edges[t2],
                evidence=f"shared_tier2_count={len(set(parents))}",
            ))
    return risks


def find_geographic_clusters(edges: list[SupplierEdge]) -> list[ConcentrationRisk]:
    """Suppliers concentrated in one location = single-region risk."""
    by_location = defaultdict(list)
    for e in edges:
        if e.location:
            by_location[e.location].append(e)

    risks = []
    for loc, group in by_location.items():
        if len(group) >= 3:
            risks.append(ConcentrationRisk(
                risk_type="geographic_cluster",
                severity="high" if len(group) >= 5 else "medium",
                description=(
                    f"{len(group)} suppliers concentrated in {loc}: "
                    f"{', '.join(sorted({e.supplier for e in group}))}. Regional "
                    f"disruption (geopolitical, natural disaster, tariff) would "
                    f"affect all of them at once."
                ),
                affected_edges=[e.id for e in group],
                evidence=f"count_in_region={len(group)}",
            ))
    return risks


def find_single_sources(edges: list[SupplierEdge]) -> list[ConcentrationRisk]:
    """A component with exactly one supplier is a single point of failure."""
    by_component = defaultdict(list)
    for e in edges:
        if e.tier == 1 and e.component:
            key = e.component.lower().strip()
            by_component[key].append(e)

    risks = []
    for comp, group in by_component.items():
        suppliers = {e.supplier for e in group}
        if len(suppliers) == 1:
            e = group[0]
            risks.append(ConcentrationRisk(
                risk_type="single_source",
                severity="high",
                description=(
                    f"Component '{e.component}' has exactly one Tier 1 supplier "
                    f"({e.supplier}). No documented redundancy."
                ),
                affected_edges=[e.id],
                evidence="tier1_supplier_count=1",
            ))
    return risks


def find_low_confidence(edges: list[SupplierEdge], threshold: float = 0.5) -> list[ConcentrationRisk]:
    """Relationships surfaced by the pipeline but with weak evidence."""
    weak = [e for e in edges if e.uncertainty_score < threshold]
    if not weak:
        return []
    return [ConcentrationRisk(
        risk_type="low_confidence",
        severity="medium",
        description=(
            f"{len(weak)} inferred relationship(s) fell below the confidence "
            f"threshold ({threshold}). Treat these as leads, not facts, and "
            f"verify manually before acting."
        ),
        affected_edges=[e.id for e in weak],
        evidence=f"low_confidence_count={len(weak)} threshold={threshold}",
    )]


def compute_all_risks(edges: list[SupplierEdge]) -> list[ConcentrationRisk]:
    return (
        find_shared_tier2(edges)
        + find_geographic_clusters(edges)
        + find_single_sources(edges)
        + find_low_confidence(edges)
    )
