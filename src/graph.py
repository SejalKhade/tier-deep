"""
LangGraph orchestration for the four-agent Tier Deep pipeline.

Graph:

    START -> discovery -> verification -> uncertainty -> aggregation -> END

Nodes exchange typed state (PipelineState). Every node appends to a
`trace` list so the UI can render the full agent history.
"""
from __future__ import annotations

from langgraph.graph import StateGraph, END

from .agents import (
    PipelineState,
    discovery_node,
    verification_node,
    uncertainty_node,
    aggregation_node,
)


def build_graph():
    g = StateGraph(PipelineState)
    g.add_node("discovery", discovery_node)
    g.add_node("verification", verification_node)
    g.add_node("uncertainty", uncertainty_node)
    g.add_node("aggregation", aggregation_node)

    g.set_entry_point("discovery")
    g.add_edge("discovery", "verification")
    g.add_edge("verification", "uncertainty")
    g.add_edge("uncertainty", "aggregation")
    g.add_edge("aggregation", END)

    return g.compile()


def run_pipeline(
    api_key: str,
    target_company: str,
    corpus: list[dict],
    use_web_search: bool = False,
) -> dict:
    graph = build_graph()
    initial: PipelineState = {
        "api_key": api_key,
        "target_company": target_company,
        "corpus": corpus,
        "use_web_search": use_web_search,
        "trace": [],
    }
    return graph.invoke(initial)
