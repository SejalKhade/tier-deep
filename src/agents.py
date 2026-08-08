"""
The four agents that make up the Tier Deep pipeline.

Each agent is a function that takes the current LangGraph state and
returns a partial state update. Every agent produces a structured
output, a natural-language reasoning trail, and never invents data
outside the sources it was given.

Model choice: Claude Haiku 4.5 is used throughout. The tasks are
extraction, verification, and scoring - not deep reasoning - so
latency and cost matter more than raw capability.
"""
from __future__ import annotations

import json
import uuid
from typing import TypedDict, Optional

from anthropic import Anthropic

from .models import Source, SupplierEdge


MODEL = "claude-haiku-4-5-20251001"


class PipelineState(TypedDict, total=False):
    """State passed between LangGraph nodes."""
    api_key: str
    target_company: str
    corpus: list[dict]                    # from curated_data
    use_web_search: bool
    discovered_edges: list[SupplierEdge]
    verified_edges: list[SupplierEdge]
    scored_edges: list[SupplierEdge]
    trace: list[dict]                     # step-by-step audit log


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------

def _client(api_key: str) -> Anthropic:
    return Anthropic(api_key=api_key)


def _extract_json(text: str) -> Optional[dict | list]:
    """
    Strip code fences and parse JSON. On failure, attempt to extract
    a partial JSON object by finding the last complete closing brace.
    Returns None only if no valid JSON can be recovered.
    """
    t = text.strip()
    # strip code fences
    if t.startswith("```"):
        t = t.strip("`")
        if t.lower().startswith("json"):
            t = t[4:].lstrip()
    # direct parse
    try:
        return json.loads(t)
    except Exception:
        pass
    # attempt partial recovery: find last complete } or ]
    for end_char, start_char in [("}", "{"), ("]", "[")]:
        idx = t.rfind(end_char)
        while idx > 0:
            candidate = t[:idx + 1]
            # find matching open
            start = candidate.find(start_char)
            if start >= 0:
                try:
                    return json.loads(candidate[start:])
                except Exception:
                    pass
            idx = t.rfind(end_char, 0, idx)
    return None


def _make_log_entry(agent: str, event: str, payload: dict) -> dict:
    return {"agent": agent, "event": event, "payload": payload}


# ------------------------------------------------------------------
# 1. Discovery Agent
# ------------------------------------------------------------------

DISCOVERY_SYSTEM = """\
You are the Discovery Agent in a supply chain intelligence pipeline.
Given a target company and a corpus of source documents, your job is
to extract every supplier relationship the corpus contains.

Rules:
- Only extract relationships explicitly supported by a document. Never
  invent suppliers not named in the corpus.
- For each relationship, cite the exact source_url and a short verbatim
  excerpt from the document that supports it.
- Classify tier: 1 = direct supplier to the target company; 2 = supplier
  to a Tier 1; 3 = supplier to a Tier 2.
- Mark extraction_mode as "explicit" if the relationship is stated
  directly, or "inferred" if you deduced it from context.

Return STRICT JSON, no prose outside it:
{
  "edges": [
    {
      "parent": "<who is being supplied>",
      "supplier": "<the supplier>",
      "tier": <1|2|3>,
      "component": "<what is supplied>",
      "location": "<country/region or empty>",
      "source_url": "<the URL from the corpus>",
      "excerpt": "<short verbatim quote>",
      "extraction_mode": "explicit"|"inferred",
      "reasoning": "<one sentence>"
    }
  ]
}
"""


def discovery_node(state: PipelineState) -> dict:
    company = state["target_company"]
    corpus = state.get("corpus", [])
    if not corpus:
        trace = list(state.get("trace", []))
        trace.append(_make_log_entry("discovery", "empty_corpus", {"company": company}))
        return {"discovered_edges": [], "trace": trace}

    corpus_text = "\n\n".join(
        f"[{i}] source_url={d['source_url']}\n"
        f"    type={d['source_type']} authority={d['authority']} date={d['date']}\n"
        f"    title={d['title']}\n"
        f"    body={d['body']}"
        for i, d in enumerate(corpus)
    )
    user_msg = f"TARGET COMPANY: {company}\n\nCORPUS:\n{corpus_text}"

    client = _client(state["api_key"])
    resp = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=DISCOVERY_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )
    raw = "".join(b.text for b in resp.content if hasattr(b, "text"))
    data = _extract_json(raw) or {"edges": []}

    edges: list[SupplierEdge] = []
    by_url = {d["source_url"]: d for d in corpus}

    for e in data.get("edges", []):
        url = e.get("source_url", "")
        # fuzzy url match: if exact match fails, try substring match
        meta = by_url.get(url)
        if meta is None:
            for corpus_url, corpus_doc in by_url.items():
                if url in corpus_url or corpus_url in url:
                    meta = corpus_doc
                    break
        meta = meta or {}
        source = Source(
            url=url or meta.get("source_url", ""),
            source_type=meta.get("source_type", "unknown"),
            authority=float(meta.get("authority", 0.5)),
            date=meta.get("date", ""),
            excerpt=e.get("excerpt", ""),
            extraction_mode=e.get("extraction_mode", "explicit"),
        )
        edges.append(
            SupplierEdge(
                id=str(uuid.uuid4())[:8],
                parent=e.get("parent", ""),
                supplier=e.get("supplier", ""),
                tier=int(e.get("tier", 1)),
                component=e.get("component", ""),
                location=e.get("location", ""),
                sources=[source],
                discovery_reasoning=e.get("reasoning", ""),
            )
        )

    trace = list(state.get("trace", []))
    trace.append(_make_log_entry("discovery", "extracted", {
        "n_edges": len(edges),
        "parse_succeeded": bool(data.get("edges")),
        "raw_output_preview": raw[:400],
    }))
    return {"discovered_edges": edges, "trace": trace}


# ------------------------------------------------------------------
# 2. Verification Agent
# ------------------------------------------------------------------

VERIFICATION_SYSTEM = """\
You are the Verification Agent. For each supplier edge, decide whether
the corpus contains INDEPENDENT confirmation from a source OTHER than
the one that originally supported it.

Rules:
- If a second source with source_url different from the discovery
  source references the same parent-supplier relationship, mark
  "verified" and cite the second source.
- If a source only partially confirms (mentions the parties but not the
  relationship), mark "partial".
- If no other source mentions the relationship, mark "unverified".
- Never invent a confirming source. Only cite URLs present in the corpus.

Return STRICT JSON:
{
  "results": [
    {
      "edge_id": "<id>",
      "status": "verified"|"partial"|"unverified",
      "confirming_source_url": "<url or empty>",
      "confirming_excerpt": "<verbatim or empty>",
      "reasoning": "<one sentence>"
    }
  ]
}
"""


def verification_node(state: PipelineState) -> dict:
    edges = state.get("discovered_edges", [])
    corpus = state.get("corpus", [])
    if not edges:
        trace = list(state.get("trace", []))
        trace.append(_make_log_entry("verification", "no_edges", {}))
        return {"verified_edges": [], "trace": trace}

    edges_text = "\n".join(
        f"- id={e.id} parent={e.parent!r} supplier={e.supplier!r} "
        f"discovery_source={e.sources[0].url if e.sources else ''}"
        for e in edges
    )
    corpus_text = "\n\n".join(
        f"source_url={d['source_url']}\nbody={d['body']}" for d in corpus
    )
    user_msg = f"EDGES TO VERIFY:\n{edges_text}\n\nCORPUS:\n{corpus_text}"

    client = _client(state["api_key"])
    resp = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=VERIFICATION_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )
    raw = "".join(b.text for b in resp.content if hasattr(b, "text"))
    data = _extract_json(raw) or {"results": []}

    by_id = {e.id: e for e in edges}
    by_url = {d["source_url"]: d for d in corpus}
    for r in data.get("results", []):
        eid = r.get("edge_id")
        if eid not in by_id:
            continue
        edge = by_id[eid]
        edge.verification_status = r.get("status", "unverified")
        edge.verification_reasoning = r.get("reasoning", "")
        confirm_url = r.get("confirming_source_url", "")
        # fuzzy match for confirming url too
        confirm_meta = by_url.get(confirm_url)
        if confirm_meta is None and confirm_url:
            for cu, cd in by_url.items():
                if confirm_url in cu or cu in confirm_url:
                    confirm_meta = cd
                    break
        if confirm_meta and edge.verification_status == "verified":
            edge.sources.append(Source(
                url=confirm_url,
                source_type=confirm_meta.get("source_type", "unknown"),
                authority=float(confirm_meta.get("authority", 0.5)),
                date=confirm_meta.get("date", ""),
                excerpt=r.get("confirming_excerpt", ""),
                extraction_mode="explicit",
            ))

    trace = list(state.get("trace", []))
    trace.append(_make_log_entry("verification", "checked", {
        "n_edges": len(edges),
        "verified": sum(1 for e in edges if e.verification_status == "verified"),
        "partial": sum(1 for e in edges if e.verification_status == "partial"),
        "unverified": sum(1 for e in edges if e.verification_status == "unverified"),
    }))
    return {"verified_edges": edges, "trace": trace}


# ------------------------------------------------------------------
# 3. Uncertainty Agent (deterministic, no LLM)
# ------------------------------------------------------------------
# We do this deterministically. Uncertainty is a math problem, not a
# generative one. If we let an LLM pick the score, we cannot audit it.

UNCERTAINTY_WEIGHTS = {
    "source_count": 0.25,      # more independent sources => higher confidence
    "source_authority": 0.30,  # average authority of sources
    "verification": 0.25,      # verified > partial > unverified
    "extraction_mode": 0.10,   # explicit > inferred
    "recency": 0.10,           # more recent => slightly more confident
}


def _recency_score(date_str: str) -> float:
    """Very simple recency proxy: 1.0 if 2025+, 0.7 if 2024, 0.4 older."""
    if not date_str:
        return 0.5
    year = date_str[:4]
    try:
        y = int(year)
        if y >= 2025:
            return 1.0
        if y == 2024:
            return 0.7
        return 0.4
    except ValueError:
        return 0.5


def _verification_score(status: str) -> float:
    return {"verified": 1.0, "partial": 0.5, "unverified": 0.15}.get(status, 0.3)


def uncertainty_node(state: PipelineState) -> dict:
    edges = state.get("verified_edges", [])

    for edge in edges:
        if not edge.sources:
            edge.uncertainty_score = 0.0
            edge.uncertainty_breakdown = {"reason": "no sources"}
            continue

        # source_count: cap at 3 for the score
        sc_raw = min(len(edge.sources), 3)
        source_count = sc_raw / 3.0

        # source_authority: mean authority across sources
        source_authority = sum(s.authority for s in edge.sources) / len(edge.sources)

        # verification
        verification = _verification_score(edge.verification_status)

        # extraction_mode: fraction of sources that were explicit
        explicit = sum(1 for s in edge.sources if s.extraction_mode == "explicit")
        extraction_mode = explicit / len(edge.sources)

        # recency: max recency across sources
        recency = max(_recency_score(s.date) for s in edge.sources)

        w = UNCERTAINTY_WEIGHTS
        score = (
            source_count * w["source_count"]
            + source_authority * w["source_authority"]
            + verification * w["verification"]
            + extraction_mode * w["extraction_mode"]
            + recency * w["recency"]
        )
        edge.uncertainty_score = round(score, 3)
        edge.uncertainty_breakdown = {
            "source_count_signal": round(source_count, 3),
            "source_authority_signal": round(source_authority, 3),
            "verification_signal": round(verification, 3),
            "extraction_mode_signal": round(extraction_mode, 3),
            "recency_signal": round(recency, 3),
            "weights": dict(w),
            "formula": (
                "score = source_count*0.25 + authority*0.30 + verification*0.25 "
                "+ extraction_mode*0.10 + recency*0.10"
            ),
        }

    trace = list(state.get("trace", []))
    trace.append(_make_log_entry("uncertainty", "scored", {
        "n_edges": len(edges),
        "mean_score": round(sum(e.uncertainty_score for e in edges) / max(len(edges), 1), 3),
    }))
    return {"scored_edges": edges, "trace": trace}


# ------------------------------------------------------------------
# 4. Aggregation Agent (deterministic concentration analytics)
# ------------------------------------------------------------------
# Also deterministic. Concentration risk is graph math.

def aggregation_node(state: PipelineState) -> dict:
    # This node does not modify edges. It exists so the graph has a
    # clean terminal node and so we can trace that aggregation happened.
    edges = state.get("scored_edges", [])
    trace = list(state.get("trace", []))
    trace.append(_make_log_entry("aggregation", "complete", {"total_edges": len(edges)}))
    return {"trace": trace}
