"""
Tier Deep - Streamlit dashboard.

Run: streamlit run app.py
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict

import pandas as pd
import streamlit as st

from src.data_sources.curated import get_corpus, TESLA_CORPUS
from src.graph import run_pipeline
from src.analytics import compute_all_risks
from src.models import edge_to_dict


# --------------------------------------------------------------------------
# Page config + professional styling. No emojis.
# --------------------------------------------------------------------------

st.set_page_config(
    page_title="Tier Deep - Supplier Graph Discovery",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #14161a;
        --ink-soft: #4a4f57;
        --line: #d9dbe0;
        --bg-panel: #f7f8fa;
        --accent: #1f3a68;
    }
    html, body, [class*="css"] {
        font-family: "Inter", "Helvetica Neue", Arial, sans-serif;
        color: var(--ink);
    }
    h1 { font-weight: 600; font-size: 1.9rem; letter-spacing: -0.01em; }
    h2 { font-weight: 600; font-size: 1.25rem; margin-top: 1.2rem; }
    h3 { font-weight: 600; font-size: 1.05rem; color: var(--ink-soft); }
    .metric-card {
        border: 1px solid var(--line);
        border-radius: 6px;
        padding: 14px 16px;
        background: var(--bg-panel);
    }
    .metric-label { font-size: 0.78rem; color: var(--ink-soft); text-transform: uppercase; letter-spacing: 0.06em; }
    .metric-value { font-size: 1.4rem; font-weight: 600; color: var(--ink); margin-top: 4px; }
    .edge-box {
        border: 1px solid var(--line);
        border-left: 3px solid var(--accent);
        border-radius: 4px;
        padding: 10px 14px;
        margin-bottom: 8px;
        background: white;
    }
    .risk-high { border-left: 3px solid #991b1b; }
    .risk-medium { border-left: 3px solid #a16207; }
    .risk-low { border-left: 3px solid #4d7c0f; }
    .agent-step {
        background: #f0f2f6;
        border-radius: 4px;
        padding: 8px 12px;
        margin-bottom: 6px;
        font-family: "SF Mono", "Monaco", monospace;
        font-size: 0.85rem;
    }
    code { font-size: 0.85em; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### Configuration")
    anthropic_key = st.text_input(
        "Anthropic API Key",
        type="password",
        help="Required. Used by Discovery and Verification agents. Not stored.",
    )
    st.markdown("---")
    st.markdown("### Target Company")
    target = st.text_input("Company name", value="Tesla, Inc.")
    mode = st.radio(
        "Data source",
        ["Curated corpus (Tesla, reproducible)", "Live mode (SEC + web, non-deterministic)"],
        index=0,
    )
    st.markdown("---")
    confidence_threshold = st.slider(
        "Low-confidence threshold",
        0.0, 1.0, 0.5, 0.05,
        help="Edges with uncertainty score below this are flagged as low-confidence.",
    )


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------

st.markdown("# Tier Deep")
st.markdown(
    "A multi-agent supplier graph discovery pipeline with per-edge uncertainty "
    "quantification. Built on LangGraph and Claude. Every score has a formula. "
    "Every claimed relationship shows its sources, its verification history, "
    "and its uncertainty breakdown."
)

with st.expander("Why this exists", expanded=False):
    st.markdown(
        "- Tier 2 and Tier 3 supplier visibility is the largest documented "
        "blind spot in modern supply chains. 93 percent of executives claim "
        "confidence in overall oversight while flagging deep tiers as their "
        "top operational blind spot."
    )
    st.markdown(
        "- Enterprise vendors (Resilinc, Interos, Sayari) map sub-tiers but do "
        "not expose per-edge uncertainty. Academic work (Helicase et al., 2026) "
        "identifies uncertainty quantification as the frontier."
    )
    st.markdown(
        "- Tier Deep is a runnable demonstration: multi-agent discovery, "
        "cross-source verification, deterministic uncertainty scoring, and a "
        "transparent audit trail per relationship."
    )


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------

ss = st.session_state
ss.setdefault("result", None)


# --------------------------------------------------------------------------
# Run pipeline
# --------------------------------------------------------------------------

st.markdown("## 1. Run pipeline")

corpus = get_corpus(target) if mode.startswith("Curated") else []

col1, col2, col3 = st.columns(3)
col1.markdown(
    f"<div class='metric-card'><div class='metric-label'>Target</div>"
    f"<div class='metric-value'>{target}</div></div>", unsafe_allow_html=True,
)
col2.markdown(
    f"<div class='metric-card'><div class='metric-label'>Corpus size</div>"
    f"<div class='metric-value'>{len(corpus)} docs</div></div>", unsafe_allow_html=True,
)
col3.markdown(
    f"<div class='metric-card'><div class='metric-label'>Mode</div>"
    f"<div class='metric-value'>{'Curated' if corpus else 'Live'}</div></div>",
    unsafe_allow_html=True,
)

if mode.startswith("Curated") and not corpus:
    st.warning(
        f"No curated corpus for '{target}'. Only Tesla is curated in this demo. "
        f"Switch to Live mode or set target to 'Tesla, Inc.'"
    )

if st.button("Run four-agent pipeline", type="primary"):
    if not anthropic_key:
        st.error("Anthropic API key required.")
    elif mode.startswith("Curated") and not corpus:
        st.error("Load a corpus or switch to Live mode.")
    else:
        with st.spinner("Running LangGraph pipeline: discovery -> verification -> uncertainty -> aggregation"):
            t0 = time.time()
            try:
                result = run_pipeline(
                    api_key=anthropic_key,
                    target_company=target,
                    corpus=corpus,
                    use_web_search=(mode.startswith("Live")),
                )
                elapsed = time.time() - t0
                ss.result = result
                st.success(f"Pipeline complete in {elapsed:.1f}s.")
            except Exception as e:
                st.error(f"Pipeline failed: {e}")
                import traceback
                st.code(traceback.format_exc())


# --------------------------------------------------------------------------
# Agent trace
# --------------------------------------------------------------------------

if ss.result:
    st.markdown("## 2. Agent trace")
    st.caption(
        "Every step every agent took, in order. This is the audit log a "
        "procurement team would review before acting on any finding."
    )
    for event in ss.result.get("trace", []):
        st.markdown(
            f"<div class='agent-step'>"
            f"<b>{event['agent']}</b> :: {event['event']} :: "
            f"{json.dumps(event['payload'], indent=None)[:200]}"
            f"</div>",
            unsafe_allow_html=True,
        )


# --------------------------------------------------------------------------
# Supplier graph
# --------------------------------------------------------------------------

if ss.result:
    edges = ss.result.get("scored_edges", [])
    st.markdown("## 3. Supplier graph")

    if not edges:
        st.info("No supplier edges were discovered. Check the trace above.")
    else:
        # summary metrics
        n_tier1 = sum(1 for e in edges if e.tier == 1)
        n_tier2 = sum(1 for e in edges if e.tier == 2)
        n_tier3 = sum(1 for e in edges if e.tier == 3)
        mean_conf = sum(e.uncertainty_score for e in edges) / len(edges)
        n_verified = sum(1 for e in edges if e.verification_status == "verified")

        cols = st.columns(5)
        for col, label, val in zip(
            cols,
            ["Total edges", "Tier 1", "Tier 2 / 3", "Verified", "Mean confidence"],
            [len(edges), n_tier1, n_tier2 + n_tier3, n_verified, round(mean_conf, 2)],
        ):
            col.markdown(
                f"<div class='metric-card'><div class='metric-label'>{label}</div>"
                f"<div class='metric-value'>{val}</div></div>",
                unsafe_allow_html=True,
            )

        st.markdown("### Edges (sorted by confidence, ascending)")
        st.caption("Lowest-confidence edges first, because those are where you look for hallucinations.")
        for e in sorted(edges, key=lambda x: x.uncertainty_score):
            status_color = {
                "verified": "#14532d",
                "partial": "#78350f",
                "unverified": "#7f1d1d",
                "unchecked": "#4a4f57",
            }.get(e.verification_status, "#4a4f57")
            st.markdown(
                f"<div class='edge-box'>"
                f"<b>{e.parent}</b> &lt;-- <b>{e.supplier}</b> "
                f"&nbsp;&nbsp;<span style='color:#4a4f57'>tier {e.tier} / {e.component or 'unspecified'}</span>"
                f"<br/>"
                f"<span style='color:{status_color}'>{e.verification_status.upper()}</span> "
                f"&nbsp;&nbsp;confidence={e.uncertainty_score} "
                f"&nbsp;&nbsp;sources={len(e.sources)}"
                f"<br/><span style='font-size:0.8em;color:#4a4f57'>id={e.id}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

            with st.expander(f"Transparency: how {e.id} was scored"):
                st.markdown("**Step 1 - Discovery Agent**")
                st.code(f"reasoning: {e.discovery_reasoning}")

                st.markdown("**Step 2 - Sources**")
                for i, s in enumerate(e.sources):
                    st.markdown(
                        f"- Source {i+1}: `{s.source_type}` "
                        f"authority={s.authority} date={s.date} mode={s.extraction_mode}"
                    )
                    st.markdown(f"  - URL: {s.url}")
                    st.markdown(f"  - Excerpt: _{s.excerpt}_")

                st.markdown("**Step 3 - Verification Agent**")
                st.code(
                    f"status: {e.verification_status}\n"
                    f"reasoning: {e.verification_reasoning}"
                )

                st.markdown("**Step 4 - Uncertainty Agent (deterministic)**")
                b = e.uncertainty_breakdown
                if b:
                    st.markdown(
                        f"- source_count_signal: {b.get('source_count_signal')}\n"
                        f"- source_authority_signal: {b.get('source_authority_signal')}\n"
                        f"- verification_signal: {b.get('verification_signal')}\n"
                        f"- extraction_mode_signal: {b.get('extraction_mode_signal')}\n"
                        f"- recency_signal: {b.get('recency_signal')}"
                    )
                    st.code(b.get("formula", ""))
                    st.markdown(f"**Final confidence: {e.uncertainty_score}**")


# --------------------------------------------------------------------------
# Concentration risks
# --------------------------------------------------------------------------

if ss.result and ss.result.get("scored_edges"):
    st.markdown("## 4. Concentration risk findings")
    st.caption(
        "Deterministic graph analytics over the verified supplier network. "
        "These are the concrete facts a procurement lead would act on."
    )
    edges = ss.result["scored_edges"]
    risks = compute_all_risks(edges)

    if not risks:
        st.info("No concentration risks detected in the current graph.")
    else:
        for r in risks:
            cls = f"risk-{r.severity}"
            st.markdown(
                f"<div class='edge-box {cls}'>"
                f"<b>{r.risk_type.replace('_',' ').title()}</b> "
                f"&nbsp;&nbsp;<span style='color:#4a4f57'>severity: {r.severity}</span>"
                f"<br/>{r.description}"
                f"<br/><span style='font-size:0.8em;color:#4a4f57'>"
                f"affected_edges={r.affected_edges} evidence={r.evidence}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )


# --------------------------------------------------------------------------
# Report download
# --------------------------------------------------------------------------

if ss.result and ss.result.get("scored_edges"):
    st.markdown("## 5. Audit report")
    edges = ss.result["scored_edges"]
    risks = compute_all_risks(edges)
    report = {
        "target_company": target,
        "mode": mode,
        "edges": [edge_to_dict(e) for e in edges],
        "risks": [asdict(r) for r in risks],
        "trace": ss.result.get("trace", []),
    }
    st.download_button(
        "Download full JSON audit report",
        data=json.dumps(report, indent=2, default=str),
        file_name=f"tier_deep_{target.replace(',', '').replace(' ', '_')}.json",
        mime="application/json",
    )

st.markdown("---")
st.caption(
    "Tier Deep makes no supplier claim it cannot cite. Every score is "
    "computed from the inputs shown. Non-deterministic model output that "
    "cannot be parsed falls back to conservative defaults."
)
