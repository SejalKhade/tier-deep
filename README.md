# Tier Deep

Multi-agent supplier graph discovery with per-edge uncertainty
quantification. Built on LangGraph and Claude. Anchored on Tesla for
a runnable, verifiable demo.

## Setup

```
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501`. Paste your Anthropic key in the sidebar.
Nothing is written to disk.

## Demo flow

1. Keep the default target (`Tesla, Inc.`) and mode (`Curated corpus`).
2. Click **Run four-agent pipeline**. Takes 20 to 60 seconds.
3. Scroll to **Agent trace**. Every step every agent took is logged.
4. **Supplier graph** shows edges sorted lowest-confidence first, so
   the graphene rumor floats to the top. Expand it: the Verification
   Agent found no confirming source, the recency signal is weak, the
   authority is a blog. Score reflects that.
5. **Concentration risk findings** surface the shared-Tier-2 chokepoints
   (Sumitomo cathode, TSMC for Nvidia).
6. Download the **JSON audit report** and hand it to a stakeholder.

## Architecture

```
START -> discovery -> verification -> uncertainty -> aggregation -> END
```

- Discovery Agent (Claude Haiku): extracts explicit supplier
  relationships with source citations.
- Verification Agent (Claude Haiku): cross-references each claim
  against a second source in the corpus.
- Uncertainty Agent (deterministic): weighted composite over five
  signals. Formula visible in the UI.
- Aggregation Agent (deterministic): terminal node.

Concentration analytics run over the verified graph and detect shared
Tier 2 dependencies, geographic clusters, single-source components,
and low-confidence relationships.

## Modes

- **Curated (default)**: Tesla-specific corpus built from real 10-K
  filings, major news, and public disclosures. Reproducible.
- **Live**: SEC EDGAR full-text search and web sources. Slower and
  non-deterministic.

## Uncertainty formula

```
score = source_count * 0.25
      + source_authority * 0.30
      + verification * 0.25
      + extraction_mode * 0.10
      + recency * 0.10
```

Every signal is bounded 0 to 1. Every weight is declared in one place
(`agents.py:UNCERTAINTY_WEIGHTS`) and echoed in the UI.

## What this does not do

- Not real-time. Building a graph takes seconds to minutes.
- US-inbound customs only. Non-US flows are not covered.
- Not calibrated against a labeled benchmark. Weights are principled
  but hand-tuned. Real calibration is future work.
