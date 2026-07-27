# CLAUDE.md

Instructions for any Claude instance working on this repository.

## What this project is

Tier Deep is a multi-agent supplier graph discovery pipeline. It takes a
target company, runs a four-agent LangGraph pipeline over a corpus of
public supply chain sources, and produces a supplier network with a
calibrated uncertainty score on every edge. Every score is auditable.

## Ground rules

1. **Never invent suppliers.** The Discovery Agent only extracts what a
   source explicitly supports. Hallucinated relationships are the exact
   failure mode this project exists to prevent.
2. **Uncertainty is deterministic.** The Uncertainty Agent is not an LLM.
   It is math. Do not replace it with a model.
3. **Transparency over convenience.** Every edge shows its sources, its
   verification history, its uncertainty breakdown, and the exact formula
   used. Do not hide intermediate values.
4. **No emojis anywhere.** UI, code comments, docs, commit messages.
5. **Model choice is Haiku 4.5.** The tasks are extraction and
   verification, not deep reasoning. Latency and cost matter.

## Architecture

```
app.py                          Streamlit dashboard
src/models.py                   Source, SupplierEdge, ConcentrationRisk
src/data_sources/curated.py     Curated Tesla corpus from public disclosures
src/data_sources/sec_edgar.py   Live SEC EDGAR full-text search
src/agents.py                   Four agent functions + shared helpers
src/graph.py                    LangGraph orchestration
src/analytics.py                Deterministic concentration risk detection
```

## Pipeline

```
START -> discovery -> verification -> uncertainty -> aggregation -> END
```

- Discovery Agent (LLM): extracts explicit supplier relationships from
  the corpus. Returns strict JSON with source citations.
- Verification Agent (LLM): for each edge, tries to find an independent
  second source in the corpus. Marks verified / partial / unverified.
- Uncertainty Agent (deterministic): composite score across five signals
  (source count, authority, verification, extraction mode, recency).
  Weights live in one place and are shown in the UI.
- Aggregation Agent (deterministic): passthrough that ensures the
  terminal node in the graph is logged. Concentration analytics run
  outside the graph in `analytics.py`.

## When making changes

- Adding a new data source: implement it under `src/data_sources/` with
  the same `{source_url, source_type, authority, date, title, body}`
  schema and add it to the corpus that gets passed to Discovery.
- Adding a signal to the uncertainty score: update `UNCERTAINTY_WEIGHTS`
  and the transparency panel in `app.py` in the same commit. The formula
  shown to the user must match the code.
- Adding a new risk type to `analytics.py`: give it a `risk_type` string,
  a severity function, and an evidence field. Keep it deterministic.
- Do not add persistence. API keys are session-only.

## What NOT to do

- Do not let the Uncertainty Agent be an LLM. It must be math.
- Do not let Discovery invent suppliers not in the corpus. If in doubt,
  return an empty edge list.
- Do not silently retry failed LLM calls. Surface the failure.
- Do not add "confidence" scores that are not derived from the five
  base signals.
- Do not remove the trace log. It is the whole point.
