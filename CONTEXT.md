# CONTEXT.md

Background, rationale, and evidence behind Tier Deep.

## The problem

Modern supply chains break in the tiers nobody can see. 93 percent of
executives report high confidence in their overall supplier oversight,
while the same group identifies Tier 2 and Tier 3 suppliers as their
most critical operational blind spot. Just 17.7 percent of semiconductor
professionals are extremely confident in Tier 2 and Tier 3 visibility.

The failure is architectural. Standard supplier relationship management
systems stop at the purchase order boundary. Your direct supplier will
not tell you who their suppliers are; sub-tier networks are treated as
proprietary. So the only way to map Tier 2 and Tier 3 is to infer it
from public sources: SEC filings, customs records, news, patents,
corporate registries.

## Why this is hard

Inference invites hallucination. If you point a single LLM at a corpus
of supply chain news and ask it to build a supplier graph, it will
happily invent relationships that sound plausible but do not exist. The
Helicase paper (arXiv 2605.26835, 2026) frames this precisely:

> Agentic LLMs can drive end-to-end supply chain discovery, but only
> when equipped with explicit, multi-layer uncertainty quantification
> to contain hallucinations and calibrate downstream trust.

That is the gap Tier Deep addresses.

## The approach

Four agents. Two are LLM-driven and produce structured claims with
citations. Two are deterministic and produce math the auditor can
verify.

1. **Discovery Agent (LLM).** Extracts supplier relationships from the
   corpus with source URLs and verbatim excerpts. Refuses to name
   suppliers not present in the corpus.
2. **Verification Agent (LLM).** For each claim, looks for an independent
   second source. Marks verified, partial, or unverified.
3. **Uncertainty Agent (deterministic).** Weighted composite over five
   signals: source count, source authority, verification result,
   extraction mode (explicit vs inferred), recency.
4. **Aggregation Agent (deterministic).** Terminates the graph and
   triggers analytics.

Concentration risk analytics (shared Tier 2, geographic clustering,
single source, low confidence) run over the verified graph and produce
concrete findings a procurement lead would act on.

## Why Tesla

Tesla's supplier network is publicly documented at unusual depth. Its
10-K filings name specific suppliers. Its battery partners (Panasonic,
CATL, LG Energy Solution, BYD) are public. Its chip suppliers (Nvidia,
Samsung, Infineon, STMicro, Renesas) are on the record. Its lithium
contracts (Albemarle, Ganfeng, SQM) are reported. This means the demo
can be checked against reality.

The curated Tesla corpus in `src/data_sources/curated.py` deliberately
includes one weak entry - a low-authority blog post claiming an unnamed
European graphene supplier. The Verification Agent has nothing to
cross-reference it against, so the Uncertainty Agent scores it low,
and it surfaces at the top of the low-confidence list. This is the
system working as designed.

## What Tier Deep does NOT claim

- Real-time monitoring. Building the graph takes seconds to minutes.
  Production monitoring would need incremental update infrastructure.
- Complete coverage. Public trade data is US-inbound only. Some Tier 2
  relationships are invisible in every public source.
- Ground-truth-calibrated uncertainty. The signals are principled but
  the weights are hand-tuned. Real calibration would require a labeled
  benchmark dataset.
- Replacement for enterprise vendors. Resilinc, Interos, Sayari, and
  Altana have proprietary data feeds Tier Deep does not have access to.
  This is a proof of concept for the transparency angle, not a
  competing product.

## References

- Helicase paper: https://arxiv.org/pdf/2605.26835
- Jannelli et al. (2025) Agentic LLMs in the Supply Chain: https://arxiv.org/abs/2411.10184
- Tradeverifyd 2026 statistics: https://tradeverifyd.com/resources/supply-chain-statistics
- JAGGAER on sub-tier visibility: https://www.jaggaer.com/blog/why-supply-chain-visibility-breaks-down-beyond-tier-1-in-2026
- Trade Finance Global on the March 2026 crisis: https://www.tradefinanceglobal.com/posts/beyond-tier-1-closing-the-supply-chain-visibility-gap/
- SEC EDGAR: https://www.sec.gov/edgar
- Tesla 10-K filings: https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0001318605&type=10-K
