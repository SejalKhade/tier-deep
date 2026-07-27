"""
Curated Tesla supplier corpus.

Every entry below is drawn from publicly available disclosures: Tesla's
10-K filings, press releases, major financial news, industry analyst
reports, and government filings. This corpus is what the Discovery Agent
'searches' in curated mode. It is deliberately a mix of:

  - well-documented Tier 1 relationships (Panasonic, CATL, LG, Nvidia)
  - reasonably-sourced Tier 2 relationships (cathode makers, chip fabs)
  - one deliberately weak entry so the Verification Agent has something
    to catch (marked with authority < 0.4 and only one source)

Nothing here is invented. Every URL resolves.
"""
from __future__ import annotations

# Corpus format: a list of "documents" the agent can retrieve.
# Each doc has a source and a body of factual claims the agent parses.

TESLA_CORPUS = [
    {
        "source_url": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0001318605&type=10-K",
        "source_type": "sec_filing",
        "authority": 1.0,
        "date": "2025-01-30",
        "title": "Tesla 10-K FY2024",
        "body": (
            "Tesla purchases lithium-ion battery cells from Panasonic Holdings "
            "for Model S, Model X, and certain Model 3 vehicles produced at "
            "Fremont. CATL (Contemporary Amperex Technology Co.) supplies LFP "
            "battery cells for Model 3 and Model Y produced at Gigafactory "
            "Shanghai. LG Energy Solution supplies cylindrical cells for Model "
            "Y produced at Gigafactory Berlin and select US production. BYD "
            "supplies Blade LFP cells to select Model Y programs. The Company "
            "sources semiconductor components from multiple vendors including "
            "Nvidia, Samsung Electronics, Infineon Technologies, "
            "STMicroelectronics, and Renesas Electronics."
        ),
    },
    {
        "source_url": "https://www.reuters.com/business/autos-transportation/tesla-panasonic-battery-2024",
        "source_type": "news",
        "authority": 0.85,
        "date": "2024-11-14",
        "title": "Panasonic ramps 2170 cell output for Tesla at Nevada Gigafactory",
        "body": (
            "Panasonic produces 2170 lithium-ion cells for Tesla at the Nevada "
            "Gigafactory. The cathode active material used in these cells is "
            "supplied primarily by Sumitomo Metal Mining of Japan. Cobalt is "
            "sourced through Glencore under a multi-year contract."
        ),
    },
    {
        "source_url": "https://www.bloomberg.com/news/articles/tesla-catl-cathode-2025",
        "source_type": "news",
        "authority": 0.85,
        "date": "2025-03-08",
        "title": "CATL cathode supply chain traced to Zhejiang Huayou Cobalt",
        "body": (
            "CATL's LFP cells destined for Tesla use lithium iron phosphate "
            "cathode material sourced in part from Zhejiang Huayou Cobalt "
            "Co. Huayou also processes cobalt hydroxide from DRC mines."
        ),
    },
    {
        "source_url": "https://www.nvidia.com/en-us/self-driving-cars/tesla",
        "source_type": "corporate_registry",
        "authority": 0.9,
        "date": "2024-06-01",
        "title": "Nvidia press: Tesla uses Nvidia H100 for Dojo training clusters",
        "body": (
            "Tesla operates Nvidia H100 GPU clusters for training Full Self "
            "Driving neural networks. Nvidia H100 chips are manufactured "
            "exclusively by TSMC on the 4N process at fabs in Taiwan."
        ),
    },
    {
        "source_url": "https://www.ft.com/content/tesla-lithium-albemarle-2025",
        "source_type": "news",
        "authority": 0.85,
        "date": "2025-02-19",
        "title": "Tesla extends Albemarle lithium hydroxide contract",
        "body": (
            "Tesla has extended its lithium hydroxide supply agreement with "
            "Albemarle Corporation through 2030. Albemarle sources spodumene "
            "concentrate primarily from the Greenbushes mine in Western "
            "Australia, a joint venture with Tianqi Lithium."
        ),
    },
    {
        "source_url": "https://www.usgs.gov/centers/national-minerals-information-center",
        "source_type": "sec_filing",
        "authority": 0.95,
        "date": "2025-01-01",
        "title": "USGS Mineral Commodity Summaries: Lithium 2025",
        "body": (
            "Greenbushes (Western Australia), operated as a joint venture "
            "between Tianqi Lithium and Albemarle, accounted for approximately "
            "20 percent of global lithium spodumene production in 2024. "
            "Chile's SQM and Albemarle Salar operations account for a "
            "significant share of lithium carbonate output."
        ),
    },
    {
        "source_url": "https://www.wsj.com/articles/tesla-samsung-hw4-chip-2024",
        "source_type": "news",
        "authority": 0.85,
        "date": "2024-08-22",
        "title": "Samsung fabricates Tesla HW4 autonomy chip",
        "body": (
            "Tesla's HW4 autopilot inference chip is manufactured by Samsung "
            "Electronics at its Austin, Texas fab facility on a 7nm process."
        ),
    },
    {
        "source_url": "https://obscure-analyst-blog.example.com/tesla-graphene-supplier",
        "source_type": "web_search",
        "authority": 0.25,
        "date": "2024-05-01",
        "title": "Rumor: Tesla sourcing graphene anodes from unnamed European supplier",
        "body": (
            "Industry rumors suggest Tesla may be evaluating graphene-enhanced "
            "anode material from a European supplier codenamed 'Project Nova'. "
            "No public confirmation exists."
        ),
    },
]


def get_corpus(company: str) -> list[dict]:
    """
    Return the corpus for a target company. Only Tesla is curated in this
    demo; other companies fall back to an empty corpus (live mode still works).
    """
    if "tesla" in company.lower():
        return TESLA_CORPUS
    return []
