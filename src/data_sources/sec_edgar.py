"""
Thin wrapper for SEC EDGAR full-text search.

EDGAR is free, no API key required. We use the full-text search endpoint
to pull recent 10-K, 10-Q, and 8-K filings for a target company and its
suspected suppliers. Rate limit: SEC asks for a User-Agent header
identifying the caller. Be a good citizen.
"""
from __future__ import annotations

import requests

EDGAR_UA = "TierDeep research prototype demo@example.com"
EDGAR_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"


def search_filings(query: str, max_results: int = 5) -> list[dict]:
    """
    Full-text search across EDGAR. Returns a list of hits with
    accession numbers, filing dates, and the entity that filed.
    """
    params = {"q": f'"{query}"', "dateRange": "custom", "forms": "10-K,10-Q,8-K"}
    headers = {"User-Agent": EDGAR_UA}
    try:
        r = requests.get(EDGAR_SEARCH_URL, params=params, headers=headers, timeout=15)
        r.raise_for_status()
        data = r.json()
        hits = data.get("hits", {}).get("hits", [])[:max_results]
        return [
            {
                "url": f"https://www.sec.gov/Archives/edgar/data/{h['_source'].get('ciks', ['?'])[0]}/{h['_source'].get('adsh', '').replace('-', '')}/{h['_source'].get('adsh', '')}-index.htm",
                "source_type": "sec_filing",
                "authority": 1.0,
                "date": h["_source"].get("file_date", ""),
                "title": h["_source"].get("display_names", ["?"])[0],
                "body": h["_source"].get("_id", ""),
            }
            for h in hits
        ]
    except Exception:
        return []
