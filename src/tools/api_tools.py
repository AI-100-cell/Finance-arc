

"""External API calls (Concept #7) — and the three ingestion data sources (Concept #8).

Three distinct sources, all eventually chunked + embedded into one ChromaDB store:
  1. Earnings call transcripts  -> fetch_transcript()   (Financial Modeling Prep)
  2. SEC EDGAR filings          -> fetch_sec_filing()    (EDGAR full-text search, no key)
  3. Live market data           -> fetch_price_data()    (Yahoo Finance via yfinance)
"""

import os
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

# Network defaults
TIMEOUT = 30  # seconds
# SEC requires a descriptive User-Agent with a contact address, or it returns 403.
SEC_HEADERS = {"User-Agent": "earnings-intelligence/1.0 (kanwalwaheed24@gmail.com)"}


# ---------------------------------------------------------------------------
# Source 1 — Earnings call transcripts (Financial Modeling Prep)
# ---------------------------------------------------------------------------
def fetch_transcript(ticker: str, year: int, quarter: int) -> str:
    """Return the full text of an earnings call transcript, or '' if unavailable.

    Requires FMP_API_KEY in the environment (.env). Free tier:
    https://financialmodelingprep.com/developer/docs
    """
    api_key = os.getenv("FMP_API_KEY")
    if not api_key:
        raise RuntimeError("FMP_API_KEY is not set — add it to your .env file.")

    url = (
        "https://financialmodelingprep.com/api/v3/"
        f"earning_call_transcript/{ticker.upper()}"
    )
    params = {"quarter": quarter, "year": year, "apikey": api_key}

    resp = requests.get(url, params=params, timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list) and data:
        return data[0].get("content", "")
    return ""


# ---------------------------------------------------------------------------
# Source 2 — SEC EDGAR filings (free, no key)
# ---------------------------------------------------------------------------
# We use EDGAR's canonical path rather than the full-text search snippet: the
# FTS endpoint isn't sorted by recency and often surfaces XBRL fragments/exhibits
# instead of the primary document. ticker -> CIK -> submissions API reliably
# yields the *latest primary* filing of a given form type.
def _resolve_cik(ticker: str) -> str | None:
    """Map a ticker symbol to its zero-padded 10-digit SEC CIK, or None."""
    resp = requests.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers=SEC_HEADERS,
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    ticker = ticker.upper()
    for entry in resp.json().values():
        if entry.get("ticker", "").upper() == ticker:
            return f"{int(entry['cik_str']):010d}"
    return None


def find_latest_filing(ticker: str, form_type: str = "10-K") -> dict | None:
    """Return metadata for the most recent filing of `form_type`, or None.

    Keys: form, file_date, accession, document, url.
    """
    cik = _resolve_cik(ticker)
    if not cik:
        return None

    resp = requests.get(
        f"https://data.sec.gov/submissions/CIK{cik}.json",
        headers=SEC_HEADERS,
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    recent = resp.json().get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    cik_int = int(cik)  # the Archives path uses the un-padded CIK

    for i, form in enumerate(forms):
        if form == form_type:
            accession = recent["accessionNumber"][i]
            document = recent["primaryDocument"][i]
            accession_nodashes = accession.replace("-", "")
            return {
                "form": form,
                "file_date": recent["filingDate"][i],
                "accession": accession,
                "document": document,
                "url": (
                    f"https://www.sec.gov/Archives/edgar/data/{cik_int}/"
                    f"{accession_nodashes}/{document}"
                ),
            }
    return None


def fetch_sec_filing(ticker: str, form_type: str = "10-K") -> str:
    """Return the plain text of the most recent matching SEC filing, or '' if none.

    (The bootcamp snippet only returned a description and a malformed URL; this
    resolves and downloads the actual primary filing document for ingestion.)
    """
    filing = find_latest_filing(ticker, form_type=form_type)
    if not filing:
        return ""

    resp = requests.get(filing["url"], headers=SEC_HEADERS, timeout=TIMEOUT)
    resp.raise_for_status()

    # Lazy import so the module loads even if bs4 isn't present.
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(resp.text, "html.parser")
    return soup.get_text(separator="\n", strip=True)


# ---------------------------------------------------------------------------
# Source 3 — Live market data (Yahoo Finance via yfinance)
# ---------------------------------------------------------------------------
def fetch_price_data(ticker: str, period: str = "1y") -> dict[str, Any]:
    """Return key quote fields plus the last 30 closing prices.

    NOTE: Yahoo's quote-summary endpoint (yf .info) is frequently rate-limited
    (HTTP 429). This degrades gracefully: price history usually still works even
    when .info does not, so missing fields come back as None rather than raising.
    """
    import yfinance as yf

    stock = yf.Ticker(ticker.upper())

    info: dict = {}
    try:
        info = stock.info or {}
    except Exception:  # noqa: BLE001 — Yahoo 429 / transient JSON errors
        info = {}

    history: dict = {}
    try:
        hist = stock.history(period=period)
        if not hist.empty and "Close" in hist.columns:
            history = hist["Close"].tail(30).to_dict()
    except Exception:  # noqa: BLE001
        history = {}

    return {
        "current_price": info.get("currentPrice"),
        "market_cap": info.get("marketCap"),
        "pe_ratio": info.get("trailingPE"),
        "revenue_growth": info.get("revenueGrowth"),
        "history": history,
    }


if __name__ == "__main__":
    # Quick smoke test. FMP needs a key; SEC + Yahoo are keyless.
    print("Latest AAPL 10-K:")
    filing = find_latest_filing("AAPL", "10-K")
    print(" ", filing)
    if filing:
        print("  text length:", len(fetch_sec_filing("AAPL", "10-K")))

    print("\nPrice data for AAPL:")
    print(" ", fetch_price_data("AAPL", period="1mo"))
