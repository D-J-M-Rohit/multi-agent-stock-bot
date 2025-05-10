# tools/data_retrieval.py
"""
Light-weight data utilities used by the agents
---------------------------------------------

▪ get_stock_price(ticker)            – current price + % change
▪ get_financial_statements(ticker)   – revenue & income summary
▪ get_recent_news(query)             – headlines for *ticker **or** company name*
▪ get_market_summary()               – major–index snapshot
"""

from __future__ import annotations

import os, time, requests
from datetime import datetime, timezone
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()                       # lets you keep keys in a .env file
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY") or ""


# ────────────────────────────────────────────────────────────────────────────
#  Helpers
# ────────────────────────────────────────────────────────────────────────────
def _lookup_symbol(name: str) -> str | None:
    """
    Convert a company name (e.g. 'Apple') to its primary ticker ('AAPL')
    using Yahoo's public search API. Returns None if nothing obvious found.
    """
    url = "https://query1.finance.yahoo.com/v1/finance/search"
    try:
        res = requests.get(url, params={"q": name, "quotesCount": 1, "newsCount": 0}, timeout=4)
        res.raise_for_status()
        quotes = res.json().get("quotes", [])
        if quotes:
            return quotes[0].get("symbol")
    except Exception:
        pass
    return None


# ────────────────────────────────────────────────────────────────────────────
#  Stock price + basic financials
# ────────────────────────────────────────────────────────────────────────────
def get_stock_price(ticker_symbol: str) -> str:
    t = yf.Ticker(ticker_symbol)
    price = change_pct = None

    # fast path
    try:
        fi = t.fast_info
        price, prev = fi["lastPrice"], fi["previousClose"]
    except Exception:
        # slow fallback
        hist = t.history(period="2d")
        if len(hist) >= 2:
            prev, price = hist["Close"][0], hist["Close"][-1]
        elif len(hist) == 1:
            price = prev = hist["Close"][0]

    if price is None:
        return f"Price data for {ticker_symbol} is not available."

    if prev and prev != 0:
        change_pct = (price - prev) / prev * 100

    sign  = "+" if change_pct and change_pct >= 0 else ""
    pct   = f" ({sign}{change_pct:.2f}%)" if change_pct is not None else ""
    return f"The current price of {ticker_symbol} is ${price:,.2f}{pct}."


def get_financial_statements(ticker_symbol: str) -> str:
    t = yf.Ticker(ticker_symbol)
    try:
        info = t.get_info()
    except Exception:
        return f"Financial information for {ticker_symbol} is not available."

    name    = info.get("shortName", ticker_symbol)
    revenue = info.get("totalRevenue")
    income  = (info.get("netIncomeToCommon") or
               info.get("netIncome") or
               info.get("incomeNet"))

    def _fmt(num):
        if num is None:
            return "N/A"
        if abs(num) >= 1e9:
            return f"${num/1e9:.2f} B"
        if abs(num) >= 1e6:
            return f"${num/1e6:.2f} M"
        return f"${num:,}"

    return (f"{name} latest financials – Revenue: {_fmt(revenue)}, "
            f"Net income: {_fmt(income)} (last annual/TTM).")


# ────────────────────────────────────────────────────────────────────────────
#  News headlines  (1️⃣ yfinance  →  2️⃣ NewsAPI if configured)
# ────────────────────────────────────────────────────────────────────────────
def _from_yfinance(ticker: str, limit: int) -> list[tuple[str, str, str]]:
    yf_ticker = yf.Ticker(ticker)
    try:
        news = yf_ticker.get_news()          # yfinance ≥ 0.2.34
    except AttributeError:
        news = getattr(yf_ticker, "news", [])
    items = []
    for item in news[:limit]:
        title  = item.get("title")
        source = item.get("publisher", "Yahoo")
        ts     = item.get("providerPublishTime")
        date   = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d") if ts else ""
        if title:
            items.append((title, source, date))
    return items


def _from_newsapi(query: str, limit: int) -> list[tuple[str, str, str]]:
    if not NEWSAPI_KEY:
        return []
    try:
        from newsapi import NewsApiClient
    except ModuleNotFoundError:
        return []

    client = NewsApiClient(api_key=NEWSAPI_KEY)
    resp   = client.get_everything(q=query,
                                   language="en",
                                   sort_by="publishedAt",
                                   page_size=limit)
    return [(a["title"],
             a["source"]["name"],
             a["publishedAt"][:10]) for a in resp.get("articles", [])]


def get_recent_news(query: str, limit: int = 3) -> str:
    """
    Return up to `limit` fresh headlines for a ticker **or** company name.
    Steps:
      1. Try yfinance directly (expects a ticker).
      2. If nothing comes back, look up a likely ticker and retry.
      3. Still nothing?  Fall back to NewsAPI (if key provided).
    """
    # First pass: assume the input *is* the ticker
    for attempt in (query, _lookup_symbol(query) or ""):
        if not attempt:
            continue
        rows = _from_yfinance(attempt, limit)
        if rows:
            lines = [f"- {t} ({s}, {d})" for t, s, d in rows]
            return f"Recent news for {attempt}:\n" + "\n".join(lines)

    # Fallback – keyword search via NewsAPI
    rows = _from_newsapi(query, limit)
    if rows:
        lines = [f"- {t} ({s}, {d})" for t, s, d in rows]
        return f"Recent news for {query}:\n" + "\n".join(lines)

    return f"No recent news found for {query}."


# ────────────────────────────────────────────────────────────────────────────
#  Brief market snapshot
# ────────────────────────────────────────────────────────────────────────────
def get_market_summary() -> str:
    indices = {"S&P 500": "^GSPC", "Dow Jones": "^DJI", "Nasdaq": "^IXIC"}
    ups = downs = 0
    parts = []

    for name, sym in indices.items():
        t    = yf.Ticker(sym)
        hist = t.history(period="2d")
        if len(hist) < 1:
            parts.append(f"{name}: N/A")
            continue
        prev, last = hist["Close"][0], hist["Close"][-1]
        chg_pct    = (last - prev) / prev * 100 if prev else 0.0
        (ups, downs)[chg_pct < 0] += 1
        sign = "+" if chg_pct >= 0 else ""
        parts.append(f"{name}: {last:,.0f} ({sign}{chg_pct:.2f}%)")

    sentiment = "positive" if ups > downs else "negative" if downs > ups else "mixed"
    return f"Market Summary – {'; '.join(parts)}. Overall sentiment: {sentiment}."
