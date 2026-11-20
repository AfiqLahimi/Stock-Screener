"""Shared calculations for Compare, Risk, Calendar and Macro (Phase 6). Daily closes from Yahoo via
core.load_one; every statistic uses only the overlapping history that actually exists (N/A otherwise)."""
import json
import os
import urllib.request

import numpy as np
import pandas as pd
import streamlit as st

from core import get_many, load_one

DEMO = os.environ.get("DEMO_DATA") == "1"
TRADING_DAYS = 252


def benchmark_of(ticker):
    return "^KLSE" if ticker.endswith(".KL") or ticker == "^KLSE" else "^GSPC"


def closes(tickers, period="2y"):
    """Daily closes, one column per ticker, aligned on dates (each stock keeps its own trading days)."""
    cols = {}
    try:
        pre = get_many(tuple(tickers), "1d", period) if len(tickers) > 1 else {}
    except Exception:
        pre = {}
    for t in tickers:
        d = pre.get(t)
        if d is None or d.empty:
            d = load_one(t, "1d", period)
        if d is not None and not d.empty:
            s = d["Close"].copy()
            s.index = pd.to_datetime(s.index).normalize()
            cols[t] = s[~s.index.duplicated(keep="last")]
    return pd.DataFrame(cols).sort_index()


def returns(px):
    return np.log(px / px.shift(1))


def ann_vol(r):
    r = r.dropna()
    return float(r.std(ddof=1) * np.sqrt(TRADING_DAYS) * 100) if len(r) > 20 else None


def max_drawdown(s):
    s = s.dropna()
    if len(s) < 2:
        return None, None, None
    peak = s.cummax()
    dd = s / peak - 1
    trough = dd.idxmin()
    start = s.loc[:trough].idxmax()
    return float(dd.min() * 100), start, trough


def beta(r_stock, r_bench):
    j = pd.concat([r_stock, r_bench], axis=1).dropna()
    if len(j) < 60 or j.iloc[:, 1].var() == 0:
        return None
    return float(j.iloc[:, 0].cov(j.iloc[:, 1]) / j.iloc[:, 1].var())


def period_return(s, days):
    s = s.dropna()
    if len(s) <= days:
        return None
    return float((s.iloc[-1] / s.iloc[-1 - days] - 1) * 100)


def hist_var(r, level=0.95):
    """One-day historical Value at Risk (as a positive % loss) and expected shortfall beyond it."""
    r = r.dropna()
    if len(r) < 100:
        return None, None
    simple = np.expm1(r)
    q = float(np.quantile(simple, 1 - level))
    es = float(simple[simple <= q].mean())
    return -q * 100, -es * 100


# ============================================================================
# Events (earnings / dividends)
# ============================================================================
@st.cache_data(ttl=6 * 3600, show_spinner=False, max_entries=300)
def events(ticker):
    """Next earnings date and dividend dates from Yahoo's calendar (None when not published)."""
    if DEMO:
        rng = np.random.default_rng(abs(hash(ticker)) % 2**32)
        now = pd.Timestamp.now().normalize()
        return {"earnings": now + pd.Timedelta(days=int(rng.integers(3, 80))),
                "ex_dividend": now + pd.Timedelta(days=int(rng.integers(-20, 60))) if rng.random() > 0.3 else None,
                "dividend_pay": None}
    out = {"earnings": None, "ex_dividend": None, "dividend_pay": None}
    try:
        import yfinance as yf
        cal = yf.Ticker(ticker).calendar or {}
        ed = cal.get("Earnings Date")
        if isinstance(ed, (list, tuple)) and ed:
            out["earnings"] = pd.Timestamp(ed[0])
        elif ed:
            out["earnings"] = pd.Timestamp(ed)
        for k, key in (("Ex-Dividend Date", "ex_dividend"), ("Dividend Date", "dividend_pay")):
            if cal.get(k):
                out[key] = pd.Timestamp(cal[k])
    except Exception:
        pass
    return out


# ============================================================================
# Macro data (free, keyless)
# ============================================================================
def _get(url, headers=None, timeout=10):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def bnm_opr():
    """Malaysia's Overnight Policy Rate from Bank Negara Malaysia's open API."""
    if DEMO:
        return {"rate": 2.75, "date": "2025-07-09", "change": -0.25}
    try:
        data = json.loads(_get("https://api.bnm.gov.my/public/opr",
                               {"Accept": "application/vnd.BNM.API.v1+json"}))["data"]
        return {"rate": float(data.get("new_opr_level")), "date": data.get("date"),
                "change": float(data.get("change_in_opr") or 0)}
    except Exception:
        return None


FRED = {
    # id: (name, transform, group, unit)  transform: level | yoy (% vs a year earlier) | diff (change vs previous)
    "CPIAUCSL": ("Inflation (CPI)", "yoy", "Inflation", "%"),
    "CPILFESL": ("Core inflation (CPI ex food & energy)", "yoy", "Inflation", "%"),
    "PCEPILFE": ("Core PCE inflation (Fed's target gauge)", "yoy", "Inflation", "%"),
    "UNRATE": ("Unemployment rate", "level", "Jobs", "%"),
    "PAYEMS": ("Jobs added (non-farm payrolls)", "diff", "Jobs", "k"),
    "ICSA": ("Weekly jobless claims", "level", "Jobs", "k"),
    "A191RL1Q225SBEA": ("Real GDP growth (annualised, quarterly)", "level", "Growth", "%"),
    "INDPRO": ("Industrial production", "yoy", "Growth", "%"),
    "RSAFS": ("Retail sales", "yoy", "Growth", "%"),
    "FEDFUNDS": ("Fed funds rate", "level", "Rates & credit", "%"),
    "DGS2": ("2-year Treasury yield", "level", "Rates & credit", "%"),
    "DGS10": ("10-year Treasury yield", "level", "Rates & credit", "%"),
    "T10Y2Y": ("10y − 2y yield spread", "level", "Rates & credit", "%"),
    "BAMLH0A0HYM2": ("High-yield credit spread", "level", "Rates & credit", "%"),
    "UMCSENT": ("Consumer sentiment (U. Michigan)", "level", "Sentiment & money", ""),
    "M2SL": ("Money supply (M2)", "yoy", "Sentiment & money", "%"),
}


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def fred(series_id):
    """A FRED series via its public CSV download (no key). Returns a Series or None."""
    if DEMO:
        idx = pd.date_range(end=pd.Timestamp.now().normalize(), periods=36, freq="MS")
        base = {"CPIAUCSL": 310, "CPILFESL": 320, "PCEPILFE": 125, "UNRATE": 4.2, "PAYEMS": 159000, "ICSA": 225,
                "A191RL1Q225SBEA": 2.1, "INDPRO": 103, "RSAFS": 720000, "FEDFUNDS": 4.1, "DGS2": 3.9, "DGS10": 4.3,
                "T10Y2Y": 0.4, "BAMLH0A0HYM2": 3.1, "UMCSENT": 68, "M2SL": 21700}.get(series_id, 100)
        rng = np.random.default_rng(len(series_id))
        vals = base * np.cumprod(1 + rng.normal(0.002 if series_id == "CPIAUCSL" else 0, 0.01, 36))
        return pd.Series(vals, index=idx)
    try:
        from io import StringIO
        txt = _get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}", timeout=12)
        df = pd.read_csv(StringIO(txt))
        df.columns = ["date", "value"]
        df["date"] = pd.to_datetime(df["date"])
        s = pd.to_numeric(df["value"], errors="coerce")
        s.index = df["date"]
        return s.dropna().tail(400)
    except Exception:
        return None
