"""Extra per-stock research data from Yahoo Finance: analyst views and price targets, ownership, insider trades,
short interest and options activity (US), quarterly statements, peers and a discounted-cash-flow model.
Everything Yahoo doesn't report comes back empty / None and is shown as N/A."""
import os

import numpy as np
import pandas as pd
import streamlit as st

DEMO = os.environ.get("DEMO_DATA") == "1"


def _tk(t):
    import yfinance as yf
    return yf.Ticker(t)


def _safe(fn, default=None):
    try:
        v = fn()
        return default if v is None else v
    except Exception:
        return default


# ============================================================================
# Analysts
# ============================================================================
@st.cache_data(ttl=6 * 3600, show_spinner=False, max_entries=150)
def analysts(ticker):
    if DEMO:
        rng = np.random.default_rng(abs(hash(ticker)) % 2**32)
        mean = float(rng.uniform(5, 30))
        recs = pd.DataFrame({"period": ["0m", "-1m", "-2m"], "strongBuy": [3, 3, 2], "buy": [6, 5, 5],
                             "hold": [4, 5, 6], "sell": [1, 1, 1], "strongSell": [0, 0, 1]})
        return {"key": "buy", "mean": 2.1, "n": 14, "target_mean": mean * 1.12, "target_high": mean * 1.35,
                "target_low": mean * 0.9, "target_median": mean * 1.1, "recs": recs,
                "eps_est": pd.DataFrame({"avg": [0.42, 0.47], "low": [0.38, 0.41], "high": [0.46, 0.52],
                                         "numberOfAnalysts": [12, 11], "growth": [0.08, 0.12]},
                                        index=["0y", "+1y"]),
                "rev_est": None, "changes": pd.DataFrame()}
    tk = _tk(ticker)
    info = _safe(lambda: tk.info, {}) or {}
    return {"key": info.get("recommendationKey"), "mean": info.get("recommendationMean"),
            "n": info.get("numberOfAnalystOpinions"), "target_mean": info.get("targetMeanPrice"),
            "target_high": info.get("targetHighPrice"), "target_low": info.get("targetLowPrice"),
            "target_median": info.get("targetMedianPrice"),
            "recs": _safe(lambda: tk.recommendations, pd.DataFrame()),
            "eps_est": _safe(lambda: tk.earnings_estimate, None),
            "rev_est": _safe(lambda: tk.revenue_estimate, None),
            "changes": _safe(lambda: tk.upgrades_downgrades, pd.DataFrame())}


# ============================================================================
# Ownership, insiders, short interest, options
# ============================================================================
@st.cache_data(ttl=12 * 3600, show_spinner=False, max_entries=150)
def ownership(ticker):
    if DEMO:
        major = pd.DataFrame({"Value": [0.42, 0.31, 0.38, 210]},
                             index=["insidersPercentHeld", "institutionsPercentHeld",
                                    "institutionsFloatPercentHeld", "institutionsCount"])
        inst = pd.DataFrame({"Holder": ["Demo Fund A", "Demo Pension B", "Demo Index C"],
                             "pctHeld": [0.08, 0.05, 0.03], "Shares": [2e8, 1.2e8, 7e7],
                             "Date Reported": pd.to_datetime(["2026-09-30"] * 3)})
        ins = pd.DataFrame() if ticker.endswith(".KL") else pd.DataFrame(
            {"Insider": ["DEMO CEO", "DEMO CFO"], "Position": ["Chief Executive Officer", "Chief Financial Officer"],
             "Transaction": ["Sale", "Purchase"], "Shares": [12000, 3000], "Value": [2.4e6, 6e5],
             "Start Date": pd.to_datetime(["2026-10-12", "2026-09-03"])})
        short = None if ticker.endswith(".KL") else {"shares": 4.1e7, "prior": 3.6e7, "pct_float": 0.021,
                                                      "ratio": 1.4, "date": "2026-10-31"}
        return {"major": major, "inst": inst, "funds": pd.DataFrame(), "insiders": ins, "short": short}
    tk = _tk(ticker)
    info = _safe(lambda: tk.info, {}) or {}
    short = None
    if info.get("sharesShort") is not None:
        d = info.get("dateShortInterest")
        short = {"shares": info.get("sharesShort"), "prior": info.get("sharesShortPriorMonth"),
                 "pct_float": info.get("shortPercentOfFloat"), "ratio": info.get("shortRatio"),
                 "date": pd.to_datetime(d, unit="s").strftime("%Y-%m-%d") if isinstance(d, (int, float)) else d}
    return {"major": _safe(lambda: tk.major_holders, pd.DataFrame()),
            "inst": _safe(lambda: tk.institutional_holders, pd.DataFrame()),
            "funds": _safe(lambda: tk.mutualfund_holders, pd.DataFrame()),
            "insiders": _safe(lambda: tk.insider_transactions, pd.DataFrame()), "short": short}


@st.cache_data(ttl=1800, show_spinner=False, max_entries=60)
def options_summary(ticker, n_exp=3):
    """Open interest and volume on the nearest expiries (US-listed stocks only; Bursa has no stock options)."""
    if ticker.endswith(".KL") or ticker.startswith("^"):
        return None
    if DEMO:
        rng = np.random.default_rng(abs(hash(ticker)) % 2**32)
        strikes = np.arange(80, 125, 5.0)
        rows = []
        for e in ("2026-11-27", "2026-12-04", "2026-12-18"):
            for k in strikes:
                rows.append({"expiry": e, "strike": k, "call_oi": int(rng.integers(100, 9000)),
                             "put_oi": int(rng.integers(100, 9000)), "call_vol": int(rng.integers(0, 3000)),
                             "put_vol": int(rng.integers(0, 3000)), "call_iv": float(rng.uniform(.25, .5)),
                             "put_iv": float(rng.uniform(.25, .55))})
        return {"chain": pd.DataFrame(rows), "spot": 100.0}
    try:
        tk = _tk(ticker)
        exps = list(tk.options or [])[:n_exp]
        if not exps:
            return None
        rows = []
        for e in exps:
            ch = tk.option_chain(e)
            c = ch.calls.set_index("strike")
            p = ch.puts.set_index("strike")
            for k in sorted(set(c.index) | set(p.index)):
                rows.append({"expiry": e, "strike": float(k),
                             "call_oi": float(c["openInterest"].get(k, 0) or 0), "put_oi": float(p["openInterest"].get(k, 0) or 0),
                             "call_vol": float(c["volume"].get(k, 0) or 0), "put_vol": float(p["volume"].get(k, 0) or 0),
                             "call_iv": float(c["impliedVolatility"].get(k, np.nan)),
                             "put_iv": float(p["impliedVolatility"].get(k, np.nan))})
        spot = _safe(lambda: tk.fast_info["lastPrice"], None)
        return {"chain": pd.DataFrame(rows).fillna({"call_oi": 0, "put_oi": 0, "call_vol": 0, "put_vol": 0}),
                "spot": spot}
    except Exception:
        return None


# ============================================================================
# Quarterly statements
# ============================================================================
@st.cache_data(ttl=6 * 3600, show_spinner=False, max_entries=100)
def quarterly(ticker):
    if DEMO:
        q = pd.date_range(end="2026-09-30", periods=5, freq="QE")
        rng = np.random.default_rng(abs(hash(ticker)) % 2**32)
        rev = 2.6e8 * np.cumprod(1 + rng.normal(0.015, 0.04, 5))
        ni = rev * rng.uniform(0.08, 0.12, 5)
        return {"income": pd.DataFrame({d: {"Total Revenue": r, "Gross Profit": r * .35, "Operating Income": r * .15,
                                            "Net Income": n, "Diluted EPS": n / 2e9} for d, r, n in zip(q, rev, ni)}),
                "balance": pd.DataFrame({d: {"Total Assets": 5e9, "Total Debt": 6e8, "Stockholders Equity": 2.4e9,
                                             "Cash And Cash Equivalents": 4e8} for d in q}),
                "cashflow": pd.DataFrame({d: {"Operating Cash Flow": n * 1.2, "Capital Expenditure": -r * .05,
                                              "Free Cash Flow": n * 1.2 - r * .05} for d, r, n in zip(q, rev, ni)})}
    tk = _tk(ticker)
    return {"income": _safe(lambda: tk.quarterly_income_stmt, pd.DataFrame()),
            "balance": _safe(lambda: tk.quarterly_balance_sheet, pd.DataFrame()),
            "cashflow": _safe(lambda: tk.quarterly_cashflow, pd.DataFrame())}


KEY_LINES = {
    "income": ["Total Revenue", "Cost Of Revenue", "Gross Profit", "Operating Expense", "Operating Income", "EBITDA",
               "EBIT", "Interest Expense", "Pretax Income", "Tax Provision", "Net Income", "Diluted EPS",
               "Diluted Average Shares"],
    "balance": ["Total Assets", "Current Assets", "Cash And Cash Equivalents", "Inventory", "Receivables",
                "Total Liabilities Net Minority Interest", "Current Liabilities", "Total Debt", "Net Debt",
                "Stockholders Equity", "Retained Earnings", "Ordinary Shares Number"],
    "cashflow": ["Operating Cash Flow", "Capital Expenditure", "Free Cash Flow", "Investing Cash Flow",
                 "Financing Cash Flow", "Cash Dividends Paid", "Repurchase Of Capital Stock",
                 "Issuance Of Debt", "Repayment Of Debt", "End Cash Position"],
}


def statement_table(df, kind, all_rows=False):
    """Statement as a display table: oldest → newest columns, key lines first (or every line)."""
    if df is None or df.empty:
        return pd.DataFrame()
    x = df.copy()
    x.columns = pd.to_datetime(x.columns)
    x = x[sorted(x.columns)]
    rows = list(x.index) if all_rows else [r for r in KEY_LINES[kind] if r in x.index]
    x = x.loc[rows].apply(pd.to_numeric, errors="coerce")
    x = x.dropna(how="all")
    x.columns = [c.strftime("%b %Y") for c in x.columns]
    return x


# ============================================================================
# Peers
# ============================================================================
def peers(ticker, bursa_all, sector_of_us=None, n=8):
    """Bursa: biggest companies in the same Yahoo sector from the Bursa-wide list. US: the app's US list."""
    if ticker.endswith(".KL") and bursa_all is not None and not bursa_all.empty:
        row = bursa_all[bursa_all["ticker"] == ticker]
        if row.empty:
            return pd.DataFrame(), None
        sec = row["sector"].iloc[0]
        p = bursa_all[bursa_all["sector"] == sec].sort_values("mcap", ascending=False)
        top = p.head(n)
        if ticker not in set(top["ticker"]):
            top = pd.concat([top.head(n - 1), row])
        return top[["ticker", "name", "price", "chg", "mcap", "pe", "div_yield"]].reset_index(drop=True), sec
    return pd.DataFrame(), sector_of_us


# ============================================================================
# DCF
# ============================================================================
def dcf(fcf0, g1, g2, r, gt, net_debt, shares, years=10):
    """Two-stage discounted cash flow. g1 for years 1–5, g2 for 6–10, gt terminal growth, r discount rate
    (all as fractions). Returns per-share value and the pieces. None if inputs make no sense."""
    if fcf0 is None or shares is None or shares <= 0 or r <= gt:
        return None
    flows, f = [], fcf0
    for y in range(1, years + 1):
        f *= 1 + (g1 if y <= 5 else g2)
        flows.append(f / (1 + r) ** y)
    tv = f * (1 + gt) / (r - gt)
    pv_tv = tv / (1 + r) ** years
    ev = sum(flows) + pv_tv
    eq = ev - (net_debt or 0)
    return {"per_share": eq / shares, "ev": ev, "equity": eq, "pv_flows": sum(flows), "pv_terminal": pv_tv,
            "terminal_share": pv_tv / ev if ev else None, "flows": flows}
