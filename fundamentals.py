"""Fundamental analysis from Yahoo Finance: valuation, profitability, financial health, growth, cash flow and
dividends, each with a plain explanation, an assessment and a category score (0–100, 50 = average).

Data comes from yfinance (company profile + annual income statement, balance sheet, cash flow, dividends).
Anything Yahoo doesn't report is shown as N/A and left out of the score; nothing is estimated.
Banks and insurers are judged differently: debt, liquidity and margin ratios built for ordinary companies
don't describe a bank, so those metrics are skipped for them.
"""
import os

import numpy as np
import pandas as pd
import streamlit as st

DEMO = os.environ.get("DEMO_DATA") == "1"
CATEGORIES = ["Valuation", "Profitability", "Financial health", "Growth", "Cash flow", "Dividends"]
DEFAULT_WEIGHTS = {"Valuation": 20, "Profitability": 20, "Financial health": 20, "Growth": 15, "Cash flow": 15,
                   "Dividends": 10}


# ============================================================================
# Data
# ============================================================================
def _demo(ticker):
    rng = np.random.default_rng(abs(hash(ticker)) % 2**32)
    years = pd.to_datetime([f"{y}-12-31" for y in range(2021, 2026)])
    rev = 1e9 * np.cumprod(1 + rng.normal(0.06, 0.08, 5))
    gm, om, nm = rng.uniform(0.25, 0.5), rng.uniform(0.08, 0.2), rng.uniform(0.05, 0.14)
    ni = rev * nm * (1 + rng.normal(0, 0.1, 5))
    inc = pd.DataFrame({d: {"Total Revenue": r, "Gross Profit": r * gm, "Operating Income": r * om, "EBIT": r * om,
                            "Net Income": n, "Diluted EPS": n / 2e9, "Interest Expense": r * 0.01,
                            "Tax Provision": n * 0.3, "Pretax Income": n * 1.3}
                        for d, r, n in zip(years, rev, ni)})
    bal = pd.DataFrame({d: {"Total Debt": 6e8, "Stockholders Equity": 2.2e9 + i * 1e8, "Total Assets": 5e9,
                            "Cash And Cash Equivalents": 4e8, "Current Assets": 1.6e9, "Current Liabilities": 1e9,
                            "Inventory": 3e8} for i, d in enumerate(years)})
    cf = pd.DataFrame({d: {"Operating Cash Flow": n * 1.2, "Capital Expenditure": -r * 0.05,
                           "Free Cash Flow": n * 1.2 - r * 0.05} for d, r, n in zip(years, rev, ni)})
    divs = pd.Series(np.repeat([0.08, 0.085, 0.09, 0.095, 0.10, 0.105], 2) / 2,
                     index=pd.date_range("2020-06-30", periods=12, freq="6MS"))
    info = {"longName": f"Demo {ticker}", "sector": "Consumer Defensive", "industry": "Packaged Foods",
            "trailingPE": 14.2, "forwardPE": 12.8, "priceToBook": 1.9, "priceToSalesTrailing12Months": 1.6,
            "enterpriseToEbitda": 8.4, "returnOnEquity": 0.135, "returnOnAssets": 0.062, "grossMargins": gm,
            "operatingMargins": om, "profitMargins": nm, "debtToEquity": 27.0, "currentRatio": 1.6, "quickRatio": 1.1,
            "dividendRate": 0.105, "trailingAnnualDividendYield": 0.034, "payoutRatio": 0.55, "currentPrice": 3.1, "marketCap": 9e9, "currency": "MYR",
            "financialCurrency": "MYR", "mostRecentQuarter": None}
    return {"info": info, "income": inc, "balance": bal, "cashflow": cf, "dividends": divs}


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load(ticker: str) -> dict:
    if DEMO:
        return _demo(ticker)
    out = {"info": {}, "income": pd.DataFrame(), "balance": pd.DataFrame(), "cashflow": pd.DataFrame(),
           "dividends": pd.Series(dtype=float)}
    try:
        import yfinance as yf
        tk = yf.Ticker(ticker)
        try:
            out["info"] = dict(tk.info or {})
        except Exception:
            pass
        for key, attr in (("income", "income_stmt"), ("balance", "balance_sheet"), ("cashflow", "cashflow")):
            try:
                df = getattr(tk, attr)
                out[key] = df if isinstance(df, pd.DataFrame) else pd.DataFrame()
            except Exception:
                pass
        try:
            dv = tk.dividends
            if dv is not None and len(dv):
                dv.index = pd.to_datetime(dv.index).tz_localize(None)
                out["dividends"] = dv
        except Exception:
            pass
    except Exception:
        pass
    return out


def _row(df, *names):
    """A statement line as a Series indexed by fiscal year end (oldest → newest), or None."""
    if df is None or df.empty:
        return None
    for n in names:
        if n in df.index:
            s = pd.to_numeric(df.loc[n], errors="coerce")
            s.index = pd.to_datetime(s.index)
            s = s.sort_index().dropna()
            return s if len(s) else None
    return None


def _last(s, k=1):
    return None if s is None or len(s) < k else float(s.iloc[-k])


def _num(v):
    try:
        v = float(v)
        return None if np.isnan(v) or np.isinf(v) else v
    except (TypeError, ValueError):
        return None


def _growth(s):
    if s is None or len(s) < 2:
        return None
    a, b = float(s.iloc[-2]), float(s.iloc[-1])
    if a <= 0:
        return None          # growth from a loss or zero is not meaningful as a %
    return (b / a - 1) * 100


def _cagr(s, years=3):
    if s is None or len(s) < years + 1:
        return None
    a, b = float(s.iloc[-1 - years]), float(s.iloc[-1])
    if a <= 0 or b <= 0:
        return None
    return ((b / a) ** (1 / years) - 1) * 100


def _band(v, cuts, higher_better=True):
    """cuts: list of (threshold, points) checked in order; higher_better uses >=, otherwise <=."""
    if v is None:
        return None
    for thr, pts in cuts:
        if (v >= thr) if higher_better else (v <= thr):
            return pts
    return -1.0


def _m(name, value, display, meaning, assessment, points, weight=1.0):
    return {"name": name, "value": value, "display": display, "meaning": meaning, "assessment": assessment,
            "points": points, "weight": weight}


def _na(name, meaning, why="not reported by Yahoo"):
    return _m(name, None, "N/A", meaning, why, None, 0.0)


def _assess(points):
    if points is None:
        return ""
    return "strong" if points >= 0.75 else "good" if points >= 0.25 else "average" if points > -0.25 else \
        "weak" if points > -0.75 else "poor"


def _pct(v):
    return "N/A" if v is None else f"{v:.1f}%"


# ============================================================================
# Analysis
# ============================================================================
def analyse(data, price=None):
    info = data.get("info") or {}
    inc, bal, cf, divs = data.get("income"), data.get("balance"), data.get("cashflow"), data.get("dividends")
    sector = (info.get("sector") or "")
    industry = (info.get("industry") or "")
    is_fin = sector == "Financial Services" and any(k in industry for k in ("Bank", "Insurance", "Credit",
                                                                             "Capital Markets", "Asset Management"))
    rev = _row(inc, "Total Revenue", "Operating Revenue")
    ebit = _row(inc, "EBIT", "Operating Income")
    ni = _row(inc, "Net Income", "Net Income Common Stockholders")
    eps = _row(inc, "Diluted EPS", "Basic EPS")
    intexp = _row(inc, "Interest Expense", "Interest Expense Non Operating")
    tax, pretax = _row(inc, "Tax Provision"), _row(inc, "Pretax Income")
    debt = _row(bal, "Total Debt")
    eq = _row(bal, "Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest")
    cash = _row(bal, "Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments")
    ocf = _row(cf, "Operating Cash Flow", "Cash Flow From Continuing Operating Activities")
    capex = _row(cf, "Capital Expenditure")
    fcf = _row(cf, "Free Cash Flow")
    if fcf is None and ocf is not None and capex is not None:
        fcf = (ocf + capex.reindex(ocf.index)).dropna()
    fy = ni.index[-1] if ni is not None else (rev.index[-1] if rev is not None else None)

    cats = {}
    # ---------------------------------------------------------------- valuation
    v = []
    pe = _num(info.get("trailingPE"))
    if pe is None and _num(info.get("trailingEps")) is not None and price and info["trailingEps"] < 0:
        v.append(_m("P/E (TTM)", None, "loss", "Price ÷ earnings per share of the last 12 months.",
                    "the company made a loss, so P/E is not meaningful", -1.0))
    elif pe is not None:
        p = 1.0 if pe <= 10 else 0.5 if pe <= 15 else 0.0 if pe <= 22 else -0.5 if pe <= 35 else -1.0
        v.append(_m("P/E (TTM)", pe, f"{pe:.1f}×", "Price ÷ earnings per share of the last 12 months. Lower = "
                    "cheaper for each ringgit/dollar of profit.", _assess(p) + " value" if p is not None else "", p))
    else:
        v.append(_na("P/E (TTM)", "Price ÷ earnings per share of the last 12 months."))
    fpe = _num(info.get("forwardPE"))
    v.append(_m("Forward P/E", fpe, f"{fpe:.1f}×", "Price ÷ analysts' expected earnings for the next year.",
                "expected earnings " + ("higher than last year's" if pe and fpe < pe else "not higher than last year's"),
                (1.0 if fpe <= 10 else 0.5 if fpe <= 15 else 0.0 if fpe <= 22 else -0.5 if fpe <= 35 else -1.0) if fpe > 0
                else -1.0, 0.5) if fpe is not None else _na("Forward P/E", "Price ÷ expected earnings for next year.",
                                                             "no analyst estimate on Yahoo (common for Bursa stocks)"))
    pb = _num(info.get("priceToBook"))
    v.append(_m("Price / book", pb, f"{pb:.2f}×", "Price ÷ net assets per share. Below 1 = trading under its book value.",
                "", (1.0 if pb <= 1 else 0.5 if pb <= 2 else 0.0 if pb <= 4 else -0.5 if pb <= 8 else -1.0) if pb > 0 else -1.0,
                1.0 if is_fin else 0.7) if pb is not None else _na("Price / book", "Price ÷ net assets per share."))
    if not is_fin:
        ps = _num(info.get("priceToSalesTrailing12Months"))
        v.append(_m("Price / sales", ps, f"{ps:.2f}×", "Market value ÷ yearly revenue. Useful when profits are small.",
                    "", 1.0 if ps <= 1 else 0.5 if ps <= 3 else 0.0 if ps <= 6 else -0.5 if ps <= 10 else -1.0, 0.7)
                 if ps is not None else _na("Price / sales", "Market value ÷ yearly revenue."))
        ev = _num(info.get("enterpriseToEbitda"))
        v.append(_m("EV / EBITDA", ev, f"{ev:.1f}×", "Company value incl. debt ÷ operating cash profit. Compares "
                    "companies with different debt levels.", "",
                    (1.0 if ev <= 6 else 0.5 if ev <= 10 else 0.0 if ev <= 15 else -0.5 if ev <= 25 else -1.0) if ev > 0 else -1.0)
                 if ev is not None else _na("EV / EBITDA", "Company value incl. debt ÷ operating cash profit."))
    cats["Valuation"] = v

    # ---------------------------------------------------------------- profitability
    p_ = []
    roe = _num(info.get("returnOnEquity"))
    if roe is None and _last(ni) is not None and _last(eq):
        roe = _last(ni) / _last(eq)
    p_.append(_m("Return on equity", roe, _pct(roe * 100), "Profit ÷ shareholders' money. How well the company "
                 "turns your capital into profit.", "", _band(roe * 100, [(20, 1.0), (15, 0.6), (10, 0.2), (5, -0.3), (0, -0.7)]))
              if roe is not None else _na("Return on equity", "Profit ÷ shareholders' equity."))
    roa = _num(info.get("returnOnAssets"))
    if roa is not None:
        cuts = [(1.2, 1.0), (0.8, 0.4), (0.5, -0.2), (0, -0.6)] if is_fin else [(10, 1.0), (5, 0.5), (2, 0.0), (0, -0.6)]
        p_.append(_m("Return on assets", roa, _pct(roa * 100), "Profit ÷ everything the company owns." +
                     (" Banks normally earn about 1%." if is_fin else ""), "", _band(roa * 100, cuts)))
    else:
        p_.append(_na("Return on assets", "Profit ÷ total assets."))
    if not is_fin:
        roic = None
        if _last(ebit) is not None and _last(eq) is not None:
            t_rate = (_last(tax) / _last(pretax)) if _last(tax) is not None and _last(pretax) else 0.24
            t_rate = min(max(t_rate, 0), 0.5)
            invested = _last(eq) + (_last(debt) or 0) - (_last(cash) or 0)
            if invested > 0:
                roic = _last(ebit) * (1 - t_rate) / invested * 100
        p_.append(_m("ROIC", roic, _pct(roic), "After-tax operating profit ÷ (equity + debt − cash). Return on all the "
                     "money invested in the business.", "", _band(roic, [(15, 1.0), (10, 0.5), (6, 0.0), (0, -0.5)]))
                  if roic is not None else _na("ROIC", "After-tax operating profit ÷ invested capital.",
                                               "statement lines missing"))
        gm = _num(info.get("grossMargins"))
        p_.append(_m("Gross margin", gm, _pct(gm * 100), "Revenue left after the direct cost of goods.", "",
                     _band(gm * 100, [(50, 1.0), (30, 0.5), (15, 0.0), (0, -0.5)]), 0.5)
                  if gm is not None and gm != 0 else _na("Gross margin", "Revenue left after cost of goods."))
        om = _num(info.get("operatingMargins"))
        p_.append(_m("Operating margin", om, _pct(om * 100), "Profit from the core business ÷ revenue.", "",
                     _band(om * 100, [(20, 1.0), (10, 0.5), (5, 0.0), (0, -0.5)]))
                  if om is not None else _na("Operating margin", "Operating profit ÷ revenue."))
    nm = _num(info.get("profitMargins"))
    p_.append(_m("Net margin", nm, _pct(nm * 100), "Final profit ÷ revenue: how many sen of each ringgit of sales is "
                 "kept.", "", _band(nm * 100, [(15, 1.0), (8, 0.5), (3, 0.0), (0, -0.5)]))
              if nm is not None else _na("Net margin", "Net profit ÷ revenue."))
    cats["Profitability"] = p_

    # ---------------------------------------------------------------- health
    h = []
    if is_fin:
        h.append(_na("Debt / equity", "Debt ÷ shareholders' equity.",
                     "not meaningful for banks/insurers (deposits count as debt); judge them by capital ratios, "
                     "which Yahoo doesn't provide"))
    else:
        de = _num(info.get("debtToEquity"))
        de = de / 100 if de is not None else ((_last(debt) / _last(eq)) if _last(debt) is not None and _last(eq) else None)
        h.append(_m("Debt / equity", de, f"{de:.2f}×", "Borrowings ÷ shareholders' money. Higher = more risk if profits "
                    "fall.", "", (1.0 if de <= 0.3 else 0.5 if de <= 0.7 else 0.0 if de <= 1.5 else -0.5 if de <= 2.5
                                  else -1.0) if de >= 0 else -1.0) if de is not None else
                 _na("Debt / equity", "Borrowings ÷ shareholders' equity."))
        cr = _num(info.get("currentRatio"))
        h.append(_m("Current ratio", cr, f"{cr:.2f}×", "Short-term assets ÷ bills due within a year. Above 1 = can "
                    "pay its near-term bills.", "", _band(cr, [(2, 1.0), (1.5, 0.5), (1, 0.0), (0, -0.8)]))
                 if cr is not None else _na("Current ratio", "Current assets ÷ current liabilities."))
        qr = _num(info.get("quickRatio"))
        h.append(_m("Quick ratio", qr, f"{qr:.2f}×", "Like the current ratio but without inventory (cash-like "
                    "assets only).", "", _band(qr, [(1.5, 1.0), (1, 0.5), (0.7, 0.0), (0, -0.7)]), 0.5)
                 if qr is not None else _na("Quick ratio", "(Current assets − inventory) ÷ current liabilities."))
        ie, eb = _last(intexp), _last(ebit)
        if eb is not None and ie is not None and abs(ie) > 0:
            cov = eb / abs(ie)
            h.append(_m("Interest coverage", cov, f"{cov:.1f}×", "Operating profit ÷ interest paid. Below 1.5 = "
                        "struggles to pay interest.", "", _band(cov, [(10, 1.0), (5, 0.5), (2.5, 0.0), (1.5, -0.5)])))
        elif eb is not None and (ie is None or ie == 0) and (_last(debt) or 0) == 0:
            h.append(_m("Interest coverage", None, "no debt", "Operating profit ÷ interest paid.",
                        "no borrowings, no interest burden", 1.0))
        else:
            h.append(_na("Interest coverage", "Operating profit ÷ interest paid.", "interest expense not reported"))
    cats["Financial health"] = h

    # ---------------------------------------------------------------- growth
    g = []
    rg = _growth(rev)
    g.append(_m("Revenue growth (last year)", rg, _pct(rg), "Change in yearly revenue vs the year before.", "",
                _band(rg, [(15, 1.0), (5, 0.5), (0, 0.0), (-10, -0.5)])) if rg is not None else
             _na("Revenue growth (last year)", "Change in yearly revenue.", "fewer than 2 years of revenue"))
    eg = _growth(eps)
    if eg is None and eps is not None and len(eps) >= 2:
        g.append(_m("EPS growth (last year)", None, "N/M", "Change in earnings per share.",
                    "previous year was a loss, so a % change is not meaningful", None, 0.0))
    else:
        g.append(_m("EPS growth (last year)", eg, _pct(eg), "Change in earnings per share vs the year before.", "",
                    _band(eg, [(15, 1.0), (5, 0.5), (0, 0.0), (-15, -0.5)])) if eg is not None else
                 _na("EPS growth (last year)", "Change in earnings per share."))
    ng = _growth(ni)
    g.append(_m("Net income growth", ng, _pct(ng), "Change in total profit. Half weight: overlaps with EPS growth.",
                "", _band(ng, [(15, 1.0), (5, 0.5), (0, 0.0), (-15, -0.5)]), 0.5) if ng is not None else
             _na("Net income growth", "Change in net profit."))
    c3 = _cagr(rev, 3)
    g.append(_m("Revenue growth, 3-year average", c3, _pct(c3), "Compound yearly revenue growth over 3 years: "
                "smooths out one-off years.", "", _band(c3, [(10, 1.0), (4, 0.5), (0, 0.0), (-100, -0.7)]))
             if c3 is not None else _na("Revenue growth, 3-year average", "3-year compound revenue growth.",
                                        "needs 4 years of revenue"))
    cats["Growth"] = g

    # ---------------------------------------------------------------- cash flow
    c = []
    o, n_ = _last(ocf), _last(ni)
    if o is not None:
        if n_ and n_ > 0:
            q = o / n_
            pts = (1.0 if q >= 1 else 0.3 if q >= 0.7 else -0.5) if o > 0 else -1.0
            c.append(_m("Operating cash flow", o, f"{_big(o)} ({q:.1f}× net profit)", "Cash the business actually "
                        "generated. At or above net profit = good-quality earnings.", "", pts))
        else:
            c.append(_m("Operating cash flow", o, _big(o), "Cash the business actually generated.", "",
                        0.5 if o > 0 else -1.0))
    else:
        c.append(_na("Operating cash flow", "Cash generated by the business."))
    f, r_ = _last(fcf), _last(rev)
    if f is not None and not is_fin:
        fm = f / r_ * 100 if r_ else None
        pts = -1.0 if f < 0 else (1.0 if fm is not None and fm >= 10 else 0.5 if fm is not None and fm >= 3 else 0.0)
        c.append(_m("Free cash flow", f, f"{_big(f)}" + (f" ({fm:.1f}% of revenue)" if fm is not None else ""),
                    "Operating cash flow minus capital spending: the cash left for dividends, debt repayment or "
                    "growth.", "", pts))
    elif is_fin:
        c.append(_na("Free cash flow", "Operating cash minus capital spending.", "not meaningful for banks"))
    else:
        c.append(_na("Free cash flow", "Operating cash minus capital spending."))
    cx = _last(capex)
    c.append(_m("Capital expenditure", cx, f"{_big(abs(cx))}" + (f" ({abs(cx) / r_ * 100:.1f}% of revenue)" if r_ else ""),
                "Money spent on equipment, buildings etc. Shown for context, not scored.", "", None, 0.0)
             if cx is not None else _na("Capital expenditure", "Spending on long-term assets."))
    og = _growth(ocf)
    c.append(_m("Operating cash flow growth", og, _pct(og), "Change in operating cash flow vs the year before.", "",
                _band(og, [(10, 1.0), (0, 0.3), (-15, -0.3), (-100, -0.8)]), 0.5) if og is not None else
             _na("Operating cash flow growth", "Change in operating cash flow."))
    cats["Cash flow"] = c

    # ---------------------------------------------------------------- dividends
    dv = []
    annual = None
    if divs is not None and len(divs):
        annual = divs.groupby(divs.index.year).sum()
        this_year = pd.Timestamp.now().year
        annual = annual[annual.index < this_year]           # full years only
    rate, tyld = _num(info.get("dividendRate")), _num(info.get("trailingAnnualDividendYield"))
    px = price or _num(info.get("currentPrice")) or _num(info.get("regularMarketPrice"))
    if rate is not None and px:
        dyp = rate / px * 100                               # indicated yearly dividend ÷ price
    elif tyld is not None:
        dyp = tyld * 100                                    # Yahoo's trailing yield (a fraction)
    else:
        dyp = (annual.iloc[-1] / px * 100) if annual is not None and len(annual) and px else None
    if (dyp is None or dyp == 0) and (annual is None or not len(annual) or annual.iloc[-1] == 0):
        dv.append(_m("Dividend yield", 0.0, "no dividend", "Yearly dividend ÷ share price.",
                     "doesn't pay a dividend (not necessarily bad: growth companies often reinvest)", None, 0.0))
    else:
        dv.append(_m("Dividend yield", dyp, _pct(dyp), "Yearly dividend ÷ share price: the cash return you get for "
                     "holding.", "", _band(dyp, [(5, 1.0), (3, 0.6), (1.5, 0.2), (0, -0.2)])))
        pr = _num(info.get("payoutRatio"))
        if pr is not None:
            prp = pr * 100
            pts = -1.0 if prp > 100 else -0.8 if prp > 90 else 0.0 if prp > 70 else 1.0 if prp >= 30 else 0.3
            dv.append(_m("Payout ratio", prp, _pct(prp), "Share of profit paid out as dividends. Above ~90% leaves "
                         "little room if profits dip.", "", pts))
        else:
            dv.append(_na("Payout ratio", "Dividends ÷ net profit."))
        if annual is not None and len(annual) >= 6 and annual.iloc[-6] > 0 and annual.iloc[-1] > 0:
            dg = ((annual.iloc[-1] / annual.iloc[-6]) ** (1 / 5) - 1) * 100
            dv.append(_m("Dividend growth (5-year average)", dg, _pct(dg), "Compound yearly growth of the dividend per "
                         "share.", "", _band(dg, [(5, 1.0), (0, 0.3), (-100, -0.7)])))
        else:
            dv.append(_na("Dividend growth (5-year average)", "Growth of dividend per share.",
                          "needs 6 full years of dividends"))
        if annual is not None and len(annual):
            last10 = annual[annual.index >= pd.Timestamp.now().year - 10]
            yrs = int((last10 > 0).sum())
            dv.append(_m("Dividend record", yrs, f"paid in {yrs} of the last 10 years", "How consistently dividends "
                         "were paid.", "", 1.0 if yrs >= 8 else 0.3 if yrs >= 5 else -0.3))
    cats["Dividends"] = dv

    # ---------------------------------------------------------------- scores
    for k, items in cats.items():
        for i in items:
            if i["points"] is not None and not i["assessment"]:
                i["assessment"] = _assess(i["points"])
    scores = {}
    for k, items in cats.items():
        sc = [(i["points"], i["weight"]) for i in items if i["points"] is not None and i["weight"] > 0]
        scores[k] = 50 + 50 * sum(p * w for p, w in sc) / sum(w for _, w in sc) if sc else None
    n_all = sum(1 for items in cats.values() for i in items if i["weight"] > 0 or i["value"] is not None
                or i["display"] == "N/A")
    n_have = sum(1 for items in cats.values() for i in items if i["points"] is not None)
    history = pd.DataFrame({"Revenue": rev, "Net income": ni, "EPS": eps, "Operating cash flow": ocf,
                            "Free cash flow": fcf}).dropna(how="all") if any(
        s is not None for s in (rev, ni, eps, ocf, fcf)) else pd.DataFrame()
    return {"categories": cats, "scores": scores, "completeness": n_have / n_all if n_all else 0.0,
            "is_financial": is_fin, "fiscal_year_end": fy, "history": history, "dividends_annual": annual,
            "currency": info.get("financialCurrency") or info.get("currency") or "", "name": info.get("longName"),
            "sector": sector, "industry": industry}


def overall(res, weights=None):
    w = dict(DEFAULT_WEIGHTS, **(weights or {}))
    avail = {k: v for k, v in res["scores"].items() if v is not None and w.get(k, 0) > 0}
    tot = sum(w[k] for k in avail)
    return sum(avail[k] * w[k] for k in avail) / tot if tot else None


def _big(v):
    if v is None:
        return "N/A"
    a = abs(v)
    for div, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if a >= div:
            return f"{v / div:,.2f}{suf}"
    return f"{v:,.0f}"
