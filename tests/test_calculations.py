"""Indicator, scoring and statistics checks against independent references and known answers."""
import numpy as np
import pandas as pd
import pytest

import core
import engine
import fundamentals as F
import indicators as ind
import research as R
import stockinfo as SI


@pytest.fixture(scope="module")
def daily():
    return core.add_indicators(core.load_one("1155.KL", "1d", "2y"))


def _close(a, b, tol):
    m = a.notna() & b.notna()
    assert m.sum() > 50
    assert float(np.max(np.abs(a[m] - b[m]))) < tol


def test_indicators_match_ta_library(daily):
    ta = pytest.importorskip("ta")
    d = daily
    x = ind.compute_all(d)
    H, L, C, V = d.High, d.Low, d.Close, d.Volume
    _close(x.CCI, ta.trend.CCIIndicator(H, L, C, 20).cci(), 1e-6)
    _close(x.MFI, ta.volume.MFIIndicator(H, L, C, V, 14).money_flow_index(), 1e-6)
    _close(x.ROC, ta.momentum.ROCIndicator(C, 12).roc(), 1e-9)
    _close(x.BBU, ta.volatility.BollingerBands(C, 20, 2).bollinger_hband(), 1e-9)
    _close(x.AD, ta.volume.AccDistIndexIndicator(H, L, C, V).acc_dist_index(), 1e-3)
    # Wilder-smoothed series start from a 14-bar average (Wilder's method); `ta` seeds from the first bar, so
    # the two agree once the seed has worn off: compare the last 300 bars.
    _close(d.RSI.iloc[-300:], ta.momentum.RSIIndicator(C, 14).rsi().iloc[-300:], 1e-4)
    _close(d.ATR.iloc[-300:], ta.volatility.AverageTrueRange(H, L, C, 14).average_true_range().iloc[-300:], 1e-4)
    _close(d.ADX.iloc[-300:], ta.trend.ADXIndicator(H, L, C, 14).adx().iloc[-300:], 1e-4)


def test_warmup_is_nan_not_estimated(daily):
    x = ind.compute_all(daily)
    assert x.E200.iloc[:199].isna().all() and x.S200.iloc[:199].isna().all()
    assert daily.RSI.iloc[:14].isna().all() and daily.ADX.iloc[:27].isna().all()


def test_risk_statistics_known_values():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2024-01-01", periods=2000)
    m = pd.Series(rng.normal(0, 0.01, 2000), index=idx)
    assert R.beta(m, m) == pytest.approx(1.0)
    assert R.beta(2 * m, m) == pytest.approx(2.0)
    assert R.ann_vol(m) == pytest.approx(15.87, abs=0.5)
    var, es = R.hist_var(m)
    assert var == pytest.approx(1.645, abs=0.1) and es == pytest.approx(2.06, abs=0.15)
    s = pd.Series([100, 120, 90, 95, 130, 104], index=pd.bdate_range("2025-01-01", periods=6))
    assert R.max_drawdown(s)[0] == pytest.approx(-25.0)
    assert R.period_return(s, 5) == pytest.approx(4.0)


def test_scan_crossovers_match_direct_calculation():
    df = core.scan_universe(tuple(core.BURSA)[:10], "1d", "2y", "trading days")
    for _, r in df.iterrows():
        c = core.load_one(r.Ticker, "1d", "2y").Close
        e50, e200 = ind.ema(c, 50), ind.ema(c, 200)
        sg = np.sign((e50 - e200).dropna().values)
        ch = [i for i in range(1, len(sg)) if sg[i] != sg[i - 1] and sg[i] and sg[i - 1]]
        exp = (int(sg[ch[-1]]), len(sg) - 1 - ch[-1]) if ch and len(sg) - 1 - ch[-1] <= 250 else (0, None)
        age = None if r["X 50/200 age"] != r["X 50/200 age"] or r["X 50/200 age"] is None else int(r["X 50/200 age"])
        assert (r["X 50/200 dir"], age) == exp
        assert r["Above EMA50"] == bool(c.iloc[-1] > e50.iloc[-1])


def _frame(close, vol=1e6):
    idx = pd.bdate_range(end="2026-11-20", periods=len(close))
    c = pd.Series(close, index=idx, dtype=float)
    return pd.DataFrame({"Open": c, "High": c * 1.01, "Low": c * 0.99, "Close": c, "Volume": vol}, index=idx)


@pytest.mark.parametrize("case", ["flat", "short", "no_volume", "gap"])
def test_engine_survives_awkward_data(case):
    n = 61 if case == "short" else 300
    close = np.linspace(1, 2, n)
    if case == "flat":
        close = np.full(n, 1.5)
    if case == "gap":
        close[-5:] *= 0.5
    df = _frame(close, 0 if case in ("flat", "no_volume") else 1e6)
    if case == "flat":
        df[["High", "Low"]] = 1.5
    d, sup, res, sc, label, _ = core.analyse(df)
    plan = core.trade_plan(d, sup, res, label, "days")
    tech = engine.assess(d, sup, res)
    hz = engine.horizons(tech["x"])
    engine.opportunities(tech["x"], sup, res, tech)
    engine.conclusion(tech["x"], sup, res, plan, tech, hz)
    assert tech["overall"] is None or 0 <= tech["overall"] <= 100


def test_fundamentals_bank_and_missing_data():
    data = F.load("1155.KL")
    bank = dict(data, info=dict(data["info"], sector="Financial Services", industry="Banks - Regional"))
    r = F.analyse(bank, 3.0)
    assert r["is_financial"] and r["scores"]["Financial health"] is None
    empty = F.analyse({"info": {}, "income": None, "balance": None, "cashflow": None, "dividends": None}, None)
    assert F.overall(empty) is None and empty["completeness"] == 0


def test_dcf_closed_form():
    # 100 a year forever at 10% = 1,000; net debt is subtracted; r <= g is rejected
    assert SI.dcf(100, 0, 0, 0.10, 0.0, 0, 1)["per_share"] == pytest.approx(1000.0)
    assert SI.dcf(100, 0, 0, 0.10, 0.0, 500, 1)["per_share"] == pytest.approx(500.0)
    assert SI.dcf(100, 0, 0, 0.03, 0.03, 0, 1) is None


def test_portfolio_math(monkeypatch):
    import streamlit as st
    from views import portfolio as PF
    st.session_state.clear() if hasattr(st.session_state, "clear") else None
    tbl, _ = PF.holdings_table([{"ticker": "1155.KL", "qty": 100.0, "cost": 1.0, "date": "2025-01-02", "note": ""},
                                {"ticker": "1155.KL", "qty": 300.0, "cost": 3.0, "date": "2025-02-02", "note": ""}])
    r = tbl.iloc[0]
    assert r["Qty"] == 400 and r["Avg cost"] == pytest.approx(2.5)
    assert r["P/L (RM)"] == pytest.approx(400 * (r["Last"] - 2.5))
    assert r["Weight %"] == pytest.approx(100.0)
