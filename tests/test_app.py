"""Every page renders on PC and phone layouts, including with awkward data (suspended stock with a flat price,
a fresh listing with little history, zero-volume indices, time-zone-aware intraday candles)."""
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import core

PAGES = ["dashboard", "macro", "commodities", "heatmap", "screener", "analysis", "compare", "picks", "notes", "portfolio",
         "risk", "terminal", "multichart", "paper", "tradingview", "calendar", "dividends", "ipo", "learn"]
APP = core.__file__.replace("core.py", "app.py")


def _problems(at):
    errs = [str(e.value)[:200] for e in at.exception]
    errs += [e.value[:200] for e in at.error if "couldn't be shown" in e.value]
    return errs


def _new():
    at = AppTest.from_file(APP, default_timeout=900)
    at.run()
    return at


@pytest.mark.parametrize("mobile", [False, True])
def test_every_page_renders(mobile):
    at = _new()
    if mobile:
        at.sidebar.toggle(key="mobile").set_value(True)
        at.run()
    bad = {}
    for p in PAGES:
        at.session_state["_goto"] = p
        at.run()
        if _problems(at):
            bad[p] = _problems(at)
    assert not bad


@pytest.fixture
def awkward(monkeypatch):
    orig = core._demo

    def quirky(ticker, interval):
        d = orig(ticker, interval).copy()
        if ticker == "9998.KL":                       # suspended: flat price, no volume
            d[["Open", "High", "Low", "Close"]] = 1.5
            d["Volume"] = 0.0
        elif ticker == "9997.KL":                     # fresh listing
            d = d.tail(45)
        elif ticker.startswith("^") or ticker.endswith("=X"):
            d["Volume"] = 0.0
        if interval in ("1h", "15m"):
            d.index = d.index.tz_localize("Asia/Kuala_Lumpur")
        return d
    monkeypatch.setattr(core, "_demo", quirky)
    core.st.cache_data.clear()
    yield
    core.st.cache_data.clear()


def test_pages_with_awkward_data(awkward):
    at = _new()
    at.sidebar.radio(key="sb_market").set_value("My own list")
    at.run()
    at.sidebar.text_area(key="sb_custom").set_value("9998.KL, 9997.KL, 1155.KL, ^KLSE, NVDA")
    at.run()
    at.session_state["term_stocks"] = ["9998.KL", "9997.KL", "1155.KL", "MYR=X"]
    at.session_state["pf_lots"] = [{"ticker": t, "qty": 100.0, "cost": 1.0, "date": "2025-01-02", "note": ""}
                                   for t in ("9998.KL", "9997.KL", "NVDA")]
    at.session_state["rk_src"] = "My portfolio"
    at.session_state["notes"] = {"9998.KL": [{"date": "2026-01-01 10:00", "tag": "Risk", "text": "suspended"}]}
    bad = {}
    for p in PAGES:
        at.session_state["_goto"] = p
        at.run()
        if _problems(at):
            bad[p] = _problems(at)
    at.session_state["_goto"] = "analysis"
    at.run()
    for t in ("9998.KL", "9997.KL", "^KLSE", "MYR=X", "NVDA"):
        at.text_input(key="an_typed").set_value(t)
        at.run()
        if _problems(at):
            bad[f"analysis {t}"] = _problems(at)
    for tf in ("1 Hour", "15 Minutes", "4 Hours"):
        at.sidebar.selectbox(key="sb_tf").set_value(tf)
        at.run()
        if _problems(at):
            bad[f"analysis {tf}"] = _problems(at)
    at.session_state["_goto"] = "terminal"
    at.session_state["term_sym"] = "9998.KL"
    at.run()
    for r in ("1D", "5D", "1Y"):
        at.segmented_control(key="term_rng").set_value(r)
        at.run()
        if _problems(at):
            bad[f"terminal {r}"] = _problems(at)
    assert not bad
