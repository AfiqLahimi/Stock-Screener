"""
Stock Screener & Analysis (Bursa Malaysia + US).
Run:  streamlit run app.py
Price data: Yahoo Finance (yfinance). Bursa ~15 min delayed, US near real-time.

This file is only the entry point: page setup, the shared sidebar, the scan universe and navigation.
Calculations live in core.py / ta_plus.py, layout pieces in ui.py and each page in views/ or its own module.
"""
import time
import types

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Live Stock Screener & Analysis", page_icon="📈", layout="wide")

import prefs  # noqa: E402
import ui  # noqa: E402
from core import (BURSA, DEMO, INTERVALS, NAMES, US, YF_SECTORS, display_name, load_bursa_all,  # noqa: E402
                  load_shariah, normalise)
from views import analysis, calendar_view, compare, dashboard, macro, others, risk, screener  # noqa: E402

ui.keep_state()
prefs.load()
AUTO_MOBILE = ui.detect_mobile()
MOBILE = bool(st.session_state.get("mobile", AUTO_MOBILE))
st.session_state["_mobile"] = MOBILE
ui.inject_css()

C = types.SimpleNamespace()   # shared context handed to every page


def _page(fn, title, icon, path, default=False):
    def run():
        fn(C)
    run.__name__ = path
    return st.Page(run, title=title, icon=icon, url_path=path, default=default)


PAGES = {
    "dashboard": _page(dashboard.render, "Dashboard", "🏠", "dashboard", default=True),
    "screener": _page(screener.render, "Screener", "🔎", "screener"),
    "analysis": _page(analysis.render, "Analysis & Forecast", "📊", "analysis"),
    "picks": _page(others.picks_page, "Picks", "🎯", "picks"),
    "compare": _page(compare.render, "Compare", "⚖️", "compare"),
    "risk": _page(risk.render, "Risk", "🛡️", "risk"),
    "calendar": _page(calendar_view.render, "Calendar", "📅", "calendar"),
    "macro": _page(macro.render, "Macro", "🌐", "macro"),
    "terminal": _page(others.terminal_page, "Terminal", "🖥️", "terminal"),
    "multichart": _page(others.multichart_page, "Multi-Chart", "🧩", "multi-chart"),
    "paper": _page(others.paper_page, "Paper Trade", "💼", "paper-trade"),
    "tradingview": _page(others.tradingview_page, "TradingView", "📺", "tradingview"),
    "heatmap": _page(others.heatmap_page, "Heatmap", "🗺️", "heatmap"),
    "dividends": _page(others.dividends_page, "Dividends", "💰", "dividends"),
    "ipo": _page(others.ipo_page, "IPO", "🚀", "ipo"),
    "learn": _page(others.learn_page, "Learn", "🎓", "learn"),
}
nav = st.navigation({
    "Markets": [PAGES[k] for k in ("dashboard", "macro", "heatmap")],
    "Research": [PAGES[k] for k in ("screener", "analysis", "compare", "picks")],
    "Trade": [PAGES[k] for k in ("terminal", "multichart", "paper", "risk", "tradingview")],
    "Income & events": [PAGES[k] for k in ("calendar", "dividends", "ipo")],
    "Learn": [PAGES["learn"]],
}, position="sidebar", expanded=True)
_go = st.session_state.pop("_goto", None)
if _go in PAGES and PAGES[_go].url_path != nav.url_path:
    st.switch_page(PAGES[_go])

with st.sidebar:
    search_slot = st.container()
    st.markdown("<div class='sec'><span>⚙️ Settings</span></div>", unsafe_allow_html=True)
    st.toggle("📱 Phone layout", value=AUTO_MOBILE, key="mobile",
              help="Smaller charts, compact tables and fewer indicator panels for phone screens. "
                   "Switches on automatically when the app is opened on a phone.")
    market = st.radio("Screener market", ["Bursa: all stocks", "Bursa: top 30 blue chips", "US", "My own list"],
                      index=0, key="sb_market")
    custom = ""
    if market == "My own list":
        custom = st.text_area("Tickers (comma separated)", "MAYBANK, TENAGA, 5398, NVDA, TSLA, AAPL", key="sb_custom")
    shariah_set, shariah_asof = load_shariah()
    shariah_mode = "All"
    if shariah_set and market.startswith("Bursa"):
        shariah_mode = st.radio("Shariah status", ["All", "Shariah-compliant only", "Non-Shariah only"],
                                horizontal=False, help=f"Securities Commission Malaysia list ({shariah_asof})", key="sb_shariah")
    bursa_filter = None
    if market == "Bursa: all stocks":
        with st.expander("🔧 Filter Bursa stocks", expanded=True):
            sectors_sel = st.multiselect("Sectors", YF_SECTORS + ["Other"], default=[],
                                         placeholder="All sectors", key="sb_sectors")
            min_price = st.number_input("Min price (RM)", 0.0, 100.0, 0.0, 0.05, format="%.2f",
                                        help="Skip very cheap (penny) stocks", key="sb_minprice")
            min_value = st.select_slider("Min daily trading value (RM)",
                                         [0, 10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000],
                                         value=0, format_func=lambda v: f"{v:,.0f}",
                                         help="Skip stocks that rarely trade (hard to buy/sell)", key="sb_minvalue")
            max_n = st.select_slider("Scan up to (most traded first)", [50, 100, 200, 300, 500, 1000, 2000],
                                     value=100, help="More stocks = slower first load", key="sb_maxn")
        bursa_filter = (tuple(sectors_sel), min_price, min_value, max_n)
    tf_name = st.selectbox("Timeframe", list(INTERVALS.keys()), index=0, key="sb_tf")
    interval, period, unit = INTERVALS[tf_name]
    closed_only = st.toggle("Signals on completed candles only", value=True, key="closed_only",
                            help="While the market is open the latest candle is still forming and can change. "
                                 "When on, the Analysis tab calculates signals from the last COMPLETED candle "
                                 "(the live price is still shown).")
    st.divider()
    auto = st.toggle("Auto refresh", value=False, key="sb_auto")
    every = st.select_slider("Refresh every (seconds)", [30, 60, 120, 300], value=60, disabled=not auto, key="sb_every")
    st.caption("Data: Yahoo Finance. Bursa is ~15 min delayed; US is near real-time during market hours. "
               "Cached for 60 seconds.")
    if DEMO:
        st.warning("DEMO MODE: fake data")

sector_map = None
with st.spinner("Loading the full Bursa list… (first time only, ~20 seconds)"):
    try:
        bursa_all = load_bursa_all()
    except Exception:
        bursa_all = pd.DataFrame()
if not bursa_all.empty:
    NAMES.update(dict(zip(bursa_all["ticker"], bursa_all["name"])))


def all_symbols():
    """Every symbol the app knows: full Bursa list (Yahoo), the blue chips and the US list."""
    out = list(BURSA)
    if not bursa_all.empty:
        out += list(bursa_all.sort_values("name")["ticker"])
    out += list(US)
    return list(dict.fromkeys(out))


def _open_everywhere():
    t = st.session_state.get("find_any")
    if t:
        ui.goto("analysis", an_typed=t, term_sym=t, pt_symbol=t, pt_other="")


with search_slot:
    _opts = all_symbols()
    st.selectbox(f"🔍 Find any stock ({len(_opts):,} listed)", _opts, index=None, key="find_any",
                 placeholder="Type a name or code…", on_change=_open_everywhere,
                 format_func=lambda t: f"{display_name(t)} · {t}",
                 help="Every Bursa stock Yahoo lists, plus the US list. Picking one opens it in Analysis (and sets it in "
                      "Terminal and the Paper Trade ticket). Not in the list? Type its 4-digit code on those pages.")

if market == "Bursa: all stocks":
    if bursa_all.empty:
        st.warning("Couldn't load the full Bursa list right now, showing the top 30 instead.")
        universe = tuple(BURSA.keys())
    else:
        secs, min_price, min_value, max_n = bursa_filter
        f = bursa_all.copy()
        f["value"] = f["price"].fillna(0) * f["avg_volume"].fillna(0)
        if secs:
            f = f[f["sector"].isin(secs)]
        f = f[(f["price"].fillna(0) >= min_price) & (f["value"] >= min_value)]
        if shariah_mode == "Shariah-compliant only":
            f = f[f["ticker"].isin(shariah_set)]
        elif shariah_mode == "Non-Shariah only":
            f = f[~f["ticker"].isin(shariah_set)]
        f = f.sort_values("value", ascending=False)
        universe = tuple(f["ticker"].head(max_n))
        sector_map = dict(zip(bursa_all["ticker"], bursa_all["sector"]))
        with st.sidebar:
            st.caption(f"📋 {len(bursa_all):,} stocks on Bursa · {len(f):,} match your filter · "
                       f"scanning {len(universe):,}")
elif market == "Bursa: top 30 blue chips":
    universe = tuple(BURSA.keys())
    if shariah_mode == "Shariah-compliant only":
        universe = tuple(t for t in universe if t in shariah_set)
    elif shariah_mode == "Non-Shariah only":
        universe = tuple(t for t in universe if t not in shariah_set)
elif market == "US":
    universe = tuple(US.keys())
else:
    universe = tuple(dict.fromkeys(t for t in (normalise(s) for s in custom.split(",")) if t))

C.__dict__.update(universe=universe, bursa_all=bursa_all, sector_map=sector_map, all_symbols=all_symbols,
                  shariah_set=shariah_set, market=market, interval=interval, period=period, unit=unit,
                  tf_name=tf_name, closed_only=closed_only, mobile=MOBILE)

ui.topbar(nav.title, nav.icon)
ui.ticker_tape()
if MOBILE:
    ui.phone_nav(PAGES, nav.url_path)
nav.run()

st.markdown("<div class='foot'>⚠️ Generated automatically from technical indicators. Not investment advice. "
            "Support/resistance, zones and forecasts are reference levels, not guarantees. Always use a stop loss and "
            "never invest money you can't afford to lose.</div>", unsafe_allow_html=True)

prefs.sync()

if auto:
    time.sleep(every)
    st.cache_data.clear()
    st.rerun()
