"""Crypto page: a board of the major coins (Yahoo, US$, 24/7 daily candles in UTC) and, for the chosen coin, the same
chart and technical analysis used for stocks. Technical read only: no fundamentals, no trading."""
import pandas as pd
import streamlit as st

import ui
from core import CRYPTO, analyse, display_name, get_many, load_one, normalise_crypto
from pxfmt import pxf


def _chg(c, n):
    c = c.dropna()
    return float((c.iloc[-1] / c.iloc[-1 - n] - 1) * 100) if len(c) > n else None


@st.cache_data(ttl=300, show_spinner=False, max_entries=4)
def _board(tickers: tuple):
    pre = get_many(tickers, "1d", "2y")
    rows = []
    for t in tickers:
        d = pre.get(t)
        if d is None or d.empty:
            try:
                d = load_one(t, "1d", "2y")
            except Exception:
                d = None
        if d is None or len(d) < 60:
            rows.append({"Coin": CRYPTO.get(t, t), "Symbol": t.split("-")[0].rstrip("0123456789"), "Ticker": t})
            continue
        c = d["Close"]
        try:
            x, sup, res, sc, label, _ = analyse(d)
            rsi, sig = float(x["RSI"].iloc[-1]), f"{label} ({sc:+d})"
        except Exception:
            rsi, sig = None, None
        rows.append({"Coin": CRYPTO.get(t, t), "Symbol": t.split("-")[0].rstrip("0123456789"), "Ticker": t,
                     "Price (US$)": float(c.iloc[-1]), "24h": _chg(c, 1), "7d": _chg(c, 7), "30d": _chg(c, 30),
                     "1Y": _chg(c, 365), "From 2Y high": float((c.iloc[-1] / c.max() - 1) * 100),
                     "RSI": rsi, "Signal": sig, "Trend": c.tail(90).tolist(),
                     "Vol (US$)": float(d["Volume"].iloc[-1] * c.iloc[-1]) if d["Volume"].iloc[-1] > 0 else None})
    return pd.DataFrame(rows)


def _pick_row():
    sel = st.session_state.get("cr_board", {}).get("selection", {}).get("rows", [])
    tbl = st.session_state.get("_cr_tbl")
    if sel and tbl is not None and sel[0] < len(tbl):
        st.session_state["cr_pick"] = tbl[sel[0]]
        st.session_state["cr_typed"] = ""


def render(C):
    tickers = tuple(CRYPTO)
    with st.spinner("Loading crypto prices…"):
        tbl = _board(tickers)
    cc = st.column_config
    with ui.panel("Crypto board", "major coins vs US$ · Yahoo Finance · daily candles close 00:00 UTC · tap a row to chart it"):
        cols = (["Symbol", "Price (US$)", "24h", "7d"] if C.mobile else
                ["Coin", "Symbol", "Price (US$)", "24h", "7d", "30d", "1Y", "From 2Y high", "RSI", "Signal", "Vol (US$)", "Trend"])
        cols = [c for c in cols if c in tbl.columns]
        st.session_state["_cr_tbl"] = tbl["Ticker"].tolist()
        st.dataframe(tbl, hide_index=True, width="stretch", height="content", column_order=cols,
                     key="cr_board", on_select=_pick_row, selection_mode="single-row",
                     column_config={"Price (US$)": cc.NumberColumn(format="%,.6g"),
                                    **{k: cc.NumberColumn(format="%+.1f%%") for k in ("24h", "7d", "30d", "1Y", "From 2Y high")},
                                    "RSI": cc.NumberColumn(format="%.0f"),
                                    "Vol (US$)": cc.NumberColumn(format="compact", help="last daily candle volume × price"),
                                    "Trend": cc.LineChartColumn("90 days", width="small")})
        st.caption("Missing values show as blank (N/A) when Yahoo has no data. Crypto trades 24/7, so the last "
                   "daily candle is still forming until 00:00 UTC (08:00 Malaysia time).")

    a, b = st.columns([1.6, 1]) if not C.mobile else (st.container(), st.container())
    if st.session_state.get("cr_pick") not in tickers:
        st.session_state["cr_pick"] = "BTC-USD"
    pick = a.selectbox("Coin", list(tickers), key="cr_pick", format_func=lambda t: f"{display_name(t)}  ·  {t}")
    typed = b.text_input("Or type any coin", placeholder="e.g. PEPE, RENDER, BTCUSDT", key="cr_typed")
    ticker = normalise_crypto(typed) if typed.strip() else pick
    if typed.strip():
        st.caption(f"Looking up **{ticker}** on Yahoo. Some coins use an id suffix (e.g. SUI20947-USD); if nothing "
                   "loads, search the coin on finance.yahoo.com and paste its symbol here.")

    with ui.panel(f"{display_name(ticker)} · {ticker}", "chart + technical read · same engine as the stock analysis"):
        v = ui.render_stock_panel(ticker, "cr")
    if v is None:
        return
    d, sup, res = v["d"], v["sup"], v["res"]
    price = v["price"]
    with ui.panel("Support & resistance", "reaction highs/lows tested 2+ times (Murphy method; see Analysis → Levels)"):
        lv = pd.DataFrame([{"Type": "Resistance", "Level": l, "Tests": n or "projected", "Distance %": (l / price - 1) * 100}
                           for l, n in reversed(res)]
                          + [{"Type": "▶ Price", "Level": price, "Tests": "", "Distance %": 0.0}]
                          + [{"Type": "Support", "Level": l, "Tests": n or "projected", "Distance %": (l / price - 1) * 100}
                             for l, n in sup])
        lv["Tests"] = lv["Tests"].astype(str)
        st.dataframe(lv, hide_index=True, width="stretch",
                     column_config={"Level": cc.NumberColumn(format=pxf(price)),
                                    "Distance %": cc.NumberColumn(format="%+.1f%%")})
    st.markdown(f"<div class='muted' style='font-size:12px'>Technical analysis only. Crypto is far more volatile than "
                f"stocks (check the ATR in the full analysis) and there are no fundamentals to fall back on. Use "
                f"<b>Open full analysis</b> above for scorecard, indicators, timeframes, volume, patterns and forecast. "
                f"Not financial advice.</div>", unsafe_allow_html=True)
