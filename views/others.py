"""Smaller pages: each one hands the shared context to an existing module."""
import streamlit as st

import extras
import learn
import multichart
import paper
import picks
import ta_plus
import terminal
import ui
from core import (CHART_PANELS, LABEL_COLOR, MC_TF, P, add_indicators, analyse, display_name, get_many,
                  make_chart, normalise, pill, rgba, stock_view, trade_plan, tradingview_widget, tv_symbol)


def picks_page(C):
    fundamentals = None
    if not C.bursa_all.empty:
        fundamentals = C.bursa_all.set_index("ticker")[["pe", "div_yield"]]
    picks.render(C.universe, display_name, P, pill, get_many, analyse, trade_plan, add_indicators,
                 learn.detect_patterns, fundamentals)


def heatmap_page(C):
    extras.render_heatmap(C.universe, display_name, P, get_many, analyse, C.sector_map)


def dividends_page(C):
    prices = {}
    if C.universe:
        with st.spinner(f"Loading prices for {len(C.universe)} stocks…"):
            data = get_many(C.universe, C.interval, C.period)
        prices = {t: float(v["Close"].iloc[-1]) for t, v in data.items() if len(v)}
    extras.render_dividends(C.universe, prices, display_name, P, pill)


def ipo_page(C):
    extras.render_ipo(P, pill)


def paper_page(C):
    paper.render(display_name, normalise, P, pill, C.all_symbols, ui.render_stock_panel)


def multichart_page(C):
    multichart.render(display_name, normalise, P, pill, C.all_symbols, C.universe, stock_view, make_chart,
                      LABEL_COLOR, CHART_PANELS, MC_TF, C.mobile)


def terminal_page(C):
    terminal.render(C)


def tradingview_page(C):
    tv_pick = st.session_state.get("an_ticker") or st.session_state.get("pick", "1155.KL")
    c1, c2 = st.columns([3, 1])
    tv_h = c1.slider("Chart height (px)", 400, 1300, 520 if C.mobile else 850, 50)
    tv_int = c2.selectbox("Interval", ["D", "W", "240", "60", "15"], index=0,
                          format_func=lambda v: {"D": "Daily", "W": "Weekly", "240": "4 hours",
                                                 "60": "1 hour", "15": "15 min"}[v])
    st.caption(f"Interactive TradingView chart for **{display_name(tv_pick)}** (the stock open in Analysis). "
               "Change the symbol inside the chart, add indicators and draw your own lines. If it says the symbol is "
               "only available on TradingView, that exchange can't be embedded; use the Analysis chart instead.")
    tradingview_widget(tv_symbol(tv_pick), height=tv_h, interval=tv_int)


def learn_page(C):
    learn.set_palette(P)
    learn.render()
