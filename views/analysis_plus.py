"""Phase 5 views inside Analysis & Forecast: Scorecard, Indicators, Fundamentals and the horizon assessment
shown on the Forecast tab. Calculations live in engine.py, fundamentals.py and indicators.py."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import engine
import fundamentals as F
import indicators as ind
import terminal
import ui
from core import P, pill, rgba

IDX_LIKE = ("^", "=X", "=F", "-USD")


def is_company(ticker):
    return not any(k in ticker for k in IDX_LIKE)


def tech_weights():
    return {g: st.session_state.get(f"w_t_{g.replace(' ', '_').lower()}", v) for g, v in engine.DEFAULT_WEIGHTS.items()}


def fund_weights():
    return {c: st.session_state.get(f"w_f_{c.replace(' ', '_').lower()}", v) for c, v in F.DEFAULT_WEIGHTS.items()}


def _col(score):
    if score is None:
        return P["muted"]
    return P["up"] if score >= 60 else P["down"] if score <= 40 else P["warn"] if score < 50 else "#9fb3c8"


def _gauge(title, score, sub=""):
    sc = "N/A" if score is None else f"{score:.0f}"
    w = 0 if score is None else score
    return (f"<div class='sg'><div class='sg-t'>{title}</div><div class='sg-v' style='color:{_col(score)}'>{sc}"
            f"<span>/100</span></div><div class='bar'><i style='width:{w:.0f}%;background:{_col(score)}'></i></div>"
            f"<div class='sg-s'>{sub}</div></div>")


def _items_table(items):
    rows = []
    for i in items:
        if i["points"] is None:
            eff = "info"
        else:
            eff = f"{i['points']:+.2f} × {i['weight']:g}"
        rows.append(f"<tr><td><b>{i['name']}</b><div class='muted' style='font-size:11.5px'>{i.get('note', '')}</div></td>"
                    f"<td class='mono'>{i['value']}</td><td>{i['reading']}</td>"
                    f"<td class='mono' style='color:{P['up'] if (i['points'] or 0) > 0 else P['down'] if (i['points'] or 0) < 0 else P['muted']}'>{eff}</td></tr>")
    return ("<table class='ev'><tr><th>Evidence</th><th>Value</th><th>Reading</th><th>Points × weight</th></tr>"
            + "".join(rows) + "</table>")


CSS = ""  # moved into ui.CSS so every page shares these styles


def inject():
    """Kept for compatibility: the styles now live in ui.CSS."""


# ============================================================================
# Scorecard
# ============================================================================
def render_scorecard(ticker, tech, plan, sup, res, hz, ops, price, mobile):
    fres = F.analyse(F.load(ticker), price) if is_company(ticker) else None
    f_over = F.overall(fres, fund_weights()) if fres else None
    split = st.session_state.get("w_combo", 60)
    t_over = tech["overall"]
    if t_over is not None and f_over is not None:
        comb = (split * t_over + (100 - split) * f_over) / 100
        comp = (tech["completeness"] + fres["completeness"]) / 2
    else:
        comb, comp = t_over, tech["completeness"]
    lab = engine.label_of(comb)
    st.markdown("<div class='sg-row'>" +
                _gauge("Technical score", t_over, f"{tech['label']} · confidence {tech['confidence']}") +
                _gauge("Fundamental score", f_over, "N/A for indices / FX" if fres is None else
                       f"{fres['completeness']:.0%} of metrics available") +
                _gauge("Overall assessment", comb, f"<b style='color:{_col(comb)}'>{lab}</b> · "
                       f"{'technical ' + str(split) + '% / fundamental ' + str(100 - split) + '%' if f_over is not None else 'technical only'}") +
                _gauge("Data completeness", comp * 100 if comp is not None else None,
                       "share of evidence that could be calculated") + "</div>", unsafe_allow_html=True)
    st.caption("Scores run 0–100 (50 = neutral). They summarise today's evidence and are **not** the probability that "
               "a trade will work. The −100…+100 Signal score used in the Screener is the quicker classic model.")

    with ui.panel("Technical score by group", "click a group to see its evidence"):
        st.markdown("<div class='gs-row'>" + "".join(
            _gauge(g, tech["scores"][g], f"weight {tech['weights'][g]}%") for g in engine.GROUPS) + "</div>",
                    unsafe_allow_html=True)
        for g in engine.GROUPS:
            sc = tech["scores"][g]
            with st.expander(f"{g}: {'N/A' if sc is None else f'{sc:.0f}/100 · ' + engine.label_of(sc)}"):
                st.markdown(_items_table(tech["groups"][g]), unsafe_allow_html=True)
        st.caption("Each piece of evidence scores −1 (bearish) to +1 (bullish) and is weighted; a group = 50 + 50 × "
                   "weighted average. Rows marked “info” are shown but not scored because they repeat other evidence "
                   "(no double counting).")

    if fres is not None:
        with ui.panel("Fundamental score by category", f"FY ending {fres['fiscal_year_end']:%b %Y}"
                      if fres["fiscal_year_end"] is not None else "Yahoo Finance"):
            st.markdown("<div class='sg-row' style='grid-template-columns:repeat(6,minmax(0,1fr))'>" + "".join(
                _gauge(c, fres["scores"][c], f"weight {fund_weights()[c]}%") for c in F.CATEGORIES) + "</div>",
                        unsafe_allow_html=True)
            st.caption("Details and explanations are on the 🏦 Fundamentals tab." +
                       (" This is a bank/insurer: debt and liquidity ratios are skipped." if fres["is_financial"] else ""))

    with ui.panel("Analysis conclusion", "conditions to watch, not predictions"):
        con = engine.conclusion(tech["x"], sup, res, plan, tech, hz)
        rows = []
        for k, v in con.items():
            if k == "Take-profit scenarios":
                v = "<br>".join(f"{lab_}: <b class='mono'>{lv:,.3f}</b> · R:R {'N/A' if r is None else f'{r:.1f}'}"
                                for lab_, lv, r in v) or "N/A (price already above the listed targets)"
            rows.append(f"<tr><td>{k}</td><td>{v}</td></tr>")
        st.markdown("<table class='concl'>" + "".join(rows) + "</table>", unsafe_allow_html=True)
        if ops:
            st.markdown("".join(f"<div class='op'><b>{t}</b> · {d}</div>" for t, d in ops), unsafe_allow_html=True)

    with st.expander("⚖️ Scoring weights (your settings are kept)"):
        st.caption("Change how much each group counts. Groups with no data are skipped and the rest re-weighted.")
        cols = st.columns(5) if not mobile else [st.container()] * 5
        for c_, (g, v) in zip(cols, engine.DEFAULT_WEIGHTS.items()):
            c_.slider(g, 0, 50, key=f"w_t_{g.replace(' ', '_').lower()}", step=5,
                      **({} if f"w_t_{g.replace(' ', '_').lower()}" in st.session_state else {"value": v}))
        cols = st.columns(6) if not mobile else [st.container()] * 6
        for c_, (cat, v) in zip(cols, F.DEFAULT_WEIGHTS.items()):
            c_.slider(cat, 0, 50, key=f"w_f_{cat.replace(' ', '_').lower()}", step=5,
                      **({} if f"w_f_{cat.replace(' ', '_').lower()}" in st.session_state else {"value": v}))
        st.slider("Overall: technical share (%)", 0, 100, key="w_combo", step=10,
                  **({} if "w_combo" in st.session_state else {"value": 60}),
                  help="The rest is fundamental. 100 = technical only.")
        st.button("↺ Default weights", key="w_reset", on_click=_reset_weights)


def _reset_weights():
    for g, v in engine.DEFAULT_WEIGHTS.items():
        st.session_state[f"w_t_{g.replace(' ', '_').lower()}"] = v
    for c, v in F.DEFAULT_WEIGHTS.items():
        st.session_state[f"w_f_{c.replace(' ', '_').lower()}"] = v
    st.session_state["w_combo"] = 60


# ============================================================================
# Indicators
# ============================================================================
def render_indicators(ticker, name, d, tech, interval, mobile):
    x = tech["x"]
    last = x.iloc[-1]
    with ui.panel("Indicator chart", "pick what to draw · same chart engine as the Terminal"):
        c1, c2 = st.columns(2) if not mobile else (st.container(), st.container())
        ov = c1.multiselect("Overlays", terminal.OVERLAYS, key="ai_ov",
                            **({} if "ai_ov" in st.session_state else
                               {"default": ["EMA 9", "EMA 20", "EMA 50", "EMA 200", "Bollinger (20, 2)"]}))
        pn = c2.multiselect("Panels", terminal.PANES, key="ai_pn",
                            **({} if "ai_pn" in st.session_state else {"default": ["Volume", "RSI", "MACD"]}))
        levels = []
        fig = terminal.build_chart(d.tail(260), name, P, rgba, "Candles", ov, pn, levels,
                                   interval in ("15m", "1h", "4h"), mobile, 360 if mobile else 480)
        st.plotly_chart(fig, width="stretch", key="ai_chart",
                        config={"displaylogo": False, "scrollZoom": True, "modeBarButtonsToAdd": terminal.DRAW_TOOLS})

    def v(col, fmt="{:,.2f}"):
        val = last.get(col)
        return "N/A" if val is None or (isinstance(val, float) and np.isnan(val)) else fmt.format(val)
    c = float(last["Close"])
    rows = {
        "Moving averages": [("EMA 9 / 20 / 50", f"{v('E9', '{:,.3f}')} / {v('E20', '{:,.3f}')} / {v('E50', '{:,.3f}')}"),
                            ("EMA 100 / 200", f"{v('E100', '{:,.3f}')} / {v('E200', '{:,.3f}')}"),
                            ("SMA 20 / 50", f"{v('S20', '{:,.3f}')} / {v('S50', '{:,.3f}')}"),
                            ("SMA 100 / 200", f"{v('S100', '{:,.3f}')} / {v('S200', '{:,.3f}')}")],
        "Momentum": [("RSI (14)", v("RSI", "{:.1f}")), ("MACD / signal (12,26,9)", f"{v('MACD', '{:+.4f}')} / {v('MACDsig', '{:+.4f}')}"),
                     ("Stochastic %K / %D (14,3,3)", f"{v('STK', '{:.1f}')} / {v('STD', '{:.1f}')}"),
                     ("Stoch RSI %K / %D (14,14,3,3)", f"{v('SRK', '{:.1f}')} / {v('SRD', '{:.1f}')}"),
                     ("ADX / +DI / −DI (14)", f"{v('ADX', '{:.1f}')} / {v('PDI', '{:.1f}')} / {v('MDI', '{:.1f}')}"),
                     ("ROC (12)", v("ROC", "{:+.2f}%")), ("CCI (20)", v("CCI", "{:.0f}"))],
        "Volatility": [("Bollinger lower / mid / upper (20, 2)", f"{v('BBL', '{:,.3f}')} / {v('BBM', '{:,.3f}')} / {v('BBU', '{:,.3f}')}"),
                       ("Bollinger width", v("BBW", "{:.2f}%")),
                       ("Keltner lower / upper (20, 2×ATR)", f"{v('KCL', '{:,.3f}')} / {v('KCU', '{:,.3f}')}"),
                       ("Squeeze (BB inside KC)", "Yes" if bool(last.get("SQZ")) else "No"),
                       ("ATR (14)", f"{v('ATR', '{:,.4f}')} ({float(last['ATR']) / c * 100:.2f}%)" if v("ATR") != "N/A" else "N/A"),
                       ("Historical volatility (20, annualised)", v("HV", "{:.1f}%"))],
        "Volume": [("Volume / SMA 20", f"{v('Volume', '{:,.0f}')} / {v('VolAvg', '{:,.0f}')}"), ("RVOL", v("RVOL", "{:.2f}×")),
                   ("OBV", v("OBV2", "{:,.0f}")), ("A/D line", v("AD", "{:,.0f}")), ("MFI (14)", v("MFI", "{:.1f}")),
                   ("VWAP (20-bar rolling)", v("VWAP20", "{:,.3f}"))],
    }
    cols = st.columns(2) if not mobile else [st.container(), st.container()]
    for i, (grp, rs) in enumerate(rows.items()):
        with cols[i % 2], ui.panel(grp, "latest completed candle"):
            st.markdown("<table class='kv'>" + "".join(f"<tr><td>{a}</td><td class='mono'>{b}</td></tr>" for a, b in rs)
                        + "</table>", unsafe_allow_html=True)
    stc = ind.structure(x)
    fib = ind.fibonacci(x)
    proj, _ = engine.projections(x)
    with ui.panel("Price action & structure", "how the levels are found"):
        a, b = st.columns(2) if not mobile else (st.container(), st.container())
        with a:
            tl_rows = []
            for kind in ("support", "resistance"):
                tl = ind.trendline(x, kind)
                tl_rows.append((f"{kind.title()} trendline now", "N/A" if not tl else f"{tl['value']:,.3f} "
                                f"({tl['slope'] / c * 100:+.2f}% per bar)"))
            st.markdown("<table class='kv'>" + "".join(f"<tr><td>{p}</td><td>{q}</td></tr>" for p, q in
                        [("Structure", stc["label"])] + tl_rows + [(p, f"{q:,.3f}") for p, q in proj]) + "</table>",
                        unsafe_allow_html=True)
            st.caption("Swing highs/lows = a bar higher (lower) than the 3 bars on each side; the last 3 bars can't be "
                       "confirmed yet. Trendlines join the last two swing lows (support) or highs (resistance).")
        with b:
            if fib:
                st.markdown(f"<div class='muted' style='font-size:12.5px'>Latest major swing: {fib['direction']} from "
                            f"{fib['low'] if fib['direction'] == 'up' else fib['high']:,.3f} to "
                            f"{fib['high'] if fib['direction'] == 'up' else fib['low']:,.3f}</div>", unsafe_allow_html=True)
                st.markdown("<table class='kv'>" + "".join(f"<tr><td>Retracement {k}</td><td class='mono'>{lv:,.3f}</td></tr>"
                            for k, lv in fib["retracements"].items()) + "".join(
                            f"<tr><td>Extension {k}</td><td class='mono'>{lv:,.3f}</td></tr>"
                            for k, lv in fib["extensions"].items()) + "</table>", unsafe_allow_html=True)
                st.caption("Fibonacci levels of the biggest move in the last 120 bars: retracements are common pullback "
                           "areas, extensions are common targets. Reference levels, not guarantees.")
            else:
                st.markdown("<div class='muted'>Fibonacci: N/A (not enough history)</div>", unsafe_allow_html=True)
    st.caption("Parameters are the same everywhere in the app: " + ", ".join(
        f"{k} {v if not isinstance(v, (list, tuple)) else '/'.join(map(str, v))}" for k, v in ind.PARAMS.items()) +
               ". Values inside an indicator's warm-up period are N/A.")


# ============================================================================
# Fundamentals
# ============================================================================
def render_fundamentals(ticker, price, mobile):
    if not is_company(ticker):
        st.info("Fundamentals apply to companies; this is an index, currency or commodity.")
        return
    with st.spinner("Loading financial statements…"):
        fres = F.analyse(F.load(ticker), price)
    over = F.overall(fres, fund_weights())
    fy = fres["fiscal_year_end"]
    st.markdown("<div class='sg-row'>" + _gauge("Fundamental score", over,
                f"{fres['completeness']:.0%} of metrics available") + "".join(
                _gauge(c, fres["scores"][c], "") for c in F.CATEGORIES[:3]) + "</div>", unsafe_allow_html=True)
    st.caption(f"{fres['name'] or ticker} · {fres['sector'] or 'sector N/A'} · {fres['industry'] or ''} · "
               f"latest annual report: {'N/A' if fy is None else f'{fy:%b %Y}'} · figures in {fres['currency'] or 'reporting currency'}. "
               "Ratios like P/E use Yahoo's latest (TTM) values; growth and cash flow use annual reports, so they can "
               "lag the latest quarter. N/A = not reported, never estimated." +
               (" Banks/insurers: debt, liquidity and margin ratios are skipped because they don't fit a bank's balance "
                "sheet." if fres["is_financial"] else ""))
    cols = st.columns(2) if not mobile else [st.container(), st.container()]
    for i, cat in enumerate(F.CATEGORIES):
        sc = fres["scores"][cat]
        with cols[i % 2], ui.panel(cat, "N/A" if sc is None else f"{sc:.0f}/100 · {engine.label_of(sc).replace('Bullish', 'strong').replace('Bearish', 'weak').replace('Neutral', 'average')}"):
            rows = "".join(
                f"<tr><td><b>{m['name']}</b><div class='muted' style='font-size:11.5px'>{m['meaning']}</div></td>"
                f"<td class='mono' style='white-space:nowrap'>{m['display']}</td>"
                f"<td style='color:{P['up'] if (m['points'] or 0) > 0.2 else P['down'] if (m['points'] or 0) < -0.2 else P['muted']}'>"
                f"{m['assessment'] or ('N/A' if m['display'] == 'N/A' else '')}</td></tr>" for m in fres["categories"][cat])
            st.markdown(f"<table class='ev'>{rows}</table>", unsafe_allow_html=True)
    h = fres["history"]
    if not h.empty:
        with ui.panel("History", "annual reports"):
            sel = st.segmented_control("Series", [c for c in h.columns if h[c].notna().any()], key="fu_hist",
                                       default="Revenue" if "Revenue" in h else None, label_visibility="collapsed")
            sel = sel or h.columns[0]
            s = h[sel].dropna()
            fig = go.Figure(go.Bar(x=[f"FY{ts:%Y}" for ts in s.index], y=s.values,
                                   marker_color=[P["up"] if v >= 0 else P["down"] for v in s.values],
                                   text=[F._big(v) if sel != "EPS" else f"{v:.3f}" for v in s.values], textposition="outside"))
            fig.update_layout(height=280, margin=dict(l=4, r=4, t=10, b=4),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font=dict(color=P["text"], size=11))
            fig.update_yaxes(gridcolor=P["grid"])
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key="fu_chart")
            yo = s.pct_change().iloc[1:] * 100 if (s > 0).all() else None
            if yo is not None and len(yo):
                st.caption("Year-on-year: " + " · ".join(f"FY{i:%Y} {v:+.1f}%" for i, v in yo.items()))
    da = fres["dividends_annual"]
    if da is not None and len(da):
        with ui.panel("Dividend history", "per share, by calendar year (full years)"):
            fig = go.Figure(go.Bar(x=[str(y) for y in da.index[-12:]], y=da.values[-12:], marker_color=P["accent"],
                                   text=[f"{v:.3f}" for v in da.values[-12:]], textposition="outside"))
            fig.update_layout(height=240, margin=dict(l=4, r=4, t=10, b=4),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font=dict(color=P["text"], size=11))
            fig.update_yaxes(gridcolor=P["grid"])
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key="fu_div")


# ============================================================================
# Forecast tab: horizons
# ============================================================================
def render_horizons(hz, ops, x):
    with ui.panel("Trend assessment by horizon", "current evidence, not a price prediction"):
        st.markdown("<div class='hz'>" + "".join(
            f"<div><div class='sg-t'>{h['name']} · {h['span']}</div>"
            f"<div style='margin:6px 0'>{pill(h['label'], _col(h['score']))}</div>"
            f"<div class='muted' style='font-size:12px;line-height:1.55'>{'<br>'.join(h['evidence'])}</div></div>"
            for h in hz) + "</div>", unsafe_allow_html=True)
        if ops:
            st.markdown("".join(f"<div class='op'><b>{t}</b> · {d}</div>" for t, d in ops), unsafe_allow_html=True)
        proj, fib = engine.projections(x)
        if proj or fib:
            rows = [(p, q) for p, q in proj]
            if fib:
                rows += [(f"Fibonacci extension {k}", v) for k, v in list(fib["extensions"].items())[:2]]
            st.markdown("<div class='muted' style='margin-top:6px;font-size:12px'>Projected reference levels: " +
                        " · ".join(f"{p} <b class='mono'>{q:,.3f}</b>" for p, q in rows) + "</div>",
                        unsafe_allow_html=True)
        st.caption("Each horizon combines a few trend checks (short: EMA 9/20, 12-bar momentum, RSI; medium: EMA 50, "
                   "60-bar change, MACD; long: EMA 200, golden/death-cross state, 52-week position). Tags are "
                   "technical scenarios to watch. The statistical forecast below is a separate model.")
