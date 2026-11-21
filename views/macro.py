"""Macro page: market regime, interest rates (US Treasury yields, Malaysia OPR), currencies, commodities and
key US economic data (FRED). All free and keyless; anything that doesn't load is shown as N/A."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import indicators as ind
import research as R
import ui
from core import P, load_one, pill
from views.dashboard import index_stats, prefetch, spark

YIELDS = [("^IRX", "US 3-month"), ("^FVX", "US 5-year"), ("^TNX", "US 10-year"), ("^TYX", "US 30-year")]
FX = [("MYR=X", "USD / MYR"), ("SGDMYR=X", "SGD / MYR"), ("EURMYR=X", "EUR / MYR"), ("CNYMYR=X", "CNY / MYR"),
      ("JPYMYR=X", "JPY / MYR"), ("DX-Y.NYB", "US dollar index")]
CMD = [("BZ=F", "Brent crude"), ("CL=F", "WTI crude"), ("NG=F", "Natural gas"), ("GC=F", "Gold"), ("HG=F", "Copper")]


def regime(t, name):
    d = load_one(t, "1d", "2y")
    if d is None or len(d) < 210:
        return None
    c = d["Close"]
    e50, e200 = ind.ema(c, 50), ind.ema(c, 200)
    sl = ind.slope(e50, 20)
    last = float(c.iloc[-1])
    up = last > e200.iloc[-1] and e50.iloc[-1] > e200.iloc[-1]
    down = last < e200.iloc[-1] and e50.iloc[-1] < e200.iloc[-1]
    trend = "Uptrend" if up and sl > 0 else "Downtrend" if down and sl < 0 else "Range / transition"
    hv = ind.hist_vol(c)
    pct = float((hv.tail(252) < hv.iloc[-1]).mean() * 100) if hv.notna().sum() > 60 else None
    vol = "N/A" if pct is None else "calm" if pct < 33 else "volatile" if pct > 66 else "normal volatility"
    return {"name": name, "trend": trend, "vol": vol, "hv": float(hv.iloc[-1]) if hv.notna().any() else None,
            "pct": pct, "e50": float(e50.iloc[-1]), "e200": float(e200.iloc[-1]), "last": last}


def _table(items, fmt="{:,.2f}", pre=None):
    rows = []
    for t, name in items:
        s = index_stats(t, pre)
        if s is None:
            rows.append(f"<tr><td><b>{name}</b></td><td colspan=5 class='muted'>N/A</td></tr>")
            continue

        def c(v):
            return f"<span style='color:{P['muted'] if v is None else P['up'] if v >= 0 else P['down']}'>" + \
                ("N/A" if v is None else f"{v:+.2f}%") + "</span>"
        rows.append(f"<tr><td><b>{name}</b></td><td class='mono'>{fmt.format(s['last'])}</td><td class='mono'>{c(s['1D'])}</td>"
                    f"<td class='mono'>{c(s['1M'])}</td><td class='mono'>{c(s['YTD'])}</td>"
                    f"<td style='width:110px'>{spark(s['spark'], P['up'] if (s['1M'] or 0) >= 0 else P['down'])}</td></tr>")
    return ("<table class='ev'><tr><th></th><th>Last</th><th>1D</th><th>1M</th><th>YTD</th><th>60 days</th></tr>"
            + "".join(rows) + "</table>")


def render(C):
    pre = prefetch([t for t, _ in YIELDS + FX + CMD] + ["^VIX", "^KLSE", "^GSPC"])
    st.markdown("<style>.ev .spark{width:100%;height:22px}.ev td:first-child{white-space:nowrap}"
                ".note{margin-bottom:8px}</style>", unsafe_allow_html=True)
    # ---------------------------------------------------------------- regime
    with ui.panel("Market regime", "trend (EMA 50/200) × volatility (20-day, vs its last year)"):
        cols = st.columns(2) if not C.mobile else [st.container(), st.container()]
        vix = index_stats("^VIX", pre)
        for col, (t, name) in zip(cols, (("^KLSE", "FBM KLCI"), ("^GSPC", "S&P 500"))):
            r = regime(t, name)
            with col:
                if r is None:
                    st.markdown(f"<b>{name}</b> <span class='muted'>N/A (not enough history)</span>", unsafe_allow_html=True)
                    continue
                colr = P["up"] if r["trend"] == "Uptrend" else P["down"] if r["trend"] == "Downtrend" else P["warn"]
                extra = ""
                if t == "^GSPC" and vix:
                    v = vix["last"]
                    extra = f" · VIX {v:.1f} ({'calm' if v < 15 else 'stressed' if v > 25 else 'normal'})"
                hv_txt = "N/A" if r["hv"] is None else "{:.1f}%".format(r["hv"])
                pct_txt = "N/A" if r["pct"] is None else "{:.0f}th percentile".format(r["pct"])
                st.markdown(f"<div style='font-weight:700;font-size:15px'>{name}</div>"
                            f"<div style='margin:6px 0'>{pill(r['trend'], colr)} {pill(r['vol'].capitalize(), P['muted'])}</div>"
                            f"<div class='muted' style='font-size:12.5px'>Price {r['last']:,.2f} · EMA50 {r['e50']:,.2f} · "
                            f"EMA200 {r['e200']:,.2f} · volatility {hv_txt} ({pct_txt}){extra}</div>",
                            unsafe_allow_html=True)
        st.caption("Uptrend = price and EMA 50 above EMA 200 with EMA 50 rising (downtrend = the reverse). Calm trends "
                   "suit trend-following; volatile or range regimes call for smaller positions and wider stops.")

    a, b = st.columns(2) if not C.mobile else (st.container(), st.container())
    # ---------------------------------------------------------------- rates
    with a, ui.panel("Interest rates", "Yahoo (US yields) · Bank Negara (OPR)"):
        opr = R.bnm_opr()
        st.markdown(f"<table class='kv'><tr><td>Malaysia OPR</td><td class='mono'>"
                    + ("N/A (Bank Negara API didn't respond)" if not opr else
                       f"{opr['rate']:.2f}% <span class='muted'>· last change {opr['change']:+.2f} on {opr['date']}</span>")
                    + "</td></tr></table>", unsafe_allow_html=True)
        st.markdown(_table(YIELDS, "{:.2f}%", pre), unsafe_allow_html=True)
        y3, y10 = index_stats("^IRX", pre), index_stats("^TNX", pre)
        if y3 and y10:
            spr = y10["last"] - y3["last"]
            st.markdown(f"<div class='note'>10-year minus 3-month: <b class='mono'>{spr:+.2f}%</b> · " +
                        ("inverted (short rates above long): historically an early recession warning"
                         if spr < 0 else "normal upward curve") + "</div>", unsafe_allow_html=True)
        st.caption("Yield change % here = % change of the yield itself, not percentage points. Higher US yields tend "
                   "to strengthen the dollar and pressure emerging-market stocks.")
    # ---------------------------------------------------------------- fx
    with b, ui.panel("Currencies", "how many ringgit per unit"):
        st.markdown(_table(FX, "{:,.4f}", pre), unsafe_allow_html=True)
        st.caption("USD/MYR going up = ringgit weaker. A weaker ringgit helps exporters (gloves, tech, plantations) "
                   "and hurts importers and companies with US-dollar debt.")
    c, d = st.columns(2) if not C.mobile else (st.container(), st.container())
    with c, ui.panel("Commodities", "futures, US$"):
        st.markdown(_table(CMD, pre=pre), unsafe_allow_html=True)
        st.caption("Crude oil matters for Malaysia's O&G stocks and government revenue. Crude palm oil futures (FCPO) "
                   "are traded on Bursa but aren't available from Yahoo, so they're not shown.")
    # ---------------------------------------------------------------- US macro
    with d, ui.panel("US economy", "FRED (St. Louis Fed)"):
        groups = list(dict.fromkeys(v[2] for v in R.FRED.values()))
        tabs = st.tabs(groups)
        for tab, grp in zip(tabs, groups):
            with tab:
                rows, charts = [], {}
                for sid, (name, kind, g, unit) in R.FRED.items():
                    if g != grp:
                        continue
                    s0 = R.fred(sid)
                    if s0 is None or len(s0) < 14:
                        rows.append(f"<tr><td>{name}</td><td class='muted'>N/A</td></tr>")
                        continue
                    if kind == "yoy":
                        step = 12 if (s0.index[-1] - s0.index[-13]).days > 300 else 52
                        s = (s0 / s0.shift(step) - 1).dropna() * 100
                    elif kind == "diff":
                        s = s0.diff().dropna()
                    else:
                        s = s0
                    last, prev = float(s.iloc[-1]), float(s.iloc[-2])
                    charts[name] = (s.tail(60), unit)
                    val = f"{last:,.0f}k" if unit == "k" else f"{last:.2f}{unit}"
                    rows.append(f"<tr><td>{name}</td><td class='mono'>{val} <span class='muted'>"
                                f"({last - prev:+,.2f} · {s.index[-1]:%b %Y})</span></td></tr>")
                st.markdown("<table class='kv'>" + "".join(rows) + "</table>", unsafe_allow_html=True)
                if charts:
                    pick = st.selectbox("Chart", list(charts), key=f"mac_fred_{grp}", label_visibility="collapsed")
                    s, unit = charts[pick]
                    fig = go.Figure(go.Scatter(x=s.index, y=s.values, line=dict(color=P["accent"], width=2)))
                    fig.update_layout(height=200, margin=dict(l=4, r=4, t=6, b=4), paper_bgcolor="rgba(0,0,0,0)",
                                      plot_bgcolor="rgba(0,0,0,0)", font=dict(size=11))
                    fig.update_yaxes(gridcolor=P["grid"], ticksuffix="%" if unit == "%" else "")
                    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key=f"mac_chart_{grp}")
        st.caption("Inflation and growth figures are % change vs a year earlier unless stated. Malaysian CPI/GDP "
                   "aren't available from a free keyless API here; see DOSM (link on the Calendar page). The full "
                   "commodity board is on the 🛢️ Commodities page.")
