"""Compare page: 2–4 stocks side by side (performance, risk, technical and fundamental scores, correlation)."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import engine
import fundamentals as F
import research as R
import ui
from core import P, analyse, display_name, load_one
from views.analysis_plus import fund_weights, is_company, tech_weights

RANGES = {"3M": 63, "6M": 126, "1Y": 252, "2Y": 504, "5Y": 1260}
LINE_COLS = ["#7b93f5", "#3fb68b", "#e3a34b", "#e88ad1"]


@st.cache_data(ttl=600, show_spinner=False, max_entries=80)
def profile(t, tw, fw):
    df = load_one(t, "1d", "2y")
    out = {"ticker": t}
    if df is None or len(df) < 60:
        return out
    d, sup, res, sc, label, _ = analyse(df)
    tech = engine.assess(d, sup, res, weights=dict(tw))
    out.update(signal=label, signal_score=sc, tech=tech["overall"], tech_label=tech["label"],
               groups={g: tech["scores"][g] for g in engine.GROUPS}, price=float(d["Close"].iloc[-1]))
    if is_company(t):
        fr = F.analyse(F.load(t), out["price"])
        out["fund"] = F.overall(fr, dict(fw))
        flat = {m["name"]: m for items in fr["categories"].values() for m in items}
        out["metrics"] = {k: flat[k]["display"] if k in flat else "N/A" for k in
                          ("P/E (TTM)", "Price / book", "Return on equity", "Net margin", "Debt / equity",
                           "Revenue growth (last year)", "Dividend yield", "Free cash flow")}
        out["metric_vals"] = {k: flat[k]["value"] if k in flat else None for k in out["metrics"]}
    return out


def render(C):
    wl = st.session_state.get("term_stocks", [])
    default = [t for t in wl if not t.startswith("^")][:3] or ["1155.KL", "1295.KL", "1023.KL"]
    c1, c2 = st.columns([3, 1.2]) if not C.mobile else (st.container(), st.container())
    opts = list(dict.fromkeys(default + list(st.session_state.get("cmp_syms") or []) + C.all_symbols()))
    syms = c1.multiselect("Stocks to compare (2–4)", opts, max_selections=4, key="cmp_syms",
                          format_func=lambda t: f"{display_name(t)} · {t}",
                          **({} if "cmp_syms" in st.session_state else {"default": default}))
    rng = c2.segmented_control("Period", list(RANGES), default="1Y", key="cmp_rng") or "1Y"
    if len(syms) < 2:
        st.info("Pick at least two stocks to compare.")
        return
    with st.spinner("Loading prices…"):
        px = R.closes(syms, "5y" if rng == "5Y" else "2y")
    px = px.dropna(how="all")
    n = RANGES[rng]
    win = px.tail(n)

    # ---------------------------------------------------------------- performance
    with ui.panel("Performance", f"rebased to 100 at the start of {rng} · each stock in its own currency"):
        fig = go.Figure()
        for i, t in enumerate(syms):
            s = win[t].dropna() if t in win else pd.Series(dtype=float)
            if len(s) < 2:
                continue
            reb = s / s.iloc[0] * 100
            fig.add_trace(go.Scatter(x=reb.index, y=reb, name=display_name(t), line=dict(color=LINE_COLS[i], width=2),
                                     hovertemplate=f"{display_name(t)} %{{y:.1f}}<extra></extra>"))
        fig.add_hline(y=100, line=dict(color=P["muted"], width=1, dash="dot"))
        fig.update_layout(template="plotly_dark", height=340, margin=dict(l=4, r=4, t=10, b=4), hovermode="x unified",
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          legend=dict(orientation="h", y=1.08, x=0), font=dict(color=P["text"], size=11))
        fig.update_yaxes(gridcolor=P["grid"], side="right")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key="cmp_perf")

    # ---------------------------------------------------------------- side by side
    tw, fw = tuple(sorted(tech_weights().items())), tuple(sorted(fund_weights().items()))
    with st.spinner("Scoring each stock…"):
        prof = {t: profile(t, tw, fw) for t in syms}
    rets = pd.DataFrame({t: R.returns(px[t].dropna()) for t in px.columns})
    rows = {}

    def pct(v):
        return "N/A" if v is None else f"{v:+.1f}%"
    for t in syms:
        s = px[t].dropna() if t in px else pd.Series(dtype=float)
        b = R.benchmark_of(t)
        bpx = R.closes([b], "2y")
        beta = R.beta(rets[t].tail(252), R.returns(bpx[b]).tail(252)) if b in bpx and t in rets else None
        dd, _, _ = R.max_drawdown(s.tail(n))
        p = prof[t]
        rows[t] = {
            "Price": "N/A" if not len(s) else f"{s.iloc[-1]:,.3f}",
            "1-month return": pct(R.period_return(s, 21)), "3-month return": pct(R.period_return(s, 63)),
            "1-year return": pct(R.period_return(s, 252)),
            "Volatility (1y, annual)": "N/A" if R.ann_vol(rets[t].tail(252)) is None else f"{R.ann_vol(rets[t].tail(252)):.1f}%",
            f"Max drawdown ({rng})": pct(dd),
            f"Beta vs {'KLCI' if b == '^KLSE' else 'S&P 500'}": "N/A" if beta is None else f"{beta:.2f}",
            "Signal (classic)": p.get("signal", "N/A") + ("" if "signal_score" not in p else f" ({p['signal_score']:+d})"),
            "Technical score": "N/A" if p.get("tech") is None else f"{p['tech']:.0f} · {p['tech_label']}",
            "Fundamental score": "N/A" if p.get("fund") is None else f"{p['fund']:.0f}",
            **(p.get("metrics") or {}),
        }
    table = pd.DataFrame(rows)
    table.columns = [display_name(t) for t in table.columns]
    with ui.panel("Side by side", "N/A = not available from Yahoo"):
        st.dataframe(table, width="stretch", height="content",
                     column_config={c: st.column_config.TextColumn(c) for c in table.columns})
        best = []
        for lab, key, hi in (("Highest technical score", "tech", True), ("Highest fundamental score", "fund", True)):
            vals = {t: prof[t].get(key) for t in syms if prof[t].get(key) is not None}
            if vals:
                w = max(vals, key=vals.get) if hi else min(vals, key=vals.get)
                best.append(f"{lab}: <b>{display_name(w)}</b> ({vals[w]:.0f})")
        vols = {t: R.ann_vol(rets[t].tail(252)) for t in syms if t in rets}
        vols = {t: v for t, v in vols.items() if v is not None}
        if vols:
            w = min(vols, key=vols.get)
            best.append(f"Lowest volatility: <b>{display_name(w)}</b> ({vols[w]:.1f}%)")
        if best:
            st.markdown("<div class='muted' style='font-size:12.5px'>" + " · ".join(best) + "</div>",
                        unsafe_allow_html=True)

    # ---------------------------------------------------------------- scores + correlation
    a, b = st.columns(2) if not C.mobile else (st.container(), st.container())
    with a, ui.panel("Technical score by group", "0–100, 50 = neutral"):
        fig = go.Figure()
        for i, t in enumerate(syms):
            g = prof[t].get("groups") or {}
            fig.add_trace(go.Bar(x=engine.GROUPS, y=[g.get(k) for k in engine.GROUPS], name=display_name(t),
                                 marker_color=LINE_COLS[i]))
        fig.add_hline(y=50, line=dict(color=P["muted"], width=1, dash="dot"))
        fig.update_layout(template="plotly_dark", height=300, barmode="group", margin=dict(l=4, r=4, t=10, b=4),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", yaxis_range=[0, 100],
                          legend=dict(orientation="h", y=1.12, x=0), font=dict(color=P["text"], size=11))
        fig.update_yaxes(gridcolor=P["grid"])
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key="cmp_groups")
    with b, ui.panel("Correlation of daily returns", f"last {rng}"):
        cr = rets[syms].tail(n).corr()
        names = [display_name(t) for t in cr.columns]
        fig = go.Figure(go.Heatmap(z=cr.values, x=names, y=names, zmin=-1, zmax=1,
                                   colorscale=[[0, P["down"]], [0.5, P["card2"]], [1, P["up"]]],
                                   text=np.round(cr.values, 2), texttemplate="%{text}", showscale=False))
        fig.update_layout(template="plotly_dark", height=300, margin=dict(l=4, r=4, t=10, b=4),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=P["text"], size=11))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key="cmp_corr")
        st.caption("1 = always move together, 0 = unrelated, −1 = opposite. Holding stocks with lower correlation "
                   "spreads risk better.")
    st.caption("Returns and drawdowns use daily closes from Yahoo (not adjusted for dividends). Beta = how much the "
               "stock moved for each 1% move of its home index over the last year.")
