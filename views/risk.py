"""Risk page: how risky is the paper portfolio (or the watchlist)?

Volatility, beta, 1-day Value at Risk, drawdown, correlation, risk contribution, sector and currency
concentration and simple stress tests, all from one year of daily closes. History describes past behaviour;
it is not a forecast of future losses.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import fundamentals as F
import paper
import research as R
import ui
from core import P, display_name, load_one
from views.analysis_plus import is_company


def _usdmyr():
    d = load_one("MYR=X", "1d", "1mo")
    return float(d["Close"].iloc[-1]) if d is not None and len(d) else None


def holdings(source, C):
    """[(ticker, value in RM, extra dict)] for the chosen source."""
    if source == "Paper Trade positions":
        acct = paper.load_account()
        pos, _ = paper.mark(acct)
        if pos.empty:
            return [], None
        fx = _usdmyr()
        out = []
        for _, r in pos.iterrows():
            v = r["Market value"] * (1 if r["Ccy"] == "MYR" else (fx or np.nan))
            out.append((r["Ticker"], v, {"sl": r["Stop loss"], "last": r["Last"], "qty": r["Qty"], "ccy": r["Ccy"]}))
        return out, fx
    if source == "My portfolio":
        from views.portfolio import holdings_table
        tbl, fx = holdings_table(st.session_state.get("pf_lots", []))
        if tbl.empty:
            return [], fx
        return [(r["Ticker"], r["Value (RM)"], {"ccy": r["Ccy"]}) for _, r in tbl.iterrows()
                if r["Value (RM)"] == r["Value (RM)"]], fx
    wl = [t for t in st.session_state.get("term_stocks", []) if is_company(t)]
    return [(t, 1.0, {}) for t in wl], None


def render(C):
    src = st.segmented_control("Portfolio", ["My portfolio", "Paper Trade positions", "Watchlist (equal weight)"],
                               default="My portfolio" if st.session_state.get("pf_lots") else "Paper Trade positions",
                               key="rk_src") or "Paper Trade positions"
    hold, fx = holdings(src, C)
    if not hold:
        st.info("Nothing to analyse here yet: add your holdings in 📁 My Portfolio, open a position in 💼 Paper Trade, "
                "or switch to “Watchlist (equal weight)”.")
        return
    if src != "Watchlist (equal weight)" and fx is None and any(h[2].get("ccy") == "USD" for h in hold):
        st.warning("USD/MYR rate N/A right now: US positions can't be converted, so they are left out.")
        hold = [h for h in hold if h[2].get("ccy") != "USD"]
    tickers = [h[0] for h in hold]
    val = pd.Series({h[0]: h[1] for h in hold}, dtype=float)
    w = val / val.sum()
    with st.spinner("Loading one year of prices…"):
        px = R.closes(tickers + ["^KLSE", "^GSPC"], "2y")
    rets = pd.DataFrame({t: np.expm1(R.returns(px[t].dropna())) for t in px.columns}).tail(252)
    stock_r = rets[[t for t in tickers if t in rets]].fillna(0)
    w = w.reindex(stock_r.columns).fillna(0)
    w = w / w.sum() if w.sum() else w
    port = (stock_r * w).sum(axis=1)
    cov = stock_r.cov() * R.TRADING_DAYS
    pvar = float(w @ cov @ w)
    pvol = np.sqrt(pvar) * 100 if pvar > 0 else None
    b_k = R.beta(port, rets["^KLSE"]) if "^KLSE" in rets else None
    b_s = R.beta(port, rets["^GSPC"]) if "^GSPC" in rets else None
    var95, es95 = R.hist_var(np.log1p(port))
    eq = (1 + port).cumprod()
    mdd, dd_start, dd_end = R.max_drawdown(eq)
    total = float(val.sum())
    top = w.idxmax()

    m = st.columns(6) if not C.mobile else st.columns(2)
    cards = [("Portfolio value", f"RM{total:,.0f}" if src != "Watchlist (equal weight)" else f"{len(tickers)} stocks"),
             ("Volatility (annual)", "N/A" if pvol is None else f"{pvol:.1f}%"),
             ("Beta vs KLCI / S&P", f"{'N/A' if b_k is None else f'{b_k:.2f}'} / {'N/A' if b_s is None else f'{b_s:.2f}'}"),
             ("1-day VaR (95%)", "N/A" if var95 is None else f"{var95:.2f}%" + (f" · RM{var95 / 100 * total:,.0f}"
                                                                              if src != "Watchlist (equal weight)" else "")),
             ("Max drawdown (1y)", "N/A" if mdd is None else f"{mdd:.1f}%"),
             ("Largest position", f"{display_name(top)} {w[top] * 100:.0f}%")]
    for i, (a, b) in enumerate(cards):
        m[i % len(m)].metric(a, b)
    st.caption("Calculated as if today's weights had been held for the last year. VaR 95% = on 19 days out of 20 the "
               "daily loss was smaller than this (expected loss on the worst 5% of days: "
               f"{'N/A' if es95 is None else f'{es95:.2f}%'}). Past behaviour, not a forecast."
               + (f" US positions converted at USD/MYR {fx:.4f}." if fx else ""))

    # ---------------------------------------------------------------- per holding
    rows = []
    mrc = (cov @ w) * w / pvar * 100 if pvar > 0 else pd.Series(0, index=w.index)
    for t in stock_r.columns:
        r_ = rets[t].dropna()
        bench = R.benchmark_of(t)
        b = R.beta(r_, rets[bench]) if bench in rets else None
        d = load_one(t, "1d", "6mo")
        atrp = None
        if d is not None and len(d) > 20:
            tr = pd.concat([d["High"] - d["Low"], (d["High"] - d["Close"].shift()).abs(),
                            (d["Low"] - d["Close"].shift()).abs()], axis=1).max(axis=1)
            atrp = float(tr.tail(14).mean() / d["Close"].iloc[-1] * 100)
        ex = next((h[2] for h in hold if h[0] == t), {})
        sl, last = ex.get("sl"), ex.get("last")
        stop_d = (last - sl) / last * 100 if sl and last else None
        risk_rm = (last - sl) * ex.get("qty", 0) * (1 if ex.get("ccy") == "MYR" else (fx or 0)) if sl and last else None
        rows.append({"Name": display_name(t), "Weight %": w[t] * 100, "Volatility %": R.ann_vol(np.log1p(r_)),
                     "Beta (own index)": b, "ATR %": atrp, "Share of risk %": float(mrc.get(t, np.nan)),
                     "To stop %": stop_d, "Loss at stop (RM)": risk_rm})
    tbl = pd.DataFrame(rows).sort_values("Weight %", ascending=False)
    for col in ("Beta (own index)", "Volatility %", "ATR %", "To stop %", "Loss at stop (RM)"):
        tbl[col] = pd.to_numeric(tbl[col], errors="coerce")
    if not src != "Watchlist (equal weight)":
        tbl = tbl.drop(columns=["To stop %", "Loss at stop (RM)"])
    with ui.panel("Holdings risk", "share of risk = how much each holding adds to total portfolio swings"):
        cc = st.column_config
        st.dataframe(tbl, hide_index=True, width="stretch", height="content",
                     column_order=["Name", "Weight %", "Share of risk %", "Volatility %"] if C.mobile else None,
                     column_config={"Weight %": cc.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
                                    "Volatility %": cc.NumberColumn(format="%.1f%%"),
                                    "Beta (own index)": cc.NumberColumn(format="%.2f"),
                                    "ATR %": cc.NumberColumn(format="%.2f%%"),
                                    "Share of risk %": cc.NumberColumn(format="%.1f%%"),
                                    "To stop %": cc.NumberColumn(format="%.1f%%"),
                                    "Loss at stop (RM)": cc.NumberColumn(format="%,.0f")})
        no_sl = [r["Name"] for r in rows if r["To stop %"] is None] if src != "Watchlist (equal weight)" else []
        if no_sl:
            st.markdown(f"<div class='note'>⚠️ No stop loss set on: {', '.join(no_sl)}. Their downside isn't capped "
                        "(set one in 💼 Paper Trade → Positions).</div>", unsafe_allow_html=True)
        heavy = tbl[tbl["Share of risk %"] > 40]
        if len(tbl) > 1 and len(heavy):
            st.markdown(f"<div class='note'>⚠️ {', '.join(heavy['Name'])} drives over 40% of the portfolio's risk.</div>",
                        unsafe_allow_html=True)

    # ---------------------------------------------------------------- concentration & stress
    a, b = st.columns(2) if not C.mobile else (st.container(), st.container())
    with a, ui.panel("Concentration", "by sector and currency"):
        sec = {}
        for t in stock_r.columns:
            s_ = (C.sector_map or {}).get(t) or (F.load(t).get("info") or {}).get("sector") or "Other"
            sec[s_] = sec.get(s_, 0) + w[t] * 100
        sec = pd.Series(sec).sort_values()
        fig = go.Figure(go.Bar(x=sec.values, y=sec.index, orientation="h", marker_color=P["accent"],
                               text=[f"{v:.0f}%" for v in sec.values], textposition="outside", cliponaxis=False))
        fig.update_layout(height=max(200, 30 * len(sec) + 50), margin=dict(l=4, r=30, t=6, b=4),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=P["text"], size=11))
        fig.update_xaxes(gridcolor=P["grid"], ticksuffix="%")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key="rk_sec")
        myr = sum(w[t] for t in stock_r.columns if t.endswith(".KL")) * 100
        st.markdown(f"<div class='muted' style='font-size:12.5px'>Currency: MYR {myr:.0f}% · USD {100 - myr:.0f}%"
                    + (" · ⚠️ over 40% in one sector" if len(sec) and sec.max() > 40 and len(tickers) > 2 else "")
                    + "</div>", unsafe_allow_html=True)
    with b, ui.panel("Stress tests", "rough estimates from beta and history"):
        tests = []
        for idx, lab in (("^KLSE", "KLCI"), ("^GSPC", "S&P 500")):
            share = sum(w[t] for t in stock_r.columns if R.benchmark_of(t) == idx)
            if share == 0:
                continue
            bsum = sum(w[t] * (R.beta(rets[t].dropna(), rets[idx]) or 1.0) for t in stock_r.columns
                       if R.benchmark_of(t) == idx)
            tests.append((f"{lab} falls 10%", -10 * bsum, total * -10 * bsum / 100))
        if len(port) > 20:
            worst = port.idxmin()
            tests.append((f"Worst day in the last year ({worst:%d %b %Y})", float(port.min() * 100),
                          total * float(port.min())))
            w5 = (1 + port).rolling(5).apply(np.prod, raw=True) - 1
            tests.append(("Worst 5-day stretch", float(w5.min() * 100), total * float(w5.min())))
        st.markdown("<table class='kv'>" + "".join(
            f"<tr><td>{lab}</td><td class='mono' style='color:{P['down']}'>{pct:+.1f}%"
            + (f" · RM{amt:,.0f}" if src != "Watchlist (equal weight)" else "") + "</td></tr>" for lab, pct, amt in tests)
                    + "</table>", unsafe_allow_html=True)
        st.caption("Index falls use each holding's beta to its own index (US holdings to the S&P 500); "
                   "real crashes can be worse because correlations rise when markets panic.")
    if len(stock_r.columns) > 1:
        with ui.panel("Correlation", "daily returns, last year"):
            cr = stock_r.corr()
            names = [display_name(t) for t in cr.columns]
            fig = go.Figure(go.Heatmap(z=cr.values, x=names, y=names, zmin=-1, zmax=1, showscale=False,
                                       colorscale=[[0, P["down"]], [0.5, P["card2"]], [1, P["up"]]],
                                       text=np.round(cr.values, 2), texttemplate="%{text}"))
            fig.update_layout(height=max(260, 34 * len(names) + 60),
                              margin=dict(l=4, r=4, t=6, b=4), paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", font=dict(color=P["text"], size=11))
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key="rk_corr")
            avg = cr.values[np.triu_indices(len(cr), 1)].mean()
            st.caption(f"Average correlation {avg:.2f}. Below ~0.3 = well spread; above ~0.6 = holdings tend to fall "
                       "together.")
