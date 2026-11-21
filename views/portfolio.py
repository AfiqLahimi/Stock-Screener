"""My Portfolio: the shares you actually own, entered by hand (no broker connection), priced live.
Lots are saved with your preferences (server + browser). Dividends received are estimated from each
stock's dividend history since the buy date."""
from pxfmt import fp, pxf  # noqa: F401
from functools import partial

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import fundamentals as F
import ui
from core import P, display_name, load_one, normalise, scan_universe


def _fx():
    d = load_one("MYR=X", "1d", "1mo")
    return float(d["Close"].iloc[-1]) if d is not None and len(d) else None


def _lots():
    return list(st.session_state.get("pf_lots", []))


def _save(lots):
    st.session_state["pf_lots"] = lots


def _add_one():
    t = st.session_state.get("pf_sym") or ""
    typed = (st.session_state.get("pf_typed") or "").strip()
    if typed:
        t = normalise(typed)
    q, c = st.session_state.get("pf_qty") or 0, st.session_state.get("pf_cost") or 0
    if t and q > 0 and c > 0:
        lots = _lots()
        lots.append({"ticker": t, "qty": float(q), "cost": float(c),
                     "date": str(st.session_state.get("pf_date") or pd.Timestamp.now().date()),
                     "note": st.session_state.get("pf_note") or ""})
        _save(lots)
        st.session_state["pf_flash"] = f"Added {q:,.0f} {display_name(t)} @ {fp(c, True)}."
        st.session_state["pf_typed"] = ""
        st.session_state["pf_note"] = ""
    else:
        st.session_state["pf_flash"] = "⚠️ Pick a stock and enter a quantity and price above zero."


def _paste():
    added, bad = 0, []
    lots = _lots()
    for line in (st.session_state.get("pf_paste") or "").splitlines():
        parts = [p.strip() for p in line.replace("\t", ",").split(",") if p.strip()]
        if not parts:
            continue
        try:
            t, q, c = normalise(parts[0]), float(parts[1]), float(parts[2])
            d = str(pd.Timestamp(parts[3]).date()) if len(parts) > 3 else str(pd.Timestamp.now().date())
            if not t or q <= 0 or c <= 0:
                raise ValueError
            lots.append({"ticker": t, "qty": q, "cost": c, "date": d, "note": ""})
            added += 1
        except Exception:
            bad.append(line)
    _save(lots)
    st.session_state["pf_paste"] = ""
    st.session_state["pf_flash"] = f"Added {added} lot(s)." + (f" Couldn't read: {'; '.join(bad[:3])}" if bad else "")


def dividends_received(t, qty, since):
    dv = F.load(t).get("dividends")
    if dv is None or not len(dv):
        return 0.0
    since = pd.Timestamp(since)
    return float(dv[(dv.index >= since) & (dv.index <= pd.Timestamp.now())].sum() * qty)


def holdings_table(lots):
    """Aggregate lots per stock and price them. Returns (table, fx)."""
    if not lots:
        return pd.DataFrame(), None
    fx = _fx()
    df = pd.DataFrame(lots)
    df["qty"], df["cost"] = df["qty"].astype(float), df["cost"].astype(float)
    tickers = tuple(dict.fromkeys(df["ticker"]))
    sc = scan_universe(tickers, "1d", "2y", "trading days")
    sc = sc.set_index("Ticker") if not sc.empty else pd.DataFrame()
    rows = []
    for t, g in df.groupby("ticker", sort=False):
        q = g["qty"].sum()
        avg = (g["qty"] * g["cost"]).sum() / q if q else np.nan
        r = sc.loc[t] if t in sc.index else None
        last = float(r["Price"]) if r is not None else np.nan
        if r is None:
            d = load_one(t, "1d", "1mo")
            last = float(d["Close"].iloc[-1]) if d is not None and len(d) else np.nan
        ccy = "MYR" if t.endswith(".KL") else "USD"
        to_rm = 1.0 if ccy == "MYR" else (fx or np.nan)
        divs = sum(dividends_received(t, lot.qty, lot.date) for lot in g.itertuples())
        val = q * last
        rows.append({"Ticker": t, "Name": display_name(t), "Ccy": ccy, "Qty": q, "Avg cost": avg, "Last": last,
                     "Day %": float(r["Change %"]) if r is not None else np.nan,
                     "Value": val, "Value (RM)": val * to_rm, "P/L": val - q * avg,
                     "P/L (RM)": (val - q * avg) * to_rm, "P/L %": (last / avg - 1) * 100 if avg else np.nan,
                     "Day P/L (RM)": (val - val / (1 + float(r["Change %"]) / 100)) * to_rm if r is not None else np.nan,
                     "Dividends (RM)": divs * to_rm, "Signal": r["Signal"] if r is not None else "N/A",
                     "Score": int(r["Score"]) if r is not None else None})
    tbl = pd.DataFrame(rows)
    tot = tbl["Value (RM)"].sum()
    tbl["Weight %"] = tbl["Value (RM)"] / tot * 100 if tot else np.nan
    return tbl.sort_values("Value (RM)", ascending=False).reset_index(drop=True), fx


def _open_row(key, tickers):
    sel = st.session_state.get(key)
    try:
        rows = sel["selection"]["rows"]
    except Exception:
        rows = []
    if rows and rows[0] < len(tickers):
        ui.goto("analysis", an_typed=tickers[rows[0]])


def render(C):
    if st.session_state.get("pf_flash"):
        msg = st.session_state.pop("pf_flash")
        (st.warning if msg.startswith("⚠️") else st.success)(msg)
    lots = _lots()
    tbl, fx = holdings_table(lots)
    if tbl.empty:
        st.info("Add the shares you own below (or paste a list). Nothing is connected to a broker: you enter the "
                "quantity and price you paid, and the app prices them live.")
    else:
        cost_rm = (tbl["Value (RM)"] - tbl["P/L (RM)"]).sum()
        m = st.columns(5) if not C.mobile else st.columns(2)
        cards = [("Portfolio value", f"RM{tbl['Value (RM)'].sum():,.0f}"),
                 ("Unrealised P/L", f"RM{tbl['P/L (RM)'].sum():,.0f}",
                  f"{tbl['P/L (RM)'].sum() / cost_rm * 100:+.2f}%" if cost_rm else None),
                 ("Today", f"RM{tbl['Day P/L (RM)'].sum():,.0f}"),
                 ("Dividends received", f"RM{tbl['Dividends (RM)'].sum():,.0f}"),
                 ("Holdings", f"{len(tbl)}")]
        for i, c in enumerate(cards):
            m[i % len(m)].metric(c[0], c[1], c[2] if len(c) > 2 else None)
        st.caption("US holdings converted at USD/MYR " + ("N/A" if fx is None else f"{fx:.4f}") +
                   ". Dividends are estimated from ex-dividend dates after each buy date (before tax). Bursa "
                   "prices are ~15 min delayed.")
        with ui.panel("Holdings", "click a row to open the stock in Analysis"):
            cc = st.column_config
            key = "pf_tbl"
            st.dataframe(tbl, hide_index=True, width="stretch", height="content", key=key,
                         on_select=partial(_open_row, key, list(tbl["Ticker"])), selection_mode="single-row",
                         column_order=(["Name", "Value (RM)", "P/L %", "Day %"] if C.mobile else
                                       ["Name", "Ticker", "Qty", "Avg cost", "Last", "Day %", "Value (RM)", "P/L (RM)",
                                        "P/L %", "Weight %", "Dividends (RM)", "Signal"]),
                         column_config={"Qty": cc.NumberColumn(format="%,.0f"), "Avg cost": cc.NumberColumn(format="%.3f"),
                                        "Last": cc.NumberColumn(format="%.3f"), "Day %": cc.NumberColumn(format="%+.2f%%"),
                                        "Value (RM)": cc.NumberColumn(format="%,.0f"),
                                        "P/L (RM)": cc.NumberColumn(format="%+,.0f"),
                                        "P/L %": cc.NumberColumn(format="%+.1f%%"),
                                        "Weight %": cc.ProgressColumn(format="%.0f%%", min_value=0, max_value=100),
                                        "Dividends (RM)": cc.NumberColumn(format="%,.0f")})
            st.button("🛡️ Analyse the risk of this portfolio", key="pf_risk", on_click=ui.goto, args=("risk",),
                      kwargs={"rk_src": "My portfolio"})
        a, b = st.columns(2) if not C.mobile else (st.container(), st.container())
        with a, ui.panel("Allocation", "by market value"):
            pal = [P["accent"], P["up"], P["warn"], P["ema200"], P["ema50"], P["down"], P["ema20"], P["muted"]]
            fig = go.Figure(go.Pie(labels=tbl["Name"], values=tbl["Value (RM)"], hole=0.55, sort=False,
                                   textinfo="label+percent", textposition="outside",
                                   marker=dict(colors=[pal[i % len(pal)] for i in range(len(tbl))],
                                               line=dict(color=P["card"], width=2))))
            fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
                              paper_bgcolor="rgba(0,0,0,0)", font=dict(size=11))
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key="pf_pie")
        with b, ui.panel("Profit / loss by holding", "RM"):
            t2 = tbl.sort_values("P/L (RM)")
            fig = go.Figure(go.Bar(x=t2["P/L (RM)"], y=t2["Name"], orientation="h",
                                   marker_color=[P["up"] if v >= 0 else P["down"] for v in t2["P/L (RM)"]],
                                   text=[f"{v:+,.0f}" for v in t2["P/L (RM)"]], textposition="inside",
                                   insidetextanchor="middle", textfont=dict(color=P["on_fill"])))
            fig.update_layout(height=max(220, 34 * len(t2) + 60), margin=dict(l=4, r=40, t=6, b=4),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(size=11))
            fig.update_xaxes(gridcolor=P["grid"])
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key="pf_pl")

    with ui.panel("Add a purchase", "one row per buy (lot); sells: edit or delete the lot below"):
        c1, c2, c3, c4, c5 = st.columns([2.2, 1, 1, 1, 0.9]) if not C.mobile else (st.container(),) * 5
        c1.selectbox("Stock", C.all_symbols(), index=None, key="pf_sym", placeholder="Search name or code…",
                     format_func=lambda t: f"{display_name(t)} · {t}")
        c2.number_input("Shares", min_value=0.0, step=100.0, key="pf_qty", help="Bursa: 1 lot = 100 shares")
        c3.number_input("Price paid", min_value=0.0, step=0.01, format="%.3f", key="pf_cost")
        c4.date_input("Bought on", key="pf_date", value=pd.Timestamp.now().date(), max_value=pd.Timestamp.now().date())
        c5.button("➕ Add", key="pf_add", on_click=_add_one, width="stretch")
        d1, d2 = st.columns([1, 2]) if not C.mobile else (st.container(), st.container())
        d1.text_input("…or type a code not in the list", key="pf_typed", placeholder="e.g. 0331, PLTR")
        d2.text_input("Note (optional)", key="pf_note", placeholder="e.g. bought for dividends")
        with st.expander("📋 Paste many at once"):
            st.text_area("One per line: code, shares, price[, date]", key="pf_paste", height=110,
                         placeholder="MAYBANK, 1000, 9.85, 2025-03-14\n5398, 2000, 4.20\nNVDA, 10, 120.5, 2024-11-02")
            st.button("Add these", key="pf_paste_b", on_click=_paste)

    if lots:
        with ui.panel("Edit lots", "change any cell or delete rows, then press Save"):
            ed = pd.DataFrame(lots)
            ed["date"] = pd.to_datetime(ed["date"]).dt.date
            ed.insert(1, "name", [display_name(t) for t in ed["ticker"]])
            new = st.data_editor(ed, key="pf_editor", num_rows="dynamic", hide_index=True, width="stretch",
                                 disabled=["name"],
                                 column_config={"ticker": st.column_config.TextColumn("Code"),
                                                "name": st.column_config.TextColumn("Name"),
                                                "qty": st.column_config.NumberColumn("Shares", min_value=0, format="%,.0f"),
                                                "cost": st.column_config.NumberColumn("Price paid", min_value=0, format="%.3f"),
                                                "date": st.column_config.DateColumn("Bought on"),
                                                "note": st.column_config.TextColumn("Note")})
            if st.button("💾 Save changes", key="pf_save"):
                out = []
                for r in new.to_dict("records"):
                    t = normalise(str(r.get("ticker") or ""))
                    try:
                        q, c = float(r.get("qty") or 0), float(r.get("cost") or 0)
                    except (TypeError, ValueError):
                        continue
                    if t and q > 0 and c > 0:
                        out.append({"ticker": t, "qty": q, "cost": c,
                                    "date": str(pd.Timestamp(r.get("date") or pd.Timestamp.now()).date()),
                                    "note": r.get("note") or ""})
                _save(out)
                st.session_state["pf_flash"] = f"Saved {len(out)} lot(s)."
                st.rerun()
            st.caption("Your portfolio is saved on the app server and in this browser (same as the watchlist).")
