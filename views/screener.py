"""Screener page: scan the chosen universe and rank by signal score."""
from core import *  # noqa: F401,F403


def render(C):
    universe, interval, period, unit = C.universe, C.interval, C.period, C.unit
    shariah_set, MOBILE = C.shariah_set, C.mobile
    rows = []
    if not universe:
        st.info("Enter at least one ticker in the sidebar.")
    else:
        with st.spinner(f"Scanning {len(universe)} stocks..."):
            data = get_many(universe, interval, period)
        for t, df in data.items():
            try:
                d, sup, res, sc, label, _ = analyse(df)
                p = trade_plan(d, sup, res, label, unit)
                last, prev = d.iloc[-1], d.iloc[-2]
                price = float(last["Close"])
                rows.append({
                    "Name": display_name(t), "Ticker": t,
                    **({"Shariah": "✅" if t in shariah_set else "–"} if shariah_set and t.endswith(".KL") else {}),
                    "Price": price,
                    "Change %": (price / float(prev["Close"]) - 1) * 100,
                    "Signal": f"{LABEL_DOT[label]} {label}", "_label": label, "Score": sc,
                    "RSI": float(last["RSI"]),
                    "RVOL": float(last["RVOL"]) if pd.notna(last["RVOL"]) else None,
                    "ADX": float(last["ADX"]) if pd.notna(last["ADX"]) else None,
                    "Trend (ADX)": adx_regime(last["ADX"], last["PDI"], last["MDI"]),
                    "Support": sup[0][0], "Resistance": res[0][0],
                    "To support %": (price / sup[0][0] - 1) * 100,
                    "Buy zone": f"{p['buy_lo']:.3f} – {p['buy_hi']:.3f}",
                    "Stop": p["stop"], "Target": p["t1"], "R:R": p["rr1"],
                })
            except Exception:
                continue
        missing = [display_name(t) for t in universe if t not in data]
        if missing:
            st.caption("No data for: " + ", ".join(missing))

    if rows:
        tbl = pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Stocks scanned", len(tbl))
        c2.metric("Buy signals", int(tbl["_label"].isin(["BUY", "STRONG BUY"]).sum()))
        c3.metric("Sell signals", int(tbl["_label"].isin(["SELL", "STRONG SELL"]).sum()))
        c4.metric("Near support (<3%)", int((tbl["To support %"] < 3).sum()))
        st.write("")
        flt = st.segmented_control("Filter", ["All", "Buy only", "Sell only", "Near support (<3%)", "R:R ≥ 2",
                                              "Unusual volume (RVOL ≥ 2)", "Strong trends (ADX ≥ 25)",
                                              "Sideways (ADX < 20)"],
                                   default="All", key="scr_filter") or "All"
        view = tbl
        if flt == "Buy only":
            view = tbl[tbl["_label"].isin(["BUY", "STRONG BUY"])]
        elif flt == "Sell only":
            view = tbl[tbl["_label"].isin(["SELL", "STRONG SELL"])]
        elif flt.startswith("Near"):
            view = tbl[tbl["To support %"] < 3]
        elif flt.startswith("R:R"):
            view = tbl[tbl["R:R"] >= 2]
        elif flt.startswith("Unusual"):
            view = tbl[tbl["RVOL"].fillna(0) >= 2].sort_values("RVOL", ascending=False)
        elif flt.startswith("Strong trends"):
            view = tbl[tbl["ADX"].fillna(0) >= ADX_STRONG].sort_values("ADX", ascending=False)
        elif flt.startswith("Sideways"):
            view = tbl[tbl["ADX"].fillna(0) < ADX_WEAK]
        cc = st.column_config
        st.dataframe(
            view.drop(columns=["_label"]), hide_index=True, width="stretch",
            column_order=(["Name", "Price", "Change %", "Signal", "Score"] if MOBILE else None),
            height=min(36 * (len(view) + 1) + 4, 900),
            column_config={
                "Name": cc.TextColumn(width="small"),
                "Price": cc.NumberColumn(format="%.3f"),
                "Change %": cc.NumberColumn(format="%+.2f%%"),
                "Score": cc.ProgressColumn(min_value=-100, max_value=100, format="%d", width="small"),
                "RSI": cc.NumberColumn(format="%.0f", width="small"),
                "ADX": cc.NumberColumn("ADX", format="%.0f", width="small",
                                       help="Trend strength: <20 sideways, 20–25 weak, 25+ strong"),
                "RVOL": cc.NumberColumn("RVOL", format="%.1f×", width="small",
                                        help="Relative volume: latest volume ÷ 20-bar average. 2×+ = spike"),
                "Support": cc.NumberColumn(format="%.3f"),
                "Resistance": cc.NumberColumn(format="%.3f"),
                "To support %": cc.NumberColumn(format="%.1f%%"),
                "Stop": cc.NumberColumn(format="%.3f"),
                "Target": cc.NumberColumn(format="%.3f"),
                "R:R": cc.NumberColumn(format="%.1f", width="small"),
            })
        st.caption("Score runs from −100 to +100 (trend, RSI, MACD, distance to support/resistance, breakouts, volume). "
                   "R:R = (target − entry) ÷ (entry − stop). Open a stock in **Analysis & Forecast** for the full picture.")
