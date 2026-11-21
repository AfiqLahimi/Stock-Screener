"""Picks tab: ranked buy / sell ideas for short-term trading and long-term investing."""
from pxfmt import fp, pxf  # noqa: F401
import numpy as np
import pandas as pd
import streamlit as st


# ----------------------------------------------------------------------------
# Short term (days to weeks): technical setup quality
# ----------------------------------------------------------------------------
def short_term(t, df, analyse, trade_plan, detect_patterns):
    d, sup, res, sc, label, why = analyse(df)
    plan = trade_plan(d, sup, res, label, "trading days")
    last = d.iloc[-1]
    price, atr, rsi = float(last["Close"]), float(last["ATR"]), float(last["RSI"])
    pats = [p for p in detect_patterns(d) if p[2] <= 1]  # latest or previous candle
    bull_pat = next((n for n, k, _ in pats if k == "bull"), None)
    bear_pat = next((n for n, k, _ in pats if k == "bear"), None)
    breakout = any("Breakout" in txt for _, txt in why)
    vol_surge = pd.notna(last["VolAvg"]) and last["Volume"] > 1.5 * last["VolAvg"]
    rr = plan["rr1"] if not np.isnan(plan["rr1"]) else 0.0
    in_zone = price <= plan["buy_hi"] * 1.02

    # setup quality: signal score + bonuses for things that make a trade actionable now
    q = sc
    q += 10 if in_zone else 0
    q += 10 if breakout else 0
    q += 8 if bull_pat else 0
    q -= 8 if bear_pat else 0
    q += 5 if (vol_surge and price >= last["Open"]) else 0
    q += min(rr, 3) * 5
    q -= 15 if rsi > 75 else 0

    buy_ok = label in ("BUY", "STRONG BUY") and rr >= 1.5 and rsi < 72 and (in_zone or breakout)
    near_res = (res[0][0] - price) <= atr and res[0][1] > 0
    sell_ok = label in ("SELL", "STRONG SELL") or (rsi > 70 and near_res) or (bear_pat is not None and sc < 0)

    reasons = [txt for pts, txt in sorted(why, key=lambda w: -w[0]) if pts > 0][:2]
    if bull_pat:
        reasons.insert(0, f"Bullish candle: {bull_pat}")
    sell_reasons = [txt for pts, txt in sorted(why, key=lambda w: w[0]) if pts < 0][:2]
    if bear_pat:
        sell_reasons.insert(0, f"Bearish candle: {bear_pat}")
    if rsi > 70 and near_res:
        sell_reasons.insert(0, f"Overbought (RSI {rsi:.0f}) right under resistance {fp(res[0][0])}")

    return {
        "Ticker": t, "Price": price, "Signal": label, "Score": sc, "Setup": round(q, 1), "RSI": rsi,
        "Buy zone": f"{fp(plan['buy_lo'])} – {fp(plan['buy_hi'])}", "Stop": plan["stop"],
        "Target": plan["t1"], "R:R": rr, "In zone": in_zone, "Breakout": breakout,
        "Pattern": bull_pat or bear_pat or "", "_buy": buy_ok, "_sell": sell_ok,
        "_why": reasons[:3], "_why_sell": sell_reasons[:3], "_action": plan["action"],
    }


# ----------------------------------------------------------------------------
# Long term (6 months+): trend, momentum, risk, valuation
# ----------------------------------------------------------------------------
def _ret(c, n):
    return (float(c.iloc[-1]) / float(c.iloc[-1 - n]) - 1) * 100 if len(c) > n else None


def long_term(t, df, add_indicators, fund=None):
    d = add_indicators(df)
    c = d["Close"].astype(float)
    last = d.iloc[-1]
    price = float(c.iloc[-1])
    s, why, risks = 0, [], []

    def add(p, txt):
        nonlocal s
        s += p
        (why if p > 0 else risks).append(txt)

    ema200 = d["EMA200"]
    if price > last["EMA200"]:
        add(20, "Above its 200-day average (long-term uptrend)")
    else:
        add(-20, "Below its 200-day average (long-term downtrend)")
    if last["EMA50"] > last["EMA200"]:
        add(10, "50-day above 200-day average")
    else:
        add(-10, "50-day below 200-day average")
    if len(ema200) > 60:
        slope = (float(ema200.iloc[-1]) / float(ema200.iloc[-61]) - 1) * 100
        if slope > 2:
            add(10, "200-day average rising steadily")
        elif slope < -2:
            add(-10, "200-day average falling")

    r6, r12 = _ret(c, 126), _ret(c, 250)
    for r, lab, big in ((r6, "6-month", 15), (r12, "1-year", 25)):
        if r is None:
            continue
        if r > big:
            add(12, f"Strong {lab} gain ({r:+.0f}%)")
        elif r > 0:
            add(5, f"Positive {lab} return ({r:+.0f}%)")
        elif r < -big:
            add(-12, f"Big {lab} loss ({r:+.0f}%)")
        else:
            add(-5, f"Negative {lab} return ({r:+.0f}%)")

    vol = float(np.log(c).diff().tail(250).std() * np.sqrt(250) * 100)
    if vol < 25:
        add(6, f"Steady price (volatility {vol:.0f}%/yr)")
    elif vol > 55:
        add(-10, f"Very volatile ({vol:.0f}%/yr)")

    hi52 = float(d["High"].tail(250).max())
    dd = (price / hi52 - 1) * 100
    if dd > -10:
        add(5, f"Near its 52-week high ({dd:.0f}%)")
    elif dd < -40:
        add(-10, f"{-dd:.0f}% below its 52-week high")

    pe = dy = None
    if fund is not None:
        pe, dy = fund.get("pe"), fund.get("div_yield")
        if pe is not None and not pd.isna(pe):
            if 0 < pe < 15:
                add(10, f"Reasonable valuation (P/E {pe:.1f})")
            elif pe > 35:
                add(-8, f"Expensive (P/E {pe:.1f})")
        if dy is not None and not pd.isna(dy) and dy >= 4:
            add(6, f"Good dividend yield ({dy:.1f}%)")

    s = int(max(-100, min(100, s)))
    if s >= 50:
        label = "BUY / ACCUMULATE"
    elif s >= 20:
        label = "GOOD · BUY ON DIPS"
    elif s > -20:
        label = "HOLD / WATCH"
    elif s > -50:
        label = "WEAK · AVOID"
    else:
        label = "SELL / AVOID"
    return {
        "Ticker": t, "Price": price, "Long-term": label, "Score": s,
        "6m %": r6, "1y %": r12, "From 52w high %": dd, "Volatility %": vol,
        "P/E": pe, "Div yield %": dy, "Above 200d": price > last["EMA200"],
        "_why": why[:3], "_risks": risks[:3],
    }


# ----------------------------------------------------------------------------
# Render
# ----------------------------------------------------------------------------
def _card(r, P, pill, color, title_txt, lines, body):
    li = "".join(f"<div class='reason'><span>•</span><span>{x}</span></div>" for x in lines)
    return f"""<div class="card" style="border-left:3px solid {color}">
<div style="display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap">
  <div><b style="font-size:16px">{r['Name']}</b> <span class="muted">{r['Ticker']}</span></div>
  {pill(title_txt, color)}</div>
<div class="muted" style="margin:6px 0 4px">{body}</div>{li}</div>"""


def render(universe, display_name, P, pill, get_many, analyse, trade_plan, add_indicators, detect_patterns,
           fundamentals=None):
    st.markdown("### 🎯 Picks")
    mode = st.segmented_control(
        "Investment style", ["⚡ Short term (days to weeks)", "🌱 Long term (6 months+)"],
        default="⚡ Short term (days to weeks)", key="picks_mode") or "⚡ Short term (days to weeks)"
    short = mode.startswith("⚡")
    if short:
        st.caption("Stocks with a good **trading setup right now**: uptrend signal, price near support or breaking "
                   "out, decent risk:reward (≥ 1.5). Plans use daily candles. Typical holding: 1 to 4 weeks. "
                   "**Always set the stop loss.**")
    else:
        st.caption("Stocks with **healthy long-term trends**: above the 200-day average, positive 6 to 12 month "
                   "momentum, manageable volatility" + (", sensible P/E and dividends" if fundamentals is not None else "")
                   + ". Typical holding: 6 months or more. Best bought on dips, in parts.")
    if not universe:
        st.info("Pick a market or enter tickers in the sidebar.")
        return

    with st.spinner(f"Ranking {len(universe)} stocks…"):
        data = get_many(tuple(universe), "1d", "2y")
        rows = []
        for t, df in data.items():
            if len(df) < (210 if not short else 80):
                continue
            try:
                if short:
                    r = short_term(t, df, analyse, trade_plan, detect_patterns)
                else:
                    fund = None
                    if fundamentals is not None and t in fundamentals.index:
                        fund = fundamentals.loc[t].to_dict()
                    r = long_term(t, df, add_indicators, fund)
                r["Name"] = display_name(t)
                rows.append(r)
            except Exception:
                continue
    if not rows:
        st.warning("Not enough price history to rank these stocks.")
        return
    df = pd.DataFrame(rows)
    top_n = st.select_slider("Show top", [5, 10, 15, 20], value=10, key="picks_n")

    left, right = st.columns(2, gap="medium")
    if short:
        buys = df[df["_buy"]].sort_values("Setup", ascending=False).head(top_n)
        sells = df[df["_sell"]].sort_values("Setup", ascending=True).head(top_n)
        with left:
            st.markdown(f"#### 🟢 Good to buy now · {len(df[df['_buy']])} found")
            if buys.empty:
                st.info("No strong buy setups right now. That's normal: waiting is also a position.")
            for _, r in buys.iterrows():
                body = (f"Price <b>{fp(r['Price'])}</b> · buy zone <b>{r['Buy zone']}</b> · stop "
                        f"<b style='color:{P['warn']}'>{fp(r['Stop'])}</b> · target "
                        f"<b style='color:{P['up']}'>{fp(r['Target'])}</b> · R:R <b>1:{r['R:R']:.1f}</b>")
                st.markdown(_card(r, P, pill, P["up"], f"{r['Signal']} · {r['Score']:+d}", r["_why"], body),
                            unsafe_allow_html=True)
        with right:
            st.markdown(f"#### 🔴 Consider selling / taking profit · {len(df[df['_sell']])} found")
            if sells.empty:
                st.info("No strong sell signals right now.")
            for _, r in sells.iterrows():
                body = (f"Price <b>{fp(r['Price'])}</b> · RSI <b>{r['RSI']:.0f}</b> · "
                        f"cut below <b style='color:{P['warn']}'>{fp(r['Stop'])}</b>")
                st.markdown(_card(r, P, pill, P["down"], f"{r['Signal']} · {r['Score']:+d}", r["_why_sell"], body),
                            unsafe_allow_html=True)
        table = df.sort_values("Setup", ascending=False)[
            ["Name", "Ticker", "Signal", "Score", "Setup", "Price", "RSI", "Buy zone", "Stop", "Target", "R:R",
             "Pattern"]]
        cfg = {"Price": st.column_config.NumberColumn(format="%.3f"),
               "Setup": st.column_config.ProgressColumn(min_value=-120, max_value=140, format="%.0f"),
               "RSI": st.column_config.NumberColumn(format="%.0f"),
               "Stop": st.column_config.NumberColumn(format="%.3f"),
               "Target": st.column_config.NumberColumn(format="%.3f"),
               "R:R": st.column_config.NumberColumn(format="%.1f")}
    else:
        buys = df[df["Score"] >= 20].sort_values("Score", ascending=False).head(top_n)
        sells = df[df["Score"] <= -20].sort_values("Score", ascending=True).head(top_n)

        def lt_body(r):
            bits = [f"Price <b>{fp(r['Price'])}</b>"]
            if r["1y %"] is not None and not pd.isna(r["1y %"]):
                col = P["up"] if r["1y %"] >= 0 else P["down"]
                bits.append(f"1y <b style='color:{col}'>{r['1y %']:+.0f}%</b>")
            if r["P/E"] is not None and not pd.isna(r["P/E"]):
                bits.append(f"P/E <b>{r['P/E']:.1f}</b>")
            if r["Div yield %"] is not None and not pd.isna(r["Div yield %"]):
                bits.append(f"yield <b>{r['Div yield %']:.1f}%</b>")
            return " · ".join(bits)
        with left:
            st.markdown(f"#### 🟢 Strong long-term picks · {int((df['Score'] >= 20).sum())} found")
            if buys.empty:
                st.info("No stocks pass the long-term test right now.")
            for _, r in buys.iterrows():
                st.markdown(_card(r, P, pill, P["up"], f"{r['Long-term']} · {r['Score']:+d}", r["_why"], lt_body(r)),
                            unsafe_allow_html=True)
        with right:
            st.markdown(f"#### 🔴 Weak: avoid or sell · {int((df['Score'] <= -20).sum())} found")
            if sells.empty:
                st.info("No clearly weak stocks right now.")
            for _, r in sells.iterrows():
                st.markdown(_card(r, P, pill, P["down"], f"{r['Long-term']} · {r['Score']:+d}", r["_risks"],
                                  lt_body(r)), unsafe_allow_html=True)
        cols = ["Name", "Ticker", "Long-term", "Score", "Price", "6m %", "1y %", "From 52w high %", "Volatility %"]
        if fundamentals is not None:
            cols += ["P/E", "Div yield %"]
        table = df.sort_values("Score", ascending=False)[cols]
        cfg = {"Price": st.column_config.NumberColumn(format="%.3f"),
               "Score": st.column_config.ProgressColumn(min_value=-100, max_value=100, format="%d"),
               "6m %": st.column_config.NumberColumn(format="%+.1f%%"),
               "1y %": st.column_config.NumberColumn(format="%+.1f%%"),
               "From 52w high %": st.column_config.NumberColumn(format="%+.1f%%"),
               "Volatility %": st.column_config.NumberColumn("Volatility / yr", format="%.0f%%"),
               "P/E": st.column_config.NumberColumn(format="%.1f"),
               "Div yield %": st.column_config.NumberColumn(format="%.1f%%")}

    with st.expander(f"📋 Full ranking ({len(table)} stocks)"):
        st.dataframe(table, hide_index=True, width="stretch", height=min(36 * (len(table) + 1) + 4, 700),
                     column_config=cfg)
    st.caption("Open any stock in **📊 Analysis & Forecast** (type its name or code) to see the chart and full plan "
               "before buying. Picks are generated automatically from price data"
               + (" and Yahoo fundamentals" if fundamentals is not None else "")
               + " and are not investment advice.")
