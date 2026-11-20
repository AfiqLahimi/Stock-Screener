"""More research tabs inside Analysis & Forecast: News & analysts, Financials (statements + peers),
Valuation (DCF), Ownership & flow (holders, insiders, short interest, options) and research Notes."""
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import fundamentals as F
import stockinfo as SI
import ui
from core import P, display_name, pill
from views.analysis_plus import is_company
from views.dashboard import load_news

NOTE_TAGS = ["Thesis", "Plan", "Risk", "Update", "Lesson"]


def _big(v):
    return F._big(v) if v is not None and v == v else "N/A"


def _clean_name(n):
    return re.sub(r"\b(Berhad|Bhd\.?|Holdings|Corporation|Corp\.?|Inc\.?|Ltd\.?|plc|Group)\b", "", n or "",
                  flags=re.I).strip(" ,.")


# ============================================================================
# News & analysts
# ============================================================================
def render_news_analysts(ticker, price, mobile):
    if not is_company(ticker):
        st.info("News and analyst coverage apply to companies.")
        return
    info = F.load(ticker).get("info") or {}
    a = SI.analysts(ticker)
    left, right = (st.container(), st.container()) if mobile else st.columns([1, 1.15], gap="small")
    with left, ui.panel("Analyst view", "Yahoo Finance consensus"):
        if not a or (a.get("n") in (None, 0) and a.get("target_mean") is None):
            st.markdown("<div class='muted'>N/A: no analyst coverage on Yahoo for this stock (common for smaller "
                        "Bursa companies).</div>", unsafe_allow_html=True)
        else:
            key = (a.get("key") or "none").replace("_", " ").title()
            col = P["up"] if "Buy" in key else P["down"] if ("Sell" in key or "Under" in key) else P["muted"]
            tm, th, tl = a.get("target_mean"), a.get("target_high"), a.get("target_low")
            up = (tm / price - 1) * 100 if tm and price else None
            st.markdown(f"<div style='display:flex;gap:10px;align-items:center;flex-wrap:wrap'>{pill(key, col)}"
                        f"<span class='muted'>{a.get('n') or 'N/A'} analysts · score {a.get('mean') or 'N/A'} "
                        f"(1 = strong buy, 5 = sell)</span></div>", unsafe_allow_html=True)
            if tm:
                st.markdown("<table class='kv' style='margin-top:8px'>" + "".join(
                    f"<tr><td>{k}</td><td class='mono'>{v}</td></tr>" for k, v in [
                        ("Average target", f"{tm:,.3f}" + ("" if up is None else f" ({up:+.1f}% vs price)")),
                        ("High / low target", f"{th:,.3f} / {tl:,.3f}" if th and tl else "N/A"),
                        ("Median target", f"{a['target_median']:,.3f}" if a.get("target_median") else "N/A")]) +
                    "</table>", unsafe_allow_html=True)
                if th and tl and price and th > tl:
                    lo, hi = min(tl, price), max(th, price)
                    pos = lambda v: (v - lo) / (hi - lo) * 100  # noqa: E731
                    st.markdown(f"<div style='position:relative;height:30px;margin:12px 4px 4px'>"
                                f"<div style='position:absolute;top:13px;left:{pos(tl):.0f}%;right:{100 - pos(th):.0f}%;"
                                f"height:4px;background:{P['accent']};opacity:.5;border-radius:3px'></div>"
                                f"<div title='price' style='position:absolute;top:6px;left:calc({pos(price):.0f}% - 2px);"
                                f"width:4px;height:18px;background:{P['text']};border-radius:2px'></div>"
                                f"<div title='average target' style='position:absolute;top:8px;left:calc({pos(tm):.0f}% - 7px);"
                                f"width:14px;height:14px;border-radius:50%;background:{P['accent']}'></div></div>"
                                f"<div class='muted' style='font-size:11.5px'>bar = analysts' low → high target · "
                                f"dot = average · line = current price</div>", unsafe_allow_html=True)
            recs = a.get("recs")
            if isinstance(recs, pd.DataFrame) and not recs.empty and "strongBuy" in recs:
                r0 = recs.iloc[0]
                cats = [("Strong buy", "strongBuy", P["up"]), ("Buy", "buy", P["up"]), ("Hold", "hold", P["muted"]),
                        ("Sell", "sell", P["down"]), ("Strong sell", "strongSell", P["down"])]
                fig = go.Figure(go.Bar(x=[c[0] for c in cats], y=[r0.get(c[1], 0) for c in cats],
                                       marker_color=[c[2] for c in cats], text=[r0.get(c[1], 0) for c in cats],
                                       textposition="outside"))
                fig.update_layout(height=200, margin=dict(l=4, r=4, t=14, b=4), paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="rgba(0,0,0,0)", font=dict(color=P["text"], size=11))
                fig.update_yaxes(visible=False)
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key="ri_recs")
            est = a.get("eps_est")
            if isinstance(est, pd.DataFrame) and not est.empty:
                show = est.rename(index={"0q": "This quarter", "+1q": "Next quarter", "0y": "This year",
                                         "+1y": "Next year"})
                cols = [c for c in ("avg", "low", "high", "numberOfAnalysts", "growth") if c in show]
                st.markdown("**Earnings-per-share estimates**")
                st.dataframe(show[cols].rename(columns={"avg": "Average", "low": "Low", "high": "High",
                                                        "numberOfAnalysts": "Analysts", "growth": "Growth"}),
                             width="stretch", height="content",
                             column_config={"Growth": st.column_config.NumberColumn(format="percent")})
            ch = a.get("changes")
            if isinstance(ch, pd.DataFrame) and not ch.empty:
                st.markdown("**Recent rating changes**")
                st.dataframe(ch.head(8), width="stretch", height="content")
        st.caption("Analyst targets are opinions, often revised after the price moves. Use them as one input.")
    with right, ui.panel("News", "Google News · last 30 days"):
        nm = display_name(ticker)
        long = _clean_name(info.get("longName") or info.get("shortName") or "")
        if ticker.endswith(".KL"):
            q = f'("{nm}" OR "{long}") Bursa when:30d' if long and long.lower() != nm.lower() else f'"{nm}" Bursa when:30d'
            items = load_news(q, "en-MY", "MY")
        else:
            items = load_news(f'"{long or nm}" stock when:30d', "en-US", "US")
        if not items:
            st.markdown("<div class='muted'>N/A: no recent headlines found.</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='newsbox'>" + "".join(
                f"<div class='news'><a href='{it['link']}' target='_blank'>{it['title']}</a>"
                f"<div class='meta'>{it['source']}{'' if it['date'] is None else ' · ' + it['date'].strftime('%d %b %H:%M')}"
                f"</div></div>" for it in items[:10]) + "</div>", unsafe_allow_html=True)
        st.caption("Headlines are matched by company name and can include unrelated stories with similar names.")


# ============================================================================
# Financials: statements + peers
# ============================================================================
def render_financials(ticker, C, mobile):
    if not is_company(ticker):
        st.info("Financial statements apply to companies.")
        return
    data = F.load(ticker)
    c1, c2, c3 = st.columns([1.6, 1.2, 1]) if not mobile else (st.container(),) * 3
    kind = c1.segmented_control("Statement", ["Income", "Balance sheet", "Cash flow"], default="Income",
                                key="fin_kind", label_visibility="collapsed") or "Income"
    per = c2.segmented_control("Period", ["Annual", "Quarterly"], default="Annual", key="fin_per",
                               label_visibility="collapsed") or "Annual"
    allr = c3.toggle("All lines", key="fin_all", value=False)
    k = {"Income": "income", "Balance sheet": "balance", "Cash flow": "cashflow"}[kind]
    src = data.get(k) if per == "Annual" else SI.quarterly(ticker).get(k)
    tbl = SI.statement_table(src, k, allr)
    cur = (data.get("info") or {}).get("financialCurrency") or ""
    with ui.panel(f"{kind} statement", f"{per.lower()} · {cur} · Yahoo Finance"):
        if tbl.empty:
            st.markdown("<div class='muted'>N/A: Yahoo has no statement data for this company.</div>",
                        unsafe_allow_html=True)
        else:
            fmt = tbl.map(lambda v: "N/A" if v != v else (f"{v:,.3f}" if abs(v) < 100 else F._big(v)))
            st.dataframe(fmt, width="stretch", height="content")
            pick = [r for r in ("Total Revenue", "Net Income", "Free Cash Flow", "Operating Cash Flow", "Total Debt",
                                "Stockholders Equity") if r in tbl.index]
            if pick:
                row = st.segmented_control("Chart a line", pick, default=pick[0], key="fin_row",
                                           label_visibility="collapsed") or pick[0]
                s = tbl.loc[row].dropna()
                fig = go.Figure(go.Bar(x=list(s.index), y=s.values, marker_color=[P["up"] if v >= 0 else P["down"] for v in s],
                                       text=[F._big(v) for v in s.values], textposition="outside"))
                fig.update_layout(height=240, margin=dict(l=4, r=4, t=14, b=4), paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="rgba(0,0,0,0)", font=dict(color=P["text"], size=11))
                fig.update_yaxes(gridcolor=P["grid"])
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key="fin_chart")
        st.caption("Yahoo usually provides 4 years of annual and 4–5 quarters of quarterly statements. Figures are "
                   "as reported; empty lines are not reported for this company.")
    pe, sec = SI.peers(ticker, C.bursa_all)
    with ui.panel("Peers", f"biggest companies in {sec}" if sec else "same sector"):
        if pe.empty:
            st.markdown("<div class='muted'>Peers N/A: shown for Bursa stocks using Yahoo's sector of every listed "
                        "company. For US stocks use the ⚖️ Compare page.</div>", unsafe_allow_html=True)
        else:
            pe = pe.rename(columns={"name": "Name", "price": "Price", "chg": "Change %", "mcap": "Market cap",
                                    "pe": "P/E", "div_yield": "Div. yield %"})
            st.dataframe(pe.drop(columns=["ticker"]), hide_index=True, width="stretch", height="content",
                         column_config={"Price": st.column_config.NumberColumn(format="%.3f"),
                                        "Change %": st.column_config.NumberColumn(format="%+.2f%%"),
                                        "Market cap": st.column_config.NumberColumn(format="compact"),
                                        "P/E": st.column_config.NumberColumn(format="%.1f"),
                                        "Div. yield %": st.column_config.NumberColumn(format="%.2f")})
            st.button("⚖️ Compare with the top 3 peers", key="fin_cmp", on_click=ui.goto, args=("compare",),
                      kwargs={"cmp_syms": list(dict.fromkeys([ticker] + [t for t in pe["ticker"] if t != ticker][:3]))})


# ============================================================================
# Valuation: DCF
# ============================================================================
def render_valuation(ticker, price, mobile):
    if not is_company(ticker):
        st.info("A cash-flow valuation applies to companies.")
        return
    data = F.load(ticker)
    info = data.get("info") or {}
    fres = F.analyse(data, price)
    if fres["is_financial"]:
        st.warning("This is a bank or insurer: free cash flow doesn't describe them well, so a DCF is unreliable "
                   "here. Use P/B, ROE and dividends (🏦 Fundamentals) instead.")
    fcf_s = F._row(data.get("cashflow"), "Free Cash Flow")
    fcf0 = float(fcf_s.iloc[-1]) if fcf_s is not None and len(fcf_s) else None
    fcf_avg = float(fcf_s.tail(3).mean()) if fcf_s is not None and len(fcf_s) >= 2 else None
    debt = F._row(data.get("balance"), "Total Debt")
    cash = F._row(data.get("balance"), "Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments")
    nd = (float(debt.iloc[-1]) if debt is not None else 0.0) - (float(cash.iloc[-1]) if cash is not None else 0.0)
    shares = info.get("sharesOutstanding")
    rev_cagr = F._cagr(F._row(data.get("income"), "Total Revenue"), 3)
    with ui.panel("Discounted cash flow", "what the business could be worth from its future cash"):
        if fcf0 is None or not shares:
            st.markdown("<div class='muted'>DCF N/A: free cash flow or share count not reported by Yahoo.</div>",
                        unsafe_allow_html=True)
            return
        a1, a2, a3 = st.columns(3) if not mobile else (st.container(),) * 3
        base = a1.selectbox("Starting cash flow", ["Latest year", "3-year average"], key="dcf_base",
                            help="Averaging smooths a one-off good or bad year")
        f0 = fcf_avg if base == "3-year average" and fcf_avg is not None else fcf0
        g1 = a2.slider("Growth, years 1–5 (%)", -10.0, 30.0, step=0.5, key="dcf_g1",
                       **({} if "dcf_g1" in st.session_state else {"value": float(np.clip(rev_cagr or 5.0, -5, 15))}))
        g2 = a3.slider("Growth, years 6–10 (%)", -5.0, 15.0, step=0.5, key="dcf_g2",
                       **({} if "dcf_g2" in st.session_state else {"value": 3.0}))
        b1, b2, b3 = st.columns(3) if not mobile else (st.container(),) * 3
        r = b1.slider("Discount rate (%)", 5.0, 18.0, step=0.5, key="dcf_r",
                      **({} if "dcf_r" in st.session_state else {"value": 9.0 if ticker.endswith(".KL") else 9.5}),
                      help="The return you require. Higher = more cautious. 8–10% is common for large companies.")
        gt = b2.slider("Terminal growth (%)", 0.0, 4.0, step=0.25, key="dcf_gt",
                       **({} if "dcf_gt" in st.session_state else {"value": 2.0}),
                       help="Growth forever after year 10. Keep it at or below long-run inflation/GDP growth.")
        mos = b3.slider("Margin of safety (%)", 0, 50, step=5, key="dcf_mos",
                        **({} if "dcf_mos" in st.session_state else {"value": 25}),
                        help="Discount to fair value before you'd consider it cheap")
        res = SI.dcf(f0, g1 / 100, g2 / 100, r / 100, gt / 100, nd, shares)
        if res is None:
            st.error("Discount rate must be above terminal growth.")
            return
        fv = res["per_share"]
        buy_below = fv * (1 - mos / 100)
        gap = (fv / price - 1) * 100 if price else None
        col = P["up"] if gap is not None and gap > mos else P["down"] if gap is not None and gap < 0 else P["warn"]
        if f0 <= 0:
            st.warning("Starting free cash flow is negative, so the model values the business below zero. A DCF "
                       "isn't meaningful until the company generates cash.")
        st.markdown("<div class='sg-row'>" + "".join(
            f"<div class='sg'><div class='sg-t'>{t}</div><div class='sg-v' style='color:{c}'>{v}</div>"
            f"<div class='sg-s'>{s}</div></div>" for t, v, c, s in [
                ("Fair value / share", f"{fv:,.3f}", col, "model estimate"),
                ("Current price", f"{price:,.3f}" if price else "N/A", P["text"], ""),
                ("Upside / downside", "N/A" if gap is None else f"{gap:+.0f}%", col, "vs fair value"),
                (f"Buy below ({mos}% safety)", f"{buy_below:,.3f}", P["accent"], "fair value × (1 − margin)")]) +
                    "</div>", unsafe_allow_html=True)
        st.markdown("<table class='kv'>" + "".join(f"<tr><td>{k}</td><td class='mono'>{v}</td></tr>" for k, v in [
            ("Starting free cash flow", _big(f0)), ("Value of cash flows, years 1–10", _big(res["pv_flows"])),
            ("Value after year 10 (terminal)", f"{_big(res['pv_terminal'])} ({res['terminal_share'] * 100:.0f}% of total)"),
            ("Less net debt", _big(nd)), ("Equity value", _big(res["equity"])), ("Shares", _big(shares))]) +
                    "</table>", unsafe_allow_html=True)
        rs, gs = [r - 1, r, r + 1], [max(gt - 0.5, 0), gt, gt + 0.5]
        grid = pd.DataFrame({f"terminal {g:.2f}%": [
            (lambda v: None if v is None else v["per_share"])(SI.dcf(f0, g1 / 100, g2 / 100, rr / 100, g / 100, nd, shares))
            for rr in rs] for g in gs}, index=[f"discount {rr:.1f}%" for rr in rs])
        st.markdown("**Sensitivity: fair value per share**")
        st.dataframe(grid, width="stretch", height="content",
                     column_config={c: st.column_config.NumberColumn(format="%.3f") for c in grid.columns})
        st.caption("A DCF is only as good as its assumptions: small changes in growth or discount rate move the "
                   "answer a lot (see the table). If most of the value comes from after year 10, treat the result "
                   "with extra care. This is an estimate, not a price target.")


# ============================================================================
# Ownership & flow
# ============================================================================
def render_ownership(ticker, mobile):
    if not is_company(ticker):
        st.info("Ownership data applies to companies.")
        return
    o = SI.ownership(ticker)
    left, right = (st.container(), st.container()) if mobile else st.columns(2, gap="small")
    with left, ui.panel("Who owns it", "Yahoo Finance"):
        mj = o.get("major")
        if isinstance(mj, pd.DataFrame) and not mj.empty:
            val = mj.iloc[:, 0] if mj.shape[1] else pd.Series(dtype=float)
            lab = {"insidersPercentHeld": "Insiders", "institutionsPercentHeld": "Institutions",
                   "institutionsFloatPercentHeld": "Institutions (of free float)", "institutionsCount": "Institutions (count)"}
            st.markdown("<table class='kv'>" + "".join(
                f"<tr><td>{lab.get(k, k)}</td><td class='mono'>{v * 100:.1f}%</td></tr>" if k != "institutionsCount"
                else f"<tr><td>{lab[k]}</td><td class='mono'>{v:,.0f}</td></tr>" for k, v in val.items() if v == v) +
                "</table>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='muted'>Holder breakdown N/A.</div>", unsafe_allow_html=True)
        inst = o.get("inst")
        if isinstance(inst, pd.DataFrame) and not inst.empty:
            st.markdown("**Largest institutional holders**")
            cols = [c for c in ("Holder", "pctHeld", "Shares", "Date Reported", "pctChange") if c in inst]
            st.dataframe(inst[cols].head(10), hide_index=True, width="stretch", height="content",
                         column_config={"Date Reported": st.column_config.DateColumn("Reported", format="DD MMM YYYY"),
                                        "pctHeld": st.column_config.NumberColumn("% held", format="percent"),
                                        "pctChange": st.column_config.NumberColumn("Change", format="percent"),
                                        "Shares": st.column_config.NumberColumn(format="compact")})
        st.caption("For Bursa companies Yahoo's holder data is often thin; the company's annual report lists its "
                   "top 30 shareholders.")
    with right, ui.panel("Insider trades", "directors and officers"):
        ins = o.get("insiders")
        if isinstance(ins, pd.DataFrame) and not ins.empty:
            cols = [c for c in ("Start Date", "Insider", "Position", "Transaction", "Text", "Shares", "Value") if c in ins]
            st.dataframe(ins[cols].head(15), hide_index=True, width="stretch", height="content",
                         column_config={"Start Date": st.column_config.DateColumn("Date", format="DD MMM YYYY"),
                                        "Shares": st.column_config.NumberColumn(format="compact"),
                                        "Value": st.column_config.NumberColumn(format="compact")})
            st.caption("Insider buying is usually more telling than selling (people sell for many reasons).")
        else:
            st.markdown("<div class='muted'>N/A: Yahoo has no insider filings for this stock. Bursa directors' "
                        "dealings are announced on Bursa Malaysia's website.</div>", unsafe_allow_html=True)
    sh = o.get("short")
    with ui.panel("Short interest", "US stocks · reported twice a month"):
        if not sh:
            st.markdown("<div class='muted'>N/A: short-interest figures are published for US stocks only.</div>",
                        unsafe_allow_html=True)
        else:
            chg = (sh["shares"] / sh["prior"] - 1) * 100 if sh.get("prior") else None
            st.markdown("<table class='kv'>" + "".join(f"<tr><td>{k}</td><td class='mono'>{v}</td></tr>" for k, v in [
                ("Shares sold short", _big(sh["shares"]) + ("" if chg is None else f" ({chg:+.1f}% vs prior month)")),
                ("Short % of free float", "N/A" if sh.get("pct_float") is None else f"{sh['pct_float'] * 100:.2f}%"),
                ("Days to cover", "N/A" if sh.get("ratio") is None else f"{sh['ratio']:.1f}"),
                ("As of", sh.get("date") or "N/A")]) + "</table>", unsafe_allow_html=True)
            st.caption("High short interest (say > 10% of float) means many traders bet on a fall; it also raises the "
                       "chance of a sharp 'short squeeze' rally if good news arrives.")
    with ui.panel("Options activity", "US stocks · nearest expiries"):
        op = SI.options_summary(ticker)
        if not op:
            st.markdown("<div class='muted'>N/A: listed stock options exist for US stocks only (Bursa has no "
                        "single-stock options).</div>", unsafe_allow_html=True)
            return
        ch = op["chain"]
        tot = ch[["call_oi", "put_oi", "call_vol", "put_vol"]].sum()
        pcr_oi = tot["put_oi"] / tot["call_oi"] if tot["call_oi"] else None
        pcr_v = tot["put_vol"] / tot["call_vol"] if tot["call_vol"] else None
        by_k = ch.groupby("strike")[["call_oi", "put_oi"]].sum()
        cw, pw = by_k["call_oi"].idxmax(), by_k["put_oi"].idxmax()
        st.markdown("<table class='kv'>" + "".join(f"<tr><td>{k}</td><td class='mono'>{v}</td></tr>" for k, v in [
            ("Expiries included", ", ".join(sorted(ch["expiry"].unique()))),
            ("Put/call ratio (open interest)", "N/A" if pcr_oi is None else f"{pcr_oi:.2f}"),
            ("Put/call ratio (today's volume)", "N/A" if pcr_v is None else f"{pcr_v:.2f}"),
            ("Biggest call open interest at", f"{cw:,.2f} (often acts as resistance)"),
            ("Biggest put open interest at", f"{pw:,.2f} (often acts as support)")]) + "</table>",
                    unsafe_allow_html=True)
        spot = op.get("spot")
        near = by_k if not spot else by_k[(by_k.index > spot * 0.75) & (by_k.index < spot * 1.25)]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=near.index, y=near["call_oi"], name="Calls", marker_color=P["up"]))
        fig.add_trace(go.Bar(x=near.index, y=-near["put_oi"], name="Puts", marker_color=P["down"]))
        if spot:
            fig.add_vline(x=spot, line=dict(color=P["text"], dash="dot"))
        fig.update_layout(height=260, barmode="relative", margin=dict(l=4, r=4, t=10, b=4),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=P["text"], size=11),
                          legend=dict(orientation="h", y=1.1))
        fig.update_yaxes(gridcolor=P["grid"], title="open interest (puts below)")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key="own_opt")
        st.caption("Put/call above ~1 = more bearish bets or hedging; below ~0.7 = more bullish bets. Open interest "
                   "is updated once a day.")


# ============================================================================
# Notes
# ============================================================================
def _add_note(ticker):
    txt = (st.session_state.get("note_txt") or "").strip()
    if not txt:
        return
    notes = dict(st.session_state.get("notes", {}))
    lst = list(notes.get(ticker, []))
    lst.insert(0, {"date": pd.Timestamp.now(tz="Asia/Kuala_Lumpur").strftime("%Y-%m-%d %H:%M"),
                   "tag": st.session_state.get("note_tag") or "Update", "text": txt})
    notes[ticker] = lst
    st.session_state["notes"] = notes
    st.session_state["note_txt"] = ""


def _del_note(ticker, i):
    notes = dict(st.session_state.get("notes", {}))
    lst = list(notes.get(ticker, []))
    if 0 <= i < len(lst):
        lst.pop(i)
    if lst:
        notes[ticker] = lst
    else:
        notes.pop(ticker, None)
    st.session_state["notes"] = notes


TAG_COL = {"Thesis": "accent", "Plan": "up", "Risk": "down", "Update": "muted", "Lesson": "warn"}


def note_html(n):
    return (f"<div class='op' style='border-left-color:{P[TAG_COL.get(n['tag'], 'muted')]}'>"
            f"<div class='muted' style='font-size:11.5px'>{n['date']} · {n['tag']}</div>"
            f"<div style='white-space:pre-wrap;margin-top:3px'>{n['text']}</div></div>")


def render_notes(ticker, mobile):
    with ui.panel(f"My notes on {display_name(ticker)}", "saved with your watchlist"):
        c1, c2 = st.columns([4, 1]) if not mobile else (st.container(), st.container())
        c1.text_area("Note", key="note_txt", height=90, label_visibility="collapsed",
                     placeholder="Why you'd buy or sell, your plan (entry, stop, target), what would change your mind…")
        c2.selectbox("Tag", NOTE_TAGS, key="note_tag", label_visibility="collapsed")
        c2.button("💾 Save note", key="note_save", on_click=_add_note, args=(ticker,), width="stretch")
        lst = st.session_state.get("notes", {}).get(ticker, [])
        if not lst:
            st.markdown("<div class='muted'>No notes yet. Writing down your reason before a trade is one of the best "
                        "habits a beginner can build.</div>", unsafe_allow_html=True)
        for i, n in enumerate(lst):
            a, b = st.columns([12, 1])
            a.markdown(note_html(n), unsafe_allow_html=True)
            b.button("🗑", key=f"note_del_{i}", on_click=_del_note, args=(ticker, i), help="Delete this note")
