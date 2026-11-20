"""Dashboard: market overview, market intelligence, watchlist and trading opportunities in one screen.

Everything is computed from free Yahoo Finance data (indices, the Bursa-wide list, the scanned universe) and
Google News RSS. Missing data is shown as N/A, never estimated.
"""
import urllib.parse
import xml.etree.ElementTree as ET
from functools import partial

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import extras
import terminal
import ui
from core import (DEMO, P, US, add_indicators, adx_regime, display_name, load_bursa_snapshot, load_one, pill,
                  scan_universe)

OVERVIEW = [("^KLSE", "FBM KLCI"), ("^GSPC", "S&P 500"), ("^IXIC", "Nasdaq"), ("^DJI", "Dow Jones"),
            ("MYR=X", "USD/MYR"), ("GC=F", "Gold"), ("CL=F", "Crude oil"), ("^VIX", "VIX")]
CHART_IDX = {"^KLSE": "FBM KLCI", "^GSPC": "S&P 500", "^IXIC": "Nasdaq", "^DJI": "Dow Jones"}
BUY, SELL = ("BUY", "STRONG BUY"), ("SELL", "STRONG SELL")
MIN_VALUE = 100_000          # RM traded per day: below this a "top gainer" is usually a thin penny stock


# ============================================================================
# Data
# ============================================================================
def index_stats(t):
    """Level, changes over several windows, trend vs EMAs and ADX regime for one index/asset."""
    d = load_one(t, "1d", "2y")
    if d is None or d.empty or len(d) < 2:
        return None
    c = d["Close"]
    last = float(c.iloc[-1])

    def chg(n):
        return (last / float(c.iloc[-1 - n]) - 1) * 100 if len(c) > n else None

    this_year = c[c.index.year == c.index[-1].year]
    ytd = (last / float(c[c.index < this_year.index[0]].iloc[-1]) - 1) * 100 \
        if len(this_year) and (c.index < this_year.index[0]).any() else None
    x = add_indicators(d) if len(d) >= 30 else None
    e50 = float(x["EMA50"].iloc[-1]) if x is not None and len(d) >= 50 else None
    e200 = float(x["EMA200"].iloc[-1]) if x is not None and len(d) >= 200 else None
    regime = adx_regime(x["ADX"].iloc[-1], x["PDI"].iloc[-1], x["MDI"].iloc[-1]) \
        if x is not None and pd.notna(x["ADX"].iloc[-1]) else None
    return {"d": d, "last": last, "1D": chg(1), "1W": chg(5), "1M": chg(21), "YTD": ytd, "e50": e50, "e200": e200,
            "regime": regime, "spark": c.tail(60).tolist(), "date": d.index[-1]}


@st.cache_data(ttl=600, show_spinner=False)
def us_movers(kind):
    """Yahoo's free predefined US screeners: day_gainers, day_losers, most_actives."""
    if DEMO:
        rng = np.random.default_rng(len(kind))
        rows = [{"Ticker": t, "Name": n, "Price": float(rng.uniform(20, 400)),
                 "Change %": float(rng.normal(0, 4)), "Volume": float(rng.lognormal(16, 1))} for t, n in US.items()]
        df = pd.DataFrame(rows)
        return df.sort_values("Change %", ascending=(kind == "day_losers")) if kind != "most_actives" \
            else df.sort_values("Volume", ascending=False)
    try:
        import yfinance as yf
        q = yf.screen(kind, count=15).get("quotes", [])
        return pd.DataFrame([{"Ticker": x.get("symbol"), "Name": x.get("shortName") or x.get("symbol"),
                              "Price": x.get("regularMarketPrice"), "Change %": x.get("regularMarketChangePercent"),
                              "Volume": x.get("regularMarketVolume")} for x in q])
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=900, show_spinner=False)
def load_news(query, hl, gl):
    if DEMO:
        now = pd.Timestamp.now()
        return [{"title": f"Demo headline {i}: market moves on {query.split()[0]} news", "source": "Demo Wire",
                 "date": now - pd.Timedelta(hours=i * 2), "link": "https://news.google.com"} for i in range(1, 9)]
    url = (f"https://news.google.com/rss/search?q={urllib.parse.quote(query)}&hl={hl}&gl={gl}&ceid={gl}:en")
    try:
        root = ET.fromstring(extras._http_get(url))
    except Exception:
        return []
    items = []
    for it in root.iter("item"):
        title = (it.findtext("title") or "").strip()
        src = (it.findtext("source") or "").strip()
        if src and title.endswith(" - " + src):
            title = title[: -len(src) - 3]
        try:
            date = pd.Timestamp(pd.to_datetime(it.findtext("pubDate"), utc=True)).tz_convert(
                "Asia/Kuala_Lumpur").tz_localize(None)
        except Exception:
            date = None
        items.append({"title": title, "source": src, "date": date, "link": it.findtext("link")})
    items.sort(key=lambda x: x["date"] or pd.Timestamp(0), reverse=True)
    return items[:12]


# ============================================================================
# Small renderers
# ============================================================================
def _fmt(v, f="{:+.2f}%"):
    return "N/A" if v is None or (isinstance(v, float) and np.isnan(v)) else f.format(v)


def _col(v):
    return P["muted"] if v is None or v != v else (P["up"] if v >= 0 else P["down"])


def spark(vals, col, w=120, h=30):
    v = np.asarray(vals, float)
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return ""
    lo, hi = v.min(), v.max()
    rng = (hi - lo) or 1
    pts = " ".join(f"{i * w / (len(v) - 1):.1f},{h - 2 - (x - lo) / rng * (h - 4):.1f}" for i, x in enumerate(v))
    return (f"<svg viewBox='0 0 {w} {h}' preserveAspectRatio='none' class='spark'>"
            f"<polyline points='{pts}' fill='none' stroke='{col}' stroke-width='1.6' "
            f"vector-effect='non-scaling-stroke'/></svg>")


def _open_row(key, tickers):
    sel = st.session_state.get(key)
    try:
        rows = sel["selection"]["rows"]
    except Exception:
        rows = []
    if rows and rows[0] < len(tickers):
        ui.goto("analysis", an_typed=tickers[rows[0]])


def table(df, key, cols, height=None, config=None):
    """Compact clickable table: picking a row opens that stock in Analysis."""
    if df is None or df.empty:
        st.markdown("<div class='muted'>N/A: nothing matches right now.</div>", unsafe_allow_html=True)
        return
    df = df.reset_index(drop=True)
    cc = st.column_config
    base = {"Name": cc.TextColumn(width="medium" if len(cols) > 5 else None), "Ticker": cc.TextColumn(width="small"),
            "Price": cc.NumberColumn(format="%.3f"), "Change %": cc.NumberColumn(format="%+.2f%%"),
            "Score": cc.NumberColumn(format="%+d", width="small"), "RVOL": cc.NumberColumn(format="%.1f×"),
            "Volume": cc.NumberColumn(format="compact"), "Value (RM)": cc.NumberColumn(format="compact"),
            "Return 20 %": cc.NumberColumn("20-day %", format="%+.1f%%"), "ADX": cc.NumberColumn(format="%.0f"),
            "To support %": cc.NumberColumn("To support", format="%.1f%%"),
            "To resistance %": cc.NumberColumn("To resistance", format="%.1f%%"),
            "RSI": cc.NumberColumn(format="%.0f")}
    st.dataframe(df, hide_index=True, width="stretch", column_order=cols, key=key,
                 column_config={**base, **(config or {})}, height=height or ("content" if len(df) <= 12 else 430),
                 on_select=partial(_open_row, key, list(df["Ticker"])), selection_mode="single-row")


# ============================================================================
# Page
# ============================================================================
def render(C):
    inject()
    mob = C.mobile
    st.caption("Market overview, breadth, movers, news, your watchlist and today's technical set-ups. "
               "Click any row to open the stock in Analysis. Prices from Yahoo Finance (Bursa ~15 min delayed).")

    # ------------------------------------------------------------ A. overview
    stats = {t: index_stats(t) for t, _ in OVERVIEW}
    cards = []
    for t, name in OVERVIEW:
        s = stats[t]
        if s is None:
            cards.append(f"<div class='idx'><div class='n'>{name}</div><div class='v'>N/A</div></div>")
            continue
        col = _col(s["1D"])
        dec = 4 if t == "MYR=X" else 2
        cards.append(f"<div class='idx'><div class='n'>{name}</div><div class='v'>{s['last']:,.{dec}f}</div>"
                     f"<div class='c' style='color:{col}'>{_fmt(s['1D'])}</div>{spark(s['spark'], col)}</div>")
    st.markdown(f"<div class='idx-grid'>{''.join(cards)}</div>", unsafe_allow_html=True)

    scan = pd.DataFrame()
    if C.universe:
        with st.spinner(f"Scanning {len(C.universe)} stocks (daily)…"):
            scan = scan_universe(C.universe, "1d", "2y", "trading days")
    is_bursa = bool(C.universe) and sum(t.endswith(".KL") for t in C.universe) >= len(C.universe) / 2
    with st.spinner("Loading today's Bursa moves…"):
        try:
            snap = load_bursa_snapshot()
        except Exception:
            snap = pd.DataFrame()
    liq = pd.DataFrame()
    if not snap.empty:
        liq = snap.dropna(subset=["chg", "price"]).copy()
        liq["Value (RM)"] = liq["price"] * liq["volume"].fillna(0)
        liq = liq[liq["price"] * liq["avg_volume"].fillna(0) >= MIN_VALUE]

    # ------------------------------------------------------------ E + B. charts | pulse
    left, right = (st.container(), st.container()) if mob else st.columns([2.45, 1], gap="small")
    with left, ui.panel("Market charts", "daily · EMA 20 / 50 / 200"):
        h1, h2 = st.columns([1.35, 1])
        idx = h1.segmented_control("Index", list(CHART_IDX), default="^KLSE", key="db_idx",
                                   format_func=CHART_IDX.get, label_visibility="collapsed") or "^KLSE"
        rng = h2.segmented_control("Range", ["3M", "6M", "1Y", "2Y"], default="6M", key="db_rng",
                                   label_visibility="collapsed") or "6M"
        s = stats.get(idx)
        if s is None:
            st.info(f"{CHART_IDX[idx]}: chart N/A (no data from Yahoo right now).")
        else:
            st.plotly_chart(index_chart(s["d"], CHART_IDX[idx], rng, mob), width="stretch",
                            config={"displaylogo": False, "displayModeBar": False}, key="db_chart")
        rows = []
        for t, name in OVERVIEW:
            s = stats[t]
            if s is None:
                rows.append({"Market": name, "Last": None, "1D": None, "1W": None, "1M": None, "YTD": None,
                             "Trend": "N/A", "ADX regime": "N/A"})
                continue
            if s["e50"] is None or s["e200"] is None:
                trend = "N/A (short history)"
            else:
                pts = (s["last"] > s["e50"]) + (s["last"] > s["e200"]) + (s["e50"] > s["e200"])
                trend = ["Downtrend", "Weak / turning", "Recovering", "Uptrend"][pts]
            rows.append({"Market": name, "Last": s["last"], "1D": s["1D"], "1W": s["1W"], "1M": s["1M"],
                         "YTD": s["YTD"], "Trend": trend, "ADX regime": s["regime"] or "N/A"})
        cc = st.column_config
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                     column_order=["Market", "Last", "1D", "1M", "Trend"] if mob else None,
                     column_config={"Last": cc.NumberColumn(format="%,.2f"),
                                    **{k: cc.NumberColumn(format="%+.2f%%") for k in ("1D", "1W", "1M", "YTD")}})
        st.caption("Trend: price vs 50- and 200-day EMAs and whether the 50 is above the 200 (3 of 3 = Uptrend). "
                   "ADX regime: trend strength (see Learn).")

    with right, ui.panel("Market pulse", "Bursa" if is_bursa else "your list"):
        pulse(scan, liq if is_bursa else pd.DataFrame(), stats.get("^KLSE" if is_bursa else "^GSPC"),
              "FBM KLCI" if is_bursa else "S&P 500", snap)

    # ------------------------------------------------------------ A. movers
    with ui.panel("Top movers", f"today · Bursa snapshot {_asof(snap)}"):
        tb, tu = st.tabs(["🇲🇾 Bursa", "🇺🇸 US"])
        with tb:
            if liq.empty:
                st.info("Bursa movers N/A: the Bursa-wide list from Yahoo isn't available right now.")
            else:
                m = liq.rename(columns={"ticker": "Ticker", "name": "Name", "price": "Price", "chg": "Change %"})
                g1, g2, g3 = (st.container(),) * 3 if mob else st.columns(3)
                with g1:
                    st.markdown("**▲ Top gainers**")
                    table(m.sort_values("Change %", ascending=False).head(8), "db_gain",
                          ["Name", "Price", "Change %"])
                with g2:
                    st.markdown("**▼ Top losers**")
                    table(m.sort_values("Change %").head(8), "db_lose", ["Name", "Price", "Change %"])
                with g3:
                    st.markdown("**🔥 Most active (by value)**")
                    table(m.sort_values("Value (RM)", ascending=False).head(8), "db_act",
                          ["Name", "Price", "Value (RM)"])
                st.caption(f"Only stocks trading at least RM{MIN_VALUE:,.0f} a day on average, so thin penny stocks "
                           f"don't fill the list.")
        with tu:
            u1, u2, u3 = (st.container(),) * 3 if mob else st.columns(3)
            for col_, kind, title in ((u1, "day_gainers", "▲ Top gainers"), (u2, "day_losers", "▼ Top losers"),
                                      (u3, "most_actives", "🔥 Most active")):
                with col_:
                    st.markdown(f"**{title}**")
                    table(us_movers(kind).head(8), f"db_us_{kind}", ["Ticker", "Name", "Price", "Change %"]
                          if kind != "most_actives" else ["Ticker", "Price", "Change %", "Volume"])
            st.caption("From Yahoo's free US screeners (large and mid caps). Refreshed every 10 minutes.")

    # ------------------------------------------------------------ B. sectors | news
    sc_col, nw_col = (st.container(), st.container()) if mob else st.columns([1, 1], gap="small")
    with sc_col, ui.panel("Bursa sectors", "today · market-cap weighted"):
        if liq.empty:
            st.info("Sector performance N/A right now.")
        else:
            st.plotly_chart(sector_chart(liq, mob), width="stretch",
                            config={"displaylogo": False, "displayModeBar": False}, key="db_sec")
            st.caption("Average of today's % change per Yahoo sector, weighted by market cap (liquid stocks only). "
                       "Number in brackets = stocks counted.")
            st.button("🗺️ Open the full heatmap", on_click=ui.goto, args=("heatmap",), key="db_hm")
    with nw_col, ui.panel("Market news", "Google News"):
        n1, n2 = st.tabs(["🇲🇾 Bursa / Malaysia", "🇺🇸 US markets"])
        with n1:
            news_list(load_news('("Bursa Malaysia" OR KLCI OR "Bursa stocks") when:3d', "en-MY", "MY"))
        with n2:
            news_list(load_news('("stock market" OR "Wall Street" OR "S&P 500") when:2d', "en-US", "US"))

    # ------------------------------------------------------------ C. watchlist
    with ui.panel("Watchlist", "shared with the Terminal"):
        watchlist(C)

    # ------------------------------------------------------------ D. opportunities
    with ui.panel("Trading opportunities", f"daily candles · {len(scan)} stocks from the sidebar universe"):
        opportunities(scan, mob)


def _asof(df):
    try:
        return pd.Timestamp(df.attrs["asof"]).strftime("%H:%M MYT")
    except Exception:
        return "N/A"


def index_chart(d, name, rng, mob):
    x = add_indicators(d) if len(d) >= 30 else d
    n = {"3M": 63, "6M": 126, "1Y": 252, "2Y": 504}[rng]
    x = x.tail(n)
    labels = [ts.strftime("%d %b %y") for ts in x.index]
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=labels, open=x["Open"], high=x["High"], low=x["Low"], close=x["Close"],
                                 name=name, increasing=dict(line=dict(color=P["up"]), fillcolor=P["up"]),
                                 decreasing=dict(line=dict(color=P["down"]), fillcolor=P["down"])))
    for k, col in (("EMA20", P["ema20"]), ("EMA50", P["ema50"]), ("EMA200", P["ema200"])):
        if k in x and x[k].notna().any():
            fig.add_trace(go.Scatter(x=labels, y=x[k], name=k, line=dict(color=col, width=1.3),
                                     hovertemplate=f"{k} %{{y:,.2f}}<extra></extra>"))
    fig.update_layout(template="plotly_dark", height=300 if mob else 360, margin=dict(l=4, r=4, t=6, b=4),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis_rangeslider_visible=False,
                      showlegend=not mob, legend=dict(orientation="h", y=1.02, x=0, font=dict(size=11)),
                      hovermode="x unified", font=dict(family="Inter, sans-serif", color=P["text"], size=11))
    fig.update_xaxes(type="category", nticks=6, showgrid=False)
    fig.update_yaxes(gridcolor=P["grid"], side="right")
    return fig


def pulse(scan, liq, idx, idx_name, snap):
    comps = []
    if not liq.empty:
        adv, dec = int((liq["chg"] > 0).sum()), int((liq["chg"] < 0).sum())
        if adv + dec:
            comps.append(("Breadth (stocks up vs down today)", adv / (adv + dec) * 100, f"{adv} up · {dec} down"))
    elif not scan.empty:
        adv, dec = int((scan["Change %"] > 0).sum()), int((scan["Change %"] < 0).sum())
        if adv + dec:
            comps.append(("Breadth (scanned stocks up vs down)", adv / (adv + dec) * 100, f"{adv} up · {dec} down"))
    if not scan.empty:
        a50 = scan["Above EMA50"].dropna()
        if len(a50):
            comps.append(("Stocks above their 50-day EMA", a50.mean() * 100, f"{int(a50.sum())} of {len(a50)}"))
        b, s_ = int(scan["_label"].isin(BUY).sum()), int(scan["_label"].isin(SELL).sum())
        if b + s_:
            comps.append(("Buy vs sell signals", b / (b + s_) * 100, f"{b} buy · {s_} sell"))
    if idx and idx["e50"] is not None and idx["e200"] is not None:
        pts = (idx["last"] > idx["e50"]) + (idx["last"] > idx["e200"]) + (idx["e50"] > idx["e200"])
        comps.append((f"{idx_name} trend (vs 50/200-day EMA)", pts / 3 * 100, f"{pts} of 3 bullish checks"))
    if not comps:
        st.info("Market pulse N/A: not enough data loaded.")
        return
    score = float(np.mean([c[1] for c in comps]))
    lab, col = next((l_, c_) for lim, l_, c_ in ((25, "Very weak", P["down"]), (45, "Weak", P["down"]),
                                                  (55, "Neutral", P["muted"]), (75, "Positive", P["up"]),
                                                  (101, "Strong", P["up"])) if score < lim)
    bars = "".join(
        f"<div class='pc'><div class='pl'><span>{name}</span><b>{val:.0f}</b></div>"
        f"<div class='bar'><i style='width:{val:.0f}%;background:{_col(val - 50)}'></i></div>"
        f"<div class='ps'>{sub}</div></div>" for name, val, sub in comps)
    st.markdown(f"""<div class='card'>
<div style='display:flex;align-items:baseline;gap:10px'><div style='font-size:34px;font-weight:700'>{score:.0f}</div>
<div class='muted'>/ 100</div><div style='margin-left:auto'>{pill(lab, col)}</div></div>
<div class='bar big'><i style='width:{score:.0f}%;background:{col}'></i></div>
<div style='margin-top:12px'>{bars}</div>
<div class='note'>Sentiment = simple average of the parts above (50 = balanced). It describes today's
breadth and trend participation; it is not a forecast.</div></div>""", unsafe_allow_html=True)
    if not scan.empty:
        hi, lo = int(scan["Breakout"].sum()), int(scan["Breakdown"].sum())
        st.markdown(f"<div class='card divider'><div class='pl'><span>New 20-day highs / lows (scanned)</span>"
                    f"<b><span style='color:{P['up']}'>{hi}</span> / <span style='color:{P['down']}'>{lo}</span></b>"
                    f"</div><div class='note'>More new highs than lows = buyers in control across many stocks."
                    f"</div></div>", unsafe_allow_html=True)


def sector_chart(liq, mob):
    g = liq.assign(w=liq["mcap"].fillna(0)).groupby("sector").apply(
        lambda x: pd.Series({"chg": np.average(x["chg"], weights=x["w"]) if x["w"].sum() > 0 else x["chg"].mean(),
                             "n": len(x)}), include_groups=False).sort_values("chg")
    fig = go.Figure(go.Bar(x=g["chg"], y=[f"{s} ({int(n)})" for s, n in zip(g.index, g["n"])], orientation="h",
                           marker_color=[P["up"] if v >= 0 else P["down"] for v in g["chg"]],
                           text=[f"{v:+.2f}%" for v in g["chg"]], textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}: %{x:+.2f}%<extra></extra>"))
    fig.update_layout(template="plotly_dark", height=max(260, 26 * len(g) + 40), margin=dict(l=4, r=40, t=4, b=4),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter, sans-serif", color=P["text"], size=11))
    lo, hi = float(min(g["chg"].min(), 0)), float(max(g["chg"].max(), 0))
    pad = (hi - lo) * 0.28 or 1
    fig.update_xaxes(gridcolor=P["grid"], zerolinecolor=P["border"], ticksuffix="%",
                     range=[lo - (pad if lo < 0 else 0), hi + pad])
    return fig


def news_list(items):
    if not items:
        st.markdown("<div class='muted'>News N/A: Google News didn't respond. Try again later.</div>",
                    unsafe_allow_html=True)
        return
    html = "".join(
        f"<div class='news'><a href='{it['link']}' target='_blank'>{it['title']}</a>"
        f"<div class='meta'>{it['source']} · {it['date']:%d %b %H:%M}</div></div>" if it["date"] is not None else
        f"<div class='news'><a href='{it['link']}' target='_blank'>{it['title']}</a>"
        f"<div class='meta'>{it['source']}</div></div>" for it in items[:8])
    st.markdown(f"<div class='newsbox'>{html}</div>", unsafe_allow_html=True)


def _wl_add():
    t = st.session_state.get("db_wl_add")
    wl = st.session_state.setdefault("term_stocks", terminal.DEFAULT_STOCKS.copy())
    if t and t not in wl:
        wl.append(t)
    st.session_state["db_wl_add"] = None


def _wl_remove():
    rm = set(st.session_state.get("db_wl_rm") or [])
    st.session_state["term_stocks"] = [t for t in st.session_state.get("term_stocks", []) if t not in rm]
    st.session_state["db_wl_rm"] = []


def watchlist(C):
    wl = st.session_state.setdefault("term_stocks", terminal.DEFAULT_STOCKS.copy())
    if wl:
        w = scan_universe(tuple(wl), "1d", "2y", "trading days")
        if not w.empty:
            w = w.set_index("Ticker").reindex([t for t in wl if t in set(w["Ticker"])]).reset_index()
            table(w, "db_wl", ["Name", "Ticker", "Price", "Change %", "Volume", "RVOL", "Signal", "Score",
                               "Trend (ADX)"] if not C.mobile else ["Name", "Price", "Change %", "Signal"])
        missing = [t for t in wl if w.empty or t not in set(w["Ticker"])]
        if missing:
            st.caption("No data for: " + ", ".join(display_name(t) for t in missing))
    else:
        st.info("Your watchlist is empty. Add a stock below.")
    a1, a2, a3, a4 = st.columns([2, 0.7, 2, 0.7]) if not C.mobile else (st, st, st, st)
    a1.selectbox("Add a stock", C.all_symbols(), index=None, key="db_wl_add", placeholder="Type a name or code…",
                 format_func=lambda t: f"{display_name(t)} · {t}", label_visibility="collapsed")
    a2.button("➕ Add", on_click=_wl_add, key="db_wl_addb", width="stretch")
    a3.multiselect("Remove", wl, key="db_wl_rm", placeholder="Pick stocks to remove…",
                   format_func=display_name, label_visibility="collapsed")
    a4.button("🗑️ Remove", on_click=_wl_remove, key="db_wl_rmb", width="stretch")


def opportunities(scan, mob):
    if scan.empty:
        st.info("No scan results yet: pick a market in the sidebar.")
        return
    base = ["Name", "Price", "Change %", "Signal", "Score"]
    tabs = st.tabs(["🟢 Bullish", "🔴 Bearish", "🚀 Breakouts", "⚡ Momentum", "🛟 Near support", "🧱 Near resistance"])
    sets = [
        (scan[scan["_label"].isin(BUY)].sort_values("Score", ascending=False), base + ["RSI", "ADX"],
         "Highest combined technical score (trend, momentum, volume, levels). Strong buy = score ≥ 40."),
        (scan[scan["_label"].isin(SELL)].sort_values("Score"), base + ["RSI", "ADX"],
         "Lowest scores: downtrends, weak momentum, breakdowns. Avoid buying until they recover."),
        (scan[scan["Breakout"]].sort_values("RVOL", ascending=False), base + ["RVOL"],
         "Closed above the highest high of the previous 20 candles. RVOL ≥ 1.5× means the move had real volume "
         "behind it; without volume many breakouts fail."),
        (scan[(scan["ADX"].fillna(0) >= 25) & (scan["Above EMA50"] == True) & (scan["MACD hist"] > 0)]  # noqa: E712
         .sort_values("Return 20 %", ascending=False), base + ["Return 20 %", "ADX"],
         "Strong trend (ADX ≥ 25), above the 50-day EMA and MACD rising, ranked by 20-day return."),
        (scan[(scan["To support %"] >= 0) & (scan["To support %"] <= 3)].sort_values("To support %"),
         base + ["To support %"], "Within 3% above a support level: a pullback entry area IF the support holds. "
                                  "A close below support cancels the idea."),
        (scan[(scan["To resistance %"] >= 0) & (scan["To resistance %"] <= 3)].sort_values("To resistance %"),
         base + ["To resistance %"], "Within 3% below resistance: either take-profit territory or a breakout watch "
                                     "(wait for a close above it on high volume)."),
    ]
    for tab, (df, cols, note), key in zip(tabs, sets, ["bull", "bear", "brk", "mom", "sup", "res"]):
        with tab:
            table(df.head(12), f"db_op_{key}", ["Name", "Price", "Change %", "Signal"] if mob else cols)
            st.caption(f"{note} {len(df)} found. These are candidates to research, not buy/sell orders.")


CSS = """<style>
.idx-grid{{display:grid;grid-template-columns:repeat(8,minmax(0,1fr));gap:10px;margin-bottom:14px}}
.newsbox .news:first-child{{padding-top:2px}}
.idx{{background:{card};border:1px solid {border};border-radius:12px;padding:11px 13px 7px}}
.idx .n{{font-size:10.5px;color:{muted};text-transform:uppercase;letter-spacing:.1em;white-space:nowrap;
  overflow:hidden;text-overflow:ellipsis;margin-bottom:5px}}
.idx .v{{font-size:18px;font-weight:700}}
.idx .c{{font-size:12.5px;font-weight:600}}
.spark{{width:100%;height:26px;display:block;margin-top:3px}}
.bar{{height:6px;background:{card2};border-radius:4px;overflow:hidden;margin-top:5px}}
.bar.big{{height:9px;margin-top:10px}}
.bar i{{display:block;height:100%;border-radius:4px}}
.pc{{margin-bottom:9px}}
.card.divider{{margin-top:10px !important;padding-top:10px !important;border-top:1px solid {border} !important}}
.pl{{display:flex;justify-content:space-between;gap:10px;font-size:13px}}
.pl span{{color:{muted}}}
.ps{{font-size:11.5px;color:{muted};margin-top:2px}}
@media (max-width:1200px){{.idx-grid{{grid-template-columns:repeat(4,minmax(0,1fr))}}}}
@media (max-width:768px){{.idx-grid{{grid-template-columns:repeat(2,minmax(0,1fr)) !important;gap:6px}}
  .idx{{padding:7px 9px 4px}} .idx .v{{font-size:15px}} .idx .c{{font-size:12px}} .spark{{height:18px}}}}
</style>"""


def inject():
    st.markdown(CSS.format(**P), unsafe_allow_html=True)
