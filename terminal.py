"""Trading Terminal: a three-panel workspace.

Left:   searchable watchlist with categories and ⭐ favourites (saved between visits).
Centre: large chart with timeframes, chart types, overlays, indicator panes below and drawing tools.
Right:  overview, technical read (daily), support/resistance, bullish & bearish scenarios, fundamentals.
Layout and indicator visibility are adjustable. All numbers come from Yahoo Finance; missing data shows N/A.
"""
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import indicators as ind
import prefs
import ui
from plotly.subplots import make_subplots

DEMO = os.environ.get("DEMO_DATA") == "1"

INDICES = {"^KLSE": "FBM KLCI", "^GSPC": "S&P 500", "^IXIC": "Nasdaq", "^DJI": "Dow Jones", "^VIX": "VIX",
           "DX-Y.NYB": "Dollar index"}
FUTURES = {"GC=F": "Gold", "CL=F": "Crude oil", "MYR=X": "USD/MYR", "BTC-USD": "Bitcoin"}
DEFAULT_STOCKS = prefs.DEFAULT_WATCH

RANGES = {"1D": ("5m", "1d"), "5D": ("15m", "5d"), "1M": ("1h", "1mo"), "3M": ("1d", "3mo"),
          "6M": ("1d", "6mo"), "YTD": ("1d", "ytd"), "1Y": ("1d", "1y"), "5Y": ("1wk", "5y"), "All": ("1mo", "max")}


def _demo_series(t, n=400):
    rng = np.random.default_rng(abs(hash(t)) % 2**32)
    c = 10 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n)))
    o = np.r_[c[0], c[:-1]] * (1 + rng.normal(0, .003, n))
    idx = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=n)
    return pd.DataFrame({"Open": o, "High": np.maximum(o, c) * 1.008, "Low": np.minimum(o, c) * 0.992,
                         "Close": c, "Volume": rng.integers(1e5, 8e6, n).astype(float)}, index=idx)


@st.cache_data(ttl=60, show_spinner=False)
def load_quotes(tickers: tuple) -> pd.DataFrame:
    """Last price, change and change % from the last two daily candles."""
    rows = []
    if DEMO:
        from core import load_one
        for t in tickers:
            d = load_one(t, "1d", "2y")
            rows.append({"t": t, "last": d.Close.iloc[-1], "prev": d.Close.iloc[-2]})
    else:
        import yfinance as yf
        try:
            raw = yf.download(list(tickers), period="5d", interval="1d", group_by="ticker", auto_adjust=False,
                              progress=False, threads=True)
        except Exception:
            raw = pd.DataFrame()
        for t in tickers:
            try:
                d = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
                c = d["Close"].dropna()
                rows.append({"t": t, "last": float(c.iloc[-1]), "prev": float(c.iloc[-2]) if len(c) > 1 else np.nan})
            except Exception:
                rows.append({"t": t, "last": np.nan, "prev": np.nan})
    q = pd.DataFrame(rows)
    q["chg"] = q["last"] - q["prev"]
    q["pct"] = q["chg"] / q["prev"] * 100
    return q.set_index("t")


@st.cache_data(ttl=3600, show_spinner=False)
def load_info(t: str) -> dict:
    if DEMO:
        return {"longName": f"Demo {t}", "sector": "Consumer Defensive", "industry": "Confectioners",
                "fullExchangeName": "KLSE" if t.endswith(".KL") else "NasdaqGS", "currency": "MYR" if t.endswith(".KL") else "USD",
                "marketCap": 3.78e9, "trailingPE": 8.2, "trailingEps": 0.16, "beta": 0.9, "dividendRate": 0.02,
                "fiftyTwoWeekHigh": 1.49, "fiftyTwoWeekLow": 0.66, "sharesOutstanding": 2.8e9,
                "_earnings": (pd.Timestamp.now() + pd.Timedelta(days=14)).date(),
                "longBusinessSummary": "Demo company used for testing the layout."}
    import yfinance as yf
    out = {}
    try:
        tk = yf.Ticker(t)
        out = dict(tk.info or {})
        try:
            cal = tk.calendar or {}
            ed = cal.get("Earnings Date")
            out["_earnings"] = ed[0] if isinstance(ed, (list, tuple)) and ed else None
        except Exception:
            out["_earnings"] = None
    except Exception:
        pass
    return out


def _num(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else v


def _big(v, cur=""):
    v = _num(v)
    if v is None:
        return "N/A"
    for div, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(v) >= div:
            return f"{cur}{v / div:.2f}{suf}"
    return f"{cur}{v:,.2f}"


def _fmt_px(v):
    v = _num(v)
    if v is None:
        return "N/A"
    return f"{v:,.3f}" if abs(v) < 10 else f"{v:,.2f}"


@st.cache_data(ttl=60, show_spinner=False)
def load_range(sym: str, iv: str, per: str) -> pd.DataFrame:
    if DEMO:
        from core import load_one
        d = load_one(sym, "1d", "2y")
        return d.tail({"1d": 1, "5d": 5, "1mo": 22, "3mo": 63, "6mo": 126, "1y": 252, "ytd": 220}.get(per, 400)
                      * (78 if iv == "5m" else 26 if iv == "15m" else 7 if iv == "1h" else 1)) if per != "max" else d
    import yfinance as yf
    try:
        d = yf.download(sym, period=per, interval=iv, auto_adjust=False, progress=False, threads=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        d = d[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
        d["Volume"] = d["Volume"].fillna(0)
        if d.index.tz is not None:
            d.index = d.index.tz_convert("Asia/Kuala_Lumpur" if sym.endswith(".KL") or sym == "^KLSE"
                                         else "America/New_York").tz_localize(None)
        return d
    except Exception:
        return pd.DataFrame()


# ============================================================================
# Chart
# ============================================================================
OVERLAYS = ["EMA 9", "EMA 20", "EMA 50", "EMA 100", "EMA 200", "SMA 50", "SMA 200", "Bollinger (20, 2)",
            "VWAP (intraday)", "Support / resistance"]
DEFAULT_OVERLAYS = ["EMA 9", "EMA 20", "EMA 50", "EMA 200"]
PANES = ["Volume", "RSI", "MACD", "ADX / DMI", "Stochastic"]
OV_COL = {"EMA 9": "#e88ad1", "EMA 20": "ema20", "EMA 50": "ema50", "EMA 100": "#7fd1c7", "EMA 200": "ema200",
          "SMA 50": "#c9c9c9", "SMA 200": "#e3a34b"}
DRAW_TOOLS = ["drawline", "drawopenpath", "drawrect", "drawcircle", "eraseshape"]


def _col(P, k):
    return P.get(OV_COL[k], OV_COL[k])


def build_chart(d, name, P, rgba, ctype, overlays, panes, levels, intraday, mobile, height):
    from core import add_indicators
    x = add_indicators(d) if len(d) >= 3 else d.copy()
    px = ind.heikin_ashi(d) if ctype == "Heikin-Ashi" else d
    labels = [ts.strftime("%d %b %H:%M" if intraday else "%d %b %y") for ts in x.index]
    panes = [p for p in PANES if p in panes]
    rows = 1 + len(panes)
    sub_h = 0.17 if rows > 1 else 0
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.025,
                        row_heights=[1 - sub_h * (rows - 1)] + [sub_h] * (rows - 1))
    c = x["Close"]
    if ctype == "Line":
        fig.add_trace(go.Scatter(x=labels, y=c, name=name, mode="lines", line=dict(color=P["accent"], width=1.8)), 1, 1)
    else:
        fig.add_trace(go.Candlestick(x=labels, open=px["Open"], high=px["High"], low=px["Low"], close=px["Close"],
                                     name=name, increasing=dict(line=dict(color=P["up"], width=1), fillcolor=P["up"]),
                                     decreasing=dict(line=dict(color=P["down"], width=1), fillcolor=P["down"])), 1, 1)
    for k in overlays:
        if k.startswith("EMA") or k.startswith("SMA"):
            n = int(k.split()[1])
            y = ind.ema(c, n) if k.startswith("EMA") else ind.sma(c, n)
            if y.notna().any():
                fig.add_trace(go.Scatter(x=labels, y=y, name=k, mode="lines", line=dict(color=_col(P, k), width=1.2),
                                         hovertemplate=f"{k} %{{y:,.3f}}<extra></extra>"), 1, 1)
        elif k.startswith("Bollinger"):
            mid, up, lo = ind.bollinger(c)
            if mid.notna().any():
                fig.add_trace(go.Scatter(x=labels, y=up, name="BB upper", line=dict(color=rgba(P["accent"], .6), width=1),
                                         hoverinfo="skip"), 1, 1)
                fig.add_trace(go.Scatter(x=labels, y=lo, name="BB lower", line=dict(color=rgba(P["accent"], .6), width=1),
                                         fill="tonexty", fillcolor=rgba(P["accent"], .06), hoverinfo="skip"), 1, 1)
                fig.add_trace(go.Scatter(x=labels, y=mid, name="BB mid", line=dict(color=rgba(P["accent"], .5), width=1,
                                                                                    dash="dot"), hoverinfo="skip"), 1, 1)
        elif k.startswith("VWAP") and intraday and d["Volume"].sum() > 0:
            fig.add_trace(go.Scatter(x=labels, y=ind.session_vwap(d), name="VWAP",
                                     line=dict(color=P["warn"], width=1.4, dash="dash")), 1, 1)
    if "Support / resistance" in overlays and levels:
        lo_, hi_ = float(x["Low"].min()), float(x["High"].max())
        span = hi_ - lo_
        for lvl, kind in levels:
            if lo_ - span * 0.15 <= lvl <= hi_ + span * 0.15:
                col = P["up"] if kind == "S" else P["down"]
                fig.add_hline(y=lvl, line=dict(color=rgba(col, .75), width=1, dash="dot"), row=1, col=1,
                              annotation_text=f"{kind} {lvl:,.3f}", annotation_position="left",
                              annotation_font=dict(color=col, size=10))
    last = float(c.iloc[-1])
    lc = P["up"] if last >= float(x["Open"].iloc[-1]) else P["down"]
    fig.add_hline(y=last, line=dict(color=lc, width=1, dash="dot"), row=1, col=1,
                  annotation_text=f" {_fmt_px(last)} ", annotation_position="right",
                  annotation_font=dict(color="#fff", size=11), annotation_bgcolor=lc)
    for i, p in enumerate(panes, start=2):
        if p == "Volume":
            up = (x["Close"] >= x["Open"]).values
            fig.add_trace(go.Bar(x=labels, y=x["Volume"], name="Volume", showlegend=False,
                                 marker_color=[rgba(P["up"], .55) if u else rgba(P["down"], .55) for u in up]), i, 1)
            fig.add_trace(go.Scatter(x=labels, y=x["Volume"].rolling(20).mean(), name="Vol avg 20", showlegend=False,
                                     line=dict(color=P["accent"], width=1)), i, 1)
        elif p == "RSI":
            fig.add_trace(go.Scatter(x=labels, y=x["RSI"], name="RSI 14", showlegend=False,
                                     line=dict(color=P["accent"], width=1.3)), i, 1)
            for lv in (30, 70):
                fig.add_hline(y=lv, line=dict(color=P["muted"], width=1, dash="dot"), row=i, col=1)
            fig.update_yaxes(range=[0, 100], row=i, col=1)
        elif p == "MACD":
            h = x["MACDhist"]
            fig.add_trace(go.Bar(x=labels, y=h, name="MACD hist", showlegend=False,
                                 marker_color=[rgba(P["up"], .6) if v >= 0 else rgba(P["down"], .6) for v in h.fillna(0)]),
                          i, 1)
            fig.add_trace(go.Scatter(x=labels, y=x["MACD"], name="MACD", showlegend=False,
                                     line=dict(color=P["ema50"], width=1.2)), i, 1)
            fig.add_trace(go.Scatter(x=labels, y=x["MACDsig"], name="Signal", showlegend=False,
                                     line=dict(color=P["warn"], width=1.2)), i, 1)
        elif p == "ADX / DMI":
            fig.add_trace(go.Scatter(x=labels, y=x["ADX"], name="ADX", showlegend=False,
                                     line=dict(color=P["text"], width=1.5)), i, 1)
            fig.add_trace(go.Scatter(x=labels, y=x["PDI"], name="+DI", showlegend=False,
                                     line=dict(color=P["up"], width=1)), i, 1)
            fig.add_trace(go.Scatter(x=labels, y=x["MDI"], name="−DI", showlegend=False,
                                     line=dict(color=P["down"], width=1)), i, 1)
            fig.add_hline(y=25, line=dict(color=P["muted"], width=1, dash="dot"), row=i, col=1)
        elif p == "Stochastic":
            k, dd = ind.stochastic(x["High"], x["Low"], x["Close"])
            fig.add_trace(go.Scatter(x=labels, y=k, name="%K", showlegend=False,
                                     line=dict(color=P["accent"], width=1.2)), i, 1)
            fig.add_trace(go.Scatter(x=labels, y=dd, name="%D", showlegend=False,
                                     line=dict(color=P["warn"], width=1.2)), i, 1)
            for lv in (20, 80):
                fig.add_hline(y=lv, line=dict(color=P["muted"], width=1, dash="dot"), row=i, col=1)
            fig.update_yaxes(range=[0, 100], row=i, col=1)
        fig.update_yaxes(title_text=p.split()[0], title_font=dict(color=P["muted"], size=10), row=i, col=1)
    fig.update_layout(template="plotly_dark", height=height + 95 * len(panes),
                      margin=dict(l=2, r=56, t=6, b=4) if mobile else dict(l=6, r=70, t=8, b=6),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis_rangeslider_visible=False,
                      hovermode="x unified", dragmode="pan", showlegend=not mobile and bool(overlays),
                      legend=dict(orientation="h", y=1.02, x=0, font=dict(size=10.5, color=P["muted"])),
                      newshape=dict(line=dict(color=P["warn"], width=2)),
                      font=dict(family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif",
                                color=P["text"], size=11))
    fig.update_xaxes(type="category", nticks=4 if mobile else 9, tickangle=0, showgrid=False,
                     tickfont=dict(color=P["muted"], size=10 if mobile else 11))
    fig.update_yaxes(gridcolor=P["grid"], side="right", tickfont=dict(color=P["muted"]))
    return fig


# ============================================================================
# Watchlist (left)
# ============================================================================
CATS = ["⭐ Favourites", "📋 My watchlist", "📈 Indices", "🛢️ FX & commodities", "🇲🇾 Bursa top 30", "🇺🇸 US"]


def _open(sym):
    st.session_state["term_sym"] = sym


def _select_cb(key, tickers):
    def cb():
        sel = st.session_state.get(key)
        try:
            rows = sel["selection"]["rows"]
        except Exception:
            rows = []
        if rows and rows[0] < len(tickers):
            st.session_state["term_sym"] = tickers[rows[0]]
    return cb


def _toggle_fav(sym):
    fav = st.session_state.setdefault("term_fav", [])
    wl = st.session_state.setdefault("term_stocks", DEFAULT_STOCKS.copy())
    if sym in fav:
        fav.remove(sym)
    else:
        fav.append(sym)
        if sym not in wl:
            wl.append(sym)


def _toggle_watch(sym):
    wl = st.session_state.setdefault("term_stocks", DEFAULT_STOCKS.copy())
    if sym in wl:
        wl.remove(sym)
        fav = st.session_state.get("term_fav", [])
        if sym in fav:
            fav.remove(sym)
    else:
        wl.append(sym)


def _jump():
    t = st.session_state.get("term_open")
    if t:
        st.session_state["term_sym"] = t
        st.session_state["term_open"] = None


def watch_panel(C, name_of, P, sym):
    from core import BURSA, US
    wl = st.session_state.setdefault("term_stocks", DEFAULT_STOCKS.copy())
    fav = st.session_state.setdefault("term_fav", [])
    opts = list(dict.fromkeys(list(INDICES) + list(FUTURES) + C.all_symbols()))
    st.selectbox("Open any symbol", opts, index=None, key="term_open", on_change=_jump,
                 placeholder="🔍 Search any stock, index, FX…", format_func=lambda t: f"{name_of(t)} · {t}",
                 label_visibility="collapsed")
    cat = st.selectbox("Category", CATS, index=1, key="term_cat", label_visibility="collapsed")
    lists = {"⭐ Favourites": [t for t in wl if t in fav] + [t for t in fav if t not in wl],
             "📋 My watchlist": [t for t in wl if t in fav] + [t for t in wl if t not in fav],
             "📈 Indices": list(INDICES), "🛢️ FX & commodities": list(FUTURES),
             "🇲🇾 Bursa top 30": list(BURSA), "🇺🇸 US": list(US)}
    tick = lists[cat]
    flt = st.text_input("Filter", placeholder="Filter this list…", key="term_filter", label_visibility="collapsed")
    if flt.strip():
        q_ = flt.strip().lower()
        tick = [t for t in tick if q_ in t.lower() or q_ in name_of(t).lower()]
    if not tick:
        st.caption("Nothing here yet. Use ☆ in the chart header to add favourites."
                   if cat.startswith("⭐") else "No symbols match.")
        return
    quotes = load_quotes(tuple(tick))
    rows, pcts = [], []
    for t in tick:
        q = quotes.loc[t] if t in quotes.index else None
        last = q["last"] if q is not None else np.nan
        pct = q["pct"] if q is not None else np.nan
        pcts.append(pct)
        rows.append({"Symbol": ("★ " if t in fav else "") + name_of(t),
                     "Last · Chg": "N/A" if pd.isna(last) else
                     f"{_fmt_px(last)}  {'' if pd.isna(pct) else f'{pct:+.2f}%'}"})
    df = pd.DataFrame(rows)
    colours = [P["muted"] if pd.isna(v) else P["up"] if v > 0 else P["down"] if v < 0 else P["muted"] for v in pcts]
    sty = df.style.apply(lambda col: [f"color:{c}" for c in colours], subset=["Last · Chg"])
    key = f"wl_{CATS.index(cat)}"
    st.dataframe(sty, hide_index=True, width="stretch", key=key, on_select=_select_cb(key, tick),
                 selection_mode="single-row", height="content" if len(df) <= 14 else 500,
                 column_config={"Symbol": st.column_config.TextColumn(width="small"),
                                "Last · Chg": st.column_config.TextColumn(width="small")})
    st.caption(f"{len(tick)} symbols · click a row to chart it. Watchlist & ★ favourites are saved.")


# ============================================================================
# Overview & technicals (right)
# ============================================================================
def _kv(rows):
    return "<table class='kv'>" + "".join(f"<tr><td>{a}</td><td>{b}</td></tr>" for a, b in rows) + "</table>"


def overview_panel(sym, name_of, P, pill, market_open, mobile):
    from core import LABEL_COLOR, regime_color, stock_view
    info = load_info(sym)
    q = load_quotes((sym,))
    q = q.loc[sym] if sym in q.index else None
    last_p = q["last"] if q is not None else np.nan
    is_idx = sym.startswith("^") or sym in FUTURES
    cur = info.get("currency") or ("MYR" if sym.endswith(".KL") else "")
    cc = P["up"] if (q is not None and q["chg"] == q["chg"] and q["chg"] >= 0) else P["down"]
    is_open = market_open("X.KL" if sym == "^KLSE" else sym)
    sector = " · ".join(x for x in (info.get("sector"), info.get("industry")) if x) or ("Index / market" if is_idx else "N/A")
    chg_txt = "" if q is None or pd.isna(q["chg"]) else f"{q['chg']:+,.3f} ({q['pct']:+.2f}%)"

    with ui.panel("Overview", "Yahoo Finance"):
        st.markdown(f"""<div style="font-weight:700;font-size:15px">{name_of(sym)} <span class="muted">· {sym}</span></div>
<div class="muted" style="font-size:12px">{info.get('longName') or info.get('shortName') or ''} {('· ' + sector) if sector else ''}</div>
<div style="margin-top:8px"><span class="mono" style="font-size:26px;font-weight:700">{_fmt_px(last_p)}</span>
 <span class="muted" style="font-size:12px">{cur}</span>
 <span class="mono" style="color:{cc};font-weight:600;margin-left:6px">{chg_txt}</span></div>
<div style="font-size:12px;margin-top:2px;color:{P['up'] if is_open else P['muted']}">● Market {'open' if is_open else 'closed'}
{' · Bursa prices ~15 min delayed' if sym.endswith('.KL') or sym == '^KLSE' else ''}</div>""", unsafe_allow_html=True)

    v = stock_view(sym, "1D")
    with ui.panel("Technical read", "daily candles"):
        if v is None:
            st.markdown("<div class='muted'>N/A: not enough daily history (need 60+ candles).</div>",
                        unsafe_allow_html=True)
        else:
            d, plan, ax = v["d"], v["plan"], v["ax"]
            last = d.iloc[-1]
            price = float(last["Close"])
            e20, e50, e200 = (float(last[k]) if pd.notna(last[k]) else None for k in ("EMA20", "EMA50", "EMA200"))
            if None in (e20, e50, e200) or len(d) < 200:
                align = "N/A (needs 200 candles)"
            elif price > e20 > e50 > e200:
                align = f"<span style='color:{P['up']}'>Bullish stack</span> (price > 20 > 50 > 200)"
            elif price < e20 < e50 < e200:
                align = f"<span style='color:{P['down']}'>Bearish stack</span> (price < 20 < 50 < 200)"
            else:
                above = sum(price > e for e in (e20, e50, e200))
                align = f"Mixed: price above {above} of 3 EMAs"
            macd_state = "Rising (above signal)" if last["MACD"] > last["MACDsig"] else "Falling (below signal)"
            rsi = float(last["RSI"])
            rsi_txt = f"{rsi:.0f} " + ("(overbought)" if rsi >= 70 else "(oversold)" if rsi <= 30 else "")
            atrp = float(last["ATR"]) / price * 100 if price else np.nan
            st.markdown(f"""<div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
{pill(v['label'], LABEL_COLOR[v['label']])}<span class="mono" style="font-weight:700">{v['sc']:+d}</span><span class="muted">/ 100</span>
{pill(ax['regime'], regime_color(ax['regime'])) if ax else ''}</div>
<div style="margin-top:8px">{_kv([("EMA alignment", align), ("RSI (14)", rsi_txt), ("MACD (12,26,9)", macd_state),
     ("ADX (14)", "N/A" if not ax else f"{ax['adx']:.0f}"), ("ATR (14)", "N/A" if atrp != atrp else f"{atrp:.2f}% of price"),
     ("RVOL", "N/A" if pd.isna(last['RVOL']) else f"{last['RVOL']:.1f}×")])}</div>""", unsafe_allow_html=True)
            sup = [f"{lv:,.3f}" for lv, _ in v["sup"][:2]]
            res = [f"{lv:,.3f}" for lv, _ in v["res"][:2]]
            st.markdown(f"""<div style="margin-top:10px">{_kv([
                ("Resistance", f"<span style='color:{P['down']}'>{' · '.join(res)}</span>"),
                ("Support", f"<span style='color:{P['up']}'>{' · '.join(sup)}</span>"),
                ("ATR stop reference", f"{plan['stop']:,.3f}")])}</div>""", unsafe_allow_html=True)
            r1, r2, s1, s2 = plan["t1"], plan["t2"], plan["s1"], plan["s2"]
            st.markdown(f"""<div class="scen up"><b>Bullish scenario</b><br>Holds above support {s1:,.3f} and closes
above resistance {r1:,.3f}, ideally on volume ≥ 1.5× average → next resistance {r2:,.3f}.</div>
<div class="scen dn"><b>Bearish scenario</b><br>Closes below support {s1:,.3f} → next support {s2:,.3f}.
{f'Losing the 50-day EMA ({e50:,.3f}) would weaken the trend further.' if e50 else ''}</div>
<div class="note">Scenarios are conditions to watch, not predictions. The plan's suggested action: {plan['action']}.</div>""",
                        unsafe_allow_html=True)

    if not is_idx:
        with ui.panel("Fundamentals", "N/A = not reported by Yahoo"):
            def pct(k):
                v_ = _num(info.get(k))
                return "N/A" if v_ is None else f"{v_ * 100:.1f}%"

            def num(k, f="{:.2f}"):
                v_ = _num(info.get(k))
                return "N/A" if v_ is None else f.format(v_)
            dy = None
            if _num(info.get("dividendRate")) and _num(last_p):
                dy = info["dividendRate"] / last_p * 100
            elif _num(info.get("trailingAnnualDividendYield")) is not None:
                dy = info["trailingAnnualDividendYield"] * 100
            ed = info.get("_earnings")
            earn = "N/A" if ed is None else pd.Timestamp(ed).strftime("%d %b %Y")
            hi52, lo52 = _num(info.get("fiftyTwoWeekHigh")), _num(info.get("fiftyTwoWeekLow"))
            st.markdown(_kv([
                ("Market cap", _big(info.get("marketCap"))), ("P/E (TTM)", num("trailingPE")),
                ("Forward P/E", num("forwardPE")), ("Price / book", num("priceToBook")),
                ("ROE", pct("returnOnEquity")), ("Net margin", pct("profitMargins")),
                ("Debt / equity", num("debtToEquity", "{:.0f}%")), ("Dividend yield", "N/A" if dy is None else f"{dy:.2f}%"),
                ("52-week range", "N/A" if not (hi52 and lo52) else f"{_fmt_px(lo52)} – {_fmt_px(hi52)}"),
                ("Next earnings", earn), ("Beta", num("beta"))]), unsafe_allow_html=True)
            summ = info.get("longBusinessSummary") or ""
            if summ:
                with st.expander("About the company"):
                    st.write(summ)
    b1, b2 = st.columns(2)
    b1.button("📊 Full analysis", width="stretch", key="term_full", on_click=ui.goto, args=("analysis",),
              kwargs={"an_typed": sym}, disabled=is_idx)
    b2.button("💼 Paper trade", width="stretch", key="term_trade", on_click=ui.goto, args=("paper",),
              kwargs={"pt_symbol": sym, "pt_other": ""}, disabled=is_idx)


# ============================================================================
# Page
# ============================================================================
LAYOUTS = ["3 panels", "Chart + overview", "Chart only"]
HEIGHTS = {"Compact": 430, "Normal": 560, "Tall": 720}


def render(C):
    from core import P, display_name, normalise, pill, rgba, tv_symbol, tradingview_widget
    from ta_plus import market_open
    st.markdown(CSS.format(**P), unsafe_allow_html=True)
    wl = st.session_state.setdefault("term_stocks", DEFAULT_STOCKS.copy())
    if "term_sym" not in st.session_state:
        st.session_state["term_sym"] = st.session_state.get("an_ticker") or (wl[0] if wl else "^KLSE")
    sym = st.session_state["term_sym"]
    name_of = lambda t: INDICES.get(t) or FUTURES.get(t) or display_name(t)  # noqa: E731
    mobile = C.mobile

    # ---- workspace settings
    with st.expander("🧰 Workspace & indicators", expanded=False):
        w1, w2, w3, w4 = st.columns([1.2, 1, 2.4, 2]) if not mobile else (st, st, st, st)
        layout = w1.segmented_control("Layout", LAYOUTS, default="3 panels", key="term_layout") or "3 panels"
        hsel = w2.segmented_control("Chart height", list(HEIGHTS), default="Normal", key="term_h") or "Normal"
        overlays = w3.multiselect("Overlays on price", OVERLAYS, default=DEFAULT_OVERLAYS, key="term_ov")
        panes = w4.multiselect("Panels under the chart", PANES,
                               default=["Volume"] if mobile else ["Volume", "RSI", "MACD"], key="term_pn")
        st.caption("Defaults: EMA 9/20/50/200 with Volume, RSI and MACD. Drawing tools (line, path, box, circle, "
                   "eraser) are in the chart's top-right toolbar; drawings are temporary and clear when the chart "
                   "reloads. VWAP shows on intraday ranges (1D/5D/1M). On a smaller screen, close the left sidebar («) "
                   "or pick 'Chart + overview' for more room.")
    if mobile:
        layout = "Chart + overview"

    if layout == "3 panels":
        left, centre, right = st.columns([1.1, 2.7, 1.25], gap="small")
    elif layout == "Chart + overview":
        left = None
        centre, right = (st.container(), st.container()) if mobile else st.columns([3, 1.2], gap="small")
    else:
        left = right = None
        centre = st.container()

    # ---- centre: chart
    with centre, ui.panel("Chart", "drag to pan · scroll to zoom"):
        is_fav = sym in st.session_state.get("term_fav", [])
        in_wl = sym in wl
        hdr = st.container(horizontal=True, vertical_alignment="center", gap="small", key="term_hdr")
        with hdr:
            typed = st.text_input("Symbol", placeholder=f"{name_of(sym)} · type a code to switch…",
                                  key="term_search", label_visibility="collapsed", width="stretch")
            if typed.strip():
                t_ = normalise(typed)
                if t_ and t_ != sym:
                    st.session_state["term_sym"] = sym = t_
                    is_fav, in_wl = sym in st.session_state.get("term_fav", []), sym in wl
            st.button("★" if is_fav else "☆", key="term_favb", on_click=_toggle_fav, args=(sym,), width="content",
                      help="Remove from favourites" if is_fav else "Add to favourites")
            st.button("✓ Listed" if in_wl else "+ List", key="term_wlb", on_click=_toggle_watch, args=(sym,),
                      width="content", help="Remove from watchlist" if in_wl else "Add to watchlist")
            engine = "Built-in" if mobile else (st.segmented_control(
                "Engine", ["Built-in", "TradingView"], default="Built-in", key="term_engine",
                label_visibility="collapsed") or "Built-in")
        with st.container(horizontal=True, gap="small", key="term_ctl"):
            rng_key = st.segmented_control("Range", list(RANGES), default="6M", key="term_rng",
                                           label_visibility="collapsed") or "6M"
            ctype = st.segmented_control("Type", ["Candles", "Heikin-Ashi", "Line"], default="Candles",
                                         key="term_type", format_func=lambda v: {"Heikin-Ashi": "H-Ashi"}.get(v, v),
                                         label_visibility="collapsed", help="H-Ashi = Heikin-Ashi smoothed candles") \
                or "Candles"
        iv, per = RANGES[rng_key]
        intraday = iv in ("5m", "15m", "1h")
        if engine == "TradingView":
            tradingview_widget(tv_symbol(sym) if not sym.startswith("^") else sym.replace("^", ""),
                               height=HEIGHTS[hsel] + 160, interval="D")
            st.caption("TradingView widget (its own drawing tools and indicators). Some Bursa symbols can't be embedded; "
                       "switch to Built-in if it says so.")
        else:
            d = load_range(sym, iv, per)
            if d is None or d.empty or len(d) < 2:
                st.warning(f"No price data for {sym} on {rng_key}. Try another range or check the symbol.")
            else:
                lr, pr = d.iloc[-1], d.iloc[-2]
                ch = float(lr["Close"] - pr["Close"])
                pc = ch / float(pr["Close"]) * 100 if float(pr["Close"]) else 0
                cc = P["up"] if ch >= 0 else P["down"]
                st.markdown(
                    f"<div class='ohlc'><b>{name_of(sym)}</b> <span class='muted'>{sym} · {rng_key} · {iv}</span>"
                    f"<span class='mono'><i>O</i>{_fmt_px(lr['Open'])} <i>H</i>{_fmt_px(lr['High'])} "
                    f"<i>L</i>{_fmt_px(lr['Low'])} <i>C</i>{_fmt_px(lr['Close'])} "
                    f"<span style='color:{cc}'>{ch:+,.3f} ({pc:+.2f}%)</span> <i>Vol</i>{_big(lr['Volume'])}</span></div>",
                    unsafe_allow_html=True)
                levels = []
                if "Support / resistance" in overlays:
                    from core import stock_view
                    v = stock_view(sym, "1D")
                    if v is not None:
                        levels = [(lv, "S") for lv, _ in v["sup"][:3]] + [(lv, "R") for lv, _ in v["res"][:3]]
                fig = build_chart(d, name_of(sym), P, rgba, ctype, overlays, panes, levels, intraday, mobile,
                                  HEIGHTS[hsel] if not mobile else 360)
                st.plotly_chart(fig, width="stretch", key="term_chart",
                                config={"displaylogo": False, "scrollZoom": True,
                                        "modeBarButtonsToAdd": DRAW_TOOLS,
                                        "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]})
                st.caption(f"Last candle {d.index[-1]:%d %b %Y %H:%M} · {len(d)} candles · Yahoo Finance"
                           + (" (Bursa ~15 min delayed)" if sym.endswith(".KL") or sym == "^KLSE" else "")
                           + ("" if ctype != "Heikin-Ashi" else " · Heikin-Ashi candles are smoothed averages, "
                              "not real prices"))

    if right is not None:
        with right:
            overview_panel(sym, name_of, P, pill, market_open, mobile)
    if left is not None:
        with left, ui.panel("Watchlist", f"{len(wl)} saved"):
            watch_panel(C, name_of, P, sym)
    elif layout == "Chart + overview":
        with (st.container() if mobile else right), ui.panel("Watchlist", f"{len(wl)} saved"):
            watch_panel(C, name_of, P, sym)


CSS = """<style>
.ohlc{{font-size:12.5px;display:flex;flex-wrap:wrap;gap:4px 12px;align-items:baseline;margin:-2px 0 -4px}}
.ohlc b{{font-size:14px}}
.ohlc i{{font-style:normal;color:{muted};margin:0 3px 0 6px}}
.scen{{font-size:12.5px;line-height:1.5;padding:8px 10px;border-radius:8px;margin-top:8px;border:1px solid {border}}}
.scen.up{{border-left:3px solid {up}}}
.scen.dn{{border-left:3px solid {down}}}
.st-key-term_favb button{{font-size:17px;padding:0 12px;min-height:38px;color:{warn}}}
.st-key-term_wlb button{{font-size:12.5px;padding:0 10px;min-height:38px;white-space:nowrap}}
.st-key-term_hdr{{flex-wrap:nowrap !important}}
.st-key-term_ctl{{flex-wrap:wrap !important;row-gap:6px !important}}
.st-key-term_ctl > div{{width:auto !important;flex:0 0 auto !important}}
</style>"""
