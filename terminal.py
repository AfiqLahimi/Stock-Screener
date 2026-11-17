"""Terminal tab: TradingView-style layout.

Left: big chart of the open stock (built-in candles + volume, or the TradingView widget).
Right top: clickable watchlist (indices, my stocks, futures & FX) with Last / Chg / Chg%.
Right bottom: details of the open stock (price, volume vs 30-day average, key stats).
All numbers come from Yahoo Finance; anything missing is shown as N/A.
"""
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

DEMO = os.environ.get("DEMO_DATA") == "1"

INDICES = {"^KLSE": "FBM KLCI", "^GSPC": "S&P 500", "^IXIC": "Nasdaq", "^DJI": "Dow Jones", "^VIX": "VIX",
           "DX-Y.NYB": "Dollar index"}
FUTURES = {"GC=F": "Gold", "CL=F": "Crude oil", "MYR=X": "USD/MYR", "BTC-USD": "Bitcoin"}
DEFAULT_STOCKS = ["1155.KL", "5347.KL", "5398.KL", "1023.KL", "AAPL", "NVDA", "TSLA", "NFLX"]

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
        for t in tickers:
            d = _demo_series(t)
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


def _watch_table(tickers, names, quotes):
    rows = []
    for t in tickers:
        q = quotes.loc[t] if t in quotes.index else None
        rows.append({"Symbol": names(t), "Last": q["last"] if q is not None else np.nan,
                     "Chg": q["chg"] if q is not None else np.nan, "Chg%": q["pct"] if q is not None else np.nan})
    return pd.DataFrame(rows)


def _style(df, P):
    def col(v):
        if pd.isna(v):
            return f"color:{P['muted']}"
        return f"color:{P['up']}" if v > 0 else (f"color:{P['down']}" if v < 0 else "")
    return (df.style.map(col, subset=["Chg", "Chg%"])
            .format({"Last": lambda v: _fmt_px(v), "Chg": lambda v: "N/A" if pd.isna(v) else f"{v:+,.3f}",
                     "Chg%": lambda v: "N/A" if pd.isna(v) else f"{v:+.2f}%"}))


def _select_cb(key, tickers):
    def cb():
        sel = st.session_state.get(key)
        rows = sel.selection.rows if sel is not None else []
        if rows:
            st.session_state["term_sym"] = tickers[rows[0]]
    return cb


def _chart(d, name, P, rgba, show_ema, interval_key, mobile=False):
    x = d.copy()
    intraday = interval_key in ("1D", "5D", "1M")
    labels = [ts.strftime("%d %b %H:%M" if intraday else "%d %b %y") for ts in x.index]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.8, 0.2])
    fig.add_trace(go.Candlestick(x=labels, open=x["Open"], high=x["High"], low=x["Low"], close=x["Close"], name=name,
                                 increasing=dict(line=dict(color=P["up"], width=1), fillcolor=P["up"]),
                                 decreasing=dict(line=dict(color=P["down"], width=1), fillcolor=P["down"])), 1, 1)
    if show_ema and len(x) > 20:
        for n, col in ((20, P["ema20"]), (50, P["ema50"]), (200, P["ema200"])):
            if len(x) >= n:
                fig.add_trace(go.Scatter(x=labels, y=x["Close"].ewm(span=n, adjust=False).mean(), name=f"EMA{n}",
                                         mode="lines", line=dict(color=col, width=1.2)), 1, 1)
    up = (x["Close"] >= x["Open"]).values
    fig.add_trace(go.Bar(x=labels, y=x["Volume"], marker_color=[rgba(P["up"], .5) if u else rgba(P["down"], .5) for u in up],
                         name="Volume", showlegend=False), 2, 1)
    last = float(x["Close"].iloc[-1])
    lc = P["up"] if last >= float(x["Open"].iloc[-1]) else P["down"]
    fig.add_hline(y=last, line=dict(color=lc, width=1, dash="dot"), row=1, col=1,
                  annotation_text=f" {_fmt_px(last)} ", annotation_position="right",
                  annotation_font=dict(color="#fff", size=11), annotation_bgcolor=lc)
    fig.update_layout(template="plotly_dark", height=430 if mobile else 720,
                      margin=dict(l=2, r=56, t=6, b=4) if mobile else dict(l=6, r=70, t=10, b=6),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis_rangeslider_visible=False,
                      hovermode="x unified", showlegend=show_ema,
                      legend=dict(orientation="h", y=1.02, x=0, font=dict(size=11, color=P["muted"])),
                      font=dict(family="Inter, sans-serif", color=P["text"], size=12))
    fig.update_xaxes(type="category", nticks=4 if mobile else 10, tickangle=0, showgrid=False,
                     tickfont=dict(color=P["muted"], size=10 if mobile else 12))
    fig.update_yaxes(gridcolor=P["grid"], side="right", tickfont=dict(color=P["muted"]))
    fig.update_yaxes(title_text="Vol", row=2, col=1, title_font=dict(color=P["muted"], size=10))
    return fig


def render(display_name, normalise, P, pill, rgba, tv_symbol, tradingview_widget, market_open, universe):
    if "term_stocks" not in st.session_state:
        st.session_state["term_stocks"] = DEFAULT_STOCKS.copy()
    if "term_sym" not in st.session_state:
        st.session_state["term_sym"] = st.session_state.get("an_ticker") or st.session_state["term_stocks"][0]

    stocks = st.session_state["term_stocks"]
    name_of = lambda t: INDICES.get(t) or FUTURES.get(t) or display_name(t)  # noqa: E731
    all_t = tuple(dict.fromkeys(list(INDICES) + stocks + list(FUTURES)))
    quotes = load_quotes(all_t)
    sym = st.session_state["term_sym"]

    mobile = bool(st.session_state.get("_mobile"))
    if mobile:
        left = st.container()
        details_box = st.container()
        right = st.container()
    else:
        left, right = st.columns([2.35, 1.45], gap="small")
        details_box = right

    # ===================== RIGHT: watchlist + details
    with right:
        st.markdown(f"<div style='font-weight:700;font-size:15px;margin-bottom:4px'>Watchlist</div>",
                    unsafe_allow_html=True)
        a1, a2 = st.columns([3, 1])
        new = a1.text_input("Add symbol", placeholder="Add: GAMUDA, 5398, AMD", label_visibility="collapsed",
                            key="term_add")
        if a2.button("➕", help="Add to watchlist", width="stretch"):
            t_new = normalise(new)
            if t_new and t_new not in stocks:
                stocks.append(t_new)
                st.session_state["term_sym"] = t_new
                st.rerun()
        cfg = {"Symbol": st.column_config.TextColumn(width=110), "Last": st.column_config.TextColumn(width=72),
               "Chg": st.column_config.TextColumn(width=66), "Chg%": st.column_config.TextColumn(width=66)}
        for title, tick, key in (("INDICES", list(INDICES), "wl_idx"), ("STOCKS", stocks, "wl_stk"),
                                 ("FUTURES & FX", list(FUTURES), "wl_fut")):
            st.markdown(f"<div class='muted' style='font-size:11px;letter-spacing:.06em;margin:6px 0 2px'>⌄ {title}</div>",
                        unsafe_allow_html=True)
            tbl = _watch_table(tick, name_of, quotes)
            st.dataframe(_style(tbl, P), hide_index=True, width="stretch", height=35 * (len(tbl) + 1) + 3,
                         key=key, on_select=_select_cb(key, tick), selection_mode="single-row", column_config=cfg)
        if sym in stocks and st.button(f"🗑️ Remove {name_of(sym)} from watchlist", width="stretch"):
            stocks.remove(sym)
            st.session_state["term_sym"] = stocks[0] if stocks else "^KLSE"
            st.rerun()
        st.caption("Click a row to open it on the chart. Watchlist is kept for this browser session. "
                   "Tip: close the left settings panel («) for a wider terminal.")

    # ===================== LEFT: chart
    with left:
        c1, c3, c4 = st.columns([2.2, 0.9, 1.6])
        typed = c1.text_input("Symbol", placeholder=f"{name_of(sym)}  ·  search…", label_visibility="collapsed",
                              key="term_search")
        if typed.strip():
            t_ = normalise(typed)
            if t_ and t_ != sym:
                st.session_state["term_sym"] = t_
                sym = t_
        show_ema = c3.toggle("EMAs", value=False, key="term_ema")
        engine = c4.segmented_control("Engine", ["Built-in", "TradingView"], default="Built-in", key="term_engine",
                                      label_visibility="collapsed") or "Built-in"
        rng_key = st.segmented_control("Range", list(RANGES), default="1Y", key="term_rng",
                                       label_visibility="collapsed") or "1Y"
        iv, per = RANGES[rng_key]
        if DEMO:
            d = _demo_series(sym)
        else:
            import yfinance as yf
            try:
                d = yf.download(sym, period=per, interval=iv, auto_adjust=False, progress=False, threads=False)
                if isinstance(d.columns, pd.MultiIndex):
                    d.columns = d.columns.get_level_values(0)
                d = d[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
                d["Volume"] = d["Volume"].fillna(0)
            except Exception:
                d = pd.DataFrame()
        if engine == "TradingView":
            tradingview_widget(tv_symbol(sym) if not sym.startswith("^") else sym.replace("^", ""),
                               height=520 if mobile else 760,
                               interval="D")
            st.caption("TradingView widget. Some Bursa symbols may not be available in embedded charts: switch "
                       "to Built-in if it says so.")
        elif d is None or d.empty:
            st.warning(f"No price data for {sym} on this range. Try another range or check the symbol.")
        else:
            lr, pr = d.iloc[-1], d.iloc[-2] if len(d) > 1 else d.iloc[-1]
            ch = float(lr["Close"] - pr["Close"])
            pc = ch / float(pr["Close"]) * 100 if float(pr["Close"]) else 0
            cc = P["up"] if ch >= 0 else P["down"]
            st.markdown(
                f"<div style='font-size:13px;margin:2px 0 -6px'><b style='font-size:15px'>{name_of(sym)}</b> "
                f"<span class='muted'>· {rng_key} · {sym}</span> &nbsp; "
                f"<span class='muted'>O</span><span style='color:{cc}'>{_fmt_px(lr['Open'])}</span> "
                f"<span class='muted'>H</span><span style='color:{cc}'>{_fmt_px(lr['High'])}</span> "
                f"<span class='muted'>L</span><span style='color:{cc}'>{_fmt_px(lr['Low'])}</span> "
                f"<span class='muted'>C</span><span style='color:{cc}'>{_fmt_px(lr['Close'])}</span> "
                f"<span style='color:{cc}'>{ch:+.3f} ({pc:+.2f}%)</span> &nbsp; "
                f"<span class='muted'>Vol</span> <span style='color:{cc}'>{_big(lr['Volume'])}</span></div>",
                unsafe_allow_html=True)
            st.plotly_chart(_chart(d, name_of(sym), P, rgba, show_ema, rng_key, mobile), width="stretch",
                            config={"displaylogo": False, "scrollZoom": True})
            st.caption(f"Last candle {d.index[-1].strftime('%d %b %Y %H:%M')} · {len(d)} candles · Yahoo Finance"
                       + (" (Bursa ~15 min delayed)" if sym.endswith(".KL") or sym == "^KLSE" else ""))

    # ===================== RIGHT (continued): details of open symbol
    with details_box:
        info = load_info(sym)
        q = quotes.loc[sym] if sym in quotes.index else None
        if q is None or pd.isna(q["last"]):
            qq = load_quotes((sym,))
            q = qq.loc[sym] if sym in qq.index else None
        last_p = q["last"] if q is not None else np.nan
        cur = info.get("currency") or ("MYR" if sym.endswith(".KL") else "")
        cc = P["up"] if (q is not None and q["chg"] >= 0) else P["down"]
        mk_ticker = "X.KL" if sym == "^KLSE" else sym
        is_open = market_open(mk_ticker)
        # 30-day volume from real daily candles
        if DEMO:
            dv = _demo_series(sym)
        else:
            import yfinance as yf
            try:
                dv = yf.download(sym, period="3mo", interval="1d", auto_adjust=False, progress=False, threads=False)
                if isinstance(dv.columns, pd.MultiIndex):
                    dv.columns = dv.columns.get_level_values(0)
            except Exception:
                dv = pd.DataFrame()
        vol_now = float(dv["Volume"].iloc[-1]) if len(dv) else np.nan
        vol30 = float(dv["Volume"].tail(31).iloc[:-1].mean()) if len(dv) > 5 else np.nan
        ratio = vol_now / vol30 if vol30 and not np.isnan(vol30) and vol30 > 0 else np.nan
        bar_w = 0 if np.isnan(ratio) else min(ratio / 2, 1) * 100
        sector = " · ".join(x for x in (info.get("sector"), info.get("industry")) if x) or "N/A"
        exch = info.get("fullExchangeName") or info.get("exchange") or ""

        dy = None
        if _num(info.get("dividendRate")) and _num(last_p):
            dy = info["dividendRate"] / last_p * 100
        elif _num(info.get("trailingAnnualDividendYield")) is not None:
            dy = info["trailingAnnualDividendYield"] * 100
        ed = info.get("_earnings")
        if ed is not None:
            days = (pd.Timestamp(ed) - pd.Timestamp.now().normalize()).days
            earn = f"in {days} days ({pd.Timestamp(ed).strftime('%d %b %Y')})" if days >= 0 else \
                pd.Timestamp(ed).strftime("%d %b %Y")
        else:
            earn = "N/A"
        hi52, lo52 = _num(info.get("fiftyTwoWeekHigh")), _num(info.get("fiftyTwoWeekLow"))
        pos52 = ((last_p - lo52) / (hi52 - lo52) * 100) if (hi52 and lo52 and hi52 > lo52 and _num(last_p)) else None
        stats = [
            ("Next earnings report", earn),
            ("Market capitalization", _big(info.get("marketCap"))),
            ("Dividend yield (indicated)", "N/A" if dy is None else f"{dy:.2f}%"),
            ("Price to earnings (TTM)", "N/A" if not _num(info.get("trailingPE")) else f"{info['trailingPE']:.2f}"),
            ("EPS (TTM)", "N/A" if _num(info.get("trailingEps")) is None else f"{info['trailingEps']:.3f}"),
            ("52-week range", "N/A" if not (hi52 and lo52) else f"{_fmt_px(lo52)} – {_fmt_px(hi52)}"),
            ("Beta", "N/A" if _num(info.get("beta")) is None else f"{info['beta']:.2f}"),
            ("Shares outstanding", _big(info.get("sharesOutstanding"))),
        ]
        rows_html = "".join(f"<tr><td>{a}</td><td>{b}</td></tr>" for a, b in stats)
        pos_html = "" if pos52 is None else (
            f"<div class='muted' style='font-size:11.5px;margin-top:10px'>Position in 52-week range</div>"
            f"<div style='position:relative;height:6px;border-radius:4px;background:{P['card2']};margin-top:4px'>"
            f"<div style='position:absolute;left:calc({pos52:.0f}% - 5px);top:-3px;width:10px;height:12px;"
            f"border-radius:3px;background:{P['accent']}'></div></div>")
        summ = info.get("longBusinessSummary") or ""
        st.markdown(f"""
<div class="card" style="margin-top:8px">
<div style="font-weight:700;font-size:16px">{name_of(sym)}</div>
<div style="font-size:13px;margin-top:2px">{info.get('longName') or info.get('shortName') or sym} <span class="muted">· {exch}</span></div>
<div class="muted" style="font-size:12px">{sector}</div>
<div style="margin-top:12px"><span style="font-size:30px;font-weight:700">{_fmt_px(last_p)}</span>
 <span class="muted" style="font-size:12px">{cur}</span>
 <span style="color:{cc};font-weight:600;margin-left:6px">{'' if q is None or pd.isna(q['chg']) else f"{q['chg']:+.3f} ({q['pct']:+.2f}%)"}</span></div>
<div style="font-size:12.5px;margin-top:2px;color:{P['up'] if is_open else P['muted']}">● Market {'open' if is_open else 'closed'}</div>
<div style="display:flex;justify-content:space-between;margin-top:14px;font-size:12px" class="muted">
 <span>Volume</span><span>Avg. volume (30D)</span></div>
<div style="display:flex;justify-content:space-between;font-weight:600;font-size:13px">
 <span>{_big(vol_now)}</span><span>{_big(vol30)}</span></div>
<div style="display:flex;height:6px;border-radius:4px;overflow:hidden;margin-top:5px;background:{P['card2']}">
 <div style="width:{bar_w:.0f}%;background:{P['accent']}"></div></div>
<div class="muted" style="font-size:11.5px;margin-top:3px">{'' if np.isnan(ratio) else f'Today is {ratio:.2f}× the 30-day average (bar full = 2×)'}</div>
<div style="font-weight:700;margin-top:16px">Key stats</div>
<table class="kv" style="margin-top:4px">{rows_html}</table>{pos_html}
</div>""", unsafe_allow_html=True)
        if summ:
            with st.expander("About the company"):
                st.write(summ)
        b1, b2 = st.columns(2)
        if b1.button("📊 Full analysis", width="stretch", help="Open this symbol in Analysis & Forecast",
                     on_click=lambda: st.session_state.update(an_typed=sym), disabled=sym.startswith("^") or sym in FUTURES):
            st.toast("Done: open the 📊 Analysis & Forecast tab, the symbol is filled in.")
        if sym not in stocks and not sym.startswith("^") and sym not in FUTURES and \
                b2.button("⭐ Add to watchlist", width="stretch"):
            stocks.append(sym)
            st.rerun()
