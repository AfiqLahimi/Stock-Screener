"""Dividends and IPO tabs."""
import datetime as dt
import os
import re
import urllib.parse
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DEMO = os.environ.get("DEMO_DATA") == "1"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/130.0 Safari/537.36")


def _today():
    return pd.Timestamp.now(tz="Asia/Kuala_Lumpur").tz_localize(None).normalize()


def _cur(ticker):
    return "RM" if ticker.endswith(".KL") else "$"


def _http_get(url, json=False, timeout=15):
    """GET with a browser fingerprint (curl_cffi ships with yfinance), falling back to requests."""
    try:
        from curl_cffi import requests as creq
        r = creq.get(url, impersonate="chrome", timeout=timeout,
                     headers={"Accept": "application/json, text/plain, */*" if json else "*/*"})
    except Exception:
        import requests
        r = requests.get(url, timeout=timeout, headers={"User-Agent": UA, "Accept": "*/*"})
    r.raise_for_status()
    return r.json() if json else r.text


# ============================================================================
# DIVIDENDS
# ============================================================================
def _demo_div(ticker):
    rng = np.random.default_rng(abs(hash(ticker)) % 2**32)
    if rng.random() < 0.2:
        return {"divs": pd.Series(dtype=float), "ex_date": None, "pay_date": None, "earnings": None}
    per_year = int(rng.choice([1, 2, 4]))
    start = _today() - pd.DateOffset(years=8)
    months = {1: [5], 2: [3, 9], 4: [2, 5, 8, 11]}[per_year]
    dates, amts = [], []
    amt = rng.uniform(0.03, 0.3)
    for y in range(start.year, _today().year + 1):
        amt *= rng.uniform(0.95, 1.12)
        for m in months:
            d = pd.Timestamp(y, m, int(rng.integers(5, 25)))
            if start <= d <= _today():
                dates.append(d)
                amts.append(round(amt / per_year, 4))
    ex = _today() + pd.Timedelta(days=int(rng.integers(3, 40))) if rng.random() < 0.4 else None
    return {"divs": pd.Series(amts, index=pd.DatetimeIndex(dates)),
            "ex_date": ex.date() if ex is not None else None,
            "pay_date": (ex + pd.Timedelta(days=20)).date() if ex is not None else None,
            "earnings": (_today() + pd.Timedelta(days=int(rng.integers(5, 80)))).date()}


def _fetch_div(ticker):
    if DEMO:
        return ticker, _demo_div(ticker)
    import yfinance as yf
    out = {"divs": pd.Series(dtype=float), "ex_date": None, "pay_date": None, "earnings": None}
    t = yf.Ticker(ticker)
    try:
        dv = t.dividends
        if dv is not None and len(dv):
            idx = pd.DatetimeIndex(dv.index)
            if idx.tz is not None:
                idx = idx.tz_localize(None)
            out["divs"] = pd.Series(dv.values.astype(float), index=idx.normalize())
    except Exception:
        pass
    try:
        cal = t.calendar or {}
        out["ex_date"] = cal.get("Ex-Dividend Date")
        out["pay_date"] = cal.get("Dividend Date")
        ed = cal.get("Earnings Date")
        out["earnings"] = ed[0] if isinstance(ed, (list, tuple)) and ed else None
    except Exception:
        pass
    return ticker, out


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_dividends(tickers: tuple) -> dict:
    with ThreadPoolExecutor(max_workers=6) as ex:
        return dict(ex.map(_fetch_div, tickers))


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def load_payout_ratio(ticker):
    if DEMO:
        return float(np.random.default_rng(abs(hash(ticker)) % 2**32).uniform(0.3, 0.9))
    try:
        import yfinance as yf
        v = yf.Ticker(ticker).info.get("payoutRatio")
        return float(v) if v is not None else None
    except Exception:
        return None


def _freq_label(n_per_year):
    if n_per_year >= 10:
        return "Monthly"
    if n_per_year >= 3.5:
        return "Quarterly"
    if n_per_year >= 2.5:
        return "3× a year"
    if n_per_year >= 1.6:
        return "Twice a year"
    if n_per_year >= 0.8:
        return "Yearly"
    return "Irregular"


def summarise(ticker, info, price):
    divs = info["divs"].sort_index()
    today = _today()
    if divs.empty:
        return None
    last_date, last_amt = divs.index[-1], float(divs.iloc[-1])
    ttm = float(divs[divs.index > today - pd.Timedelta(days=365)].sum())
    recent = divs[divs.index > today - pd.Timedelta(days=730)]
    freq = _freq_label(len(recent) / 2)

    # next ex-date: announced (confirmed) or estimated from the same cycle last year
    status, nxt, exp_amt = "—", None, None
    ex = info.get("ex_date")
    if ex is not None and pd.Timestamp(ex) >= today:
        nxt, status, exp_amt = pd.Timestamp(ex), "Confirmed", last_amt
    elif len(divs) >= 2 and (today - last_date).days < 550:
        cands = [(d + pd.DateOffset(years=1), a) for d, a in divs.items() if d + pd.DateOffset(years=1) >= today]
        if cands:
            nxt, exp_amt = min(cands, key=lambda c: c[0])
            nxt = pd.Timestamp(nxt)
            status = "Estimated"
        else:
            gap = pd.Series(divs.index[-6:]).diff().median()
            if pd.notna(gap) and gap.days > 20:
                nxt = last_date + gap
                while nxt < today:
                    nxt += gap
                status, exp_amt = "Estimated", last_amt
    if nxt is not None and status == "Estimated":
        nxt = pd.offsets.BDay().rollforward(nxt)  # ex-dates fall on trading days
    if (today - last_date).days >= 550:
        status = "Stopped?"

    yearly = divs.groupby(divs.index.year).sum()
    full = yearly[yearly.index < today.year]
    growth = None
    if len(full) >= 6 and full.iloc[-6] > 0:
        growth = ((full.iloc[-1] / full.iloc[-6]) ** (1 / 5) - 1) * 100
    streak = 0
    for y in range(today.year - 1, today.year - 40, -1):
        if yearly.get(y, 0) > 0:
            streak += 1
        else:
            break

    return {
        "Name": None, "Ticker": ticker, "Price": price,
        "Yield %": ttm / price * 100 if price else None,
        "Div / yr (TTM)": ttm, "Frequency": freq,
        "Last ex-date": last_date, "Last amount": last_amt,
        "Next ex-date": nxt, "Status": status,
        "Days to ex": (nxt - today).days if nxt is not None else None,
        "Expected amount": exp_amt,
        "5y growth %": growth, "Years paid": streak,
        "Pay date": info.get("pay_date") if status == "Confirmed" else None,
        "Next results": info.get("earnings"),
    }, yearly, divs


def render_dividends(universe, prices, display_name, P, pill):
    st.markdown("### 💰 Dividends")
    st.caption("Dividend history and next ex-dates for the stocks in your sidebar list. **Confirmed** = announced by "
               "the company. **Estimated** = same time as last year's payment (companies usually pay in the same "
               "months each year), so check the official announcement before relying on it.")
    if not universe:
        st.info("Pick a market or enter tickers in the sidebar.")
        return
    with st.spinner("Loading dividend history… (first load can take ~20 seconds)"):
        raw = load_dividends(tuple(universe))
    rows, series = [], {}
    for t in universe:
        if t not in raw or t not in prices:
            continue
        res = summarise(t, raw[t], prices[t])
        if res:
            s, yearly, divs = res
            s["Name"] = display_name(t)
            rows.append(s)
            series[t] = (yearly, divs)
    nonpayers = [display_name(t) for t in universe if t in raw and not len(raw[t]["divs"])]
    if not rows:
        st.info("None of these stocks have paid dividends recently.")
        return
    df = pd.DataFrame(rows)
    for c in ("Pay date", "Next results", "Next ex-date", "Last ex-date"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    today = _today()

    soon = df[(df["Days to ex"].notna()) & (df["Days to ex"] >= 0) & (df["Days to ex"] <= 30)]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dividend payers", f"{len(df)} / {len(universe)}")
    top = df.sort_values("Yield %", ascending=False).iloc[0]
    c2.metric("Highest yield", f"{top['Yield %']:.1f}%", top["Name"], delta_color="off")
    c3.metric("Ex-dates in next 30 days", len(soon))
    c4.metric("Average yield", f"{df['Yield %'].mean():.1f}%")

    cc = st.column_config
    # ---------------- upcoming calendar
    st.markdown("#### 📅 Upcoming ex-dividend dates")
    up = df[df["Next ex-date"].notna() & (df["Days to ex"] >= 0)].sort_values("Next ex-date")
    if up.empty:
        st.caption("No upcoming ex-dates found.")
    else:
        view = up[["Name", "Ticker", "Next ex-date", "Days to ex", "Status", "Expected amount", "Price", "Yield %", "Pay date"]].copy()
        view["Status"] = view["Status"].map({"Confirmed": "✅ Confirmed", "Estimated": "🕒 Estimated"}).fillna(view["Status"])
        st.dataframe(view, hide_index=True, width="stretch", height=min(36 * (len(view) + 1) + 4, 460),
                     column_config={
                         "Next ex-date": cc.DateColumn(format="ddd, D MMM YYYY"),
                         "Days to ex": cc.NumberColumn("Days left", format="%d"),
                         "Expected amount": cc.NumberColumn("Expected / share", format="%.4f"),
                         "Price": cc.NumberColumn(format="%.3f"),
                         "Yield %": cc.NumberColumn("Yield (TTM)", format="%.2f%%"),
                         "Pay date": cc.DateColumn("Payment date", format="D MMM YYYY"),
                     })
        st.caption("To get the dividend you must **own the shares before the ex-date**. Buy on or after the ex-date and "
                   "the dividend goes to the seller.")

    # ---------------- all payers
    st.markdown("#### 🏆 All dividend stocks")
    sort_by = st.segmented_control("Sort by", ["Highest yield", "Next ex-date", "Most consistent", "Fastest growth"],
                                   default="Highest yield", key="div_sort") or "Highest yield"
    key, asc = {"Highest yield": ("Yield %", False), "Next ex-date": ("Days to ex", True),
                "Most consistent": ("Years paid", False), "Fastest growth": ("5y growth %", False)}[sort_by]
    allv = df.sort_values(key, ascending=asc, na_position="last")
    allv = allv[["Name", "Ticker", "Yield %", "Div / yr (TTM)", "Frequency", "Last ex-date", "Last amount",
                 "Next ex-date", "Status", "5y growth %", "Years paid", "Next results"]]
    st.dataframe(allv, hide_index=True, width="stretch", height=min(36 * (len(allv) + 1) + 4, 820),
                 column_config={
                     "Yield %": cc.ProgressColumn("Yield (TTM)", min_value=0, max_value=max(10.0, float(df["Yield %"].max())),
                                                  format="%.2f%%"),
                     "Div / yr (TTM)": cc.NumberColumn("Div / share (12m)", format="%.4f"),
                     "Last ex-date": cc.DateColumn(format="D MMM YYYY"),
                     "Last amount": cc.NumberColumn(format="%.4f"),
                     "Next ex-date": cc.DateColumn(format="D MMM YYYY"),
                     "5y growth %": cc.NumberColumn("5y growth / yr", format="%+.1f%%"),
                     "Years paid": cc.NumberColumn("Years in a row", format="%d"),
                     "Next results": cc.DateColumn("Next results", format="D MMM YYYY"),
                 })
    if nonpayers:
        st.caption("No dividends: " + ", ".join(nonpayers))

    # ---------------- detail
    st.markdown("#### 🔍 Dividend details")
    opts = list(df.sort_values("Yield %", ascending=False)["Ticker"])
    pick = st.selectbox("Stock", opts, format_func=lambda t: f"{display_name(t)}  ·  {t}", key="div_pick")
    r = df[df["Ticker"] == pick].iloc[0]
    cur = _cur(pick)
    payout = load_payout_ratio(pick)
    left, right = st.columns([1.6, 1], gap="medium")
    with left:
        yearly = series[pick][0].tail(10)
        colors = [P["accent"] if y < today.year else P["muted"] for y in yearly.index]
        fig = go.Figure(go.Bar(x=[str(y) for y in yearly.index], y=yearly.values, marker_color=colors,
                               text=[f"{v:.3f}" for v in yearly.values], textposition="outside",
                               textfont=dict(color=P["muted"], size=11)))
        fig.update_layout(template="plotly_dark", height=330, margin=dict(l=8, r=8, t=40, b=8),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          title=dict(text=f"Dividend per share by year ({cur})", font=dict(size=14, color=P["text"])),
                          font=dict(family="Inter, sans-serif", color=P["text"]), showlegend=False)
        fig.update_yaxes(gridcolor=P["grid"], zeroline=False, tickfont=dict(color=P["muted"]))
        fig.update_xaxes(tickfont=dict(color=P["muted"]))
        st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
        st.caption(f"Grey bar = {today.year} so far (not a full year yet).")
        hist = series[pick][1].sort_index(ascending=False).head(12)
        st.dataframe(pd.DataFrame({"Ex-date": hist.index, f"Amount ({cur})": hist.values}), hide_index=True,
                     width="stretch", column_config={"Ex-date": cc.DateColumn(format="D MMM YYYY"),
                                                     f"Amount ({cur})": cc.NumberColumn(format="%.4f")})
    with right:
        nxt = r["Next ex-date"]
        nxt_txt = "Not known" if pd.isna(nxt) else pd.Timestamp(nxt).strftime("%a, %d %b %Y")
        st_col = {"Confirmed": P["up"], "Estimated": P["warn"], "Stopped?": P["down"]}.get(r["Status"], P["muted"])
        warn = ""
        if r["Yield %"] and r["Yield %"] > 9:
            warn = (f"<div class='note' style='color:{P['warn']}'>⚠️ Very high yield. Check that profits can sustain "
                    f"it: sometimes a high yield means the share price has fallen sharply (a 'dividend trap').</div>")
        if payout is not None and payout > 1:
            warn += (f"<div class='note' style='color:{P['warn']}'>⚠️ Paying out more than it earns "
                     f"(payout {payout * 100:.0f}%). The dividend may be cut.</div>")
        g = "–" if pd.isna(r["5y growth %"]) else f"{r['5y growth %']:+.1f}% / yr"
        pr = "–" if payout is None else f"{payout * 100:.0f}%"
        exp_amt = r["Expected amount"]
        exp_txt = "" if pd.isna(exp_amt) else f" · ~{cur}{exp_amt:.4f}/share"
        st.markdown(f"""
<div class="card"><h3>{display_name(pick)}</h3>
<div style="margin-bottom:10px">{pill(f"Next ex-date: {nxt_txt}", st_col)}</div>
<div class="muted" style="margin-bottom:8px">{r['Status']}{exp_txt}</div>
<table class="kv">
<tr><td>Dividend yield (12m)</td><td>{r['Yield %']:.2f}%</td></tr>
<tr><td>Dividend per share (12m)</td><td>{cur}{r['Div / yr (TTM)']:.4f}</td></tr>
<tr><td>Pays</td><td>{r['Frequency']}</td></tr>
<tr><td>Last payment</td><td>{cur}{r['Last amount']:.4f} · {pd.Timestamp(r['Last ex-date']).strftime('%d %b %Y')}</td></tr>
<tr><td>5-year growth</td><td>{g}</td></tr>
<tr><td>Paid every year for</td><td>{int(r['Years paid'])} years</td></tr>
<tr><td>Payout ratio</td><td>{pr}</td></tr>
</table>{warn}</div>""", unsafe_allow_html=True)

        st.markdown("<div class='card'><h3>💵 How much would I get?</h3>", unsafe_allow_html=True)
        is_bursa = pick.endswith(".KL")
        mode = st.radio("I have", ["Shares / lots", "An amount to invest"], horizontal=True, key="div_mode",
                        label_visibility="collapsed")
        if mode.startswith("Shares"):
            if is_bursa:
                lots = st.number_input("Lots (1 lot = 100 shares)", min_value=1, value=10, step=1, key="div_lots")
                shares = lots * 100
            else:
                shares = st.number_input("Shares", min_value=1, value=100, step=1, key="div_shares")
        else:
            amt = st.number_input(f"Amount ({cur})", min_value=100.0, value=10000.0, step=500.0, key="div_amt")
            shares = int(amt // r["Price"])
            if is_bursa:
                shares = shares // 100 * 100
            st.caption(f"≈ {shares:,} shares at {cur}{r['Price']:.3f}")
        per_year = shares * r["Div / yr (TTM)"]
        per_pay = shares * (exp_amt if not pd.isna(exp_amt) else r["Last amount"])
        st.markdown(f"""
<table class="kv">
<tr><td>Cost</td><td>{cur}{shares * r['Price']:,.2f}</td></tr>
<tr><td>Next payment (approx.)</td><td style="color:{P['up']}">{cur}{per_pay:,.2f}</td></tr>
<tr><td>Per year (last 12m rate)</td><td style="color:{P['up']}">{cur}{per_year:,.2f}</td></tr>
<tr><td>Per month (average)</td><td>{cur}{per_year / 12:,.2f}</td></tr>
</table>
<div class="note">Based on past payments; future dividends can be higher, lower or cut. US dividends paid to
Malaysian residents usually have 30% withholding tax deducted.</div></div>""", unsafe_allow_html=True)


# ============================================================================
# IPO
# ============================================================================
def _num(s):
    if s is None:
        return None
    m = re.findall(r"[\d.]+", str(s).replace(",", ""))
    vals = [float(x) for x in m if x not in (".",)]
    return sum(vals) / len(vals) if vals else None


def _demo_ipos():
    today = _today()
    up = [{"proposedTickerSymbol": f"DMO{i}", "companyName": f"Demo Company {i} Inc.", "proposedExchange": "NASDAQ Global",
           "proposedSharePrice": "14.00-16.00", "sharesOffered": "12,500,000",
           "expectedPriceDate": (today + pd.Timedelta(days=3 * i)).strftime("%m/%d/%Y"),
           "dollarValueOfSharesOffered": "$200,000,000"} for i in range(1, 6)]
    pr = [{"proposedTickerSymbol": "AAPL", "companyName": "Demo Priced Co.", "proposedExchange": "NYSE",
           "proposedSharePrice": "21.00", "sharesOffered": "20,000,000",
           "pricedDate": (today - pd.Timedelta(days=4)).strftime("%m/%d/%Y"),
           "dollarValueOfSharesOffered": "$420,000,000", "dealStatus": "Priced"}]
    fl = [{"proposedTickerSymbol": "NEWCO", "companyName": "Demo Filed Corp", "filedDate": today.strftime("%m/%d/%Y"),
           "dollarValueOfSharesOffered": "$100,000,000"}]
    return {"upcoming": up, "priced": pr, "filed": fl, "error": None}


@st.cache_data(ttl=3 * 3600, show_spinner=False)
def load_us_ipos():
    if DEMO:
        return _demo_ipos()
    today = _today()
    months = sorted({(today + pd.DateOffset(months=k)).strftime("%Y-%m") for k in (-1, 0, 1)})
    out = {"upcoming": [], "priced": [], "filed": [], "error": None}
    for m in months:
        try:
            j = _http_get(f"https://api.nasdaq.com/api/ipo/calendar?date={m}", json=True)
            d = (j or {}).get("data") or {}
            out["upcoming"] += (((d.get("upcoming") or {}).get("upcomingTable") or {}).get("rows") or [])
            out["priced"] += ((d.get("priced") or {}).get("rows") or [])
            out["filed"] += ((d.get("filed") or {}).get("rows") or [])
        except Exception as e:
            out["error"] = str(e)[:200]
    for k in ("upcoming", "priced", "filed"):
        seen, uniq = set(), []
        for r in out[k]:
            key = (r.get("dealID"), r.get("companyName"))
            if key not in seen:
                seen.add(key)
                uniq.append(r)
        out[k] = uniq
    return out


@st.cache_data(ttl=600, show_spinner=False)
def load_last_prices(tickers: tuple):
    if not tickers:
        return {}
    if DEMO:
        return {t: 25.0 for t in tickers}
    try:
        import yfinance as yf
        raw = yf.download(list(tickers), period="5d", interval="1d", progress=False, group_by="ticker", threads=True)
        out = {}
        for t in tickers:
            try:
                s = (raw[t] if len(tickers) > 1 else raw)["Close"]
                if isinstance(s, pd.DataFrame):
                    s = s.iloc[:, 0]
                s = s.dropna()
                if len(s):
                    out[t] = float(s.iloc[-1])
            except Exception:
                pass
        return out
    except Exception:
        return {}


@st.cache_data(ttl=1800, show_spinner=False)
def load_bursa_ipo_news():
    if DEMO:
        today = _today()
        return [{"title": f"Demo Bhd IPO oversubscribed {i * 7.5:.1f} times ahead of ACE Market listing",
                 "source": "Demo News", "date": today - pd.Timedelta(days=i), "link": "https://www.bursamalaysia.com"}
                for i in range(1, 9)], None
    q = '"Bursa Malaysia" IPO (listing OR oversubscribed OR prospectus OR debut) when:45d'
    url = ("https://news.google.com/rss/search?q=" + urllib.parse.quote(q) + "&hl=en-MY&gl=MY&ceid=MY:en")
    try:
        root = ET.fromstring(_http_get(url))
        items = []
        for it in root.iter("item"):
            title = (it.findtext("title") or "").strip()
            src = (it.findtext("source") or "").strip()
            if src and title.endswith(" - " + src):
                title = title[: -len(src) - 3]
            try:
                date = pd.Timestamp(pd.to_datetime(it.findtext("pubDate"), utc=True)).tz_convert("Asia/Kuala_Lumpur").tz_localize(None)
            except Exception:
                date = None
            items.append({"title": title, "source": src, "date": date, "link": it.findtext("link")})
        items.sort(key=lambda x: x["date"] or pd.Timestamp(0), reverse=True)
        return items[:40], None
    except Exception as e:
        return [], str(e)[:200]


def render_ipo(P, pill):
    st.markdown("### 🚀 IPO")
    st.caption("New companies listing on the stock market.")
    t_my, t_us, t_learn = st.tabs(["🇲🇾 Bursa Malaysia", "🇺🇸 US (Nasdaq / NYSE)", "📘 How IPOs work"])
    cc = st.column_config

    with t_my:
        items, err = load_bursa_ipo_news()
        st.markdown(f"""
<div class="card"><div class="muted">Bursa doesn't publish a free IPO data feed, so this shows the <b>latest IPO news</b>
(last 45 days, refreshed every 30 minutes). For the full list with prospectus, IPO price and dates see
<a href="https://www.rakutentrade.my/ipo-hub" target="_blank">Rakuten Trade IPO Hub</a> or
<a href="https://www.bursamalaysia.com" target="_blank">bursamalaysia.com</a>. Prospectuses are also on your bank's
e-IPO page (e.g. Maybank2u).</div></div>""", unsafe_allow_html=True)
        if err:
            st.warning("Couldn't load IPO news right now. Check your internet connection and try again later.")
        kw = st.text_input("Filter headlines", placeholder="e.g. ACE, Main Market, oversubscribed, REIT", key="ipo_kw")
        shown = [i for i in items if not kw or kw.lower() in i["title"].lower()]
        tags = [("oversubscribed", P["up"]), ("prospectus", P["accent"]), ("debut", P["warn"]),
                ("listing", P["accent"]), ("ace market", P["muted"]), ("main market", P["muted"]), ("reit", P["muted"])]
        html = []
        for i in shown:
            t_low = i["title"].lower()
            badges = "".join(pill(k.title(), c) + " " for k, c in tags if k in t_low)[:600]
            d = i["date"].strftime("%a %d %b, %H:%M") if i["date"] is not None else ""
            html.append(f"<div class='news'><a href='{i['link']}' target='_blank'>{i['title']}</a>"
                        f"<div class='meta'>{i['source']} · {d} &nbsp;{badges}</div></div>")
        st.markdown("<div class='card'>" + ("".join(html) or "<div class='muted'>No headlines found.</div>") + "</div>",
                    unsafe_allow_html=True)

    with t_us:
        data = load_us_ipos()
        if data.get("error") and not (data["upcoming"] or data["priced"]):
            st.warning("Couldn't reach the Nasdaq IPO calendar right now. Try again later, or see "
                       "nasdaq.com/market-activity/ipos.")
        today = _today()

        st.markdown("#### 📅 Upcoming (expected to price soon)")
        up = pd.DataFrame(data["upcoming"])
        if up.empty:
            st.caption("No upcoming IPOs listed right now.")
        else:
            up = up.rename(columns={"proposedTickerSymbol": "Ticker", "companyName": "Company",
                                    "proposedExchange": "Exchange", "proposedSharePrice": "Price range ($)",
                                    "sharesOffered": "Shares offered", "expectedPriceDate": "Expected date",
                                    "dollarValueOfSharesOffered": "Deal size"})
            if "Expected date" in up:
                up["Expected date"] = pd.to_datetime(up["Expected date"], errors="coerce")
                up = up.sort_values("Expected date")
            cols = [c for c in ["Expected date", "Company", "Ticker", "Exchange", "Price range ($)", "Shares offered",
                                "Deal size"] if c in up]
            st.dataframe(up[cols], hide_index=True, width="stretch",
                         column_config={"Expected date": cc.DateColumn(format="ddd, D MMM YYYY")})

        st.markdown("#### ✅ Recently priced: how are they doing?")
        pr = pd.DataFrame(data["priced"])
        if pr.empty:
            st.caption("No recently priced IPOs.")
        else:
            pr = pr.rename(columns={"proposedTickerSymbol": "Ticker", "companyName": "Company",
                                    "proposedExchange": "Exchange", "proposedSharePrice": "IPO price ($)",
                                    "pricedDate": "Priced on", "dollarValueOfSharesOffered": "Deal size"})
            if "Priced on" in pr:
                pr["Priced on"] = pd.to_datetime(pr["Priced on"], errors="coerce")
                pr = pr.sort_values("Priced on", ascending=False)
            tick = tuple(t for t in pr.get("Ticker", pd.Series(dtype=str)).dropna().astype(str) if t)
            last = load_last_prices(tick)
            pr["Price now ($)"] = pr["Ticker"].map(last) if "Ticker" in pr else None
            ipo_p = pr["IPO price ($)"].map(_num) if "IPO price ($)" in pr else None
            if ipo_p is not None:
                pr["Since IPO %"] = (pr["Price now ($)"] / ipo_p - 1) * 100
            cols = [c for c in ["Priced on", "Company", "Ticker", "Exchange", "IPO price ($)", "Price now ($)",
                                "Since IPO %", "Deal size"] if c in pr]
            st.dataframe(pr[cols], hide_index=True, width="stretch",
                         column_config={"Priced on": cc.DateColumn(format="D MMM YYYY"),
                                        "Price now ($)": cc.NumberColumn(format="%.2f"),
                                        "Since IPO %": cc.NumberColumn(format="%+.1f%%")})
            st.caption("Tip: type any of these tickers in the Analysis tab. New listings have little history, so "
                       "support/resistance and forecasts are less reliable for the first few months.")

        fl = pd.DataFrame(data["filed"])
        with st.expander(f"📝 Newly filed (planning to IPO, no date yet) · {len(fl)}"):
            if fl.empty:
                st.caption("None.")
            else:
                fl = fl.rename(columns={"proposedTickerSymbol": "Ticker", "companyName": "Company",
                                        "filedDate": "Filed on", "dollarValueOfSharesOffered": "Planned size"})
                if "Filed on" in fl:
                    fl["Filed on"] = pd.to_datetime(fl["Filed on"], errors="coerce")
                    fl = fl.sort_values("Filed on", ascending=False)
                cols = [c for c in ["Filed on", "Company", "Ticker", "Planned size"] if c in fl]
                st.dataframe(fl[cols], hide_index=True, width="stretch",
                             column_config={"Filed on": cc.DateColumn(format="D MMM YYYY")})
        st.caption("Source: Nasdaq IPO calendar, refreshed every 3 hours. Can a Malaysian buy US IPO shares before "
                   "listing? Usually not through local brokers. Most people buy on the first trading day instead.")

    with t_learn:
        st.markdown("""
**What is an IPO?** *Initial Public Offering*: the first time a company sells its shares to the public and lists on
the stock exchange.

**Bursa Malaysia markets**
- **Main Market**: larger, established companies with a profit track record.
- **ACE Market**: smaller growth companies. More volatile, higher risk.
- **LEAP Market**: only for sophisticated investors; ordinary retail investors can't buy.

**How to apply for a Bursa IPO (public portion)**
1. You need a **CDS account** (opened through your stockbroker, e.g. moomoo MY, Rakuten Trade, M+).
2. Read the **prospectus** (key info: business, IPO price, use of proceeds, risks, P/E at IPO price).
3. Apply through your bank's **e-IPO** (e.g. Maybank2u, CIMB Clicks) or your broker before the closing date.
4. If the IPO is **oversubscribed**, there's a **balloting** (lucky draw). You may get some or none; unsuccessful
   money is refunded.
5. Shares are credited to your CDS account and you can trade from **listing day**.

**Key terms**
- **IPO price**: price you pay if you apply before listing.
- **Oversubscribed X times**: demand was X times the shares available. High numbers = very popular (but not a
  guarantee of first-day gains).
- **Listing date / debut**: first trading day.
- **Lock-up**: period when founders/major holders can't sell. When it ends, extra selling can push price down.
- **Cornerstone investors**: big funds that commit to buy a block of shares before the IPO, usually a positive sign.

**Beginner tips**
- Hot IPOs can jump on day one and then fall for weeks. Don't chase the first-day spike.
- Compare the IPO **P/E** with similar listed companies: is it cheap or expensive?
- Check **use of proceeds**: expansion is better than mostly repaying debt or cashing out owners.
- New listings have no trading history, so technical signals in this app need a few months of data to be useful.
""")


# ============================================================================
# HEATMAP
# ============================================================================
SECTORS = {
    # Bursa
    "1155.KL": "Financials", "1295.KL": "Financials", "1023.KL": "Financials", "1066.KL": "Financials",
    "5819.KL": "Financials", "5347.KL": "Utilities", "4677.KL": "Utilities", "6033.KL": "Utilities",
    "5225.KL": "Healthcare", "5168.KL": "Healthcare", "7113.KL": "Healthcare",
    "5183.KL": "Energy & Chemicals", "5681.KL": "Energy & Chemicals", "3816.KL": "Transport",
    "5285.KL": "Plantation", "1961.KL": "Plantation", "2445.KL": "Plantation",
    "5398.KL": "Construction & Property", "5211.KL": "Construction & Property",
    "0166.KL": "Technology", "0128.KL": "Technology", "0138.KL": "Technology",
    "6947.KL": "Telecom", "6888.KL": "Telecom", "4863.KL": "Telecom",
    "4707.KL": "Consumer", "7084.KL": "Consumer", "5296.KL": "Consumer", "3182.KL": "Consumer", "4715.KL": "Consumer",
    # US
    "NVDA": "Technology", "AAPL": "Technology", "MSFT": "Technology", "AVGO": "Technology", "AMD": "Technology",
    "ORCL": "Technology", "CRM": "Technology", "INTC": "Technology", "MU": "Technology", "PLTR": "Technology",
    "GOOGL": "Communication", "META": "Communication", "NFLX": "Communication", "DIS": "Communication",
    "AMZN": "Consumer", "TSLA": "Consumer", "COST": "Consumer", "WMT": "Consumer",
    "JPM": "Financials", "V": "Financials", "MA": "Financials", "LLY": "Healthcare", "UNH": "Healthcare",
    "XOM": "Energy", "BA": "Industrials", "SPY": "ETF", "QQQ": "ETF",
}


def _pct(close, n):
    if len(close) <= n:
        return None
    return (float(close.iloc[-1]) / float(close.iloc[-1 - n]) - 1) * 100


def _tv_heatmap(source, height, P):
    html = f"""<!DOCTYPE html><html><head><style>html,body{{margin:0;height:{height}px;background:{P['bg']}}}</style></head>
<body><div class="tradingview-widget-container" style="height:{height}px;width:100%">
<div class="tradingview-widget-container__widget" style="height:{height}px;width:100%"></div>
<script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-stock-heatmap.js" async>
{{"exchanges": [], "dataSource": "{source}", "grouping": "sector", "blockSize": "market_cap_basic",
  "blockColor": "change", "locale": "en", "symbolUrl": "", "colorTheme": "dark", "hasTopBar": true,
  "isDataSetEnabled": true, "isZoomEnabled": true, "hasSymbolTooltip": true, "isMonoSize": false,
  "width": "100%", "height": {height}}}
</script></div></body></html>"""
    if hasattr(st, "iframe"):
        st.iframe(html, height=height + 4)
    else:
        import streamlit.components.v1 as components
        components.html(html, height=height + 4)


def render_heatmap(universe, display_name, P, load_many, analyse):
    st.markdown("### 🗺️ Market heatmap")
    t_mine, t_tv = st.tabs(["📋 My list (Bursa / US)", "🇺🇸 Whole US market (TradingView)"])

    with t_mine:
        _heatmap_mine(universe, display_name, P, load_many, analyse)

    with t_tv:
        c1, c2 = st.columns([1, 2])
        src = c1.selectbox("Index", ["SPX500", "NASDAQ100"], key="tvhm_src",
                           format_func=lambda v: {"SPX500": "S&P 500", "NASDAQ100": "Nasdaq 100"}[v])
        tv_h = c2.slider("Height (px)", 500, 1200, 750, 50, key="tvhm_h")
        st.caption("TradingView heatmap of the whole index, sized by market cap. Data may be delayed ~15 min. "
                   "Hover a box for details.")
        _tv_heatmap(src, tv_h, P)


def _heatmap_mine(universe, display_name, P, load_many, analyse):
    if True:
        if not universe:
            st.info("Pick a market or enter tickers in the sidebar.")
            return
        c1, c2 = st.columns([2, 1])
        metric = c1.segmented_control("Colour by", ["Today", "1 week", "1 month", "3 months", "Signal score"],
                                      default="Today", key="hm_metric") or "Today"
        size_by = c2.segmented_control("Box size", ["Trading value", "Equal"], default="Trading value",
                                       key="hm_size") or "Trading value"
        with st.spinner("Building heatmap…"):
            data = load_many(tuple(universe), "1d", "1y")
        rows = []
        for t, df in data.items():
            c = df["Close"].dropna()
            if len(c) < 30:
                continue
            row = {"t": t, "name": display_name(t), "sector": SECTORS.get(t, "Other"),
                   "price": float(c.iloc[-1]),
                   "Today": _pct(c, 1), "1 week": _pct(c, 5), "1 month": _pct(c, 21), "3 months": _pct(c, 63),
                   "value": float((df["Close"] * df["Volume"]).tail(20).mean())}
            try:
                row["Signal score"] = float(analyse(df)[3])
            except Exception:
                row["Signal score"] = None
            rows.append(row)
        if not rows:
            st.warning("No data available.")
            return
        h = pd.DataFrame(rows)
        h["metric"] = h[metric].astype(float)
        h["size"] = h["value"].clip(lower=1) if size_by == "Trading value" else 1.0
        if size_by == "Trading value":  # dampen giants so small caps stay visible
            h["size"] = np.sqrt(h["size"])

        lim = {"Today": 3, "1 week": 6, "1 month": 12, "3 months": 25, "Signal score": 60}[metric]
        unit = "" if metric == "Signal score" else "%"
        sec_rows = []
        for name, g in h.groupby("sector"):
            sec_rows.append({"sector": name, "size": float(g["size"].sum()),
                             "metric": float(np.average(g["metric"].fillna(0), weights=g["size"]))})
        sec = pd.DataFrame(sec_rows)
        ids = list(sec["sector"]) + list(h["t"])
        labels = list(sec["sector"]) + list(h["name"])
        parents = [""] * len(sec) + list(h["sector"])
        values = list(sec["size"]) + list(h["size"])
        colors = list(sec["metric"]) + list(h["metric"].fillna(0))
        fmt = (lambda v: f"{v:+.0f}") if metric == "Signal score" else (lambda v: f"{v:+.2f}%")
        text = [f"{fmt(v)}" for v in sec["metric"]] + \
               [f"{fmt(v) if pd.notna(v) else '–'}<br>{p:,.3f}" for v, p in zip(h["metric"], h["price"])]
        scale = [[0, P["down"]], [0.5, "#2b3242"], [1, P["up"]]]
        fig = go.Figure(go.Treemap(
            ids=ids, labels=labels, parents=parents, values=values, branchvalues="total",
            text=text, texttemplate="<b>%{label}</b><br>%{text}", textposition="middle center",
            hovertemplate="<b>%{label}</b><br>%{text}<extra></extra>",
            marker=dict(colors=colors, colorscale=scale, cmid=0, cmin=-lim, cmax=lim,
                        line=dict(color=P["bg"], width=2), pad=dict(t=22, l=2, r=2, b=2),
                        colorbar=dict(title=dict(text=unit or "score", font=dict(color=P["muted"])),
                                      tickfont=dict(color=P["muted"]), thickness=10, len=0.6)),
            tiling=dict(pad=2), pathbar=dict(visible=False),
            textfont=dict(family="Inter, sans-serif", color="#ffffff", size=13),
            insidetextfont=dict(color="#ffffff"),
        ))
        fig.update_layout(height=620, margin=dict(l=0, r=0, t=4, b=0), paper_bgcolor="rgba(0,0,0,0)",
                          font=dict(family="Inter, sans-serif", color=P["text"]))
        st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
        st.caption(f"Green = up, red = down ({metric.lower()}). Box size = "
                   f"{'average daily trading value (last 20 days)' if size_by == 'Trading value' else 'equal'}. "
                   "Click a sector to zoom in, click again to zoom out. Uses daily data.")

        a, b, c = st.columns(3, gap="medium")
        hs = h.dropna(subset=["metric"]).sort_values("metric", ascending=False)

        def lst(df):
            return "".join(
                f"<div class='reason'><span style='flex:1'>{r['name']}</span>"
                f"<b style='color:{P['up'] if r['metric'] >= 0 else P['down']}'>{fmt(r['metric'])}</b></div>"
                for _, r in df.iterrows())
        a.markdown(f"<div class='card'><h3>🚀 Top gainers · {metric.lower()}</h3>{lst(hs.head(5))}</div>",
                   unsafe_allow_html=True)
        b.markdown(f"<div class='card'><h3>📉 Top losers · {metric.lower()}</h3>{lst(hs.tail(5).iloc[::-1])}</div>",
                   unsafe_allow_html=True)
        secs = sec.sort_values("metric", ascending=False)
        srows = "".join(
            f"<div class='reason'><span style='flex:1'>{r['sector']}</span>"
            f"<b style='color:{P['up'] if r['metric'] >= 0 else P['down']}'>{fmt(r['metric'])}</b></div>"
            for _, r in secs.iterrows())
        c.markdown(f"<div class='card'><h3>🏭 Sectors · {metric.lower()}</h3>{srows}</div>", unsafe_allow_html=True)

