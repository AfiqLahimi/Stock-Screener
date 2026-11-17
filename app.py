"""
Live Stock Screener & Analysis (Bursa Malaysia + US)
Run:  streamlit run app.py
Price data: Yahoo Finance (yfinance). Bursa ~15 min delayed, US near real-time.
"""
import os
import re
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots

import extras
import learn
import picks
import ta_plus
import terminal

st.set_page_config(page_title="Live Stock Screener & Analysis", page_icon="📈", layout="wide")

DEMO = os.environ.get("DEMO_DATA") == "1"  # for testing without internet
if not DEMO:
    import yfinance as yf

# ----------------------------------------------------------------------------
# Stock lists
# ----------------------------------------------------------------------------
BURSA = {
    "1155.KL": "MAYBANK", "1295.KL": "PBBANK", "1023.KL": "CIMB", "1066.KL": "RHBBANK",
    "5819.KL": "HLBANK", "5347.KL": "TENAGA", "5225.KL": "IHH", "5183.KL": "PCHEM",
    "5285.KL": "SDG", "5398.KL": "GAMUDA", "4677.KL": "YTL", "6947.KL": "CDB",
    "0166.KL": "INARI", "0128.KL": "FRONTKN", "6888.KL": "AXIATA", "4863.KL": "TM",
    "4707.KL": "NESTLE", "3816.KL": "MISC", "5681.KL": "PETDAG", "6033.KL": "PETGAS",
    "1961.KL": "IOICORP", "2445.KL": "KLK", "5211.KL": "SUNWAY", "3182.KL": "GENTING",
    "4715.KL": "GENM", "7084.KL": "QL", "5296.KL": "MRDIY", "0138.KL": "MYEG",
    "5168.KL": "HARTA", "7113.KL": "TOPGLOV",
}
US = {
    "NVDA": "Nvidia", "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Alphabet",
    "AMZN": "Amazon", "META": "Meta", "TSLA": "Tesla", "AVGO": "Broadcom", "AMD": "AMD",
    "NFLX": "Netflix", "PLTR": "Palantir", "ORCL": "Oracle", "CRM": "Salesforce",
    "INTC": "Intel", "MU": "Micron", "JPM": "JPMorgan", "V": "Visa", "MA": "Mastercard",
    "COST": "Costco", "WMT": "Walmart", "LLY": "Eli Lilly", "UNH": "UnitedHealth",
    "XOM": "Exxon", "BA": "Boeing", "DIS": "Disney", "SPY": "S&P 500 ETF", "QQQ": "Nasdaq 100 ETF",
}
NAME_TO_BURSA = {v: k for k, v in BURSA.items()}

INTERVALS = {
    "Daily (1D)": ("1d", "2y", "trading days"),
    "Weekly (1W)": ("1wk", "10y", "weeks"),
    "Monthly (1M)": ("1mo", "max", "months"),
    "4 Hours": ("4h", "730d", "4-hour bars"),
    "1 Hour": ("1h", "6mo", "trading hours"),
    "15 Minutes": ("15m", "1mo", "15-minute bars"),
}


def normalise(sym: str) -> str:
    s = (sym or "").strip().upper().replace(" ", "")
    if not s:
        return ""
    if s in NAME_TO_BURSA:
        return NAME_TO_BURSA[s]
    if s.startswith("MYX:"):
        s = s[4:]
        return NAME_TO_BURSA.get(s, s if s.endswith(".KL") else s + ".KL")
    if ":" in s:  # NASDAQ:NVDA -> NVDA
        s = s.split(":", 1)[1]
    if s.isdigit() and len(s) == 4:
        return s + ".KL"
    # full-Bursa names (e.g. "SKA" or "SKA CAPITAL")
    if NAMES:
        up = {v.upper().replace(" ", ""): k for k, v in NAMES.items()}
        if s in up:
            return up[s]
        hits = [k for n, k in up.items() if n.startswith(s)]
        if len(hits) == 1:
            return hits[0]
    return s


NAMES = {}  # filled from the Bursa-wide screener (ticker -> short name)


@st.cache_data(show_spinner=False)
def load_shariah():
    """SC Shariah Advisory Council list (bundled as shariah.csv). Returns (set of tickers, as-of text)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shariah.csv")
    if not os.path.exists(path):
        return set(), None
    df = pd.read_csv(path, dtype=str)
    asof = df["as_of"].dropna().iloc[0] if "as_of" in df and df["as_of"].notna().any() else None
    return {c.strip().zfill(4) + ".KL" for c in df["code"].dropna()}, asof


def display_name(t: str) -> str:
    return BURSA.get(t) or US.get(t) or NAMES.get(t) or t


def tv_symbol(t: str) -> str:
    if t.endswith(".KL"):
        return "MYX:" + BURSA.get(t, t[:-3])
    return t


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def _demo(ticker: str, interval: str) -> pd.DataFrame:
    rng = np.random.default_rng(abs(hash((ticker, interval))) % 2**32)
    n = 500
    freq = {"1d": "B", "1wk": "W-FRI", "1mo": "MS", "1h": "h", "15m": "15min"}[interval]
    idx = pd.date_range(end=pd.Timestamp.now().floor("h"), periods=n, freq=freq)
    close = 20 * np.exp(np.cumsum(rng.normal(0.0004, 0.018, n)))
    op = close * (1 + rng.normal(0, 0.006, n))
    hi = np.maximum(op, close) * (1 + np.abs(rng.normal(0, 0.008, n)))
    lo = np.minimum(op, close) * (1 - np.abs(rng.normal(0, 0.008, n)))
    vol = rng.integers(100_000, 5_000_000, n).astype(float)
    return pd.DataFrame({"Open": op, "High": hi, "Low": lo, "Close": close, "Volume": vol}, index=idx)


def _clean(d: pd.DataFrame) -> pd.DataFrame:
    """Data-integrity pass: keep complete, ordered, de-duplicated OHLCV rows only."""
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d = d[["Open", "High", "Low", "Close", "Volume"]].copy()
    n0 = len(d)
    d = d[~d.index.duplicated(keep="last")].sort_index()
    d = d.dropna(subset=["Open", "High", "Low", "Close"])
    d = d[(d[["Open", "High", "Low", "Close"]] > 0).all(axis=1) & (d["High"] >= d["Low"])]
    d["Volume"] = d["Volume"].fillna(0)
    d.attrs["dropped_rows"] = int(n0 - len(d))
    return d


def _resample_4h(h1: pd.DataFrame) -> pd.DataFrame:
    """Yahoo has no 4-hour candles: build them from real 1-hour candles, 4 per block within each trading day."""
    if h1.empty:
        return h1
    x = h1.copy()
    day = x.index.normalize() if x.index.tz is None else x.index.tz_localize(None).normalize()
    x["_day"] = day
    x["_blk"] = x.groupby("_day").cumcount() // 4
    g = x.groupby(["_day", "_blk"])
    out = pd.DataFrame({"Open": g["Open"].first(), "High": g["High"].max(), "Low": g["Low"].min(),
                        "Close": g["Close"].last(), "Volume": g["Volume"].sum()})
    x["_ts"] = x.index
    out.index = pd.DatetimeIndex(x.groupby(["_day", "_blk"])["_ts"].first().tolist())  # keeps exchange timezone
    out.attrs.update(h1.attrs)
    return out


@st.cache_data(ttl=60, show_spinner=False)
def load_one(ticker: str, interval: str, period: str) -> pd.DataFrame:
    if DEMO:
        d = _demo(ticker, "1h" if interval == "4h" else interval)
        if interval == "4h":
            d = _resample_4h(d)
        d.attrs["fetched"] = pd.Timestamp.now(tz="Asia/Kuala_Lumpur").isoformat()
        d.attrs["source"] = "DEMO (synthetic)"
        return d
    src_iv = "1h" if interval == "4h" else interval
    try:
        d = yf.download(ticker, period=period, interval=src_iv, auto_adjust=False,
                        progress=False, threads=False)
    except Exception:
        d = None
    if d is None or d.empty:
        return pd.DataFrame()
    d = _clean(d)
    if interval == "4h":
        d = _resample_4h(d)
    d.attrs["fetched"] = pd.Timestamp.now(tz="Asia/Kuala_Lumpur").isoformat()
    d.attrs["source"] = "Yahoo Finance (yfinance), unadjusted OHLC" + (" · 4H built from 1H" if interval == "4h" else "")
    return d


def _download_many(tickers: tuple, interval: str, period: str) -> dict:
    if DEMO:
        return {t: (_resample_4h(_demo(t, "1h")) if interval == "4h" else _demo(t, interval)) for t in tickers}
    src_iv = "1h" if interval == "4h" else interval
    raw = yf.download(list(tickers), period=period, interval=src_iv, group_by="ticker",
                      auto_adjust=False, progress=False, threads=True)
    out = {}
    for t in tickers:
        try:
            d = raw[t] if isinstance(raw.columns, pd.MultiIndex) and len(tickers) > 1 else raw
            d = _clean(d.copy())
            if interval == "4h":
                d = _resample_4h(d)
            if len(d) > 60:
                out[t] = d
        except Exception:
            pass
    return out


@st.cache_data(ttl=60, show_spinner=False)
def load_many(tickers: tuple, interval: str, period: str) -> dict:
    return _download_many(tickers, interval, period)


@st.cache_data(ttl=600, show_spinner=False)
def load_many_big(tickers: tuple, interval: str, period: str) -> dict:
    """Same as load_many, cached for 10 minutes, for big lists (Bursa-wide scans)."""
    return _download_many(tickers, interval, period)


def get_many(tickers: tuple, interval: str, period: str) -> dict:
    """Pick the cache by list size: small lists refresh every minute, big ones every 10 minutes."""
    return load_many(tickers, interval, period) if len(tickers) <= 60 else load_many_big(tickers, interval, period)


YF_SECTORS = ["Basic Materials", "Communication Services", "Consumer Cyclical", "Consumer Defensive", "Energy",
              "Financial Services", "Healthcare", "Industrials", "Real Estate", "Technology", "Utilities"]


def _short_name(n):
    n = (n or "").strip()
    for suf in (" BERHAD", " BHD", " Berhad", " Bhd", " BHD."):
        if n.endswith(suf):
            n = n[: -len(suf)]
    return n[:22]


def _demo_bursa_all():
    rng = np.random.default_rng(7)
    rows = []
    for i in range(420):
        code = f"{rng.integers(0, 9999):04d}.KL"
        price = float(np.round(np.exp(rng.normal(0, 1.1)), 3))
        vol = float(rng.lognormal(13, 1.5))
        rows.append({"ticker": code, "name": f"DEMO {i}", "sector": YF_SECTORS[i % len(YF_SECTORS)],
                     "price": price, "chg": float(rng.normal(0, 2)), "volume": vol,
                     "avg_volume": vol, "mcap": price * float(rng.lognormal(19, 1.2)),
                     "pe": float(rng.uniform(5, 40)), "div_yield": float(rng.uniform(0, 7))})
    for t, nm in BURSA.items():
        rows.append({"ticker": t, "name": nm, "sector": "Financial Services", "price": 10.0, "chg": 0.5,
                     "volume": 5e6, "avg_volume": 5e6, "mcap": 5e10, "pe": 12.0, "div_yield": 4.0})
    return pd.DataFrame(rows).drop_duplicates("ticker")


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def load_bursa_all() -> pd.DataFrame:
    """Every stock listed on Bursa Malaysia (Yahoo exchange code KLS), with sector, via Yahoo's screener."""
    if DEMO:
        return _demo_bursa_all()
    from concurrent.futures import ThreadPoolExecutor
    from yfinance import EquityQuery

    base = EquityQuery("eq", ["exchange", "KLS"])

    def pull(q):
        out, off = [], 0
        while off < 3000:
            try:
                r = yf.screen(q, offset=off, size=250, sortField="intradaymarketcap", sortAsc=False)
            except Exception:
                break
            qs = (r or {}).get("quotes") or []
            out += qs
            off += len(qs)
            if not qs or off >= int((r or {}).get("total") or 0):
                break
        return out

    rows = {}
    for q in pull(base):
        sym = q.get("symbol")
        if not sym or not sym.endswith(".KL"):
            continue
        price = q.get("regularMarketPrice")
        rows[sym] = {"ticker": sym, "name": _short_name(q.get("shortName") or q.get("longName") or sym),
                     "sector": "Other", "price": price, "chg": q.get("regularMarketChangePercent"),
                     "volume": q.get("regularMarketVolume"), "avg_volume": q.get("averageDailyVolume3Month"),
                     "mcap": q.get("marketCap"), "pe": q.get("trailingPE"),
                     "div_yield": q.get("dividendYield") if q.get("dividendYield") is not None
                     else (q.get("trailingAnnualDividendYield") or 0) * 100}

    def by_sector(sec):
        return sec, pull(EquityQuery("and", [base, EquityQuery("eq", ["sector", sec])]))

    with ThreadPoolExecutor(max_workers=4) as ex:
        for sec, qs in ex.map(by_sector, YF_SECTORS):
            for q in qs:
                sym = q.get("symbol")
                if sym in rows:
                    rows[sym]["sector"] = sec
    df = pd.DataFrame(list(rows.values()))
    if df.empty:
        return df
    for c in ("price", "chg", "volume", "avg_volume", "mcap", "pe", "div_yield"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ----------------------------------------------------------------------------
# Technical analysis
# ----------------------------------------------------------------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    c = d["Close"]
    for n in (20, 50, 200):
        d[f"EMA{n}"] = c.ewm(span=n, adjust=False).mean()
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    d["RSI"] = (100 - 100 / (1 + rs)).fillna(50)
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    d["MACD"] = ema12 - ema26
    d["MACDsig"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACDhist"] = d["MACD"] - d["MACDsig"]
    tr = pd.concat([d["High"] - d["Low"], (d["High"] - c.shift()).abs(), (d["Low"] - c.shift()).abs()], axis=1).max(axis=1)
    d["ATR"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    d["VolAvg"] = d["Volume"].rolling(20).mean()
    d["VolAvg20"] = d["Volume"].shift(1).rolling(20).mean()          # average of the previous 20 bars
    d["RVOL"] = (d["Volume"] / d["VolAvg20"]).replace([np.inf, -np.inf], np.nan)
    d["OBV"] = (np.sign(c.diff()).fillna(0) * d["Volume"]).cumsum()
    # ADX (14) with +DI / -DI, Wilder smoothing
    up_move = d["High"].diff()
    down_move = -d["Low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    a = 1 / 14
    atr_w = tr.ewm(alpha=a, adjust=False).mean()
    d["PDI"] = 100 * pd.Series(plus_dm, index=d.index).ewm(alpha=a, adjust=False).mean() / atr_w.replace(0, np.nan)
    d["MDI"] = 100 * pd.Series(minus_dm, index=d.index).ewm(alpha=a, adjust=False).mean() / atr_w.replace(0, np.nan)
    dx = 100 * (d["PDI"] - d["MDI"]).abs() / (d["PDI"] + d["MDI"]).replace(0, np.nan)
    d["ADX"] = dx.ewm(alpha=a, adjust=False).mean()
    d["HH20"] = d["High"].shift(1).rolling(20).max()                 # prior 20-bar high / low (breakout levels)
    d["LL20"] = d["Low"].shift(1).rolling(20).min()
    return d


def find_levels(d: pd.DataFrame, lookback: int = 250, w: int = 5):
    """Support/resistance from clustered swing highs/lows."""
    x = d.tail(lookback)
    hi, lo = x["High"].values, x["Low"].values
    price, atr = float(x["Close"].iloc[-1]), float(x["ATR"].iloc[-1])
    pivots = []
    for i in range(w, len(x) - w):
        if hi[i] == hi[i - w:i + w + 1].max():
            pivots.append(hi[i])
        if lo[i] == lo[i - w:i + w + 1].min():
            pivots.append(lo[i])
    # include the latest swing (not enough bars to its right yet)
    pivots += [float(x["High"].tail(w).max()), float(x["Low"].tail(w).min())]
    pivots.sort()
    tol = max(0.5 * atr, price * 0.008)
    clusters = []
    for p in pivots:
        if clusters and p - clusters[-1][-1] <= tol:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    lv = [(float(np.mean(c)), len(c)) for c in clusters]
    gap = 0.15 * atr  # levels too close to price are treated as "being tested"
    sup = sorted([l for l in lv if l[0] < price - gap], key=lambda l: -l[0])[:3]
    res = sorted([l for l in lv if l[0] > price + gap], key=lambda l: l[0])[:3]
    if not sup:
        sup = [(price - 2 * atr, 0)]
    if not res:
        res = [(price + 2 * atr, 0)]  # 0 touches = ATR projection (price at highs)
    while len(res) < 2:
        res.append((res[-1][0] + 2 * atr, 0))
    while len(sup) < 2:
        sup.append((sup[-1][0] - 2 * atr, 0))
    return sup, res


def score_factors(d: pd.DataFrame, sup, res):
    """The scoring rules, one row per factor, with the actual reading and the points it earned.
    Points are identical to the original scoring system; only the descriptions were made exact."""
    last, prev = d.iloc[-1], d.iloc[-2]
    price, atr = float(last["Close"]), float(last["ATR"])
    F = []

    def add(group, factor, reading, rule, pts, text):
        F.append({"Group": group, "Factor": factor, "Reading": reading, "Rule": rule, "Points": pts, "text": text})

    e20, e50, e200 = float(last["EMA20"]), float(last["EMA50"]), float(last["EMA200"])
    add("Trend", "Price vs EMA200", f"{price:.3f} vs {e200:.3f}", "above: +15 · below: −15",
        15 if price > e200 else -15,
        "Price above EMA200: long-term uptrend" if price > e200 else "Price below EMA200: long-term downtrend")
    add("Trend", "EMA50 vs EMA200", f"{e50:.3f} vs {e200:.3f}", "above: +10 · below: −10",
        10 if e50 > e200 else -10,
        "EMA50 above EMA200 (golden cross structure)" if e50 > e200 else "EMA50 below EMA200 (death cross structure)")
    add("Trend", "Price vs EMA20", f"{price:.3f} vs {e20:.3f}", "above: +10 · below: −10",
        10 if price > e20 else -10,
        "Price above EMA20: short-term momentum positive" if price > e20 else "Price below EMA20: short-term momentum weak")

    rsi = float(last["RSI"])
    if rsi < 30:
        p_, t_ = 15, f"RSI {rsi:.0f}: oversold, potential bounce"
    elif rsi > 70:
        p_, t_ = -15, f"RSI {rsi:.0f}: overbought, pullback risk"
    elif rsi >= 50:
        p_, t_ = 5, f"RSI {rsi:.0f}: buyers have momentum"
    else:
        p_, t_ = -5, f"RSI {rsi:.0f}: sellers have momentum"
    add("Momentum", "RSI (14)", f"{rsi:.1f}", "<30: +15 · 30–50: −5 · 50–70: +5 · >70: −15", p_, t_)

    m, sg, h, hp = float(last["MACD"]), float(last["MACDsig"]), float(last["MACDhist"]), float(prev["MACDhist"])
    reading = f"MACD {m:.4f} · signal {sg:.4f} · hist {h:+.4f}"
    if h > 0 and hp <= 0:
        p_, t_ = 15, "MACD line just crossed ABOVE the signal line (bullish crossover)"
    elif h < 0 and hp >= 0:
        p_, t_ = -15, "MACD line just crossed BELOW the signal line (bearish crossover)"
    elif h > 0:
        p_, t_ = 5, "MACD line above the signal line (bullish momentum)"
    else:
        p_, t_ = -5, "MACD line below the signal line (bearish momentum)"
    add("Momentum", "MACD vs signal line", reading,
        "cross above: +15 · cross below: −15 · above: +5 · below: −5", p_, t_)

    s1, r1 = sup[0][0], res[0][0]
    if price - s1 <= atr:
        add("Levels", "Near support", f"{price - s1:.3f} above S {s1:.3f} (ATR {atr:.3f})", "within 1 ATR: +15", 15,
            f"Price near support {s1:.3f} (low-risk entry area)")
    if r1 - price <= atr and res[0][1] > 0:
        add("Levels", "Near resistance", f"{r1 - price:.3f} below R {r1:.3f} (ATR {atr:.3f})", "within 1 ATR: −15", -15,
            f"Price near resistance {r1:.3f} (limited upside)")
    if prev["Close"] < s1 <= price:
        add("Levels", "Breakout", f"prev close {float(prev['Close']):.3f} → {price:.3f} over {s1:.3f}",
            "closed above a level: +10", 10, f"Breakout: just closed above {s1:.3f}")
    if prev["Close"] > r1 >= price:
        add("Levels", "Breakdown", f"prev close {float(prev['Close']):.3f} → {price:.3f} under {r1:.3f}",
            "closed below a level: −10", -10, f"Breakdown: just closed below {r1:.3f}")

    if pd.notna(last["VolAvg"]) and last["VolAvg"] > 0 and last["Volume"] > 1.5 * last["VolAvg"]:
        green = price >= last["Open"]
        add("Volume", "High-volume candle", f"{last['Volume'] / last['VolAvg']:.1f}× 20-bar avg on a "
            f"{'green' if green else 'red'} candle", ">1.5× avg: green +5 · red −5", 5 if green else -5,
            "High volume on a green candle (strong buyers)" if green else "High volume on a red candle (strong sellers)")
    return F


def score_signal(d: pd.DataFrame, sup, res):
    F = score_factors(d, sup, res)
    s = int(max(-100, min(100, sum(f["Points"] for f in F))))
    why = [(f["Points"], f["text"]) for f in F]
    if s >= 40:
        label = "STRONG BUY"
    elif s >= 15:
        label = "BUY"
    elif s <= -40:
        label = "STRONG SELL"
    elif s <= -15:
        label = "SELL"
    else:
        label = "NEUTRAL"
    return s, label, why


def trade_plan(d, sup, res, label, unit, atr_mult: float = 1.0):
    last = d.iloc[-1]
    price, atr = float(last["Close"]), float(last["ATR"])
    s1, s2 = sup[0][0], sup[1][0]
    r1, r2 = res[0][0], res[1][0]
    buy_lo, buy_hi = s1, s1 + 0.5 * atr
    stop = s1 - atr_mult * atr
    sell_lo, sell_hi = r1 - 0.5 * atr, r1
    entry = price if price <= buy_hi else (buy_lo + buy_hi) / 2
    risk = entry - stop
    rr1 = (r1 - entry) / risk if risk > 0 else np.nan
    rr2 = (r2 - entry) / risk if risk > 0 else np.nan
    bars_t1 = max(1, round((r1 - entry) / atr)) if atr > 0 else None
    bars_back = max(1, round((price - buy_hi) / atr)) if price > buy_hi and atr > 0 else 0

    in_zone = price <= buy_hi
    bullish = label in ("BUY", "STRONG BUY")
    bearish = label in ("SELL", "STRONG SELL")

    if bullish and in_zone:
        action = "🟢 CONSIDER BUYING"
        when = (f"Price is in the buy zone now. Entry around **{price:.3f}**, stop loss **{stop:.3f}**. "
                f"Take half profit at **{r1:.3f}**, the rest at **{r2:.3f}**.")
    elif bullish:
        action = "🟡 UPTREND, BUT DON'T CHASE"
        when = (f"Price is far above support. Two options: **(a)** wait for a pullback to **{buy_lo:.3f}–{buy_hi:.3f}** "
                f"(rough estimate ~{bars_back} {unit} at a normal pace), or **(b)** buy when price "
                f"**closes above {r1:.3f}** on high volume (breakout), stop below {r1 - atr:.3f}.")
    elif bearish:
        action = "🔴 AVOID BUYING / CONSIDER SELLING"
        when = (f"If you hold it: consider selling or reducing, especially if price **closes below {s1:.3f}**. "
                f"A bounce to **{sell_lo:.3f}–{sell_hi:.3f}** is a better place to sell. "
                f"Don't buy until it closes back above EMA20 ({last['EMA20']:.3f}).")
    else:
        action = "⚪ WAIT"
        when = (f"No clear direction. A **BUY** signal appears if price bounces off **{s1:.3f}** with RSI rising / MACD crossing up, "
                f"**or** price closes above **{r1:.3f}**. A **SELL** signal if price closes below **{s1:.3f}** "
                f"(next downside target {s2:.3f}).")

    return {
        "action": action, "when": when, "entry": entry, "buy_lo": buy_lo, "buy_hi": buy_hi,
        "stop": stop, "t1": r1, "t2": r2, "sell_lo": sell_lo, "sell_hi": sell_hi,
        "rr1": rr1, "rr2": rr2, "bars_t1": bars_t1, "atr": atr, "atr_mult": atr_mult, "s1": s1, "s2": s2,
    }


def analyse(df):
    d = add_indicators(df)
    sup, res = find_levels(d)
    sc, label, why = score_signal(d, sup, res)
    return d, sup, res, sc, label, why


# ----------------------------------------------------------------------------
# Trend forecast
# ----------------------------------------------------------------------------
def trend_forecast(d: pd.DataFrame, horizon: int = 10, lookback: int = 60):
    """Project the recent trend forward.

    Centre line: log-linear regression slope over `lookback` bars, continued from the last close.
    Bands: volatility cone from the std-dev of log returns (68% ≈ ±1σ√h, 95% ≈ ±1.96σ√h).
    This is a statistical projection of the current trend, not a prediction.
    """
    close = d["Close"].tail(lookback).astype(float).values
    y = np.log(close)
    x = np.arange(len(y))
    slope, intercept = np.polyfit(x, y, 1)
    fit = intercept + slope * x
    var_y = y.var()
    r2 = float(1 - (y - fit).var() / var_y) if var_y > 0 else 0.0

    short = np.log(d["Close"].tail(20).astype(float).values)
    slope_short = float(np.polyfit(np.arange(len(short)), short, 1)[0])

    rets = np.diff(np.log(d["Close"].tail(lookback + 1).astype(float).values))
    sigma = float(rets.std()) if len(rets) > 2 else 0.0

    last = float(d["Close"].iloc[-1])
    h = np.arange(1, horizon + 1)
    centre = last * np.exp(slope * h)
    band1 = sigma * np.sqrt(h)
    band2 = 1.96 * sigma * np.sqrt(h)

    per_bar = (np.exp(slope) - 1) * 100
    total = (np.exp(slope * horizon) - 1) * 100
    if r2 < 0.25 or abs(total) < 0.5 * band1[-1] * 100:
        direction, arrow = "Sideways", "→"
    elif slope > 0:
        direction, arrow = "Uptrend", "↗"
    else:
        direction, arrow = "Downtrend", "↘"
    strength = "Strong" if r2 >= 0.7 else ("Moderate" if r2 >= 0.45 else "Weak")
    if direction == "Sideways":
        strength = "Flat"  # a clean but tiny slope is still no usable trend
    if direction == "Sideways":
        momentum = "No clear direction recently"
    elif np.sign(slope_short) != np.sign(slope):
        momentum = "Short-term move is going against the trend: possible turn"
    elif abs(slope_short) > 1.5 * abs(slope):
        momentum = "Accelerating (last 20 bars steeper than the trend)"
    elif abs(slope_short) < 0.5 * abs(slope):
        momentum = "Slowing down (last 20 bars flatter than the trend)"
    else:
        momentum = "Steady"

    return {
        "horizon": horizon, "lookback": lookback,
        "fit": np.exp(fit), "centre": centre,
        "lo1": centre * np.exp(-band1), "hi1": centre * np.exp(band1),
        "lo2": centre * np.exp(-band2), "hi2": centre * np.exp(band2),
        "r2": r2, "per_bar": per_bar, "total": total,
        "direction": direction, "arrow": arrow, "strength": strength, "momentum": momentum,
    }


def first_cross(path, level, up=True):
    for i, v in enumerate(path, 1):
        if (up and v >= level) or (not up and v <= level):
            return i
    return None


def future_labels(d, n, interval):
    last = d.index[-1]
    if interval == "1d":
        idx = pd.bdate_range(last + pd.Timedelta(days=1), periods=n)
        return [f"{t.strftime('%d %b %y')} ·" for t in idx]
    if interval == "1wk":
        idx = pd.date_range(last + pd.Timedelta(days=7), periods=n, freq="7D")
        return [f"{t.strftime('%d %b %y')} ·" for t in idx]
    if interval == "1mo":
        idx = pd.date_range(pd.Timestamp(last).tz_localize(None) + pd.offsets.MonthBegin(1), periods=n, freq="MS")
        return [f"{t.strftime('%b %Y')} ·" for t in idx]
    step = {"1h": "h", "4h": "×4h"}.get(interval, "×15m")
    return [f"+{k}{step}" for k in range(1, n + 1)]


# ----------------------------------------------------------------------------
# Theme (soft dark, easy on the eyes)
# ----------------------------------------------------------------------------
P = dict(
    bg="#10151f", card="#171d29", card2="#1c2331", border="#262f40", text="#dfe3ea",
    muted="#8890a2", up="#3fb68b", down="#e0646a", accent="#7b93f5", warn="#e3a34b",
    grid="#1f2737", ema20="#e3c46a", ema50="#6fa8ef", ema200="#b48ae0",
)
LABEL_COLOR = {"STRONG BUY": "#3fb68b", "BUY": "#62c49e", "NEUTRAL": "#8890a2",
               "SELL": "#e88a8e", "STRONG SELL": "#e0646a"}
LABEL_DOT = {"STRONG BUY": "🟢", "BUY": "🟢", "NEUTRAL": "⚪", "SELL": "🔴", "STRONG SELL": "🔴"}


def rgba(hex_color, a):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a})"


def pill(text, color):
    return (f"<span class='pill' style='background:{rgba(color, .16)};color:{color};"
            f"border:1px solid {rgba(color, .35)}'>{text}</span>")


# ----------------------------------------------------------------------------
# Volume analysis
# ----------------------------------------------------------------------------
SPIKE_RVOL, UNUSUAL_RVOL = 2.0, 3.0


def volume_analysis(d: pd.DataFrame, bar_word: str = "day"):
    """Is the price action backed by meaningful trading volume?"""
    x = d.dropna(subset=["VolAvg20"])
    if len(x) < 25 or float(x["Volume"].tail(20).sum()) == 0:
        return None
    last, prev = x.iloc[-1], x.iloc[-2]
    rvol = float(last["RVOL"]) if pd.notna(last["RVOL"]) else 0.0
    atr = float(last["ATR"])
    chg = (float(last["Close"]) / float(prev["Close"]) - 1) * 100
    rng = float(last["High"] - last["Low"])
    body = abs(float(last["Close"] - last["Open"]))

    # participation: last 5 bars vs 20-bar average
    part = float(x["Volume"].tail(5).mean() / last["VolAvg20"]) if last["VolAvg20"] else np.nan

    # up-volume vs down-volume (accumulation / distribution) over 20 bars
    t20 = x.tail(20)
    up_v = float(t20.loc[t20["Close"] > t20["Open"], "Volume"].sum())
    dn_v = float(t20.loc[t20["Close"] < t20["Open"], "Volume"].sum())
    ud = up_v / dn_v if dn_v > 0 else (np.inf if up_v > 0 else 1.0)

    # OBV vs price over 20 bars
    p_chg = float(t20["Close"].iloc[-1] - t20["Close"].iloc[0])
    o_chg = float(t20["OBV"].iloc[-1] - t20["OBV"].iloc[0])
    obv_div = (p_chg > 0 > o_chg) or (p_chg < 0 < o_chg)

    # spikes in the last 60 bars
    t60 = x.tail(60)
    spikes = []
    for ts, r in t60[t60["RVOL"] >= SPIKE_RVOL].iterrows():
        i = x.index.get_loc(ts)
        pc = (float(r["Close"]) / float(x["Close"].iloc[i - 1]) - 1) * 100 if i > 0 else 0.0
        spikes.append({"date": ts, "rvol": float(r["RVOL"]), "chg": pc})

    # most recent 20-bar breakout / breakdown within the last 10 bars, and its volume
    brk = None
    t10 = x.tail(10)
    for ts, r in t10.iloc[::-1].iterrows():
        if pd.notna(r["HH20"]) and r["Close"] > r["HH20"]:
            brk = {"type": "Breakout", "date": ts, "level": float(r["HH20"]),
                   "rvol": float(r["RVOL"]) if pd.notna(r["RVOL"]) else 0.0}
            break
        if pd.notna(r["LL20"]) and r["Close"] < r["LL20"]:
            brk = {"type": "Breakdown", "date": ts, "level": float(r["LL20"]),
                   "rvol": float(r["RVOL"]) if pd.notna(r["RVOL"]) else 0.0}
            break
    if brk:
        brk["bars_ago"] = len(x) - 1 - x.index.get_loc(brk["date"])
        brk["grade"] = ("Confirmed" if brk["rvol"] >= 1.5 else "Moderate" if brk["rvol"] >= 1.0 else "Weak")

    # unusual activity flags (latest bar)
    flags = []
    if rvol >= UNUSUAL_RVOL:
        flags.append(("🚨", f"Unusual activity: volume is {rvol:.1f}× normal", P["warn"]))
    elif rvol >= SPIKE_RVOL:
        flags.append(("⚡", f"Volume spike: {rvol:.1f}× normal", P["warn"]))
    if rvol >= SPIKE_RVOL and rng > 0 and body < 0.3 * rng and abs(chg) < 1:
        flags.append(("🧲", "Heavy volume but price barely moved (churning): big players may be quietly "
                            "buying or selling. Watch which way it breaks.", P["accent"]))
    if rng > 1.5 * atr and rvol < 0.8:
        flags.append(("🫥", f"Big price move on light volume ({rvol:.1f}×): less reliable, may reverse.", P["muted"]))
    if obv_div:
        flags.append(("↔️", "OBV divergence: volume flow disagrees with the price trend (early warning).",
                      P["warn"]))
    recent_unusual = [s_ for s_ in spikes if s_["rvol"] >= UNUSUAL_RVOL and (x.index[-1] - s_["date"]).days <= 14]
    if recent_unusual and rvol < UNUSUAL_RVOL:
        s0 = recent_unusual[-1]
        flags.append(("🔎", f"Unusual volume {s0['rvol']:.1f}× on {s0['date'].strftime('%d %b')} "
                            f"({s0['chg']:+.1f}%): something happened; check the news.", P["accent"]))

    # verdict: is the move supported?
    pts, why = 0, []
    move_up = float(x["Close"].iloc[-1]) >= float(x["Close"].iloc[-6])
    direction = "up" if move_up else "down"
    if abs(chg) >= 0.5:
        if rvol >= 1.2:
            pts += 2
            why.append(f"Latest {bar_word} moved {chg:+.1f}% on {rvol:.1f}× normal volume: a real move.")
        elif rvol < 0.8:
            pts -= 2
            why.append(f"Latest {bar_word} moved {chg:+.1f}% on only {rvol:.1f}× normal volume: weak conviction.")
        else:
            why.append(f"Latest {bar_word} moved {chg:+.1f}% on about normal volume ({rvol:.1f}×).")
    else:
        why.append(f"Latest {bar_word} was quiet ({chg:+.1f}%, volume {rvol:.1f}× normal).")
    if (move_up and ud >= 1.3) or (not move_up and ud <= 0.77):
        pts += 2
        why.append(f"Volume on {'up' if move_up else 'down'} candles dominates (up/down volume {ud:.2f}): "
                   f"{'buyers are accumulating' if move_up else 'sellers are distributing'}, which supports the "
                   f"{direction}-move.")
    elif (move_up and ud <= 0.9) or (not move_up and ud >= 1.1):
        pts -= 2
        why.append(f"Price is going {direction} but the volume is mostly on the opposite side (up/down volume "
                   f"{ud:.2f}): the move isn't backed by real money.")
    if pd.notna(part):
        if part >= 1.2:
            pts += 1
            why.append(f"Participation is rising: last 5 {bar_word}s average {part:.1f}× the 20-{bar_word} norm.")
        elif part <= 0.7:
            pts -= 1
            why.append(f"Participation is fading: last 5 {bar_word}s average only {part:.1f}× the norm.")
    if brk:
        if brk["grade"] == "Confirmed":
            pts += 2
        elif brk["grade"] == "Weak":
            pts -= 2
    if obv_div:
        pts -= 1

    if pts >= 3:
        verdict, vcol = f"✅ Volume supports the {direction}-move", P["up"] if move_up else P["down"]
    elif pts <= -2:
        verdict, vcol = f"⚠️ The {direction}-move lacks volume support", P["warn"]
    else:
        verdict, vcol = "➖ Mixed: volume neither confirms nor rejects the move", P["muted"]

    return {"rvol": rvol, "vol": float(last["Volume"]), "avg": float(last["VolAvg20"]), "chg": chg,
            "part": part, "ud": ud, "obv_div": obv_div, "spikes": spikes, "brk": brk, "flags": flags,
            "verdict": verdict, "vcol": vcol, "why": why, "score": pts}


def fmt_vol(v):
    v = float(v)
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(v) >= div:
            return f"{v / div:.2f}{suf}"
    return f"{v:,.0f}"


# ----------------------------------------------------------------------------
# ADX trend regime
# ----------------------------------------------------------------------------
ADX_STRONG, ADX_WEAK = 25, 20
REGIME_COLOR_KEY = {"Strong Bullish Trend": "up", "Weak Bullish Trend": "up_soft", "Strong Bearish Trend": "down",
                    "Weak Bearish Trend": "down_soft", "Sideways Market": "muted"}


def adx_regime(adx, pdi, mdi):
    """ADX = strength only; +DI vs -DI = direction."""
    if pd.isna(adx) or pd.isna(pdi) or pd.isna(mdi):
        return "Sideways Market"
    if adx < ADX_WEAK:
        return "Sideways Market"
    bull = pdi >= mdi
    if adx >= ADX_STRONG:
        return "Strong Bullish Trend" if bull else "Strong Bearish Trend"
    return "Weak Bullish Trend" if bull else "Weak Bearish Trend"


def regime_color(regime):
    k = REGIME_COLOR_KEY.get(regime, "muted")
    return {"up": P["up"], "down": P["down"], "up_soft": "#7fcfae", "down_soft": "#e9989c"}.get(k, P["muted"])


def adx_analysis(d: pd.DataFrame, bar_word="day"):
    x = d.dropna(subset=["ADX", "PDI", "MDI"])
    if len(x) < 30:
        return None
    last = x.iloc[-1]
    adx, pdi, mdi = float(last["ADX"]), float(last["PDI"]), float(last["MDI"])
    regime = adx_regime(adx, pdi, mdi)
    adx5 = float(x["ADX"].iloc[-6])
    slope = adx - adx5
    rising = slope > 1.0
    falling = slope < -1.0
    # most recent DI crossover in the last 20 bars
    cross = None
    diff = (x["PDI"] - x["MDI"]).tail(21)
    sgn = np.sign(diff.values)
    for i in range(len(sgn) - 1, 0, -1):
        if sgn[i] != sgn[i - 1] and sgn[i] != 0:
            cross = {"bull": sgn[i] > 0, "bars_ago": len(sgn) - 1 - i, "adx": float(x["ADX"].iloc[-(len(sgn) - i)])}
            break
    band = ("above 25: a strong trend" if adx >= ADX_STRONG else
            "between 20 and 25: a weak trend" if adx >= ADX_WEAK else "below 20: no real trend")
    side = ("buyers (+DI) lead sellers (−DI)" if pdi >= mdi else "sellers (−DI) lead buyers (+DI)")
    notes = [f"ADX {adx:.1f} is {band}; {side} by {abs(pdi - mdi):.1f} points → <b>{regime}</b>."
             if regime != "Sideways Market" else
             f"ADX {adx:.1f} is {band}, so the +DI/−DI direction is ignored → <b>{regime}</b>."]
    if adx >= 40:
        notes.append(f"ADX {adx:.0f} is very high: a powerful trend, but trends this stretched can be late-stage.")
    if regime != "Sideways Market":
        if rising:
            notes.append(f"ADX is rising ({adx5:.0f} → {adx:.0f}): the trend is getting stronger.")
        elif not falling:
            notes.append(f"ADX is steady ({adx5:.0f} → {adx:.0f}): trend strength isn't changing much.")
        elif falling:
            notes.append(f"ADX is falling ({adx5:.0f} → {adx:.0f}): the trend is losing strength. Watch for a pause "
                         f"or reversal.")
    else:
        if rising:
            notes.append(f"ADX is rising from low levels ({adx5:.0f} → {adx:.0f}): a new trend may be starting.")
        else:
            notes.append("No trend: price is moving sideways, so breakouts often fail in this condition.")
    if cross:
        side = "+DI crossed above −DI (bullish)" if cross["bull"] else "−DI crossed above +DI (bearish)"
        when = "on the latest bar" if cross["bars_ago"] == 0 else f"{cross['bars_ago']} {bar_word}(s) ago"
        strength = "with ADX already above 20, so it carries weight" if cross["adx"] >= ADX_WEAK else \
                   "but ADX was low, so treat it with caution"
        notes.append(f"{side} {when}, {strength}.")
    playbook = {
        "Strong Bullish Trend": "Trend-following works best: buy pullbacks to support / EMA20, hold winners, "
                                "use a trailing stop. Avoid shorting.",
        "Weak Bullish Trend": "Up-bias but not strong yet: smaller positions, wait for ADX to climb above 25 or buy "
                              "only near support.",
        "Strong Bearish Trend": "Sellers in control: avoid buying dips, protect existing positions with stops. "
                                "Wait for ADX to fall and +DI to cross back up.",
        "Weak Bearish Trend": "Down-bias but weak: avoid new buys until +DI crosses above −DI; tighten stops.",
        "Sideways Market": "Range conditions: buy near support, sell near resistance, and be careful with "
                           "breakouts (many fail). Trend signals are less reliable here.",
    }[regime]
    return {"adx": adx, "pdi": pdi, "mdi": mdi, "regime": regime, "rising": rising, "falling": falling,
            "adx5": adx5, "notes": notes, "playbook": playbook, "cross": cross}


# ----------------------------------------------------------------------------
# Extra support/resistance methods (optional overlays)
# ----------------------------------------------------------------------------
EXTRA_OPTS = ["Swing highs/lows", "Repeated touches (3+)", "Volume profile", "Fibonacci retracement"]
FIB_RATIOS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]


def _pivots(x: pd.DataFrame, w: int = 5):
    hi, lo = x["High"].values, x["Low"].values
    highs, lows = [], []
    for i in range(w, len(x) - w):
        if hi[i] == hi[i - w:i + w + 1].max():
            highs.append((i, float(hi[i])))
        if lo[i] == lo[i - w:i + w + 1].min():
            lows.append((i, float(lo[i])))
    return highs, lows


def compute_extra_levels(d: pd.DataFrame, sel, show_bars: int = 120):
    """Optional extra S/R levels. Returns (overlay data for the chart, rows for the levels table)."""
    x = d.tail(show_bars)
    price, atr = float(d["Close"].iloc[-1]), float(d["ATR"].iloc[-1])
    out, rows = {}, []

    if "Swing highs/lows" in sel:
        highs, lows = _pivots(x, 5)
        out["swings"] = {"hi": highs, "lo": lows,
                         "lv_hi": [p for _, p in highs[-3:]], "lv_lo": [p for _, p in lows[-3:]]}
        rows += [{"Method": "Swing high", "Level": p, "Note": "recent swing"} for p in out["swings"]["lv_hi"]]
        rows += [{"Method": "Swing low", "Level": p, "Note": "recent swing"} for p in out["swings"]["lv_lo"]]

    if "Repeated touches (3+)" in sel:
        full = d.tail(250)
        hs, ls = _pivots(full, 3)
        pts = sorted([p for _, p in hs + ls])
        tol = max(0.4 * atr, price * 0.006)
        clusters = []
        for p in pts:
            if clusters and p - clusters[-1][-1] <= tol:
                clusters[-1].append(p)
            else:
                clusters.append([p])
        lo_v, hi_v = float(x["Low"].min()) * 0.97, float(x["High"].max()) * 1.03
        touch = [(float(np.mean(c)), len(c)) for c in clusters if len(c) >= 3 and lo_v <= np.mean(c) <= hi_v]
        touch = sorted(touch, key=lambda t: -t[1])[:6]
        out["touch"] = touch
        rows += [{"Method": "Repeated touches", "Level": l, "Note": f"touched {n}x"} for l, n in touch]

    if "Volume profile" in sel:
        tp = ((x["High"] + x["Low"] + x["Close"]) / 3).values
        vol = x["Volume"].fillna(0).values
        lo_p, hi_p = float(x["Low"].min()), float(x["High"].max())
        if hi_p > lo_p and vol.sum() > 0:
            edges = np.linspace(lo_p, hi_p, 31)
            idx = np.clip(np.digitize(tp, edges) - 1, 0, 29)
            prof = np.bincount(idx, weights=vol, minlength=30)
            poc_i = int(prof.argmax())
            # value area: expand from POC until 70% of volume is covered
            lo_i = hi_i = poc_i
            covered = prof[poc_i]
            while covered < 0.7 * prof.sum() and (lo_i > 0 or hi_i < 29):
                up = prof[hi_i + 1] if hi_i < 29 else -1
                dn = prof[lo_i - 1] if lo_i > 0 else -1
                if up >= dn:
                    hi_i += 1
                    covered += prof[hi_i]
                else:
                    lo_i -= 1
                    covered += prof[lo_i]
            mid = (edges[:-1] + edges[1:]) / 2
            out["vp"] = {"edges": edges, "vol": prof, "poc": float(mid[poc_i]),
                         "vah": float(edges[hi_i + 1]), "val": float(edges[lo_i])}
            rows += [{"Method": "Volume profile", "Level": out["vp"]["poc"], "Note": "POC (most traded price)"},
                     {"Method": "Volume profile", "Level": out["vp"]["vah"], "Note": "Value area high"},
                     {"Method": "Volume profile", "Level": out["vp"]["val"], "Note": "Value area low"}]

    if "Fibonacci retracement" in sel:
        y = x.tail(min(len(x), 120))
        hi_pos, lo_pos = int(y["High"].values.argmax()), int(y["Low"].values.argmin())
        hi_p, lo_p = float(y["High"].iloc[hi_pos]), float(y["Low"].iloc[lo_pos])
        up_move = lo_pos < hi_pos  # low came first -> retracing an up-move
        lv = [(r, hi_p - r * (hi_p - lo_p)) if up_move else (r, lo_p + r * (hi_p - lo_p)) for r in FIB_RATIOS]
        out["fib"] = {"levels": lv, "up": up_move, "hi": hi_p, "lo": lo_p}
        rows += [{"Method": "Fibonacci", "Level": p, "Note": f"{r * 100:.1f}% of the {'up' if up_move else 'down'}-move"}
                 for r, p in lv if 0 < r < 1]

    for r in rows:
        r["Distance %"] = (r["Level"] / price - 1) * 100
        r["Role"] = "Resistance (above)" if r["Level"] > price else "Support (below)"
    return out, rows


# ----------------------------------------------------------------------------
# Chart
# ----------------------------------------------------------------------------
def _draw_extras(fig, ex, labels):
    gold, teal, pink = "#e3c46a", "#5ec8d8", "#d68ad6"
    if "vp" in ex:
        vp = ex["vp"]
        mx = vp["vol"].max() or 1
        for i, v in enumerate(vp["vol"]):
            if v <= 0:
                continue
            w = 0.22 * v / mx
            mid = (vp["edges"][i] + vp["edges"][i + 1]) / 2
            col = P["warn"] if abs(mid - vp["poc"]) < 1e-9 else (
                P["accent"] if vp["val"] <= mid <= vp["vah"] else P["muted"])
            fig.add_shape(type="rect", xref="x domain", yref="y", x0=1 - w, x1=1,
                          y0=vp["edges"][i], y1=vp["edges"][i + 1], fillcolor=col, opacity=0.28,
                          line_width=0, layer="below")
        fig.add_hline(y=vp["poc"], line=dict(color=P["warn"], width=1.4, dash="dashdot"), row=1, col=1,
                      annotation_text=f"POC {vp['poc']:.3f}", annotation_position="bottom left",
                      annotation_font=dict(color=P["warn"], size=11))
        for k, lab in (("vah", "VAH"), ("val", "VAL")):
            fig.add_hline(y=vp[k], line=dict(color=rgba(P["accent"], .55), width=1, dash="dot"), row=1, col=1,
                          annotation_text=f"{lab} {vp[k]:.3f}", annotation_position="bottom left",
                          annotation_font=dict(color=P["accent"], size=10))
    if "fib" in ex:
        for r, p in ex["fib"]["levels"]:
            strong = r in (0.382, 0.5, 0.618)
            fig.add_hline(y=p, line=dict(color=rgba(gold, .75 if strong else .4), width=1.2 if strong else 0.9,
                                         dash="dot"), row=1, col=1,
                          annotation_text=f"Fib {r * 100:.1f}% · {p:.3f}", annotation_position="top left",
                          annotation_font=dict(color=gold, size=10))
    if "touch" in ex:
        for lvl, n in ex["touch"]:
            fig.add_hline(y=lvl, line=dict(color=rgba(teal, .8), width=0.8 + min(n, 8) * 0.15, dash="longdash"),
                          row=1, col=1, annotation_text=f"⟷ {n}x · {lvl:.3f}", annotation_position="top left",
                          annotation_font=dict(color=teal, size=10))
    if "swings" in ex:
        sw = ex["swings"]
        fig.add_trace(go.Scatter(mode="markers", x=[labels[i] for i, _ in sw["hi"]], y=[p * 1.006 for _, p in sw["hi"]],
                                 marker=dict(symbol="triangle-down", size=9, color=pink), name="Swing high",
                                 hovertemplate="Swing high %{customdata:.3f}<extra></extra>",
                                 customdata=[p for _, p in sw["hi"]]), 1, 1)
        fig.add_trace(go.Scatter(mode="markers", x=[labels[i] for i, _ in sw["lo"]], y=[p * 0.994 for _, p in sw["lo"]],
                                 marker=dict(symbol="triangle-up", size=9, color=teal), name="Swing low",
                                 hovertemplate="Swing low %{customdata:.3f}<extra></extra>",
                                 customdata=[p for _, p in sw["lo"]]), 1, 1)
        for p in sw["lv_hi"]:
            fig.add_hline(y=p, line=dict(color=rgba(pink, .45), width=1, dash="dot"), row=1, col=1)
        for p in sw["lv_lo"]:
            fig.add_hline(y=p, line=dict(color=rgba(teal, .45), width=1, dash="dot"), row=1, col=1)


CHART_PANELS = ["Volume", "RSI", "MACD", "ADX"]
CHART_OVERLAYS = ["EMA 20/50/200", "Support/resistance & zones", "Breakout markers"]


def make_chart(d, sup, res, plan, name, interval, fc=None, show_bars=120, extra=None,
               panels=tuple(CHART_PANELS), overlays=tuple(CHART_OVERLAYS)):
    x = d.tail(show_bars)
    fmt = {"1d": "%d %b %y", "1wk": "%d %b %y", "1mo": "%b %Y"}.get(interval, "%d %b %H:%M")
    labels = [ts.strftime(fmt) for ts in x.index]
    flabels = future_labels(d, fc["horizon"], interval) if fc else []

    panels = [p_ for p_ in CHART_PANELS if p_ in panels]
    rows = 1 + len(panels)
    ROW = {p_: i + 2 for i, p_ in enumerate(panels)}
    sub_h = 0.13 if rows > 1 else 0
    heights = [1 - sub_h * (rows - 1)] + [sub_h] * (rows - 1)
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.025 if rows > 1 else 0.02,
                        row_heights=heights)
    fig.add_trace(go.Candlestick(
        x=labels, open=x["Open"], high=x["High"], low=x["Low"], close=x["Close"], name=name,
        increasing=dict(line=dict(color=P["up"], width=1), fillcolor=P["up"]),
        decreasing=dict(line=dict(color=P["down"], width=1), fillcolor=P["down"])), 1, 1)
    if "EMA 20/50/200" in overlays:
        for n in (20, 50, 200):
            fig.add_trace(go.Scatter(x=labels, y=x[f"EMA{n}"], name=f"EMA{n}",
                                     line=dict(width=1.4, color=P[f"ema{n}"]), opacity=0.9), 1, 1)

    if extra:
        _draw_extras(fig, extra, labels)

    if fc:
        lb = min(fc["lookback"], len(labels))
        fig.add_trace(go.Scatter(mode="lines", x=labels[-lb:], y=fc["fit"][-lb:], name="Trend line",
                                 line=dict(color=P["accent"], width=1.2, dash="dot"), opacity=0.7), 1, 1)
        fx = [labels[-1]] + flabels
        last = float(x["Close"].iloc[-1])
        lo2, hi2 = [last] + list(fc["lo2"]), [last] + list(fc["hi2"])
        lo1, hi1 = [last] + list(fc["lo1"]), [last] + list(fc["hi1"])
        fig.add_trace(go.Scatter(mode="lines", x=fx, y=hi2, line=dict(width=0), showlegend=False, hoverinfo="skip"), 1, 1)
        fig.add_trace(go.Scatter(mode="lines", x=fx, y=lo2, fill="tonexty", fillcolor=rgba(P["accent"], .08),
                                 line=dict(width=0), name="95% range"), 1, 1)
        fig.add_trace(go.Scatter(mode="lines", x=fx, y=hi1, line=dict(width=0), showlegend=False, hoverinfo="skip"), 1, 1)
        fig.add_trace(go.Scatter(mode="lines", x=fx, y=lo1, fill="tonexty", fillcolor=rgba(P["accent"], .18),
                                 line=dict(width=0), name="68% range"), 1, 1)
        fig.add_trace(go.Scatter(mode="lines", x=fx, y=[last] + list(fc["centre"]), name="Forecast",
                                 line=dict(color=P["accent"], width=2.2, dash="dash")), 1, 1)
        # keep RSI/MACD x-axes aligned with the extended price axis
        for row in ROW.values():
            fig.add_trace(go.Scatter(mode="lines", x=flabels, y=[None] * len(flabels), showlegend=False, hoverinfo="skip"), row, 1)

    if "Support/resistance & zones" in overlays:
        fig.add_hrect(y0=plan["buy_lo"], y1=plan["buy_hi"], fillcolor=P["up"], opacity=0.10, line_width=0, row=1, col=1)
        fig.add_hrect(y0=plan["sell_lo"], y1=plan["sell_hi"], fillcolor=P["down"], opacity=0.10, line_width=0, row=1, col=1)
        for lvl, n in sup:
            fig.add_hline(y=lvl, line=dict(color=rgba(P["up"], .7), width=1.2), row=1, col=1,
                          annotation_text=f"S {lvl:.3f}" + (f" · {n}x" if n else ""), annotation_position="right",
                          annotation_font=dict(color=P["up"], size=11))
        for lvl, n in res:
            fig.add_hline(y=lvl, line=dict(color=rgba(P["down"], .7), width=1.2), row=1, col=1,
                          annotation_text=f"R {lvl:.3f}" + (f" · {n}x" if n else " · proj."), annotation_position="right",
                          annotation_font=dict(color=P["down"], size=11))
        fig.add_hline(y=plan["stop"], line=dict(color=P["warn"], width=1.1, dash="dash"), row=1, col=1,
                      annotation_text=f"Stop {plan['stop']:.3f}", annotation_position="bottom right",
                      annotation_font=dict(color=P["warn"], size=11))

    rv = x["RVOL"].fillna(0).values
    if "Volume" in ROW:
        # ---- volume panel
        rv = x["RVOL"].fillna(0).values
        up_bar = (x["Close"] >= x["Open"]).values
        vcolors = [P["warn"] if r >= SPIKE_RVOL else (rgba(P["up"], .55) if u else rgba(P["down"], .55))
                   for r, u in zip(rv, up_bar)]
        fig.add_trace(go.Bar(x=labels, y=x["Volume"], name="Volume", marker_color=vcolors, showlegend=False,
                             customdata=np.stack([rv], axis=-1),
                             hovertemplate="Vol %{y:,.0f} · RVOL %{customdata[0]:.1f}×<extra></extra>"), ROW['Volume'], 1)
        fig.add_trace(go.Scatter(x=labels, y=x["VolAvg20"], name="Avg volume (20)", mode="lines",
                                 line=dict(color=P["accent"], width=1.4)), ROW['Volume'], 1)
        sp = [i for i, r in enumerate(rv) if r >= SPIKE_RVOL]
        if sp:
            fig.add_trace(go.Scatter(mode="text", x=[labels[i] for i in sp], y=[float(x["Volume"].iloc[i]) for i in sp],
                                     text=["🚨" if rv[i] >= UNUSUAL_RVOL else "⚡" for i in sp], textposition="top center",
                                     showlegend=False, hoverinfo="skip", textfont=dict(size=11)), ROW['Volume'], 1)

    if "Breakout markers" in overlays:
        # breakout / breakdown markers on price, coloured by volume confirmation
        bo_x, bo_y, bo_t, bo_c, bd_x, bd_y, bd_t, bd_c = [], [], [], [], [], [], [], []
        for i in range(1, len(x)):
            r = x.iloc[i]
            if pd.isna(r["HH20"]) or pd.isna(r["LL20"]):
                continue
            rvi = float(r["RVOL"]) if pd.notna(r["RVOL"]) else 0
            prev_c = float(x["Close"].iloc[i - 1])
            if r["Close"] > r["HH20"] and prev_c <= float(x["HH20"].iloc[i - 1] or np.inf):
                bo_x.append(labels[i]); bo_y.append(float(r["Low"]) * 0.985)
                bo_t.append(f"Breakout · RVOL {rvi:.1f}× · {'confirmed' if rvi >= 1.5 else 'weak volume'}")
                bo_c.append(P["up"] if rvi >= 1.5 else P["muted"])
            elif r["Close"] < r["LL20"] and prev_c >= float(x["LL20"].iloc[i - 1] or -np.inf):
                bd_x.append(labels[i]); bd_y.append(float(r["High"]) * 1.015)
                bd_t.append(f"Breakdown · RVOL {rvi:.1f}× · {'confirmed' if rvi >= 1.5 else 'weak volume'}")
                bd_c.append(P["down"] if rvi >= 1.5 else P["muted"])
        if bo_x:
            fig.add_trace(go.Scatter(mode="markers", x=bo_x, y=bo_y, name="20-bar breakout", text=bo_t,
                                     hovertemplate="%{text}<extra></extra>",
                                     marker=dict(symbol="star-triangle-up", size=11, color=bo_c,
                                                 line=dict(width=1, color=P["bg"]))), 1, 1)
        if bd_x:
            fig.add_trace(go.Scatter(mode="markers", x=bd_x, y=bd_y, name="20-bar breakdown", text=bd_t,
                                     hovertemplate="%{text}<extra></extra>",
                                     marker=dict(symbol="star-triangle-down", size=11, color=bd_c,
                                                 line=dict(width=1, color=P["bg"]))), 1, 1)

    if "RSI" in ROW:
        fig.add_trace(go.Scatter(x=labels, y=x["RSI"], name="RSI", line=dict(color=P["ema200"], width=1.4),
                                 showlegend=False), ROW['RSI'], 1)
        fig.add_hrect(y0=30, y1=70, fillcolor=P["muted"], opacity=0.06, line_width=0, row=ROW['RSI'], col=1)
        fig.add_hline(y=70, line=dict(color=rgba(P["down"], .6), width=1, dash="dot"), row=ROW['RSI'], col=1)
        fig.add_hline(y=30, line=dict(color=rgba(P["up"], .6), width=1, dash="dot"), row=ROW['RSI'], col=1)
    if "MACD" in ROW:
        colors = np.where(x["MACDhist"] >= 0, rgba(P["up"], .75), rgba(P["down"], .75))
        fig.add_trace(go.Bar(x=labels, y=x["MACDhist"], name="MACD hist", marker_color=colors, showlegend=False), ROW['MACD'], 1)
        fig.add_trace(go.Scatter(x=labels, y=x["MACD"], name="MACD", line=dict(color=P["ema50"], width=1.2),
                                 showlegend=False), ROW['MACD'], 1)
        fig.add_trace(go.Scatter(x=labels, y=x["MACDsig"], name="Signal", line=dict(color=P["warn"], width=1.2),
                                 showlegend=False), ROW['MACD'], 1)
    if "ADX" in ROW:
        # ---- ADX panel
        fig.add_hrect(y0=0, y1=ADX_WEAK, fillcolor=P["muted"], opacity=0.07, line_width=0, row=ROW['ADX'], col=1)
        fig.add_hline(y=ADX_STRONG, line=dict(color=rgba(P["text"], .35), width=1, dash="dot"), row=ROW['ADX'], col=1,
                      annotation_text="25 strong", annotation_position="right",
                      annotation_font=dict(color=P["muted"], size=10))
        fig.add_hline(y=ADX_WEAK, line=dict(color=rgba(P["text"], .25), width=1, dash="dot"), row=ROW['ADX'], col=1,
                      annotation_text="20 trend / no trend", annotation_position="right",
                      annotation_font=dict(color=P["muted"], size=10))
        fig.add_trace(go.Scatter(x=labels, y=x["PDI"], name="+DI", line=dict(color=P["up"], width=1.2),
                                 showlegend=False, hovertemplate="+DI %{y:.1f}<extra></extra>"), ROW['ADX'], 1)
        fig.add_trace(go.Scatter(x=labels, y=x["MDI"], name="−DI", line=dict(color=P["down"], width=1.2),
                                 showlegend=False, hovertemplate="−DI %{y:.1f}<extra></extra>"), ROW['ADX'], 1)
        fig.add_trace(go.Scatter(x=labels, y=x["ADX"], name="ADX", line=dict(color=P["text"], width=2),
                                 showlegend=False, hovertemplate="ADX %{y:.1f}<extra></extra>"), ROW['ADX'], 1)

    fig.update_layout(
        template="plotly_dark", height=560 + 115 * (rows - 1), margin=dict(l=8, r=96, t=36, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=P["text"], size=12),
        xaxis_rangeslider_visible=False, hovermode="x unified",
        hoverlabel=dict(bgcolor=P["card2"], bordercolor=P["border"], font=dict(color=P["text"])),
        legend=dict(orientation="h", y=1.05, x=0, font=dict(size=11, color=P["muted"]), bgcolor="rgba(0,0,0,0)"),
        bargap=0.25,
    )
    fig.update_xaxes(type="category", nticks=8, showgrid=False, linecolor=P["border"], tickfont=dict(color=P["muted"]),
                     showspikes=True, spikecolor=P["muted"], spikethickness=1, spikedash="dot")
    fig.update_yaxes(gridcolor=P["grid"], zeroline=False, linecolor=P["border"], tickfont=dict(color=P["muted"]))
    # focus the price axis on price action + key levels (a far-away EMA200 shouldn't squash the candles)
    lo_v = [float(x["Low"].min()), sup[0][0], plan["stop"]]
    hi_v = [float(x["High"].max()), res[0][0]]
    if fc:
        lo_v.append(float(fc["lo1"][-1]))
        hi_v.append(float(fc["hi1"][-1]))
    lo_r, hi_r = min(lo_v), max(hi_v)
    pad = (hi_r - lo_r) * 0.05
    fig.update_yaxes(range=[lo_r - pad, hi_r + pad], row=1, col=1)
    tf_ = dict(color=P["muted"], size=11)
    if "Volume" in ROW:
        fig.update_yaxes(title_text="Volume", row=ROW["Volume"], col=1, title_font=tf_, rangemode="tozero")
    if "RSI" in ROW:
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=ROW["RSI"], col=1, title_font=tf_)
    if "MACD" in ROW:
        fig.update_yaxes(title_text="MACD", row=ROW["MACD"], col=1, title_font=tf_)
    if "ADX" in ROW:
        fig.update_yaxes(title_text="ADX / DI", row=ROW["ADX"], col=1, title_font=tf_, rangemode="tozero")
    return fig


def tradingview_widget(symbol: str, height=800, interval="D"):
    # Fixed pixel height (not autosize): autosize collapses inside Streamlit's iframe.
    html = f"""<!DOCTYPE html><html><head><style>
      html,body{{margin:0;padding:0;height:{height}px;background:{P['bg']};overflow:hidden}}
    </style></head><body>
    <div class="tradingview-widget-container" style="height:{height}px;width:100%">
      <div class="tradingview-widget-container__widget" style="height:{height}px;width:100%"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>
      {{"autosize": false, "width": "100%", "height": {height}, "symbol": "{symbol}", "interval": "{interval}",
        "timezone": "Asia/Kuala_Lumpur", "theme": "dark", "style": "1", "locale": "en",
        "backgroundColor": "{P['bg']}", "gridColor": "{P['grid']}",
        "allow_symbol_change": true, "withdateranges": true, "hide_side_toolbar": false,
        "details": true, "studies": ["RSI@tv-basicstudies", "MACD@tv-basicstudies"],
        "support_host": "https://www.tradingview.com"}}
      </script>
    </div></body></html>"""
    if hasattr(st, "iframe"):
        st.iframe(html, height=height + 4)
    else:
        components.html(html, height=height + 4)


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.markdown(f"""
<style>
.block-container{{padding-top:4.5rem;padding-bottom:3rem;max-width:1520px}}
h1,h2,h3,h4{{letter-spacing:-0.01em}}
/* metric cards */
[data-testid="stMetric"]{{background:{P['card']};border:1px solid {P['border']};border-radius:14px;padding:14px 18px}}
[data-testid="stMetricLabel"] p{{color:{P['muted']} !important;font-size:12px !important;text-transform:uppercase;letter-spacing:.05em}}
[data-testid="stMetricValue"]{{font-size:26px !important;font-weight:600}}
/* tabs */
.stTabs [data-baseweb="tab-list"]{{gap:4px;border-bottom:1px solid {P['border']}}}
.stTabs [data-baseweb="tab"]{{padding:10px 18px;border-radius:10px 10px 0 0;font-weight:600;color:{P['muted']}}}
.stTabs [aria-selected="true"]{{color:{P['text']} !important;background:{P['card']}}}
/* sidebar */
section[data-testid="stSidebar"]{{background:{P['card']};border-right:1px solid {P['border']}}}
/* generic pieces */
.hero{{display:flex;align-items:center;gap:14px;margin-bottom:6px}}
.hero .logo{{width:42px;height:42px;border-radius:12px;display:flex;align-items:center;justify-content:center;
  background:linear-gradient(135deg,{P['accent']},{P['up']});font-size:22px}}
.hero h1{{font-size:26px;margin:0;padding:0;font-weight:700}}
.hero .sub{{color:{P['muted']};font-size:13px;margin-top:2px}}
.live{{margin-left:auto;font-size:12px;color:{P['muted']};display:flex;align-items:center;gap:7px;
  background:{P['card']};border:1px solid {P['border']};padding:6px 12px;border-radius:999px}}
.live i{{width:8px;height:8px;border-radius:50%;background:{P['up']};display:inline-block;box-shadow:0 0 0 3px {rgba(P['up'], .2)}}}
.card{{background:{P['card']};border:1px solid {P['border']};border-radius:16px;padding:18px 20px;margin-bottom:14px}}
.card h3{{font-size:16px;margin:0 0 10px;font-weight:600}}
.card .muted{{color:{P['muted']};font-size:13px}}
.pill{{display:inline-block;padding:4px 12px;border-radius:999px;font-size:13px;font-weight:600;letter-spacing:.02em}}
.head{{display:flex;flex-wrap:wrap;align-items:flex-end;gap:18px 34px}}
.head .name{{font-size:15px;color:{P['muted']}}}
.head .price{{font-size:38px;font-weight:700;line-height:1.1}}
.head .chg{{font-size:16px;font-weight:600;margin-left:8px}}
.kv{{width:100%;border-collapse:collapse;border:none !important;margin:0 !important;display:table !important}}
.kv tr{{background:transparent !important;border:none !important}}
.kv td{{padding:8px 0 !important;border:none !important;border-bottom:1px solid {P['border']} !important;font-size:14px;background:transparent !important}}
.kv tr:last-child td{{border-bottom:none}}
.kv td:first-child{{color:{P['muted']}}}
.kv td:last-child{{text-align:right;font-weight:600}}
.reason{{display:flex;gap:10px;align-items:flex-start;padding:6px 0;font-size:14px;line-height:1.45}}
.reason b{{min-width:34px;text-align:right;font-weight:600}}
.note{{color:{P['muted']};font-size:12px;line-height:1.5;margin-top:8px}}
.action{{font-size:15px;font-weight:700;margin:2px 0 8px;line-height:1.35}}
.when{{font-size:13.5px;line-height:1.6;margin-top:12px;color:{P['text']};opacity:.9}}
a{{color:{P['accent']}}}
.news{{padding:10px 0;border-bottom:1px solid {P['border']}}}
.news:last-child{{border-bottom:none}}
.news a{{color:{P['text']};text-decoration:none;font-weight:600;font-size:14.5px;line-height:1.45}}
.news a:hover{{color:{P['accent']}}}
.news .meta{{color:{P['muted']};font-size:12px;margin-top:4px}}
.news .pill{{font-size:11px;padding:1px 8px}}
.just{{margin-top:auto;padding-top:10px;font-size:13px;line-height:1.55;color:{P['text']};opacity:.82;
  border-top:1px dashed {P['border']};margin-top:10px}}
.foot{{color:{P['muted']};font-size:12px;line-height:1.6;border-top:1px solid {P['border']};padding-top:14px;margin-top:26px}}
</style>""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    market = st.radio("Screener market", ["Bursa: all stocks", "Bursa: top 30 blue chips", "US", "My own list"],
                      index=0)
    custom = ""
    if market == "My own list":
        custom = st.text_area("Tickers (comma separated)", "MAYBANK, TENAGA, 5398, NVDA, TSLA, AAPL")
    shariah_set, shariah_asof = load_shariah()
    shariah_mode = "All"
    if shariah_set and market.startswith("Bursa"):
        shariah_mode = st.radio("Shariah status", ["All", "Shariah-compliant only", "Non-Shariah only"],
                                horizontal=False, help=f"Securities Commission Malaysia list ({shariah_asof})")
    bursa_filter = None
    if market == "Bursa: all stocks":
        with st.expander("🔧 Filter Bursa stocks", expanded=True):
            sectors_sel = st.multiselect("Sectors", YF_SECTORS + ["Other"], default=[],
                                         placeholder="All sectors")
            min_price = st.number_input("Min price (RM)", 0.0, 100.0, 0.10, 0.05, format="%.2f",
                                        help="Skip very cheap (penny) stocks")
            min_value = st.select_slider("Min daily trading value (RM)",
                                         [0, 10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000],
                                         value=100_000, format_func=lambda v: f"{v:,.0f}",
                                         help="Skip stocks that rarely trade (hard to buy/sell)")
            max_n = st.select_slider("Scan up to (most traded first)", [50, 100, 200, 300, 500, 1000],
                                     value=100, help="More stocks = slower first load")
        bursa_filter = (tuple(sectors_sel), min_price, min_value, max_n)
    tf_name = st.selectbox("Timeframe", list(INTERVALS.keys()), index=0)
    interval, period, unit = INTERVALS[tf_name]
    closed_only = st.toggle("Signals on completed candles only", value=True, key="closed_only",
                            help="While the market is open the latest candle is still forming and can change. "
                                 "When on, the Analysis tab calculates signals from the last COMPLETED candle "
                                 "(the live price is still shown).")
    st.divider()
    auto = st.toggle("Auto refresh", value=False)
    every = st.select_slider("Refresh every (seconds)", [30, 60, 120, 300], value=60, disabled=not auto)
    st.caption("Data: Yahoo Finance. Bursa is ~15 min delayed; US is near real-time during market hours. "
               "Cached for 60 seconds.")
    if DEMO:
        st.warning("DEMO MODE: fake data")

now_txt = pd.Timestamp.now(tz="Asia/Kuala_Lumpur").strftime("%d %b %Y, %H:%M")
st.markdown(f"""
<div class="hero"><div class="logo">📈</div>
<div><h1>Stock Screener & Analysis</h1>
<div class="sub">Bursa Malaysia & US · signals, support/resistance, trend forecast</div></div>
<div class="live"><i></i>Updated {now_txt} MYT</div></div>""", unsafe_allow_html=True)

bursa_all = pd.DataFrame()
sector_map = None
if market == "Bursa: all stocks":
    with st.spinner("Loading the full Bursa list… (first time only, ~20 seconds)"):
        bursa_all = load_bursa_all()
    if bursa_all.empty:
        st.warning("Couldn't load the full Bursa list right now, showing the top 30 instead.")
        universe = tuple(BURSA.keys())
    else:
        NAMES.update(dict(zip(bursa_all["ticker"], bursa_all["name"])))
        secs, min_price, min_value, max_n = bursa_filter
        f = bursa_all.copy()
        f["value"] = f["price"].fillna(0) * f["avg_volume"].fillna(0)
        if secs:
            f = f[f["sector"].isin(secs)]
        f = f[(f["price"].fillna(0) >= min_price) & (f["value"] >= min_value)]
        if shariah_mode == "Shariah-compliant only":
            f = f[f["ticker"].isin(shariah_set)]
        elif shariah_mode == "Non-Shariah only":
            f = f[~f["ticker"].isin(shariah_set)]
        f = f.sort_values("value", ascending=False)
        universe = tuple(f["ticker"].head(max_n))
        sector_map = dict(zip(bursa_all["ticker"], bursa_all["sector"]))
        with st.sidebar:
            st.caption(f"📋 {len(bursa_all):,} stocks on Bursa · {len(f):,} match your filter · "
                       f"scanning {len(universe):,}")
elif market == "Bursa: top 30 blue chips":
    universe = tuple(BURSA.keys())
    if shariah_mode == "Shariah-compliant only":
        universe = tuple(t for t in universe if t in shariah_set)
    elif shariah_mode == "Non-Shariah only":
        universe = tuple(t for t in universe if t not in shariah_set)
elif market == "US":
    universe = tuple(US.keys())
else:
    universe = tuple(dict.fromkeys(t for t in (normalise(s) for s in custom.split(",")) if t))

tab_scr, tab_term, tab_pk, tab_hm, tab_an, tab_div, tab_ipo, tab_tv, tab_learn = st.tabs(
    ["🔎  Screener", "🖥️  Terminal", "🎯  Picks", "🗺️  Heatmap", "📊  Analysis & Forecast", "💰  Dividends", "🚀  IPO",
     "📺  TradingView", "🎓  Learn"])
prices = {}

# ---------------- Screener ----------------
with tab_scr:
    rows = []
    if not universe:
        st.info("Enter at least one ticker in the sidebar.")
    else:
        with st.spinner(f"Scanning {len(universe)} stocks..."):
            data = get_many(universe, interval, period)
        prices = {t: float(v["Close"].iloc[-1]) for t, v in data.items() if len(v)}
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

# ---------------- Analysis ----------------
with tab_an:
    opts = list(universe) if universe else ["1155.KL"]
    c1, c2, c3, c4 = st.columns([2.2, 1.4, 1, 1])
    pick = c1.selectbox("Stock", opts, format_func=lambda t: f"{display_name(t)}  ·  {t}", key="pick")
    typed = c2.text_input("Or type a ticker", placeholder="e.g. GAMUDA, 5398, NVDA", key="an_typed")
    horizon = c3.selectbox("Forecast ahead", [5, 10, 20, 30], index=1,
                           format_func=lambda v: f"{v} {unit.split()[-1]}")
    lookback = c4.selectbox("Trend based on", [30, 60, 120], index=1, format_func=lambda v: f"last {v} bars")
    ticker = normalise(typed) if typed.strip() else pick
    st.session_state["an_ticker"] = ticker
    cur = "RM" if ticker.endswith(".KL") else "$"

    with st.spinner(f"Fetching {ticker}..."):
        df_raw = load_one(ticker, interval, period)

    if df_raw.empty or len(df_raw) < 60:
        st.error(f"Not enough data found for **{ticker}** on this timeframe ({len(df_raw)} candles). For Bursa use the "
                 f"4-digit code + .KL (e.g. 1155.KL) or a name from the list; for US use the normal ticker (e.g. NVDA).")
    else:
        dq, cstat = ta_plus.data_quality(df_raw, interval, ticker)
        live_price = float(df_raw["Close"].iloc[-1])
        df = df_raw.iloc[:-1] if (closed_only and cstat["forming"]) else df_raw
        d, sup, res, sc, label, why = analyse(df)
        atr_mult = float(st.session_state.get("ts_atr", 1.0))
        plan = trade_plan(d, sup, res, label, unit, atr_mult=atr_mult)
        fc = trend_forecast(d, horizon=horizon, lookback=min(lookback, len(d) - 2))
        bar_word_a = {"1d": "day", "1wk": "week", "1mo": "month", "4h": "4-hour bar", "1h": "hour",
                      "15m": "bar"}.get(interval, "bar")
        bar_word = bar_word_a
        ax = adx_analysis(d, bar_word_a)
        va = volume_analysis(d, bar_word)
        vol = ta_plus.volatility(d)
        last, prev = d.iloc[-1], d.iloc[-2]
        price = float(last["Close"])
        chg = (price / float(prev["Close"]) - 1) * 100
        chg_col = P["up"] if chg >= 0 else P["down"]
        dir_col = {"Uptrend": P["up"], "Downtrend": P["down"]}.get(fc["direction"], P["muted"])
        vol_col = {"Low": P["up"], "Moderate": P["warn"], "High": P["down"]}.get(vol["cls"] if vol else "", P["muted"])

        # ---- status badges (data integrity)
        badges = []
        if cstat["forming"]:
            badges.append(pill(f"● Live candle forming · live {cur}{live_price:,.3f}"
                               + (" · signals use last completed candle" if closed_only else ""), P["warn"]))
        if cstat["stale"]:
            badges.append(pill("⚠ Data looks stale: check the symbol or try again later", P["down"]))
        badges.append(pill(f"Market {'open' if cstat['open'] else 'closed'} · {cstat['exchange']}", P["muted"]))
        st.markdown(f"""
<div class="card"><div class="head">
  <div><div class="name">{display_name(ticker)} · {ticker}</div>
       <div><span class="price">{price:,.3f}</span><span class="chg" style="color:{chg_col}">{chg:+.2f}%</span></div></div>
  <div><div class="name">Signal</div>{pill(label, LABEL_COLOR[label])}</div>
  <div><div class="name">Score</div><div style="font-size:20px;font-weight:600">{sc:+d}<span class="muted"> / 100</span></div></div>
  <div><div class="name">Trend strength (ADX)</div>{pill(f"{ax['regime']} · {ax['adx']:.0f}", regime_color(ax['regime'])) if ax else pill("N/A", P['muted'])}</div>
  <div><div class="name">Volatility (ATR)</div>{pill(f"{vol['cls']} · {vol['atrp']:.2f}%", vol_col) if vol else pill("N/A", P['muted'])}</div>
  <div><div class="name">RSI</div><div style="font-size:20px;font-weight:600">{last['RSI']:.0f}</div></div>
  <div style="margin-left:auto" class="muted">Last candle used: {d.index[-1].strftime('%d %b %Y %H:%M')}<br>{tf_name} · {len(d)} candles
   · fetched {pd.Timestamp(dq['Fetched']).strftime('%H:%M:%S') if dq['Fetched'] != 'n/a' else 'n/a'}</div>
</div><div style="margin-top:10px;display:flex;gap:6px;flex-wrap:wrap">{''.join(badges)}</div></div>""",
                    unsafe_allow_html=True)

        # ---- chart controls
        k1, k2, k3, k4 = st.columns([0.8, 1.6, 1.4, 1.5])
        show_fc = k1.toggle("Trend forecast", value=True)
        extra_sel = k2.multiselect("Extra support / resistance", EXTRA_OPTS, default=[],
                                   placeholder="Swings, touches, volume profile, Fibonacci…", key="extra_sel")
        panels = k3.multiselect("Indicator panels", CHART_PANELS, default=CHART_PANELS, key="panels")
        overlays = k4.multiselect("Price overlays", CHART_OVERLAYS, default=CHART_OVERLAYS, key="overlays")
        extra, extra_rows = compute_extra_levels(d, extra_sel) if extra_sel else ({}, [])
        st.plotly_chart(make_chart(d, sup, res, plan, display_name(ticker), interval, fc if show_fc else None,
                                   extra=extra, panels=panels, overlays=overlays),
                        width="stretch", config={"displaylogo": False, "scrollZoom": True})
        legend_bits = []
        if "Support/resistance & zones" in overlays:
            legend_bits.append("green/red lines = support/resistance (×N touches) · green band = buy zone · red band = "
                               "sell zone · orange dashed = stop loss")
        if show_fc:
            legend_bits.append("blue dashed = trend forecast with nominal 68%/95% bands (see Forecast accuracy)")
        if "Volume" in panels:
            legend_bits.append(f"volume: blue line = 20-bar average, <span style='color:{P['warn']}'>orange ⚡</span> = "
                               f"spike 2×+, 🚨 = 3×+")
        if "Breakout markers" in overlays:
            legend_bits.append("★ = 20-bar breakout/breakdown (coloured = volume-confirmed, grey = weak volume)")
        if "ADX" in panels:
            legend_bits.append("ADX panel: white = ADX, green = +DI, red = −DI")
        extra_legend = {
            "Swing highs/lows": "▼▲ = swing highs/lows",
            "Repeated touches (3+)": "teal ⟷ = levels touched 3+ times",
            "Volume profile": "right-side bars = volume by price, POC/VAH/VAL",
            "Fibonacci retracement": "gold dotted = Fibonacci levels",
        }
        legend_bits += [extra_legend[k] for k in extra_sel]
        st.markdown("<div class='note'>" + " · ".join(legend_bits) + "</div>", unsafe_allow_html=True)

        # ---- other timeframes (real data only; used by MTF tab, audit and summary)
        with st.spinner("Loading other timeframes…"):
            tf_rows, tf_data = {}, {}
            for key, lab, iv, per in ta_plus.TIMEFRAMES:
                tdf = df_raw if key == interval else load_one(ticker, iv, per)
                if tdf is None or tdf.empty:
                    tf_rows[key] = {"available": False, "candles": 0, "reason": "no data from the source"}
                    continue
                tst = ta_plus.candle_status(tdf, key, ticker)
                use = tdf.iloc[:-1] if (closed_only and tst["forming"]) else tdf
                tf_data[key] = use
                r_ = ta_plus.tf_analysis(use, add_indicators)
                r_["forming"] = tst["forming"]
                r_["stale"] = tst["stale"]
                tf_rows[key] = r_
        mtf = ta_plus.mtf_summary(tf_rows)
        pats_full = ta_plus.pattern_report(d, sup, res)
        bt_df, bt_m = ta_plus.walk_forward(d["Close"].values, fc["horizon"], fc["lookback"])

        F = score_factors(d, sup, res)
        tabs = st.tabs(["🧠 Summary", "🧮 Audit", "📋 Trade plan", "⏱️ Timeframes", "📊 Volume", "📐 ADX",
                        "🕯️ Patterns", "🎯 Forecast", "📏 Levels", "🧾 Data"])

        # ================= Summary
        with tabs[0]:
            head, secs = ta_plus.narrative(display_name(ticker), ticker, d, sup, res, plan, va, ax, mtf, fc, vol,
                                           pats_full, cstat, (bt_df, bt_m))
            st.markdown(f"""
<div class="card" style="border-left:3px solid {P['accent']}">
<h3>🧠 Analysis summary</h3>
<div style="font-size:15.5px;line-height:1.7">{head}</div>
<div class="note">Written automatically from the indicator values calculated on this page (rule-based; every number
comes from the data, nothing is invented). Not investment advice.</div></div>""", unsafe_allow_html=True)
            # ---------------- Key numbers at a glance (every analysis, with justification)
            def kcard(title, verdict, vcol, rows, why):
                trs = "".join(f"<tr><td>{a}</td><td>{b}</td></tr>" for a, b in rows)
                return (f"<div class='card' style='margin:0;border-top:3px solid {vcol};display:flex;flex-direction:column'>"
                        f"<div style='display:flex;justify-content:space-between;align-items:center;gap:8px;"
                        f"flex-wrap:wrap'><h3 style='margin:0'>{title}</h3>{pill(verdict, vcol)}</div>"
                        f"<table class='kv' style='margin-top:10px'>{trs}</table>"
                        f"<div class='just'><b>Justification:</b> {why}</div></div>")

            nz = lambda v, f_="{:.3f}", na="N/A": na if v is None or (isinstance(v, float) and np.isnan(v)) else f_.format(v)  # noqa: E731
            cards_k = []

            # 1. Trade plan
            rr1v = plan["rr1"]
            rr_txt = "–" if np.isnan(rr1v) else f"1 : {rr1v:.2f}"
            rr_q = ("good (≥ 2)" if rr1v >= 2 else "acceptable (1.5–2)" if rr1v >= 1.5 else "poor (< 1.5)") \
                if not np.isnan(rr1v) else "not available"
            act_col = (P["up"] if "BUY" in plan["action"] and "DON'T" not in plan["action"] else
                       P["down"] if "AVOID" in plan["action"] else P["warn"] if "DON'T" in plan["action"] else P["muted"])
            cards_k.append(kcard(
                "📋 Trade plan", plan["action"].split(" ", 1)[1].title(), act_col,
                [("Buy zone", f"{cur}{plan['buy_lo']:.3f} – {plan['buy_hi']:.3f}"),
                 ("Reference entry", f"{cur}{plan['entry']:.3f}"),
                 ("Stop loss", f"{cur}{plan['stop']:.3f} ({(plan['stop'] / plan['entry'] - 1) * 100:.1f}%)"),
                 ("Target 1 / Target 2", f"{cur}{plan['t1']:.3f} / {cur}{plan['t2']:.3f}"),
                 ("Risk : Reward (T1)", rr_txt),
                 ("Sell zone", f"{cur}{plan['sell_lo']:.3f} – {plan['sell_hi']:.3f}")],
                f"Buy zone = nearest support {cur}{plan['s1']:.3f} ({sup[0][1] or 'projected'} touches) up to +0.5 ATR. "
                f"Stop sits {plan['atr_mult']:g} × ATR ({cur}{plan['atr']:.3f}) below support so normal daily noise "
                f"doesn't trigger it. Target 1 = nearest resistance ({res[0][1] or 'projected'} touches). "
                f"R:R is {rr_q}."))

            # 2. Trend forecast
            fc_col = {"Uptrend": P["up"], "Downtrend": P["down"]}.get(fc["direction"], P["muted"])
            bt_txt = (f"Back-tested direction accuracy {bt_m['dir_acc']:.0f}% over {bt_m['n']} past forecasts "
                      f"(skill vs 'no change' {bt_m['skill']:+.0f}%); the 95% band actually held the price "
                      f"{bt_m['cov95']:.0f}% of the time." if bt_m else "No back-test available.")
            reliab = ("reliable-ish" if bt_m and bt_m["dir_acc"] >= 60 and bt_m["skill"] > 0 else
                      "weak" if bt_m else "untested")
            cards_k.append(kcard(
                f"📈 Trend forecast · {fc['horizon']} {bar_word}s", f"{fc['arrow']} {fc['direction']}", fc_col,
                [("Projected price", f"{cur}{fc['centre'][-1]:.3f} ({fc['total']:+.1f}%)"),
                 ("Likely range (68%)", f"{cur}{fc['lo1'][-1]:.3f} – {fc['hi1'][-1]:.3f}"),
                 ("Wide range (95%)", f"{cur}{fc['lo2'][-1]:.3f} – {fc['hi2'][-1]:.3f}"),
                 ("Trend fit R²", f"{fc['r2']:.2f} ({fc['strength'].lower()})"),
                 ("Momentum", fc["momentum"]),
                 ("Forecast track record", reliab)],
                f"A straight trend line fitted to the last {fc['lookback']} {bar_word}s, extended forward. "
                f"R² {fc['r2']:.2f} = {'clean' if fc['r2'] >= 0.7 else 'moderate' if fc['r2'] >= 0.45 else 'choppy'} trend. "
                + bt_txt))

            # 3. Why this signal (original scoring)
            top = sorted(F, key=lambda f_: -abs(f_["Points"]))
            sig_rows = [(f_["Factor"], f"<span style='color:{P['up'] if f_['Points'] > 0 else P['down']}'>"
                                       f"{f_['Points']:+d}</span> · {f_['Reading']}") for f_ in top[:6]]
            cards_k.append(kcard(
                "🧮 Why this signal", f"{label} · {sc:+d}", LABEL_COLOR[label], sig_rows,
                f"Sum of {len(F)} scored factors = {sum(f_['Points'] for f_ in F):+d} (capped ±100). ≥ +40 Strong Buy, "
                f"≥ +15 Buy, ≤ −15 Sell, ≤ −40 Strong Sell. Trend (EMAs) "
                f"{sum(f_['Points'] for f_ in F if f_['Group'] == 'Trend'):+d}, momentum (RSI/MACD) "
                f"{sum(f_['Points'] for f_ in F if f_['Group'] == 'Momentum'):+d}, levels "
                f"{sum(f_['Points'] for f_ in F if f_['Group'] == 'Levels'):+d}. Full table in 🧮 Audit."))

            # 4. Volume
            if va:
                brk_v = (f"{va['brk']['type']} · {va['brk']['grade']} ({va['brk']['rvol']:.1f}×)" if va["brk"]
                         else "none in last 10 bars")
                ud_v = "∞" if np.isinf(va["ud"]) else f"{va['ud']:.2f}"
                cards_k.append(kcard(
                    "📊 Volume", va["verdict"].split(" ", 1)[1].split(":")[0], va["vcol"],
                    [("Latest volume", fmt_vol(va["vol"])), ("20-bar average", fmt_vol(va["avg"])),
                     ("Relative volume (RVOL)", f"{va['rvol']:.2f}×"),
                     ("Last 5 vs average", nz(va["part"], "{:.2f}×")),
                     ("Up / down volume (20)", ud_v), ("Breakout check", brk_v)],
                    " ".join(va["why"][:2]) + (" " + va["flags"][0][1] + "." if va["flags"] else "")))

            # 5. ADX
            if ax:
                cards_k.append(kcard(
                    "📐 Trend strength (ADX)", ax["regime"], regime_color(ax["regime"]),
                    [("ADX (14)", f"{ax['adx']:.1f} ({'rising' if ax['rising'] else 'falling' if ax['falling'] else 'flat'},"
                                  f" was {ax['adx5']:.1f})"),
                     ("+DI (buyers)", f"{ax['pdi']:.1f}"), ("−DI (sellers)", f"{ax['mdi']:.1f}"),
                     ("DI gap", f"{abs(ax['pdi'] - ax['mdi']):.1f} → {'buyers' if ax['pdi'] >= ax['mdi'] else 'sellers'} lead")],
                    f"ADX measures strength only: < 20 sideways, 20–25 weak, ≥ 25 strong. Direction comes from +DI vs −DI. "
                    + (ax["notes"][1] if len(ax["notes"]) > 1 else ax["notes"][0])))

            # 6. Volatility & position size
            if vol:
                cfg_s = {"capital": float(st.session_state.get("ts_capital", 10000.0)),
                         "risk_pct": float(st.session_state.get("ts_risk", 1.0)),
                         "max_loss": float(st.session_state.get("ts_maxloss", 0.0)),
                         "brk_pct": float(st.session_state.get("ts_brk", 0.08)),
                         "brk_min": float(st.session_state.get("ts_brkmin", 8.0)),
                         "platform": float(st.session_state.get("ts_plat", 0.0)),
                         "us_commission": float(st.session_state.get("ts_uscomm", 0.0))}
                e_ = ta_plus.round_tick(plan["entry"], ticker)
                s_ = ta_plus.round_tick(plan["stop"], ticker, "down")
                ps_ = ta_plus.position_size(e_, s_, ticker, cfg_s)
                sh_ = ps_["shares"] if ps_ else 0
                size_txt = (f"{sh_:,} shares" + (f" ({sh_ // 100} lots)" if ticker.endswith('.KL') else "")) if sh_ \
                    else "0 (stop too far for budget)"
                cards_k.append(kcard(
                    "🌡️ Volatility & position size", f"{vol['cls']} volatility", vol_col,
                    [("ATR (14)", f"{cur}{vol['atr']:.3f}"), ("ATR % of price", f"{vol['atrp']:.2f}%"),
                     ("Volatility rank", nz(vol["pct"], "{:.0f}th percentile")),
                     ("Risk budget", f"{cur}{ps_['budget']:,.2f}" if ps_ else "N/A"),
                     ("Position size", size_txt),
                     ("Required capital / max loss",
                      f"{cur}{ps_['cost']:,.0f} / {cur}{ps_['loss']:,.2f}" if sh_ else "–")],
                    f"Today's ATR% ranks {nz(vol['pct'], '{:.0f}')}th against this stock's last {vol['n']} candles. "
                    f"Risk per share = entry {cur}{e_:.3f} − stop {cur}{s_:.3f} = {cur}{e_ - s_:.3f}; budget "
                    f"{cfg_s['risk_pct']:g}% of {cur}{cfg_s['capital']:,.0f} ÷ that, rounded down to "
                    f"{'100-share lots' if ticker.endswith('.KL') else 'whole shares'}, fees included. Change the "
                    f"settings in 📋 Trade plan."))

            # 7. Multi-timeframe
            lab_k = {k: lab for k, lab, _, _ in ta_plus.TIMEFRAMES}
            mt_rows = [(lab_k[k], f"<span style='color:{P['up'] if 'Bullish' in r_['signal'] else P['down'] if 'Bearish' in r_['signal'] else P['muted']}'>"
                                  f"{r_['signal']}</span> · {r_['score']:+d}")
                       for k, r_ in tf_rows.items() if r_.get("available")]
            mt_col = {"Aligned bullish": P["up"], "Aligned bearish": P["down"], "Conflict": P["warn"]}.get(mtf["swing"], P["muted"])
            cards_k.append(kcard(
                "⏱️ Multi-timeframe", f"Swing: {mtf['swing']}", mt_col, mt_rows or [("Timeframes", "N/A")],
                f"Each timeframe scores trend, EMA alignment, RSI and MACD (±7). Weekly ×3, Daily ×2, 4H ×1 give a swing "
                f"score of {mtf['swing_score']:+d} / ±{mtf['swing_max']}. "
                + ("Timeframes disagree: follow the higher timeframe and wait for the lower one to line up."
                   if mtf["disagree"] else "No timeframes point in opposite directions.")))

            # 8. Candlestick patterns
            if pats_full:
                p0 = pats_full[0]
                perf0 = p0["perf"]
                pk_col = {"bull": P["up"], "bear": P["down"]}.get(p0["kind"], P["muted"])
                cards_k.append(kcard(
                    "🕯️ Candlestick pattern", f"{p0['name']}", pk_col,
                    [("When", "latest candle" if p0["bars_ago"] == 0 else f"{p0['bars_ago']} candle(s) ago"),
                     ("Type", {"bull": "Bullish", "bear": "Bearish"}.get(p0["kind"], "Neutral")),
                     ("Context", f"{p0['context']} ({p0['ctx_grade'].lower()})"),
                     ("Confirmation", p0["confirmation"]),
                     ("History in this stock", f"{perf0['win']:.0f}% worked over {perf0['n']} cases (random {perf0['base']:.0f}%)"
                      if perf0.get("win") is not None else f"insufficient (n={perf0['n']})"),
                     ("Other patterns (5 candles)", str(len(pats_full) - 1))],
                    p0["use"] + f" Detected because: {p0['rule']}."))
            else:
                cards_k.append(kcard("🕯️ Candlestick pattern", "None", P["muted"],
                                     [("Last 5 candles", "no recognised pattern")],
                                     "No pattern means no candle-based signal: rely on levels, trend and volume."))

            # 9. Forecast accuracy
            if bt_m:
                acc_col = P["up"] if bt_m["dir_acc"] >= 60 and bt_m["skill"] > 0 else (
                    P["warn"] if bt_m["dir_acc"] >= 50 else P["down"])
                cards_k.append(kcard(
                    "🎯 Forecast accuracy (back-test)", f"{bt_m['dir_acc']:.0f}% direction", acc_col,
                    [("Forecasts tested", f"{bt_m['n']}"), ("Directional accuracy", f"{bt_m['dir_acc']:.0f}% "
                                                                                   f"(always-up {bt_m['up_rate']:.0f}%)"),
                     ("MAE / RMSE", f"{cur}{bt_m['mae']:.3f} / {cur}{bt_m['rmse']:.3f} ({bt_m['mape']:.1f}%)"),
                     ("Skill vs 'no change'", f"{bt_m['skill']:+.0f}%"),
                     ("Band coverage 68% / 95%", f"{bt_m['cov68']:.0f}% / {bt_m['cov95']:.0f}%")],
                    f"Walk-forward test: at each past date the model saw only earlier data. Above 50% direction and "
                    f"positive skill mean it added value; bands are {ta_plus.calibration_note(bt_m)}."))

            st.markdown("<div class='sech' style='display:flex;align-items:center;gap:10px;margin:18px 0 8px'>"
                        "<span style='font-size:18px;font-weight:650'>📌 Key numbers from every analysis</span>"
                        f"<i style='flex:1;height:1px;background:{P['border']}'></i></div>", unsafe_allow_html=True)
            st.markdown("<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:12px'>"
                        + "".join(cards_k) + "</div>", unsafe_allow_html=True)
            st.markdown("<div class='sech' style='display:flex;align-items:center;gap:10px;margin:22px 0 8px'>"
                        "<span style='font-size:18px;font-weight:650'>📝 Detailed explanation</span>"
                        f"<i style='flex:1;height:1px;background:{P['border']}'></i></div>", unsafe_allow_html=True)
            order = ["Overall trend", "Short-term momentum", "Long-term trend", "Support & resistance",
                     "Volume behaviour", "RSI & MACD", "ADX trend strength", "Multi-timeframe", "Volatility",
                     "Bullish scenario", "Bearish scenario", "Suggested entry conditions",
                     "Risk factors & invalidation"]
            icons = {"Bullish scenario": "🟢", "Bearish scenario": "🔴", "Suggested entry conditions": "🎯",
                     "Risk factors & invalidation": "⚠️"}
            cards = "".join(
                f"<div class='card' style='margin:0'><div class='muted' style='font-size:12px;text-transform:uppercase;"
                f"letter-spacing:.05em'>{icons.get(k, '•')} {k}</div><div style='margin-top:6px;line-height:1.6'>"
                f"{secs[k]}</div></div>" for k in order if k in secs)
            st.markdown(f"<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:12px'>"
                        f"{cards}</div>", unsafe_allow_html=True)

        # ================= Signal audit
        with tabs[1]:
            F = score_factors(d, sup, res)
            total = sum(f["Points"] for f in F)
            st.markdown(f"""
<div class="card"><div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px">
<h3 style="margin:0">🧮 How the score was calculated</h3>{pill(f"{label} · {sc:+d}", LABEL_COLOR[label])}</div>
<div class="muted" style="margin-top:8px">Raw total {total:+d} (capped to ±100). Labels: ≥ +40 STRONG BUY · +15…+39 BUY ·
−14…+14 NEUTRAL · −15…−39 SELL · ≤ −40 STRONG SELL. Each row shows the actual reading, the rule and the points.</div></div>""",
                        unsafe_allow_html=True)
            aud = pd.DataFrame([{k: f[k] for k in ("Group", "Factor", "Reading", "Rule", "Points")} for f in F])
            st.dataframe(aud, hide_index=True, width="stretch",
                         column_config={"Points": st.column_config.NumberColumn(format="%+d", width="small")})
            grp = aud.groupby("Group", sort=False)["Points"].sum()
            gtxt = " · ".join(f"<b>{g}</b> {v:+d}" for g, v in grp.items())
            st.markdown(f"<div class='note'>Subtotals: {gtxt}. The three EMA checks share one <b>Trend</b> group "
                        f"(max ±35) because they tend to move together: read them as one trend vote.</div>",
                        unsafe_allow_html=True)
            # context rows (not scored, avoids double counting)
            m_, sg_, h_ = float(last["MACD"]), float(last["MACDsig"]), float(last["MACDhist"])
            zero_cross = None
            mz = d["MACD"].tail(6).values
            for k_ in range(len(mz) - 1, 0, -1):
                if mz[k_] > 0 >= mz[k_ - 1]:
                    zero_cross = f"MACD line crossed ABOVE zero {len(mz) - 1 - k_} bar(s) ago"
                    break
                if mz[k_] < 0 <= mz[k_ - 1]:
                    zero_cross = f"MACD line crossed BELOW zero {len(mz) - 1 - k_} bar(s) ago"
                    break
            ctx_rows = [
                {"Context": "MACD line", "Reading": f"{m_:.4f}", "Meaning": "fast EMA(12) − slow EMA(26)"},
                {"Context": "MACD signal line", "Reading": f"{sg_:.4f}", "Meaning": "9-period EMA of the MACD line"},
                {"Context": "MACD histogram", "Reading": f"{h_:+.4f}",
                 "Meaning": "MACD − signal: above 0 = line above signal (bullish momentum); below 0 = bearish momentum"},
                {"Context": "MACD vs zero line", "Reading": "above zero" if m_ > 0 else "below zero",
                 "Meaning": ("positive MACD regime (fast EMA above slow EMA)" if m_ > 0 else
                             "negative MACD regime (fast EMA below slow EMA)")
                            + (f" · {zero_cross} (zero-line cross ≠ signal-line crossover)" if zero_cross else "")},
                {"Context": "ADX trend strength", "Reading": f"{ax['adx']:.1f} · {ax['regime']}" if ax else "N/A",
                 "Meaning": "strength only; direction from +DI/−DI"},
                {"Context": "Volume confirmation", "Reading": va["verdict"] if va else "N/A",
                 "Meaning": f"RVOL {va['rvol']:.2f}×" if va else ""},
                {"Context": "Multi-timeframe (swing)", "Reading": mtf["swing"],
                 "Meaning": f"bullish on {', '.join(mtf['bulls']) or 'none'} · bearish on {', '.join(mtf['bears']) or 'none'}"},
            ]
            st.markdown("**Context checks (shown for transparency, not added to the score)**")
            st.dataframe(pd.DataFrame(ctx_rows), hide_index=True, width="stretch")
            st.markdown("<div class='note'>Why not scored: the MACD zero line and ADX direction largely repeat what the EMA "
                        "trend factors already measure, and volume / multi-timeframe are reported separately so the "
                        "original score stays comparable. MACD rules used: line above signal = bullish momentum; line "
                        "crossing above signal = bullish crossover; line above zero = positive regime. These are "
                        "different events and are reported separately.</div>", unsafe_allow_html=True)

        # ================= Trade plan & sizing
        with tabs[2]:
            st.markdown("##### ⚙️ Your trade settings")
            s1c, s2c, s3c, s4c = st.columns(4)
            capital = s1c.number_input(f"Trading capital ({cur})", min_value=100.0, value=10000.0, step=500.0,
                                       key="ts_capital")
            risk_pct = s2c.number_input("Risk per trade (%)", min_value=0.1, max_value=10.0, value=1.0, step=0.25,
                                        key="ts_risk")
            max_loss = s3c.number_input(f"Max loss per trade ({cur}, 0 = no cap)", min_value=0.0, value=0.0,
                                        step=10.0, key="ts_maxloss")
            s4c.select_slider("Stop distance (× ATR below support)", [0.5, 1.0, 1.5, 2.0, 2.5, 3.0], value=1.0,
                              key="ts_atr")
            with st.expander("Broker fees" + (" (Bursa)" if ticker.endswith(".KL") else " (US)")):
                f1, f2, f3, f4 = st.columns(4)
                brk_pct = f1.number_input("Brokerage (%)", 0.0, 1.0, 0.08, 0.01, format="%.3f", key="ts_brk")
                brk_min = f2.number_input("Minimum brokerage (RM)", 0.0, 50.0, 8.0, 1.0, key="ts_brkmin")
                platform = f3.number_input("Platform fee per order (RM)", 0.0, 20.0, 0.0, 0.5, key="ts_plat")
                us_comm = f4.number_input("US commission per order ($)", 0.0, 20.0, 0.0, 0.5, key="ts_uscomm")
                st.caption("Bursa: clearing 0.03% (max RM1,000) and stamp duty RM1 per RM1,000 (max RM1,000) are "
                           "added automatically on both buy and sell. Ordinary-share brokerage is SST-exempt; the "
                           "platform fee gets 8% SST. Set brokerage to your broker's rate.")
            cfg = {"capital": capital, "risk_pct": risk_pct, "max_loss": max_loss, "brk_pct": brk_pct,
                   "brk_min": brk_min, "platform": platform, "us_commission": us_comm}

            rt = lambda p_, m_="nearest": ta_plus.round_tick(p_, ticker, m_)  # noqa: E731
            entry = rt(plan["entry"])
            stop = rt(plan["stop"], "down")
            t1, t2 = rt(plan["t1"], "down"), rt(plan["t2"], "down")
            buy_lo, buy_hi = rt(plan["buy_lo"]), rt(plan["buy_hi"])
            ps = ta_plus.position_size(entry, stop, ticker, cfg)
            atr_v = plan["atr"]
            if vol:
                st.markdown(f"""
<div class="card"><h3>🌡️ Volatility (ATR 14)</h3>
<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px">
 <div><div class="muted">ATR (14)</div><div style="font-size:20px;font-weight:600">{cur}{vol['atr']:.3f}</div></div>
 <div><div class="muted">ATR % of price</div><div style="font-size:20px;font-weight:600">{vol['atrp']:.2f}%</div></div>
 <div><div class="muted">Volatility</div>{pill(vol['cls'], vol_col)}<div class="muted" style="font-size:12px">
   {f"{vol['pct']:.0f}th percentile of last {vol['n']} candles" if vol['pct'] is not None else 'needs ≥ 60 candles'}</div></div>
 <div><div class="muted">ATR stop</div><div style="font-size:20px;font-weight:600">{cur}{stop:.3f}</div>
   <div class="muted" style="font-size:12px">support {cur}{plan['s1']:.3f} − {atr_mult:g} × ATR</div></div>
</div>
<div class="note">ATR = average size of one {bar_word}'s move (Wilder, 14). Classification compares today's ATR% with this
stock's own last {vol['n']} candles: bottom third = Low, middle = Moderate, top third = High. Because the stop sits a fixed
number of ATRs away, position size automatically shrinks when the stock is more volatile.</div></div>""",
                            unsafe_allow_html=True)
            if ps is None or ps["shares"] == 0:
                st.warning("With these settings the position size is 0: the stop is too far for your risk budget "
                           "(or price exceeds capital). Lower the ATR multiple, raise risk %, or choose a cheaper stock.")
            if ps:
                sh = ps["shares"]
                lot = ps["lot"]
                be = ta_plus.break_even(sh, entry, ticker, cfg) if sh else None
                p1 = ta_plus.net_pnl(sh, entry, t1, ticker, cfg) if sh else 0
                p2 = ta_plus.net_pnl(sh, entry, t2, ticker, cfg) if sh else 0
                half1, half2 = ta_plus.split_partial(sh, lot)
                part = (ta_plus.net_pnl(half1, entry, t1, ticker, cfg) + ta_plus.net_pnl(half2, entry, t2, ticker, cfg)
                        if half2 else p1)
                part_be = (ta_plus.net_pnl(half1, entry, t1, ticker, cfg) +
                           (ta_plus.net_pnl(half2, entry, be, ticker, cfg) if be else 0)) if half2 else p1
                loss = ps["loss"]
                rr_net1 = p1 / loss if loss > 0 else np.nan
                rr_net2 = p2 / loss if loss > 0 else np.nan
                trail_now = max(stop, float(d["Close"].tail(10).max()) - 2 * atr_v)
                lots_txt = f" ({sh // 100} lots)" if lot == 100 else ""
                fb = ps["buy_fees"]["total"]
                plan_rows = [
                    ("Entry price (reference)", f"{cur}{entry:.3f}"),
                    ("Buy zone", f"{cur}{buy_lo:.3f} – {cur}{buy_hi:.3f}"),
                    ("Stop loss", f"{cur}{stop:.3f} ({(stop / entry - 1) * 100:.1f}%)"),
                    ("Target 1", f"{cur}{t1:.3f} ({(t1 / entry - 1) * 100:+.1f}%)"),
                    ("Target 2", f"{cur}{t2:.3f} ({(t2 / entry - 1) * 100:+.1f}%)"),
                    ("Risk : Reward (gross)", "–" if np.isnan(plan['rr1']) else f"1 : {plan['rr1']:.2f} / 1 : {plan['rr2']:.2f}"),
                    ("Risk : Reward (after fees)", "–" if np.isnan(rr_net1) else f"1 : {rr_net1:.2f} / 1 : {rr_net2:.2f}"),
                ]
                size_rows = [
                    ("Risk budget", f"{cur}{ps['budget']:,.2f} ({ps['binding']})"),
                    ("Position size", f"{sh:,} shares{lots_txt}"),
                    ("Required capital", f"{cur}{ps['cost']:,.2f} (incl. {cur}{fb:,.2f} buy fees)"),
                    ("Maximum monetary risk", f"{cur}{loss:,.2f} if stopped (incl. fees both ways)"),
                    ("Break-even price", f"{cur}{be:.3f}" if be else "N/A"),
                    ("Net profit at T1 / T2", f"{cur}{p1:,.2f} / {cur}{p2:,.2f}"),
                ]
                kv = lambda rows: "".join(f"<tr><td>{a}</td><td>{b}</td></tr>" for a, b in rows)  # noqa: E731
                pc1, pc2 = st.columns(2, gap="medium")
                when_html = re.sub(r"[*][*](.+?)[*][*]", r"<b>\1</b>", plan["when"])
                pc1.markdown(f"<div class='card'><h3>📋 Trade plan</h3><div class='action'>{plan['action']}</div>"
                             f"<table class='kv'>{kv(plan_rows)}</table><div class='when'>"
                             f"{when_html}</div>"
                             f"<div class='note'>Prices rounded to the exchange tick size"
                             f"{' (Bursa: 0.5 sen < RM1, 1 sen < RM10, 2 sen < RM100)' if ticker.endswith('.KL') else ''}."
                             f"</div></div>", unsafe_allow_html=True)
                pc2.markdown(f"<div class='card'><h3>🧮 Position sizing</h3><table class='kv'>{kv(size_rows)}</table>"
                             f"<div class='note'>Size = largest {'board-lot (100-share) ' if lot == 100 else ''}quantity "
                             f"whose loss at the stop, including buy and sell fees, stays within the smaller of "
                             f"{risk_pct:g}% of capital and your max-loss cap, and whose cost fits your capital.</div></div>",
                             unsafe_allow_html=True)
                mgmt = [
                    ("💰 Partial profit-taking",
                     f"Sell {half1:,} shares at T1 {cur}{t1:.3f} and {half2:,} at T2 {cur}{t2:.3f} → net "
                     f"{cur}{part:,.2f}. If the rest is stopped at break-even after T1: net {cur}{part_be:,.2f}."
                     if half2 else f"Position too small to split into board lots: sell all at T1 (net {cur}{p1:,.2f})."),
                    ("🔒 Trailing stop",
                     f"After T1, move the stop to break-even ({cur}{be:.3f}). Then trail it at the highest close − 2 × ATR "
                     f"(right now that would be {cur}{trail_now:.3f}); only ever move it up." if be else "N/A"),
                    ("⛔ Trade invalidation",
                     f"Exit if a {bar_word} closes below the stop {cur}{stop:.3f}; reassess if price closes below support "
                     f"{cur}{plan['s1']:.3f}, if ADX turns Strong Bearish, if a breakout entry falls back below "
                     f"{cur}{plan['t1']:.3f} on volume, or if T1 isn't reached within about "
                     f"{2 * (plan['bars_t1'] or 5)} {bar_word}s (time stop)."),
                ]
                st.markdown("<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px'>"
                            + "".join(f"<div class='card' style='margin:0'><b>{a}</b><div style='margin-top:6px;"
                                      f"line-height:1.6'>{b}</div></div>" for a, b in mgmt) + "</div>",
                            unsafe_allow_html=True)
                if ps["buy_fees"].get("brokerage") is not None and sh:
                    fb_ = ps["buy_fees"]
                    st.caption(f"Buy-side fees for {sh:,} shares: brokerage {cur}{fb_['brokerage']:.2f} · clearing "
                               f"{cur}{fb_['clearing']:.2f} · stamp duty {cur}{fb_['stamp']:.2f}"
                               + (f" · platform {cur}{fb_['platform']:.2f}" if fb_['platform'] else "")
                               + ". Sell-side fees are similar and are included in the risk and profit numbers.")

                # ---- paper trading journal
                st.markdown("##### 📝 Paper-trading journal")
                st.caption("Manual and paper trading only: this app never places real orders.")
                jcol1, jcol2 = st.columns([1, 3])
                if jcol1.button("➕ Add this plan as a paper trade", disabled=not sh):
                    st.session_state.setdefault("paper", []).append({
                        "Opened": pd.Timestamp.now(tz="Asia/Kuala_Lumpur").strftime("%Y-%m-%d %H:%M"),
                        "Ticker": ticker, "Name": display_name(ticker), "Entry": entry, "Stop": stop,
                        "T1": t1, "T2": t2, "Shares": sh})
                up = jcol2.file_uploader("Restore a saved journal (CSV)", type="csv", label_visibility="collapsed")
                if up is not None and not st.session_state.get("_paper_loaded"):
                    st.session_state["paper"] = pd.read_csv(up).to_dict("records")
                    st.session_state["_paper_loaded"] = True
                paper = st.session_state.get("paper", [])
                if paper:
                    rows_ = []
                    for tr_ in paper:
                        px = load_one(tr_["Ticker"], "1d", "1y")
                        now_p = float(px["Close"].iloc[-1]) if not px.empty else np.nan
                        since = px[px.index >= pd.Timestamp(tr_["Opened"][:10])] if not px.empty else px
                        status = "Open"
                        if len(since):
                            if float(since["Low"].min()) <= tr_["Stop"]:
                                status = "Stop hit"
                            elif float(since["High"].max()) >= tr_["T2"]:
                                status = "T2 hit"
                            elif float(since["High"].max()) >= tr_["T1"]:
                                status = "T1 hit"
                        pnl = ta_plus.net_pnl(int(tr_["Shares"]), tr_["Entry"], now_p, tr_["Ticker"], cfg) \
                            if not np.isnan(now_p) else np.nan
                        rows_.append({**tr_, "Now": now_p, "Status": status, "Net P/L (if sold now)": pnl})
                    jdf = pd.DataFrame(rows_)
                    st.dataframe(jdf, hide_index=True, width="stretch",
                                 column_config={"Net P/L (if sold now)": st.column_config.NumberColumn(format="%.2f"),
                                                "Now": st.column_config.NumberColumn(format="%.3f")})
                    st.caption("Journal is kept for this browser session. Copy it to keep it: select the table → "
                               "Download (top-right of the table), then restore it next time with the uploader above.")
                    if st.button("🗑️ Clear journal"):
                        st.session_state["paper"] = []
                        st.session_state["_paper_loaded"] = False
                        st.rerun()

        # ================= Multi-timeframe
        with tabs[3]:
            lab_of = {k: lab for k, lab, _, _ in ta_plus.TIMEFRAMES}
            avail = [k for k, _, _, _ in ta_plus.TIMEFRAMES if tf_rows.get(k, {}).get("available")]
            missing = [f"{lab_of[k]} ({tf_rows[k].get('reason', 'n/a')})" for k, _, _, _ in ta_plus.TIMEFRAMES
                       if not tf_rows.get(k, {}).get("available")]
            sw_col = {"Aligned bullish": P["up"], "Aligned bearish": P["down"], "Conflict": P["warn"]}.get(
                mtf["swing"], P["muted"])
            sig_col = lambda s_: P["up"] if "Bullish" in s_ else (P["down"] if "Bearish" in s_ else P["muted"])  # noqa: E731
            cards = ""
            for k in avail:
                r_ = tf_rows[k]
                sw = " · swing priority" if k in ta_plus.SWING_WEIGHTS else ""
                cards += f"""<div class='card' style='margin:0;border-top:3px solid {sig_col(r_['signal'])}'>
<div style='display:flex;justify-content:space-between;align-items:center'><b>{lab_of[k]}</b>{pill(r_['signal'], sig_col(r_['signal']))}</div>
<div class='muted' style='font-size:12px;margin:2px 0 8px'>{r_['candles']} candles{sw}{' · last candle excluded (forming)' if (r_.get('forming') and closed_only) else ''}{' · ⚠ stale' if r_.get('stale') else ''}</div>
<table class='kv'><tr><td>Trend</td><td>{r_['trend']}</td></tr><tr><td>EMA alignment</td><td style='font-weight:500'>{r_['ema']}</td></tr>
<tr><td>RSI</td><td>{r_['rsi']:.1f} · {r_['rsi_txt']}</td></tr><tr><td>MACD</td><td style='font-weight:500'>{r_['macd_txt']}</td></tr>
<tr><td>Score</td><td>{r_['score']:+d} / ±7</td></tr></table></div>"""
            warn = ""
            if mtf["disagree"]:
                warn = (f"<div class='reason'><span>⚠️</span><span style='color:{P['warn']}'>Timeframes disagree: bullish on "
                        f"{', '.join(lab_of[k] for k in mtf['bulls'])}; bearish on "
                        f"{', '.join(lab_of[k] for k in mtf['bears'])}. Lower timeframes often pull back against "
                        f"the bigger trend; trade in the direction of the higher timeframe.</span></div>")
            st.markdown(f"""
<div class="card" style="border-left:3px solid {sw_col}">
<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px">
<h3 style="margin:0">⏱️ Multi-timeframe analysis</h3>{pill('Swing (W/D/4H): ' + mtf['swing'], sw_col)}</div>
<div class="muted" style="margin-top:8px">For swing trading the Weekly (×3), Daily (×2) and 4-Hour (×1) timeframes are weighted
most: weighted swing score {mtf['swing_score']:+d} / ±{mtf['swing_max']}.</div>{warn}</div>""", unsafe_allow_html=True)
            st.markdown(f"<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px'>"
                        f"{cards}</div>", unsafe_allow_html=True)
            if missing:
                st.caption("Not shown (insufficient real data, nothing is filled in): " + "; ".join(missing))
            st.markdown("<div class='note'>Per-timeframe score (±7): trend ±2 (price vs EMA50 and its slope) · EMA "
                        "alignment ±2 (±1 if EMA200 unavailable) · RSI +1 (55–70) / −1 (30–45) · MACD ±1 vs signal line "
                        "and ±1 vs zero line. ≥ +2 Bullish, ≥ +5 Strong Bullish (same for bearish). EMA200 needs ≥ 250 "
                        "candles and EMA50 ≥ 100, otherwise shown as N/A. 4-Hour candles are built from real 1-hour "
                        "candles (Yahoo has no 4H feed).</div>", unsafe_allow_html=True)

        # ================= Volume
        with tabs[4]:
            # ---- Volume analysis card
            if va is None:
                st.info("Not enough volume data for a volume analysis.")
            else:
                rv_col = P["warn"] if va["rvol"] >= SPIKE_RVOL else (P["up"] if va["rvol"] >= 1.2 else
                                                                     (P["muted"] if va["rvol"] >= 0.8 else P["down"]))
                ud_txt = "∞" if np.isinf(va["ud"]) else f"{va['ud']:.2f}"
                ud_col = P["up"] if va["ud"] >= 1.3 else (P["down"] if va["ud"] <= 0.77 else P["muted"])
                ud_lab = "accumulation" if va["ud"] >= 1.3 else ("distribution" if va["ud"] <= 0.77 else "balanced")
                part_txt = "–" if pd.isna(va["part"]) else f"{va['part']:.2f}×"
                if va["brk"]:
                    b = va["brk"]
                    gcol = {"Confirmed": P["up"] if b["type"] == "Breakout" else P["down"], "Moderate": P["warn"],
                            "Weak": P["muted"]}[b["grade"]]
                    when = "latest bar" if b["bars_ago"] == 0 else f"{b['bars_ago']} bar(s) ago"
                    brk_html = (f"{pill(b['type'] + ' · ' + b['grade'], gcol)}<div class='muted' style='margin-top:6px'>"
                                f"{b['type']} of the 20-bar {'high' if b['type'] == 'Breakout' else 'low'} "
                                f"{b['level']:.3f} ({when}) on <b>{b['rvol']:.1f}×</b> normal volume. "
                                + ("Strong volume = real buyers behind it." if b['grade'] == 'Confirmed' and b['type'] == 'Breakout'
                                   else "Strong volume = real selling behind it." if b['grade'] == 'Confirmed'
                                   else "Volume is so-so: wait for follow-through." if b['grade'] == 'Moderate'
                                   else "Low volume: higher risk of a fakeout.") + "</div>")
                else:
                    brk_html = "<div class='muted'>No 20-bar breakout or breakdown in the last 10 bars.</div>"
                flags_html = "".join(f"<div class='reason'><span>{ic}</span><span style='color:{c}'>{t}</span></div>"
                                     for ic, t, c in va["flags"]) or "<div class='muted'>Nothing unusual right now.</div>"
                why_html = "".join(f"<div class='reason'><span>•</span><span>{w}</span></div>" for w in va["why"])
                spk = va["spikes"][-6:][::-1]
                spk_html = "".join(
                    f"<div class='reason'><span style='min-width:70px'>{z['date'].strftime('%d %b')}</span>"
                    f"<b style='color:{P['warn']};min-width:52px'>{z['rvol']:.1f}×</b>"
                    f"<span style='color:{P['up'] if z['chg'] >= 0 else P['down']}'>{z['chg']:+.1f}%</span>"
                    f"<span class='muted'>{'🚨 unusual' if z['rvol'] >= UNUSUAL_RVOL else '⚡ spike'}</span></div>"
                    for z in spk) or "<div class='muted'>No spikes (2×+) in the last 60 bars.</div>"
                st.markdown(f"""
    <div class="card" style="border-left:3px solid {va['vcol']}">
    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px">
      <h3 style="margin:0">📊 Volume analysis</h3>{pill(va['verdict'], va['vcol'])}</div>
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:14px 0">
      <div><div class="muted">Volume (latest)</div><div style="font-size:20px;font-weight:600">{fmt_vol(va['vol'])}</div></div>
      <div><div class="muted">20-{bar_word} average</div><div style="font-size:20px;font-weight:600">{fmt_vol(va['avg'])}</div></div>
      <div><div class="muted">Relative volume (RVOL)</div><div style="font-size:20px;font-weight:600;color:{rv_col}">{va['rvol']:.2f}×</div></div>
      <div><div class="muted">Last 5 vs average</div><div style="font-size:20px;font-weight:600">{part_txt}</div></div>
      <div><div class="muted">Up / down volume (20)</div><div style="font-size:20px;font-weight:600;color:{ud_col}">{ud_txt}</div>
           <div class="muted" style="font-size:12px">{ud_lab}</div></div>
    </div>
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:18px">
      <div><b>Is the move backed by volume?</b>{why_html}</div>
      <div><b>Breakout confirmation</b><div style="margin-top:8px">{brk_html}</div>
           <div style="margin-top:14px"><b>Unusual activity</b></div>{flags_html}</div>
      <div><b>Recent volume spikes (last 60 bars)</b>{spk_html}</div>
    </div>
    <div class="note">RVOL = today's volume ÷ average of the previous 20 {bar_word}s. 1× = normal, 2×+ = spike, 3×+ = unusual.
    Price moves on high volume are more trustworthy; moves on low volume often fade. Up/down volume above 1 = more shares
    traded on green candles than red (buyers in control).</div></div>""", unsafe_allow_html=True)

        # ================= ADX
        with tabs[5]:
            if not ax:
                st.info("Not enough candles for ADX (needs about 30).")
            # ---- ADX card
            if ax:
                rc = regime_color(ax["regime"])
                gauge_w = min(ax["adx"], 60) / 60 * 100
                di_total = ax["pdi"] + ax["mdi"] or 1
                pdi_w = ax["pdi"] / di_total * 100
                trend_arrow = "↗ rising" if ax["rising"] else ("↘ falling" if ax["falling"] else "→ flat")
                notes_html = "".join(f"<div class='reason'><span>•</span><span>{n}</span></div>" for n in ax["notes"])
                regimes = ["Strong Bearish Trend", "Weak Bearish Trend", "Sideways Market", "Weak Bullish Trend",
                           "Strong Bullish Trend"]
                scale = "".join(
                    f"<div style='flex:1;text-align:center;padding:7px 4px;border-radius:8px;font-size:12px;"
                    f"background:{rgba(regime_color(r_), .22 if r_ == ax['regime'] else .06)};"
                    f"color:{regime_color(r_) if r_ == ax['regime'] else P['muted']};"
                    f"font-weight:{700 if r_ == ax['regime'] else 500};"
                    f"border:1px solid {rgba(regime_color(r_), .6) if r_ == ax['regime'] else P['border']}'>{r_}</div>"
                    for r_ in regimes)
                st.markdown(f"""
    <div class="card" style="border-left:3px solid {rc}">
    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px">
      <h3 style="margin:0">📐 Trend strength (ADX 14)</h3>{pill(ax['regime'], rc)}</div>
    <div style="display:flex;gap:6px;margin:14px 0 16px;flex-wrap:wrap">{scale}</div>
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:20px">
     <div>
      <div class="muted">ADX: how strong (not which way)</div>
      <div style="display:flex;align-items:baseline;gap:10px"><span style="font-size:28px;font-weight:700">{ax['adx']:.1f}</span>
       <span class="muted">{trend_arrow} (5 {bar_word_a}s ago {ax['adx5']:.1f})</span></div>
      <div style="position:relative;height:10px;border-radius:6px;background:{P['card2']};margin:8px 0 4px;overflow:hidden">
       <div style="position:absolute;left:0;top:0;bottom:0;width:{gauge_w:.0f}%;background:{rc};opacity:.85"></div>
       <div style="position:absolute;left:{20/60*100:.1f}%;top:0;bottom:0;width:2px;background:{P['bg']}"></div>
       <div style="position:absolute;left:{25/60*100:.1f}%;top:0;bottom:0;width:2px;background:{P['bg']}"></div></div>
      <div class="muted" style="font-size:11.5px;display:flex;justify-content:space-between">
       <span>0 · no trend</span><span>20</span><span>25 · strong</span><span>40+ · very strong</span></div>
     </div>
     <div>
      <div class="muted">+DI vs −DI: which side is winning (direction)</div>
      <div style="display:flex;justify-content:space-between;margin-top:6px;font-weight:600">
       <span style="color:{P['up']}">+DI {ax['pdi']:.1f} buyers</span><span style="color:{P['down']}">sellers −DI {ax['mdi']:.1f}</span></div>
      <div style="display:flex;height:10px;border-radius:6px;overflow:hidden;margin-top:6px">
       <div style="width:{pdi_w:.0f}%;background:{P['up']};opacity:.8"></div>
       <div style="flex:1;background:{P['down']};opacity:.8"></div></div>
      <div class="muted" style="font-size:11.5px;margin-top:4px">{'Buyers' if ax['pdi'] >= ax['mdi'] else 'Sellers'} have the upper hand
       (gap {abs(ax['pdi'] - ax['mdi']):.1f})</div>
     </div>
    </div>
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:20px;margin-top:14px">
     <div><b>What the indicators say</b>{notes_html}</div>
     <div><b>How to trade this condition</b><div class="reason"><span>🧭</span><span>{ax['playbook']}</span></div></div>
    </div>
    <div class="note">ADX below 20 = sideways (direction ignored) · 20–25 = weak trend · 25+ = strong trend. The direction comes from
    +DI (buying pressure) vs −DI (selling pressure). ADX rises in strong up-trends <i>and</i> strong down-trends, so it never
    tells direction on its own. Chart: bottom panel, white = ADX, green = +DI, red = −DI.</div></div>""", unsafe_allow_html=True)
            st.write("")

        # ================= Patterns
        with tabs[6]:
            st.markdown("<div class='note'>Patterns in the last 5 candles. Each is checked against the trend before "
                        "it and nearby support/resistance; reversal patterns in the wrong place are marked low "
                        "reliability rather than turned into signals.</div>", unsafe_allow_html=True)
            if not pats_full:
                st.info("No candlestick pattern in the last 5 candles.")
            for p_ in pats_full:
                kc = {"bull": P["up"], "bear": P["down"]}.get(p_["kind"], P["muted"])
                kind_txt = {"bull": "Bullish", "bear": "Bearish"}.get(p_["kind"], "Neutral")
                perf = p_["perf"]
                if perf.get("win") is not None:
                    perf_txt = (f"{perf['n']} earlier times in this stock's history: price was "
                                f"{'higher' if p_['kind'] != 'bear' else 'lower'} {p_['fwd']} {bar_word}s later "
                                f"<b>{perf['win']:.0f}%</b> of the time (vs {perf['base']:.0f}% for any random "
                                f"{bar_word}), average move {perf['avg']:+.2f}%.")
                else:
                    perf_txt = f"Insufficient history (only {perf['n']} earlier occurrence(s); needs ≥ 5)."
                when = "latest candle" if p_["bars_ago"] == 0 else f"{p_['bars_ago']} candle(s) ago"
                use_col = (P["up"] if p_["use"].startswith("Actionable") else
                           P["down"] if p_["use"].startswith(("Ignore", "Low")) else P["warn"])
                st.markdown(f"""
<div class="card" style="border-left:3px solid {kc}">
<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px">
 <div><b style="font-size:16px">{p_['name']}</b> <span class="muted">· {when}</span></div>
 <div>{pill(kind_txt, kc)} {pill(p_['use'].split(':')[0], use_col)}</div></div>
<table class="kv" style="margin-top:10px">
<tr><td>Detection</td><td style="font-weight:500">{p_['rule']}</td></tr>
<tr><td>Context</td><td style="font-weight:500">{p_['context']} · context {p_['ctx_grade'].lower()}</td></tr>
<tr><td>Confirmation</td><td style="font-weight:500">{p_['confirmation']}</td></tr>
<tr><td>Historical performance</td><td style="font-weight:500">{perf_txt}</td></tr>
<tr><td>How to use</td><td style="font-weight:500">{p_['use']}</td></tr></table></div>""", unsafe_allow_html=True)
                with st.expander(f"Candle formation details · {p_['name']}"):
                    st.dataframe(pd.DataFrame(p_["candles"]), hide_index=True, width="stretch",
                                 column_config={c_: st.column_config.NumberColumn(format="%.3f")
                                                for c_ in ("Open", "High", "Low", "Close")}
                                 | {c_: st.column_config.NumberColumn(format="%.0f%%")
                                    for c_ in ("Body % of range", "Upper wick %", "Lower wick %")})
            st.markdown("<div class='note'>Confirmation = a later candle (within 3) closes beyond the pattern's high "
                        "(bullish) or low (bearish). Historical performance only uses earlier occurrences with a full "
                        "5-candle window afterwards (no look-ahead). See 🎓 Learn → Candlestick Patterns.</div>",
                        unsafe_allow_html=True)

        # ================= Forecast & accuracy
        with tabs[7]:
            # ---- Forecast
            h = fc["horizon"]
            unit_short = unit.split()[-1]
            cross_up = first_cross(fc["centre"], plan["t1"], up=True)
            cross_dn = first_cross(fc["centre"], sup[0][0], up=False)
            if fc["direction"] == "Uptrend":
                hint = (f"If this pace holds, price could reach resistance <b>{plan['t1']:.3f}</b> in about "
                        f"<b>{cross_up} {unit_short}</b>, a likely area for sellers / taking profit."
                        if cross_up else f"At this pace, price stays below resistance {plan['t1']:.3f} within {h} {unit_short}.")
            elif fc["direction"] == "Downtrend":
                hint = (f"If this pace holds, price could drop to support <b>{sup[0][0]:.3f}</b> in about "
                        f"<b>{cross_dn} {unit_short}</b>. Watch there for a bounce (buy) or a break (more downside)."
                        if cross_dn else f"At this pace, price stays above support {sup[0][0]:.3f} within {h} {unit_short}.")
            else:
                hint = (f"No reliable trend: price is moving sideways. Range traders buy near support "
                        f"{sup[0][0]:.3f} and sell near resistance {plan['t1']:.3f}.")
            st.markdown(f"""
    <div class="card"><h3>Trend forecast · next {h} {unit_short}</h3>
    <div style="margin-bottom:10px">{pill(f"{fc['arrow']} {fc['direction']}", dir_col)} {pill(f"{fc['strength']} trend · R² {fc['r2']:.2f}", P['muted'])}</div>
    <table class="kv">
    <tr><td>Projected price</td><td>{fc['centre'][-1]:.3f} <span class="muted">({fc['total']:+.1f}%)</span></td></tr>
    <tr><td>Likely range (68%)</td><td>{fc['lo1'][-1]:.3f} – {fc['hi1'][-1]:.3f}</td></tr>
    <tr><td>Wide range (95%)</td><td>{fc['lo2'][-1]:.3f} – {fc['hi2'][-1]:.3f}</td></tr>
    <tr><td>Trend speed</td><td>{fc['per_bar']:+.2f}% per bar</td></tr>
    <tr><td>Momentum</td><td style="font-weight:500">{fc['momentum']}</td></tr>
    </table>
    <div style="margin-top:12px;font-size:14px;line-height:1.5">{hint}</div>
    <div class="note">How it works: a trend line is fitted to the last {fc['lookback']} bars and extended forward; the bands
    come from the stock's normal volatility. It shows where price goes <i>if the current trend continues</i>. News, results and
    market moves can break any trend, so treat it as a guide, not a prediction. R² near 1 = clean trend, near 0 = choppy.</div>
    </div>""", unsafe_allow_html=True)
            st.markdown("#### 🎯 How accurate has this forecast been? (walk-forward test)")
            if bt_m is None:
                st.info("Not enough history for an out-of-sample test.")
            else:
                cov_note = ta_plus.calibration_note(bt_m)
                skill_col = P["up"] if bt_m["skill"] > 0 else P["down"]
                st.markdown(f"""
<div class="card"><div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px">
 <div><div class="muted">Forecasts tested</div><div style="font-size:20px;font-weight:600">{bt_m['n']}</div></div>
 <div><div class="muted">Directional accuracy</div><div style="font-size:20px;font-weight:600">{bt_m['dir_acc']:.0f}%</div>
   <div class="muted" style="font-size:12px">'always up' would score {bt_m['up_rate']:.0f}%</div></div>
 <div><div class="muted">MAE</div><div style="font-size:20px;font-weight:600">{cur}{bt_m['mae']:.3f}</div>
   <div class="muted" style="font-size:12px">{bt_m['mape']:.1f}% of price</div></div>
 <div><div class="muted">RMSE</div><div style="font-size:20px;font-weight:600">{cur}{bt_m['rmse']:.3f}</div></div>
 <div><div class="muted">vs 'no change' guess</div><div style="font-size:20px;font-weight:600;color:{skill_col}">{bt_m['skill']:+.0f}%</div>
   <div class="muted" style="font-size:12px">naive MAE {cur}{bt_m['naive_mae']:.3f}</div></div>
 <div><div class="muted">Band coverage 68% / 95%</div><div style="font-size:20px;font-weight:600">{bt_m['cov68']:.0f}% / {bt_m['cov95']:.0f}%</div>
   <div class="muted" style="font-size:12px">{cov_note}</div></div>
</div>
<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px;margin-top:14px">
 <div><b>1 · Model fit (in-sample)</b><div class="muted" style="margin-top:4px">R² {fc['r2']:.2f} = how neatly a straight
   trend line fits the last {fc['lookback']} {bar_word}s. It says nothing about the future.</div></div>
 <div><b>2 · Forecast accuracy (out-of-sample)</b><div class="muted" style="margin-top:4px">At {bt_m['n']} past dates the
   model was fitted using only data up to that date and asked to predict {fc['horizon']} {bar_word}s ahead; results above
   compare those predictions with what actually happened.</div></div>
 <div><b>3 · Uncertainty (bands)</b><div class="muted" style="margin-top:4px">The 68%/95% bands are nominal ranges from
   recent volatility, not guaranteed probabilities. Historically they contained the real price {bt_m['cov68']:.0f}% /
   {bt_m['cov95']:.0f}% of the time ({cov_note}).</div></div>
</div></div>""", unsafe_allow_html=True)
                x_lab = [str(d.index[int(t_ + fc['horizon'])])[:10] for t_ in bt_df["t"]]
                figb = go.Figure()
                figb.add_trace(go.Scatter(x=x_lab, y=bt_df["hi95"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
                figb.add_trace(go.Scatter(x=x_lab, y=bt_df["lo95"], fill="tonexty", fillcolor=rgba(P["accent"], .12),
                                          line=dict(width=0), name="95% band (nominal)"))
                figb.add_trace(go.Scatter(x=x_lab, y=bt_df["pred"], name="Forecast", mode="lines+markers",
                                          line=dict(color=P["accent"], dash="dash"), marker=dict(size=4)))
                figb.add_trace(go.Scatter(x=x_lab, y=bt_df["actual"], name="Actual", mode="lines+markers",
                                          line=dict(color=P["text"]), marker=dict(size=4)))
                figb.update_layout(template="plotly_dark", height=340, margin=dict(l=8, r=8, t=30, b=8),
                                   paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                   title=dict(text=f"Forecast vs actual ({fc['horizon']} {bar_word}s ahead, out-of-sample)",
                                              font=dict(size=13)),
                                   legend=dict(orientation="h", y=1.12), hovermode="x unified",
                                   font=dict(family="Inter, sans-serif", color=P["text"]))
                figb.update_yaxes(gridcolor=P["grid"])
                st.plotly_chart(figb, width="stretch", config={"displaylogo": False})

                rows_tf = []
                for k, lab, _, _ in ta_plus.TIMEFRAMES:
                    if k not in tf_data or len(tf_data[k]) < fc["lookback"] + fc["horizon"] + 10:
                        continue
                    _, m_tf = ta_plus.walk_forward(tf_data[k]["Close"].values, fc["horizon"], fc["lookback"])
                    if m_tf:
                        rows_tf.append({"Timeframe": lab, "Tests": m_tf["n"], "Direction %": m_tf["dir_acc"],
                                        "MAE %": m_tf["mape"], "Skill vs no-change %": m_tf["skill"],
                                        "68% coverage": m_tf["cov68"], "95% coverage": m_tf["cov95"]})
                if rows_tf:
                    st.markdown(f"**Performance by timeframe** · {display_name(ticker)} · {fc['horizon']} bars ahead, "
                                f"{fc['lookback']}-bar fit")
                    pct = st.column_config.NumberColumn(format="%.0f%%")
                    st.dataframe(pd.DataFrame(rows_tf), hide_index=True, width="stretch",
                                 column_config={"Direction %": pct, "MAE %": st.column_config.NumberColumn(format="%.1f%%"),
                                                "Skill vs no-change %": st.column_config.NumberColumn(format="%+.0f%%"),
                                                "68% coverage": pct, "95% coverage": pct})
                if st.button(f"Run the same test for every stock in my list ({len(universe)})"):
                    with st.spinner("Back-testing…"):
                        data_u = get_many(tuple(universe), interval, period)
                        res_u = []
                        for t_, dfu in data_u.items():
                            if len(dfu) < fc["lookback"] + fc["horizon"] + 10:
                                continue
                            _, m_u = ta_plus.walk_forward(dfu["Close"].values, fc["horizon"], fc["lookback"])
                            if m_u:
                                res_u.append({"Name": display_name(t_), "Ticker": t_, "Tests": m_u["n"],
                                              "Direction %": m_u["dir_acc"], "MAE %": m_u["mape"],
                                              "Skill vs no-change %": m_u["skill"], "95% coverage": m_u["cov95"]})
                        st.session_state["bt_universe"] = (tf_name, fc["horizon"], fc["lookback"], res_u)
                if st.session_state.get("bt_universe"):
                    tfn, hz, lb, res_u = st.session_state["bt_universe"]
                    st.markdown(f"**Performance by stock** · {tfn} · {hz} bars ahead · {lb}-bar fit")
                    st.dataframe(pd.DataFrame(res_u).sort_values("Direction %", ascending=False), hide_index=True,
                                 width="stretch",
                                 column_config={"Direction %": st.column_config.NumberColumn(format="%.0f%%"),
                                                "MAE %": st.column_config.NumberColumn(format="%.1f%%"),
                                                "Skill vs no-change %": st.column_config.NumberColumn(format="%+.0f%%"),
                                                "95% coverage": st.column_config.NumberColumn(format="%.0f%%")})
                st.markdown("<div class='note'>Method: walk-forward, out-of-sample. Each test origin uses only candles up to "
                            "that date (no look-ahead), predicts the close N bars later, then moves forward by half the "
                            "horizon. Skill > 0% means the trend line beat simply guessing 'no change'. A directional "
                            "accuracy near 50% means the forecast has little predictive value for this stock.</div>",
                            unsafe_allow_html=True)

        # ================= Levels
        with tabs[8]:
            with st.expander("Support & resistance levels (table)", expanded=bool(extra_rows)):
                lv = pd.DataFrame(
                    [{"Type": "Resistance", "Level": l, "Touches": str(n) if n else "projected", "Distance %": (l / price - 1) * 100} for l, n in reversed(res)]
                    + [{"Type": "▶ Current price", "Level": price, "Touches": "", "Distance %": 0.0}]
                    + [{"Type": "Support", "Level": l, "Touches": str(n) if n else "projected", "Distance %": (l / price - 1) * 100} for l, n in sup])
                st.dataframe(lv, hide_index=True, width="stretch",
                             column_config={"Level": st.column_config.NumberColumn(format="%.3f"),
                                            "Distance %": st.column_config.NumberColumn(format="%+.1f%%")})
                if extra_rows:
                    st.markdown("**Extra levels**")
                    ex_df = pd.DataFrame(extra_rows).sort_values("Level", ascending=False)
                    st.dataframe(ex_df[["Method", "Role", "Level", "Distance %", "Note"]], hide_index=True,
                                 width="stretch",
                                 column_config={"Level": st.column_config.NumberColumn(format="%.3f"),
                                                "Distance %": st.column_config.NumberColumn(format="%+.1f%%")})

        # ================= Data
        with tabs[9]:
            st.markdown("#### 🧾 Data quality & source audit")
            st.dataframe(pd.DataFrame([{"Check": k, "Value": str(v)} for k, v in dq.items()]), hide_index=True,
                         width="stretch")
            st.markdown("""<div class='note'>
• <b>Source</b>: Yahoo Finance via the free <code>yfinance</code> library (no paid APIs). Bursa quotes are delayed about
15 minutes; US quotes are near real-time in market hours.<br>
• <b>Prices</b> are unadjusted for dividends (they match your broker's chart); Yahoo already adjusts for splits/bonus issues.<br>
• <b>Cleaning</b>: duplicate timestamps, rows with missing/zero prices and rows with High &lt; Low are removed; missing
volume is treated as 0.<br>
• <b>Forming candle</b>: while the market is open the latest candle can still change. With “Signals on completed
candles only” on, all indicators, patterns and signals use the last completed candle.<br>
• <b>Indicators</b> are calculated locally and deterministically (EMA, Wilder RSI/ATR/ADX, MACD 12/26/9). Values
that need more history than available are shown as N/A instead of being estimated.<br>
• <b>4-Hour</b> candles are aggregated from real 1-hour candles (4 per block within each trading day).</div>""",
                        unsafe_allow_html=True)


# ---------------- Picks ----------------
with tab_pk:
    fundamentals = None
    if not bursa_all.empty:
        fundamentals = bursa_all.set_index("ticker")[["pe", "div_yield"]]
    picks.render(universe, display_name, P, pill, get_many, analyse, trade_plan, add_indicators,
                 learn.detect_patterns, fundamentals)

# ---------------- Heatmap ----------------
with tab_hm:
    extras.render_heatmap(universe, display_name, P, get_many, analyse, sector_map)

# ---------------- Dividends ----------------
with tab_div:
    extras.render_dividends(universe, prices, display_name, P, pill)

# ---------------- IPO ----------------
with tab_ipo:
    extras.render_ipo(P, pill)

# ---------------- Terminal ----------------
with tab_term:
    terminal.render(display_name, normalise, P, pill, rgba, tv_symbol, tradingview_widget, ta_plus.market_open,
                    universe)

# ---------------- TradingView ----------------
with tab_tv:
    tv_pick = st.session_state.get("an_ticker") or st.session_state.get("pick", "1155.KL")
    c1, c2 = st.columns([3, 1])
    tv_h = c1.slider("Chart height (px)", 500, 1300, 850, 50)
    tv_int = c2.selectbox("Interval", ["D", "W", "240", "60", "15"], index=0,
                          format_func=lambda v: {"D": "Daily", "W": "Weekly", "240": "4 hours",
                                                 "60": "1 hour", "15": "15 min"}[v])
    st.caption(f"Interactive TradingView chart for **{display_name(tv_pick)}** (the stock picked in Analysis). "
               "Change the symbol inside the chart, add indicators and draw your own lines. If it says the symbol is "
               "only available on TradingView, that exchange can't be embedded; use the Analysis chart instead.")
    tradingview_widget(tv_symbol(tv_pick), height=tv_h, interval=tv_int)

# ---------------- Learn ----------------
with tab_learn:
    learn.set_palette(P)
    learn.render()

st.markdown("<div class='foot'>⚠️ Generated automatically from technical indicators. Not investment advice. "
            "Support/resistance, zones and forecasts are reference levels, not guarantees. Always use a stop loss and "
            "never invest money you can't afford to lose.</div>", unsafe_allow_html=True)

if auto:
    time.sleep(every)
    st.cache_data.clear()
    st.rerun()
