"""Core of the Stock Screener: stock lists, data loading (Yahoo Finance), indicators, signal scoring,
trade plans, forecasts, volume/ADX analysis, extra levels and charts. No page layout lives here.
Price data: Yahoo Finance (yfinance). Bursa ~15 min delayed, US near real-time.
"""
from pxfmt import fp, pxf  # noqa: F401
import os
import re
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots

import indicators as ind



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


CRYPTO = {
    "BTC-USD": "Bitcoin", "ETH-USD": "Ethereum", "XRP-USD": "XRP", "BNB-USD": "BNB", "SOL-USD": "Solana",
    "DOGE-USD": "Dogecoin", "TRX-USD": "TRON", "ADA-USD": "Cardano", "LINK-USD": "Chainlink",
    "AVAX-USD": "Avalanche", "XLM-USD": "Stellar", "SUI20947-USD": "Sui", "BCH-USD": "Bitcoin Cash",
    "HBAR-USD": "Hedera", "LTC-USD": "Litecoin", "TON11419-USD": "Toncoin", "DOT-USD": "Polkadot",
    "SHIB-USD": "Shiba Inu", "UNI7083-USD": "Uniswap", "NEAR-USD": "NEAR Protocol", "AAVE-USD": "Aave",
    "ETC-USD": "Ethereum Classic", "ATOM-USD": "Cosmos", "ARB11841-USD": "Arbitrum", "PEPE24478-USD": "Pepe",
}
CRYPTO_ALIAS = {k.split("-")[0].rstrip("0123456789"): k for k in CRYPTO}
CRYPTO_ALIAS.update({v.upper().replace(" ", ""): k for k, v in CRYPTO.items()})


def is_crypto(t: str) -> bool:
    return str(t).upper().endswith(("-USD", "-USDT", "-MYR", "-EUR", "-BTC", "-ETH"))


def normalise_crypto(sym: str) -> str:
    """'btc', 'Bitcoin', 'BTCUSD', 'BINANCE:BTCUSDT' or 'BTC-USD' -> Yahoo's 'BTC-USD'."""
    s = (sym or "").strip().upper().replace(" ", "")
    if ":" in s:
        s = s.split(":", 1)[1]
    if not s:
        return ""
    if is_crypto(s):
        return s
    for q in ("USDT", "USD"):
        if s.endswith(q) and len(s) > len(q):
            s = s[: -len(q)]
            break
    return CRYPTO_ALIAS.get(s, s + "-USD")


def display_name(t: str) -> str:
    return BURSA.get(t) or US.get(t) or CRYPTO.get(t) or NAMES.get(t) or (t[:-4] if is_crypto(t) and t.endswith("-USD") else t)


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


@st.cache_data(ttl=60, show_spinner=False, max_entries=400)
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


@st.cache_data(ttl=60, show_spinner=False, max_entries=30)
def load_many(tickers: tuple, interval: str, period: str) -> dict:
    return _download_many(tickers, interval, period)


@st.cache_data(ttl=600, show_spinner=False, max_entries=30)
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
    """Every stock listed on Bursa Malaysia (names, sectors). Cached 12 h: use load_bursa_snapshot for prices."""
    return _fetch_bursa_all()


@st.cache_data(ttl=600, show_spinner=False)
def load_bursa_snapshot() -> pd.DataFrame:
    """Same list, refreshed every 10 minutes, for today's moves (breadth, gainers/losers, sectors)."""
    df = _fetch_bursa_all()
    df.attrs["asof"] = pd.Timestamp.now(tz="Asia/Kuala_Lumpur").isoformat()
    return df


def _fetch_bursa_all() -> pd.DataFrame:
    """Every stock listed on Bursa Malaysia (Yahoo exchange code KLS), with sector, via Yahoo's screener."""
    if DEMO:
        d = _demo_bursa_all()
        return d
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
def _wilder(s: pd.Series, n: int) -> pd.Series:
    """Wilder's smoothing as in his original RSI/ATR/ADX: the first value is the simple average of the first n
    valid inputs, then avg = (prev * (n - 1) + x) / n. Values before that are NaN (warm-up), never estimated."""
    v = s.values.astype(float)
    out = np.full(len(v), np.nan)
    idx = np.flatnonzero(~np.isnan(v))
    if len(idx) < n:
        return pd.Series(out, index=s.index)
    start = idx[n - 1]
    out[start] = np.nanmean(v[idx[0]:start + 1])
    for i in range(start + 1, len(v)):
        x = v[i]
        out[i] = out[i - 1] if np.isnan(x) else (out[i - 1] * (n - 1) + x) / n
    return pd.Series(out, index=s.index)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    c = d["Close"]
    for n in (20, 50, 200):
        d[f"EMA{n}"] = c.ewm(span=n, adjust=False).mean()
    delta = c.diff()
    gain = _wilder(delta.clip(lower=0), 14)
    loss = _wilder(-delta.clip(upper=0), 14)
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    d["RSI"] = rsi.where(~(loss == 0) | gain.isna(), 100.0)      # no losses at all = 100; warm-up stays NaN
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    d["MACD"] = ema12 - ema26
    d["MACDsig"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACDhist"] = d["MACD"] - d["MACDsig"]
    tr = pd.concat([d["High"] - d["Low"], (d["High"] - c.shift()).abs(), (d["Low"] - c.shift()).abs()], axis=1).max(axis=1)
    tr.iloc[0] = np.nan                                              # no previous close on the first bar
    d["ATR"] = _wilder(tr, 14)
    d["VolAvg"] = d["Volume"].rolling(20).mean()
    d["VolAvg20"] = d["Volume"].shift(1).rolling(20).mean()          # average of the previous 20 bars
    d["RVOL"] = (d["Volume"] / d["VolAvg20"]).replace([np.inf, -np.inf], np.nan)
    d["OBV"] = (np.sign(c.diff()).fillna(0) * d["Volume"]).cumsum()
    # ADX (14) with +DI / -DI, Wilder smoothing
    up_move = d["High"].diff()
    down_move = -d["Low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    pdm = pd.Series(plus_dm, index=d.index)
    mdm = pd.Series(minus_dm, index=d.index)
    pdm.iloc[0] = mdm.iloc[0] = np.nan
    atr_w = d["ATR"]
    d["PDI"] = 100 * _wilder(pdm, 14) / atr_w.replace(0, np.nan)
    d["MDI"] = 100 * _wilder(mdm, 14) / atr_w.replace(0, np.nan)
    dx = 100 * (d["PDI"] - d["MDI"]).abs() / (d["PDI"] + d["MDI"]).replace(0, np.nan)
    d["ADX"] = _wilder(dx, 14)
    d["HH20"] = d["High"].shift(1).rolling(20).max()                 # prior 20-bar high / low (breakout levels)
    d["LL20"] = d["Low"].shift(1).rolling(20).min()
    return d


# ----------------------------------------------------------------------------
# Support & resistance (John J. Murphy, "Technical Analysis of the Financial Markets", ch. 4):
#   support is usually a previous reaction low and resistance a previous reaction high; a level is more
#   significant the more often it was tested, the longer prices traded there, the heavier the volume there and
#   the more recent it is; a broken level tends to swap roles (polarity / role reversal); round numbers matter.
# ----------------------------------------------------------------------------
REACTION_W = 5           # a swing high/low must be the extreme of the 5 candles on each side (confirmed)
MIN_REACTION_ATR = 1.0   # ...and price must have moved at least 1 ATR away from it afterwards (a real reaction)
MAJOR_REACTION_ATR = 2.0 # a single-touch level is only kept if its reaction was at least 2 ATR (a major turn)


def _round_step(price):
    return 0.05 if price < 1 else 0.5 if price < 10 else 5 if price < 100 else 10 if price < 1000 else 50


def level_zones(d: pd.DataFrame, lookback: int = 250, w: int = REACTION_W):
    """All support/resistance zones found from confirmed reaction highs/lows, with the evidence for each."""
    x = d.tail(lookback)
    n = len(x)
    if n < 2 * w + 5:
        return [], None
    H, L, C, V = (x[k].values.astype(float) for k in ("High", "Low", "Close", "Volume"))
    atr = float(x["ATR"].iloc[-1]) if "ATR" in x and pd.notna(x["ATR"].iloc[-1]) else float(np.nanmean(H - L))
    price = float(C[-1])
    tol = max(0.5 * atr, price * 0.006)
    pts = []
    for i in range(w, n - w):                       # the last w candles can't be confirmed yet (no look-ahead)
        j = min(n, i + 16)
        if H[i] == H[i - w:i + w + 1].max():
            react = H[i] - L[i + 1:j].min()
            if react >= MIN_REACTION_ATR * atr:
                pts.append({"i": i, "price": H[i], "kind": "peak", "react": react / atr})
        if L[i] == L[i - w:i + w + 1].min():
            react = H[i + 1:j].max() - L[i]
            if react >= MIN_REACTION_ATR * atr:
                pts.append({"i": i, "price": L[i], "kind": "trough", "react": react / atr})
    pts.sort(key=lambda p: p["price"])
    groups = []
    for p_ in pts:                                  # zones are at most `tol` wide (no chaining drift)
        if groups and p_["price"] - groups[-1][0]["price"] <= tol:
            groups[-1].append(p_)
        else:
            groups.append([p_])
    vol_avg = float(np.nanmean(V)) if np.nansum(V) > 0 else 0.0
    zones = []
    for g in groups:
        prices = [q["price"] for q in g]
        lvl = float(np.median(prices))
        lo, hi = min(prices), max(prices)
        band_lo, band_hi = min(lo, lvl - tol / 4), max(hi, lvl + tol / 4)
        inside = (L <= band_hi) & (H >= band_lo)
        time_bars = int(inside.sum())
        vol_ratio = float(np.nanmean(V[inside]) / vol_avg) if vol_avg and inside.any() else None
        last_i = max(q["i"] for q in g)
        age = n - 1 - last_i
        recency = 0.5 ** (age / 120)
        kinds = {q["kind"] for q in g}
        touches = len(g)
        big = max(q["react"] for q in g)
        raw = (1 + (touches - 1)) * (1 + min(time_bars, 40) / 40) * (1 + min(vol_ratio or 1, 2) / 4) * (0.4 + 0.6 * recency)
        step = _round_step(lvl)
        zones.append({
            "level": lvl, "lo": band_lo, "hi": band_hi, "touches": touches, "time_bars": time_bars,
            "vol_ratio": vol_ratio, "last_date": x.index[last_i], "age": age, "recency": recency, "raw": raw,
            "biggest_reaction_atr": big, "kinds": kinds,
            "members": [(x.index[q["i"]], q["price"], q["kind"], q["react"]) for q in g],
            "round": abs(lvl - round(lvl / step) * step) <= tol / 2,
            "testing": band_lo - tol / 2 <= price <= band_hi + tol / 2,
            "grade": "strong" if touches >= 3 or (touches >= 2 and time_bars >= 15) else "medium" if touches >= 2
            else "major swing" if big >= MAJOR_REACTION_ATR else "weak",
        })
    top = max((z["raw"] for z in zones), default=1) or 1
    for z in zones:
        z["strength"] = round(z["raw"] / top * 100)
        role = "support" if z["level"] < price else "resistance"
        z["role"] = role
        flipped = (role == "support" and z["kinds"] == {"peak"}) or (role == "resistance" and z["kinds"] == {"trough"})
        z["polarity"] = "old resistance, now support" if flipped and role == "support" else \
            "old support, now resistance" if flipped else ("acted as both" if len(z["kinds"]) == 2 else "")
    return zones, {"atr": atr, "tol": tol, "price": price, "lookback": n, "points": len(pts)}


def find_levels(d: pd.DataFrame, lookback: int = 250, w: int = REACTION_W):
    """Nearest meaningful support and resistance below/above the price as [(level, touches)], nearest first.
    A zone counts if it was tested at least twice, or once by a major (≥ 2 ATR) reaction. Zones the price is
    sitting inside are 'being tested' and skipped. With nothing above (new highs) or below, an ATR projection
    is returned with 0 touches."""
    zones, meta = level_zones(d, lookback, w)
    x = d.tail(lookback)
    price = float(x["Close"].iloc[-1])
    atr = meta["atr"] if meta else float(x["ATR"].iloc[-1])
    ok = [z for z in zones if z["grade"] != "weak" and not z["testing"]]

    def pick(cands):
        """Nearest first, but levels closer than 1 ATR to each other are one area: keep the stronger."""
        out = []
        for z in cands:
            if out and abs(z["level"] - out[-1]["level"]) < atr:
                if z["strength"] > out[-1]["strength"]:
                    out[-1] = z
                continue
            out.append(z)
        return [(z["level"], z["touches"]) for z in out[:3]]

    sup = pick(sorted((z for z in ok if z["hi"] < price), key=lambda z: -z["level"]))
    res = pick(sorted((z for z in ok if z["lo"] > price), key=lambda z: z["level"]))
    if not sup:
        sup = [(price - 2 * atr, 0)]
    if not res:
        res = [(price + 2 * atr, 0)]  # 0 touches = ATR projection (price at highs)
    while len(res) < 2:
        res.append((res[-1][0] + 2 * atr, 0))
    while len(sup) < 2:
        sup.append((sup[-1][0] - 2 * atr, 0))
    return sup, res


def zone_for(zones, level):
    """The zone a chosen level came from (None for ATR projections)."""
    for z in zones or []:
        if abs(z["level"] - level) < 1e-9:
            return z
    return None


def score_factors(d: pd.DataFrame, sup, res):
    """The scoring rules, one row per factor, with the actual reading and the points it earned.
    Points are identical to the original scoring system; only the descriptions were made exact."""
    last, prev = d.iloc[-1], d.iloc[-2]
    price, atr = float(last["Close"]), float(last["ATR"])
    F = []

    def add(group, factor, reading, rule, pts, text):
        F.append({"Group": group, "Factor": factor, "Reading": reading, "Rule": rule, "Points": pts, "text": text})

    e20, e50, e200 = float(last["EMA20"]), float(last["EMA50"]), float(last["EMA200"])
    add("Trend", "Price vs EMA200", f"{fp(price)} vs {fp(e200)}", "above: +15 · below: −15",
        15 if price > e200 else -15,
        "Price above EMA200: long-term uptrend" if price > e200 else "Price below EMA200: long-term downtrend")
    add("Trend", "EMA50 vs EMA200", f"{fp(e50)} vs {fp(e200)}", "above: +10 · below: −10",
        10 if e50 > e200 else -10,
        "EMA50 above EMA200 (golden cross structure)" if e50 > e200 else "EMA50 below EMA200 (death cross structure)")
    add("Trend", "Price vs EMA20", f"{fp(price)} vs {fp(e20)}", "above: +10 · below: −10",
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
        add("Levels", "Near support", f"{fp(price - s1)} above S {fp(s1)} (ATR {fp(atr)})", "within 1 ATR: +15", 15,
            f"Price near support {fp(s1)} (low-risk entry area)")
    if r1 - price <= atr and res[0][1] > 0:
        add("Levels", "Near resistance", f"{fp(r1 - price)} below R {fp(r1)} (ATR {fp(atr)})", "within 1 ATR: −15", -15,
            f"Price near resistance {fp(r1)} (limited upside)")
    if prev["Close"] < s1 <= price:
        add("Levels", "Breakout", f"prev close {fp(float(prev['Close']))} → {fp(price)} over {fp(s1)}",
            "closed above a level: +10", 10, f"Breakout: just closed above {fp(s1)}")
    if prev["Close"] > r1 >= price:
        add("Levels", "Breakdown", f"prev close {fp(float(prev['Close']))} → {fp(price)} under {fp(r1)}",
            "closed below a level: −10", -10, f"Breakdown: just closed below {fp(r1)}")

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
        when = (f"Price is in the buy zone now. Entry around **{fp(price)}**, stop loss **{fp(stop)}**. "
                f"Take half profit at **{fp(r1)}**, the rest at **{fp(r2)}**.")
    elif bullish:
        action = "🟡 UPTREND, BUT DON'T CHASE"
        when = (f"Price is far above support. Two options: **(a)** wait for a pullback to **{fp(buy_lo)}–{fp(buy_hi)}** "
                f"(rough estimate ~{bars_back} {unit} at a normal pace), or **(b)** buy when price "
                f"**closes above {fp(r1)}** on high volume (breakout), stop below {fp(r1 - atr)}.")
    elif bearish:
        action = "🔴 AVOID BUYING / CONSIDER SELLING"
        when = (f"If you hold it: consider selling or reducing, especially if price **closes below {fp(s1)}**. "
                f"A bounce to **{fp(sell_lo)}–{fp(sell_hi)}** is a better place to sell. "
                f"Don't buy until it closes back above EMA20 ({fp(last['EMA20'])}).")
    else:
        action = "⚪ WAIT"
        when = (f"No clear direction. A **BUY** signal appears if price bounces off **{fp(s1)}** with RSI rising / MACD crossing up, "
                f"**or** price closes above **{fp(r1)}**. A **SELL** signal if price closes below **{fp(s1)}** "
                f"(next downside target {fp(s2)}).")

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
# Structural colours (cards, borders, grid, muted text) are the same translucent greys in both themes so they
# look right the instant the viewer switches theme (Streamlit doesn't rerun the script on a theme switch).
# Page CSS uses var(--ss-*) colours computed in the browser from the current text colour.
_NEUTRAL = dict(muted="#8a93a6", grid="rgba(138,147,166,0.16)", border="rgba(138,147,166,0.28)", card="rgba(138,147,166,0.06)", card2="rgba(138,147,166,0.12)", bg="rgba(0,0,0,0)", heat_mid="rgba(138,147,166,0.22)")
DARK_P = dict(_NEUTRAL, text="#dfe3ea", up="#3fb68b", down="#e0646a", accent="#7b93f5", warn="#e3a34b",
              ema20="#e3c46a", ema50="#6fa8ef", ema200="#b48ae0", on_fill="#ffffff", neutral="#9aa3b5")
LIGHT_P = dict(_NEUTRAL, text="#1c2230", up="#16955f", down="#d0414e", accent="#4f6be8", warn="#b97a12",
               ema20="#c29a12", ema50="#2f74d0", ema200="#8650c2", on_fill="#ffffff", neutral="#7a8394")
P = dict(DARK_P)


def apply_theme(kind):
    """Switch the shared palette in place (every module holds this same dict) to the viewer's theme."""
    P.clear()
    P.update(LIGHT_P if kind == "light" else DARK_P)
    return P
LABEL_COLOR = {"STRONG BUY": "#3fb68b", "BUY": "#62c49e", "NEUTRAL": "#8890a2",
               "SELL": "#e88a8e", "STRONG SELL": "#e0646a"}
LABEL_DOT = {"STRONG BUY": "🟢", "BUY": "🟢", "NEUTRAL": "⚪", "SELL": "🔴", "STRONG SELL": "🔴"}


def rgba(hex_color, a):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a})"


def mix(c1, c2, t):
    """Blend two hex colours: t = 0 gives c1, t = 1 gives c2."""
    a, b = c1.lstrip("#"), c2.lstrip("#")
    rgb = [round(int(a[i:i + 2], 16) * (1 - t) + int(b[i:i + 2], 16) * t) for i in (0, 2, 4)]
    return "#" + "".join(f"{v:02x}" for v in rgb)


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
    return {"up": P["up"], "down": P["down"], "up_soft": mix(P["up"], "#8a93a6", 0.3),
            "down_soft": mix(P["down"], "#8a93a6", 0.3)}.get(k, P["muted"])


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
                      annotation_text=f"POC {fp(vp['poc'])}", annotation_position="bottom left",
                      annotation_font=dict(color=P["warn"], size=11))
        for k, lab in (("vah", "VAH"), ("val", "VAL")):
            fig.add_hline(y=vp[k], line=dict(color=rgba(P["accent"], .55), width=1, dash="dot"), row=1, col=1,
                          annotation_text=f"{lab} {fp(vp[k])}", annotation_position="bottom left",
                          annotation_font=dict(color=P["accent"], size=10))
    if "fib" in ex:
        for r, p in ex["fib"]["levels"]:
            strong = r in (0.382, 0.5, 0.618)
            fig.add_hline(y=p, line=dict(color=rgba(gold, .75 if strong else .4), width=1.2 if strong else 0.9,
                                         dash="dot"), row=1, col=1,
                          annotation_text=f"Fib {r * 100:.1f}% · {fp(p)}", annotation_position="top left",
                          annotation_font=dict(color=gold, size=10))
    if "touch" in ex:
        for lvl, n in ex["touch"]:
            fig.add_hline(y=lvl, line=dict(color=rgba(teal, .8), width=0.8 + min(n, 8) * 0.15, dash="longdash"),
                          row=1, col=1, annotation_text=f"⟷ {n}x · {fp(lvl)}", annotation_position="top left",
                          annotation_font=dict(color=teal, size=10))
    if "swings" in ex:
        sw = ex["swings"]
        fig.add_trace(go.Scatter(mode="markers", x=[labels[i] for i, _ in sw["hi"]], y=[p * 1.006 for _, p in sw["hi"]],
                                 marker=dict(symbol="triangle-down", size=9, color=pink), name="Swing high",
                                 hovertemplate="Swing high %{customdata:.6~g}<extra></extra>",
                                 customdata=[p for _, p in sw["hi"]]), 1, 1)
        fig.add_trace(go.Scatter(mode="markers", x=[labels[i] for i, _ in sw["lo"]], y=[p * 0.994 for _, p in sw["lo"]],
                                 marker=dict(symbol="triangle-up", size=9, color=teal), name="Swing low",
                                 hovertemplate="Swing low %{customdata:.6~g}<extra></extra>",
                                 customdata=[p for _, p in sw["lo"]]), 1, 1)
        for p in sw["lv_hi"]:
            fig.add_hline(y=p, line=dict(color=rgba(pink, .45), width=1, dash="dot"), row=1, col=1)
        for p in sw["lv_lo"]:
            fig.add_hline(y=p, line=dict(color=rgba(teal, .45), width=1, dash="dot"), row=1, col=1)


CHART_PANELS = ["Volume", "RSI", "MACD", "ADX"]
CHART_OVERLAYS = ["EMA 20/50/200", "Support/resistance & zones", "Breakout markers"]


def make_chart(d, sup, res, plan, name, interval, fc=None, show_bars=120, extra=None,
               panels=tuple(CHART_PANELS), overlays=tuple(CHART_OVERLAYS), mobile=False, height=None):
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
        zones, _ = level_zones(d)
        shown = set(labels)
        for role, lst, col in (("S", sup, P["up"]), ("R", res, P["down"])):
            for lvl, n in lst:
                z = zone_for(zones, lvl)
                if z is None:
                    fig.add_hline(y=lvl, line=dict(color=rgba(col, .55), width=1, dash="dot"), row=1, col=1,
                                  annotation_text=f"{role} {fp(lvl)} · proj.", annotation_position="right",
                                  annotation_font=dict(color=col, size=10.5))
                    continue
                fig.add_hrect(y0=z["lo"], y1=z["hi"], fillcolor=col, opacity=0.13, line_width=0, row=1, col=1)
                fig.add_hline(y=lvl, line=dict(color=rgba(col, .8), width=1.1), row=1, col=1,
                              annotation_text=f"{role} {fp(lvl)} · {n}×" + (" ⇄" if z["polarity"].startswith("old") else ""),
                              annotation_position="right", annotation_font=dict(color=col, size=10.5))
                mx = [(ts, p_, k, r_) for ts, p_, k, r_ in z["members"]
                      if ts.strftime(fmt) in shown]
                if mx:
                    fig.add_trace(go.Scatter(
                        x=[ts.strftime(fmt) for ts, _, _, _ in mx], y=[p_ for _, p_, _, _ in mx], mode="markers",
                        marker=dict(symbol=["triangle-up" if k == "trough" else "triangle-down" for _, _, k, _ in mx],
                                    size=10, color=col, line=dict(width=1, color=P["on_fill"])),
                        name=f"{role} {fp(lvl)} reactions", showlegend=False,
                        customdata=[[k, r_] for _, _, k, r_ in mx],
                        hovertemplate=f"{role} {fp(lvl)}: reaction %{{customdata[0]}} %{{y:.6~g}} "
                                      "(moved %{customdata[1]:.1f} ATR after)<extra></extra>"), 1, 1)
        fig.add_hline(y=plan["stop"], line=dict(color=P["warn"], width=1.1, dash="dash"), row=1, col=1,
                      annotation_text=f"Stop {fp(plan['stop'])}", annotation_position="bottom right",
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
        fig.add_hline(y=ADX_STRONG, line=dict(color=rgba(P["muted"], .6), width=1, dash="dot"), row=ROW['ADX'], col=1,
                      annotation_text="25 strong", annotation_position="right",
                      annotation_font=dict(color=P["muted"], size=10))
        fig.add_hline(y=ADX_WEAK, line=dict(color=rgba(P["muted"], .45), width=1, dash="dot"), row=ROW['ADX'], col=1,
                      annotation_text="20 trend / no trend", annotation_position="right",
                      annotation_font=dict(color=P["muted"], size=10))
        fig.add_trace(go.Scatter(x=labels, y=x["PDI"], name="+DI", line=dict(color=P["up"], width=1.2),
                                 showlegend=False, hovertemplate="+DI %{y:.1f}<extra></extra>"), ROW['ADX'], 1)
        fig.add_trace(go.Scatter(x=labels, y=x["MDI"], name="−DI", line=dict(color=P["down"], width=1.2),
                                 showlegend=False, hovertemplate="−DI %{y:.1f}<extra></extra>"), ROW['ADX'], 1)
        fig.add_trace(go.Scatter(x=labels, y=x["ADX"], name="ADX", line=dict(color=P["neutral"], width=2),
                                 showlegend=False, hovertemplate="ADX %{y:.1f}<extra></extra>"), ROW['ADX'], 1)

    fig.update_layout(
        
        height=height or ((380 + 90 * (rows - 1)) if mobile else (560 + 115 * (rows - 1))),
        margin=dict(l=2, r=62, t=8, b=4) if mobile else dict(l=8, r=96, t=36, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=12),
        xaxis_rangeslider_visible=False, hovermode="x unified",
        legend=dict(orientation="h", y=1.05, x=0, font=dict(size=11, color=P["muted"]), bgcolor="rgba(0,0,0,0)"),
        bargap=0.25,
    )
    if mobile:
        fig.update_layout(showlegend=False, font=dict(size=10))
    fig.update_xaxes(type="category", nticks=4 if mobile else 8, tickangle=0, showgrid=False, linecolor=P["border"], tickfont=dict(color=P["muted"]),
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
# Compact chart + analysis (used by Paper Trade and Multi-Chart)
# ----------------------------------------------------------------------------
MC_TF = {"1D": ("1d", "2y", 120, "days"), "1W": ("1wk", "10y", 104, "weeks"),
         "4H": ("4h", "730d", 120, "4-hour bars"), "1H": ("1h", "6mo", 120, "hours")}


def stock_view(ticker, tf="1D"):
    iv, per, bars, unit = MC_TF[tf]
    try:
        df = load_one(ticker, iv, per)
    except Exception:
        df = pd.DataFrame()
    if df is None or df.empty or len(df) < 60:
        return None
    d, sup, res, sc, label, why = analyse(df)
    plan = trade_plan(d, sup, res, label, unit)
    chg = (float(d["Close"].iloc[-1]) / float(d["Close"].iloc[-2]) - 1) * 100
    return {"d": d, "sup": sup, "res": res, "sc": sc, "label": label, "why": why, "plan": plan, "iv": iv,
            "bars": bars, "ax": adx_analysis(d), "chg": chg, "price": float(d["Close"].iloc[-1])}


# ----------------------------------------------------------------------------
# Universe scan (shared by Screener, Dashboard and others)
# ----------------------------------------------------------------------------
def _cross(fast, slow, look=60):
    """(direction, bars ago) of the most recent cross of fast over slow within `look` bars, else (0, None).
    direction +1 = fast crossed above slow (bullish), -1 = crossed below."""
    diff = (fast - slow).dropna().tail(look + 1)
    sg = np.sign(diff.values)
    for i in range(len(sg) - 1, 0, -1):
        if sg[i] != 0 and sg[i - 1] != 0 and sg[i] != sg[i - 1]:
            return int(sg[i]), len(sg) - 1 - i
    return 0, None


def _above(a, b):
    return None if pd.isna(a) or pd.isna(b) else bool(a > b)


@st.cache_data(ttl=60, show_spinner=False, max_entries=20)
def scan_universe(universe: tuple, interval: str, period: str, unit: str) -> pd.DataFrame:
    """Analyse every stock in the universe once (shared by Screener, Dashboard, watchlists).
    All values use completed history only; anything inside an indicator's warm-up is None (shown as N/A)."""
    data = get_many(universe, interval, period)
    rows = []
    for t, df in data.items():
        try:
            d, sup, res, sc, label, _ = analyse(df)
            p = trade_plan(d, sup, res, label, unit)
            last, prev = d.iloc[-1], d.iloc[-2]
            c = d["Close"]
            price = float(last["Close"])
            e9, e20, e50, e200 = ind.ema(c, 9), ind.ema(c, 20), ind.ema(c, 50), ind.ema(c, 200)
            x920, x2050, x50200 = _cross(e9, e20), _cross(e20, e50), _cross(e50, e200, 250)
            xm = _cross(d["MACD"], d["MACDsig"])
            k, _d = ind.stochastic(d["High"], d["Low"], c)

            def ret(n):
                return (price / float(c.iloc[-1 - n]) - 1) * 100 if len(c) > n and c.iloc[-1 - n] else None
            hi252 = float(d["High"].iloc[-253:-1].max()) if len(d) >= 120 else None
            lo252 = float(d["Low"].iloc[-253:-1].min()) if len(d) >= 120 else None
            vavg = float(d["VolAvg20"].iloc[-1]) if pd.notna(d["VolAvg20"].iloc[-1]) else None
            rows.append({
                "Name": display_name(t), "Ticker": t, "Price": price,
                "Change %": (price / float(prev["Close"]) - 1) * 100,
                "Signal": f"{LABEL_DOT[label]} {label}", "_label": label, "Score": sc,
                "RSI": float(last["RSI"]),
                "RVOL": float(last["RVOL"]) if pd.notna(last["RVOL"]) else None,
                "ADX": float(last["ADX"]) if pd.notna(last["ADX"]) else None,
                "Trend (ADX)": adx_regime(last["ADX"], last["PDI"], last["MDI"]),
                "Support": sup[0][0], "Resistance": res[0][0],
                "To support %": (price / sup[0][0] - 1) * 100,
                "Buy zone": f"{fp(p['buy_lo'])} – {fp(p['buy_hi'])}",
                "Stop": p["stop"], "Target": p["t1"], "R:R": p["rr1"],
                # ---- extras (Dashboard + advanced Screener)
                "To resistance %": (res[0][0] / price - 1) * 100,
                "Volume": float(last["Volume"]) if pd.notna(last["Volume"]) else None,
                "Avg value": price * vavg if vavg else None,
                "Return 5 %": ret(5), "Return 20 %": ret(20), "Return 60 %": ret(60),
                "Above EMA20": _above(price, e20.iloc[-1]), "Above EMA50": _above(price, e50.iloc[-1]),
                "Above EMA200": _above(price, e200.iloc[-1]),
                "EMA9>20": _above(e9.iloc[-1], e20.iloc[-1]), "EMA20>50": _above(e20.iloc[-1], e50.iloc[-1]),
                "EMA50>200": _above(e50.iloc[-1], e200.iloc[-1]),
                "X 9/20 dir": x920[0], "X 9/20 age": x920[1], "X 20/50 dir": x2050[0], "X 20/50 age": x2050[1],
                "X 50/200 dir": x50200[0], "X 50/200 age": x50200[1],
                "MACD>signal": _above(last["MACD"], last["MACDsig"]),
                "MACD hist rising": bool(last["MACDhist"] > prev["MACDhist"]),
                "X MACD dir": xm[0], "X MACD age": xm[1], "MACD hist": float(last["MACDhist"]),
                "+DI>-DI": _above(last["PDI"], last["MDI"]),
                "ATR %": float(last["ATR"]) / price * 100 if price else None,
                "Stoch %K": float(k.iloc[-1]) if pd.notna(k.iloc[-1]) else None,
                "From 52w high %": (price / hi252 - 1) * 100 if hi252 else None,
                "From 52w low %": (price / lo252 - 1) * 100 if lo252 else None,
                "New 52w high": bool(hi252 and price > hi252),
                "Breakout": bool(pd.notna(last["HH20"]) and price > last["HH20"]),
                "Breakdown": bool(pd.notna(last["LL20"]) and price < last["LL20"]),
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


SCAN_BASE = ["Name", "Ticker", "Price", "Change %", "Signal", "_label", "Score", "RSI", "RVOL", "ADX", "Trend (ADX)",
             "Support", "Resistance", "To support %", "Buy zone", "Stop", "Target", "R:R"]

