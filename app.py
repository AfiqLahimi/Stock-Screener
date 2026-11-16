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
    return s


def display_name(t: str) -> str:
    return BURSA.get(t) or US.get(t) or t


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
    freq = {"1d": "B", "1wk": "W-FRI", "1h": "h", "15m": "15min"}[interval]
    idx = pd.date_range(end=pd.Timestamp.now().floor("h"), periods=n, freq=freq)
    close = 20 * np.exp(np.cumsum(rng.normal(0.0004, 0.018, n)))
    op = close * (1 + rng.normal(0, 0.006, n))
    hi = np.maximum(op, close) * (1 + np.abs(rng.normal(0, 0.008, n)))
    lo = np.minimum(op, close) * (1 - np.abs(rng.normal(0, 0.008, n)))
    vol = rng.integers(100_000, 5_000_000, n).astype(float)
    return pd.DataFrame({"Open": op, "High": hi, "Low": lo, "Close": close, "Volume": vol}, index=idx)


def _clean(d: pd.DataFrame) -> pd.DataFrame:
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d = d[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
    return d


@st.cache_data(ttl=60, show_spinner=False)
def load_one(ticker: str, interval: str, period: str) -> pd.DataFrame:
    if DEMO:
        return _demo(ticker, interval)
    d = yf.download(ticker, period=period, interval=interval, auto_adjust=False,
                    progress=False, threads=False)
    if d is None or d.empty:
        return pd.DataFrame()
    return _clean(d)


@st.cache_data(ttl=60, show_spinner=False)
def load_many(tickers: tuple, interval: str, period: str) -> dict:
    if DEMO:
        return {t: _demo(t, interval) for t in tickers}
    raw = yf.download(list(tickers), period=period, interval=interval, group_by="ticker",
                      auto_adjust=False, progress=False, threads=True)
    out = {}
    for t in tickers:
        try:
            d = raw[t] if isinstance(raw.columns, pd.MultiIndex) and len(tickers) > 1 else raw
            d = _clean(d.copy())
            if len(d) > 60:
                out[t] = d
        except Exception:
            pass
    return out


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


def score_signal(d: pd.DataFrame, sup, res):
    last, prev = d.iloc[-1], d.iloc[-2]
    price, atr = float(last["Close"]), float(last["ATR"])
    s, why = 0, []

    def add(pts, txt):
        nonlocal s
        s += pts
        why.append((pts, txt))

    if price > last["EMA200"]:
        add(15, "Price above EMA200: long-term uptrend")
    else:
        add(-15, "Price below EMA200: long-term downtrend")
    if last["EMA50"] > last["EMA200"]:
        add(10, "EMA50 above EMA200 (golden cross structure)")
    else:
        add(-10, "EMA50 below EMA200 (death cross structure)")
    if price > last["EMA20"]:
        add(10, "Price above EMA20: short-term momentum positive")
    else:
        add(-10, "Price below EMA20: short-term momentum weak")

    rsi = float(last["RSI"])
    if rsi < 30:
        add(15, f"RSI {rsi:.0f}: oversold, potential bounce")
    elif rsi > 70:
        add(-15, f"RSI {rsi:.0f}: overbought, pullback risk")
    elif rsi >= 50:
        add(5, f"RSI {rsi:.0f}: buyers have momentum")
    else:
        add(-5, f"RSI {rsi:.0f}: sellers have momentum")

    if last["MACDhist"] > 0 and prev["MACDhist"] <= 0:
        add(15, "MACD just crossed ABOVE the signal line")
    elif last["MACDhist"] < 0 and prev["MACDhist"] >= 0:
        add(-15, "MACD just crossed BELOW the signal line")
    elif last["MACDhist"] > 0:
        add(5, "MACD positive")
    else:
        add(-5, "MACD negative")

    s1, r1 = sup[0][0], res[0][0]
    if price - s1 <= atr:
        add(15, f"Price near support {s1:.3f} (low-risk entry area)")
    if r1 - price <= atr and res[0][1] > 0:
        add(-15, f"Price near resistance {r1:.3f} (limited upside)")
    if prev["Close"] < s1 <= price:
        add(10, f"Breakout: just closed above {s1:.3f}")
    if prev["Close"] > r1 >= price:
        add(-10, f"Breakdown: just closed below {r1:.3f}")

    if pd.notna(last["VolAvg"]) and last["Volume"] > 1.5 * last["VolAvg"]:
        if price >= last["Open"]:
            add(5, "High volume on a green candle (strong buyers)")
        else:
            add(-5, "High volume on a red candle (strong sellers)")

    s = int(max(-100, min(100, s)))
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


def trade_plan(d, sup, res, label, unit):
    last = d.iloc[-1]
    price, atr = float(last["Close"]), float(last["ATR"])
    s1, s2 = sup[0][0], sup[1][0]
    r1, r2 = res[0][0], res[1][0]
    buy_lo, buy_hi = s1, s1 + 0.5 * atr
    stop = s1 - atr
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
        "rr1": rr1, "rr2": rr2, "bars_t1": bars_t1,
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
    step = "h" if interval == "1h" else "×15m"
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
# Chart
# ----------------------------------------------------------------------------
def make_chart(d, sup, res, plan, name, interval, fc=None, show_bars=120):
    x = d.tail(show_bars)
    fmt = "%d %b %y" if interval in ("1d", "1wk") else "%d %b %H:%M"
    labels = [ts.strftime(fmt) for ts in x.index]
    flabels = future_labels(d, fc["horizon"], interval) if fc else []

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        row_heights=[0.64, 0.18, 0.18])
    fig.add_trace(go.Candlestick(
        x=labels, open=x["Open"], high=x["High"], low=x["Low"], close=x["Close"], name=name,
        increasing=dict(line=dict(color=P["up"], width=1), fillcolor=P["up"]),
        decreasing=dict(line=dict(color=P["down"], width=1), fillcolor=P["down"])), 1, 1)
    for n in (20, 50, 200):
        fig.add_trace(go.Scatter(x=labels, y=x[f"EMA{n}"], name=f"EMA{n}",
                                 line=dict(width=1.4, color=P[f"ema{n}"]), opacity=0.9), 1, 1)

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
        for row in (2, 3):
            fig.add_trace(go.Scatter(mode="lines", x=flabels, y=[None] * len(flabels), showlegend=False, hoverinfo="skip"), row, 1)

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

    fig.add_trace(go.Scatter(x=labels, y=x["RSI"], name="RSI", line=dict(color=P["ema200"], width=1.4),
                             showlegend=False), 2, 1)
    fig.add_hrect(y0=30, y1=70, fillcolor=P["muted"], opacity=0.06, line_width=0, row=2, col=1)
    fig.add_hline(y=70, line=dict(color=rgba(P["down"], .6), width=1, dash="dot"), row=2, col=1)
    fig.add_hline(y=30, line=dict(color=rgba(P["up"], .6), width=1, dash="dot"), row=2, col=1)
    colors = np.where(x["MACDhist"] >= 0, rgba(P["up"], .75), rgba(P["down"], .75))
    fig.add_trace(go.Bar(x=labels, y=x["MACDhist"], name="MACD hist", marker_color=colors, showlegend=False), 3, 1)
    fig.add_trace(go.Scatter(x=labels, y=x["MACD"], name="MACD", line=dict(color=P["ema50"], width=1.2),
                             showlegend=False), 3, 1)
    fig.add_trace(go.Scatter(x=labels, y=x["MACDsig"], name="Signal", line=dict(color=P["warn"], width=1.2),
                             showlegend=False), 3, 1)

    fig.update_layout(
        template="plotly_dark", height=780, margin=dict(l=8, r=96, t=36, b=8),
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
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1, title_font=dict(color=P["muted"], size=11))
    fig.update_yaxes(title_text="MACD", row=3, col=1, title_font=dict(color=P["muted"], size=11))
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
.block-container{{padding-top:1.6rem;padding-bottom:3rem;max-width:1520px}}
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
.foot{{color:{P['muted']};font-size:12px;line-height:1.6;border-top:1px solid {P['border']};padding-top:14px;margin-top:26px}}
</style>""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    market = st.radio("Screener market", ["Bursa Malaysia", "US", "My own list"], index=0)
    custom = ""
    if market == "My own list":
        custom = st.text_area("Tickers (comma separated)", "MAYBANK, TENAGA, 5398, NVDA, TSLA, AAPL")
    tf_name = st.selectbox("Timeframe", list(INTERVALS.keys()), index=0)
    interval, period, unit = INTERVALS[tf_name]
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

if market == "Bursa Malaysia":
    universe = tuple(BURSA.keys())
elif market == "US":
    universe = tuple(US.keys())
else:
    universe = tuple(dict.fromkeys(t for t in (normalise(s) for s in custom.split(",")) if t))

tab_scr, tab_an, tab_div, tab_ipo, tab_tv, tab_learn = st.tabs(
    ["🔎  Screener", "📊  Analysis & Forecast", "💰  Dividends", "🚀  IPO", "📺  TradingView", "🎓  Learn"])
prices = {}

# ---------------- Screener ----------------
with tab_scr:
    rows = []
    if not universe:
        st.info("Enter at least one ticker in the sidebar.")
    else:
        with st.spinner(f"Scanning {len(universe)} stocks..."):
            data = load_many(universe, interval, period)
        prices = {t: float(v["Close"].iloc[-1]) for t, v in data.items() if len(v)}
        for t, df in data.items():
            try:
                d, sup, res, sc, label, _ = analyse(df)
                p = trade_plan(d, sup, res, label, unit)
                last, prev = d.iloc[-1], d.iloc[-2]
                price = float(last["Close"])
                rows.append({
                    "Name": display_name(t), "Ticker": t, "Price": price,
                    "Change %": (price / float(prev["Close"]) - 1) * 100,
                    "Signal": f"{LABEL_DOT[label]} {label}", "_label": label, "Score": sc,
                    "RSI": float(last["RSI"]),
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
        flt = st.segmented_control("Filter", ["All", "Buy only", "Sell only", "Near support (<3%)", "R:R ≥ 2"],
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
    typed = c2.text_input("Or type a ticker", placeholder="e.g. GAMUDA, 5398, NVDA")
    horizon = c3.selectbox("Forecast ahead", [5, 10, 20, 30], index=1,
                           format_func=lambda v: f"{v} {unit.split()[-1]}")
    lookback = c4.selectbox("Trend based on", [30, 60, 120], index=1, format_func=lambda v: f"last {v} bars")
    ticker = normalise(typed) if typed.strip() else pick
    st.session_state["an_ticker"] = ticker

    with st.spinner(f"Fetching {ticker}..."):
        df = load_one(ticker, interval, period)

    if df.empty or len(df) < 60:
        st.error(f"Not enough data found for **{ticker}**. For Bursa use the 4-digit code + .KL (e.g. 1155.KL) or a name "
                 f"from the list; for US use the normal ticker (e.g. NVDA).")
    else:
        d, sup, res, sc, label, why = analyse(df)
        plan = trade_plan(d, sup, res, label, unit)
        fc = trend_forecast(d, horizon=horizon, lookback=min(lookback, len(d) - 2))
        last, prev = d.iloc[-1], d.iloc[-2]
        price = float(last["Close"])
        chg = (price / float(prev["Close"]) - 1) * 100
        chg_col = P["up"] if chg >= 0 else P["down"]
        dir_col = {"Uptrend": P["up"], "Downtrend": P["down"]}.get(fc["direction"], P["muted"])

        st.markdown(f"""
<div class="card"><div class="head">
  <div><div class="name">{display_name(ticker)} · {ticker}</div>
       <div><span class="price">{price:,.3f}</span><span class="chg" style="color:{chg_col}">{chg:+.2f}%</span></div></div>
  <div><div class="name">Signal</div>{pill(label, LABEL_COLOR[label])}</div>
  <div><div class="name">Score</div><div style="font-size:20px;font-weight:600">{sc:+d}<span class="muted"> / 100</span></div></div>
  <div><div class="name">Trend forecast</div>{pill(f"{fc['arrow']} {fc['direction']} · {fc['strength']}", dir_col)}</div>
  <div><div class="name">RSI</div><div style="font-size:20px;font-weight:600">{last['RSI']:.0f}</div></div>
  <div style="margin-left:auto" class="muted">Last data {d.index[-1].strftime('%d %b %Y %H:%M')}<br>{tf_name}</div>
</div></div>""", unsafe_allow_html=True)

        show_fc = st.toggle("Show trend forecast on chart", value=True)
        st.plotly_chart(make_chart(d, sup, res, plan, display_name(ticker), interval, fc if show_fc else None),
                        width="stretch", config={"displaylogo": False, "scrollZoom": True})
        st.markdown(f"<div class='note'>Green band = buy zone · red band = sell zone · dashed orange = stop loss · "
                    f"S/R lines show how many times price touched them · blue dashed = forecast with 68% (darker) "
                    f"and 95% (lighter) ranges</div>", unsafe_allow_html=True)
        st.write("")

        col1, col2, col3 = st.columns(3, gap="medium")

        # ---- Trade plan
        rr1 = "–" if np.isnan(plan["rr1"]) else f"1 : {plan['rr1']:.1f}"
        rr2 = "–" if np.isnan(plan["rr2"]) else f"1 : {plan['rr2']:.1f}"
        e = plan["entry"]
        col1.markdown(f"""
<div class="card"><h3>Trade plan</h3>
<div class="action">{plan['action']}</div>
<table class="kv">
<tr><td>Buy zone</td><td style="color:{P['up']}">{plan['buy_lo']:.3f} – {plan['buy_hi']:.3f}</td></tr>
<tr><td>Reference entry</td><td>{e:.3f}</td></tr>
<tr><td>Stop loss</td><td style="color:{P['warn']}">{plan['stop']:.3f} <span class="muted">({(plan['stop']/e-1)*100:.1f}%)</span></td></tr>
<tr><td>Target 1 · sell half</td><td style="color:{P['down']}">{plan['t1']:.3f} <span class="muted">({(plan['t1']/e-1)*100:+.1f}%)</span></td></tr>
<tr><td>Target 2</td><td style="color:{P['down']}">{plan['t2']:.3f} <span class="muted">({(plan['t2']/e-1)*100:+.1f}%)</span></td></tr>
<tr><td>Risk : Reward</td><td>{rr1} &nbsp;/&nbsp; {rr2}</td></tr>
</table>
<div class="when">{re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", plan['when'])}</div></div>""", unsafe_allow_html=True)

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
        col2.markdown(f"""
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

        # ---- Why + patterns
        reasons = "".join(
            f"<div class='reason'><b style='color:{P['up'] if pts > 0 else P['down']}'>{pts:+d}</b><span>{txt}</span></div>"
            for pts, txt in sorted(why, key=lambda w: -abs(w[0])))
        pats = learn.detect_patterns(d)
        if pats:
            prow = "".join(
                f"<div class='reason'><span>{ {'bull': '🟢', 'bear': '🔴'}.get(k, '⚪') }</span>"
                f"<span><b style='min-width:0;text-align:left'>{nm}</b> <span class='muted'>· "
                f"{'latest candle' if ago == 0 else f'{ago} candle' + ('s' if ago > 1 else '') + ' ago'}</span></span></div>"
                for nm, k, ago in pats)
        else:
            prow = "<div class='muted'>No clear pattern in the last 5 candles.</div>"
        col3.markdown(f"""
<div class="card"><h3>Why this signal</h3>{reasons}</div>
<div class="card"><h3>Candlestick patterns · last 5 candles</h3>{prow}
<div class="note">Meanings are in 🎓 Learn → Candlestick Patterns.</div></div>""", unsafe_allow_html=True)

        with st.expander("Support & resistance levels (table)"):
            lv = pd.DataFrame(
                [{"Type": "Resistance", "Level": l, "Touches": str(n) if n else "projected", "Distance %": (l / price - 1) * 100} for l, n in reversed(res)]
                + [{"Type": "▶ Current price", "Level": price, "Touches": "", "Distance %": 0.0}]
                + [{"Type": "Support", "Level": l, "Touches": str(n) if n else "projected", "Distance %": (l / price - 1) * 100} for l, n in sup])
            st.dataframe(lv, hide_index=True, width="stretch",
                         column_config={"Level": st.column_config.NumberColumn(format="%.3f"),
                                        "Distance %": st.column_config.NumberColumn(format="%+.1f%%")})

# ---------------- Dividends ----------------
with tab_div:
    extras.render_dividends(universe, prices, display_name, P, pill)

# ---------------- IPO ----------------
with tab_ipo:
    extras.render_ipo(P, pill)

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
