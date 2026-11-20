"""Technical indicators computed from OHLCV data (pandas only, no look-ahead: every value at bar t uses
bars up to and including t). Values inside an indicator's warm-up period are NaN, never estimated.

core.add_indicators already provides EMA 20/50/200, RSI(14), MACD(12,26,9), ATR(14), ADX/DMI(14), RVOL, OBV.
This module adds the rest and is the base of the Phase 5 analysis engine.
"""
import numpy as np
import pandas as pd


def ema(s, n):
    out = s.ewm(span=n, adjust=False, min_periods=n).mean()
    return out


def sma(s, n):
    return s.rolling(n, min_periods=n).mean()


def bollinger(c, n=20, k=2.0):
    mid = sma(c, n)
    sd = c.rolling(n, min_periods=n).std(ddof=0)
    return mid, mid + k * sd, mid - k * sd


def stochastic(h, l, c, n=14, k_smooth=3, d_smooth=3):
    """Slow stochastic: %K = SMA(k_smooth) of raw %K, %D = SMA(d_smooth) of %K."""
    ll, hh = l.rolling(n, min_periods=n).min(), h.rolling(n, min_periods=n).max()
    raw = 100 * (c - ll) / (hh - ll).replace(0, np.nan)
    k = raw.rolling(k_smooth, min_periods=k_smooth).mean()
    return k, k.rolling(d_smooth, min_periods=d_smooth).mean()


def heikin_ashi(d):
    ha = pd.DataFrame(index=d.index)
    ha["Close"] = (d["Open"] + d["High"] + d["Low"] + d["Close"]) / 4
    o = np.empty(len(d))
    o[0] = (d["Open"].iloc[0] + d["Close"].iloc[0]) / 2
    hc = ha["Close"].values
    for i in range(1, len(d)):
        o[i] = (o[i - 1] + hc[i - 1]) / 2
    ha["Open"] = o
    ha["High"] = pd.concat([d["High"], ha["Open"], ha["Close"]], axis=1).max(axis=1)
    ha["Low"] = pd.concat([d["Low"], ha["Open"], ha["Close"]], axis=1).min(axis=1)
    ha["Volume"] = d["Volume"]
    return ha


def session_vwap(d):
    """VWAP that restarts every trading day (meaningful on intraday candles)."""
    tp = (d["High"] + d["Low"] + d["Close"]) / 3
    v = d["Volume"].fillna(0)
    day = pd.Series(d.index.date, index=d.index)
    cum_pv = (tp * v).groupby(day).cumsum()
    cum_v = v.groupby(day).cumsum().replace(0, np.nan)
    return cum_pv / cum_v


# ============================================================================
# Full indicator set (Phase 5). Parameters are fixed and listed here so every page uses the same ones.
# ============================================================================
PARAMS = {
    "EMA": [9, 20, 50, 100, 200], "SMA": [20, 50, 100, 200], "RSI": 14, "MACD": (12, 26, 9),
    "Stochastic": (14, 3, 3), "Stoch RSI": (14, 14, 3, 3), "ADX": 14, "ROC": 12, "CCI": 20,
    "Bollinger": (20, 2.0), "ATR": 14, "Keltner": (20, 2.0), "Historical volatility": 20,
    "Volume SMA": 20, "RVOL": 20, "VWAP (rolling)": 20, "MFI": 14,
}


def stoch_rsi(rsi, n=14, k_smooth=3, d_smooth=3):
    lo, hi = rsi.rolling(n, min_periods=n).min(), rsi.rolling(n, min_periods=n).max()
    raw = 100 * (rsi - lo) / (hi - lo).replace(0, np.nan)
    k = raw.rolling(k_smooth, min_periods=k_smooth).mean()
    return k, k.rolling(d_smooth, min_periods=d_smooth).mean()


def roc(c, n=12):
    return (c / c.shift(n) - 1) * 100


def cci(h, l, c, n=20):
    tp = (h + l + c) / 3
    ma = tp.rolling(n, min_periods=n).mean()
    md = tp.rolling(n, min_periods=n).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - ma) / (0.015 * md.replace(0, np.nan))


def keltner(c, atr, n=20, k=2.0):
    mid = ema(c, n)
    return mid, mid + k * atr, mid - k * atr


def hist_vol(c, n=20, periods_per_year=252):
    """Annualised standard deviation of daily log returns, in %."""
    r = np.log(c / c.shift(1))
    return r.rolling(n, min_periods=n).std(ddof=1) * np.sqrt(periods_per_year) * 100


def rolling_vwap(d, n=20):
    """Volume-weighted average price of the last n candles (used on daily charts, where a session VWAP
    would just be the day's typical price)."""
    tp = (d["High"] + d["Low"] + d["Close"]) / 3
    v = d["Volume"].fillna(0)
    return (tp * v).rolling(n, min_periods=n).sum() / v.rolling(n, min_periods=n).sum().replace(0, np.nan)


def ad_line(d):
    """Accumulation/Distribution line (Chaikin): cumulative money-flow volume."""
    rng = (d["High"] - d["Low"]).replace(0, np.nan)
    mfm = ((d["Close"] - d["Low"]) - (d["High"] - d["Close"])) / rng
    return (mfm.fillna(0) * d["Volume"].fillna(0)).cumsum()


def mfi(d, n=14):
    tp = (d["High"] + d["Low"] + d["Close"]) / 3
    mf = tp * d["Volume"].fillna(0)
    up = mf.where(tp > tp.shift(1), 0.0)
    dn = mf.where(tp < tp.shift(1), 0.0)
    pos, neg = up.rolling(n, min_periods=n).sum(), dn.rolling(n, min_periods=n).sum()
    return 100 - 100 / (1 + pos / neg.replace(0, np.nan))


def obv(d):
    return (np.sign(d["Close"].diff()).fillna(0) * d["Volume"].fillna(0)).cumsum()


def slope(s, n=20):
    """Least-squares slope of the last n values per bar (NaN if not enough data)."""
    y = s.dropna().tail(n).values
    if len(y) < n:
        return np.nan
    x = np.arange(n)
    return float(np.polyfit(x, y, 1)[0])


def pivots(d, w=3):
    """Confirmed swing highs/lows: a bar whose high (low) is the highest (lowest) of the w bars on each side.
    The last w bars can't be confirmed yet, so they are never pivots (no look-ahead)."""
    h, l = d["High"].values, d["Low"].values
    hi, lo = [], []
    for i in range(w, len(d) - w):
        if h[i] == h[i - w:i + w + 1].max():
            hi.append(i)
        if l[i] == l[i - w:i + w + 1].min():
            lo.append(i)
    return hi, lo


def structure(d, w=3):
    """Market structure from the last two confirmed swing highs and lows."""
    hi, lo = pivots(d, w)
    if len(hi) < 2 or len(lo) < 2:
        return {"label": "N/A", "hh": None, "hl": None, "hi": hi, "lo": lo}
    h, l = d["High"].values, d["Low"].values
    hh, hl = h[hi[-1]] > h[hi[-2]], l[lo[-1]] > l[lo[-2]]
    if hh and hl:
        lab = "Higher highs & higher lows (uptrend structure)"
    elif not hh and not hl:
        lab = "Lower highs & lower lows (downtrend structure)"
    elif hh and not hl:
        lab = "Higher high but lower low (expanding / unstable)"
    else:
        lab = "Lower high but higher low (contracting / triangle)"
    return {"label": lab, "hh": bool(hh), "hl": bool(hl), "hi": hi, "lo": lo,
            "last_hi": float(h[hi[-1]]), "prev_hi": float(h[hi[-2]]), "last_lo": float(l[lo[-1]]),
            "prev_lo": float(l[lo[-2]])}


def trendline(d, kind="support", w=3, lookback=120):
    """Line through the last two confirmed swing lows (support) or highs (resistance) inside the lookback.
    Returns slope per bar, value at the last bar and the two anchor indices, or None."""
    x = d.tail(lookback)
    hi, lo = pivots(x, w)
    pts = lo if kind == "support" else hi
    if len(pts) < 2:
        return None
    i1, i2 = pts[-2], pts[-1]
    y = x["Low"].values if kind == "support" else x["High"].values
    m = (y[i2] - y[i1]) / (i2 - i1)
    now = y[i2] + m * (len(x) - 1 - i2)
    off = len(d) - len(x)
    return {"slope": float(m), "value": float(now), "i1": i1 + off, "i2": i2 + off, "y1": float(y[i1]),
            "y2": float(y[i2])}


def divergence(price, osc, w=3, lookback=60):
    """Regular divergence between price and an oscillator on the last two confirmed swing points.
    bearish: price higher high but oscillator lower high; bullish: price lower low but oscillator higher low."""
    x = pd.DataFrame({"High": price, "Low": price}).tail(lookback)
    o = osc.tail(lookback).values
    hi, lo = pivots(x, w)
    out = None
    if len(hi) >= 2 and not np.isnan(o[hi[-2]]) and not np.isnan(o[hi[-1]]):
        if x["High"].values[hi[-1]] > x["High"].values[hi[-2]] and o[hi[-1]] < o[hi[-2]]:
            out = ("bearish", len(x) - 1 - hi[-1])
    if len(lo) >= 2 and not np.isnan(o[lo[-2]]) and not np.isnan(o[lo[-1]]):
        if x["Low"].values[lo[-1]] < x["Low"].values[lo[-2]] and o[lo[-1]] > o[lo[-2]]:
            b = ("bullish", len(x) - 1 - lo[-1])
            out = b if out is None or b[1] < out[1] else out
    return out


FIB_RET = [0.236, 0.382, 0.5, 0.618, 0.786]
FIB_EXT = [1.272, 1.618, 2.0, 2.618]


def fibonacci(d, lookback=120):
    """Retracement and extension levels of the latest major swing inside the lookback.
    If the high came after the low the swing is up: retracements are pullback levels below the high and
    extensions are targets above it; the reverse for a down swing."""
    x = d.tail(lookback)
    if len(x) < 20:
        return None
    ih, il = int(np.argmax(x["High"].values)), int(np.argmin(x["Low"].values))
    hi, lo = float(x["High"].iloc[ih]), float(x["Low"].iloc[il])
    rng = hi - lo
    if rng <= 0:
        return None
    up = ih > il
    if up:
        ret = {f"{r * 100:.1f}%": hi - r * rng for r in FIB_RET}
        ext = {f"{e * 100:.1f}%": lo + e * rng for e in FIB_EXT}
    else:
        ret = {f"{r * 100:.1f}%": lo + r * rng for r in FIB_RET}
        ext = {f"{e * 100:.1f}%": hi - e * rng for e in FIB_EXT}
    return {"direction": "up" if up else "down", "high": hi, "low": lo, "high_date": x.index[ih],
            "low_date": x.index[il], "retracements": ret, "extensions": ext}


def compute_all(d):
    """Add every indicator column to a copy of an OHLCV frame that already went through core.add_indicators."""
    x = d.copy()
    c = x["Close"]
    for n in PARAMS["EMA"]:
        x[f"E{n}"] = ema(c, n)
    for n in PARAMS["SMA"]:
        x[f"S{n}"] = sma(c, n)
    x["STK"], x["STD"] = stochastic(x["High"], x["Low"], c)
    x["SRK"], x["SRD"] = stoch_rsi(x["RSI"])
    x["ROC"] = roc(c, PARAMS["ROC"])
    x["CCI"] = cci(x["High"], x["Low"], c, PARAMS["CCI"])
    x["BBM"], x["BBU"], x["BBL"] = bollinger(c, *PARAMS["Bollinger"])
    x["BBW"] = (x["BBU"] - x["BBL"]) / x["BBM"] * 100
    x["KCM"], x["KCU"], x["KCL"] = keltner(c, x["ATR"], *PARAMS["Keltner"])
    x["SQZ"] = (x["BBU"] < x["KCU"]) & (x["BBL"] > x["KCL"])
    x["HV"] = hist_vol(c, PARAMS["Historical volatility"])
    x["VWAP20"] = rolling_vwap(x, PARAMS["VWAP (rolling)"])
    x["OBV2"] = obv(x)
    x["AD"] = ad_line(x)
    x["MFI"] = mfi(x, PARAMS["MFI"])
    return x
