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
