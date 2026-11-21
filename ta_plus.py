"""Advanced analysis engine (pure calculations, no Streamlit UI).

Data integrity, ATR volatility & risk, multi-timeframe analysis, candlestick context and history,
Bursa/US transaction costs and position sizing, walk-forward forecast backtesting, and the
rule-based narrative summary. Everything is computed locally from the downloaded candles.
"""
from pxfmt import fp, pxf  # noqa: F401
import math
from datetime import time as dtime

import numpy as np
import pandas as pd

import learn

# ============================================================================
# Data integrity
# ============================================================================
EXCHANGES = {
    "KL": ("Asia/Kuala_Lumpur", [(dtime(9, 0), dtime(12, 30)), (dtime(14, 30), dtime(17, 0))], "Bursa Malaysia"),
    "US": ("America/New_York", [(dtime(9, 30), dtime(16, 0))], "US market"),
    "CRYPTO": ("UTC", [(dtime(0, 0), dtime(23, 59, 59, 999999))], "Crypto (24/7, UTC candles)"),
}
IV_MINUTES = {"15m": 15, "1h": 60, "4h": 240}


def is_crypto(ticker):
    return str(ticker).upper().endswith(("-USD", "-USDT", "-MYR", "-EUR", "-BTC", "-ETH"))


def exchange_of(ticker):
    return "KL" if ticker.endswith(".KL") else "CRYPTO" if is_crypto(ticker) else "US"


def market_open(ticker, now=None):
    tz, sessions, _ = EXCHANGES[exchange_of(ticker)]
    now = now or pd.Timestamp.now(tz=tz)
    if exchange_of(ticker) == "CRYPTO":
        return True
    if now.weekday() >= 5:
        return False
    t = now.time()
    return any(a <= t < b for a, b in sessions)


def _to_exchange_tz(ts, tz):
    ts = pd.Timestamp(ts)
    return ts.tz_convert(tz) if ts.tzinfo is not None else ts.tz_localize(tz)


def candle_status(d: pd.DataFrame, interval: str, ticker: str, now=None):
    """Is the latest candle still forming? Is the data stale? Uses the exchange's own clock."""
    tz, _, ex_name = EXCHANGES[exchange_of(ticker)]
    now = now or pd.Timestamp.now(tz=tz)
    is_open = market_open(ticker, now)
    if d is None or d.empty:
        return {"forming": False, "stale": True, "last": None, "open": is_open, "exchange": ex_name,
                "note": "No data"}
    last = _to_exchange_tz(d.index[-1], tz)
    forming, stale = False, False
    crypto = exchange_of(ticker) == "CRYPTO"
    if interval == "1d":
        forming = is_open and last.date() == now.date()
        stale = ((now.normalize() - last.normalize()).days > 2) if crypto else \
            np.busday_count(np.datetime64(last.date()), np.datetime64(now.date())) > 3
    elif interval == "1wk":
        forming = (now.weekday() < 5 or is_open) and last.isocalendar()[:2] == now.isocalendar()[:2]
        stale = (now - last).days > 14
    elif interval == "1mo":
        forming = (last.year, last.month) == (now.year, now.month)
        stale = (now - last).days > 45
    else:
        mins = IV_MINUTES.get(interval, 60)
        bar_end = last + pd.Timedelta(minutes=mins)
        forming = is_open and now < bar_end
        if is_open:
            stale = (now - bar_end) > pd.Timedelta(minutes=3 * mins + 20)  # Yahoo Bursa feed is ~15 min delayed
        else:
            stale = np.busday_count(np.datetime64(last.date()), np.datetime64(now.date())) > 3
    age = now - last
    if age.days >= 1:
        age_txt = f"{age.days} day(s) ago"
    else:
        age_txt = f"{int(age.total_seconds() // 3600)}h {int(age.total_seconds() % 3600 // 60)}m ago"
    return {"forming": forming, "stale": stale, "last": last, "open": is_open, "exchange": ex_name,
            "age": age_txt}


def data_quality(d: pd.DataFrame, interval: str, ticker: str):
    st_ = candle_status(d, interval, ticker)
    zero_vol = int((d["Volume"] <= 0).sum()) if len(d) else 0
    gaps = 0
    if interval == "1d" and len(d) > 2:
        dd_ = np.array(d.index.date, dtype="datetime64[D]")
        bd = np.busday_count(dd_[:-1], dd_[1:])
        gaps = int((bd > 3).sum())  # more than ~3 business days between candles (beyond public holidays)
    return {
        "Source": d.attrs.get("source", "Yahoo Finance (yfinance)"),
        "Fetched": d.attrs.get("fetched", "n/a"),
        "Timeframe": interval, "Candles": len(d),
        "First candle": str(d.index[0]) if len(d) else "N/A",
        "Last candle": str(d.index[-1]) if len(d) else "N/A",
        "Rows removed in cleaning": d.attrs.get("dropped_rows", 0),
        "Zero-volume candles": zero_vol, "Unusual gaps between daily candles": gaps,
        "Latest candle": "forming (live)" if st_["forming"] else "complete",
        "Stale": "YES" if st_["stale"] else "no", "Market now": "open" if st_["open"] else "closed",
    }, st_


# ============================================================================
# ATR volatility & risk
# ============================================================================
def volatility(d: pd.DataFrame):
    atr = d["ATR"]
    atrp = (atr / d["Close"] * 100).dropna()
    if len(atrp) < 20:
        return None
    cur = float(atrp.iloc[-1])
    hist = atrp.tail(250)
    enough = len(hist) >= 60
    pct = float((hist < cur).mean() * 100) if enough else None
    if pct is None:
        cls = "N/A"
    elif pct < 100 / 3:
        cls = "Low"
    elif pct < 200 / 3:
        cls = "Moderate"
    else:
        cls = "High"
    return {"atr": float(atr.iloc[-1]), "atrp": cur, "pct": pct, "cls": cls, "median": float(hist.median()),
            "n": len(hist)}


# ============================================================================
# Transaction costs, tick sizes, position sizing
# ============================================================================
def bursa_tick(price):
    if price < 1:
        return 0.005
    if price < 10:
        return 0.01
    if price < 100:
        return 0.02
    return 0.10


def round_tick(price, ticker, mode="nearest"):
    tick = bursa_tick(price) if ticker.endswith(".KL") else 0.01
    f = {"down": math.floor, "up": math.ceil}.get(mode, round)
    return round(f(price / tick - (1e-9 if mode == "up" else 0)) * tick, 4)


def fees(value, ticker, cfg):
    """Cost of ONE side (buy or sell) of a trade."""
    if value <= 0:
        return {"total": 0.0}
    if ticker.endswith(".KL"):
        brokerage = max(value * cfg["brk_pct"] / 100, cfg["brk_min"])
        clearing = min(value * 0.0003, 1000.0)                 # 0.03%, capped RM1,000
        stamp = min(float(math.ceil(value / 1000.0)), 1000.0)  # RM1 per RM1,000 or part, capped RM1,000
        platform = cfg["platform"] * 1.08                      # platform fee carries 8% SST
        tot = brokerage + clearing + stamp + platform
        return {"brokerage": brokerage, "clearing": clearing, "stamp": stamp, "platform": platform, "total": tot}
    commission = cfg["us_commission"]
    return {"commission": commission, "total": commission}


def position_size(entry, stop, ticker, cfg):
    """Largest lot-sized position whose loss at the stop (incl. both sides' fees) stays within the risk budget
    and whose cost stays within capital."""
    lot = 100 if ticker.endswith(".KL") else 1
    per = entry - stop
    if per <= 0 or entry <= 0:
        return None
    budget_pct = cfg["capital"] * cfg["risk_pct"] / 100
    budget = min(budget_pct, cfg["max_loss"]) if cfg["max_loss"] > 0 else budget_pct
    binding = "risk %" if budget == budget_pct else "max loss cap"
    shares = int(budget / per) // lot * lot
    shares = min(shares, int(cfg["capital"] / entry) // lot * lot)
    while shares > 0:
        fb, fs = fees(shares * entry, ticker, cfg), fees(shares * stop, ticker, cfg)
        loss = shares * per + fb["total"] + fs["total"]
        cost = shares * entry + fb["total"]
        if loss <= budget and cost <= cfg["capital"]:
            if cost > 0.98 * cfg["capital"]:
                binding = "capital"
            return {"shares": shares, "lots": shares // lot if lot == 100 else None, "lot": lot, "cost": cost,
                    "loss": loss, "buy_fees": fb, "sell_fees_stop": fs, "budget": budget, "binding": binding,
                    "per_share_risk": per}
        shares -= lot
    return {"shares": 0, "lots": 0, "lot": lot, "cost": 0.0, "loss": 0.0, "budget": budget, "binding": binding,
            "per_share_risk": per, "buy_fees": {"total": 0}, "sell_fees_stop": {"total": 0}}


def net_pnl(shares, entry, exit_price, ticker, cfg):
    buy = shares * entry + fees(shares * entry, ticker, cfg)["total"]
    sell = shares * exit_price - fees(shares * exit_price, ticker, cfg)["total"]
    return sell - buy


def break_even(shares, entry, ticker, cfg):
    if shares <= 0:
        return None
    lo, hi = entry, entry * 1.5
    for _ in range(60):
        mid = (lo + hi) / 2
        if net_pnl(shares, entry, mid, ticker, cfg) >= 0:
            hi = mid
        else:
            lo = mid
    return hi


def split_partial(shares, lot):
    first = (shares // 2) // lot * lot
    if first == 0:
        return shares, 0
    return first, shares - first


# ============================================================================
# Multi-timeframe analysis
# ============================================================================
TIMEFRAMES = [  # key, label, yahoo interval (4h is built from 1h), period
    ("15m", "15 Minutes", "15m", "60d"),
    ("1h", "1 Hour", "1h", "730d"),
    ("4h", "4 Hours", "4h", "730d"),
    ("1d", "Daily", "1d", "5y"),
    ("1wk", "Weekly", "1wk", "10y"),
    ("1mo", "Monthly", "1mo", "max"),
]
SWING_WEIGHTS = {"1wk": 3, "1d": 2, "4h": 1}
MIN_BARS = {"rsi": 30, "macd": 60, "ema50": 100, "ema200": 250}


def rsi_condition(r):
    if r > 70:
        return "Overbought", 0
    if r >= 55:
        return "Bullish", 1
    if r > 45:
        return "Neutral", 0
    if r >= 30:
        return "Bearish", -1
    return "Oversold", 0


def macd_condition(d):
    last = d.iloc[-1]
    above_sig = last["MACD"] > last["MACDsig"]
    above_zero = last["MACD"] > 0
    h = d["MACDhist"].tail(4).values
    cross = None
    for k in range(len(h) - 1, 0, -1):
        if h[k] > 0 >= h[k - 1]:
            cross = "bullish crossover"
            break
        if h[k] < 0 <= h[k - 1]:
            cross = "bearish crossover"
            break
    txt = ("MACD > signal" if above_sig else "MACD < signal") + " · " + ("above zero" if above_zero else "below zero")
    if cross:
        txt += f" · {cross} (last 3 bars)"
    return txt, (1 if above_sig else -1) + (1 if above_zero else -1)


def tf_analysis(df: pd.DataFrame, add_indicators):
    """One timeframe → trend, EMA alignment, RSI, MACD and an overall signal. N/A where data is insufficient."""
    n = len(df)
    if n < MIN_BARS["macd"]:
        return {"candles": n, "available": False, "reason": f"only {n} candles (needs ≥ {MIN_BARS['macd']})"}
    d = add_indicators(df)
    last = d.iloc[-1]
    price = float(last["Close"])
    ok50, ok200 = n >= MIN_BARS["ema50"], n >= MIN_BARS["ema200"]
    e20 = float(last["EMA20"])
    e50 = float(last["EMA50"]) if ok50 else None
    e200 = float(last["EMA200"]) if ok200 else None
    score = 0
    # trend direction
    ref = d["EMA50"] if ok50 else d["EMA20"]
    slope = float(ref.iloc[-1] - ref.iloc[-11]) if len(ref) > 11 else 0.0
    refv = float(ref.iloc[-1])
    if price > refv and slope > 0:
        trend, tp = "Up", 2
    elif price < refv and slope < 0:
        trend, tp = "Down", -2
    else:
        trend, tp = "Sideways", 0
    score += tp
    # EMA alignment
    if ok200:
        if e20 > e50 > e200:
            ema, ep = "Bullish stack (20 > 50 > 200)", 2
        elif e20 < e50 < e200:
            ema, ep = "Bearish stack (20 < 50 < 200)", -2
        else:
            ema, ep = "Mixed", 0
    elif ok50:
        ema, ep = (("20 > 50 (bullish)", 1) if e20 > e50 else ("20 < 50 (bearish)", -1))
        ema += " · EMA200 N/A"
    else:
        ema, ep = "N/A (too few candles)", 0
    score += ep
    rsi = float(last["RSI"])
    rsi_txt, rp = rsi_condition(rsi)
    score += rp
    macd_txt, mp = macd_condition(d)
    score += mp
    if score >= 5:
        sig = "Strong Bullish"
    elif score >= 2:
        sig = "Bullish"
    elif score <= -5:
        sig = "Strong Bearish"
    elif score <= -2:
        sig = "Bearish"
    else:
        sig = "Neutral"
    return {"candles": n, "available": True, "price": price, "trend": trend, "ema": ema, "rsi": rsi,
            "rsi_txt": rsi_txt, "macd_txt": macd_txt, "score": score, "signal": sig, "last": d.index[-1],
            "e20": e20, "e50": e50, "e200": e200}


def mtf_summary(rows: dict):
    """rows: key -> tf_analysis dict. Detect disagreement and swing (W/D/4H) alignment."""
    avail = {k: r for k, r in rows.items() if r.get("available")}
    bulls = [k for k, r in avail.items() if "Bullish" in r["signal"]]
    bears = [k for k, r in avail.items() if "Bearish" in r["signal"]]
    disagree = bool(bulls and bears)
    sw = {k: avail[k] for k in SWING_WEIGHTS if k in avail}
    if sw:
        wscore = sum(SWING_WEIGHTS[k] * np.sign(r["score"]) * (1 if abs(r["score"]) >= 2 else 0) for k, r in sw.items())
        wmax = sum(SWING_WEIGHTS[k] for k in sw)
        sigs = {k: ("bull" if "Bullish" in r["signal"] else "bear" if "Bearish" in r["signal"] else "neutral")
                for k, r in sw.items()}
        if all(v == "bull" for v in sigs.values()):
            swing = "Aligned bullish"
        elif all(v == "bear" for v in sigs.values()):
            swing = "Aligned bearish"
        elif "bull" in sigs.values() and "bear" in sigs.values():
            swing = "Conflict"
        else:
            swing = "Partly aligned"
    else:
        wscore, wmax, swing, sigs = 0, 0, "N/A", {}
    return {"bulls": bulls, "bears": bears, "disagree": disagree, "swing": swing, "swing_score": wscore,
            "swing_max": wmax, "swing_sigs": sigs}


# ============================================================================
# Candlestick patterns: context, confirmation, history
# ============================================================================
def pattern_report(d: pd.DataFrame, sup, res, lookback=5, fwd=5):
    o, h, l, c = learn._arrays(d)
    n = len(d)
    hist = learn.scan_history(d)
    atr = d["ATR"].values
    levels_s = [lv for lv, _ in sup]
    levels_r = [lv for lv, _ in res]
    out = []
    for i, name, kind in hist:
        if i < n - lookback:
            continue
        det = learn.detect_at(o, h, l, c, i)
        nb, rule = det[2], det[3]
        # formation details
        candles = []
        for k in range(i - nb + 1, i + 1):
            rg = max(h[k] - l[k], 1e-9)
            candles.append({
                "Candle": str(d.index[k])[:16], "Open": o[k], "High": h[k], "Low": l[k], "Close": c[k],
                "Colour": "green" if c[k] > o[k] else "red" if c[k] < o[k] else "doji",
                "Body % of range": abs(c[k] - o[k]) / rg * 100,
                "Upper wick %": (h[k] - max(o[k], c[k])) / rg * 100,
                "Lower wick %": (min(o[k], c[k]) - l[k]) / rg * 100})
        # context
        prior = "decline" if c[i - 1] < c[i - 6] else "rise"
        a = atr[i]
        lv_all = levels_s + levels_r
        body_hi, body_lo = max(o[i], c[i]), min(o[i], c[i])
        d_below = min([abs(l[i] - lv) for lv in lv_all if lv <= body_hi] or [np.inf])   # level under the candle
        d_above = min([abs(h[i] - lv) for lv in lv_all if lv >= body_lo] or [np.inf])   # level over the candle
        near_s = d_below <= a and d_below < d_above
        near_r = d_above <= a and d_above <= d_below
        if kind == "bull":
            ctx_ok = prior == "decline" or near_s
            ctx = (f"after a {prior}" + (", at support" if near_s else "") + (", but right under resistance" if near_r else ""))
            ctx_grade = "Good" if (prior == "decline" and near_s and not near_r) else ("OK" if ctx_ok and not near_r else "Weak")
        elif kind == "bear":
            ctx_ok = prior == "rise" or near_r
            ctx = (f"after a {prior}" + (", at resistance" if near_r else "") + (", but right on support" if near_s else ""))
            ctx_grade = "Good" if (prior == "rise" and near_r and not near_s) else ("OK" if ctx_ok and not near_s else "Weak")
        else:
            ctx, ctx_grade = f"after a {prior} (indecision)", "Neutral"
        # confirmation: next up-to-3 candles must close beyond the pattern's high (bull) or low (bear)
        top, bot = max(h[i - nb + 1:i + 1]), min(l[i - nb + 1:i + 1])
        if i == n - 1:
            conf = "Pending: needs the next candle"
        elif kind == "neutral":
            nxt = c[i + 1]
            conf = ("Resolved up (closed above its high)" if nxt > h[i] else
                    "Resolved down (closed below its low)" if nxt < l[i] else "Still undecided")
        else:
            conf = "Not confirmed yet"
            for k in range(i + 1, min(n, i + 4)):
                if (kind == "bull" and c[k] > top) or (kind == "bear" and c[k] < bot):
                    conf = f"Confirmed (closed beyond the pattern {k - i} candle(s) later)"
                    break
                if (kind == "bull" and c[k] < bot) or (kind == "bear" and c[k] > top):
                    conf = f"Failed (closed the wrong way {k - i} candle(s) later)"
                    break
        # historical performance: earlier occurrences only, with a full forward window (no look-ahead)
        past = [j for j, nm, _ in hist if nm == name and j < i and j + fwd < n]
        rets = np.array([(c[j + fwd] / c[j] - 1) * 100 for j in past])
        if len(rets) >= 5:
            win = float((rets > 0).mean() * 100) if kind != "bear" else float((rets < 0).mean() * 100)
            base_all = np.array([(c[j + fwd] / c[j] - 1) * 100 for j in range(10, i - fwd)])
            base = float((base_all > 0).mean() * 100) if kind != "bear" else float((base_all < 0).mean() * 100)
            perf = {"n": len(rets), "win": win, "avg": float(rets.mean()), "median": float(np.median(rets)),
                    "base": base}
        else:
            perf = {"n": len(rets)}
        # how to use it
        if kind == "neutral":
            use = "Indecision only: wait for the next candle to show direction."
        elif conf.startswith("Confirmed") and ctx_grade in ("Good", "OK"):
            use = "Actionable: confirmed in a sensible location."
        elif conf.startswith("Failed"):
            use = "Ignore: the pattern already failed."
        elif ctx_grade == "Weak":
            use = "Low reliability: wrong location for this pattern; don't trade it on its own."
        else:
            use = "Watch: wait for confirmation before acting."
        out.append({"name": name, "kind": kind, "bars_ago": n - 1 - i, "rule": rule, "candles": candles,
                    "context": ctx, "ctx_grade": ctx_grade, "confirmation": conf, "perf": perf, "use": use,
                    "fwd": fwd})
    out.sort(key=lambda r: r["bars_ago"])
    return out


# ============================================================================
# Forecast backtesting (walk-forward, out-of-sample)
# ============================================================================
def walk_forward(close: np.ndarray, horizon: int, lookback: int, step: int | None = None):
    """At each origin t, fit only on close[t-lookback+1 .. t] and predict close[t+horizon]."""
    close = np.asarray(close, float)
    step = step or max(1, horizon // 2)
    rows = []
    for t in range(lookback, len(close) - horizon, step):
        w = np.log(close[t - lookback + 1:t + 1])
        slope = np.polyfit(np.arange(lookback), w, 1)[0]
        rets = np.diff(np.log(close[t - lookback:t + 1]))
        sigma = rets.std()
        c0 = close[t]
        pred = c0 * math.exp(slope * horizon)
        b1, b2 = sigma * math.sqrt(horizon), 1.96 * sigma * math.sqrt(horizon)
        rows.append((t, c0, pred, close[t + horizon], pred * math.exp(-b1), pred * math.exp(b1),
                     pred * math.exp(-b2), pred * math.exp(b2)))
    if not rows:
        return None, None
    df = pd.DataFrame(rows, columns=["t", "origin", "pred", "actual", "lo68", "hi68", "lo95", "hi95"])
    pdir = np.sign(df["pred"] - df["origin"])
    adir = np.sign(df["actual"] - df["origin"])
    valid = (pdir != 0) & (adir != 0)
    err = df["pred"] - df["actual"]
    naive_err = df["origin"] - df["actual"]
    m = {
        "n": len(df),
        "dir_acc": float((pdir[valid] == adir[valid]).mean() * 100) if valid.any() else np.nan,
        "up_rate": float((adir[valid] > 0).mean() * 100) if valid.any() else np.nan,
        "mae": float(err.abs().mean()), "rmse": float(np.sqrt((err ** 2).mean())),
        "mape": float((err.abs() / df["actual"]).mean() * 100),
        "naive_mae": float(naive_err.abs().mean()),
        "cov68": float(((df["actual"] >= df["lo68"]) & (df["actual"] <= df["hi68"])).mean() * 100),
        "cov95": float(((df["actual"] >= df["lo95"]) & (df["actual"] <= df["hi95"])).mean() * 100),
    }
    m["skill"] = (1 - m["mae"] / m["naive_mae"]) * 100 if m["naive_mae"] > 0 else np.nan
    return df, m


def calibration_note(m):
    if m is None:
        return "N/A"
    c68, c95 = m["cov68"], m["cov95"]
    if c95 < 85 or c68 < 55:
        return "too narrow: real outcomes landed outside the bands more often than the labels suggest"
    if c95 > 99 and c68 > 85:
        return "too wide: bands were overly cautious"
    return "reasonably calibrated on this history"


# ============================================================================
# Narrative summary (rule-based, from calculated values only)
# ============================================================================
def _cur(ticker):
    return "RM" if ticker.endswith(".KL") else "$"


def narrative(name, ticker, d, sup, res, plan, va, ax, mtf, fc, vol, pats, status, bt=None):
    cur = _cur(ticker)
    last = d.iloc[-1]
    price = float(last["Close"])
    e20, e50, e200 = float(last["EMA20"]), float(last["EMA50"]), float(last["EMA200"])
    enough200 = len(d) >= MIN_BARS["ema200"]
    rsi = float(last["RSI"])
    macd_above = last["MACD"] > last["MACDsig"]
    macd_pos = last["MACD"] > 0
    atr = float(last["ATR"])
    s1, r1 = sup[0][0], res[0][0]
    s2, r2 = sup[1][0], res[1][0]

    # long-term trend
    if not enough200:
        lt = "a long-term trend that can't be judged yet (fewer than 250 candles)"
        lt_short = "unknown long-term trend"
    elif price > e200 and e50 > e200:
        lt, lt_short = "a longer-term uptrend", "uptrend"
    elif price < e200 and e50 < e200:
        lt, lt_short = "a longer-term downtrend", "downtrend"
    else:
        lt, lt_short = "a mixed longer-term picture", "mixed"
    # short-term momentum
    votes = sum([price > e20, rsi > 50, macd_above])
    mom = {3: "positive", 2: "mildly positive", 1: "mildly negative", 0: "negative"}[votes]
    if 45 <= rsi <= 55 and votes in (1, 2):
        mom = "neutral"
    # location
    near_r = (r1 - price) <= atr and res[0][1] > 0
    near_s = (price - s1) <= atr
    if near_r:
        loc = f"Price is approaching resistance at {cur}{fp(r1)}"
    elif near_s:
        loc = f"Price is sitting near support at {cur}{fp(s1)}"
    else:
        loc = f"Price is between support at {cur}{fp(s1)} and resistance at {cur}{fp(r1)}"
    # volume confirmation
    if va is None:
        vtxt, vconf = "Volume data is not available.", None
    else:
        vconf = va["score"] >= 3
        vtxt = ("Volume confirmation is established." if vconf else
                "Volume confirmation is not yet established." if va["score"] > -2 else
                "Volume is contradicting the move.")
    head = (f"{name} is currently in {lt}, but short-term momentum is {mom}." if lt_short in ("uptrend", "downtrend")
            and ((lt_short == "uptrend") != (votes >= 2)) else
            f"{name} is currently in {lt}, and short-term momentum is {mom}.")
    head += f" {loc}. {vtxt} "
    head += (f"A breakout above {cur}{fp(r1)} may provide a potential entry opportunity, while a rejection could lead "
             f"to a retest of {cur}{fp(s1)}.")

    sec = {}
    sec["Overall trend"] = (f"Price {cur}{fp(price)} vs EMA20 {fp(e20)} / EMA50 {fp(e50)} / EMA200 "
                            + (f"{fp(e200)}" if enough200 else "N/A") + f" → {lt}.")
    sec["Short-term momentum"] = (f"Price is {'above' if price > e20 else 'below'} EMA20, RSI {rsi:.1f}, MACD line "
                                  f"{'above' if macd_above else 'below'} its signal line → {mom}.")
    sec["Long-term trend"] = (f"EMA50 is {'above' if e50 > e200 else 'below'} EMA200 and price is "
                              f"{'above' if price > e200 else 'below'} EMA200." if enough200 else
                              "Not enough history for EMA200 (needs ≥ 250 candles).")
    sec["Support & resistance"] = (f"Nearest support {cur}{fp(s1)} ({(s1 / price - 1) * 100:+.1f}%, "
                                   f"{sup[0][1] or 'projected'} touches), next {cur}{fp(s2)}. Nearest resistance "
                                   f"{cur}{fp(r1)} ({(r1 / price - 1) * 100:+.1f}%), next {cur}{fp(r2)}.")
    if va:
        ud_txt = "∞" if np.isinf(va["ud"]) else f"{va['ud']:.2f}"
        sec["Volume behaviour"] = (f"RVOL {va['rvol']:.2f}× (latest vs 20-bar average), up/down volume {ud_txt}"
                                   f" → {va['verdict'].split(' ', 1)[1]}.")
    else:
        sec["Volume behaviour"] = "N/A"
    rsi_c = rsi_condition(rsi)[0]
    sec["RSI & MACD"] = (f"RSI {rsi:.1f} ({rsi_c.lower()}). MACD line {float(last['MACD']):.4f} is "
                         f"{'above' if macd_above else 'below'} the signal line {float(last['MACDsig']):.4f} "
                         f"({'bullish' if macd_above else 'bearish'} momentum) and {'above' if macd_pos else 'below'} "
                         f"the zero line ({'positive' if macd_pos else 'negative'} MACD regime).")
    sec["ADX trend strength"] = (f"ADX {ax['adx']:.1f} with +DI {ax['pdi']:.1f} / −DI {ax['mdi']:.1f} → "
                                 f"{ax['regime']}" + (" (strengthening)." if ax["rising"] else
                                                      " (weakening)." if ax["falling"] else ".")) if ax else "N/A"
    if mtf:
        sec["Multi-timeframe"] = (f"Swing timeframes (Weekly/Daily/4H): {mtf['swing']}."
                                  + (f" Bullish on {', '.join(mtf['bulls'])}; bearish on {', '.join(mtf['bears'])}."
                                     if mtf["disagree"] else ""))
    else:
        sec["Multi-timeframe"] = "N/A"
    if vol:
        sec["Volatility"] = (f"ATR {fp(vol['atr'])} = {vol['atrp']:.2f}% of price per candle → {vol['cls']} "
                             f"volatility" + (f" ({vol['pct']:.0f}th percentile of the last {vol['n']} candles)."
                                              if vol["pct"] is not None else "."))
    vol_need = (f" with volume ≥ 1.5× average (≥ {va['avg'] * 1.5:,.0f} shares)" if va else "")
    sec["Bullish scenario"] = (f"A close above {cur}{fp(r1)}{vol_need} would confirm a breakout and open the way "
                               f"to {cur}{fp(r2)} ({(r2 / price - 1) * 100:+.1f}%).")
    sec["Bearish scenario"] = (f"A close below {cur}{fp(s1)} would expose {cur}{fp(s2)} "
                               f"({(s2 / price - 1) * 100:+.1f}%); the trade plan is invalid below {cur}{fp(plan['stop'])}.")
    conds = []
    if price <= plan["buy_hi"]:
        conds.append(f"Price is inside the buy zone {cur}{fp(plan['buy_lo'])}–{fp(plan['buy_hi'])}")
    else:
        conds.append(f"Pullback into {cur}{fp(plan['buy_lo'])}–{fp(plan['buy_hi'])}, or a breakout close above "
                     f"{cur}{fp(r1)}{vol_need}")
    if ax and ax["regime"] in ("Sideways Market",):
        conds.append("ADX below 20: prefer range entries near support over breakouts")
    elif ax and "Bearish" in ax["regime"]:
        conds.append("wait for +DI to cross back above −DI before buying")
    if not macd_above:
        conds.append("ideally a MACD bullish crossover (line crossing above signal)")
    bull_conf = [p for p in pats if p["kind"] == "bull" and p["confirmation"].startswith("Confirmed")]
    if bull_conf:
        conds.append(f"confirmed {bull_conf[0]['name']} supports an entry")
    sec["Suggested entry conditions"] = "; ".join(conds) + "."
    risks = [f"Invalidation: a close below {cur}{fp(plan['stop'])}"]
    if rsi > 70:
        risks.append(f"RSI {rsi:.0f} is overbought")
    if va and va["score"] <= -2:
        risks.append("the move lacks volume support")
    if ax and ax["falling"] and ax["regime"] != "Sideways Market":
        risks.append("ADX is falling (trend losing strength)")
    if mtf and mtf["disagree"]:
        risks.append("timeframes disagree")
    if vol and vol["cls"] == "High":
        risks.append("volatility is high: use smaller size")
    if status and status.get("stale"):
        risks.append("market data looks stale")
    if bt and bt[1] and bt[1]["dir_acc"] < 55:
        risks.append(f"the trend forecast has only been right on direction {bt[1]['dir_acc']:.0f}% of the time "
                     f"historically for this stock")
    sec["Risk factors & invalidation"] = "; ".join(risks) + "."
    return head, sec
