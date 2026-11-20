"""Explainable technical assessment.

Five groups are scored separately from 0 (bearish) to 100 (bullish), 50 = neutral:
  Trend, Momentum, Volume, Volatility, Price action.
Each group is built from a few *distinct* pieces of evidence, every one mapped to −1 … +1 and weighted.
Indicators that carry the same information are shown for reference but not scored again (for example the
individual EMA crosses are folded into one moving-average structure item; Stoch RSI and CCI repeat what RSI
and the stochastic already say). The Technical Score is the weighted average of the available groups; the
weights are configurable. Scores describe the current evidence; they are not probabilities of success.
"""
import numpy as np
import pandas as pd

import indicators as ind

DEFAULT_WEIGHTS = {"Trend": 30, "Momentum": 25, "Volume": 20, "Price action": 20, "Volatility": 5}
GROUPS = list(DEFAULT_WEIGHTS)


def _ok(*v):
    return all(x is not None and not (isinstance(x, float) and np.isnan(x)) for x in v)


def _clip(v):
    return float(np.clip(v, -1, 1))


def _item(name, value, reading, points=None, weight=1.0, note=""):
    return {"name": name, "value": value, "reading": reading, "points": points, "weight": weight, "note": note}


def _info(name, value, reading, note=""):
    return _item(name, value, reading, None, 0.0, note)


def group_score(items):
    sc = [(i["points"], i["weight"]) for i in items if i["points"] is not None and i["weight"] > 0]
    if not sc:
        return None
    w = sum(x[1] for x in sc)
    return 50 + 50 * sum(p * wt for p, wt in sc) / w


def label_of(score):
    if score is None:
        return "N/A"
    return "Bullish" if score >= 60 else "Bearish" if score <= 40 else "Neutral"


def _f(v, fmt="{:,.3f}"):
    return "N/A" if not _ok(v) else fmt.format(v)


# ============================================================================
# Groups
# ============================================================================
def trend_group(x, mtf=None):
    last = x.iloc[-1]
    c = float(last["Close"])
    items = []
    checks = []
    for k in ("E20", "E50", "E200"):
        if _ok(last[k]):
            checks.append(c > last[k])
    for a, b in (("E20", "E50"), ("E50", "E200")):
        if _ok(last[a], last[b]):
            checks.append(last[a] > last[b])
    if checks:
        bull = sum(checks)
        pts = (bull - (len(checks) - bull)) / len(checks)
        parts = [f"price {'above' if c > last[k] else 'below'} EMA{k[1:]}" for k in ("E20", "E50", "E200") if _ok(last[k])]
        items.append(_item("Moving-average structure", f"{bull} of {len(checks)} bullish",
                           "; ".join(parts) + ("" if not _ok(last["E200"]) else
                                               f"; EMA50 {'above' if last['E50'] > last['E200'] else 'below'} EMA200"),
                           _clip(pts), 1.0, "Price vs EMA 20/50/200 and their order. Crosses are part of this, so "
                                            "they are not scored again."))
    s50 = ind.slope(x["E50"], 20)
    atr = float(last["ATR"]) if _ok(last["ATR"]) else None
    if _ok(s50) and atr:
        z = s50 * 20 / atr
        items.append(_item("Trend slope (EMA 50, 20 bars)", f"{z:+.2f} ATR",
                           "rising" if z > 0.3 else "falling" if z < -0.3 else "flat", _clip(z / 2), 1.0,
                           "How far the 50-bar average moved over 20 bars, in ATRs (volatility-adjusted)."))
    if _ok(last["ADX"], last["PDI"], last["MDI"]):
        dir_ = 1 if last["PDI"] > last["MDI"] else -1
        strength = float(np.clip((last["ADX"] - 15) / 20, 0, 1))
        items.append(_item("ADX / DMI", f"ADX {last['ADX']:.0f}, +DI {last['PDI']:.0f} / −DI {last['MDI']:.0f}",
                           ("strong " if last["ADX"] >= 25 else "weak " if last["ADX"] >= 20 else "no ") +
                           ("up" if dir_ > 0 else "down") + "trend" if last["ADX"] >= 20 else "sideways (ADX < 20)",
                           dir_ * strength, 1.0, "ADX = strength (not direction); +DI vs −DI = direction."))
    if mtf and mtf.get("swing_max"):
        z = mtf["swing_score"] / mtf["swing_max"]
        items.append(_item("Multi-timeframe confirmation", f"{mtf['swing']} ({mtf['swing_score']:+d})",
                           "weekly, daily and 4-hour trends agree" if abs(z) > 0.6 else "timeframes disagree",
                           _clip(z), 0.5, "Weighted agreement of the weekly, daily and 4-hour trend signals."))
    # reference only
    sma_txt = ", ".join(f"{'above' if c > last[f'S{n}'] else 'below'} SMA{n}" for n in (20, 50, 100, 200)
                        if _ok(last[f"S{n}"])) or "N/A (not enough history)"
    items.append(_info("Simple moving averages", "", sma_txt, "Shown for reference; same information as the EMAs."))
    for a, b, lab in ((9, 20, "EMA 9/20"), (20, 50, "EMA 20/50"), (50, 200, "EMA 50/200")):
        dir_, age = _cross(x[f"E{a}"], x[f"E{b}"], 250 if b == 200 else 60)
        name = {1: "golden cross" if b == 200 else "bullish cross", -1: "death cross" if b == 200 else "bearish cross"}
        items.append(_info(f"{lab} cross", "", "none in range" if not dir_ else f"{name[dir_]} {age} bar{'' if age == 1 else 's'} ago"))
    if _ok(last["E9"], last["E100"]):
        items.append(_info("EMA 9 / EMA 100", f"{_f(last['E9'])} / {_f(last['E100'])}",
                           f"price {'above' if c > last['E9'] else 'below'} EMA9, "
                           f"{'above' if c > last['E100'] else 'below'} EMA100"))
    return items


def _cross(fast, slow, look=60):
    diff = (fast - slow).dropna().tail(look + 1)
    sg = np.sign(diff.values)
    for i in range(len(sg) - 1, 0, -1):
        if sg[i] != 0 and sg[i - 1] != 0 and sg[i] != sg[i - 1]:
            return int(sg[i]), len(sg) - 1 - i
    return 0, None


def momentum_group(x):
    last, prev = x.iloc[-1], x.iloc[-2]
    items = []
    if _ok(last["RSI"]):
        r = float(last["RSI"])
        state = "overbought (>70): strong but stretched" if r >= 70 else "oversold (<30): weak but stretched" \
            if r <= 30 else "bullish zone (50–70)" if r > 50 else "bearish zone (30–50)"
        items.append(_item("RSI (14)", f"{r:.0f}", state, _clip((r - 50) / 25), 1.0))
    if _ok(last["MACD"], last["MACDsig"], last["MACDhist"], prev["MACDhist"]):
        above = last["MACD"] > last["MACDsig"]
        rising = last["MACDhist"] > prev["MACDhist"]
        pts = (0.5 if above else -0.5) + (0.5 if rising else -0.5)
        items.append(_item("MACD (12, 26, 9)", f"hist {last['MACDhist']:+.4f}",
                           f"{'above' if above else 'below'} signal, histogram {'rising (accelerating)' if rising else 'falling (weakening)'}",
                           _clip(pts), 1.0, "Above/below signal = direction; histogram slope = acceleration."))
    if _ok(last["ROC"], last["ATR"]) and last["Close"]:
        atrp = last["ATR"] / last["Close"] * 100
        z = last["ROC"] / (atrp * np.sqrt(12)) if atrp else 0
        items.append(_item("Rate of change (12)", f"{last['ROC']:+.2f}%",
                           "strong upward" if z > 1 else "strong downward" if z < -1 else "modest", _clip(z / 1.5), 1.0,
                           "12-bar % change compared with the stock's normal volatility."))
    if _ok(last["STK"], last["STD"]):
        k = float(last["STK"])
        items.append(_item("Stochastic (14, 3, 3)", f"%K {k:.0f} / %D {last['STD']:.0f}",
                           ("overbought" if k > 80 else "oversold" if k < 20 else "mid-range") +
                           (", %K above %D" if k > last["STD"] else ", %K below %D"), _clip((k - 50) / 30), 0.5,
                           "Half weight: it overlaps with RSI."))
    div_r = ind.divergence(x["Close"], x["RSI"])
    div_m = ind.divergence(x["Close"], x["MACD"])
    divs = [(dv, src) for dv, src in ((div_r, "RSI"), (div_m, "MACD")) if dv and dv[1] <= 15]
    if divs:
        (kind, ago), src = sorted(divs, key=lambda z: z[0][1])[0]
        both = len(divs) == 2 and divs[0][0][0] == divs[1][0][0]
        items.append(_item("Divergence", f"{kind} ({'RSI & MACD' if both else src})", f"swing point {ago} bars ago: "
                           + ("price made a lower low but momentum a higher low" if kind == "bullish"
                              else "price made a higher high but momentum a lower high"),
                           0.8 if kind == "bullish" else -0.8, 1.0,
                           "Counted once even if RSI and MACD both show it. Divergence warns early but needs price "
                           "confirmation."))
    else:
        items.append(_info("Divergence", "none", "no regular RSI/MACD divergence on the last swings"))
    if _ok(last["SRK"]):
        items.append(_info("Stoch RSI (14, 14, 3, 3)", f"{last['SRK']:.0f}",
                           "overbought" if last["SRK"] > 80 else "oversold" if last["SRK"] < 20 else "mid-range",
                           "Reference only: a faster version of RSI."))
    if _ok(last["CCI"]):
        items.append(_info("CCI (20)", f"{last['CCI']:.0f}",
                           "above +100 (strong / overbought)" if last["CCI"] > 100 else
                           "below −100 (weak / oversold)" if last["CCI"] < -100 else "normal range",
                           "Reference only: overlaps with RSI and stochastic."))
    return items


def volume_group(x):
    last = x.iloc[-1]
    items = []
    if x["Volume"].tail(20).sum() <= 0:
        return [_info("Volume", "N/A", "no volume data for this symbol")]
    chg = float(last["Close"] / x["Close"].iloc[-2] - 1) * 100
    if _ok(last["RVOL"]):
        rv = float(last["RVOL"])
        if rv >= 1.5:
            pts = np.sign(chg) * min((rv - 1) / 1.5, 1)
            reading = f"{'buying' if chg > 0 else 'selling'} on {rv:.1f}× normal volume: move is supported"
        else:
            pts = 0.0
            reading = "normal volume: today's move is not confirmed by volume"
        items.append(_item("Volume confirmation (RVOL)", f"{rv:.1f}× · {chg:+.2f}%", reading, _clip(pts), 1.0,
                           "Today's volume ÷ average of the previous 20 bars, signed by the price move."))
    avgv = x["Volume"].tail(20).mean()
    for col, lab in (("OBV2", "On-balance volume (20 bars)"), ("AD", "Accumulation / distribution (20 bars)")):
        if len(x) > 21 and avgv:
            z = (x[col].iloc[-1] - x[col].iloc[-21]) / (avgv * 20)
            pr = x["Close"].iloc[-1] / x["Close"].iloc[-21] - 1
            div = (pr > 0 and z < -0.05) or (pr < 0 and z > 0.05)
            reading = ("accumulation (volume flowing in)" if z > 0.05 else "distribution (volume flowing out)"
                       if z < -0.05 else "flat") + ("; diverges from price" if div else "")
            items.append(_item(lab, f"{z:+.2f}", reading, _clip(z * 2), 1.0 if col == "OBV2" else 0.7,
                               "Net volume flow over 20 bars ÷ normal 20-bar volume. A/D weights each bar by where "
                               "it closed in its range."))
    if _ok(last["MFI"]):
        m = float(last["MFI"])
        items.append(_item("Money Flow Index (14)", f"{m:.0f}", "overbought (>80)" if m > 80 else "oversold (<20)"
                           if m < 20 else "buying pressure" if m > 50 else "selling pressure", _clip((m - 50) / 30), 0.7,
                           "Volume-weighted RSI."))
    if _ok(last["VWAP20"], last["ATR"]):
        z = (last["Close"] - last["VWAP20"]) / last["ATR"]
        items.append(_item("Price vs VWAP (20 bars)", f"{_f(last['VWAP20'])}",
                           f"price {'above' if z > 0 else 'below'} the volume-weighted average ({z:+.1f} ATR)",
                           _clip(z / 2), 0.5, "Rolling 20-bar VWAP: the average price weighted by volume."))
    items.append(_info("Volume SMA (20)", f"{avgv:,.0f}", "average of the last 20 bars"))
    return items


def volatility_group(x):
    last = x.iloc[-1]
    items = []
    c = float(last["Close"])
    if _ok(last["ATR"]):
        atrp = float(last["ATR"]) / c * 100
        hist = (x["ATR"] / x["Close"] * 100).dropna().tail(252)
        pct = float((hist < atrp).mean() * 100) if len(hist) > 20 else None
        items.append(_info("ATR (14)", f"{_f(last['ATR'])} ({atrp:.2f}% of price)",
                           "N/A" if pct is None else f"{pct:.0f}th percentile of the last year: " +
                           ("low" if pct < 33 else "high" if pct > 66 else "normal") + " volatility"))
    if _ok(last["HV"]):
        items.append(_info("Historical volatility (20)", f"{last['HV']:.1f}% a year", "annualised std. dev. of daily returns"))
    pts = None
    if _ok(last["BBW"]):
        w_avg = x["BBW"].tail(120).mean()
        expanding = last["BBW"] > x["BBW"].iloc[-6] * 1.1 if _ok(x["BBW"].iloc[-6]) else False
        state = "squeeze (Bollinger inside Keltner): energy building, breakout possible" if bool(last["SQZ"]) else \
            "expanding" if expanding else ("contracting" if last["BBW"] < w_avg * 0.8 else "normal")
        if c > last["BBU"] and expanding:
            pts, rd = 1.0, "expanding with price above the upper band: upside volatility breakout"
        elif c < last["BBL"] and expanding:
            pts, rd = -1.0, "expanding with price below the lower band: downside volatility breakout"
        else:
            pts, rd = 0.0, state
        items.append(_item("Volatility breakout", f"band width {last['BBW']:.1f}%", rd, pts, 1.0,
                           "Only a band breakout during expansion counts as directional; otherwise neutral."))
        pb = (c - last["BBL"]) / (last["BBU"] - last["BBL"]) if last["BBU"] > last["BBL"] else np.nan
        items.append(_info("Bollinger (20, 2)", f"%B {pb:.2f}" if pb == pb else "N/A",
                           f"{_f(last['BBL'])} – {_f(last['BBU'])}"))
        items.append(_info("Keltner (20, 2×ATR)", f"{_f(last['KCL'])} – {_f(last['KCU'])}",
                           "squeeze on" if bool(last["SQZ"]) else "no squeeze"))
    if _ok(last["ATR"]):
        a = float(last["ATR"])
        items.append(_info("ATR stop references", "", f"1.5× {c - 1.5 * a:,.3f} · 2× {c - 2 * a:,.3f} · 3× {c - 3 * a:,.3f}",
                           "Distance below the last close; wider stops suit more volatile stocks."))
    return items


def price_action_group(x, sup, res, patterns=None):
    last = x.iloc[-1]
    c = float(last["Close"])
    atr = float(last["ATR"]) if _ok(last["ATR"]) else None
    items = []
    stc = ind.structure(x)
    if stc["hh"] is not None:
        pts = 1.0 if stc["hh"] and stc["hl"] else -1.0 if not stc["hh"] and not stc["hl"] else 0.0
        items.append(_item("Market structure", f"highs {stc['prev_hi']:,.3f}→{stc['last_hi']:,.3f}, lows "
                           f"{stc['prev_lo']:,.3f}→{stc['last_lo']:,.3f}", stc["label"], pts, 1.0,
                           "Last two confirmed swing highs and lows (3 bars each side)."))
    hh, ll = last["HH20"], last["LL20"]
    if _ok(hh, ll):
        rv = float(last["RVOL"]) if _ok(last["RVOL"]) else 1.0
        if c > hh:
            items.append(_item("Breakout", f"above 20-bar high {hh:,.3f}", "breakout" +
                               (" on strong volume (confirmed)" if rv >= 1.5 else " on normal volume (unconfirmed)"),
                               1.0 if rv >= 1.5 else 0.6, 1.0))
        elif c < ll:
            items.append(_item("Breakdown", f"below 20-bar low {ll:,.3f}", "breakdown" +
                               (" on strong volume" if rv >= 1.5 else " on normal volume"), -1.0 if rv >= 1.5 else -0.6, 1.0))
        else:
            rt = _retest(x, atr)
            if rt:
                items.append(_item("Retest", rt["value"], rt["reading"], rt["points"], 1.0,
                                   "A broken level that price comes back to and respects."))
            else:
                items.append(_item("Range position", f"{ll:,.3f} – {hh:,.3f}",
                                   f"inside the 20-bar range ({(c - ll) / (hh - ll) * 100:.0f}% of the way up)" if hh > ll
                                   else "inside the 20-bar range", 0.0, 0.5))
    if sup and res and atr:
        ds, dr = (c - sup[0][0]) / atr, (res[0][0] - c) / atr
        if ds <= 1 and (stc.get("hl") or stc.get("hh") is None):
            pts, rd = 0.5, f"{ds:.1f} ATR above support {sup[0][0]:,.3f}: buyers defended it before"
        elif dr <= 1:
            pts, rd = -0.3, f"{dr:.1f} ATR below resistance {res[0][0]:,.3f}: sellers may appear"
        else:
            pts, rd = 0.0, f"{ds:.1f} ATR above support, {dr:.1f} ATR below resistance"
        items.append(_item("Support / resistance position", f"S {sup[0][0]:,.3f} · R {res[0][0]:,.3f}", rd, pts, 0.5,
                           "Levels from clustered swing points (more touches = stronger)."))
    for kind in ("support", "resistance"):
        tl = ind.trendline(x, kind)
        if tl and atr:
            above = c > tl["value"]
            if kind == "support":
                pts = (0.5 if above else -0.7) if tl["slope"] > 0 else (0.0 if above else -0.3)
                rd = f"rising support line at {tl['value']:,.3f}: price {'above' if above else 'below (broken)'}" \
                    if tl["slope"] > 0 else f"falling lows line at {tl['value']:,.3f}"
            else:
                pts = (0.7 if above else -0.3) if tl["slope"] < 0 else (0.3 if above else 0.0)
                rd = f"falling resistance line at {tl['value']:,.3f}: price {'above (broken out)' if above else 'below'}" \
                    if tl["slope"] < 0 else f"rising highs line at {tl['value']:,.3f}"
            items.append(_item(f"Trendline ({kind})", f"{tl['slope'] / c * 100:+.2f}%/bar", rd, pts, 0.5,
                               "Line through the last two confirmed swing lows (support) or highs (resistance)."))
    if patterns:
        name, kind, ago = patterns[0]
        pts = 0.6 if kind == "bull" else -0.6 if kind == "bear" else 0.0
        items.append(_item("Candlestick pattern", name, f"{ago} bars ago · {kind}ish signal, needs confirmation by the "
                           "next candle", pts, 0.5, "A single pattern is never treated as a confirmed signal."))
    else:
        items.append(_info("Candlestick pattern", "none", "no recognised pattern in the last 3 candles"))
    if len(x) >= 120:
        hi, lo = x["High"].tail(252).max(), x["Low"].tail(252).min()
        items.append(_info("52-week range", f"{lo:,.3f} – {hi:,.3f}", f"price {((c - lo) / (hi - lo) * 100) if hi > lo else 0:.0f}% "
                           "of the way from the low to the high"))
    return items


def _retest(x, atr):
    if not atr or len(x) < 35:
        return None
    c = x["Close"]
    for j in range(2, 11):
        i = len(x) - 1 - j
        lvl_hi, lvl_lo = x["HH20"].iloc[i], x["LL20"].iloc[i]
        if _ok(lvl_hi) and c.iloc[i] > lvl_hi and 0 <= c.iloc[-1] - lvl_hi <= atr:
            return {"value": f"broke {lvl_hi:,.3f} {j} bars ago", "points": 0.6,
                    "reading": "pulled back to the breakout level and is holding above it"}
        if _ok(lvl_lo) and c.iloc[i] < lvl_lo and 0 <= lvl_lo - c.iloc[-1] <= atr:
            return {"value": f"broke {lvl_lo:,.3f} {j} bars ago", "points": -0.6,
                    "reading": "bounced back up to the broken support and is failing below it"}
    return None


# ============================================================================
# Assessment
# ============================================================================
def assess(d, sup, res, mtf=None, patterns=None, weights=None):
    """d: frame from core.add_indicators (completed candles). Returns groups, scores, overall and confidence."""
    x = ind.compute_all(d)
    groups = {
        "Trend": trend_group(x, mtf),
        "Momentum": momentum_group(x),
        "Volume": volume_group(x),
        "Volatility": volatility_group(x),
        "Price action": price_action_group(x, sup, res, patterns),
    }
    w = dict(DEFAULT_WEIGHTS, **(weights or {}))
    scores = {g: group_score(it) for g, it in groups.items()}
    avail = {g: s for g, s in scores.items() if s is not None and w.get(g, 0) > 0}
    total = sum(w[g] for g in avail)
    overall = sum(avail[g] * w[g] for g in avail) / total if total else None
    possible = 4 + 6 + 5 + 1 + 6           # scored items each group could have at most
    have = sum(1 for it in groups.values() for i in it if i["points"] is not None)
    completeness = min(have / possible, 1.0)
    spread = float(np.std(list(avail.values()))) if len(avail) > 1 else 0.0
    if overall is None:
        conf = "N/A"
    elif completeness < 0.55 or spread > 20 or abs(overall - 50) < 7:
        conf = "Low"
    elif completeness >= 0.75 and spread <= 13 and abs(overall - 50) >= 15:
        conf = "High"
    else:
        conf = "Medium"
    return {"x": x, "groups": groups, "scores": scores, "weights": w, "overall": overall,
            "label": label_of(overall), "completeness": completeness, "spread": spread, "confidence": conf}


# ============================================================================
# Conclusion, horizons and opportunities
# ============================================================================
def horizons(x):
    last = x.iloc[-1]
    c = float(last["Close"])
    out = []

    def mk(name, span, checks):
        ok = [(t, v) for t, v in checks if v is not None]
        if not ok:
            return {"name": name, "span": span, "label": "N/A", "score": None, "evidence": ["not enough history"]}
        s = sum(v for _, v in ok) / len(ok)
        return {"name": name, "span": span, "score": 50 + 50 * s,
                "label": "Bullish" if s > 0.2 else "Bearish" if s < -0.2 else "Neutral",
                "evidence": [t for t, _ in ok]}

    sl20 = ind.slope(x["E20"], 10)
    out.append(mk("Short term", "≈ 2–4 weeks", [
        (f"EMA 9 {'above' if last['E9'] > last['E20'] else 'below'} EMA 20",
         (1 if last["E9"] > last["E20"] else -1) if _ok(last["E9"], last["E20"]) else None),
        (f"EMA 20 {'rising' if sl20 > 0 else 'falling'}", (1 if sl20 > 0 else -1) if _ok(sl20) else None),
        (f"12-bar change {last['ROC']:+.1f}%", _clip(last["ROC"] / 5) if _ok(last["ROC"]) else None),
        (f"RSI {last['RSI']:.0f}", _clip((last["RSI"] - 50) / 20) if _ok(last["RSI"]) else None),
    ]))
    sl50 = ind.slope(x["E50"], 20)
    r60 = (c / x["Close"].iloc[-61] - 1) * 100 if len(x) > 61 else None
    out.append(mk("Medium term", "≈ 1–3 months", [
        (f"price {'above' if c > last['E50'] else 'below'} EMA 50", (1 if c > last["E50"] else -1) if _ok(last["E50"]) else None),
        (f"EMA 50 {'rising' if sl50 > 0 else 'falling'}", (1 if sl50 > 0 else -1) if _ok(sl50) else None),
        (f"60-bar change {r60:+.1f}%" if r60 is not None else "60-bar change N/A", _clip(r60 / 15) if r60 is not None else None),
        (f"MACD {'above' if last['MACD'] > last['MACDsig'] else 'below'} signal",
         (1 if last["MACD"] > last["MACDsig"] else -1) if _ok(last["MACD"], last["MACDsig"]) else None),
    ]))
    sl200 = ind.slope(x["E200"], 50)
    hi, lo = (x["High"].tail(252).max(), x["Low"].tail(252).min()) if len(x) >= 200 else (None, None)
    out.append(mk("Long term", "≈ 6–12 months", [
        (f"price {'above' if c > last['E200'] else 'below'} EMA 200", (1 if c > last["E200"] else -1) if _ok(last["E200"]) else None),
        (f"EMA 200 {'rising' if sl200 > 0 else 'falling'}", (1 if sl200 > 0 else -1) if _ok(sl200) else None),
        (f"EMA 50 {'above' if last['E50'] > last['E200'] else 'below'} EMA 200",
         (1 if last["E50"] > last["E200"] else -1) if _ok(last["E50"], last["E200"]) else None),
        (f"{(c - lo) / (hi - lo) * 100:.0f}% of the 52-week range" if hi and hi > lo else "52-week range N/A",
         _clip(((c - lo) / (hi - lo) - 0.5) * 2) if hi and hi > lo else None),
    ]))
    return out


def opportunities(x, sup, res, tech):
    last = x.iloc[-1]
    c = float(last["Close"])
    atr = float(last["ATR"]) if _ok(last["ATR"]) else None
    tags = []
    if res and atr and 0 <= (res[0][0] - c) / c * 100 <= 3 and (tech["scores"].get("Trend") or 0) >= 50:
        sq = bool(last["SQZ"]) if _ok(last["SQZ"]) else False
        tags.append(("🚀 Breakout candidate", f"within 3% of resistance {res[0][0]:,.3f} in a non-bearish trend"
                     + (" with a volatility squeeze" if sq else "") + ". Watch for a close above it on ≥ 1.5× volume."))
    if _ok(last["E200"], last["E50"], last["E20"]) and c > last["E200"] and last["E50"] > last["E200"] and atr:
        near = min(abs(c - last["E20"]), abs(c - last["E50"])) <= atr
        if near and _ok(last["RSI"]) and 35 <= last["RSI"] <= 55:
            tags.append(("📉 Pullback opportunity", "long-term uptrend, price back near its 20/50-day average with "
                         "cooled-off RSI. Wait for the pullback to stop (a higher low or bullish candle)."))
    if _ok(last["ADX"], last["PDI"], last["MDI"], last["E20"]) and last["ADX"] >= 25 and last["PDI"] > last["MDI"] \
            and last["MACD"] > last["MACDsig"] and c > last["E20"]:
        tags.append(("⚡ Momentum continuation", "strong uptrend (ADX ≥ 25, +DI on top) with MACD above signal and "
                     "price above EMA 20. Trends tend to persist until these flip."))
    if _ok(last["ADX"], last["PDI"], last["MDI"], last["E20"]) and last["ADX"] >= 25 and last["MDI"] > last["PDI"] \
            and last["MACD"] < last["MACDsig"] and c < last["E20"]:
        tags.append(("🔻 Downtrend continuation", "strong downtrend (ADX ≥ 25, −DI on top) with MACD below signal "
                     "and price below EMA 20."))
    warn = []
    for g in ("Momentum",):
        for i in tech["groups"][g]:
            if i["name"] == "Divergence" and i["points"] is not None:
                warn.append(f"{i['value']} divergence")
    d50 = _cross(x["E50"], x["E200"], 30)
    if d50[0]:
        warn.append(("golden" if d50[0] > 0 else "death") + f" cross {d50[1]} bars ago")
    stc = ind.structure(x)
    if stc["hh"] is not None and stc["hh"] != stc["hl"]:
        warn.append("swing structure is changing")
    if warn:
        tags.append(("🔄 Possible trend reversal", "; ".join(warn) + ". Reversal signals need confirmation by price "
                     "(a break of the last swing high/low)."))
    if sup and atr and 0 <= (c - sup[0][0]) / c * 100 <= 3 and (tech["scores"].get("Momentum") or 50) < 45:
        tags.append(("⚠️ Breakdown risk", f"sitting just above support {sup[0][0]:,.3f} with weak momentum. A close "
                     "below it opens the next support."))
    return tags


def projections(x, lookahead=10):
    """Where the trendlines will be in `lookahead` bars, and Fibonacci targets."""
    out = []
    for kind in ("support", "resistance"):
        tl = ind.trendline(x, kind)
        if tl:
            out.append((f"{kind.title()} trendline in {lookahead} bars", tl["value"] + tl["slope"] * lookahead))
    fib = ind.fibonacci(x)
    return out, fib


def conclusion(x, sup, res, plan, tech, hz):
    last = x.iloc[-1]
    c = float(last["Close"])
    atr = float(last["ATR"]) if _ok(last["ATR"]) else 0
    g = tech["groups"]

    def first(group, name):
        return next((i for i in g[group] if i["name"].startswith(name)), None)
    ma = first("Trend", "Moving-average")
    adx = first("Trend", "ADX")
    rsi, macd = first("Momentum", "RSI"), first("Momentum", "MACD")
    vc = first("Volume", "Volume confirmation")
    vb = first("Volatility", "Volatility breakout")
    atr_i = first("Volatility", "ATR (14)")
    s1, s2, r1, r2 = plan["s1"], plan["s2"], plan["t1"], plan["t2"]
    fib = ind.fibonacci(x)
    ext = []
    if fib and fib["direction"] == "up":
        ext = [(k, v) for k, v in fib["extensions"].items() if v > c][:2]
    entry_pb = f"{plan['buy_lo']:,.3f} – {plan['buy_hi']:,.3f} (support zone)"
    entry_bo = f"daily close above {r1:,.3f} on ≥ 1.5× volume (breakout entry)"
    stop = plan["stop"]

    def rr(target, entry):
        return (target - entry) / (entry - stop) if entry > stop else None
    tps = [("T1 (first resistance)", r1), ("T2 (second resistance)", r2)] + \
          [(f"Fibonacci extension {k}", v) for k, v in ext]
    tp_rows = [(lab, v, rr(v, plan["entry"])) for lab, v in tps if v > plan["entry"]]
    return {
        "Trend direction": f"{tech['label']} technical picture ({tech['overall']:.0f}/100). "
                           f"Short term {hz[0]['label'].lower()}, medium term {hz[1]['label'].lower()}, "
                           f"long term {hz[2]['label'].lower()}." if tech["overall"] is not None else "N/A",
        "EMA / SMA alignment": f"{ma['reading']} ({ma['value']})." if ma else "N/A",
        "Trend strength": f"{adx['reading']} ({adx['value']})." if adx else "N/A",
        "Momentum": "; ".join(i for i in (rsi and f"RSI {rsi['value']} {rsi['reading']}",
                                         macd and f"MACD {macd['reading']}") if i) or "N/A",
        "Volume confirmation": vc["reading"] if vc else "N/A",
        "Volatility": "; ".join(i for i in (vb and vb["reading"], atr_i and atr_i["reading"]) if i) or "N/A",
        "Support / resistance": f"Support {s1:,.3f} then {s2:,.3f}; resistance {r1:,.3f} then {r2:,.3f}.",
        "Bullish scenario": f"Price holds above {s1:,.3f} and closes above {r1:,.3f} on strong volume → room "
                            f"towards {r2:,.3f}" + (f" and the Fibonacci {ext[0][0]} extension {ext[0][1]:,.3f}" if ext else "")
                            + ".",
        "Bearish scenario": f"A close below {s1:,.3f} breaks support → next support {s2:,.3f}"
                            + (f"; losing EMA 50 ({last['E50']:,.3f}) would also weaken the medium-term trend"
                               if _ok(last["E50"]) and c > last["E50"] else "") + ".",
        "Entry zones": f"Pullback: {entry_pb}. Breakout: {entry_bo}.",
        "Stop-loss reference": f"{stop:,.3f} (support − {plan['atr_mult']:.1f}× ATR). A 2× ATR trailing stop would sit "
                               f"at {c - 2 * atr:,.3f}.",
        "Take-profit scenarios": tp_rows,
        "Invalidation": f"The bullish case is invalid on a daily close below {s1:,.3f} (or below the stop {stop:,.3f}). "
                        f"The bearish case is invalid on a close above {r1:,.3f}. If ADX falls below 20 the trend "
                        "signals lose weight (market turning sideways).",
    }
