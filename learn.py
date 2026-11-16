"""Learn tab: tips & tricks, trading glossary, candlestick patterns (with diagrams),
plus a simple candlestick pattern detector used by the Analysis tab."""
import math

import pandas as pd
import streamlit as st

BULL, BEAR, NEUT = "#3fb68b", "#e0646a", "#c9ced8"
PAL = dict(card="#171d29", card2="#1c2331", border="#262f40", text="#dfe3ea", muted="#8890a2", accent="#7b93f5")


def set_palette(p):
    """Use the app's colour palette for the Learn tab."""
    global BULL, BEAR
    BULL, BEAR = p.get("up", BULL), p.get("down", BEAR)
    PAL.update({k: p[k] for k in PAL if k in p})

# ----------------------------------------------------------------------------
# Candlestick diagrams (inline SVG)
# ----------------------------------------------------------------------------

def _context(first_open, kind):
    """Three faded candles showing the trend before the pattern."""
    out = []
    for i in range(3):
        if kind == "down":
            o, c = first_open + 24 - 8 * i, first_open + 17 - 8 * i
        else:
            o, c = first_open - 24 + 8 * i, first_open - 17 + 8 * i
        out.append((o, max(o, c) + 2, min(o, c) - 2, c))
    return out


def candle_svg(candles, context=None, width=300, height=170):
    ctx = _context(candles[0][0], context) if context else []
    allc = ctx + list(candles)
    lo = min(c[2] for c in allc)
    hi = max(c[1] for c in allc)
    pad = 12
    span = (hi - lo) or 1

    def y(v):
        return pad + (hi - v) / span * (height - 2 * pad)

    n = len(allc)
    step = min(46, (width - 30) / n)
    bw = step * 0.6
    x0 = (width - step * n) / 2 + step / 2
    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" xmlns="http://www.w3.org/2000/svg">',
             f'<rect width="{width}" height="{height}" rx="10" fill="{PAL["card2"]}"/>']
    for i, (o, h, l, c) in enumerate(allc):
        faded = i < len(ctx)
        col = NEUT if abs(c - o) < 0.6 else (BULL if c > o else BEAR)
        op = 0.35 if faded else 1
        cx = x0 + i * step
        top, bot = y(max(o, c)), y(min(o, c))
        parts.append(f'<line x1="{cx:.1f}" y1="{y(h):.1f}" x2="{cx:.1f}" y2="{y(l):.1f}" '
                     f'stroke="{col}" stroke-width="2.5" opacity="{op}"/>')
        bh = max(bot - top, 2)
        parts.append(f'<rect x="{cx - bw / 2:.1f}" y="{top:.1f}" width="{bw:.1f}" height="{bh:.1f}" '
                     f'fill="{col}" opacity="{op}" rx="1"/>')
    parts.append("</svg>")
    return "".join(parts)


def anatomy_svg():
    svg = """
<svg viewBox="0 0 520 246" width="100%" height="246" xmlns="http://www.w3.org/2000/svg" font-family="sans-serif" font-size="13">
 <rect width="520" height="246" rx="10" fill="#161b26"/>
 <!-- bullish -->
 <line x1="110" y1="20" x2="110" y2="210" stroke="#26a69a" stroke-width="3"/>
 <rect x="85" y="60" width="50" height="110" fill="#26a69a" rx="2"/>
 <text x="110" y="226" fill="#26a69a" text-anchor="middle" font-weight="bold">Bullish (green)</text>
 <!-- bearish -->
 <line x1="400" y1="20" x2="400" y2="210" stroke="#ef5350" stroke-width="3"/>
 <rect x="375" y="60" width="50" height="110" fill="#ef5350" rx="2"/>
 <text x="400" y="226" fill="#ef5350" text-anchor="middle" font-weight="bold">Bearish (red)</text>
 <!-- labels -->
 <g fill="#d1d4dc">
  <text x="140" y="24">← High</text>
  <text x="140" y="64">← Close</text>
  <text x="140" y="174">← Open</text>
  <text x="140" y="212">← Low</text>
  <text x="370" y="24" text-anchor="end">High →</text>
  <text x="370" y="64" text-anchor="end">Open →</text>
  <text x="370" y="174" text-anchor="end">Close →</text>
  <text x="370" y="212" text-anchor="end">Low →</text>
 </g>
 <g fill="#8a8f9c" font-size="12">
  <text x="255" y="44" text-anchor="middle">upper wick (shadow)</text>
  <text x="255" y="118" text-anchor="middle">BODY</text>
  <text x="255" y="134" text-anchor="middle">open ↔ close</text>
  <text x="255" y="196" text-anchor="middle">lower wick (shadow)</text>
 </g>
</svg>"""
    return (svg.replace("#161b26", PAL["card2"]).replace("#26a69a", BULL).replace("#ef5350", BEAR)
               .replace("#d1d4dc", PAL["text"]).replace("#8a8f9c", PAL["muted"]))


# name, kind, context, candles (o,h,l,c), what it looks like, what it means, how to use it
PATTERNS = [
    # ---- single candle ----
    ("Doji", "neutral", None, [(50, 70, 30, 50)],
     "Open and close are almost the same, so the body is a thin line.",
     "Buyers and sellers are evenly matched: indecision. After a strong trend it can warn the trend is tiring.",
     "Never trade a doji alone. Wait for the next candle to show which side won."),
    ("Dragonfly Doji", "bull", "down", [(60, 61, 25, 60)],
     "A doji with a long lower wick and no upper wick (looks like a T).",
     "Sellers pushed price down hard, but buyers pushed it all the way back. Bullish at the bottom of a downtrend.",
     "Stronger if it forms on a support level. Confirm with a green candle next."),
    ("Gravestone Doji", "bear", "up", [(40, 75, 39, 40)],
     "A doji with a long upper wick and no lower wick (upside-down T).",
     "Buyers pushed price up but got rejected all the way back. Bearish at the top of an uptrend.",
     "Stronger at a resistance level. Confirm with a red candle next."),
    ("Hammer", "bull", "down", [(52, 59, 25, 58)],
     "Small body at the top, long lower wick (at least 2× the body), little or no upper wick. Appears after a fall.",
     "Sellers were in control, but buyers stepped in and closed near the high. Possible bottom.",
     "Buy idea: enter above the hammer's high, stop loss below its low."),
    ("Inverted Hammer", "bull", "down", [(42, 75, 41, 48)],
     "Small body at the bottom, long upper wick. Appears after a fall.",
     "Buyers are starting to test higher prices. Possible bottom, but weaker than a hammer.",
     "Needs a strong green candle next to confirm."),
    ("Hanging Man", "bear", "up", [(58, 59, 28, 52)],
     "Same shape as a hammer, but it appears after a rise.",
     "Sellers managed a big push down during the session: a warning that the uptrend may be weakening.",
     "If you hold the stock, tighten your stop. Confirmed if the next candle closes lower."),
    ("Shooting Star", "bear", "up", [(48, 78, 41, 42)],
     "Small body at the bottom, long upper wick. Appears after a rise.",
     "Buyers pushed up, then sellers slammed it back down. Common at tops and resistance.",
     "Sell / take-profit idea, especially at resistance. Confirm with a red candle next."),
    ("Spinning Top", "neutral", None, [(48, 70, 32, 54)],
     "Small body in the middle, wicks on both sides.",
     "Indecision. Neither side is in control.",
     "Treat it as a 'pause'. Watch what the next candles do."),
    ("Bullish Marubozu", "bull", None, [(30, 75, 30, 75)],
     "Big green body with no (or tiny) wicks.",
     "Buyers controlled the whole session from open to close. Strong bullish momentum.",
     "Often starts or continues an uptrend. Pullbacks to its midpoint can be entries."),
    ("Bearish Marubozu", "bear", None, [(75, 75, 30, 30)],
     "Big red body with no (or tiny) wicks.",
     "Sellers controlled the whole session. Strong bearish momentum.",
     "Avoid buying right after one. Wait for selling to calm down."),
    # ---- two candles ----
    ("Bullish Engulfing", "bull", "down", [(50, 52, 40, 42), (38, 62, 37, 60)],
     "A small red candle followed by a bigger green candle whose body completely covers the red body.",
     "Buyers overpowered sellers completely. One of the most reliable reversal signals at a bottom.",
     "Best at support with higher-than-normal volume. Stop below the green candle's low."),
    ("Bearish Engulfing", "bear", "up", [(50, 60, 48, 58), (62, 63, 38, 40)],
     "A small green candle followed by a bigger red candle that covers it.",
     "Sellers overpowered buyers. Reliable reversal signal at a top.",
     "Take profit / avoid buying, especially at resistance."),
    ("Bullish Harami", "bull", "down", [(70, 72, 30, 32), (42, 55, 40, 52)],
     "A big red candle followed by a small green candle that sits inside the red body ('pregnant' in Japanese).",
     "The selling is losing strength. Early hint of a reversal, weaker than engulfing.",
     "Wait for a close above the big red candle's open for confirmation."),
    ("Bearish Harami", "bear", "up", [(30, 72, 28, 70), (60, 62, 45, 48)],
     "A big green candle followed by a small red candle inside it.",
     "The buying is losing strength. Early warning at a top.",
     "Tighten stops; confirmed if price closes below the big green candle's open."),
    ("Piercing Line", "bull", "down", [(70, 71, 40, 42), (36, 62, 35, 60)],
     "A red candle, then a green candle that opens lower but closes above the middle of the red body.",
     "Strong buying came in after a weak open. Bullish reversal.",
     "Stronger the deeper the green candle closes into the red one."),
    ("Dark Cloud Cover", "bear", "up", [(40, 70, 39, 68), (74, 75, 50, 52)],
     "A green candle, then a red candle that opens higher but closes below the middle of the green body.",
     "Selling came in after a strong open. Bearish reversal (opposite of piercing line).",
     "Consider taking profit; confirmed by another red candle."),
    ("Tweezer Bottom", "bull", "down", [(55, 56, 30, 35), (35, 52, 30, 50)],
     "Two candles with (almost) the same low: a red one then a green one.",
     "Price tested the same low twice and failed to break it: buyers defend that level.",
     "That low becomes support. Stop loss just below it."),
    ("Tweezer Top", "bear", "up", [(45, 70, 44, 65), (65, 70, 48, 50)],
     "Two candles with (almost) the same high: a green one then a red one.",
     "Price failed twice at the same high: sellers defend that level.",
     "That high becomes resistance. Good take-profit area."),
    # ---- three candles ----
    ("Morning Star", "bull", "down", [(75, 76, 42, 45), (40, 44, 33, 38), (42, 70, 41, 68)],
     "Big red candle → small candle (the 'star') → big green candle closing above the middle of the first.",
     "Selling, then indecision, then strong buying. Classic bottom reversal.",
     "One of the stronger buy signals, especially at support with rising volume."),
    ("Evening Star", "bear", "up", [(35, 68, 34, 65), (70, 77, 67, 72), (68, 69, 38, 40)],
     "Big green candle → small candle → big red candle closing below the middle of the first.",
     "Buying, then indecision, then strong selling. Classic top reversal.",
     "Strong sell / take-profit signal, especially at resistance."),
    ("Three White Soldiers", "bull", "down", [(30, 46, 29, 44), (40, 60, 39, 58), (54, 74, 53, 72)],
     "Three strong green candles in a row, each closing higher.",
     "Steady, determined buying. Often marks the start of an uptrend.",
     "Don't chase after the third candle; wait for a small pullback."),
    ("Three Black Crows", "bear", "up", [(72, 73, 56, 58), (60, 61, 42, 44), (46, 47, 28, 30)],
     "Three strong red candles in a row, each closing lower.",
     "Steady, determined selling. Often marks the start of a downtrend.",
     "Avoid buying until a reversal pattern appears."),
]

def _badge(kind):
    return {"bull": ("Bullish", BULL), "bear": ("Bearish", BEAR), "neutral": ("Indecision", PAL["muted"])}[kind]

# ----------------------------------------------------------------------------
# Glossary
# ----------------------------------------------------------------------------
GLOSSARY = [
    # Basics
    ("Stock / Share", "Basics", "A small piece of ownership in a company. If the company does well, the share price usually rises."),
    ("Ticker", "Basics", "The short code for a stock. US: letters (AAPL = Apple). Bursa: a 4-digit code or name (1155 / MAYBANK)."),
    ("Bid / Ask", "Basics", "Bid = highest price a buyer is willing to pay right now. Ask (offer) = lowest price a seller will accept."),
    ("Spread", "Basics", "The gap between bid and ask. Small spread = easy to trade. Big spread = you lose a bit just by buying and selling."),
    ("Volume", "Basics", "How many shares changed hands in a period. High volume means many people care about the move, so it's more trustworthy."),
    ("Liquidity", "Basics", "How easily you can buy/sell without moving the price. Big caps are liquid; small penny stocks often aren't."),
    ("Market cap", "Basics", "Share price × number of shares = the total value of the company."),
    ("Blue chip", "Basics", "Large, established, financially strong companies (e.g. Maybank, Tenaga, Apple)."),
    ("Penny stock", "Basics", "Very low-priced shares (often below RM1 / US$5). Cheap, but very volatile and easy to manipulate."),
    ("ETF", "Basics", "Exchange-Traded Fund: one ticker that holds a basket of stocks (e.g. SPY = 500 US companies). Good for beginners to diversify."),
    ("Lot / Board lot", "Basics", "On Bursa Malaysia, 1 lot = 100 shares. You buy in multiples of 100. US stocks can be bought 1 share at a time (or fractions on some brokers)."),
    ("Bull market / Bear market", "Basics", "Bull = prices rising over a long period. Bear = prices falling (commonly 20%+ from the top)."),
    ("Bullish / Bearish", "Basics", "Bullish = expecting price to go up. Bearish = expecting price to go down."),
    ("Market hours", "Basics", "Bursa: 9:00am–12:30pm and 2:30pm–5:00pm MYT. US: 9:30am–4:00pm New York time, which is about 10:30pm–5:00am MYT in winter and 9:30pm–4:00am MYT in US summer."),
    ("T+2 / T+1 settlement", "Basics", "Days after the trade until shares and cash officially change hands. Bursa uses T+2; the US moved to T+1 in 2024."),
    ("Contra (Bursa)", "Basics", "Buying and selling the same shares within the settlement period without paying the full amount. Risky: losses must be paid quickly."),
    # Orders
    ("Market order", "Orders", "Buy/sell immediately at the best available price. Fast, but price may be worse than you expected (slippage)."),
    ("Limit order", "Orders", "Buy/sell only at your price or better. You control the price, but it may not get filled."),
    ("Stop loss", "Orders", "An order that sells automatically if price falls to your chosen level, to cap your loss. The #1 tool for survival."),
    ("Take profit", "Orders", "A target price where you sell to lock in gains."),
    ("Trailing stop", "Orders", "A stop loss that moves up as price rises (e.g. always 5% below the highest price), locking in profit."),
    ("Slippage", "Orders", "The difference between the price you expected and the price you actually got."),
    ("Long / Short", "Orders", "Long = buy, profit if price rises. Short = sell borrowed shares, profit if price falls (advanced and riskier)."),
    # Chart & technical analysis
    ("Candlestick", "Technical analysis", "A bar showing the open, high, low and close for one period. Green = closed higher than it opened, red = closed lower."),
    ("Timeframe", "Technical analysis", "How much time each candle represents (15 min, 1 hour, daily, weekly). Higher timeframes give more reliable signals."),
    ("Trend", "Technical analysis", "The general direction. Uptrend = higher highs and higher lows. Downtrend = lower highs and lower lows."),
    ("Support", "Technical analysis", "A price level where buyers have stepped in before, so price tends to stop falling there. Think of it as a floor."),
    ("Resistance", "Technical analysis", "A price level where sellers have appeared before, so price tends to stop rising there. Think of it as a ceiling."),
    ("Breakout", "Technical analysis", "Price closes above resistance. Often leads to a bigger move up, especially with high volume."),
    ("Breakdown", "Technical analysis", "Price closes below support. Often leads to further falls."),
    ("Pullback", "Technical analysis", "A short dip within an uptrend. Often a better place to buy than chasing the top."),
    ("Fakeout", "Technical analysis", "A breakout that quickly reverses back. Waiting for a candle to close beyond the level helps avoid them."),
    ("Gap", "Technical analysis", "When a candle opens far above/below the previous close, leaving an empty space on the chart. Usually caused by news."),
    ("Moving average (SMA / EMA)", "Technical analysis", "The average price over N periods, drawn as a line. EMA reacts faster to recent prices than SMA. Price above the line = uptrend."),
    ("EMA 20 / 50 / 200", "Technical analysis", "Common settings: EMA20 = short-term trend, EMA50 = medium-term, EMA200 = long-term. Many traders only buy when price is above EMA200."),
    ("Golden cross / Death cross", "Technical analysis", "Golden = EMA/SMA 50 crosses above 200 (bullish). Death = 50 crosses below 200 (bearish)."),
    ("RSI", "Technical analysis", "Relative Strength Index, 0–100. Above 70 = overbought (may pull back). Below 30 = oversold (may bounce). Above 50 = buyers have momentum."),
    ("Overbought / Oversold", "Technical analysis", "Price has moved up (or down) too far, too fast. Not an instant sell/buy signal; strong trends can stay overbought for a long time."),
    ("MACD", "Technical analysis", "Momentum indicator. When the MACD line crosses above its signal line, momentum is turning up; crossing below, turning down. Histogram bars show the gap between them."),
    ("ATR", "Technical analysis", "Average True Range: how much a stock typically moves per candle. Used to place stop losses (e.g. 1× ATR below support)."),
    ("Volatility", "Technical analysis", "How wildly price swings. High volatility = bigger profits and bigger losses."),
    ("Swing high / Swing low", "Technical analysis", "A swing high is a peak with lower highs on both sides; a swing low is a dip with higher lows on both sides. Old swing highs often act as resistance and old swing lows as support."),
    ("Repeated touches", "Technical analysis", "A price level that price has bounced off (or been rejected at) several times. The more touches, the more traders watch that level, so the stronger it tends to be."),
    ("Volume Profile", "Technical analysis", "A sideways histogram showing how much volume traded at each price. Prices with lots of volume act like magnets and as support/resistance; thin areas are where price can move fast."),
    ("POC (Point of Control)", "Technical analysis", "The single price with the most volume traded in the Volume Profile. A very strong support/resistance level."),
    ("Value Area (VAH / VAL)", "Technical analysis", "The price range where about 70% of volume traded. VAH = top of that range, VAL = bottom. Moving out of the value area often starts a new trend."),
    ("Fibonacci retracement", "Technical analysis", "Levels at 23.6%, 38.2%, 50%, 61.8% and 78.6% of a big move. After a rise, pullbacks often pause at 38.2%-61.8% before continuing. 61.8% is called the 'golden ratio'."),
    ("Shariah-compliant", "Basics", "A stock that the Securities Commission Malaysia's Shariah Advisory Council has approved under Islamic principles (business activities and debt/cash levels within limits). The list is updated every May and November."),
    ("Divergence", "Technical analysis", "When price makes a new high but RSI/MACD doesn't (or vice versa). A warning that the trend may be weakening."),
    # Risk
    ("Risk : Reward (R:R)", "Risk management", "How much you could gain vs how much you could lose. 1:2 means risking RM100 to make RM200. Aim for at least 1:2."),
    ("Position sizing", "Risk management", "Deciding how many shares to buy so that if your stop loss is hit, you only lose a small % of your capital (often 1–2%)."),
    ("Diversification", "Risk management", "Spreading money across different stocks/sectors so one bad stock doesn't wreck your account."),
    ("Drawdown", "Risk management", "How far your account has fallen from its highest point. A 50% drawdown needs a 100% gain just to recover."),
    ("Averaging down", "Risk management", "Buying more of a losing stock to lower your average price. Dangerous for beginners: it can turn a small loss into a big one."),
    ("DCA (Dollar-cost averaging)", "Risk management", "Investing a fixed amount regularly (e.g. every month) regardless of price. Simple and low-stress for long-term investing."),
    ("FOMO", "Risk management", "Fear Of Missing Out: buying because price is flying and everyone is talking about it. Usually leads to buying the top."),
    ("Paper trading", "Risk management", "Practising with fake money. moomoo and TradingView both have paper trading: use it before risking real money."),
    # Fundamentals
    ("EPS", "Fundamentals", "Earnings Per Share = company profit ÷ number of shares."),
    ("P/E ratio", "Fundamentals", "Price ÷ EPS. How many years of current profit you're paying for. Compare it with similar companies, not across sectors."),
    ("Dividend", "Fundamentals", "Cash a company pays to shareholders from its profits."),
    ("Dividend yield", "Fundamentals", "Yearly dividend ÷ share price. A 5% yield on RM10 = RM0.50 per share per year."),
    ("Ex-dividend date", "Fundamentals", "You must own the stock before this date to receive the next dividend. Price often drops by about the dividend amount on this day."),
    ("Earnings / Quarterly results", "Fundamentals", "When a company reports its profits (every 3 months). Prices can gap up or down sharply; many traders avoid holding through results."),
]

# ----------------------------------------------------------------------------
# Pattern detection (for the Analysis tab)
# ----------------------------------------------------------------------------

def detect_patterns(d: pd.DataFrame, lookback: int = 5):
    """Return [(name, kind, bars_ago)] for patterns ending in the last `lookback` candles."""
    if len(d) < lookback + 10:
        return []
    o, h, l, c = (d[k].values for k in ("Open", "High", "Low", "Close"))
    found = []
    n = len(d)
    for end in range(n - lookback, n):
        ago = n - 1 - end

        def body(i): return abs(c[i] - o[i])
        def rng(i): return max(h[i] - l[i], 1e-9)
        def upper(i): return h[i] - max(o[i], c[i])
        def lower(i): return min(o[i], c[i]) - l[i]
        def bull(i): return c[i] > o[i]
        def bear(i): return c[i] < o[i]

        i = end
        prior_down = c[i - 1] < c[i - 6]
        prior_up = c[i - 1] > c[i - 6]
        avg_rng = float(pd.Series(h[i - 10:i] - l[i - 10:i]).mean())

        # --- three-candle ---
        a, b = i - 2, i - 1
        if bear(a) and body(a) >= 0.5 * rng(a) and body(b) <= 0.35 * body(a) and bull(i) \
                and c[i] > (o[a] + c[a]) / 2 and prior_down:
            found.append(("Morning Star", "bull", ago)); continue
        if bull(a) and body(a) >= 0.5 * rng(a) and body(b) <= 0.35 * body(a) and bear(i) \
                and c[i] < (o[a] + c[a]) / 2 and prior_up:
            found.append(("Evening Star", "bear", ago)); continue
        if all(bull(k) and body(k) >= 0.5 * rng(k) for k in (a, b, i)) and c[a] < c[b] < c[i]:
            found.append(("Three White Soldiers", "bull", ago)); continue
        if all(bear(k) and body(k) >= 0.5 * rng(k) for k in (a, b, i)) and c[a] > c[b] > c[i]:
            found.append(("Three Black Crows", "bear", ago)); continue

        # --- two-candle ---
        p = i - 1
        strong = body(i) >= 0.5 * rng(i)
        if bear(p) and bull(i) and strong and o[i] <= c[p] and c[i] >= o[p] and body(i) > body(p):
            found.append(("Bullish Engulfing", "bull", ago)); continue
        if bull(p) and bear(i) and strong and o[i] >= c[p] and c[i] <= o[p] and body(i) > body(p):
            found.append(("Bearish Engulfing", "bear", ago)); continue
        big_prev = body(p) >= 0.6 * rng(p) and rng(p) >= avg_rng
        inside = max(o[i], c[i]) < max(o[p], c[p]) and min(o[i], c[i]) > min(o[p], c[p]) and body(i) <= 0.5 * body(p)
        if big_prev and inside and bear(p) and bull(i) and prior_down:
            found.append(("Bullish Harami", "bull", ago)); continue
        if big_prev and inside and bull(p) and bear(i) and prior_up:
            found.append(("Bearish Harami", "bear", ago)); continue
        if bear(p) and bull(i) and abs(l[p] - l[i]) <= 0.05 * avg_rng and prior_down and body(p) >= 0.3 * rng(p):
            found.append(("Tweezer Bottom", "bull", ago)); continue
        if bull(p) and bear(i) and abs(h[p] - h[i]) <= 0.05 * avg_rng and prior_up and body(p) >= 0.3 * rng(p):
            found.append(("Tweezer Top", "bear", ago)); continue
        if bear(p) and bull(i) and o[i] < c[p] and c[i] > (o[p] + c[p]) / 2 and c[i] < o[p] and prior_down:
            found.append(("Piercing Line", "bull", ago)); continue
        if bull(p) and bear(i) and o[i] > c[p] and c[i] < (o[p] + c[p]) / 2 and c[i] > o[p] and prior_up:
            found.append(("Dark Cloud Cover", "bear", ago)); continue

        # --- single candle ---
        small = max(body(i), 0.1 * rng(i))
        if rng(i) < 0.3 * avg_rng:
            continue  # too tiny to mean anything
        if body(i) <= 0.1 * rng(i):
            if lower(i) >= 0.6 * rng(i):
                found.append(("Dragonfly Doji", "bull" if prior_down else "neutral", ago))
            elif upper(i) >= 0.6 * rng(i):
                found.append(("Gravestone Doji", "bear" if prior_up else "neutral", ago))
            else:
                found.append(("Doji", "neutral", ago))
            continue
        if lower(i) >= 2 * small and upper(i) <= 0.15 * rng(i):
            if prior_down:
                found.append(("Hammer", "bull", ago))
            elif prior_up:
                found.append(("Hanging Man", "bear", ago))
            continue
        if upper(i) >= 2 * small and lower(i) <= 0.15 * rng(i):
            if prior_up:
                found.append(("Shooting Star", "bear", ago))
            elif prior_down:
                found.append(("Inverted Hammer", "bull", ago))
            continue
        if body(i) >= 0.9 * rng(i) and rng(i) >= 1.3 * avg_rng:
            found.append(("Bullish Marubozu" if bull(i) else "Bearish Marubozu", "bull" if bull(i) else "bear", ago))
    found.sort(key=lambda x: x[2])
    return found


# ----------------------------------------------------------------------------
# Render
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Learn tab: look & feel
# ----------------------------------------------------------------------------
CAT_STYLE = {  # category -> (icon, colour key)
    "Basics": ("📘", "accent"),
    "Orders": ("🧾", "warn"),
    "Technical analysis": ("📈", "teal"),
    "Risk management": ("🛡️", "pink"),
    "Fundamentals": ("🏢", "up"),
}


def _col(key):
    extra = {"teal": "#5ec8d8", "pink": "#d68ad6", "warn": "#e3a34b", "up": BULL, "down": BEAR}
    return extra.get(key) or PAL.get(key, PAL["accent"])


def _css():
    p = PAL
    return f"""
<style>
.lhero{{background:linear-gradient(135deg,{p['card']} 0%,{p['card2']} 100%);border:1px solid {p['border']};
  border-radius:18px;padding:22px 24px;margin:4px 0 18px}}
.lhero h2{{margin:0 0 4px;font-size:22px}}
.lhero .sub{{color:{p['muted']};font-size:14px;line-height:1.6}}
.path{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin-top:16px}}
.path div{{background:{p['card']};border:1px solid {p['border']};border-radius:12px;padding:12px 14px;font-size:13px;
  color:{p['muted']};line-height:1.45}}
.path b{{display:block;color:{p['text']};font-size:14px;margin-bottom:2px}}
.path .n{{display:inline-flex;width:22px;height:22px;border-radius:50%;align-items:center;justify-content:center;
  background:{p['accent']};color:#fff;font-size:12px;font-weight:700;margin-right:6px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(310px,1fr));gap:14px;margin:6px 0 22px}}
.tcard{{background:{p['card']};border:1px solid {p['border']};border-radius:16px;padding:16px 16px 14px;
  display:flex;flex-direction:column;gap:8px}}
.thead{{display:flex;align-items:center;gap:12px}}
.abbr{{min-width:46px;height:46px;padding:0 8px;border-radius:12px;display:flex;align-items:center;justify-content:center;
  font-weight:800;font-size:15px;letter-spacing:.02em}}
.tname{{font-weight:650;font-size:15.5px;line-height:1.3}}
.tcat{{font-size:11.5px;color:{p['muted']};margin-top:2px}}
.tdesc{{font-size:13.8px;line-height:1.6;color:{p['text']};opacity:.86}}
.ex{{background:{p['card2']};border:1px dashed {p['border']};border-radius:10px;padding:8px 11px;font-size:13px;
  line-height:1.5;color:{p['text']}}}
.ex b{{color:{p['accent']}}}
.viz{{border-radius:10px;overflow:hidden;line-height:0}}
.viz svg{{width:100%;height:auto;display:block}}
.sech{{display:flex;align-items:center;gap:10px;margin:18px 0 4px}}
.sech span{{font-size:18px;font-weight:650}}
.sech i{{flex:1;height:1px;background:{p['border']}}}
.chip{{display:inline-block;font-size:11.5px;font-weight:600;border-radius:999px;padding:3px 10px;margin:0 6px 6px 0;
  border:1px solid {p['border']};color:{p['muted']}}}
.pcard{{background:{p['card']};border:1px solid {p['border']};border-radius:16px;padding:12px 14px 14px;margin-bottom:16px}}
.pcard h4{{margin:10px 0 6px;font-size:16px;font-weight:600}}
.badge{{display:inline-block;font-size:11px;font-weight:600;border-radius:999px;padding:2px 9px;margin:0 4px 4px 0;
  border:1px solid {p['border']};color:{p['muted']}}}
.pcard p{{font-size:13.5px;line-height:1.55;margin:7px 0;color:{p['text']};opacity:.88}}
.legend{{display:flex;flex-wrap:wrap;gap:8px 18px;background:{p['card']};border:1px solid {p['border']};
  border-radius:12px;padding:12px 16px;margin:10px 0 16px;font-size:13px;color:{p['muted']}}}
.legend span{{display:inline-flex;align-items:center;gap:7px}}
.legend i{{width:14px;height:14px;border-radius:3px;display:inline-block}}
.cheat{{width:100%;border-collapse:separate;border-spacing:0;font-size:13.5px;margin:4px 0 14px}}
.cheat td,.cheat th{{padding:9px 12px;border-bottom:1px solid {p['border']} !important;border-left:none !important;
  border-right:none !important;border-top:none !important;text-align:left;background:transparent !important}}
.cheat th{{color:{p['muted']};font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.04em}}
.step{{background:{p['card']};border:1px solid {p['border']};border-radius:16px;padding:16px 18px}}
.step .ic{{font-size:22px}}
.step h4{{margin:6px 0 6px;font-size:16px}}
.step li{{font-size:13.6px;line-height:1.6;color:{p['text']};opacity:.88}}
</style>"""


# ----------------------------------------------------------------------------
# Mini diagrams for glossary cards
# ----------------------------------------------------------------------------
def _svg(inner, h=92):
    return (f'<svg viewBox="0 0 300 {h}" width="100%" height="{h}" xmlns="http://www.w3.org/2000/svg" '
            f'font-family="sans-serif" font-size="10"><rect width="300" height="{h}" fill="{PAL["card2"]}"/>{inner}</svg>')


def _path(pts, col, w=2, dash=""):
    d = " ".join(f"{'M' if i == 0 else 'L'}{x},{y}" for i, (x, y) in enumerate(pts))
    da = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<path d="{d}" fill="none" stroke="{col}" stroke-width="{w}" stroke-linejoin="round"{da}/>'


def _hl(y, col, label="", dash="5 4", x0=10, x1=290, anchor_right=True):
    t = (f'<text x="{x1 - 2 if anchor_right else x0 + 2}" y="{y - 4}" fill="{col}" '
         f'text-anchor="{"end" if anchor_right else "start"}">{label}</text>') if label else ""
    return f'<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" stroke="{col}" stroke-width="1.3" stroke-dasharray="{dash}"/>{t}'


def _viz(key):
    t, mu, up, dn, ac, gold = PAL["text"], PAL["muted"], BULL, BEAR, PAL["accent"], "#e3c46a"
    if key == "sr":
        pts = [(70, 60), (95, 26), (120, 66), (145, 24), (170, 64), (195, 25), (220, 62), (245, 27), (270, 65), (290, 44)]
        return _svg(_hl(22, dn, "Resistance", "5 4", 10, 290, False) + _hl(68, up, "Support", "5 4", 10, 290, False)
                    + _path(pts, t, 1.8))
    if key == "breakout":
        pts = [(10, 60), (40, 34), (70, 58), (100, 33), (130, 56), (160, 34), (190, 50), (220, 22), (250, 14), (290, 8)]
        bars = "".join(f'<rect x="{x}" y="{92 - h}" width="10" height="{h}" fill="{mu}" opacity=".45"/>'
                       for x, h in [(35, 8), (65, 6), (95, 9), (125, 7), (155, 8), (185, 10)])
        bars += f'<rect x="215" y="66" width="10" height="26" fill="{up}"/><rect x="245" y="70" width="10" height="22" fill="{up}"/>'
        return _svg(bars + _hl(31, dn, "old resistance") + _path(pts, t, 1.8) +
                    f'<text x="232" y="60" fill="{up}" text-anchor="middle">high vol</text>')
    if key == "pullback":
        pts = [(10, 80), (60, 50), (90, 62), (140, 30), (175, 46), (230, 14), (290, 6)]
        return _svg(_path(pts, t, 1.8) + _path([(10, 88), (290, 20)], ac, 1.2, "4 4") +
                    f'<circle cx="90" cy="62" r="5" fill="none" stroke="{up}" stroke-width="2"/>'
                    f'<circle cx="175" cy="46" r="5" fill="none" stroke="{up}" stroke-width="2"/>'
                    f'<text x="182" y="62" fill="{up}">pullback = buy area</text>')
    if key == "trend":
        pts = [(10, 82), (50, 54), (80, 66), (130, 34), (160, 48), (215, 16), (245, 28), (290, 8)]
        return _svg(_path(pts, up, 2) + f'<text x="50" y="48" fill="{mu}">HH</text><text x="80" y="80" fill="{mu}">HL</text>'
                    f'<text x="130" y="28" fill="{mu}">HH</text><text x="160" y="62" fill="{mu}">HL</text>'
                    f'<text x="215" y="12" fill="{mu}">HH</text>')
    if key == "rsi":
        ys = [50, 40, 28, 18, 22, 38, 55, 70, 76, 66, 50, 40, 46]
        pts = [(10 + i * 23, y) for i, y in enumerate(ys)]
        return _svg(f'<rect x="10" y="20" width="280" height="50" fill="{mu}" opacity=".07"/>' +
                    _hl(20, dn, "70 overbought", "3 3") + _hl(70, up, "30 oversold", "3 3") + _path(pts, "#b48ae0", 2))
    if key == "macd":
        a = [(10 + i * 20, 46 + 22 * math.sin(i / 2.2)) for i in range(15)]
        b = [(10 + i * 20, 46 + 18 * math.sin((i - 1.3) / 2.2)) for i in range(15)]
        bars = "".join(f'<rect x="{x - 4}" y="{min(46, 46 + (ay - by) * 1.6)}" width="8" height="{abs(ay - by) * 1.6}" '
                       f'fill="{up if ay < by else dn}" opacity=".6"/>' for (x, ay), (_, by) in zip(a, b))
        return _svg(bars + _path(a, "#6fa8ef", 1.8) + _path(b, "#e3a34b", 1.6) +
                    f'<text x="292" y="12" fill="{mu}" text-anchor="end">blue = MACD · orange = signal</text>')
    if key == "ma":
        import random
        rnd = random.Random(3)
        px = [(10 + i * 7, 70 - i * 1.0 + rnd.uniform(-9, 9)) for i in range(41)]
        ma = [(x, sum(p[1] for p in px[max(0, i - 6):i + 1]) / len(px[max(0, i - 6):i + 1])) for i, (x, _) in enumerate(px)]
        return _svg(_path(px, mu, 1.2) + _path(ma, gold, 2.2) + f'<text x="292" y="86" fill="{gold}" text-anchor="end">moving average</text>')
    if key == "cross":
        fast = [(10 + i * 14, 70 - 40 / (1 + math.exp(-(i - 10) / 2.5))) for i in range(21)]
        slow = [(10 + i * 14, 58 - 14 / (1 + math.exp(-(i - 12) / 4))) for i in range(21)]
        return _svg(_path(slow, "#b48ae0", 2) + _path(fast, "#6fa8ef", 2) +
                    f'<circle cx="150" cy="51" r="7" fill="none" stroke="{gold}" stroke-width="2"/>'
                    f'<text x="160" y="80" fill="{gold}">golden cross ✨</text>'
                    f'<text x="292" y="12" fill="{mu}" text-anchor="end">blue = 50-day · purple = 200-day</text>')
    if key == "bidask":
        rows = [("4.54", "12,000", dn), ("4.53", "8,500", dn), ("4.52", "5,100", dn), ("4.50", "9,800", up),
                ("4.49", "15,200", up), ("4.48", "7,300", up)]
        inner = "".join(f'<rect x="70" y="{4 + i * 14}" width="160" height="13" fill="{c}" opacity=".12"/>'
                        f'<text x="110" y="{14 + i * 14}" fill="{c}" text-anchor="middle">{p_}</text>'
                        f'<text x="190" y="{14 + i * 14}" fill="{mu}" text-anchor="middle">{v}</text>'
                        for i, (p_, v, c) in enumerate(rows))
        return _svg(inner + f'<text x="62" y="28" fill="{dn}" text-anchor="end">ASK ↑</text>'
                    f'<text x="62" y="70" fill="{up}" text-anchor="end">BID ↓</text>'
                    f'<text x="238" y="48" fill="{mu}">spread 2 sen</text>', 90)
    if key == "rr":
        return _svg(f'<rect x="40" y="38" width="70" height="18" fill="{dn}" opacity=".75" rx="3"/>'
                    f'<rect x="110" y="38" width="150" height="18" fill="{up}" opacity=".75" rx="3"/>'
                    f'<line x1="110" y1="28" x2="110" y2="66" stroke="{t}" stroke-width="2"/>'
                    f'<text x="75" y="32" fill="{dn}" text-anchor="middle">risk 10 sen</text>'
                    f'<text x="185" y="32" fill="{up}" text-anchor="middle">reward 25 sen</text>'
                    f'<text x="110" y="80" fill="{t}" text-anchor="middle">entry</text>'
                    f'<text x="40" y="80" fill="{mu}" text-anchor="middle">stop</text>'
                    f'<text x="260" y="80" fill="{mu}" text-anchor="middle">target</text>'
                    f'<text x="292" y="14" fill="{t}" text-anchor="end" font-weight="bold">R:R = 1 : 2.5</text>')
    if key == "gap":
        return candle_svg([(30, 52, 26, 48), (64, 86, 60, 82)], None, 300, 92).replace(
            "</svg>", f'<rect x="120" y="0" width="60" height="92" fill="{gold}" opacity="0"/>'
                      f'<text x="8" y="86" fill="{gold}" font-size="10" font-family="sans-serif" '
                      f'text-anchor="start">gap up: empty space between candles</text></svg>')
    if key == "fib":
        lv = [(0, "100%"), (0.236, "78.6%"), (0.382, "61.8%"), (0.5, "50%"), (0.618, "38.2%"), (0.786, "23.6%"), (1, "0%")]
        inner = "".join(_hl(10 + r * 72, gold if lab in ("38.2%", "50%", "61.8%") else mu, lab, "3 3", 140, 290)
                        for r, lab in lv)
        return _svg(inner + _path([(10, 82), (70, 30), (120, 10), (160, 38), (200, 46)], t, 2) +
                    f'<circle cx="200" cy="46" r="5" fill="none" stroke="{up}" stroke-width="2"/>')
    if key == "vp":
        vols = [6, 10, 18, 30, 46, 62, 50, 34, 20, 12, 8]
        inner = "".join(f'<rect x="{290 - v * 2.4}" y="{6 + i * 7.5}" width="{v * 2.4}" height="6.5" '
                        f'fill="{"#e3a34b" if v == 62 else ac}" opacity="{.85 if v == 62 else .4}"/>'
                        for i, v in enumerate(vols))
        return _svg(inner + _hl(6 + 5 * 7.5 + 3, "#e3a34b", "POC", "4 3", 10, 140, False) +
                    _path([(10, 70), (40, 30), (70, 50), (100, 40), (130, 48)], t, 1.6))
    if key == "atr":
        cs = [(50, 62, 38, 56), (56, 66, 46, 50), (50, 60, 36, 44), (44, 58, 34, 54)]
        svg = candle_svg(cs, None, 300, 92)
        return svg.replace("</svg>", f'<text x="292" y="14" fill="{gold}" font-size="10" font-family="sans-serif" '
                                     f'text-anchor="end">ATR ≈ average height of candles</text></svg>')
    if key == "divergence":
        return _svg(_path([(10, 50), (70, 30), (110, 44), (170, 22), (210, 36), (270, 14)], t, 1.8) +
                    _path([(70, 30), (170, 22), (270, 14)], up, 1, "3 3") +
                    f'<text x="292" y="46" fill="{mu}" text-anchor="end">price: higher highs</text>' +
                    _path([(10, 84), (70, 60), (110, 76), (170, 66), (210, 80), (270, 74)], "#b48ae0", 1.8) +
                    _path([(70, 60), (170, 66), (270, 74)], dn, 1, "3 3") +
                    f'<text x="292" y="90" fill="#b48ae0" text-anchor="end">RSI: lower highs ⚠️</text>', 96)
    if key == "drawdown":
        return _svg(_path([(10, 60), (70, 14), (150, 74), (290, 30)], t, 2) +
                    f'<line x1="70" y1="14" x2="70" y2="74" stroke="{dn}" stroke-dasharray="3 3"/>'
                    f'<text x="76" y="50" fill="{dn}">−50%</text><text x="200" y="70" fill="{up}">needs +100% to recover</text>')
    if key == "dca":
        inner = "".join(f'<rect x="{18 + i * 23}" y="{78 - 12}" width="16" height="12" fill="{ac}" opacity=".7" rx="2"/>'
                        for i in range(12))
        return _svg(inner + _path([(18 + i * 23 + 8, 50 + 16 * math.sin(i / 1.7)) for i in range(12)], t, 1.6) +
                    f'<text x="10" y="14" fill="{mu}">same RM amount every month → buys more when cheap</text>')
    if key == "candle":
        return candle_svg([(40, 70, 28, 62), (62, 66, 44, 48)], None, 300, 92)
    if key == "volume":
        hs = [10, 14, 8, 12, 9, 11, 13, 10, 44, 38, 16, 12]
        inner = "".join(f'<rect x="{14 + i * 23}" y="{86 - h}" width="15" height="{h}" fill="{up if h > 30 else mu}" '
                        f'opacity="{.85 if h > 30 else .45}" rx="2"/>' for i, h in enumerate(hs))
        return _svg(inner + f'<text x="292" y="16" fill="{up}" text-anchor="end">volume spike = big interest</text>')
    return ""


# term -> (abbreviation tile, worked example, diagram key)
TERM_EXTRA = {
    "Stock / Share": ("📄", "Company has 1 billion shares. You own 1,000 → you own <b>0.0001%</b> of it.", None),
    "Ticker": ("TKR", "<b>MAYBANK</b> = 1155.KL on Bursa · <b>AAPL</b> = Apple on Nasdaq.", None),
    "Bid / Ask": ("B/A", "Bid <b>4.50</b> / Ask <b>4.52</b> → buy now at 4.52, sell now at 4.50.", "bidask"),
    "Spread": ("SPR", "Ask 4.52 − Bid 4.50 = <b>2 sen</b>. Thin stocks can have 5–10 sen spreads.", None),
    "Volume": ("VOL", "Normally 2M shares/day, today <b>8M</b> → 4× normal: something is happening.", "volume"),
    "Liquidity": ("LIQ", "Maybank trades RM100M+ a day (easy to exit). A tiny stock may trade RM20k (hard to exit).", None),
    "Market cap": ("CAP", "RM10 × 1 billion shares = <b>RM10 billion</b> market cap.", None),
    "Blue chip": ("BC", "Maybank, Tenaga, Public Bank, Apple, Microsoft.", None),
    "Penny stock": ("¢", "A RM0.05 stock moving 1 sen = <b>20%</b> swing in one tick.", None),
    "ETF": ("ETF", "Buy <b>1 unit of SPY</b> → own a slice of all 500 S&P companies.", None),
    "Lot / Board lot": ("LOT", "3 lots at RM2.00 = 300 shares = <b>RM600</b> (+ fees).", None),
    "Bull market / Bear market": ("🐂🐻", "Index falls from 1,600 to 1,250 (−22%) → bear market.", None),
    "Bullish / Bearish": ("↑↓", "'I'm bullish on Tenaga' = I think Tenaga will go up.", None),
    "Market hours": ("⏰", "Bursa: <b>9:00–12:30</b> & <b>2:30–5:00pm</b>. US in winter: <b>10:30pm–5:00am MYT</b>.", None),
    "T+2 / T+1 settlement": ("T+2", "Buy on Bursa <b>Monday</b> → settled <b>Wednesday</b>.", None),
    "Contra (Bursa)": ("CON", "Buy Mon, sell Tue at a gain → you only receive/pay the difference.", None),
    "Market order": ("MKT", "Ask is 4.52 → your market buy fills at ~4.52 (maybe 4.53 if it moves).", None),
    "Limit order": ("LMT", "Limit buy at <b>4.45</b> → only fills if someone sells at 4.45 or lower.", None),
    "Stop loss": ("SL", "Buy 2.00, stop <b>1.90</b> → max loss 10 sen/share (−5%).", None),
    "Take profit": ("TP", "Buy 2.00, take profit <b>2.30</b> → locks in +15% automatically.", None),
    "Trailing stop": ("TS", "5% trailing: price hits 2.40 → stop rises to <b>2.28</b>.", None),
    "Slippage": ("SLP", "Wanted 2.00, filled at <b>2.02</b> → 2 sen slippage.", None),
    "Long / Short": ("L/S", "Long at 2.00 → sell 2.30 = +30 sen. Short at 2.00 → buy back 1.70 = +30 sen.", None),
    "Candlestick": ("🕯️", "Green = closed higher than it opened. Red = closed lower.", "candle"),
    "Timeframe": ("TF", "Daily chart: 1 candle = 1 day. 15-min chart: 1 candle = 15 minutes.", None),
    "Trend": ("HH", "Uptrend = <b>H</b>igher <b>H</b>ighs + <b>H</b>igher <b>L</b>ows.", "trend"),
    "Support": ("S", "Price bounced at <b>RM1.80</b> three times → strong support.", "sr"),
    "Resistance": ("R", "Price failed at <b>RM2.50</b> twice → resistance.", "sr"),
    "Breakout": ("BO", "Resistance 2.50 → closes at <b>2.58</b> with 3× volume = breakout.", "breakout"),
    "Breakdown": ("BD", "Support 1.80 → closes at <b>1.72</b> = breakdown. Next support is lower.", None),
    "Pullback": ("PB", "Rises 2.00 → 2.40, dips to <b>2.25</b> near support → common buy spot.", "pullback"),
    "Fakeout": ("FO", "Pops to 2.55 above 2.50 at noon, closes at 2.46 → fakeout.", None),
    "Gap": ("GAP", "Closed 2.00 Friday, opens <b>2.20</b> Monday after news → gap up.", "gap"),
    "Moving average (SMA / EMA)": ("MA", "20-day MA = average close of the last 20 days.", "ma"),
    "EMA 20 / 50 / 200": ("EMA", "Price 5.20, EMA200 4.80 → above EMA200 = long-term uptrend.", None),
    "Golden cross / Death cross": ("GC", "50-day crosses <b>above</b> 200-day = golden cross (bullish).", "cross"),
    "RSI": ("RSI", "RSI <b>25</b> = oversold (bounce?) · RSI <b>78</b> = overbought (pullback?).", "rsi"),
    "Overbought / Oversold": ("OB/OS", "Up 30% in 6 days with RSI 82 → overbought. Can still go higher!", None),
    "MACD": ("MACD", "MACD line crosses <b>above</b> the signal line → momentum turning up.", "macd"),
    "ATR": ("ATR", "ATR <b>0.08</b> → moves ~8 sen a day. Stop = 1 ATR below support.", "atr"),
    "Volatility": ("σ", "Moves ±5%/day = high volatility. ±0.5%/day = low.", None),
    "Swing high / Swing low": ("SW", "Peak at 2.60 with lower highs either side → swing high = resistance.", "sr"),
    "Repeated touches": ("3×", "Bounced at 1.80 in Mar, Jun and Sep → <b>3 touches</b>: strong level.", None),
    "Volume Profile": ("VP", "Most shares traded around 2.10–2.20 → price tends to stick there.", "vp"),
    "POC (Point of Control)": ("POC", "POC at <b>2.15</b> → the most-traded price, often a magnet.", "vp"),
    "Value Area (VAH / VAL)": ("VA", "70% of volume between <b>2.05 (VAL)</b> and <b>2.28 (VAH)</b>.", None),
    "Fibonacci retracement": ("FIB", "Rose 1.00 → 2.00. 61.8% pullback = <b>1.38</b>, 50% = 1.50.", "fib"),
    "Divergence": ("DIV", "Price makes a new high but RSI doesn't → uptrend may be tiring.", "divergence"),
    "Risk : Reward (R:R)": ("R:R", "Risk 10 sen (stop) to make 25 sen (target) → <b>1 : 2.5</b>.", "rr"),
    "Position sizing": ("PS", "RM10,000 × 1% = RM100 risk. Stop 20 sen away → <b>500 shares (5 lots)</b>.", None),
    "Diversification": ("DIV+", "RM10k split across 6 stocks in 4 sectors, not all in one.", None),
    "Drawdown": ("DD", "RM10k → RM5k is −50%. You now need <b>+100%</b> to get back.", "drawdown"),
    "Averaging down": ("AVG↓", "Bought at 2.00, buy more at 1.50 → avg 1.75, but double the exposure to a loser.", None),
    "DCA (Dollar-cost averaging)": ("DCA", "RM300 into an ETF every month, rain or shine.", "dca"),
    "FOMO": ("😱", "Stock +40% this week, everyone on TikTok talking → classic FOMO top.", None),
    "Paper trading": ("📝", "Practise on moomoo's paper account with RM100k of fake money.", None),
    "EPS": ("EPS", "Profit RM100M ÷ 500M shares = EPS <b>RM0.20</b>.", None),
    "P/E ratio": ("P/E", "Price RM5.00 ÷ EPS RM0.50 = P/E <b>10</b> (pay 10 years of profit).", None),
    "Dividend": ("DIV", "Company pays <b>RM0.10</b>/share → 1,000 shares = RM100.", None),
    "Dividend yield": ("YLD", "RM0.25/year on a RM5 share = <b>5%</b> yield.", None),
    "Ex-dividend date": ("EX", "Ex-date <b>Wednesday</b> → you must own it by Tuesday's close.", None),
    "Earnings / Quarterly results": ("QR", "Results beat forecast → stock gaps up 8% next morning.", None),
    "Shariah-compliant": ("☪", "~80% of Bursa stocks are on the SC Shariah list; banks are mostly not.", None),
}

TIPS = [
    ("protect", "🛡️", "Risk only 1–2% per trade", "Your #1 job is not losing big.", "RM5,000 account → max loss <b>RM50–100</b> per trade."),
    ("protect", "🛑", "Set the stop loss before you buy", "Decide your exit if wrong, then place the stop order immediately.", "Buy 2.00, stop 1.90 → worst case −5%, not −40%."),
    ("protect", "⚖️", "Only take R:R of 1:2 or better", "Then you can be wrong half the time and still make money.", "Risk RM100 to make RM200+. 5 wins, 5 losses = <b>+RM500</b>."),
    ("protect", "📅", "Avoid holding through results", "Quarterly results can gap price 10–20% overnight, past your stop.", "Stop at −5%, but results gap it <b>−15%</b> at the open."),
    ("protect", "🚫", "Don't average down on losers", "Adding to a loser multiplies the damage. Add to winners instead.", "Bought 2.00, now 1.50: buying more doubles the bet on a broken stock."),
    ("entry", "📈", "Trade with the trend", "Mostly buy stocks above EMA200 with EMA50 above EMA200.", "Price 5.20 > EMA200 4.80 ✅ · price 3.10 < EMA200 3.60 ❌"),
    ("entry", "🎯", "Buy near support, sell near resistance", "Close stop below support = small risk; room to run = big reward.", "Support 1.80, resistance 2.20 → buy ~1.82, stop 1.74, target 2.18."),
    ("entry", "🕯️", "Wait for the candle to close", "Signals only count once the candle has closed.", "Breaks 2.50 at 11am, closes 2.47 → no breakout."),
    ("entry", "📊", "Volume confirms the move", "Breakouts with high volume are far more reliable.", "Breakout on <b>3× average volume</b> ✅ vs 0.5× ❌"),
    ("entry", "🔭", "Check a higher timeframe", "Best setups agree on daily and weekly.", "Daily BUY + weekly uptrend ✅ · daily BUY + weekly downtrend ⚠️"),
    ("entry", "🧩", "Look for 2–3 reasons, not 1", "Pattern + level + indicator beats any one alone.", "Hammer <b>+</b> at support <b>+</b> RSI 28 = strong case."),
    ("mind", "😤", "Don't chase (beat FOMO)", "If it already ran 20% today, you're late. Wait for a pullback.", "Missed 2.00 → 2.40? Wait for a dip to ~2.25 or skip it."),
    ("mind", "📓", "Keep a trading journal", "Write why you entered, your stop, what happened and how you felt.", "After 30 trades: 'I lose most on Friday afternoon FOMO buys.'"),
    ("mind", "🧪", "Paper trade first", "Practise a month on moomoo's paper account before real money.", "Follow this app's Picks on paper for 4 weeks, then review."),
    ("mind", "🎲", "Signals are probabilities", "No indicator is always right. Small losses, bigger wins.", "Win 45% of trades at 1:2.5 R:R → still profitable."),
]
TIP_GROUPS = [("protect", "🛡️ Protect your money"), ("entry", "🎯 Better entries"), ("mind", "🧠 Mindset")]


def _term_card(term, cat, desc):
    abbr, ex, viz = TERM_EXTRA.get(term, (term[:3].upper(), None, None))
    icon, ckey = CAT_STYLE.get(cat, ("•", "accent"))
    col = _col(ckey)
    from_hex = col.lstrip("#")
    r, g, b = int(from_hex[0:2], 16), int(from_hex[2:4], 16), int(from_hex[4:6], 16)
    tile = (f"<div class='abbr' style='background:rgba({r},{g},{b},.16);color:{col};"
            f"border:1px solid rgba({r},{g},{b},.4);font-size:{15 if len(abbr) <= 3 else 12}px'>{abbr}</div>")
    v = _viz(viz) if viz else ""
    return (f"<div class='tcard'><div class='thead'>{tile}<div><div class='tname'>{term}</div>"
            f"<div class='tcat'>{icon} {cat}</div></div></div>"
            f"<div class='tdesc'>{desc}</div>"
            + (f"<div class='viz'>{v}</div>" if v else "")
            + (f"<div class='ex'>💡 <b>Example:</b> {ex}</div>" if ex else "")
            + "</div>")


def _tip_card(i, icon, title, body, ex):
    return (f"<div class='tcard'><div class='thead'><div class='abbr' style='background:{PAL['card2']};"
            f"border:1px solid {PAL['border']};font-size:20px'>{icon}</div><div><div class='tname'>{i}. {title}</div>"
            f"</div></div><div class='tdesc'>{body}</div><div class='ex'>💡 {ex}</div></div>")


def _section(title):
    st.markdown(f"<div class='sech'><span>{title}</span><i></i></div>", unsafe_allow_html=True)


def render():
    st.markdown(_css(), unsafe_allow_html=True)
    st.markdown(f"""
<div class="lhero"><h2>🎓 Learn trading from zero</h2>
<div class="sub">Short, simple lessons with real examples. Start with <b>How to use this app</b>, then
<b>Candlesticks</b>, then the <b>Trading Terms</b> as you need them. Keep the <b>Tips</b> in mind for every trade.</div>
<div class="path">
 <div><span class="n">1</span><b>How to use this app</b>What every number and colour means</div>
 <div><span class="n">2</span><b>Candlestick patterns</b>{len(PATTERNS)} patterns with pictures</div>
 <div><span class="n">3</span><b>Trading terms</b>{len(GLOSSARY)} terms with examples</div>
 <div><span class="n">4</span><b>Tips & tricks</b>{len(TIPS)} rules + position-size calculator</div>
</div></div>""", unsafe_allow_html=True)

    t4, t3, t2, t1 = st.tabs(["🧭 How to use this app", "🕯️ Candlestick Patterns", "📖 Trading Terms",
                              "💡 Tips & Tricks"])

    # ---------------- Trading terms
    with t2:
        c1, c2 = st.columns([1, 2])
        q = c1.text_input("🔍 Search", placeholder="e.g. RSI, stop loss, lot", key="gl_q")
        cats = list(dict.fromkeys(g[1] for g in GLOSSARY))
        pick = c2.segmented_control("Category", ["All"] + cats, default="All", key="gl_cat") or "All"
        hits = [g for g in GLOSSARY
                if (pick == "All" or g[1] == pick)
                and (not q or q.lower() in g[0].lower() or q.lower() in g[2].lower()
                     or q.lower() in TERM_EXTRA.get(g[0], ("",))[0].lower())]
        if not hits:
            st.info("No term found. Try another word.")
        for cat in cats:
            items = [g for g in hits if g[1] == cat]
            if not items:
                continue
            icon, _ = CAT_STYLE.get(cat, ("•", "accent"))
            _section(f"{icon} {cat} · {len(items)}")
            st.markdown("<div class='grid'>" + "".join(_term_card(*g) for g in items) + "</div>",
                        unsafe_allow_html=True)

    # ---------------- Tips
    with t1:
        n = 0
        for key, title in TIP_GROUPS:
            _section(title)
            cards = []
            for g, icon, ttl, body, ex in TIPS:
                if g == key:
                    n += 1
                    cards.append(_tip_card(n, icon, ttl, body, ex))
            st.markdown("<div class='grid'>" + "".join(cards) + "</div>", unsafe_allow_html=True)
        _section("🧮 Position size calculator")
        _position_calculator()

    # ---------------- Candles
    with t3:
        a, b = st.columns([1.1, 1], gap="large")
        with a:
            st.markdown(anatomy_svg(), unsafe_allow_html=True)
        with b:
            st.markdown(f"""
<div class="tcard"><div class="tname">How to read one candle</div>
<div class="tdesc">Each candle = 4 prices for one period: <b>Open, High, Low, Close</b>.<br>
• <b>Body</b> (thick part) = open → close.<br>
• <b>Wicks</b> (thin lines) = how far price went beyond.<br>
• <b>Long wick = rejection</b>: price went there but couldn't stay.</div>
<div class="ex">💡 <b>Example:</b> Opens 2.00, high 2.15, low 1.95, closes 2.10 → green candle, body 2.00–2.10,
upper wick to 2.15, lower wick to 1.95.</div></div>""", unsafe_allow_html=True)
        st.markdown(f"""
<div class="legend">
 <span><i style="background:{BULL}"></i>Bullish candle / pattern</span>
 <span><i style="background:{BEAR}"></i>Bearish candle / pattern</span>
 <span><i style="background:{NEUT}"></i>Indecision (doji)</span>
 <span><i style="background:{PAL['muted']};opacity:.4"></i>Faded = trend <b>before</b> the pattern</span>
</div>""", unsafe_allow_html=True)
        _section("⚡ Cheat sheet")
        st.markdown(f"""
<table class="cheat"><tr><th>If you see…</th><th>After a…</th><th>It usually means</th><th>Do this</th></tr>
<tr><td>Hammer, Bullish Engulfing, Morning Star</td><td>Downtrend</td><td style="color:{BULL}">Possible bottom ↑</td><td>Wait for a green confirm candle, stop below the low</td></tr>
<tr><td>Shooting Star, Bearish Engulfing, Evening Star</td><td>Uptrend</td><td style="color:{BEAR}">Possible top ↓</td><td>Take profit / tighten stop</td></tr>
<tr><td>Doji, Spinning Top</td><td>Anywhere</td><td style="color:{PAL['muted']}">Indecision</td><td>Wait for the next candle</td></tr>
<tr><td>Marubozu, Three White Soldiers</td><td>Anywhere</td><td style="color:{BULL}">Strong buying</td><td>Don't chase: buy the pullback</td></tr>
<tr><td>Three Black Crows, Bearish Marubozu</td><td>Anywhere</td><td style="color:{BEAR}">Strong selling</td><td>Avoid buying for now</td></tr>
</table>""", unsafe_allow_html=True)
        flt = st.segmented_control("Show", ["All", "Bullish", "Bearish", "Indecision"], default="All",
                                   key="pat_filter") or "All"
        want = {"All": None, "Bullish": "bull", "Bearish": "bear", "Indecision": "neutral"}[flt]
        for group, size in (("Single-candle patterns", 1), ("Two-candle patterns", 2), ("Three-candle patterns", 3)):
            items = [p for p in PATTERNS if len(p[3]) == size and (want is None or p[1] == want)]
            if not items:
                continue
            _section(f"{group} · {len(items)}")
            cols = st.columns(3)
            for k, p in enumerate(items):
                cols[k % 3].markdown(_pattern_card(p), unsafe_allow_html=True)

    # ---------------- How to use
    with t4:
        steps = [
            ("🔎", "Screener", ["Every stock gets a <b>Signal</b> and a <b>Score</b> (−100 to +100).",
                                "<b>To support %</b>: small = closer to a low-risk entry.",
                                "<b>R:R</b>: reward vs risk. Prefer <b>2 or more</b>.",
                                "Filter: <i>Buy only</i> + <i>R:R ≥ 2</i> for a shortlist."]),
            ("🎯", "Picks", ["<b>Short term</b>: setups for 1–4 weeks with buy zone, stop and target.",
                             "<b>Long term</b>: healthy trends to hold 6 months+.",
                             "Empty list? No good setups now: waiting is fine."]),
            ("🗺️", "Heatmap", ["Green = up, red = down. Bigger box = more trading.",
                               "Switch to 1 week / 1 month to see trends, not noise.",
                               "Check which <b>sectors</b> are strong."]),
            ("📊", "Analysis & Forecast", ["<b style='color:" + BULL + "'>Green lines</b> = support, "
                                          "<b style='color:" + BEAR + "'>red</b> = resistance, (3x) = touches.",
                                          "Green box = buy zone · red box = sell zone · orange dashed = stop.",
                                          "Blue dashed = trend forecast with likely range.",
                                          "Add extra levels: swings, touches, volume profile, Fibonacci."]),
            ("💰", "Dividends", ["Yield, how often they pay, next <b>ex-date</b>.",
                                 "✅ Confirmed = announced · 🕒 Estimated = same time as last year.",
                                 "Must own the shares <b>before</b> the ex-date."]),
            ("🚀", "IPO", ["Latest Bursa IPO news + US IPO calendar.",
                           "Recently listed US IPOs show % since IPO."]),
        ]
        cards = "".join(
            f"<div class='step'><div class='ic'>{ic}</div><h4>{ttl}</h4><ul>"
            + "".join(f"<li>{x}</li>" for x in lis) + "</ul></div>" for ic, ttl, lis in steps)
        st.markdown(f"<div class='grid'>{cards}</div>", unsafe_allow_html=True)
        _section("🗓️ A simple beginner routine")
        routine = [
            ("1", "Shortlist", "Picks → Short term, or Screener → <i>Buy only</i> + <i>R:R ≥ 2</i>."),
            ("2", "Check the chart", "Analysis: above EMA200? Near the green buy zone? Bullish candle?"),
            ("3", "Zoom out", "Switch the sidebar timeframe to <b>Weekly</b>. Does it agree?"),
            ("4", "Size it", "Tips → position size calculator: risk 1% of your account."),
            ("5", "Place orders", "Limit buy in the buy zone, <b>stop loss straight away</b>, take profit at Target 1."),
            ("6", "Practise first", "Do steps 1–5 on moomoo <b>paper trading</b> for a few weeks."),
        ]
        st.markdown("<div class='path'>" + "".join(
            f"<div><span class='n'>{n}</span><b style='display:inline'>{t}</b><br>{d}</div>" for n, t, d in routine)
            + "</div>", unsafe_allow_html=True)


def _pattern_card(p):
    name, kind, ctx, candles, looks, means, use = p
    label, col = _badge(kind)
    size = {1: "Single candle", 2: "Two candles", 3: "Three candles"}[len(candles)]
    where = {"down": "after a downtrend", "up": "after an uptrend", None: "anywhere"}[ctx]
    return f"""<div class="pcard">{candle_svg(candles, ctx)}
<h4>{name}</h4>
<span class="badge" style="color:{col};border-color:{col}">{label}</span><span class="badge">{size}</span><span class="badge">Appears {where}</span>
<p><b>Looks like:</b> {looks}</p>
<p><b>Meaning:</b> {means}</p>
<p><b>How to use:</b> {use}</p></div>"""



def _position_calculator():
    st.markdown("#### 🧮 Position size calculator")
    st.caption("How many shares should I buy so that, if my stop loss is hit, I only lose a small % of my account?")
    c1, c2, c3 = st.columns(3)
    capital = c1.number_input("Account size", min_value=100.0, value=5000.0, step=500.0)
    risk_pct = c2.slider("Risk per trade (%)", 0.5, 5.0, 1.0, 0.5)
    mkt = c3.radio("Market", ["Bursa (lots of 100)", "US (single shares)"], horizontal=False)
    c4, c5 = st.columns(2)
    entry = c4.number_input("Entry price", min_value=0.001, value=10.00, step=0.01, format="%.3f")
    stop = c5.number_input("Stop loss price", min_value=0.0, value=9.50, step=0.01, format="%.3f")
    if stop >= entry:
        st.warning("Stop loss must be below the entry price (for a buy trade).")
        return
    risk_amt = capital * risk_pct / 100
    per_share = entry - stop
    shares = math.floor(risk_amt / per_share)
    lot = 100 if mkt.startswith("Bursa") else 1
    shares = (shares // lot) * lot
    cost = shares * entry
    if cost > capital:
        shares = int((capital // entry) // lot * lot)
        cost = shares * entry
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Max loss allowed", f"{risk_amt:,.2f}")
    m2.metric("Shares to buy", f"{shares:,}" + (f" ({shares // 100} lots)" if lot == 100 else ""))
    m3.metric("Position cost", f"{cost:,.2f}")
    m4.metric("Loss if stop hit", f"{shares * per_share:,.2f}")
    if shares == 0:
        st.info("Your stop is too far away for this account size and risk %. Choose a closer stop or a cheaper stock.")
    st.caption("Fees and slippage are not included. Bursa brokerage + stamp duty + clearing fees add a little to every trade.")


