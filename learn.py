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
def _css():
    p = PAL
    return f"""
<style>
.pcard{{background:{p['card']};border:1px solid {p['border']};border-radius:16px;padding:12px 14px 14px;margin-bottom:16px}}
.pcard h4{{margin:10px 0 6px;font-size:16px;font-weight:600}}
.badge{{display:inline-block;font-size:11px;font-weight:600;border-radius:999px;padding:2px 9px;margin:0 4px 4px 0;
  border:1px solid {p['border']};color:{p['muted']}}}
.pcard p{{font-size:13.5px;line-height:1.55;margin:7px 0;color:{p['text']};opacity:.88}}
.pcard p b{{opacity:1}}
.tip{{background:{p['card']};border:1px solid {p['border']};border-left:3px solid {p['accent']};border-radius:12px;
  padding:12px 16px;margin-bottom:10px;font-size:14px;line-height:1.6}}
</style>"""


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


TIPS = [
    ("Protect your capital first", "Your #1 job is not losing big. Never risk more than <b>1–2% of your account</b> on one trade. With RM5,000, that's a max loss of RM50–100 per trade. Use the calculator below."),
    ("Always set a stop loss before you buy", "Decide where you'll exit if you're wrong <b>before</b> entering. Put the stop order in immediately. Hoping a loser comes back is how small losses become big ones."),
    ("Trade with the trend", "Easiest money is made going with the trend. Beginners should mostly buy stocks <b>above EMA200</b> with EMA50 above EMA200, and avoid catching falling knives."),
    ("Buy near support, sell near resistance", "Buying near support gives you a close stop loss (small risk) and room to run to resistance (bigger reward). Buying right under resistance is the opposite."),
    ("Only take trades with Risk:Reward of at least 1:2", "If you risk RM100, aim to make RM200+. Then you can be wrong more than half the time and still make money."),
    ("Wait for the candle to close", "Signals, breakouts and patterns only count once the candle has <b>closed</b>. A candle can look great mid-day and finish ugly."),
    ("Volume confirms the move", "A breakout on high volume is far more trustworthy than one on low volume. Low-volume moves often fake out."),
    ("Check a higher timeframe", "If the daily chart says BUY but the weekly is in a clear downtrend, you're swimming against the current. Best setups agree on both."),
    ("One pattern is not enough", "Candlestick patterns work best when they line up with something else: a support/resistance level, RSI oversold/overbought, or a trend. Look for 2–3 reasons, not 1."),
    ("Don't chase (beat FOMO)", "If a stock already ran 20% today, you're late. There is always another trade. Wait for a pullback or move on."),
    ("Avoid trading right before earnings", "Quarterly results can gap the price 10–20% overnight, jumping right over your stop loss. Know the result dates of stocks you hold."),
    ("Don't average down on losers", "Adding to a losing position feels smart but multiplies the damage. Add to winners, not losers."),
    ("Keep a trading journal", "Write down every trade: why you entered, where your stop was, what happened, how you felt. After 20–30 trades you'll see your own patterns."),
    ("Practise with paper trading first", "moomoo has a paper trading account. Practise for a month with this app's signals before using real money."),
    ("Signals are probabilities, not promises", "No indicator is right every time. The goal is to win more than you lose over many trades, with small losses and bigger wins."),
]


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


def render():
    st.markdown(_css(), unsafe_allow_html=True)
    t1, t2, t3, t4 = st.tabs(["💡 Tips & Tricks", "📖 Trading Terms", "🕯️ Candlestick Patterns", "🧭 How to use this app"])

    with t1:
        st.markdown("### Rules that keep beginners alive")
        for i, (title, body) in enumerate(TIPS, 1):
            st.markdown(f'<div class="tip"><b>{i}. {title}</b><br>{body}</div>', unsafe_allow_html=True)
        st.divider()
        _position_calculator()

    with t2:
        q = st.text_input("🔍 Search a term", placeholder="e.g. RSI, stop loss, dividend, lot")
        cats = list(dict.fromkeys(g[1] for g in GLOSSARY))
        hits = [g for g in GLOSSARY if not q or q.lower() in g[0].lower() or q.lower() in g[2].lower()]
        if not hits:
            st.info("No term found. Try a different word.")
        for cat in cats:
            items = [g for g in hits if g[1] == cat]
            if not items:
                continue
            with st.expander(f"{cat} ({len(items)})", expanded=bool(q) or cat == "Basics"):
                for term, _, desc in items:
                    st.markdown(f"**{term}**: {desc}")

    with t3:
        st.markdown("### Anatomy of a candle")
        st.markdown(anatomy_svg(), unsafe_allow_html=True)
        st.markdown(
            "Each candle shows 4 prices for one period: **Open**, **High**, **Low**, **Close**. "
            "The thick part (body) is between open and close; the thin lines (wicks) show how far price went beyond. "
            "**Long wicks = rejection**: price went there but couldn't stay. "
            "In the diagrams below, the faded candles show the trend *before* the pattern, which matters: "
            "a hammer only means something after a fall.")
        flt = st.radio("Show", ["All", "Bullish", "Bearish", "Indecision"], horizontal=True, key="pat_filter")
        want = {"All": None, "Bullish": "bull", "Bearish": "bear", "Indecision": "neutral"}[flt]
        for group, size in (("Single-candle patterns", 1), ("Two-candle patterns", 2), ("Three-candle patterns", 3)):
            items = [p for p in PATTERNS if len(p[3]) == size and (want is None or p[1] == want)]
            if not items:
                continue
            st.markdown(f"#### {group}")
            cols = st.columns(3)
            for k, p in enumerate(items):
                cols[k % 3].markdown(_pattern_card(p), unsafe_allow_html=True)
        st.info("Patterns are hints, not guarantees. They work best at support/resistance, with volume, and when "
                "the next candle confirms them. The Analysis tab automatically lists any patterns found in the last 5 candles.")

    with t4:
        st.markdown("""
### How to read this app

**1. Screener tab: find candidates**
- **Signal / Score**: a summary of trend (EMAs), momentum (RSI, MACD), distance to support/resistance, breakouts and volume. +100 = everything bullish, −100 = everything bearish.
- **To support %**: how far price is above the nearest support. Small = closer to a low-risk entry.
- **R:R**: potential reward vs risk if you buy in the buy zone with the suggested stop. Prefer **2 or more**.
- Use the filters (e.g. *Buy only* + *R:R ≥ 2*) to shortlist.

**2. Analysis tab: check one stock properly**
- **Green lines** = support, **red lines** = resistance. *(3x)* means price touched that level 3 times, so the more touches, the stronger the level.
- **Green box** = buy zone, **red box** = sell / take-profit zone, **dashed orange line** = stop loss.
- **Trade plan**: the exact numbers to use. *Target 1* = sell half, *Target 2* = sell the rest.
- **Why this signal**: every reason behind the score, so you learn what the indicators are saying.
- **Candlestick patterns**: recent patterns spotted automatically. Look them up in this Learn tab.

**🎯 Picks tab: let the app shortlist for you**
- **Short term**: setups to trade over 1 to 4 weeks. A stock only appears under *Good to buy now* if the signal is a
  buy, risk:reward is at least 1.5, and price is near support or breaking out. Each card shows the buy zone, stop
  and target.
- **Long term**: stocks to hold 6 months+. Scored on the 200-day trend, 6 to 12 month momentum, volatility,
  distance from the 52-week high and, for Bursa, P/E and dividend yield.

**3. TradingView tab**: a full interactive chart where you can draw your own lines and add more indicators.

**4. Dividends tab**: which stocks pay dividends, how much (yield), how often, and the **next ex-date**.
✅ Confirmed = announced by the company; 🕒 Estimated = based on the same time last year. Use the calculator to see
how much you'd receive for a number of lots.

**5. IPO tab**: latest Bursa IPO news, plus the US IPO calendar (upcoming, recently priced and how they've performed).

**A simple beginner routine**
1. Screener → filter *Buy only* and *R:R ≥ 2*.
2. Open each one in Analysis. Is price above EMA200? Near the green buy zone? Any bullish candlestick pattern?
3. Check the **Weekly** timeframe in the sidebar too. Does it agree?
4. Use the position size calculator (Tips tab) to decide how many shares.
5. Enter with a limit order in the buy zone, set the stop loss straight away, and set a take-profit at Target 1.
6. Practise this on moomoo **paper trading** first.
""")
