# Stock Screener & Analysis

A personal research terminal for **Bursa Malaysia** and **US** stocks, built with Streamlit. Live prices, no
CSV uploads, no paid APIs, no broker connection: everything is calculated in the app from free public data.

> Not investment advice. Signals, scores, scenarios and forecasts are generated automatically from market data
> and describe current evidence, not guaranteed outcomes. Paper trading only: the app never places real orders.

## Pages

**Markets**
- **Dashboard**: index / FX / commodity strip with sparklines, index chart with EMA 20/50/200 and trend table,
  market pulse (breadth, % of stocks above EMA 50, signal balance, index trend), Bursa and US top movers,
  Bursa sector performance, market news, your watchlist and six lists of technical set-ups. Click any row to
  open the stock.
- **Macro**: market regime (trend × volatility) for KLCI and S&P 500, US Treasury yields and curve, Malaysia OPR,
  ringgit crosses, and 16 US economic series from FRED (inflation, jobs, growth, rates and credit, sentiment).
- **Commodities**: about 20 energy, metal and farm futures with 1D–1Y changes and distance from the 5-year high, a
  chart, and which Bursa companies a price rise typically helps or hurts, with the measured correlation as evidence.
- **Crypto**: board of 25 major coins (price in US$, 24h/7d/30d/1Y, RSI, signal) and, for any coin, the same chart,
  trade plan and support/resistance as for stocks, with full technical analysis one click away. Technical only.
- **Heatmap**: market heatmap by sector.

**Research**
- **Screener**: 12 one-click screens plus a custom builder (signal, score, price vs EMA 20/50/200, EMA stack,
  EMA 9/20, 20/50, 50/200 crosses, ADX/DMI, RSI, MACD, stochastic, returns, volatility, relative volume,
  liquidity, distance to support/resistance, risk:reward, 20-bar and 52-week breakouts), combined with
  *match all* or *match any*; save your own screens.
- **Analysis & Forecast**: full single-stock analysis.
  - *Summary*, *Audit*, *Trade plan* (ATR stops, position sizing, Bursa fees), *Timeframes* (15m to monthly),
    *Volume*, *ADX*, *Patterns*, *Forecast* (with walk-forward accuracy), *Levels*, *Data* quality.
  - *Scorecard*: Technical score (trend, momentum, volume, volatility, price action), Fundamental score,
    overall assessment, confidence and data completeness, with the evidence behind every score, configurable
    weights, and a conclusion (scenarios, entry zones, stop, targets with R:R, invalidation).
  - *Indicators*: EMA 9/20/50/100/200, SMA 20/50/100/200, RSI, MACD, stochastic, Stoch RSI, ADX/DMI, ROC, CCI,
    Bollinger, Keltner, squeeze, ATR, historical volatility, volume SMA, RVOL, OBV, VWAP, A/D, MFI, swing
    structure, trendlines, divergence, Fibonacci retracement and extension.
  - *Fundamentals*: valuation, profitability (incl. ROIC), financial health, growth, cash flow and dividends,
    with plain-language explanations; banks are judged without debt/liquidity ratios.
  - *News & analysts*: company headlines, consensus rating, price targets and estimates.
  - *Financials*: annual and quarterly income statement, balance sheet and cash flow, plus sector peers.
  - *Valuation*: a two-stage discounted-cash-flow model with adjustable assumptions and a sensitivity table.
  - *Ownership*: major and institutional holders, insider trades, and (US stocks) short interest and options
    activity.
  - *Notes*: your own research notes on the stock.
- **Compare**: 2–4 stocks side by side (performance, risk, scores, key ratios, correlation).
- **Picks**: short-term and long-term ideas, ranked.
- **Notes**: all your research notes in one place, searchable.

**Portfolio & trade**
- **My Portfolio**: the shares you actually own, entered by hand or pasted as a list (no broker connection),
  priced live with profit/loss, today's change, weights, estimated dividends received and allocation charts.
- **Terminal**: three-panel workspace: watchlist with categories and favourites, chart with ranges, chart
  types, overlays, indicator panes and drawing tools, and an overview with technicals, scenarios and
  fundamentals. Layout and indicators are adjustable.
- **Multi-Chart**: up to six charts at once.
- **Paper Trade**: virtual MYR and USD accounts with market/limit/stop orders, stop-loss/take-profit,
  Bursa board lots and fees, positions, history, stats and an equity curve.
- **Risk**: volatility, beta, 1-day Value at Risk, drawdown, risk contribution, stop distances, sector and
  currency concentration and stress tests for your real portfolio, paper portfolio or watchlist.
- **TradingView**: the TradingView chart widget.

**Income & events**
- **Calendar**: earnings and dividend dates for your watchlist and positions, a month view and daily
  "watchlist intelligence" alerts.
- **Dividends**: yields, history, next ex-dates and an income calculator.
- **IPO**: Bursa IPO news and the US IPO calendar.

**Learn**: glossary, tips and candlestick patterns.

## Data sources and limits

| Source | Used for | Notes |
|---|---|---|
| Yahoo Finance (`yfinance`) | prices, the full Bursa list, company data, statements, dividends, calendars, US movers | Bursa prices are ~15 minutes delayed; US near real time. Fundamentals for small Bursa companies are often incomplete (shown as N/A). |
| Bank Negara Malaysia open API | Overnight Policy Rate | keyless |
| FRED (St. Louis Fed) CSV | 16 US economic series | keyless |
| Nasdaq IPO calendar, Google News RSS | IPOs, news | headlines only |

Values that need more history than exists (for example EMA 200 on a new listing) are shown as N/A, never
estimated. Signals use completed candles only (setting in the sidebar), and backtests are walk-forward with no
look-ahead. Crude palm oil futures and a free economic-release calendar are not available, so they are not
shown. Short interest, insider filings and options exist on Yahoo for US stocks only. Your watchlist, favourites,
saved screens, scoring weights, notes, portfolio and paper account are saved on the app server and in your
browser; download a paper-account backup now and then.

**Light or dark**: the app follows your device setting; pick Light or Dark in the ⋮ menu → Settings to fix it.

## Run locally

Mac: double-click `run_mac.command`. Windows: double-click `run_windows.bat`. Or:

```
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`. Set `DEMO_DATA=1` to run on synthetic data without internet.

## Project structure

```
app.py            entry point: page setup, shared sidebar, navigation
ui.py             design system (theme CSS, panels, top bar, ticker tape), page switching, error guard
core.py           stock lists, data loading, core indicators, signal score, trade plan, forecast, charts, scan
indicators.py     full indicator library (warm-up aware, no look-ahead)
engine.py         explainable technical scoring, conclusion, horizons, opportunities
fundamentals.py   fundamental metrics and scores
research.py       risk statistics, events, macro data
stockinfo.py      analysts, ownership, insiders, short interest, options, quarterly statements, peers, DCF
ta_plus.py        fees, position sizing, multi-timeframe, patterns, walk-forward test
terminal.py, paper.py, multichart.py, picks.py, extras.py, learn.py, prefs.py
views/            one module per page
tests/            offline tests (indicator accuracy, statistics, every page, awkward data)
```

## Tests

```
pip install -r requirements-dev.txt
python -m pytest -q tests
```

Tests run offline on synthetic data: indicators are checked against the `ta` library, risk statistics against
known answers, and every page is rendered on PC and phone layouts, including with a suspended (flat-price)
stock, a fresh listing, zero-volume indices and time-zone-aware intraday candles.
