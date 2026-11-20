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
  ringgit crosses, commodities, US inflation / unemployment / Fed funds.
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
- **Compare**: 2–4 stocks side by side (performance, risk, scores, key ratios, correlation).
- **Picks**: short-term and long-term ideas, ranked.

**Trade**
- **Terminal**: three-panel workspace: watchlist with categories and favourites, chart with ranges, chart
  types, overlays, indicator panes and drawing tools, and an overview with technicals, scenarios and
  fundamentals. Layout and indicators are adjustable.
- **Multi-Chart**: up to six charts at once.
- **Paper Trade**: virtual MYR and USD accounts with market/limit/stop orders, stop-loss/take-profit,
  Bursa board lots and fees, positions, history, stats and an equity curve.
- **Risk**: volatility, beta, 1-day Value at Risk, drawdown, risk contribution, stop distances, sector and
  currency concentration and stress tests for your paper portfolio or watchlist.
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
| FRED (St. Louis Fed) CSV | US CPI, unemployment, Fed funds, 10y–2y spread | keyless |
| Nasdaq IPO calendar, Google News RSS | IPOs, news | headlines only |

Values that need more history than exists (for example EMA 200 on a new listing) are shown as N/A, never
estimated. Signals use completed candles only (setting in the sidebar), and backtests are walk-forward with no
look-ahead. Crude palm oil futures and a free economic-release calendar are not available, so they are not
shown. Your watchlist, favourites, saved screens, scoring weights and paper account are saved on the app server
and in your browser; download a paper-account backup now and then.

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
