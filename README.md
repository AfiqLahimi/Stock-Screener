# Stock Screener & Analysis

Live stock screener for **Bursa Malaysia** and **US** stocks, built with Streamlit.

- **Screener**: Buy/Sell signals with a score from −100 to +100, plus support, resistance and risk:reward
- **Picks**: short-term (days to weeks) and long-term (6 months+) buy/sell ideas, ranked
- **Heatmap**: market heatmap by sector, coloured by daily, weekly, monthly or 3-month change, or signal score
- **Analysis & Forecast**: chart with switchable panels (volume, RSI, MACD, ADX), plus sections for a written summary, signal audit, trade plan with ATR stops, position sizing and Bursa fees, paper-trading journal, multi-timeframe analysis (15m–Monthly), candlestick context, confirmation and history, walk-forward forecast accuracy, levels, and a data-quality report
- **Analysis & Forecast (chart)**: chart with support/resistance, buy/sell zones, stop loss, trend forecast and candlestick pattern detection
- **Dividends**: yield, history, next ex-dates (confirmed or estimated) and an income calculator
- **IPO**: Bursa IPO news and the US IPO calendar from Nasdaq
- **TradingView**: interactive TradingView chart
- **Learn**: tips, a trading glossary and candlestick patterns

Data: Yahoo Finance (via `yfinance`), the Nasdaq IPO calendar and Google News. All indicators are calculated locally; values that need more history than is available are shown as N/A.

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

> Not investment advice. Signals and forecasts are generated automatically from technical indicators.
