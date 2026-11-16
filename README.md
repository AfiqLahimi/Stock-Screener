# Stock Screener & Analysis

Live stock screener for **Bursa Malaysia** and **US** stocks, built with Streamlit.

- **Screener**: Buy/Sell signals with a score from −100 to +100, plus support, resistance and risk:reward
- **Analysis & Forecast**: chart with support/resistance, buy/sell zones, stop loss, trend forecast and candlestick pattern detection
- **Dividends**: yield, history, next ex-dates (confirmed or estimated) and an income calculator
- **IPO**: Bursa IPO news and the US IPO calendar from Nasdaq
- **TradingView**: interactive TradingView chart
- **Learn**: tips, a trading glossary and candlestick patterns

Data: Yahoo Finance (via `yfinance`), the Nasdaq IPO calendar and Google News.

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

> Not investment advice. Signals and forecasts are generated automatically from technical indicators.
