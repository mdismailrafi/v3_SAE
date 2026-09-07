# Indian Market Opportunity Scanner

A separate market-wide discovery app that sits before `IndianStockEngine V3.3`.

## V1 pipeline

`Universe → Daily OHLCV → Indicators → Market regime → Horizon scores → Top 20 → Breakout board → V3.3 deep analysis`

### Boards
- Intraday Top 20
- Short-term Top 20
- Long-term Top 20
- Fresh Breakouts (today / previous session)
- Risk / Watch

## Run

From the repository root:

```bash
pip install -r requirements.txt
streamlit run scanner/app.py
```

## Important design rule

This scanner discovers candidates; it does **not** replace V3.3's final decision. V3.3 remains the deep-analysis layer for technicals, fundamentals, backtesting, risk and entry planning.

## Data architecture

The V1 provider is Yahoo Finance for daily OHLCV with a provider boundary in `scanner/data.py`. That boundary is deliberate: NSE/OpenChart can be promoted to primary market-wide data later without rewriting the scoring/UI layer.

Fundamentals, news intelligence and IPO intelligence are separate modules in the planned V2/V3 pipeline. V1 does not invent those fields or pretend that price-only scores are fundamental scores.
