# Indian Stock Analysis Engine — V3.3 FINAL

## Purpose
V3.3 fixes the specific failure in V3.1/V3.2: **Intraday analysis can no longer fall through to Alpha Vantage.**

### Intraday data flow
```text
User enters NSE symbol
        ↓
NSE charting / OpenChart-compatible API
        ↓
NSE symbol resolution (EQ)
        ↓
OHLCV normalization + integrity checks
        ↓
Canonical CSV written automatically
        ↓
V3 intraday indicators / scoring / validation
```

If NSE is temporarily unavailable:
```text
NSE live request fails
        ↓
Previously fetched canonical CSV is checked automatically
        ↓
If usable → analyse cached candles
If not usable → DATA UNAVAILABLE
```

**There is no Alpha Vantage fallback in `load_intraday()`.**

## Automatic CSV
You do NOT need to find or download an OHLCV CSV manually.

After a successful NSE fetch, the engine writes:
```text
.data_cache/intraday_csv/RVNL_5m.csv
```

The CSV is a local canonical recovery layer, not the primary source. It lets the engine survive temporary NSE rate limits or connectivity failures after at least one successful fetch.

## Source verification
OpenChart's current documentation describes it as a Python library for NSE intraday/EOD historical OHLCV, with equity symbols such as `RELIANCE-EQ`, and timeframes including 1m, 5m, 10m, 15m, 30m and 1h. The project is unofficial and uses NSE's publicly available charting APIs.

This build uses the NSE charting endpoints directly rather than requiring the OpenChart package at runtime.

## V3.3 changes
- Intraday path hard-isolated from Alpha Vantage.
- Automatic canonical CSV creation after successful NSE fetch.
- Automatic CSV recovery if live NSE fetch fails.
- Absolute cache path based on the application folder, so Streamlit working-directory changes do not break the cache.
- Stronger OHLC integrity validation.
- NSE chart home is primed before API requests.
- Clear provider/source status.
- No trading verdict is produced when there is no usable market data.
- Existing short-term and long-term Alpha Vantage path remains separate.

## Install
```bash
pip install -r requirements.txt
streamlit run app.py
```

For daily/monthly/fundamental analysis only, configure `ALPHAVANTAGE_API_KEY` in Streamlit Secrets or the environment.

## Deploy
Upload:
- `app.py`
- `engine.py`
- `data_provider.py`
- `requirements.txt`
- `README.md`
- `test_intraday.py`

Do not commit `.data_cache/` or `__pycache__/`.

## Important limitation
NSE charting access is unofficial and can be rate-limited or changed. V3.3 handles that as a provider problem and may recover from an existing local CSV. It never disguises missing data as a trading signal.

## Quick verification
```bash
python test_intraday.py
```
The structural tests verify that the intraday entry point contains no Alpha Vantage fallback. A live fetch additionally verifies NSE connectivity.
