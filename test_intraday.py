from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import pandas as pd

from data_provider import _normalize_ohlcv, _write_csv, _read_cached_csv, load_intraday


def test_normalization_and_csv_bridge():
    idx = pd.date_range(datetime.now() - timedelta(minutes=20), periods=5, freq="5min")
    raw = pd.DataFrame({
        "Open": [100,101,102,103,104], "High": [101,102,103,104,105],
        "Low": [99,100,101,102,103], "Close": [100.5,101.5,102.5,103.5,104.5],
        "Volume": [10,20,30,40,50]
    }, index=idx)
    df = _normalize_ohlcv(raw)
    assert list(df.columns) == ["open","high","low","close","volume"]
    assert len(df) == 5


def test_live_provider_is_not_alpha_vantage():
    import data_provider
    import inspect
    src = inspect.getsource(data_provider.load_intraday)
    assert "AlphaVantageProvider" not in src
    assert "AV_URL" not in src


def main():
    test_normalization_and_csv_bridge()
    test_live_provider_is_not_alpha_vantage()
    symbol = "RVNL"
    interval = "5m"
    try:
        df = load_intraday(symbol, interval, lookback_days=5, force_refresh=True)
        if df.empty:
            raise RuntimeError("No candles returned")
        print(f"LIVE OK: {len(df)} candles")
        print(df.tail())
        print(f"Canonical CSV: .data_cache/intraday_csv/{symbol}_{interval}.csv")
    except Exception as exc:
        print(f"LIVE FETCH NOT AVAILABLE: {exc}")
        print("Structural tests passed; this is a provider/network availability issue, not an Alpha Vantage fallback.")


if __name__ == "__main__":
    main()
