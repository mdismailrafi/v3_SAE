from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import requests

try:
    import streamlit as st
except Exception:
    st = None

try:
    from openchart import NSEData
except Exception:
    NSEData = None

try:
    import yfinance as yf
except Exception:
    yf = None

AV_URL = "https://www.alphavantage.co/query"
ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / ".data_cache"
CSV_CACHE_DIR = CACHE_DIR / "intraday_csv"
CACHE_DIR.mkdir(exist_ok=True)
CSV_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _get_secret(name: str) -> str:
    if st is not None:
        try:
            value = st.secrets.get(name, "")
            if value:
                return str(value).strip()
        except Exception:
            pass
    return os.getenv(name, "").strip()


def _safe(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(text))


def _cache_path(prefix: str, key: str, suffix: str = "pkl") -> Path:
    return CACHE_DIR / f"{prefix}_{_safe(key)}.{suffix}"


def _load_pickle(prefix: str, key: str, ttl_seconds: int) -> Optional[Any]:
    p = _cache_path(prefix, key)
    if not p.exists() or time.time() - p.stat().st_mtime > ttl_seconds:
        return None
    try:
        return pd.read_pickle(p)
    except Exception:
        return None


def _save_pickle(prefix: str, key: str, obj: Any) -> None:
    try:
        obj.to_pickle(_cache_path(prefix, key))
    except Exception:
        pass


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["open", "high", "low", "close", "volume"]
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)

    out = df.copy()
    lookup = {str(c).strip().lower(): c for c in out.columns}
    if "timestamp" in lookup:
        out = out.set_index(lookup["timestamp"])
    elif "date" in lookup:
        out = out.set_index(lookup["date"])

    data = pd.DataFrame(index=out.index)
    for target in cols:
        src = lookup.get(target)
        if src is not None:
            data[target] = pd.to_numeric(out[src], errors="coerce")
    for c in ["open", "high", "low", "close"]:
        if c not in data.columns:
            raise ValueError(f"Missing OHLC column: {c}")
    if "volume" not in data.columns:
        data["volume"] = 0.0

    idx = pd.Index(data.index)
    if pd.api.types.is_numeric_dtype(pd.Series(idx)):
        vals = pd.to_numeric(idx, errors="coerce")
        finite = vals[pd.notna(vals)]
        magnitude = float(pd.Series(finite).abs().median()) if len(finite) else 0
        unit = "ms" if magnitude >= 1e11 else "s"
        data.index = pd.to_datetime(vals, unit=unit, errors="coerce", utc=True)
        data.index = data.index.tz_convert("Asia/Kolkata").tz_localize(None)
    else:
        data.index = pd.to_datetime(data.index, errors="coerce")
        if getattr(data.index, "tz", None) is not None:
            data.index = data.index.tz_convert("Asia/Kolkata").tz_localize(None)

    data = data[~data.index.isna()]
    data = data[~data.index.duplicated(keep="last")].sort_index()
    data = data.dropna(subset=["open", "high", "low", "close"])
    bad = (
        (data.high < data[["open", "close", "low"]].max(axis=1)) |
        (data.low > data[["open", "close", "high"]].min(axis=1)) |
        (data[["open", "high", "low", "close"]] <= 0).any(axis=1)
    )
    return data.loc[~bad, cols]


def _read_cached_csv(symbol: str, interval: str, max_age_days: int = 7) -> pd.DataFrame:
    path = CSV_CACHE_DIR / f"{_safe(symbol.upper())}_{interval}.csv"
    if not path.exists() or time.time() - path.stat().st_mtime > max_age_days * 86400:
        return pd.DataFrame()
    try:
        return _normalize_ohlcv(pd.read_csv(path))
    except Exception:
        return pd.DataFrame()


def _write_csv(df: pd.DataFrame, symbol: str, interval: str) -> Path:
    path = CSV_CACHE_DIR / f"{_safe(symbol.upper())}_{interval}.csv"
    export = df.copy().reset_index()
    export = export.rename(columns={export.columns[0]: "timestamp"})
    export.to_csv(path, index=False)
    return path


class OpenChartNSEProvider:
    """Primary intraday provider: OpenChart -> NSE charting."""

    INTERVALS = {"1m", "5m", "10m", "15m", "30m", "1h"}

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.last_status: Dict[str, Any] = {}
        if NSEData is None:
            raise RuntimeError("OpenChart is not installed. Redeploy so requirements.txt installs it.")
        self.client = NSEData()

    def historical(self, symbol: str, start: datetime, end: datetime, interval: str = "5m", force_refresh: bool = False) -> pd.DataFrame:
        if interval not in self.INTERVALS:
            raise ValueError(f"Unsupported intraday interval: {interval}")

        s = symbol.strip().upper()
        cache_key = f"{s}_{interval}_{start:%Y%m%d%H%M}_{end:%Y%m%d%H%M}"
        if not force_refresh:
            cached = _load_pickle("nse_hist", cache_key, 60)
            if cached is not None and not cached.empty:
                self.last_status = {"provider": "OpenChart", "source_type": "disk cache", "rows": len(cached)}
                return cached

        # The package's documented historical() path performs its own symbol
        # resolution and constructs the NSE charting request. This is more
        # robust than manually supplying the token to historical_direct().
        try:
            raw = self.client.historical(s, "EQ", start, end, interval)
        except Exception as exc:
            raise RuntimeError(f"OpenChart/NSE request failed: {exc}") from exc

        df = _normalize_ohlcv(pd.DataFrame(raw) if raw is not None else pd.DataFrame())
        if df.empty:
            raise RuntimeError("OpenChart/NSE returned no intraday candles")

        path = _write_csv(df, s, interval)
        _save_pickle("nse_hist", cache_key, df)
        self.last_status = {
            "provider": "OpenChart/NSE",
            "source_type": "fresh NSE fetch",
            "rows": len(df),
            "csv_path": str(path),
        }
        return df


class YahooIntradayProvider:
    """Secondary recovery provider when NSE charting is unavailable.

    Yahoo is not used for fundamentals/daily analysis here. It is only a
    recovery transport for intraday OHLCV, and its use is explicitly surfaced
    in the UI so the user knows the source changed.
    """

    INTERVAL_MAP = {"1m": "1m", "5m": "5m", "10m": "5m", "15m": "15m", "30m": "30m", "1h": "60m"}

    def historical(self, symbol: str, start: datetime, end: datetime, interval: str) -> pd.DataFrame:
        if yf is None:
            raise RuntimeError("yfinance is not installed")
        yahoo_interval = self.INTERVAL_MAP[interval]
        period_days = max(2, (end.date() - start.date()).days + 1)
        # Yahoo's intraday retention is limited; request only what can be
        # reasonably returned, then let normalization/caching handle it.
        if yahoo_interval == "1m":
            period = f"{min(period_days, 7)}d"
        else:
            period = f"{min(period_days, 60)}d"
        ticker = yf.Ticker(f"{symbol.upper()}.NS")
        raw = ticker.history(period=period, interval=yahoo_interval, auto_adjust=False, prepost=False, timeout=20)
        if raw is None or raw.empty:
            raise RuntimeError("Yahoo Finance returned no intraday candles")
        raw = raw.reset_index()
        df = _normalize_ohlcv(raw)
        if df.empty:
            raise RuntimeError("Yahoo Finance returned no usable intraday candles")
        if interval in {"10m", "1h"}:
            rule = "10min" if interval == "10m" else "60min"
            df = df.resample(rule).agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
        return df


def load_intraday(symbol: str, interval: str = "5m", lookback_days: int = 5, force_refresh: bool = False) -> pd.DataFrame:
    """Intraday pipeline: OpenChart/NSE -> Yahoo recovery -> canonical CSV.

    Alpha Vantage is deliberately never called from this function.
    """
    symbol = symbol.strip().upper()
    lookback = max(2, int(lookback_days))
    end = datetime.now()
    start = end - timedelta(days=lookback)
    errors = []

    try:
        provider = OpenChartNSEProvider()
        df = provider.historical(symbol, start, end, interval, force_refresh=force_refresh)
        if not df.empty:
            df.attrs["provider_status"] = dict(provider.last_status)
            return df
    except Exception as exc:
        errors.append(f"OpenChart/NSE: {exc}")

    try:
        provider = YahooIntradayProvider()
        df = provider.historical(symbol, start, end, interval)
        if not df.empty:
            path = _write_csv(df, symbol, interval)
            status = {
                "provider": "Yahoo Finance",
                "source_type": "secondary intraday recovery",
                "rows": len(df),
                "csv_path": str(path),
                "live_error": " | ".join(errors),
            }
            df.attrs["provider_status"] = status
            return df
    except Exception as exc:
        errors.append(f"Yahoo Finance: {exc}")

    cached = _read_cached_csv(symbol, interval, max_age_days=max(7, lookback + 2))
    if not cached.empty:
        status = {
            "provider": "Cached canonical CSV",
            "source_type": "automatic CSV recovery",
            "rows": len(cached),
            "live_error": " | ".join(errors),
        }
        cached.attrs["provider_status"] = status
        return cached

    raise RuntimeError(
        "Intraday data unavailable from OpenChart/NSE and recovery sources. "
        f"Detail: {' | '.join(errors)}"
    )


class AlphaVantageProvider:
    """Only for daily/monthly/fundamental context; never used by intraday."""

    def __init__(self, timeout: int = 20):
        self.key = _get_secret("ALPHAVANTAGE_API_KEY")
        self.timeout = timeout

    def _query(self, params: Dict[str, Any], cache_key: str, ttl: int = 1800) -> Dict[str, Any]:
        if not self.key:
            raise RuntimeError("ALPHAVANTAGE_API_KEY is not configured.")
        cached = _load_pickle("av_json", cache_key, ttl)
        if cached is not None:
            try:
                return cached.iloc[0].to_dict()
            except Exception:
                pass
        p = dict(params)
        p["apikey"] = self.key
        last = None
        for i in range(3):
            try:
                r = requests.get(AV_URL, params=p, timeout=self.timeout)
                r.raise_for_status()
                data = r.json()
                if "Note" in data:
                    raise RuntimeError("Alpha Vantage rate limit reached.")
                if "Error Message" in data:
                    raise RuntimeError(str(data["Error Message"]))
                pd.DataFrame([data]).to_pickle(_cache_path("av_json", cache_key))
                return data
            except Exception as exc:
                last = exc
                time.sleep(i + 1)
        raise RuntimeError(str(last))

    def daily(self, symbol: str) -> pd.DataFrame:
        for candidate in [f"{symbol.upper()}.BSE", symbol.upper()]:
            try:
                data = self._query({"function": "TIME_SERIES_DAILY", "symbol": candidate, "outputsize": "compact"}, f"daily_{candidate}", 1800)
                series = data.get("Time Series (Daily)") or {}
                if series:
                    df = pd.DataFrame.from_dict(series, orient="index").rename(columns={"1. open": "open", "2. high": "high", "3. low": "low", "4. close": "close", "5. volume": "volume"})
                    return _normalize_ohlcv(df.sort_index())
            except Exception:
                continue
        return pd.DataFrame()

    def monthly(self, symbol: str) -> pd.DataFrame:
        for candidate in [f"{symbol.upper()}.BSE", symbol.upper()]:
            try:
                data = self._query({"function": "TIME_SERIES_MONTHLY", "symbol": candidate}, f"monthly_{candidate}", 24 * 3600)
                series = data.get("Monthly Time Series") or {}
                if series:
                    df = pd.DataFrame.from_dict(series, orient="index").rename(columns={"1. open": "open", "2. high": "high", "3. low": "low", "4. close": "close", "5. volume": "volume"})
                    return _normalize_ohlcv(df.sort_index())
            except Exception:
                continue
        return pd.DataFrame()

    def overview(self, symbol: str) -> Dict[str, Any]:
        for candidate in [f"{symbol.upper()}.BSE", symbol.upper()]:
            try:
                data = self._query({"function": "OVERVIEW", "symbol": candidate}, f"overview_{candidate}", 12 * 3600)
                if data and "Symbol" in data:
                    return data
            except Exception:
                continue
        return {}


def load_daily(symbol: str) -> pd.DataFrame:
    return AlphaVantageProvider().daily(symbol)


def load_monthly(symbol: str) -> pd.DataFrame:
    return AlphaVantageProvider().monthly(symbol)


def load_overview(symbol: str) -> Dict[str, Any]:
    return AlphaVantageProvider().overview(symbol)
