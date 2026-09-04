from __future__ import annotations
import os, time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import requests

try:
    import streamlit as st
except Exception:
    st = None

AV_URL = "https://www.alphavantage.co/query"
NSE_HOME = "https://www.nseindia.com"
NSE_CHART_HOME = "https://charting.nseindia.com/"
NSE_SYMBOL_URL = "https://charting.nseindia.com/v1/exchanges/symbolsDynamic"
NSE_HIST_URL = "https://charting.nseindia.com/v1/charts/symbolHistoricalData"

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
        unit = "ms" if magnitude >= 1e11 else "s" if magnitude >= 1e9 else "ms"
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
    if not path.exists():
        return pd.DataFrame()
    if time.time() - path.stat().st_mtime > max_age_days * 86400:
        return pd.DataFrame()
    try:
        raw = pd.read_csv(path)
        if "timestamp" not in raw.columns:
            return pd.DataFrame()
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], errors="coerce")
        raw = raw.dropna(subset=["timestamp"]).set_index("timestamp")
        return _normalize_ohlcv(raw)
    except Exception:
        return pd.DataFrame()


def _write_csv(df: pd.DataFrame, symbol: str, interval: str) -> Path:
    path = CSV_CACHE_DIR / f"{_safe(symbol.upper())}_{interval}.csv"
    export = df.copy().reset_index().rename(columns={df.index.name or "index": "timestamp"})
    export.to_csv(path, index=False)
    return path


class OpenChartNSEProvider:
    """Primary intraday provider: NSE charting endpoints documented/used by OpenChart.

    Alpha Vantage is deliberately absent from this class and from load_intraday().
    """

    INTERVALS = {"1m": (1, "I"), "5m": (5, "I"), "10m": (10, "I"), "15m": (15, "I"), "30m": (30, "I"), "1h": (60, "I")}

    def __init__(self, timeout: int = 20):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Origin": NSE_CHART_HOME.rstrip("/"),
            "Referer": NSE_CHART_HOME,
        })
        self._primed = False
        self.last_status: Dict[str, Any] = {}

    def _prime(self) -> None:
        if self._primed:
            return
        try:
            self.session.get(NSE_HOME, timeout=self.timeout)
            self.session.get(NSE_CHART_HOME, timeout=self.timeout)
        except requests.RequestException:
            pass
        self._primed = True

    def _post_json(self, url: str, payload: Dict[str, Any], attempts: int = 4) -> Dict[str, Any]:
        self._prime()
        last = None
        for attempt in range(1, attempts + 1):
            try:
                r = self.session.post(url, json=payload, timeout=self.timeout)
                if r.status_code in {403, 429, 500, 502, 503, 504}:
                    raise RuntimeError(f"NSE charting HTTP {r.status_code}")
                r.raise_for_status()
                data = r.json()
                if not isinstance(data, dict):
                    raise RuntimeError("Unexpected NSE charting response format")
                return data
            except Exception as exc:
                last = exc
                if attempt < attempts:
                    time.sleep(min(5.0, 0.75 * attempt))
        raise RuntimeError(f"NSE charting request failed after {attempts} attempts: {last}")

    def search(self, symbol: str) -> pd.DataFrame:
        symbol = symbol.strip().upper()
        cached = _load_pickle("nse_search", symbol, 24 * 3600)
        if cached is not None:
            return cached
        result = self._post_json(NSE_SYMBOL_URL, {"symbol": symbol, "segment": "EQ"})
        rows = result.get("data") or []
        if not result.get("status") or not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        keep = [c for c in ["symbol", "scripcode", "description", "type", "exchange"] if c in df.columns]
        df = df[keep]
        _save_pickle("nse_search", symbol, df)
        return df

    def resolve(self, symbol: str) -> Dict[str, Any]:
        s = symbol.strip().upper()
        df = self.search(s)
        if df.empty:
            raise ValueError(f"NSE could not resolve '{s}'. Use an NSE equity symbol such as RELIANCE, TCS, INFY or RVNL.")
        u = df.symbol.astype(str).str.upper()
        exact = df[u == s]
        suffixed = df[u == f"{s}-EQ"]
        row = exact.iloc[0] if not exact.empty else suffixed.iloc[0] if not suffixed.empty else df.iloc[0]
        return row.to_dict()

    @staticmethod
    def _parse_rows(rows: Any) -> pd.DataFrame:
        if not isinstance(rows, list):
            return pd.DataFrame()
        records = []
        for r in rows:
            if isinstance(r, dict):
                records.append({
                    "timestamp": r.get("time", r.get("timestamp", r.get("Timestamp", r.get("date")))),
                    "open": r.get("open", r.get("Open")), "high": r.get("high", r.get("High")),
                    "low": r.get("low", r.get("Low")), "close": r.get("close", r.get("Close")),
                    "volume": r.get("volume", r.get("Volume", 0)),
                })
            elif isinstance(r, (list, tuple)) and len(r) >= 5:
                records.append({"timestamp": r[0], "open": r[1], "high": r[2], "low": r[3], "close": r[4], "volume": r[5] if len(r) > 5 else 0})
        return pd.DataFrame(records).set_index("timestamp") if records else pd.DataFrame()

    def historical(self, symbol: str, start: datetime, end: datetime, interval: str = "5m", force_refresh: bool = False) -> pd.DataFrame:
        if interval not in self.INTERVALS:
            raise ValueError(f"Unsupported intraday interval: {interval}")
        info = self.resolve(symbol)
        minutes, chart_type = self.INTERVALS[interval]
        cache_key = f"{symbol}_{interval}_{start:%Y%m%d%H%M}_{end:%Y%m%d%H%M}"
        if not force_refresh:
            cached = _load_pickle("nse_hist", cache_key, 60)
            if cached is not None and not cached.empty:
                self.last_status = {"provider": "OpenChart/NSE charting", "source_type": "disk cache", "rows": len(cached)}
                return cached

        payload = {
            "token": str(info["scripcode"]), "fromDate": int(start.timestamp()), "toDate": int(end.timestamp()),
            "symbol": info["symbol"], "symbolType": info.get("type", "Equity"), "chartType": chart_type, "timeInterval": minutes,
        }
        result = self._post_json(NSE_HIST_URL, payload)
        rows = result.get("data") or []
        if not result.get("status") or not rows:
            raise RuntimeError("NSE charting returned no intraday candles")
        df = _normalize_ohlcv(self._parse_rows(rows))
        if df.empty:
            raise RuntimeError("NSE returned data, but it could not be normalized into OHLCV candles")
        path = _write_csv(df, symbol, interval)
        _save_pickle("nse_hist", cache_key, df)
        self.last_status = {"provider": "OpenChart/NSE charting", "source_type": "fresh NSE fetch", "rows": len(df), "csv_path": str(path)}
        return df


def load_intraday(symbol: str, interval: str = "5m", lookback_days: int = 10, force_refresh: bool = False) -> pd.DataFrame:
    """Only intraday entry point. No Alpha Vantage fallback exists here.

    Order:
      1. NSE charting/OpenChart-compatible API when refresh is requested or cache is absent.
      2. Previously fetched canonical CSV if the live request fails.
      3. Raise a data-provider error. Never manufacture candles and never call Alpha Vantage.
    """
    symbol = symbol.strip().upper()
    end = datetime.now()
    start = end - timedelta(days=max(2, int(lookback_days)))
    provider = OpenChartNSEProvider()
    live_error = None
    try:
        df = provider.historical(symbol, start, end, interval, force_refresh=force_refresh)
        if not df.empty:
            df.attrs["provider_status"] = dict(provider.last_status)
            return df
    except Exception as exc:
        live_error = exc

    cached = _read_cached_csv(symbol, interval, max_age_days=max(7, int(lookback_days) + 2))
    if not cached.empty:
        provider.last_status = {"provider": "OpenChart/NSE charting", "source_type": "automatic CSV recovery", "rows": len(cached), "live_error": str(live_error) if live_error else ""}
        cached.attrs["provider_status"] = dict(provider.last_status)
        return cached
    if live_error:
        raise RuntimeError(f"Intraday data unavailable from NSE charting, and no usable cached CSV exists. Detail: {live_error}")
    return pd.DataFrame()


class AlphaVantageProvider:
    """Only for daily/monthly/fundamental context; never used by load_intraday()."""
    def __init__(self, timeout: int = 20):
        self.key = _get_secret("ALPHAVANTAGE_API_KEY")
        self.timeout = timeout

    def _query(self, params: Dict[str, Any], cache_key: str, ttl: int = 1800) -> Dict[str, Any]:
        if not self.key:
            raise RuntimeError("ALPHAVANTAGE_API_KEY is not configured.")
        cached = _load_pickle("av_json", cache_key, ttl)
        if cached is not None:
            try: return cached.iloc[0].to_dict()
            except Exception: pass
        p = dict(params); p["apikey"] = self.key
        last = None
        for i in range(3):
            try:
                r = requests.get(AV_URL, params=p, timeout=self.timeout); r.raise_for_status(); data = r.json()
                if "Note" in data: raise RuntimeError("Alpha Vantage rate limit reached.")
                if "Error Message" in data: raise RuntimeError(str(data["Error Message"]))
                pd.DataFrame([data]).to_pickle(_cache_path("av_json", cache_key)); return data
            except Exception as exc:
                last = exc; time.sleep(i + 1)
        raise RuntimeError(str(last))

    def daily(self, symbol: str) -> pd.DataFrame:
        for candidate in [f"{symbol.upper()}.BSE", symbol.upper()]:
            try:
                data = self._query({"function":"TIME_SERIES_DAILY","symbol":candidate,"outputsize":"compact"}, f"daily_{candidate}", 1800)
                series = data.get("Time Series (Daily)") or {}
                if series:
                    df = pd.DataFrame.from_dict(series, orient="index").rename(columns={"1. open":"open","2. high":"high","3. low":"low","4. close":"close","5. volume":"volume"})
                    return _normalize_ohlcv(df.sort_index())
            except Exception: continue
        return pd.DataFrame()

    def monthly(self, symbol: str) -> pd.DataFrame:
        for candidate in [f"{symbol.upper()}.BSE", symbol.upper()]:
            try:
                data = self._query({"function":"TIME_SERIES_MONTHLY","symbol":candidate}, f"monthly_{candidate}", 24 * 3600)
                series = data.get("Monthly Time Series") or {}
                if series:
                    df = pd.DataFrame.from_dict(series, orient="index").rename(columns={"1. open":"open","2. high":"high","3. low":"low","4. close":"close","5. volume":"volume"})
                    return _normalize_ohlcv(df.sort_index())
            except Exception: continue
        return pd.DataFrame()

    def overview(self, symbol: str) -> Dict[str, Any]:
        for candidate in [f"{symbol.upper()}.BSE", symbol.upper()]:
            try:
                data = self._query({"function":"OVERVIEW","symbol":candidate}, f"overview_{candidate}", 12 * 3600)
                if data and "Symbol" in data: return data
            except Exception: continue
        return {}


def load_daily(symbol: str) -> pd.DataFrame: return AlphaVantageProvider().daily(symbol)
def load_monthly(symbol: str) -> pd.DataFrame: return AlphaVantageProvider().monthly(symbol)
def load_overview(symbol: str) -> Dict[str, Any]: return AlphaVantageProvider().overview(symbol)
