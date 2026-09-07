from __future__ import annotations

import io
import time
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import requests
import yfinance as yf


DEFAULT_UNIVERSE = [
    "RELIANCE","HDFCBANK","ICICIBANK","SBIN","INFY","TCS","BHARTIARTL","ITC",
    "LT","AXISBANK","KOTAKBANK","M&M","MARUTI","SUNPHARMA","NTPC","POWERGRID",
    "TATASTEEL","HINDALCO","ADANIENT","ADANIPORTS","BAJFINANCE","TITAN","BEL",
    "HAL","RVNL","IRFC","NATIONALUM","MAHABANK","TMPV","DPSCLTD"
]

@dataclass
class FetchResult:
    symbol: str
    frame: pd.DataFrame
    source: str
    error: str = ""


def yahoo_symbol(symbol: str) -> str:
    return symbol.strip().upper().replace("&", "").replace(" ", "-") + ".NS"


def load_symbols(uploaded=None) -> list[str]:
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            col = next((c for c in df.columns if c.lower() in {"symbol", "ticker", "scrip"}), df.columns[0])
            vals = df[col].dropna().astype(str).str.upper().str.strip().tolist()
            return [x.replace(".NS", "") for x in vals if x and x != "NAN"]
        except Exception:
            pass
    return DEFAULT_UNIVERSE.copy()


def fetch_daily(symbol: str, period: str = "6mo") -> FetchResult:
    try:
        df = yf.download(yahoo_symbol(symbol), period=period, interval="1d", auto_adjust=False, progress=False, threads=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns={"Adj Close":"Adj_Close"})
        needed = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
        df = df[needed].dropna(subset=["Close"])
        if len(df) < 60:
            return FetchResult(symbol, pd.DataFrame(), "Yahoo", "Insufficient daily history")
        return FetchResult(symbol, df, "Yahoo Finance", "")
    except Exception as exc:
        return FetchResult(symbol, pd.DataFrame(), "Yahoo Finance", str(exc))


def fetch_market_snapshot() -> dict:
    out = {}
    for name, ticker in {"NIFTY50":"^NSEI", "NIFTY500":"^CRSLDX", "INDIAVIX":"^INDIAVIX"}.items():
        try:
            df = yf.download(ticker, period="3mo", interval="1d", auto_adjust=False, progress=False, threads=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            close = df["Close"].dropna()
            out[name] = {"last": float(close.iloc[-1]), "ret20": float(close.iloc[-1] / close.iloc[-21] - 1) if len(close)>21 else None,
                         "ret60": float(close.iloc[-1] / close.iloc[-61] - 1) if len(close)>61 else None}
        except Exception as exc:
            out[name] = {"error": str(exc)}
    return out
