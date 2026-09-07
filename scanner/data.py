from __future__ import annotations

import io
from dataclasses import dataclass
import pandas as pd
import requests
import yfinance as yf

DEFAULT_UNIVERSE = [
    "RELIANCE","HDFCBANK","ICICIBANK","SBIN","INFY","TCS","BHARTIARTL","ITC","LT",
    "AXISBANK","KOTAKBANK","M&M","MARUTI","SUNPHARMA","NTPC","POWERGRID","TATASTEEL",
    "HINDALCO","ADANIENT","ADANIPORTS","BAJFINANCE","TITAN","BEL","HAL","RVNL","IRFC",
    "NATIONALUM","MAHABANK","TMPV","DPSCLTD"
]
NSE_INDEX_URL = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20500"

@dataclass
class FetchResult:
    symbol: str
    frame: pd.DataFrame
    source: str
    error: str = ""

def yahoo_symbol(symbol: str) -> str:
    return symbol.strip().upper().replace("&", "").replace(" ", "-") + ".NS"

def fetch_nse_universe(index: str = "NIFTY 500") -> list[str]:
    """Best-effort NSE universe. Falls back cleanly when NSE blocks automated requests."""
    try:
        url = f"https://www.nseindia.com/api/equity-stockIndices?index={requests.utils.quote(index)}"
        headers = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/128 Safari/537.36", "Accept":"application/json,text/plain,*/*", "Referer":"https://www.nseindia.com/market-data/live-equity-market"}
        s = requests.Session(); s.headers.update(headers)
        s.get("https://www.nseindia.com", timeout=10)
        r = s.get(url, timeout=15); r.raise_for_status(); data = r.json()
        vals = [x.get("symbol") for x in data.get("data", []) if x.get("symbol")]
        vals = [v for v in vals if v not in {index.replace(" ",""), "NIFTY 500"}]
        return sorted(set(vals))
    except Exception:
        return []

def load_symbols(uploaded=None, max_default=500) -> list[str]:
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            col = next((c for c in df.columns if c.lower() in {"symbol","ticker","scrip"}), df.columns[0])
            vals = df[col].dropna().astype(str).str.upper().str.strip().tolist()
            return [x.replace(".NS","") for x in vals if x and x != "NAN"]
        except Exception:
            pass
    nse = fetch_nse_universe("NIFTY 500")
    return nse[:max_default] if nse else DEFAULT_UNIVERSE.copy()

def fetch_daily(symbol: str, period: str = "1y") -> FetchResult:
    try:
        df = yf.download(yahoo_symbol(symbol), period=period, interval="1d", auto_adjust=False, progress=False, threads=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df = df.rename(columns={"Adj Close":"Adj_Close"})
        needed = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
        df = df[needed].dropna(subset=["Close"])
        if len(df) < 60: return FetchResult(symbol, pd.DataFrame(), "Yahoo Finance", "Insufficient daily history")
        return FetchResult(symbol, df, "Yahoo Finance", "")
    except Exception as exc:
        return FetchResult(symbol, pd.DataFrame(), "Yahoo Finance", str(exc))

def fetch_market_snapshot() -> dict:
    out = {}
    for name,ticker in {"NIFTY50":"^NSEI","NIFTY500":"^CRSLDX","INDIAVIX":"^INDIAVIX"}.items():
        try:
            df=yf.download(ticker,period="6mo",interval="1d",auto_adjust=False,progress=False,threads=False)
            if isinstance(df.columns,pd.MultiIndex): df.columns=df.columns.get_level_values(0)
            close=df["Close"].dropna()
            out[name]={"last":float(close.iloc[-1]),"ret20":float(close.iloc[-1]/close.iloc[-21]-1) if len(close)>21 else None,"ret60":float(close.iloc[-1]/close.iloc[-61]-1) if len(close)>61 else None}
        except Exception as exc: out[name]={"error":str(exc)}
    return out

def fetch_fundamentals(symbol: str) -> dict:
    """Yahoo fundamentals are enrichment only; V3.3 remains the final fundamental authority."""
    try:
        info=yf.Ticker(yahoo_symbol(symbol)).info
        keys={"marketCap":"market_cap","trailingPE":"pe","forwardPE":"forward_pe","priceToBook":"pb","returnOnEquity":"roe","returnOnAssets":"roa","debtToEquity":"debt_equity","profitMargins":"profit_margin","operatingMargins":"operating_margin","revenueGrowth":"revenue_growth","earningsGrowth":"earnings_growth"}
        out={}
        for src,dst in keys.items():
            v=info.get(src)
            if v is not None: out[dst]=float(v)
        return out
    except Exception:
        return {}
