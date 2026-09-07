from __future__ import annotations
from datetime import datetime
import requests
import pandas as pd

NSE_IPO_URL="https://www.nseindia.com/api/all-upcoming-issues?index=ipo"
HEADERS={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/128 Safari/537.36","Accept":"application/json,text/plain,*/*","Referer":"https://www.nseindia.com/market-data/all-upcoming-issues-ipo"}

def fetch_ipo_data():
    try:
        s=requests.Session(); s.headers.update(HEADERS); s.get("https://www.nseindia.com",timeout=10)
        r=s.get(NSE_IPO_URL,timeout=15); r.raise_for_status(); data=r.json()
        if isinstance(data,dict):
            for key in ("data","ipo","ipoData","records"):
                if isinstance(data.get(key),list): data=data[key]; break
        return pd.DataFrame(data if isinstance(data,list) else [])
    except Exception:
        return pd.DataFrame()

def score_ipo(df):
    if df.empty:return df
    x=df.copy()
    def txt(row): return " ".join(str(v) for v in row.values).lower()
    x["ipo_score"]=x.apply(lambda r: 50,axis=1)
    x["decision"]="WATCH"
    x["data_status"]="NSE data retrieved"
    # We intentionally do not invent valuation, promoter, GMP or financial data when NSE does not provide it.
    return x
