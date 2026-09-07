from __future__ import annotations
from datetime import datetime, timedelta
import requests

HEADERS={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/128 Safari/537.36","Accept":"application/json,text/plain,*/*","Referer":"https://www.nseindia.com/companies-listing/corporate-filings-announcements"}
POSITIVE=("order","contract","award","acquisition","approval","capacity","expansion","partnership","dividend","buyback","funding","investment","profit","guidance","growth")
NEGATIVE=("fraud","default","downgrade","resignation","penalty","fine","investigation","insolvency","loss","delay","pledge","litigation","warning","restatement")

def _session():
    s=requests.Session(); s.headers.update(HEADERS); s.get("https://www.nseindia.com",timeout=10); return s

def fetch_announcements(symbol: str, days: int = 3) -> list[dict]:
    try:
        end=datetime.now(); start=end-timedelta(days=days)
        url="https://www.nseindia.com/api/corporate-announcements"
        params={"index":"equities","from_date":start.strftime("%d-%m-%Y"),"to_date":end.strftime("%d-%m-%Y"),"symbol":symbol}
        r=_session().get(url,params=params,timeout=15); r.raise_for_status(); data=r.json()
        if isinstance(data,dict): data=data.get("data",[])
        return [x for x in data if str(x.get("symbol","")).upper()==symbol.upper()][:10]
    except Exception:
        return []

def classify_announcements(items: list[dict]) -> tuple[int,str,str]:
    if not items:return 0,"NONE","No recent NSE announcement retrieved"
    score=0; labels=[]
    for x in items:
        text=" ".join(str(x.get(k,"")) for k in ("subject","desc","attchmntText")).lower()
        p=sum(w in text for w in POSITIVE); n=sum(w in text for w in NEGATIVE)
        if p>n: score+=min(10,p*2); labels.append("POSITIVE")
        elif n>p: score-=min(10,n*3); labels.append("NEGATIVE")
        else: labels.append("NEUTRAL")
    score=max(-20,min(20,score)); sentiment="POSITIVE" if score>3 else "NEGATIVE" if score<-3 else "NEUTRAL"
    return score,sentiment, f"{len(items)} recent NSE filing(s)"
