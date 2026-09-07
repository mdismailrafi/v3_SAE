from __future__ import annotations
import numpy as np
import pandas as pd

def ema(s,n): return s.ewm(span=n,adjust=False).mean()
def rsi(s,n=14):
    d=s.diff(); up=d.clip(lower=0); dn=-d.clip(upper=0)
    rs=up.ewm(alpha=1/n,adjust=False).mean()/dn.ewm(alpha=1/n,adjust=False).mean().replace(0,np.nan)
    return 100-(100/(1+rs))
def atr(df,n=14):
    pc=df.Close.shift(1); tr=pd.concat([df.High-df.Low,(df.High-pc).abs(),(df.Low-pc).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n,adjust=False).mean()
def enrich(df):
    x=df.copy(); x["EMA20"]=ema(x.Close,20); x["EMA50"]=ema(x.Close,50); x["EMA200"]=ema(x.Close,200)
    x["RSI"]=rsi(x.Close); x["ATR"]=atr(x); x["ATR_pct"]=x.ATR/x.Close; x["VOL20"]=x.Volume.rolling(20).mean(); x["RVOL"]=x.Volume/x.VOL20
    x["HH20"]=x.High.rolling(20).max().shift(1); x["HH55"]=x.High.rolling(55).max().shift(1); x["LL20"]=x.Low.rolling(20).min().shift(1)
    x["RET5"]=x.Close.pct_change(5); x["RET20"]=x.Close.pct_change(20); x["RET60"]=x.Close.pct_change(60)
    x["RANGE_POS20"]=(x.Close-x.LL20)/(x.HH20-x.LL20).replace(0,np.nan); x["DOLLAR_VOL"]=x.Close*x.Volume
    return x

def market_regime(snapshot):
    n=snapshot.get("NIFTY50",{}); r20=n.get("ret20"); r60=n.get("ret60")
    if r20 is None:return "UNKNOWN",50
    score=50
    score += 15 if r20>0 else -15
    score += 15 if r60 and r60>0 else -15
    vix=snapshot.get("INDIAVIX",{}).get("last")
    if vix is not None: score += 10 if vix<15 else (-10 if vix>22 else 0)
    score=max(0,min(100,score)); return ("RISK-ON" if score>=65 else "RISK-OFF" if score<=35 else "NEUTRAL"),score

def _fundamental_score(f):
    if not f:return 50
    s=50
    if f.get("roe") is not None:s += 15 if f["roe"]>.15 else (-10 if f["roe"]<0 else 0)
    if f.get("revenue_growth") is not None:s += 10 if f["revenue_growth"]>0.10 else (-5 if f["revenue_growth"]<0 else 0)
    if f.get("earnings_growth") is not None:s += 10 if f["earnings_growth"]>0.10 else (-5 if f["earnings_growth"]<0 else 0)
    if f.get("debt_equity") is not None:s += 10 if f["debt_equity"]<80 else (-10 if f["debt_equity"]>150 else 0)
    return max(0,min(100,s))

def score_row(x,regime_score=50,fundamentals=None):
    last=x.iloc[-1]; prev=x.iloc[-2]
    trend=(20 if last.Close>last.EMA20 else 0)+(15 if last.EMA20>last.EMA50 else 0)+(15 if last.EMA50>last.EMA200 else 0)+(10 if last.RET20>0 else 0)
    momentum=min(20,max(0,(last.RSI-45)*.8))+min(20,max(0,last.RVOL*10)); quality=min(100,trend+momentum)
    breakout=bool(last.Close>last.HH20) if pd.notna(last.HH20) else False; prev_breakout=bool(prev.Close>prev.HH20) if pd.notna(prev.HH20) else False
    fresh=breakout or (prev_breakout and last.Close>=prev.Close*.985); fscore=_fundamental_score(fundamentals or {})
    intraday=min(100,.60*quality+.20*regime_score+.20*(100 if fresh else 50))
    short=min(100,.40*quality+.20*regime_score+.20*(100 if fresh else 50)+.20*fscore)
    long=min(100,.30*(100 if last.Close>last.EMA200 else 35)+.15*(100 if last.EMA50>last.EMA200 else 35)+.15*regime_score+.40*fscore)
    return {"price":float(last.Close),"rsi":float(last.RSI),"rvol":float(last.RVOL),"ret5":float(last.RET5),"ret20":float(last.RET20),"trend_score":round(quality,1),"fundamental_score":round(fscore,1),"intraday_score":round(intraday,1),"short_score":round(short,1),"long_score":round(long,1),"breakout_today":breakout,"breakout_previous":prev_breakout,"fresh_breakout":fresh,"above_20ema":bool(last.Close>last.EMA20),"above_50ema":bool(last.Close>last.EMA50),"above_200ema":bool(last.Close>last.EMA200),"atr_pct":float(last.ATR_pct),"risk_flag":"HIGH" if last.ATR_pct>.06 else ("MEDIUM" if last.ATR_pct>.035 else "NORMAL")}
