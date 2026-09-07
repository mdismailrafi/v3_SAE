from __future__ import annotations

import math
import numpy as np
import pandas as pd


def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def rsi(s, n=14):
    d=s.diff(); up=d.clip(lower=0); dn=-d.clip(upper=0)
    rs=up.ewm(alpha=1/n, adjust=False).mean()/dn.ewm(alpha=1/n, adjust=False).mean().replace(0,np.nan)
    return 100-(100/(1+rs))

def atr(df, n=14):
    pc=df.Close.shift(1)
    tr=pd.concat([df.High-df.Low,(df.High-pc).abs(),(df.Low-pc).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n,adjust=False).mean()

def enrich(df):
    x=df.copy()
    x["EMA20"]=ema(x.Close,20); x["EMA50"]=ema(x.Close,50); x["EMA200"]=ema(x.Close,200)
    x["RSI"]=rsi(x.Close); x["ATR"]=atr(x); x["ATR_pct"]=x.ATR/x.Close
    x["VOL20"]=x.Volume.rolling(20).mean(); x["RVOL"]=x.Volume/x.VOL20
    x["HH20"]=x.High.rolling(20).max().shift(1); x["HH55"]=x.High.rolling(55).max().shift(1)
    x["LL20"]=x.Low.rolling(20).min().shift(1)
    x["RET5"]=x.Close.pct_change(5); x["RET20"]=x.Close.pct_change(20); x["RET60"]=x.Close.pct_change(60)
    x["RANGE_POS20"]=(x.Close-x.LL20)/(x.HH20-x.LL20).replace(0,np.nan)
    x["DOLLAR_VOL"]=x.Close*x.Volume
    return x


def market_regime(snapshot):
    n=snapshot.get("NIFTY50",{})
    r20=n.get("ret20")
    if r20 is None: return "UNKNOWN", 50
    if r20 > .03: return "RISK-ON", 75
    if r20 < -.03: return "RISK-OFF", 25
    return "NEUTRAL", 50


def score_row(x, regime_score=50):
    last=x.iloc[-1]; prev=x.iloc[-2]
    trend=0
    trend += 20 if last.Close>last.EMA20 else 0
    trend += 15 if last.EMA20>last.EMA50 else 0
    trend += 15 if last.EMA50>last.EMA200 else 0
    trend += 10 if last.RET20>0 else 0
    momentum=0
    momentum += min(20,max(0,(last.RSI-45)*0.8))
    momentum += min(20,max(0,last.RVOL*10))
    breakout = bool(last.Close>last.HH20) if pd.notna(last.HH20) else False
    prev_breakout = bool(prev.Close>prev.HH20) if pd.notna(prev.HH20) else False
    fresh_breakout = breakout or (prev_breakout and last.Close>=prev.Close*0.985)
    quality = min(100, max(0, trend+momentum))
    intraday = min(100, .60*quality + .20*regime_score + .20*(100 if fresh_breakout else 50))
    short = min(100, .45*quality + .20*regime_score + .25*(100 if fresh_breakout else 50) + .10*min(100,max(0,last.RANGE_POS20*100)))
    long = min(100, .45*(100 if last.Close>last.EMA200 else 35) + .25*(100 if last.EMA50>last.EMA200 else 35) + .20*regime_score + .10*min(100,max(0,last.RSI)))
    return {
        "price":float(last.Close),"rsi":float(last.RSI),"rvol":float(last.RVOL),"ret5":float(last.RET5),"ret20":float(last.RET20),
        "trend_score":round(quality,1),"intraday_score":round(intraday,1),"short_score":round(short,1),"long_score":round(long,1),
        "breakout_today":breakout,"breakout_previous":prev_breakout,"fresh_breakout":fresh_breakout,
        "above_20ema":bool(last.Close>last.EMA20),"above_50ema":bool(last.Close>last.EMA50),"above_200ema":bool(last.Close>last.EMA200),
        "atr_pct":float(last.ATR_pct),"risk_flag": "HIGH" if last.ATR_pct>.06 else ("MEDIUM" if last.ATR_pct>.035 else "NORMAL")
    }
