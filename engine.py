from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev = df["close"].shift(1)
    tr = pd.concat([(df["high"] - df["low"]), (df["high"] - prev).abs(), (df["low"] - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def add_indicators(df: pd.DataFrame, intraday: bool = False) -> pd.DataFrame:
    x = df.copy().sort_index()
    c, h, l, v = x["close"], x["high"], x["low"], x["volume"]
    x["ema9"] = c.ewm(span=9, adjust=False).mean()
    x["ema20"] = c.ewm(span=20, adjust=False).mean()
    x["ema50"] = c.ewm(span=50, adjust=False).mean()
    x["ema200"] = c.ewm(span=200, adjust=False).mean() if len(x) >= 200 else np.nan
    x["rsi14"] = _rsi(c)
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    x["macd"] = ema12 - ema26
    x["macd_signal"] = x["macd"].ewm(span=9, adjust=False).mean()
    x["macd_hist"] = x["macd"] - x["macd_signal"]
    x["atr14"] = _atr(x)
    x["bb_mid"] = c.rolling(20, min_periods=20).mean()
    std = c.rolling(20, min_periods=20).std()
    x["bb_upper"] = x["bb_mid"] + 2 * std
    x["bb_lower"] = x["bb_mid"] - 2 * std
    x["ret1"] = c.pct_change()
    x["vol_ma20"] = v.rolling(20, min_periods=5).mean()
    x["rel_volume"] = v / x["vol_ma20"].replace(0, np.nan)
    if intraday:
        tp = (h + l + c) / 3
        pv = tp * v
        day = pd.Series(x.index.date, index=x.index)
        x["vwap"] = pv.groupby(day).cumsum() / v.groupby(day).cumsum().replace(0, np.nan)
        x["day_high"] = h.groupby(day).cummax()
        x["day_low"] = l.groupby(day).cummin()
        x["prev_day_high"] = h.groupby(day).transform("max").shift(1)
        x["prev_day_low"] = l.groupby(day).transform("min").shift(1)
        x["range_pct"] = (h - l) / c.replace(0, np.nan)
    return x


def data_quality(df: pd.DataFrame, interval: str, intraday: bool = False) -> Dict[str, object]:
    n = len(df)
    req = {"1m": 300, "5m": 120, "10m": 80, "15m": 60, "30m": 50, "1h": 40}.get(interval, 50) if intraday else 50
    missing = int(df[["open", "high", "low", "close", "volume"]].isna().sum().sum()) if not df.empty else 999
    dup = int(df.index.duplicated().sum()) if not df.empty else 0
    quality = "A" if n >= req and missing == 0 and dup == 0 else "B" if n >= max(30, req // 2) and missing == 0 else "C" if n else "D"
    return {"rows": n, "required": req, "missing_values": missing, "duplicate_timestamps": dup, "start": str(df.index.min()) if n else "—", "end": str(df.index.max()) if n else "—", "grade": quality}


def technical_snapshot(x: pd.DataFrame, intraday: bool = False) -> Dict[str, object]:
    if x.empty:
        return {"score": np.nan, "confidence": "None", "items": {}}
    last = x.iloc[-1]
    score = weight = 0.0
    items = {}

    def add(name, value, pts, w=1):
        nonlocal score, weight
        items[name] = value
        if pts is not None:
            score += pts * w
            weight += w

    c = last["close"]
    if pd.notna(last.get("ema20")): add("Price vs EMA20", "Above" if c > last.ema20 else "Below", 1 if c > last.ema20 else -1)
    if pd.notna(last.get("ema50")): add("Price vs EMA50", "Above" if c > last.ema50 else "Below", 1 if c > last.ema50 else -1)
    else: items["Price vs EMA50"] = "Unavailable"
    if pd.notna(last.get("ema200")): add("Price vs EMA200", "Above" if c > last.ema200 else "Below", 1 if c > last.ema200 else -1)
    else: items["EMA200"] = "Insufficient history (<200 bars)"
    rsi = last.get("rsi14")
    if pd.notna(rsi):
        pts = 1 if 50 <= rsi <= 68 else 0.5 if 45 <= rsi < 50 or 68 < rsi <= 72 else -1 if rsi < 35 or rsi > 80 else -0.25
        add("RSI14", round(float(rsi), 1), pts)
    else: items["RSI14"] = "Unavailable"
    mh = last.get("macd_hist")
    if pd.notna(mh): add("MACD histogram", "Positive" if mh > 0 else "Negative", 1 if mh > 0 else -1)
    else: items["MACD"] = "Unavailable"
    rv = last.get("rel_volume")
    if pd.notna(rv): add("Relative volume", round(float(rv), 2), 1 if rv >= 1.2 else 0 if rv >= 0.8 else -0.5)
    else: items["Relative volume"] = "Unavailable"
    if intraday and pd.notna(last.get("vwap")): add("VWAP", "Above" if c > last.vwap else "Below", 1 if c > last.vwap else -1)
    elif intraday: items["VWAP"] = "Unavailable"
    pct = 50 + 50 * (score / weight) if weight else np.nan
    conf = "High" if weight >= 5 else "Medium" if weight >= 3 else "Low"
    return {"score": round(pct, 1) if pd.notna(pct) else np.nan, "confidence": conf, "items": items}


def monthly_snapshot(monthly: pd.DataFrame) -> Dict[str, object]:
    """Long-term market-regime context from monthly OHLCV; deliberately modest in weight."""
    if monthly is None or monthly.empty:
        return {"score": np.nan, "confidence": "None", "items": {}, "observations": 0}
    x = add_indicators(monthly, intraday=False)
    last = x.iloc[-1]
    c = float(last.close)
    score = weight = 0.0
    items = {}

    def add(name, value, pts):
        nonlocal score, weight
        items[name] = value
        if pts is not None:
            score += pts
            weight += 1

    for col, label in [("ema12", "Price vs monthly EMA12"), ("ema24", "Price vs monthly EMA24")]:
        if col not in x:
            span = 12 if col == "ema12" else 24
            x[col] = x["close"].ewm(span=span, adjust=False).mean()
        val = x[col].iloc[-1]
        if pd.notna(val): add(label, "Above" if c > val else "Below", 1 if c > val else -1)
    rsi = last.get("rsi14")
    if pd.notna(rsi): add("Monthly RSI14", round(float(rsi), 1), 1 if 50 <= rsi <= 70 else 0.5 if 45 <= rsi < 50 or 70 < rsi <= 75 else -1 if rsi < 35 or rsi > 80 else -0.25)
    mh = last.get("macd_hist")
    if pd.notna(mh): add("Monthly MACD histogram", "Positive" if mh > 0 else "Negative", 1 if mh > 0 else -1)
    if len(x) >= 13:
        ret12 = c / float(x["close"].iloc[-13]) - 1
        add("12-month price return", round(100 * ret12, 1), 1 if ret12 > 0.10 else 0.5 if ret12 > 0 else -1)
    pct = 50 + 50 * score / weight if weight else np.nan
    return {"score": round(float(pct), 1) if pd.notna(pct) else np.nan, "confidence": "High" if weight >= 4 else "Medium" if weight >= 2 else "Low", "items": items, "observations": len(x), "start": str(x.index.min()), "end": str(x.index.max())}


def support_resistance(x: pd.DataFrame, window: int = 30) -> Tuple[float, float]:
    if x.empty: return np.nan, np.nan
    tail = x.tail(window)
    return float(tail["low"].quantile(0.15)), float(tail["high"].quantile(0.85))


def candlestick_pattern(row: pd.Series) -> str:
    o, h, l, c = [float(row[k]) for k in ["open", "high", "low", "close"]]
    body = abs(c-o); rng = max(h-l, 1e-9)
    upper, lower = h-max(o,c), min(o,c)-l
    if body/rng < 0.12 and upper/rng > .25 and lower/rng > .25: return "Doji / indecision"
    if lower > body*2 and upper < body*0.8: return "Hammer-like rejection"
    if upper > body*2 and lower < body*0.8: return "Shooting-star-like rejection"
    if c > o and body/rng > .65: return "Strong bullish candle"
    if o > c and body/rng > .65: return "Strong bearish candle"
    return "No dominant single-candle pattern"


def entry_plan(x: pd.DataFrame, horizon: str, intraday: bool = False) -> Dict[str, float | str]:
    if x.empty: return {}
    last = x.iloc[-1]; price = float(last.close)
    atr = float(last.atr14) if pd.notna(last.atr14) else price * (0.01 if intraday else 0.03)
    support, resistance = support_resistance(x, 50 if not intraday else 60)
    if intraday:
        entry_low, entry_high = price - 0.20*atr, price + 0.10*atr; stop = min(price-1.1*atr, support) if pd.notna(support) and support < price else price-1.1*atr; risk=max(price-stop, .5*atr); t1,t2=price+1.5*risk,price+2.2*risk
    elif horizon == "Short-term":
        entry_low, entry_high = price - .5*atr, price + .15*atr; stop=min(price-1.5*atr,support) if pd.notna(support) and support<price else price-1.5*atr; risk=max(price-stop,.75*atr); t1,t2=price+1.5*risk,price+2.5*risk
    else:
        entry_low, entry_high = price-atr, price+.25*atr; stop=price-2*atr; risk=max(price-stop,atr); t1,t2=price+2*risk,price+3*risk
    return {"current":price,"entry_low":entry_low,"entry_high":entry_high,"stop":stop,"target1":t1,"target2":t2,"rr1":(t1-price)/risk if risk else np.nan,"rr2":(t2-price)/risk if risk else np.nan,"support":support,"resistance":resistance,"atr":atr}


def _num(d, key):
    try:
        v=float(d.get(key,"nan")); return v if math.isfinite(v) else np.nan
    except Exception: return np.nan


def fundamental_snapshot(o: Dict[str, object]) -> Dict[str, object]:
    if not o: return {"score":np.nan,"items":{},"status":"Unavailable"}
    fields={"MarketCap":"Market Capitalization","PERatio":"P/E","PriceToBookRatio":"P/B","ReturnOnEquityTTM":"ROE","ReturnOnAssetsTTM":"ROA","ProfitMargin":"Profit Margin","OperatingMarginTTM":"Operating Margin","QuarterlyRevenueGrowthYOY":"Revenue growth YoY","QuarterlyEarningsGrowthYOY":"Earnings growth YoY","DebtToEquity":"Debt/Equity","DividendYield":"Dividend yield","Beta":"Beta","BookValue":"Book value","EPS":"EPS"}
    items={}; points=[]
    for k,label in fields.items():
        v=_num(o,k)
        if pd.notna(v): items[label]=v
    for k,lo,hi in [("PERatio",0,25),("PriceToBookRatio",0,4),("DebtToEquity",0,1.5)]:
        v=_num(o,k)
        if pd.notna(v): points.append(1 if lo<v<=hi else -1)
    for k in ["ReturnOnEquityTTM","ProfitMargin","OperatingMarginTTM","QuarterlyRevenueGrowthYOY","QuarterlyEarningsGrowthYOY"]:
        v=_num(o,k)
        if pd.notna(v): points.append(1 if v>0 else -1)
    score=50+50*np.mean(points) if points else np.nan
    return {"score":round(float(score),1) if pd.notna(score) else np.nan,"items":items,"status":"Available" if items else "Unavailable"}


def backtest_signal(x: pd.DataFrame, horizon: str, intraday: bool=False, hold_bars: int|None=None, cost_bps: float=10.0, slippage_bps: float=5.0) -> Dict[str, object]:
    if x.empty: return {"status":"No data"}
    bars=5 if intraday else 10 if horizon=="Short-term" else 20
    if hold_bars is not None: bars=hold_bars
    z=x.copy()
    signal=(z.close>z.ema20)&(z.macd_hist>0)&(z.rsi14.between(50,72))
    # Signal is known at bar t. Enter at the next bar close (t+1), then hold for `bars` additional bars.
    entry=z.close.shift(-1)
    exit_price=z.close.shift(-(bars+1))
    fwd=exit_price/entry-1
    trades=fwd[signal & entry.notna() & exit_price.notna()].dropna()
    n=len(trades)
    if n==0: return {"status":"Insufficient signal observations","observations":0,"minimum":30}
    friction=(cost_bps+slippage_bps)/10000; net=trades-friction; wins=net>0
    gross_profit=net[net>0].sum(); gross_loss=-net[net<0].sum(); equity=(1+net).cumprod(); dd=(equity/equity.cummax()-1).min()
    expectancy=float(net.mean()); pf=float(gross_profit/gross_loss) if gross_loss>0 else np.inf
    status="Exploratory" if n<30 else "Preliminary" if n<100 else "Moderate evidence" if n<250 else "Stronger evidence"
    return {"status":status,"observations":n,"minimum":30,"win_rate":float(wins.mean()),"expectancy":expectancy,"profit_factor":pf,"max_drawdown":float(dd),"net_return_sum":float(net.sum()),"friction_bps":cost_bps+slippage_bps}


def decision(tech: Dict[str, object], fund: Dict[str, object], horizon: str, backtest: Dict[str, object]|None=None, hard_risk: bool=False, monthly: Dict[str, object]|None=None) -> Dict[str, object]:
    ts=tech.get("score",np.nan); fs=fund.get("score",np.nan); ms=(monthly or {}).get("score",np.nan)
    if horizon=="Long-term":
        if pd.notna(ts) and pd.notna(ms):
            market_tech=0.60*ts+0.40*ms
        else: market_tech=ms if pd.isna(ts) else ts
        score=fund.get("score",np.nan) if pd.isna(market_tech) else (0.35*market_tech+0.65*fs if pd.notna(fs) else market_tech)
    elif horizon=="Short-term": score=ts if pd.isna(fs) else (0.70*ts+0.30*fs if pd.notna(ts) else fs)
    else: score=ts
    bt=backtest or {}; n=int(bt.get("observations",0) or 0); exp=bt.get("expectancy",np.nan); validation_penalty=False
    if horizon in {"Intraday","Short-term"} and n>=30 and pd.notna(exp) and float(exp)<0:
        validation_penalty=True; score=min(score,57.9) if pd.notna(score) else score
    if hard_risk: verdict="PASS / RISK FLAG"
    elif pd.isna(score): verdict="INSUFFICIENT DATA"
    elif validation_penalty: verdict="PASS / NEGATIVE VALIDATION"
    elif score>=72: verdict="BUY / SETUP"
    elif score>=58: verdict="WATCH / WAIT FOR ENTRY"
    else: verdict="PASS"
    return {"score":round(float(score),1) if pd.notna(score) else np.nan,"verdict":verdict,"validation_penalty":validation_penalty}


def analyze(df: pd.DataFrame, horizon: str, intraday: bool=False, fundamentals: Dict[str, object]|None=None, interval: str="5m", monthly: pd.DataFrame|None=None) -> Dict[str, object]:
    x=add_indicators(df,intraday=intraday); q=data_quality(x,interval,intraday); tech=technical_snapshot(x,intraday); fund=fundamental_snapshot(fundamentals or {})
    month=monthly_snapshot(monthly) if horizon=="Long-term" else {"score":np.nan,"confidence":"None","items":{},"observations":0}
    plan=entry_plan(x,horizon,intraday); bt=backtest_signal(x,horizon,intraday); hard_risk=False
    d=fundamentals or {}; de=_num(d,"DebtToEquity"); pm=_num(d,"ProfitMargin")
    if pd.notna(de) and de>4: hard_risk=True
    if pd.notna(pm) and pm<0: hard_risk=True
    dec=decision(tech,fund,horizon,backtest=bt,hard_risk=hard_risk,monthly=month)
    return {"data":x,"quality":q,"technical":tech,"monthly":month,"fundamental":fund,"plan":plan,"backtest":bt,"decision":dec,"candle_pattern":candlestick_pattern(x.iloc[-1]) if not x.empty else "—"}
