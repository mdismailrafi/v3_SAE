from __future__ import annotations
import numpy as np
import pandas as pd


def ema(s, n): return s.ewm(span=n, adjust=False).mean()


def rsi(s, n=14):
    d = s.diff(); up = d.clip(lower=0); dn = -d.clip(upper=0)
    rs = up.ewm(alpha=1/n, adjust=False).mean() / dn.ewm(alpha=1/n, adjust=False).mean().replace(0, np.nan)
    return 100 - (100/(1+rs))


def atr(df, n=14):
    pc = df.Close.shift(1)
    tr = pd.concat([df.High-df.Low, (df.High-pc).abs(), (df.Low-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()


def _obv(close, volume):
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def enrich(df):
    x = df.copy()
    for c in ["Open", "High", "Low", "Close", "Volume"]: x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.dropna(subset=["High", "Low", "Close", "Volume"])
    x["EMA20"] = ema(x.Close, 20); x["EMA50"] = ema(x.Close, 50); x["EMA200"] = ema(x.Close, 200)
    x["RSI"] = rsi(x.Close); x["ATR"] = atr(x); x["ATR_pct"] = x.ATR/x.Close
    x["VOL20"] = x.Volume.rolling(20).mean(); x["RVOL"] = x.Volume/x.VOL20
    x["HH20"] = x.High.rolling(20).max().shift(1); x["HH55"] = x.High.rolling(55).max().shift(1)
    x["LL20"] = x.Low.rolling(20).min().shift(1)
    x["RET5"] = x.Close.pct_change(5); x["RET20"] = x.Close.pct_change(20); x["RET60"] = x.Close.pct_change(60)
    x["RANGE_POS20"] = (x.Close-x.LL20)/(x.HH20-x.LL20).replace(0, np.nan); x["DOLLAR_VOL"] = x.Close*x.Volume
    x["MACD"] = ema(x.Close, 12)-ema(x.Close, 26); x["MACD_SIGNAL"] = ema(x.MACD, 9); x["MACD_HIST"] = x.MACD-x.MACD_SIGNAL
    x["OBV"] = _obv(x.Close, x.Volume); x["OBV20"] = x.OBV.diff(20)
    x["VOL_RATIO_5_20"] = x.Volume.rolling(5).mean()/x.Volume.rolling(20).mean()
    x["ATR_SMA20"] = x.ATR_pct.rolling(20).mean()
    # Swing lows/highs used to form support/resistance zones rather than treating one exact price as support.
    x["SWING_LOW"] = (x.Low <= x.Low.shift(1)) & (x.Low <= x.Low.shift(-1))
    x["SWING_HIGH"] = (x.High >= x.High.shift(1)) & (x.High >= x.High.shift(-1))
    return x


def market_regime(snapshot):
    n = snapshot.get("NIFTY50", {}); r20 = n.get("ret20"); r60 = n.get("ret60")
    if r20 is None: return "UNKNOWN", 50
    score = 50
    score += 15 if r20 > 0 else -15; score += 15 if r60 and r60 > 0 else -15
    vix = snapshot.get("INDIAVIX", {}).get("last")
    if vix is not None: score += 10 if vix < 15 else (-10 if vix > 22 else 0)
    score = max(0, min(100, score))
    return ("RISK-ON" if score >= 65 else "RISK-OFF" if score <= 35 else "NEUTRAL"), score


def _fundamental_score(f):
    if not f: return 50
    s = 50
    if f.get("roe") is not None: s += 15 if f["roe"] > .15 else (-10 if f["roe"] < 0 else 0)
    if f.get("revenue_growth") is not None: s += 10 if f["revenue_growth"] > .10 else (-5 if f["revenue_growth"] < 0 else 0)
    if f.get("earnings_growth") is not None: s += 10 if f["earnings_growth"] > .10 else (-5 if f["earnings_growth"] < 0 else 0)
    if f.get("debt_equity") is not None: s += 10 if f["debt_equity"] < 80 else (-10 if f["debt_equity"] > 150 else 0)
    return max(0, min(100, s))


def _zone_stats(x):
    last = x.iloc[-1]; price = float(last.Close)
    # Look back 120 sessions; cluster swing lows/highs within 3% to create practical zones.
    lows = x.loc[x.SWING_LOW & (x.index >= x.index[-1] - pd.Timedelta(days=190)), "Low"].dropna().tail(30)
    highs = x.loc[x.SWING_HIGH & (x.index >= x.index[-1] - pd.Timedelta(days=190)), "High"].dropna().tail(30)
    support_candidates = [v for v in lows if v <= price * 1.04]
    resistance_candidates = [v for v in highs if v >= price * .96]
    # EMA supports are valid only when price is close enough to them.
    for e in [last.EMA20, last.EMA50, last.EMA200]:
        if pd.notna(e) and e <= price * 1.05: support_candidates.append(float(e))
    for e in [last.EMA20, last.EMA50]:
        if pd.notna(e) and e >= price * .96: resistance_candidates.append(float(e))
    support = max(support_candidates) if support_candidates else np.nan
    resistance = min(resistance_candidates) if resistance_candidates else np.nan
    if pd.notna(resistance) and resistance <= price * 1.005:
        # Use prior 55-day high when the nearest swing high is effectively at price.
        resistance = max(float(last.HH55) if pd.notna(last.HH55) else price, price)
    tests = 0
    if pd.notna(support):
        band = .03
        tests = int(((x.Low >= support*(1-band)) & (x.Low <= support*(1+band))).sum())
    return support, resistance, tests


def _prebreakout(x, fundamentals=None, regime_score=50):
    last = x.iloc[-1]; prev = x.iloc[-2]; price = float(last.Close)
    support, resistance, tests = _zone_stats(x)
    support_dist = (price-support)/price if pd.notna(support) else np.nan
    resistance_dist = (resistance-price)/price if pd.notna(resistance) and resistance > price else np.nan
    rsi_slope = float(last.RSI-prev.RSI) if pd.notna(last.RSI) and pd.notna(prev.RSI) else 0
    ema20_slope = float(last.EMA20-x.EMA20.iloc[-6]) if len(x)>=6 and pd.notna(last.EMA20) else 0
    trend = 0
    trend += 8 if price > last.EMA200 else 0
    trend += 6 if last.EMA50 > last.EMA200 else 0
    trend += 3 if ema20_slope > 0 else 0
    trend += 3 if last.RET20 > 0 and last.RET5 > -0.05 else 0
    support_score = 0
    if pd.notna(support_dist):
        if support_dist <= .03: support_score += 8
        elif support_dist <= .06: support_score += 5
        elif support_dist <= .10: support_score += 2
    support_score += min(5, tests)
    support_score += 4 if min(abs(price-last.EMA20), abs(price-last.EMA50))/price <= .03 else 0
    prev_res_support = bool(pd.notna(last.HH20) and abs(price-last.HH20)/price <= .03)
    support_score += 3 if prev_res_support else 0
    momentum = 0
    if pd.notna(last.RSI):
        momentum += 4 if 40 <= last.RSI <= 58 else (2 if 35 <= last.RSI < 65 else 0)
    momentum += 4 if rsi_slope > 0 else 0
    momentum += 3 if last.MACD_HIST > prev.MACD_HIST else 0
    momentum += 4 if last.RET5 > 0 and last.RET20 > 0 else (2 if last.RET5 > 0 else 0)
    volume = 0
    contraction = bool(pd.notna(last.VOL_RATIO_5_20) and last.VOL_RATIO_5_20 < .85)
    volume += 5 if contraction else 0
    volume += 5 if last.OBV20 > 0 else 0
    volume += 5 if last.RVOL >= 1.2 and last.RET5 > 0 else 0
    volume += 5 if last.RVOL >= 1.0 and last.RET5 > 0 else 0
    proximity = 0
    if pd.notna(resistance_dist):
        proximity += 5 if resistance_dist <= .05 else (3 if resistance_dist <= .10 else 0)
        proximity += 5 if resistance_dist <= .03 else 0
    relative = 5 if last.RET20 > 0 else 0
    relative += 5 if last.RET60 > 0 else 0
    fscore = _fundamental_score(fundamentals or {})
    fundamental = (2 if fscore >= 70 else 1 if fscore >= 55 else 0) + (2 if fscore >= 70 else 0) + (1 if fscore >= 60 else 0)
    raw = trend + support_score + momentum + volume + proximity + relative + fundamental
    # In weak markets, do not manufacture high-confidence scores: demand stronger evidence.
    market_penalty = 8 if regime_score < 36 else (3 if regime_score < 50 else 0)
    score = max(0, min(100, raw-market_penalty))
    if score >= 80: grade = "A — High-quality pre-breakout"
    elif score >= 70: grade = "B — Strong watchlist"
    elif score >= 60: grade = "C — Needs confirmation"
    else: grade = "D — Ignore"
    setup = "SPRINGBOARD" if score >= 70 and pd.notna(support_dist) and support_dist <= .06 else ("NEAR BREAKOUT" if score >= 70 and pd.notna(resistance_dist) and resistance_dist <= .05 else "WATCH")
    return {
        "support": support, "resistance": resistance, "support_distance_pct": support_dist,
        "resistance_distance_pct": resistance_dist, "support_tests": tests, "rsi_slope": rsi_slope,
        "volume_contraction": contraction, "prebreakout_score": float(score), "setup_grade": grade,
        "setup_type": setup, "fscore": fscore
    }


def score_row(x, regime_score=50, fundamentals=None):
    last = x.iloc[-1]; prev = x.iloc[-2]
    trend = (20 if last.Close>last.EMA20 else 0)+(15 if last.EMA20>last.EMA50 else 0)+(15 if last.EMA50>last.EMA200 else 0)+(10 if last.RET20>0 else 0)
    momentum = min(20,max(0,(last.RSI-45)*.8))+min(20,max(0,last.RVOL*10)); quality=min(100,trend+momentum)
    breakout = bool(last.Close>last.HH20) if pd.notna(last.HH20) else False; prev_breakout = bool(prev.Close>prev.HH20) if pd.notna(prev.HH20) else False
    fresh = breakout or (prev_breakout and last.Close>=prev.Close*.985); fscore=_fundamental_score(fundamentals or {})
    pb = _prebreakout(x, fundamentals, regime_score)
    intraday=min(100,.50*quality+.15*regime_score+.15*(100 if fresh else 50)+.20*pb["prebreakout_score"])
    short=min(100,.30*quality+.15*regime_score+.15*(100 if fresh else 50)+.20*fscore+.20*pb["prebreakout_score"])
    long=min(100,.20*(100 if last.Close>last.EMA200 else 35)+.10*(100 if last.EMA50>last.EMA200 else 35)+.10*regime_score+.40*fscore+.20*pb["prebreakout_score"])
    return {
        "price":float(last.Close),"rsi":float(last.RSI),"rvol":float(last.RVOL),"ret5":float(last.RET5),"ret20":float(last.RET20),"trend_score":round(quality,1),"fundamental_score":round(fscore,1),
        "intraday_score":round(intraday,1),"short_score":round(short,1),"long_score":round(long,1),"breakout_today":breakout,"breakout_previous":prev_breakout,"fresh_breakout":fresh,
        "above_20ema":bool(last.Close>last.EMA20),"above_50ema":bool(last.Close>last.EMA50),"above_200ema":bool(last.Close>last.EMA200),"atr_pct":float(last.ATR_pct),
        "risk_flag":"HIGH" if last.ATR_pct>.06 else ("MEDIUM" if last.ATR_pct>.035 else "NORMAL"), **pb
    }
