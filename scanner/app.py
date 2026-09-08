from __future__ import annotations
import pandas as pd
import streamlit as st

from data import load_symbols, fetch_daily, fetch_market_snapshot, fetch_fundamentals
from scoring import enrich, score_row, market_regime
from events import fetch_announcements, classify_announcements
from ipo import fetch_ipo_data, score_ipo

st.set_page_config(page_title="Indian Market Scanner", page_icon="📡", layout="wide")
st.title("📡 Indian Market Opportunity Scanner")
st.caption("Market-wide discovery → support + accumulation + breakout ranking → V3.3 deep analysis")
with st.sidebar:
    st.header("Scanner")
    uploaded = st.file_uploader("Optional symbols CSV", type=["csv"])
    max_symbols = st.number_input("Stocks to scan", min_value=5, max_value=500, value=100, step=25)
    period = st.selectbox("History", ["6mo", "1y", "2y"], index=1)
    event_days = st.number_input("Event lookback (days)", min_value=1, max_value=10, value=3)
    run = st.button("🔎 Scan market", type="primary", use_container_width=True)
    st.info("NIFTY 500 best-effort universe • Yahoo daily feed • NSE corporate events • fundamentals enrichment. Missing data is marked unknown, never invented.")

snapshot = fetch_market_snapshot(); regime, regime_score = market_regime(snapshot)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Market regime", regime)
c2.metric("NIFTY 50 20D", f"{snapshot.get('NIFTY50', {}).get('ret20', 0) * 100:.1f}%" if snapshot.get('NIFTY50', {}).get('ret20') is not None else "—")
c3.metric("NIFTY 50 60D", f"{snapshot.get('NIFTY50', {}).get('ret60', 0) * 100:.1f}%" if snapshot.get('NIFTY50', {}).get('ret60') is not None else "—")
c4.metric("Regime score", f"{regime_score}/100")

if run:
    symbols = load_symbols(uploaded)[:int(max_symbols)]; rows = []; errors = []; bar = st.progress(0)
    total = max(1, len(symbols))
    for i, symbol in enumerate(symbols, 1):
        result = fetch_daily(symbol, period)
        if result.frame.empty:
            errors.append((symbol, result.error)); bar.progress(i / total); continue
        try:
            enriched = enrich(result.frame); fundamentals = fetch_fundamentals(symbol); row = score_row(enriched, regime_score, fundamentals)
            event_items = fetch_announcements(symbol, days=int(event_days)); event_score, event_sentiment, event_note = classify_announcements(event_items)
            row["event_score"] = event_score; row["event_sentiment"] = event_sentiment; row["event_note"] = event_note
            row["intraday_score"] = round(max(0, min(100, row["intraday_score"] + event_score * .5)), 1)
            row["short_score"] = round(max(0, min(100, row["short_score"] + event_score * .75)), 1)
            row["long_score"] = round(max(0, min(100, row["long_score"] + event_score * .25)), 1)
            row["symbol"] = symbol; row["source"] = result.source; rows.append(row)
        except Exception as exc:
            errors.append((symbol, str(exc)))
        bar.progress(i / total)
    bar.empty(); st.session_state["scan_df"] = pd.DataFrame(rows); st.session_state["errors"] = errors

if "scan_df" not in st.session_state:
    st.warning("Press **Scan market** to build the opportunity board."); st.stop()
df = st.session_state["scan_df"]; errors = st.session_state.get("errors", [])
if df.empty: st.error("No stocks produced usable data."); st.stop()

def board(title, score, desc):
    st.subheader(title); st.caption(desc)
    cols = ["symbol", score, "price", "rsi", "rvol", "ret5", "ret20", "fundamental_score", "event_sentiment", "fresh_breakout", "risk_flag"]
    out = df.sort_values(score, ascending=False).head(20)[cols].copy(); out.columns = ["Symbol", "Score", "Price", "RSI", "RVOL", "5D %", "20D %", "Fundamental", "Event", "Breakout", "Risk"]
    out[["5D %", "20D %"]] = (out[["5D %", "20D %"]] * 100).round(1); st.dataframe(out, use_container_width=True, hide_index=True)

def setup_board(title, frame, desc):
    st.subheader(title); st.caption(desc)
    cols = ["symbol", "prebreakout_score", "setup_grade", "setup_type", "price", "support", "support_distance_pct", "support_tests", "resistance", "resistance_distance_pct", "rsi", "rsi_slope", "rvol", "volume_contraction", "fundamental_score", "short_score", "risk_flag"]
    out = frame.sort_values("prebreakout_score", ascending=False).head(20)[cols].copy()
    out.columns = ["Symbol", "Pre-Breakout", "Grade", "Setup", "Price", "Support", "Support Dist %", "Tests", "Resistance", "Resistance Dist %", "RSI", "RSI Δ", "RVOL", "Vol Contract", "Fundamental", "Short Score", "Risk"]
    out[["Support Dist %", "Resistance Dist %"]] *= 100; out[["Support Dist %", "Resistance Dist %"]] = out[["Support Dist %", "Resistance Dist %"]].round(1)
    out[["Pre-Breakout", "RSI", "RSI Δ", "RVOL", "Fundamental", "Short Score"]] = out[["Pre-Breakout", "RSI", "RSI Δ", "RVOL", "Fundamental", "Short Score"]].round(1)
    st.dataframe(out, use_container_width=True, hide_index=True)

# The Springboard board deliberately separates setup discovery from a buy call.
tabs = st.tabs(["🔥 Intraday Top 20", "📈 Short-term Top 20", "🏦 Long-term Top 20", "🟢 Springboard", "🚀 Near Breakout", "💰 Accumulation", "🚀 Breakouts", "📰 Events", "🏷️ IPO Watch", "⚠️ Risk"])
with tabs[0]: board("Intraday Top 20", "intraday_score", "Price action + momentum + market regime + recent corporate catalyst.")
with tabs[1]: board("Short-term Top 20", "short_score", "Technical strength + pre-breakout setup + regime + fundamentals + catalyst.")
with tabs[2]: board("Long-term Top 20", "long_score", "Fundamental enrichment + long trend + accumulation setup + regime. V3.3 is final authority.")
with tabs[3]:
    s = df[(df.setup_type == "SPRINGBOARD") & (df.prebreakout_score >= 70)].copy()
    setup_board("🟢 Springboard — strong stock sitting near support", s, "Looks for an established trend, nearby support, repeated tests, improving momentum and constructive volume. Best used as a watchlist for a confirmed reversal.")
with tabs[4]:
    s = df[(df.setup_type == "NEAR BREAKOUT") & (df.prebreakout_score >= 70)].copy()
    setup_board("🚀 Near Breakout — strong stock close to resistance", s, "Candidates are within roughly 5% of a detected resistance zone. Prefer a decisive close above resistance with volume rather than anticipating blindly.")
with tabs[5]:
    s = df[(df.volume_contraction) & (df.prebreakout_score >= 60) & (df.fundamental_score >= 55)].copy()
    setup_board("💰 Accumulation — supply appears to be drying up", s, "Volume contraction + positive OBV/momentum + acceptable fundamentals. This is an accumulation watchlist, not proof of institutional buying.")
with tabs[6]:
    b = df[df.fresh_breakout].sort_values("short_score", ascending=False).head(20)
    st.dataframe(b[["symbol", "price", "rsi", "rvol", "ret5", "ret20", "breakout_today", "breakout_previous", "atr_pct", "fundamental_score", "event_sentiment", "risk_flag"]], use_container_width=True, hide_index=True)
with tabs[7]:
    e = df[df.event_sentiment != "NONE"].copy().sort_values("event_score", ascending=False)
    st.dataframe(e[["symbol", "event_score", "event_sentiment", "event_note", "short_score", "fresh_breakout"]].head(50), use_container_width=True, hide_index=True)
with tabs[8]:
    ipo = score_ipo(fetch_ipo_data())
    if ipo.empty: st.info("NSE IPO endpoint did not return data. No IPO values are fabricated; retry later or use the NSE IPO page.")
    else: st.dataframe(ipo, use_container_width=True, hide_index=True)
with tabs[9]:
    r = df.sort_values("atr_pct", ascending=False).head(20); st.dataframe(r[["symbol", "price", "atr_pct", "rsi", "rvol", "ret20", "fundamental_score", "event_sentiment", "risk_flag"]], use_container_width=True, hide_index=True)

st.divider(); st.subheader("🧭 How to use the new setup boards")
st.write("1) Start with 🟢 Springboard or 💰 Accumulation. 2) Open the strongest names in V3.3. 3) Wait for price-action confirmation: higher low/reclaim of support, improving RSI/MACD and preferably expanding volume. 4) For a breakout trade, wait for a close above resistance and define the stop before entry. A high score is a candidate, not a guaranteed take-off.")
st.divider(); st.subheader("🔗 V3.3 hand-off")
st.write("Scanner = discovery/ranking. V3.3 = final deep analysis, backtest, risk and entry plan. Never treat a scanner score alone as a buy signal.")
if st.button("Export current scan CSV"): st.download_button("Download CSV", df.to_csv(index=False).encode(), "market_scanner_results.csv", "text/csv")
if errors:
    with st.expander(f"Data issues ({len(errors)})"): st.dataframe(pd.DataFrame(errors, columns=["Symbol", "Error"]), use_container_width=True, hide_index=True)
