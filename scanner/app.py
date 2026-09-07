from __future__ import annotations
import pandas as pd
import streamlit as st

# Streamlit Cloud executes this file from the scanner directory. Local imports
# keep this entrypoint compatible with both Cloud and direct local execution.
from data import load_symbols, fetch_daily, fetch_market_snapshot, fetch_fundamentals
from scoring import enrich, score_row, market_regime
from events import fetch_announcements, classify_announcements
from ipo import fetch_ipo_data, score_ipo

st.set_page_config(page_title="Indian Market Scanner", page_icon="📡", layout="wide")
st.title("📡 Indian Market Opportunity Scanner")
st.caption("Market-wide discovery → catalyst + fundamentals + technical ranking → V3.3 deep analysis")
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

tabs = st.tabs(["🔥 Intraday Top 20", "📈 Short-term Top 20", "🏦 Long-term Top 20", "🚀 Breakouts", "📰 Events", "🏷️ IPO Watch", "⚠️ Risk"])
with tabs[0]: board("Intraday Top 20", "intraday_score", "Price action + momentum + market regime + recent corporate catalyst.")
with tabs[1]: board("Short-term Top 20", "short_score", "Technical strength + regime + breakout + fundamentals + catalyst.")
with tabs[2]: board("Long-term Top 20", "long_score", "Fundamental enrichment + long trend + regime + corporate catalyst. V3.3 is final authority.")
with tabs[3]:
    b = df[df.fresh_breakout].sort_values("short_score", ascending=False).head(20)
    st.dataframe(b[["symbol", "price", "rsi", "rvol", "ret5", "ret20", "breakout_today", "breakout_previous", "atr_pct", "fundamental_score", "event_sentiment", "risk_flag"]], use_container_width=True, hide_index=True)
with tabs[4]:
    e = df[df.event_sentiment != "NONE"].copy().sort_values("event_score", ascending=False)
    st.dataframe(e[["symbol", "event_score", "event_sentiment", "event_note", "short_score", "fresh_breakout"]].head(50), use_container_width=True, hide_index=True)
with tabs[5]:
    ipo = score_ipo(fetch_ipo_data())
    if ipo.empty: st.info("NSE IPO endpoint did not return data. No IPO values are fabricated; retry later or use the NSE IPO page.")
    else: st.dataframe(ipo, use_container_width=True, hide_index=True)
with tabs[6]:
    r = df.sort_values("atr_pct", ascending=False).head(20); st.dataframe(r[["symbol", "price", "atr_pct", "rsi", "rvol", "ret20", "fundamental_score", "event_sentiment", "risk_flag"]], use_container_width=True, hide_index=True)

st.divider(); st.subheader("🔗 V3.3 hand-off")
st.write("Scanner = discovery/ranking. V3.3 = final deep analysis, backtest, risk and entry plan. Never treat a scanner score alone as a buy signal.")
if st.button("Export current scan CSV"): st.download_button("Download CSV", df.to_csv(index=False).encode(), "market_scanner_results.csv", "text/csv")
if errors:
    with st.expander(f"Data issues ({len(errors)})"): st.dataframe(pd.DataFrame(errors, columns=["Symbol", "Error"]), use_container_width=True, hide_index=True)
