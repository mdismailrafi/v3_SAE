from __future__ import annotations
import pandas as pd
import streamlit as st
from scanner.data import load_symbols,fetch_daily,fetch_market_snapshot,fetch_fundamentals
from scanner.scoring import enrich,score_row,market_regime

st.set_page_config(page_title="Indian Market Scanner",page_icon="📡",layout="wide")
st.title("📡 Indian Market Opportunity Scanner")
st.caption("Market-wide discovery → ranked candidates → V3.3 deep analysis")
with st.sidebar:
    st.header("Universe")
    uploaded=st.file_uploader("Optional symbols CSV",type=["csv"])
    max_symbols=st.number_input("Stocks to scan",min_value=5,max_value=500,value=100,step=25)
    period=st.selectbox("History",["6mo","1y","2y"],index=1)
    run=st.button("🔎 Scan market",type="primary",use_container_width=True)
    st.info("Universe: NSE NIFTY 500 best-effort, with manual CSV fallback. Daily recovery feed: Yahoo Finance. Fundamentals are enrichment; V3.3 remains final decision layer.")

snapshot=fetch_market_snapshot(); regime,regime_score=market_regime(snapshot)
c1,c2,c3,c4=st.columns(4)
c1.metric("Market regime",regime); c2.metric("NIFTY 50 20D",f"{snapshot.get('NIFTY50',{}).get('ret20',0)*100:.1f}%" if snapshot.get('NIFTY50',{}).get('ret20') is not None else "—"); c3.metric("NIFTY 50 60D",f"{snapshot.get('NIFTY50',{}).get('ret60',0)*100:.1f}%" if snapshot.get('NIFTY50',{}).get('ret60') is not None else "—"); c4.metric("Regime score",f"{regime_score}/100")

if run:
    symbols=load_symbols(uploaded)[:int(max_symbols)]; rows=[]; errors=[]; bar=st.progress(0)
    for i,symbol in enumerate(symbols,1):
        result=fetch_daily(symbol,period)
        if result.frame.empty: errors.append((symbol,result.error)); bar.progress(i/len(symbols)); continue
        try:
            enriched=enrich(result.frame); fundamentals=fetch_fundamentals(symbol); row=score_row(enriched,regime_score,fundamentals)
            row["symbol"]=symbol; row["source"]=result.source; rows.append(row)
        except Exception as exc: errors.append((symbol,str(exc)))
        bar.progress(i/len(symbols))
    bar.empty(); st.session_state["scan_df"]=pd.DataFrame(rows); st.session_state["errors"]=errors

if "scan_df" not in st.session_state:
    st.warning("Press **Scan market** to build the opportunity board."); st.stop()
df=st.session_state["scan_df"]; errors=st.session_state.get("errors",[])
if df.empty: st.error("No stocks produced usable data."); st.stop()

def board(title,score,desc):
    st.subheader(title); st.caption(desc)
    cols=["symbol",score,"price","rsi","rvol","ret5","ret20","fundamental_score","fresh_breakout","risk_flag"]
    out=df.sort_values(score,ascending=False).head(20)[cols].copy(); out.columns=["Symbol","Score","Price","RSI","RVOL","5D %","20D %","Fundamental","Fresh breakout","Risk"]
    out[["5D %","20D %"]]=(out[["5D %","20D %"]]*100).round(1); st.dataframe(out,use_container_width=True,hide_index=True)

tabs=st.tabs(["🔥 Intraday Top 20","📈 Short-term Top 20","🏦 Long-term Top 20","🚀 Fresh Breakouts","⚠️ Risk / Watch"])
with tabs[0]: board("Intraday Top 20","intraday_score","Price action + momentum + market regime. Run V3.3 Intraday before trading.")
with tabs[1]: board("Short-term Top 20","short_score","Days-to-weeks ranking using technical strength, regime, breakout and fundamental enrichment.")
with tabs[2]: board("Long-term Top 20","long_score","Business/fundamental enrichment + trend + regime. V3.3 performs the final deep fundamental decision.")
with tabs[3]:
    st.subheader("Fresh Breakouts — today or previous session")
    b=df[df.fresh_breakout].sort_values("short_score",ascending=False).head(20)
    st.dataframe(b[["symbol","price","rsi","rvol","ret5","ret20","breakout_today","breakout_previous","atr_pct","fundamental_score","risk_flag"]],use_container_width=True,hide_index=True)
with tabs[4]:
    r=df.sort_values("atr_pct",ascending=False).head(20); st.dataframe(r[["symbol","price","atr_pct","rsi","rvol","ret20","fundamental_score","risk_flag"]],use_container_width=True,hide_index=True)

st.divider(); st.subheader("🔗 V3.3 hand-off"); st.write("The scanner discovers and ranks. V3.3 decides after deeper technical, fundamental, backtest, risk and entry-plan analysis.")
if st.button("Export current scan CSV"): st.download_button("Download CSV",df.to_csv(index=False).encode(),"market_scanner_results.csv","text/csv")
if errors:
    with st.expander(f"Data issues ({len(errors)})"): st.dataframe(pd.DataFrame(errors,columns=["Symbol","Error"]),use_container_width=True,hide_index=True)
