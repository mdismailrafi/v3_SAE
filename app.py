
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_provider import load_daily, load_monthly, load_overview, load_intraday
from engine import analyze

st.set_page_config(
    page_title="Indian Stock Analysis Engine V3.3",
    page_icon="📈",
    layout="wide",
)

st.markdown("# 📈 Indian Stock Analysis Engine")
st.caption("V3.3 • Intraday-first data pipeline • NSE/OpenChart charting • automatic CSV recovery • no Alpha Vantage in the intraday path")

with st.sidebar:
    st.header("Analysis")
    symbol = st.text_input(
        "Indian stock symbol",
        value="RVNL",
        help="Use the NSE trading symbol, e.g. RVNL, RELIANCE, TCS, INFY."
    ).strip().upper()

    horizon = st.selectbox("Horizon", ["Intraday", "Short-term", "Long-term"])

    # Keep the UI exactly aligned with the intervals supported by the NSE provider.
    interval = st.selectbox(
        "Intraday timeframe",
        ["1m", "5m", "10m", "15m", "30m", "1h"],
        index=1,
        disabled=horizon != "Intraday",
    )

    lookback = st.slider(
        "Intraday history (calendar days)",
        2, 30, 10,
        disabled=horizon != "Intraday",
    )

    force_refresh = st.checkbox(
        "Force fresh NSE fetch",
        value=True,
        disabled=horizon != "Intraday",
        help="Requests fresh NSE candles. If NSE fails, the engine automatically checks its previously fetched CSV cache."
    )

    analyze_btn = st.button("Analyze", type="primary", use_container_width=True)

    st.divider()
    if horizon == "Intraday":
        st.success("INTRADAY SOURCE\nOpenChart-compatible NSE charting API")
        st.caption(
            "The intraday pipeline never calls Alpha Vantage. "
            "Fetched OHLCV is automatically written to .data_cache/intraday_csv/."
        )
    else:
        st.info("Daily/monthly/fundamental context uses Alpha Vantage.")

if analyze_btn:
    if not symbol:
        st.error("Enter an NSE stock symbol.")
        st.stop()

    with st.spinner(f"Fetching and analysing {symbol}…"):
        try:
            if horizon == "Intraday":
                source = "NSE charting / OpenChart-compatible API (automatic CSV recovery enabled)"
                df = load_intraday(symbol, interval, lookback, force_refresh=force_refresh)
                provider_status = df.attrs.get("provider_status", {})

                if df.empty:
                    st.error("No usable intraday candles are available from NSE or the automatic CSV recovery cache.")
                    st.warning(
                        "This is a DATA FAILURE, not a trading PASS. "
                        "Try another supported timeframe or refresh later."
                    )
                    st.stop()

                fundamentals = {}
                result = analyze(
                    df, horizon, intraday=True,
                    fundamentals=fundamentals, interval=interval
                )
                monthly = pd.DataFrame()

            else:
                df = load_daily(symbol)
                if df.empty:
                    st.error("No daily data returned. Check the symbol or Alpha Vantage entitlement/rate limit.")
                    st.stop()
                fundamentals = load_overview(symbol)
                result = analyze(
                    df, horizon, intraday=False,
                    fundamentals=fundamentals, interval="1d"
                )
                source = "Alpha Vantage daily + overview"
                monthly = load_monthly(symbol) if horizon == "Long-term" else pd.DataFrame()
                provider_status = {}

        except Exception as exc:
            st.exception(exc)
            st.stop()

    q = result["quality"]
    dec = result["decision"]
    tech = result["technical"]
    fund = result["fundamental"]
    plan = result["plan"]
    bt = result["backtest"]

    st.success(f"Data source: {source}")
    if horizon == "Intraday" and provider_status:
        src_type = provider_status.get("source_type", "")
        if src_type:
            st.caption(f"Actual intraday transport: {src_type} • {provider_status.get('rows', len(df))} candles")

    if horizon == "Intraday":
        st.caption(
            "Intraday flow: NSE charting → OHLCV normalization → canonical CSV → V3 analysis. "
            "If live NSE fails, the canonical CSV is checked automatically; Alpha Vantage is never used."
        )

    st.markdown(f"## {symbol} — {horizon}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Decision", dec["verdict"])
    c2.metric("Composite score", f"{dec['score']:.1f}" if pd.notna(dec["score"]) else "NA")
    c3.metric("Last price", f"₹{df.close.iloc[-1]:,.2f}")
    c4.metric("Data quality", q["grade"])

    tabs = st.tabs(["Decision", "Chart & Signals", "Fundamentals", "Data Quality", "Validation"])

    with tabs[0]:
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Preferred entry", f"₹{plan.get('entry_low', float('nan')):,.2f}–₹{plan.get('entry_high', float('nan')):,.2f}")
        p2.metric("Stop", f"₹{plan.get('stop', float('nan')):,.2f}")
        p3.metric("Target 1", f"₹{plan.get('target1', float('nan')):,.2f}")
        p4.metric("R:R", f"1:{plan.get('rr1', float('nan')):.2f}")

        st.markdown("### Why")
        st.write(f"Technical score: **{tech.get('score', 'NA')}** ({tech.get('confidence', '—')} confidence)")
        if pd.notna(fund.get("score", float("nan"))):
            st.write(f"Fundamental score: **{fund['score']}**")
        st.write(f"Latest candle: **{result['candle_pattern']}**")
        st.caption(
            "A BUY/SETUP verdict is a research signal, not a guarantee. "
            "The engine separates signal strength from statistical evidence."
        )

    with tabs[1]:
        z = result["data"]
        fig = go.Figure(
            data=[go.Candlestick(
                x=z.index, open=z.open, high=z.high,
                low=z.low, close=z.close, name="Price"
            )]
        )
        for col, label in [
            ("ema9", "EMA9"), ("ema20", "EMA20"),
            ("ema50", "EMA50"), ("ema200", "EMA200")
        ]:
            if col in z and z[col].notna().any():
                fig.add_trace(go.Scatter(x=z.index, y=z[col], name=label, mode="lines"))
        if horizon == "Intraday" and "vwap" in z:
            fig.add_trace(go.Scatter(x=z.index, y=z.vwap, name="VWAP", mode="lines"))
        fig.update_layout(
            height=620,
            xaxis_rangeslider_visible=False,
            margin=dict(l=10, r=10, t=30, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("### Signals")
        st.json(tech["items"])

    with tabs[2]:
        if horizon == "Intraday":
            st.info("Fundamentals are intentionally not used to force an intraday trade.")
        else:
            st.metric(
                "Fundamental score",
                f"{fund['score']:.1f}" if pd.notna(fund.get("score")) else "Unavailable"
            )
            st.dataframe(
                pd.DataFrame(
                    list(fund.get("items", {}).items()),
                    columns=["Metric", "Value"]
                ),
                use_container_width=True, hide_index=True
            )
            if horizon == "Long-term" and not monthly.empty:
                st.caption(
                    f"Monthly context: {len(monthly)} observations available from "
                    f"{monthly.index.min().date()} to {monthly.index.max().date()}."
                )

    with tabs[3]:
        st.dataframe(pd.DataFrame([q]), use_container_width=True, hide_index=True)
        st.write("**Indicator availability**")
        avail = {
            k: ("Available" if result["data"][k].notna().any() else "Unavailable")
            for k in ["ema9", "ema20", "ema50", "ema200", "rsi14", "macd", "atr14", "rel_volume"]
            if k in result["data"]
        }
        st.dataframe(
            pd.DataFrame(list(avail.items()), columns=["Indicator", "Status"]),
            use_container_width=True, hide_index=True
        )
        if q["rows"] < q["required"]:
            st.warning(
                "The dataset is below the engine's preferred size for validation. "
                "Indicator availability and validation sufficiency are treated separately."
            )

    with tabs[4]:
        st.metric("Validation status", bt.get("status", "—"))
        vcols = st.columns(5)
        vcols[0].metric("Observations", bt.get("observations", 0))
        vcols[1].metric("Win rate", f"{100 * bt['win_rate']:.1f}%" if "win_rate" in bt else "—")
        vcols[2].metric("Expectancy", f"{100 * bt['expectancy']:.2f}%" if "expectancy" in bt else "—")
        vcols[3].metric("Profit factor", f"{bt['profit_factor']:.2f}" if "profit_factor" in bt and bt['profit_factor'] != float('inf') else "∞")
        vcols[4].metric("Max drawdown", f"{100 * bt['max_drawdown']:.1f}%" if "max_drawdown" in bt else "—")
        st.caption(
            "Validation is conservative: fewer than 30 signal observations is exploratory. "
            "Costs/slippage are included in the displayed test."
        )

else:
    st.markdown("### Start here")
    st.write(
        "Enter an NSE stock symbol and choose a horizon. "
        "For Intraday, V3.3 fetches candles from NSE charting/OpenChart-compatible endpoints, "
        "automatically stores them as a canonical CSV, and can recover from that CSV if NSE is temporarily unavailable."
    )
