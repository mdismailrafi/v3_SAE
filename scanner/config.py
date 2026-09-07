"""Central scanner configuration. Scores are deliberately explicit and auditable."""
HORIZON_WEIGHTS={
    "intraday":{"technical":.60,"regime":.20,"event":.15,"fundamental":.05},
    "short":{"technical":.40,"regime":.15,"event":.20,"fundamental":.20,"risk":.05},
    "long":{"fundamental":.50,"valuation":.20,"technical":.15,"regime":.10,"event":.05},
}
BREAKOUT_RULES={"lookback":20,"min_rvol":1.5,"max_extension_atr":2.5,"retest_tolerance":.02}
