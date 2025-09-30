# ---------------------------------------------------------------------------
# registry.py — catalog of indicators and provider priority
# ---------------------------------------------------------------------------
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Literal, Any

@dataclass
class IndicatorMeta:
    kind: Literal["tech","fund"]
    inputs: List[str]              # tech: e.g. ["close"], fund: []
    outputs: List[str]             # canonical outputs (e.g. ["value"], or ["upper","middle","lower"])
    providers: List[str]           # priority order keys that match adapters dict
    defaults: dict[str, object] = None  # canonical defaults (optional)

INDICATOR_META = {
    # --- Technical ---
    "rsi":         IndicatorMeta("tech", ["close"], ["value"], ["ta-lib","pandas-ta"]),
    "ema":         IndicatorMeta("tech", ["close"], ["value"], ["ta-lib","pandas-ta"]),
    "sma":         IndicatorMeta("tech", ["close"], ["value"], ["ta-lib","pandas-ta"]),
    "macd":        IndicatorMeta("tech", ["close"], ["macd","signal","hist"], ["ta-lib","pandas-ta"]),
    "bbands":      IndicatorMeta("tech", ["close"], ["upper","middle","lower"], ["ta-lib","pandas-ta"]),
    "adx":         IndicatorMeta("tech", ["high","low","close"], ["value"], ["ta-lib","pandas-ta"]),
    "plus_di":     IndicatorMeta("tech", ["high","low","close"], ["value"], ["ta-lib","pandas-ta"]),
    "minus_di":    IndicatorMeta("tech", ["high","low","close"], ["value"], ["ta-lib","pandas-ta"]),
    "atr":         IndicatorMeta("tech", ["high","low","close"], ["value"], ["ta-lib","pandas-ta"]),
    "stoch":       IndicatorMeta("tech", ["high","low","close"], ["k","d"], ["ta-lib","pandas-ta"]),
    "obv":         IndicatorMeta("tech", ["close","volume"], ["value"], ["ta-lib","pandas-ta"]),
    # --- Fundamental (point-in-time / latest snapshot) ---
    "pe":          IndicatorMeta("fund", [], ["value"], ["fundamentals"]),
    "forward_pe":  IndicatorMeta("fund", [], ["value"], ["fundamentals"]),
    "pb":          IndicatorMeta("fund", [], ["value"], ["fundamentals"]),
    "ps":          IndicatorMeta("fund", [], ["value"], ["fundamentals"]),
    "peg":         IndicatorMeta("fund", [], ["value"], ["fundamentals"]),
    "roe":         IndicatorMeta("fund", [], ["value"], ["fundamentals"]),
    "roa":         IndicatorMeta("fund", [], ["value"], ["fundamentals"]),
    "de_ratio":    IndicatorMeta("fund", [], ["value"], ["fundamentals"]),
    "current_ratio": IndicatorMeta("fund", [], ["value"], ["fundamentals"]),
    "quick_ratio": IndicatorMeta("fund", [], ["value"], ["fundamentals"]),
    "div_yield":   IndicatorMeta("fund", [], ["value"], ["fundamentals"]),
    "payout_ratio":IndicatorMeta("fund", [], ["value"], ["fundamentals"]),
    "market_cap":  IndicatorMeta("fund", [], ["value"], ["fundamentals"]),
    "enterprise_value": IndicatorMeta("fund", [], ["value"], ["fundamentals"]),
}

