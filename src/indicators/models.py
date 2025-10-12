from __future__ import annotations
from typing import Dict, List, Optional, Union, Literal, Any
from pydantic import BaseModel, Field, validator

# --- Shared ---
OutputName = Union[str, Dict[str, str]]

class FillNASpec(BaseModel):
    method: Optional[Literal["ffill","bfill","zero"]] = None
    limit: Optional[int] = None

class WarmupSpec(BaseModel):
    min_bars: int = 0
    mask_to_nan: bool = True

class IndicatorSpec(BaseModel):
    name: str                     # canonical name, e.g. "rsi", "pe" (see registry)
    params: Dict[str, Any] = Field(default_factory=dict)
    input_map: Dict[str, str] = Field(default_factory=dict)  # map std inputs→df cols
    output: OutputName            # final column name or mapping for multi-output
    depends_on: List[str] = Field(default_factory=list)      # names of prior outputs required
    timeframe: Optional[str] = None                          # optional per-indicator TF (tech only)

class IndicatorBatchConfig(BaseModel):
    timeframe: Optional[str] = None   # batch TF (applies to tech indicators without own TF)
    fillna: Optional[FillNASpec] = None
    warmup: Optional[WarmupSpec] = None
    dropna_after: bool = False
    indicators: List[IndicatorSpec]

    @field_validator("indicators")
    def unique_outputs(cls, v):
        seen = set()
        for spec in v:
            outs = spec.output if isinstance(spec.output, dict) else {"value": spec.output}
            for name in outs.values():
                if name in seen:
                    raise ValueError(f"Duplicate output column: {name}")
                seen.add(name)
        return v

# Request model for ticker-based computation (tech + fundamentals)
class TickerIndicatorsRequest(BaseModel):
    ticker: str
    timeframe: str = "1d"            # for OHLCV
    period: str = "1y"
    provider: Optional[str] = None
    indicators: List[str]            # list of names from registry (both tech + fund)
    fillna: Optional[FillNASpec] = None
    warmup: Optional[WarmupSpec] = None
    include_recommendations: bool = True
    force_refresh: bool = False

# Result container (minimalist)
class IndicatorValue(BaseModel):
    name: str
    value: Any
    source: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)

class IndicatorResultSet(BaseModel):
    ticker: Optional[str] = None
    technical: Dict[str, IndicatorValue] = Field(default_factory=dict)
    fundamental: Dict[str, IndicatorValue] = Field(default_factory=dict)
    overall: Optional[Dict[str, Any]] = None
