from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

TF = Literal["1m", "5m", "15m", "1h", "4h", "1d"]


class IndicatorSpec(BaseModel):
    type: Literal["RSI", "MACD", "BBANDS", "SMA", "EMA", "ADX", "ATR", "STOCH", "WILLR", "CCI", "ROC", "MFI"]
    output: str | None = None
    params: Dict[str, Any] = Field(default_factory=dict)


class Operand(BaseModel):
    value: float | None = None
    field: Literal["open", "high", "low", "close", "volume"] | None = None
    indicator: IndicatorSpec | None = None


class Node(BaseModel):
    # comparators
    gt: Dict[str, Operand] | None = None
    gte: Dict[str, Operand] | None = None
    lt: Dict[str, Operand] | None = None
    lte: Dict[str, Operand] | None = None
    eq: Dict[str, Operand] | None = None
    ne: Dict[str, Operand] | None = None
    between: Dict[str, Operand] | None = None
    outside: Dict[str, Operand] | None = None
    inside_band: Dict[str, Operand] | None = None
    outside_band: Dict[str, Operand] | None = None
    crosses_above: Dict[str, Operand] | None = None
    crosses_below: Dict[str, Operand] | None = None
    # logic
    and_: List["Node"] | None = Field(default=None, alias="and")
    or_: List["Node"] | None = Field(default=None, alias="or")
    not_: Optional["Node"] = Field(default=None, alias="not")
    model_config = ConfigDict(allow_population_by_field_name=True)


Node.update_forward_refs()


class NotifyCfg(BaseModel):
    telegram: bool = True
    email: bool = False


class OptionsCfg(BaseModel):
    evaluate_once_per_bar: bool = True
    on_error: Literal["ERROR", "IGNORE"] = "ERROR"
    snapshot_payload: bool = False


class AlertConfig(BaseModel):
    version: int = 1
    name: str
    ticker: str
    timeframe: TF
    rule: Node
    re_arm: Node | None = None
    notify: NotifyCfg = NotifyCfg()
    options: OptionsCfg = OptionsCfg()
    meta: Dict[str, Any] = Field(default_factory=dict)
