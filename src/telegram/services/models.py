from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Literal, Dict, Any, List

TF = Literal["1m","5m","15m","1h","4h","1d"]

class IndicatorSpec(BaseModel):
    type: Literal["RSI","MACD","BBANDS","SMA","EMA","ADX","ATR","STOCH","WILLR","CCI","ROC","MFI"]
    output: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)

class Operand(BaseModel):
    value: Optional[float] = None
    field: Optional[Literal["open","high","low","close","volume"]] = None
    indicator: Optional[IndicatorSpec] = None

class Node(BaseModel):
    # comparators
    gt: Optional[Dict[str, Operand]] = None
    gte: Optional[Dict[str, Operand]] = None
    lt: Optional[Dict[str, Operand]] = None
    lte: Optional[Dict[str, Operand]] = None
    eq: Optional[Dict[str, Operand]] = None
    ne: Optional[Dict[str, Operand]] = None
    between: Optional[Dict[str, Operand]] = None
    outside: Optional[Dict[str, Operand]] = None
    inside_band: Optional[Dict[str, Operand]] = None
    outside_band: Optional[Dict[str, Operand]] = None
    crosses_above: Optional[Dict[str, Operand]] = None
    crosses_below: Optional[Dict[str, Operand]] = None
    # logic
    and_: Optional[List["Node"]] = Field(default=None, alias="and")
    or_: Optional[List["Node"]] = Field(default=None, alias="or")
    not_: Optional["Node"] = Field(default=None, alias="not")
    model_config = ConfigDict(allow_population_by_field_name=True)

Node.update_forward_refs()

class NotifyCfg(BaseModel):
    telegram: bool = True
    email: bool = False

class OptionsCfg(BaseModel):
    evaluate_once_per_bar: bool = True
    on_error: Literal["ERROR","IGNORE"] = "ERROR"
    snapshot_payload: bool = False

class AlertConfig(BaseModel):
    version: int = 1
    name: str
    ticker: str
    timeframe: TF
    rule: Node
    re_arm: Optional[Node] = None
    notify: NotifyCfg = NotifyCfg()
    options: OptionsCfg = OptionsCfg()
    meta: Dict[str, Any] = Field(default_factory=dict)
