"""
Models for trading strategy logic.

Includes:
- Aggregation methods and market regime enums
- Trading signal and composite signal dataclasses
"""
from enum import Enum
from typing import Any, Dict, List
from dataclasses import dataclass
from datetime import datetime

class AggregationMethod(Enum):
    """Methods for aggregating signals from multiple strategies."""
    WEIGHTED_VOTING = "weighted_voting"
    CONSENSUS = "consensus"
    MAJORITY = "majority"
    WEIGHTED_AVERAGE = "weighted_average"


class MarketRegime(Enum):
    """Market regime classifications."""
    TRENDING_VOLATILE = "trending_volatile"
    TRENDING_STABLE = "trending_stable"
    RANGING_VOLATILE = "ranging_volatile"
    RANGING_STABLE = "ranging_stable"
    CRISIS = "crisis"


@dataclass
class StrategySignal:
    """Represents a trading signal from a strategy."""
    strategy_name: str
    signal_type: str  # 'buy', 'sell', 'hold'
    confidence: float  # 0.0 to 1.0
    weight: float
    timestamp: datetime
    metadata: Dict[str, Any]


@dataclass
class CompositeSignal:
    """Represents an aggregated signal from multiple strategies."""
    signal_type: str
    confidence: float
    contributing_strategies: List[str]
    timestamp: datetime
    metadata: Dict[str, Any]
