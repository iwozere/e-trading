"""
Comprehensive type definitions for the unified indicator system.

This module provides TypeScript-style type hints, validation schemas,
and runtime type checking for all indicator-related interfaces.
"""

from __future__ import annotations
from typing import (
    Dict, List, Optional, Union, Any, Callable, Protocol, TypeVar, Generic,
    Literal, TypedDict, NewType, runtime_checkable
)
from typing_extensions import NotRequired
from datetime import datetime
from enum import Enum
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, field_validator
from abc import ABC, abstractmethod

# Type variables
T = TypeVar('T')
IndicatorValueType = TypeVar('IndicatorValueType', float, Dict[str, float])

# New types for better type safety
TickerSymbol = NewType('TickerSymbol', str)
IndicatorName = NewType('IndicatorName', str)
TimeFrame = NewType('TimeFrame', str)
Period = NewType('Period', str)
ProviderName = NewType('ProviderName', str)

# Literal types for better IDE support
RecommendationLiteral = Literal["BUY", "SELL", "HOLD", "STRONG_BUY", "STRONG_SELL", "WEAK_BUY", "WEAK_SELL"]
IndicatorCategoryLiteral = Literal["technical", "fundamental"]
ProviderLiteral = Literal["ta-lib", "pandas-ta", "fundamentals", "backtrader"]
FillMethodLiteral = Literal["ffill", "bfill", "zero"]

# TypedDict definitions for structured data
class OHLCVData(TypedDict):
    """OHLCV data structure."""
    open: Union[pd.Series, np.ndarray]
    high: Union[pd.Series, np.ndarray]
    low: Union[pd.Series, np.ndarray]
    close: Union[pd.Series, np.ndarray]
    volume: Union[pd.Series, np.ndarray]

class IndicatorParameters(TypedDict, total=False):
    """Common indicator parameters."""
    timeperiod: int
    fastperiod: int
    slowperiod: int
    signalperiod: int
    nbdevup: float
    nbdevdn: float
    fastk_period: int
    slowk_period: int
    slowd_period: int
    acceleration: float
    maximum: float
    length: int
    multiplier: float

class CalculationContext(TypedDict, total=False):
    """Context for indicator calculations."""
    current_price: float
    macd: float
    macd_signal: float
    macd_histogram: float
    bb_upper: float
    bb_middle: float
    bb_lower: float
    stoch_k: float
    stoch_d: float
    plus_di: float
    minus_di: float

class CacheMetrics(TypedDict):
    """Cache performance metrics."""
    cache_size: int
    max_size: int
    ttl: int
    hit_rate: str
    hits: int
    misses: int

class ServiceInfo(TypedDict):
    """Service information structure."""
    service: str
    version: str
    cache_stats: CacheMetrics
    available_indicators: Dict[str, int]

# Protocol definitions for duck typing
@runtime_checkable
class IndicatorAdapter(Protocol):
    """Protocol for indicator calculation adapters."""

    async def compute(
        self,
        name: str,
        df: pd.DataFrame,
        inputs: Dict[str, pd.Series],
        params: Dict[str, Any]
    ) -> Dict[str, pd.Series]:
        """Compute indicator values asynchronously."""
        ...

    def supports(self, indicator_name: str) -> bool:
        """Check if adapter supports the indicator."""
        ...

@runtime_checkable
class RecommendationProvider(Protocol):
    """Protocol for recommendation providers."""

    def get_recommendation(
        self,
        indicator_name: str,
        value: Union[float, Dict[str, float]],
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Get recommendation for indicator value."""
        ...

@runtime_checkable
class CacheProvider(Protocol):
    """Protocol for cache providers."""

    def get(self, key: str) -> Optional[Any]:
        """Get cached value."""
        ...

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set cached value."""
        ...

    def clear(self) -> None:
        """Clear cache."""
        ...

# Generic types for better type safety
class IndicatorResult(BaseModel, Generic[IndicatorValueType]):
    """Generic indicator result."""
    name: IndicatorName
    value: IndicatorValueType
    timestamp: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)

# Validation schemas using Pydantic
class IndicatorNameValidator(BaseModel):
    """Validator for indicator names."""
    name: str = Field(min_length=1, max_length=50, pattern=r'^[a-z][a-z0-9_]*$')

    @field_validator('name')
    @classmethod
    def validate_indicator_name(cls, v):
        from src.indicators.constants import validate_indicator_name
        if not validate_indicator_name(v):
            raise ValueError(f"Unknown indicator name: {v}")
        return v

class TickerValidator(BaseModel):
    """Validator for ticker symbols."""
    ticker: str = Field(min_length=1, max_length=10, pattern=r'^[A-Z][A-Z0-9]*$')

class TimeFrameValidator(BaseModel):
    """Validator for timeframes."""
    timeframe: str = Field(pattern=r'^(1|5|15|30|60)m|1h|4h|1d|1w|1M$')

class PeriodValidator(BaseModel):
    """Validator for periods."""
    period: str = Field(pattern=r'^(\d+)(d|w|m|y)$')

class ParameterValidator(BaseModel):
    """Validator for indicator parameters."""
    timeperiod: Optional[int] = Field(None, ge=1, le=200)
    fastperiod: Optional[int] = Field(None, ge=1, le=100)
    slowperiod: Optional[int] = Field(None, ge=1, le=200)
    signalperiod: Optional[int] = Field(None, ge=1, le=50)
    nbdevup: Optional[float] = Field(None, ge=0.1, le=5.0)
    nbdevdn: Optional[float] = Field(None, ge=0.1, le=5.0)

    @field_validator('slowperiod')
    @classmethod
    def validate_slow_greater_than_fast(cls, v, values):
        if v is not None and 'fastperiod' in values and values['fastperiod'] is not None:
            if v <= values['fastperiod']:
                raise ValueError('slowperiod must be greater than fastperiod')
        return v

# Runtime type checking functions
def is_valid_ohlcv_data(data: Any) -> bool:
    """Check if data is valid OHLCV format."""
    if not isinstance(data, pd.DataFrame):
        return False

    required_columns = {'open', 'high', 'low', 'close', 'volume'}
    return required_columns.issubset(set(data.columns))

def is_valid_indicator_value(value: Any) -> bool:
    """Check if value is a valid indicator value."""
    if isinstance(value, (int, float)):
        return not (np.isnan(value) or np.isinf(value))
    elif isinstance(value, dict):
        return all(
            isinstance(k, str) and isinstance(v, (int, float)) and not (np.isnan(v) or np.isinf(v))
            for k, v in value.items()
        )
    return False

def validate_indicator_parameters(indicator_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize indicator parameters."""
    from src.indicators.constants import get_default_parameters, normalize_parameter_name

    # Get default parameters
    defaults = get_default_parameters(indicator_name)

    # Normalize parameter names
    normalized_params = {}
    for key, value in params.items():
        normalized_key = normalize_parameter_name(indicator_name, key)
        normalized_params[normalized_key] = value

    # Merge with defaults
    final_params = {**defaults, **normalized_params}

    # Validate using Pydantic
    try:
        validator = ParameterValidator(**final_params)
        return validator.dict(exclude_none=True)
    except Exception as e:
        raise ValueError(f"Invalid parameters for {indicator_name}: {e}")

# Type guards for runtime type checking
def is_technical_indicator_result(result: Any) -> bool:
    """Type guard for technical indicator results."""
    return (
        hasattr(result, 'category') and
        result.category == 'technical' and
        hasattr(result, 'value') and
        is_valid_indicator_value(result.value)
    )

def is_fundamental_indicator_result(result: Any) -> bool:
    """Type guard for fundamental indicator results."""
    return (
        hasattr(result, 'category') and
        result.category == 'fundamental' and
        hasattr(result, 'value') and
        is_valid_indicator_value(result.value)
    )

# Abstract base classes for better inheritance
class BaseIndicatorCalculator(ABC):
    """Abstract base class for indicator calculators."""

    @abstractmethod
    async def calculate(
        self,
        data: pd.DataFrame,
        parameters: Dict[str, Any]
    ) -> Union[float, Dict[str, float]]:
        """Calculate indicator value."""
        pass

    @abstractmethod
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data."""
        pass

    @abstractmethod
    def get_required_columns(self) -> List[str]:
        """Get required data columns."""
        pass

class BaseRecommendationEngine(ABC):
    """Abstract base class for recommendation engines."""

    @abstractmethod
    def get_recommendation(
        self,
        indicator_name: str,
        value: Union[float, Dict[str, float]],
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Get recommendation for indicator value."""
        pass

    @abstractmethod
    def get_composite_recommendation(
        self,
        indicators: Dict[str, Any]
    ) -> Any:
        """Get composite recommendation from multiple indicators."""
        pass

# Callback types
IndicatorCalculationCallback = Callable[[str, Union[float, Dict[str, float]]], None]
ErrorCallback = Callable[[str, Exception], None]
ProgressCallback = Callable[[int, int], None]

# Configuration types
class AdapterConfig(TypedDict, total=False):
    """Configuration for indicator adapters."""
    provider: ProviderLiteral
    timeout: int
    retry_count: int
    fallback_providers: List[ProviderLiteral]

class CacheConfig(TypedDict, total=False):
    """Configuration for caching."""
    max_size: int
    ttl: int
    enabled: bool

class ServiceConfig(TypedDict, total=False):
    """Configuration for indicator service."""
    cache: CacheConfig
    adapters: Dict[str, AdapterConfig]
    default_provider: ProviderLiteral
    max_concurrent: int
    timeout: int

# Export all types for easy importing
__all__ = [
    # Type variables
    'T', 'IndicatorValueType',

    # New types
    'TickerSymbol', 'IndicatorName', 'TimeFrame', 'Period', 'ProviderName',

    # Literal types
    'RecommendationLiteral', 'IndicatorCategoryLiteral', 'ProviderLiteral', 'FillMethodLiteral',

    # TypedDict classes
    'OHLCVData', 'IndicatorParameters', 'CalculationContext', 'CacheMetrics', 'ServiceInfo',
    'AdapterConfig', 'CacheConfig', 'ServiceConfig',

    # Protocols
    'IndicatorAdapter', 'RecommendationProvider', 'CacheProvider',

    # Generic types
    'IndicatorResult',

    # Validators
    'IndicatorNameValidator', 'TickerValidator', 'TimeFrameValidator', 'PeriodValidator', 'ParameterValidator',

    # Type checking functions
    'is_valid_ohlcv_data', 'is_valid_indicator_value', 'validate_indicator_parameters',
    'is_technical_indicator_result', 'is_fundamental_indicator_result',

    # Abstract base classes
    'BaseIndicatorCalculator', 'BaseRecommendationEngine',

    # Callback types
    'IndicatorCalculationCallback', 'ErrorCallback', 'ProgressCallback'
]