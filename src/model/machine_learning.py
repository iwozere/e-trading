"""
Models for machine learning and feature engineering.

Includes:
- Feature types, training triggers, and model types enums
- Feature, model metadata, training config, and performance metrics dataclasses
"""
from enum import Enum
from typing import Any, Dict, List
from dataclasses import dataclass
from datetime import datetime

class FeatureType(Enum):
    """Types of features that can be generated."""
    TECHNICAL_INDICATOR = "technical_indicator"
    MARKET_MICROSTRUCTURE = "market_microstructure"
    STATISTICAL = "statistical"
    TIME_BASED = "time_based"
    CROSS_ASSET = "cross_asset"

class TrainingTrigger(Enum):
    """Types of training triggers."""
    SCHEDULED = "scheduled"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    DATA_DRIFT = "data_drift"
    MANUAL = "manual"


class ModelType(Enum):
    """Supported model types."""
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    LINEAR_REGRESSION = "linear_regression"

@dataclass
class FeatureConfig:
    """Configuration for feature generation."""
    name: str
    feature_type: FeatureType
    parameters: Dict[str, Any]
    description: str
    enabled: bool = True


@dataclass
class ModelMetadata:
    """Metadata for model tracking and registry."""
    model_name: str
    version: str
    model_type: str  # 'sklearn', 'pytorch', 'tensorflow', 'xgboost', 'lightgbm'
    framework_version: str
    created_at: datetime
    author: str
    description: str
    tags: Dict[str, str]
    hyperparameters: Dict[str, Any]
    metrics: Dict[str, float]
    feature_names: List[str]
    target_column: str
    data_version: str
    git_commit: str


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    model_type: ModelType
    hyperparameters: Dict[str, Any]
    training_schedule: str  # Cron expression
    performance_threshold: float
    retrain_threshold: float
    max_training_time: int  # minutes
    validation_split: float
    cross_validation_folds: int
    feature_selection_method: str
    n_features: int
    scaler_type: str


@dataclass
class PerformanceMetrics:
    """Performance metrics for model evaluation."""
    mse: float
    mae: float
    r2: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    timestamp: datetime
