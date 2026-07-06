import sys
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Dict, Any, List
from datetime import datetime

# Set up project root imports using pathlib
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.indicators.models import (
    IndicatorSet,
    IndicatorResult,
    Recommendation,
    RecommendationType,
    IndicatorCategory,
)
from src.common.recommendation.engine import RecommendationEngine

def test_engine():
    engine = RecommendationEngine()
    
    # Create indicator set where one indicator result has recommendation = None
    indicator_set = IndicatorSet(
        ticker="AAPL",
        technical_indicators={
            "RSI": IndicatorResult(
                name="RSI",
                value=50.0,
                recommendation=None, # This is the case that fails!
                category=IndicatorCategory.TECHNICAL,
                source="test"
            ),
            "MACD": IndicatorResult(
                name="MACD",
                value=1.5,
                recommendation=Recommendation(
                    recommendation=RecommendationType.BUY,
                    confidence=0.8,
                    reason="MACD Bullish Cross"
                ),
                category=IndicatorCategory.TECHNICAL,
                source="test"
            )
        }
    )
    
    composite = engine.get_composite_recommendation(indicator_set)
    print("Composite recommendation computed successfully!")
    print(f"Rec: {composite.recommendation}, Confidence: {composite.confidence}")

if __name__ == "__main__":
    test_engine()
