# src/common/sentiments/processing/__init__.py
"""
Sentiment processing and analysis components.

This package contains enhanced sentiment analysis algorithms including:
- Advanced heuristic sentiment analysis with context awareness
- Enhanced HuggingFace integration with multiple models
- Bot detection algorithms
- Virality and engagement metrics
- Sentiment aggregation and weighting strategies
"""

from .bot_detector import BotDetectionResult, BotDetector, PostMetrics, UserProfile
from .enhanced_hf_analyzer import ContentType, EnhancedHFAnalyzer, SentimentPrediction
from .heuristic_analyzer import HeuristicSentimentAnalyzer, SentimentResult
from .sentiment_aggregator import AggregatedSentiment, SentimentAggregator, SourceSentiment
from .virality_calculator import (
    AuthorInfluence,
    EngagementMetrics,
    Platform,
    PostData,
    ViralityCalculator,
    ViralityResult,
)

__all__ = [
    "HeuristicSentimentAnalyzer",
    "SentimentResult",
    "EnhancedHFAnalyzer",
    "SentimentPrediction",
    "ContentType",
    "BotDetector",
    "BotDetectionResult",
    "UserProfile",
    "PostMetrics",
    "ViralityCalculator",
    "ViralityResult",
    "PostData",
    "EngagementMetrics",
    "AuthorInfluence",
    "Platform",
    "SentimentAggregator",
    "AggregatedSentiment",
    "SourceSentiment",
]
