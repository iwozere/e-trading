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

from .heuristic_analyzer import HeuristicSentimentAnalyzer, SentimentResult
from .enhanced_hf_analyzer import EnhancedHFAnalyzer, SentimentPrediction, ContentType
from .bot_detector import BotDetector, BotDetectionResult, UserProfile, PostMetrics
from .virality_calculator import ViralityCalculator, ViralityResult, PostData, EngagementMetrics, AuthorInfluence, Platform
from .sentiment_aggregator import SentimentAggregator, AggregatedSentiment, SourceSentiment

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
    "SourceSentiment"
]