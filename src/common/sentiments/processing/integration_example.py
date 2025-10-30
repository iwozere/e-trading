# src/common/sentiments/processing/integration_example.py
"""
Example demonstrating integration of enhanced sentiment processing components.

This example shows how to use the new processing modules together:
- HeuristicSentimentAnalyzer for enhanced keyword-based analysis
- EnhancedHFAnalyzer for multi-model ML sentiment analysis
- BotDetector for identifying automated content
- ViralityCalculator for engagement and virality metrics
- SentimentAggregator for combining multiple sentiment sources
"""

import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any
from pathlib import Path
import sys

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from src.notification.logger import setup_logger
from . import (
    HeuristicSentimentAnalyzer, SentimentResult,
    EnhancedHFAnalyzer, SentimentPrediction, ContentType,
    BotDetector, BotDetectionResult, UserProfile, PostMetrics,
    ViralityCalculator, ViralityResult, PostData, EngagementMetrics, AuthorInfluence, Platform,
    SentimentAggregator, AggregatedSentiment, SourceSentiment
)

_logger = setup_logger(__name__)

async def demonstrate_enhanced_processing():
    """Demonstrate the enhanced sentiment processing pipeline."""

    # Sample data for demonstration
    sample_texts = [
        "AAPL to the moon! 🚀 Diamond hands holding strong! #AAPL #ToTheMoon",
        "Bearish on AAPL, expecting a major correction soon. Selling my position.",
        "AAPL earnings beat expectations! Strong buy signal here 📈",
        "Not sure about AAPL direction, mixed signals in the market",
        "AAPL pump and dump scheme, be careful! This looks like manipulation."
    ]

    sample_users = [
        UserProfile("trader_pro_2024", 45, 1200, 5000, 500, True, False, 150),
        UserProfile("bear_market_guru", 120, 800, 15000, 200, False, False, 80),
        UserProfile("earnings_watcher", 200, 2000, 8000, 1000, True, False, 120),
        UserProfile("market_newbie", 15, 50, 100, 2000, False, True, 20),
        UserProfile("pump_bot_123", 5, 500, 50, 5000, False, True, 10)
    ]

    print("=== Enhanced Sentiment Processing Demo ===\n")

    # 1. Heuristic Sentiment Analysis
    print("1. Enhanced Heuristic Analysis:")
    heuristic_analyzer = HeuristicSentimentAnalyzer()

    heuristic_results = []
    for i, text in enumerate(sample_texts):
        result = heuristic_analyzer.analyze_sentiment(text)
        heuristic_results.append(result)
        print(f"   Text {i+1}: Score={result.score:.3f}, Confidence={result.confidence:.3f}")
        print(f"           Positive: {result.positive_signals}")
        print(f"           Negative: {result.negative_signals}")
        print(f"           Negation: {result.negation_detected}")
        print(f"           Emoji: {result.emoji_sentiment:.3f}, Slang: {result.slang_sentiment:.3f}")
        print()

    # 2. Enhanced HuggingFace Analysis (if available)
    print("2. Enhanced HuggingFace Analysis:")
    try:
        hf_analyzer = EnhancedHFAnalyzer({
            "models": {
                "financial": {
                    "model_path": "ProsusAI/finbert",
                    "content_types": ["financial"],
                    "batch_size": 4
                }
            }
        })

        hf_results = await hf_analyzer.predict_batch(sample_texts, ContentType.FINANCIAL)

        for i, result in enumerate(hf_results):
            print(f"   Text {i+1}: Label={result.label}, Score={result.score:.3f}")
            print(f"           Confidence={result.confidence:.3f}, Model={result.model_used}")

        await hf_analyzer.close()

    except Exception as e:
        print(f"   HuggingFace analysis not available: {e}")
        hf_results = []

    print()

    # 3. Bot Detection Analysis
    print("3. Bot Detection Analysis:")
    bot_detector = BotDetector()

    # Create sample post metrics
    sample_posts = []
    for i, (text, user) in enumerate(zip(sample_texts, sample_users)):
        post = PostMetrics(
            timestamp=datetime.now(timezone.utc) - timedelta(hours=i),
            content=text,
            likes=max(0, 100 - i * 20),
            replies=max(0, 10 - i * 2),
            retweets=max(0, 50 - i * 10),
            content_hash=f"hash_{i}"
        )
        sample_posts.append(post)

    bot_results = []
    for i, (user, posts) in enumerate(zip(sample_users, [[post] for post in sample_posts])):
        result = bot_detector.analyze_user(user, posts)
        bot_results.append(result)
        print(f"   User {i+1} ({user.username}): Bot={result.is_bot}, Score={result.bot_score:.3f}")
        print(f"           Confidence={result.confidence:.3f}")
        print(f"           Reasons: {result.detection_reasons}")
        print()

    # 4. Virality Analysis
    print("4. Virality Analysis:")
    virality_calculator = ViralityCalculator()

    # Create PostData objects for virality analysis
    post_data_list = []
    for i, (text, user, post) in enumerate(zip(sample_texts, sample_users, sample_posts)):
        author_influence = AuthorInfluence(
            username=user.username,
            followers_count=user.followers_count or 0,
            following_count=user.following_count or 0,
            verified=user.verified,
            account_age_days=user.account_age_days,
            total_posts=user.total_posts or 0
        )

        engagement = EngagementMetrics(
            likes=post.likes,
            replies=post.replies,
            retweets=post.retweets
        )

        post_data = PostData(
            id=f"post_{i}",
            content=text,
            author=author_influence,
            timestamp=post.timestamp,
            engagement=engagement,
            platform=Platform.TWITTER,
            hashtags=["AAPL", "ToTheMoon"] if "moon" in text.lower() else ["AAPL"]
        )
        post_data_list.append(post_data)

    virality_result = virality_calculator.calculate_virality(post_data_list, "AAPL")

    print(f"   Virality Index: {virality_result.virality_index:.3f}")
    print(f"   Engagement Score: {virality_result.engagement_score:.3f}")
    print(f"   Velocity Score: {virality_result.velocity_score:.3f}")
    print(f"   Reach Score: {virality_result.reach_score:.3f}")
    print(f"   Influence Score: {virality_result.influence_score:.3f}")
    print(f"   Trending Score: {virality_result.trending_score:.3f}")
    print(f"   Top Contributors: {virality_result.top_contributors[:3]}")
    print()

    # 5. Sentiment Aggregation
    print("5. Sentiment Aggregation:")
    aggregator = SentimentAggregator()

    # Create source sentiments from different analyses
    sources = []

    # Add heuristic results
    if heuristic_results:
        avg_heuristic = sum(r.score for r in heuristic_results) / len(heuristic_results)
        avg_confidence = sum(r.confidence for r in heuristic_results) / len(heuristic_results)

        sources.append(aggregator.create_source_sentiment(
            "heuristic", avg_heuristic, avg_confidence, "good", len(heuristic_results)
        ))

    # Add HF results if available
    if hf_results:
        avg_hf = sum(r.score for r in hf_results) / len(hf_results)
        avg_hf_confidence = sum(r.confidence for r in hf_results) / len(hf_results)

        sources.append(aggregator.create_source_sentiment(
            "huggingface", avg_hf, avg_hf_confidence, "excellent", len(hf_results)
        ))

    # Add virality-weighted sentiment
    virality_weighted_sentiment = 0.0
    for i, result in enumerate(heuristic_results):
        weight = post_data_list[i].engagement.total_engagement() + 1
        virality_weighted_sentiment += result.score * weight

    if post_data_list:
        total_weight = sum(p.engagement.total_engagement() + 1 for p in post_data_list)
        virality_weighted_sentiment /= total_weight

        sources.append(aggregator.create_source_sentiment(
            "virality_weighted", virality_weighted_sentiment, 0.8, "good", len(post_data_list)
        ))

    # Aggregate all sources
    if sources:
        final_result = aggregator.aggregate_sentiment(sources)

        print(f"   Final Sentiment Score: {final_result.final_score:.3f}")
        print(f"   Confidence: {final_result.confidence:.3f}")
        print(f"   Quality Score: {final_result.quality_score:.3f}")
        print(f"   Confidence Interval: ({final_result.confidence_interval[0]:.3f}, {final_result.confidence_interval[1]:.3f})")
        print(f"   Temporal Trend: {final_result.temporal_trend}")
        print(f"   Aggregation Method: {final_result.aggregation_method}")
        print(f"   Source Breakdown:")
        for source, breakdown in final_result.source_breakdown.items():
            print(f"     {source}: Score={breakdown['sentiment_score']:.3f}, Weight={breakdown['weight']:.3f}")

    print("\n=== Demo Complete ===")

if __name__ == "__main__":
    asyncio.run(demonstrate_enhanced_processing())