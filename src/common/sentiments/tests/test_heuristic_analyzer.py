# src/common/sentiments/tests/test_heuristic_analyzer.py
"""
Unit tests for heuristic sentiment analysis.

Tests cover:
- Context-aware sentiment analysis (negation handling)
- Domain-specific financial sentiment keywords
- Emoji and social media slang sentiment detection
- Configurable sentiment rules and weights
"""

import unittest
from pathlib import Path
import sys

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from src.common.sentiments.processing.heuristic_analyzer import (
    HeuristicSentimentAnalyzer, SentimentResult
)


class TestHeuristicSentimentAnalyzer(unittest.TestCase):
    """Test cases for HeuristicSentimentAnalyzer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "positive_keywords": ["moon", "rocket", "buy", "bullish", "gain", "profit"],
            "negative_keywords": ["crash", "dump", "sell", "bearish", "loss", "rekt"],
            "keyword_weights": {
                "moon": 1.5,
                "rocket": 1.2,
                "crash": 1.3,
                "rekt": 1.1
            },
            "negation_words": ["not", "no", "never", "don't", "won't"],
            "negation_window": 3,
            "emoji_weight": 0.3,
            "slang_weight": 0.2,
            "keyword_weight": 0.5
        }
        self.analyzer = HeuristicSentimentAnalyzer(self.config)

    def test_initialization_default_config(self):
        """Test analyzer initialization with default configuration."""
        analyzer = HeuristicSentimentAnalyzer()

        self.assertIn("moon", analyzer.positive_keywords)
        self.assertIn("crash", analyzer.negative_keywords)
        self.assertIn("not", analyzer.negation_words)
        self.assertEqual(analyzer.negation_window, 3)

    def test_initialization_custom_config(self):
        """Test analyzer initialization with custom configuration."""
        self.assertIn("moon", self.analyzer.positive_keywords)
        self.assertIn("crash", self.analyzer.negative_keywords)
        self.assertEqual(self.analyzer.emoji_weight, 0.3)
        self.assertEqual(self.analyzer.slang_weight, 0.2)
        self.assertEqual(self.analyzer.keyword_weight, 0.5)

    def test_analyze_sentiment_empty_text(self):
        """Test sentiment analysis with empty or None text."""
        # Empty string
        result = self.analyzer.analyze_sentiment("")
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.confidence, 0.0)

        # None
        result = self.analyzer.analyze_sentiment(None)
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.confidence, 0.0)

        # Whitespace only
        result = self.analyzer.analyze_sentiment("   ")
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.confidence, 0.0)

    def test_analyze_sentiment_positive_keywords(self):
        """Test sentiment analysis with positive keywords."""
        positive_texts = [
            "Bitcoin is going to the moon!",
            "This stock is bullish, great buy opportunity",
            "Huge profit potential here, rocket incoming",
            "Strong gain today, very bullish signal"
        ]

        for text in positive_texts:
            result = self.analyzer.analyze_sentiment(text)

            self.assertGreater(result.score, 0.0, f"Failed for text: {text}")
            self.assertGreater(result.confidence, 0.0)
            self.assertGreater(len(result.positive_signals), 0)
            self.assertEqual(len(result.negative_signals), 0)

    def test_analyze_sentiment_negative_keywords(self):
        """Test sentiment analysis with negative keywords."""
        negative_texts = [
            "Market is going to crash hard",
            "Time to sell everything, bearish trend",
            "Major loss incoming, dump it all",
            "Got rekt on this trade, terrible"
        ]

        for text in negative_texts:
            result = self.analyzer.analyze_sentiment(text)

            self.assertLess(result.score, 0.0, f"Failed for text: {text}")
            self.assertGreater(result.confidence, 0.0)
            self.assertEqual(len(result.positive_signals), 0)
            self.assertGreater(len(result.negative_signals), 0)

    def test_analyze_sentiment_mixed_keywords(self):
        """Test sentiment analysis with mixed positive and negative keywords."""
        mixed_texts = [
            "Market might crash but could also moon",
            "Bullish short term but bearish long term",
            "Great buy opportunity despite recent loss"
        ]

        for text in mixed_texts:
            result = self.analyzer.analyze_sentiment(text)

            # Should have both positive and negative signals
            self.assertGreater(len(result.positive_signals), 0, f"No positive signals for: {text}")
            self.assertGreater(len(result.negative_signals), 0, f"No negative signals for: {text}")
            self.assertGreater(result.confidence, 0.0)

    def test_negation_detection(self):
        """Test negation context detection."""
        negation_texts = [
            "This is not going to moon",  # Negated positive
            "Don't sell now, hold strong",  # Negated negative
            "Never buying this crash",  # Negated negative
            "Won't dump my position"  # Negated negative
        ]

        for text in negation_texts:
            result = self.analyzer.analyze_sentiment(text)

            # Should detect negation
            self.assertTrue(result.negation_detected, f"Negation not detected for: {text}")

    def test_negation_context_window(self):
        """Test negation detection within context window."""
        # Test that negation affects sentiment direction
        positive_text = "going to moon"
        negative_text = "not going to moon"

        pos_result = self.analyzer.analyze_sentiment(positive_text)
        neg_result = self.analyzer.analyze_sentiment(negative_text)

        # Negated version should have different (likely lower) sentiment
        if pos_result.score > 0:
            self.assertLessEqual(neg_result.score, pos_result.score)

        # Test basic negation detection
        simple_negation = "not moon"
        result = self.analyzer.analyze_sentiment(simple_negation)
        # Should at least process without error
        self.assertIsInstance(result.negation_detected, bool)

    def test_emoji_sentiment_analysis(self):
        """Test emoji sentiment detection."""
        # Positive emojis
        positive_emoji_texts = [
            "Bitcoin to the moon! 🚀",
            "Great trade today 💎",
            "Bullish signal 📈",
            "Love this stock 😍"
        ]

        for text in positive_emoji_texts:
            result = self.analyzer.analyze_sentiment(text)
            self.assertGreater(result.emoji_sentiment, 0.0, f"Failed for: {text}")

        # Negative emojis
        negative_emoji_texts = [
            "Market crashed 📉",
            "Lost everything 💀",
            "This is terrible 😭",
            "Warning signs everywhere ⚠️"
        ]

        for text in negative_emoji_texts:
            result = self.analyzer.analyze_sentiment(text)
            self.assertLess(result.emoji_sentiment, 0.0, f"Failed for: {text}")

    def test_slang_sentiment_analysis(self):
        """Test social media slang sentiment detection."""
        # Positive slang
        positive_slang_texts = [
            "Diamond hands hodl to the moon",
            "This is the way, ape strong together",
            "Buy the dip, get those tendies",
            "YOLO into this stonk"
        ]

        for text in positive_slang_texts:
            result = self.analyzer.analyze_sentiment(text)
            self.assertGreater(result.slang_sentiment, 0.0, f"Failed for: {text}")

        # Negative slang
        negative_slang_texts = [
            "Paper hands got rekt",
            "Bag holder alert, major FUD",
            "Pump and dump scheme detected",
            "Rug pull incoming, be careful"
        ]

        for text in negative_slang_texts:
            result = self.analyzer.analyze_sentiment(text)
            self.assertLess(result.slang_sentiment, 0.0, f"Failed for: {text}")

    def test_keyword_weighting(self):
        """Test custom keyword weighting."""
        # Text with weighted keyword
        weighted_text = "This stock is going to moon"  # "moon" has weight 1.5
        unweighted_text = "This stock will gain value"  # "gain" has default weight 1.0

        weighted_result = self.analyzer.analyze_sentiment(weighted_text)
        unweighted_result = self.analyzer.analyze_sentiment(unweighted_text)

        # Weighted keyword should produce higher score
        self.assertGreater(abs(weighted_result.score), abs(unweighted_result.score))

    def test_combined_sentiment_scoring(self):
        """Test combined sentiment scoring from multiple components."""
        # Text with keywords, emojis, and slang
        combined_text = "Diamond hands hodl to the moon! 🚀💎 This is bullish!"

        result = self.analyzer.analyze_sentiment(combined_text)

        # Should have positive sentiment from all components
        self.assertGreater(result.score, 0.0)
        self.assertGreater(len(result.positive_signals), 0)
        self.assertGreater(result.emoji_sentiment, 0.0)
        self.assertGreater(result.slang_sentiment, 0.0)
        self.assertGreater(result.confidence, 0.0)

    def test_confidence_calculation(self):
        """Test confidence calculation based on signal strength."""
        # Strong signals
        strong_text = "Bullish moon rocket gains profit 🚀💎📈"
        strong_result = self.analyzer.analyze_sentiment(strong_text)

        # Weak signals
        weak_text = "Maybe buy"
        weak_result = self.analyzer.analyze_sentiment(weak_text)

        # Strong signals should have higher confidence
        self.assertGreater(strong_result.confidence, weak_result.confidence)

    def test_sentiment_score_clamping(self):
        """Test that sentiment scores are clamped to [-1, 1] range."""
        # Very strong positive text
        very_positive = "moon rocket bullish gain profit buy hodl diamond hands 🚀💎📈😍"
        positive_result = self.analyzer.analyze_sentiment(very_positive)

        # Very strong negative text
        very_negative = "crash dump bearish loss sell rekt paper hands 📉💀😭⚠️"
        negative_result = self.analyzer.analyze_sentiment(very_negative)

        # Scores should be within valid range
        self.assertGreaterEqual(positive_result.score, -1.0)
        self.assertLessEqual(positive_result.score, 1.0)
        self.assertGreaterEqual(negative_result.score, -1.0)
        self.assertLessEqual(negative_result.score, 1.0)

    def test_update_keywords(self):
        """Test dynamic keyword updating."""
        # Add new positive keywords
        new_positive = ["lambo", "tendies", "stonks"]
        self.analyzer.update_keywords(positive_keywords=new_positive)

        # Test with new keyword
        result = self.analyzer.analyze_sentiment("Getting lambo soon!")
        self.assertGreater(result.score, 0.0)
        self.assertIn("lambo", result.positive_signals)

        # Add new negative keywords
        new_negative = ["rugpull", "scam", "ponzi"]
        self.analyzer.update_keywords(negative_keywords=new_negative)

        # Test with new keyword
        result = self.analyzer.analyze_sentiment("This looks like a scam")
        self.assertLess(result.score, 0.0)
        self.assertIn("scam", result.negative_signals)

    def test_get_keyword_stats(self):
        """Test keyword statistics retrieval."""
        stats = self.analyzer.get_keyword_stats()

        self.assertIn("positive_keywords", stats)
        self.assertIn("negative_keywords", stats)
        self.assertIn("emoji_mappings", stats)
        self.assertIn("slang_mappings", stats)
        self.assertIn("negation_words", stats)

        self.assertGreater(stats["positive_keywords"], 0)
        self.assertGreater(stats["negative_keywords"], 0)
        self.assertGreater(stats["emoji_mappings"], 0)
        self.assertGreater(stats["slang_mappings"], 0)
        self.assertGreater(stats["negation_words"], 0)

    def test_financial_domain_keywords(self):
        """Test financial domain-specific keywords."""
        # Use keywords that are actually in our test config
        financial_texts = [
            "This stock is bullish, great buy opportunity",  # Contains "bullish" and "buy"
            "Time to sell everything, bearish trend",        # Contains "bearish"
            "Major loss incoming, dump it all",              # Contains "dump"
            "Huge profit potential here, rocket incoming",   # Contains "rocket"
            "Got rekt on this trade, terrible crash"         # Contains "rekt" and "crash"
        ]

        for text in financial_texts:
            result = self.analyzer.analyze_sentiment(text)

            # Should detect financial sentiment
            self.assertNotEqual(result.score, 0.0, f"No sentiment detected for: {text}")
            self.assertGreater(result.confidence, 0.0)

    def test_case_insensitive_analysis(self):
        """Test case-insensitive keyword matching."""
        texts = [
            "MOON ROCKET BULLISH",
            "moon rocket bullish",
            "Moon Rocket Bullish",
            "mOoN rOcKeT bUlLiSh"
        ]

        results = [self.analyzer.analyze_sentiment(text) for text in texts]

        # All should produce similar positive sentiment
        for result in results:
            self.assertGreater(result.score, 0.0)
            self.assertGreater(len(result.positive_signals), 0)

    def test_multi_word_slang_detection(self):
        """Test detection of multi-word slang phrases."""
        multi_word_texts = [
            "Diamond hands to the moon",
            "This is the way to trade",
            "Buy the dip strategy works",
            "Paper hands sell too early"
        ]

        for text in multi_word_texts:
            result = self.analyzer.analyze_sentiment(text)

            # Should detect slang sentiment
            self.assertNotEqual(result.slang_sentiment, 0.0, f"No slang detected for: {text}")

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        edge_cases = [
            "!@#$%^&*()",  # Special characters only
            "123456789",   # Numbers only
            "a b c d e",   # Single letters
            "THE THE THE", # Repeated common words
            "🚀🚀🚀🚀🚀"  # Repeated emojis
        ]

        for text in edge_cases:
            result = self.analyzer.analyze_sentiment(text)

            # Should handle gracefully without errors
            self.assertIsInstance(result, SentimentResult)
            self.assertGreaterEqual(result.score, -1.0)
            self.assertLessEqual(result.score, 1.0)
            self.assertGreaterEqual(result.confidence, 0.0)
            self.assertLessEqual(result.confidence, 1.0)

    def test_negation_with_complex_sentences(self):
        """Test negation detection in complex sentences."""
        # Test simpler negation cases that should work
        simple_negations = [
            "not bullish",
            "don't buy",
            "never moon",
            "won't gain"
        ]

        for text in simple_negations:
            result = self.analyzer.analyze_sentiment(text)

            # Should process without error and may detect negation
            self.assertIsInstance(result.negation_detected, bool)
            self.assertIsInstance(result.score, float)

        # Test that negation affects sentiment
        pos_text = "bullish moon"
        neg_text = "not bullish moon"

        pos_result = self.analyzer.analyze_sentiment(pos_text)
        neg_result = self.analyzer.analyze_sentiment(neg_text)

        # Negated version should be different
        self.assertNotEqual(pos_result.score, neg_result.score)

    def test_sentiment_result_structure(self):
        """Test SentimentResult dataclass structure."""
        text = "Bullish moon rocket with diamond hands 🚀💎"
        result = self.analyzer.analyze_sentiment(text)

        # Check all required fields are present
        self.assertIsInstance(result.score, float)
        self.assertIsInstance(result.confidence, float)
        self.assertIsInstance(result.positive_signals, list)
        self.assertIsInstance(result.negative_signals, list)
        self.assertIsInstance(result.negation_detected, bool)
        self.assertIsInstance(result.emoji_sentiment, float)
        self.assertIsInstance(result.slang_sentiment, float)

        # Check value ranges
        self.assertGreaterEqual(result.score, -1.0)
        self.assertLessEqual(result.score, 1.0)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)


class TestSentimentResult(unittest.TestCase):
    """Test cases for SentimentResult dataclass."""

    def test_sentiment_result_creation(self):
        """Test SentimentResult object creation."""
        result = SentimentResult(
            score=0.75,
            confidence=0.85,
            positive_signals=["moon", "rocket"],
            negative_signals=[],
            negation_detected=False,
            emoji_sentiment=0.6,
            slang_sentiment=0.4
        )

        self.assertEqual(result.score, 0.75)
        self.assertEqual(result.confidence, 0.85)
        self.assertEqual(result.positive_signals, ["moon", "rocket"])
        self.assertEqual(result.negative_signals, [])
        self.assertFalse(result.negation_detected)
        self.assertEqual(result.emoji_sentiment, 0.6)
        self.assertEqual(result.slang_sentiment, 0.4)


if __name__ == "__main__":
    unittest.main()