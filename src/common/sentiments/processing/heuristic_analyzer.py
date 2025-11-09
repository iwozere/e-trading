# src/common/sentiments/processing/heuristic_analyzer.py
"""
Enhanced heuristic sentiment analysis with context awareness and financial domain expertise.

This module provides sophisticated keyword-based sentiment detection with:
- Context-aware sentiment analysis (negation handling)
- Domain-specific financial sentiment keywords
- Emoji and social media slang sentiment detection
- Configurable sentiment rules and weights
"""

import re
from typing import Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path
import sys

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

@dataclass
class SentimentResult:
    """Result of heuristic sentiment analysis."""
    score: float  # -1.0 to 1.0
    confidence: float  # 0.0 to 1.0
    positive_signals: List[str]
    negative_signals: List[str]
    negation_detected: bool
    emoji_sentiment: float
    slang_sentiment: float


class HeuristicSentimentAnalyzer:
    """
    Enhanced heuristic sentiment analyzer with context awareness and financial domain expertise.

    Features:
    - Expanded financial sentiment keywords
    - Context-aware negation handling
    - Emoji and social media slang detection
    - Configurable sentiment weights and rules
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the heuristic sentiment analyzer.

        Args:
            config: Configuration dictionary with sentiment rules and keywords
        """
        self.config = config or {}

        # Load sentiment keywords and rules
        self._load_sentiment_keywords()
        self._load_negation_patterns()
        self._load_emoji_mappings()
        self._load_slang_mappings()

        # Configuration parameters
        self.negation_window = self.config.get("negation_window", 3)
        self.emoji_weight = self.config.get("emoji_weight", 0.3)
        self.slang_weight = self.config.get("slang_weight", 0.2)
        self.keyword_weight = self.config.get("keyword_weight", 0.5)

    def _load_sentiment_keywords(self) -> None:
        """Load financial domain-specific sentiment keywords."""
        # Enhanced positive keywords for financial context
        default_positive = [
            # Basic positive
            "moon", "rocket", "diamond", "buy", "long", "hold", "bullish", "bull",
            "up", "rise", "gain", "profit", "win", "winning", "strong", "solid",

            # Financial positive
            "breakout", "rally", "surge", "pump", "squeeze", "momentum", "uptrend",
            "support", "resistance", "bounce", "recovery", "growth", "earnings beat",
            "upgrade", "outperform", "overweight", "accumulate", "target raised",

            # Social media positive
            "to the moon", "diamond hands", "hodl", "ape", "lambo", "tendies",
            "stonks", "this is the way", "buy the dip", "btfd", "yolo",

            # Emojis as text
            "🚀", "💎", "🌙", "📈", "💰", "🔥", "💪", "👍", "✅", "🎯"
        ]

        # Enhanced negative keywords for financial context
        default_negative = [
            # Basic negative
            "short", "sell", "dump", "crash", "fall", "drop", "down", "bearish", "bear",
            "loss", "lose", "losing", "weak", "bad", "terrible", "awful",

            # Financial negative
            "breakdown", "collapse", "plunge", "tank", "correction", "pullback",
            "downtrend", "resistance", "rejection", "decline", "selloff", "panic",
            "earnings miss", "downgrade", "underperform", "underweight", "reduce",
            "target lowered", "bankruptcy", "delisting", "fraud", "investigation",

            # Social media negative
            "paper hands", "bag holder", "bagholder", "rekt", "fud", "shill",
            "pump and dump", "rug pull", "dead cat bounce", "falling knife",

            # Emojis as text
            "📉", "💀", "🔻", "❌", "😭", "😱", "🤡", "💩", "⚠️", "🚨"
        ]

        # Load from config or use defaults
        self.positive_keywords = set(self.config.get("positive_keywords", default_positive))
        self.negative_keywords = set(self.config.get("negative_keywords", default_negative))

        # Create weighted keyword mappings
        self.keyword_weights = self.config.get("keyword_weights", {})

        _logger.debug("Loaded %d positive and %d negative keywords",
                     len(self.positive_keywords), len(self.negative_keywords))

    def _load_negation_patterns(self) -> None:
        """Load negation patterns for context-aware analysis."""
        default_negations = [
            "not", "no", "never", "none", "nothing", "nowhere", "neither", "nor",
            "don't", "doesn't", "didn't", "won't", "wouldn't", "can't", "cannot",
            "couldn't", "shouldn't", "mustn't", "isn't", "aren't", "wasn't", "weren't",
            "haven't", "hasn't", "hadn't", "without", "lack", "lacking", "absent",
            "fail", "failed", "failing", "unable", "impossible", "hardly", "barely",
            "scarcely", "seldom", "rarely"
        ]

        self.negation_words = set(self.config.get("negation_words", default_negations))

        # Compile negation patterns
        negation_pattern = r'\b(?:' + '|'.join(re.escape(word) for word in self.negation_words) + r')\b'
        self.negation_regex = re.compile(negation_pattern, re.IGNORECASE)

    def _load_emoji_mappings(self) -> None:
        """Load emoji sentiment mappings."""
        default_emoji_sentiment = {
            # Very positive
            "🚀": 1.0, "💎": 0.9, "🌙": 0.8, "📈": 0.9, "💰": 0.8,
            "🔥": 0.7, "💪": 0.7, "👍": 0.6, "✅": 0.6, "🎯": 0.7,
            "😍": 0.8, "🤩": 0.8, "😎": 0.6, "🥳": 0.8, "🎉": 0.7,

            # Positive
            "😊": 0.5, "😄": 0.6, "😃": 0.5, "🙂": 0.4, "👌": 0.5,
            "💯": 0.7, "⭐": 0.6, "🌟": 0.6, "⚡": 0.6, "🎊": 0.6,

            # Negative
            "😢": -0.6, "😭": -0.7, "😱": -0.8, "😰": -0.6, "😨": -0.7,
            "🤡": -0.8, "💩": -0.9, "😡": -0.8, "🤬": -0.9, "😤": -0.6,

            # Very negative
            "📉": -0.9, "💀": -1.0, "🔻": -0.8, "❌": -0.7, "⚠️": -0.6,
            "🚨": -0.8, "💸": -0.7, "🔥": -0.5,  # Fire can be negative in "money burning" context
        }

        self.emoji_sentiment = self.config.get("emoji_sentiment", default_emoji_sentiment)

    def _load_slang_mappings(self) -> None:
        """Load social media slang sentiment mappings."""
        default_slang_sentiment = {
            # Very positive slang
            "hodl": 0.8, "diamond hands": 0.9, "to the moon": 1.0, "lambo": 0.8,
            "tendies": 0.7, "stonks": 0.6, "this is the way": 0.7, "ape": 0.6,
            "btfd": 0.8, "buy the dip": 0.8, "yolo": 0.5, "fomo": 0.3,

            # Positive slang
            "bullish af": 0.9, "moon mission": 0.9, "rocket fuel": 0.8,
            "diamond handed": 0.8, "ape strong": 0.7, "hold the line": 0.7,

            # Negative slang
            "paper hands": -0.8, "bag holder": -0.7, "bagholder": -0.7,
            "rekt": -0.9, "fud": -0.6, "shill": -0.7, "cope": -0.5,

            # Very negative slang
            "rug pull": -1.0, "pump and dump": -0.9, "exit scam": -1.0,
            "dead cat bounce": -0.8, "falling knife": -0.8, "bag holding": -0.7,
        }

        self.slang_sentiment = self.config.get("slang_sentiment", default_slang_sentiment)

        # Create regex patterns for multi-word slang
        self.slang_patterns = {}
        for phrase, sentiment in self.slang_sentiment.items():
            if len(phrase.split()) > 1:
                pattern = re.compile(r'\b' + re.escape(phrase) + r'\b', re.IGNORECASE)
                self.slang_patterns[pattern] = sentiment

    def analyze_sentiment(self, text: str) -> SentimentResult:
        """
        Analyze sentiment of text using enhanced heuristic methods.

        Args:
            text: Text to analyze for sentiment

        Returns:
            SentimentResult with detailed sentiment analysis
        """
        if not text or not text.strip():
            return SentimentResult(0.0, 0.0, [], [], False, 0.0, 0.0)

        text_lower = text.lower().strip()

        # Detect sentiment signals
        positive_signals = self._find_positive_signals(text_lower)
        negative_signals = self._find_negative_signals(text_lower)

        # Analyze emoji sentiment
        emoji_sentiment = self._analyze_emoji_sentiment(text)

        # Analyze slang sentiment
        slang_sentiment = self._analyze_slang_sentiment(text_lower)

        # Detect negation context
        negation_detected = self._detect_negation_context(text_lower, positive_signals + negative_signals)

        # Calculate base sentiment score
        pos_score = sum(self._get_keyword_weight(signal) for signal in positive_signals)
        neg_score = sum(self._get_keyword_weight(signal) for signal in negative_signals)

        # Combine different sentiment components
        keyword_sentiment = pos_score - neg_score

        # Apply negation if detected
        if negation_detected:
            keyword_sentiment *= -0.8  # Flip and slightly reduce intensity

        # Weighted combination of sentiment components
        final_score = (
            self.keyword_weight * keyword_sentiment +
            self.emoji_weight * emoji_sentiment +
            self.slang_weight * slang_sentiment
        )

        # Normalize to [-1, 1] range
        final_score = max(-1.0, min(1.0, final_score))

        # Calculate confidence based on signal strength
        signal_count = len(positive_signals) + len(negative_signals)
        confidence = min(1.0, signal_count * 0.2 + abs(emoji_sentiment) * 0.3 + abs(slang_sentiment) * 0.3)

        return SentimentResult(
            score=final_score,
            confidence=confidence,
            positive_signals=positive_signals,
            negative_signals=negative_signals,
            negation_detected=negation_detected,
            emoji_sentiment=emoji_sentiment,
            slang_sentiment=slang_sentiment
        )

    def _find_positive_signals(self, text: str) -> List[str]:
        """Find positive sentiment signals in text."""
        signals = []
        for keyword in self.positive_keywords:
            if keyword.lower() in text:
                signals.append(keyword)
        return signals

    def _find_negative_signals(self, text: str) -> List[str]:
        """Find negative sentiment signals in text."""
        signals = []
        for keyword in self.negative_keywords:
            if keyword.lower() in text:
                signals.append(keyword)
        return signals

    def _analyze_emoji_sentiment(self, text: str) -> float:
        """Analyze emoji sentiment in text."""
        total_sentiment = 0.0
        emoji_count = 0

        for emoji, sentiment in self.emoji_sentiment.items():
            count = text.count(emoji)
            if count > 0:
                total_sentiment += sentiment * count
                emoji_count += count

        return total_sentiment / max(1, emoji_count) if emoji_count > 0 else 0.0

    def _analyze_slang_sentiment(self, text: str) -> float:
        """Analyze social media slang sentiment in text."""
        total_sentiment = 0.0
        slang_count = 0

        # Check multi-word slang patterns
        for pattern, sentiment in self.slang_patterns.items():
            matches = pattern.findall(text)
            if matches:
                total_sentiment += sentiment * len(matches)
                slang_count += len(matches)

        # Check single-word slang
        words = text.split()
        for word in words:
            if word in self.slang_sentiment:
                total_sentiment += self.slang_sentiment[word]
                slang_count += 1

        return total_sentiment / max(1, slang_count) if slang_count > 0 else 0.0

    def _detect_negation_context(self, text: str, signals: List[str]) -> bool:
        """
        Detect if sentiment signals appear in negation context.

        Args:
            text: Text to analyze
            signals: List of sentiment signals found

        Returns:
            True if negation context detected
        """
        if not signals:
            return False

        words = text.split()

        for signal in signals:
            signal_words = signal.split()

            # Find signal position in text
            for i in range(len(words) - len(signal_words) + 1):
                if ' '.join(words[i:i+len(signal_words)]) == signal:
                    # Check negation window before signal
                    start_idx = max(0, i - self.negation_window)
                    context_words = words[start_idx:i]

                    # Check if any negation words in context
                    for word in context_words:
                        if word in self.negation_words:
                            return True

        return False

    def _get_keyword_weight(self, keyword: str) -> float:
        """Get weight for a specific keyword."""
        return self.keyword_weights.get(keyword, 1.0)

    def update_keywords(self, positive_keywords: Optional[List[str]] = None,
                       negative_keywords: Optional[List[str]] = None) -> None:
        """
        Update sentiment keywords dynamically.

        Args:
            positive_keywords: New positive keywords to add
            negative_keywords: New negative keywords to add
        """
        if positive_keywords:
            self.positive_keywords.update(positive_keywords)
            _logger.info("Added %d positive keywords", len(positive_keywords))

        if negative_keywords:
            self.negative_keywords.update(negative_keywords)
            _logger.info("Added %d negative keywords", len(negative_keywords))

    def get_keyword_stats(self) -> Dict[str, int]:
        """Get statistics about loaded keywords."""
        return {
            "positive_keywords": len(self.positive_keywords),
            "negative_keywords": len(self.negative_keywords),
            "emoji_mappings": len(self.emoji_sentiment),
            "slang_mappings": len(self.slang_sentiment),
            "negation_words": len(self.negation_words)
        }