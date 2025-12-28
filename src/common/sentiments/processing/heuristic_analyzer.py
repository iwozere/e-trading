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
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import sys
from urllib.parse import urlparse

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

CONFIG_PATH = PROJECT_ROOT / "config" / "sentiments" / "sentiments.json"

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
        self.config = config or self._load_default_config()

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

    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration from JSON file or fallback to internal defaults."""
        try:
            if CONFIG_PATH.exists():
                with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                _logger.warning("Sentiment config not found at %s, using minimal defaults", CONFIG_PATH)
                return {}
        except Exception as e:
            _logger.error("Error loading sentiment config: %s", e)
            return {}

    def _load_sentiment_keywords(self) -> None:
        """Load financial domain-specific sentiment keywords."""
        # Load from config or use internal defaults (abbreviated here for brevity, assuming sentiments.json is primary)
        self.positive_keywords = set(self.config.get("positive_keywords", ["moon", "rocket", "buy", "long", "bull"]))
        self.negative_keywords = set(self.config.get("negative_keywords", ["short", "sell", "dump", "crash", "bear"]))

        # Create weighted keyword mappings
        self.keyword_weights = self.config.get("keyword_weights", {})

        _logger.debug("Loaded %d positive and %d negative keywords",
                     len(self.positive_keywords), len(self.negative_keywords))

    def _load_negation_patterns(self) -> None:
        """Load negation patterns for context-aware analysis."""
        default_negations = ["not", "no", "never", "don't", "doesn't", "isn't", "wasn't"]
        self.negation_words = set(self.config.get("negation_words", default_negations))

        # Compile negation patterns
        negation_pattern = r'\b(?:' + '|'.join(re.escape(word) for word in self.negation_words) + r')\b'
        self.negation_regex = re.compile(negation_pattern, re.IGNORECASE)

    def _load_emoji_mappings(self) -> None:
        """Load emoji sentiment mappings."""
        # Default emoji sentiment if not in config
        default_emoji_sentiment = {
            "🚀": 1.0, "💎": 0.9, "🌙": 0.8, "📈": 0.9, "📉": -0.9, "💀": -1.0
        }
        self.emoji_sentiment = self.config.get("emoji_sentiment", default_emoji_sentiment)

    def _load_slang_mappings(self) -> None:
        """Load social media slang sentiment mappings."""
        default_slang_sentiment = {
            "hodl": 0.8, "lambo": 0.8, "rekt": -0.9, "fud": -0.6
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
        """
        if not text or not text.strip():
            return SentimentResult(0.0, 0.0, [], [], False, 0.0, 0.0)

        text_lower = text.lower().strip()

        # Detect sentiment signals
        positive_signals = self._find_positive_signals(text_lower)
        negative_signals = self._find_negative_signals(text_lower)

        # Analyze emoji sentiment
        emoji_count = 0
        total_emoji_sentiment = 0.0
        for emoji, sentiment in self.emoji_sentiment.items():
            count = text.count(emoji)
            if count > 0:
                total_emoji_sentiment += sentiment * count
                emoji_count += count
        emoji_sentiment = total_emoji_sentiment / max(1, emoji_count) if emoji_count > 0 else 0.0

        # Analyze slang sentiment
        total_slang_sentiment = 0.0
        slang_count = 0
        for pattern, sentiment in self.slang_patterns.items():
            matches = pattern.findall(text_lower)
            if matches:
                total_slang_sentiment += sentiment * len(matches)
                slang_count += len(matches)
        for word in text_lower.split():
            if word in self.slang_sentiment:
                total_slang_sentiment += self.slang_sentiment[word]
                slang_count += 1
        slang_sentiment = total_slang_sentiment / max(1, slang_count) if slang_count > 0 else 0.0

        # Detect negation context
        negation_detected = self._detect_negation_context(text_lower, positive_signals + negative_signals)

        # Calculate base sentiment score
        pos_score = sum(self.keyword_weights.get(signal, 1.0) for signal in positive_signals)
        neg_score = sum(self.keyword_weights.get(signal, 1.0) for signal in negative_signals)
        keyword_sentiment = pos_score - neg_score

        if negation_detected:
            keyword_sentiment *= -0.8

        final_score = (
            self.keyword_weight * keyword_sentiment +
            self.emoji_weight * emoji_sentiment +
            self.slang_weight * slang_sentiment
        )
        final_score = max(-1.0, min(1.0, final_score))

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
        return [kw for kw in self.positive_keywords if kw.lower() in text]

    def _find_negative_signals(self, text: str) -> List[str]:
        return [kw for kw in self.negative_keywords if kw.lower() in text]

    def _detect_negation_context(self, text: str, signals: List[str]) -> bool:
        if not signals: return False
        words = text.split()
        for signal in signals:
            try:
                # Find the index of the signal word(s) in the list of words
                signal_words = signal.lower().split()
                for i in range(len(words) - len(signal_words) + 1):
                    if words[i:i+len(signal_words)] == signal_words:
                        idx = i
                        start_idx = max(0, idx - self.negation_window)
                        if any(word in self.negation_words for word in words[start_idx:idx]):
                            return True
                        break
            except Exception:
                continue
        return False

    def analyze_bias(self, text: str) -> Dict[str, bool]:
        """Detect potential bias indicators in text."""
        text_lower = text.lower()
        bias_indicators = self.config.get("bias_indicators", {})
        bias_detected = {}
        for bias_type, keywords in bias_indicators.items():
            bias_detected[bias_type] = any(keyword in text_lower for keyword in keywords)
        return bias_detected

    def analyze_trend_queries(self, queries: List[str]) -> Dict[str, int]:
        """Analyze sentiment of trend-related queries using group matching."""
        sentiment_counts = {'bullish': 0, 'bearish': 0, 'neutral': 0}
        trend_terms = self.config.get("trend_terms", {})

        for query in queries:
            query_lower = query.lower()
            bullish_found = any(all(term in query_lower for term in term_group)
                               for term_group in trend_terms.get('bullish', []))
            bearish_found = any(all(term in query_lower for term in term_group)
                               for term_group in trend_terms.get('bearish', []))

            if bullish_found and not bearish_found:
                sentiment_counts['bullish'] += 1
            elif bearish_found and not bullish_found:
                sentiment_counts['bearish'] += 1
            else:
                sentiment_counts['neutral'] += 1
        return sentiment_counts

    def get_credibility(self, url: str) -> float:
        """Get credibility score for a news source URL."""
        try:
            domain = urlparse(url).netloc.lower().replace('www.', '')
            cred_mapping = self.config.get("source_credibility", {})
            return cred_mapping.get(domain, cred_mapping.get("default", 0.5))
        except Exception:
            return 0.5

    def get_subreddits(self) -> List[str]:
        """Get list of monitored subreddits."""
        return self.config.get("subreddits", ["wallstreetbets", "stocks", "pennystocks", "options"])

    def get_discord_channel_keywords(self) -> List[str]:
        """Get keywords to identify financial Discord channels."""
        return self.config.get("discord_channel_keywords", ["trading", "stocks", "market"])
