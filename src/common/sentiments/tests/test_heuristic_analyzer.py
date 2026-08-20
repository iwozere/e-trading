import sys
import unittest
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.sentiments.processing.heuristic_analyzer import HeuristicSentimentAnalyzer


class TestHeuristicSentimentAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = HeuristicSentimentAnalyzer()

    def test_load_config(self):
        # Verify config is loaded (not empty if sentiments.json exists)
        self.assertTrue(len(self.analyzer.positive_keywords) > 5)
        self.assertTrue(len(self.analyzer.negative_keywords) > 5)
        self.assertIn("wallstreetbets", self.analyzer.get_subreddits())

    def test_analyze_sentiment(self):
        # Positive
        res = self.analyzer.analyze_sentiment("NVDA is going to the moon 🚀🚀🚀")
        self.assertTrue(res.score > 0.5)

        # Negative
        res = self.analyzer.analyze_sentiment("NVDA is crashing, total dump 📉")
        self.assertTrue(res.score < -0.5)

        # Negation
        res = self.analyzer.analyze_sentiment("I am not bullish on NVDA")
        self.assertTrue(res.score < 0)

    def test_analyze_bias(self):
        res = self.analyzer.analyze_bias("This is a sponsored post about a must-see stock.")
        self.assertTrue(res["promotional"])
        self.assertTrue(res["emotional"])
        self.assertFalse(res["speculative"])

    def test_analyze_trend_queries(self):
        queries = ["buy NVDA", "NVDA stock price target", "short sell TSLA", "market analysis"]
        counts = self.analyzer.analyze_trend_queries(queries)
        self.assertEqual(counts["bullish"], 2)
        self.assertEqual(counts["bearish"], 1)
        self.assertEqual(counts["neutral"], 1)

    def test_get_credibility(self):
        self.assertEqual(self.analyzer.get_credibility("https://www.reuters.com/business"), 0.95)
        self.assertEqual(self.analyzer.get_credibility("https://unknown.com"), 0.50)


class TestPerSourceLexicons(unittest.TestCase):
    """
    signal_class routes to a different lexicon bucket (sentiment-spec-rev2.md §2.5.3): retail
    measures hype, tech_discourse measures engineering reputation. Sharing one lexicon across
    both is the exact Rev 1 failure mode this split exists to fix.
    """

    def test_default_signal_class_is_retail(self):
        analyzer = HeuristicSentimentAnalyzer()
        self.assertEqual(analyzer.signal_class, "retail")
        self.assertIn("moon", analyzer.positive_keywords)

    def test_tech_discourse_uses_engineering_vocabulary(self):
        analyzer = HeuristicSentimentAnalyzer(signal_class="tech_discourse")
        self.assertIn("elegant", analyzer.positive_keywords)
        self.assertIn("outage", analyzer.negative_keywords)

    def test_tech_discourse_does_not_use_retail_hype_words(self):
        # "moon"/"diamond hands" are WSB slang with near-zero frequency on HN -- a shared lexicon
        # would silently return neutral for almost every HN message (spec §0).
        analyzer = HeuristicSentimentAnalyzer(signal_class="tech_discourse")
        self.assertNotIn("moon", analyzer.positive_keywords)

    def test_retail_does_not_use_tech_discourse_vocabulary(self):
        analyzer = HeuristicSentimentAnalyzer()
        self.assertNotIn("outage", analyzer.negative_keywords)

    def test_tech_discourse_scores_engineering_reputation(self):
        analyzer = HeuristicSentimentAnalyzer(signal_class="tech_discourse")
        positive = analyzer.analyze_sentiment("The new release is solid and well-designed, ships reliably.")
        negative = analyzer.analyze_sentiment("Another regression, the outage was caused by vendor lock-in.")
        self.assertGreater(positive.score, 0.0)
        self.assertLess(negative.score, 0.0)


if __name__ == "__main__":
    unittest.main()
