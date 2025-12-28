import unittest
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

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
        self.assertTrue(res['promotional'])
        self.assertTrue(res['emotional'])
        self.assertFalse(res['speculative'])

    def test_analyze_trend_queries(self):
        queries = ["buy NVDA", "NVDA stock price target", "short sell TSLA", "market analysis"]
        counts = self.analyzer.analyze_trend_queries(queries)
        self.assertEqual(counts['bullish'], 2)
        self.assertEqual(counts['bearish'], 1)
        self.assertEqual(counts['neutral'], 1)

    def test_get_credibility(self):
        self.assertEqual(self.analyzer.get_credibility("https://www.reuters.com/business"), 0.95)
        self.assertEqual(self.analyzer.get_credibility("https://unknown.com"), 0.50)

if __name__ == "__main__":
    unittest.main()