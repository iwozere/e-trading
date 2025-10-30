# src/common/sentiments/tests/test_virality_calculator.py
"""
Unit tests for virality and engagement calculations.

Tests cover:
- Virality index calculation based on engagement patterns
- Platform-specific engagement weighting
- Trending sentiment detection algorithms
- Influence scoring for high-impact accounts
"""

import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone, timedelta
from pathlib import Path
import sys

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from src.common.sentiments.processing.virality_calculator import (
    ViralityCalculator, ViralityResult, PostData, EngagementMetrics,
    AuthorInfluence, Platform
)


class TestViralityCalculator(unittest.TestCase):
    """Test cases for ViralityCalculator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "time_decay_factor": 0.1,
            "velocity_window_hours": 6,
            "trending_threshold": 0.7,
            "min_posts_for_trending": 10,
            "follower_weight": 0.3,
            "engagement_weight": 0.4,
            "verification_bonus": 0.2,
            "account_age_weight": 0.1,
            "virality_weights": {
                "engagement": 0.3,
                "velocity": 0.2,
                "reach": 0.2,
                "influence": 0.15,
                "trending": 0.15
            }
        }
        self.calculator = ViralityCalculator(self.config)

    def test_initialization_default_config(self):
        """Test calculator initialization with default configuration."""
        calculator = ViralityCalculator()

        self.assertEqual(calculator.time_decay_factor, 0.1)
        self.assertEqual(calculator.velocity_window_hours, 6)
        self.assertEqual(calculator.trending_threshold, 0.7)
        self.assertIn(Platform.TWITTER, calculator.platform_weights)

    def test_initialization_custom_config(self):
        """Test calculator initialization with custom configuration."""
        self.assertEqual(self.calculator.time_decay_factor, 0.1)
        self.assertEqual(self.calculator.velocity_window_hours, 6)
        self.assertEqual(self.calculator.follower_weight, 0.3)
        self.assertEqual(self.calculator.engagement_weight, 0.4)

    def test_calculate_virality_empty_posts(self):
        """Test virality calculation with empty posts list."""
        result = self.calculator.calculate_virality([])

        self.assertEqual(result.virality_index, 0.0)
        self.assertEqual(result.engagement_score, 0.0)
        self.assertEqual(result.velocity_score, 0.0)
        self.assertEqual(result.reach_score, 0.0)
        self.assertEqual(result.influence_score, 0.0)
        self.assertEqual(result.trending_score, 0.0)
        self.assertEqual(len(result.top_contributors), 0)

    def test_calculate_virality_single_post(self):
        """Test virality calculation with single post."""
        author = AuthorInfluence(
            username="test_user",
            followers_count=1000,
            following_count=500,
            verified=False,
            account_age_days=365,
            total_posts=100,
            avg_engagement=10.0
        )

        engagement = EngagementMetrics(
            likes=50,
            replies=10,
            retweets=5,
            shares=2,
            views=1000
        )

        post = PostData(
            id="post_1",
            content="Great trading opportunity! #crypto #bitcoin",
            author=author,
            timestamp=datetime.now(timezone.utc),
            engagement=engagement,
            platform=Platform.TWITTER,
            hashtags=["crypto", "bitcoin"],
            mentions=[]
        )

        result = self.calculator.calculate_virality([post])

        self.assertGreater(result.virality_index, 0.0)
        self.assertLessEqual(result.virality_index, 1.0)
        self.assertGreater(result.engagement_score, 0.0)
        self.assertGreaterEqual(result.reach_score, 0.0)
        self.assertGreater(result.influence_score, 0.0)
        self.assertEqual(len(result.top_contributors), 1)
        self.assertEqual(result.top_contributors[0][0], "test_user")

    def test_calculate_virality_multiple_posts(self):
        """Test virality calculation with multiple posts."""
        posts = []

        for i in range(5):
            author = AuthorInfluence(
                username=f"user_{i}",
                followers_count=1000 + i * 500,
                following_count=500,
                verified=i == 0,  # First user is verified
                account_age_days=365 - i * 30,
                total_posts=100 + i * 50,
                avg_engagement=10.0 + i * 5
            )

            engagement = EngagementMetrics(
                likes=20 + i * 10,
                replies=5 + i * 2,
                retweets=2 + i,
                shares=1,
                views=500 + i * 200
            )

            post = PostData(
                id=f"post_{i}",
                content=f"Trading post {i} #crypto #trading",
                author=author,
                timestamp=datetime.now(timezone.utc) - timedelta(hours=i),
                engagement=engagement,
                platform=Platform.TWITTER,
                hashtags=["crypto", "trading"],
                mentions=[]
            )
            posts.append(post)

        result = self.calculator.calculate_virality(posts)

        self.assertGreater(result.virality_index, 0.0)
        self.assertLessEqual(result.virality_index, 1.0)
        self.assertGreater(result.engagement_score, 0.0)
        self.assertGreaterEqual(result.velocity_score, 0.0)
        self.assertGreater(result.reach_score, 0.0)
        self.assertGreater(result.influence_score, 0.0)
        self.assertGreaterEqual(result.trending_score, 0.0)
        self.assertLessEqual(len(result.top_contributors), 5)

    def test_platform_specific_weighting(self):
        """Test platform-specific engagement weighting."""
        author = AuthorInfluence(
            username="test_user",
            followers_count=1000,
            verified=False,
            account_age_days=365
        )

        # Same engagement metrics on different platforms
        engagement = EngagementMetrics(likes=10, replies=5, retweets=3)

        twitter_post = PostData(
            id="twitter_post",
            content="Test post",
            author=author,
            timestamp=datetime.now(timezone.utc),
            engagement=engagement,
            platform=Platform.TWITTER
        )

        reddit_post = PostData(
            id="reddit_post",
            content="Test post",
            author=author,
            timestamp=datetime.now(timezone.utc),
            engagement=EngagementMetrics(upvotes=10, comments=5),
            platform=Platform.REDDIT
        )

        twitter_result = self.calculator.calculate_virality([twitter_post])
        reddit_result = self.calculator.calculate_virality([reddit_post])

        # Results should differ due to platform-specific weighting
        self.assertNotEqual(twitter_result.engagement_score, reddit_result.engagement_score)

    def test_engagement_score_calculation(self):
        """Test engagement score calculation logic."""
        author = AuthorInfluence(username="test_user", followers_count=1000)

        # High engagement post
        high_engagement = EngagementMetrics(
            likes=100,
            replies=20,
            retweets=15,
            shares=5,
            views=5000
        )

        high_post = PostData(
            id="high_post",
            content="Viral post",
            author=author,
            timestamp=datetime.now(timezone.utc),
            engagement=high_engagement,
            platform=Platform.TWITTER
        )

        # Low engagement post
        low_engagement = EngagementMetrics(
            likes=2,
            replies=0,
            retweets=0,
            shares=0,
            views=50
        )

        low_post = PostData(
            id="low_post",
            content="Low engagement post",
            author=author,
            timestamp=datetime.now(timezone.utc),
            engagement=low_engagement,
            platform=Platform.TWITTER
        )

        high_result = self.calculator.calculate_virality([high_post])
        low_result = self.calculator.calculate_virality([low_post])

        self.assertGreater(high_result.engagement_score, low_result.engagement_score)

    def test_velocity_score_calculation(self):
        """Test velocity score calculation for engagement growth."""
        author = AuthorInfluence(username="test_user", followers_count=1000)

        # Create posts with increasing engagement over time
        posts = []
        base_time = datetime.now(timezone.utc) - timedelta(hours=12)

        for i in range(6):
            engagement = EngagementMetrics(
                likes=10 + i * 20,  # Increasing engagement
                replies=2 + i * 4,
                retweets=1 + i * 2
            )

            post = PostData(
                id=f"post_{i}",
                content=f"Post {i}",
                author=author,
                timestamp=base_time + timedelta(hours=i * 2),
                engagement=engagement,
                platform=Platform.TWITTER
            )
            posts.append(post)

        result = self.calculator.calculate_virality(posts)

        # Should detect positive velocity due to increasing engagement
        self.assertGreater(result.velocity_score, 0.0)

    def test_reach_score_calculation(self):
        """Test reach score calculation based on followers and engagement."""
        # High-reach scenario: verified user with many followers
        high_reach_author = AuthorInfluence(
            username="influencer",
            followers_count=100000,
            verified=True,
            account_age_days=1000
        )

        high_reach_post = PostData(
            id="high_reach",
            content="Influencer post",
            author=high_reach_author,
            timestamp=datetime.now(timezone.utc),
            engagement=EngagementMetrics(likes=1000, replies=100, retweets=200),
            platform=Platform.TWITTER
        )

        # Low-reach scenario: new user with few followers
        low_reach_author = AuthorInfluence(
            username="newbie",
            followers_count=50,
            verified=False,
            account_age_days=30
        )

        low_reach_post = PostData(
            id="low_reach",
            content="Newbie post",
            author=low_reach_author,
            timestamp=datetime.now(timezone.utc),
            engagement=EngagementMetrics(likes=5, replies=1, retweets=0),
            platform=Platform.TWITTER
        )

        high_result = self.calculator.calculate_virality([high_reach_post])
        low_result = self.calculator.calculate_virality([low_reach_post])

        self.assertGreater(high_result.reach_score, low_result.reach_score)

    def test_influence_score_calculation(self):
        """Test influence score calculation based on author metrics."""
        # High influence: verified, many followers, high engagement
        high_influence_author = AuthorInfluence(
            username="crypto_expert",
            followers_count=50000,
            following_count=1000,
            verified=True,
            account_age_days=2000,
            total_posts=5000,
            avg_engagement=100.0
        )

        # Low influence: new account, few followers
        low_influence_author = AuthorInfluence(
            username="newbie123",
            followers_count=10,
            following_count=500,
            verified=False,
            account_age_days=10,
            total_posts=5,
            avg_engagement=1.0
        )

        high_post = PostData(
            id="high_influence",
            content="Expert analysis",
            author=high_influence_author,
            timestamp=datetime.now(timezone.utc),
            engagement=EngagementMetrics(likes=200, replies=50, retweets=30),
            platform=Platform.TWITTER
        )

        low_post = PostData(
            id="low_influence",
            content="Newbie question",
            author=low_influence_author,
            timestamp=datetime.now(timezone.utc),
            engagement=EngagementMetrics(likes=2, replies=1, retweets=0),
            platform=Platform.TWITTER
        )

        high_result = self.calculator.calculate_virality([high_post])
        low_result = self.calculator.calculate_virality([low_post])

        self.assertGreater(high_result.influence_score, low_result.influence_score)

    def test_trending_score_calculation(self):
        """Test trending score calculation based on hashtags and mentions."""
        author = AuthorInfluence(username="test_user", followers_count=1000)

        # Posts with trending hashtags
        trending_posts = []
        for i in range(15):  # Above min_posts_for_trending
            post = PostData(
                id=f"trending_post_{i}",
                content=f"Post {i} #bitcoin #crypto #moon",
                author=author,
                timestamp=datetime.now(timezone.utc) - timedelta(hours=i),
                engagement=EngagementMetrics(likes=10, replies=2, retweets=1),
                platform=Platform.TWITTER,
                hashtags=["bitcoin", "crypto", "moon"],
                mentions=["elonmusk"] if i % 3 == 0 else []
            )
            trending_posts.append(post)

        # Posts without trending elements
        non_trending_posts = []
        for i in range(5):
            post = PostData(
                id=f"normal_post_{i}",
                content=f"Normal post {i}",
                author=author,
                timestamp=datetime.now(timezone.utc) - timedelta(hours=i),
                engagement=EngagementMetrics(likes=10, replies=2, retweets=1),
                platform=Platform.TWITTER,
                hashtags=[],
                mentions=[]
            )
            non_trending_posts.append(post)

        trending_result = self.calculator.calculate_virality(trending_posts)
        non_trending_result = self.calculator.calculate_virality(non_trending_posts)

        self.assertGreater(trending_result.trending_score, non_trending_result.trending_score)

    def test_time_decay_factor(self):
        """Test time decay factor for older posts."""
        author = AuthorInfluence(username="test_user", followers_count=1000)
        engagement = EngagementMetrics(likes=50, replies=10, retweets=5)

        # Recent post
        recent_post = PostData(
            id="recent",
            content="Recent post",
            author=author,
            timestamp=datetime.now(timezone.utc),
            engagement=engagement,
            platform=Platform.TWITTER
        )

        # Old post
        old_post = PostData(
            id="old",
            content="Old post",
            author=author,
            timestamp=datetime.now(timezone.utc) - timedelta(days=1),
            engagement=engagement,
            platform=Platform.TWITTER
        )

        recent_result = self.calculator.calculate_virality([recent_post])
        old_result = self.calculator.calculate_virality([old_post])

        # Recent post should have higher engagement score due to time decay
        self.assertGreater(recent_result.engagement_score, old_result.engagement_score)

    def test_analyze_trending_topics(self):
        """Test trending topics analysis."""
        author = AuthorInfluence(username="test_user", followers_count=1000)

        posts = []
        for i in range(10):
            post = PostData(
                id=f"post_{i}",
                content=f"Post {i} #bitcoin #crypto",
                author=author,
                timestamp=datetime.now(timezone.utc) - timedelta(hours=i),
                engagement=EngagementMetrics(likes=10 + i, replies=2, retweets=1),
                platform=Platform.TWITTER,
                hashtags=["bitcoin", "crypto"] + (["moon"] if i < 3 else []),
                mentions=["elonmusk"] if i < 5 else []
            )
            posts.append(post)

        trending_topics = self.calculator.analyze_trending_topics(posts, min_mentions=3)

        self.assertIn("#bitcoin", trending_topics)
        self.assertIn("#crypto", trending_topics)
        self.assertIn("@elonmusk", trending_topics)

        # Bitcoin should be most trending (appears in all posts)
        self.assertGreater(trending_topics["#bitcoin"], trending_topics.get("#moon", 0))

    def test_get_virality_stats(self):
        """Test virality statistics retrieval."""
        # Calculate some virality to populate cache
        author = AuthorInfluence(username="test_user", followers_count=1000)
        post = PostData(
            id="test_post",
            content="Test post",
            author=author,
            timestamp=datetime.now(timezone.utc),
            engagement=EngagementMetrics(likes=10, replies=2, retweets=1),
            platform=Platform.TWITTER
        )

        self.calculator.calculate_virality([post])

        stats = self.calculator.get_virality_stats()

        self.assertIn("influence_cache_size", stats)
        self.assertIn("platform_baselines", stats)
        self.assertIn("config", stats)

        self.assertGreaterEqual(stats["influence_cache_size"], 1)

    def test_clear_cache(self):
        """Test cache clearing functionality."""
        # Calculate virality to populate cache
        author = AuthorInfluence(username="test_user", followers_count=1000)
        post = PostData(
            id="test_post",
            content="Test post",
            author=author,
            timestamp=datetime.now(timezone.utc),
            engagement=EngagementMetrics(likes=10, replies=2, retweets=1),
            platform=Platform.TWITTER
        )

        self.calculator.calculate_virality([post])

        # Verify cache has content
        stats_before = self.calculator.get_virality_stats()
        self.assertGreater(stats_before["influence_cache_size"], 0)

        # Clear cache
        self.calculator.clear_cache()

        # Verify cache is empty
        stats_after = self.calculator.get_virality_stats()
        self.assertEqual(stats_after["influence_cache_size"], 0)

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with posts having zero engagement
        author = AuthorInfluence(username="test_user", followers_count=0)
        zero_engagement_post = PostData(
            id="zero_post",
            content="Zero engagement post",
            author=author,
            timestamp=datetime.now(timezone.utc),
            engagement=EngagementMetrics(),  # All zeros
            platform=Platform.TWITTER
        )

        result = self.calculator.calculate_virality([zero_engagement_post])

        # Should handle gracefully
        self.assertIsInstance(result, ViralityResult)
        self.assertGreaterEqual(result.virality_index, 0.0)
        self.assertLessEqual(result.virality_index, 1.0)


class TestEngagementMetrics(unittest.TestCase):
    """Test cases for EngagementMetrics dataclass."""

    def test_engagement_metrics_creation(self):
        """Test EngagementMetrics object creation."""
        engagement = EngagementMetrics(
            likes=50,
            replies=10,
            retweets=5,
            shares=3,
            views=1000,
            comments=8,
            upvotes=25,
            downvotes=2,
            reactions=15
        )

        self.assertEqual(engagement.likes, 50)
        self.assertEqual(engagement.replies, 10)
        self.assertEqual(engagement.retweets, 5)
        self.assertEqual(engagement.shares, 3)
        self.assertEqual(engagement.views, 1000)
        self.assertEqual(engagement.comments, 8)
        self.assertEqual(engagement.upvotes, 25)
        self.assertEqual(engagement.downvotes, 2)
        self.assertEqual(engagement.reactions, 15)

    def test_total_engagement_calculation(self):
        """Test total engagement calculation."""
        engagement = EngagementMetrics(
            likes=10,
            replies=5,
            retweets=3,
            shares=2,
            comments=4,
            upvotes=8,
            reactions=6
        )

        expected_total = 10 + 5 + 3 + 2 + 4 + 8 + 6
        self.assertEqual(engagement.total_engagement(), expected_total)

    def test_default_values(self):
        """Test default values for EngagementMetrics."""
        engagement = EngagementMetrics()

        self.assertEqual(engagement.likes, 0)
        self.assertEqual(engagement.replies, 0)
        self.assertEqual(engagement.retweets, 0)
        self.assertEqual(engagement.total_engagement(), 0)


class TestAuthorInfluence(unittest.TestCase):
    """Test cases for AuthorInfluence dataclass."""

    def test_author_influence_creation(self):
        """Test AuthorInfluence object creation."""
        author = AuthorInfluence(
            username="crypto_expert",
            followers_count=50000,
            following_count=1000,
            verified=True,
            account_age_days=1500,
            total_posts=2000,
            avg_engagement=75.5,
            influence_score=0.85
        )

        self.assertEqual(author.username, "crypto_expert")
        self.assertEqual(author.followers_count, 50000)
        self.assertEqual(author.following_count, 1000)
        self.assertTrue(author.verified)
        self.assertEqual(author.account_age_days, 1500)
        self.assertEqual(author.total_posts, 2000)
        self.assertEqual(author.avg_engagement, 75.5)
        self.assertEqual(author.influence_score, 0.85)


class TestPostData(unittest.TestCase):
    """Test cases for PostData dataclass."""

    def test_post_data_creation(self):
        """Test PostData object creation."""
        author = AuthorInfluence(username="test_user", followers_count=1000)
        engagement = EngagementMetrics(likes=50, replies=10, retweets=5)
        timestamp = datetime.now(timezone.utc)

        post = PostData(
            id="test_post_123",
            content="This is a test post about #crypto trading",
            author=author,
            timestamp=timestamp,
            engagement=engagement,
            platform=Platform.TWITTER,
            hashtags=["crypto"],
            mentions=["elonmusk"]
        )

        self.assertEqual(post.id, "test_post_123")
        self.assertEqual(post.content, "This is a test post about #crypto trading")
        self.assertEqual(post.author, author)
        self.assertEqual(post.timestamp, timestamp)
        self.assertEqual(post.engagement, engagement)
        self.assertEqual(post.platform, Platform.TWITTER)
        self.assertEqual(post.hashtags, ["crypto"])
        self.assertEqual(post.mentions, ["elonmusk"])

    def test_post_data_default_lists(self):
        """Test PostData with default hashtags and mentions."""
        author = AuthorInfluence(username="test_user", followers_count=1000)
        engagement = EngagementMetrics(likes=50, replies=10, retweets=5)

        post = PostData(
            id="test_post",
            content="Test post",
            author=author,
            timestamp=datetime.now(timezone.utc),
            engagement=engagement,
            platform=Platform.TWITTER
        )

        self.assertEqual(post.hashtags, [])
        self.assertEqual(post.mentions, [])


if __name__ == "__main__":
    unittest.main()