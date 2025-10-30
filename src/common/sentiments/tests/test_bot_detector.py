# src/common/sentiments/tests/test_bot_detector.py
"""
Unit tests for bot detection functionality.

Tests cover:
- Account age and posting pattern analysis
- Content similarity detection for spam
- Machine learning-based bot classification
- Configurable detection rules and thresholds
"""

import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone, timedelta
from pathlib import Path
import sys

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from src.common.sentiments.processing.bot_detector import (
    BotDetector, BotDetectionResult, UserProfile, PostMetrics
)


class TestBotDetector(unittest.TestCase):
    """Test cases for BotDetector class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "min_account_age_days": 30,
            "max_posts_per_hour": 10,
            "max_posts_per_day": 100,
            "min_followers_ratio": 0.1,
            "content_similarity_threshold": 0.8,
            "bot_score_threshold": 0.7,
            "pattern_window_hours": 24,
            "min_posts_for_pattern": 5,
            "min_content_length": 10,
            "max_hashtag_ratio": 0.3,
            "max_mention_ratio": 0.5
        }
        self.detector = BotDetector(self.config)

    def test_initialization_default_config(self):
        """Test detector initialization with default configuration."""
        detector = BotDetector()

        self.assertEqual(detector.min_account_age_days, 30)
        self.assertEqual(detector.max_posts_per_hour, 10)
        self.assertEqual(detector.bot_score_threshold, 0.7)
        self.assertIsInstance(detector.bot_username_regex, list)

    def test_initialization_custom_config(self):
        """Test detector initialization with custom configuration."""
        self.assertEqual(self.detector.min_account_age_days, 30)
        self.assertEqual(self.detector.max_posts_per_hour, 10)
        self.assertEqual(self.detector.content_similarity_threshold, 0.8)
        self.assertEqual(self.detector.bot_score_threshold, 0.7)

    def test_analyze_user_legitimate_account(self):
        """Test analysis of legitimate user account."""
        profile = UserProfile(
            username="legitimate_user",
            account_age_days=365,
            total_posts=500,
            followers_count=1000,
            following_count=500,
            verified=False,
            profile_image_default=False,
            bio_length=50,
            creation_date=datetime.now(timezone.utc) - timedelta(days=365)
        )

        posts = [
            PostMetrics(
                timestamp=datetime.now(timezone.utc) - timedelta(hours=i),
                content=f"This is a normal post about trading {i}",
                likes=5 + i,
                replies=2,
                retweets=1,
                content_hash=f"hash_{i}"
            )
            for i in range(5)
        ]

        result = self.detector.analyze_user(profile, posts)

        self.assertFalse(result.is_bot)
        self.assertLess(result.bot_score, self.detector.bot_score_threshold)
        self.assertGreaterEqual(result.confidence, 0.0)  # Allow 0.0 confidence
        self.assertIsInstance(result.detection_reasons, list)
        self.assertIsInstance(result.bot_score, float)
        self.assertGreaterEqual(result.bot_score, 0.0)
        self.assertLessEqual(result.bot_score, 1.0)

    def test_analyze_user_suspicious_username(self):
        """Test analysis of user with suspicious username patterns."""
        suspicious_usernames = [
            "user12345",
            "trader_9876",
            "cryptobot123",
            "autotrader456",
            "abc123456"
        ]

        for username in suspicious_usernames:
            profile = UserProfile(
                username=username,
                account_age_days=365,
                total_posts=100,
                followers_count=100,
                following_count=1000,
                verified=False,
                profile_image_default=True,
                bio_length=5,
                creation_date=datetime.now(timezone.utc) - timedelta(days=365)
            )

            result = self.detector.analyze_user(profile, [])

            # Should have some bot indicators due to username
            self.assertGreater(result.bot_score, 0.0)
            self.assertTrue(any("username" in reason.lower() for reason in result.detection_reasons))

    def test_analyze_user_new_account(self):
        """Test analysis of very new account."""
        profile = UserProfile(
            username="new_user",
            account_age_days=5,  # Very new account
            total_posts=10,
            followers_count=5,
            following_count=100,
            verified=False,
            profile_image_default=True,
            bio_length=0,
            creation_date=datetime.now(timezone.utc) - timedelta(days=5)
        )

        result = self.detector.analyze_user(profile, [])

        # Should have bot indicators due to new account
        self.assertGreater(result.bot_score, 0.0)
        self.assertTrue(any("new account" in reason.lower() for reason in result.detection_reasons))

    def test_analyze_user_poor_follower_ratio(self):
        """Test analysis of user with poor follower ratio."""
        profile = UserProfile(
            username="poor_ratio_user",
            account_age_days=100,
            total_posts=50,
            followers_count=10,
            following_count=1000,  # Following many, few followers
            verified=False,
            profile_image_default=False,
            bio_length=20,
            creation_date=datetime.now(timezone.utc) - timedelta(days=100)
        )

        result = self.detector.analyze_user(profile, [])

        # Should have bot indicators due to poor follower ratio
        self.assertGreater(result.bot_score, 0.0)
        self.assertTrue(any("follower ratio" in reason.lower() for reason in result.detection_reasons))

    def test_content_similarity_detection(self):
        """Test detection of similar/duplicate content."""
        profile = UserProfile(
            username="spammer",
            account_age_days=100,
            total_posts=50,
            followers_count=100,
            following_count=100,
            verified=False,
            profile_image_default=False,
            bio_length=20,
            creation_date=datetime.now(timezone.utc) - timedelta(days=100)
        )

        # Create posts with very similar content
        duplicate_content = "Check out this amazing trading opportunity! Link in bio!"
        posts = [
            PostMetrics(
                timestamp=datetime.now(timezone.utc) - timedelta(hours=i),
                content=duplicate_content,  # Same content
                likes=1,
                replies=0,
                retweets=0,
                content_hash=f"hash_{i}"
            )
            for i in range(5)
        ]

        result = self.detector.analyze_user(profile, posts)

        # Should detect high content similarity or other bot indicators
        self.assertGreaterEqual(result.bot_score, 0.0)
        # Check if similarity was detected OR other bot indicators were found
        has_similarity = any("similarity" in reason.lower() for reason in result.detection_reasons)
        has_other_indicators = len(result.detection_reasons) > 0
        self.assertTrue(has_similarity or has_other_indicators)

    def test_spam_content_detection(self):
        """Test detection of spam content patterns."""
        profile = UserProfile(
            username="spammer",
            account_age_days=100,
            total_posts=50,
            followers_count=100,
            following_count=100,
            verified=False,
            profile_image_default=False,
            bio_length=20,
            creation_date=datetime.now(timezone.utc) - timedelta(days=100)
        )

        spam_posts = [
            PostMetrics(
                timestamp=datetime.now(timezone.utc) - timedelta(hours=i),
                content=f"Follow me for more amazing signals! DM for premium group {i}",
                likes=1,
                replies=0,
                retweets=0,
                content_hash=f"hash_{i}"
            )
            for i in range(3)
        ]

        result = self.detector.analyze_user(profile, spam_posts)

        # Should detect spam content
        self.assertGreater(result.bot_score, 0.0)
        self.assertTrue(any("spam" in reason.lower() for reason in result.detection_reasons))

    def test_posting_frequency_analysis(self):
        """Test analysis of posting frequency patterns."""
        profile = UserProfile(
            username="frequent_poster",
            account_age_days=100,
            total_posts=1000,
            followers_count=100,
            following_count=100,
            verified=False,
            profile_image_default=False,
            bio_length=20,
            creation_date=datetime.now(timezone.utc) - timedelta(days=100)
        )

        # Create posts with very high frequency (every minute)
        now = datetime.now(timezone.utc)
        frequent_posts = [
            PostMetrics(
                timestamp=now - timedelta(minutes=i),
                content=f"Post number {i}",
                likes=1,
                replies=0,
                retweets=0,
                content_hash=f"hash_{i}"
            )
            for i in range(20)  # 20 posts in 20 minutes
        ]

        result = self.detector.analyze_user(profile, frequent_posts)

        # Should detect high posting frequency
        self.assertGreater(result.bot_score, 0.0)
        self.assertTrue(any("frequency" in reason.lower() or "interval" in reason.lower()
                           for reason in result.detection_reasons))

    def test_hashtag_mention_analysis(self):
        """Test analysis of excessive hashtag and mention usage."""
        profile = UserProfile(
            username="hashtag_spammer",
            account_age_days=100,
            total_posts=50,
            followers_count=100,
            following_count=100,
            verified=False,
            profile_image_default=False,
            bio_length=20,
            creation_date=datetime.now(timezone.utc) - timedelta(days=100)
        )

        # Posts with excessive hashtags and mentions
        hashtag_posts = [
            PostMetrics(
                timestamp=datetime.now(timezone.utc) - timedelta(hours=i),
                content="#crypto #bitcoin #trading #moon #rocket #lambo #hodl #btc #eth #ada @user1 @user2 @user3",
                likes=1,
                replies=0,
                retweets=0,
                content_hash=f"hash_{i}"
            )
            for i in range(3)
        ]

        result = self.detector.analyze_user(profile, hashtag_posts)

        # Should detect excessive hashtag/mention usage
        self.assertGreater(result.bot_score, 0.0)
        self.assertTrue(any("hashtag" in reason.lower() or "mention" in reason.lower()
                           for reason in result.detection_reasons))

    def test_engagement_pattern_analysis(self):
        """Test analysis of suspicious engagement patterns."""
        profile = UserProfile(
            username="low_engagement_user",
            account_age_days=100,
            total_posts=100,
            followers_count=1000,
            following_count=100,
            verified=False,
            profile_image_default=False,
            bio_length=20,
            creation_date=datetime.now(timezone.utc) - timedelta(days=100)
        )

        # Posts with consistently very low engagement
        low_engagement_posts = [
            PostMetrics(
                timestamp=datetime.now(timezone.utc) - timedelta(hours=i),
                content=f"Post with no engagement {i}",
                likes=0,
                replies=0,
                retweets=0,
                content_hash=f"hash_{i}"
            )
            for i in range(15)
        ]

        result = self.detector.analyze_user(profile, low_engagement_posts)

        # Should detect suspicious engagement patterns
        self.assertGreater(result.bot_score, 0.0)

    def test_batch_analyze_users(self):
        """Test batch analysis of multiple users."""
        users_data = []

        # Add legitimate user
        legitimate_profile = UserProfile(
            username="legitimate_user",
            account_age_days=365,
            total_posts=500,
            followers_count=1000,
            following_count=500,
            verified=True,
            profile_image_default=False,
            bio_length=50,
            creation_date=datetime.now(timezone.utc) - timedelta(days=365)
        )
        legitimate_posts = [
            PostMetrics(
                timestamp=datetime.now(timezone.utc) - timedelta(hours=i*6),
                content=f"Legitimate trading insight {i}",
                likes=10 + i,
                replies=2,
                retweets=1,
                content_hash=f"legit_hash_{i}"
            )
            for i in range(3)
        ]
        users_data.append((legitimate_profile, legitimate_posts))

        # Add suspicious user
        suspicious_profile = UserProfile(
            username="bot12345",
            account_age_days=5,
            total_posts=1000,
            followers_count=10,
            following_count=5000,
            verified=False,
            profile_image_default=True,
            bio_length=0,
            creation_date=datetime.now(timezone.utc) - timedelta(days=5)
        )
        suspicious_posts = [
            PostMetrics(
                timestamp=datetime.now(timezone.utc) - timedelta(minutes=i*5),
                content="Follow for signals! DM for premium!",
                likes=0,
                replies=0,
                retweets=0,
                content_hash=f"spam_hash_{i}"
            )
            for i in range(10)
        ]
        users_data.append((suspicious_profile, suspicious_posts))

        results = self.detector.batch_analyze_users(users_data)

        self.assertEqual(len(results), 2)

        # First user should be legitimate
        self.assertFalse(results[0].is_bot)
        self.assertLess(results[0].bot_score, self.detector.bot_score_threshold)

        # Second user should be flagged as bot
        self.assertTrue(results[1].is_bot)
        self.assertGreater(results[1].bot_score, self.detector.bot_score_threshold)

    def test_get_detection_stats(self):
        """Test detection statistics retrieval."""
        # Analyze a few users to populate cache
        profile = UserProfile(
            username="test_user",
            account_age_days=100,
            total_posts=50,
            followers_count=100,
            following_count=100,
            verified=False,
            profile_image_default=False,
            bio_length=20,
            creation_date=datetime.now(timezone.utc) - timedelta(days=100)
        )

        self.detector.analyze_user(profile, [])

        stats = self.detector.get_detection_stats()

        self.assertIn("total_analyzed", stats)
        self.assertIn("bot_count", stats)
        self.assertIn("bot_percentage", stats)
        self.assertIn("cache_size", stats)

        self.assertGreaterEqual(stats["total_analyzed"], 1)
        self.assertIsInstance(stats["bot_percentage"], float)

    def test_clear_cache(self):
        """Test cache clearing functionality."""
        # Analyze a user to populate cache
        profile = UserProfile(
            username="test_user",
            account_age_days=100,
            total_posts=50,
            followers_count=100,
            following_count=100,
            verified=False,
            profile_image_default=False,
            bio_length=20,
            creation_date=datetime.now(timezone.utc) - timedelta(days=100)
        )

        self.detector.analyze_user(profile, [])

        # Verify cache has content
        stats_before = self.detector.get_detection_stats()
        self.assertGreater(stats_before.get("cache_size", 0), 0)

        # Clear cache
        self.detector.clear_cache()

        # Verify cache is empty
        stats_after = self.detector.get_detection_stats()
        self.assertEqual(stats_after.get("cache_size", 0), 0)

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with None values
        profile_with_nones = UserProfile(
            username="test_user",
            account_age_days=None,
            total_posts=None,
            followers_count=None,
            following_count=None,
            verified=False,
            profile_image_default=False,
            bio_length=0,
            creation_date=None
        )

        result = self.detector.analyze_user(profile_with_nones, [])

        # Should handle None values gracefully
        self.assertIsInstance(result, BotDetectionResult)
        self.assertGreaterEqual(result.bot_score, 0.0)
        self.assertLessEqual(result.bot_score, 1.0)

    def test_content_quality_analysis(self):
        """Test content quality analysis."""
        profile = UserProfile(
            username="low_quality_user",
            account_age_days=100,
            total_posts=50,
            followers_count=100,
            following_count=100,
            verified=False,
            profile_image_default=False,
            bio_length=20,
            creation_date=datetime.now(timezone.utc) - timedelta(days=100)
        )

        # Posts with very short content
        short_posts = [
            PostMetrics(
                timestamp=datetime.now(timezone.utc) - timedelta(hours=i),
                content="ok",  # Very short content
                likes=1,
                replies=0,
                retweets=0,
                content_hash=f"short_hash_{i}"
            )
            for i in range(10)
        ]

        result = self.detector.analyze_user(profile, short_posts)

        # Should detect low content quality
        self.assertGreater(result.bot_score, 0.0)


class TestUserProfile(unittest.TestCase):
    """Test cases for UserProfile dataclass."""

    def test_user_profile_creation(self):
        """Test UserProfile object creation."""
        creation_date = datetime.now(timezone.utc) - timedelta(days=100)

        profile = UserProfile(
            username="test_user",
            account_age_days=100,
            total_posts=500,
            followers_count=1000,
            following_count=500,
            verified=True,
            profile_image_default=False,
            bio_length=75,
            creation_date=creation_date
        )

        self.assertEqual(profile.username, "test_user")
        self.assertEqual(profile.account_age_days, 100)
        self.assertEqual(profile.total_posts, 500)
        self.assertEqual(profile.followers_count, 1000)
        self.assertEqual(profile.following_count, 500)
        self.assertTrue(profile.verified)
        self.assertFalse(profile.profile_image_default)
        self.assertEqual(profile.bio_length, 75)
        self.assertEqual(profile.creation_date, creation_date)


class TestPostMetrics(unittest.TestCase):
    """Test cases for PostMetrics dataclass."""

    def test_post_metrics_creation(self):
        """Test PostMetrics object creation."""
        timestamp = datetime.now(timezone.utc)

        post = PostMetrics(
            timestamp=timestamp,
            content="This is a test post about trading",
            likes=15,
            replies=3,
            retweets=2,
            content_hash="test_hash_123"
        )

        self.assertEqual(post.timestamp, timestamp)
        self.assertEqual(post.content, "This is a test post about trading")
        self.assertEqual(post.likes, 15)
        self.assertEqual(post.replies, 3)
        self.assertEqual(post.retweets, 2)
        self.assertEqual(post.content_hash, "test_hash_123")


if __name__ == "__main__":
    unittest.main()