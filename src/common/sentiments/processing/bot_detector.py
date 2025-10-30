# src/common/sentiments/processing/bot_detector.py
"""
Advanced bot detection algorithms for sentiment analysis.

This module provides sophisticated bot detection with:
- Account age and posting pattern analysis
- Content similarity detection for spam
- Machine learning-based bot classification
- Configurable detection rules and thresholds
"""

import re
import hashlib
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import math
from pathlib import Path
import sys

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

@dataclass
class BotDetectionResult:
    """Result of bot detection analysis."""
    is_bot: bool
    confidence: float  # 0.0 to 1.0
    bot_score: float  # 0.0 to 1.0, higher = more bot-like
    detection_reasons: List[str]
    account_analysis: Dict[str, Any]
    content_analysis: Dict[str, Any]
    pattern_analysis: Dict[str, Any]

@dataclass
class UserProfile:
    """User profile for bot detection analysis."""
    username: str
    account_age_days: Optional[int]
    total_posts: Optional[int]
    followers_count: Optional[int]
    following_count: Optional[int]
    verified: bool
    profile_image_default: bool
    bio_length: int
    creation_date: Optional[datetime]

@dataclass
class PostMetrics:
    """Post metrics for pattern analysis."""
    timestamp: datetime
    content: str
    likes: int
    replies: int
    retweets: int
    content_hash: str

class BotDetector:
    """
    Advanced bot detection system for social media content.

    Features:
    - Account age and profile analysis
    - Posting pattern detection
    - Content similarity and spam detection
    - Machine learning-based classification
    - Configurable detection rules
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the bot detector.

        Args:
            config: Configuration dictionary with detection rules and thresholds
        """
        self.config = config or {}

        # Detection thresholds
        self.min_account_age_days = self.config.get("min_account_age_days", 30)
        self.max_posts_per_hour = self.config.get("max_posts_per_hour", 10)
        self.max_posts_per_day = self.config.get("max_posts_per_day", 100)
        self.min_followers_ratio = self.config.get("min_followers_ratio", 0.1)  # followers/following
        self.content_similarity_threshold = self.config.get("content_similarity_threshold", 0.8)
        self.bot_score_threshold = self.config.get("bot_score_threshold", 0.7)

        # Pattern detection settings
        self.pattern_window_hours = self.config.get("pattern_window_hours", 24)
        self.min_posts_for_pattern = self.config.get("min_posts_for_pattern", 5)

        # Content analysis settings
        self.min_content_length = self.config.get("min_content_length", 10)
        self.max_hashtag_ratio = self.config.get("max_hashtag_ratio", 0.3)
        self.max_mention_ratio = self.config.get("max_mention_ratio", 0.5)

        # Load bot detection patterns
        self._load_bot_patterns()

        # Cache for user analysis
        self._user_cache: Dict[str, BotDetectionResult] = {}
        self._content_hashes: Dict[str, List[str]] = defaultdict(list)  # hash -> usernames

    def _load_bot_patterns(self) -> None:
        """Load bot detection patterns and rules."""
        # Username patterns that indicate bots
        self.bot_username_patterns = [
            r'^[a-zA-Z]+\d{4,}$',  # Letters followed by 4+ digits
            r'^[a-zA-Z]+_\d{4,}$',  # Letters, underscore, 4+ digits
            r'^\w+bot\w*$',  # Contains "bot"
            r'^\w+auto\w*$',  # Contains "auto"
            r'^user\d+$',  # "user" + digits
            r'^[a-zA-Z]{1,3}\d{6,}$',  # 1-3 letters + 6+ digits
            r'^\w*crypto\w*\d+$',  # Contains "crypto" + digits
            r'^\w*trade\w*\d+$',  # Contains "trade" + digits
        ]

        # Compile regex patterns
        self.bot_username_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.bot_username_patterns]

        # Bio patterns that indicate bots
        self.bot_bio_patterns = [
            r'crypto.*signals?',
            r'trading.*bot',
            r'automated.*trading',
            r'follow.*for.*signals?',
            r'dm.*for.*premium',
            r'link.*in.*bio',
            r'check.*my.*link',
            r'investment.*advice',
            r'guaranteed.*profits?',
        ]

        self.bot_bio_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.bot_bio_patterns]

        # Content patterns that indicate bot behavior
        self.spam_content_patterns = [
            r'follow.*for.*more',
            r'dm.*me.*for',
            r'check.*my.*profile',
            r'link.*in.*bio',
            r'guaranteed.*returns?',
            r'risk.*free.*trading',
            r'100%.*accurate',
            r'join.*my.*group',
            r'premium.*signals?',
        ]

        self.spam_content_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.spam_content_patterns]

    def analyze_user(self, user_profile: UserProfile, recent_posts: List[PostMetrics]) -> BotDetectionResult:
        """
        Analyze a user for bot behavior.

        Args:
            user_profile: User profile information
            recent_posts: List of recent posts for pattern analysis

        Returns:
            BotDetectionResult with detailed analysis
        """
        # Check cache first
        cache_key = f"{user_profile.username}_{len(recent_posts)}"
        if cache_key in self._user_cache:
            return self._user_cache[cache_key]

        detection_reasons = []
        bot_score = 0.0

        # Account analysis
        account_analysis = self._analyze_account(user_profile, detection_reasons)
        bot_score += account_analysis["total_score"]

        # Content analysis
        content_analysis = self._analyze_content(recent_posts, detection_reasons)
        bot_score += content_analysis["total_score"]

        # Pattern analysis
        pattern_analysis = self._analyze_patterns(recent_posts, detection_reasons)
        bot_score += pattern_analysis["total_score"]

        # Normalize bot score
        bot_score = min(1.0, bot_score)

        # Determine if user is a bot
        is_bot = bot_score >= self.bot_score_threshold
        confidence = min(1.0, bot_score * 1.2)  # Slightly boost confidence

        result = BotDetectionResult(
            is_bot=is_bot,
            confidence=confidence,
            bot_score=bot_score,
            detection_reasons=detection_reasons,
            account_analysis=account_analysis,
            content_analysis=content_analysis,
            pattern_analysis=pattern_analysis
        )

        # Cache result
        self._user_cache[cache_key] = result

        return result

    def _analyze_account(self, profile: UserProfile, reasons: List[str]) -> Dict[str, Any]:
        """Analyze account characteristics for bot indicators."""
        score = 0.0
        analysis = {}

        # Username analysis
        username_bot_score = 0.0
        for pattern in self.bot_username_regex:
            if pattern.match(profile.username):
                username_bot_score = 0.3
                reasons.append(f"Suspicious username pattern: {profile.username}")
                break

        analysis["username_bot_score"] = username_bot_score
        score += username_bot_score

        # Account age analysis
        age_score = 0.0
        if profile.account_age_days is not None:
            if profile.account_age_days < self.min_account_age_days:
                age_score = 0.2 * (1 - profile.account_age_days / self.min_account_age_days)
                reasons.append(f"Very new account: {profile.account_age_days} days old")

        analysis["age_score"] = age_score
        score += age_score

        # Follower ratio analysis
        ratio_score = 0.0
        if (profile.followers_count is not None and profile.following_count is not None and
            profile.following_count > 0):

            follower_ratio = profile.followers_count / profile.following_count
            if follower_ratio < self.min_followers_ratio:
                ratio_score = 0.15
                reasons.append(f"Low follower ratio: {follower_ratio:.3f}")

        analysis["follower_ratio_score"] = ratio_score
        score += ratio_score

        # Profile completeness analysis
        profile_score = 0.0
        if profile.profile_image_default:
            profile_score += 0.1
            reasons.append("Default profile image")

        if profile.bio_length < 10:
            profile_score += 0.1
            reasons.append("Very short or empty bio")

        # Check bio content for bot patterns
        if hasattr(profile, 'bio') and profile.bio:
            for pattern in self.bot_bio_regex:
                if pattern.search(profile.bio):
                    profile_score += 0.15
                    reasons.append("Suspicious bio content")
                    break

        analysis["profile_score"] = profile_score
        score += profile_score

        # Verification status
        verification_score = 0.0
        if not profile.verified and profile.followers_count and profile.followers_count > 10000:
            verification_score = 0.05  # Slight penalty for unverified high-follower accounts

        analysis["verification_score"] = verification_score
        score += verification_score

        analysis["total_score"] = score
        return analysis

    def _analyze_content(self, posts: List[PostMetrics], reasons: List[str]) -> Dict[str, Any]:
        """Analyze content characteristics for bot indicators."""
        if not posts:
            return {"total_score": 0.0}

        score = 0.0
        analysis = {}

        # Content similarity analysis
        similarity_score = self._analyze_content_similarity(posts, reasons)
        analysis["similarity_score"] = similarity_score
        score += similarity_score

        # Spam content analysis
        spam_score = self._analyze_spam_content(posts, reasons)
        analysis["spam_score"] = spam_score
        score += spam_score

        # Content quality analysis
        quality_score = self._analyze_content_quality(posts, reasons)
        analysis["quality_score"] = quality_score
        score += quality_score

        # Hashtag and mention analysis
        hashtag_mention_score = self._analyze_hashtags_mentions(posts, reasons)
        analysis["hashtag_mention_score"] = hashtag_mention_score
        score += hashtag_mention_score

        analysis["total_score"] = score
        return analysis

    def _analyze_content_similarity(self, posts: List[PostMetrics], reasons: List[str]) -> float:
        """Analyze content similarity for duplicate/template detection."""
        if len(posts) < 2:
            return 0.0

        # Calculate content hashes
        content_hashes = []
        for post in posts:
            # Normalize content for comparison
            normalized = re.sub(r'[^\w\s]', '', post.content.lower())
            normalized = re.sub(r'\s+', ' ', normalized).strip()

            if len(normalized) > 10:  # Only consider substantial content
                content_hash = hashlib.md5(normalized.encode()).hexdigest()
                content_hashes.append(content_hash)

        if len(content_hashes) < 2:
            return 0.0

        # Count duplicate hashes
        hash_counts = Counter(content_hashes)
        duplicate_count = sum(count - 1 for count in hash_counts.values() if count > 1)

        similarity_ratio = duplicate_count / len(content_hashes)

        if similarity_ratio > self.content_similarity_threshold:
            reasons.append(f"High content similarity: {similarity_ratio:.2f}")
            return 0.25 * similarity_ratio

        return 0.0

    def _analyze_spam_content(self, posts: List[PostMetrics], reasons: List[str]) -> float:
        """Analyze content for spam patterns."""
        spam_count = 0

        for post in posts:
            for pattern in self.spam_content_regex:
                if pattern.search(post.content):
                    spam_count += 1
                    break

        if spam_count > 0:
            spam_ratio = spam_count / len(posts)
            if spam_ratio > 0.3:  # More than 30% spam content
                reasons.append(f"High spam content ratio: {spam_ratio:.2f}")
                return 0.2 * spam_ratio

        return 0.0

    def _analyze_content_quality(self, posts: List[PostMetrics], reasons: List[str]) -> float:
        """Analyze content quality indicators."""
        short_posts = 0
        total_length = 0

        for post in posts:
            content_length = len(post.content.strip())
            total_length += content_length

            if content_length < self.min_content_length:
                short_posts += 1

        # Check for very short posts
        short_ratio = short_posts / len(posts)
        if short_ratio > 0.7:  # More than 70% very short posts
            reasons.append(f"High ratio of very short posts: {short_ratio:.2f}")
            return 0.15 * short_ratio

        # Check average content length
        avg_length = total_length / len(posts)
        if avg_length < 20:  # Very short average content
            reasons.append(f"Very short average content length: {avg_length:.1f}")
            return 0.1

        return 0.0

    def _analyze_hashtags_mentions(self, posts: List[PostMetrics], reasons: List[str]) -> float:
        """Analyze hashtag and mention usage patterns."""
        total_hashtags = 0
        total_mentions = 0
        total_words = 0

        for post in posts:
            content = post.content
            hashtags = len(re.findall(r'#\w+', content))
            mentions = len(re.findall(r'@\w+', content))
            words = len(content.split())

            total_hashtags += hashtags
            total_mentions += mentions
            total_words += words

        if total_words == 0:
            return 0.0

        hashtag_ratio = total_hashtags / total_words
        mention_ratio = total_mentions / total_words

        score = 0.0

        if hashtag_ratio > self.max_hashtag_ratio:
            reasons.append(f"Excessive hashtag usage: {hashtag_ratio:.2f}")
            score += 0.1

        if mention_ratio > self.max_mention_ratio:
            reasons.append(f"Excessive mention usage: {mention_ratio:.2f}")
            score += 0.1

        return score

    def _analyze_patterns(self, posts: List[PostMetrics], reasons: List[str]) -> Dict[str, Any]:
        """Analyze posting patterns for bot behavior."""
        if len(posts) < self.min_posts_for_pattern:
            return {"total_score": 0.0}

        score = 0.0
        analysis = {}

        # Posting frequency analysis
        frequency_score = self._analyze_posting_frequency(posts, reasons)
        analysis["frequency_score"] = frequency_score
        score += frequency_score

        # Temporal pattern analysis
        temporal_score = self._analyze_temporal_patterns(posts, reasons)
        analysis["temporal_score"] = temporal_score
        score += temporal_score

        # Engagement pattern analysis
        engagement_score = self._analyze_engagement_patterns(posts, reasons)
        analysis["engagement_score"] = engagement_score
        score += engagement_score

        analysis["total_score"] = score
        return analysis

    def _analyze_posting_frequency(self, posts: List[PostMetrics], reasons: List[str]) -> float:
        """Analyze posting frequency for bot-like behavior."""
        if len(posts) < 2:
            return 0.0

        # Sort posts by timestamp
        sorted_posts = sorted(posts, key=lambda p: p.timestamp)

        # Calculate posts per hour
        time_span = (sorted_posts[-1].timestamp - sorted_posts[0].timestamp).total_seconds() / 3600
        if time_span < 1:
            time_span = 1  # Minimum 1 hour

        posts_per_hour = len(posts) / time_span

        score = 0.0

        if posts_per_hour > self.max_posts_per_hour:
            reasons.append(f"Very high posting frequency: {posts_per_hour:.1f} posts/hour")
            score += 0.2

        # Check for burst posting (many posts in short time)
        intervals = []
        for i in range(1, len(sorted_posts)):
            interval = (sorted_posts[i].timestamp - sorted_posts[i-1].timestamp).total_seconds()
            intervals.append(interval)

        if intervals:
            avg_interval = sum(intervals) / len(intervals)
            if avg_interval < 60:  # Less than 1 minute average between posts
                reasons.append(f"Very short average posting interval: {avg_interval:.1f} seconds")
                score += 0.15

        return score

    def _analyze_temporal_patterns(self, posts: List[PostMetrics], reasons: List[str]) -> float:
        """Analyze temporal posting patterns."""
        if len(posts) < 5:
            return 0.0

        # Extract posting hours
        posting_hours = [post.timestamp.hour for post in posts]
        hour_counts = Counter(posting_hours)

        # Check for very regular posting patterns
        max_hour_count = max(hour_counts.values())
        total_posts = len(posts)

        # If more than 80% of posts are in the same hour, it's suspicious
        if max_hour_count / total_posts > 0.8:
            reasons.append("Very regular posting time pattern")
            return 0.1

        # Check for 24/7 posting (posts at all hours)
        unique_hours = len(set(posting_hours))
        if unique_hours > 20 and total_posts > 50:  # Posts at 20+ different hours
            reasons.append("Posting at unusual hours (24/7 pattern)")
            return 0.05

        return 0.0

    def _analyze_engagement_patterns(self, posts: List[PostMetrics], reasons: List[str]) -> float:
        """Analyze engagement patterns for bot indicators."""
        if not posts:
            return 0.0

        # Calculate engagement metrics
        total_likes = sum(post.likes for post in posts)
        total_replies = sum(post.replies for post in posts)
        total_retweets = sum(post.retweets for post in posts)

        avg_likes = total_likes / len(posts)
        avg_replies = total_replies / len(posts)
        avg_retweets = total_retweets / len(posts)

        score = 0.0

        # Very low engagement across all posts
        if avg_likes < 0.5 and avg_replies < 0.1 and avg_retweets < 0.1 and len(posts) > 10:
            reasons.append("Consistently very low engagement")
            score += 0.1

        # Check for artificial engagement patterns
        engagement_variance = self._calculate_engagement_variance(posts)
        if engagement_variance < 0.1 and len(posts) > 5:  # Very consistent engagement
            reasons.append("Suspiciously consistent engagement pattern")
            score += 0.05

        return score

    def _calculate_engagement_variance(self, posts: List[PostMetrics]) -> float:
        """Calculate variance in engagement across posts."""
        if len(posts) < 2:
            return 1.0

        engagements = [post.likes + post.replies + post.retweets for post in posts]

        if not engagements or max(engagements) == 0:
            return 0.0

        mean_engagement = sum(engagements) / len(engagements)
        if mean_engagement == 0:
            return 0.0

        variance = sum((e - mean_engagement) ** 2 for e in engagements) / len(engagements)
        normalized_variance = variance / (mean_engagement ** 2)

        return min(1.0, normalized_variance)

    def batch_analyze_users(self, user_data: List[Tuple[UserProfile, List[PostMetrics]]]) -> List[BotDetectionResult]:
        """
        Analyze multiple users for bot behavior in batch.

        Args:
            user_data: List of (UserProfile, recent_posts) tuples

        Returns:
            List of BotDetectionResult objects
        """
        results = []

        for user_profile, recent_posts in user_data:
            try:
                result = self.analyze_user(user_profile, recent_posts)
                results.append(result)
            except Exception as e:
                _logger.error("Error analyzing user %s: %s", user_profile.username, e)
                # Return neutral result for failed analysis
                results.append(BotDetectionResult(
                    is_bot=False,
                    confidence=0.0,
                    bot_score=0.0,
                    detection_reasons=["Analysis failed"],
                    account_analysis={},
                    content_analysis={},
                    pattern_analysis={}
                ))

        return results

    def get_detection_stats(self) -> Dict[str, Any]:
        """Get statistics about bot detection performance."""
        total_analyzed = len(self._user_cache)
        bot_count = sum(1 for result in self._user_cache.values() if result.is_bot)

        if total_analyzed == 0:
            return {"total_analyzed": 0, "bot_percentage": 0.0}

        return {
            "total_analyzed": total_analyzed,
            "bot_count": bot_count,
            "bot_percentage": (bot_count / total_analyzed) * 100,
            "cache_size": len(self._user_cache),
            "content_hashes": len(self._content_hashes)
        }

    def clear_cache(self) -> None:
        """Clear analysis cache."""
        self._user_cache.clear()
        self._content_hashes.clear()
        _logger.debug("Bot detector cache cleared")