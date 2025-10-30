# src/common/sentiments/processing/virality_calculator.py
"""
Virality and engagement metrics calculation for sentiment analysis.

This module provides comprehensive virality analysis with:
- Virality index calculation based on engagement patterns
- Platform-specific engagement weighting
- Trending sentiment detection algorithms
- Influence scoring for high-impact accounts
"""

import math
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from collections import defaultdict, Counter
from enum import Enum
from pathlib import Path
import sys

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

class Platform(Enum):
    """Social media platform types."""
    TWITTER = "twitter"
    REDDIT = "reddit"
    STOCKTWITS = "stocktwits"
    DISCORD = "discord"
    NEWS = "news"
    GENERAL = "general"

@dataclass
class EngagementMetrics:
    """Engagement metrics for a post or message."""
    likes: int = 0
    replies: int = 0
    retweets: int = 0
    shares: int = 0
    views: int = 0
    comments: int = 0
    upvotes: int = 0
    downvotes: int = 0
    reactions: int = 0

    def total_engagement(self) -> int:
        """Calculate total engagement across all metrics."""
        return (self.likes + self.replies + self.retweets + self.shares +
                self.comments + self.upvotes + self.reactions)

@dataclass
class AuthorInfluence:
    """Author influence metrics."""
    username: str
    followers_count: int = 0
    following_count: int = 0
    verified: bool = False
    account_age_days: Optional[int] = None
    total_posts: int = 0
    avg_engagement: float = 0.0
    influence_score: float = 0.0

@dataclass
class ViralityResult:
    """Result of virality analysis."""
    virality_index: float  # 0.0 to 1.0+
    engagement_score: float
    velocity_score: float
    reach_score: float
    influence_score: float
    trending_score: float
    platform_normalized_score: float
    breakdown: Dict[str, float]
    top_contributors: List[Tuple[str, float]]  # (username, contribution_score)

@dataclass
class PostData:
    """Post data for virality analysis."""
    id: str
    content: str
    author: AuthorInfluence
    timestamp: datetime
    engagement: EngagementMetrics
    platform: Platform
    hashtags: List[str] = None
    mentions: List[str] = None

    def __post_init__(self):
        if self.hashtags is None:
            self.hashtags = []
        if self.mentions is None:
            self.mentions = []

class ViralityCalculator:
    """
    Advanced virality and engagement metrics calculator.

    Features:
    - Multi-platform virality index calculation
    - Engagement velocity and acceleration analysis
    - Influence-weighted virality scoring
    - Trending detection algorithms
    - Platform-specific normalization
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the virality calculator.

        Args:
            config: Configuration dictionary with calculation parameters
        """
        self.config = config or {}

        # Platform-specific engagement weights
        self.platform_weights = self._load_platform_weights()

        # Virality calculation parameters
        self.time_decay_factor = self.config.get("time_decay_factor", 0.1)
        self.velocity_window_hours = self.config.get("velocity_window_hours", 6)
        self.trending_threshold = self.config.get("trending_threshold", 0.7)
        self.min_posts_for_trending = self.config.get("min_posts_for_trending", 10)

        # Influence scoring parameters
        self.follower_weight = self.config.get("follower_weight", 0.3)
        self.engagement_weight = self.config.get("engagement_weight", 0.4)
        self.verification_bonus = self.config.get("verification_bonus", 0.2)
        self.account_age_weight = self.config.get("account_age_weight", 0.1)

        # Caching for performance
        self._influence_cache: Dict[str, float] = {}
        self._platform_baselines: Dict[Platform, Dict[str, float]] = {}

    def _load_platform_weights(self) -> Dict[Platform, Dict[str, float]]:
        """Load platform-specific engagement weights."""
        default_weights = {
            Platform.TWITTER: {
                "likes": 1.0,
                "retweets": 3.0,
                "replies": 2.0,
                "views": 0.01,
                "base_multiplier": 1.0
            },
            Platform.REDDIT: {
                "upvotes": 2.0,
                "downvotes": -1.0,
                "comments": 3.0,
                "shares": 2.5,
                "base_multiplier": 1.2
            },
            Platform.STOCKTWITS: {
                "likes": 1.5,
                "replies": 2.5,
                "retweets": 3.5,
                "base_multiplier": 1.3
            },
            Platform.DISCORD: {
                "reactions": 1.0,
                "replies": 2.0,
                "base_multiplier": 0.8
            },
            Platform.NEWS: {
                "shares": 4.0,
                "comments": 2.0,
                "likes": 1.0,
                "base_multiplier": 2.0
            },
            Platform.GENERAL: {
                "likes": 1.0,
                "replies": 2.0,
                "shares": 2.5,
                "base_multiplier": 1.0
            }
        }

        # Override with config values
        custom_weights = self.config.get("platform_weights", {})
        for platform_name, weights in custom_weights.items():
            try:
                platform = Platform(platform_name)
                if platform in default_weights:
                    default_weights[platform].update(weights)
                else:
                    default_weights[platform] = weights
            except ValueError:
                _logger.warning("Unknown platform in config: %s", platform_name)

        return default_weights

    def calculate_virality(self, posts: List[PostData], ticker: Optional[str] = None) -> ViralityResult:
        """
        Calculate comprehensive virality metrics for a collection of posts.

        Args:
            posts: List of posts to analyze
            ticker: Optional ticker symbol for context

        Returns:
            ViralityResult with detailed virality analysis
        """
        if not posts:
            return self._create_empty_result()

        # Sort posts by timestamp
        sorted_posts = sorted(posts, key=lambda p: p.timestamp)

        # Calculate individual components
        engagement_score = self._calculate_engagement_score(sorted_posts)
        velocity_score = self._calculate_velocity_score(sorted_posts)
        reach_score = self._calculate_reach_score(sorted_posts)
        influence_score = self._calculate_influence_score(sorted_posts)
        trending_score = self._calculate_trending_score(sorted_posts)

        # Platform normalization
        platform_normalized_score = self._normalize_by_platform(sorted_posts, engagement_score)

        # Calculate overall virality index
        virality_index = self._calculate_virality_index(
            engagement_score, velocity_score, reach_score, influence_score, trending_score
        )

        # Find top contributors
        top_contributors = self._find_top_contributors(sorted_posts)

        # Create breakdown
        breakdown = {
            "engagement": engagement_score,
            "velocity": velocity_score,
            "reach": reach_score,
            "influence": influence_score,
            "trending": trending_score,
            "platform_normalized": platform_normalized_score
        }

        return ViralityResult(
            virality_index=virality_index,
            engagement_score=engagement_score,
            velocity_score=velocity_score,
            reach_score=reach_score,
            influence_score=influence_score,
            trending_score=trending_score,
            platform_normalized_score=platform_normalized_score,
            breakdown=breakdown,
            top_contributors=top_contributors
        )

    def _calculate_engagement_score(self, posts: List[PostData]) -> float:
        """Calculate engagement score based on platform-weighted metrics."""
        if not posts:
            return 0.0

        total_weighted_engagement = 0.0
        total_posts = len(posts)

        for post in posts:
            platform_weights = self.platform_weights.get(post.platform, self.platform_weights[Platform.GENERAL])

            # Calculate weighted engagement for this post
            weighted_engagement = 0.0
            engagement = post.engagement

            # Apply platform-specific weights
            weighted_engagement += engagement.likes * platform_weights.get("likes", 1.0)
            weighted_engagement += engagement.replies * platform_weights.get("replies", 2.0)
            weighted_engagement += engagement.retweets * platform_weights.get("retweets", 3.0)
            weighted_engagement += engagement.shares * platform_weights.get("shares", 2.5)
            weighted_engagement += engagement.comments * platform_weights.get("comments", 2.0)
            weighted_engagement += engagement.upvotes * platform_weights.get("upvotes", 2.0)
            weighted_engagement += engagement.downvotes * platform_weights.get("downvotes", -1.0)
            weighted_engagement += engagement.reactions * platform_weights.get("reactions", 1.0)
            weighted_engagement += engagement.views * platform_weights.get("views", 0.01)

            # Apply platform base multiplier
            base_multiplier = platform_weights.get("base_multiplier", 1.0)
            weighted_engagement *= base_multiplier

            # Apply time decay
            time_decay = self._calculate_time_decay(post.timestamp)
            weighted_engagement *= time_decay

            total_weighted_engagement += weighted_engagement

        # Normalize by number of posts and apply logarithmic scaling
        avg_engagement = total_weighted_engagement / total_posts

        # Apply logarithmic scaling to handle wide range of engagement values
        if avg_engagement > 0:
            engagement_score = math.log10(avg_engagement + 1) / 5.0  # Normalize to roughly 0-1 range
        else:
            engagement_score = 0.0

        return min(1.0, engagement_score)

    def _calculate_velocity_score(self, posts: List[PostData]) -> float:
        """Calculate engagement velocity (rate of engagement growth)."""
        if len(posts) < 2:
            return 0.0

        # Group posts by time windows
        window_hours = self.velocity_window_hours
        time_windows = defaultdict(list)

        base_time = posts[0].timestamp
        for post in posts:
            hours_diff = (post.timestamp - base_time).total_seconds() / 3600
            window_idx = int(hours_diff / window_hours)
            time_windows[window_idx].append(post)

        if len(time_windows) < 2:
            return 0.0

        # Calculate engagement per window
        window_engagements = []
        for window_idx in sorted(time_windows.keys()):
            window_posts = time_windows[window_idx]
            total_engagement = sum(post.engagement.total_engagement() for post in window_posts)
            window_engagements.append(total_engagement)

        # Calculate velocity (acceleration of engagement)
        velocities = []
        for i in range(1, len(window_engagements)):
            if window_engagements[i-1] > 0:
                velocity = (window_engagements[i] - window_engagements[i-1]) / window_engagements[i-1]
            else:
                velocity = 1.0 if window_engagements[i] > 0 else 0.0
            velocities.append(velocity)

        if not velocities:
            return 0.0

        # Average positive velocity (growth rate)
        positive_velocities = [v for v in velocities if v > 0]
        if not positive_velocities:
            return 0.0

        avg_velocity = sum(positive_velocities) / len(positive_velocities)

        # Normalize to 0-1 range
        velocity_score = min(1.0, avg_velocity / 2.0)  # Divide by 2 to normalize 100% growth to 0.5

        return velocity_score

    def _calculate_reach_score(self, posts: List[PostData]) -> float:
        """Calculate reach score based on unique authors and potential audience."""
        if not posts:
            return 0.0

        # Count unique authors
        unique_authors = set(post.author.username for post in posts)
        author_diversity = len(unique_authors) / len(posts)

        # Calculate potential reach based on follower counts
        total_potential_reach = 0
        for post in posts:
            # Estimate reach based on engagement and follower count
            follower_reach = post.author.followers_count * 0.1  # Assume 10% reach rate
            engagement_amplification = post.engagement.total_engagement() * 5  # Each engagement reaches 5 more people
            post_reach = follower_reach + engagement_amplification
            total_potential_reach += post_reach

        # Normalize reach score
        avg_reach = total_potential_reach / len(posts)

        # Apply logarithmic scaling
        if avg_reach > 0:
            reach_score = math.log10(avg_reach + 1) / 6.0  # Normalize to roughly 0-1 range
        else:
            reach_score = 0.0

        # Combine with author diversity
        final_reach_score = (reach_score * 0.7) + (author_diversity * 0.3)

        return min(1.0, final_reach_score)

    def _calculate_influence_score(self, posts: List[PostData]) -> float:
        """Calculate influence score based on author influence metrics."""
        if not posts:
            return 0.0

        total_influence = 0.0

        for post in posts:
            author_influence = self._get_author_influence_score(post.author)

            # Weight by post engagement
            post_engagement = post.engagement.total_engagement()
            engagement_weight = math.sqrt(post_engagement + 1)  # Square root to reduce extreme values

            weighted_influence = author_influence * engagement_weight
            total_influence += weighted_influence

        # Normalize by number of posts
        avg_influence = total_influence / len(posts)

        # Apply scaling to fit 0-1 range
        influence_score = min(1.0, avg_influence / 100.0)  # Assuming max weighted influence around 100

        return influence_score

    def _get_author_influence_score(self, author: AuthorInfluence) -> float:
        """Calculate influence score for an individual author."""
        # Check cache first
        if author.username in self._influence_cache:
            return self._influence_cache[author.username]

        influence_score = 0.0

        # Follower count component (logarithmic scaling)
        if author.followers_count > 0:
            follower_score = math.log10(author.followers_count + 1) * self.follower_weight
            influence_score += follower_score

        # Engagement rate component
        if author.total_posts > 0 and author.avg_engagement > 0:
            engagement_rate = author.avg_engagement / max(1, author.followers_count)
            engagement_score = min(10.0, engagement_rate * 1000) * self.engagement_weight
            influence_score += engagement_score

        # Verification bonus
        if author.verified:
            influence_score += self.verification_bonus * 10

        # Account age component (older accounts are more trustworthy)
        if author.account_age_days:
            age_score = min(5.0, author.account_age_days / 365) * self.account_age_weight * 10
            influence_score += age_score

        # Cache the result
        self._influence_cache[author.username] = influence_score

        return influence_score

    def _calculate_trending_score(self, posts: List[PostData]) -> float:
        """Calculate trending score based on hashtag and mention patterns."""
        if len(posts) < self.min_posts_for_trending:
            return 0.0

        # Analyze hashtag frequency
        hashtag_counts = Counter()
        mention_counts = Counter()

        for post in posts:
            hashtag_counts.update(post.hashtags)
            mention_counts.update(post.mentions)

        # Calculate hashtag trending score
        hashtag_score = 0.0
        if hashtag_counts:
            # Find most common hashtags
            top_hashtags = hashtag_counts.most_common(5)
            total_hashtag_usage = sum(hashtag_counts.values())

            for hashtag, count in top_hashtags:
                # Score based on frequency and concentration
                frequency_score = count / len(posts)  # How often this hashtag appears
                concentration_score = count / total_hashtag_usage  # How dominant this hashtag is
                hashtag_score += (frequency_score * concentration_score) * 0.5

        # Calculate mention trending score
        mention_score = 0.0
        if mention_counts:
            top_mentions = mention_counts.most_common(3)
            total_mention_usage = sum(mention_counts.values())

            for mention, count in top_mentions:
                frequency_score = count / len(posts)
                concentration_score = count / total_mention_usage
                mention_score += (frequency_score * concentration_score) * 0.3

        # Combine scores
        trending_score = hashtag_score + mention_score

        return min(1.0, trending_score)

    def _normalize_by_platform(self, posts: List[PostData], base_score: float) -> float:
        """Normalize score by platform-specific baselines."""
        if not posts:
            return base_score

        # Group posts by platform
        platform_counts = Counter(post.platform for post in posts)

        # Calculate weighted normalization
        normalized_score = 0.0
        total_weight = 0.0

        for platform, count in platform_counts.items():
            platform_baseline = self._get_platform_baseline(platform)
            platform_weight = count / len(posts)

            # Normalize base score by platform baseline
            if platform_baseline > 0:
                platform_normalized = base_score / platform_baseline
            else:
                platform_normalized = base_score

            normalized_score += platform_normalized * platform_weight
            total_weight += platform_weight

        if total_weight > 0:
            normalized_score /= total_weight

        return min(1.0, normalized_score)

    def _get_platform_baseline(self, platform: Platform) -> float:
        """Get baseline engagement score for a platform."""
        # These are rough baseline values for normalization
        baselines = {
            Platform.TWITTER: 1.0,
            Platform.REDDIT: 1.2,
            Platform.STOCKTWITS: 0.8,
            Platform.DISCORD: 0.6,
            Platform.NEWS: 2.0,
            Platform.GENERAL: 1.0
        }

        return baselines.get(platform, 1.0)

    def _calculate_virality_index(self, engagement: float, velocity: float, reach: float,
                                 influence: float, trending: float) -> float:
        """Calculate overall virality index from component scores."""
        # Weighted combination of components
        weights = {
            "engagement": 0.3,
            "velocity": 0.2,
            "reach": 0.2,
            "influence": 0.15,
            "trending": 0.15
        }

        # Override with config weights if provided
        config_weights = self.config.get("virality_weights", {})
        weights.update(config_weights)

        virality_index = (
            engagement * weights["engagement"] +
            velocity * weights["velocity"] +
            reach * weights["reach"] +
            influence * weights["influence"] +
            trending * weights["trending"]
        )

        return min(1.0, virality_index)

    def _find_top_contributors(self, posts: List[PostData]) -> List[Tuple[str, float]]:
        """Find top contributing authors by influence and engagement."""
        author_contributions = defaultdict(float)

        for post in posts:
            author_influence = self._get_author_influence_score(post.author)
            post_engagement = post.engagement.total_engagement()

            # Contribution score combines influence and engagement
            contribution = author_influence * math.sqrt(post_engagement + 1)
            author_contributions[post.author.username] += contribution

        # Sort by contribution and return top 10
        sorted_contributors = sorted(
            author_contributions.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_contributors[:10]

    def _calculate_time_decay(self, timestamp: datetime) -> float:
        """Calculate time decay factor for engagement."""
        now = datetime.now(timezone.utc)
        hours_ago = (now - timestamp).total_seconds() / 3600

        # Exponential decay: newer posts have higher weight
        decay_factor = math.exp(-self.time_decay_factor * hours_ago)

        return max(0.1, decay_factor)  # Minimum 10% weight for old posts

    def _create_empty_result(self) -> ViralityResult:
        """Create empty virality result."""
        return ViralityResult(
            virality_index=0.0,
            engagement_score=0.0,
            velocity_score=0.0,
            reach_score=0.0,
            influence_score=0.0,
            trending_score=0.0,
            platform_normalized_score=0.0,
            breakdown={},
            top_contributors=[]
        )

    def analyze_trending_topics(self, posts: List[PostData], min_mentions: int = 5) -> Dict[str, float]:
        """
        Analyze trending topics from hashtags and mentions.

        Args:
            posts: List of posts to analyze
            min_mentions: Minimum mentions required for a topic to be considered trending

        Returns:
            Dictionary of trending topics with their trending scores
        """
        if not posts:
            return {}

        # Collect all hashtags and mentions
        all_topics = Counter()

        for post in posts:
            # Add hashtags
            for hashtag in post.hashtags:
                all_topics[f"#{hashtag}"] += 1

            # Add mentions
            for mention in post.mentions:
                all_topics[f"@{mention}"] += 1

        # Filter by minimum mentions and calculate trending scores
        trending_topics = {}

        for topic, count in all_topics.items():
            if count >= min_mentions:
                # Calculate trending score based on frequency and recency
                recent_mentions = 0
                total_engagement = 0

                cutoff_time = datetime.now(timezone.utc) - timedelta(hours=6)  # Last 6 hours

                for post in posts:
                    topic_clean = topic[1:]  # Remove # or @ prefix

                    if (topic.startswith('#') and topic_clean in post.hashtags) or \
                       (topic.startswith('@') and topic_clean in post.mentions):

                        if post.timestamp >= cutoff_time:
                            recent_mentions += 1

                        total_engagement += post.engagement.total_engagement()

                # Trending score combines frequency, recency, and engagement
                frequency_score = count / len(posts)
                recency_score = recent_mentions / max(1, count)
                engagement_score = total_engagement / max(1, count)

                trending_score = (
                    frequency_score * 0.4 +
                    recency_score * 0.3 +
                    min(1.0, engagement_score / 100) * 0.3
                )

                trending_topics[topic] = trending_score

        # Sort by trending score
        return dict(sorted(trending_topics.items(), key=lambda x: x[1], reverse=True))

    def get_virality_stats(self) -> Dict[str, Any]:
        """Get statistics about virality calculations."""
        return {
            "influence_cache_size": len(self._influence_cache),
            "platform_baselines": dict(self._platform_baselines),
            "config": self.config
        }

    def clear_cache(self) -> None:
        """Clear calculation cache."""
        self._influence_cache.clear()
        self._platform_baselines.clear()
        _logger.debug("Virality calculator cache cleared")