# Sentiment Integration Implementation Plan
## p04_short_squeeze Pipeline Enhancement

**Document Version**: 1.0
**Created**: 2025-11-17
**Author**: System Architecture Team
**Status**: Planning Phase

---

## Executive Summary

This document outlines the integration of the `src/common/sentiments` multi-source sentiment module into the p04_short_squeeze pipeline, replacing the current single-source Finnhub sentiment with a comprehensive 7-provider sentiment system.

**Key Benefits**:
- 7 data sources vs 1 (StockTwits, Reddit, News APIs, Google Trends, Twitter, Discord, HuggingFace ML)
- Advanced analytics: bot detection, virality scoring, mention growth trends
- Reduces Finnhub API usage by ~33%
- Richer signal quality with multi-source aggregation
- Quality weighting and graceful degradation

**Estimated Effort**: 6-8 hours
**Risk Level**: LOW (well-tested module, async-ready, standardized interface)

---

## Table of Contents

1. [Current State Analysis](#1-current-state-analysis)
2. [Target Architecture](#2-target-architecture)
3. [Implementation Phases](#3-implementation-phases)
4. [Database Schema Changes](#4-database-schema-changes)
5. [Code Modifications](#5-code-modifications)
6. [Configuration Changes](#6-configuration-changes)
7. [Testing Strategy](#7-testing-strategy)
8. [Rollback Plan](#8-rollback-plan)
9. [Performance Considerations](#9-performance-considerations)
10. [Future Enhancements](#10-future-enhancements)

---

## 1. Current State Analysis

### 1.1 Existing Sentiment Usage

**Location**: `src/ml/pipeline/p04_short_squeeze/core/daily_deep_scan.py:551`

```python
# Current implementation
sentiment_score = await self._safe_api_call(
    self.finnhub_downloader.aggregate_24h_sentiment,
    ticker,
    api_name="Finnhub Sentiment"
)
candidate.sentiment_24h = sentiment_score if sentiment_score is not None else 0.0
```

**Current Metrics**:
- Single source: Finnhub news sentiment only
- Single metric: `sentiment_24h` (0-1 normalized)
- Weight in scoring: 12.5% (1/4 transient metrics × 50% transient weight)
- API calls: ~50-200/day depending on candidate count
- No temporal trends or growth analysis
- No bot detection or quality filtering

### 1.2 Current Scoring Model

**From**: `src/ml/pipeline/p04_short_squeeze/core/scoring_engine.py`

```python
# Transient metrics (50% of final score)
transient_score = (
    volume_spike_score * 0.25 +      # 12.5% of total
    sentiment_score * 0.25 +         # 12.5% of total
    call_put_score * 0.25 +          # 12.5% of total
    borrow_fee_score * 0.25          # 12.5% of total
)

# FINRA metrics (30% of final score)
finra_score = si_score * 0.6 + dtc_score * 0.4

# Screener score (20% of final score)
final_score = transient_score * 0.5 + finra_score * 0.3 + screener_score * 0.2
```

### 1.3 Database Schema (Current)

**Table**: `ss_deep_metrics`

```sql
-- Existing sentiment field
sentiment_24h FLOAT,  -- Single score from Finnhub
```

### 1.4 Dependencies

**Current**:
- `src/data/downloader/finnhub_data_downloader.py`
- Environment: `FINNHUB_API_KEY`

**New**:
- `src/common/sentiments/collect_sentiment_async.py`
- Environment: `SENTIMENT_STOCKTWITS_ENABLED`, `SENTIMENT_REDDIT_ENABLED`, etc.

---

## 2. Target Architecture

### 2.1 Enhanced Sentiment Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    Daily Deep Scan (10:00 AM)                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              Load Candidates from ss_snapshot                    │
│                  (~50-200 tickers)                               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│         Batch Sentiment Collection (NEW)                         │
│   collect_sentiment_batch(tickers, lookback_hours=24)           │
└─────────────────────────────────────────────────────────────────┘
          ↓              ↓              ↓              ↓
   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
   │StockTwits│   │  Reddit  │   │   News   │   │  Trends  │
   │  (0.4)   │   │  (0.3)   │   │  (0.2)   │   │  (0.1)   │
   └──────────┘   └──────────┘   └──────────┘   └──────────┘
          ↓              ↓              ↓              ↓
┌─────────────────────────────────────────────────────────────────┐
│              Sentiment Aggregator + Bot Detector                 │
│  - Multi-source weighted average                                │
│  - Bot percentage filtering                                     │
│  - Virality index calculation                                   │
│  - Mention growth vs 7-day average                              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                  SentimentFeatures Output                        │
│  - sentiment_normalized (0-1)                                   │
│  - mentions_24h, unique_authors_24h                             │
│  - mentions_growth_7d (vs historical avg)                       │
│  - virality_index (engagement-weighted)                         │
│  - bot_pct (estimated bot percentage)                           │
│  - data_quality (per-provider status)                           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              Store in ss_deep_metrics (Enhanced)                 │
│  + Enhanced Scoring Engine Calculation                          │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 New Metrics Integration

**Enhanced Metrics**:
1. **sentiment_24h** (existing) - Multi-source aggregated sentiment (0-1)
2. **mentions_24h** (new) - Total mention count across all sources
3. **mentions_growth_7d** (new) - Percentage growth vs 7-day average
4. **virality_index** (new) - Engagement-weighted virality score (0-1)
5. **bot_pct** (new) - Estimated bot percentage (0-1)
6. **sentiment_data_quality** (new) - JSON field with per-provider status

### 2.3 Enhanced Scoring Model

**Proposed Weights**:

```python
# Transient metrics (55% of final score - increased from 50%)
transient_score = (
    volume_spike_score * 0.30 +      # 16.5% of total (was 12.5%)
    sentiment_score * 0.35 +         # 19.25% of total (was 12.5%) ← INCREASED
    virality_score * 0.15 +          # 8.25% of total (NEW)
    call_put_score * 0.10 +          # 5.5% of total (was 12.5%)
    borrow_fee_score * 0.10          # 5.5% of total (was 12.5%)
)

# FINRA metrics (30% of final score - unchanged)
finra_score = si_score * 0.6 + dtc_score * 0.4

# Screener score (15% of final score - decreased from 20%)
final_score = transient_score * 0.55 + finra_score * 0.3 + screener_score * 0.15
```

**Rationale**:
- Sentiment now has more reliable multi-source data → increase weight
- Virality is a strong early indicator → add new dimension
- Volume spike remains most important transient signal
- Reduce call/put and borrow fee (often noisy/incomplete data)

---

## 3. Implementation Phases

### Phase 1: Database Schema Updates (1 hour)
- Add new columns to `ss_deep_metrics` table
- Create migration script with backward compatibility
- Test migration on development database

### Phase 2: Sentiment Module Integration (2 hours)
- Update `daily_deep_scan.py` to use `collect_sentiment_batch()`
- Add historical lookup function for mention growth
- Implement batch processing for candidates
- Add error handling and fallback logic

### Phase 3: Scoring Engine Updates (1 hour)
- Update `scoring_engine.py` with new weights
- Add virality scoring function
- Update normalization logic for new metrics
- Add metric validation

### Phase 4: Configuration Updates (1 hour)
- Add sentiment module config to pipeline YAML
- Add environment variable documentation
- Create example configuration file
- Update config loader and validation

### Phase 5: Testing & Validation (2 hours)
- Unit tests for new scoring logic
- Integration tests with mock sentiment data
- End-to-end test with small candidate set
- Performance benchmarking

### Phase 6: Documentation & Deployment (1 hour)
- Update pipeline documentation
- Create operations runbook
- Add monitoring alerts
- Deploy to production with feature flag

---

## 4. Database Schema Changes

### 4.1 Migration Script

**File**: `src/ml/pipeline/p04_short_squeeze/data/migrations/add_sentiment_metrics.sql`

```sql
-- Migration: Add enhanced sentiment metrics to ss_deep_metrics
-- Version: 1.0
-- Date: 2025-11-17

BEGIN;

-- Add new columns for enhanced sentiment
ALTER TABLE ss_deep_metrics
ADD COLUMN IF NOT EXISTS mentions_24h INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS mentions_growth_7d FLOAT DEFAULT NULL,
ADD COLUMN IF NOT EXISTS virality_index FLOAT DEFAULT 0.0,
ADD COLUMN IF NOT EXISTS bot_pct FLOAT DEFAULT 0.0,
ADD COLUMN IF NOT EXISTS sentiment_data_quality JSONB DEFAULT '{}'::jsonb;

-- Add indexes for common queries
CREATE INDEX IF NOT EXISTS idx_ss_deep_metrics_virality
ON ss_deep_metrics(virality_index DESC)
WHERE virality_index > 0.5;

CREATE INDEX IF NOT EXISTS idx_ss_deep_metrics_mentions_growth
ON ss_deep_metrics(mentions_growth_7d DESC)
WHERE mentions_growth_7d IS NOT NULL;

-- Add check constraints for data validation
ALTER TABLE ss_deep_metrics
ADD CONSTRAINT check_virality_range CHECK (virality_index >= 0 AND virality_index <= 1),
ADD CONSTRAINT check_bot_pct_range CHECK (bot_pct >= 0 AND bot_pct <= 1),
ADD CONSTRAINT check_mentions_positive CHECK (mentions_24h >= 0);

-- Add comment for documentation
COMMENT ON COLUMN ss_deep_metrics.mentions_24h IS
'Total mention count across all sentiment sources (StockTwits, Reddit, News, etc.)';

COMMENT ON COLUMN ss_deep_metrics.mentions_growth_7d IS
'Percentage growth in mentions compared to 7-day historical average. NULL if no historical data.';

COMMENT ON COLUMN ss_deep_metrics.virality_index IS
'Engagement-weighted virality score (0-1) based on likes, replies, retweets, and influence.';

COMMENT ON COLUMN ss_deep_metrics.bot_pct IS
'Estimated percentage of bot activity (0-1) based on account age, posting patterns, and content similarity.';

COMMENT ON COLUMN ss_deep_metrics.sentiment_data_quality IS
'JSON object with per-provider status: {"stocktwits": "ok", "reddit": "partial", "news": "failed"}';

COMMIT;
```

### 4.2 Historical Mentions Table (Optional)

For calculating `mentions_growth_7d`, we need historical data:

```sql
-- Create table for historical mention tracking
CREATE TABLE IF NOT EXISTS ss_sentiment_history (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(20) NOT NULL,
    date DATE NOT NULL,
    mentions_count INTEGER NOT NULL DEFAULT 0,
    unique_authors INTEGER NOT NULL DEFAULT 0,
    sentiment_avg FLOAT NOT NULL DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT unique_ticker_date UNIQUE (ticker, date)
);

-- Indexes for efficient lookups
CREATE INDEX idx_ss_sentiment_history_ticker_date
ON ss_sentiment_history(ticker, date DESC);

CREATE INDEX idx_ss_sentiment_history_date
ON ss_sentiment_history(date DESC);

-- Retention policy: Keep 30 days of data
-- (Implement via scheduled job or trigger)
```

### 4.3 Rollback Script

```sql
-- Rollback script (if needed)
BEGIN;

ALTER TABLE ss_deep_metrics
DROP COLUMN IF EXISTS mentions_24h,
DROP COLUMN IF EXISTS mentions_growth_7d,
DROP COLUMN IF EXISTS virality_index,
DROP COLUMN IF EXISTS bot_pct,
DROP COLUMN IF EXISTS sentiment_data_quality;

DROP INDEX IF EXISTS idx_ss_deep_metrics_virality;
DROP INDEX IF EXISTS idx_ss_deep_metrics_mentions_growth;

DROP TABLE IF EXISTS ss_sentiment_history;

COMMIT;
```

---

## 5. Code Modifications

### 5.1 Update `daily_deep_scan.py`

**Location**: `src/ml/pipeline/p04_short_squeeze/core/daily_deep_scan.py`

#### 5.1.1 Add Import

```python
# Add to imports section (around line 30)
from src.common.sentiments.collect_sentiment_async import collect_sentiment_batch
```

#### 5.1.2 Add Historical Lookup Function

```python
# Add after class initialization
async def _get_historical_mentions(self, ticker: str) -> Optional[float]:
    """
    Get 7-day average mention count for growth calculation.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Average mentions over past 7 days, or None if insufficient data
    """
    try:
        # Query ss_sentiment_history for 7-day average
        query = """
            SELECT AVG(mentions_count) as avg_mentions
            FROM ss_sentiment_history
            WHERE ticker = %s
            AND date >= CURRENT_DATE - INTERVAL '7 days'
            AND date < CURRENT_DATE
            HAVING COUNT(*) >= 3  -- Require at least 3 days of data
        """
        result = await self.db.fetch_one(query, (ticker,))

        if result and result['avg_mentions']:
            return float(result['avg_mentions'])
        return None

    except Exception as e:
        self.logger.warning(f"Failed to get historical mentions for {ticker}: {e}")
        return None
```

#### 5.1.3 Replace Sentiment Collection (MAIN CHANGE)

**BEFORE** (lines ~540-560):

```python
# Fetch transient metrics
self.logger.info(f"Fetching transient metrics for {len(candidates)} candidates...")

for candidate in candidates:
    ticker = candidate.ticker

    # Sentiment
    sentiment_score = await self._safe_api_call(
        self.finnhub_downloader.aggregate_24h_sentiment,
        ticker,
        api_name="Finnhub Sentiment"
    )
    candidate.sentiment_24h = sentiment_score if sentiment_score is not None else 0.0

    # ... other metrics
```

**AFTER**:

```python
# Fetch transient metrics
self.logger.info(f"Fetching transient metrics for {len(candidates)} candidates...")

# BATCH SENTIMENT COLLECTION (replaces individual Finnhub calls)
ticker_list = [c.ticker for c in candidates]

try:
    # Collect sentiment for all candidates in batch
    sentiment_config = self.config.sentiment_config if hasattr(self.config, 'sentiment_config') else None

    self.logger.info(f"Collecting sentiment data for {len(ticker_list)} tickers...")
    sentiment_map = await collect_sentiment_batch(
        tickers=ticker_list,
        lookback_hours=24,
        config=sentiment_config,
        history_lookup=self._get_historical_mentions,
        output_format="dataclass"
    )

    self.logger.info(f"Sentiment collection complete. Success: {sum(1 for v in sentiment_map.values() if v)}/{len(ticker_list)}")

except Exception as e:
    self.logger.error(f"Batch sentiment collection failed: {e}", exc_info=True)
    sentiment_map = {}  # Empty map, will use defaults

# Process candidates with sentiment data
for candidate in candidates:
    ticker = candidate.ticker

    # Enhanced sentiment metrics
    sentiment_features = sentiment_map.get(ticker)

    if sentiment_features:
        candidate.sentiment_24h = sentiment_features.sentiment_normalized
        candidate.mentions_24h = sentiment_features.mentions_24h
        candidate.mentions_growth_7d = sentiment_features.mentions_growth_7d
        candidate.virality_index = sentiment_features.virality_index
        candidate.bot_pct = sentiment_features.bot_pct
        candidate.sentiment_data_quality = sentiment_features.data_quality

        # Log notable signals
        if sentiment_features.virality_index > 0.7:
            self.logger.info(f"  {ticker}: HIGH VIRALITY detected ({sentiment_features.virality_index:.2f})")
        if sentiment_features.mentions_growth_7d and sentiment_features.mentions_growth_7d > 2.0:
            self.logger.info(f"  {ticker}: MENTION SURGE detected (+{sentiment_features.mentions_growth_7d*100:.0f}%)")
        if sentiment_features.bot_pct > 0.5:
            self.logger.warning(f"  {ticker}: High bot activity detected ({sentiment_features.bot_pct*100:.0f}%)")
    else:
        # Fallback to defaults if sentiment collection failed
        candidate.sentiment_24h = 0.0
        candidate.mentions_24h = 0
        candidate.mentions_growth_7d = None
        candidate.virality_index = 0.0
        candidate.bot_pct = 0.0
        candidate.sentiment_data_quality = {"error": "collection_failed"}

        self.logger.warning(f"  {ticker}: No sentiment data available, using defaults")

    # Continue with other transient metrics (volume spike, call/put ratio, borrow fee)
    # ... existing code for other metrics
```

#### 5.1.4 Update Database Storage

**Location**: Around line 650 in `_store_deep_metrics()`

```python
# Update the INSERT statement to include new fields
insert_query = """
    INSERT INTO ss_deep_metrics (
        ticker, scan_date,
        sentiment_24h, mentions_24h, mentions_growth_7d,
        virality_index, bot_pct, sentiment_data_quality,
        volume_spike_ratio, call_put_ratio, borrow_fee_pct,
        short_interest_pct, days_to_cover,
        screener_score, transient_score, finra_score, final_score,
        alert_level, created_at
    ) VALUES (
        %s, %s,
        %s, %s, %s,
        %s, %s, %s,
        %s, %s, %s,
        %s, %s,
        %s, %s, %s, %s,
        %s, %s
    )
    ON CONFLICT (ticker, scan_date) DO UPDATE SET
        sentiment_24h = EXCLUDED.sentiment_24h,
        mentions_24h = EXCLUDED.mentions_24h,
        mentions_growth_7d = EXCLUDED.mentions_growth_7d,
        virality_index = EXCLUDED.virality_index,
        bot_pct = EXCLUDED.bot_pct,
        sentiment_data_quality = EXCLUDED.sentiment_data_quality,
        -- ... other fields
        updated_at = CURRENT_TIMESTAMP
"""

# Update the parameters tuple
params = (
    candidate.ticker, scan_date,
    candidate.sentiment_24h, candidate.mentions_24h, candidate.mentions_growth_7d,
    candidate.virality_index, candidate.bot_pct, json.dumps(candidate.sentiment_data_quality),
    # ... other parameters
)
```

#### 5.1.5 Add Sentiment History Tracking

```python
# Add after storing deep metrics
async def _update_sentiment_history(self, candidates: List[CandidateModel]):
    """
    Update sentiment history table for trend analysis.

    Args:
        candidates: List of candidates with sentiment data
    """
    try:
        today = datetime.now().date()

        for candidate in candidates:
            if candidate.mentions_24h > 0:  # Only store if we have data
                query = """
                    INSERT INTO ss_sentiment_history (
                        ticker, date, mentions_count, unique_authors, sentiment_avg
                    ) VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (ticker, date) DO UPDATE SET
                        mentions_count = EXCLUDED.mentions_count,
                        unique_authors = EXCLUDED.unique_authors,
                        sentiment_avg = EXCLUDED.sentiment_avg
                """

                await self.db.execute(query, (
                    candidate.ticker,
                    today,
                    candidate.mentions_24h,
                    getattr(candidate, 'unique_authors_24h', 0),
                    candidate.sentiment_24h
                ))

        self.logger.info(f"Updated sentiment history for {len(candidates)} tickers")

    except Exception as e:
        self.logger.error(f"Failed to update sentiment history: {e}", exc_info=True)
```

### 5.2 Update `models.py`

**Location**: `src/ml/pipeline/p04_short_squeeze/core/models.py`

```python
@dataclass
class CandidateModel:
    """Model for short squeeze candidate with metrics."""

    ticker: str

    # ... existing fields

    # Enhanced sentiment metrics
    sentiment_24h: float = 0.0
    mentions_24h: int = 0                          # NEW
    mentions_growth_7d: Optional[float] = None     # NEW
    virality_index: float = 0.0                    # NEW
    bot_pct: float = 0.0                           # NEW
    sentiment_data_quality: Dict[str, str] = None  # NEW

    # ... other fields

    def __post_init__(self):
        """Initialize default values."""
        if self.sentiment_data_quality is None:
            self.sentiment_data_quality = {}
```

### 5.3 Update `scoring_engine.py`

**Location**: `src/ml/pipeline/p04_short_squeeze/core/scoring_engine.py`

```python
class ScoringEngine:
    """Enhanced scoring engine with multi-source sentiment."""

    # Update weights
    TRANSIENT_WEIGHT = 0.55  # Increased from 0.50
    FINRA_WEIGHT = 0.30      # Unchanged
    SCREENER_WEIGHT = 0.15   # Decreased from 0.20

    # Transient metric sub-weights
    VOLUME_SPIKE_WEIGHT = 0.30   # Increased from 0.25
    SENTIMENT_WEIGHT = 0.35      # Increased from 0.25
    VIRALITY_WEIGHT = 0.15       # NEW
    CALL_PUT_WEIGHT = 0.10       # Decreased from 0.25
    BORROW_FEE_WEIGHT = 0.10     # Decreased from 0.25

    def calculate_virality_score(self, virality_index: float, bot_pct: float) -> float:
        """
        Calculate virality score with bot penalty.

        Args:
            virality_index: Raw virality index (0-1)
            bot_pct: Bot percentage (0-1)

        Returns:
            Adjusted virality score (0-1)
        """
        # Penalize high bot activity
        bot_penalty = 1.0 - (bot_pct * 0.5)  # Max 50% penalty

        adjusted_score = virality_index * bot_penalty

        return max(0.0, min(1.0, adjusted_score))

    def calculate_transient_score(self, candidate: CandidateModel) -> float:
        """
        Calculate transient metrics score with enhanced sentiment.

        Args:
            candidate: Candidate with all metrics populated

        Returns:
            Weighted transient score (0-1)
        """
        # Normalize individual metrics
        volume_score = self._normalize_volume_spike(candidate.volume_spike_ratio)
        sentiment_score = candidate.sentiment_24h  # Already normalized
        virality_score = self.calculate_virality_score(
            candidate.virality_index,
            candidate.bot_pct
        )
        call_put_score = self._normalize_call_put_ratio(candidate.call_put_ratio)
        borrow_fee_score = self._normalize_borrow_fee(candidate.borrow_fee_pct)

        # Weighted combination
        transient_score = (
            volume_score * self.VOLUME_SPIKE_WEIGHT +
            sentiment_score * self.SENTIMENT_WEIGHT +
            virality_score * self.VIRALITY_WEIGHT +
            call_put_score * self.CALL_PUT_WEIGHT +
            borrow_fee_score * self.BORROW_FEE_WEIGHT
        )

        # Boost if mentions are growing rapidly
        if candidate.mentions_growth_7d and candidate.mentions_growth_7d > 3.0:
            # Up to 10% boost for 3x+ growth
            growth_boost = min(0.1, (candidate.mentions_growth_7d - 3.0) / 20.0)
            transient_score = min(1.0, transient_score * (1.0 + growth_boost))

        return transient_score

    # ... rest of scoring logic
```

### 5.4 Update Configuration

**File**: `src/ml/pipeline/p04_short_squeeze/config/pipeline_config.yaml`

```yaml
# Enhanced Sentiment Configuration
sentiment_config:
  # Provider toggles
  providers:
    stocktwits: true
    reddit_pushshift: true
    news: true
    google_trends: false  # Optional, conservative rate limit
    twitter: false         # Requires API access
    discord: false         # Requires channel access
    hf_enabled: false      # ML enhancement (CPU intensive)

  # Batch processing
  batching:
    concurrency: 8         # Parallel requests
    rate_limit_delay_sec: 0.3
    batch_size: 50         # Tickers per batch

  # Source weighting
  weights:
    stocktwits: 0.4        # High quality, trading-focused
    reddit: 0.3            # Good coverage, some noise
    news: 0.2              # Credible but lagging
    google_trends: 0.1     # Supplementary
    heuristic_vs_hf: 0.5   # 50/50 if HF enabled

  # Quality thresholds
  thresholds:
    min_mentions_for_hf: 20        # Only use ML if sufficient data
    bot_pct_warning: 0.5           # Warn if >50% bot activity
    min_data_quality_sources: 1    # Require at least 1 source

  # Caching
  cache:
    enabled: true
    ttl_seconds: 1800      # 30 minutes
    redis_enabled: false   # Use in-memory cache

  # Monitoring
  monitoring:
    log_failures: true
    alert_on_all_providers_down: true
    performance_profiling: false
```

---

## 6. Configuration Changes

### 6.1 Environment Variables

Add to `.env` or environment configuration:

```bash
# Sentiment Module Configuration
# ---------------------------------

# Provider API Keys (optional - some work without keys)
STOCKTWITS_ACCESS_TOKEN=your_token_here  # Optional, higher rate limits
NEWSAPI_API_KEY=your_key_here                 # Optional for NewsAPI
ALPHAVANTAGE_KEY=your_key_here            # Optional for Alpha Vantage

# Provider Toggles
SENTIMENT_STOCKTWITS_ENABLED=true
SENTIMENT_REDDIT_ENABLED=true
SENTIMENT_NEWS_ENABLED=true
SENTIMENT_GOOGLE_TRENDS_ENABLED=false  # Conservative, low limits
SENTIMENT_TWITTER_ENABLED=false         # Requires Twitter API
SENTIMENT_DISCORD_ENABLED=false         # Requires Discord access

# ML Enhancement
SENTIMENT_HF_ENABLED=false                    # CPU intensive
SENTIMENT_HF_MODEL=cardiffnlp/twitter-roberta-base-sentiment
SENTIMENT_HF_DEVICE=cpu                       # or 'cuda' if GPU available

# Performance
SENTIMENT_CONCURRENCY=8
SENTIMENT_RATE_LIMIT_DELAY=0.3
SENTIMENT_CACHE_TTL=1800

# Redis (optional)
SENTIMENT_REDIS_ENABLED=false
SENTIMENT_REDIS_HOST=localhost
SENTIMENT_REDIS_PORT=6379
SENTIMENT_REDIS_DB=2
```

### 6.2 Config Loader Updates

**File**: `src/ml/pipeline/p04_short_squeeze/config/config_manager.py`

```python
from src.ml.pipeline.p04_short_squeeze.config.data_classes import SentimentConfig

class ConfigManager:
    """Enhanced config manager with sentiment config."""

    def load_sentiment_config(self) -> SentimentConfig:
        """
        Load sentiment module configuration.

        Returns:
            SentimentConfig instance
        """
        sentiment_section = self.config.get('sentiment_config', {})

        return SentimentConfig(
            providers=sentiment_section.get('providers', {}),
            batching=sentiment_section.get('batching', {}),
            weights=sentiment_section.get('weights', {}),
            thresholds=sentiment_section.get('thresholds', {}),
            cache=sentiment_section.get('cache', {}),
            monitoring=sentiment_section.get('monitoring', {})
        )
```

**File**: `src/ml/pipeline/p04_short_squeeze/config/data_classes.py`

```python
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class SentimentConfig:
    """Configuration for sentiment module integration."""

    providers: Dict[str, bool] = field(default_factory=dict)
    batching: Dict[str, Any] = field(default_factory=dict)
    weights: Dict[str, float] = field(default_factory=dict)
    thresholds: Dict[str, float] = field(default_factory=dict)
    cache: Dict[str, Any] = field(default_factory=dict)
    monitoring: Dict[str, bool] = field(default_factory=dict)

    def __post_init__(self):
        """Set defaults for missing values."""
        if not self.providers:
            self.providers = {
                'stocktwits': True,
                'reddit_pushshift': True,
                'news': True,
                'hf_enabled': False
            }

        if not self.batching:
            self.batching = {
                'concurrency': 8,
                'rate_limit_delay_sec': 0.3
            }

        if not self.weights:
            self.weights = {
                'stocktwits': 0.4,
                'reddit': 0.3,
                'news': 0.2,
                'google_trends': 0.1
            }
```

---

## 7. Testing Strategy

### 7.1 Unit Tests

**File**: `src/ml/pipeline/p04_short_squeeze/tests/test_sentiment_integration.py`

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.ml.pipeline.p04_short_squeeze.core.daily_deep_scan import DailyDeepScan
from src.ml.pipeline.p04_short_squeeze.core.models import CandidateModel
from src.common.sentiments.collect_sentiment_async import SentimentFeatures

@pytest.mark.asyncio
async def test_sentiment_batch_collection():
    """Test batch sentiment collection."""

    # Mock sentiment response
    mock_sentiment = {
        'AAPL': SentimentFeatures(
            ticker='AAPL',
            mentions_24h=150,
            unique_authors_24h=80,
            mentions_growth_7d=2.5,
            positive_ratio_24h=0.7,
            sentiment_score_24h=0.65,
            sentiment_normalized=0.65,
            virality_index=0.75,
            bot_pct=0.15,
            data_quality={'stocktwits': 'ok', 'reddit': 'ok'},
            raw_payload={}
        )
    }

    with patch('src.common.sentiments.collect_sentiment_async.collect_sentiment_batch',
               new_callable=AsyncMock, return_value=mock_sentiment):

        scanner = DailyDeepScan(config=MagicMock())
        candidates = [CandidateModel(ticker='AAPL')]

        # This would call the mocked sentiment function
        # Test implementation here

        assert candidates[0].sentiment_24h == 0.65
        assert candidates[0].mentions_24h == 150
        assert candidates[0].virality_index == 0.75

@pytest.mark.asyncio
async def test_sentiment_fallback_on_error():
    """Test graceful degradation when sentiment fails."""

    with patch('src.common.sentiments.collect_sentiment_async.collect_sentiment_batch',
               new_callable=AsyncMock, side_effect=Exception("API Error")):

        scanner = DailyDeepScan(config=MagicMock())
        candidates = [CandidateModel(ticker='AAPL')]

        # Should use default values without crashing
        assert candidates[0].sentiment_24h == 0.0
        assert candidates[0].virality_index == 0.0

def test_virality_score_calculation():
    """Test virality score with bot penalty."""
    from src.ml.pipeline.p04_short_squeeze.core.scoring_engine import ScoringEngine

    engine = ScoringEngine()

    # High virality, low bots
    score1 = engine.calculate_virality_score(virality_index=0.8, bot_pct=0.1)
    assert score1 > 0.75

    # High virality, high bots (should be penalized)
    score2 = engine.calculate_virality_score(virality_index=0.8, bot_pct=0.6)
    assert score2 < score1
    assert score2 < 0.7

def test_mention_growth_boost():
    """Test scoring boost for rapid mention growth."""
    from src.ml.pipeline.p04_short_squeeze.core.scoring_engine import ScoringEngine

    engine = ScoringEngine()

    # Candidate with high growth
    candidate = CandidateModel(
        ticker='GME',
        sentiment_24h=0.7,
        virality_index=0.6,
        mentions_growth_7d=5.0,  # 5x growth
        # ... other fields
    )

    score = engine.calculate_transient_score(candidate)

    # Should receive growth boost
    assert score > 0.6
```

### 7.2 Integration Tests

**File**: `src/ml/pipeline/p04_short_squeeze/tests/integration/test_sentiment_pipeline.py`

```python
import pytest
from src.ml.pipeline.p04_short_squeeze.scripts.run_daily_deep_scan import main

@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_pipeline_with_sentiment(test_config, test_db):
    """Test full pipeline execution with sentiment integration."""

    # Insert test candidates
    test_db.execute("""
        INSERT INTO ss_snapshot (ticker, screener_score, scan_date)
        VALUES ('AAPL', 0.6, CURRENT_DATE)
    """)

    # Run pipeline
    await main(config_path=test_config)

    # Verify results
    result = test_db.fetch_one("""
        SELECT * FROM ss_deep_metrics
        WHERE ticker = 'AAPL' AND scan_date = CURRENT_DATE
    """)

    assert result is not None
    assert result['sentiment_24h'] >= 0
    assert result['mentions_24h'] >= 0
    assert result['virality_index'] >= 0
    assert result['sentiment_data_quality'] is not None

@pytest.mark.integration
def test_sentiment_history_tracking(test_db):
    """Test that sentiment history is properly tracked."""

    # Run deep scan
    # ... (run pipeline)

    # Check history table
    history = test_db.fetch_all("""
        SELECT * FROM ss_sentiment_history
        WHERE date = CURRENT_DATE
    """)

    assert len(history) > 0
    assert all(h['mentions_count'] >= 0 for h in history)
```

### 7.3 Performance Tests

**File**: `src/ml/pipeline/p04_short_squeeze/tests/performance/test_sentiment_performance.py`

```python
import pytest
import time
from src.common.sentiments.collect_sentiment_async import collect_sentiment_batch

@pytest.mark.performance
@pytest.mark.asyncio
async def test_batch_sentiment_performance():
    """Test sentiment collection performance with realistic load."""

    # 100 tickers (realistic daily load)
    tickers = [f'TICK{i:03d}' for i in range(100)]

    start_time = time.time()

    results = await collect_sentiment_batch(
        tickers=tickers,
        lookback_hours=24,
        config={'batching': {'concurrency': 8}}
    )

    elapsed = time.time() - start_time

    # Should complete in reasonable time (< 2 minutes)
    assert elapsed < 120

    # Should have results for most tickers
    success_rate = sum(1 for v in results.values() if v) / len(tickers)
    assert success_rate > 0.7  # At least 70% success

    print(f"Processed {len(tickers)} tickers in {elapsed:.2f}s")
    print(f"Success rate: {success_rate*100:.1f}%")
```

### 7.4 Test Execution Plan

```bash
# 1. Run unit tests
pytest src/ml/pipeline/p04_short_squeeze/tests/test_sentiment_integration.py -v

# 2. Run integration tests (requires test database)
pytest src/ml/pipeline/p04_short_squeeze/tests/integration/ -v -m integration

# 3. Run performance tests
pytest src/ml/pipeline/p04_short_squeeze/tests/performance/ -v -m performance

# 4. Run full test suite
pytest src/ml/pipeline/p04_short_squeeze/tests/ -v --cov=src/ml/pipeline/p04_short_squeeze

# 5. Manual smoke test with small candidate set
python src/ml/pipeline/p04_short_squeeze/scripts/run_daily_deep_scan.py --dry-run --limit 5
```

---

## 8. Rollback Plan

### 8.1 Feature Flag Approach

Implement feature flag for gradual rollout:

```python
# In daily_deep_scan.py
USE_ENHANCED_SENTIMENT = os.getenv('FEATURE_ENHANCED_SENTIMENT', 'false').lower() == 'true'

if USE_ENHANCED_SENTIMENT:
    # Use new sentiment module
    sentiment_map = await collect_sentiment_batch(...)
else:
    # Use legacy Finnhub sentiment
    sentiment_score = await self.finnhub_downloader.aggregate_24h_sentiment(ticker)
```

### 8.2 Rollback Procedure

If issues arise:

1. **Immediate Rollback** (< 5 minutes):
   ```bash
   # Set feature flag to false
   export FEATURE_ENHANCED_SENTIMENT=false

   # Restart daily deep scan service
   systemctl restart p04_daily_deep_scan
   ```

2. **Database Rollback** (if schema changes cause issues):
   ```bash
   # Run rollback migration
   psql -d trading_db -f src/ml/pipeline/p04_short_squeeze/data/migrations/rollback_sentiment_metrics.sql
   ```

3. **Code Rollback** (if feature flag not working):
   ```bash
   # Revert to previous commit
   git revert <commit_hash>
   git push

   # Redeploy
   ./deploy_pipeline.sh
   ```

### 8.3 Monitoring for Issues

Watch for these indicators:

- **Performance degradation**: Daily scan takes >5 minutes (baseline: 2-3 min)
- **API errors**: >50% failure rate in sentiment collection
- **Data quality**: >30% of candidates have empty sentiment data
- **Scoring anomalies**: Final scores significantly different from historical patterns
- **Database errors**: Insert failures, constraint violations

---

## 9. Performance Considerations

### 9.1 Expected Performance

**Current (Finnhub only)**:
- API calls: 200 calls/day (1 per ticker)
- Time: ~2 seconds per ticker (sequential) = ~6-7 minutes for 200 tickers
- Rate limit: 60/min = plenty of headroom

**New (Multi-source sentiment)**:
- API calls: 200 tickers × 3 sources = 600 calls/day (but batched)
- Time: ~0.5 seconds per ticker (parallel, concurrency=8) = ~2 minutes for 200 tickers
- Rate limit: StockTwits 200/hr, Reddit unlimited = acceptable

**Expected Improvement**: 3-4 minutes faster (due to batching)

### 9.2 Optimization Strategies

1. **Batch Processing**:
   - Process 50 tickers at a time
   - Concurrent requests (8 parallel)
   - Reduces total API calls via batch endpoints

2. **Caching**:
   - 30-minute TTL for sentiment summaries
   - Reduces duplicate calls for repeated scans
   - Cache hit rate expected: ~20-30%

3. **Progressive Enhancement**:
   - Basic sentiment from all sources (fast)
   - ML analysis only if mentions > 20 (expensive)
   - Skip Google Trends by default (slow, low value)

4. **Connection Pooling**:
   - Reuse HTTP connections across requests
   - Async I/O for non-blocking requests
   - Timeout handling to prevent hanging

### 9.3 Resource Requirements

**CPU**:
- Current: ~5-10% during scan
- New (without HF): ~10-15% (more async processing)
- New (with HF): ~40-60% (ML inference)
- **Recommendation**: Disable HF by default, enable only if needed

**Memory**:
- Current: ~200 MB
- New: ~300-400 MB (additional data structures, caching)
- **Recommendation**: Acceptable for modern servers

**Network**:
- Current: ~1-2 MB total
- New: ~5-10 MB total (more API responses)
- **Recommendation**: Negligible impact

### 9.4 Rate Limit Management

**Provider Limits**:
| Provider | Free Tier | With Key | Current Use | Safety Margin |
|----------|-----------|----------|-------------|---------------|
| StockTwits | 200/hr | 400/hr | ~50/hr | 4x buffer |
| Reddit | Unlimited* | Unlimited* | ~50/hr | No limit |
| NewsAPI | 100/day | 1000/day | ~50/day | 2x buffer |
| Google Trends | ~30/hr | N/A | 0 (disabled) | N/A |

*Unofficial, respect 0.5s delay

**Mitigation**:
- Adaptive rate limiting: Slow down if errors detected
- Circuit breaker: Disable provider after 5 failures
- Retry logic: Exponential backoff
- Fallback: Continue with remaining providers if one fails

---

## 10. Future Enhancements

### 10.1 Phase 2 Enhancements (Post-Launch)

1. **Volume Detector Integration**:
   - Add sentiment check to volume spike detection
   - Quick 3-hour lookback for immediate signals
   - Filter: Volume spike + positive sentiment = stronger alert

2. **Sentiment-Driven Alerts**:
   - New alert type: "Viral Mention Surge"
   - Trigger: mentions_growth_7d > 5x AND virality > 0.7
   - Independent of screener/volume signals

3. **Historical Trend Analysis**:
   - Sentiment momentum: 7-day rolling average
   - Sentiment reversal detection: Positive → negative shift
   - Correlation with price movements

### 10.2 Phase 3 Enhancements (Future)

1. **Custom ML Model**:
   - Train on labeled short squeeze outcomes
   - Features: sentiment + volume + FINRA data
   - Replace HuggingFace with domain-specific model

2. **Real-time Sentiment Streaming**:
   - WebSocket connections to StockTwits/Twitter
   - Intraday sentiment shifts
   - Trigger immediate alerts on viral events

3. **Sentiment Quality Scoring**:
   - Source credibility weighting (WSJ > unknown blogs)
   - Author influence scoring (verified accounts)
   - Content quality analysis (depth, originality)

4. **Cross-asset Sentiment**:
   - Sector sentiment analysis
   - Market-wide sentiment index
   - Relative sentiment (ticker vs sector average)

### 10.3 Monitoring & Analytics

1. **Sentiment Dashboard**:
   - Top movers by mention growth
   - Virality index leaderboard
   - Bot activity heatmap
   - Data quality metrics

2. **Backtesting Framework**:
   - Historical sentiment vs squeeze outcomes
   - Optimize scoring weights based on data
   - A/B testing for different sentiment sources

3. **Alert Effectiveness Tracking**:
   - Track alert → squeeze outcome correlation
   - Measure precision/recall of sentiment signals
   - Continuous improvement of thresholds

---

## Appendix A: Migration Checklist

- [ ] Review and approve implementation plan
- [ ] Create feature branch: `feature/sentiment-integration`
- [ ] Run database migration script on dev database
- [ ] Verify migration with test data
- [ ] Implement code changes in daily_deep_scan.py
- [ ] Update models.py with new fields
- [ ] Update scoring_engine.py with new weights
- [ ] Add sentiment configuration to YAML
- [ ] Update environment variables documentation
- [ ] Write unit tests (target: >80% coverage)
- [ ] Write integration tests
- [ ] Run performance tests
- [ ] Test rollback procedure
- [ ] Update pipeline documentation
- [ ] Code review and approval
- [ ] Deploy to staging environment
- [ ] Run smoke tests on staging
- [ ] Deploy to production with feature flag OFF
- [ ] Enable feature flag for 10% of candidates (canary)
- [ ] Monitor for 24 hours
- [ ] Enable feature flag for 100% if no issues
- [ ] Remove feature flag after 1 week of stable operation
- [ ] Archive legacy Finnhub sentiment code

---

## Appendix B: Deployment Schedule

**Week 1: Development**
- Mon-Tue: Database migration + code implementation
- Wed-Thu: Testing (unit, integration, performance)
- Fri: Code review and staging deployment

**Week 2: Staging & Canary**
- Mon: Staging smoke tests
- Tue: Production deployment (feature flag OFF)
- Wed: Enable 10% canary rollout
- Thu-Fri: Monitor canary, adjust if needed

**Week 3: Full Rollout**
- Mon: Enable 50% rollout if canary successful
- Tue-Wed: Monitor and validate
- Thu: Enable 100% rollout
- Fri: Remove feature flag code

**Week 4: Cleanup**
- Mon-Tue: Remove legacy code
- Wed: Documentation finalization
- Thu: Team training on new metrics
- Fri: Retrospective and lessons learned

---

## Appendix C: Contact & Escalation

**Technical Owner**: Data Engineering Team
**Product Owner**: Trading Strategy Team
**On-call**: #trading-alerts Slack channel

**Escalation Path**:
1. Alert detected → #trading-alerts
2. No response in 15 min → Page on-call engineer
3. Critical issue → Escalate to Engineering Manager

**Documentation**:
- Architecture: `src/ml/pipeline/p04_short_squeeze/docs/Design.md`
- Sentiment Module: `src/common/sentiments/docs/README.md`
- Runbook: `src/ml/pipeline/p04_short_squeeze/docs/OPERATIONS.md` (to be created)

---

**END OF IMPLEMENTATION PLAN**
