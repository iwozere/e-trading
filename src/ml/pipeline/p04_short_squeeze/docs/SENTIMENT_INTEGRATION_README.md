# Sentiment Integration for Short Squeeze Detection Pipeline

## Overview

The p04_short_squeeze pipeline now supports **multi-source sentiment analysis** to enhance short squeeze detection. This integration replaces the single-source Finnhub sentiment with a comprehensive system that aggregates data from 7 different sources.

## Key Features

### 🎯 Multi-Source Sentiment
- **StockTwits**: Trading-focused social sentiment
- **Reddit**: Community discussion analysis via Pushshift API
- **News APIs**: Credible news sources (Finnhub, NewsAPI, Alpha Vantage)
- **Google Trends**: Search volume and trending analysis
- **Twitter**: Social media sentiment (optional)
- **Discord**: Community chat monitoring (optional)
- **HuggingFace ML**: Machine learning-based sentiment analysis (optional)

### 📊 Enhanced Metrics
- **Virality Index** (0-1): Engagement-weighted virality score
- **Mentions Growth**: 7-day growth trend analysis
- **Bot Detection**: Automated bot activity filtering
- **Multi-source Aggregation**: Quality-weighted sentiment combination
- **Data Quality Tracking**: Per-provider status monitoring

### ⚡ Performance Optimizations
- **Batch Processing**: Collect sentiment for multiple tickers simultaneously
- **Caching**: 30-minute TTL reduces duplicate API calls
- **Concurrent Requests**: Process 8 requests in parallel
- **Graceful Degradation**: Continues with available sources if some fail

## Installation

### 1. Database Migration

Run the database migration to add new sentiment fields:

```bash
psql -U your_user -d trading_db -f src/ml/pipeline/p04_short_squeeze/data/migrations/001_add_sentiment_metrics.sql
```

This creates:
- 5 new columns in `ss_deep_metrics` table
- New `ss_sentiment_history` table for trend tracking
- Indexes for performance
- Validation constraints

### 2. Environment Variables

Copy the environment template and configure:

```bash
cp src/ml/pipeline/p04_short_squeeze/config/.env.sentiment.example .env.sentiment
# Edit .env.sentiment with your API keys and preferences
```

Required environment variables:
```bash
# Feature flag (enable/disable)
FEATURE_ENHANCED_SENTIMENT=true

# Provider toggles
SENTIMENT_STOCKTWITS_ENABLED=true
SENTIMENT_REDDIT_ENABLED=true
SENTIMENT_NEWS_ENABLED=true

# Performance settings
SENTIMENT_CONCURRENCY=8
SENTIMENT_CACHE_TTL=1800
```

### 3. Configuration File

Update your pipeline configuration:

```yaml
# config/pipeline_config.yaml

sentiment:
  providers:
    stocktwits: true
    reddit_pushshift: true
    news: true
    hf_enabled: false  # Disable ML by default (CPU intensive)

  batching:
    concurrency: 8
    rate_limit_delay_sec: 0.3

  weights:
    stocktwits: 0.4
    reddit: 0.3
    news: 0.2
    google_trends: 0.1
```

See [pipeline_config_example.yaml](../config/pipeline_config_example.yaml) for full configuration options.

## Usage

### Basic Usage

The sentiment integration is **automatic** once configured. The daily deep scan will use multi-source sentiment by default:

```python
from src.ml.pipeline.p04_short_squeeze.core.daily_deep_scan import DailyDeepScan
from src.ml.pipeline.p04_short_squeeze.config.data_classes import (
    DeepScanConfig, SentimentConfig
)

# Initialize with sentiment config
scanner = DailyDeepScan(
    fmp_downloader=fmp,
    finnhub_downloader=finnhub,
    config=DeepScanConfig(),
    sentiment_config=SentimentConfig()  # Auto-loads from config
)

# Run deep scan (sentiment collected automatically)
results = scanner.run_deep_scan()
```

### Feature Flag

To temporarily disable enhanced sentiment and use legacy Finnhub:

```bash
export FEATURE_ENHANCED_SENTIMENT=false
```

Or in code:

```python
scanner = DailyDeepScan(
    fmp_downloader=fmp,
    finnhub_downloader=finnhub,
    config=DeepScanConfig(),
    sentiment_config=None  # Will use legacy Finnhub
)
```

### Accessing Enhanced Metrics

Enhanced sentiment metrics are available in `TransientMetrics`:

```python
for scored_candidate in results.scored_candidates:
    transient = scored_candidate.transient_metrics

    print(f"Ticker: {scored_candidate.candidate.ticker}")
    print(f"  Sentiment: {transient.sentiment_24h:.2f}")
    print(f"  Mentions (24h): {transient.mentions_24h}")
    print(f"  Mention Growth (7d): {transient.mentions_growth_7d:.1f}x")
    print(f"  Virality Index: {transient.virality_index:.2f}")
    print(f"  Bot Activity: {transient.bot_pct:.1%}")
    print(f"  Data Quality: {transient.sentiment_data_quality}")
```

## Architecture

### Data Flow

```
┌─────────────────────────────────────────────┐
│     Daily Deep Scan (10:00 AM)              │
└─────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────┐
│  Load Candidates (50-200 tickers)           │
└─────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────┐
│  Batch Sentiment Collection                 │
│  - Concurrent requests (8 parallel)         │
│  - Multi-source aggregation                 │
│  - Bot detection & filtering                │
│  - Quality weighting                        │
└─────────────────────────────────────────────┘
         ↓              ↓              ↓
   ┌──────────┐   ┌──────────┐   ┌──────────┐
   │StockTwits│   │  Reddit  │   │   News   │
   │  (0.4)   │   │  (0.3)   │   │  (0.2)   │
   └──────────┘   └──────────┘   └──────────┘
                     ↓
┌─────────────────────────────────────────────┐
│  SentimentFeatures Output                   │
│  - sentiment_normalized (0-1)               │
│  - mentions_24h                             │
│  - mentions_growth_7d                       │
│  - virality_index                           │
│  - bot_pct                                  │
│  - data_quality                             │
└─────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────┐
│  Store in ss_deep_metrics                   │
│  + Enhanced Scoring                         │
└─────────────────────────────────────────────┘
```

### Scoring Integration

Enhanced sentiment is integrated into the scoring engine with increased weights:

```python
# New scoring weights
transient_score = (
    volume_spike * 0.30 +      # 16.5% of total (increased)
    sentiment * 0.35 +         # 19.25% of total (increased from 12.5%)
    virality * 0.15 +          # 8.25% of total (NEW)
    call_put * 0.10 +          # 5.5% of total
    borrow_fee * 0.10          # 5.5% of total
)

final_score = (
    transient_score * 0.55 +   # Increased from 50%
    finra_score * 0.30 +       # Unchanged
    screener_score * 0.15      # Decreased from 20%
)
```

**Mention Growth Boost**: Rapid mention growth provides an additional boost:
- 2x growth: +5% boost
- 5x growth: +10% boost
- 10x+ growth: +15% boost

**Bot Penalty**: High bot activity reduces virality score by up to 60%.

## Database Schema

### New Columns in `ss_deep_metrics`

| Column | Type | Description |
|--------|------|-------------|
| `mentions_24h` | INTEGER | Total mentions across all sources |
| `mentions_growth_7d` | FLOAT | Growth vs 7-day average (NULL if no data) |
| `virality_index` | FLOAT | Engagement-weighted virality (0-1) |
| `bot_pct` | FLOAT | Estimated bot percentage (0-1) |
| `sentiment_data_quality` | JSONB | Per-provider status |

### New Table: `ss_sentiment_history`

Tracks daily sentiment for growth calculations:

| Column | Type | Description |
|--------|------|-------------|
| `ticker` | VARCHAR(20) | Stock ticker |
| `date` | DATE | Date of record |
| `mentions_count` | INTEGER | Daily total mentions |
| `unique_authors` | INTEGER | Unique contributors |
| `sentiment_avg` | FLOAT | Average sentiment |
| `virality_avg` | FLOAT | Average virality |

Retention: 30 days (automatic cleanup)

## Performance

### Expected Performance

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Daily scan time | 6-7 min | 2-3 min | **3-4 min faster** |
| Sentiment API calls | 200/day | ~600/day | 3x increase (but free) |
| FMP API usage | 200/day | 50/day | **75% reduction** |
| Finnhub API usage | 400/day | 200/day | **50% reduction** |

### API Rate Limits

| Provider | Free Tier | With Key | Current Use | Buffer |
|----------|-----------|----------|-------------|--------|
| StockTwits | 200/hr | 400/hr | ~50/hr | 4x |
| Reddit | Unlimited* | Unlimited* | ~50/hr | ∞ |
| NewsAPI | 100/day | 1000/day | ~50/day | 2x |

*Unofficial limits, respect 0.5s delay

### Optimization Tips

1. **Disable HuggingFace** by default (CPU intensive):
   ```yaml
   sentiment:
     providers:
       hf_enabled: false
   ```

2. **Enable Redis caching** for better performance:
   ```bash
   SENTIMENT_REDIS_ENABLED=true
   SENTIMENT_REDIS_HOST=localhost
   ```

3. **Adjust concurrency** based on resources:
   ```yaml
   sentiment:
     batching:
       concurrency: 16  # Increase if you have resources
   ```

## Monitoring

### Logs

Enhanced sentiment logs notable events:

```
INFO: Batch sentiment collection complete: 47/50 successful
INFO:   GME: HIGH VIRALITY detected (0.85)
INFO:   AMC: MENTION SURGE detected (+450%)
WARN:   XYZ: High bot activity detected (62%)
```

### Data Quality

Check data quality in results:

```python
for candidate in results.scored_candidates:
    quality = candidate.transient_metrics.sentiment_data_quality

    if 'error' in quality:
        print(f"Sentiment collection failed for {candidate.ticker}")
    else:
        providers = [k for k, v in quality.items() if v == 'ok']
        print(f"Successful providers for {candidate.ticker}: {providers}")
```

### Metrics

Track sentiment metrics in database:

```sql
-- Average sentiment data quality
SELECT
    AVG(CASE WHEN mentions_24h > 0 THEN 1.0 ELSE 0.0 END) as sentiment_success_rate,
    AVG(mentions_24h) as avg_mentions,
    AVG(virality_index) as avg_virality,
    AVG(bot_pct) as avg_bot_activity
FROM ss_deep_metrics
WHERE scan_date >= CURRENT_DATE - INTERVAL '7 days';

-- Top viral candidates
SELECT ticker, virality_index, mentions_24h, mentions_growth_7d, bot_pct
FROM ss_deep_metrics
WHERE scan_date = CURRENT_DATE
AND virality_index > 0.7
ORDER BY virality_index DESC
LIMIT 10;
```

## Troubleshooting

### Issue: No Sentiment Data Collected

**Symptom**: All candidates have `mentions_24h = 0`

**Possible Causes**:
1. Feature flag disabled: Check `FEATURE_ENHANCED_SENTIMENT=true`
2. Sentiment config missing: Pass `SentimentConfig()` to `DailyDeepScan`
3. All providers disabled: Enable at least one provider
4. API rate limit: Check logs for rate limit errors

**Solution**:
```python
# Verify configuration
scanner = DailyDeepScan(
    fmp_downloader=fmp,
    finnhub_downloader=finnhub,
    config=DeepScanConfig(),
    sentiment_config=SentimentConfig()  # Must not be None
)

# Check if enabled
print(f"Enhanced sentiment enabled: {scanner.use_enhanced_sentiment}")
```

### Issue: High Bot Percentage

**Symptom**: Many candidates have `bot_pct > 0.5`

**Explanation**: This is expected for certain tickers (meme stocks, penny stocks). Bot detection helps filter noise.

**Action**: Use bot percentage in scoring logic (already implemented - virality penalty).

### Issue: Slow Performance

**Symptom**: Daily scan takes > 5 minutes

**Solutions**:
1. Reduce concurrency if CPU/memory limited:
   ```yaml
   sentiment:
     batching:
       concurrency: 4  # Reduce from 8
   ```

2. Disable optional providers:
   ```yaml
   sentiment:
     providers:
       google_trends: false  # Slow provider
       hf_enabled: false      # CPU intensive
   ```

3. Enable Redis caching:
   ```bash
   SENTIMENT_REDIS_ENABLED=true
   ```

### Issue: API Rate Limit Errors

**Symptom**: Logs show "429 Too Many Requests"

**Solutions**:
1. Increase rate limit delay:
   ```yaml
   sentiment:
     batching:
       rate_limit_delay_sec: 0.5  # Increase from 0.3
   ```

2. Reduce concurrency:
   ```yaml
   sentiment:
     batching:
       concurrency: 4  # Reduce from 8
   ```

3. Add API keys for higher limits:
   ```bash
   STOCKTWITS_ACCESS_TOKEN=your_token
   NEWS API_KEY=your_key
   ```

## Rollback

If you need to rollback the sentiment integration:

### 1. Disable Feature Flag

```bash
export FEATURE_ENHANCED_SENTIMENT=false
```

Or set in code:
```python
scanner = DailyDeepScan(..., sentiment_config=None)
```

### 2. Database Rollback (Optional)

```bash
psql -U your_user -d trading_db -f src/ml/pipeline/p04_short_squeeze/data/migrations/001_rollback_sentiment_metrics.sql
```

**Warning**: This removes all sentiment history data.

## Testing

Run the test suite:

```bash
# Run all tests
pytest src/ml/pipeline/p04_short_squeeze/tests/test_sentiment_integration.py -v

# Run specific test
pytest src/ml/pipeline/p04_short_squeeze/tests/test_sentiment_integration.py::TestScoringEngineEnhanced::test_calculate_virality_score_with_bots -v

# Run with coverage
pytest src/ml/pipeline/p04_short_squeeze/tests/test_sentiment_integration.py --cov=src/ml/pipeline/p04_short_squeeze/core --cov-report=html
```

## References

- **Implementation Plan**: [SENTIMENT_INTEGRATION_PLAN.md](SENTIMENT_INTEGRATION_PLAN.md)
- **Sentiment Module Docs**: [src/common/sentiments/docs/README.md](../../../common/sentiments/docs/README.md)
- **Example Config**: [pipeline_config_example.yaml](../config/pipeline_config_example.yaml)
- **Environment Template**: [.env.sentiment.example](../config/.env.sentiment.example)

## Support

For issues or questions:
- Check logs: `logs/p04_short_squeeze/daily_deep_scan.log`
- Review configuration: Ensure all required fields are set
- Test connectivity: Verify API keys and network access
- Consult docs: See references above

---

**Version**: 1.0
**Last Updated**: 2025-11-17
**Status**: ✅ Production Ready
