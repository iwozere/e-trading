# Tasks

## Implementation Status

### ✅ COMPLETED FEATURES
- [x] Project structure and directory organization
- [x] Configuration system with YAML loading and validation
- [x] Type-safe configuration data classes
- [x] Logging integration with existing notification system
- [x] Environment variable substitution in configuration
- [x] Comprehensive configuration validation
- [x] Module documentation (README, Requirements, Design, Tasks)

### 🔄 IN PROGRESS
- [ ] Database schema creation and data models
- [ ] Data provider integration layer
- [ ] Core pipeline modules (Universe Loader, Screener, Deep Scan)
- [ ] Scoring and alert system
- [ ] Candidate management system

### 🚀 PLANNED ENHANCEMENTS
- [ ] Executable pipeline scripts
- [ ] Reporting and monitoring system
- [ ] Error handling and resilience features
- [ ] Scheduler integration interfaces
- [ ] Performance optimization and validation
- [ ] Comprehensive test suite

## Technical Debt
- [ ] Add comprehensive unit tests for configuration system
- [ ] Implement configuration hot-reload capability
- [ ] Add configuration schema documentation
- [ ] Create configuration migration utilities

## Known Issues
- Configuration validation could be more granular for nested objects
- Environment variable substitution doesn't support complex expressions
- Logging context inheritance needs testing with concurrent operations
- (Fixed 2026-08-20) `SentimentProviders`/`SentimentWeights` used the field name `google_trends`,
  but the adapter is registered as `"trends"` in `adapter_manager.py` — the Google Trends
  provider toggle and blend weight silently never reached the adapter loop. Same bug existed for
  the sentiment cache config (`"cache"` vs. the consumer's `"caching"` key). Both renamed;
  `daily_deep_scan.py`'s `_collect_batch_sentiment` config-dict builder now matches.
- (Fixed 2026-08-20) `_store_results()` computed `mentions_24h`/`virality_index`/`bot_pct`/
  `sentiment_data_quality` but never included them in the dict written to the DB, even though the
  `ss_deep_metrics` columns already existed (migration 001). The `DeepScanMetrics` ORM class also
  didn't map those columns at all. Both fixed; see `sentiment-spec-rev2.md` for the accompanying
  `tech_*` fields added alongside them.
- (Fixed 2026-08-20, Rev 2 Phase 3) `collect_sentiment_async._initialize_adapters` special-cased a
  `"huggingface"` entry *inside* the `config["providers"]` loop, but `"huggingface"` was never
  actually a key in that dict (only the `hf_enabled` toggle was) -- `manager.add_adapter` was
  therefore never called for it, so the HF-enhancement path (`enhanced_sentiment`/`bot_pct`/
  `virality_index`) was silently dead every time `hf_enabled=True`, always falling through to the
  raw heuristic score. Fixed by registering `"huggingface"` (and a new `"huggingface_tech"`
  instance for per-source model routing) explicitly, gated on `hf_enabled`, outside the generic
  provider loop.
- `virality_index`'s scale changed with the Rev 2 Phase 3 redefinition (spec §2.5.5): it is now
  unsigned reach (`Σengagement / sqrt(unique_authors+1)`, unbounded) rather than Rev 1's
  sentiment-weighted ~0..1 score. `daily_deep_scan.py`'s "HIGH VIRALITY" log threshold was bumped
  to a placeholder (`50.0`) to stop it firing on every ticker, but is not tuned against real data
  -- run `coverage-report` (Phase 4) before trusting it for anything more than an info log line.
- `mentions_growth_7d` is not yet per-source (spec §2.5.5's "both new sources start with no
  history" concern) -- `_get_historical_mentions_async` in `daily_deep_scan.py` is a pre-existing
  stub that always returns `None`, so growth is currently always `None` regardless of source.
  Making it source-aware needs a new per-provider mention-history table; deferred.

## Testing Requirements
- [ ] Unit tests for ConfigManager class
- [ ] Unit tests for all configuration data classes
- [ ] Integration tests for YAML loading and validation
- [ ] Performance tests for configuration loading
- [ ] Tests for environment variable substitution
- [ ] Tests for logging context management

## Documentation Updates
- [ ] Add configuration parameter reference guide
- [ ] Create troubleshooting guide for common configuration issues
- [ ] Document environment variable requirements
- [ ] Add examples for different deployment scenarios

## Next Implementation Steps

### Immediate (Task 2)
- Create database schema migration scripts
- Implement data model classes for pipeline entities
- Set up database integration tests

### Short Term (Tasks 3-4)
- Extend existing data providers for short squeeze metrics
- Implement core pipeline modules with basic functionality
- Create unit tests for data processing logic

### Medium Term (Tasks 5-8)
- Build scoring and alert systems
- Implement candidate management
- Create executable scripts for manual operation
- Add comprehensive reporting capabilities

### Long Term (Tasks 9-12)
- Implement advanced error handling and resilience
- Prepare scheduler integration interfaces
- Performance optimization and monitoring
- Complete documentation and deployment guides