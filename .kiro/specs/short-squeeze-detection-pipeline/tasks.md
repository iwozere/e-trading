# Short Squeeze Detection Pipeline Implementation Plan

- [x] 1. Set up project structure and configuration system
  - Create directory structure for pipeline modules, configuration, and tests
  - Implement YAML configuration loader with validation
  - Create configuration data classes for type safety
  - Set up logging integration with existing notification system
  - _Requirements: 5.1, 5.2, 6.4, 6.5_

- [x] 2. Create database schema and data models using centralized infrastructure
  - [x] 2.1 Implement database schema creation scripts
    - Write SQL migration scripts for the five tables (ss_snapshot, ss_deep_metrics, ss_alerts, ss_ad_hoc_candidates, ss_finra_short_interest)
    - Create database indexes for performance optimization including FINRA data lookups
    - Integrate with centralized migration system in `src/data/db/migrations/`
    - _Requirements: 8.2_

  - [x] 2.2 Create data model classes
    - Implement SQLAlchemy models in `src/data/db/models/model_short_squeeze.py` including FINRA table
    - Create repository layer in `src/data/db/repos/repo_short_squeeze.py`
    - Implement service layer in `src/data/db/services/short_squeeze_service.py` (consolidated with FINRA functionality)
    - Create business logic dataclasses in `src/ml/pipeline/p04_short_squeeze/core/models.py` with volume and FINRA metrics
    - Pipeline modules use centralized services directly (no facade layer needed)
    - _Requirements: 8.2_

  - [x] 2.3 Write database integration tests
    - Create test database setup and teardown utilities using centralized test patterns
    - Write tests for CRUD operations on all tables including FINRA data operations
    - Test direct usage of centralized services from pipeline modules
    - Test FINRA data storage and retrieval with date-based queries
    - _Requirements: 8.2_

- [x] 3. Implement data provider integration layer
  - [x] 3.1 Extend existing FMP data downloader
    - Add methods for fetching universe data, market cap, and volume information
    - Implement universe loading functionality from FMP stock screener with market cap filtering
    - Add volume and price data fetching for volume analysis
    - Add error handling and rate limiting specific to short squeeze data needs
    - _Requirements: 1.1, 1.3, 1.4, 8.1_

  - [x] 3.2 Create FINRA data downloader
    - Implement FINRADataDownloader class for official short interest data collection
    - Add FINRA file download and parsing functionality
    - Implement data validation and error handling for FINRA format
    - Create integration with FINRA service for data storage
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

  - [x] 3.3 Extend existing Finnhub data downloader
    - Add methods for fetching sentiment data, options data, and borrow rates
    - Implement call/put ratio calculations from options data
    - Add 24-hour sentiment aggregation functionality
    - _Requirements: 4.2, 4.4, 8.1_

  - [x] 3.4 Create data provider integration tests
    - Write mock API response tests for FMP and Finnhub extensions
    - Test FINRA data download and parsing with sample files
    - Test rate limiting and error handling scenarios
    - _Requirements: 8.1_

- [x] 4. Build core pipeline modules
  - [x] 4.1 Implement Universe Loader
    - Create UniverseLoader class with FMP integration
    - Implement market cap, volume, and exchange filtering
    - Add universe caching for performance optimization
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

  - [x] 4.2 Implement Volume-Based Squeeze Detector
    - Create VolumeSqueezeDetector class with volume pattern analysis
    - Implement volume spike ratio calculation (current vs 20-day average)
    - Add RSI and momentum indicator calculations
    - Implement volume-based candidate scoring and filtering
    - Store volume detector results in ss_snapshot table
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

  - [x] 4.3 Implement Daily Deep Scan Module with Hybrid Analysis
    - Create DailyDeepScan class combining volume analysis with FINRA data
    - Implement hybrid metrics calculation using both volume patterns and FINRA short interest
    - Add sentiment score aggregation from 24-hour data
    - Implement call-to-put ratio calculation from options data
    - Handle cases where FINRA data is unavailable or outdated
    - Store daily results in ss_deep_metrics table
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

  - [x] 4.4 Write core module unit tests
    - Test volume detector scoring algorithms with known inputs
    - Test hybrid deep scan metric calculations with and without FINRA data
    - Test data storage and retrieval operations for all modules
    - Test FINRA data integration and fallback scenarios
    - _Requirements: 1.1-1.5, 2.1-2.5, 3.1-3.5, 4.1-4.5_

- [x] 5. Implement scoring and alert system
  - [x] 5.1 Create Hybrid Scoring Engine
    - Implement ScoringEngine class with hybrid metric normalization
    - Create weighted scoring algorithm combining volume metrics, FINRA data, and transient metrics
    - Add configurable weight system for hybrid scoring (with/without FINRA data scenarios)
    - Implement score validation and bounds checking for hybrid approach
    - _Requirements: 4.1-4.5_

  - [x] 5.2 Implement Alert Engine with Hybrid Criteria
    - Create AlertEngine class with hybrid threshold evaluation
    - Implement three-tier alert system (high, medium, low) using volume and FINRA criteria
    - Add cooldown period enforcement with database tracking
    - Integrate with existing notification system for Telegram and email alerts
    - Include both volume spike and FINRA short interest data in alert messages
    - Store alert events in ss_alerts table
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 8.3_

  - [x] 5.3 Write scoring and alert tests
    - Test hybrid scoring algorithm with various metric combinations
    - Test alert threshold evaluation with volume and FINRA criteria
    - Test cooldown logic and notification system integration
    - Test scenarios with missing or outdated FINRA data
    - _Requirements: 5.1-5.5_

- [x] 6. Build candidate management system
  - [x] 6.1 Implement Candidate Store
    - Create CandidateStore class with PostgreSQL integration
    - Implement screener snapshot storage and retrieval
    - Add deep scan results storage with date-based updates
    - Create candidate lifecycle management (creation, updates, expiration)
    - _Requirements: 1.4, 2.1, 6.2_

  - [x] 6.2 Implement Ad-hoc Candidate Manager
    - Create AdHocManager class for manual candidate additions
    - Implement TTL-based expiration system (default 7 days)
    - Add candidate activation/deactivation functionality
    - Integrate with deep scan processing pipeline
    - Store ad-hoc candidates in ss_ad_hoc_candidates table
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

  - [x] 6.3 Write candidate management tests
    - Test candidate storage and retrieval operations
    - Test ad-hoc candidate lifecycle management
    - Test TTL expiration functionality
    - _Requirements: 4.1-4.5_

- [x] 7. Create executable pipeline scripts
  - [x] 7.1 Implement weekly universe loader script
    - Create standalone executable script for weekly universe loading
    - Add command-line argument parsing for configuration options
    - Implement comprehensive error handling and logging
    - Add performance metrics collection and reporting
    - _Requirements: 10.1, 10.2, 7.3, 7.4_

  - [x] 7.2 Implement bi-weekly FINRA collector script



    - Create standalone executable script for FINRA data collection
    - Add command-line argument parsing and configuration loading
    - Implement FINRA file download, parsing, and storage
    - Add error handling for FINRA data format changes



    - _Requirements: 10.1, 10.2, 7.3, 7.4_


  - [x] 7.3 Implement daily volume detector script
    - Create standalone executable script for volume-based squeeze detection
    - Add command-line argument parsing and configuration loading
    - Implement batch processing for universe analysis
    - Add progress tracking and performance monitoring
    - _Requirements: 10.1, 10.2, 7.3, 7.4_

  - [x] 7.4 Implement daily deep scan script
    - Create standalone executable script for hybrid deep scan runs
    - Add command-line argument parsing and configuration loading
    - Implement batch processing for candidate analysis with FINRA integration
    - Add progress tracking and performance monitoring
    - _Requirements: 10.1, 10.2, 7.3, 7.4_

  - [x] 7.5 Create ad-hoc candidate management script
    - Implement command-line utility for adding/removing ad-hoc candidates
    - Add candidate status checking and expiration management
    - Create bulk import functionality for multiple candidates
    - _Requirements: 6.1, 6.2, 6.5, 10.2_

  - [x] 7.6 Write script integration tests
    - Test end-to-end pipeline execution with sample data including FINRA integration
    - Test command-line argument parsing and error handling for all scripts
    - Test script performance and resource usage
    - Test hybrid pipeline workflow (universe → volume detection → FINRA integration → deep scan)
    - _Requirements: 10.1, 10.2_

- [ ] 8. Implement reporting and monitoring
  - [ ] 8.1 Create Reporting Engine
    - Implement ReportingEngine class with weekly and daily report generation
    - Add HTML report generation with candidate rankings and metrics
    - Implement CSV export functionality for data analysis
    - Create trend analysis and performance visualization
    - _Requirements: 7.3, 7.4, 7.5_

  - [ ] 8.2 Add performance monitoring
    - Implement runtime metrics collection (duration, API calls, errors)
    - Add data quality tracking (valid payloads, non-null fields)
    - Create performance dashboard data export
    - Integrate with existing logging system for metrics storage
    - _Requirements: 5.2, 5.3, 7.1, 7.2_

  - [ ] 8.3 Write reporting tests
    - Test report generation with sample data
    - Test HTML and CSV export functionality
    - Test performance metrics collection
    - _Requirements: 7.1-7.5_

- [ ] 9. Add error handling and resilience
  - [ ] 9.1 Implement API error handling
    - Create retry logic with exponential backoff for API failures
    - Add rate limiting compliance for FMP (300/min) and Finnhub (60/min)
    - Implement circuit breaker pattern for repeated API failures
    - Add graceful degradation for partial data availability
    - _Requirements: 5.4, 5.5_

  - [ ] 9.2 Add data quality validation
    - Implement JSON payload validation with 99% success target
    - Add key field presence validation with 95% non-null target
    - Create data quality scoring and reporting
    - Add data quality alerts for significant degradation
    - _Requirements: 7.1, 7.2_

  - [ ] 9.3 Write error handling tests
    - Test API failure scenarios and retry logic
    - Test data quality validation with corrupted data
    - Test graceful degradation scenarios
    - _Requirements: 5.4, 5.5, 7.1, 7.2_

- [ ] 10. Prepare for scheduler integration
  - [ ] 10.1 Create scheduler interface adapters
    - Implement SchedulerInterface abstract class for future integration
    - Create PipelineSchedulerAdapter for multi-tier job registration and management
    - Add timezone handling for European trading hours (Europe/Zurich) and FINRA reporting schedule
    - Implement job failure handling and notification for all pipeline components
    - _Requirements: 10.3, 10.4, 10.5_

  - [ ] 10.2 Add configuration for automated scheduling
    - Extend configuration system with multi-tier scheduling parameters (weekly, bi-weekly, daily)
    - Add cron expression support for flexible scheduling of all pipeline components
    - Implement schedule validation and conflict detection
    - Create scheduling documentation and examples for hybrid pipeline
    - _Requirements: 10.4, 10.5_

  - [ ] 10.3 Write scheduler integration tests
    - Test scheduler interface implementation for all pipeline components
    - Test job registration and management functionality
    - Test timezone handling and schedule validation
    - Test multi-tier scheduling coordination (universe → FINRA → volume → deep scan)
    - _Requirements: 10.3, 10.4, 10.5_

- [ ] 11. Create documentation and deployment guides
  - [ ] 11.1 Write user documentation
    - Create comprehensive README with setup and usage instructions
    - Document configuration options and parameter tuning
    - Add troubleshooting guide for common issues
    - Create example configurations for different use cases
    - _Requirements: 5.1, 5.6_

  - [ ] 11.2 Create deployment documentation
    - Document database setup and migration procedures
    - Create API key setup and configuration guide
    - Add performance tuning and monitoring recommendations
    - Document integration with existing platform components
    - _Requirements: 6.1, 6.2, 6.3, 6.4_

- [ ] 12. Performance optimization and validation
  - [ ] 12.1 Optimize database queries and indexing
    - Analyze query performance and add additional indexes if needed
    - Implement query optimization for large datasets
    - Add database connection pooling and optimization
    - _Requirements: 6.2_

  - [ ] 12.2 Validate performance requirements
    - Test weekly screener completion within 3-hour target
    - Test daily deep scan completion within 30-minute target
    - Validate API rate limiting compliance
    - Test system performance with realistic data volumes
    - _Requirements: 5.3, 5.4_