# Refactoring Plan: Unified Financial Data Services Module

## 1. Executive Summary & Goals

This document outlines a comprehensive refactoring and design plan for the data module. The primary objective is to create a robust, provider-agnostic data access layer that simplifies data acquisition for both historical OHLCV data and live feeds. The architecture will be unified around a single, intelligent data manager that leverages a provider-agnostic file-based cache.

**Key Goals:**
1.  **Unified Architecture:** Establish a single, coherent architecture by implementing a provider-agnostic "Unified Cache" (`SYMBOL/TIMEFRAME/year.csv.gz`) and deprecating the conflicting provider-specific cache structure.
2.  **Simplified Data Access:** Introduce a `DataManager` facade as the sole entry point for all data requests, abstracting away the complexities of provider selection, rate limiting, and caching from the application layer.
3.  **Enhanced Live Feeds:** Ensure live data feeds are seamlessly integrated, using the unified cache for efficient historical data backfilling.

## 2. Current Situation Analysis

The existing data module has a strong foundation with several well-designed components:
-   Modular, provider-specific data downloaders inheriting from a `BaseDataDownloader`.
-   Dedicated utility modules for crucial cross-cutting concerns like rate limiting (`rate_limiting.py`), retries (`retry.py`), and data validation (`validation.py`).
-   Extensive documentation and a comprehensive test suite.

## 3. Proposed Solution / Refactoring Strategy

### 3.1. High-Level Design / Architectural Overview

The proposed architecture centralizes data access through a `DataManager` facade. This component orchestrates the process of retrieving data, ensuring that consumers (e.g., trading strategies, analytics tools) are completely decoupled from the underlying data sources and caching mechanisms.

The data retrieval flow will be as follows:

```mermaid
graph TD
    A[Application Layer] --> B{DataManager};
    B --> C{UnifiedCache.get()};
    C -- Cache Miss --> D{ProviderSelector};
    C -- Cache Hit --> E[Return Data];
    D -- Selects Provider --> F[Provider-Specific Downloader];
    F -- Fetches Raw Data --> G{DataHandler};
    G -- Standardizes & Validates --> H{UnifiedCache.put()};
    H --> E;
    E --> A;

    subgraph Data Services Core
        B; C; D; F; G; H;
    end

    subgraph External
        I[Data Provider APIs];
    end

    F --> I;
```

**Description of Flow:**
1.  The **Application Layer** requests data from the `DataManager`.
2.  The `DataManager` first attempts to retrieve the data from the `UnifiedCache`.
3.  **On a cache hit**, the data is returned immediately.
4.  **On a cache miss**, the `DataManager` consults the `ProviderSelector` to determine the best provider for the given symbol and timeframe.
5.  The `DataManager` invokes the appropriate provider-specific **Downloader** to fetch the data from the external API.
6.  The raw data is passed to a `DataHandler` for standardization (ensuring a consistent OHLCV format) and validation.
7.  The clean, standardized data is then stored in the `UnifiedCache` for future requests.
8.  Finally, the data is returned to the application.

### 3.2. Key Components / Modules

1.  **`DataManager` (New Facade):**
    -   **Responsibilities:** Acts as the single public entry point for all data operations. Implements the logic shown in the diagram above. Manages interactions between the cache, provider selector, and downloaders.
    -   **Key Methods:** `get_ohlcv(symbol, timeframe, start, end)`, `get_live_feed(symbol, timeframe)`.

2.  **`UnifiedCache` (Refactored `utils/file_based_cache.py`):**
    -   **Responsibilities:** Manages the physical storage and retrieval of data. Strictly adheres to the `SYMBOL/TIMEFRAME/year.csv.gz` structure. Handles data serialization (to gzipped CSV) and deserialization. Manages associated metadata files (`.json`).
    -   **Key Methods:** `get(symbol, timeframe, year)`, `put(symbol, timeframe, year, data, metadata)`, `exists(...)`.

3.  **`ProviderSelector` (New Module):**
    -   **Responsibilities:** Encapsulates the logic for choosing the best data provider based on rules defined in `PROVIDER_COMPARISON.md` and `Design.md` (e.g., asset type, timeframe, data quality).
    -   **Key Methods:** `get_best_provider(symbol, timeframe)`.

4.  **`LiveFeedManager` (Refactored `DataFeedFactory`):**
    -   **Responsibilities:** Manages the lifecycle of live data feeds.
    -   **Key Change:** It will be refactored to use the `DataManager` to acquire historical data for backfilling, ensuring that live feeds benefit from the unified cache.

5.  **Data Downloaders (Refactored):**
    -   **Responsibilities:** Each downloader class (e.g., `BinanceDataDownloader`) will be simplified to a single responsibility: fetching raw data from its specific API. All caching, validation, and rate-limiting logic will be removed from them and handled by the `DataManager` and its utilities.

### 3.3. Detailed Action Plan / Phases

#### Phase 1: Foundational Refactoring & Unification

-   **Objective(s):** Establish the core components of the new architecture.
-   **Priority:** High

-   **Task 1.1: Implement `UnifiedCache`**
    -   **Rationale/Goal:** Create the cornerstone of the new architecture, aligning with the user's specified cache structure.
    -   **Estimated Effort:** M
    -   **Deliverable/Criteria for Completion:** A class in a new `data/cache.py` module that can read and write gzipped pandas DataFrames to a `[cache_root]/[SYMBOL]/[TIMEFRAME]/[year].csv.gz` path. It must also handle a corresponding `[year].metadata.json` file. Unit tests must pass.

-   **Task 1.2: Implement `ProviderSelector`**
    -   **Rationale/Goal:** Centralize the provider selection logic into a testable and maintainable component.
    -   **Estimated Effort:** S
    -   **Deliverable/Criteria for Completion:** A class in a new `data/providers/selector.py` that implements the selection logic from `Design.md` (e.g., crypto -> binance, stock daily -> yfinance). Unit tests covering various symbols and timeframes must pass.

-   **Task 1.3: Create the `DataManager` Facade**
    -   **Rationale/Goal:** Establish the single entry point for data access.
    -   **Estimated Effort:** M
    -   **Deliverable/Criteria for Completion:** A new `DataManager` class in `data/manager.py`. It should have a `get_ohlcv` method that correctly orchestrates `UnifiedCache` and `ProviderSelector`. Initially, it can be tested with a mock downloader.

#### Phase 2: Integrating Historical Downloaders

-   **Objective(s):** Connect the existing data downloaders to the new `DataManager` and migrate existing cached data.
-   **Priority:** High (Dependent on Phase 1)

-   **Task 2.1: Refactor `BaseDataDownloader` and Concrete Implementations**
    -   **Rationale/Goal:** Decouple downloaders from caching and other concerns, enforcing the Single Responsibility Principle.
    -   **Estimated Effort:** L
    -   **Deliverable/Criteria for Completion:** All downloader classes are updated. They only contain logic for API communication and returning a raw pandas DataFrame. All caching calls (`self.cache.put`, etc.) are removed. The `DataManager` now calls the appropriate downloader on a cache miss.

-   **Task 2.2: Implement Rate Limiting and Retries in `DataManager`**
    -   **Rationale/Goal:** Centralize control over external API interactions to ensure compliance and robustness.
    -   **Estimated Effort:** M
    -   **Deliverable/Criteria for Completion:** The `DataManager` uses the existing `utils/rate_limiting.py` and `utils/retry.py` modules before calling a downloader. Provider-specific limits from `PROVIDER_COMPARISON.md` are configured.

#### Phase 3: Integrating Live Feeds

-   **Objective(s):** Align live data feeds with the new unified data access layer.
-   **Priority:** Medium (Dependent on Phase 1 & 2)

-   **Task 3.1: Refactor `BaseLiveDataFeed`**
    -   **Rationale/Goal:** Ensure live feeds use the same robust, cached data source for historical backfilling as the rest of the application.
    -   **Estimated Effort:** M
    -   **Deliverable/Criteria for Completion:** The `_load_historical_data` method in `BaseLiveDataFeed` is refactored to call `DataManager.get_ohlcv` instead of performing its own download.

-   **Task 3.2: Update Concrete Live Feeds**
    -   **Rationale/Goal:** Ensure all live feeds conform to the updated base class.
    -   **Estimated Effort:** S
    -   **Deliverable/Criteria for Completion:** `BinanceLiveFeed`, `YahooLiveFeed`, etc., are updated and tested to ensure they correctly backfill data from the unified cache.

#### Phase 4: Finalization and Cleanup

-   **Objective(s):** Solidify the new architecture with comprehensive testing and updated documentation.
-   **Priority:** Medium

-   **Task 4.1: Create Integration Tests for `DataManager`**
    -   **Rationale/Goal:** Verify the entire data retrieval flow works as designed.
    -   **Estimated Effort:** M
    -   **Deliverable/Criteria for Completion:** New integration tests that cover scenarios like: first-time download (cache miss), subsequent request (cache hit), and data requests spanning multiple years.

-   **Task 4.2: Update and Unify Documentation**
    -   **Rationale/Goal:** Eliminate architectural ambiguity and provide a single source of truth for developers.
    -   **Estimated Effort:** M
    -   **Deliverable/Criteria for Completion:** `README.md`, `Design.md`, and other key documents are updated to reflect the `DataManager` and `UnifiedCache` architecture. Conflicting documents like `PHASE4_DOCUMENTATION.md` are either removed or heavily revised to align with the new standard.

### 3.4. Data Model Changes

The primary data model change is the **cache structure on the file system** (done).

-   **Current :** `[cache_root]/[SYMBOL]/[TIMEFRAME]/[year].csv.gz`

The **metadata file** (`[year].metadata.json`) associated with each data file will be crucial. It should be standardized to include:
```json
{
  "symbol": "LTCUSDT",
  "timeframe": "15m",
  "year": 2019,
  "data_source": "binance",
  "created_at": "2025-09-04T10:54:50.186152",
  "last_updated": "2025-09-04T10:54:50.186158",
  "start_date": "2019-12-31T23:00:00",
  "end_date": "2019-12-31T23:45:00",
  "data_quality": {
    "score": 1.0,
    "validation_errors": [],
    "gaps": 0,
    "duplicates": 0
  },
  "file_info": {
    "format": "csv.gz",
    "size_bytes": 345,
    "rows": 4,
    "columns": [
      "open",
      "high",
      "low",
      "close",
      "volume",
      "close_time",
      "quote_asset_volume",
      "number_of_trades",
      "taker_buy_base_asset_volume",
      "taker_buy_quote_asset_volume",
      "ignore"
    ]
  },
  "provider_info": {
    "name": "binance",
    "reliability": 0.95,
    "rate_limit": "unknown"
  }
}
```

### 3.5. API Design / Interface Changes

The main public-facing API change will be the introduction of the `DataManager`. All other components will become internal implementation details.

**Proposed `DataManager` Interface:**
```python
class DataManager:
    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Retrieves historical OHLCV data, handling caching and provider selection.
        """
        # ... implementation ...

    def get_live_feed(
        self,
        symbol: str,
        timeframe: str,
        # ... other relevant params ...
    ) -> BaseLiveDataFeed:
        """
        Creates and returns a live data feed instance, pre-filled with historical data.
        """
        # ... implementation ...
```

## 4. Key Considerations & Risk Mitigation

### 4.1. Technical Risks & Challenges

-   **Cache Migration Complexity:** Migrating existing data from the provider-specific cache can be complex, especially if metadata formats differ.
    -   **Mitigation:** The migration script (Task 2.3) must be developed carefully and tested on a copy of the cache. It should be idempotent and include a dry-run mode.
-   **Data Standardization:** Different providers return data in slightly different formats (e.g., column names, timestamp formats).
    -   **Mitigation:** The `DataHandler` component is critical. It must contain robust logic to transform all incoming data into the single, canonical format before validation and caching.
-   **Performance of Gzipped CSV:** While `gzip` provides good compression, reading large gzipped CSV files can be slower than binary formats like Parquet.
    -   **Mitigation:** Data is partitioned by year, which keeps individual file sizes manageable. For performance-critical applications, the `UnifiedCache` could be extended in the future to support Parquet as an alternative format, selectable via configuration.

### 4.2. Dependencies

-   The successful completion of **Phase 1** is a hard dependency for all subsequent phases.
-   **Phase 2** (Historical Downloaders) must be completed before **Phase 3** (Live Feeds), as the live feeds will depend on the `DataManager` for backfilling.
-   The project relies on external provider APIs. Any changes to these APIs could impact the downloader implementations.

### 4.3. Non-Functional Requirements (NFRs) Addressed

-   **Maintainability:** Greatly improved by centralizing data access logic in the `DataManager` and enforcing the Single Responsibility Principle for downloaders.
-   **Reliability:** Centralized error handling, retry logic, and validation within the `DataManager` will lead to more consistent and robust behavior.
-   **Performance:** The `UnifiedCache` ensures that data is downloaded only once, significantly speeding up subsequent requests. Partitioning by year prevents individual files from becoming excessively large.
-   **Extensibility:** Adding a new data provider becomes much simpler: only a new downloader class needs to be created and registered with the `ProviderSelector`. No changes to caching or other core logic are required.

## 5. Success Metrics / Validation Criteria

The success of this refactoring plan will be measured by the following criteria:
1.  **Architectural Unification:** The codebase exclusively uses the `DataManager` for data access.
2.  **Functional Correctness:** All existing unit and integration tests pass after the refactoring. New integration tests for the `DataManager` (covering cache hits/misses) are implemented and pass.
3.  **Performance:** For a given symbol and timeframe, the second request for data is at least 10x faster than the first (demonstrating a successful cache hit).
4.  **Developer Experience:** A developer can add a new data provider by implementing a single downloader class and adding one line to the `ProviderSelector` configuration.

## 6. Assumptions Made

-   The "Unified Cache" architecture (`SYMBOL/TIMEFRAME/year.gz`) is the desired end state, as it aligns with the user's request and the strategic goal of provider-agnosticism.
-   The existing utility modules (`rate_limiting.py`, `retry.py`, `validation.py`) are functionally sound and can be integrated into the `DataManager` without major rewrites.
-   The performance of reading gzipped CSV files partitioned by year is acceptable for the system's requirements.
-   API keys will continue to be managed via environment variables.

## 7. Implementation Progress

### ✅ Completed (Phase 1 - Foundational Refactoring)

-   **Task 1.1: Implement `UnifiedCache`** ✅
    -   **Status**: Completed
    -   **Location**: `src/data/cache/unified_cache.py`
    -   **Features**: Provider-agnostic caching with `SYMBOL/TIMEFRAME/year.csv.gz` structure

-   **Task 1.2: Implement `ProviderSelector`** ✅
    -   **Status**: Completed and Enhanced
    -   **Location**: `src/data/data_manager.py` (ProviderSelector class)
    -   **Features**: Configuration-driven provider selection with comprehensive ticker classification
    -   **Enhancement**: Replaced `TickerClassifier` functionality with improved, configurable approach

-   **Task 1.3: Create the `DataManager` Facade** ✅
    -   **Status**: Completed
    -   **Location**: `src/data/data_manager.py` (DataManager class)
    -   **Features**: Single entry point for all data operations with orchestration logic

### ✅ Additional Completed Tasks

-   **Base Classes Reorganization** ✅
    -   **Status**: Completed
    -   **Changes**: Moved `BaseDataSource` to `src/data/sources/` subfolder
    -   **Benefits**: Better code organization and clearer module structure

-   **TickerClassifier Replacement** ✅
    -   **Status**: Completed
    -   **Replacement**: Enhanced `ProviderSelector` with configuration-driven rules
    -   **Benefits**: Better maintainability, extensibility, and integration with unified architecture
    -   **Migration Guide**: `src/data/docs/TICKER_CLASSIFIER_MIGRATION.md`

### ✅ Completed (Phase 2 - Historical Downloaders)

-   **Task 2.1: Refactor `BaseDataDownloader` and Concrete Implementations** ✅
    -   **Status**: Completed
    -   **Changes**: Removed caching, file management, and rate limiting logic from downloaders
    -   **Benefits**: Downloaders now focus solely on API communication, following Single Responsibility Principle

-   **Task 2.2: Implement Rate Limiting and Retries in `DataManager`** ✅
    -   **Status**: Completed
    -   **Implementation**: DataManager now handles rate limiting and retries centrally
    -   **Benefits**: Consistent rate limiting across all providers, centralized error handling

-   **Task 2.3: Create Cache Migration Script** ✅
    -   **Status**: Completed (but not needed - cache structure already correct)
    -   **Note**: Migration script created but cache structure was already in correct format

### ✅ Completed (Phase 3 - Live Feed Integration)

-   **Task 3.1: Refactor `BaseLiveDataFeed`** ✅
    -   **Status**: Completed
    -   **Changes**: Updated `BaseLiveDataFeed` to use `DataManager.get_ohlcv()` for historical data loading
    -   **Benefits**: Consistent data access and caching across all live feeds

-   **Task 3.2: Update Concrete Live Feeds** ✅
    -   **Status**: Completed
    -   **Changes**: Updated `YahooLiveDataFeed` and `BinanceLiveDataFeed` to inherit the new base implementation
    -   **Benefits**: All live feeds now use unified data access layer

### ✅ Completed (Phase 4 - Finalization)

-   **Task 4.1: Create Integration Tests for `DataManager`** ✅
    -   **Status**: Completed
    -   **Location**: `src/data/tests/integration/test_data_manager_integration.py`
    -   **Coverage**: Cache hit/miss scenarios, provider selection, data validation, live feed integration, error handling

-   **Performance Validation** ✅
    -   **Status**: Completed
    -   **Location**: `src/data/tests/performance/test_cache_performance.py`
    -   **Validation**: Cache performance improvements, concurrent access, memory usage, compression efficiency

### 📋 Remaining Tasks

-   **Documentation Updates**: Update all documentation to reflect new architecture
-   **Final Testing**: Run comprehensive test suite to ensure all functionality works

## 8. Open Questions / Areas for Further Investigation

-   **Provider Failover Strategy:** ✅ **RESOLVED** - Implemented in `ProviderSelector.get_provider_with_failover()`
-   **Data Quality Conflict Resolution:** If cached data from one provider has a low quality score, should a new request automatically try to fetch from a higher-quality provider? (Recommendation: This should be a configurable behavior).
-   **Configuration of `ProviderSelector`:** ✅ **RESOLVED** - Now uses `config/data/provider_rules.yaml` for all rules