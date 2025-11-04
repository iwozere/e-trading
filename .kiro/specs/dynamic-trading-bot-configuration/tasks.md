# Implementation Plan

## Architecture Principles (UPDATED)

**Key principle: Single source of configuration from database via `strategy_manager.py` ONLY.**

### Component Responsibilities

| Component | Purpose | Config Loading | Current Status |
|-----------|---------|----------------|----------------|
| `trading_runner.py` | Service orchestrator | ❌ NO (delegates to StrategyManager) | Needs simplification |
| `strategy_manager.py` | Bot manager | ✅ YES (SOLE CONFIG LOADER) | Mostly complete, needs StrategyHandler integration |
| `strategy_handler.py` | Strategy factory | ❌ NO (receives configs) | NEW - needs creation |
| `StrategyInstance` | Bot wrapper | ❌ NO (receives configs) | Needs LiveTradingBot refactor |
| `trading_bot.py` | Single-bot runner | N/A | DEPRECATED (replaced by trading_runner.py) |
| `live_trading_bot.py` | Old bot implementation | N/A | REFACTOR into StrategyInstance |

### Data Flow

```
Database (trading_bots table)
    ↓
trading_service.get_enabled_bots()
    ↓
strategy_manager.load_strategies_from_db() ← SOLE CONFIG LOADER
    ↓
StrategyInstance objects created
    ↓
strategy_handler.get_strategy_class() → Strategy instantiation
    ↓
Bot execution with BaseTradingBot
```

---

## Implementation Tasks

- [x] 1. Create database integration layer for bot configuration management
  - Implement database connection and query methods for trading_bots table
  - Create configuration validation and parsing utilities
  - Add database schema validation for bot configuration JSON
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [x] 2. Implement Strategy Handler for dynamic strategy loading
  - [x] 2.1 Create StrategyHandler class with plugin architecture
    - Design strategy registry system for different strategy types
    - Implement dynamic import mechanism for strategy classes
    - Create strategy validation framework
    - _Requirements: 3.1, 3.2, 3.5_

  - [x] 2.2 Implement CustomStrategy integration and fallback logic
    - Add CustomStrategy loading with entry/exit mixin support
    - Implement parameter validation for mixin configurations
    - Create fallback mechanism for unknown strategy types
    - _Requirements: 3.2, 3.3, 3.5_

  - [x] 2.3 Add plugin architecture for future strategy types
    - Design extensible strategy registration system
    - Create base interfaces for new strategy types
    - Implement strategy type discovery mechanism
    - _Requirements: 3.4, 3.5_

- [-] 3. Simplify trading_runner.py as service orchestrator (NO CONFIG LOADING)
  - [x] 3.1 Remove JSON configuration loading completely
    - **Remove** all JSON file loading logic
    - **Remove** any database configuration loading (delegate to StrategyManager)
    - Simplify to pure orchestration (start/stop coordination)
    - _Requirements: 6.1, 6.2, 6.3_
    - _Current state: Has JSON loading (lines 58-77) - MUST BE REMOVED_

  - [x] 3.2 Delegate all bot management to StrategyManager
    - Replace direct bot creation with StrategyManager calls
    - Call `strategy_manager.load_strategies_from_db()`
    - Call `strategy_manager.start_all_strategies()`
    - Call `strategy_manager.shutdown()` on stop
    - _Requirements: 6.2, 6.3, 6.4, 6.5_
    - _Current state: Partially delegates, needs cleanup_

  - [ ] 3.3 Add system service integration
    - Implement systemd service file for Linux
    - Add Windows service wrapper integration
    - Implement signal handlers (SIGINT, SIGTERM)
    - Create graceful shutdown coordination
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_
    - _Current state: Signal handlers exist (lines 325-332), needs enhancement_

- [x] 4. Enhance strategy_manager.py as SOLE CONFIG LOADER ✅ COMPLETED
  - [x] 4.1 Ensure strategy_manager is ONLY config loader (already mostly done)
    - **Verify** `load_strategies_from_db()` is complete (lines 356-400)
    - **Remove** any JSON-based loading (already removed)
    - **Ensure** StrategyInstance uses configs from manager only
    - **Integrate** with StrategyHandler for strategy class loading
    - _Requirements: 7.1, 7.2, 8.1, 8.2, 8.3_
    - _Status: ✅ COMPLETED - StrategyHandler integration complete_

  - [x] 4.2 Enhance StrategyInstance with LiveTradingBot logic
    - **Refactor** LiveTradingBot logic into StrategyInstance
    - Add data feed management (from LiveTradingBot._create_data_feed)
    - Add Backtrader setup (from LiveTradingBot._setup_backtrader)
    - Maintain async bot execution with isolation
    - _Requirements: 2.1, 2.2, 2.3, 7.3_
    - _Status: ✅ COMPLETED - All LiveTradingBot logic refactored into StrategyInstance_

  - [x] 4.3 Enhance database status updates and monitoring
    - **Improve** bot status updates in DB polling (lines 471-477)
    - Add performance metrics recording to trading_performance_metrics
    - Enhance heartbeat system (framework exists)
    - Add trade recording via trading_service
    - _Requirements: 7.4, 9.1, 9.2, 9.3, 9.5, 10.1, 10.2_
    - _Status: ✅ COMPLETED - Heartbeat system, performance metrics, and trade recording all implemented_

- [-] 5. Implement notification integration for trade events
  - [x] 5.1 Create notification service integration layer
    - Design notification event system for trade events
    - Implement notification service client integration
    - Add notification configuration parsing from bot config
    - _Requirements: 4.1, 4.2, 4.4_

  - [ ] 5.2 Add trade event notification triggers
    - Implement buy/sell order notifications
    - Add error notification system
    - Create notification filtering based on bot configuration
    - _Requirements: 4.1, 4.2, 4.3, 4.5_

  - [ ] 5.3 Add notification testing and validation
    - Create notification service integration tests
    - Add notification delivery validation
    - Implement notification retry and error handling
    - _Requirements: 4.4, 4.5_

- [ ] 6. Implement comprehensive error handling and validation
  - [ ] 6.1 Create configuration validation framework
    - Implement JSON schema validation for bot configurations
    - Add broker and strategy compatibility validation
    - Create detailed error reporting and logging
    - _Requirements: 8.1, 8.2, 8.3, 8.5_

  - [ ] 6.2 Add runtime error handling and recovery
    - Implement retry logic with exponential backoff
    - Add automatic bot recovery mechanisms
    - Create error isolation and containment
    - _Requirements: 8.4, 8.5, 2.3_

  - [ ] 6.3 Add comprehensive error handling tests
    - Create error scenario test cases
    - Add validation error testing
    - Implement recovery mechanism testing
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

- [x] 7. Implement database state management and persistence ✅ COMPLETED
  - [x] 7.1 Create bot status and lifecycle management
    - ✅ Implement bot status updates (running, stopped, error)
    - ✅ Add started_at and last_heartbeat timestamp management
    - ✅ Create bot lifecycle state persistence
    - _Requirements: 9.1, 9.4, 9.5_
    - _Status: ✅ COMPLETED in Task 4.3 - Heartbeat system, status updates, lifecycle tracking all implemented_

  - [x] 7.2 Add trade recording and performance tracking
    - ✅ Implement trade data recording to trading_trades table
    - ✅ Add performance metrics calculation and storage
    - ✅ Create balance and P&L tracking
    - _Requirements: 9.2, 9.3_
    - _Status: ✅ COMPLETED in Task 4.3 - Trade recording, performance metrics, P&L tracking all implemented_

  - [x] 7.3 Implement service restart recovery
    - ✅ Add bot state recovery on service restart
    - ✅ Implement configuration reload mechanisms
    - ✅ Create graceful bot resumption logic
    - _Requirements: 9.4, 5.5_
    - _Status: ✅ COMPLETED - Crash detection, smart resume, state recovery, graceful shutdown all implemented_

- [ ] 8. Add monitoring and health management system
  - [ ] 8.1 Implement bot health monitoring
    - Create heartbeat monitoring system
    - Add bot responsiveness detection
    - Implement automatic recovery for unresponsive bots
    - _Requirements: 10.1, 10.2, 10.5_

  - [ ] 8.2 Add system resource monitoring
    - Implement CPU and memory usage monitoring
    - Add resource constraint detection and warnings
    - Create performance metrics collection
    - _Requirements: 10.3, 10.5_

  - [ ] 8.3 Create structured logging and metrics
    - Implement structured logging with bot correlation IDs
    - Add metrics collection for external monitoring systems
    - Create log aggregation and analysis capabilities
    - _Requirements: 10.4, 10.5_

- [ ] 9. Deprecate trading_bot.py and refactor live_trading_bot.py
  - [ ] 9.1 Deprecate trading_bot.py (single-bot runner)
    - **Mark** trading_bot.py as deprecated
    - **Document** that trading_runner.py is the new service entry point
    - **Optionally** keep for backward compatibility or single-bot scenarios
    - Create migration guide from trading_bot.py to trading_runner.py
    - _Requirements: 5.1, 5.2, 6.1_
    - _Current state: trading_bot.py exists, runs single LiveTradingBot_

  - [ ] 9.2 Refactor LiveTradingBot logic into StrategyInstance
    - **Extract** Backtrader setup logic from LiveTradingBot
    - **Extract** data feed management from LiveTradingBot
    - **Extract** trading loop execution from LiveTradingBot
    - **Integrate** into StrategyInstance.start()
    - **Optionally** keep LiveTradingBot for reference or deprecate
    - _Requirements: 2.1, 2.2, 7.1_
    - _Current state: LiveTradingBot has needed logic, needs extraction_

  - [ ] 9.3 Add service integration testing
    - Create service startup and shutdown tests
    - Add multi-bot execution testing
    - Implement service recovery testing
    - Test DB polling and hot-reload
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [ ] 10. Integration testing and system validation
  - [ ] 10.1 Create end-to-end integration tests
    - Implement database-to-bot execution testing
    - Add multi-bot concurrent execution validation
    - Create notification integration testing
    - _Requirements: All requirements integration_

  - [ ] 10.2 Add performance and load testing
    - Create multi-bot performance testing
    - Add resource usage validation
    - Implement scalability testing
    - _Requirements: 2.1, 2.2, 10.3, 10.5_

  - [ ] 10.3 Add system reliability testing
    - Create failure recovery testing
    - Add database connectivity failure testing
    - Implement service restart and recovery validation
    - _Requirements: 8.4, 8.5, 9.4, 10.1, 10.2_
