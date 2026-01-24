# Modular Configuration Architecture Proposal

## Vision
To achieve the "LEGO-like" architecture, we will decouple the "Bot Definition" from its "Component Configurations". A Bot Configuration (whether in a file or DB) effectively becomes a **Mainboard** that plugs in various **Modules** (Broker, Risk, Strategy, etc.).

We will introduce a `ConfigurationFactory` that performs "Assembly" + "Validation".

## 1. The "Contract" Layer (YAML Schemas)
You have started this correctly in `config/contracts/*.yaml`. These act as the **Interface Definitions**.
- `broker.yaml`
- `strategy.yaml`
- `risk-management.yaml`
- `notifications.yaml`

## 2. The "Instance" Layer (JSON Components)
These are the implementation "parts".
- `config/contracts/instances/brokers/binance-paper.json`
- `config/contracts/instances/strategies/rsi-bb.json`
- `config/contracts/instances/risk/conservative.json`

## 3. The "Assembly" Layer (The Bot Config)
The Bot Configuration file (or DB record) becomes a **Reference Manifest**.
Instead of defining the full object, it *references* the instance IDs.

**Example `config/bots/my-cool-bot.json`:**
```json
{
  "bot_id": "my-cool-bot",
  "modules": {
    "broker": "instances/brokers/binance_paper",
    "strategy": "instances/strategies/rsi_bb",
    "risk": "instances/risk/conservative",
    "notifications": "instances/notifications/standard"
  },
  "overrides": {
    "symbol": "BTCUSDT",
    "initial_balance": 50000
  }
}
```

## 4. The Engineering Implementation

### A. `ConfigurationFactory` (New Service)
This service is responsible for:
1.  **Loading**: Reading the "Mainboard" config.
2.  **Resolving**: For each module reference, loading the corresponding JSON file from `config/contracts/instances/`.
3.  **Merging**: Applying any "overrides" from the Mainboard to the component configs (e.g., changing `initial_balance` on the Broker config).
4.  **Validating**: Using the YAML schemas (Contracts) to validate that the *assembled* object meets the requirements.
5.  **Outputting**: Returning a fully hydrated `TradingBotConfig` object (compatible with your existing Pydantic models).

### B. Database Integration
The `trading_bots` table currently stores a big JSON blob.
- **Immediate Path**: The DB can store the *Reference Manifest* JSON (The "Assembly" Layer). The `ConfigurationFactory` handles the expansion at runtime.
- **Future Path**: We could normalize the DB to have `brokers`, `strategies` tables, but storing the Manifest JSON is more flexible and simpler to implement now.

## Implementation Steps

1.  **Standardize Directory Structure**:
    - `config/contracts/schemas/` (The .yaml files)
    - `config/contracts/instances/` (The .json parts folders)
    - `config/contracts/manifests/` (The bot definitions)

2.  **Develop `ConfigurationFactory`**:
    - `load_manifest(manifest_path_or_json)`
    - `resolve_references(manifest)`
    - `validate_against_contracts(assembled_config)`

3.  **Refactor `StrategyManager`**:
    - Update `load_strategies_from_db` and `load_strategies_from_config` to use `ConfigurationFactory.build_config()`.

4.  **Refactor `LiveTradingBot`**:
    - Update `_load_configuration` to use `ConfigurationFactory`.

## "Senior Architect" Add-ons
- **Hot-Swapping**: Since components are decoupled, you could theoretically validatate a "Risk Module" swap at runtime.
- **Validation CLI**: A tool `python tools/validate_bot.py my-bot` that reports "Your Strategy module violates the StrategyContract: missing 'exit_logic'".

This approach maximizes code reuse because `LiveTradingBot` doesn't care *how* the config was built, only that it receives a valid one. The complexity is contained entirely in the `ConfigurationFactory`.
