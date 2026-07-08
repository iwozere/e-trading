"""
Configuration Loader
===================

Simple configuration loader that loads YAML/JSON files and validates them
using Pydantic models. Provides error handling and validation feedback.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Union

import yaml
from pydantic import ValidationError

from src.model.config_models import (
    BrokerConfig,
    DataConfig,
    DataSourceType,
    OptimizerConfig,
    StrategyParamsConfig,
    TradingBotConfig,
)


def load_config(config_path: Union[str, Path]) -> TradingBotConfig:
    """
    Load and validate a trading bot configuration from file.

    Args:
        config_path: Path to configuration file (JSON or YAML)

    Returns:
        Validated TradingBotConfig instance

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValidationError: If config validation fails
        ValueError: If file format is not supported
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Load raw config data
    raw_config = _load_raw_config(config_path)

    # Validate and return TradingBotConfig
    try:
        return TradingBotConfig(**raw_config)
    except ValidationError as e:
        print(f"Configuration validation error: {e}")
        raise


def load_optimizer_config(config_path: Union[str, Path]) -> OptimizerConfig:
    """
    Load and validate an optimizer configuration from file.

    Args:
        config_path: Path to configuration file (JSON or YAML)

    Returns:
        Validated OptimizerConfig instance
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    raw_config = _load_raw_config(config_path)

    try:
        return OptimizerConfig(**raw_config)
    except ValidationError as e:
        print(f"Optimizer configuration validation error: {e}")
        raise


def load_data_config(config_path: Union[str, Path]) -> DataConfig:
    """
    Load and validate a data configuration from file.

    Args:
        config_path: Path to configuration file (JSON or YAML)

    Returns:
        Validated DataConfig instance
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    raw_config = _load_raw_config(config_path)

    try:
        return DataConfig(**raw_config)
    except ValidationError as e:
        print(f"Data configuration validation error: {e}")
        raise


def _load_raw_config(config_path: Path) -> Dict[str, Any]:
    """
    Load raw configuration data from file.

    Args:
        config_path: Path to configuration file

    Returns:
        Raw configuration dictionary

    Raises:
        ValueError: If file format is not supported
    """
    suffix = config_path.suffix.lower()

    try:
        if suffix in [".yaml", ".yml"]:
            with open(config_path, encoding="utf-8") as f:
                return yaml.safe_load(f)
        elif suffix == ".json":
            with open(config_path, encoding="utf-8") as f:
                return json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {suffix}. Use .json, .yaml, or .yml")
    except (yaml.YAMLError, json.JSONDecodeError) as e:
        raise ValueError(f"Error parsing configuration file {config_path}: {e}")


def save_config(config: Union[TradingBotConfig, OptimizerConfig, DataConfig], output_path: Union[str, Path]) -> None:
    """
    Save a configuration to file.

    Args:
        config: Configuration instance to save
        output_path: Output file path
    """
    output_path = Path(output_path)

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dictionary
    config_dict = config.model_dump()

    # Save based on file extension
    suffix = output_path.suffix.lower()

    if suffix in [".yaml", ".yml"]:
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    elif suffix == ".json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    else:
        raise ValueError(f"Unsupported output format: {suffix}. Use .json, .yaml, or .yml")


def validate_config_file(config_path: Union[str, Path]) -> tuple[bool, list[str], list[str]]:
    """
    Validate a configuration file without loading it.

    Args:
        config_path: Path to configuration file

    Returns:
        Tuple of (is_valid, errors, warnings)
    """
    errors: List[str] = []
    warnings: List[str] = []

    try:
        # Try to load the config
        config = load_config(config_path)

        # Additional validation checks
        take_profit = config.risk.get("take_profit_pct", 0.0)
        stop_loss = config.risk.get("stop_loss_pct", 0.0)
        if take_profit <= stop_loss:
            warnings.append("Take profit should be greater than stop loss")

        risk_per_trade = config.risk.get("risk_per_trade", 0.0)
        if risk_per_trade > 0.05:
            warnings.append("Risk per trade is quite high (>5%)")

        max_open_trades = config.trading.get("max_open_trades", 0)
        if max_open_trades > 10:
            warnings.append("Maximum open trades is quite high (>10)")

        return True, errors, warnings

    except Exception as e:
        errors.append(str(e))
        return False, errors, warnings


def create_sample_config(output_path: Union[str, Path], config_type: str = "trading") -> None:
    """
    Create a sample configuration file.

    Args:
        output_path: Output file path
        config_type: Type of configuration ("trading", "optimizer", "data")
    """
    sample_config: Union[TradingBotConfig, OptimizerConfig, DataConfig]
    if config_type == "trading":
        sample_config = TradingBotConfig(
            bot_id="sample_bot_001",
            symbol="BTCUSDT",
            broker=BrokerConfig(type="binance_paper"),
            strategy=StrategyParamsConfig(type="rsi_bb"),
            data={"data_source": "binance"},
        )
    elif config_type == "optimizer":
        sample_config = OptimizerConfig(
            optimizer_id="sample_opt_001",
            name="Sample Optimizer",
            optimizer_type="optuna",
            initial_capital=10000.0,
            n_trials=100,
            strategy_name="rsi_bb",
            param_ranges={"rsi_period": {"low": 10, "high": 20}},
            symbol="BTCUSDT",
            start_date="2023-01-01",
            end_date="2023-12-31",
        )
    elif config_type == "data":
        sample_config = DataConfig(
            data_id="sample_data_001",
            name="Sample Data Config",
            data_source=DataSourceType.BINANCE,
            symbols=["BTCUSDT"],
        )
    else:
        raise ValueError(f"Unknown config type: {config_type}")

    save_config(sample_config, output_path)
    print(f"Sample {config_type} configuration created: {output_path}")


def convert_old_config(old_config_path: Union[str, Path], new_config_path: Union[str, Path]) -> None:
    """
    Convert old configuration format to new simplified format.

    Args:
        old_config_path: Path to old configuration file
        new_config_path: Path for new configuration file
    """
    # Load old config
    old_config = _load_raw_config(Path(old_config_path))

    # Convert to new format
    new_config = _convert_old_to_new_format(old_config)

    # Save new config
    save_config(new_config, new_config_path)
    print(f"Converted configuration saved: {new_config_path}")


def _convert_old_to_new_format(old_config: Dict[str, Any]) -> TradingBotConfig:
    """
    Convert old configuration format to new TradingBotConfig.

    Args:
        old_config: Old configuration dictionary

    Returns:
        New TradingBotConfig instance
    """
    # Extract values from old format
    bot_id = old_config.get("bot_id", "converted_bot_001")
    symbol = old_config.get("trading", {}).get("symbol", "BTCUSDT")
    broker_type = old_config.get("broker", {}).get("type", "binance_paper")
    data_source = old_config.get("data", {}).get("data_source", "binance")

    strategy_type = old_config.get("strategy", {}).get("type", "custom")
    strategy_params = old_config.get("strategy", {}).get("parameters", {})
    if not strategy_type and "strategy_params" in old_config:
        strategy_type = "custom"
        strategy_params = old_config["strategy_params"]

    # Create new config
    new_config = TradingBotConfig(
        bot_id=bot_id,
        symbol=symbol,
        broker=BrokerConfig(type=broker_type),
        strategy=StrategyParamsConfig(type=strategy_type, parameters=strategy_params),
        data={"data_source": data_source},
    )

    # Copy over other values if they exist
    if "trading" in old_config:
        trading = old_config["trading"]
        if "position_size" in trading:
            new_config.trading["position_size"] = trading["position_size"]
        if "max_positions" in trading:
            new_config.trading["max_open_trades"] = trading["max_positions"]

    if "risk_management" in old_config:
        risk = old_config["risk_management"]
        if "stop_loss_pct" in risk:
            new_config.risk["stop_loss_pct"] = risk["stop_loss_pct"]
        if "take_profit_pct" in risk:
            new_config.risk["take_profit_pct"] = risk["take_profit_pct"]

    if "logging" in old_config:
        logging = old_config["logging"]
        if "level" in logging:
            new_config.logging["level"] = logging["level"]

    return new_config
