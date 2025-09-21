#!/usr/bin/env python3
"""
Configuration utilities for the e-trading project.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Union


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def validate_config(config: Dict[str, Any], schema: Dict[str, Any]) -> bool:
    """
    Validate configuration against schema.

    Args:
        config: Configuration dictionary
        schema: Schema dictionary defining expected structure

    Returns:
        True if valid, raises ValueError if invalid
    """
    for key, expected_type in schema.items():
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")

        if not isinstance(config[key], expected_type):
            raise ValueError(f"Invalid type for {key}: expected {expected_type}, got {type(config[key])}")

    return True


def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        config_path: Path to save configuration file
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
