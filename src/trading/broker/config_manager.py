#!/usr/bin/env python3
"""
Broker Configuration Management System
-------------------------------------

This module provides comprehensive configuration management for trading brokers
including configuration validation, templates, hot-reloading, and environment
management.

Features:
- Configuration templates and presets
- Environment-specific configurations (dev, staging, prod)
- Configuration validation and normalization
- Hot-reloading and dynamic updates
- Configuration versioning and rollback
- Secure credential management

Classes:
- ConfigManager: Main configuration management class
- ConfigTemplate: Configuration template system
- EnvironmentManager: Environment-specific configuration management
"""

import json
import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib

from src.trading.broker.config_validator import validate_and_create_broker_config
from src.trading.broker.binance_utils import create_binance_config_template
from src.trading.broker.ibkr_utils import create_ibkr_config_template

from src.notification.logger import setup_logger
_logger = setup_logger(__name__)


class Environment(Enum):
    """Environment enumeration."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class ConfigFormat(Enum):
    """Configuration file format enumeration."""
    JSON = "json"
    YAML = "yaml"
    TOML = "toml"


@dataclass
class ConfigVersion:
    """Configuration version information."""
    version: str
    timestamp: datetime
    description: str
    config_hash: str
    author: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'version': self.version,
            'timestamp': self.timestamp.isoformat(),
            'description': self.description,
            'config_hash': self.config_hash,
            'author': self.author
        }


class ConfigTemplate:
    """Configuration template system."""

    def __init__(self):
        self.templates: Dict[str, Dict[str, Any]] = {}
        self._load_builtin_templates()

    def _load_builtin_templates(self):
        """Load built-in configuration templates."""
        # Binance templates
        self.templates['binance_paper'] = create_binance_config_template('paper')
        self.templates['binance_live'] = create_binance_config_template('live')

        # IBKR templates
        self.templates['ibkr_paper'] = create_ibkr_config_template('paper')
        self.templates['ibkr_live'] = create_ibkr_config_template('live')

        # Mock template
        self.templates['mock_dev'] = {
            'type': 'mock',
            'trading_mode': 'paper',
            'cash': 100000.0,
            'paper_trading_config': {
                'mode': 'basic',
                'commission_rate': 0.0,
                'slippage_model': 'fixed',
                'base_slippage': 0.0,
                'latency_simulation': False
            },
            'notifications': {
                'position_opened': False,
                'position_closed': False,
                'email_enabled': False,
                'telegram_enabled': False
            }
        }

        # Conservative template
        self.templates['conservative'] = {
            'type': 'ibkr',
            'trading_mode': 'paper',
            'cash': 10000.0,
            'paper_trading_config': {
                'mode': 'realistic',
                'commission_rate': 0.0005,
                'base_slippage': 0.0002,
                'partial_fill_probability': 0.05,
                'reject_probability': 0.001
            },
            'risk_management': {
                'max_position_size': 200.0,
                'max_daily_loss': 100.0,
                'max_portfolio_risk': 0.005
            }
        }

        # Aggressive template
        self.templates['aggressive'] = {
            'type': 'binance',
            'trading_mode': 'paper',
            'cash': 50000.0,
            'paper_trading_config': {
                'mode': 'advanced',
                'commission_rate': 0.001,
                'base_slippage': 0.001,
                'partial_fill_probability': 0.25,
                'reject_probability': 0.05,
                'market_impact_enabled': True
            },
            'risk_management': {
                'max_position_size': 5000.0,
                'max_daily_loss': 2500.0,
                'max_portfolio_risk': 0.05
            }
        }

    def get_template(self, template_name: str) -> Optional[Dict[str, Any]]:
        """Get a configuration template."""
        return self.templates.get(template_name)

    def list_templates(self) -> List[str]:
        """List available templates."""
        return list(self.templates.keys())

    def add_template(self, name: str, config: Dict[str, Any]):
        """Add a custom template."""
        self.templates[name] = config.copy()
        _logger.info("Added configuration template: %s", name)

    def remove_template(self, name: str) -> bool:
        """Remove a template."""
        if name in self.templates:
            del self.templates[name]
            _logger.info("Removed configuration template: %s", name)
            return True
        return False

    def create_from_template(self, template_name: str, overrides: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create configuration from template with optional overrides."""
        template = self.get_template(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found")

        config = template.copy()

        if overrides:
            config.update(overrides)

        return config


class EnvironmentManager:
    """Environment-specific configuration management."""

    def __init__(self, config_dir: str = "config/brokers"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)

        self.environments: Dict[Environment, Dict[str, Any]] = {}
        self.current_environment = Environment.DEVELOPMENT

        self._load_environment_configs()

    def _load_environment_configs(self):
        """Load environment-specific configurations."""
        for env in Environment:
            config_file = self.config_dir / f"{env.value}.json"
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        self.environments[env] = json.load(f)
                    _logger.info("Loaded %s environment configuration", env.value)
                except Exception as e:
                    _logger.exception("Failed to load %s config:", env.value)

    def set_environment(self, environment: Environment):
        """Set the current environment."""
        self.current_environment = environment
        _logger.info("Set environment to %s", environment.value)

    def get_environment_config(self, environment: Environment = None) -> Dict[str, Any]:
        """Get configuration for an environment."""
        env = environment or self.current_environment
        return self.environments.get(env, {})

    def save_environment_config(self, environment: Environment, config: Dict[str, Any]):
        """Save configuration for an environment."""
        config_file = self.config_dir / f"{environment.value}.json"

        try:
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2, default=str)

            self.environments[environment] = config
            _logger.info("Saved %s environment configuration", environment.value)

        except Exception as e:
            _logger.exception("Failed to save %s config:", environment.value)
            raise

    def apply_environment_overrides(self, base_config: Dict[str, Any],
                                  environment: Environment = None) -> Dict[str, Any]:
        """Apply environment-specific overrides to base configuration."""
        env = environment or self.current_environment
        env_config = self.get_environment_config(env)

        if not env_config:
            return base_config

        # Deep merge configurations
        result = base_config.copy()

        def deep_merge(target: Dict, source: Dict):
            for key, value in source.items():
                if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                    deep_merge(target[key], value)
                else:
                    target[key] = value

        deep_merge(result, env_config)
        return result


class ConfigManager:
    """
    Comprehensive configuration management system.

    Features:
    - Configuration templates and presets
    - Environment-specific configurations
    - Configuration validation and normalization
    - Hot-reloading and dynamic updates
    - Configuration versioning and rollback
    - Secure credential management
    """

    def __init__(self, config_dir: str = "config/brokers"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.template_manager = ConfigTemplate()
        self.environment_manager = EnvironmentManager(str(self.config_dir))

        # Configuration storage
        self.configurations: Dict[str, Dict[str, Any]] = {}
        self.config_versions: Dict[str, List[ConfigVersion]] = {}
        self.config_watchers: Dict[str, Any] = {}

        # Load existing configurations
        self._load_configurations()

        _logger.info("Configuration manager initialized")

    def _load_configurations(self):
        """Load existing configurations from disk."""
        try:
            for config_file in self.config_dir.glob("*.json"):
                if config_file.stem in ['development', 'staging', 'production', 'testing']:
                    continue  # Skip environment files

                try:
                    with open(config_file, 'r') as f:
                        config = json.load(f)

                    config_name = config_file.stem
                    self.configurations[config_name] = config
                    _logger.debug("Loaded configuration: %s", config_name)

                except Exception as e:
                    _logger.exception("Failed to load config %s:", config_file)

        except Exception as e:
            _logger.exception("Error loading configurations:")

    def create_configuration(self, name: str, config: Dict[str, Any],
                           validate: bool = True) -> bool:
        """
        Create a new configuration.

        Args:
            name: Configuration name
            config: Configuration dictionary
            validate: Whether to validate the configuration

        Returns:
            True if configuration was created successfully
        """
        try:
            if validate:
                # Validate configuration
                validated_config = validate_and_create_broker_config(config)
            else:
                validated_config = config

            # Store configuration
            self.configurations[name] = validated_config

            # Save to disk
            self._save_configuration(name, validated_config)

            # Create version entry
            self._create_version(name, validated_config, "Initial configuration")

            _logger.info("Created configuration: %s", name)
            return True

        except Exception as e:
            _logger.exception("Failed to create configuration %s:", name)
            return False

    def update_configuration(self, name: str, config: Dict[str, Any],
                           description: str = "Configuration update") -> bool:
        """
        Update an existing configuration.

        Args:
            name: Configuration name
            config: Updated configuration dictionary
            description: Update description

        Returns:
            True if configuration was updated successfully
        """
        try:
            # Validate configuration
            validated_config = validate_and_create_broker_config(config)

            # Store configuration
            self.configurations[name] = validated_config

            # Save to disk
            self._save_configuration(name, validated_config)

            # Create version entry
            self._create_version(name, validated_config, description)

            _logger.info("Updated configuration: %s", name)
            return True

        except Exception as e:
            _logger.exception("Failed to update configuration %s:", name)
            return False

    def get_configuration(self, name: str, environment: Environment = None) -> Optional[Dict[str, Any]]:
        """
        Get a configuration with optional environment overrides.

        Args:
            name: Configuration name
            environment: Environment for overrides

        Returns:
            Configuration dictionary or None if not found
        """
        base_config = self.configurations.get(name)
        if not base_config:
            return None

        # Apply environment overrides if specified
        if environment:
            return self.environment_manager.apply_environment_overrides(base_config, environment)

        return base_config.copy()

    def delete_configuration(self, name: str) -> bool:
        """
        Delete a configuration.

        Args:
            name: Configuration name

        Returns:
            True if configuration was deleted successfully
        """
        try:
            if name in self.configurations:
                del self.configurations[name]

            # Delete from disk
            config_file = self.config_dir / f"{name}.json"
            if config_file.exists():
                config_file.unlink()

            # Delete versions
            if name in self.config_versions:
                del self.config_versions[name]

            _logger.info("Deleted configuration: %s", name)
            return True

        except Exception as e:
            _logger.exception("Failed to delete configuration %s:", name)
            return False

    def list_configurations(self) -> List[Dict[str, Any]]:
        """List all configurations with metadata."""
        configs = []

        for name, config in self.configurations.items():
            versions = self.config_versions.get(name, [])
            latest_version = versions[-1] if versions else None

            config_info = {
                'name': name,
                'type': config.get('type', 'unknown'),
                'trading_mode': config.get('trading_mode', 'unknown'),
                'version_count': len(versions),
                'latest_version': latest_version.version if latest_version else None,
                'last_updated': latest_version.timestamp.isoformat() if latest_version else None
            }

            configs.append(config_info)

        return configs

    def create_from_template(self, name: str, template_name: str,
                           overrides: Dict[str, Any] = None) -> bool:
        """
        Create configuration from template.

        Args:
            name: New configuration name
            template_name: Template to use
            overrides: Optional configuration overrides

        Returns:
            True if configuration was created successfully
        """
        try:
            config = self.template_manager.create_from_template(template_name, overrides)
            return self.create_configuration(name, config)

        except Exception as e:
            _logger.exception("Failed to create configuration from template:")
            return False

    def get_configuration_versions(self, name: str) -> List[ConfigVersion]:
        """Get version history for a configuration."""
        return self.config_versions.get(name, [])

    def rollback_configuration(self, name: str, version: str) -> bool:
        """
        Rollback configuration to a specific version.

        Args:
            name: Configuration name
            version: Version to rollback to

        Returns:
            True if rollback was successful
        """
        try:
            versions = self.config_versions.get(name, [])
            target_version = None

            for v in versions:
                if v.version == version:
                    target_version = v
                    break

            if not target_version:
                _logger.error("Version %s not found for configuration %s", version, name)
                return False

            # Load configuration from version file
            version_file = self.config_dir / "versions" / f"{name}_{version}.json"
            if not version_file.exists():
                _logger.error("Version file not found: %s", version_file)
                return False

            with open(version_file, 'r') as f:
                config = json.load(f)

            # Update current configuration
            return self.update_configuration(name, config, f"Rollback to version {version}")

        except Exception as e:
            _logger.exception("Failed to rollback configuration %s to %s:", name, version)
            return False

    def _save_configuration(self, name: str, config: Dict[str, Any]):
        """Save configuration to disk."""
        config_file = self.config_dir / f"{name}.json"

        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2, default=str)

    def _create_version(self, name: str, config: Dict[str, Any], description: str):
        """Create a new version entry for a configuration."""
        # Calculate config hash
        config_str = json.dumps(config, sort_keys=True, default=str)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]

        # Create version
        timestamp = datetime.now(timezone.utc)
        version_num = len(self.config_versions.get(name, [])) + 1
        version = f"v{version_num:03d}"

        config_version = ConfigVersion(
            version=version,
            timestamp=timestamp,
            description=description,
            config_hash=config_hash
        )

        # Store version
        if name not in self.config_versions:
            self.config_versions[name] = []

        self.config_versions[name].append(config_version)

        # Save version file
        versions_dir = self.config_dir / "versions"
        versions_dir.mkdir(exist_ok=True)

        version_file = versions_dir / f"{name}_{version}.json"
        with open(version_file, 'w') as f:
            json.dump(config, f, indent=2, default=str)

        # Save version metadata
        metadata_file = versions_dir / f"{name}_versions.json"
        versions_data = [v.to_dict() for v in self.config_versions[name]]
        with open(metadata_file, 'w') as f:
            json.dump(versions_data, f, indent=2, default=str)

    def export_configuration(self, name: str, format: ConfigFormat = ConfigFormat.JSON) -> str:
        """
        Export configuration in specified format.

        Args:
            name: Configuration name
            format: Export format

        Returns:
            Configuration as string in specified format
        """
        config = self.get_configuration(name)
        if not config:
            raise ValueError(f"Configuration '{name}' not found")

        if format == ConfigFormat.JSON:
            return json.dumps(config, indent=2, default=str)
        elif format == ConfigFormat.YAML:
            return yaml.dump(config, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def import_configuration(self, name: str, config_str: str,
                           format: ConfigFormat = ConfigFormat.JSON) -> bool:
        """
        Import configuration from string.

        Args:
            name: Configuration name
            config_str: Configuration as string
            format: Configuration format

        Returns:
            True if import was successful
        """
        try:
            if format == ConfigFormat.JSON:
                config = json.loads(config_str)
            elif format == ConfigFormat.YAML:
                config = yaml.safe_load(config_str)
            else:
                raise ValueError(f"Unsupported format: {format}")

            return self.create_configuration(name, config)

        except Exception as e:
            _logger.exception("Failed to import configuration:")
            return False

    def get_template_manager(self) -> ConfigTemplate:
        """Get the template manager."""
        return self.template_manager

    def get_environment_manager(self) -> EnvironmentManager:
        """Get the environment manager."""
        return self.environment_manager