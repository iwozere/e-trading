"""
Configuration Migration Script
=============================

This script helps migrate from the old scattered configuration system
to the new centralized configuration management system.

Features:
- Automatic discovery of existing configurations
- Migration to new schema format
- Environment-specific organization
- Validation of migrated configurations
"""

import os
import json
import yaml
import shutil
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import logging

from .config_manager import ConfigManager


class ConfigMigrator:
    """
    Configuration migration utility.
    
    Migrates from old scattered configuration files to the new
    centralized configuration management system.
    """
    
    def __init__(self, old_config_dir: str = "config", new_config_dir: str = "config_new"):
        """
        Initialize the config migrator.
        
        Args:
            old_config_dir: Directory containing old configuration files
            new_config_dir: Directory for new centralized configurations
        """
        self.old_config_dir = Path(old_config_dir)
        self.new_config_dir = Path(new_config_dir)
        self.logger = logging.getLogger(__name__)
        
        # Ensure directories exist
        self.new_config_dir.mkdir(parents=True, exist_ok=True)
        
        # Migration statistics
        self.migration_stats = {
            "total_files": 0,
            "migrated_files": 0,
            "failed_files": 0,
            "errors": []
        }
    
    def discover_old_configs(self) -> Dict[str, List[Path]]:
        """
        Discover all configuration files in the old structure.
        
        Returns:
            Dictionary mapping config types to file paths
        """
        configs = {
            "trading": [],
            "optimizer": [],
            "data": [],
            "plotter": [],
            "unknown": []
        }
        
        if not self.old_config_dir.exists():
            self.logger.warning(f"Old config directory not found: {self.old_config_dir}")
            return configs
        
        # Scan for configuration files
        for config_file in self.old_config_dir.rglob("*"):
            if config_file.is_file() and config_file.suffix.lower() in ['.json', '.yaml', '.yml']:
                config_type = self._detect_config_type(config_file)
                configs[config_type].append(config_file)
                self.migration_stats["total_files"] += 1
        
        self.logger.info(f"Discovered {self.migration_stats['total_files']} configuration files")
        for config_type, files in configs.items():
            if files:
                self.logger.info(f"  {config_type}: {len(files)} files")
        
        return configs
    
    def _detect_config_type(self, config_path: Path) -> str:
        """Detect the type of configuration file"""
        try:
            # Load the configuration
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
            else:
                with open(config_path, 'r') as f:
                    config = json.load(f)
            
            # Determine type based on content
            if 'bot_id' in config or 'broker' in config:
                return "trading"
            elif 'optimizer_type' in config or 'n_trials' in config:
                return "optimizer"
            elif 'data_source' in config:
                return "data"
            elif 'plot' in config or 'visualization' in config:
                return "plotter"
            else:
                return "unknown"
                
        except Exception as e:
            self.logger.warning(f"Error detecting config type for {config_path}: {e}")
            return "unknown"
    
    def migrate_configs(self, environment: str = "development") -> bool:
        """
        Migrate all discovered configurations.
        
        Args:
            environment: Target environment for migrated configs
            
        Returns:
            True if migration was successful
        """
        self.logger.info("Starting configuration migration...")
        
        # Discover old configurations
        old_configs = self.discover_old_configs()
        
        # Create environment directory
        env_dir = self.new_config_dir / environment
        env_dir.mkdir(parents=True, exist_ok=True)
        
        # Migrate each type of configuration
        for config_type, config_files in old_configs.items():
            if config_files:
                self.logger.info(f"Migrating {config_type} configurations...")
                for config_file in config_files:
                    try:
                        self._migrate_single_config(config_file, config_type, env_dir)
                        self.migration_stats["migrated_files"] += 1
                    except Exception as e:
                        self.migration_stats["failed_files"] += 1
                        self.migration_stats["errors"].append({
                            "file": str(config_file),
                            "error": str(e)
                        })
                        self.logger.error(f"Failed to migrate {config_file}: {e}")
        
        # Generate migration report
        self._generate_migration_report(env_dir)
        
        return self.migration_stats["failed_files"] == 0
    
    def _migrate_single_config(self, config_path: Path, config_type: str, env_dir: Path):
        """Migrate a single configuration file"""
        # Load old configuration
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path, 'r') as f:
                old_config = yaml.safe_load(f)
        else:
            with open(config_path, 'r') as f:
                old_config = json.load(f)
        
        # Transform configuration based on type
        if config_type == "trading":
            new_config = self._transform_trading_config(old_config, config_path)
        elif config_type == "optimizer":
            new_config = self._transform_optimizer_config(old_config, config_path)
        elif config_type == "data":
            new_config = self._transform_data_config(old_config, config_path)
        else:
            # For unknown types, just copy with metadata
            new_config = self._transform_generic_config(old_config, config_path)
        
        # Add migration metadata
        new_config.update({
            "migrated_from": str(config_path),
            "migrated_at": datetime.now().isoformat(),
            "original_config_type": config_type
        })
        
        # Generate new filename
        new_filename = self._generate_new_filename(config_path, config_type)
        new_config_path = env_dir / new_filename
        
        # Save new configuration
        with open(new_config_path, 'w') as f:
            json.dump(new_config, f, indent=2, default=str)
        
        self.logger.debug(f"Migrated {config_path} -> {new_config_path}")
    
    def _transform_trading_config(self, old_config: Dict[str, Any], config_path: Path) -> Dict[str, Any]:
        """Transform old trading configuration to new format"""
        new_config = {
            "environment": "development",
            "version": "1.0.0",
            "description": old_config.get("description", f"Migrated from {config_path.name}"),
            "bot_id": old_config.get("bot_id", config_path.stem),
        }
        
        # Transform broker configuration
        if "broker" in old_config:
            new_config["broker"] = old_config["broker"]
        elif "type" in old_config:
            new_config["broker"] = {
                "type": old_config["type"],
                "initial_balance": old_config.get("initial_balance", 1000.0),
                "commission": old_config.get("commission", 0.001)
            }
        
        # Transform trading parameters
        if "trading" in old_config:
            new_config["trading"] = old_config["trading"]
        else:
            new_config["trading"] = {
                "symbol": old_config.get("trading_pair", "BTCUSDT"),
                "position_size": old_config.get("position_size", 0.1),
                "max_positions": old_config.get("max_positions", 1),
                "max_drawdown_pct": old_config.get("max_drawdown_pct", 20.0),
                "max_exposure": old_config.get("max_exposure", 1.0)
            }
        
        # Transform data configuration
        if "data" in old_config:
            new_config["data"] = old_config["data"]
        else:
            new_config["data"] = {
                "data_source": "binance",
                "symbol": old_config.get("trading_pair", "BTCUSDT"),
                "interval": old_config.get("interval", "1h"),
                "lookback_bars": old_config.get("lookback_bars", 1000),
                "retry_interval": old_config.get("retry_interval", 60)
            }
        
        # Transform strategy configuration
        if "strategy" in old_config:
            new_config["strategy"] = old_config["strategy"]
        elif "strategy_params" in old_config:
            new_config["strategy"] = {
                "type": "custom",
                "entry_logic": {
                    "name": old_config.get("entry_logic", "RSIBBVolumeEntryMixin"),
                    "params": old_config.get("strategy_params", {})
                },
                "exit_logic": {
                    "name": old_config.get("exit_logic", "RSIBBExitMixin"),
                    "params": old_config.get("strategy_params", {})
                }
            }
        
        # Add default sections
        new_config.update({
            "risk_management": old_config.get("risk_management", {}),
            "logging": old_config.get("logging", {}),
            "scheduling": old_config.get("scheduling", {}),
            "performance": old_config.get("performance", {}),
            "notifications": old_config.get("notifications", {})
        })
        
        return new_config
    
    def _transform_optimizer_config(self, old_config: Dict[str, Any], config_path: Path) -> Dict[str, Any]:
        """Transform old optimizer configuration to new format"""
        new_config = {
            "environment": "development",
            "version": "1.0.0",
            "description": f"Migrated optimizer config from {config_path.name}",
        }
        
        # Copy optimizer settings
        if "optimizer_settings" in old_config:
            new_config.update(old_config["optimizer_settings"])
        else:
            new_config.update(old_config)
        
        # Ensure required fields
        new_config.setdefault("optimizer_type", "optuna")
        new_config.setdefault("initial_capital", 1000.0)
        new_config.setdefault("commission", 0.001)
        new_config.setdefault("n_trials", 100)
        
        return new_config
    
    def _transform_data_config(self, old_config: Dict[str, Any], config_path: Path) -> Dict[str, Any]:
        """Transform old data configuration to new format"""
        new_config = {
            "environment": "development",
            "version": "1.0.0",
            "description": f"Migrated data config from {config_path.name}",
        }
        
        # Copy data configuration
        new_config.update(old_config)
        
        return new_config
    
    def _transform_generic_config(self, old_config: Dict[str, Any], config_path: Path) -> Dict[str, Any]:
        """Transform generic configuration to new format"""
        return {
            "environment": "development",
            "version": "1.0.0",
            "description": f"Migrated config from {config_path.name}",
            "config_type": "generic",
            "data": old_config
        }
    
    def _generate_new_filename(self, config_path: Path, config_type: str) -> str:
        """Generate new filename for migrated configuration"""
        base_name = config_path.stem
        
        # Add type prefix if not already present
        if not base_name.startswith(config_type):
            base_name = f"{config_type}_{base_name}"
        
        # Add timestamp to avoid conflicts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{base_name}_{timestamp}.json"
    
    def _generate_migration_report(self, env_dir: Path):
        """Generate migration report"""
        report = {
            "migration_summary": {
                "timestamp": datetime.now().isoformat(),
                "old_config_dir": str(self.old_config_dir),
                "new_config_dir": str(self.new_config_dir),
                "environment": env_dir.name
            },
            "statistics": self.migration_stats,
            "migrated_files": []
        }
        
        # List all migrated files
        for config_file in env_dir.glob("*.json"):
            with open(config_file, 'r') as f:
                config = json.load(f)
                report["migrated_files"].append({
                    "file": config_file.name,
                    "original_file": config.get("migrated_from"),
                    "config_type": config.get("original_config_type"),
                    "description": config.get("description")
                })
        
        # Save report
        report_path = self.new_config_dir / "migration_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Migration report saved to: {report_path}")
    
    def validate_migrated_configs(self) -> List[str]:
        """Validate all migrated configurations"""
        errors = []
        
        # Initialize config manager with new directory
        config_manager = ConfigManager(str(self.new_config_dir))
        
        # Validate each configuration
        for config_file in self.new_config_dir.rglob("*.json"):
            if config_file.name == "migration_report.json":
                continue
                
            try:
                is_valid, config_errors, warnings = config_manager.validate_config_file(str(config_file))
                if not is_valid:
                    errors.extend([f"{config_file}: {error}" for error in config_errors])
            except Exception as e:
                errors.append(f"{config_file}: {e}")
        
        return errors
    
    def create_backup(self) -> str:
        """Create backup of old configuration directory"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = f"{self.old_config_dir}_backup_{timestamp}"
        
        if self.old_config_dir.exists():
            shutil.copytree(self.old_config_dir, backup_dir)
            self.logger.info(f"Backup created: {backup_dir}")
            return backup_dir
        else:
            self.logger.warning("No old config directory to backup")
            return ""


def main():
    """Main migration function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate old configuration files to new centralized system")
    parser.add_argument("--old-dir", default="config", help="Old configuration directory")
    parser.add_argument("--new-dir", default="config_new", help="New configuration directory")
    parser.add_argument("--environment", default="development", help="Target environment")
    parser.add_argument("--backup", action="store_true", help="Create backup before migration")
    parser.add_argument("--validate", action="store_true", help="Validate migrated configurations")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create migrator
    migrator = ConfigMigrator(args.old_dir, args.new_dir)
    
    # Create backup if requested
    if args.backup:
        backup_dir = migrator.create_backup()
        if backup_dir:
            print(f"Backup created: {backup_dir}")
    
    # Run migration
    success = migrator.migrate_configs(args.environment)
    
    if success:
        print("Migration completed successfully!")
        
        # Validate if requested
        if args.validate:
            print("Validating migrated configurations...")
            errors = migrator.validate_migrated_configs()
            if errors:
                print("Validation errors found:")
                for error in errors:
                    print(f"  - {error}")
            else:
                print("All configurations validated successfully!")
    else:
        print("Migration completed with errors. Check the migration report for details.")
    
    # Print statistics
    stats = migrator.migration_stats
    print(f"\nMigration Statistics:")
    print(f"  Total files: {stats['total_files']}")
    print(f"  Migrated: {stats['migrated_files']}")
    print(f"  Failed: {stats['failed_files']}")


if __name__ == "__main__":
    main() 