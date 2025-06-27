"""
Configuration Manager
====================

Main configuration manager that handles:
- Loading configurations from files
- Environment-specific configuration management
- Schema validation
- Configuration hot-reloading
- Configuration templates
"""

import os
import json
import yaml
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from .schemas import (
    ConfigSchema,
    TradingConfig,
    OptimizerConfig,
    DataConfig,
)
from .registry import ConfigRegistry
from .templates import ConfigTemplates


class ConfigFileHandler(FileSystemEventHandler):
    """File system event handler for configuration hot-reloading"""
    
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
    
    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith(('.json', '.yaml', '.yml')):
            self.logger.info(f"Configuration file modified: {event.src_path}")
            self.config_manager.reload_config(event.src_path)


class ConfigManager:
    """
    Main configuration manager for the trading platform.
    
    Features:
    - Environment-specific configuration loading
    - Schema validation with detailed error messages
    - Hot-reload support for development
    - Configuration templates
    - Configuration registry and discovery
    """
    
    def __init__(self, config_dir: str = "config", environment: str = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_dir: Base configuration directory
            environment: Environment to load (dev/staging/prod)
        """
        self.config_dir = Path(config_dir)
        self.environment = environment or os.getenv("TRADING_ENV", "development")
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.registry = ConfigRegistry()
        self.templates = ConfigTemplates()
        
        # Configuration cache
        self._config_cache: Dict[str, Any] = {}
        self._config_files: Dict[str, str] = {}
        
        # Hot-reload support
        self._observer = None
        self._hot_reload_enabled = False
        
        # Ensure config directory exists
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Load environment-specific configurations
        self._load_environment_configs()
    
    def _load_environment_configs(self):
        """Load environment-specific configuration files"""
        env_dir = self.config_dir / self.environment
        if env_dir.exists():
            self.logger.info(f"Loading environment-specific configs from: {env_dir}")
            for config_file in env_dir.rglob("*.json"):
                self._load_config_file(config_file)
            for config_file in env_dir.rglob("*.yaml"):
                self._load_config_file(config_file)
            for config_file in env_dir.rglob("*.yml"):
                self._load_config_file(config_file)
    
    def _load_config_file(self, config_path: Path) -> Optional[Dict[str, Any]]:
        """Load a single configuration file"""
        try:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
            else:
                with open(config_path, 'r') as f:
                    config = json.load(f)
            
            # Add metadata
            config['_file_path'] = str(config_path)
            config['_loaded_at'] = datetime.now().isoformat()
            
            # Validate and cache
            self._validate_and_cache_config(config, config_path)
            
            return config
            
        except Exception as e:
            self.logger.error(f"Error loading config file {config_path}: {e}")
            return None
    
    def _validate_and_cache_config(self, config: Dict[str, Any], config_path: Path):
        """Validate configuration and add to cache"""
        try:
            # Determine config type and validate
            config_type = self._detect_config_type(config)
            
            if config_type == "trading":
                validated_config = TradingConfig(**config)
            elif config_type == "optimizer":
                validated_config = OptimizerConfig(**config)
            elif config_type == "data":
                validated_config = DataConfig(**config)
            else:
                # Generic config validation
                validated_config = ConfigSchema(**config)
            
            # Cache the validated config
            config_id = self._generate_config_id(config_path)
            self._config_cache[config_id] = validated_config
            self._config_files[config_id] = str(config_path)
            
            # Register in registry
            self.registry.register_config(config_id, validated_config, config_type)
            
            self.logger.info(f"Loaded and validated config: {config_id}")
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed for {config_path}: {e}")
            raise
    
    def _detect_config_type(self, config: Dict[str, Any]) -> str:
        """Detect configuration type based on content"""
        if 'bot_id' in config and 'broker' in config:
            return "trading"
        elif 'optimizer_type' in config or 'n_trials' in config:
            return "optimizer"
        elif 'data_source' in config and 'symbol' in config:
            return "data"
        else:
            return "generic"
    
    def _generate_config_id(self, config_path: Path) -> str:
        """Generate a unique configuration ID"""
        # Use relative path from config directory
        relative_path = config_path.relative_to(self.config_dir)
        return str(relative_path).replace('/', '_').replace('\\', '_').replace('.', '_')
    
    def get_config(self, config_id: str) -> Optional[Any]:
        """Get a configuration by ID"""
        return self._config_cache.get(config_id)
    
    def get_config_by_type(self, config_type: str) -> List[Any]:
        """Get all configurations of a specific type"""
        return self.registry.get_configs_by_type(config_type)
    
    def get_trading_configs(self) -> List[TradingConfig]:
        """Get all trading configurations"""
        return self.get_config_by_type("trading")
    
    def get_optimizer_configs(self) -> List[OptimizerConfig]:
        """Get all optimizer configurations"""
        return self.get_config_by_type("optimizer")
    
    def get_data_configs(self) -> List[DataConfig]:
        """Get all data configurations"""
        return self.get_config_by_type("data")
    
    def create_config(self, config_type: str, **kwargs) -> Any:
        """Create a new configuration using templates"""
        template = self.templates.get_template(config_type)
        if not template:
            raise ValueError(f"No template found for config type: {config_type}")
        
        # Merge template with provided kwargs
        config_data = template.copy()
        config_data.update(kwargs)
        
        # Validate the configuration
        if config_type == "trading":
            return TradingConfig(**config_data)
        elif config_type == "optimizer":
            return OptimizerConfig(**config_data)
        elif config_type == "data":
            return DataConfig(**config_data)
        else:
            return ConfigSchema(**config_data)
    
    def save_config(self, config: Any, filename: str = None) -> str:
        """Save a configuration to file"""
        if not filename:
            if hasattr(config, 'bot_id'):
                filename = f"{config.bot_id}.json"
            else:
                filename = f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        config_path = self.config_dir / self.environment / filename
        
        # Ensure directory exists
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dict and save
        config_dict = config.dict()
        config_dict['updated_at'] = datetime.now().isoformat()
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
        
        # Reload the config
        self._load_config_file(config_path)
        
        self.logger.info(f"Saved configuration: {config_path}")
        return str(config_path)
    
    def delete_config(self, config_id: str) -> bool:
        """Delete a configuration"""
        if config_id in self._config_files:
            config_path = Path(self._config_files[config_id])
            try:
                config_path.unlink()
                del self._config_cache[config_id]
                del self._config_files[config_id]
                self.registry.unregister_config(config_id)
                self.logger.info(f"Deleted configuration: {config_id}")
                return True
            except Exception as e:
                self.logger.error(f"Error deleting config {config_id}: {e}")
                return False
        return False
    
    def reload_config(self, config_path: str = None):
        """Reload configuration(s)"""
        if config_path:
            # Reload specific file
            self._load_config_file(Path(config_path))
        else:
            # Reload all configs
            self._config_cache.clear()
            self._config_files.clear()
            self.registry.clear()
            self._load_environment_configs()
    
    def enable_hot_reload(self, enabled: bool = True):
        """Enable/disable hot-reload for configuration files"""
        if enabled and not self._hot_reload_enabled:
            self._observer = Observer()
            handler = ConfigFileHandler(self)
            self._observer.schedule(handler, str(self.config_dir), recursive=True)
            self._observer.start()
            self._hot_reload_enabled = True
            self.logger.info("Hot-reload enabled for configuration files")
        
        elif not enabled and self._hot_reload_enabled:
            if self._observer:
                self._observer.stop()
                self._observer.join()
            self._hot_reload_enabled = False
            self.logger.info("Hot-reload disabled")
    
    def get_environment_config(self, key: str, default: Any = None) -> Any:
        """Get environment-specific configuration value"""
        env_config_file = self.config_dir / f"{self.environment}.json"
        
        if env_config_file.exists():
            with open(env_config_file, 'r') as f:
                env_config = json.load(f)
                return env_config.get(key, default)
        
        return default
    
    def list_configs(self) -> Dict[str, List[str]]:
        """List all available configurations by type"""
        return self.registry.list_configs()
    
    def validate_config_file(self, config_path: str) -> tuple[bool, List[str]]:
        """Validate a configuration file"""
        errors = []
        try:
            config = self._load_config_file(Path(config_path))
            if config is None:
                errors.append("Failed to load configuration file")
                return False, errors
            
            # Validation is done in _validate_and_cache_config
            return True, []
            
        except Exception as e:
            errors.append(str(e))
            return False, errors
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of all configurations"""
        summary = {
            "environment": self.environment,
            "config_dir": str(self.config_dir),
            "total_configs": len(self._config_cache),
            "configs_by_type": self.list_configs(),
            "hot_reload_enabled": self._hot_reload_enabled
        }
        return summary
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.enable_hot_reload(False) 