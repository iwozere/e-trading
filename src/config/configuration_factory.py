import json
import logging
from pathlib import Path
from typing import Any, Dict, Union

from src.trading.services.schema_validator import validate_bot_configuration

logger = logging.getLogger(__name__)


class ConfigurationFactory:
    """
    Factory for assembling and validating trading bot configurations.

    Unified flat-config loader:
    1. Load bot configuration JSON (file or dict)
    2. Resolve optional instance references
    3. Validate using the single runtime schema validator
    """

    def __init__(self, root_path: str | None = None):
        """
        Initialize the factory.

        Args:
            root_path: Root of the project (defaults to finding it relative to this file)
        """
        if root_path:
            self.root_path = Path(root_path)
        else:
            # Assuming src/config/configuration_factory.py -> ../../..
            self.root_path = Path(__file__).parent.parent.parent.resolve()

        self.contracts_path = self.root_path / "config" / "contracts"

    def load_manifest(self, manifest_source: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Load a bot manifest, resolve all references, and validatate.

        Args:
            manifest_source: Path to JSON manifest file OR dictionary object.

        Returns:
            Fully hydrated and validated configuration dictionary.
        """
        # 1. Load raw manifest
        if isinstance(manifest_source, (str, Path)):
            with open(manifest_source) as f:
                manifest = json.load(f)
        else:
            manifest = manifest_source.copy()

        # Unified format only: flat configuration, no modules container.
        if "modules" in manifest:
            raise ValueError(
                "Unsupported configuration format: 'modules' is not allowed. "
                "Use flat top-level keys: broker, strategy, risk_management, notifications, data, trading."
            )

        # 2. Resolve references (hydrating the config)
        hydrated_config = self._resolve_references(manifest)

        # 3. Apply Overrides (if any)
        if "overrides" in hydrated_config and isinstance(hydrated_config["overrides"], dict):
            overrides = hydrated_config.pop("overrides")
            for k, v in overrides.items():
                if k == "initial_balance" and "broker" in hydrated_config:
                    hydrated_config["broker"]["cash"] = v
                else:
                    hydrated_config[k] = v

        # 4. Validate via single runtime schema
        is_valid, errors, _warnings = validate_bot_configuration(hydrated_config)
        if not is_valid:
            raise ValueError("; ".join(errors))

        return hydrated_config

    def _resolve_references(self, config_node: Any) -> Any:
        """
        Recursively walk the config. If a string value looks like a path to an instance,
        load it and merge it in.
        """
        if isinstance(config_node, dict):
            resolved_dict = {}
            for key, value in config_node.items():
                resolved_dict[key] = self._resolve_references(value)
            return resolved_dict

        elif isinstance(config_node, list):
            return [self._resolve_references(item) for item in config_node]

        elif isinstance(config_node, str):
            # Check if string is a reference
            if self._is_reference(config_node):
                return self._load_instance(config_node)
            return config_node

        else:
            return config_node

    def _is_reference(self, value: str) -> bool:
        """Heuristic to check if a string is a file reference."""
        if value.startswith("instances/") or value.startswith("config/contracts/instances/"):
            return True
        return False

    def _load_instance(self, ref_path: str) -> Dict[str, Any]:
        """Load a JSON instance from a path."""
        # Normalize path
        # If starts with "instances/", map to self.contracts_path / ...
        if ref_path.startswith("instances/"):
            full_path = self.contracts_path / ref_path
        elif ref_path.startswith("config/"):
            full_path = self.root_path / ref_path
        else:
            full_path = Path(ref_path)

        if not full_path.suffix:
            full_path = full_path.with_suffix(".json")

        if not full_path.exists():
            raise FileNotFoundError(f"Module reference not found: {full_path}")

        with open(full_path) as f:
            data = json.load(f)

        # Recursively resolve any nested references inside the instance
        return self._resolve_references(data)

    def validate(self, config: Dict[str, Any]) -> bool:
        """Validate config using the single runtime schema validator."""
        is_valid, errors, _warnings = validate_bot_configuration(config)
        if not is_valid:
            raise ValueError("; ".join(errors))
        return True


# Singleton instance for easy import
config_factory = ConfigurationFactory()
