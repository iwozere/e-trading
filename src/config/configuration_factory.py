import json
import yaml
import os
from pathlib import Path
from typing import Dict, Any, Union, Optional
import logging
import jsonschema

# Configure logger
logger = logging.getLogger(__name__)

class ConfigurationFactory:
    """
    Factory for assembling and validating trading bot configurations.

    Implements the "Modular Architecture":
    1.  Loads a Manifest (Bot Config)
    2.  Resolves "Module References" (loading JSONs from config/contracts/instances)
    3.  Validates the result against YAML Contracts (config/contracts/*.yaml)
    """

    def __init__(self, root_path: str = None):
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
        self.instances_path = self.contracts_path / "instances"
        self.schemas = {}
        self._load_schemas()

    def _load_schemas(self):
        """Load all YAML contracts/schemas into memory."""
        try:
            schema_files = {
                "manifest": "trading_bot.yaml",
                "broker": "broker.yaml",
                "strategy": "strategy.yaml",
                "risk": "risk-management.yaml",
                "notifications": "notifications.yaml",
                "strategy_risk": "strategy-risk.yaml"
            }

            for key, filename in schema_files.items():
                schema_path = self.contracts_path / filename
                if schema_path.exists():
                    with open(schema_path, 'r') as f:
                        self.schemas[key] = yaml.safe_load(f)
                else:
                    logger.warning(f"Schema not found: {schema_path}")
        except Exception as e:
            logger.error(f"Failed to load schemas: {e}")

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
            with open(manifest_source, 'r') as f:
                manifest = json.load(f)
        else:
            manifest = manifest_source.copy()

        # Validate the raw manifest structure itself if schema exists
        if "manifest" in self.schemas:
            try:
                jsonschema.validate(instance=manifest, schema=self.schemas["manifest"])
            except jsonschema.ValidationError as e:
                logger.error(f"Manifest Structure Error: {e.message}")
                raise

        # 2. Resolve references (hydrating the config)
        hydrated_config = self._resolve_references(manifest)

        # 3. Apply Overrides (if any)
        # TODO: Implement deep merge override logic if needed
        # For now, simple top-level overrides are handled by the manifest structure usually

        # 4. Validate
        self.validate(hydrated_config)

        return hydrated_config

    def _resolve_references(self, config_node: Any) -> Any:
        """
        Recursively walk the config. If a string value looks like a path to an instance,
        load it and merge it in.
        """
        if isinstance(config_node, dict):
            # Special case: check if this dict IS a reference pointer?
            # Or usually references are values like "broker": "instances/brokers/binance"

            resolved_dict = {}
            for key, value in config_node.items():
                if key == "modules" and isinstance(value, dict):
                    # "modules": { "broker": "..." } -> Lift these up or resolve them
                    # The architectural proposal said:
                    # "modules": { "broker": "instances/..." }
                    # expected result: "broker": { ... content ... }
                    for module_type, module_ref in value.items():
                        resolved_module = self._load_instance(module_ref)

                        # Auto-unwrap if the instance is wrapped in its own key matching the module type
                        # e.g. module_type="broker", resolved_module={"broker": {...}} -> unwrap to {...}
                        if isinstance(resolved_module, dict) and len(resolved_module) == 1 and module_type in resolved_module:
                             resolved_dict[module_type] = resolved_module[module_type]
                        else:
                             resolved_dict[module_type] = resolved_module
                else:
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
        # We look for "instances/" or specific path patterns
        # Or maybe it's an absolute path?
        # Let's support relative to config/contracts/instances or absolute
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

        with open(full_path, 'r') as f:
            data = json.load(f)

        # Recursively resolve any nested references inside the instance
        return self._resolve_references(data)

    def validate(self, config: Dict[str, Any]) -> bool:
        """
        Validate the full config against known schemas.
        """
        # Validate Broker
        if "broker" in config and "broker" in self.schemas:
            try:
                jsonschema.validate(instance=config, schema=self.schemas["broker"])
            except jsonschema.ValidationError as e:
                logger.error(f"Broker Validation Error: {e.message}")
                # We might want to raise or collect errors. For now log.
                raise

        # Validate Strategy
        if "strategy" in config:
             # Strategy schema expects { "strategy": { ... } }
             # Our config has { "strategy": ... } so we validate the root config against the schema
             # assuming schema structure matches root structure.
             if "strategy" in self.schemas:
                 try:
                    jsonschema.validate(instance=config, schema=self.schemas["strategy"])
                 except jsonschema.ValidationError as e:
                    logger.error(f"Strategy Validation Error: {e.message}")
                    raise

        # Validate Risk
        if "risk_management" in config and "risk" in self.schemas:
             try:
                jsonschema.validate(instance=config, schema=self.schemas["risk"])
             except jsonschema.ValidationError as e:
                logger.error(f"Risk Management Validation Error: {e.message}")
                raise

        return True

# Singleton instance for easy import
config_factory = ConfigurationFactory()
