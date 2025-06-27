"""
Configuration Registry
=====================

Registry for tracking and organizing configurations by type, metadata,
and relationships. Provides discovery and query capabilities.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import logging


class ConfigRegistry:
    """
    Registry for managing and discovering configurations.
    
    Features:
    - Configuration indexing by type
    - Metadata tracking
    - Relationship mapping
    - Search and filtering
    """
    
    def __init__(self):
        """Initialize the configuration registry"""
        self.logger = logging.getLogger(__name__)
        
        # Configuration storage by type
        self._configs_by_type: Dict[str, Dict[str, Any]] = {}
        
        # Configuration metadata
        self._metadata: Dict[str, Dict[str, Any]] = {}
        
        # Configuration relationships
        self._relationships: Dict[str, List[str]] = {}
        
        # Configuration tags
        self._tags: Dict[str, List[str]] = {}
    
    def register_config(self, config_id: str, config: Any, config_type: str, 
                       metadata: Dict[str, Any] = None, tags: List[str] = None):
        """
        Register a configuration in the registry.
        
        Args:
            config_id: Unique configuration identifier
            config: Configuration object
            config_type: Type of configuration
            metadata: Additional metadata
            tags: Configuration tags
        """
        # Initialize type storage if needed
        if config_type not in self._configs_by_type:
            self._configs_by_type[config_type] = {}
        
        # Store configuration
        self._configs_by_type[config_type][config_id] = config
        
        # Store metadata
        self._metadata[config_id] = {
            "type": config_type,
            "registered_at": datetime.now().isoformat(),
            "file_path": getattr(config, '_file_path', None),
            "loaded_at": getattr(config, '_loaded_at', None)
        }
        
        if metadata:
            self._metadata[config_id].update(metadata)
        
        # Store tags
        if tags:
            self._tags[config_id] = tags
        
        # Initialize relationships
        self._relationships[config_id] = []
        
        self.logger.debug(f"Registered config: {config_id} (type: {config_type})")
    
    def unregister_config(self, config_id: str):
        """Unregister a configuration from the registry"""
        # Remove from type storage
        for config_type, configs in self._configs_by_type.items():
            if config_id in configs:
                del configs[config_id]
                break
        
        # Remove metadata
        if config_id in self._metadata:
            del self._metadata[config_id]
        
        # Remove relationships
        if config_id in self._relationships:
            del self._relationships[config_id]
        
        # Remove tags
        if config_id in self._tags:
            del self._tags[config_id]
        
        self.logger.debug(f"Unregistered config: {config_id}")
    
    def get_config(self, config_id: str) -> Optional[Any]:
        """Get a configuration by ID"""
        for configs in self._configs_by_type.values():
            if config_id in configs:
                return configs[config_id]
        return None
    
    def get_configs_by_type(self, config_type: str) -> List[Any]:
        """Get all configurations of a specific type"""
        return list(self._configs_by_type.get(config_type, {}).values())
    
    def get_config_ids_by_type(self, config_type: str) -> List[str]:
        """Get all configuration IDs of a specific type"""
        return list(self._configs_by_type.get(config_type, {}).keys())
    
    def get_config_metadata(self, config_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a configuration"""
        return self._metadata.get(config_id)
    
    def get_configs_by_tag(self, tag: str) -> List[str]:
        """Get configuration IDs that have a specific tag"""
        config_ids = []
        for config_id, tags in self._tags.items():
            if tag in tags:
                config_ids.append(config_id)
        return config_ids
    
    def add_relationship(self, config_id: str, related_config_id: str):
        """Add a relationship between two configurations"""
        if config_id not in self._relationships:
            self._relationships[config_id] = []
        
        if related_config_id not in self._relationships[config_id]:
            self._relationships[config_id].append(related_config_id)
    
    def get_related_configs(self, config_id: str) -> List[str]:
        """Get related configuration IDs"""
        return self._relationships.get(config_id, [])
    
    def search_configs(self, query: str, config_type: str = None) -> List[str]:
        """
        Search configurations by query string.
        
        Args:
            query: Search query
            config_type: Optional filter by configuration type
            
        Returns:
            List of matching configuration IDs
        """
        results = []
        query_lower = query.lower()
        
        for config_id, metadata in self._metadata.items():
            # Filter by type if specified
            if config_type and metadata.get('type') != config_type:
                continue
            
            # Search in config_id
            if query_lower in config_id.lower():
                results.append(config_id)
                continue
            
            # Search in description
            config = self.get_config(config_id)
            if config and hasattr(config, 'description') and config.description:
                if query_lower in config.description.lower():
                    results.append(config_id)
                    continue
            
            # Search in tags
            tags = self._tags.get(config_id, [])
            for tag in tags:
                if query_lower in tag.lower():
                    results.append(config_id)
                    break
        
        return list(set(results))  # Remove duplicates
    
    def list_configs(self) -> Dict[str, List[str]]:
        """List all configurations by type"""
        return {
            config_type: list(configs.keys())
            for config_type, configs in self._configs_by_type.items()
        }
    
    def get_config_stats(self) -> Dict[str, Any]:
        """Get statistics about registered configurations"""
        stats = {
            "total_configs": sum(len(configs) for configs in self._configs_by_type.values()),
            "configs_by_type": {
                config_type: len(configs)
                for config_type, configs in self._configs_by_type.items()
            },
            "total_relationships": sum(len(rels) for rels in self._relationships.values()),
            "total_tags": len(set(tag for tags in self._tags.values() for tag in tags))
        }
        return stats
    
    def export_registry(self) -> Dict[str, Any]:
        """Export the entire registry as a dictionary"""
        return {
            "configs_by_type": {
                config_type: {
                    config_id: {
                        "metadata": self._metadata.get(config_id, {}),
                        "tags": self._tags.get(config_id, []),
                        "relationships": self._relationships.get(config_id, [])
                    }
                    for config_id in configs.keys()
                }
                for config_type, configs in self._configs_by_type.items()
            },
            "stats": self.get_config_stats()
        }
    
    def clear(self):
        """Clear all registered configurations"""
        self._configs_by_type.clear()
        self._metadata.clear()
        self._relationships.clear()
        self._tags.clear()
        self.logger.info("Configuration registry cleared")
    
    def validate_registry(self) -> List[str]:
        """
        Validate the registry for consistency.
        
        Returns:
            List of validation errors
        """
        errors = []
        
        # Check for orphaned metadata
        for config_id in self._metadata:
            if not self.get_config(config_id):
                errors.append(f"Orphaned metadata for config: {config_id}")
        
        # Check for orphaned tags
        for config_id in self._tags:
            if not self.get_config(config_id):
                errors.append(f"Orphaned tags for config: {config_id}")
        
        # Check for orphaned relationships
        for config_id in self._relationships:
            if not self.get_config(config_id):
                errors.append(f"Orphaned relationships for config: {config_id}")
        
        # Check for invalid relationships
        for config_id, related_ids in self._relationships.items():
            for related_id in related_ids:
                if not self.get_config(related_id):
                    errors.append(f"Invalid relationship: {config_id} -> {related_id}")
        
        return errors 