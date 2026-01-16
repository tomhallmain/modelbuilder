"""
Configuration management for Model Builder.

This module handles loading and managing configuration from YAML files
and command-line arguments.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class Config:
    """
    Configuration manager for Model Builder.
    
    Loads configuration from YAML files and provides access to
    configuration values with defaults.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to YAML configuration file. If None, uses defaults.
        """
        self._config: Dict[str, Any] = {}
        self._defaults = self._get_defaults()
        
        if config_path and config_path.exists():
            self.load_from_file(config_path)
        else:
            self._config = self._defaults.copy()
            if config_path:
                logger.warning(f"Config file not found: {config_path}, using defaults")
    
    def _get_defaults(self) -> Dict[str, Any]:
        """Get default configuration values."""
        return {
            "model": {
                "default_type": "image_classification",
                "default_framework": "pytorch",
                "default_architecture": "resnet34",
            },
            "data": {
                "raw_data_dir": "raw_data",
                "data_dir": "data",
                "test_images_per_class": 1000,
                "image_size": 224,
                "batch_size": None,  # auto-detect
            },
            "training": {
                "frozen_epochs": 5,
                "unfrozen_epochs": 20,
                "frozen_lr": 0.001,
                "unfrozen_lr_max": 0.0003,
                "unfrozen_lr_min": 0.00001,
                "num_workers": 12,
            },
            "paths": {
                "models_dir": "data/models",
                "logs_dir": "logs",
                "timing_dir": "timing_data",
            },
        }
    
    def load_from_file(self, config_path: Path):
        """
        Load configuration from a YAML file.
        
        Args:
            config_path: Path to YAML configuration file
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                file_config = yaml.safe_load(f) or {}
            
            # Merge with defaults (file config takes precedence)
            self._config = self._deep_merge(self._defaults.copy(), file_config)
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            logger.info("Using default configuration")
            self._config = self._defaults.copy()
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """
        Deep merge two dictionaries.
        
        Args:
            base: Base dictionary
            override: Dictionary to merge in (takes precedence)
            
        Returns:
            Merged dictionary
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to config value (e.g., 'model.default_type')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self._config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any):
        """
        Set a configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to config value
            value: Value to set
        """
        keys = key_path.split('.')
        config = self._config
        
        # Navigate/create nested dictionaries
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # Set the final value
        config[keys[-1]] = value
    
    def update(self, updates: Dict[str, Any]):
        """
        Update configuration with a dictionary of values.
        
        Args:
            updates: Dictionary of updates (can use dot notation keys)
        """
        for key, value in updates.items():
            self.set(key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Get the full configuration as a dictionary.
        
        Returns:
            Configuration dictionary
        """
        return self._config.copy()
    
    def save_to_file(self, config_path: Path):
        """
        Save configuration to a YAML file.
        
        Args:
            config_path: Path to save configuration
        """
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)
            logger.info(f"Saved configuration to {config_path}")
        except Exception as e:
            logger.error(f"Failed to save config to {config_path}: {e}")
            raise


# Global configuration instance
_global_config: Optional[Config] = None


def get_config(config_path: Optional[Path] = None) -> Config:
    """
    Get the global configuration instance.
    
    Args:
        config_path: Optional path to config file (only used on first call)
        
    Returns:
        Config instance
    """
    global _global_config
    
    if _global_config is None:
        if config_path is None:
            # Try to find default config file
            default_config = Path("configs/default.yaml")
            if default_config.exists():
                config_path = default_config
        
        _global_config = Config(config_path)
    
    return _global_config


def reset_config():
    """Reset the global configuration (useful for testing)."""
    global _global_config
    _global_config = None
