"""
Hyperparameter management for training.

This module provides utilities for managing and merging hyperparameters
from various sources (config, CLI, defaults).
"""

from typing import Any, Dict, Optional, TYPE_CHECKING

import logging

if TYPE_CHECKING:
    from mb.pipeline_config import PipelineConfig

logger = logging.getLogger(__name__)


class HyperparameterManager:
    """
    Manages hyperparameters from multiple sources.
    
    Priority order (highest to lowest):
    1. CLI arguments
    2. Config file
    3. Model type defaults
    4. Framework defaults
    """
    
    def __init__(self):
        """Initialize the hyperparameter manager."""
        self._hyperparams: Dict[str, Any] = {}
    
    def merge_hyperparams(
        self,
        model_type_defaults: Dict[str, Any],
        config_hyperparams: Optional[Dict[str, Any]] = None,
        cli_hyperparams: Optional[Dict[str, Any]] = None,
        *,
        pipeline_config: Optional["PipelineConfig"] = None,
    ) -> Dict[str, Any]:
        """
        Merge hyperparameters from multiple sources.
        
        Args:
            model_type_defaults: Default hyperparameters from model type handler
            config_hyperparams: Explicit dict (used when ``pipeline_config`` is omitted)
            cli_hyperparams: Hyperparameters from CLI arguments
            pipeline_config: If set, overrides *config_hyperparams* with
                :meth:`~mb.pipeline_config.PipelineConfig.training_hyperparams`
            
        Returns:
            Merged hyperparameters dictionary
        """
        # Start with model type defaults
        merged = model_type_defaults.copy()
        
        if pipeline_config is not None:
            merged.update(pipeline_config.training_hyperparams())
        elif config_hyperparams:
            merged.update(config_hyperparams)
        
        # Override with CLI values (highest priority)
        if cli_hyperparams:
            # Only include non-None values from CLI
            cli_filtered = {k: v for k, v in cli_hyperparams.items() if v is not None}
            merged.update(cli_filtered)
        
        self._hyperparams = merged
        return merged
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a hyperparameter value.
        
        Args:
            key: Hyperparameter key
            default: Default value if key not found
            
        Returns:
            Hyperparameter value or default
        """
        return self._hyperparams.get(key, default)
    
    def set(self, key: str, value: Any):
        """
        Set a hyperparameter value.
        
        Args:
            key: Hyperparameter key
            value: Value to set
        """
        self._hyperparams[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Get all hyperparameters as a dictionary.
        
        Returns:
            Dictionary of all hyperparameters
        """
        return self._hyperparams.copy()
    
    def validate(self) -> bool:
        """
        Validate hyperparameters.
        
        Returns:
            True if valid, False otherwise
        """
        # Check required hyperparameters
        required = ['frozen_epochs', 'unfrozen_epochs', 'frozen_lr']
        missing = [key for key in required if key not in self._hyperparams]
        
        if missing:
            logger.error(f"Missing required hyperparameters: {missing}")
            return False
        
        # Validate ranges
        if self._hyperparams.get('frozen_epochs', 0) < 0:
            logger.error("frozen_epochs must be >= 0")
            return False
        
        if self._hyperparams.get('unfrozen_epochs', 0) < 0:
            logger.error("unfrozen_epochs must be >= 0")
            return False
        
        if self._hyperparams.get('frozen_lr', 0) <= 0:
            logger.error("frozen_lr must be > 0")
            return False
        
        return True


def get_training_hyperparams(
    model_type_defaults: Dict[str, Any],
    pipeline_config: Optional[Any] = None,
    cli_args: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Get training hyperparameters from multiple sources.
    
    Args:
        model_type_defaults: Default hyperparameters from model type handler
        pipeline_config: :class:`~mb.pipeline_config.PipelineConfig` (optional)
        cli_args: CLI arguments dictionary (optional)
        
    Returns:
        Merged hyperparameters dictionary
    """
    manager = HyperparameterManager()

    hyperparams = manager.merge_hyperparams(
        model_type_defaults=model_type_defaults,
        cli_hyperparams=cli_args,
        pipeline_config=pipeline_config,
    )
    
    # Validate
    if not manager.validate():
        raise ValueError("Invalid hyperparameters")
    
    return hyperparams
