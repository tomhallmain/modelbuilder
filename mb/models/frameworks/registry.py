"""
Architecture registry for framework-specific model architectures.

This module provides a registry system for mapping architecture names
to their implementations across different frameworks.
"""

from typing import Dict, Callable, Optional, Any
import logging

logger = logging.getLogger(__name__)


class ArchitectureRegistry:
    """
    Registry for model architectures across frameworks.
    
    This allows easy lookup of architecture implementations by framework
    and architecture name.
    """
    
    def __init__(self):
        """Initialize the registry."""
        self._architectures: Dict[str, Dict[str, Callable]] = {}
    
    def register(
        self,
        framework: str,
        architecture: str,
        factory: Callable,
        overwrite: bool = False
    ):
        """
        Register an architecture factory function.
        
        Args:
            framework: Framework name (e.g., 'pytorch', 'keras')
            architecture: Architecture name (e.g., 'resnet34')
            factory: Callable that creates the architecture
            overwrite: Whether to overwrite existing registration
            
        Raises:
            ValueError: If architecture already registered and overwrite=False
        """
        if framework not in self._architectures:
            self._architectures[framework] = {}
        
        if architecture in self._architectures[framework] and not overwrite:
            raise ValueError(
                f"Architecture '{architecture}' already registered for framework '{framework}'"
            )
        
        self._architectures[framework][architecture] = factory
        logger.debug(f"Registered {framework}.{architecture}")
    
    def get(
        self,
        framework: str,
        architecture: str
    ) -> Optional[Callable]:
        """
        Get an architecture factory function.
        
        Args:
            framework: Framework name
            architecture: Architecture name
            
        Returns:
            Factory function or None if not found
        """
        return self._architectures.get(framework, {}).get(architecture)
    
    def list_architectures(self, framework: Optional[str] = None) -> Dict[str, list]:
        """
        List all registered architectures.
        
        Args:
            framework: If provided, only list architectures for this framework
            
        Returns:
            Dictionary mapping framework names to lists of architecture names
        """
        if framework:
            return {framework: list(self._architectures.get(framework, {}).keys())}
        
        return {
            fw: list(archs.keys())
            for fw, archs in self._architectures.items()
        }
    
    def is_registered(self, framework: str, architecture: str) -> bool:
        """
        Check if an architecture is registered.
        
        Args:
            framework: Framework name
            architecture: Architecture name
            
        Returns:
            True if registered, False otherwise
        """
        return architecture in self._architectures.get(framework, {})


# Global registry instance
_registry = ArchitectureRegistry()


def register_architecture(
    framework: str,
    architecture: str,
    factory: Callable,
    overwrite: bool = False
):
    """
    Register an architecture in the global registry.
    
    Args:
        framework: Framework name
        architecture: Architecture name
        factory: Factory function
        overwrite: Whether to overwrite existing registration
    """
    _registry.register(framework, architecture, factory, overwrite)


def get_architecture(framework: str, architecture: str) -> Optional[Callable]:
    """
    Get an architecture factory from the global registry.
    
    Args:
        framework: Framework name
        architecture: Architecture name
        
    Returns:
        Factory function or None if not found
    """
    return _registry.get(framework, architecture)


def list_architectures(framework: Optional[str] = None) -> Dict[str, list]:
    """
    List all registered architectures.
    
    Args:
        framework: If provided, only list for this framework
        
    Returns:
        Dictionary mapping framework names to architecture lists
    """
    return _registry.list_architectures(framework)


def is_architecture_registered(framework: str, architecture: str) -> bool:
    """
    Check if an architecture is registered.
    
    Args:
        framework: Framework name
        architecture: Architecture name
        
    Returns:
        True if registered, False otherwise
    """
    return _registry.is_registered(framework, architecture)
