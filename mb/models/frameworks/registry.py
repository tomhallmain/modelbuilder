"""
Architecture registry for framework-specific model architectures.

This module provides a registry system for mapping architecture names
to their implementations across different frameworks.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Union

from mb.models.types import ArchitectureType, FrameworkType
from mb.utils.logging_setup import get_logger

logger = get_logger(__name__)

FrameworkKey = Union[FrameworkType, str]
ArchitectureKey = Union[ArchitectureType, str]


def _framework_key(framework: FrameworkKey) -> str:
    if isinstance(framework, FrameworkType):
        return framework.value
    return str(framework).strip().lower()


def _architecture_key(architecture: ArchitectureKey) -> str:
    if isinstance(architecture, ArchitectureType):
        return architecture.value
    return str(architecture).strip().lower()


class ArchitectureRegistry:
    """
    Registry for model architectures across frameworks.

    This allows easy lookup of architecture implementations by framework
    and architecture name.
    """

    def __init__(self) -> None:
        """Initialize the registry."""
        self._architectures: Dict[str, Dict[str, Callable[..., Any]]] = {}

    def register(
        self,
        framework: FrameworkKey,
        architecture: ArchitectureKey,
        factory: Callable[..., Any],
        overwrite: bool = False,
    ) -> None:
        """
        Register an architecture factory function.

        Args:
            framework: Training framework
            architecture: Canonical architecture id (registry key, lowercase)
            factory: Callable that creates the architecture
            overwrite: Whether to overwrite existing registration

        Raises:
            ValueError: If architecture already registered and overwrite=False
        """
        fw = _framework_key(framework)
        arch = _architecture_key(architecture)
        if fw not in self._architectures:
            self._architectures[fw] = {}

        if arch in self._architectures[fw] and not overwrite:
            raise ValueError(
                f"Architecture '{arch}' already registered for framework '{fw}'"
            )

        self._architectures[fw][arch] = factory
        logger.debug("Registered %s.%s", fw, arch)

    def get(
        self,
        framework: FrameworkKey,
        architecture: ArchitectureKey,
    ) -> Optional[Callable[..., Any]]:
        """
        Get an architecture factory function.

        Args:
            framework: Training framework
            architecture: Architecture name

        Returns:
            Factory function or None if not found
        """
        fw = _framework_key(framework)
        arch = _architecture_key(architecture)
        return self._architectures.get(fw, {}).get(arch)

    def list_architectures(
        self, framework: Optional[FrameworkKey] = None
    ) -> Dict[str, list]:
        """
        List all registered architectures.

        Args:
            framework: If provided, only list architectures for this framework

        Returns:
            Dictionary mapping framework name strings to lists of architecture name strings
        """
        if framework is not None:
            fw = _framework_key(framework)
            return {fw: list(self._architectures.get(fw, {}).keys())}

        return {
            fwk: list(archs.keys())
            for fwk, archs in self._architectures.items()
        }

    def is_registered(self, framework: FrameworkKey, architecture: ArchitectureKey) -> bool:
        """
        Check if an architecture is registered.

        Args:
            framework: Training framework
            architecture: Architecture name

        Returns:
            True if registered, False otherwise
        """
        fw = _framework_key(framework)
        arch = _architecture_key(architecture)
        return arch in self._architectures.get(fw, {})


# Global registry instance
_registry = ArchitectureRegistry()


def register_architecture(
    framework: FrameworkKey,
    architecture: ArchitectureKey,
    factory: Callable[..., Any],
    overwrite: bool = False,
) -> None:
    """
    Register an architecture in the global registry.

    Args:
        framework: Training framework
        architecture: Canonical architecture id
        factory: Factory function
        overwrite: Whether to overwrite existing registration
    """
    _registry.register(framework, architecture, factory, overwrite)


def get_architecture(
    framework: FrameworkKey, architecture: ArchitectureKey
) -> Optional[Callable[..., Any]]:
    """
    Get an architecture factory from the global registry.

    Args:
        framework: Training framework
        architecture: Architecture name (enum or string from config/CLI)

    Returns:
        Factory function or None if not found
    """
    return _registry.get(framework, architecture)


def list_architectures(
    framework: Optional[FrameworkKey] = None,
) -> Dict[str, list]:
    """
    List all registered architectures.

    Args:
        framework: If provided, only list for this framework

    Returns:
        Dictionary mapping framework name strings to architecture lists
    """
    return _registry.list_architectures(framework)


def is_architecture_registered(
    framework: FrameworkKey, architecture: ArchitectureKey
) -> bool:
    """
    Check if an architecture is registered.

    Args:
        framework: Training framework
        architecture: Architecture name

    Returns:
        True if registered, False otherwise
    """
    return _registry.is_registered(framework, architecture)
