"""
PyTorch architecture definitions and registration.

This module registers PyTorch model architectures with the global registry.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional, Union
import logging

from mb.models.frameworks.registry import register_architecture
from mb.models.types import ArchitectureType, FrameworkType

_FW = FrameworkType.PYTORCH

logger = logging.getLogger(__name__)


def _architecture_str(architecture: Union[ArchitectureType, str]) -> str:
    return architecture.value if isinstance(architecture, ArchitectureType) else str(architecture).strip().lower()


def create_resnet(
    architecture: Union[ArchitectureType, str],
    num_classes: int,
    pretrained: bool = True,
    **kwargs
) -> nn.Module:
    """
    Create a ResNet model.

    Args:
        architecture: Canonical id (e.g. :class:`~mb.models.types.ArchitectureType.RESNET34`) or string
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        **kwargs: Additional arguments (ignored for now)

    Returns:
        ResNet model instance
    """
    arch_s = _architecture_str(architecture)
    # Map architecture names to model constructors (string keys = ``ArchitectureType`` values)
    resnet_models = {
        ArchitectureType.RESNET18.value: models.resnet18,
        ArchitectureType.RESNET34.value: models.resnet34,
        ArchitectureType.RESNET50.value: models.resnet50,
        ArchitectureType.RESNET101.value: models.resnet101,
        ArchitectureType.RESNET152.value: models.resnet152,
    }

    if arch_s not in resnet_models:
        raise ValueError(
            f"Unknown ResNet architecture: {arch_s}. "
            f"Supported: {list(resnet_models.keys())}"
        )

    model_fn = resnet_models[arch_s]
    model = model_fn(pretrained=pretrained, **kwargs)

    # Replace the final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    logger.info(f"Created {arch_s} with {num_classes} classes (pretrained={pretrained})")
    
    return model


def create_efficientnet(
    architecture: Union[ArchitectureType, str],
    num_classes: int,
    pretrained: bool = True,
    **kwargs
) -> nn.Module:
    """
    Create an EfficientNet model.

    Args:
        architecture: Canonical id or string (e.g. ``efficientnet_b0``)
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        **kwargs: Additional arguments (ignored for now)

    Returns:
        EfficientNet model instance
    """
    arch_s = _architecture_str(architecture)
    try:
        from torchvision.models import efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3

        efficientnet_models = {
            ArchitectureType.EFFICIENTNET_B0.value: efficientnet_b0,
            ArchitectureType.EFFICIENTNET_B1.value: efficientnet_b1,
            ArchitectureType.EFFICIENTNET_B2.value: efficientnet_b2,
            ArchitectureType.EFFICIENTNET_B3.value: efficientnet_b3,
        }

        if arch_s not in efficientnet_models:
            raise ValueError(
                f"Unknown EfficientNet architecture: {arch_s}. "
                f"Supported: {list(efficientnet_models.keys())}"
            )

        model_fn = efficientnet_models[arch_s]
        model = model_fn(pretrained=pretrained, **kwargs)

        # Replace the classifier
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)

        logger.info(f"Created {arch_s} with {num_classes} classes (pretrained={pretrained})")
        
        return model
    except ImportError:
        raise ImportError(
            "EfficientNet models require torchvision >= 0.13.0. "
            "Please upgrade: pip install torchvision>=0.13.0"
        )


# Register architectures
def _make_resnet_factory(arch_name):
    """Create a factory function for a ResNet architecture."""
    return lambda num_classes, pretrained=True, **kwargs: create_resnet(arch_name, num_classes, pretrained, **kwargs)

def _make_efficientnet_factory(arch_name):
    """Create a factory function for an EfficientNet architecture."""
    return lambda num_classes, pretrained=True, **kwargs: create_efficientnet(arch_name, num_classes, pretrained, **kwargs)

register_architecture(_FW, ArchitectureType.RESNET18, _make_resnet_factory(ArchitectureType.RESNET18.value))
register_architecture(_FW, ArchitectureType.RESNET34, _make_resnet_factory(ArchitectureType.RESNET34.value))
register_architecture(_FW, ArchitectureType.RESNET50, _make_resnet_factory(ArchitectureType.RESNET50.value))
register_architecture(_FW, ArchitectureType.RESNET101, _make_resnet_factory(ArchitectureType.RESNET101.value))
register_architecture(_FW, ArchitectureType.RESNET152, _make_resnet_factory(ArchitectureType.RESNET152.value))

# Try to register EfficientNet (may not be available in older torchvision versions)
try:
    register_architecture(_FW, ArchitectureType.EFFICIENTNET_B0, _make_efficientnet_factory(ArchitectureType.EFFICIENTNET_B0.value))
    register_architecture(_FW, ArchitectureType.EFFICIENTNET_B1, _make_efficientnet_factory(ArchitectureType.EFFICIENTNET_B1.value))
    register_architecture(_FW, ArchitectureType.EFFICIENTNET_B2, _make_efficientnet_factory(ArchitectureType.EFFICIENTNET_B2.value))
    register_architecture(_FW, ArchitectureType.EFFICIENTNET_B3, _make_efficientnet_factory(ArchitectureType.EFFICIENTNET_B3.value))
except Exception as e:
    logger.debug(f"Could not register EfficientNet architectures: {e}")
