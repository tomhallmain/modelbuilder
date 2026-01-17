"""
PyTorch architecture definitions and registration.

This module registers PyTorch model architectures with the global registry.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional
import logging

from mb.models.frameworks.registry import register_architecture

logger = logging.getLogger(__name__)


def create_resnet(
    architecture: str,
    num_classes: int,
    pretrained: bool = True,
    **kwargs
) -> nn.Module:
    """
    Create a ResNet model.
    
    Args:
        architecture: Architecture name (e.g., 'resnet18', 'resnet34', 'resnet50')
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        **kwargs: Additional arguments (ignored for now)
        
    Returns:
        ResNet model instance
    """
    # Map architecture names to model constructors
    resnet_models = {
        'resnet18': models.resnet18,
        'resnet34': models.resnet34,
        'resnet50': models.resnet50,
        'resnet101': models.resnet101,
        'resnet152': models.resnet152,
    }
    
    if architecture not in resnet_models:
        raise ValueError(
            f"Unknown ResNet architecture: {architecture}. "
            f"Supported: {list(resnet_models.keys())}"
        )
    
    model_fn = resnet_models[architecture]
    model = model_fn(pretrained=pretrained, **kwargs)
    
    # Replace the final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    logger.info(f"Created {architecture} with {num_classes} classes (pretrained={pretrained})")
    
    return model


def create_efficientnet(
    architecture: str,
    num_classes: int,
    pretrained: bool = True,
    **kwargs
) -> nn.Module:
    """
    Create an EfficientNet model.
    
    Args:
        architecture: Architecture name (e.g., 'efficientnet_b0', 'efficientnet_b1')
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        **kwargs: Additional arguments (ignored for now)
        
    Returns:
        EfficientNet model instance
    """
    try:
        from torchvision.models import efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3
        
        efficientnet_models = {
            'efficientnet_b0': efficientnet_b0,
            'efficientnet_b1': efficientnet_b1,
            'efficientnet_b2': efficientnet_b2,
            'efficientnet_b3': efficientnet_b3,
        }
        
        if architecture not in efficientnet_models:
            raise ValueError(
                f"Unknown EfficientNet architecture: {architecture}. "
                f"Supported: {list(efficientnet_models.keys())}"
            )
        
        model_fn = efficientnet_models[architecture]
        model = model_fn(pretrained=pretrained, **kwargs)
        
        # Replace the classifier
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)
        
        logger.info(f"Created {architecture} with {num_classes} classes (pretrained={pretrained})")
        
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

register_architecture('pytorch', 'resnet18', _make_resnet_factory('resnet18'))
register_architecture('pytorch', 'resnet34', _make_resnet_factory('resnet34'))
register_architecture('pytorch', 'resnet50', _make_resnet_factory('resnet50'))
register_architecture('pytorch', 'resnet101', _make_resnet_factory('resnet101'))
register_architecture('pytorch', 'resnet152', _make_resnet_factory('resnet152'))

# Try to register EfficientNet (may not be available in older torchvision versions)
try:
    register_architecture('pytorch', 'efficientnet_b0', _make_efficientnet_factory('efficientnet_b0'))
    register_architecture('pytorch', 'efficientnet_b1', _make_efficientnet_factory('efficientnet_b1'))
except Exception as e:
    logger.debug(f"Could not register EfficientNet architectures: {e}")
