"""
PyTorch architecture definitions and registration.

This module registers PyTorch model architectures with the global registry.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional, Union

from mb.models.frameworks.registry import register_architecture
from mb.models.types import ArchitectureType, FrameworkType
from mb.utils.logging_setup import get_logger

_FW = FrameworkType.PYTORCH

logger = get_logger(__name__)


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


def create_mobilenet(
    architecture: Union[ArchitectureType, str],
    num_classes: int,
    pretrained: bool = True,
    **kwargs,
) -> nn.Module:
    """Create MobileNet V2 / V3 (``torchvision.models``)."""
    arch_s = _architecture_str(architecture)
    mobilenet_models = {
        ArchitectureType.MOBILENET_V2.value: models.mobilenet_v2,
        ArchitectureType.MOBILENET_V3_LARGE.value: models.mobilenet_v3_large,
        ArchitectureType.MOBILENET_V3_SMALL.value: models.mobilenet_v3_small,
    }
    if arch_s not in mobilenet_models:
        raise ValueError(
            f"Unknown MobileNet architecture: {arch_s}. Supported: {list(mobilenet_models.keys())}"
        )
    model = mobilenet_models[arch_s](pretrained=pretrained, **kwargs)
    if arch_s == ArchitectureType.MOBILENET_V2.value:
        in_f = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_f, num_classes)
    else:
        in_f = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_f, num_classes)
    logger.info(f"Created {arch_s} with {num_classes} classes (pretrained={pretrained})")
    return model


def create_densenet(
    architecture: Union[ArchitectureType, str],
    num_classes: int,
    pretrained: bool = True,
    **kwargs,
) -> nn.Module:
    """Create DenseNet (``torchvision.models``)."""
    arch_s = _architecture_str(architecture)
    densenet_models = {
        ArchitectureType.DENSENET121.value: models.densenet121,
        ArchitectureType.DENSENET169.value: models.densenet169,
        ArchitectureType.DENSENET201.value: models.densenet201,
    }
    if arch_s not in densenet_models:
        raise ValueError(
            f"Unknown DenseNet architecture: {arch_s}. Supported: {list(densenet_models.keys())}"
        )
    model = densenet_models[arch_s](pretrained=pretrained, **kwargs)
    in_f = model.classifier.in_features
    model.classifier = nn.Linear(in_f, num_classes)
    logger.info(f"Created {arch_s} with {num_classes} classes (pretrained={pretrained})")
    return model


def create_vgg(
    architecture: Union[ArchitectureType, str],
    num_classes: int,
    pretrained: bool = True,
    **kwargs,
) -> nn.Module:
    """Create VGG (``torchvision.models``)."""
    arch_s = _architecture_str(architecture)
    vgg_models = {
        ArchitectureType.VGG16.value: models.vgg16,
        ArchitectureType.VGG19.value: models.vgg19,
    }
    if arch_s not in vgg_models:
        raise ValueError(
            f"Unknown VGG architecture: {arch_s}. Supported: {list(vgg_models.keys())}"
        )
    model = vgg_models[arch_s](pretrained=pretrained, **kwargs)
    in_f = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_f, num_classes)
    logger.info(f"Created {arch_s} with {num_classes} classes (pretrained={pretrained})")
    return model


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


def _make_mobilenet_factory(arch_name: str):
    return lambda num_classes, pretrained=True, **kwargs: create_mobilenet(
        arch_name, num_classes, pretrained, **kwargs
    )


def _make_densenet_factory(arch_name: str):
    return lambda num_classes, pretrained=True, **kwargs: create_densenet(
        arch_name, num_classes, pretrained, **kwargs
    )


def _make_vgg_factory(arch_name: str):
    return lambda num_classes, pretrained=True, **kwargs: create_vgg(arch_name, num_classes, pretrained, **kwargs)


for _arch in (
    ArchitectureType.MOBILENET_V2,
    ArchitectureType.MOBILENET_V3_LARGE,
    ArchitectureType.MOBILENET_V3_SMALL,
):
    register_architecture(_FW, _arch, _make_mobilenet_factory(_arch.value))
for _arch in (
    ArchitectureType.DENSENET121,
    ArchitectureType.DENSENET169,
    ArchitectureType.DENSENET201,
):
    register_architecture(_FW, _arch, _make_densenet_factory(_arch.value))
for _arch in (ArchitectureType.VGG16, ArchitectureType.VGG19):
    register_architecture(_FW, _arch, _make_vgg_factory(_arch.value))
