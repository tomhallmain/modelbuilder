"""
Keras/TensorFlow architecture definitions and registration.

This module registers Keras model architectures with the global registry.
"""

import logging
from typing import Union

from mb.models.types import ArchitectureType

logger = logging.getLogger(__name__)


def _architecture_str(architecture: Union[ArchitectureType, str]) -> str:
    return architecture.value if isinstance(architecture, ArchitectureType) else str(architecture).strip().lower()

try:
    from tensorflow import keras
    from tensorflow.keras import applications
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not available. Keras architectures will not work.")


def create_resnet(
    architecture: Union[ArchitectureType, str],
    num_classes: int,
    pretrained: bool = True,
    **kwargs
):
    """
    Create a ResNet model using Keras.

    Args:
        architecture: Canonical id (e.g. :class:`~mb.models.types.ArchitectureType.RESNET50`) or string
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights (ImageNet)
        **kwargs: Additional arguments

    Returns:
        Keras model instance
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required for Keras models")

    arch_s = _architecture_str(architecture)
    # Map canonical lowercase names to Keras applications
    resnet_models = {
        ArchitectureType.RESNET50.value: applications.ResNet50,
        ArchitectureType.RESNET101.value: applications.ResNet101,
        ArchitectureType.RESNET152.value: applications.ResNet152,
    }

    if arch_s not in resnet_models:
        raise ValueError(
            f"Unknown ResNet architecture: {arch_s}. "
            f"Supported: {list(resnet_models.keys())}"
        )

    model_fn = resnet_models[arch_s]
    
    # Create base model
    base_model = model_fn(
        weights='imagenet' if pretrained else None,
        include_top=False,
        input_shape=(224, 224, 3),
        **kwargs
    )
    
    # Add custom classifier
    x = keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = keras.layers.Dense(512, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    predictions = keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=base_model.input, outputs=predictions)
    
    logger.info(f"Created {arch_s} with {num_classes} classes (pretrained={pretrained})")

    return model


def create_efficientnet(
    architecture: Union[ArchitectureType, str],
    num_classes: int,
    pretrained: bool = True,
    **kwargs
):
    """
    Create an EfficientNet model using Keras.

    Args:
        architecture: Canonical id or string (e.g. ``efficientnet_b0``)
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights (ImageNet)
        **kwargs: Additional arguments

    Returns:
        Keras model instance
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required for Keras models")

    arch_s = _architecture_str(architecture)

    try:
        from tensorflow.keras.applications import efficientnet
    except ImportError:
        raise ImportError("EfficientNet requires TensorFlow >= 2.3.0")

    efficientnet_models = {
        ArchitectureType.EFFICIENTNET_B0.value: efficientnet.EfficientNetB0,
        ArchitectureType.EFFICIENTNET_B1.value: efficientnet.EfficientNetB1,
        ArchitectureType.EFFICIENTNET_B2.value: efficientnet.EfficientNetB2,
        ArchitectureType.EFFICIENTNET_B3.value: efficientnet.EfficientNetB3,
    }

    if arch_s not in efficientnet_models:
        raise ValueError(
            f"Unknown EfficientNet architecture: {arch_s}. "
            f"Supported: {list(efficientnet_models.keys())}"
        )

    model_fn = efficientnet_models[arch_s]
    
    # Create base model
    base_model = model_fn(
        weights='imagenet' if pretrained else None,
        include_top=False,
        input_shape=(224, 224, 3),
        **kwargs
    )
    
    # Add custom classifier
    x = keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = keras.layers.Dense(512, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    predictions = keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=base_model.input, outputs=predictions)
    
    logger.info(f"Created {arch_s} with {num_classes} classes (pretrained={pretrained})")

    return model


def _keras_gap_classifier(base_model, num_classes: int):
    """GlobalAveragePooling2D + small MLP head (same pattern as :func:`create_resnet`)."""
    x = keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = keras.layers.Dense(512, activation="relu")(x)
    x = keras.layers.Dropout(0.5)(x)
    predictions = keras.layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs=base_model.input, outputs=predictions)


def create_mobilenet(
    architecture: Union[ArchitectureType, str],
    num_classes: int,
    pretrained: bool = True,
    **kwargs
):
    """Create MobileNet V2 / V3 via ``keras.applications``."""
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required for Keras models")
    arch_s = _architecture_str(architecture)
    mobilenet_models = {
        ArchitectureType.MOBILENET_V2.value: applications.MobileNetV2,
        ArchitectureType.MOBILENET_V3_LARGE.value: applications.MobileNetV3Large,
        ArchitectureType.MOBILENET_V3_SMALL.value: applications.MobileNetV3Small,
    }
    if arch_s not in mobilenet_models:
        raise ValueError(
            f"Unknown MobileNet architecture: {arch_s}. Supported: {list(mobilenet_models.keys())}"
        )
    model_fn = mobilenet_models[arch_s]
    base_model = model_fn(
        weights="imagenet" if pretrained else None,
        include_top=False,
        input_shape=(224, 224, 3),
        **kwargs,
    )
    model = _keras_gap_classifier(base_model, num_classes)
    logger.info(f"Created {arch_s} with {num_classes} classes (pretrained={pretrained})")
    return model


def create_densenet(
    architecture: Union[ArchitectureType, str],
    num_classes: int,
    pretrained: bool = True,
    **kwargs
):
    """Create DenseNet via ``keras.applications``."""
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required for Keras models")
    arch_s = _architecture_str(architecture)
    densenet_models = {
        ArchitectureType.DENSENET121.value: applications.DenseNet121,
        ArchitectureType.DENSENET169.value: applications.DenseNet169,
        ArchitectureType.DENSENET201.value: applications.DenseNet201,
    }
    if arch_s not in densenet_models:
        raise ValueError(
            f"Unknown DenseNet architecture: {arch_s}. Supported: {list(densenet_models.keys())}"
        )
    model_fn = densenet_models[arch_s]
    base_model = model_fn(
        weights="imagenet" if pretrained else None,
        include_top=False,
        input_shape=(224, 224, 3),
        **kwargs,
    )
    model = _keras_gap_classifier(base_model, num_classes)
    logger.info(f"Created {arch_s} with {num_classes} classes (pretrained={pretrained})")
    return model


def create_vgg(
    architecture: Union[ArchitectureType, str],
    num_classes: int,
    pretrained: bool = True,
    **kwargs
):
    """Create VGG via ``keras.applications``."""
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required for Keras models")
    arch_s = _architecture_str(architecture)
    vgg_models = {
        ArchitectureType.VGG16.value: applications.VGG16,
        ArchitectureType.VGG19.value: applications.VGG19,
    }
    if arch_s not in vgg_models:
        raise ValueError(
            f"Unknown VGG architecture: {arch_s}. Supported: {list(vgg_models.keys())}"
        )
    model_fn = vgg_models[arch_s]
    base_model = model_fn(
        weights="imagenet" if pretrained else None,
        include_top=False,
        input_shape=(224, 224, 3),
        **kwargs,
    )
    model = _keras_gap_classifier(base_model, num_classes)
    logger.info(f"Created {arch_s} with {num_classes} classes (pretrained={pretrained})")
    return model


# Register architectures (only if TensorFlow is available)
if TF_AVAILABLE:
    from mb.models.frameworks.registry import register_architecture
    from mb.models.types import FrameworkType

    _FW = FrameworkType.KERAS

    def _make_resnet_factory(arch_name):
        """Create a factory function for a ResNet architecture."""
        return lambda num_classes, pretrained=True, **kwargs: create_resnet(arch_name, num_classes, pretrained, **kwargs)

    def _make_efficientnet_factory(arch_name):
        """Create a factory function for an EfficientNet architecture."""
        return lambda num_classes, pretrained=True, **kwargs: create_efficientnet(arch_name, num_classes, pretrained, **kwargs)

    register_architecture(_FW, ArchitectureType.RESNET50, _make_resnet_factory(ArchitectureType.RESNET50.value))
    register_architecture(_FW, ArchitectureType.RESNET101, _make_resnet_factory(ArchitectureType.RESNET101.value))
    register_architecture(_FW, ArchitectureType.RESNET152, _make_resnet_factory(ArchitectureType.RESNET152.value))

    # Try to register EfficientNet
    try:
        register_architecture(_FW, ArchitectureType.EFFICIENTNET_B0, _make_efficientnet_factory(ArchitectureType.EFFICIENTNET_B0.value))
        register_architecture(_FW, ArchitectureType.EFFICIENTNET_B1, _make_efficientnet_factory(ArchitectureType.EFFICIENTNET_B1.value))
        register_architecture(_FW, ArchitectureType.EFFICIENTNET_B2, _make_efficientnet_factory(ArchitectureType.EFFICIENTNET_B2.value))
        register_architecture(_FW, ArchitectureType.EFFICIENTNET_B3, _make_efficientnet_factory(ArchitectureType.EFFICIENTNET_B3.value))
    except Exception as e:
        logger.debug(f"Could not register EfficientNet architectures: {e}")

    def _make_mobilenet_factory(arch_name: str):
        return lambda num_classes, pretrained=True, **kw: create_mobilenet(
            arch_name, num_classes, pretrained, **kw
        )

    def _make_densenet_factory(arch_name: str):
        return lambda num_classes, pretrained=True, **kw: create_densenet(
            arch_name, num_classes, pretrained, **kw
        )

    def _make_vgg_factory(arch_name: str):
        return lambda num_classes, pretrained=True, **kw: create_vgg(
            arch_name, num_classes, pretrained, **kw
        )

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
