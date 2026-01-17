"""
Keras/TensorFlow architecture definitions and registration.

This module registers Keras model architectures with the global registry.
"""

import logging

logger = logging.getLogger(__name__)

try:
    from tensorflow import keras
    from tensorflow.keras import applications
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not available. Keras architectures will not work.")


def create_resnet(
    architecture: str,
    num_classes: int,
    pretrained: bool = True,
    **kwargs
):
    """
    Create a ResNet model using Keras.
    
    Args:
        architecture: Architecture name (e.g., 'ResNet50', 'ResNet101')
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights (ImageNet)
        **kwargs: Additional arguments
        
    Returns:
        Keras model instance
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required for Keras models")
    
    # Map architecture names to Keras applications
    resnet_models = {
        'resnet50': applications.ResNet50,
        'resnet101': applications.ResNet101,
        'resnet152': applications.ResNet152,
    }
    
    # Also support capitalized versions
    resnet_models.update({
        'ResNet50': applications.ResNet50,
        'ResNet101': applications.ResNet101,
        'ResNet152': applications.ResNet152,
    })
    
    if architecture not in resnet_models:
        raise ValueError(
            f"Unknown ResNet architecture: {architecture}. "
            f"Supported: {list(set(k.lower() for k in resnet_models.keys()))}"
        )
    
    model_fn = resnet_models[architecture]
    
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
    
    logger.info(f"Created {architecture} with {num_classes} classes (pretrained={pretrained})")
    
    return model


def create_efficientnet(
    architecture: str,
    num_classes: int,
    pretrained: bool = True,
    **kwargs
):
    """
    Create an EfficientNet model using Keras.
    
    Args:
        architecture: Architecture name (e.g., 'EfficientNetB0', 'EfficientNetB1')
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights (ImageNet)
        **kwargs: Additional arguments
        
    Returns:
        Keras model instance
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required for Keras models")
    
    try:
        from tensorflow.keras.applications import efficientnet
    except ImportError:
        raise ImportError("EfficientNet requires TensorFlow >= 2.3.0")
    
    # Map architecture names
    efficientnet_models = {
        'efficientnet_b0': efficientnet.EfficientNetB0,
        'efficientnet_b1': efficientnet.EfficientNetB1,
        'efficientnet_b2': efficientnet.EfficientNetB2,
        'efficientnet_b3': efficientnet.EfficientNetB3,
        'EfficientNetB0': efficientnet.EfficientNetB0,
        'EfficientNetB1': efficientnet.EfficientNetB1,
        'EfficientNetB2': efficientnet.EfficientNetB2,
        'EfficientNetB3': efficientnet.EfficientNetB3,
    }
    
    if architecture not in efficientnet_models:
        raise ValueError(
            f"Unknown EfficientNet architecture: {architecture}. "
            f"Supported: {list(set(k.lower() for k in efficientnet_models.keys()))}"
        )
    
    model_fn = efficientnet_models[architecture]
    
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
    
    logger.info(f"Created {architecture} with {num_classes} classes (pretrained={pretrained})")
    
    return model


# Register architectures (only if TensorFlow is available)
if TF_AVAILABLE:
    from mb.models.frameworks.registry import register_architecture
    
    def _make_resnet_factory(arch_name):
        """Create a factory function for a ResNet architecture."""
        return lambda num_classes, pretrained=True, **kwargs: create_resnet(arch_name, num_classes, pretrained, **kwargs)
    
    def _make_efficientnet_factory(arch_name):
        """Create a factory function for an EfficientNet architecture."""
        return lambda num_classes, pretrained=True, **kwargs: create_efficientnet(arch_name, num_classes, pretrained, **kwargs)
    
    register_architecture('keras', 'resnet50', _make_resnet_factory('resnet50'))
    register_architecture('keras', 'resnet101', _make_resnet_factory('resnet101'))
    register_architecture('keras', 'resnet152', _make_resnet_factory('resnet152'))
    
    # Try to register EfficientNet
    try:
        register_architecture('keras', 'efficientnet_b0', _make_efficientnet_factory('efficientnet_b0'))
        register_architecture('keras', 'efficientnet_b1', _make_efficientnet_factory('efficientnet_b1'))
    except Exception as e:
        logger.debug(f"Could not register EfficientNet architectures: {e}")
