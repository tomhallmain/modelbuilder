"""
Keras/TensorFlow data loading utilities for image classification.

This module provides data generators for Keras training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional

from mb.utils.logging_setup import get_logger
from mb.utils.translations import _

logger = get_logger(__name__)

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning(_("TensorFlow not available. Keras data loaders will not work."))


def create_data_generators(
    train_dir: Path,
    val_dir: Path,
    batch_size: int,
    image_size: int = 224,
    num_workers: int = 0,
    **kwargs
) -> Tuple[ImageDataGenerator, ImageDataGenerator]:
    """
    Create Keras data generators for training and validation.
    
    Args:
        train_dir: Path to training data directory
        val_dir: Path to validation/test data directory
        batch_size: Batch size for data loading
        image_size: Target image size (assumes square)
        num_workers: Number of worker processes (not used in Keras)
        **kwargs: Additional arguments
        
    Returns:
        Tuple of (train_generator, val_generator)
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required for Keras data generators")
    
    # Training data generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        zoom_range=0.1,
        fill_mode='nearest'
    )
    
    # Validation data generator (no augmentation)
    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0
    )
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        directory=str(train_dir),
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        seed=42
    )
    
    val_generator = val_datagen.flow_from_directory(
        directory=str(val_dir),
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        seed=42
    )
    
    logger.info(_("Created Keras data generators:"))
    logger.info(
        _(
            "  Train: {samples} samples, {n_train_generator} batches"
        ).format(samples=train_generator.samples, n_train_generator=len(train_generator)))
    logger.info(
        _(
            "  Val: {samples} samples, {n_val_generator} batches"
        ).format(samples=val_generator.samples, n_val_generator=len(val_generator)))
    logger.info(_("  Classes: {class_indices}").format(class_indices=train_generator.class_indices))
    
    return train_generator, val_generator
