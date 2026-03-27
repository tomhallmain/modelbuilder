"""
Keras/TensorFlow data loading utilities for image classification.

This module provides data generators for Keras training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not available. Keras data loaders will not work.")


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
    
    logger.info(f"Created Keras data generators:")
    logger.info(f"  Train: {train_generator.samples} samples, {len(train_generator)} batches")
    logger.info(f"  Val: {val_generator.samples} samples, {len(val_generator)} batches")
    logger.info(f"  Classes: {train_generator.class_indices}")
    
    return train_generator, val_generator
