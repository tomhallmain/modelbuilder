"""
Keras/TensorFlow trainer implementation.

This module implements the FrameworkTrainer interface for Keras/TensorFlow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union
import json
import logging
import os
import threading
from datetime import datetime

# Before importing TensorFlow: cut C++ INFO/WARN noise (oneDNN, absl pre-init) on stderr.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    from tensorflow import keras
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logging.getLogger(__name__).warning("TensorFlow not available. Keras trainer will not work.")

from mb.cancellation import check_cancel_event
from mb.models.base import FrameworkTrainer
from mb.training.gui_progress import (
    make_keras_frozen_gui_progress,
    make_keras_unfrozen_gui_progress,
)
from mb.models.frameworks.keras.data_loader import create_data_generators
from mb.models.frameworks.keras.architectures import create_resnet, create_efficientnet
from mb.models.frameworks.registry import get_architecture, list_architectures
from mb.models.types import ArchitectureType, FrameworkType

logger = logging.getLogger(__name__)


class KerasTrainer(FrameworkTrainer):
    """
    Keras/TensorFlow implementation of FrameworkTrainer.
    
    Supports transfer learning with frozen/unfrozen phases,
    callbacks (checkpointing, early stopping), and evaluation.
    """
    
    def __init__(self):
        """Initialize the Keras trainer."""
        super().__init__(FrameworkType.KERAS)
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for Keras trainer")
        logger.info("Keras trainer initialized")
    
    def get_supported_architectures(self) -> list:
        """Get list of supported architectures."""
        fw = FrameworkType.KERAS
        return list_architectures(fw).get(fw.value, [])
    
    def create_model(
        self,
        architecture: Union[ArchitectureType, str],
        num_classes: int,
        pretrained: bool = True,
        **kwargs
    ) -> keras.Model:
        """
        Create a Keras model.
        
        Args:
            architecture: Architecture name (e.g., 'resnet50')
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            **kwargs: Additional architecture-specific arguments
            
        Returns:
            Keras model instance
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for Keras models")
        
        arch_s = architecture.value if isinstance(architecture, ArchitectureType) else str(architecture)
        # Try to get from registry first
        factory = get_architecture(FrameworkType.KERAS, architecture)

        if factory:
            model = factory(num_classes=num_classes, pretrained=pretrained, **kwargs)
        else:
            # Fallback to direct creation
            if arch_s.lower().startswith('resnet'):
                model = create_resnet(arch_s, num_classes, pretrained, **kwargs)
            elif arch_s.lower().startswith('efficientnet'):
                model = create_efficientnet(arch_s, num_classes, pretrained, **kwargs)
            else:
                raise ValueError(f"Unknown architecture: {arch_s}")
        
        logger.info(f"Created {arch_s} model with {num_classes} classes")
        
        return model
    
    def create_data_loaders(
        self,
        train_dir: Path,
        val_dir: Path,
        batch_size: int,
        image_size: int = 224,
        num_workers: int = 0,
        **kwargs
    ) -> Tuple[ImageDataGenerator, ImageDataGenerator]:
        """
        Create Keras data generators.
        
        Args:
            train_dir: Path to training data directory
            val_dir: Path to validation/test data directory
            batch_size: Batch size for data loading
            image_size: Target image size (assumes square)
            num_workers: Number of worker processes (not used in Keras)
            **kwargs: Additional data generator arguments
            
        Returns:
            Tuple of (train_generator, val_generator)
        """
        return create_data_generators(
            train_dir=train_dir,
            val_dir=val_dir,
            batch_size=batch_size,
            image_size=image_size,
            num_workers=num_workers,
            **kwargs
        )
    
    def train(
        self,
        model: keras.Model,
        train_loader: ImageDataGenerator,
        val_loader: ImageDataGenerator,
        hyperparams: Dict[str, Any],
        output_dir: Path,
        resume_from_checkpoint: Optional[Path] = None,
        cancel_event: Optional[threading.Event] = None,
        progress_cb: Optional[Callable[[str, Optional[float]], None]] = None,
        **kwargs
    ) -> keras.Model:
        """
        Train the Keras model.
        
        Args:
            model: Model instance to train
            train_loader: Training data generator
            val_loader: Validation data generator
            hyperparams: Dictionary of hyperparameters
            output_dir: Directory to save checkpoints and final model
            resume_from_checkpoint: Optional path to checkpoint to resume from
            **kwargs: Additional training arguments
            
        Returns:
            Trained model instance
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for Keras training")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract hyperparameters
        frozen_epochs = hyperparams.get('frozen_epochs', 5)
        unfrozen_epochs = hyperparams.get('unfrozen_epochs', 20)
        total_plan_epochs = frozen_epochs + unfrozen_epochs
        frozen_lr = hyperparams.get('frozen_lr', 0.001)
        unfrozen_lr_max = hyperparams.get('unfrozen_lr_max', 0.0003)
        unfrozen_lr_min = hyperparams.get('unfrozen_lr_min', 0.00001)
        
        # Load checkpoint if resuming
        frozen_epochs_completed = 0
        unfrozen_epochs_completed = 0
        
        if resume_from_checkpoint and resume_from_checkpoint.exists():
            checkpoint_data = self._load_checkpoint(resume_from_checkpoint, model)
            if checkpoint_data:
                frozen_epochs_completed = checkpoint_data.get('frozen_epochs_completed', 0)
                unfrozen_epochs_completed = checkpoint_data.get('unfrozen_epochs_completed', 0)
                logger.info(f"Resumed from checkpoint: "
                          f"frozen={frozen_epochs_completed}/{frozen_epochs}, "
                          f"unfrozen={unfrozen_epochs_completed}/{unfrozen_epochs}")
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=frozen_lr),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Phase 1: Frozen backbone training
        if frozen_epochs_completed < frozen_epochs:
            logger.info(f"Phase 1: Training with frozen backbone ({frozen_epochs} epochs)")
            
            # Freeze all layers except the classifier
            for layer in model.layers:
                layer.trainable = False
            
            # Unfreeze classifier layers (last few layers)
            for layer in model.layers[-3:]:  # Last 3 layers (dense layers)
                layer.trainable = True
            
            # Recompile with frozen learning rate
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=frozen_lr),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Callbacks
            checkpoint_cb = ModelCheckpoint(
                filepath=str(output_dir / "checkpoint_frozen_epoch_{epoch:02d}.h5"),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
            
            callbacks = [checkpoint_cb]
            remaining_frozen = frozen_epochs - frozen_epochs_completed

            callbacks.append(
                make_keras_frozen_gui_progress(
                    progress_cb,
                    cancel_event,
                    total_plan_epochs,
                    frozen_epochs,
                    frozen_epochs_completed,
                    train_loader,
                    val_loader,
                )
            )

            # Train
            history = model.fit(
                train_loader,
                epochs=remaining_frozen,
                validation_data=val_loader,
                callbacks=callbacks,
                verbose=1
            )
            
            frozen_epochs_completed = frozen_epochs
        else:
            logger.info("Frozen phase already completed, skipping")
        
        # Phase 2: Unfrozen fine-tuning
        if unfrozen_epochs_completed < unfrozen_epochs:
            logger.info(f"Phase 2: Fine-tuning all layers ({unfrozen_epochs} epochs)")
            
            # Unfreeze all layers
            for layer in model.layers:
                layer.trainable = True
            
            # Recompile with learning rate schedule
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=unfrozen_lr_max),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Callbacks
            checkpoint_cb = ModelCheckpoint(
                filepath=str(output_dir / "checkpoint_unfrozen_epoch_{epoch:02d}.h5"),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
            
            reduce_lr_cb = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=unfrozen_lr_min,
                verbose=1
            )
            
            early_stop_cb = EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            )
            
            callbacks = [checkpoint_cb, reduce_lr_cb, early_stop_cb]
            remaining_unfrozen = unfrozen_epochs - unfrozen_epochs_completed

            callbacks.append(
                make_keras_unfrozen_gui_progress(
                    progress_cb,
                    cancel_event,
                    total_plan_epochs,
                    frozen_epochs,
                    unfrozen_epochs,
                    unfrozen_epochs_completed,
                    train_loader,
                    val_loader,
                )
            )

            # Train
            history = model.fit(
                train_loader,
                epochs=remaining_unfrozen,
                validation_data=val_loader,
                callbacks=callbacks,
                verbose=1
            )
            
            unfrozen_epochs_completed = unfrozen_epochs
        else:
            logger.info("Unfrozen phase already completed, skipping")
        
        logger.info("Training completed")
        
        return model
    
    def evaluate(
        self,
        model: keras.Model,
        val_loader: ImageDataGenerator,
        **kwargs
    ) -> Dict[str, float]:
        """
        Evaluate the model on validation data.
        
        Args:
            model: Model instance to evaluate
            val_loader: Validation data generator
            **kwargs: Additional evaluation arguments
            
        Returns:
            Dictionary of metric names to values
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for Keras evaluation")

        cancel_event: Optional[threading.Event] = kwargs.get("cancel_event")
        progress_cb: Optional[Callable[[str, Optional[float]], None]] = kwargs.get("progress_cb")
        check_cancel_event(cancel_event)
        if progress_cb is not None:
            progress_cb("Evaluating…", 0.0)
        results = model.evaluate(val_loader, verbose=0)
        check_cancel_event(cancel_event)
        if progress_cb is not None:
            progress_cb("Evaluating…", 1.0)
        
        # Results are [loss, accuracy, ...] based on metrics
        metrics = {}
        if len(results) >= 1:
            metrics['loss'] = float(results[0])
        if len(results) >= 2:
            metrics['accuracy'] = float(results[1])
        
        return metrics
    
    def save_model(
        self,
        model: keras.Model,
        path: Path,
        format: str = "native",
        **kwargs
    ) -> Path:
        """
        Save the Keras model.
        
        Args:
            model: Model instance to save
            path: Path to save the model
            format: Format to save in ('native' for .h5, 'saved_model' for SavedModel format)
            **kwargs: Additional save arguments
            
        Returns:
            Path where model was saved
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for saving Keras models")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "native" or format == "h5":
            model.save(str(path))
            logger.info(f"Saved Keras model to {path}")
        elif format == "saved_model":
            model.save(str(path))
            logger.info(f"Saved Keras model (SavedModel format) to {path}")
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return path
    
    def load_model(
        self,
        path: Path,
        architecture: str,
        num_classes: int,
        **kwargs
    ) -> keras.Model:
        """
        Load a saved Keras model.
        
        Args:
            path: Path to saved model
            architecture: Model architecture (may be needed for some formats)
            num_classes: Number of output classes (may be needed for some formats)
            **kwargs: Additional load arguments
            
        Returns:
            Loaded model instance
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for loading Keras models")
        
        try:
            # Try loading as SavedModel first
            model = keras.models.load_model(str(path))
            logger.info(f"Loaded Keras model from {path}")
            return model
        except Exception:
            # If that fails, might need to rebuild architecture
            logger.warning(f"Could not load model directly, rebuilding architecture: {architecture}")
            model = self.create_model(architecture, num_classes, pretrained=False)
            model.load_weights(str(path))
            logger.info(f"Loaded Keras model weights from {path}")
            return model
    
    def _load_checkpoint(
        self,
        checkpoint_path: Path,
        model: keras.Model
    ) -> Optional[Dict]:
        """Load checkpoint metadata."""
        metadata_path = checkpoint_path.with_suffix('.json')
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                # Try to load model weights
                try:
                    model.load_weights(str(checkpoint_path))
                    logger.info(f"Loaded checkpoint from {checkpoint_path}")
                except Exception as e:
                    logger.warning(f"Could not load model weights: {e}")
                return metadata
            except Exception as e:
                logger.error(f"Failed to load checkpoint metadata: {e}")
        return None
