"""
Generic training orchestrator.

This module provides a framework-agnostic interface for training models.
"""

from pathlib import Path
from typing import Any, Callable, Optional
import logging
import threading

from mb.models.base import FrameworkTrainer
from mb.models.types import ModelType, get_model_type_handler
from mb.models.frameworks.pytorch.trainer import PyTorchTrainer
from mb.models.frameworks.keras.trainer import KerasTrainer
from mb.training.hyperparams import get_training_hyperparams
from mb.training.run_args import TrainingRunArgs
from mb.training.snapshot_integration import update_training_snapshot
from mb.utils.snapshot import (
    find_unified_snapshot, save_unified_snapshot, preload_gather_cache
)

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Generic training orchestrator that works with any framework.
    
    This class handles:
    - Framework selection
    - Model creation
    - Data loading
    - Training execution
    - Model saving
    """
    
    def __init__(
        self,
        framework: str,
        model_type: ModelType = ModelType.IMAGE_CLASSIFICATION,
        pipeline_config: Optional[Any] = None,
    ):
        """
        Initialize the model trainer.
        
        Args:
            framework: Framework name ('pytorch' or 'keras')
            model_type: Model type enum
            pipeline_config: Optional :class:`~mb.pipeline_config.PipelineConfig`
        """
        self.framework_name = framework.lower()
        self.model_type = model_type
        self.pipeline_config = pipeline_config
        
        # Get framework trainer
        if self.framework_name == 'pytorch':
            self.framework_trainer: FrameworkTrainer = PyTorchTrainer()
        elif self.framework_name == 'keras':
            self.framework_trainer = KerasTrainer()
        else:
            raise ValueError(f"Unsupported framework: {framework}")
        
        # Get model type handler
        self.model_type_handler = get_model_type_handler(model_type)
        
        logger.info(f"Initialized ModelTrainer: framework={framework}, model_type={model_type.value}")
    
    def train(
        self,
        args: TrainingRunArgs,
        *,
        cancel_event: Optional[threading.Event] = None,
        progress_cb: Optional[Callable[[str, Optional[float]], None]] = None,
    ) -> Path:
        """
        Train a model from structured :class:`~mb.training.run_args.TrainingRunArgs`.

        Args:
            args: Paths, architecture, hyperparams, and snapshot options (``framework`` must
                match the framework this :class:`ModelTrainer` was constructed with).
            cancel_event: When set, framework training loops stop cooperatively between
                training/validation batches (and at epoch starts in Keras)
            progress_cb: Optional callback ``(message, percent_or_none)`` for GUI progress

        Returns:
            Path to saved model
        """
        if args.framework.lower() != self.framework_name:
            raise ValueError(
                f"TrainingRunArgs.framework ({args.framework!r}) does not match "
                f"this trainer ({self.framework_name!r})"
            )

        data_dir = args.data_dir
        architecture = args.architecture
        output_dir = args.output_dir
        resume_from_checkpoint = args.resume_from
        run_id = args.run_id
        update_snapshot = args.update_snapshot
        cli_hyperparams = args.cli_hyperparams

        def _progress(msg: str, pct: Optional[float] = None) -> None:
            if progress_cb is not None:
                progress_cb(msg, pct)

        # Validate data structure
        _progress("Validating data…", None)
        if not self.model_type_handler.validate_data(data_dir):
            raise ValueError(f"Invalid data structure for {self.model_type.value}")
        
        # Get number of classes
        num_classes = self.model_type_handler.get_num_classes(data_dir)
        logger.info(f"Number of classes: {num_classes}")
        
        # Get hyperparameters
        model_type_defaults = self.model_type_handler.get_default_hyperparams()
        cli_for_hp = cli_hyperparams if cli_hyperparams else None
        hyperparams = get_training_hyperparams(
            model_type_defaults=model_type_defaults,
            pipeline_config=self.pipeline_config,
            cli_args=cli_for_hp,
        )
        
        # Override with CLI args if provided
        if cli_hyperparams:
            if 'image_size' in cli_hyperparams and cli_hyperparams['image_size']:
                hyperparams['image_size'] = cli_hyperparams['image_size']
            if 'batch_size' in cli_hyperparams and cli_hyperparams['batch_size']:
                hyperparams['batch_size'] = cli_hyperparams['batch_size']
        
        # Set defaults if not specified
        image_size = hyperparams.get('image_size', 224)
        batch_size = hyperparams.get('batch_size', 32)
        num_workers = hyperparams.get('num_workers', 0)
        
        logger.info("Hyperparameters:")
        for key, value in hyperparams.items():
            logger.info(f"  {key}: {value}")
        
        # Create model
        _progress("Creating model…", None)
        logger.info(f"Creating {architecture} model...")
        model = self.framework_trainer.create_model(
            architecture=architecture,
            num_classes=num_classes,
            pretrained=True
        )
        
        # Create data loaders
        train_dir = data_dir / "train"
        val_dir = data_dir / "test"
        
        _progress("Loading data…", None)
        logger.info("Creating data loaders...")
        train_loader, val_loader = self.framework_trainer.create_data_loaders(
            train_dir=train_dir,
            val_dir=val_dir,
            batch_size=batch_size,
            image_size=image_size,
            num_workers=num_workers
        )
        
        # Update unified snapshot if requested
        unified_snapshot = None
        if update_snapshot:
            _progress("Loading snapshot…", None)
            logger.info("Loading unified snapshot...")
            search_paths = [data_dir, data_dir.parent]
            unified_snapshot = find_unified_snapshot(search_paths, run_id=run_id, logger=logger)
            
            if unified_snapshot:
                logger.info(f"Loaded unified snapshot with run_id: {unified_snapshot.run_id}")
                
                # Preload gather cache for faster hash lookups
                raw_data_dir = Path(unified_snapshot.raw_data_directory)
                cache_loaded = preload_gather_cache(raw_data_dir)
                if cache_loaded:
                    logger.info(
                        "Gather cache loaded successfully - hash lookups will be faster"
                    )
            else:
                logger.warning("No unified snapshot found. Training will proceed without snapshot update.")
                logger.warning("Run data conversion and dataset creation first to create a snapshot.")
        
        # Train model
        _progress("Training…", None)
        logger.info("Starting training...")
        trained_model = self.framework_trainer.train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            hyperparams=hyperparams,
            output_dir=output_dir,
            resume_from_checkpoint=resume_from_checkpoint,
            cancel_event=cancel_event,
            progress_cb=_progress,
        )
        
        # Evaluate model
        _progress("Evaluating…", None)
        logger.info("Evaluating model...")
        metrics = self.framework_trainer.evaluate(trained_model, val_loader)
        logger.info("Evaluation metrics:")
        for metric_name, metric_value in metrics.items():
            logger.info(f"  {metric_name}: {metric_value:.4f}")
        
        # Update snapshot with training data
        if unified_snapshot and update_snapshot:
            logger.info("Updating unified snapshot with training data...")
            update_training_snapshot(data_dir, unified_snapshot)
            
            # Save updated snapshot
            snapshot_path = save_unified_snapshot(unified_snapshot, data_dir, logger)
            if snapshot_path:
                summary = unified_snapshot.to_dict().get('summary', {})
                train_total = summary.get('training_train_count', 0)
                test_total = summary.get('training_test_count', 0)
                logger.info(
                    f"Unified snapshot updated: {train_total} train, {test_total} test images"
                )
        
        # Save final model
        output_dir.mkdir(parents=True, exist_ok=True)
        model_name = f"{architecture}_model"
        
        if self.framework_name == 'pytorch':
            model_path = output_dir / f"{model_name}.pth"
            self.framework_trainer.save_model(trained_model, model_path, format="native")
        elif self.framework_name == 'keras':
            model_path = output_dir / f"{model_name}.h5"
            self.framework_trainer.save_model(trained_model, model_path, format="h5")
        else:
            raise ValueError(f"Unknown framework: {self.framework_name}")
        
        logger.info(f"Model saved to: {model_path}")
        
        return model_path
    
    def get_supported_architectures(self) -> list:
        """Get list of supported architectures for this framework."""
        return self.framework_trainer.get_supported_architectures()
