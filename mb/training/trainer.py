"""
Generic training orchestrator.

This module provides a framework-agnostic interface for training models.
"""

from pathlib import Path
from typing import Any, Callable, Optional
import threading
import time
from datetime import datetime, timezone

from mb.models.base import FrameworkTrainer
from mb.models.types import FrameworkType, ModelType, get_model_type_handler
from mb.models.frameworks.pytorch.trainer import PyTorchTrainer
from mb.models.frameworks.keras.trainer import KerasTrainer
from mb.training.hyperparams import get_training_hyperparams
from mb.training.run_args import TrainingRunArgs
from mb.training.snapshot_integration import update_training_snapshot
from mb.utils.constants import ModelBuilderTaskType
from mb.utils.logging_setup import get_logger
from mb.utils.snapshot import (
    find_unified_snapshot,
    preload_gather_cache,
    save_unified_snapshot,
    set_step_errors_for_invocation,
)

logger = get_logger(__name__)


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
        framework: FrameworkType,
        model_type: ModelType = ModelType.IMAGE_CLASSIFICATION,
        pipeline_config: Optional[Any] = None,
    ):
        """
        Initialize the model trainer.
        
        Args:
            framework: Training framework
            model_type: Model type enum
            pipeline_config: Optional :class:`~mb.pipeline_config.PipelineConfig`
        """
        self.framework = framework
        self.model_type = model_type
        self.pipeline_config = pipeline_config

        # Get framework trainer
        if framework == FrameworkType.PYTORCH:
            self.framework_trainer: FrameworkTrainer = PyTorchTrainer()
        elif framework == FrameworkType.KERAS:
            self.framework_trainer = KerasTrainer()
        else:
            raise ValueError(f"Unsupported framework: {framework!r}")

        # Get model type handler
        self.model_type_handler = get_model_type_handler(model_type)

        logger.info(
            "Initialized ModelTrainer: framework=%s, model_type=%s",
            framework.value,
            model_type.value,
        )
    
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
            progress_cb: Optional callback ``(message, percent_or_none)`` for GUI progress.
                *percent* is an overall job fraction in ``[0, 1]`` when known (setup, training,
                and evaluation use heuristic bands; the training loop refines the middle band).

        Returns:
            Path to saved model
        """
        if args.framework != self.framework:
            raise ValueError(
                f"TrainingRunArgs.framework ({args.framework!r}) does not match "
                f"this trainer ({self.framework!r})"
            )

        data_dir = args.data_dir
        architecture = args.architecture
        output_dir = args.output_dir
        resume_from_checkpoint = args.resume_from
        run_id = args.run_id
        update_snapshot = args.update_snapshot
        cli_hyperparams = args.cli_hyperparams

        def _emit(msg: str, pct: Optional[float]) -> None:
            if progress_cb is not None:
                progress_cb(msg, pct)

        # Overall [0, 1] job fraction for the GUI (heuristic: most time is in the framework
        # training loop; setup, snapshot load, and post-train eval are smaller bands).
        p_after_validate = 0.04
        p_after_model = 0.09
        p_after_data = 0.16
        if update_snapshot:
            p_after_snapshot = 0.20
        else:
            p_after_snapshot = p_after_data

        # Validate data structure
        _emit("Validating data…", p_after_validate)
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
        
        # Set defaults and normalize optional/nullable values from config/UI.
        image_size = hyperparams.get('image_size', 224)
        try:
            batch_size = int(hyperparams.get('batch_size') or 32)
        except (TypeError, ValueError):
            batch_size = 32
        if batch_size < 1:
            batch_size = 32
        try:
            num_workers = int(hyperparams.get('num_workers') or 0)
        except (TypeError, ValueError):
            num_workers = 0
        if num_workers < 0:
            num_workers = 0
        
        logger.info("Hyperparameters:")
        for key, value in hyperparams.items():
            logger.info(f"  {key}: {value}")
        
        # Create model
        _emit("Creating model…", p_after_model)
        logger.info(f"Creating {architecture.value} model...")
        model = self.framework_trainer.create_model(
            architecture=architecture,
            num_classes=num_classes,
            pretrained=True
        )
        
        # Create data loaders
        train_dir = data_dir / "train"
        val_dir = data_dir / "test"
        
        _emit("Loading data…", p_after_data)
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
        train_invocation_started: Optional[str] = None
        if update_snapshot:
            train_invocation_started = datetime.now(timezone.utc).isoformat()
            _emit("Loading snapshot…", p_after_snapshot)
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

        train_lo = p_after_snapshot
        train_hi = 0.88
        eval_end = 0.96

        def _training_progress(msg: str, pct: Optional[float]) -> None:
            if pct is None:
                _emit(msg, None)
            else:
                overall = train_lo + (train_hi - train_lo) * float(pct)
                _emit(msg, overall)

        def _evaluate_progress(msg: str, pct: Optional[float]) -> None:
            if pct is None:
                _emit(msg, None)
            else:
                overall = train_hi + (eval_end - train_hi) * float(pct)
                _emit(msg, overall)

        snapshot_end = 0.99

        def _snapshot_progress(msg: str, pct: Optional[float]) -> None:
            if pct is None:
                _emit(msg, None)
            else:
                overall = eval_end + (snapshot_end - eval_end) * float(pct)
                _emit(msg, overall)

        # Train model (framework reports 0..1 within the training loop only)
        _emit("Training…", train_lo)
        logger.info("Starting training...")
        t_train0 = time.perf_counter()
        trained_model = self.framework_trainer.train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            hyperparams=hyperparams,
            output_dir=output_dir,
            resume_from_checkpoint=resume_from_checkpoint,
            cancel_event=cancel_event,
            progress_cb=_training_progress,
        )
        t_train1 = time.perf_counter()

        # Evaluate model
        _emit("Evaluating…", train_hi)
        logger.info("Evaluating model...")
        metrics = self.framework_trainer.evaluate(
            trained_model,
            val_loader,
            cancel_event=cancel_event,
            progress_cb=_evaluate_progress,
        )
        t_eval1 = time.perf_counter()
        _emit("Evaluating…", eval_end)
        logger.info("Evaluation metrics:")
        for metric_name, metric_value in metrics.items():
            logger.info(f"  {metric_name}: {metric_value:.4f}")
        
        # Update snapshot with training data
        if unified_snapshot and update_snapshot:
            train_s = max(0.0, t_train1 - t_train0)
            eval_s = max(0.0, t_eval1 - t_train1)
            unified_snapshot.training_timing = {
                "version": 1,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "framework": self.framework.value,
                "architecture": architecture.value,
                "model_type": self.model_type.value,
                "seconds": {
                    "train": round(train_s, 4),
                    "evaluate": round(eval_s, 4),
                    "total": round(train_s + eval_s, 4),
                },
            }
            logger.info("Updating unified snapshot with training data...")
            update_training_snapshot(
                data_dir,
                unified_snapshot,
                cancel_event=cancel_event,
                progress_cb=_snapshot_progress,
            )

            if train_invocation_started:
                set_step_errors_for_invocation(
                    unified_snapshot,
                    ModelBuilderTaskType.TRAIN.value,
                    train_invocation_started,
                    [],
                )

            # Save updated snapshot
            snapshot_path = save_unified_snapshot(unified_snapshot, data_dir, logger)
            if snapshot_path:
                summary = unified_snapshot.to_dict().get('summary', {})
                train_total = summary.get('training_train_count', 0)
                test_total = summary.get('training_test_count', 0)
                logger.info(
                    f"Unified snapshot updated: {train_total} train, {test_total} test images"
                )
            _emit("Updating snapshot…", snapshot_end)
        
        # Save final model
        output_dir.mkdir(parents=True, exist_ok=True)
        model_name = f"{architecture.value}_model"

        if self.framework == FrameworkType.PYTORCH:
            model_path = output_dir / f"{model_name}.pth"
            self.framework_trainer.save_model(trained_model, model_path, format="native")
        elif self.framework == FrameworkType.KERAS:
            model_path = output_dir / f"{model_name}.h5"
            self.framework_trainer.save_model(trained_model, model_path, format="h5")
        else:
            raise ValueError(f"Unknown framework: {self.framework!r}")
        
        logger.info(f"Model saved to: {model_path}")
        
        return model_path
    
    def get_supported_architectures(self) -> list:
        """Get list of supported architectures for this framework."""
        return self.framework_trainer.get_supported_architectures()
