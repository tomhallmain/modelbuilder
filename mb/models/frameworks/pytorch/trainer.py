"""
PyTorch trainer implementation.

This module implements the FrameworkTrainer interface for PyTorch.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union
import json
import math
import threading
from datetime import datetime

from mb.cancellation import check_cancel_event
from mb.models.base import FrameworkTrainer
from mb.training.gui_progress import subepoch_progress_emit
from mb.models.frameworks.pytorch.data_loader import create_data_loaders
from mb.models.frameworks.pytorch.architectures import create_resnet, create_efficientnet
from mb.models.frameworks.pytorch.optimization import (
    autocast_context,
    build_one_cycle,
    build_optimizer,
    load_optimizer_moments,
    make_grad_scaler,
    scheduler_checkpoint_state,
    optimizer_steps_per_epoch,
    resolve_autocast_dtype,
    resolve_optimization_settings,
)
from mb.models.frameworks.pytorch.objectives import (
    ClassificationObjective,
    MultiLabelF1,
    SingleLabelAccuracy,
    build_multi_label_objective,
    build_single_label_objective,
)
from mb.models.frameworks.registry import get_architecture, list_architectures
from mb.models.types import ArchitectureType, ClassWeightingMode, FrameworkType, LabelMode
from mb.training.class_weights import DEFAULT_MAX_CLASS_WEIGHT, resolve_class_weights
from mb.utils.logging_setup import get_logger
from mb.utils.translations import _

logger = get_logger(__name__)

# Written to the run's output directory when training.restore_best is on.
BEST_WEIGHTS_FILENAME = "best_model_weights.pth"


class PyTorchTrainer(FrameworkTrainer):
    """
    PyTorch implementation of FrameworkTrainer.
    
    Supports transfer learning with frozen/unfrozen phases,
    learning rate scheduling, checkpointing, and evaluation.
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the PyTorch trainer.
        
        Args:
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        super().__init__(FrameworkType.PYTORCH)
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = torch.device(device)
        logger.info(f"PyTorch trainer initialized on device: {self.device}")
    
    def get_supported_architectures(self) -> list:
        """Get list of supported architectures."""
        fw = FrameworkType.PYTORCH
        return list_architectures(fw).get(fw.value, [])
    
    def create_model(
        self,
        architecture: Union[ArchitectureType, str],
        num_classes: int,
        pretrained: bool = True,
        **kwargs
    ) -> nn.Module:
        """
        Create a PyTorch model.
        
        Args:
            architecture: Architecture name (e.g., 'resnet34')
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            **kwargs: Additional architecture-specific arguments
            
        Returns:
            PyTorch model instance
        """
        arch_s = architecture.value if isinstance(architecture, ArchitectureType) else str(architecture)
        # Try to get from registry first
        factory = get_architecture(FrameworkType.PYTORCH, architecture)

        if factory:
            model = factory(num_classes=num_classes, pretrained=pretrained, **kwargs)
        else:
            # Fallback to direct creation
            if arch_s.startswith('resnet'):
                model = create_resnet(arch_s, num_classes, pretrained, **kwargs)
            elif arch_s.startswith('efficientnet'):
                model = create_efficientnet(arch_s, num_classes, pretrained, **kwargs)
            else:
                raise ValueError(f"Unknown architecture: {arch_s}")

        model = model.to(self.device)
        logger.info(
            _(
                "Created {arch_s} model with {num_classes} classes"
            ).format(arch_s=arch_s, num_classes=num_classes))
        
        return model
    
    def create_data_loaders(
        self,
        train_dir: Path,
        val_dir: Path,
        batch_size: int,
        image_size: int = 224,
        num_workers: int = 0,
        **kwargs
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create PyTorch data loaders.
        
        Args:
            train_dir: Path to training data directory
            val_dir: Path to validation/test data directory
            batch_size: Batch size for data loading
            image_size: Target image size (assumes square)
            num_workers: Number of worker processes for data loading
            **kwargs: Additional data loader arguments
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        return create_data_loaders(
            train_dir=train_dir,
            val_dir=val_dir,
            batch_size=batch_size,
            image_size=image_size,
            num_workers=num_workers,
            **kwargs
        )
    
    def train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        hyperparams: Dict[str, Any],
        output_dir: Path,
        resume_from_checkpoint: Optional[Path] = None,
        cancel_event: Optional[threading.Event] = None,
        progress_cb: Optional[Callable[[str, Optional[float]], None]] = None,
        **kwargs
    ) -> nn.Module:
        """
        Train the PyTorch model.
        
        Args:
            model: Model instance to train
            train_loader: Training data loader
            val_loader: Validation data loader
            hyperparams: Dictionary of hyperparameters
            output_dir: Directory to save checkpoints and final model
            resume_from_checkpoint: Optional path to checkpoint to resume from
            **kwargs: Additional training arguments
            
        Returns:
            Trained model instance
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract hyperparameters
        frozen_epochs = hyperparams.get('frozen_epochs', 5)
        unfrozen_epochs = hyperparams.get('unfrozen_epochs', 20)
        total_plan_epochs = frozen_epochs + unfrozen_epochs
        frozen_lr = hyperparams.get('frozen_lr', 0.001)
        unfrozen_lr_max = hyperparams.get('unfrozen_lr_max', 0.0003)
        unfrozen_lr_min = hyperparams.get('unfrozen_lr_min', 0.00001)
        unfrozen_pct_start = hyperparams.get('unfrozen_pct_start', 0.05)
        
        # Load checkpoint if resuming
        start_epoch = 0
        frozen_epochs_completed = 0
        unfrozen_epochs_completed = 0
        best_val_acc = 0.0
        # Checkpoint being resumed from; its optimizer, scheduler and scaler state are
        # restored into the phase it was saved in.
        resume_state: Optional[Dict[str, Any]] = None
        
        if resume_from_checkpoint and resume_from_checkpoint.exists():
            checkpoint = self._load_checkpoint(resume_from_checkpoint, model)
            if checkpoint:
                resume_state = checkpoint
                start_epoch = checkpoint.get('epoch', 0)
                frozen_epochs_completed = checkpoint.get('frozen_epochs_completed', 0)
                unfrozen_epochs_completed = checkpoint.get('unfrozen_epochs_completed', 0)
                best_val_acc = checkpoint.get('best_val_acc', 0.0)
                logger.info(
                    _(
                        "Resumed from checkpoint: epoch {start_epoch}, "
                        "frozen={frozen_epochs_completed}/{frozen_epochs}, "
                        "unfrozen={unfrozen_epochs_completed}/{unfrozen_epochs}"
                    ).format(
                        start_epoch=start_epoch,
                        frozen_epochs_completed=frozen_epochs_completed,
                        frozen_epochs=frozen_epochs,
                        unfrozen_epochs_completed=unfrozen_epochs_completed,
                        unfrozen_epochs=unfrozen_epochs,
                    ))
        
        # Loss, epoch metric, and weighting are chosen together: they are three views of the
        # same decision about what the output layer means. Weighting applies to the training
        # loss only, so validation loss stays comparable across weighting settings.
        objective = self._build_objective(train_loader, hyperparams)
        criterion = objective.criterion
        eval_criterion = objective.eval_criterion
        class_weights = objective.class_weights
        make_accumulator = objective.make_accumulator
        logger.info("Objective: %s, tracking %s", objective.label_mode.value, objective.primary_metric)

        settings = resolve_optimization_settings(hyperparams)
        autocast_dtype = resolve_autocast_dtype(settings.mixed_precision, self.device)
        accum = settings.grad_accum_steps
        logger.info(
            _(
                "Optimization: optimizer={optimizer}, weight_decay={wd}, layer_decay={ld}, "
                "grad_accum_steps={accum}, mixed_precision={precision}"
            ).format(
                optimizer=settings.optimizer.value,
                wd=settings.weight_decay,
                ld=settings.layer_decay,
                accum=accum,
                precision="fp32" if autocast_dtype is None else str(autocast_dtype).replace("torch.", ""),
            ))
        if settings.gradient_checkpointing:
            enable_gc = getattr(model, "enable_gradient_checkpointing", None)
            if callable(enable_gc) and enable_gc():
                logger.info(_("Gradient checkpointing enabled"))
            else:
                logger.warning(
                    _("Gradient checkpointing is not supported by this architecture; ignoring"))
        best_weights_path = output_dir / BEST_WEIGHTS_FILENAME
        # training.store_checkpoints: true keeps every epoch's checkpoint; false keeps only
        # the most recent one this run wrote, which is still enough for --resume-from.
        keep_all_checkpoints = bool(hyperparams.get('store_checkpoints', False))
        self._last_checkpoint_path = None
        epoch_kwargs = {"grad_accum_steps": accum, "autocast_dtype": autocast_dtype}

        # Phase 1: Frozen backbone training
        if frozen_epochs_completed < frozen_epochs:
            logger.info(
                _(
                    "Phase 1: Training with frozen backbone ({frozen_epochs} epochs)"
                ).format(frozen_epochs=frozen_epochs))
            
            # Freeze all layers except the classifier
            for param in model.parameters():
                param.requires_grad = False

            # Unfreeze the classification head. Models that name their own head explicitly
            # are asked directly; matching on 'fc'/'classifier' works for torchvision but
            # silently trains nothing on a backbone that names its head anything else.
            head_params = getattr(model, 'head_parameters', None)
            if callable(head_params):
                for param in head_params():
                    param.requires_grad = True
            elif hasattr(model, 'fc'):
                for param in model.fc.parameters():
                    param.requires_grad = True
            elif hasattr(model, 'classifier'):
                for param in model.classifier.parameters():
                    param.requires_grad = True
            else:
                raise ValueError(
                    f"Cannot locate a classification head to unfreeze on "
                    f"{type(model).__name__}; the frozen phase would train nothing"
                )

            optimizer, _group_lrs = build_optimizer(
                model, frozen_lr, settings, trainable_only=True
            )
            scaler = make_grad_scaler(autocast_dtype)
            if (
                resume_state is not None
                and resume_state.get('phase') == 'frozen'
                and frozen_epochs_completed > 0
            ):
                self._restore_training_state(optimizer, scaler, resume_state)
            
            remaining_frozen = frozen_epochs - frozen_epochs_completed
            frozen_start_completed = frozen_epochs_completed
            spe = len(train_loader) + len(val_loader)
            for epoch in range(remaining_frozen):
                check_cancel_event(cancel_event)
                epoch_num = frozen_start_completed + epoch + 1
                logger.info(f"Frozen epoch {epoch_num}/{frozen_epochs}")
                epochs_done_before = frozen_start_completed + epoch
                emit = subepoch_progress_emit(
                    progress_cb,
                    total_plan_epochs,
                    epochs_done_before,
                    spe,
                    f"Frozen phase: epoch {epoch_num}/{frozen_epochs}",
                )

                # Train
                train_loss, train_acc = self._train_epoch(
                    model, train_loader, criterion, optimizer, epoch_num,
                    cancel_event=cancel_event,
                    on_batch_step=emit,
                    make_accumulator=make_accumulator,
                    scaler=scaler,
                    **epoch_kwargs,
                )

                # Validate
                val_loss, val_acc = self._validate(
                    model, val_loader, eval_criterion,
                    cancel_event=cancel_event,
                    train_step_count=len(train_loader),
                    on_batch_step=emit,
                    make_accumulator=make_accumulator,
                    autocast_dtype=autocast_dtype,
                )
                
                logger.info(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                logger.info(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                
                frozen_epochs_completed += 1
                if settings.restore_best and val_acc > best_val_acc:
                    self._save_best_weights(model, best_weights_path, val_acc, epoch_num)
                best_val_acc = max(best_val_acc, val_acc)
                
                # Save checkpoint
                self._save_checkpoint(
                    model, optimizer, output_dir,
                    epoch=epoch_num,
                    frozen_epochs_completed=frozen_epochs_completed,
                    unfrozen_epochs_completed=unfrozen_epochs_completed,
                    best_val_acc=best_val_acc,
                    phase='frozen',
                    class_weights=class_weights,
                    primary_metric=objective.primary_metric,
                    keep_previous=keep_all_checkpoints,
                    scaler=scaler,
                )
        else:
            logger.info(_("Frozen phase already completed, skipping"))
        
        # Phase 2: Unfrozen fine-tuning
        if unfrozen_epochs_completed < unfrozen_epochs:
            logger.info(
                _(
                    "Phase 2: Fine-tuning all layers ({unfrozen_epochs} epochs)"
                ).format(unfrozen_epochs=unfrozen_epochs))
            
            # Unfreeze all layers
            for param in model.parameters():
                param.requires_grad = True
            
            # OneCycle over the whole fine-tuning phase. A resume continues the saved schedule
            # when it was planned over the same number of steps; otherwise (an older
            # checkpoint, or changed epochs, batch size or accumulation) it restarts over the
            # remaining epochs.
            steps_per_epoch = optimizer_steps_per_epoch(len(train_loader), accum)
            full_total = steps_per_epoch * unfrozen_epochs
            resuming_unfrozen = (
                resume_state is not None
                and resume_state.get('phase') == 'unfrozen'
                and unfrozen_epochs_completed > 0
            )
            saved_schedule = (
                resume_state.get('scheduler_state_dict') if resuming_unfrozen else None
            )
            continue_schedule = (
                isinstance(saved_schedule, dict)
                and saved_schedule.get('total_steps') == full_total
            )
            total_steps = (
                full_total
                if continue_schedule
                else steps_per_epoch * (unfrozen_epochs - unfrozen_epochs_completed)
            )
            optimizer, group_lrs = build_optimizer(
                model, unfrozen_lr_max, settings, apply_layer_decay=True
            )
            scaler = make_grad_scaler(autocast_dtype)
            scheduler = build_one_cycle(
                optimizer,
                group_lrs,
                unfrozen_lr_max,
                unfrozen_lr_min,
                total_steps,
                unfrozen_pct_start,
            )
            if resuming_unfrozen:
                self._restore_training_state(optimizer, scaler, resume_state)
                if continue_schedule:
                    scheduler.load_state_dict(saved_schedule)
                    # OneCycle sets both the rate and Adam's beta1 (momentum) per step, and the
                    # groups still hold step-0 values. Re-taking the saved step through step()
                    # recomputes both from this run's settings; the restored step count keeps
                    # it from warning that the optimizer has not stepped yet.
                    scheduler.last_epoch -= 1
                    scheduler.step()
                    logger.info(
                        _("Resumed the learning-rate schedule at step {step}/{total}").format(
                            step=scheduler.last_epoch, total=full_total))
                else:
                    logger.info(
                        _("The saved learning-rate schedule does not match this run; "
                          "restarting it over the remaining {n} epochs").format(
                            n=unfrozen_epochs - unfrozen_epochs_completed))
            
            remaining_unfrozen = unfrozen_epochs - unfrozen_epochs_completed
            unfrozen_start_completed = unfrozen_epochs_completed
            spe = len(train_loader) + len(val_loader)
            for epoch in range(remaining_unfrozen):
                check_cancel_event(cancel_event)
                epoch_num = unfrozen_start_completed + epoch + 1
                logger.info(f"Unfrozen epoch {epoch_num}/{unfrozen_epochs}")
                epochs_done_before = frozen_epochs_completed + unfrozen_start_completed + epoch
                emit = subepoch_progress_emit(
                    progress_cb,
                    total_plan_epochs,
                    epochs_done_before,
                    spe,
                    f"Fine-tune: epoch {epoch_num}/{unfrozen_epochs}",
                )

                # Train
                train_loss, train_acc = self._train_epoch(
                    model, train_loader, criterion, optimizer, epoch_num, scheduler,
                    cancel_event=cancel_event,
                    on_batch_step=emit,
                    make_accumulator=make_accumulator,
                    scaler=scaler,
                    **epoch_kwargs,
                )

                # Validate
                val_loss, val_acc = self._validate(
                    model, val_loader, eval_criterion,
                    cancel_event=cancel_event,
                    train_step_count=len(train_loader),
                    on_batch_step=emit,
                    make_accumulator=make_accumulator,
                    autocast_dtype=autocast_dtype,
                )
                
                logger.info(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                logger.info(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                logger.info(f"  LR: {scheduler.get_last_lr()[-1]:.6f}")
                
                unfrozen_epochs_completed += 1
                if settings.restore_best and val_acc > best_val_acc:
                    self._save_best_weights(
                        model, best_weights_path, val_acc, frozen_epochs + epoch_num
                    )
                best_val_acc = max(best_val_acc, val_acc)
                
                # Save checkpoint
                self._save_checkpoint(
                    model, optimizer, output_dir,
                    epoch=frozen_epochs + epoch_num,
                    frozen_epochs_completed=frozen_epochs_completed,
                    unfrozen_epochs_completed=unfrozen_epochs_completed,
                    best_val_acc=best_val_acc,
                    phase='unfrozen',
                    class_weights=class_weights,
                    primary_metric=objective.primary_metric,
                    keep_previous=keep_all_checkpoints,
                    scheduler=scheduler,
                    scaler=scaler,
                )
        else:
            logger.info(_("Unfrozen phase already completed, skipping"))
        
        logger.info(
            _(
                "Training completed. Best validation accuracy: {best_val_acc:.4f}"
            ).format(best_val_acc=best_val_acc))

        if settings.restore_best:
            self._restore_best_weights(model, best_weights_path, best_val_acc)

        return model

    def _restore_training_state(
        self, optimizer: optim.Optimizer, scaler: Optional[Any], state: Dict[str, Any]
    ) -> bool:
        """
        Restore optimizer moments and fp16 scaler state saved in a checkpoint.

        Returns False, with a warning, when the saved optimizer does not fit this run's
        parameter groups (a different model, or optimizer options changed between runs);
        training then continues with fresh moments, as an older checkpoint does.
        """
        saved = state.get('optimizer_state_dict')
        restored = False
        if saved:
            try:
                load_optimizer_moments(optimizer, saved)
                restored = True
            except (ValueError, KeyError, RuntimeError) as e:
                logger.warning(
                    _("Could not restore optimizer state from the checkpoint ({err}); "
                      "continuing with a fresh optimizer").format(err=e))
        scaler_state = state.get('scaler_state_dict')
        if scaler is not None and scaler_state:
            try:
                scaler.load_state_dict(scaler_state)
            except (ValueError, KeyError, RuntimeError) as e:
                logger.warning(
                    _("Could not restore the fp16 loss scaler ({err})").format(err=e))
        return restored

    def _save_best_weights(
        self, model: nn.Module, path: Path, metric: float, epoch: int
    ) -> None:
        """Write the weights of the best epoch so far, with the metric that earned it."""
        torch.save(
            {"model_state_dict": model.state_dict(), "best_val_acc": float(metric), "epoch": epoch},
            path,
        )
        logger.info(
            _("New best validation metric {metric:.4f} at epoch {epoch}; saved {path}").format(
                metric=metric, epoch=epoch, path=path))

    def _restore_best_weights(self, model: nn.Module, path: Path, best_val_acc: float) -> None:
        """
        Load the best epoch's weights into *model*, keeping the final weights if unavailable.

        The stored metric must equal this run's best: a file left by an earlier run in the
        same output directory, or a resume into a directory that never saw the best epoch,
        would otherwise restore weights this run did not produce.
        """
        if not path.is_file():
            logger.warning(
                _("No best-epoch weights at {path}; keeping the final epoch's weights").format(
                    path=path))
            return
        saved = torch.load(path, map_location=self.device)
        stored = float(saved.get("best_val_acc", float("nan")))
        if not math.isclose(stored, float(best_val_acc), rel_tol=0.0, abs_tol=1e-9):
            logger.warning(
                _(
                    "Best-epoch weights at {path} record {stored:.4f}, not this run's best "
                    "{best:.4f}; keeping the final epoch's weights"
                ).format(path=path, stored=stored, best=best_val_acc))
            return
        model.load_state_dict(saved["model_state_dict"])
        logger.info(
            _("Restored best-epoch weights (epoch {epoch}, metric {metric:.4f})").format(
                epoch=saved.get("epoch"), metric=stored))
    
    def evaluate(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        **kwargs
    ) -> Dict[str, float]:
        """
        Evaluate the model on validation data.
        
        Args:
            model: Model instance to evaluate
            val_loader: Validation data loader
            **kwargs: Additional evaluation arguments
            
        Returns:
            Dictionary of metric names to values. The metric key follows the label mode:
            ``accuracy`` for single-label, ``micro_f1`` for multi-label, since accuracy over
            independent labels is dominated by correctly predicted negatives.
        """
        model.eval()

        # Detected from the dataset rather than passed in: the loader already knows which
        # kind of target it produces, and a mismatch here would apply the wrong loss.
        schema = getattr(getattr(val_loader, "dataset", None), "schema", None)
        if schema is not None:
            criterion: nn.Module = nn.BCEWithLogitsLoss()
            thresholds = schema.thresholds()
            make_accumulator: Optional[Callable[[], Any]] = lambda: MultiLabelF1(thresholds)
            metric_name = MultiLabelF1.name
        else:
            criterion = nn.CrossEntropyLoss()
            make_accumulator = None
            metric_name = SingleLabelAccuracy.name

        cancel_event: Optional[threading.Event] = kwargs.get("cancel_event")
        progress_cb: Optional[Callable[[str, Optional[float]], None]] = kwargs.get("progress_cb")
        total_steps = max(len(val_loader), 1)

        def _emit_eval_step(step: int) -> None:
            if progress_cb is None:
                return
            pct = min(max(step / total_steps, 0.0), 1.0)
            progress_cb(f"Evaluating… — batch {step}/{total_steps}", pct)

        val_loss, val_metric = self._validate(
            model,
            val_loader,
            criterion,
            cancel_event=cancel_event,
            train_step_count=0,
            on_batch_step=_emit_eval_step,
            make_accumulator=make_accumulator,
        )

        return {
            'loss': val_loss,
            metric_name: val_metric,
        }
    
    def save_model(
        self,
        model: nn.Module,
        path: Path,
        format: str = "native",
        **kwargs
    ) -> Path:
        """
        Save the PyTorch model.
        
        Args:
            model: Model instance to save
            path: Path to save the model
            format: Format to save in ('native' for .pth, 'onnx' for ONNX)
            **kwargs: Additional save arguments
            
        Returns:
            Path where model was saved
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "native" or format == "pth":
            # Save PyTorch state dict
            torch.save(model.state_dict(), path)
            logger.info(_("Saved PyTorch model to {path}").format(path=path))
        elif format == "onnx":
            from mb.conversion.converters import export_torch_module_to_onnx

            from mb.models.preprocessing import default_image_size_for

            model.eval()
            # Hugging Face wrappers record their architecture; torchvision models do not, and
            # keep the 224 default unless image_size is passed.
            image_size = kwargs.get("image_size") or default_image_size_for(
                getattr(model, "architecture", None), 224
            )
            export_torch_module_to_onnx(model, path, image_size=int(image_size))
            logger.info(_("Exported model to ONNX: {path}").format(path=path))
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return path
    
    def load_model(
        self,
        path: Path,
        architecture: str,
        num_classes: int,
        **kwargs
    ) -> nn.Module:
        """
        Load a saved PyTorch model.
        
        Args:
            path: Path to saved model
            architecture: Model architecture (needed to rebuild model)
            num_classes: Number of output classes
            **kwargs: Additional load arguments
            
        Returns:
            Loaded model instance
        """
        model = self.create_model(architecture, num_classes, pretrained=False)
        model.load_state_dict(torch.load(path, map_location=self.device))
        model = model.to(self.device)
        logger.info(_("Loaded model from {path}").format(path=path))
        return model
    
    def _build_objective(
        self,
        train_loader: DataLoader,
        hyperparams: Dict[str, Any],
    ) -> ClassificationObjective:
        """
        Pick the loss and epoch metric for this run's label mode.

        Multi-label needs the training split's per-label positive counts and the schema's
        thresholds, both of which the dataset already carries; single-label resolves the
        class weights exactly as before.
        """
        label_mode = LabelMode.try_from(hyperparams.get('label_mode')) or LabelMode.get_default()
        dataset = getattr(train_loader, 'dataset', None)
        max_weight = float(hyperparams.get('class_weight_max') or DEFAULT_MAX_CLASS_WEIGHT)

        if label_mode == LabelMode.MULTI_LABEL:
            counts_fn = getattr(dataset, 'label_positive_counts', None)
            schema = getattr(dataset, 'schema', None)
            if counts_fn is None or schema is None:
                raise ValueError(
                    "Multi-label training needs a multi-label dataset; "
                    f"got {type(dataset).__name__}"
                )
            positive_counts = counts_fn()
            weighting = (
                ClassWeightingMode.try_from(hyperparams.get('class_weighting'))
                or ClassWeightingMode.get_default()
            )
            for name, count in zip(schema.labels, positive_counts):
                logger.info("  label %s: %d positives", name, count)
            return build_multi_label_objective(
                positive_counts=positive_counts,
                n_samples=len(dataset),
                thresholds=schema.thresholds(),
                device=self.device,
                max_weight=max_weight,
                weighted=weighting == ClassWeightingMode.INVERSE_FREQUENCY,
            )

        class_weights = resolve_class_weights(
            train_loader,
            ClassWeightingMode.try_from(hyperparams.get('class_weighting'))
            or ClassWeightingMode.get_default(),
            max_weight=max_weight,
            class_names=getattr(dataset, 'classes', None),
        )
        return build_single_label_objective(class_weights, self.device)

    def _train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int,
        scheduler: Optional[Any] = None,
        cancel_event: Optional[threading.Event] = None,
        on_batch_step: Optional[Callable[[int], None]] = None,
        make_accumulator: Optional[Callable[[], Any]] = None,
        grad_accum_steps: int = 1,
        autocast_dtype: Optional[torch.dtype] = None,
        scaler: Optional[Any] = None,
    ) -> Tuple[float, float]:
        """Train for one epoch.

        *make_accumulator* builds the epoch metric; the default is top-1 accuracy, which is
        what the metric was before label mode became selectable.

        With *grad_accum_steps* > 1, gradients from that many batches are summed before each
        optimizer (and scheduler) step; a trailing partial group still steps, which is what
        :func:`optimizer_steps_per_epoch` counts. *autocast_dtype* enables mixed precision,
        and *scaler* is the fp16 loss scaler.
        """
        model.train()
        running_loss = 0.0
        metric = (make_accumulator or SingleLabelAccuracy)()
        accum = max(1, int(grad_accum_steps))
        num_batches = len(train_loader)

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            check_cancel_event(cancel_event)
            inputs = inputs.to(self.device)
            if not torch.is_tensor(targets):
                targets = torch.as_tensor(targets)
            targets = targets.to(self.device)

            # Forward pass
            if batch_idx % accum == 0:
                optimizer.zero_grad()
            with autocast_context(self.device, autocast_dtype):
                outputs = model(inputs)
            if autocast_dtype is not None:
                outputs = outputs.float()
            loss = criterion(outputs, targets)

            # Backward pass. The loss is averaged over the batches in this accumulation
            # group, so the summed gradient matches one batch of the combined size.
            group_size = min(accum, num_batches - (batch_idx // accum) * accum)
            backward_loss = loss / group_size if group_size > 1 else loss
            if scaler is not None:
                scaler.scale(backward_loss).backward()
            else:
                backward_loss.backward()

            if (batch_idx + 1) % accum == 0 or batch_idx + 1 == num_batches:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                # Update scheduler if provided
                if scheduler:
                    scheduler.step()

            # Statistics
            running_loss += loss.item()
            metric.update(outputs, targets)
            if on_batch_step is not None:
                on_batch_step(batch_idx + 1)

        epoch_loss = running_loss / max(len(train_loader), 1)
        epoch_acc = metric.value()

        return epoch_loss, epoch_acc
    
    def _validate(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        criterion: nn.Module,
        cancel_event: Optional[threading.Event] = None,
        train_step_count: int = 0,
        on_batch_step: Optional[Callable[[int], None]] = None,
        make_accumulator: Optional[Callable[[], Any]] = None,
        autocast_dtype: Optional[torch.dtype] = None,
    ) -> Tuple[float, float]:
        """Validate the model (under autocast when *autocast_dtype* is set)."""
        model.eval()
        running_loss = 0.0
        metric = (make_accumulator or SingleLabelAccuracy)()

        step = train_step_count
        with torch.no_grad():
            for inputs, targets in val_loader:
                check_cancel_event(cancel_event)
                inputs = inputs.to(self.device)
                if not torch.is_tensor(targets):
                    targets = torch.as_tensor(targets)
                targets = targets.to(self.device)

                with autocast_context(self.device, autocast_dtype):
                    outputs = model(inputs)
                if autocast_dtype is not None:
                    outputs = outputs.float()
                loss = criterion(outputs, targets)

                running_loss += loss.item()
                metric.update(outputs, targets)
                step += 1
                if on_batch_step is not None:
                    on_batch_step(step)

        epoch_loss = running_loss / max(len(val_loader), 1)
        epoch_acc = metric.value()

        return epoch_loss, epoch_acc
    
    def _save_checkpoint(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        output_dir: Path,
        epoch: int,
        frozen_epochs_completed: int,
        unfrozen_epochs_completed: int,
        best_val_acc: float,
        phase: str,
        class_weights: Optional[list] = None,
        primary_metric: str = SingleLabelAccuracy.name,
        keep_previous: bool = True,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
    ):
        """Save a training checkpoint.

        The optimizer, *scheduler* and fp16 *scaler* states are stored so a resume continues
        where this epoch left off rather than restarting warmup with fresh optimizer moments.

        With *keep_previous* False, the checkpoint this run wrote before this one is deleted
        once the new one is on disk, so at most one is kept and an interrupted write never
        leaves none. Only files written by this run are removed: a checkpoint resumed from,
        or left by an earlier run, is never touched.

        ``best_val_acc`` keeps its name so older checkpoints still resume, but its meaning
        now depends on the label mode. ``primary_metric`` records which meaning applies, so
        a reader is never left guessing whether a stored figure is accuracy or F1.
        """
        # A loss recorded under class weighting is not comparable to one recorded without
        # it, so the weights that produced these numbers travel with them.
        weights = [float(w) for w in class_weights] if class_weights else None
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': (
                scheduler_checkpoint_state(scheduler) if scheduler is not None else None
            ),
            'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
            'frozen_epochs_completed': frozen_epochs_completed,
            'unfrozen_epochs_completed': unfrozen_epochs_completed,
            'best_val_acc': best_val_acc,
            'phase': phase,
            'class_weights': weights,
            'primary_metric': primary_metric,
            'saved_at': datetime.now().isoformat()
        }

        # Save model checkpoint
        checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)

        # Save human-readable metadata only (state dicts contain Tensors and are not JSON-serializable)
        metadata = {
            'epoch': epoch,
            'frozen_epochs_completed': frozen_epochs_completed,
            'unfrozen_epochs_completed': unfrozen_epochs_completed,
            'best_val_acc': float(best_val_acc),
            'phase': phase,
            'class_weights': weights,
            'primary_metric': primary_metric,
            'saved_at': checkpoint['saved_at'],
        }
        metadata_path = checkpoint_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.debug(f"Saved checkpoint: {checkpoint_path}")

        previous = getattr(self, "_last_checkpoint_path", None)
        if not keep_previous and previous is not None and previous != checkpoint_path:
            for stale in (previous, previous.with_suffix('.json')):
                try:
                    stale.unlink(missing_ok=True)
                except OSError as e:
                    logger.warning(
                        _("Could not remove previous checkpoint {path}: {err}").format(
                            path=stale, err=e))
        self._last_checkpoint_path = checkpoint_path
    
    def _load_checkpoint(
        self,
        checkpoint_path: Path,
        model: nn.Module
    ) -> Optional[Dict]:
        """Load a training checkpoint."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
            return checkpoint
        except Exception as e:
            logger.error(_("Failed to load checkpoint: {e}").format(e=e))
            return None
