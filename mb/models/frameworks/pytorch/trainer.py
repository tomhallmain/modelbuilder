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
import logging
import threading
from datetime import datetime

from mb.cancellation import check_cancel_event
from mb.models.base import FrameworkTrainer
from mb.training.gui_progress import subepoch_progress_emit
from mb.models.frameworks.pytorch.data_loader import create_data_loaders
from mb.models.frameworks.pytorch.architectures import create_resnet, create_efficientnet
from mb.models.frameworks.registry import get_architecture, list_architectures
from mb.models.types import ArchitectureType, FrameworkType

logger = logging.getLogger(__name__)


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
        
        if resume_from_checkpoint and resume_from_checkpoint.exists():
            checkpoint = self._load_checkpoint(resume_from_checkpoint, model)
            if checkpoint:
                start_epoch = checkpoint.get('epoch', 0)
                frozen_epochs_completed = checkpoint.get('frozen_epochs_completed', 0)
                unfrozen_epochs_completed = checkpoint.get('unfrozen_epochs_completed', 0)
                best_val_acc = checkpoint.get('best_val_acc', 0.0)
                logger.info(f"Resumed from checkpoint: epoch {start_epoch}, "
                          f"frozen={frozen_epochs_completed}/{frozen_epochs}, "
                          f"unfrozen={unfrozen_epochs_completed}/{unfrozen_epochs}")
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        
        # Phase 1: Frozen backbone training
        if frozen_epochs_completed < frozen_epochs:
            logger.info(f"Phase 1: Training with frozen backbone ({frozen_epochs} epochs)")
            
            # Freeze all layers except the classifier
            for param in model.parameters():
                param.requires_grad = False
            
            # Unfreeze classifier
            if hasattr(model, 'fc'):
                for param in model.fc.parameters():
                    param.requires_grad = True
            elif hasattr(model, 'classifier'):
                for param in model.classifier.parameters():
                    param.requires_grad = True
            
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=frozen_lr
            )
            
            remaining_frozen = frozen_epochs - frozen_epochs_completed
            spe = len(train_loader) + len(val_loader)
            for epoch in range(remaining_frozen):
                check_cancel_event(cancel_event)
                epoch_num = frozen_epochs_completed + epoch + 1
                logger.info(f"Frozen epoch {epoch_num}/{frozen_epochs}")
                epochs_done_before = frozen_epochs_completed + epoch
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
                )

                # Validate
                val_loss, val_acc = self._validate(
                    model, val_loader, criterion,
                    cancel_event=cancel_event,
                    train_step_count=len(train_loader),
                    on_batch_step=emit,
                )
                
                logger.info(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                logger.info(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                
                frozen_epochs_completed += 1
                best_val_acc = max(best_val_acc, val_acc)
                
                # Save checkpoint
                self._save_checkpoint(
                    model, optimizer, output_dir,
                    epoch=epoch_num,
                    frozen_epochs_completed=frozen_epochs_completed,
                    unfrozen_epochs_completed=unfrozen_epochs_completed,
                    best_val_acc=best_val_acc,
                    phase='frozen'
                )
        else:
            logger.info("Frozen phase already completed, skipping")
        
        # Phase 2: Unfrozen fine-tuning
        if unfrozen_epochs_completed < unfrozen_epochs:
            logger.info(f"Phase 2: Fine-tuning all layers ({unfrozen_epochs} epochs)")
            
            # Unfreeze all layers
            for param in model.parameters():
                param.requires_grad = True
            
            # Use OneCycleLR scheduler for learning rate scheduling
            total_steps = len(train_loader) * (unfrozen_epochs - unfrozen_epochs_completed)
            optimizer = optim.Adam(model.parameters(), lr=unfrozen_lr_max)
            scheduler = OneCycleLR(
                optimizer,
                max_lr=unfrozen_lr_max,
                total_steps=total_steps,
                pct_start=unfrozen_pct_start,
                anneal_strategy='cos'
            )
            
            remaining_unfrozen = unfrozen_epochs - unfrozen_epochs_completed
            spe = len(train_loader) + len(val_loader)
            for epoch in range(remaining_unfrozen):
                check_cancel_event(cancel_event)
                epoch_num = unfrozen_epochs_completed + epoch + 1
                logger.info(f"Unfrozen epoch {epoch_num}/{unfrozen_epochs}")
                epochs_done_before = frozen_epochs_completed + unfrozen_epochs_completed + epoch
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
                )

                # Validate
                val_loss, val_acc = self._validate(
                    model, val_loader, criterion,
                    cancel_event=cancel_event,
                    train_step_count=len(train_loader),
                    on_batch_step=emit,
                )
                
                logger.info(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                logger.info(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                logger.info(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
                
                unfrozen_epochs_completed += 1
                best_val_acc = max(best_val_acc, val_acc)
                
                # Save checkpoint
                self._save_checkpoint(
                    model, optimizer, output_dir,
                    epoch=frozen_epochs + epoch_num,
                    frozen_epochs_completed=frozen_epochs_completed,
                    unfrozen_epochs_completed=unfrozen_epochs_completed,
                    best_val_acc=best_val_acc,
                    phase='unfrozen'
                )
        else:
            logger.info("Unfrozen phase already completed, skipping")
        
        logger.info(f"Training completed. Best validation accuracy: {best_val_acc:.4f}")
        
        return model
    
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
            Dictionary of metric names to values
        """
        model.eval()
        criterion = nn.CrossEntropyLoss()
        
        val_loss, val_acc = self._validate(model, val_loader, criterion)
        
        return {
            'loss': val_loss,
            'accuracy': val_acc
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
            logger.info(f"Saved PyTorch model to {path}")
        elif format == "onnx":
            # Export to ONNX
            model.eval()
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            torch.onnx.export(
                model,
                dummy_input,
                path,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )
            logger.info(f"Exported model to ONNX: {path}")
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
        logger.info(f"Loaded model from {path}")
        return model
    
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
    ) -> Tuple[float, float]:
        """Train for one epoch."""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            check_cancel_event(cancel_event)
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Update scheduler if provided
            if scheduler:
                scheduler.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if on_batch_step is not None:
                on_batch_step(batch_idx + 1)

        epoch_loss = running_loss / max(len(train_loader), 1)
        epoch_acc = 100.0 * correct / max(total, 1)

        return epoch_loss, epoch_acc
    
    def _validate(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        criterion: nn.Module,
        cancel_event: Optional[threading.Event] = None,
        train_step_count: int = 0,
        on_batch_step: Optional[Callable[[int], None]] = None,
    ) -> Tuple[float, float]:
        """Validate the model."""
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        step = train_step_count
        with torch.no_grad():
            for inputs, targets in val_loader:
                check_cancel_event(cancel_event)
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                step += 1
                if on_batch_step is not None:
                    on_batch_step(step)

        epoch_loss = running_loss / max(len(val_loader), 1)
        epoch_acc = 100.0 * correct / max(total, 1)

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
        phase: str
    ):
        """Save a training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'frozen_epochs_completed': frozen_epochs_completed,
            'unfrozen_epochs_completed': unfrozen_epochs_completed,
            'best_val_acc': best_val_acc,
            'phase': phase,
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
            'saved_at': checkpoint['saved_at'],
        }
        metadata_path = checkpoint_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.debug(f"Saved checkpoint: {checkpoint_path}")
    
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
            logger.error(f"Failed to load checkpoint: {e}")
            return None
