"""
Base classes for framework trainers and model abstractions.

This module defines the abstract interfaces that framework-specific
implementations must follow.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from mb.models.types import ArchitectureType, FrameworkType


class FrameworkTrainer(ABC):
    """
    Abstract base class for framework-specific trainers.

    Subclasses must call ``super().__init__(framework)`` with their fixed
    :class:`~mb.models.types.FrameworkType`.
    """

    def __init__(self, framework: FrameworkType) -> None:
        self._framework = framework

    @property
    def framework(self) -> FrameworkType:
        """Framework this trainer implementation uses."""
        return self._framework

    def get_framework_name(self) -> str:
        """Return the framework id string (same as :attr:`framework` ``.value``)."""
        return self.framework.value

    @abstractmethod
    def create_model(
        self,
        architecture: Union[ArchitectureType, str],
        num_classes: int,
        pretrained: bool = True,
        **kwargs
    ) -> Any:
        """
        Create a model instance.
        
        Args:
            architecture: Canonical architecture id or string from config/CLI
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            **kwargs: Additional architecture-specific arguments
            
        Returns:
            Model instance (framework-specific type)
        """
        pass
    
    @abstractmethod
    def create_data_loaders(
        self,
        train_dir: Path,
        val_dir: Path,
        batch_size: int,
        image_size: int,
        num_workers: int = 0,
        **kwargs
    ) -> Tuple[Any, Any]:
        """
        Create training and validation data loaders.
        
        Args:
            train_dir: Path to training data directory
            val_dir: Path to validation/test data directory
            batch_size: Batch size for training
            image_size: Target image size (assumes square)
            num_workers: Number of worker processes for data loading
            **kwargs: Additional data loader arguments
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        pass
    
    @abstractmethod
    def train(
        self,
        model: Any,
        train_loader: Any,
        val_loader: Any,
        hyperparams: Dict[str, Any],
        output_dir: Path,
        **kwargs
    ) -> Any:
        """
        Train the model.
        
        Args:
            model: Model instance to train
            train_loader: Training data loader
            val_loader: Validation data loader
            hyperparams: Dictionary of hyperparameters
            output_dir: Directory to save checkpoints and final model
            **kwargs: Additional training arguments
            
        Returns:
            Trained model instance
        """
        pass
    
    @abstractmethod
    def evaluate(
        self,
        model: Any,
        val_loader: Any,
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
        pass
    
    @abstractmethod
    def save_model(
        self,
        model: Any,
        path: Path,
        format: str = "native",
        **kwargs
    ) -> Path:
        """
        Save the model to disk.
        
        Args:
            model: Model instance to save
            path: Path to save the model
            format: Format to save in ('native', 'h5', 'onnx', etc.)
            **kwargs: Additional save arguments
            
        Returns:
            Path where model was saved
        """
        pass
    
    @abstractmethod
    def load_model(
        self,
        path: Path,
        **kwargs
    ) -> Any:
        """
        Load a saved model from disk.
        
        Args:
            path: Path to saved model
            **kwargs: Additional load arguments
            
        Returns:
            Loaded model instance
        """
        pass
    
    @abstractmethod
    def get_supported_architectures(self) -> list:
        """
        Get list of supported architectures for this framework.
        
        Returns:
            List of architecture names
        """
        pass
