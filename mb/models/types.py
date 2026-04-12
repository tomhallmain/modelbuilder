"""
Model type definitions and base classes.

This module defines the supported model types and provides base classes
for model type handlers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Dict, FrozenSet, Optional

# --- CLI / pipeline step enums (used by ``mb data``, Data page, training metadata) ---


class ModelBuildStepCommand(str, Enum):
    """
    ``mb data <subcommand>`` values (CLI, :func:`~mb.cli.run_data_subcommand_cli`).

    The Data page uses the first five; ``estimate-space`` is CLI-only.
    """

    GATHER = "gather"
    CONVERT = "convert"
    DEDUPLICATE = "deduplicate"
    UPSCALE = "upscale"
    CREATE_DATASET = "create-dataset"
    ESTIMATE_SPACE = "estimate-space"
    FIX_JPEG_EXTENSION_MISMATCH = "fix-jpeg-extension-mismatch"

    @classmethod
    def try_from(cls, raw: object) -> ModelBuildStepCommand | None:
        if raw is None:
            return None
        s = str(raw).strip().lower()
        try:
            return cls(s)
        except ValueError:
            return None

    @classmethod
    def data_page_tab_values(cls) -> tuple[ModelBuildStepCommand, ...]:
        """Tab order on :class:`~ui.pages.data_page.DataPage` (no ``estimate-space``)."""
        return (
            cls.GATHER,
            cls.CONVERT,
            cls.DEDUPLICATE,
            cls.UPSCALE,
            cls.CREATE_DATASET,
        )

    @classmethod
    def gui_wildcard_command_values(cls) -> tuple[ModelBuildStepCommand, ...]:
        """Commands offered on the Data page Wildcard tab (excludes CLI-only ``estimate-space``)."""
        return (
            cls.GATHER,
            cls.CONVERT,
            cls.DEDUPLICATE,
            cls.UPSCALE,
            cls.CREATE_DATASET,
            cls.FIX_JPEG_EXTENSION_MISMATCH,
        )


class FrameworkType(str, Enum):
    """Training / export framework (``mb train``, ``mb convert``, registry keys)."""

    PYTORCH = "pytorch"
    KERAS = "keras"

    @classmethod
    def get_default(cls) -> FrameworkType:
        """Default pipeline framework (matches packaged :file:`mb/config/default_pipeline.yaml`)."""
        return cls.PYTORCH

    @classmethod
    def registered_values(cls) -> FrozenSet[str]:
        """Set of all enum values (for validation against static registrations)."""
        return frozenset(m.value for m in cls)

    @classmethod
    def try_from(cls, raw: object) -> FrameworkType | None:
        if raw is None:
            return None
        s = str(raw).strip().lower()
        try:
            return cls(s)
        except ValueError:
            return None


class ArchitectureType(str, Enum):
    """
    Canonical architecture ids registered in :mod:`mb.models.frameworks`.

    Values match registry keys passed as :class:`ArchitectureType` to
    :func:`mb.models.frameworks.registry.register_architecture` (lowercase). Families
    include ResNet, EfficientNet, MobileNet, DenseNet, and VGG where implemented per
    framework. Not every framework implements every member; use the registry or trainer
    helpers to list supported names per framework.
    """

    RESNET18 = "resnet18"
    RESNET34 = "resnet34"
    RESNET50 = "resnet50"
    RESNET101 = "resnet101"
    RESNET152 = "resnet152"
    EFFICIENTNET_B0 = "efficientnet_b0"
    EFFICIENTNET_B1 = "efficientnet_b1"
    EFFICIENTNET_B2 = "efficientnet_b2"
    EFFICIENTNET_B3 = "efficientnet_b3"
    # MobileNet (``torchvision.models`` / ``keras.applications``)
    MOBILENET_V2 = "mobilenet_v2"
    MOBILENET_V3_LARGE = "mobilenet_v3_large"
    MOBILENET_V3_SMALL = "mobilenet_v3_small"
    # DenseNet
    DENSENET121 = "densenet121"
    DENSENET169 = "densenet169"
    DENSENET201 = "densenet201"
    # VGG
    VGG16 = "vgg16"
    VGG19 = "vgg19"

    @classmethod
    def get_default(cls) -> ArchitectureType:
        """Default pipeline architecture (matches packaged :file:`mb/config/default_pipeline.yaml`)."""
        return cls.RESNET34

    @classmethod
    def registered_values(cls) -> FrozenSet[str]:
        """Set of all enum values (for validation against static registrations)."""
        return frozenset(m.value for m in cls)

    @classmethod
    def try_from(cls, raw: object) -> ArchitectureType | None:
        if raw is None:
            return None
        s = str(raw).strip().lower()
        try:
            return cls(s)
        except ValueError:
            return None


class InfoSubcommand(str, Enum):
    """``mb info <subcommand>`` (model vs dataset)."""

    MODEL = "model"
    DATASET = "dataset"

    @classmethod
    def try_from(cls, raw: object) -> InfoSubcommand | None:
        if raw is None:
            return None
        s = str(raw).strip().lower()
        try:
            return cls(s)
        except ValueError:
            return None


class ConversionTargetFormat(str, Enum):
    """``mb convert --target`` output format."""

    ONNX = "onnx"
    SAFETENSORS = "safetensors"

    @classmethod
    def registered_values(cls) -> FrozenSet[str]:
        """Set of all enum values (CLI ``--target`` choices)."""
        return frozenset(m.value for m in cls)

    @classmethod
    def try_from(cls, raw: object) -> ConversionTargetFormat | None:
        if raw is None:
            return None
        s = str(raw).strip().lower()
        try:
            return cls(s)
        except ValueError:
            return None


class ModelType(str, Enum):
    """Pipeline / training model type (YAML ``model.default_type``, CLI, gather, convert)."""

    IMAGE_CLASSIFICATION = "image_classification"
    # Reserved for future handlers; gather/convert treat as non-image-classification until wired.
    OBJECT_DETECTION = "object_detection"

    @classmethod
    def from_pipeline_value(cls, value: Optional[object]) -> ModelType:
        """Resolve ``model.default_type`` or CLI string to a member (unknown → image classification)."""
        if value is None:
            return cls.IMAGE_CLASSIFICATION
        if isinstance(value, ModelType):
            return value
        s = str(value).strip()
        if not s:
            return cls.IMAGE_CLASSIFICATION
        try:
            return cls(s)
        except ValueError:
            return cls.IMAGE_CLASSIFICATION


class VisualMediaSourceType(str, Enum):
    """
    How a raw media file is handled by :func:`~mb.data.media_utils.classify_convert_source`
    (image-classification convert path).
    """

    STATIC = "static"
    """Still image path: convert or copy JPEG as today (includes single-frame GIF)."""
    VIDEO_EXTRACT = "video"
    """Random frame from a configured video type."""
    ANIMATED_GIF_EXTRACT = "animated_gif"
    """Random frame from a multi-frame GIF."""


class ModelTypeHandler(ABC):
    """
    Abstract base class for model type handlers.
    
    Each model type (e.g., image classification) should have a handler
    that knows how to validate data, determine number of classes, etc.
    """
    
    @abstractmethod
    def get_num_classes(self, data_dir: Path) -> int:
        """
        Determine the number of classes from the data directory structure.
        
        Args:
            data_dir: Path to the data directory (typically containing train/test subdirs)
            
        Returns:
            Number of classes
        """
        pass
    
    @abstractmethod
    def validate_data(self, data_dir: Path) -> bool:
        """
        Validate that the data directory structure is correct for this model type.
        
        Args:
            data_dir: Path to the data directory
            
        Returns:
            True if data structure is valid, False otherwise
        """
        pass
    
    @abstractmethod
    def get_default_hyperparams(self) -> Dict[str, Any]:
        """
        Get default hyperparameters for this model type.
        
        Returns:
            Dictionary of default hyperparameters
        """
        pass
    
    @abstractmethod
    def get_data_structure_info(self) -> str:
        """
        Get information about the expected data directory structure.
        
        Returns:
            Description of expected data structure
        """
        pass


class ImageClassificationHandler(ModelTypeHandler):
    """Handler for image classification model type."""
    
    def get_num_classes(self, data_dir: Path) -> int:
        """
        Determine number of classes from train directory structure.
        
        For image classification, classes are determined by subdirectories
        in the train/ directory.
        """
        train_dir = data_dir / "train"
        if not train_dir.exists():
            raise ValueError(f"Train directory not found: {train_dir}")
        
        # Count subdirectories (each represents a class)
        class_dirs = [d for d in train_dir.iterdir() if d.is_dir()]
        num_classes = len(class_dirs)
        
        if num_classes == 0:
            raise ValueError(f"No class directories found in {train_dir}")
        
        return num_classes
    
    def validate_data(self, data_dir: Path) -> bool:
        """
        Validate image classification data structure.
        
        Expected structure:
        data_dir/
          train/
            class1/
            class2/
            ...
          test/
            class1/
            class2/
            ...
        """
        train_dir = data_dir / "train"
        test_dir = data_dir / "test"
        
        if not train_dir.exists():
            return False
        
        if not test_dir.exists():
            return False
        
        # Check that train and test have the same classes
        train_classes = {d.name for d in train_dir.iterdir() if d.is_dir()}
        test_classes = {d.name for d in test_dir.iterdir() if d.is_dir()}
        
        if len(train_classes) == 0:
            return False
        
        # Test set should have same classes (or subset)
        if not test_classes.issubset(train_classes):
            return False
        
        return True
    
    def get_default_hyperparams(self) -> Dict[str, Any]:
        """Get default hyperparameters for image classification."""
        return {
            "image_size": 224,
            "batch_size": None,  # Auto-detect
            "frozen_epochs": 5,
            "unfrozen_epochs": 20,
            "frozen_lr": 0.001,
            "unfrozen_lr_max": 0.0003,
            "unfrozen_lr_min": 0.00001,
            "num_workers": 12,
        }
    
    def get_data_structure_info(self) -> str:
        """Get information about expected data structure."""
        return """
Image Classification Data Structure:
  data_dir/
    train/
      class1/
        image1.jpg
        image2.jpg
        ...
      class2/
        ...
    test/
      class1/
        ...
      class2/
        ...
"""


# Registry for model type handlers
_MODEL_TYPE_HANDLERS: Dict[ModelType, ModelTypeHandler] = {
    ModelType.IMAGE_CLASSIFICATION: ImageClassificationHandler(),
}


def get_model_type_handler(model_type: ModelType) -> ModelTypeHandler:
    """
    Get the handler for a specific model type.
    
    Args:
        model_type: The model type enum value
        
    Returns:
        ModelTypeHandler instance
        
    Raises:
        ValueError: If model type is not supported
    """
    if model_type not in _MODEL_TYPE_HANDLERS:
        raise ValueError(f"Model type {model_type} is not supported")
    
    return _MODEL_TYPE_HANDLERS[model_type]


def register_model_type_handler(model_type: ModelType, handler: ModelTypeHandler):
    """
    Register a new model type handler.
    
    Args:
        model_type: The model type enum value
        handler: The handler instance
    """
    _MODEL_TYPE_HANDLERS[model_type] = handler
