"""
Shared types for ``mb.evaluate`` (metrics, misclassified, compare).

``MetricsRequest`` / ``ClassificationMetricsReport`` are model-type–aware entry points:
add sibling request/report dataclasses when new :class:`~mb.models.types.ModelType` values
gain evaluation support.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from mb.models.types import FrameworkType, ModelType


@dataclass(frozen=True)
class MetricsRequest:
    """Inputs for ``mb evaluate metrics`` / ``misclassified`` (extensible per :attr:`model_type`)."""

    model_path: Path
    data_dir: Path
    model_type: ModelType
    framework: Optional[FrameworkType] = None
    architecture: Optional[str] = None
    num_classes: Optional[int] = None
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 0
    device: Optional[str] = None
    dry_run: bool = False


@dataclass
class ClassificationMetricsReport:
    """Image-classification metrics on an ImageFolder-style split."""

    model_type: ModelType
    framework: FrameworkType
    model_path: Path
    data_dir: Path
    n_samples: int
    class_names: list[str]
    accuracy_percent: float
    avg_loss: Optional[float] = None
    per_class_correct: list[int] = field(default_factory=list)
    per_class_total: list[int] = field(default_factory=list)
    confusion_matrix: list[list[int]] = field(default_factory=list)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type.value,
            "framework": self.framework.value,
            "model_path": str(self.model_path),
            "data_dir": str(self.data_dir),
            "n_samples": self.n_samples,
            "class_names": list(self.class_names),
            "accuracy_percent": self.accuracy_percent,
            "avg_loss": self.avg_loss,
            "per_class_correct": list(self.per_class_correct),
            "per_class_total": list(self.per_class_total),
            "confusion_matrix": [list(row) for row in self.confusion_matrix],
        }


@dataclass
class MisclassifiedSample:
    """One image whose predicted class differs from the on-disk folder label."""

    path: str
    true_label: str
    predicted_label: str
    confidence: float


@dataclass
class MisclassifiedListing:
    """Result of ``mb evaluate misclassified`` for image classification."""

    model_type: ModelType
    framework: FrameworkType
    model_path: Path
    data_dir: Path
    n_scanned: int
    n_misclassified: int
    samples: list[MisclassifiedSample] = field(default_factory=list)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type.value,
            "framework": self.framework.value,
            "model_path": str(self.model_path),
            "data_dir": str(self.data_dir),
            "n_scanned": self.n_scanned,
            "n_misclassified": self.n_misclassified,
            "samples": [
                {
                    "path": s.path,
                    "true_label": s.true_label,
                    "predicted_label": s.predicted_label,
                    "confidence": s.confidence,
                }
                for s in self.samples
            ],
        }
