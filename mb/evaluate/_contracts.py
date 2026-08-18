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

from mb.models.types import FrameworkType, LabelMode, ModelType


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
    label_mode: LabelMode = LabelMode.SINGLE_LABEL
    """Defaulted so existing callers keep single-label scoring."""
    tune_thresholds: bool = False
    """Multi-label only: sweep per-label thresholds and write the result back to the schema."""


@dataclass(frozen=True)
class PerClassMetrics:
    """Precision/recall/F1 for one class, derived from a confusion matrix."""

    name: str
    support: int
    """Samples whose true label is this class (confusion-matrix row sum)."""
    predicted: int
    """Samples the model assigned to this class (confusion-matrix column sum)."""
    true_positives: int
    precision: float
    recall: float
    f1: float

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "support": self.support,
            "predicted": self.predicted,
            "true_positives": self.true_positives,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


@dataclass(frozen=True)
class AveragedMetrics:
    """Precision/recall/F1 averaged across classes."""

    precision: float
    recall: float
    f1: float
    n_classes: int
    """Classes the average was taken over (zero-support classes are excluded)."""

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "n_classes": self.n_classes,
        }


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Zero when the denominator is zero — see :meth:`ClassificationMetricsReport.per_class_metrics`."""
    return float(numerator) / float(denominator) if denominator else 0.0


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

    def per_class_metrics(self) -> list[PerClassMetrics]:
        """
        Per-class precision/recall/F1 derived from :attr:`confusion_matrix`.

        A metric whose denominator is zero is reported as 0.0 rather than omitted or NaN:
        a class the model never predicts has no meaningful precision, and 0.0 is the value
        that keeps the number sortable and comparable across runs. Distinguish the two
        zero cases by reading :attr:`PerClassMetrics.support` and
        :attr:`PerClassMetrics.predicted` alongside it — support 0 means the class was
        absent from the evaluation split, predicted 0 means the model never chose it.

        Returns an empty list when no confusion matrix was recorded.
        """
        cm = self.confusion_matrix
        if not cm:
            return []

        n = len(cm)
        out: list[PerClassMetrics] = []
        for i in range(n):
            name = self.class_names[i] if i < len(self.class_names) else str(i)
            tp = int(cm[i][i]) if i < len(cm[i]) else 0
            support = int(sum(cm[i]))
            predicted = int(sum(row[i] for row in cm if i < len(row)))
            precision = _safe_ratio(tp, predicted)
            recall = _safe_ratio(tp, support)
            f1 = _safe_ratio(2.0 * precision * recall, precision + recall)
            out.append(
                PerClassMetrics(
                    name=name,
                    support=support,
                    predicted=predicted,
                    true_positives=tp,
                    precision=precision,
                    recall=recall,
                    f1=f1,
                )
            )
        return out

    def macro_averages(self) -> Optional[AveragedMetrics]:
        """
        Unweighted mean of per-class metrics over classes present in the split.

        Classes with zero support are excluded rather than averaged in as 0.0. A class
        absent from the evaluation split has no measured performance, so counting it as a
        perfect failure would understate the model by an amount that depends only on how
        the split was built. The per-class table still lists those classes.

        Micro averages are deliberately not reported: for single-label classification,
        micro precision, recall, and F1 all equal :attr:`accuracy_percent`.
        """
        scored = [m for m in self.per_class_metrics() if m.support > 0]
        if not scored:
            return None
        k = len(scored)
        return AveragedMetrics(
            precision=sum(m.precision for m in scored) / k,
            recall=sum(m.recall for m in scored) / k,
            f1=sum(m.f1 for m in scored) / k,
            n_classes=k,
        )

    def weighted_averages(self) -> Optional[AveragedMetrics]:
        """Support-weighted mean of per-class metrics (zero-support classes contribute nothing)."""
        scored = [m for m in self.per_class_metrics() if m.support > 0]
        if not scored:
            return None
        total = sum(m.support for m in scored)
        if total <= 0:
            return None
        return AveragedMetrics(
            precision=sum(m.precision * m.support for m in scored) / total,
            recall=sum(m.recall * m.support for m in scored) / total,
            f1=sum(m.f1 * m.support for m in scored) / total,
            n_classes=len(scored),
        )

    def to_jsonable(self) -> dict[str, Any]:
        macro = self.macro_averages()
        weighted = self.weighted_averages()
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
            "per_class_metrics": [m.to_jsonable() for m in self.per_class_metrics()],
            "macro_avg": macro.to_jsonable() if macro is not None else None,
            "weighted_avg": weighted.to_jsonable() if weighted is not None else None,
        }


@dataclass(frozen=True)
class PerLabelMetrics:
    """Binary counts and derived rates for one label of a multi-label model."""

    name: str
    support: int
    predicted: int
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    precision: float
    recall: float
    f1: float
    threshold: float
    """Score above which this label counts as predicted."""

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "support": self.support,
            "predicted": self.predicted,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "true_negatives": self.true_negatives,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "threshold": self.threshold,
        }


@dataclass
class MultiLabelClassificationMetricsReport:
    """
    Multi-label metrics on a prepared split.

    There is no confusion matrix: predictions are independent, so the analogue of an n x n
    matrix is a 2 x 2 count per label. Accuracy is absent for the same reason it is not the
    training metric — with most labels negative on most images, predicting nothing scores
    well by that measure.
    """

    model_type: ModelType
    framework: FrameworkType
    model_path: Path
    data_dir: Path
    n_samples: int
    label_names: list[str]
    micro_f1: float
    macro_f1: float
    avg_loss: Optional[float] = None
    per_label: list[PerLabelMetrics] = field(default_factory=list)
    axis_macro_f1: dict[str, float] = field(default_factory=dict)
    n_labels_with_support: int = 0

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type.value,
            "framework": self.framework.value,
            "label_mode": "multi_label",
            "model_path": str(self.model_path),
            "data_dir": str(self.data_dir),
            "n_samples": self.n_samples,
            "label_names": list(self.label_names),
            "micro_f1": self.micro_f1,
            "macro_f1": self.macro_f1,
            "avg_loss": self.avg_loss,
            "per_label": [m.to_jsonable() for m in self.per_label],
            "axis_macro_f1": dict(self.axis_macro_f1),
            "n_labels_with_support": self.n_labels_with_support,
        }


@dataclass
class MisclassifiedSample:
    """One image whose predicted class differs from the on-disk folder label."""

    path: str
    true_label: str
    predicted_label: str
    confidence: float


@dataclass
class DisagreementSample:
    """One image where model A and model B disagree on the predicted class."""

    path: str
    true_label: str
    predicted_a: str
    predicted_b: str
    confidence_a: float
    confidence_b: float


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


@dataclass(frozen=True)
class CompareRequest:
    """Inputs for ``mb evaluate compare`` (paired checkpoints on one split)."""

    model_path_a: Path
    model_path_b: Path
    data_dir: Path
    model_type: ModelType
    framework_a: Optional[FrameworkType] = None
    framework_b: Optional[FrameworkType] = None
    architecture: Optional[str] = None
    architecture_b: Optional[str] = None
    num_classes: Optional[int] = None
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 0
    device: Optional[str] = None
    max_disagreement_report: Optional[int] = None


@dataclass
class ClassificationCompareReport:
    """Paired image-classification comparison on one ImageFolder split."""

    model_type: ModelType
    framework_a: FrameworkType
    framework_b: FrameworkType
    model_path_a: Path
    model_path_b: Path
    data_dir: Path
    n_samples: int
    class_names: list[str]
    accuracy_a_percent: float
    accuracy_b_percent: float
    avg_loss_a: Optional[float] = None
    avg_loss_b: Optional[float] = None
    both_correct: int = 0
    only_a_correct: int = 0
    only_b_correct: int = 0
    both_wrong: int = 0
    pred_disagreement: int = 0
    disagreement_samples: list[DisagreementSample] = field(default_factory=list)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type.value,
            "framework_a": self.framework_a.value,
            "framework_b": self.framework_b.value,
            "model_path_a": str(self.model_path_a),
            "model_path_b": str(self.model_path_b),
            "data_dir": str(self.data_dir),
            "n_samples": self.n_samples,
            "class_names": list(self.class_names),
            "accuracy_a_percent": self.accuracy_a_percent,
            "accuracy_b_percent": self.accuracy_b_percent,
            "avg_loss_a": self.avg_loss_a,
            "avg_loss_b": self.avg_loss_b,
            "both_correct": self.both_correct,
            "only_a_correct": self.only_a_correct,
            "only_b_correct": self.only_b_correct,
            "both_wrong": self.both_wrong,
            "pred_disagreement": self.pred_disagreement,
            "disagreement_samples": [
                {
                    "path": s.path,
                    "true_label": s.true_label,
                    "predicted_a": s.predicted_a,
                    "predicted_b": s.predicted_b,
                    "confidence_a": s.confidence_a,
                    "confidence_b": s.confidence_b,
                }
                for s in self.disagreement_samples
            ],
        }
