"""
Loss and epoch-metric selection for the PyTorch classification trainer.

Single-label and multi-label training differ in three coupled ways — the loss, how a score
becomes a prediction, and what "how good is this epoch" means — so they are resolved
together here rather than as three independent branches scattered through the loop.

Accuracy is the right epoch metric for a softmax over mutually exclusive classes and the
wrong one for independent sigmoids: with most labels negative on most images, predicting
nothing scores extremely well. Multi-label training therefore tracks micro F1 instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import torch
import torch.nn as nn

from mb.evaluate._multilabel import MultiLabelCounter, positive_weights
from mb.models.types import LabelMode
from mb.utils.logging_setup import get_logger

logger = get_logger(__name__)


class SingleLabelAccuracy:
    """Top-1 accuracy over mutually exclusive classes, as a percentage."""

    name = "accuracy"

    def __init__(self) -> None:
        self.correct = 0
        self.total = 0

    def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        _, predicted = outputs.max(1)
        self.total += int(targets.size(0))
        self.correct += int(predicted.eq(targets).sum().item())

    def value(self) -> float:
        return 100.0 * self.correct / max(self.total, 1)


class MultiLabelF1:
    """Micro F1 at per-label thresholds, as a percentage."""

    name = "micro_f1"

    def __init__(self, thresholds: Sequence[float]) -> None:
        self.thresholds = list(thresholds)
        self.counter = MultiLabelCounter(len(self.thresholds))
        self._threshold_tensor: Optional[torch.Tensor] = None

    def _thresholds_on(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self._threshold_tensor is None or self._threshold_tensor.device != device:
            self._threshold_tensor = torch.tensor(
                self.thresholds, dtype=dtype, device=device
            )
        return self._threshold_tensor

    def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        # The loss operates on raw logits; scoring needs probabilities, so the sigmoid is
        # applied here rather than in the model.
        probabilities = torch.sigmoid(outputs.detach())
        thresholds = self._thresholds_on(probabilities.device, probabilities.dtype)
        predicted = probabilities >= thresholds
        actual = targets.detach() > 0.5

        # Reduce on-device to per-label totals; a Python loop over the batch would dominate
        # step time on a large label set.
        self.counter.add_label_increments(
            true_positives=(predicted & actual).sum(dim=0).tolist(),
            false_positives=(predicted & ~actual).sum(dim=0).tolist(),
            false_negatives=(~predicted & actual).sum(dim=0).tolist(),
            true_negatives=(~predicted & ~actual).sum(dim=0).tolist(),
            n_samples=int(targets.size(0)),
        )

    def value(self) -> float:
        return 100.0 * self.counter.micro_f1()

    def macro_value(self) -> float:
        return 100.0 * self.counter.macro_f1()


@dataclass
class ClassificationObjective:
    """Loss functions and epoch metric for one training run."""

    criterion: nn.Module
    """Training loss; carries any class weighting."""
    eval_criterion: nn.Module
    """Unweighted loss, so validation stays comparable across weighting settings."""
    label_mode: LabelMode
    primary_metric: str
    """Name of the value tracked as the run's best score."""
    make_accumulator: Callable[[], object]
    class_weights: Optional[List[float]] = None
    """Per-class weights (single-label) or per-label pos_weight (multi-label), for logging."""


def build_single_label_objective(
    class_weights: Optional[Sequence[float]],
    device: torch.device,
) -> ClassificationObjective:
    """Cross-entropy over mutually exclusive classes, optionally class-weighted."""
    weight_tensor = (
        torch.tensor(list(class_weights), dtype=torch.float32, device=device)
        if class_weights
        else None
    )
    return ClassificationObjective(
        criterion=nn.CrossEntropyLoss(weight=weight_tensor),
        eval_criterion=nn.CrossEntropyLoss(),
        label_mode=LabelMode.SINGLE_LABEL,
        primary_metric=SingleLabelAccuracy.name,
        make_accumulator=SingleLabelAccuracy,
        class_weights=list(class_weights) if class_weights else None,
    )


def build_multi_label_objective(
    positive_counts: Sequence[int],
    n_samples: int,
    thresholds: Sequence[float],
    device: torch.device,
    *,
    max_weight: float,
    weighted: bool = True,
) -> ClassificationObjective:
    """
    Binary cross-entropy over independent labels, with per-label positive weighting.

    Each label is its own binary problem, so imbalance is handled per label rather than
    across a shared softmax.
    """
    weights: Optional[List[float]] = None
    pos_weight_tensor = None
    if weighted:
        weights = positive_weights(positive_counts, n_samples, max_weight=max_weight)
        pos_weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)

    return ClassificationObjective(
        criterion=nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor),
        eval_criterion=nn.BCEWithLogitsLoss(),
        label_mode=LabelMode.MULTI_LABEL,
        primary_metric=MultiLabelF1.name,
        make_accumulator=lambda: MultiLabelF1(thresholds),
        class_weights=weights,
    )
