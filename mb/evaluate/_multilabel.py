"""
Per-label counting and averaging for multi-label classification.

Kept free of any framework import so the same arithmetic serves the training loop's epoch
metric and ``mb evaluate metrics``, and so it can be tested without torch.

A multi-label result has no n x n confusion matrix: predictions are independent, so the
analogue is a 2x2 count per label. Accuracy is not reported at all — with most labels
negative on most images, predicting nothing scores extremely well by that measure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Zero when the denominator is zero, matching the single-label report's convention."""
    return float(numerator) / float(denominator) if denominator else 0.0


def f1_from_precision_recall(precision: float, recall: float) -> float:
    """Harmonic mean, zero when both inputs are zero."""
    return _safe_ratio(2.0 * precision * recall, precision + recall)


@dataclass
class LabelCounts:
    """Binary confusion counts for one label."""

    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_negatives: int = 0

    @property
    def support(self) -> int:
        """Images that genuinely carry this label."""
        return self.true_positives + self.false_negatives

    @property
    def predicted(self) -> int:
        """Images the model assigned this label."""
        return self.true_positives + self.false_positives

    @property
    def precision(self) -> float:
        return _safe_ratio(self.true_positives, self.predicted)

    @property
    def recall(self) -> float:
        return _safe_ratio(self.true_positives, self.support)

    @property
    def f1(self) -> float:
        return f1_from_precision_recall(self.precision, self.recall)


class MultiLabelCounter:
    """
    Accumulates per-label binary counts across batches.

    Callers convert scores to booleans at whatever thresholds apply and pass the increments
    in; this class holds no notion of a threshold itself.
    """

    def __init__(self, num_labels: int) -> None:
        self.num_labels = num_labels
        self.counts: List[LabelCounts] = [LabelCounts() for _ in range(num_labels)]
        self.n_samples = 0

    def add_batch(
        self,
        predicted: Sequence[Sequence[bool]],
        actual: Sequence[Sequence[bool]],
    ) -> None:
        """Fold in one batch of per-label boolean predictions and truths."""
        for pred_row, true_row in zip(predicted, actual):
            self.n_samples += 1
            for index in range(self.num_labels):
                p = bool(pred_row[index])
                t = bool(true_row[index])
                counts = self.counts[index]
                if p and t:
                    counts.true_positives += 1
                elif p and not t:
                    counts.false_positives += 1
                elif t:
                    counts.false_negatives += 1
                else:
                    counts.true_negatives += 1

    def add_label_increments(
        self,
        true_positives: Sequence[int],
        false_positives: Sequence[int],
        false_negatives: Sequence[int],
        true_negatives: Sequence[int],
        n_samples: int,
    ) -> None:
        """Fold in per-label totals already reduced by a framework (avoids a Python loop)."""
        self.n_samples += int(n_samples)
        for index in range(self.num_labels):
            counts = self.counts[index]
            counts.true_positives += int(true_positives[index])
            counts.false_positives += int(false_positives[index])
            counts.false_negatives += int(false_negatives[index])
            counts.true_negatives += int(true_negatives[index])

    def micro_f1(self) -> float:
        """F1 over pooled counts: dominated by the common labels."""
        tp = sum(c.true_positives for c in self.counts)
        fp = sum(c.false_positives for c in self.counts)
        fn = sum(c.false_negatives for c in self.counts)
        precision = _safe_ratio(tp, tp + fp)
        recall = _safe_ratio(tp, tp + fn)
        return f1_from_precision_recall(precision, recall)

    def macro_f1(self) -> float:
        """
        Mean per-label F1 over labels present in the data.

        Labels with no positives are excluded, matching the single-label report: a label
        absent from the split has no measured performance, and averaging in a zero would
        understate the model by an amount that depends only on how the split was built.
        """
        scored = [c for c in self.counts if c.support > 0]
        if not scored:
            return 0.0
        return sum(c.f1 for c in scored) / len(scored)

    def scored_label_count(self) -> int:
        """Number of labels with at least one positive in the data."""
        return sum(1 for c in self.counts if c.support > 0)

    def axis_macro_f1(
        self,
        label_names: Sequence[str],
        axis_labels: Sequence[str],
    ) -> Optional[float]:
        """Macro F1 restricted to one axis, or None when no axis label has support."""
        wanted = set(axis_labels)
        scored = [
            counts
            for name, counts in zip(label_names, self.counts)
            if name in wanted and counts.support > 0
        ]
        if not scored:
            return None
        return sum(c.f1 for c in scored) / len(scored)


def positive_weights(
    positive_counts: Sequence[int],
    n_samples: int,
    *,
    max_weight: float,
) -> List[float]:
    """
    ``pos_weight`` per label for a binary cross-entropy loss.

    Each label is its own binary problem, so the weight is the negative-to-positive ratio
    for that label alone: ``(N - n_positive) / n_positive``. A label with no positives gets
    1.0, since it contributes no positive term to weight.

    The ceiling is absolute and applied last, matching the single-label class weighting: a
    label at a fraction of a percent support would otherwise produce a weight large enough
    to destabilize training by itself.
    """
    ceiling = float(max_weight) if max_weight and max_weight > 0 else float("inf")
    total = int(n_samples)
    weights: List[float] = []
    for count in positive_counts:
        positives = int(count)
        if positives <= 0 or total <= positives:
            weights.append(1.0)
            continue
        weights.append(min((total - positives) / positives, ceiling))
    return weights


def thresholds_for_labels(
    label_names: Sequence[str],
    per_label: Dict[str, float],
    default: float,
) -> List[float]:
    """Threshold per label in output-index order."""
    return [float(per_label.get(name, default)) for name in label_names]
