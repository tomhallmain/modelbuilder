"""
What a trained model's outputs mean, recorded alongside the weights.

An exported graph carries logits and nothing else: the label order, how to turn logits into
scores, and where the decision boundary sits all live outside it. A consumer that guesses
wrong here does not get an error — applying softmax to independent sigmoid logits yields
plausible-looking normalized numbers, and reading labels in the wrong order yields confident
predictions of the wrong class. Both survive a smoke test that only checks output shape.

This block is written into the export manifest and into the ONNX file's metadata, so an
artifact that travels on its own still describes how to decode it.

All fields are always present, including the ones single-label models leave null. A
consumer can then branch on ``activation`` without having to distinguish "field absent
because this is an older artifact" from "field absent because it does not apply".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

# ``label_mode`` values.
SINGLE_LABEL = "single_label"
MULTI_LABEL = "multi_label"

# ``activation`` values: how to turn logits into per-class scores.
SOFTMAX = "softmax"
"""Scores compete and sum to 1; the prediction is the argmax."""
SIGMOID = "sigmoid"
"""Scores are independent; every class over its own threshold is predicted."""

CONTRACT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class OutputContract:
    """
    Decode contract for a classification model's output tensor.

    Args:
        label_mode: :data:`SINGLE_LABEL` or :data:`MULTI_LABEL`.
        activation: :data:`SOFTMAX` or :data:`SIGMOID`.
        labels: Class names in output-index order. None when they could not be resolved,
            which is not the same as an empty list.
        thresholds: Per-label decision thresholds; multi-label only.
        axes: Named groupings over ``labels``; multi-label only.
    """

    label_mode: str = SINGLE_LABEL
    activation: str = SOFTMAX
    labels: Optional[List[str]] = None
    thresholds: Optional[Dict[str, float]] = None
    axes: Optional[Dict[str, Any]] = None

    def to_manifest_dict(self) -> Dict[str, Any]:
        """The ``output`` block, with every field present."""
        return {
            "label_mode": self.label_mode,
            "activation": self.activation,
            "labels": list(self.labels) if self.labels is not None else None,
            "thresholds": dict(self.thresholds) if self.thresholds is not None else None,
            "axes": dict(self.axes) if self.axes is not None else None,
        }


def single_label_contract(labels: Optional[Sequence[str]] = None) -> OutputContract:
    """
    Contract for a softmax classifier over mutually exclusive classes.

    Thresholds and axes stay null: a softmax argmax has no per-class threshold to tune, and
    the classes form one flat set rather than separable groups.
    """
    resolved = [str(name) for name in labels] if labels else None
    return OutputContract(
        label_mode=SINGLE_LABEL,
        activation=SOFTMAX,
        labels=resolved,
        thresholds=None,
        axes=None,
    )
