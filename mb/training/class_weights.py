"""
Class weighting for imbalanced single-label training sets.

A rare class contributes almost nothing to an unweighted cross-entropy gradient, so a
classifier trained on a corpus where one label holds a fraction of a percent of the
support will systematically lose that label's borderline cases to whichever neighbouring
class is statistically dominant. Weighting the loss by inverse class frequency restores
the rare class's influence on the gradient without touching the data or the head.

The weights here are the standard balanced form, ``N / (K * n_i)``, where ``K`` counts the
classes that actually have samples. Its per-sample mean is exactly 1.0, so a weighted loss
curve stays on the same scale as an unweighted one and the two remain readable side by
side.

The ceiling is applied last and is absolute. Renormalizing after clamping to restore the
mean would push the clamped weight back above the ceiling (with counts ``[1, 9999]`` and a
ceiling of 10, renormalization lands at ~20), which defeats the point of having one. When
the ceiling does bite, the per-sample mean therefore falls below 1.0 and the weighted loss
sits on a slightly smaller scale — a bounded, visible trade, unlike a ceiling that silently
does not hold.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from mb.models.types import ClassWeightingMode
from mb.utils.logging_setup import get_logger

logger = get_logger(__name__)

# Fallback ceiling on any single class weight, used when the pipeline config supplies no
# training.class_weight_max. Without a ceiling, a class with a handful of images in a large
# corpus produces a weight big enough to destabilize training on its own.
DEFAULT_MAX_CLASS_WEIGHT = 50.0


def compute_class_weights(
    counts: Sequence[int],
    *,
    max_weight: float = DEFAULT_MAX_CLASS_WEIGHT,
) -> List[float]:
    """
    Balanced class weights for per-class sample *counts*.

    Classes with zero samples get weight 1.0. The value is arbitrary — such a class
    contributes no term to the loss either way — and 1.0 keeps it from standing out as
    meaningful when the weights are logged.

    Args:
        counts: Number of training samples per class, in class-index order.
        max_weight: Absolute ceiling on any single weight.

    Returns:
        One weight per class, in the same order. Per-sample mean is 1.0 unless the ceiling
        binds, in which case it is lower.
    """
    if len(counts) == 0:
        return []

    total = sum(int(c) for c in counts)
    if total <= 0:
        return [1.0] * len(counts)

    # Divide by the number of classes that actually have samples, not the taxonomy size.
    # An empty class in the denominator would scale every other weight down and shift the
    # whole loss, which is exactly the confound this is meant to avoid.
    n_populated = sum(1 for c in counts if int(c) > 0)
    ceiling = float(max_weight) if max_weight and max_weight > 0 else float("inf")

    weights: List[float] = []
    for count in counts:
        c = int(count)
        if c <= 0:
            weights.append(1.0)
            continue
        weights.append(min(total / (n_populated * c), ceiling))

    return weights


def class_counts_from_loader(loader: Any) -> Optional[List[int]]:
    """
    Per-class training sample counts from a PyTorch loader or a Keras generator.

    Returns ``None`` when the counts cannot be determined, which callers treat as "cannot
    weight" rather than as an error — an unweighted run is a valid outcome.
    """
    # PyTorch: DataLoader wrapping ImageFolderDataset (samples is a list of (path, label)).
    dataset = getattr(loader, "dataset", None)
    samples = getattr(dataset, "samples", None)
    classes = getattr(dataset, "classes", None)
    if samples is not None and classes is not None:
        counts = [0] * len(classes)
        for _path, label in samples:
            idx = int(label)
            if 0 <= idx < len(counts):
                counts[idx] += 1
        return counts

    # Keras: ImageDataGenerator iterator (classes is a per-sample label array).
    gen_classes = getattr(loader, "classes", None)
    class_indices = getattr(loader, "class_indices", None)
    if gen_classes is not None and class_indices:
        counts = [0] * len(class_indices)
        for label in gen_classes:
            idx = int(label)
            if 0 <= idx < len(counts):
                counts[idx] += 1
        return counts

    return None


def resolve_class_weights(
    loader: Any,
    mode: ClassWeightingMode,
    *,
    max_weight: float = DEFAULT_MAX_CLASS_WEIGHT,
    class_names: Optional[Sequence[str]] = None,
) -> Optional[List[float]]:
    """
    Class weights for *loader* under *mode*, or ``None`` for an unweighted loss.

    Logs the resolved weights: a weighted loss value is not comparable to an unweighted one
    or to a run with a different ceiling, so the regime that produced a number has to be
    recoverable from the training log.
    """
    if mode != ClassWeightingMode.INVERSE_FREQUENCY:
        return None

    counts = class_counts_from_loader(loader)
    if not counts:
        logger.warning("Class weighting requested but per-class counts are unavailable; using unweighted loss")
        return None

    weights = compute_class_weights(counts, max_weight=max_weight)
    names = list(class_names) if class_names else [str(i) for i in range(len(weights))]
    logger.info("Class weighting (%s, max=%.1f):", mode.value, max_weight)
    for name, count, weight in zip(names, counts, weights):
        logger.info("  %s: n=%d weight=%.4f", name, count, weight)
    return weights


def class_weight_dict(weights: Optional[Sequence[float]]) -> Optional[Dict[int, float]]:
    """Keras ``model.fit(class_weight=…)`` mapping, or ``None`` when unweighted."""
    if not weights:
        return None
    return {i: float(w) for i, w in enumerate(weights)}
