"""
Inverse-frequency class weighting (``mb.training.class_weights``).

Pure arithmetic plus loader introspection — no torch, no TensorFlow, no disk I/O.
"""

from __future__ import annotations

from types import SimpleNamespace

from mb.models.types import ClassWeightingMode
from mb.training.class_weights import (
    class_counts_from_loader,
    class_weight_dict,
    compute_class_weights,
    resolve_class_weights,
)


def _per_sample_mean(weights: list[float], counts: list[int]) -> float:
    return sum(w * c for w, c in zip(weights, counts)) / sum(counts)


def test_balanced_counts_give_uniform_weights() -> None:
    assert compute_class_weights([10, 10, 10]) == [1.0, 1.0, 1.0]


def test_rare_class_is_weighted_up_and_common_class_down() -> None:
    """
    The ``gore``-style case: ~0.2% support against a ~25.8% majority, over 11 classes.

    Also pins the default ceiling as fit for this taxonomy: at 11 classes a 0.2% class
    lands at ~45, just under the default 50, so the default does not silently clamp the
    very class the weighting exists to rescue.
    """
    counts = [2] + [258] + [74] * 9  # rare / majority / nine mid-sized classes
    weights = compute_class_weights(counts)

    assert weights[0] > weights[2] > weights[1]
    assert weights[0] == sum(counts) / (11 * 2)  # balanced weight is N / (K * n_i)
    assert weights[0] < 50.0  # default ceiling does not bind
    assert abs(_per_sample_mean(weights, counts) - 1.0) < 1e-9


def test_ceiling_is_absolute() -> None:
    """
    The ceiling holds exactly; it is not relaxed to preserve the mean.

    Renormalizing after the clamp would push this weight back to ~20 against a ceiling of
    10, so no renormalization happens and the per-sample mean drops below 1.0 instead.
    """
    counts = [1, 9999]
    weights = compute_class_weights(counts, max_weight=10.0)

    assert weights[0] == 10.0  # unclamped this would be 10000 / (2 * 1) = 5000
    assert _per_sample_mean(weights, counts) < 1.0


def test_mean_is_one_when_the_ceiling_does_not_bind() -> None:
    counts = [2, 258, 740]
    weights = compute_class_weights(counts, max_weight=1000.0)
    assert abs(_per_sample_mean(weights, counts) - 1.0) < 1e-9


def test_empty_class_does_not_scale_down_the_populated_ones() -> None:
    """
    A class present in the taxonomy but absent from the split must not shift the loss.

    Dividing by the taxonomy size rather than the populated-class count would drag every
    other weight below 1.0 and quietly rescale the whole training loss.
    """
    with_empty = compute_class_weights([50, 0, 50])
    without_empty = compute_class_weights([50, 50])

    assert with_empty[0] == without_empty[0]
    assert with_empty[2] == without_empty[1]


def test_zero_count_class_does_not_divide_by_zero() -> None:
    counts = [50, 0, 50]
    weights = compute_class_weights(counts)

    assert len(weights) == 3
    assert weights[1] == 1.0
    assert all(w > 0 for w in weights)
    # A zero-count class contributes no samples, so the mean is unaffected by its weight.
    assert abs(_per_sample_mean(weights, counts) - 1.0) < 1e-9


def test_degenerate_inputs() -> None:
    assert compute_class_weights([]) == []
    assert compute_class_weights([0, 0]) == [1.0, 1.0]


def test_counts_from_pytorch_style_loader() -> None:
    dataset = SimpleNamespace(
        classes=["a", "b", "c"],
        samples=[("x.jpg", 0), ("y.jpg", 2), ("z.jpg", 2)],
    )
    loader = SimpleNamespace(dataset=dataset)
    assert class_counts_from_loader(loader) == [1, 0, 2]


def test_counts_from_keras_style_generator() -> None:
    generator = SimpleNamespace(classes=[0, 1, 1, 1], class_indices={"a": 0, "b": 1})
    assert class_counts_from_loader(generator) == [1, 3]


def test_counts_unavailable_returns_none() -> None:
    assert class_counts_from_loader(SimpleNamespace()) is None


def test_resolve_returns_none_when_mode_is_none() -> None:
    dataset = SimpleNamespace(classes=["a", "b"], samples=[("x.jpg", 0), ("y.jpg", 1)])
    loader = SimpleNamespace(dataset=dataset)
    assert resolve_class_weights(loader, ClassWeightingMode.NONE) is None


def test_resolve_returns_none_when_counts_cannot_be_determined() -> None:
    """An unweighted run is the fallback, not an error."""
    assert resolve_class_weights(SimpleNamespace(), ClassWeightingMode.INVERSE_FREQUENCY) is None


def test_resolve_computes_weights_for_inverse_frequency() -> None:
    dataset = SimpleNamespace(
        classes=["rare", "common"],
        samples=[("a.jpg", 0)] + [(f"{i}.jpg", 1) for i in range(9)],
    )
    loader = SimpleNamespace(dataset=dataset)
    weights = resolve_class_weights(loader, ClassWeightingMode.INVERSE_FREQUENCY)

    assert weights is not None
    assert weights[0] > weights[1]


def test_class_weight_dict_shape() -> None:
    assert class_weight_dict([2.0, 0.5]) == {0: 2.0, 1: 0.5}
    assert class_weight_dict(None) is None
    assert class_weight_dict([]) is None
