"""Unit tests: modulated test split counts (:mod:`mb.data.dataset`)."""

from __future__ import annotations

import pytest

from mb.data.dataset import modulated_test_count
from mb.utils.constants import DatasetSplitMode


def test_dataset_split_mode_normalize() -> None:
    assert DatasetSplitMode.normalize(None) == DatasetSplitMode.FIXED
    assert DatasetSplitMode.normalize("fixed") == DatasetSplitMode.FIXED
    assert DatasetSplitMode.normalize("dataset_weighted") == DatasetSplitMode.DATASET_WEIGHTED
    assert DatasetSplitMode.normalize("dataset-weighted") == DatasetSplitMode.DATASET_WEIGHTED
    assert DatasetSplitMode.normalize("modulated") == DatasetSplitMode.DATASET_WEIGHTED
    assert DatasetSplitMode.normalize(DatasetSplitMode.DATASET_WEIGHTED) == DatasetSplitMode.DATASET_WEIGHTED


def test_modulated_user_example_abc_anchor_5000() -> None:
    """Illustrative A/B/C totals from product discussion (anchor = threshold = 5000)."""
    n_total = 221_500
    anchor = 5000
    thr = 5000
    # Class A: anchor + anchor * share
    assert modulated_test_count(165_000, n_total, anchor=anchor, small_class_threshold=thr) == 8724
    assert modulated_test_count(50_000, n_total, anchor=anchor, small_class_threshold=thr) == 6128
    # Class C: proportional only (n_c < threshold)
    assert modulated_test_count(1500, n_total, anchor=anchor, small_class_threshold=thr) == 33


def test_modulated_small_integration_style_counts() -> None:
    """Unbalanced three-class layout; threshold 100, anchor 100."""
    n_total = 550
    anchor = 100
    thr = 100
    assert modulated_test_count(50, n_total, anchor=anchor, small_class_threshold=thr) == 9
    assert modulated_test_count(100, n_total, anchor=anchor, small_class_threshold=thr) == 99
    assert modulated_test_count(400, n_total, anchor=anchor, small_class_threshold=thr) == 172


def test_modulated_single_image_class() -> None:
    assert modulated_test_count(1, 100, anchor=1000, small_class_threshold=5000) == 1


def test_modulated_two_images_floor() -> None:
    """Ensure at least one image remains for training when n_c > 1."""
    assert modulated_test_count(2, 10, anchor=100, small_class_threshold=1) == 1


@pytest.mark.parametrize(
    ("n_c", "n_total", "expected"),
    [
        (0, 100, 0),
        (5, 0, 0),
    ],
)
def test_modulated_empty_edge(n_c: int, n_total: int, expected: int) -> None:
    assert modulated_test_count(n_c, n_total, anchor=100, small_class_threshold=50) == expected
