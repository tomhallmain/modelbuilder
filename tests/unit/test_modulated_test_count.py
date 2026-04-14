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
    mean_class_size = n_total / 3
    # Class A: above mean — full anchor + share
    assert modulated_test_count(
        165_000, n_total, anchor=anchor, small_class_threshold=thr, mean_class_size=mean_class_size
    ) == 8724
    # Class B: below mean — capped to anchor * (n_c / mean)
    assert modulated_test_count(
        50_000, n_total, anchor=anchor, small_class_threshold=thr, mean_class_size=mean_class_size
    ) == 3386
    # Class C: proportional branch, then floor to ~⅓ of class (1500 → ≥500 test)
    assert modulated_test_count(
        1500, n_total, anchor=anchor, small_class_threshold=thr, mean_class_size=mean_class_size
    ) == 500


def test_modulated_small_integration_style_counts() -> None:
    """Unbalanced three-class layout; threshold 100, anchor 100; mean caps below-average classes."""
    n_total = 550
    anchor = 100
    thr = 100
    mean_class_size = n_total / 3
    assert modulated_test_count(
        50, n_total, anchor=anchor, small_class_threshold=thr, mean_class_size=mean_class_size
    ) == 17
    assert modulated_test_count(
        100, n_total, anchor=anchor, small_class_threshold=thr, mean_class_size=mean_class_size
    ) == 54
    assert modulated_test_count(
        400, n_total, anchor=anchor, small_class_threshold=thr, mean_class_size=mean_class_size
    ) == 172


def test_modulated_eleven_class_user_counts_anchor_5000() -> None:
    """
    Per-class image counts (classes A–K); ``n_total`` = 638_540.

    With ``mean_class_size = n_total / 11``, classes below that mean use the additive branch
    but are capped so test holdouts stay near ``anchor * (n_c / mean)`` instead of a full anchor.
    """
    anchor = 5000
    thr = 5000
    # (label, n_c images, expected test count)
    cases: list[tuple[str, int, int]] = [
        ("A", 47735, 4111),
        ("B", 15854, 1365),
        ("C", 35189, 3030),
        ("D", 93735, 5733),
        ("E", 112608, 5881),
        ("F", 164741, 6289),
        ("G", 117435, 5919),
        ("H", 27304, 2351),
        ("I", 8126, 699),
        ("J", 1517, 506),
        ("K", 14296, 1231),
    ]
    n_total = sum(nc for _, nc, _ in cases)
    assert n_total == 638_540
    mean_class_size = n_total / 11
    for label, n_c, expected in cases:
        got = modulated_test_count(
            n_c,
            n_total,
            anchor=anchor,
            small_class_threshold=thr,
            mean_class_size=mean_class_size,
        )
        assert got == expected, f"class {label}: n_c={n_c}, expected test={expected}, got {got}"


def test_modulated_tiny_mid_huge_tiers_anchor_100() -> None:
    """
    Same totals as ``test_dataset_weighted_near_default_cutoff`` (50 / 1000 / 4000): a small
    class, one at the old default cap, and a large class—static expectations for modulation.
    """
    n_total = 5050
    anchor = 100
    thr = 100
    mean_class_size = n_total / 3
    assert modulated_test_count(
        50, n_total, anchor=anchor, small_class_threshold=thr, mean_class_size=mean_class_size
    ) == 17
    assert modulated_test_count(
        1000, n_total, anchor=anchor, small_class_threshold=thr, mean_class_size=mean_class_size
    ) == 59
    assert modulated_test_count(
        4000, n_total, anchor=anchor, small_class_threshold=thr, mean_class_size=mean_class_size
    ) == 179


def test_modulated_proportional_branch_at_least_one_third_test_images() -> None:
    """Below-threshold classes use the proportional branch; test count is floored to ~ceil(n_c/3)."""
    n_total = 638_540
    anchor = thr = 5000
    # Class J–sized: proportional alone would be tiny; floor raises to ceil(1517/3) == 506
    assert modulated_test_count(1517, n_total, anchor=anchor, small_class_threshold=thr) == 506


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
