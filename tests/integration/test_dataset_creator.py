"""Integration: :class:`mb.data.dataset.DatasetCreator` on synthetic raw data."""

from __future__ import annotations

import random
from pathlib import Path

from mb.data.class_layout import SYNTHETIC_DEFAULT_CLASS_NAMES
from mb.data.dataset import DatasetCreator, modulated_test_count
from mb.utils.constants import DatasetSplitMode

from tests.test_utils import prepare_synthetic_raw_with_snapshot


def test_dataset_creator_produces_train_and_test_splits(tmp_path: Path) -> None:
    random.seed(42)
    raw = prepare_synthetic_raw_with_snapshot(tmp_path, total_images=100)
    data_dir = tmp_path / "data"
    test_per_class = 10

    creator = DatasetCreator(
        raw_data_dir=raw,
        data_dir=data_dir,
        test_per_class=test_per_class,
        class_names=list(SYNTHETIC_DEFAULT_CLASS_NAMES),
    )
    assert creator.run() is True

    # Deterministic split for total_images=100, seed=42 (34/33/33 per class)
    expected_train = {"coherent": 24, "semi-incoherent": 23, "incoherent": 23}
    expected_test = {name: test_per_class for name in SYNTHETIC_DEFAULT_CLASS_NAMES}

    for class_name in SYNTHETIC_DEFAULT_CLASS_NAMES:
        train_c = data_dir / "train" / class_name
        test_c = data_dir / "test" / class_name
        assert train_c.is_dir(), f"missing {train_c}"
        assert test_c.is_dir(), f"missing {test_c}"
        train_n = len(list(train_c.glob("*.jpg")))
        test_n = len(list(test_c.glob("*.jpg")))
        assert train_n == expected_train[class_name]
        assert test_n == expected_test[class_name]


def test_fixed_split_uses_all_when_below_test_per_class(tmp_path: Path) -> None:
    """Default-style fixed count: if a class has fewer images than test_per_class, all go to test."""
    random.seed(42)
    raw = prepare_synthetic_raw_with_snapshot(
        tmp_path,
        per_class_counts={"solo": 7},
        snapshot_run_id="solo_split",
    )
    data_dir = tmp_path / "data_fixed"
    creator = DatasetCreator(
        raw_data_dir=raw,
        data_dir=data_dir,
        test_per_class=1000,
        class_names=["solo"],
    )
    assert creator.run() is True
    assert len(list((data_dir / "test" / "solo").glob("*.jpg"))) == 7
    assert len(list((data_dir / "train" / "solo").glob("*.jpg"))) == 0


def test_dataset_weighted_near_default_cutoff(tmp_path: Path) -> None:
    """
    One class has ~the old default test cap (1000): fixed mode would nearly empty train;
    dataset_weighted uses proportional + anchor/share branches instead.
    """
    random.seed(42)
    counts = {"tiny": 50, "mid": 1000, "huge": 400}
    raw = prepare_synthetic_raw_with_snapshot(
        tmp_path,
        per_class_counts=counts,
        snapshot_run_id="weighted_split",
    )
    data_dir = tmp_path / "data_weighted"
    n_total = sum(counts.values())
    anchor = 100
    thr = 100
    expected_test = {
        "tiny": modulated_test_count(counts["tiny"], n_total, anchor=anchor, small_class_threshold=thr),
        "mid": modulated_test_count(counts["mid"], n_total, anchor=anchor, small_class_threshold=thr),
        "huge": modulated_test_count(counts["huge"], n_total, anchor=anchor, small_class_threshold=thr),
    }
    creator = DatasetCreator(
        raw_data_dir=raw,
        data_dir=data_dir,
        test_per_class=anchor,
        class_names=list(counts.keys()),
        test_split_mode=DatasetSplitMode.DATASET_WEIGHTED,
        test_small_class_threshold=thr,
    )
    assert creator.run() is True
    for name, exp in expected_test.items():
        te = len(list((data_dir / "test" / name).glob("*.jpg")))
        tr = len(list((data_dir / "train" / name).glob("*.jpg")))
        assert te == exp, f"{name}: expected {exp} test, got {te}"
        assert te + tr == counts[name]
