"""Integration: :class:`mb.data.dataset.DatasetCreator` on synthetic raw data."""

from __future__ import annotations

import random
from pathlib import Path

from mb.data.class_layout import SYNTHETIC_DEFAULT_CLASS_NAMES
from mb.data.dataset import DatasetCreator

from tests.test_utils import prepare_synthetic_raw_with_snapshot


def test_dataset_creator_produces_train_and_test_splits(tmp_path: Path) -> None:
    random.seed(42)
    raw = prepare_synthetic_raw_with_snapshot(tmp_path, total_images=100)
    data_dir = tmp_path / "data"
    test_per_class = 10

    creator = DatasetCreator(
        raw_data_dir=raw,
        data_dir=data_dir,
        test_images_per_class=test_per_class,
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
