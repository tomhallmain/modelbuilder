"""Integration: :class:`mb.data.dataset.DatasetCreator` on synthetic raw data."""

from __future__ import annotations

import random
from pathlib import Path

from mb.data.dataset import CLASS_NAMES, DatasetCreator

from tests.test_utils import prepare_synthetic_raw_with_snapshot


def test_dataset_creator_produces_train_and_test_splits(tmp_path: Path) -> None:
    random.seed(42)
    raw = prepare_synthetic_raw_with_snapshot(tmp_path, total_images=100)
    data_dir = tmp_path / "data"

    creator = DatasetCreator(
        raw_data_dir=raw,
        data_dir=data_dir,
        test_images_per_class=10,
    )
    assert creator.run() is True

    for class_name in CLASS_NAMES:
        train_c = data_dir / "train" / class_name
        test_c = data_dir / "test" / class_name
        assert train_c.is_dir(), f"missing {train_c}"
        assert test_c.is_dir(), f"missing {test_c}"
        train_n = len(list(train_c.glob("*.jpg")))
        test_n = len(list(test_c.glob("*.jpg")))
        assert train_n >= 1, f"train empty for {class_name}"
        assert test_n >= 1, f"test empty for {class_name}"
