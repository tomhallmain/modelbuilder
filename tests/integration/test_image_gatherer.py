"""Integration: :class:`mb.data.gather.ImageGatherer` copy-only scenario (no network)."""

from __future__ import annotations

import random
from pathlib import Path

from PIL import Image

from mb.data.gather import ImageGatherer


def _write_jpg(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (24, 24), (15, 99, 200)).save(path, format="JPEG")


def test_image_gatherer_copies_up_to_target_from_subdirs(tmp_path: Path) -> None:
    random.seed(0)
    src = tmp_path / "source"
    sub = src / "batch_a"
    for i in range(3):
        _write_jpg(sub / f"img_{i}.jpg")

    raw_root = tmp_path / "raw_data"
    target = raw_root / "coherent"

    gatherer = ImageGatherer(
        source_dir=str(src),
        valid_subdirs=["batch_a"],
        target_dir=target,
        target_count=10,
        rejected_dir=None,
        subdir_weights=None,
    )
    gatherer.raw_data_dir = raw_root

    assert gatherer.run() is True
    assert gatherer.stats["files_copied"] == 3

    copied = list(target.rglob("*.jpg"))
    assert len(copied) == 3
    for p in copied:
        assert p.stat().st_size > 0
