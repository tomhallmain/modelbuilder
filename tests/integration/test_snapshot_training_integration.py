"""
Integration: :func:`mb.training.snapshot_integration.update_training_snapshot` with a
minimal train/test tree and a pre-seeded :class:`~mb.utils.snapshot.UnifiedSnapshot`.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from mb.training.snapshot_integration import update_training_snapshot
from mb.utils.snapshot import UnifiedSnapshot


def _write_jpeg(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (10, 20, 30)).save(path, format="JPEG", quality=90)


def test_update_training_snapshot_attaches_training_metadata(tmp_path: Path) -> None:
    """
    Snapshot records must have ``dataset.path`` matching paths relative to ``data_dir``
    (e.g. ``train/cat/a.jpg``) so :meth:`~mb.utils.snapshot.UnifiedSnapshot.add_training_image`
    can attach the ``training`` block.
    """
    data_dir = tmp_path / "data"
    train_cat = data_dir / "train" / "cat"
    test_cat = data_dir / "test" / "cat"
    _write_jpeg(train_cat / "a.jpg")
    _write_jpeg(test_cat / "b.jpg")

    snap = UnifiedSnapshot(run_id="integration_run", raw_data_dir=str(tmp_path / "raw_data"))
    snap.images["key_a"] = {
        "original": {
            "hash": "key_a",
            "basename": "a.jpg",
            "path": "cat/a.jpg",
            "format": ".jpg",
        },
        "converted": {"md5": "m1", "path": "cat/CONVERTED/a.jpg", "basename": "a.jpg"},
        "dataset": {
            "class": "cat",
            "path": "train/cat/a.jpg",
            "basename": "a.jpg",
            "split": "train",
            "sha256": "s",
        },
        "training": None,
    }
    snap.images["key_b"] = {
        "original": {
            "hash": "key_b",
            "basename": "b.jpg",
            "path": "cat/b.jpg",
            "format": ".jpg",
        },
        "converted": {"md5": "m2", "path": "cat/CONVERTED/b.jpg", "basename": "b.jpg"},
        "dataset": {
            "class": "cat",
            "path": "test/cat/b.jpg",
            "basename": "b.jpg",
            "split": "test",
            "sha256": "s",
        },
        "training": None,
    }

    update_training_snapshot(data_dir, snap)

    ta = snap.images["key_a"].get("training")
    tb = snap.images["key_b"].get("training")
    assert ta is not None and ta.get("split") == "train"
    assert ta.get("path") == "train/cat/a.jpg"
    assert tb is not None and tb.get("split") == "test"
    assert tb.get("path") == "test/cat/b.jpg"

    summary = snap.to_dict().get("summary", {})
    assert summary.get("training_train_count") == 1
    assert summary.get("training_test_count") == 1
