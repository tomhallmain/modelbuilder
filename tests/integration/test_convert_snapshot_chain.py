"""Integration: convert reruns, snapshot ``converted`` metadata, and downstream path matching."""

from __future__ import annotations

import json
import random
from pathlib import Path

from PIL import Image

from mb.data.class_layout import CONVERTED_MEDIA_SUBDIR
from mb.data.dataset import MIN_FILE_SIZE
from mb.data.convert import ImageConverter
from mb.data.dataset import DatasetCreator
from mb.utils.snapshot import UnifiedSnapshot, calculate_file_hash, find_unified_snapshot
from mb.utils.utils import convert_output_jpeg_filename


def _rng_png(path: Path, seed: int = 42) -> None:
    """Large enough PNG that convert → JPEG still meets :data:`MIN_FILE_SIZE` for dataset."""
    rng = random.Random(seed)
    w = h = 512
    data = bytes(rng.randrange(256) for _ in range(w * h * 3))
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.frombytes("RGB", (w, h), data).save(path, format="PNG")


def test_convert_records_converted_paths_for_hash_suffixed_outputs(tmp_path: Path) -> None:
    """Still-image convert must populate ``converted`` so create-dataset can match by path (not stem-only)."""
    raw = tmp_path / "raw_data"
    cdir = raw / "solo"
    png = cdir / "photo.png"
    _rng_png(png)

    conv = ImageConverter(raw_data_dir=raw)
    assert conv.run(skip_space_check=True) is True

    rid = conv.run_id
    assert rid
    snap_path = next(raw.glob(f"snapshot_{rid}.json"))
    blob = json.loads(snap_path.read_text(encoding="utf-8"))
    assert len(blob["images"]) == 1
    rec = blob["images"][0]
    assert rec.get("converted") is not None
    out_dir = cdir / CONVERTED_MEDIA_SUBDIR
    expected_name = convert_output_jpeg_filename(png, output_dir=out_dir)
    assert rec["converted"]["basename"] == expected_name
    rel = rec["converted"]["path"].replace("\\", "/")
    assert rel.endswith(f"solo/{CONVERTED_MEDIA_SUBDIR}/{expected_name}")


def test_convert_resume_skip_backfills_converted_metadata(tmp_path: Path) -> None:
    """Re-run with same run ID: skips refresh ``converted``; new sources still convert (``run()`` requires work)."""
    raw = tmp_path / "raw_data"
    cdir = raw / "solo"
    png_a = cdir / "photo.png"
    _rng_png(png_a, seed=1)

    first = ImageConverter(raw_data_dir=raw)
    assert first.run(skip_space_check=True) is True
    rid = first.run_id

    png_b = cdir / "other.png"
    _rng_png(png_b, seed=2)

    second = ImageConverter(raw_data_dir=raw)
    assert second.run(skip_space_check=True, run_id=rid) is True

    snap = find_unified_snapshot([raw], run_id=rid)
    assert snap is not None
    assert len(snap.images) == 2
    for rec in snap.images.values():
        assert rec.get("converted") is not None
        assert rec["converted"]["md5"]


def test_convert_then_dataset_preserves_single_snapshot_record(tmp_path: Path) -> None:
    raw = tmp_path / "raw_data"
    cdir = raw / "solo"
    png = cdir / "img.png"
    _rng_png(png, seed=99)

    conv = ImageConverter(raw_data_dir=raw)
    assert conv.run(skip_space_check=True) is True
    rid = conv.run_id

    out_jpg = cdir / CONVERTED_MEDIA_SUBDIR / convert_output_jpeg_filename(
        png, output_dir=cdir / CONVERTED_MEDIA_SUBDIR
    )
    assert out_jpg.stat().st_size >= MIN_FILE_SIZE

    data_dir = tmp_path / "data"
    ds = DatasetCreator(
        raw_data_dir=raw,
        data_dir=data_dir,
        test_per_class=1,
        class_names=["solo"],
        run_id=rid,
        skip_space_check=True,
    )
    assert ds.run() is True

    # Dataset writes the updated snapshot under ``data_dir``; ``raw_data`` still has convert-only JSON.
    snap = find_unified_snapshot([data_dir, raw], run_id=rid)
    assert snap is not None
    assert len(snap.images) == 1
    rec = next(iter(snap.images.values()))
    assert rec.get("dataset") is not None
    assert rec["dataset"]["class"] == "solo"


def test_calculate_file_hash_returns_converted_md5_via_dataset_path(tmp_path: Path) -> None:
    """:func:`calculate_file_hash` matches ``dataset.path`` like training snapshot integration."""
    raw = tmp_path / "raw_data"
    snap = UnifiedSnapshot(run_id="t", raw_data_dir=str(raw))
    data_rel = "train/c/abc123.jpg"
    snap.images["origkey"] = {
        "original": {"hash": "origkey", "basename": "x.png", "path": "c/x.png", "format": ".png"},
        "converted": {
            "md5": "deadbeefcafe",
            "path": f"c/{CONVERTED_MEDIA_SUBDIR}/stem_hash.jpg",
            "basename": "stem_hash.jpg",
        },
        "dataset": {
            "class": "c",
            "path": data_rel,
            "basename": "abc123.jpg",
            "split": "train",
            "sha256": "x",
        },
        "training": None,
    }
    phantom = tmp_path / "data" / data_rel
    h = calculate_file_hash(
        phantom,
        algorithm="md5",
        unified_snapshot=snap,
        relative_path=data_rel,
        logger=None,
    )
    assert h == "deadbeefcafe"
