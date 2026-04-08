"""Integration: :class:`mb.data.convert.ImageConverter` on real PNG/JPEG files."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from mb.data.class_layout import CONVERTED_MEDIA_SUBDIR
from mb.data.convert import ImageConverter


def _write_png(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color).save(path, format="PNG")


def test_image_converter_pngs_to_converted_subdir(tmp_path: Path) -> None:
    raw = tmp_path / "raw_data"
    cls_dir = raw / "coherent"
    _write_png(cls_dir / "a.png", (40, 80, 120))
    _write_png(cls_dir / "b.png", (200, 10, 30))

    converter = ImageConverter(raw_data_dir=raw)
    assert converter.run() is True

    out = cls_dir / CONVERTED_MEDIA_SUBDIR
    assert out.is_dir()
    jpgs = sorted(out.glob("*.jpg"))
    assert len(jpgs) == 2
    for p in jpgs:
        assert p.stat().st_size > 0
        with Image.open(p) as im:
            assert im.format == "JPEG"

    assert converter.stats["files_converted"] == 2
    assert converter.stats["files_copied"] == 0


def test_image_converter_resume_updates_same_snapshot_file(tmp_path: Path) -> None:
    """``run_id=`` loads ``snapshot_<run_id>.json`` and saves back to the same path."""
    raw = tmp_path / "raw_data"
    cls_dir = raw / "coherent"
    _write_png(cls_dir / "a.png", (40, 80, 120))

    first = ImageConverter(raw_data_dir=raw)
    assert first.run(skip_space_check=True) is True
    snaps = sorted(raw.glob("snapshot_*.json"))
    assert len(snaps) == 1
    rid = snaps[0].stem.removeprefix("snapshot_")
    mtime_before = snaps[0].stat().st_mtime_ns

    _write_png(cls_dir / "b.png", (1, 2, 3))
    second = ImageConverter(raw_data_dir=raw)
    assert second.run(skip_space_check=True, run_id=rid) is True
    snaps_after = sorted(raw.glob("snapshot_*.json"))
    assert len(snaps_after) == 1
    assert snaps_after[0].resolve() == snaps[0].resolve()
    assert snaps_after[0].stat().st_mtime_ns >= mtime_before
