"""Integration: :class:`mb.data.convert.ImageConverter` on real PNG/JPEG files."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from mb.data.convert import JPEG_IMAGES_DIR, ImageConverter


def _write_png(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color).save(path, format="PNG")


def test_image_converter_pngs_to_jpeg_images_subdir(tmp_path: Path) -> None:
    raw = tmp_path / "raw_data"
    cls_dir = raw / "coherent"
    _write_png(cls_dir / "a.png", (40, 80, 120))
    _write_png(cls_dir / "b.png", (200, 10, 30))

    converter = ImageConverter(raw_data_dir=raw)
    assert converter.run() is True

    out = cls_dir / JPEG_IMAGES_DIR
    assert out.is_dir()
    jpgs = sorted(out.glob("*.jpg"))
    assert len(jpgs) == 2
    for p in jpgs:
        assert p.stat().st_size > 0
        with Image.open(p) as im:
            assert im.format == "JPEG"

    assert converter.stats["files_converted"] == 2
    assert converter.stats["files_copied"] == 0
