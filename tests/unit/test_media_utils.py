"""Unit tests for :mod:`mb.data.media_utils`."""

from __future__ import annotations

import random
from pathlib import Path

from PIL import Image

from mb.data.class_layout import CONVERTED_MEDIA_SUBDIR, VISUAL_MEDIA_REVIEW_SUBDIR
from mb.data.convert import ImageConverter
from mb.models.types import ModelType, VisualMediaSourceType
from mb.data.media_utils import (
    classify_convert_source,
    extract_random_gif_frame_to_jpeg,
    pil_gif_frame_count,
)


def test_pil_gif_frame_count_single(tmp_path: Path) -> None:
    p = tmp_path / "one.gif"
    Image.new("RGB", (4, 4), (1, 2, 3)).save(p, format="GIF")
    assert pil_gif_frame_count(p) == 1


def test_pil_gif_frame_count_multi(tmp_path: Path) -> None:
    p = tmp_path / "two.gif"
    a = Image.new("RGB", (4, 4), (255, 0, 0))
    b = Image.new("RGB", (4, 4), (0, 255, 0))
    a.save(p, save_all=True, append_images=[b], duration=100, loop=0, format="GIF")
    assert pil_gif_frame_count(p) == 2


def test_classify_convert_source(tmp_path: Path) -> None:
    png = tmp_path / "x.png"
    Image.new("RGB", (2, 2), (0, 0, 0)).save(png)
    assert classify_convert_source(png, model_type=ModelType.IMAGE_CLASSIFICATION) == VisualMediaSourceType.STATIC

    mp4 = tmp_path / "c.mp4"
    mp4.touch()
    assert classify_convert_source(mp4, model_type=ModelType.IMAGE_CLASSIFICATION) == VisualMediaSourceType.VIDEO_EXTRACT
    assert classify_convert_source(mp4, model_type=ModelType.OBJECT_DETECTION) == VisualMediaSourceType.STATIC

    g1 = tmp_path / "sg.gif"
    Image.new("RGB", (2, 2), (1, 1, 1)).save(g1, format="GIF")
    assert classify_convert_source(g1, model_type=ModelType.IMAGE_CLASSIFICATION) == VisualMediaSourceType.STATIC

    gm = tmp_path / "mg.gif"
    a = Image.new("RGB", (2, 2), (10, 0, 0))
    b = Image.new("RGB", (2, 2), (0, 10, 0))
    a.save(gm, save_all=True, append_images=[b], duration=50, loop=0, format="GIF")
    assert classify_convert_source(gm, model_type=ModelType.IMAGE_CLASSIFICATION) == VisualMediaSourceType.ANIMATED_GIF_EXTRACT


def test_extract_random_gif_frame_to_jpeg(tmp_path: Path) -> None:
    src = tmp_path / "m.gif"
    a = Image.new("RGB", (8, 8), (200, 0, 0))
    b = Image.new("RGB", (8, 8), (0, 200, 0))
    a.save(src, save_all=True, append_images=[b], duration=50, loop=0, format="GIF")
    out = tmp_path / "o.jpg"
    rng = random.Random(42)
    assert extract_random_gif_frame_to_jpeg(src, out, rng) is True
    assert out.exists() and out.stat().st_size > 0
    with Image.open(out) as im:
        assert im.format == "JPEG"


def test_image_converter_animated_gif_writes_converted_and_review(tmp_path: Path) -> None:
    raw = tmp_path / "raw_data"
    cls_dir = raw / "cls_a"
    cls_dir.mkdir(parents=True)
    gif_path = cls_dir / "anim.gif"
    f0 = Image.new("RGB", (12, 12), (100, 0, 0))
    f1 = Image.new("RGB", (12, 12), (0, 100, 0))
    f0.save(gif_path, save_all=True, append_images=[f1], duration=80, loop=0, format="GIF")

    c = ImageConverter(raw_data_dir=raw, model_type=ModelType.IMAGE_CLASSIFICATION)
    assert c.run() is True

    jpg = cls_dir / CONVERTED_MEDIA_SUBDIR / "anim.jpg"
    rev = cls_dir / VISUAL_MEDIA_REVIEW_SUBDIR / "anim.jpg"
    assert jpg.is_file() and jpg.stat().st_size > 0
    assert rev.is_file() and rev.stat().st_size > 0
    assert c.stats["files_visual_extracted"] == 1
    assert c.stats["files_converted"] == 0
