"""Integration: :class:`mb.data.upscale.ImageUpscaler` on review-tree layout (no network)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from mb.data.upscale import MIN_DIMENSION_TARGET, ImageUpscaler


def test_image_upscaler_makes_small_review_image_at_least_min_target(tmp_path: Path) -> None:
    review = tmp_path / "small_images_review"
    cat_dir = review / "coherent"
    cat_dir.mkdir(parents=True)
    # Below MIN_DIMENSION_TARGET so upscale path runs (not copy-only skip)
    small = cat_dir / "tiny.jpg"
    Image.new("RGB", (120, 100), (40, 200, 100)).save(small, quality=92)

    upscaler = ImageUpscaler(review_dir=review)
    assert upscaler.run() is True
    assert upscaler.stats["upscaled"]["coherent"] >= 1

    out_dir = review / "upscaled_small_images" / "coherent"
    outs = list(out_dir.glob("*.jpg"))
    assert len(outs) >= 1
    with Image.open(outs[0]) as im:
        w, h = im.size
        assert min(w, h) >= MIN_DIMENSION_TARGET
