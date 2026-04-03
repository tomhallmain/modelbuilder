"""
Synthetic image trees for tests — matches :class:`mb.data.dataset.DatasetCreator` raw layout.

Layout (under *raw_data_dir*):

    <class_name>/JPEG_IMAGES/*.jpg

Class names default to :data:`mb.data.dataset.CLASS_NAMES`. JPEG files are random RGB
noise saved large enough to satisfy :data:`mb.data.dataset.MIN_FILE_SIZE` (6 KiB).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Mapping, Sequence

from mb.data.dataset import CLASS_NAMES, MIN_FILE_SIZE

try:
    from PIL import Image
except ImportError as e:  # pragma: no cover
    raise ImportError("tests require Pillow") from e


def _split_counts(total: int, n_classes: int) -> list[int]:
    if total < n_classes:
        raise ValueError(f"total_images ({total}) must be >= number of classes ({n_classes})")
    base, rem = divmod(total, n_classes)
    return [base + (1 if i < rem else 0) for i in range(n_classes)]


def _random_rgb_image(width: int, height: int, seed: int) -> Image.Image:
    rng = random.Random(seed)
    data = bytes(rng.randrange(256) for _ in range(width * height * 3))
    return Image.frombytes("RGB", (width, height), data)


def _save_jpeg_meeting_min_size(path: Path, seed: int, min_bytes: int = MIN_FILE_SIZE) -> int:
    """Write a JPEG at *path* with size >= *min_bytes*. Returns file size in bytes."""
    width = height = 96
    for attempt in range(12):
        img = _random_rgb_image(width, height, seed + attempt * 9973)
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(path, format="JPEG", quality=92, optimize=False)
        size = path.stat().st_size
        if size >= min_bytes:
            return size
        width += 64
        height += 64
    raise RuntimeError(f"Could not write JPEG >= {min_bytes} bytes at {path}")


def build_synthetic_raw_data_dir(
    raw_data_dir: Path,
    *,
    total_images: int = 100,
    seed: int = 42,
    class_names: Sequence[str] | None = None,
    min_bytes: int = MIN_FILE_SIZE,
) -> Mapping[str, int]:
    """
    Create a raw data tree compatible with :class:`~mb.data.dataset.DatasetCreator`.

    Args:
        raw_data_dir: Root directory (e.g. ``tmp_path / "raw_data"``).
        total_images: Total ``.jpg`` files to create (split evenly across classes).
        seed: Base seed for reproducible image bytes.
        class_names: Defaults to ``mb.data.dataset.CLASS_NAMES`` (three classes).
        min_bytes: Minimum JPEG file size (must be ``>= MIN_FILE_SIZE`` for dataset creation).

    Returns:
        Mapping of class name → number of images written.
    """
    names = list(class_names) if class_names is not None else list(CLASS_NAMES)
    n_classes = len(names)
    if n_classes == 0:
        raise ValueError("class_names must not be empty")
    counts = _split_counts(total_images, n_classes)
    out: dict[str, int] = {}
    image_index = 0
    for class_name, n in zip(names, counts):
        jpeg_dir = raw_data_dir / class_name / "JPEG_IMAGES"
        jpeg_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            path = jpeg_dir / f"synth_{image_index:04d}.jpg"
            _save_jpeg_meeting_min_size(path, seed=seed + image_index, min_bytes=min_bytes)
            image_index += 1
        out[class_name] = n
    return out
