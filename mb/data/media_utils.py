"""
Shared helpers for visual media (GIF frames, video frames) on the image-classification path.

Used by :mod:`mb.data.convert` and :mod:`mb.data.gather`. Video frame extraction uses
``imageio`` (see project requirements).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional, Tuple

if TYPE_CHECKING:
    from PIL import Image as PILImage

from mb.data.file_types import configured_video_suffixes

# Match mb.data.convert: max edge before downscale
_MAX_JPEG_EDGE = 4000
_JPEG_QUALITY = 95


def pil_gif_frame_count(path: Path) -> Optional[int]:
    """Return Pillow's ``n_frames`` for a GIF, or ``None`` if not a readable GIF."""
    try:
        from PIL import Image

        with Image.open(path) as im:
            if im.format != "GIF":
                return None
            return int(getattr(im, "n_frames", 1) or 1)
    except OSError:
        return None


def is_configured_video_suffix(path: Path) -> bool:
    return path.suffix.lower() in configured_video_suffixes()


def classify_convert_source(
    path: Path,
    *,
    image_classification: bool,
) -> Tuple[Literal["static", "extract"], Optional[str]]:
    """
    Decide whether *path* is converted as a normal still image or needs a random frame.

    When *image_classification* is False, every file is treated as ``static`` (videos
    are not expected in the scan).

    Returns:
        (``static`` | ``extract``, reason for extract or ``None``)
    """
    if not image_classification:
        return ("static", None)
    suf = path.suffix.lower()
    if suf in configured_video_suffixes():
        return ("extract", "video")
    if suf == ".gif":
        n = pil_gif_frame_count(path)
        if n is None:
            return ("static", None)
        if n <= 1:
            return ("static", None)
        return ("extract", "animated_gif")
    return ("static", None)


def pil_image_to_jpeg_normalized(img: "PILImage.Image", target_path: Path) -> bool:
    """
    Save a PIL image as JPEG with the same RGB/downscale rules as :class:`~mb.data.convert.ImageConverter`.
    """
    try:
        from PIL import Image

        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGB")
        elif img.mode != "RGB":
            img = img.convert("RGB")

        width, height = img.size
        max_dimension = max(width, height)
        if max_dimension > _MAX_JPEG_EDGE:
            scale = _MAX_JPEG_EDGE / max_dimension
            new_w = int(width * scale)
            new_h = int(height * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)

        target_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(target_path, "JPEG", quality=_JPEG_QUALITY, optimize=True)
        return target_path.exists() and target_path.stat().st_size > 0
    except OSError:
        return False


def extract_random_gif_frame_to_jpeg(
    source_path: Path,
    target_path: Path,
    rng: random.Random,
) -> bool:
    """Pick a random frame from a multi-frame GIF and write normalized JPEG to *target_path*."""
    try:
        from PIL import Image

        with Image.open(source_path) as im:
            n = int(getattr(im, "n_frames", 1) or 1)
            if n <= 1:
                return False
            idx = rng.randrange(0, n)
            im.seek(idx)
            frame = im.convert("RGB")
        return pil_image_to_jpeg_normalized(frame, target_path)
    except OSError:
        return False


def extract_random_video_frame_to_jpeg(
    source_path: Path,
    target_path: Path,
    rng: random.Random,
) -> bool:
    """
    Decode one random frame from a video file and write normalized JPEG.

    Requires ``imageio`` and typically ``imageio-ffmpeg`` for common containers.
    """
    try:
        import imageio.v2 as imageio
        import numpy as np
        from PIL import Image
    except ImportError:
        return False

    reader = imageio.get_reader(source_path)
    try:
        try:
            n = reader.count_frames()
        except Exception:
            n = None
        if n is None or n == float("inf") or n < 1:
            data = reader.get_data(0)
        else:
            ni = int(n)
            idx = rng.randrange(0, ni)
            data = reader.get_data(idx)
    except Exception:
        try:
            data = reader.get_data(0)
        except Exception:
            return False
    finally:
        try:
            reader.close()
        except Exception:
            pass

    arr = np.asarray(data)
    if arr.ndim == 2:
        return False
    if arr.shape[-1] == 4:
        img = Image.fromarray(arr[..., :3], mode="RGB")
    elif arr.shape[-1] == 3:
        img = Image.fromarray(arr, mode="RGB")
    else:
        img = Image.fromarray(arr).convert("RGB")
    return pil_image_to_jpeg_normalized(img, target_path)
