"""
Caption sidecar-file convention for image-generation LoRA training data
(:attr:`~mb.models.types.ModelType.IMAGE_GENERATION_LORA`).

Each training image may have a paired plain-text caption file with the same stem
(``image123.jpg`` + ``image123.txt``) in the same directory — the convention used by most
LoRA/diffusion trainers (kohya-ss, diffusers examples, etc.). Captions are optional: a
missing sidecar means "no caption for this image" (some LoRA workflows train caption-less,
relying only on a trigger word), not an error.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

CAPTION_SUFFIX = ".txt"


def caption_path_for(image_path: Path) -> Path:
    """Sidecar caption path for *image_path* (same directory, same stem, ``.txt``)."""
    return image_path.with_suffix(CAPTION_SUFFIX)


def read_caption(image_path: Path) -> Optional[str]:
    """Read the caption sidecar for *image_path*, or ``None`` if it doesn't exist/is empty."""
    cap_path = caption_path_for(image_path)
    if not cap_path.is_file():
        return None
    try:
        text = cap_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return text or None


def write_caption(image_path: Path, text: str) -> Path:
    """Write *text* to the caption sidecar for *image_path* (overwrites if present)."""
    cap_path = caption_path_for(image_path)
    cap_path.parent.mkdir(parents=True, exist_ok=True)
    cap_path.write_text(text.strip() + "\n", encoding="utf-8")
    return cap_path
