"""
Keep a page's image-size field consistent with its architecture field.

A resolution-locked backbone (SigLIP2, ViT) accepts only its native input size, while the
pipeline's ``data.image_size`` default is shared by every architecture. Pages that pre-fill
the size from the pipeline, or start from a fixed 224, would otherwise hand such a backbone a
size it cannot take: an error at training or conversion time, or a wrong preprocessing record
in an exported bundle.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import QLineEdit, QSpinBox

from mb.models.preprocessing import is_resolution_locked, preprocessing_spec_for
from mb.models.types import ArchitectureType


def native_image_size(architecture_text: str) -> Optional[int]:
    """Native input size when *architecture_text* names a resolution-locked architecture."""
    arch = ArchitectureType.try_from(architecture_text)
    if arch is None or not is_resolution_locked(arch):
        return None
    return preprocessing_spec_for(arch).image_size


def apply_native_image_size(architecture_text: str, image_size: QSpinBox) -> bool:
    """Set *image_size* to the architecture's native size if it has one; True if it did."""
    native = native_image_size(architecture_text)
    if native is None:
        return False
    image_size.setValue(native)
    return True


def sync_image_size_to_architecture(architecture: QLineEdit, image_size: QSpinBox) -> None:
    """
    Set the size to the native one whenever the architecture becomes a locked one.

    Only an exact architecture id triggers it, so typing a name, or entering a LoRA base
    model path on the Train page, leaves the size alone. A size restored or typed after the
    architecture is set still wins, which is what a model trained with
    ``--allow-resolution-mismatch`` needs.
    """
    architecture.textChanged.connect(lambda text: apply_native_image_size(text, image_size))
