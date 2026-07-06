"""Composes the per-architecture modules into a lookup + best-effort detection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from mb.models.generation_architectures import (
    chroma,
    flux,
    stable_diffusion_1x,
    stable_diffusion_3,
    stable_diffusion_xl,
    z_image_turbo,
)
from mb.models.generation_architectures.types import (
    BaseArchitectureDescriptor,
    BaseGenerationArchitecture,
)

# Order matters only in that the first matching module wins; each module's
# ``matches_model_index`` is written to be mutually exclusive with the others in practice
# (UNet vs. transformer, encoder count, ``_class_name`` substring).
_ARCHITECTURE_MODULES = (
    stable_diffusion_1x,
    stable_diffusion_xl,
    stable_diffusion_3,
    flux,
    chroma,
    z_image_turbo,
)

_DESCRIPTORS: Dict[BaseGenerationArchitecture, BaseArchitectureDescriptor] = {
    module.DESCRIPTOR.architecture: module.DESCRIPTOR for module in _ARCHITECTURE_MODULES
}


def get_descriptor(architecture: BaseGenerationArchitecture) -> BaseArchitectureDescriptor:
    return _DESCRIPTORS[architecture]


def detect_base_architecture(base_model: str) -> Optional[BaseGenerationArchitecture]:
    """
    Best-effort detection from a local diffusers checkpoint directory's ``model_index.json``.

    Returns ``None`` when detection isn't possible or the shape isn't recognized — a hub id
    (e.g. ``"runwayml/stable-diffusion-v1-5"``, not a local path) isn't resolved here (would
    need a network call to fetch the file), and callers should treat ``None`` as "the user
    must pass ``--base-model-architecture`` explicitly," not as any particular architecture.
    """
    index_path = Path(base_model) / "model_index.json"
    if not index_path.is_file():
        return None
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(index, dict):
        return None

    for module in _ARCHITECTURE_MODULES:
        if module.matches_model_index(index):
            return module.DESCRIPTOR.architecture
    return None
