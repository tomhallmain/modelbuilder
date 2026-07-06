"""
Base image-generation model architecture registry for
:attr:`~mb.models.types.ModelType.IMAGE_GENERATION` /
:attr:`~mb.models.types.ModelType.IMAGE_GENERATION_LORA`.

Unlike an image classifier (one backbone shape: images in, class logits out), diffusion
base checkpoints have meaningfully different component structures depending on
architecture — Stable Diffusion 1.x/2.x has a UNet denoiser and one CLIP text encoder;
SDXL adds a second text encoder plus extra conditioning; SD3/Flux/Chroma replace the UNet
with a transformer denoiser entirely. Which components to load, which layers a LoRA
adapter should target, and how conditioning is built are all architecture-specific — none
of that can be a single hardcoded constant the way it can for a classifier backbone. This
package is the equivalent of :mod:`mb.models.frameworks.registry` for the classification
side: one module per recognized base architecture (descriptor + `model_index.json`
detection heuristic), composed by :mod:`~mb.models.generation_architectures.registry`.

:attr:`BaseGenerationArchitecture.FLUX` and :attr:`BaseGenerationArchitecture.STABLE_DIFFUSION_1X`
are implemented end to end (see :mod:`mb.training.lora_diffusion_trainer`) — Flux is the
recommended default for new LoRAs (a current-generation model with meaningfully better
output quality/coherence than SD 1.x); SD 1.x is kept working alongside it (smaller,
faster to iterate on, and still the most widely available LoRA training data/tooling to
compare against), not as the recommended starting point.
:attr:`BaseGenerationArchitecture.CHROMA` is also implemented, reusing the Flux training
path under an explicit, flagged assumption (see ``chroma.py``) that Chroma checkpoints are
distributed in Flux-compatible ``diffusers`` format — this has not been verified against a
real Chroma checkpoint. :attr:`BaseGenerationArchitecture.STABLE_DIFFUSION_XL`,
:attr:`BaseGenerationArchitecture.STABLE_DIFFUSION_3`, and
:attr:`BaseGenerationArchitecture.Z_IMAGE_TURBO` remain reserved, unimplemented members —
mirroring how :attr:`~mb.models.types.ModelType.OBJECT_DETECTION` is a reserved,
unimplemented ``ModelType`` today. Z_IMAGE_TURBO in particular is a placeholder name only:
there isn't yet enough confirmed information about its actual architecture/``diffusers``
support to write real loading code against, and guessing at API details for a model this
under-documented would very likely be wrong rather than merely incomplete — implementing
it needs a concrete reference (e.g. a HF repo id and/or a working reference training
script) rather than another best-effort guess.
"""

from __future__ import annotations

from mb.models.generation_architectures.adapter_detection import looks_like_lora_adapter
from mb.models.generation_architectures.registry import detect_base_architecture, get_descriptor
from mb.models.generation_architectures.types import (
    BaseArchitectureDescriptor,
    BaseGenerationArchitecture,
)

__all__ = [
    "BaseArchitectureDescriptor",
    "BaseGenerationArchitecture",
    "detect_base_architecture",
    "get_descriptor",
    "looks_like_lora_adapter",
]
