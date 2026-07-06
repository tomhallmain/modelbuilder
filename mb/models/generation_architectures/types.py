"""Shared types for the base image-generation architecture registry (see package docstring)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


class BaseGenerationArchitecture(str, Enum):
    """Recognized base image-generation model architectures."""

    STABLE_DIFFUSION_1X = "stable_diffusion_1x"
    """SD 1.x/2.x: UNet denoiser + a single CLIP text encoder. Implemented; kept for
    compatibility/small-scale iteration, not the recommended starting point (see FLUX)."""

    STABLE_DIFFUSION_XL = "stable_diffusion_xl"
    """SDXL: UNet denoiser + two text encoders (CLIP ViT-L + OpenCLIP ViT-bigG). Reserved."""

    STABLE_DIFFUSION_3 = "stable_diffusion_3"
    """SD3: MMDiT transformer denoiser (no UNet) + up to three text encoders. Reserved."""

    FLUX = "flux"
    """Flux: transformer denoiser (no UNet) + CLIP (pooled) + T5 (sequence) text encoders.
    Implemented; the recommended default base architecture for new LoRAs."""

    CHROMA = "chroma"
    """Community Flux-derived model (pruned/retrained transformer denoiser, not
    guidance-distilled). Implemented by reusing the Flux training path, under the
    explicit, flagged assumption that Chroma checkpoints are Flux-compatible in
    ``diffusers`` — see ``chroma.py``; not verified against a real checkpoint."""

    Z_IMAGE_TURBO = "z_image_turbo"
    """Reserved — named per user request, but not enough confirmed architecture/diffusers
    support detail is available to implement real loading code responsibly; needs a
    concrete reference (repo id / training script) rather than a guess."""

    @classmethod
    def try_from(cls, raw: object) -> Optional["BaseGenerationArchitecture"]:
        if raw is None:
            return None
        s = str(raw).strip().lower()
        if not s:
            return None
        try:
            return cls(s)
        except ValueError:
            return None


@dataclass(frozen=True)
class BaseArchitectureDescriptor:
    """What a LoRA trainer/evaluator needs to know about one base architecture."""

    architecture: BaseGenerationArchitecture
    pipeline_class_hint: str
    """Informational: the diffusers pipeline class this architecture is normally loaded via."""
    text_encoder_subfolders: Tuple[str, ...]
    """Component subfolder names under the base model root, in encoder order."""
    text_encoder_types: Tuple[str, ...]
    """Encoder family per subfolder, same order (``"clip"`` or ``"t5"``) — they use different
    model/tokenizer classes and conditioning roles (e.g. Flux: pooled CLIP + sequence T5)."""
    unet_subfolder: Optional[str]
    """Denoiser subfolder name, or ``None`` for a transformer-based architecture."""
    transformer_subfolder: Optional[str]
    """Denoiser subfolder name for a transformer-based architecture, or ``None`` for a UNet one."""
    lora_target_modules: Tuple[str, ...]
    """Module name suffixes ``peft.LoraConfig(target_modules=...)`` should match."""
    implemented: bool
    """Whether :mod:`mb.training.lora_diffusion_trainer` actually supports this today."""
