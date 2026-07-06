"""SD3: MMDiT transformer denoiser (no UNet) + up to three text encoders. Reserved."""

from __future__ import annotations

from typing import Any, Dict

from mb.models.generation_architectures.types import (
    BaseArchitectureDescriptor,
    BaseGenerationArchitecture,
)

DESCRIPTOR = BaseArchitectureDescriptor(
    architecture=BaseGenerationArchitecture.STABLE_DIFFUSION_3,
    pipeline_class_hint="StableDiffusion3Pipeline",
    text_encoder_subfolders=("text_encoder", "text_encoder_2", "text_encoder_3"),
    text_encoder_types=("clip", "clip", "t5"),
    unet_subfolder=None,
    transformer_subfolder="transformer",
    lora_target_modules=(),
    implemented=False,
)


def matches_model_index(index: Dict[str, Any]) -> bool:
    class_name = str(index.get("_class_name", ""))
    has_transformer = "transformer" in index
    has_unet = "unet" in index
    return has_transformer and not has_unet and "StableDiffusion3" in class_name
