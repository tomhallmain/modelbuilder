"""Stable Diffusion 1.x/2.x: UNet denoiser + a single CLIP text encoder."""

from __future__ import annotations

from typing import Any, Dict

from mb.models.generation_architectures.types import (
    BaseArchitectureDescriptor,
    BaseGenerationArchitecture,
)

DESCRIPTOR = BaseArchitectureDescriptor(
    architecture=BaseGenerationArchitecture.STABLE_DIFFUSION_1X,
    pipeline_class_hint="StableDiffusionPipeline",
    text_encoder_subfolders=("text_encoder",),
    text_encoder_types=("clip",),
    unet_subfolder="unet",
    transformer_subfolder=None,
    lora_target_modules=("to_k", "to_q", "to_v", "to_out.0"),
    implemented=True,
)


def matches_model_index(index: Dict[str, Any]) -> bool:
    has_unet = "unet" in index
    n_text_encoders = sum(1 for k in index if k == "text_encoder" or k.startswith("text_encoder_"))
    return has_unet and n_text_encoders == 1
