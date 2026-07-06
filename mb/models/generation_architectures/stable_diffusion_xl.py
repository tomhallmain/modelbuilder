"""SDXL: UNet denoiser + two text encoders (CLIP ViT-L + OpenCLIP ViT-bigG). Reserved."""

from __future__ import annotations

from typing import Any, Dict

from mb.models.generation_architectures.types import (
    BaseArchitectureDescriptor,
    BaseGenerationArchitecture,
)

DESCRIPTOR = BaseArchitectureDescriptor(
    architecture=BaseGenerationArchitecture.STABLE_DIFFUSION_XL,
    pipeline_class_hint="StableDiffusionXLPipeline",
    text_encoder_subfolders=("text_encoder", "text_encoder_2"),
    text_encoder_types=("clip", "clip"),
    unet_subfolder="unet",
    transformer_subfolder=None,
    lora_target_modules=("to_k", "to_q", "to_v", "to_out.0"),
    implemented=False,
)


def matches_model_index(index: Dict[str, Any]) -> bool:
    has_unet = "unet" in index
    n_text_encoders = sum(1 for k in index if k == "text_encoder" or k.startswith("text_encoder_"))
    return has_unet and n_text_encoders >= 2
