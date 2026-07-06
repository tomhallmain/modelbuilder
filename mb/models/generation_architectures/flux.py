"""
Flux: transformer denoiser (no UNet) + CLIP (pooled) + T5 (sequence) text encoders.
Implemented; the recommended default base architecture for new LoRAs — see
:mod:`mb.training.lora_diffusion_trainer` for the training loop (flow-matching loss,
latent packing) and its confidence note on this specific path.
"""

from __future__ import annotations

from typing import Any, Dict

from mb.models.generation_architectures.types import (
    BaseArchitectureDescriptor,
    BaseGenerationArchitecture,
)

DESCRIPTOR = BaseArchitectureDescriptor(
    architecture=BaseGenerationArchitecture.FLUX,
    pipeline_class_hint="FluxPipeline",
    text_encoder_subfolders=("text_encoder", "text_encoder_2"),
    text_encoder_types=("clip", "t5"),
    unet_subfolder=None,
    transformer_subfolder="transformer",
    # Attention projection layers only (present in both Flux's dual-stream
    # "transformer_blocks" and single-stream "single_transformer_blocks"). Many Flux
    # LoRA trainers also target the single-stream blocks' fused MLP projections
    # ("proj_mlp"/"proj_out") and the dual-stream cross-branch projections
    # ("add_q_proj"/"add_k_proj"/"add_v_proj"/"to_add_out") for a stronger adapter —
    # left out here to keep this first pass to the subset common to every block type.
    lora_target_modules=("to_k", "to_q", "to_v", "to_out.0"),
    implemented=True,
)


def matches_model_index(index: Dict[str, Any]) -> bool:
    class_name = str(index.get("_class_name", ""))
    has_transformer = "transformer" in index
    has_unet = "unet" in index
    return has_transformer and not has_unet and "Flux" in class_name
