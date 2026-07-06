"""
Chroma: a community-trained model derived from Flux.1 Schnell's transformer architecture
(reportedly pruned/retrained relative to the original Flux, and — unlike Flux-dev — not
guidance-distilled, so it has no meaningful ``guidance_embeds`` conditioning).

**Confidence caveat (important):** this descriptor assumes Chroma checkpoints are
distributed in Flux-compatible ``diffusers`` format — i.e. loadable via the same
``FluxTransformer2DModel``/``FluxPipeline`` classes used for Flux itself, just with
different trained weights and (per public discussion of the project) a modified/pruned
transformer config. This assumption has **not** been verified against a real Chroma
checkpoint or a specific ``diffusers`` release's Chroma support. If a checkpoint instead
ships a dedicated ``ChromaTransformer2DModel``/``ChromaPipeline`` with a different API
shape, or if key attributes (e.g. ``guidance_embeds``, block counts) differ enough to
break ``FluxTransformer2DModel.from_pretrained``, this file and the shared Flux/Chroma
training path in :mod:`mb.training.lora_diffusion_trainer` need revisiting against an
actual checkpoint before relying on them.
"""

from __future__ import annotations

from typing import Any, Dict

from mb.models.generation_architectures.types import (
    BaseArchitectureDescriptor,
    BaseGenerationArchitecture,
)

DESCRIPTOR = BaseArchitectureDescriptor(
    architecture=BaseGenerationArchitecture.CHROMA,
    pipeline_class_hint="FluxPipeline (assumed Flux-compatible loading; not verified — see module docstring)",
    text_encoder_subfolders=("text_encoder", "text_encoder_2"),
    text_encoder_types=("clip", "t5"),
    unet_subfolder=None,
    transformer_subfolder="transformer",
    lora_target_modules=("to_k", "to_q", "to_v", "to_out.0"),
    implemented=True,
)


def matches_model_index(index: Dict[str, Any]) -> bool:
    class_name = str(index.get("_class_name", ""))
    has_transformer = "transformer" in index
    has_unet = "unet" in index
    return has_transformer and not has_unet and "Chroma" in class_name
