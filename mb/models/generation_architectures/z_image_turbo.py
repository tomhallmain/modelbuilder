"""
Z-Image-Turbo — reserved, **not implemented**.

This module exists only to give the enum member a name and a place to fill in real
details later. Unlike the other reserved architectures in this package (SDXL, SD3), whose
component shapes are well-documented and confidently known even though no training code
has been written for them yet, there isn't enough reliably confirmed information about
Z-Image-Turbo's actual architecture or its ``diffusers`` integration (pipeline/model class
names, component subfolder layout, training objective) to write real loading code against
— doing so would be guessing at a fabricated API rather than implementing a known one, and
would very likely produce confidently wrong code rather than merely incomplete code.

Implementing this needs a concrete reference first: a Hugging Face repo id for a real
checkpoint, and/or a link to an official or community training script, so the descriptor
below can be filled in against actual facts rather than assumptions.
"""

from __future__ import annotations

from typing import Any, Dict

from mb.models.generation_architectures.types import (
    BaseArchitectureDescriptor,
    BaseGenerationArchitecture,
)

DESCRIPTOR = BaseArchitectureDescriptor(
    architecture=BaseGenerationArchitecture.Z_IMAGE_TURBO,
    pipeline_class_hint="",
    text_encoder_subfolders=(),
    text_encoder_types=(),
    unet_subfolder=None,
    transformer_subfolder=None,
    lora_target_modules=(),
    implemented=False,
)


def matches_model_index(index: Dict[str, Any]) -> bool:
    # No confirmed detection signature — always declines to match; see module docstring.
    return False
