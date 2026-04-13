"""
Shared ``data.image_size`` values for tests above the default 224.

The codebase does not pick a single globally optimal resolution: that depends on the
architecture (patch size for ViTs, stride alignment for CNNs), GPU memory, and the
original capture resolution. For **regression tests** we use a fixed size **above 300px**
(as requested for models trained above the usual 224 baseline):

``HIGH_RES_PIPELINE_IMAGE_SIZE`` (**320**) — divisible by 32 (common CNN alignment),
modest memory vs 384/448, and typical for ResNet-style fine-tuning with adaptive pooling.
"""

from __future__ import annotations

HIGH_RES_PIPELINE_IMAGE_SIZE = 320
