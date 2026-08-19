"""
Per-architecture input preprocessing contract (resolution, normalization).

One :class:`PreprocessingSpec` is the single source of truth for three things that must
agree or inference silently degrades: the transforms training applies
(:mod:`mb.models.frameworks.pytorch.data_loader`), the ``preprocessing`` block recorded in
an exported bundle manifest (:mod:`mb.export.bundle`), and the size validation applied to
``--image-size``.

torchvision/Keras backbones use ImageNet statistics, which is what every architecture in
the registry resolved to before this module existed — so architectures without an explicit
registration keep exactly that behavior. Architectures whose pretrained weights expect
different statistics (SigLIP and other Hugging Face vision backbones) register their own
spec, sourced from the checkpoint's own preprocessor config rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

from mb.models.types import ArchitectureType, ResizeMode
from mb.utils.logging_setup import get_logger
from mb.utils.translations import _

logger = get_logger(__name__)

ArchitectureKey = Union[ArchitectureType, str]

# torchvision/Keras pretrained weights are trained against these; they were hardcoded in
# the transform builders and the export manifest before this module.
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

DEFAULT_IMAGE_SIZE = 224


@dataclass(frozen=True)
class PreprocessingSpec:
    """
    What a trained model expects its input to look like.

    Args:
        image_size: Native square input size for the backbone's pretrained weights.
        normalize_mean: Per-channel mean subtracted after scaling to ``[0, 1]``.
        normalize_std: Per-channel standard deviation divided after mean subtraction.
        channels: Input channel count (3 for every architecture registered today).
        resize_mode: How the source image is fitted. Recorded so a consumer reproducing
            preprocessing does not have to guess between the two, which produce visibly
            different crops.
        resize_size: Shortest-edge target before cropping, used only by
            :attr:`~mb.models.types.ResizeMode.SHORTEST_EDGE_CROP`. For that mode
            :attr:`image_size` is the *crop* size, and this is the larger intermediate.
        resolution_locked: True when the backbone cannot accept an input size other than
            :attr:`image_size` without interpolating pretrained position embeddings — true
            for fixed-grid vision transformers, false for fully convolutional backbones.
    """

    image_size: int = DEFAULT_IMAGE_SIZE
    normalize_mean: Tuple[float, float, float] = IMAGENET_MEAN
    normalize_std: Tuple[float, float, float] = IMAGENET_STD
    channels: int = 3
    resize_mode: ResizeMode = ResizeMode.SQUASH
    resize_size: Optional[int] = None
    resolution_locked: bool = False

    def resize_target_for(self, image_size: int) -> int:
        """
        Shortest-edge target when cropping to *image_size*.

        Scaled by the ratio this spec records rather than fixed, so the policy still holds
        when a backbone that tolerates other resolutions is trained at one: FocalNet's
        256-then-crop-224 becomes 366-then-crop-320 rather than cropping most of the frame
        away.
        """
        if self.resize_size is None or self.image_size <= 0:
            return int(image_size)
        return max(int(image_size), round(int(image_size) * self.resize_size / self.image_size))

    def with_image_size(self, image_size: int) -> "PreprocessingSpec":
        """Copy of this spec at *image_size* (normalization and resize policy unchanged)."""
        if int(image_size) == self.image_size:
            return self
        return PreprocessingSpec(
            image_size=int(image_size),
            normalize_mean=self.normalize_mean,
            normalize_std=self.normalize_std,
            channels=self.channels,
            resize_mode=self.resize_mode,
            # Rescaled so the recorded ratio survives, matching resize_target_for.
            resize_size=(
                None if self.resize_size is None else self.resize_target_for(int(image_size))
            ),
            resolution_locked=self.resolution_locked,
        )

    def to_manifest_dict(self) -> Dict[str, Any]:
        """The ``preprocessing`` block of an export manifest."""
        return {
            "image_size": int(self.image_size),
            "channels": int(self.channels),
            "normalize_mean": list(self.normalize_mean),
            "normalize_std": list(self.normalize_std),
            "resize_mode": self.resize_mode.value,
            "resize_size": int(self.resize_size) if self.resize_size is not None else None,
        }


DEFAULT_PREPROCESSING = PreprocessingSpec()

# SigLIP and ViT checkpoints normalize to [-1, 1] rather than by ImageNet channel
# statistics. Values read from each checkpoint's own preprocessor_config.json (image_mean,
# image_std, size, resample=2 => bilinear, which matches transforms.Resize's default).
# Hugging Face backbones do not share a preprocessing contract, so every one of them must
# appear below — inheriting the ImageNet default would be silently wrong rather than absent.
# Named for the values rather than for a model family: SigLIP introduced them here, but ViT
# uses the same ones, and a family-specific name would go stale on the next backbone.
HALF_MEAN: Tuple[float, float, float] = (0.5, 0.5, 0.5)
HALF_STD: Tuple[float, float, float] = (0.5, 0.5, 0.5)


def _fixed_grid_half_norm_spec(image_size: int) -> PreprocessingSpec:
    """Square-resize, half-normalized contract for a fixed patch-grid transformer."""
    return PreprocessingSpec(
        image_size=image_size,
        normalize_mean=HALF_MEAN,
        normalize_std=HALF_STD,
        # Position embeddings are learned per patch position on a fixed grid, so a
        # different input size changes the patch count and no longer matches the
        # pretrained embeddings.
        resolution_locked=True,
    )


def _shortest_edge_crop_spec(crop_size: int, resize_size: int) -> PreprocessingSpec:
    """ImageNet-normalized contract that preserves aspect ratio and centre-crops."""
    return PreprocessingSpec(
        image_size=crop_size,
        normalize_mean=IMAGENET_MEAN,
        normalize_std=IMAGENET_STD,
        resize_mode=ResizeMode.SHORTEST_EDGE_CROP,
        resize_size=resize_size,
    )


# Architectures whose pretrained weights need something other than the ImageNet defaults —
# which includes a different resize policy, not only different normalization.
_SPECS: Dict[str, PreprocessingSpec] = {
    ArchitectureType.SIGLIP2_BASE_PATCH16_224.value: _fixed_grid_half_norm_spec(224),
    ArchitectureType.SIGLIP2_BASE_PATCH16_256.value: _fixed_grid_half_norm_spec(256),
    ArchitectureType.SIGLIP2_BASE_PATCH16_384.value: _fixed_grid_half_norm_spec(384),
    ArchitectureType.SIGLIP2_BASE_PATCH16_512.value: _fixed_grid_half_norm_spec(512),
    ArchitectureType.VIT_BASE_PATCH16_224.value: _fixed_grid_half_norm_spec(224),
    # FocalNet keeps ImageNet statistics but crops rather than squashing, and has no absolute
    # position embeddings, so it tolerates other input sizes.
    ArchitectureType.FOCALNET_TINY.value: _shortest_edge_crop_spec(224, 256),
    ArchitectureType.FOCALNET_BASE.value: _shortest_edge_crop_spec(224, 256),
}


def _architecture_key(architecture: ArchitectureKey) -> str:
    if isinstance(architecture, ArchitectureType):
        return architecture.value
    return str(architecture).strip().lower()


def register_preprocessing_spec(
    architecture: ArchitectureKey,
    spec: PreprocessingSpec,
    overwrite: bool = False,
) -> None:
    """
    Register *spec* as the preprocessing contract for *architecture*.

    Raises:
        ValueError: If already registered and *overwrite* is False.
    """
    key = _architecture_key(architecture)
    if key in _SPECS and not overwrite:
        raise ValueError(f"Preprocessing spec already registered for '{key}'")
    _SPECS[key] = spec
    logger.debug("Registered preprocessing spec for %s", key)


def has_registered_spec(architecture: Optional[ArchitectureKey]) -> bool:
    """
    Whether *architecture* declares its own preprocessing contract.

    False means "no opinion", not "expects the ImageNet default at 224". Fully
    convolutional backbones accept any reasonable input size, so an unregistered
    architecture must not be size-validated against the default spec's 224.
    """
    if architecture is None:
        return False
    return _architecture_key(architecture) in _SPECS


def preprocessing_spec_for(
    architecture: Optional[ArchitectureKey],
    image_size: Optional[int] = None,
) -> PreprocessingSpec:
    """
    Resolve the preprocessing contract for *architecture*.

    Unregistered (and unknown) architectures resolve to the ImageNet default, which is what
    every torchvision and Keras backbone here expects. When *image_size* is given it
    replaces the spec's native size — call :func:`resolve_image_size` first if the request
    came from a user and should be validated rather than trusted.
    """
    spec = DEFAULT_PREPROCESSING
    if architecture is not None:
        spec = _SPECS.get(_architecture_key(architecture), DEFAULT_PREPROCESSING)
    if image_size is None:
        return spec
    return spec.with_image_size(image_size)


def resolve_image_size(
    architecture: Optional[ArchitectureKey],
    requested_image_size: Optional[int],
    *,
    allow_mismatch: bool = False,
) -> int:
    """
    Validate a requested input size against what *architecture* expects.

    A size differing from the architecture's native size is a warning for backbones that
    tolerate it and an error for :attr:`PreprocessingSpec.resolution_locked` ones, where
    training at the wrong size silently interpolates pretrained position embeddings and
    costs accuracy without failing.

    Architectures with no registered spec are not validated at all: a fully convolutional
    backbone has no single correct input size, and the baseline models this pipeline has
    already produced were trained at 320 on a ResNet whose torchvision weights are
    nominally 224.

    Returns:
        The image size to train/evaluate at.

    Raises:
        ValueError: If the architecture is resolution-locked, the sizes differ, and
            *allow_mismatch* is False.
    """
    spec = preprocessing_spec_for(architecture)
    if requested_image_size is None:
        return spec.image_size

    requested = int(requested_image_size)
    if requested == spec.image_size or not has_registered_spec(architecture):
        return requested

    arch_name = _architecture_key(architecture) if architecture is not None else "?"
    if spec.resolution_locked and not allow_mismatch:
        raise ValueError(
            _(
                "Architecture {arch} expects {native}x{native} input; {requested} was requested. "
                "Pass --allow-resolution-mismatch to train at a different size anyway."
            ).format(arch=arch_name, native=spec.image_size, requested=requested)
        )
    logger.warning(
        "Training %s at %dx%d instead of its native %dx%d",
        arch_name,
        requested,
        requested,
        spec.image_size,
        spec.image_size,
    )
    return requested
