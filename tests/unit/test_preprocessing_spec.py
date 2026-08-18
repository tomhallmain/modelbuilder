"""
Per-architecture preprocessing contract (``mb.models.preprocessing``).

The behavior that matters most here is what happens for an architecture with no
registration: it must keep the ImageNet defaults and must not be size-validated, because
the models this pipeline has already produced train a fully convolutional ResNet at 320.
"""

from __future__ import annotations

import pytest

from mb.models.preprocessing import (
    DEFAULT_IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    PreprocessingSpec,
    has_registered_spec,
    preprocessing_spec_for,
    register_preprocessing_spec,
    resolve_image_size,
)
from mb.models.types import ArchitectureType


@pytest.fixture
def registered_locked(monkeypatch: pytest.MonkeyPatch):
    """A resolution-locked architecture registered into an isolated registry."""
    import mb.models.preprocessing as mod

    monkeypatch.setattr(mod, "_SPECS", {}, raising=True)
    spec = PreprocessingSpec(
        image_size=256,
        normalize_mean=(0.5, 0.5, 0.5),
        normalize_std=(0.5, 0.5, 0.5),
        resolution_locked=True,
    )
    register_preprocessing_spec("locked_backbone", spec)
    return spec


def test_unregistered_architecture_gets_imagenet_defaults() -> None:
    spec = preprocessing_spec_for(ArchitectureType.RESNET34)
    assert spec.normalize_mean == IMAGENET_MEAN
    assert spec.normalize_std == IMAGENET_STD
    assert spec.image_size == DEFAULT_IMAGE_SIZE


def test_unregistered_architecture_is_not_size_validated() -> None:
    """
    The existing 320px ResNet workflow must not start warning or failing.

    A fully convolutional backbone has no single correct input size, so "no registration"
    means "no opinion" rather than "expects 224".
    """
    assert not has_registered_spec(ArchitectureType.RESNET34)
    assert resolve_image_size(ArchitectureType.RESNET34, 320) == 320
    assert resolve_image_size(None, 320) == 320


def test_none_image_size_falls_back_to_the_spec_size() -> None:
    assert resolve_image_size(ArchitectureType.RESNET34, None) == DEFAULT_IMAGE_SIZE


def test_registered_spec_is_returned_and_size_overridden(registered_locked) -> None:
    spec = preprocessing_spec_for("locked_backbone")
    assert spec.normalize_mean == (0.5, 0.5, 0.5)
    assert spec.image_size == 256

    resized = preprocessing_spec_for("locked_backbone", 384)
    assert resized.image_size == 384
    assert resized.normalize_mean == (0.5, 0.5, 0.5)  # normalization survives the resize


def test_resolution_locked_mismatch_raises(registered_locked) -> None:
    with pytest.raises(ValueError):
        resolve_image_size("locked_backbone", 224)


def test_resolution_locked_mismatch_allowed_explicitly(registered_locked) -> None:
    assert resolve_image_size("locked_backbone", 224, allow_mismatch=True) == 224


def test_resolution_locked_matching_size_is_fine(registered_locked) -> None:
    assert resolve_image_size("locked_backbone", 256) == 256


def test_manifest_dict_shape() -> None:
    spec = preprocessing_spec_for(ArchitectureType.RESNET34, 320)
    manifest = spec.to_manifest_dict()

    assert manifest["image_size"] == 320
    assert manifest["channels"] == 3
    # Lists, not tuples: this block is serialized straight into model_manifest.json.
    assert manifest["normalize_mean"] == list(IMAGENET_MEAN)
    assert manifest["normalize_std"] == list(IMAGENET_STD)
    assert manifest["resize_mode"] == "squash"


def test_with_image_size_returns_self_when_unchanged() -> None:
    spec = PreprocessingSpec(image_size=224)
    assert spec.with_image_size(224) is spec
    assert spec.with_image_size(256) is not spec


def test_duplicate_registration_requires_overwrite(registered_locked) -> None:
    with pytest.raises(ValueError):
        register_preprocessing_spec("locked_backbone", PreprocessingSpec())
    register_preprocessing_spec("locked_backbone", PreprocessingSpec(), overwrite=True)
    assert preprocessing_spec_for("locked_backbone").normalize_mean == IMAGENET_MEAN
