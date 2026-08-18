"""
Hugging Face backbones: registry wiring, offline construction, and the preprocessing
contract read from each checkpoint.

Requires optional ``transformers`` (skipped via marker + importorskip). The specs and hub
ids are plain data, so those are checked unconditionally.
"""

from __future__ import annotations

import pytest

from mb.models.frameworks.pytorch.hf_architectures import SIGLIP2_HUB_IDS
from mb.models.preprocessing import (
    SIGLIP_MEAN,
    SIGLIP_STD,
    preprocessing_spec_for,
    resolve_image_size,
)
from mb.models.types import ArchitectureType, FrameworkType

_SIGLIP2 = (
    ArchitectureType.SIGLIP2_BASE_PATCH16_224,
    ArchitectureType.SIGLIP2_BASE_PATCH16_256,
    ArchitectureType.SIGLIP2_BASE_PATCH16_384,
)

# Native size encoded in each architecture id, which must match the checkpoint's
# preprocessor_config.json.
_NATIVE_SIZES = {
    ArchitectureType.SIGLIP2_BASE_PATCH16_224: 224,
    ArchitectureType.SIGLIP2_BASE_PATCH16_256: 256,
    ArchitectureType.SIGLIP2_BASE_PATCH16_384: 384,
}


@pytest.mark.parametrize("arch", _SIGLIP2)
def test_every_siglip2_architecture_has_a_hub_id(arch: ArchitectureType) -> None:
    """The architecture id is the hub id with dashes, so the two cannot drift apart."""
    assert SIGLIP2_HUB_IDS[arch.value] == f"google/{arch.value.replace('_', '-')}"


@pytest.mark.parametrize("arch", _SIGLIP2)
def test_siglip2_preprocessing_is_not_imagenet(arch: ArchitectureType) -> None:
    """
    SigLIP normalizes to [-1, 1], not by ImageNet channel statistics.

    Training these through the ImageNet defaults would fit the model on a different input
    distribution than the pretrained weights expect and record the wrong contract in the
    export manifest, with nothing raising.
    """
    spec = preprocessing_spec_for(arch)
    assert spec.normalize_mean == SIGLIP_MEAN == (0.5, 0.5, 0.5)
    assert spec.normalize_std == SIGLIP_STD == (0.5, 0.5, 0.5)
    assert spec.image_size == _NATIVE_SIZES[arch]
    assert spec.resolution_locked is True


@pytest.mark.parametrize("arch", _SIGLIP2)
def test_wrong_image_size_is_rejected_unless_explicitly_allowed(arch: ArchitectureType) -> None:
    native = _NATIVE_SIZES[arch]
    other = 320 if native != 320 else 224

    with pytest.raises(ValueError):
        resolve_image_size(arch, other)

    assert resolve_image_size(arch, other, allow_mismatch=True) == other
    assert resolve_image_size(arch, native) == native


def test_resnet_is_unaffected_by_siglip_registration() -> None:
    """Registering fixed-resolution backbones must not start validating the ResNet path."""
    spec = preprocessing_spec_for(ArchitectureType.RESNET34)
    assert spec.normalize_mean != SIGLIP_MEAN
    assert spec.resolution_locked is False
    assert resolve_image_size(ArchitectureType.RESNET34, 320) == 320


@pytest.mark.requires_transformers
@pytest.mark.parametrize("arch", _SIGLIP2)
def test_registry_factory_forward_offline(arch: ArchitectureType) -> None:
    """
    Each architecture is registered and runs a forward pass at its native resolution.

    ``pretrained=False`` must not touch the network: the suite has to run offline, so a
    factory that reached the hub here would make it fail without connectivity.
    """
    pytest.importorskip("transformers")
    torch = pytest.importorskip("torch")
    import mb.models.frameworks.pytorch.hf_architectures  # noqa: F401 — register side effects
    from mb.models.frameworks.registry import get_architecture

    factory = get_architecture(FrameworkType.PYTORCH, arch)
    assert factory is not None

    model = factory(num_classes=3, pretrained=False)
    size = _NATIVE_SIZES[arch]
    y = model(torch.randn(1, 3, size, size))

    # A bare tensor, not an ImageClassifierOutput: the training loop feeds this straight
    # into a loss function.
    assert torch.is_tensor(y)
    assert y.shape == (1, 3)


@pytest.mark.requires_transformers
def test_head_parameters_are_a_strict_subset_of_all_parameters() -> None:
    """The frozen phase unfreezes exactly the head, so it must be identifiable and smaller."""
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    from mb.models.frameworks.pytorch.hf_architectures import create_siglip2

    model = create_siglip2(
        ArchitectureType.SIGLIP2_BASE_PATCH16_224, num_classes=5, pretrained=False
    )
    head = list(model.head_parameters())
    all_params = list(model.parameters())

    assert head, "no head parameters exposed"
    assert len(head) < len(all_params)
    head_ids = {id(p) for p in head}
    assert head_ids.issubset({id(p) for p in all_params})
    # The head is the layer sized to the class count.
    assert any(p.shape[0] == 5 for p in head)


@pytest.mark.requires_transformers
def test_frozen_phase_unfreezes_only_the_head() -> None:
    """
    Reproduces what ``PyTorchTrainer.train`` does in its frozen phase.

    Matching on ``fc``/``classifier`` alone would work here by luck; this pins the
    ``head_parameters()`` contract the trainer actually prefers.
    """
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    from mb.models.frameworks.pytorch.hf_architectures import create_siglip2

    model = create_siglip2(
        ArchitectureType.SIGLIP2_BASE_PATCH16_224, num_classes=2, pretrained=False
    )
    for param in model.parameters():
        param.requires_grad = False
    for param in model.head_parameters():
        param.requires_grad = True

    trainable = [p for p in model.parameters() if p.requires_grad]
    assert trainable
    assert len(trainable) == len(list(model.head_parameters()))


def test_unknown_siglip2_architecture_is_rejected() -> None:
    """Rejected on the id, before transformers is needed — so this runs without it."""
    from mb.models.frameworks.pytorch.hf_architectures import create_siglip2

    with pytest.raises(ValueError):
        create_siglip2("siglip2_base_patch16_999", num_classes=2, pretrained=False)
