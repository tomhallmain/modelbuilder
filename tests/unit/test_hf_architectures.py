"""
Hugging Face backbones: registry wiring, preprocessing contracts, and offline construction.

Parametrized over the backbone table rather than a hand-maintained list, so a newly declared
backbone is covered the moment it is added and cannot be merged untested.

Requires optional ``transformers`` for anything that builds a model; the table and the
preprocessing contracts are plain data and are checked unconditionally.
"""

from __future__ import annotations

import pytest

from mb.models.frameworks.pytorch.hf_architectures import (
    HF_BACKBONES,
    HF_HUB_IDS,
    problem_type_for,
)
from mb.models.preprocessing import (
    IMAGENET_MEAN,
    has_registered_spec,
    preprocessing_spec_for,
    resolve_image_size,
)
from mb.models.types import ArchitectureType, FrameworkType, LabelMode

_HF_ARCHITECTURES = sorted(HF_BACKBONES)


# --- Guards: the traps that breadth invites (see the backbone spec's testing section) ---


@pytest.mark.parametrize("arch", _HF_ARCHITECTURES)
def test_every_hf_backbone_declares_its_preprocessing(arch: str) -> None:
    """
    The highest-value test here: Hugging Face backbones share no preprocessing contract.

    SigLIP and ViT normalize to [-1, 1]; torchvision backbones use ImageNet statistics.
    Inheriting the registry default would train the model on a different input distribution
    than its pretrained weights expect and record the wrong contract in the export manifest —
    wrong twice in the same direction, and silent both times.
    """
    assert has_registered_spec(arch), f"{arch} has no registered PreprocessingSpec"


def test_backbone_table_and_preprocessing_table_agree() -> None:
    """
    The two tables live in different modules on purpose, so a cross-check replaces coupling.

    Preprocessing specs are registered on import of ``mb.models.preprocessing``, with no
    dependency on ``transformers`` — the export and evaluation paths resolve a spec without
    importing the PyTorch framework package at all. Moving the specs next to the hub ids
    would make a correct lookup depend on import order and fail silently.
    """
    import mb.models.preprocessing as preprocessing

    specs = set(preprocessing._SPECS)
    assert set(HF_BACKBONES) <= specs, f"backbones missing specs: {set(HF_BACKBONES) - specs}"


@pytest.mark.parametrize("arch", _HF_ARCHITECTURES)
def test_hf_backbones_are_not_imagenet_normalized(arch: str) -> None:
    """Every backbone currently in the table normalizes to [-1, 1], not by ImageNet stats."""
    spec = preprocessing_spec_for(arch)
    assert spec.normalize_mean != IMAGENET_MEAN
    assert spec.normalize_mean == (0.5, 0.5, 0.5)
    assert spec.normalize_std == (0.5, 0.5, 0.5)


@pytest.mark.parametrize("arch", _HF_ARCHITECTURES)
def test_every_hf_backbone_is_an_architecture_enum_value(arch: str) -> None:
    """A registry key that is not an enum member cannot survive TrainingRunArgs JSON."""
    assert ArchitectureType.try_from(arch) is not None


@pytest.mark.parametrize("arch", _HF_ARCHITECTURES)
def test_hub_id_matches_the_architecture_id(arch: str) -> None:
    """
    The architecture id names the checkpoint it loads, so the two cannot drift apart.

    ViT is the deliberate exception: its hub id carries an ``-in21k`` suffix because the
    pretraining checkpoint, which has no classification head, is what fine-tuning wants.
    """
    hub_id = HF_HUB_IDS[arch]
    expected = f"google/{arch.replace('_', '-')}"
    assert hub_id in (expected, f"{expected}-in21k")


# --- Resolution locking ---


@pytest.mark.parametrize("arch", _HF_ARCHITECTURES)
def test_fixed_grid_backbones_reject_a_mismatched_size(arch: str) -> None:
    """
    Position embeddings are learned per patch position, so the input size is not free.

    Training at another size silently interpolates them and costs accuracy without failing.
    """
    spec = preprocessing_spec_for(arch)
    assert spec.resolution_locked is True

    other = 320 if spec.image_size != 320 else 224
    with pytest.raises(ValueError):
        resolve_image_size(arch, other)
    assert resolve_image_size(arch, other, allow_mismatch=True) == other
    assert resolve_image_size(arch, spec.image_size) == spec.image_size


def test_torchvision_backbones_are_unaffected() -> None:
    """Registering fixed-resolution backbones must not start validating the ResNet path."""
    spec = preprocessing_spec_for(ArchitectureType.RESNET34)
    assert spec.normalize_mean == IMAGENET_MEAN
    assert spec.resolution_locked is False
    assert resolve_image_size(ArchitectureType.RESNET34, 320) == 320


# --- problem_type metadata ---


def test_problem_type_follows_the_label_mode() -> None:
    assert problem_type_for(LabelMode.SINGLE_LABEL) == "single_label_classification"
    assert problem_type_for(LabelMode.MULTI_LABEL) == "multi_label_classification"
    assert problem_type_for("multi_label") == "multi_label_classification"
    # Unknown and absent values fall back rather than producing an invalid config value.
    assert problem_type_for(None) == "single_label_classification"
    assert problem_type_for("nonsense") == "single_label_classification"


# --- Construction (needs transformers) ---


@pytest.mark.requires_transformers
@pytest.mark.parametrize("arch", _HF_ARCHITECTURES)
def test_registry_factory_forward_returns_requested_class_count(arch: str) -> None:
    """
    Each backbone is registered and produces logits for exactly the requested class count.

    The class count is the assertion that matters: a forward pass that merely runs proves
    little, while one returning ``num_classes`` proves the head was actually replaced.

    ``pretrained=False`` must not touch the network — the suite has to run offline, and with
    several backbones a hub fetch per test is hundreds of megabytes each.
    """
    pytest.importorskip("transformers")
    torch = pytest.importorskip("torch")
    import mb.models.frameworks.pytorch.hf_architectures  # noqa: F401 — register side effects
    from mb.models.frameworks.registry import get_architecture

    factory = get_architecture(FrameworkType.PYTORCH, arch)
    assert factory is not None

    model = factory(num_classes=7, pretrained=False)
    size = preprocessing_spec_for(arch).image_size
    y = model(torch.randn(1, 3, size, size))

    # A bare tensor, not an ImageClassifierOutput: the training loop feeds this to a loss.
    assert torch.is_tensor(y)
    assert y.shape == (1, 7)


@pytest.mark.requires_transformers
@pytest.mark.parametrize("arch", _HF_ARCHITECTURES)
def test_head_parameters_are_a_strict_subset(arch: str) -> None:
    """The frozen phase unfreezes exactly the head, so it must be identifiable and smaller."""
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    from mb.models.frameworks.pytorch.hf_architectures import create_hf_classifier

    model = create_hf_classifier(arch, num_classes=5, pretrained=False)
    head = list(model.head_parameters())
    all_params = list(model.parameters())

    assert head
    assert len(head) < len(all_params)
    assert {id(p) for p in head}.issubset({id(p) for p in all_params})
    assert any(p.shape[0] == 5 for p in head)


@pytest.mark.requires_transformers
def test_problem_type_is_recorded_on_the_config() -> None:
    """A checkpoint loaded elsewhere picks its loss from this, so it must match training."""
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    from mb.models.frameworks.pytorch.hf_architectures import create_hf_classifier

    model = create_hf_classifier(
        ArchitectureType.VIT_BASE_PATCH16_224, num_classes=3, pretrained=False
    )
    assert model.hf.config.problem_type == "single_label_classification"

    model.set_problem_type(LabelMode.MULTI_LABEL)
    assert model.hf.config.problem_type == "multi_label_classification"


def test_unknown_architecture_is_rejected() -> None:
    """Rejected on the id, before transformers is needed — so this runs without it."""
    from mb.models.frameworks.pytorch.hf_architectures import create_hf_classifier

    with pytest.raises(ValueError):
        create_hf_classifier("vit_base_patch16_999", num_classes=2, pretrained=False)
