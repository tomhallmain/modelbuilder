"""
torchvision architecture families beyond ResNet: registry wiring and one forward/predict each.

Requires optional ``torch`` / ``tensorflow`` (skipped via markers + importorskip).
"""

from __future__ import annotations

import numpy as np
import pytest

from mb.models.types import ArchitectureType, FrameworkType

# Implemented for both frameworks.
_EXTRA = (
    ArchitectureType.MOBILENET_V2,
    ArchitectureType.MOBILENET_V3_LARGE,
    ArchitectureType.MOBILENET_V3_SMALL,
    ArchitectureType.DENSENET121,
    ArchitectureType.DENSENET169,
    ArchitectureType.DENSENET201,
    ArchitectureType.VGG16,
    ArchitectureType.VGG19,
)

# torchvision only — ``keras.applications`` offers counterparts for some of these, but they
# are not registered on the Keras path, and the enum does not require every framework to
# implement every member.
_PYTORCH_ONLY = (
    ArchitectureType.EFFICIENTNET_B4,
    ArchitectureType.CONVNEXT_TINY,
    ArchitectureType.CONVNEXT_SMALL,
    ArchitectureType.CONVNEXT_BASE,
    ArchitectureType.CONVNEXT_LARGE,
)


@pytest.mark.requires_torch
@pytest.mark.parametrize("arch", _EXTRA)
def test_pytorch_registry_factory_forward(arch: ArchitectureType) -> None:
    """Each extra architecture is registered and runs a single forward pass."""
    torch = pytest.importorskip("torch")
    import mb.models.frameworks.pytorch.architectures  # noqa: F401 — register side effects
    from mb.models.frameworks.registry import get_architecture

    factory = get_architecture(FrameworkType.PYTORCH, arch)
    assert factory is not None
    model = factory(num_classes=2, pretrained=False)
    y = model(torch.randn(1, 3, 224, 224))
    assert y.shape == (1, 2)


@pytest.mark.requires_torch
@pytest.mark.parametrize("arch", _PYTORCH_ONLY, ids=lambda a: a.value)
def test_pytorch_only_registry_factory_forward(arch: ArchitectureType) -> None:
    """
    Each PyTorch-only architecture is registered and returns the requested class count.

    The class count is the assertion that matters. Every torchvision family stores its head
    somewhere different — ResNet at ``.fc``, EfficientNet at ``classifier[1]``, MobileNetV3
    at ``classifier[3]``, ConvNeXt at the end of its ``classifier`` — so a head-replacement
    that quietly did nothing would leave 1000 ImageNet outputs and still run.
    """
    torch = pytest.importorskip("torch")
    import mb.models.frameworks.pytorch.architectures  # noqa: F401 — register side effects
    from mb.models.frameworks.registry import get_architecture

    factory = get_architecture(FrameworkType.PYTORCH, arch)
    assert factory is not None
    model = factory(num_classes=7, pretrained=False)
    y = model(torch.randn(1, 3, 224, 224))
    assert y.shape == (1, 7)


def test_every_registered_pytorch_architecture_is_an_enum_member() -> None:
    """
    A registry key that is not an enum value cannot round-trip through TrainingRunArgs JSON,
    so the GUI's subprocess training would reject it. Cheap to check and easy to get wrong
    when registering a new family by string.
    """
    import mb.models.frameworks.pytorch.architectures  # noqa: F401 — register side effects
    from mb.models.frameworks.registry import list_architectures

    registered = list_architectures(FrameworkType.PYTORCH)[FrameworkType.PYTORCH.value]
    unknown = [name for name in registered if ArchitectureType.try_from(name) is None]
    assert not unknown, f"registered but not in ArchitectureType: {unknown}"


@pytest.mark.requires_torch
def test_replace_final_linear_finds_the_last_linear() -> None:
    """
    The helper searches backwards rather than trusting an index.

    A head whose Linear is not where a literal index expects is exactly the case that would
    otherwise leave the model at its pretrained class count without raising.
    """
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    from mb.models.frameworks.pytorch.architectures import _replace_final_linear

    head = nn.Sequential(nn.LayerNorm(8), nn.Flatten(1), nn.Linear(8, 1000))
    _replace_final_linear(head, 3)
    assert head[2].out_features == 3
    assert head(torch.randn(1, 8)).shape == (1, 3)

    with pytest.raises(ValueError):
        _replace_final_linear(nn.Sequential(nn.Flatten(1)), 3)
    with pytest.raises(ValueError):
        _replace_final_linear(nn.Linear(8, 4), 3)


@pytest.mark.requires_tf
@pytest.mark.parametrize("arch", _EXTRA)
def test_keras_registry_factory_predict(arch: ArchitectureType) -> None:
    """Each extra architecture is registered and runs a single predict step."""
    pytest.importorskip("tensorflow")
    import mb.models.frameworks.keras.architectures  # noqa: F401 — register side effects
    from mb.models.frameworks.registry import get_architecture

    factory = get_architecture(FrameworkType.KERAS, arch)
    assert factory is not None
    model = factory(num_classes=2, pretrained=False)
    y = model.predict(np.zeros((1, 224, 224, 3), dtype=np.float32), verbose=0)
    assert y.shape == (1, 2)
