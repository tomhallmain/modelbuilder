"""
MobileNet, DenseNet, VGG: registry wiring and one forward/predict per architecture.

Requires optional ``torch`` / ``tensorflow`` (skipped via markers + importorskip).
"""

from __future__ import annotations

import numpy as np
import pytest

from mb.models.types import ArchitectureType, FrameworkType

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
