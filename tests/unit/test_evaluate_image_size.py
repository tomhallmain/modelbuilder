"""
Input size for ``mb evaluate`` when ``--image-size`` is not given, and per-model sizes in compare.

The fixed default is 224, which a resolution-locked backbone such as SigLIP2 at 512 cannot
take. In compare, each model must also be scored on its own preprocessing contract rather
than on model A's.
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from mb.evaluate.compare import build_compare_request
from mb.evaluate.metrics import build_metrics_request, evaluation_image_size
from mb.models.types import FrameworkType, ModelType

_SO400M_512 = "siglip2_so400m_patch16_512"


def test_explicit_size_is_used_as_given() -> None:
    assert evaluation_image_size(320, "resnet18") == 320
    assert evaluation_image_size(384, _SO400M_512) == 384


def test_unrequested_size_is_native_for_locked_and_224_elsewhere() -> None:
    assert evaluation_image_size(None, _SO400M_512) == 512
    assert evaluation_image_size(None, "resnet18") == 224
    assert evaluation_image_size(None, None) == 224


def _metrics_ns(tmp_path: Path, **overrides) -> Namespace:
    values = dict(
        model=tmp_path / "m.pth",
        data_dir=tmp_path,
        model_type=ModelType.IMAGE_CLASSIFICATION.value,
        framework=FrameworkType.PYTORCH.value,
        architecture="resnet18",
        num_classes=None,
        image_size=None,
        batch_size=2,
        num_workers=0,
        device=None,
    )
    values.update(overrides)
    return Namespace(**values)


def test_metrics_request_sizes(tmp_path: Path) -> None:
    assert build_metrics_request(_metrics_ns(tmp_path)).image_size == 224
    assert build_metrics_request(_metrics_ns(tmp_path, architecture=_SO400M_512)).image_size == 512
    assert build_metrics_request(_metrics_ns(tmp_path, image_size=300)).image_size == 300


def _compare_ns(tmp_path: Path, **overrides) -> Namespace:
    values = dict(
        model_a=tmp_path / "a.pth",
        model_b=tmp_path / "b.pth",
        data_dir=tmp_path,
        model_type=ModelType.IMAGE_CLASSIFICATION.value,
        framework=FrameworkType.PYTORCH.value,
        framework_b=None,
        architecture="resnet18",
        architecture_b=None,
        num_classes=None,
        image_size=None,
        batch_size=2,
        num_workers=0,
        device=None,
        max_disagreement_report=None,
    )
    values.update(overrides)
    return Namespace(**values)


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({}, (224, 224)),
        ({"image_size": 320}, (320, 320)),
        ({"architecture_b": _SO400M_512}, (224, 512)),
        ({"architecture_b": _SO400M_512, "image_size": 320}, (320, 512)),
        ({"architecture": _SO400M_512}, (512, 512)),
        ({"architecture_b": "convnext_large"}, (224, 224)),
        ({"image_size_b": 96}, (224, 96)),
    ],
)
def test_compare_request_sizes(tmp_path: Path, overrides, expected) -> None:
    req = build_compare_request(_compare_ns(tmp_path, **overrides))
    assert (req.image_size, req.image_size_b) == expected


@pytest.mark.requires_torch
@pytest.mark.parametrize("size_b, expected_builds", [(64, [64]), (96, [64, 96])])
def test_compare_builds_a_separate_view_only_when_preprocessing_differs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, size_b: int, expected_builds: list[int]
) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from PIL import Image

    import mb.evaluate.classification.image_compare as image_compare
    from mb.evaluate.compare import run_compare
    from mb.models.frameworks.pytorch.trainer import PyTorchTrainer

    data = tmp_path / "data"
    for cls in ("a", "b"):
        (data / cls).mkdir(parents=True)
        for i in range(2):
            Image.new("RGB", (40, 30), (i * 90, 60, 120)).save(data / cls / f"{i}.png")

    model = PyTorchTrainer(device="cpu").create_model("resnet18", 2, pretrained=False)
    for name in ("a.pth", "b.pth"):
        torch.save(model.state_dict(), tmp_path / name)

    built: list[int] = []
    original = image_compare._build_imagefolder_torch

    def _recording(data_dir, image_size, *args, **kwargs):
        built.append(int(image_size))
        return original(data_dir, image_size, *args, **kwargs)

    monkeypatch.setattr(image_compare, "_build_imagefolder_torch", _recording)

    req = build_compare_request(
        _compare_ns(tmp_path, data_dir=data, image_size=64, image_size_b=size_b)
    )
    report = run_compare(req)
    assert built == expected_builds
    assert report.n_samples == 4
