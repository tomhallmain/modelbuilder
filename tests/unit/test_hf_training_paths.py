"""
Training and export paths for Hugging Face backbones that otherwise only meet synthetic
models in the tests: layer-wise decay on real parameter names, gradient checkpointing on
SigLIP, and a non-base-sized SigLIP surviving the export bundle and ONNX.

The non-base-sized cases use a tiny SigLIP registered just for the test (2 layers, width
32), which takes the same code paths as SO400M at a size that runs in a fraction of a second.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytestmark = [pytest.mark.requires_torch, pytest.mark.requires_transformers]

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from mb.models.frameworks.pytorch.hf_architectures import (  # noqa: E402
    HF_BACKBONES,
    HFBackbone,
    create_hf_classifier,
)
from mb.models.frameworks.pytorch.optimization import (  # noqa: E402
    OptimizationSettings,
    build_optimizer,
)

_TINY = "tiny_siglip_for_tests"
_TINY_SIZE = 32
_TINY_DIMS = (
    ("hidden_size", 32),
    ("intermediate_size", 64),
    ("num_hidden_layers", 2),
    ("num_attention_heads", 2),
)


@pytest.fixture
def tiny_siglip(monkeypatch: pytest.MonkeyPatch) -> str:
    """Register a small, non-base-sized SigLIP backbone and its preprocessing spec."""
    import mb.models.preprocessing as preprocessing

    monkeypatch.setitem(
        HF_BACKBONES,
        _TINY,
        HFBackbone("not/downloaded", "SiglipConfig", nested_vision_config=True, vision_dims=_TINY_DIMS),
    )
    monkeypatch.setitem(
        preprocessing._SPECS, _TINY, preprocessing._fixed_grid_half_norm_spec(_TINY_SIZE)
    )
    return _TINY


# --- Layer-wise decay on real parameter names ---


@pytest.mark.parametrize(
    "arch, n_blocks",
    [("siglip2_base_patch16_224", 12), ("vit_base_patch16_224", 12)],
)
def test_layer_decay_finds_every_encoder_block(arch: str, n_blocks: int) -> None:
    """
    SigLIP names blocks ``encoder.layers.N`` and ViT ``encoder.layer.N``; both must resolve
    to embeddings + N blocks + top, with the head at the full rate.
    """
    model = create_hf_classifier(arch, num_classes=3, pretrained=False)
    optimizer, lrs = build_optimizer(
        model, 1.0, OptimizationSettings(layer_decay=0.5), apply_layer_decay=True
    )
    assert len(lrs) == n_blocks + 2
    lr_of = {id(p): g["lr"] for g in optimizer.param_groups for p in g["params"]}
    assert sorted(lr_of) == sorted(id(p) for p in model.parameters())
    for p in model.head_parameters():
        assert lr_of[id(p)] == pytest.approx(1.0)
    embeddings = [p for n, p in model.named_parameters() if "embeddings" in n]
    assert embeddings
    for p in embeddings:
        assert lr_of[id(p)] == pytest.approx(0.5 ** (n_blocks + 1))


def test_layer_decay_on_the_so400m_layout(tiny_siglip: str) -> None:
    """A non-base depth comes from vision_dims, and the grouping follows it."""
    model = create_hf_classifier(tiny_siglip, num_classes=3, pretrained=False)
    _optimizer, lrs = build_optimizer(
        model, 1.0, OptimizationSettings(layer_decay=0.5), apply_layer_decay=True
    )
    assert len(lrs) == dict(_TINY_DIMS)["num_hidden_layers"] + 2


# --- Gradient checkpointing on SigLIP ---


def test_gradient_checkpointing_trains_siglip(tiny_siglip: str) -> None:
    model = create_hf_classifier(tiny_siglip, num_classes=3, pretrained=False)
    assert model.enable_gradient_checkpointing() is True
    assert model.hf.is_gradient_checkpointing
    model.train()
    x = torch.randn(2, 3, _TINY_SIZE, _TINY_SIZE)
    loss = torch.nn.functional.cross_entropy(model(x), torch.tensor([0, 2]))
    loss.backward()
    block = next(p for n, p in model.named_parameters() if ".encoder.layers.0." in n)
    assert block.grad is not None and torch.count_nonzero(block.grad) > 0


def test_gradient_checkpointing_matches_plain_gradients(tiny_siglip: str) -> None:
    """Recomputing activations must not change the gradient."""
    torch.manual_seed(0)
    plain = create_hf_classifier(tiny_siglip, num_classes=3, pretrained=False)
    ckpt = create_hf_classifier(tiny_siglip, num_classes=3, pretrained=False)
    ckpt.load_state_dict(plain.state_dict())
    ckpt.enable_gradient_checkpointing()
    x = torch.randn(2, 3, _TINY_SIZE, _TINY_SIZE)
    y = torch.tensor([1, 0])
    for m in (plain, ckpt):
        m.train()
        torch.nn.functional.cross_entropy(m(x), y).backward()
    for a, b in zip(plain.parameters(), ckpt.parameters()):
        if a.grad is None:
            assert b.grad is None
        else:
            assert torch.allclose(a.grad, b.grad, atol=1e-6)


# --- Export bundle round trip for a non-base-sized SigLIP ---


def _load_stub_module(path: Path):
    spec = importlib.util.spec_from_file_location(f"stub_{path.parent.name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_bundle_stub_rebuilds_a_non_base_sized_siglip(tiny_siglip: str, tmp_path: Path) -> None:
    """
    The generated model_architecture.py must rebuild the trained size, load the weights with
    an exact key and shape match, and reproduce the trained model's outputs.
    """
    pytest.importorskip("safetensors")
    pytest.importorskip("torchvision")
    import json

    from mb.export.bundle import export_bundle

    torch.manual_seed(0)
    model = create_hf_classifier(tiny_siglip, num_classes=3, pretrained=False).eval()
    pth = tmp_path / "model.pth"
    torch.save(model.state_dict(), pth)

    result = export_bundle(
        input_model=pth,
        output_dir=tmp_path / "bundle",
        architecture=tiny_siglip,
        num_classes=3,
        class_names=["a", "b", "c"],
        data_dir=None,
        image_size=None,
        include_architecture_py=True,
        pipeline_config={"data": {"image_size": 224}},
        snapshot_path=None,
        run_id="rid",
    )
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["preprocessing"]["image_size"] == _TINY_SIZE

    stub = _load_stub_module(Path(result["architecture_path"]))
    rebuilt, missing, unexpected = stub.load_bundle_model(
        result["weights_path"], architecture=tiny_siglip, num_classes=3, device="cpu", strict=True
    )
    assert not missing and not unexpected

    x = torch.randn(2, 3, _TINY_SIZE, _TINY_SIZE)
    with torch.no_grad():
        assert torch.allclose(model(x), rebuilt(x), atol=1e-5)


# --- ONNX export at a non-224 size ---


@pytest.mark.requires_onnx
def test_onnx_export_of_a_non_base_sized_siglip_matches_pytorch(
    tiny_siglip: str, tmp_path: Path
) -> None:
    """
    The exported graph takes the backbone's own input size and gives the same logits.

    Graph size (SO400M at fp32 is about 1.6 GB, under protobuf's 2 GB single-file limit) is
    not something a unit test can establish; this covers the export path itself.
    """
    onnxruntime = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    import numpy as np

    from mb.conversion.converters import export_torch_module_to_onnx

    torch.manual_seed(0)
    model = create_hf_classifier(tiny_siglip, num_classes=3, pretrained=False).eval()
    out = tmp_path / "tiny.onnx"
    export_torch_module_to_onnx(model, out, image_size=_TINY_SIZE)

    x = torch.randn(2, 3, _TINY_SIZE, _TINY_SIZE)
    session = onnxruntime.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    (onnx_logits,) = session.run(None, {session.get_inputs()[0].name: x.numpy()})
    with torch.no_grad():
        expected = model(x).numpy()
    assert onnx_logits.shape == (2, 3)
    assert np.allclose(onnx_logits, expected, atol=1e-4)
