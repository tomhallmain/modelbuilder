"""Integration coverage for SafeTensors conversion and bundle export."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mb.conversion.converters import convert_model, inline_onnx_external_data
from mb.export.bundle import export_bundle


def _write_tiny_pytorch_checkpoint(path: Path) -> None:
    torch = pytest.importorskip("torch", reason="requires PyTorch")
    model = torch.nn.Sequential(torch.nn.Linear(4, 3))
    state = {"model_state_dict": model.state_dict()}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def _make_tiny_onnx_model(weights):
    """Single-initializer identity-graph ONNX model wrapping *weights*."""
    from onnx import TensorProto, helper, numpy_helper

    init = numpy_helper.from_array(weights, name="W")
    graph = helper.make_graph(
        [helper.make_node("Identity", ["X"], ["Y"])],
        "g",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1])],
        [init],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])


def test_inline_onnx_external_data_embeds_sidecar(tmp_path: Path) -> None:
    onnx = pytest.importorskip("onnx", reason="requires onnx")
    import numpy as np

    weights = np.arange(1000, dtype=np.float32)
    model = _make_tiny_onnx_model(weights)
    path = tmp_path / "split.onnx"
    data_name = "split.onnx.data"
    onnx.save_model(
        model,
        str(path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_name,
        size_threshold=0,
    )
    data_path = tmp_path / data_name
    assert data_path.is_file()

    assert inline_onnx_external_data(path) is True
    assert not data_path.exists()

    embedded = onnx.load(str(path), load_external_data=False)
    assert not any(t.external_data for t in embedded.graph.initializer)
    assert sum(len(t.raw_data) for t in embedded.graph.initializer) == weights.nbytes


def test_inline_onnx_external_data_keeps_external_when_too_large(
    tmp_path: Path, monkeypatch
) -> None:
    """When the embedded size would exceed the protobuf limit, leave the sidecar alone."""
    onnx = pytest.importorskip("onnx", reason="requires onnx")
    import numpy as np
    from mb.conversion import converters as converters_module

    weights = np.arange(1000, dtype=np.float32)
    model = _make_tiny_onnx_model(weights)
    path = tmp_path / "split.onnx"
    data_name = "split.onnx.data"
    onnx.save_model(
        model,
        str(path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_name,
        size_threshold=0,
    )
    data_path = tmp_path / data_name
    assert data_path.is_file()

    # Avoid needing a multi-GB fixture: force the "too large to embed" branch directly.
    monkeypatch.setattr(converters_module, "_ONNX_EMBED_MAX_BYTES", 10)

    assert inline_onnx_external_data(path) is False

    # Nothing should have been touched: sidecar still present, main file still split.
    assert data_path.is_file()
    reloaded = onnx.load(str(path), load_external_data=False)
    assert any(t.external_data for t in reloaded.graph.initializer)


def test_inline_onnx_external_data_noop_when_already_embedded(tmp_path: Path) -> None:
    """A model with no external data at all should be reported OK and left untouched."""
    onnx = pytest.importorskip("onnx", reason="requires onnx")
    import numpy as np

    weights = np.arange(1000, dtype=np.float32)
    model = _make_tiny_onnx_model(weights)
    path = tmp_path / "embedded.onnx"
    onnx.save_model(model, str(path), save_as_external_data=False)
    before_mtime_ns = path.stat().st_mtime_ns

    assert inline_onnx_external_data(path) is True

    # Early-return path: the file must not have been rewritten.
    assert path.stat().st_mtime_ns == before_mtime_ns
    assert list(tmp_path.iterdir()) == [path]
    reloaded = onnx.load(str(path), load_external_data=False)
    assert not any(t.external_data for t in reloaded.graph.initializer)
    assert sum(len(t.raw_data) for t in reloaded.graph.initializer) == weights.nbytes


def test_convert_model_pytorch_to_safetensors(tmp_path: Path) -> None:
    pytest.importorskip("safetensors", reason="requires safetensors")
    input_pth = tmp_path / "tiny.pth"
    out_st = tmp_path / "tiny.safetensors"
    _write_tiny_pytorch_checkpoint(input_pth)

    ok = convert_model(
        input_path=input_pth,
        output_path=out_st,
        source_framework="pytorch",
        target_format="safetensors",
    )
    assert ok is True
    assert out_st.is_file() and out_st.stat().st_size > 0


def test_export_bundle_writes_manifest_and_safetensors(tmp_path: Path) -> None:
    pytest.importorskip("safetensors", reason="requires safetensors")
    input_pth = tmp_path / "tiny.pth"
    _write_tiny_pytorch_checkpoint(input_pth)

    data_dir = tmp_path / "data"
    (data_dir / "train" / "a").mkdir(parents=True, exist_ok=True)
    (data_dir / "train" / "b").mkdir(parents=True, exist_ok=True)
    out_dir = tmp_path / "bundle"

    result = export_bundle(
        input_model=input_pth,
        output_dir=out_dir,
        architecture="resnet18",
        num_classes=2,
        class_names=None,
        data_dir=data_dir,
        image_size=224,
        include_architecture_py=True,
        pipeline_config={},
        snapshot_path=None,
        run_id="rid_test",
    )

    weights_path = Path(result["weights_path"])
    manifest_path = Path(result["manifest_path"])
    assert weights_path.is_file() and weights_path.suffix == ".safetensors"
    assert manifest_path.is_file()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["model"]["weights_format"] == "safetensors"
    assert manifest["model"]["architecture"] == "resnet18"
    assert manifest["model"]["num_classes"] == 2
    assert manifest["model"]["class_names"] == ["a", "b"]
    assert manifest["export"]["run_id"] == "rid_test"
    assert manifest["artifacts"]["weights_file"] == "model.safetensors"
    assert isinstance(manifest["artifacts"]["weights_sha256"], str)
    assert len(manifest["artifacts"]["weights_sha256"]) == 64

