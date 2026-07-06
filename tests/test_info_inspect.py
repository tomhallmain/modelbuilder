"""Unit tests for :mod:`mb.info_inspect` (shared with CLI and Info page)."""

from __future__ import annotations

from pathlib import Path

import pytest

from mb.info_inspect import dataset_info_text, model_info_text
from mb.utils.translations import _

_NO_WEIGHTS_WARNING = _(
    "Warning: no weight tensors found in this ONNX file (graph-only or incomplete export)."
)


def test_model_info_text_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "nope.onnx"
    with pytest.raises(FileNotFoundError):
        model_info_text(missing)


def test_model_info_text_rejects_directory(tmp_path: Path) -> None:
    d = tmp_path / "dir"
    d.mkdir()
    with pytest.raises(ValueError) as ei:
        model_info_text(d)
    # Message is translated; assert path and non-file semantics without English-only regex.
    msg = str(ei.value)
    assert str(d.resolve()) in msg or d.name in msg


def test_model_info_text_onnx_minimal(tmp_path: Path) -> None:
    onnx = tmp_path / "m.onnx"
    onnx.write_bytes(b"")
    text = model_info_text(onnx)
    assert str(onnx.resolve()) in text or onnx.name in text
    assert "onnx" in text.lower()


def test_model_info_text_onnx_reports_initializers_and_weight_bytes(tmp_path: Path) -> None:
    onnx = pytest.importorskip("onnx")
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    weights = np.arange(100, dtype=np.float32).reshape(10, 10)
    init = numpy_helper.from_array(weights, name="W")
    graph = helper.make_graph(
        [helper.make_node("Identity", ["X"], ["Y"])],
        "g",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1])],
        [init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    path = tmp_path / "with_weights.onnx"
    onnx.save(model, str(path))

    text = model_info_text(path)
    assert "1" in text  # initializer count
    assert f"{weights.nbytes:,}" in text or str(weights.nbytes) in text
    assert _NO_WEIGHTS_WARNING.lower() not in text.lower()


def test_model_info_text_onnx_graph_only_warns(tmp_path: Path) -> None:
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        [helper.make_node("Identity", ["X"], ["Y"])],
        "g",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1])],
        [],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    path = tmp_path / "graph_only.onnx"
    onnx.save(model, str(path))

    text = model_info_text(path)
    assert "0" in text
    assert _NO_WEIGHTS_WARNING.lower() in text.lower()


def test_model_info_text_pytorch_state_dict(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    state = {"layer.weight": torch.zeros(4, 4), "layer.bias": torch.zeros(4)}
    path = tmp_path / "m.pth"
    torch.save(state, path)

    text = model_info_text(path)
    assert "2" in text
    # 4*4 + 4 float32 params
    assert "20" in text
    assert f"{20 * 4:,}" in text or str(20 * 4) in text


def test_dataset_info_text_two_class_layout(two_class_classification_data_dir: Path) -> None:
    text = dataset_info_text(two_class_classification_data_dir)
    assert "train" in text.lower()
    assert "class_a" in text or "class_b" in text


def test_dataset_info_text_missing_dir(tmp_path: Path) -> None:
    missing = tmp_path / "nodata"
    with pytest.raises(FileNotFoundError):
        dataset_info_text(missing)
