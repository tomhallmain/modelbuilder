"""
ONNX decode metadata and quantized variants.

The metadata is what lets a standalone ``.onnx`` file be decoded without being handed the
label order separately, so most of these check that it survives the round trip — including
across quantization, which does not always carry metadata over on its own.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mb.conversion.converters import build_onnx_metadata
from mb.conversion.quantize import quantized_output_path
from mb.models.types import ArchitectureType, OnnxQuantizationMode


def test_build_metadata_is_all_strings() -> None:
    """ONNX ``metadata_props`` is a string/string map; non-strings would fail at write."""
    meta = build_onnx_metadata(
        architecture=ArchitectureType.RESNET34.value,
        class_names=["a", "b"],
        image_size=320,
    )
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in meta.items())


def test_build_metadata_carries_labels_and_activation() -> None:
    meta = build_onnx_metadata(
        architecture=ArchitectureType.RESNET34.value,
        class_names=["neutral", "gore"],
        image_size=320,
    )
    output = json.loads(meta["mb.output"])

    assert output["labels"] == ["neutral", "gore"]
    assert output["activation"] == "softmax"
    assert output["label_mode"] == "single_label"


def test_build_metadata_preprocessing_follows_the_architecture() -> None:
    """The graph records the normalization its own weights were trained against."""
    resnet = json.loads(
        build_onnx_metadata(architecture=ArchitectureType.RESNET34.value, image_size=320)[
            "mb.preprocessing"
        ]
    )
    siglip = json.loads(
        build_onnx_metadata(
            architecture=ArchitectureType.SIGLIP2_BASE_PATCH16_384.value, image_size=384
        )["mb.preprocessing"]
    )

    assert resnet["normalize_mean"] == [0.485, 0.456, 0.406]
    assert resnet["image_size"] == 320
    assert siglip["normalize_mean"] == [0.5, 0.5, 0.5]
    assert siglip["image_size"] == 384


def test_build_metadata_without_class_names_is_still_valid() -> None:
    meta = build_onnx_metadata(architecture=ArchitectureType.RESNET34.value)
    output = json.loads(meta["mb.output"])
    assert output["labels"] is None
    assert output["activation"] == "softmax"


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (OnnxQuantizationMode.FP16, "model.fp16.onnx"),
        (OnnxQuantizationMode.INT8, "model.int8.onnx"),
        (OnnxQuantizationMode.UINT8, "model.uint8.onnx"),
    ],
)
def test_quantized_paths_are_siblings(mode: OnnxQuantizationMode, expected: str) -> None:
    """Variants sit next to the source so several can ship together."""
    assert quantized_output_path(Path("/out/model.onnx"), mode).name == expected


def test_quantization_mode_parsing() -> None:
    assert OnnxQuantizationMode.try_from("fp16") == OnnxQuantizationMode.FP16
    assert OnnxQuantizationMode.try_from("INT8") == OnnxQuantizationMode.INT8
    assert OnnxQuantizationMode.try_from(OnnxQuantizationMode.UINT8) == OnnxQuantizationMode.UINT8
    assert OnnxQuantizationMode.try_from("bnb4") is None
    assert OnnxQuantizationMode.try_from(None) is None


def test_quantize_missing_source_returns_none(tmp_path: Path) -> None:
    from mb.conversion.quantize import quantize_onnx_model

    assert quantize_onnx_model(tmp_path / "nope.onnx", OnnxQuantizationMode.INT8) is None


def test_quantize_unknown_mode_returns_none(tmp_path: Path) -> None:
    from mb.conversion.quantize import quantize_onnx_model

    src = tmp_path / "model.onnx"
    src.write_bytes(b"")
    assert quantize_onnx_model(src, "q4f16") is None


@pytest.mark.requires_onnx
def test_metadata_round_trips_through_an_onnx_file(tmp_path: Path) -> None:
    """Write, reload, and read back — the path a JS consumer actually takes."""
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    from mb.conversion.converters import write_onnx_metadata

    # Smallest valid graph: one identity op.
    tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3])
    graph = helper.make_graph(
        [helper.make_node("Identity", ["input"], ["output"])], "g", [tensor], [out]
    )
    model_path = tmp_path / "model.onnx"
    onnx.save_model(helper.make_model(graph), str(model_path))

    meta = build_onnx_metadata(
        architecture=ArchitectureType.RESNET34.value, class_names=["a", "b"], image_size=320
    )
    assert write_onnx_metadata(model_path, meta) is True

    reloaded = {p.key: p.value for p in onnx.load(str(model_path)).metadata_props}
    assert json.loads(reloaded["mb.output"])["labels"] == ["a", "b"]
    assert json.loads(reloaded["mb.preprocessing"])["image_size"] == 320

    # Writing again replaces rather than duplicating keys.
    assert write_onnx_metadata(model_path, {"mb.architecture": "resnet50"}) is True
    props = onnx.load(str(model_path)).metadata_props
    assert len([p for p in props if p.key == "mb.architecture"]) == 1
    assert {p.key: p.value for p in props}["mb.architecture"] == "resnet50"


def test_write_metadata_with_empty_dict_is_a_noop(tmp_path: Path) -> None:
    from mb.conversion.converters import write_onnx_metadata

    assert write_onnx_metadata(tmp_path / "missing.onnx", {}) is False
