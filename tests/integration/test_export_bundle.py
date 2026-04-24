"""Integration coverage for SafeTensors conversion and bundle export."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mb.conversion.converters import convert_model
from mb.export.bundle import export_bundle


def _write_tiny_pytorch_checkpoint(path: Path) -> None:
    torch = pytest.importorskip("torch", reason="requires PyTorch")
    model = torch.nn.Sequential(torch.nn.Linear(4, 3))
    state = {"model_state_dict": model.state_dict()}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


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

