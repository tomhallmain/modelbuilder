"""
Tests for LoRA-adapter detection and the ``mb convert``/``mb export bundle`` rejections
built on it — neither command has a real conversion/export step to offer a LoRA adapter
(it's already the final, portable artifact from training), so pointing them at one should
be a clean, immediate rejection rather than a generic message or a raw traceback from
deeper file-loading code.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mb.cli import main
from mb.models.generation_architectures import looks_like_lora_adapter
from mb.utils.constants import ModelBuilderTaskType


def test_looks_like_lora_adapter_directory_with_config(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    assert looks_like_lora_adapter(adapter_dir) is True


def test_looks_like_lora_adapter_directory_without_config(tmp_path: Path) -> None:
    plain_dir = tmp_path / "not_an_adapter"
    plain_dir.mkdir()
    (plain_dir / "some_other_file.json").write_text("{}", encoding="utf-8")
    assert looks_like_lora_adapter(plain_dir) is False


def test_looks_like_lora_adapter_non_safetensors_file(tmp_path: Path) -> None:
    f = tmp_path / "model.pth"
    f.write_bytes(b"")
    assert looks_like_lora_adapter(f) is False


def test_looks_like_lora_adapter_missing_path(tmp_path: Path) -> None:
    assert looks_like_lora_adapter(tmp_path / "does_not_exist") is False


def test_looks_like_lora_adapter_safetensors_with_lora_keys(tmp_path: Path) -> None:
    pytest.importorskip("safetensors")
    import numpy as np
    import safetensors.numpy

    path = tmp_path / "adapter_model.safetensors"
    safetensors.numpy.save_file(
        {
            "base_model.model.transformer_blocks.0.attn.to_q.lora_A.weight": np.zeros(
                (4, 4), dtype=np.float32
            ),
            "base_model.model.transformer_blocks.0.attn.to_q.lora_B.weight": np.zeros(
                (4, 4), dtype=np.float32
            ),
        },
        str(path),
    )
    assert looks_like_lora_adapter(path) is True


def test_looks_like_lora_adapter_safetensors_without_lora_keys(tmp_path: Path) -> None:
    pytest.importorskip("safetensors")
    import numpy as np
    import safetensors.numpy

    path = tmp_path / "not_lora.safetensors"
    safetensors.numpy.save_file({"weight": np.zeros((4, 4), dtype=np.float32)}, str(path))
    assert looks_like_lora_adapter(path) is False


def test_cli_convert_rejects_lora_adapter_directory(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    assert (
        main(
            [
                str(ModelBuilderTaskType.CONVERT.value),
                "--input",
                str(adapter_dir),
                "--output",
                str(tmp_path / "out.safetensors"),
                "--target",
                "safetensors",
            ]
        )
        == 1
    )


def test_cli_export_bundle_rejects_lora_adapter_directory(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    assert (
        main(
            [
                str(ModelBuilderTaskType.EXPORT.value),
                "bundle",
                "--input",
                str(adapter_dir),
                "--output-dir",
                str(tmp_path / "bundle_out"),
            ]
        )
        == 1
    )
