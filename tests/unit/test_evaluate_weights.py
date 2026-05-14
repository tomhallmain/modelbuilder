"""Tests for ``mb.evaluate._weights`` (PyTorch checkpoint normalization for evaluation)."""

from __future__ import annotations

import pytest

from mb.evaluate._weights import extract_pytorch_state_dict


def test_extract_pytorch_state_dict_plain_state_dict() -> None:
    pytest.importorskip("torch")
    import torch

    sd = {"layer.weight": torch.zeros(1)}
    assert extract_pytorch_state_dict(sd) is sd


def test_extract_pytorch_state_dict_nested() -> None:
    pytest.importorskip("torch")
    import torch

    inner = {"w": torch.ones(2)}
    blob = {"model_state_dict": inner, "epoch": 3}
    assert extract_pytorch_state_dict(blob) is inner
