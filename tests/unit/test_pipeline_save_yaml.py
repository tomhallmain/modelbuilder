"""Tests for :func:`mb.pipeline_config.save_pipeline_yaml` merge behavior."""

from __future__ import annotations

from pathlib import Path

import yaml

from mb.pipeline_config import save_pipeline_yaml


def test_save_pipeline_yaml_creates_subset(tmp_path: Path) -> None:
    p = tmp_path / "pipeline.yaml"
    save_pipeline_yaml(
        p,
        {
            "model": {"default_type": "image_classification"},
            "data": {"raw_data_dir": "raw_data"},
            "training": {"frozen_epochs": 1},
            "paths": {"logs_dir": "logs"},
        },
    )
    with open(p, encoding="utf-8") as f:
        d = yaml.safe_load(f)
    assert set(d.keys()) == {"model", "data", "training", "paths"}


def test_save_pipeline_yaml_merges_into_existing(tmp_path: Path) -> None:
    p = tmp_path / "combined.yaml"
    p.write_text(
        yaml.dump({"gui": {"locale": "en"}, "model": {"default_type": "x"}}, default_flow_style=False),
        encoding="utf-8",
    )
    save_pipeline_yaml(
        p,
        {
            "model": {"default_type": "image_classification"},
            "data": {"raw_data_dir": "r"},
            "training": {"frozen_epochs": 2},
            "paths": {"logs_dir": "l"},
        },
    )
    with open(p, encoding="utf-8") as f:
        d = yaml.safe_load(f)
    assert d.get("gui", {}).get("locale") == "en"
    assert d["model"]["default_type"] == "image_classification"
