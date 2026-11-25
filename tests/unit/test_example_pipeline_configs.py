"""
The shipped image-classification example pipelines load, and every value in them takes effect.

The pipeline loader merges unknown keys in without complaint, and the trainer falls back to
a default (with only a warning) for an option that does not apply, so a mistake in an
example would silently train something other than what the file says. These tests make each
key and value accountable, for every ``image_classification_*.example.yaml``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mb.models.preprocessing import is_resolution_locked, preprocessing_spec_for, resolve_image_size
from mb.models.types import ArchitectureType
from mb.pipeline_config import PipelineConfig
from mb.utils.app_paths import resource_path

_CONFIG_DIR = resource_path("mb", "config")
_EXAMPLES = sorted(Path(_CONFIG_DIR).glob("image_classification_*.example.yaml"))
_IDS = [p.name.removesuffix(".example.yaml") for p in _EXAMPLES]

# Shipped examples, by file name: a deleted or renamed one fails here rather than vanishing
# from the parametrized tests below.
_EXPECTED = {
    "image_classification_convnext",
    "image_classification_efficientnet",
    "image_classification_focalnet",
    "image_classification_mobilenet",
    "image_classification_resnet",
    "image_classification_siglip2_base",
    "image_classification_siglip2_so400m",
    "image_classification_vit",
}


def _raw(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _architecture(path: Path) -> ArchitectureType:
    arch = ArchitectureType.try_from(PipelineConfig(config_path=path).get("model.default_architecture"))
    assert arch is not None, path.name
    return arch


def test_the_expected_examples_ship() -> None:
    assert set(_IDS) == _EXPECTED


@pytest.mark.parametrize("path", _EXAMPLES, ids=_IDS)
def test_every_key_is_a_known_pipeline_key(path: Path) -> None:
    defaults = PipelineConfig(config_path=None).to_dict()
    for section, values in _raw(path).items():
        assert section in defaults, section
        for key in values:
            assert key in defaults[section], f"{section}.{key} is not a pipeline key"


@pytest.mark.parametrize("path", _EXAMPLES, ids=_IDS)
def test_values_reach_the_trainer_unchanged(path: Path) -> None:
    raw = _raw(path)
    hp = PipelineConfig(config_path=path).training_hyperparams()
    for key, value in raw.get("training", {}).items():
        assert hp[key] == value, key
    for key in ("image_size", "batch_size"):
        if key in raw.get("data", {}):
            assert hp[key] == raw["data"][key], key


@pytest.mark.parametrize("path", _EXAMPLES, ids=_IDS)
def test_architecture_and_image_size_agree(path: Path) -> None:
    arch = _architecture(path)
    size = PipelineConfig(config_path=path).get("data.image_size")
    if is_resolution_locked(arch):
        assert size == preprocessing_spec_for(arch).image_size
    # Raises if the size does not fit the backbone.
    assert resolve_image_size(arch, size) == size


@pytest.mark.parametrize("path", _EXAMPLES, ids=_IDS)
def test_options_are_accepted_without_fallback(path: Path) -> None:
    """Each option resolves to the value written, not to a default after a warning."""
    pytest.importorskip("torch")
    from mb.models.frameworks.pytorch.optimization import resolve_optimization_settings

    training = _raw(path).get("training", {})
    hp = PipelineConfig(config_path=path).training_hyperparams()
    settings = resolve_optimization_settings(hp)
    for key in ("weight_decay", "layer_decay", "grad_accum_steps"):
        if key in training:
            assert getattr(settings, key) == training[key], key
    for key in ("optimizer", "mixed_precision"):
        if key in training:
            assert getattr(settings, key).value == training[key], key
    for key in ("gradient_checkpointing", "restore_best"):
        if key in training:
            assert getattr(settings, key) is training[key], key
    assert 0 < hp["unfrozen_lr_min"] < hp["unfrozen_lr_max"]


@pytest.mark.parametrize("path", _EXAMPLES, ids=_IDS)
def test_transformer_only_options_are_set_only_on_backbones_they_affect(path: Path) -> None:
    """
    layer_decay applies to plain transformer encoders (SigLIP, ViT) and gradient
    checkpointing to Hugging Face backbones; anywhere else the trainer only warns and
    ignores them, so an example setting them would describe a run that does not happen.
    """
    pytest.importorskip("torch")
    from mb.models.frameworks.pytorch.hf_architectures import HF_BACKBONES

    training = _raw(path).get("training", {})
    arch = _architecture(path).value
    backbone = HF_BACKBONES.get(arch)
    if training.get("layer_decay", 1.0) < 1.0:
        assert backbone is not None and backbone.config_class in ("SiglipConfig", "ViTConfig"), arch
    if training.get("gradient_checkpointing"):
        assert backbone is not None, arch
