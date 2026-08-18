"""
Export manifest: preprocessing and output blocks.

The manifest's declared preprocessing must match what training actually applied. The two
are now derived from one spec, and this is the test that keeps them that way — a mismatch
here degrades inference without raising anything.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from mb.export.bundle import _architecture_stub_text
from mb.models.preprocessing import IMAGENET_MEAN, IMAGENET_STD, preprocessing_spec_for
from mb.models.types import ArchitectureType


def _rendered_stub(architecture: str = "resnet34", num_classes: int = 2) -> str:
    """The ``model_architecture.py`` text a bundle export would emit."""
    return _architecture_stub_text(
        architecture=architecture, num_classes=num_classes, framework="pytorch"
    )


def _stub_function(name: str, source: str):
    """
    Pull one function out of the generated stub and make it callable.

    The stub as a whole imports torch and torchvision at module scope, so it cannot simply
    be exec'd here. Extracting a single self-contained function keeps the test dependency
    free while still exercising the code that actually ships in the bundle.
    """
    tree = ast.parse(source)
    node = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    namespace: dict = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<stub>", "exec"), namespace)
    return namespace[name]


@pytest.mark.requires_torch
def test_manifest_preprocessing_matches_training_transforms(tmp_path: Path) -> None:
    """
    What the manifest promises is what ``get_val_transforms`` actually does.

    Read the Normalize step straight out of the composed transform rather than re-deriving
    it, so this fails if either side drifts.
    """
    pytest.importorskip("torch")
    from torchvision import transforms

    from mb.models.frameworks.pytorch.data_loader import get_val_transforms

    for arch in (
        ArchitectureType.RESNET34,
        ArchitectureType.SIGLIP2_BASE_PATCH16_384,
    ):
        spec = preprocessing_spec_for(arch)
        composed = get_val_transforms(spec.image_size, preprocessing=spec)
        normalize = next(t for t in composed.transforms if isinstance(t, transforms.Normalize))

        assert tuple(normalize.mean) == tuple(spec.normalize_mean)
        assert tuple(normalize.std) == tuple(spec.normalize_std)

        manifest_block = spec.to_manifest_dict()
        assert manifest_block["normalize_mean"] == list(normalize.mean)
        assert manifest_block["normalize_std"] == list(normalize.std)


def test_siglip_and_resnet_manifests_differ() -> None:
    """Guards against a regression that silently reinstates ImageNet stats everywhere."""
    resnet = preprocessing_spec_for(ArchitectureType.RESNET34).to_manifest_dict()
    siglip = preprocessing_spec_for(ArchitectureType.SIGLIP2_BASE_PATCH16_384).to_manifest_dict()

    assert resnet["normalize_mean"] == list(IMAGENET_MEAN)
    assert resnet["normalize_std"] == list(IMAGENET_STD)
    assert siglip["normalize_mean"] == [0.5, 0.5, 0.5]
    assert siglip["image_size"] == 384
    assert resnet != siglip


def test_manifest_blocks_are_json_serializable() -> None:
    from mb.models.output_contract import single_label_contract

    payload = {
        "preprocessing": preprocessing_spec_for(ArchitectureType.RESNET34, 320).to_manifest_dict(),
        "output": single_label_contract(["a", "b"]).to_manifest_dict(),
    }
    assert json.loads(json.dumps(payload)) == payload


def test_generated_stub_is_valid_python() -> None:
    """
    The stub is built by string interpolation, so a brace-escaping slip is a live hazard.

    It ships as a standalone file that only runs at bundle-load time, which is far too late
    to discover a syntax error.
    """
    for architecture in ("resnet34", "siglip2_base_patch16_384"):
        ast.parse(_rendered_stub(architecture=architecture, num_classes=11))


def test_stub_layout_inference_detects_transformers_wrapper() -> None:
    """
    A Hugging Face checkpoint's keys are prefixed by the wrapper's submodule attribute.

    Without this branch the stub tries to rebuild the model through torchvision and fails
    at load time with an unhelpful error.
    """
    infer_layout = _stub_function("infer_layout_from_state_dict", _rendered_stub())
    hf_keys = {
        "hf.vision_model.embeddings.patch_embedding.weight": None,
        "hf.classifier.weight": None,
        "hf.classifier.bias": None,
    }
    assert infer_layout(hf_keys) == "transformers"


def test_stub_layout_inference_still_detects_torchvision_and_fastai() -> None:
    infer_layout = _stub_function("infer_layout_from_state_dict", _rendered_stub())
    torchvision_keys = {"conv1.weight": None, "layer1.0.conv1.weight": None, "fc.weight": None}
    fastai_keys = {"0.0.weight": None, "1.2.weight": None}

    assert infer_layout(torchvision_keys) == "torchvision"
    assert infer_layout(fastai_keys) == "fastai_sequential"
    assert infer_layout({}) == "torchvision"


def test_stub_can_rebuild_every_hugging_face_backbone() -> None:
    """
    The stub rebuilds the model offline, so it carries each backbone's description literally.

    Checked against the live table rather than hardcoded values, so adding a backbone that
    the stub cannot rebuild fails here rather than at a consumer's bundle-load time. A wrong
    size means a patch grid that does not match the saved weights; a wrong config class means
    the wrong architecture entirely.
    """
    from mb.models.frameworks.pytorch.hf_architectures import HF_BACKBONES

    tree = ast.parse(_rendered_stub(architecture="siglip2_base_patch16_384", num_classes=3))
    assign = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(getattr(t, "id", None) == "HF_BACKBONES" for t in node.targets)
    )
    stub_table = ast.literal_eval(assign.value)

    # Every trainable Hugging Face backbone is rebuildable from the stub, at the same input
    # size its preprocessing spec declares.
    assert set(stub_table) == set(HF_BACKBONES)
    for name, backbone in HF_BACKBONES.items():
        entry = stub_table[name]
        assert entry["image_size"] == preprocessing_spec_for(name).image_size
        assert entry["config_class"] == backbone.config_class
        assert entry["nested_vision_config"] == backbone.nested_vision_config
        assert entry["patch_size"] == backbone.patch_size
