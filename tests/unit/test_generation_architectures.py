"""Tests for ``mb.models.generation_architectures``."""

from __future__ import annotations

from pathlib import Path

from mb.models.generation_architectures import (
    BaseGenerationArchitecture,
    detect_base_architecture,
    get_descriptor,
)


def _write_index(base_model_dir: Path, index: dict) -> None:
    import json

    base_model_dir.mkdir(parents=True, exist_ok=True)
    (base_model_dir / "model_index.json").write_text(json.dumps(index), encoding="utf-8")


def test_try_from_accepts_known_values() -> None:
    assert (
        BaseGenerationArchitecture.try_from("stable_diffusion_1x")
        == BaseGenerationArchitecture.STABLE_DIFFUSION_1X
    )
    assert (
        BaseGenerationArchitecture.try_from("STABLE_DIFFUSION_XL")
        == BaseGenerationArchitecture.STABLE_DIFFUSION_XL
    )
    assert BaseGenerationArchitecture.try_from("flux") == BaseGenerationArchitecture.FLUX
    assert BaseGenerationArchitecture.try_from("chroma") == BaseGenerationArchitecture.CHROMA
    assert (
        BaseGenerationArchitecture.try_from("z_image_turbo")
        == BaseGenerationArchitecture.Z_IMAGE_TURBO
    )


def test_try_from_rejects_unknown_or_empty() -> None:
    assert BaseGenerationArchitecture.try_from(None) is None
    assert BaseGenerationArchitecture.try_from("") is None
    assert BaseGenerationArchitecture.try_from("not_a_real_architecture") is None


def test_get_descriptor_stable_diffusion_1x_is_implemented() -> None:
    d = get_descriptor(BaseGenerationArchitecture.STABLE_DIFFUSION_1X)
    assert d.implemented is True
    assert d.text_encoder_subfolders == ("text_encoder",)
    assert d.text_encoder_types == ("clip",)
    assert d.unet_subfolder == "unet"
    assert d.transformer_subfolder is None
    assert d.lora_target_modules


def test_get_descriptor_flux_is_implemented() -> None:
    """Flux is the recommended default for new LoRAs — see module docstring."""
    d = get_descriptor(BaseGenerationArchitecture.FLUX)
    assert d.implemented is True
    assert d.text_encoder_subfolders == ("text_encoder", "text_encoder_2")
    assert d.text_encoder_types == ("clip", "t5")
    assert d.unet_subfolder is None
    assert d.transformer_subfolder == "transformer"
    assert d.lora_target_modules


def test_get_descriptor_chroma_is_implemented() -> None:
    """Chroma reuses the Flux training path (see chroma.py's Flux-compatibility assumption)."""
    d = get_descriptor(BaseGenerationArchitecture.CHROMA)
    assert d.implemented is True
    assert d.text_encoder_subfolders == ("text_encoder", "text_encoder_2")
    assert d.text_encoder_types == ("clip", "t5")
    assert d.unet_subfolder is None
    assert d.transformer_subfolder == "transformer"
    assert d.lora_target_modules


def test_get_descriptor_others_reserved_not_implemented() -> None:
    for arch in (
        BaseGenerationArchitecture.STABLE_DIFFUSION_XL,
        BaseGenerationArchitecture.STABLE_DIFFUSION_3,
        BaseGenerationArchitecture.Z_IMAGE_TURBO,
    ):
        assert get_descriptor(arch).implemented is False


def test_detect_missing_model_index_returns_none(tmp_path: Path) -> None:
    assert detect_base_architecture(str(tmp_path)) is None


def test_detect_unreadable_json_returns_none(tmp_path: Path) -> None:
    (tmp_path / "model_index.json").write_text("not valid json{{{", encoding="utf-8")
    assert detect_base_architecture(str(tmp_path)) is None


def test_detect_sd1x_single_text_encoder_unet(tmp_path: Path) -> None:
    _write_index(
        tmp_path,
        {
            "_class_name": "StableDiffusionPipeline",
            "unet": ["diffusers", "UNet2DConditionModel"],
            "text_encoder": ["transformers", "CLIPTextModel"],
            "vae": ["diffusers", "AutoencoderKL"],
        },
    )
    assert detect_base_architecture(str(tmp_path)) == BaseGenerationArchitecture.STABLE_DIFFUSION_1X


def test_detect_sdxl_dual_text_encoder_unet(tmp_path: Path) -> None:
    _write_index(
        tmp_path,
        {
            "_class_name": "StableDiffusionXLPipeline",
            "unet": ["diffusers", "UNet2DConditionModel"],
            "text_encoder": ["transformers", "CLIPTextModel"],
            "text_encoder_2": ["transformers", "CLIPTextModelWithProjection"],
        },
    )
    assert detect_base_architecture(str(tmp_path)) == BaseGenerationArchitecture.STABLE_DIFFUSION_XL


def test_detect_sd3_transformer_no_unet(tmp_path: Path) -> None:
    _write_index(
        tmp_path,
        {
            "_class_name": "StableDiffusion3Pipeline",
            "transformer": ["diffusers", "SD3Transformer2DModel"],
            "text_encoder": ["transformers", "CLIPTextModel"],
        },
    )
    assert detect_base_architecture(str(tmp_path)) == BaseGenerationArchitecture.STABLE_DIFFUSION_3


def test_detect_flux_transformer_no_unet(tmp_path: Path) -> None:
    _write_index(
        tmp_path,
        {
            "_class_name": "FluxPipeline",
            "transformer": ["diffusers", "FluxTransformer2DModel"],
            "text_encoder": ["transformers", "CLIPTextModel"],
        },
    )
    assert detect_base_architecture(str(tmp_path)) == BaseGenerationArchitecture.FLUX


def test_detect_chroma_transformer_no_unet(tmp_path: Path) -> None:
    _write_index(
        tmp_path,
        {
            "_class_name": "ChromaPipeline",
            "transformer": ["diffusers", "ChromaTransformer2DModel"],
            "text_encoder": ["transformers", "CLIPTextModel"],
        },
    )
    assert detect_base_architecture(str(tmp_path)) == BaseGenerationArchitecture.CHROMA


def test_detect_unrecognized_shape_returns_none(tmp_path: Path) -> None:
    _write_index(tmp_path, {"_class_name": "SomeUnknownPipeline"})
    assert detect_base_architecture(str(tmp_path)) is None
