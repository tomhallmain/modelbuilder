"""
``mb train --model-type image_generation_lora`` — LoRA fine-tuning of a base
image-generation (diffusion) model on the flat image+caption training set produced by
``mb data create-dataset --model-type image_generation_lora``
(:class:`~mb.data.lora_dataset.LoraDatasetCreator`).

PyTorch + `diffusers <https://github.com/huggingface/diffusers>`_ +
`peft <https://github.com/huggingface/peft>`_ only — no mature first-party diffusion/PEFT
toolchain exists for Keras/TensorFlow. Install with ``pip install -e ".[lora]"``
(``requirements-ml.txt`` documents the same extras).

Three base architectures are implemented (see :mod:`mb.models.generation_architectures`
for the full registry): **Flux** (transformer denoiser + CLIP/T5 text encoders) is the
recommended default for new LoRAs — a current-generation model, meaningfully more coherent
than earlier Stable Diffusion versions. **Chroma** (a community Flux-derived model) reuses
the same training function under an explicit, flagged assumption that it's Flux-compatible
in ``diffusers`` — see ``mb/models/generation_architectures/chroma.py``, not verified
against a real checkpoint. **Stable Diffusion 1.x/2.x** (UNet denoiser + single CLIP text
encoder) is kept working alongside them — smaller and faster to iterate on for quick
smoke-testing, and still the architecture with the most existing LoRA training
data/tooling to compare against — but is not the recommended starting point.

This is deliberately a standalone module, not a branch inside
:class:`~mb.training.trainer.ModelTrainer` / :class:`~mb.models.base.FrameworkTrainer` —
those assume a classifier (``num_classes``, a registered ``ArchitectureType``, a
frozen/unfrozen epoch schedule, a single complete saved model). None of that fits a
frozen base diffusion pipeline + trainable low-rank adapter deltas, so retrofitting it in
would mean conditionals threaded through nearly every line of the classification trainer.

**Scope of this first slice** (deliberately, to keep the implementation reviewable):
single-process, single-phase, step-based training loop; no ``accelerate`` (no gradient
accumulation / mixed precision / multi-GPU); no EMA; no periodic validation-sample
generation during training. These are natural follow-ups, not required for a first
working LoRA adapter.

**Confidence note**: the Stable Diffusion 1.x path is a well-established, "textbook"
diffusers training loop. The Flux path is structurally faithful to the published
diffusers/peft approach for Flux LoRA training, but its low-level tensor plumbing (latent
packing, image/text position-id tensors, the transformer's exact forward signature) is
more version-sensitive and has a real chance of drifting from whatever ``diffusers``
version ends up installed — treat it as the higher-risk part of this module to verify
first against a real checkpoint. The Chroma path is the same code again, plus an
*additional* unverified assumption on top (that Chroma checkpoints load via the same
``FluxTransformer2DModel``/``FluxPipeline`` classes) — treat it as higher-risk still.
"""

from __future__ import annotations

import threading
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from mb.cancellation import check_cancel_event
from mb.data.lora_captions import read_caption
from mb.models.generation_architectures import (
    BaseArchitectureDescriptor,
    BaseGenerationArchitecture,
    detect_base_architecture,
    get_descriptor,
)
from mb.utils.logging_setup import get_logger
from mb.utils.translations import _

logger = get_logger(__name__)

_IMPLEMENTED_ARCHITECTURES = (
    BaseGenerationArchitecture.FLUX,
    BaseGenerationArchitecture.CHROMA,
    BaseGenerationArchitecture.STABLE_DIFFUSION_1X,
)

# Chroma reuses the Flux training function verbatim (see mb/models/generation_architectures
# /chroma.py for the assumption this rests on: Flux-compatible diffusers component classes).
_FLUX_FAMILY_ARCHITECTURES = (
    BaseGenerationArchitecture.FLUX,
    BaseGenerationArchitecture.CHROMA,
)

_DEFAULT_BASE_MODEL_ERROR = _(
    "--architecture is required for --model-type image_generation_lora (the base "
    "image-generation model to fine-tune, e.g. a local path or hub id)."
)
_BASE_ARCHITECTURE_UNKNOWN_ERROR = _(
    "Could not determine the base model's architecture from {path} (no readable "
    "model_index.json, or a hub id rather than a local path). Pass "
    "--base-model-architecture explicitly (one of: {choices})."
)
_BASE_ARCHITECTURE_UNIMPLEMENTED_ERROR = _(
    "Base architecture '{arch}' is recognized but not implemented yet — implemented "
    "architectures today: {implemented}."
)


@dataclass
class LoraTrainingConfig:
    """Resolved inputs for :func:`train_image_generation_lora`."""

    base_model: str
    base_architecture: BaseGenerationArchitecture
    data_dir: Path
    output_dir: Path
    rank: int = 16
    alpha: int = 16
    resolution: int = 512
    batch_size: int = 1
    learning_rate: float = 1e-4
    max_train_steps: int = 1000
    num_workers: int = 0
    seed: Optional[int] = None
    device: Optional[str] = None


def build_lora_training_config_from_args(args: Namespace, pipeline: Any) -> LoraTrainingConfig:
    """
    Merge ``mb train --model-type image_generation_lora`` CLI flags with pipeline defaults.

    Resolving ``base_architecture`` is mandatory (LoRA target layers, and which components
    to load, differ by base architecture — see :mod:`mb.models.generation_architectures`):
    an explicit ``--base-model-architecture`` wins; otherwise it's detected from
    ``base_model``'s ``model_index.json`` when that's a local directory. Raises
    ``ValueError`` if neither resolves anything, rather than silently guessing.
    """
    base_model = args.architecture or pipeline.get("model.default_architecture")
    if not base_model:
        raise ValueError(_DEFAULT_BASE_MODEL_ERROR)
    base_model = str(base_model)

    base_architecture = BaseGenerationArchitecture.try_from(
        getattr(args, "base_model_architecture", None)
    )
    if base_architecture is None:
        base_architecture = detect_base_architecture(base_model)
    if base_architecture is None:
        raise ValueError(
            _BASE_ARCHITECTURE_UNKNOWN_ERROR.format(
                path=base_model,
                choices=", ".join(a.value for a in BaseGenerationArchitecture),
            )
        )

    data_dir = args.data_dir or Path(pipeline.get("data.data_dir", "data"))
    output_dir = args.output_dir or Path(pipeline.get("paths.models_dir", "data/models"))

    rank = args.lora_rank if args.lora_rank is not None else 16
    alpha = args.lora_alpha if args.lora_alpha is not None else rank

    return LoraTrainingConfig(
        base_model=base_model,
        base_architecture=base_architecture,
        data_dir=Path(data_dir),
        output_dir=Path(output_dir),
        rank=rank,
        alpha=alpha,
        resolution=args.image_size if args.image_size is not None else 512,
        batch_size=args.batch_size if args.batch_size is not None else 1,
        learning_rate=args.learning_rate if args.learning_rate is not None else 1e-4,
        max_train_steps=args.max_train_steps if args.max_train_steps is not None else 1000,
        num_workers=args.num_workers if args.num_workers is not None else 0,
        seed=args.seed,
    )


class _LoraImageCaptionDataset:
    """
    Reads the flat ``image.jpg`` (+ optional ``image.txt`` caption) layout produced by
    :class:`~mb.data.lora_dataset.LoraDatasetCreator`. Constructed lazily inside the
    architecture-specific training functions since it needs ``torch``/``torchvision``.

    ``tokenizers`` maps an arbitrary name (used as the ``input_ids_<name>`` batch key) to
    ``(tokenizer, max_length)`` — one entry for a single-encoder architecture (Stable
    Diffusion 1.x), two for a dual-encoder one (Flux: CLIP + T5, each with a different
    natural max sequence length).
    """

    def __init__(
        self,
        data_dir: Path,
        tokenizers: Dict[str, Tuple[Any, int]],
        resolution: int,
    ) -> None:
        from PIL import Image  # noqa: F401 — import-availability check, used in __getitem__
        from torchvision import transforms

        self._Image = Image
        self.data_dir = Path(data_dir)
        self.tokenizers = tokenizers
        self.image_paths = sorted(self.data_dir.glob("*.jpg"))
        if not self.image_paths:
            raise ValueError(_("No prepared images found under {path}").format(path=self.data_dir))
        self.transform = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.image_paths[idx]
        image = self._Image.open(path).convert("RGB")
        pixel_values = self.transform(image)
        caption = read_caption(path) or ""
        item: Dict[str, Any] = {"pixel_values": pixel_values}
        for name, (tokenizer, max_length) in self.tokenizers.items():
            input_ids = tokenizer(
                caption,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).input_ids[0]
            item[f"input_ids_{name}"] = input_ids
        return item


def train_image_generation_lora(
    config: LoraTrainingConfig,
    *,
    cancel_event: Optional[threading.Event] = None,
) -> Path:
    """
    Fine-tune a LoRA adapter for ``config.base_model`` on ``config.data_dir``.

    Dispatches to the architecture-specific training function for
    ``config.base_architecture``. Only the adapter weights are saved (small
    ``.safetensors`` + ``adapter_config.json``), not the base model. Returns the directory
    the adapter was saved to (``config.output_dir``).

    Raises ``ImportError`` with a clear message if ``diffusers``/``peft``/``transformers``
    aren't installed (``pip install -e ".[lora]"``). Raises ``NotImplementedError`` if
    ``config.base_architecture`` isn't (yet) one this function knows how to load — see
    :mod:`mb.models.generation_architectures`.
    """
    descriptor = get_descriptor(config.base_architecture)
    if not descriptor.implemented:
        raise NotImplementedError(
            _BASE_ARCHITECTURE_UNIMPLEMENTED_ERROR.format(
                arch=config.base_architecture.value,
                implemented=", ".join(a.value for a in _IMPLEMENTED_ARCHITECTURES),
            )
        )
    if config.base_architecture in _FLUX_FAMILY_ARCHITECTURES:
        return _train_flux(config, descriptor, cancel_event)
    return _train_stable_diffusion_1x(config, descriptor, cancel_event)


def _train_stable_diffusion_1x(
    config: LoraTrainingConfig,
    descriptor: BaseArchitectureDescriptor,
    cancel_event: Optional[threading.Event],
) -> Path:
    assert len(descriptor.text_encoder_subfolders) == 1 and descriptor.unet_subfolder, (
        f"Descriptor for {config.base_architecture.value} isn't a single-text-encoder UNet "
        "architecture; _train_stable_diffusion_1x needs updating to load it."
    )

    try:
        import torch
        import torch.nn.functional as F
        from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
        from peft import LoraConfig, get_peft_model
        from transformers import CLIPTextModel, CLIPTokenizer
    except ImportError as e:
        raise ImportError(_lora_extras_import_error(e)) from e

    device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if config.seed is not None:
        torch.manual_seed(config.seed)

    logger.info(
        f"Loading base model components ({config.base_architecture.value}) from: {config.base_model}"
    )
    text_encoder_subfolder = descriptor.text_encoder_subfolders[0]
    tokenizer = CLIPTokenizer.from_pretrained(config.base_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(config.base_model, subfolder=text_encoder_subfolder)
    vae = AutoencoderKL.from_pretrained(config.base_model, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(config.base_model, subfolder=descriptor.unet_subfolder)
    noise_scheduler = DDPMScheduler.from_pretrained(config.base_model, subfolder="scheduler")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    logger.info(f"Injecting LoRA adapters (rank={config.rank}, alpha={config.alpha})")
    lora_config = LoraConfig(
        r=config.rank,
        lora_alpha=config.alpha,
        init_lora_weights="gaussian",
        target_modules=list(descriptor.lora_target_modules),
    )
    unet = get_peft_model(unet, lora_config)

    unet.to(device)
    vae.to(device)
    text_encoder.to(device)

    dataset = _LoraImageCaptionDataset(
        config.data_dir,
        tokenizers={"clip": (tokenizer, tokenizer.model_max_length)},
        resolution=config.resolution,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    logger.info(f"Training on {len(dataset)} images, batch_size={config.batch_size}")

    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config.learning_rate)

    unet.train()
    global_step = 0
    log_startup_info(config)
    while global_step < config.max_train_steps:
        for batch in dataloader:
            if global_step % 100 == 0:
                check_cancel_event(cancel_event)

            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids_clip"].to(device)

            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                encoder_hidden_states = text_encoder(input_ids)[0]

            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            if global_step % 50 == 0 or global_step == config.max_train_steps:
                logger.info(f"Step {global_step}/{config.max_train_steps}: loss={loss.item():.4f}")
            if global_step >= config.max_train_steps:
                break

    config.output_dir.mkdir(parents=True, exist_ok=True)
    unet.save_pretrained(str(config.output_dir))
    logger.info(f"LoRA adapter saved to: {config.output_dir}")
    return config.output_dir


def _train_flux(
    config: LoraTrainingConfig,
    descriptor: BaseArchitectureDescriptor,
    cancel_event: Optional[threading.Event],
) -> Path:
    """
    Flux-family LoRA training (Flux, and Chroma under its documented Flux-compatibility
    assumption — see :data:`_FLUX_FAMILY_ARCHITECTURES`): rectified-flow (flow-matching)
    objective on a transformer denoiser, conditioned on a pooled CLIP embedding plus a T5
    sequence embedding. Loads via ``FluxTransformer2DModel``/``FluxPipeline`` regardless of
    which of the two architectures this is called for.

    See this module's docstring — the tensor plumbing here (latent packing, image/text
    position ids, the transformer's forward signature) is the highest-uncertainty part of
    this feature; verify against the installed ``diffusers`` version's own Flux training
    example before relying on it. For Chroma specifically, this also rests on the
    additional, separately-flagged assumption in ``chroma.py`` that it loads via the same
    classes as Flux.
    """
    assert (
        descriptor.transformer_subfolder
        and descriptor.text_encoder_types == ("clip", "t5")
        and len(descriptor.text_encoder_subfolders) == 2
    ), (
        f"Descriptor for {config.base_architecture.value} isn't a CLIP+T5 transformer "
        "architecture; _train_flux needs updating to load it."
    )

    try:
        import torch
        import torch.nn.functional as F
        from diffusers import AutoencoderKL, FluxTransformer2DModel
        from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
        from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
        from peft import LoraConfig, get_peft_model
        from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
    except ImportError as e:
        raise ImportError(_lora_extras_import_error(e)) from e

    device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if config.seed is not None:
        torch.manual_seed(config.seed)

    logger.info(
        f"Loading base model components ({config.base_architecture.value}) from: {config.base_model}"
    )
    clip_subfolder, t5_subfolder = descriptor.text_encoder_subfolders
    clip_tokenizer = CLIPTokenizer.from_pretrained(config.base_model, subfolder="tokenizer")
    clip_text_encoder = CLIPTextModel.from_pretrained(config.base_model, subfolder=clip_subfolder)
    t5_tokenizer = T5TokenizerFast.from_pretrained(config.base_model, subfolder="tokenizer_2")
    t5_text_encoder = T5EncoderModel.from_pretrained(config.base_model, subfolder=t5_subfolder)
    vae = AutoencoderKL.from_pretrained(config.base_model, subfolder="vae")
    transformer = FluxTransformer2DModel.from_pretrained(
        config.base_model, subfolder=descriptor.transformer_subfolder
    )
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        config.base_model, subfolder="scheduler"
    )

    clip_text_encoder.requires_grad_(False)
    t5_text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    transformer.requires_grad_(False)

    logger.info(f"Injecting LoRA adapters (rank={config.rank}, alpha={config.alpha})")
    lora_config = LoraConfig(
        r=config.rank,
        lora_alpha=config.alpha,
        init_lora_weights="gaussian",
        target_modules=list(descriptor.lora_target_modules),
    )
    transformer = get_peft_model(transformer, lora_config)

    transformer.to(device)
    vae.to(device)
    clip_text_encoder.to(device)
    t5_text_encoder.to(device)

    # 512 is the standard Flux max sequence length for the T5 encoder (both the "dev" and
    # "schnell" checkpoints); unlike CLIP, T5 tokenizers don't carry a usable model_max_length.
    t5_max_sequence_length = 512
    dataset = _LoraImageCaptionDataset(
        config.data_dir,
        tokenizers={
            "clip": (clip_tokenizer, clip_tokenizer.model_max_length),
            "t5": (t5_tokenizer, t5_max_sequence_length),
        },
        resolution=config.resolution,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    logger.info(f"Training on {len(dataset)} images, batch_size={config.batch_size}")

    trainable_params = [p for p in transformer.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config.learning_rate)

    # Guidance-distilled ("dev") checkpoints take a guidance-scale embedding at train time;
    # the un-distilled ("schnell") variant doesn't have this transformer input at all.
    guidance_embeds = bool(getattr(transformer.config, "guidance_embeds", False))

    transformer.train()
    global_step = 0
    log_startup_info(config)
    while global_step < config.max_train_steps:
        for batch in dataloader:
            if global_step % 100 == 0:
                check_cancel_event(cancel_event)

            pixel_values = batch["pixel_values"].to(device)
            clip_ids = batch["input_ids_clip"].to(device)
            t5_ids = batch["input_ids_t5"].to(device)
            bsz = pixel_values.shape[0]

            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
                pooled_prompt_embeds = clip_text_encoder(clip_ids).pooler_output
                encoder_hidden_states = t5_text_encoder(t5_ids)[0]

            height, width = latents.shape[-2], latents.shape[-1]
            packed_latents = FluxPipeline._pack_latents(latents, bsz, latents.shape[1], height, width)
            latent_image_ids = FluxPipeline._prepare_latent_image_ids(
                bsz, height // 2, width // 2, device, packed_latents.dtype
            )
            text_ids = torch.zeros(t5_ids.shape[1], 3, device=device, dtype=packed_latents.dtype)

            noise = torch.randn_like(packed_latents)
            t = torch.rand(bsz, device=device)
            sigmas = t.view(-1, 1, 1)
            noisy_latents = (1.0 - sigmas) * packed_latents + sigmas * noise
            # Flow-matching target: the velocity from data to noise (not epsilon-noise as in DDPM).
            target = noise - packed_latents
            timesteps = t * noise_scheduler.config.num_train_timesteps

            guidance = None
            if guidance_embeds:
                guidance = torch.full((bsz,), 1.0, device=device, dtype=packed_latents.dtype)

            model_pred = transformer(
                hidden_states=noisy_latents,
                timestep=timesteps / noise_scheduler.config.num_train_timesteps,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=encoder_hidden_states,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                return_dict=False,
            )[0]

            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            if global_step % 50 == 0 or global_step == config.max_train_steps:
                logger.info(f"Step {global_step}/{config.max_train_steps}: loss={loss.item():.4f}")
            if global_step >= config.max_train_steps:
                break

    config.output_dir.mkdir(parents=True, exist_ok=True)
    transformer.save_pretrained(str(config.output_dir))
    logger.info(f"LoRA adapter saved to: {config.output_dir}")
    return config.output_dir


def _lora_extras_import_error(e: ImportError) -> str:
    return _(
        "Image-generation LoRA training requires the 'lora' extras: "
        "pip install -e \".[lora]\" (diffusers, peft, transformers). Underlying error: {err}"
    ).format(err=e)


def log_startup_info(config: LoraTrainingConfig) -> None:
    logger.info("=" * 80)
    logger.info("LORA TRAINING STARTUP")
    logger.info("=" * 80)
    logger.info(f"Base model: {config.base_model}")
    logger.info(f"Base architecture: {config.base_architecture.value}")
    logger.info(f"Data dir: {config.data_dir}")
    logger.info(f"Output dir: {config.output_dir}")
    logger.info(f"Rank/alpha: {config.rank}/{config.alpha}")
    logger.info(f"Resolution: {config.resolution}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Learning rate: {config.learning_rate}")
    logger.info(f"Max train steps: {config.max_train_steps}")


def run_train_image_generation_lora_cli(args: Namespace, pipeline: Any) -> int:
    """CLI implementation for ``mb train --model-type image_generation_lora`` (returns exit code)."""
    # Only reject an *explicit* non-pytorch request; there is exactly one implementation
    # (this module, PyTorch/diffusers-only), so an unset --framework proceeds even if the
    # pipeline's classification-oriented default framework is keras — that default doesn't
    # apply to this model type.
    fw_raw = getattr(args, "framework", None)
    if fw_raw and str(fw_raw).strip().lower() != "pytorch":
        logger.error(
            _(
                "--model-type image_generation_lora only supports --framework pytorch "
                "(got: {fw})."
            ).format(fw=fw_raw)
        )
        return 1

    try:
        config = build_lora_training_config_from_args(args, pipeline)
    except ValueError as e:
        logger.error(str(e))
        return 1

    descriptor = get_descriptor(config.base_architecture)
    if not descriptor.implemented:
        logger.error(
            _BASE_ARCHITECTURE_UNIMPLEMENTED_ERROR.format(
                arch=config.base_architecture.value,
                implemented=", ".join(a.value for a in _IMPLEMENTED_ARCHITECTURES),
            )
        )
        return 1

    if not config.data_dir.exists():
        logger.error(_("Data directory does not exist: {path}").format(path=config.data_dir))
        return 1

    try:
        output_dir = train_image_generation_lora(config)
    except ImportError as e:
        logger.error(str(e))
        return 1
    except NotImplementedError as e:
        logger.error(str(e))
        return 1
    except ValueError as e:
        logger.error(str(e))
        return 1
    except Exception as e:
        logger.error(
            _("LoRA training failed: {err}").format(err=e),
            exc_info=getattr(args, "verbose", False),
        )
        return 1

    logger.info(_("Training completed successfully. Adapter saved to: {path}").format(path=output_dir))
    return 0
