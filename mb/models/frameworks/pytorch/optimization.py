"""
Optimizer, learning-rate grouping, and precision settings for the PyTorch trainer.

Every setting here defaults to the trainer's original behavior: Adam with no weight decay,
one learning rate for all parameters, fp32, one optimizer step per batch. With those
defaults :func:`build_optimizer` returns exactly the optimizer the trainer always built, so a
run that sets none of these options trains identically.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from mb.models.types import MixedPrecisionMode, OptimizerType
from mb.utils.logging_setup import get_logger
from mb.utils.translations import _

logger = get_logger(__name__)

def _block_index(name: str) -> Optional[int]:
    """
    Transformer block index in a parameter name, for layer-wise decay; None if not in one.

    A block is the first ``layers.N`` / ``layer.N`` segment pair with no index earlier in the
    name. That matches SigLIP (``encoder.layers.N``) and ViT in both ``transformers`` 4.x
    (``encoder.layer.N``) and 5.x (``vit.layers.N``). Hierarchical backbones (FocalNet's
    ``stages.N.layers.M``) and torchvision CNNs (``layer1.N``) do not match, so layer-wise
    decay is not applied to them rather than applied with a wrong depth ordering.
    """
    parts = name.split(".")
    for i, part in enumerate(parts[:-1]):
        if part.isdigit():
            return None
        if part in ("layers", "layer") and parts[i + 1].isdigit():
            return int(parts[i + 1])
    return None


@dataclass(frozen=True)
class OptimizationSettings:
    """Resolved optimization options for one training run."""

    optimizer: OptimizerType = OptimizerType.ADAM
    weight_decay: float = 0.0
    layer_decay: float = 1.0
    grad_accum_steps: int = 1
    mixed_precision: MixedPrecisionMode = MixedPrecisionMode.NONE
    gradient_checkpointing: bool = False
    restore_best: bool = False


def _float_setting(hyperparams: Dict[str, Any], key: str, default: float) -> float:
    raw = hyperparams.get(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        logger.warning(_("Ignoring invalid {key}={value!r}; using {default}").format(
            key=key, value=raw, default=default))
        return default


def resolve_optimization_settings(hyperparams: Dict[str, Any]) -> OptimizationSettings:
    """
    Read and validate the optimization options from merged hyperparameters.

    Invalid values fall back to the default with a warning, matching how the trainer treats
    other malformed config values, so a typo cannot stop a long run from starting.
    """
    optimizer = OptimizerType.try_from(hyperparams.get("optimizer"))
    if optimizer is None:
        if hyperparams.get("optimizer") is not None:
            logger.warning(_("Unknown optimizer {value!r}; using {default}").format(
                value=hyperparams.get("optimizer"), default=OptimizerType.get_default().value))
        optimizer = OptimizerType.get_default()

    precision = MixedPrecisionMode.try_from(hyperparams.get("mixed_precision"))
    if precision is None:
        if hyperparams.get("mixed_precision") is not None:
            logger.warning(_("Unknown mixed_precision {value!r}; using {default}").format(
                value=hyperparams.get("mixed_precision"),
                default=MixedPrecisionMode.get_default().value))
        precision = MixedPrecisionMode.get_default()

    weight_decay = _float_setting(hyperparams, "weight_decay", 0.0)
    if weight_decay < 0:
        logger.warning(_("weight_decay must be >= 0; using 0"))
        weight_decay = 0.0

    layer_decay = _float_setting(hyperparams, "layer_decay", 1.0)
    if not 0.0 < layer_decay <= 1.0:
        logger.warning(_("layer_decay must be in (0, 1]; using 1.0 (disabled)"))
        layer_decay = 1.0

    raw_accum = hyperparams.get("grad_accum_steps")
    try:
        grad_accum_steps = int(raw_accum) if raw_accum is not None else 1
    except (TypeError, ValueError):
        grad_accum_steps = 0
    if grad_accum_steps < 1:
        logger.warning(_("grad_accum_steps must be >= 1; using 1"))
        grad_accum_steps = 1

    return OptimizationSettings(
        optimizer=optimizer,
        weight_decay=weight_decay,
        layer_decay=layer_decay,
        grad_accum_steps=grad_accum_steps,
        mixed_precision=precision,
        gradient_checkpointing=bool(hyperparams.get("gradient_checkpointing", False)),
        restore_best=bool(hyperparams.get("restore_best", False)),
    )


def optimizer_steps_per_epoch(num_batches: int, grad_accum_steps: int) -> int:
    """Optimizer steps one epoch takes; a trailing partial accumulation group still steps."""
    return math.ceil(num_batches / max(1, grad_accum_steps))


def _head_parameter_ids(model: nn.Module) -> Set[int]:
    head_params = getattr(model, "head_parameters", None)
    if callable(head_params):
        return {id(p) for p in head_params()}
    for attr in ("fc", "classifier"):
        head = getattr(model, attr, None)
        if isinstance(head, nn.Module):
            return {id(p) for p in head.parameters()}
    return set()


def _layer_ids(model: nn.Module) -> Tuple[Dict[int, int], int]:
    """
    Depth index per parameter for layer-wise decay, and the deepest index.

    Embeddings are 0, encoder block N is N + 1, and everything after the encoder (final
    norm, pooling, classification head) shares the top index. Returns an empty mapping when
    the model has no recognizable encoder blocks.
    """
    named = list(model.named_parameters())
    blocks = {
        id(p): index + 1
        for name, p in named
        if (index := _block_index(name)) is not None
    }
    if not blocks:
        return {}, 0
    top = max(blocks.values()) + 1
    head = _head_parameter_ids(model)
    ids: Dict[int, int] = {}
    for name, p in named:
        if id(p) in head:
            ids[id(p)] = top
        elif id(p) in blocks:
            ids[id(p)] = blocks[id(p)]
        elif "embeddings" in name:
            ids[id(p)] = 0
        else:
            ids[id(p)] = top
    return ids, top


def _decays(name: str, param: nn.Parameter) -> bool:
    """Weight decay applies to weight matrices, not to biases or normalization scales."""
    return param.ndim > 1 and not name.endswith(".bias")


def build_optimizer(
    model: nn.Module,
    lr: float,
    settings: OptimizationSettings,
    *,
    trainable_only: bool = False,
    apply_layer_decay: bool = False,
) -> Tuple[optim.Optimizer, List[float]]:
    """
    Build the optimizer and return it with each parameter group's base learning rate.

    The learning-rate list is what a scheduler such as ``OneCycleLR`` takes as ``max_lr``
    so per-group scaling survives scheduling. With default settings this is a single-group
    optimizer over the same parameters the trainer always used.
    """
    named = [
        (n, p) for n, p in model.named_parameters() if p.requires_grad or not trainable_only
    ]
    opt_cls = optim.AdamW if settings.optimizer == OptimizerType.ADAMW else optim.Adam

    layer_ids: Dict[int, int] = {}
    top = 0
    if apply_layer_decay and settings.layer_decay < 1.0:
        layer_ids, top = _layer_ids(model)
        if not layer_ids:
            logger.warning(
                _("layer_decay is set but this architecture has no recognizable encoder "
                  "blocks; training with one learning rate for all layers"))

    use_wd = settings.weight_decay > 0
    if not use_wd and not layer_ids:
        params: Iterable[nn.Parameter] = [p for _n, p in named]
        # AdamW's own default weight_decay is 0.01, so it has to be pinned to the setting.
        kwargs = {"weight_decay": settings.weight_decay} if opt_cls is optim.AdamW else {}
        return opt_cls(params, lr=lr, **kwargs), [lr]

    groups: Dict[Tuple[int, bool], Dict[str, Any]] = {}
    for name, p in named:
        depth = layer_ids.get(id(p), top)
        decays = use_wd and _decays(name, p)
        key = (depth, decays)
        if key not in groups:
            scale = settings.layer_decay ** (top - depth) if layer_ids else 1.0
            groups[key] = {
                "params": [],
                "lr": lr * scale,
                "weight_decay": settings.weight_decay if decays else 0.0,
            }
        groups[key]["params"].append(p)

    ordered = [groups[k] for k in sorted(groups)]
    optimizer = opt_cls(ordered, lr=lr)
    return optimizer, [g["lr"] for g in ordered]


def resolve_autocast_dtype(
    mode: MixedPrecisionMode, device: torch.device
) -> Optional[torch.dtype]:
    """
    The autocast dtype for *mode* on *device*, or None for fp32.

    Unsupported combinations fall back to fp32 with a warning rather than failing, since the
    run is still correct at full precision, just slower and larger.
    """
    if mode == MixedPrecisionMode.NONE:
        return None
    if mode == MixedPrecisionMode.FP16:
        if device.type != "cuda":
            logger.warning(_("fp16 mixed precision needs CUDA; training in fp32"))
            return None
        return torch.float16
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        logger.warning(_("This GPU does not support bf16; training in fp32. Try fp16 instead."))
        return None
    return torch.bfloat16


def make_grad_scaler(dtype: Optional[torch.dtype]) -> Optional[Any]:
    """Loss scaler for fp16 autocast (bf16 has fp32's exponent range and needs none)."""
    if dtype != torch.float16:
        return None
    amp = getattr(torch, "amp", None)
    if amp is not None and hasattr(amp, "GradScaler"):
        try:
            return amp.GradScaler("cuda")
        except TypeError:
            pass
    return torch.cuda.amp.GradScaler()


def autocast_context(device: torch.device, dtype: Optional[torch.dtype]):
    """Autocast for *dtype* on *device*, or a no-op when *dtype* is None."""
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


# OneCycleLR's warmup starts at max_lr / this; kept at the library default so the warmup
# phase is unchanged.
ONE_CYCLE_DIV_FACTOR = 25.0
# Library default for the final factor, used when unfrozen_lr_min is unusable.
_ONE_CYCLE_DEFAULT_FINAL_DIV = 1e4


def build_one_cycle(
    optimizer: optim.Optimizer,
    group_lrs: List[float],
    max_lr: float,
    min_lr: Optional[float],
    total_steps: int,
    pct_start: float,
) -> torch.optim.lr_scheduler.OneCycleLR:
    """
    OneCycle schedule from ``max_lr / 25`` up to *max_lr* and down to *min_lr*.

    The final rate is set through ``final_div_factor``, which OneCycleLR applies to every
    parameter group, so with layer-wise decay each group ends at the same fraction of its
    own peak. A *min_lr* that is missing, non-positive, or not below *max_lr* falls back to
    the library default ending (``max_lr / 25 / 1e4``) with a warning.
    """
    # OneCycleLR ends its warmup at step pct_start * total_steps - 1. When that is exactly 0,
    # its first step divides by zero (e.g. 20 steps at the default pct_start of 0.05). A
    # negligible nudge keeps the warmup end off 0 without changing the schedule.
    if float(pct_start) * total_steps == 1.0:
        pct_start = float(pct_start) * (1.0 + 1e-6)
    initial = float(max_lr) / ONE_CYCLE_DIV_FACTOR
    final_div = _ONE_CYCLE_DEFAULT_FINAL_DIV
    if min_lr is not None and 0.0 < float(min_lr) < float(max_lr):
        final_div = initial / float(min_lr)
    else:
        logger.warning(
            _("unfrozen_lr_min={min_lr!r} must be > 0 and below unfrozen_lr_max={max_lr}; "
              "ending the schedule at the library default instead").format(
                min_lr=min_lr, max_lr=max_lr))
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        # A single group keeps the scalar form.
        max_lr=group_lrs if len(group_lrs) > 1 else float(max_lr),
        total_steps=total_steps,
        pct_start=pct_start,
        anneal_strategy="cos",
        div_factor=ONE_CYCLE_DIV_FACTOR,
        final_div_factor=final_div,
    )


def scheduler_checkpoint_state(scheduler: Any) -> Dict[str, Any]:
    """
    A scheduler's state without callables, so it survives ``torch.load(weights_only=True)``.

    Some PyTorch versions keep OneCycleLR's annealing function as a bound method in its
    state; the scheduler rebuilt on resume already has its own, so dropping it loses nothing.
    """
    return {k: v for k, v in scheduler.state_dict().items() if not callable(v)}


def load_optimizer_moments(optimizer: optim.Optimizer, state: Dict[str, Any]) -> None:
    """
    Load *state*'s per-parameter moments while keeping *optimizer*'s current group settings.

    Learning rate, weight decay and the scheduler's per-group bookkeeping come from this
    run's configuration, so a value changed between runs takes effect after resume. Raises
    ``ValueError`` when the parameter groups do not line up (different model or options).
    """
    current = [{k: v for k, v in g.items() if k != "params"} for g in optimizer.param_groups]
    optimizer.load_state_dict(state)
    for group, settings in zip(optimizer.param_groups, current):
        group.update(settings)
