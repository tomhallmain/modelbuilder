"""Torch helpers shared by the neural text backends (imported only by those backends)."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from mb.utils.translations import _


def import_torch():
    try:
        import torch
    except ImportError as e:
        raise ImportError(
            _(
                "Neural text backends require PyTorch and transformers: "
                "pip install -e \".[pytorch,text]\". Underlying error: {err}"
            ).format(err=e)
        ) from e
    return torch


def import_transformers():
    try:
        import transformers
    except ImportError as e:
        raise ImportError(
            _(
                "Neural text backends require transformers: pip install -e \".[text]\". "
                "Underlying error: {err}"
            ).format(err=e)
        ) from e
    return transformers


def seed_everything(seed: int) -> None:
    torch = import_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def autocast_dtype(device: str, precision: str):
    """``torch.bfloat16`` / ``torch.float16`` for mixed precision, or ``None`` for fp32."""
    torch = import_torch()
    if not device.startswith("cuda") or precision == "fp32":
        return None
    if precision == "fp16":
        return torch.float16
    if precision in ("bf16", "auto") and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return None


def weighted_bce(logits, labels, weights):
    """Weighted mean binary cross-entropy: ``sum(w · bce) / sum(w)``."""
    torch = import_torch()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits.float(), labels.float(), reduction="none"
    )
    return (loss * weights).sum() / weights.sum().clamp_min(1e-12)


@dataclass
class EarlyStopper:
    """Tracks the best selection metric and how many evaluations since it improved."""

    patience: int
    best: Optional[float] = None
    best_step: Optional[int] = None
    bad_evals: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)

    def update(self, metric: float, step: int, epoch: float) -> bool:
        """Record an evaluation; returns ``True`` when it is the new best."""
        self.history.append({"step": step, "epoch": epoch, "val_metric": metric})
        if self.best is None or metric > self.best:
            self.best, self.best_step, self.bad_evals = metric, step, 0
            return True
        self.bad_evals += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.bad_evals >= max(self.patience, 1)

    def summary(self) -> Dict[str, Any]:
        return {"best_val_metric": self.best, "best_step": self.best_step, "evaluations": self.history}
