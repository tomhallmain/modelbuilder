"""
Interface every text-classification backend implements.

A backend only fits a model and produces *uncalibrated* probabilities; the harness in
:mod:`mb.training.text_trainer` owns data selection, weighting, calibration, thresholds,
evaluation and reporting, so all backends are scored identically.
"""

from __future__ import annotations

import json
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from mb.models.types import TextBackendType, TextCalibrationMethod
from mb.utils.translations import _

BACKEND_META_FILE = "backend.json"

ProgressFn = Callable[[str, Optional[float]], None]


@dataclass
class TextFitData:
    """Rows a backend trains or validates on. ``weights`` already combine class × tier."""

    texts: List[str]
    labels: np.ndarray
    weights: np.ndarray


@dataclass
class TextFitContext:
    """Everything a backend needs from the run besides the data."""

    model_id: Optional[str]
    seed: int
    max_length: int
    lr: Optional[float]
    epochs: Optional[int]
    batch_size: Optional[int]
    warmup_ratio: float
    weight_decay: float
    patience: int
    options: Dict[str, Any]
    device: str
    run_dir: Path
    # Model-selection score (higher is better) for uncalibrated val probabilities, in val
    # row order; backends with epochs use it for early stopping and checkpoint choice.
    selection_metric: Callable[[np.ndarray], float]
    cancel_event: Optional[threading.Event] = None
    progress: Optional[ProgressFn] = None

    def report(self, message: str, fraction: Optional[float] = None) -> None:
        if self.progress is not None:
            self.progress(message, fraction)


class TextBackend(ABC):
    """One model family (see :class:`~mb.models.types.TextBackendType`)."""

    name: ClassVar[TextBackendType]
    # Neural backends default to temperature scaling, linear ones to Platt scaling.
    neural: ClassVar[bool]
    default_model_id: ClassVar[Optional[str]] = None
    default_lr: ClassVar[Optional[float]] = None
    default_epochs: ClassVar[Optional[int]] = None
    default_batch_size: ClassVar[Optional[int]] = None
    # Allowed ``backend_options`` keys and their defaults.
    option_defaults: ClassVar[Dict[str, Any]] = {}
    # Python source for the run's standalone ``predict.py``: defines
    # ``load_model(model_dir, meta, device)`` returning an object whose
    # ``predict_proba(texts)`` reproduces :meth:`predict_proba` with only the backend's
    # third-party libraries (no ``mb`` import). It may use ``json``, ``Path``, ``np``,
    # ``_sigmoid`` and ``_auto_device`` from the script's shared part; placeholders are
    # filled from :meth:`standalone_values`.
    standalone_template: ClassVar[str]
    # pip packages ``predict.py`` needs besides numpy.
    standalone_requirements: ClassVar[Tuple[str, ...]] = ()

    @classmethod
    def default_calibration(cls) -> TextCalibrationMethod:
        return TextCalibrationMethod.TEMPERATURE if cls.neural else TextCalibrationMethod.PLATT

    @classmethod
    def resolve_options(cls, raw: Mapping[str, Any]) -> Dict[str, Any]:
        """Defaults merged with *raw*; unknown keys raise ``ValueError``."""
        unknown = sorted(k for k in raw if k not in cls.option_defaults)
        if unknown:
            raise ValueError(
                _("Unknown backend_options for {backend}: {bad} (allowed: {allowed})").format(
                    backend=cls.name.value,
                    bad=", ".join(unknown),
                    allowed=", ".join(cls.option_defaults) or _("none"),
                )
            )
        return {**cls.option_defaults, **dict(raw)}

    @classmethod
    def standalone_values(cls) -> Dict[str, str]:
        """Placeholder → Python literal substitutions for :attr:`standalone_template`."""
        return {"__OPTION_DEFAULTS__": repr(cls.option_defaults)}

    @classmethod
    def standalone_source(cls) -> str:
        src = cls.standalone_template
        for key, value in cls.standalone_values().items():
            src = src.replace(key, value)
        return src

    @abstractmethod
    def fit(self, train: TextFitData, val: TextFitData, ctx: TextFitContext) -> Dict[str, Any]:
        """Train; returns a JSON-able summary (e.g. per-epoch val metric, chosen epoch)."""

    @abstractmethod
    def predict_proba(
        self,
        texts: Sequence[str],
        *,
        cancel_event: Optional[threading.Event] = None,
        progress: Optional[ProgressFn] = None,
    ) -> np.ndarray:
        """Uncalibrated ``p(label = 1)`` per text, shape ``(n,)``, in input order."""

    def count_truncated(self, texts: Sequence[str]) -> Optional[int]:
        """Rows longer than the model's input limit; ``None`` when nothing is truncated."""
        return None

    @abstractmethod
    def save_weights(self, model_dir: Path) -> None:
        """Write everything :meth:`load_weights` needs into *model_dir*."""

    @classmethod
    @abstractmethod
    def load_weights(cls, model_dir: Path, meta: Dict[str, Any], device: str) -> TextBackend:
        """Rebuild a fitted backend from :meth:`save_weights` output and its metadata."""

    def metadata(self) -> Dict[str, Any]:
        """Extra fields stored in ``backend.json`` beside the backend name."""
        return {}

    def save(self, model_dir: Path) -> None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        self.save_weights(model_dir)
        meta = {"backend": self.name.value, **self.metadata()}
        (model_dir / BACKEND_META_FILE).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
