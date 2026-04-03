"""
Pipeline (ML job) configuration: model, data, training, paths.

This is separate from the desktop shell settings in ``utils.config`` (``gui`` /
``app``). Use :func:`get_pipeline_config` / :func:`reload_pipeline_config` for
``mb train``, :class:`~mb.training.trainer.ModelTrainer`, and training UI.

Default file: ``mb/config/default_pipeline.yaml`` (shipped with the package).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from mb.utils.logging_setup import get_logger

logger = get_logger("mb.pipeline_config")

_MB_ROOT = Path(__file__).resolve().parent
DEFAULT_PIPELINE_YAML = _MB_ROOT / "config" / "default_pipeline.yaml"

_PIPELINE_KEYS = frozenset({"model", "data", "training", "paths"})

_global_pipeline: Optional["PipelineConfig"] = None


class PipelineConfig:
    """YAML-backed pipeline defaults merged over built-in defaults."""

    def __init__(self, config_path: Optional[Path] = None) -> None:
        self._active_path: Optional[Path] = None
        self._defaults = self._get_defaults()
        self._config: Dict[str, Any] = self._deep_merge({}, self._defaults)

        if config_path and config_path.exists():
            self.load_from_file(config_path)
        elif config_path:
            logger.warning("Pipeline config file not found: %s, using defaults", config_path)

    def _get_defaults(self) -> Dict[str, Any]:
        return {
            "model": {
                "default_type": "image_classification",
                "default_framework": "pytorch",
                "default_architecture": "resnet34",
            },
            "data": {
                "raw_data_dir": "raw_data",
                "data_dir": "data",
                "test_images_per_class": 1000,
                "image_size": 224,
                "batch_size": None,
                "image_types": [
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".tif",
                    ".tiff",
                    ".webp",
                    ".bmp",
                    ".heic",
                    ".avif",
                ],
                "video_types": [".mp4", ".mkv", ".avi", ".wmv", ".mov", ".flv"],
            },
            "training": {
                "frozen_epochs": 5,
                "unfrozen_epochs": 20,
                "frozen_lr": 0.001,
                "unfrozen_lr_max": 0.0003,
                "unfrozen_lr_min": 0.00001,
                "num_workers": 12,
                "store_checkpoints": False,
            },
            "paths": {
                "models_dir": "data/models",
                "logs_dir": "logs",
                "timing_dir": "timing_data",
            },
        }

    def load_from_file(self, config_path: Path) -> None:
        try:
            with open(config_path, encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            filtered = {k: v for k, v in raw.items() if k in _PIPELINE_KEYS}
            self._config = self._deep_merge(self._deep_merge({}, self._defaults), filtered)
            self._active_path = config_path
            logger.info("Loaded pipeline configuration from %s", config_path)
        except Exception:
            logger.exception("Failed to load pipeline config from %s", config_path)
            self._config = self._deep_merge({}, self._defaults)
            self._active_path = None

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(base)
        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def get(self, key_path: str, default: Any = None) -> Any:
        keys = key_path.split(".")
        value: Any = self._config
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._config)

    def training_hyperparams(self) -> Dict[str, Any]:
        """
        Flat hyperparameter keys for :meth:`HyperparameterManager.merge_hyperparams`.

        Pulls training-related values from ``training.*`` plus ``data.image_size`` and
        ``data.batch_size`` (merged into the same dict used by the trainer). Keys
        with value ``None`` are omitted so defaults from the model type can apply.
        """
        raw = {
            "frozen_epochs": self.get("training.frozen_epochs"),
            "unfrozen_epochs": self.get("training.unfrozen_epochs"),
            "frozen_lr": self.get("training.frozen_lr"),
            "unfrozen_lr_max": self.get("training.unfrozen_lr_max"),
            "unfrozen_lr_min": self.get("training.unfrozen_lr_min"),
            "num_workers": self.get("training.num_workers"),
            "image_size": self.get("data.image_size"),
            "batch_size": self.get("data.batch_size"),
        }
        return {k: v for k, v in raw.items() if v is not None}

    @property
    def active_path(self) -> Optional[Path]:
        return self._active_path

    @property
    def file_types(self) -> List[str]:
        """Image extensions plus video extensions when using pipeline video types in UI helpers."""
        ft = list(self.get("data.image_types") or [])
        return ft


def _resolve_default_pipeline_path(config_path: Optional[Path]) -> Optional[Path]:
    if config_path is not None:
        return config_path
    if DEFAULT_PIPELINE_YAML.is_file():
        return DEFAULT_PIPELINE_YAML
    return None


def _pipeline_paths_equivalent(a: Optional[Path], b: Optional[Path]) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    try:
        return a.resolve() == b.resolve()
    except OSError:
        return a == b


def reload_pipeline_config(
    config_path: Optional[Path] = None,
    *,
    force: bool = False,
) -> PipelineConfig:
    """Replace the global pipeline config. ``None`` loads packaged defaults when present."""
    global _global_pipeline
    path = _resolve_default_pipeline_path(config_path)
    if (
        not force
        and _global_pipeline is not None
        and _pipeline_paths_equivalent(_global_pipeline.active_path, path)
    ):
        return _global_pipeline
    _global_pipeline = PipelineConfig(path)
    return _global_pipeline


def get_pipeline_config(config_path: Optional[Path] = None) -> PipelineConfig:
    """
    Return the active :class:`PipelineConfig` singleton.

    Same resolution pattern as :func:`utils.config.get_application_config`: an
    explicit ``Path`` always reloads; ``None`` returns the cached instance or
    loads defaults.
    """
    global _global_pipeline
    if config_path is not None:
        reload_pipeline_config(config_path, force=True)
    elif _global_pipeline is None:
        reload_pipeline_config(None)
    assert _global_pipeline is not None
    return _global_pipeline


def reset_pipeline_config() -> None:
    global _global_pipeline
    _global_pipeline = None
