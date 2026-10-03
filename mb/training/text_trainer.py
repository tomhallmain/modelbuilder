"""
``mb train --model-type text_classification``: fit, calibrate and evaluate one run.

A run is configured entirely by the pipeline's ``text_classification`` section
(:mod:`mb.training.text_config`) plus CLI/GUI overrides, and writes one directory::

    <runs_dir>/<timestamp>_<backend>_<model-slug>/
      config.yaml          the section as run, backend defaults filled in
      environment.json     library versions, device, seed, dataset sha256s, timings
      model/               backend weights + backend.json
      calibrator.json      fitted on val
      thresholds.json, metrics.json, predictions_test.tsv, review_queue.tsv,
      cut_rescore.tsv, extra_reports.json, MODEL_CARD.md   (see text_evaluation)
      predict.py           standalone scorer (see text_predict_script)

The dataset is verified before the run directory is created, and training stops on any
failed check. Like the LoRA trainer this is a standalone module rather than a branch of
:class:`~mb.training.trainer.ModelTrainer`, whose image-folder, frozen/unfrozen-epoch
assumptions do not apply to string inputs.
"""

from __future__ import annotations

import importlib.metadata
import platform
import re
import sys
import threading
import time
from argparse import Namespace
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import yaml

from mb import __version__ as MB_VERSION
from mb.data.text_dataset import (
    DATASET_FILE,
    TextDatasetError,
    compute_sample_weights,
    gold_mask,
    load_text_dataset,
    select_training_indices,
    sha256_file,
    verify_text_dataset,
)
from mb.evaluate.classification.text_calibration import fit_calibrator, save_calibrator
from mb.evaluate.classification.text_evaluation import (
    CALIBRATOR_FILE,
    CONFIG_FILE,
    ENVIRONMENT_FILE,
    MODEL_DIR,
    evaluate_text_run,
    resolve_device,
    write_json,
)
from mb.evaluate.classification.text_metrics import average_precision
from mb.models.text_backends import TextFitContext, TextFitData, get_backend_class
from mb.models.types import TextCalibrationMethod
from mb.training.text_config import TextConfigError, TextRunConfig, resolve_text_run_config
from mb.utils.logging_setup import get_logger
from mb.utils.translations import _

logger = get_logger(__name__)

ProgressFn = Callable[[str, Optional[float]], None]

_LIBRARIES = {
    "numpy": "numpy",
    "scipy": "scipy",
    "scikit-learn": "scikit-learn",
    "torch": "torch",
    "transformers": "transformers",
    "tokenizers": "tokenizers",
    "safetensors": "safetensors",
}


def resolve_backend_defaults(config: TextRunConfig) -> TextRunConfig:
    """Fill backend defaults (model, optimizer, options, calibration) into *config*."""
    cls = get_backend_class(config.backend)
    options = cls.resolve_options(config.backend_options)
    config = config.with_backend_defaults(
        model_id=cls.default_model_id,
        lr=cls.default_lr,
        epochs=cls.default_epochs,
        batch_size=cls.default_batch_size,
        backend_options=options,
    )
    if config.calibration == TextCalibrationMethod.AUTO:
        config = replace(config, calibration=cls.default_calibration())
    return config


def _model_slug(config: TextRunConfig) -> str:
    raw = (config.model_id or "default").replace("\\", "/").rstrip("/").split("/")[-1]
    return re.sub(r"[^A-Za-z0-9._-]+", "-", raw).strip("-") or "model"


def _new_run_dir(config: TextRunConfig) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = config.runs_dir / f"{stamp}_{config.backend.value}_{_model_slug(config)}"
    run_dir, n = base, 2
    while run_dir.exists():
        run_dir = base.with_name(f"{base.name}_{n}")
        n += 1
    run_dir.mkdir(parents=True)
    return run_dir


def environment_info(config: TextRunConfig, device: str) -> Dict[str, Any]:
    libs: Dict[str, Optional[str]] = {}
    for label, dist in _LIBRARIES.items():
        try:
            libs[label] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            libs[label] = None
    gpu = None
    if device.startswith("cuda"):
        try:
            import torch

            gpu = torch.cuda.get_device_name(torch.device(device))
        except Exception:
            gpu = None
    return {
        "mb_version": MB_VERSION,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "libraries": libs,
        "device": device,
        "gpu": gpu,
        "seed": config.seed,
    }


def train_text_classifier(
    config: TextRunConfig,
    *,
    cancel_event: Optional[threading.Event] = None,
    progress: Optional[ProgressFn] = None,
) -> Path:
    """
    Verify the dataset, fit the configured backend, calibrate on val, evaluate, and write
    the run directory. Returns the run directory.

    Raises:
        TextDatasetError / ValueError: verification failed or the data cannot be used.
        ImportError: the backend's optional dependencies are missing.
        OperationCancelled: *cancel_event* was set.
    """
    say = progress or (lambda _m, _f: None)
    started = time.time()
    config = resolve_backend_defaults(config)
    backend_cls = get_backend_class(config.backend)
    device = resolve_device(config.device, neural=backend_cls.neural)
    if backend_cls.neural and config.device is None and device == "cpu":
        _warn_no_cuda(config)

    say(_("Loading dataset…"), None)
    ds = load_text_dataset(config.data_dir, keep_groups=True, cancel_event=cancel_event)
    say(_("Verifying dataset…"), None)
    report = verify_text_dataset(config.data_dir, dataset=ds, cancel_event=cancel_event)
    logger.info("Dataset verification:\n%s", report.format())
    if not report.ok:
        raise TextDatasetError(_("Dataset verification failed:\n{report}").format(report=report.format()))
    ds.drop_groups()
    sha = dict(report.file_sha256) or {DATASET_FILE: sha256_file(config.data_dir / DATASET_FILE)}

    train_idx, subsample = select_training_indices(
        ds,
        exclude_conflicts=config.exclude_conflicts,
        subsample_keep_unreviewed=config.subsample_keep_unreviewed,
        seed=config.seed,
    )
    weights, class_weights = compute_sample_weights(
        ds, train_idx, class_weight=config.class_weight, tier_weight=config.tier_weight
    )
    val_idx = ds.split_indices("val", exclude_conflicts=config.exclude_conflicts)
    val_labels = ds.labels[val_idx].astype(np.int64)
    # Fail on unknown gold tiers now rather than after training.
    gold_mask(ds, val_idx, config.gold_tiers)
    if val_labels.min(initial=1) == val_labels.max(initial=0):
        raise TextDatasetError(_("The val split needs both labels for model selection and calibration."))

    train = TextFitData(
        texts=[ds.texts[i] for i in train_idx],
        labels=ds.labels[train_idx].astype(np.float32),
        weights=weights,
    )
    val = TextFitData(
        texts=[ds.texts[i] for i in val_idx],
        labels=val_labels.astype(np.float32),
        weights=np.ones(len(val_idx), dtype=np.float32),
    )

    run_dir = _new_run_dir(config)
    logger.info("Text-classification run directory: %s", run_dir)
    with open(run_dir / CONFIG_FILE, "w", encoding="utf-8") as f:
        yaml.safe_dump(config.to_dict(), f, sort_keys=False, allow_unicode=True)
    env = environment_info(config, device)
    env["started"] = datetime.now().isoformat(timespec="seconds")
    env["dataset"] = {"dir": str(config.data_dir), "sha256": sha}
    env["training"] = {
        "n_train": len(train_idx),
        "n_train_pos": int(train.labels.sum()),
        "n_val": len(val_idx),
        "class_weights": {str(k): v for k, v in class_weights.items()},
        "subsample": None
        if subsample is None
        else {"ratio": subsample.ratio, "seed": subsample.seed, "available": subsample.available, "kept": subsample.kept},
    }
    write_json(run_dir / ENVIRONMENT_FILE, env)

    def selection_metric(probs: np.ndarray) -> float:
        ap = average_precision(val_labels, probs)
        return float(ap) if ap is not None else 0.0

    ctx = TextFitContext(
        model_id=config.model_id,
        seed=config.seed,
        max_length=config.max_length,
        lr=config.optim.lr,
        epochs=config.optim.epochs,
        batch_size=config.optim.batch_size,
        warmup_ratio=config.optim.warmup_ratio,
        weight_decay=config.optim.weight_decay,
        patience=config.early_stopping.patience,
        options=dict(config.backend_options),
        device=device,
        run_dir=run_dir,
        selection_metric=selection_metric,
        cancel_event=cancel_event,
        progress=(lambda m, f: say(m, None if f is None else 0.05 + 0.75 * f)),
    )
    backend = backend_cls()
    fit_started = time.time()
    fit_summary = backend.fit(train, val, ctx)
    env["training"]["fit"] = fit_summary
    env["training"]["fit_seconds"] = round(time.time() - fit_started, 1)
    backend.save(run_dir / MODEL_DIR)

    say(_("Calibrating on val ({method})…").format(method=config.calibration.value), 0.8)
    val_raw = backend.predict_proba(val.texts, cancel_event=cancel_event)
    calibrator = fit_calibrator(config.calibration, val_raw, val_labels)
    save_calibrator(calibrator, run_dir / CALIBRATOR_FILE)
    write_json(run_dir / ENVIRONMENT_FILE, env)

    evaluate_text_run(
        run_dir,
        backend=backend,
        calibrator=calibrator,
        dataset=ds,
        val_raw=val_raw,
        cancel_event=cancel_event,
        progress=(lambda m, f: say(m, None if f is None else 0.82 + 0.18 * f)),
    )
    env["finished"] = datetime.now().isoformat(timespec="seconds")
    env["total_seconds"] = round(time.time() - started, 1)
    write_json(run_dir / ENVIRONMENT_FILE, env)
    return run_dir


def _warn_no_cuda(config: TextRunConfig) -> None:
    """A neural backend fell back to CPU without being asked to; say why and how to fix it."""
    try:
        import torch

        build = torch.version.cuda
    except ImportError:
        build = None
    reason = (
        _("this PyTorch build has no CUDA support (on Windows, plain `pip install torch` installs "
          "the CPU-only build; install a CUDA build from https://pytorch.org/get-started/locally/)")
        if build is None
        else _("PyTorch was built for CUDA {v} but no usable GPU was found (check the driver)").format(v=build)
    )
    logger.warning(
        _(
            "No GPU available for {backend} ({model}): training on CPU will be very slow. Cause: "
            "{reason}. Set text_classification.device: cpu to run on CPU without this warning."
        ).format(backend=config.backend.value, model=config.model_id, reason=reason)
    )


def cli_progress_logger(min_interval: float = 30.0) -> ProgressFn:
    """
    Progress callback for CLI runs: logs phase changes (no fraction) immediately and step
    progress at most every *min_interval* seconds.
    """
    started = time.monotonic()
    last = [float("-inf")]

    def report(message: str, fraction: Optional[float]) -> None:
        now = time.monotonic()
        if fraction is not None and fraction < 1.0 and now - last[0] < min_interval:
            return
        last[0] = now
        elapsed = int(now - started)
        stamp = f"{elapsed // 3600}:{elapsed // 60 % 60:02d}:{elapsed % 60:02d}"
        if fraction is None:
            logger.info("[%s] %s", stamp, message)
        else:
            logger.info("[%s] %s (%.1f%%)", stamp, message, 100.0 * fraction)

    return report


def text_train_overrides(args: Namespace) -> Dict[str, Any]:
    """``mb train`` flags that override ``text_classification`` keys (``None`` = not given)."""

    def path(v: Any) -> Optional[str]:
        return None if v is None else str(v)

    return {
        "data_dir": path(getattr(args, "data_dir", None)),
        "runs_dir": path(getattr(args, "output_dir", None)),
        "backend": getattr(args, "text_backend", None),
        "model_id": getattr(args, "text_model_id", None),
        "seed": getattr(args, "seed", None),
        "optim.lr": getattr(args, "learning_rate", None),
        "optim.epochs": getattr(args, "epochs", None),
        "optim.batch_size": getattr(args, "batch_size", None),
    }


def run_train_text_classification_cli(args: Namespace, pipeline: Any) -> int:
    """CLI implementation for ``mb train --model-type text_classification`` (exit code)."""
    try:
        config = resolve_text_run_config(pipeline, text_train_overrides(args))
    except TextConfigError as e:
        logger.error(str(e))
        return 1
    if not (config.data_dir / DATASET_FILE).is_file():
        logger.error(
            _("No {file} in the text-classification data directory: {path}").format(
                file=DATASET_FILE, path=config.data_dir
            )
        )
        return 1
    try:
        run_dir = train_text_classifier(config, progress=cli_progress_logger())
    except ImportError as e:
        logger.error(str(e))
        return 1
    except (TextDatasetError, ValueError) as e:
        logger.error(str(e))
        return 1
    logger.info(_("Training completed successfully. Run directory: {path}").format(path=run_dir))
    print(run_dir)
    return 0
