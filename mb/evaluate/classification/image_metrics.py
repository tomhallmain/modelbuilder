"""
Image-classification metrics (PyTorch and Keras).

Uses the same ImageFolder layout and transforms conventions as
``mb.models.frameworks.{pytorch,keras}.data_loader`` so evaluation matches training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from mb.conversion.converters import detect_model_framework
from mb.data.file_types import configured_media_suffixes
from mb.evaluate._contracts import ClassificationMetricsReport, MetricsRequest
from mb.evaluate._weights import extract_pytorch_state_dict
from mb.models.types import FrameworkType, ModelType
from mb.utils.logging_setup import get_logger
from mb.utils.translations import _

logger = get_logger(__name__)


def _extensions_tuple() -> tuple[str, ...]:
    return tuple(sorted(configured_media_suffixes()))


def _build_imagefolder_torch(data_dir: Path, image_size: int, batch_size: int, num_workers: int):
    from torch.utils.data import DataLoader

    from mb.models.frameworks.pytorch.data_loader import ImageFolderDataset, get_val_transforms

    exts = _extensions_tuple()
    ds = ImageFolderDataset(
        root=data_dir,
        transform=get_val_transforms(image_size),
        extensions=exts,
    )
    if len(ds) == 0:
        raise ValueError(_("No images found under {path}").format(path=data_dir))
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )
    return ds, loader


def _resolve_framework(model_path: Path, hint: Optional[FrameworkType]) -> FrameworkType:
    if hint is not None:
        return hint
    raw = detect_model_framework(model_path)
    if raw is None:
        raise ValueError(_("Could not detect model framework from {path}").format(path=model_path))
    fw = FrameworkType.try_from(raw)
    if fw is None or fw not in (FrameworkType.PYTORCH, FrameworkType.KERAS):
        raise ValueError(
            _("Framework {fw} is not supported for metrics (use PyTorch or Keras).").format(fw=raw)
        )
    return fw


def run_image_classification_metrics_pytorch(req: MetricsRequest) -> ClassificationMetricsReport:
    import torch
    import torch.nn as nn

    from mb.models.frameworks.pytorch.trainer import PyTorchTrainer

    if not req.architecture:
        raise ValueError(_("--architecture is required for PyTorch metrics evaluation."))

    ds, loader = _build_imagefolder_torch(
        req.data_dir, req.image_size, req.batch_size, req.num_workers
    )
    class_names = list(ds.classes)
    n_classes = len(class_names)
    if req.num_classes is not None and int(req.num_classes) != n_classes:
        raise ValueError(
            _("--num-classes ({n}) does not match directory class count ({c}).").format(
                n=req.num_classes, c=n_classes
            )
        )

    trainer = PyTorchTrainer(device=req.device)
    raw = torch.load(req.model_path, map_location=trainer.device)
    state_dict = extract_pytorch_state_dict(raw)
    model = trainer.create_model(req.architecture, n_classes, pretrained=False)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    running_loss = 0.0
    n_seen = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(trainer.device)
            if not torch.is_tensor(targets):
                targets = torch.as_tensor(targets)
            targets = targets.to(trainer.device)
            logits = model(inputs)
            loss = criterion(logits, targets)
            bs = int(targets.size(0))
            running_loss += float(loss.item()) * bs
            n_seen += bs
            pred = logits.argmax(dim=1)
            for t, p in zip(targets.view(-1).cpu().numpy(), pred.view(-1).cpu().numpy()):
                cm[int(t), int(p)] += 1

    correct = int(np.trace(cm))
    accuracy_percent = 100.0 * correct / max(n_seen, 1)
    avg_loss = running_loss / max(n_seen, 1)
    per_class_total = [int(cm[i].sum()) for i in range(n_classes)]
    per_class_correct = [int(cm[i, i]) for i in range(n_classes)]

    return ClassificationMetricsReport(
        model_type=ModelType.IMAGE_CLASSIFICATION,
        framework=FrameworkType.PYTORCH,
        model_path=req.model_path,
        data_dir=req.data_dir,
        n_samples=n_seen,
        class_names=class_names,
        accuracy_percent=accuracy_percent,
        avg_loss=avg_loss,
        per_class_correct=per_class_correct,
        per_class_total=per_class_total,
        confusion_matrix=cm.tolist(),
    )


def run_image_classification_metrics_keras(req: MetricsRequest) -> ClassificationMetricsReport:
    try:
        from tensorflow import keras
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
    except ImportError as e:
        raise RuntimeError(_("TensorFlow is required for Keras metrics: {err}").format(err=e)) from e

    val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    val_gen = val_datagen.flow_from_directory(
        directory=str(req.data_dir),
        target_size=(req.image_size, req.image_size),
        batch_size=req.batch_size,
        class_mode="categorical",
        shuffle=False,
    )
    if val_gen.samples == 0:
        raise ValueError(_("No images found under {path}").format(path=req.data_dir))

    class_names = sorted(val_gen.class_indices.keys(), key=lambda n: val_gen.class_indices[n])
    n_classes = len(class_names)
    if req.num_classes is not None and int(req.num_classes) != n_classes:
        raise ValueError(
            _("--num-classes ({n}) does not match directory class count ({c}).").format(
                n=req.num_classes, c=n_classes
            )
        )

    model = keras.models.load_model(str(req.model_path))
    steps = int(np.ceil(val_gen.samples / req.batch_size))
    prob = model.predict(val_gen, steps=steps, verbose=0)
    prob = prob[: val_gen.samples]
    y_pred = np.argmax(prob, axis=1)
    y_true = np.asarray(val_gen.classes, dtype=np.int64)[: val_gen.samples]

    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1

    correct = int(np.trace(cm))
    n_seen = int(val_gen.samples)
    accuracy_percent = 100.0 * correct / max(n_seen, 1)
    per_class_total = [int(cm[i].sum()) for i in range(n_classes)]
    per_class_correct = [int(cm[i, i]) for i in range(n_classes)]

    return ClassificationMetricsReport(
        model_type=ModelType.IMAGE_CLASSIFICATION,
        framework=FrameworkType.KERAS,
        model_path=req.model_path,
        data_dir=req.data_dir,
        n_samples=n_seen,
        class_names=class_names,
        accuracy_percent=accuracy_percent,
        avg_loss=None,
        per_class_correct=per_class_correct,
        per_class_total=per_class_total,
        confusion_matrix=cm.tolist(),
    )


def run_image_classification_metrics(req: MetricsRequest) -> ClassificationMetricsReport:
    """Dispatch image-classification metrics for the resolved framework."""
    fw = _resolve_framework(req.model_path, req.framework)
    if fw == FrameworkType.PYTORCH:
        return run_image_classification_metrics_pytorch(req)
    if fw == FrameworkType.KERAS:
        return run_image_classification_metrics_keras(req)
    raise ValueError(_("Unsupported framework for metrics: {fw}").format(fw=fw.value))
