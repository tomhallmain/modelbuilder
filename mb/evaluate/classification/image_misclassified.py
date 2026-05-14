"""
Image-classification misclassified listing (PyTorch and Keras).

Reuses loader and framework resolution from :mod:`mb.evaluate.classification.image_metrics`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from mb.evaluate._contracts import MisclassifiedListing, MisclassifiedSample, MetricsRequest
from mb.evaluate._weights import extract_pytorch_state_dict
from mb.evaluate.classification.image_metrics import _build_imagefolder_torch, _resolve_framework
from mb.models.types import FrameworkType, ModelType
from mb.utils.translations import _


def run_image_classification_misclassified_pytorch(
    req: MetricsRequest, *, max_report: Optional[int] = None
) -> MisclassifiedListing:
    import torch
    import torch.nn.functional as F

    from mb.models.frameworks.pytorch.trainer import PyTorchTrainer

    if not req.architecture:
        raise ValueError(_("--architecture is required for PyTorch misclassified listing."))

    ds, _ = _build_imagefolder_torch(
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

    rows: list[MisclassifiedSample] = []
    n_misclassified = 0
    n_scanned = len(ds)

    for start in range(0, n_scanned, req.batch_size):
        end = min(start + req.batch_size, n_scanned)
        batch_paths: list[str] = []
        xs: list[torch.Tensor] = []
        ys: list[int] = []
        for idx in range(start, end):
            path, y_int = ds.samples[idx]
            img, _y = ds[idx]
            xs.append(img)
            ys.append(int(y_int))
            batch_paths.append(str(Path(path).resolve()))

        x = torch.stack(xs, dim=0).to(trainer.device)
        with torch.no_grad():
            logits = model(x)
            prob = F.softmax(logits, dim=1)
            conf, pred = prob.max(dim=1)

        pred_np = pred.detach().cpu().numpy()
        conf_np = conf.detach().cpu().numpy()
        for pth, yi, pi, cconf in zip(batch_paths, ys, pred_np.tolist(), conf_np.tolist()):
            if int(pi) != int(yi):
                n_misclassified += 1
                if max_report is None or len(rows) < max_report:
                    rows.append(
                        MisclassifiedSample(
                            path=pth,
                            true_label=class_names[int(yi)],
                            predicted_label=class_names[int(pi)],
                            confidence=float(cconf),
                        )
                    )

    return MisclassifiedListing(
        model_type=ModelType.IMAGE_CLASSIFICATION,
        framework=FrameworkType.PYTORCH,
        model_path=req.model_path,
        data_dir=req.data_dir,
        n_scanned=n_scanned,
        n_misclassified=n_misclassified,
        samples=rows,
    )


def run_image_classification_misclassified_keras(
    req: MetricsRequest, *, max_report: Optional[int] = None
) -> MisclassifiedListing:
    try:
        from tensorflow import keras
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
    except ImportError as e:
        raise RuntimeError(_("TensorFlow is required for Keras misclassified listing: {err}").format(err=e)) from e

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
    paths = list(val_gen.filepaths)[: val_gen.samples]

    rows: list[MisclassifiedSample] = []
    n_misclassified = 0
    for i, (pth, yi, pi) in enumerate(zip(paths, y_true, y_pred)):
        if int(pi) != int(yi):
            n_misclassified += 1
            if max_report is None or len(rows) < max_report:
                rows.append(
                    MisclassifiedSample(
                        path=str(Path(pth).resolve()),
                        true_label=class_names[int(yi)],
                        predicted_label=class_names[int(pi)],
                        confidence=float(prob[i, int(pi)]),
                    )
                )

    return MisclassifiedListing(
        model_type=ModelType.IMAGE_CLASSIFICATION,
        framework=FrameworkType.KERAS,
        model_path=req.model_path,
        data_dir=req.data_dir,
        n_scanned=int(val_gen.samples),
        n_misclassified=n_misclassified,
        samples=rows,
    )


def run_image_classification_misclassified(
    req: MetricsRequest, *, max_report: Optional[int] = None
) -> MisclassifiedListing:
    """Dispatch image-classification misclassified listing for the resolved framework."""
    fw = _resolve_framework(req.model_path, req.framework)
    if fw == FrameworkType.PYTORCH:
        return run_image_classification_misclassified_pytorch(req, max_report=max_report)
    if fw == FrameworkType.KERAS:
        return run_image_classification_misclassified_keras(req, max_report=max_report)
    raise ValueError(_("Unsupported framework for misclassified: {fw}").format(fw=fw.value))
