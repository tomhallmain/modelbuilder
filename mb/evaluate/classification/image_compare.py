"""
Paired image-classification comparison (PyTorch and Keras).

Runs both checkpoints on the same ImageFolder ordering as training / metrics so
contingency counts (who is right when) and prediction disagreements are meaningful.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mb.evaluate._contracts import (
    ClassificationCompareReport,
    CompareRequest,
    DisagreementSample,
)
from mb.evaluate._weights import extract_pytorch_state_dict
from mb.evaluate.classification.image_metrics import _build_imagefolder_torch, _resolve_framework
from mb.models.types import FrameworkType, ModelType
from mb.utils.translations import _


def _require_same_framework(fw_a: FrameworkType, fw_b: FrameworkType) -> None:
    if fw_a != fw_b:
        raise ValueError(
            _(
                "Compare requires both checkpoints to use the same framework "
                "(got {a} vs {b}). Mixed PyTorch / Keras is not supported yet."
            ).format(a=fw_a.value, b=fw_b.value)
        )


def run_image_classification_compare_pytorch(req: CompareRequest) -> ClassificationCompareReport:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from mb.models.frameworks.pytorch.trainer import PyTorchTrainer

    arch_a = req.architecture
    arch_b = req.architecture_b or req.architecture
    if not arch_a:
        raise ValueError(_("--architecture is required for PyTorch compare (model A)."))
    if not arch_b:
        raise ValueError(
            _("Model B needs --architecture-b or the same --architecture when B is PyTorch.")
        )

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
    device = trainer.device

    raw_a = torch.load(req.model_path_a, map_location=device)
    state_a = extract_pytorch_state_dict(raw_a)
    model_a = trainer.create_model(arch_a, n_classes, pretrained=False)
    model_a.load_state_dict(state_a, strict=True)
    model_a.eval()

    raw_b = torch.load(req.model_path_b, map_location=device)
    state_b = extract_pytorch_state_dict(raw_b)
    model_b = trainer.create_model(arch_b, n_classes, pretrained=False)
    model_b.load_state_dict(state_b, strict=True)
    model_b.eval()

    criterion = nn.CrossEntropyLoss(reduction="sum")
    n_scanned = len(ds)
    loss_sum_a = 0.0
    loss_sum_b = 0.0
    both_correct = only_a_correct = only_b_correct = both_wrong = 0
    pred_disagreement = 0
    disagreement_samples: list[DisagreementSample] = []
    cap = req.max_disagreement_report

    with torch.no_grad():
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

            x = torch.stack(xs, dim=0).to(device)
            targets = torch.as_tensor(ys, dtype=torch.long, device=device)

            logits_a = model_a(x)
            logits_b = model_b(x)
            loss_sum_a += float(criterion(logits_a, targets).item())
            loss_sum_b += float(criterion(logits_b, targets).item())

            prob_a = F.softmax(logits_a, dim=1)
            prob_b = F.softmax(logits_b, dim=1)
            conf_a, pred_a = prob_a.max(dim=1)
            conf_b, pred_b = prob_b.max(dim=1)

            pa = pred_a.detach().cpu().numpy().astype(np.int64)
            pb = pred_b.detach().cpu().numpy().astype(np.int64)
            ca = conf_a.detach().cpu().numpy()
            cb = conf_b.detach().cpu().numpy()
            y_np = np.asarray(ys, dtype=np.int64)

            for pth, yi, pia, pib, cfa, cfb in zip(batch_paths, y_np, pa, pb, ca, cb):
                ok_a = int(pia) == int(yi)
                ok_b = int(pib) == int(yi)
                if ok_a and ok_b:
                    both_correct += 1
                elif ok_a and not ok_b:
                    only_a_correct += 1
                elif not ok_a and ok_b:
                    only_b_correct += 1
                else:
                    both_wrong += 1

                if int(pia) != int(pib):
                    pred_disagreement += 1
                    if cap is None or len(disagreement_samples) < cap:
                        disagreement_samples.append(
                            DisagreementSample(
                                path=pth,
                                true_label=class_names[int(yi)],
                                predicted_a=class_names[int(pia)],
                                predicted_b=class_names[int(pib)],
                                confidence_a=float(cfa),
                                confidence_b=float(cfb),
                            )
                        )

    n = max(n_scanned, 1)
    correct_a = both_correct + only_a_correct
    correct_b = both_correct + only_b_correct
    return ClassificationCompareReport(
        model_type=ModelType.IMAGE_CLASSIFICATION,
        framework_a=FrameworkType.PYTORCH,
        framework_b=FrameworkType.PYTORCH,
        model_path_a=req.model_path_a,
        model_path_b=req.model_path_b,
        data_dir=req.data_dir,
        n_samples=n_scanned,
        class_names=class_names,
        accuracy_a_percent=100.0 * correct_a / n,
        accuracy_b_percent=100.0 * correct_b / n,
        avg_loss_a=loss_sum_a / n,
        avg_loss_b=loss_sum_b / n,
        both_correct=both_correct,
        only_a_correct=only_a_correct,
        only_b_correct=only_b_correct,
        both_wrong=both_wrong,
        pred_disagreement=pred_disagreement,
        disagreement_samples=disagreement_samples,
    )


def run_image_classification_compare_keras(req: CompareRequest) -> ClassificationCompareReport:
    try:
        from tensorflow import keras
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
    except ImportError as e:
        raise RuntimeError(_("TensorFlow is required for Keras compare: {err}").format(err=e)) from e

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

    model_a = keras.models.load_model(str(req.model_path_a))
    model_b = keras.models.load_model(str(req.model_path_b))

    n_scanned = int(val_gen.samples)
    steps = int(np.ceil(n_scanned / req.batch_size))

    both_correct = only_a_correct = only_b_correct = both_wrong = 0
    pred_disagreement = 0
    disagreement_samples: list[DisagreementSample] = []
    cap = req.max_disagreement_report
    offset = 0

    val_gen.reset()
    for _ in range(steps):
        x_batch, y_batch = next(val_gen)
        bs = len(x_batch)
        y_true = np.argmax(y_batch, axis=1).astype(np.int64)

        prob_a = model_a.predict_on_batch(x_batch)
        prob_b = model_b.predict_on_batch(x_batch)
        pred_a = np.argmax(prob_a, axis=1).astype(np.int64)
        pred_b = np.argmax(prob_b, axis=1).astype(np.int64)

        paths_batch = [str(Path(p).resolve()) for p in val_gen.filepaths[offset : offset + bs]]
        offset += bs

        for j, (pth, yi, pia, pib) in enumerate(zip(paths_batch, y_true, pred_a, pred_b)):
            ok_a = int(pia) == int(yi)
            ok_b = int(pib) == int(yi)
            if ok_a and ok_b:
                both_correct += 1
            elif ok_a and not ok_b:
                only_a_correct += 1
            elif not ok_a and ok_b:
                only_b_correct += 1
            else:
                both_wrong += 1

            if int(pia) != int(pib):
                pred_disagreement += 1
                if cap is None or len(disagreement_samples) < cap:
                    disagreement_samples.append(
                        DisagreementSample(
                            path=pth,
                            true_label=class_names[int(yi)],
                            predicted_a=class_names[int(pia)],
                            predicted_b=class_names[int(pib)],
                            confidence_a=float(prob_a[j, int(pia)]),
                            confidence_b=float(prob_b[j, int(pib)]),
                        )
                    )

    n = max(n_scanned, 1)
    correct_a = both_correct + only_a_correct
    correct_b = both_correct + only_b_correct
    return ClassificationCompareReport(
        model_type=ModelType.IMAGE_CLASSIFICATION,
        framework_a=FrameworkType.KERAS,
        framework_b=FrameworkType.KERAS,
        model_path_a=req.model_path_a,
        model_path_b=req.model_path_b,
        data_dir=req.data_dir,
        n_samples=n_scanned,
        class_names=class_names,
        accuracy_a_percent=100.0 * correct_a / n,
        accuracy_b_percent=100.0 * correct_b / n,
        avg_loss_a=None,
        avg_loss_b=None,
        both_correct=both_correct,
        only_a_correct=only_a_correct,
        only_b_correct=only_b_correct,
        both_wrong=both_wrong,
        pred_disagreement=pred_disagreement,
        disagreement_samples=disagreement_samples,
    )


def run_image_classification_compare(req: CompareRequest) -> ClassificationCompareReport:
    """Paired compare for image classification; both checkpoints must share a framework."""
    fw_a = _resolve_framework(req.model_path_a, req.framework_a)
    fw_b = _resolve_framework(req.model_path_b, req.framework_b)
    _require_same_framework(fw_a, fw_b)
    if fw_a == FrameworkType.PYTORCH:
        return run_image_classification_compare_pytorch(req)
    if fw_a == FrameworkType.KERAS:
        return run_image_classification_compare_keras(req)
    raise ValueError(_("Unsupported framework for compare: {fw}").format(fw=fw_a.value))
