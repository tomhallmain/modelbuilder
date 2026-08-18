"""
Multi-label image-classification metrics (PyTorch).

Uses the same folder layout, transforms, and label schema as training, so a score here is
measured on the input distribution the model was fitted on.

Scores are collected once and thresholded afterwards. That makes the optional threshold
sweep nearly free — it re-thresholds cached scores rather than re-running inference per
candidate — and guarantees the reported metrics and the tuned thresholds come from exactly
the same predictions.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from mb.data.label_schema import (
    LabelSchema,
    load_label_manifest,
    load_label_schema,
    write_label_schema,
)
from mb.evaluate._contracts import (
    MetricsRequest,
    MultiLabelClassificationMetricsReport,
    PerLabelMetrics,
)
from mb.evaluate._multilabel import MultiLabelCounter, f1_from_precision_recall
from mb.evaluate._weights import extract_pytorch_state_dict
from mb.models.preprocessing import preprocessing_spec_for
from mb.models.types import FrameworkType, ModelType
from mb.utils.logging_setup import get_logger
from mb.utils.translations import _

logger = get_logger(__name__)

# Candidate thresholds for the sweep: 0.05 to 0.95 in steps of 0.05. Finer steps chase
# noise on a small evaluation split without changing the decision in any useful way.
_SWEEP_STEPS = [round(0.05 * i, 2) for i in range(1, 20)]


def _collect_scores(
    req: MetricsRequest,
    schema: LabelSchema,
) -> Tuple[List[List[float]], List[List[bool]], float, int]:
    """Run inference once, returning per-sample scores, truths, mean loss, and sample count."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    from mb.data.file_types import configured_media_suffixes
    from mb.models.frameworks.pytorch.data_loader import (
        MultiLabelImageFolderDataset,
        get_val_transforms,
    )
    from mb.models.frameworks.pytorch.trainer import PyTorchTrainer

    manifest = load_label_manifest(req.data_dir.parent, schema)
    spec = preprocessing_spec_for(req.architecture, req.image_size)
    dataset = MultiLabelImageFolderDataset(
        root=req.data_dir,
        schema=schema,
        manifest=manifest,
        manifest_root=req.data_dir.parent,
        transform=get_val_transforms(req.image_size, preprocessing=spec),
        extensions=tuple(sorted(configured_media_suffixes())),
    )
    if len(dataset) == 0:
        raise ValueError(_("No images found under {path}").format(path=req.data_dir))

    loader = DataLoader(
        dataset,
        batch_size=req.batch_size,
        shuffle=False,
        num_workers=req.num_workers,
        pin_memory=False,
        drop_last=False,
    )

    trainer = PyTorchTrainer(device=req.device)
    raw = torch.load(req.model_path, map_location=trainer.device)
    state_dict = extract_pytorch_state_dict(raw)
    model = trainer.create_model(req.architecture, schema.num_labels, pretrained=False)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    criterion = nn.BCEWithLogitsLoss()
    scores: List[List[float]] = []
    truths: List[List[bool]] = []
    running_loss = 0.0
    n_seen = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(trainer.device)
            targets = targets.to(trainer.device)
            logits = model(inputs)
            loss = criterion(logits, targets)
            batch = int(targets.size(0))
            running_loss += float(loss.item()) * batch
            n_seen += batch
            scores.extend(torch.sigmoid(logits).cpu().tolist())
            truths.extend((targets > 0.5).cpu().tolist())

    return scores, truths, running_loss / max(n_seen, 1), n_seen


def _count_at_thresholds(
    scores: Sequence[Sequence[float]],
    truths: Sequence[Sequence[bool]],
    thresholds: Sequence[float],
) -> MultiLabelCounter:
    """Fold cached scores into per-label counts at the given thresholds."""
    counter = MultiLabelCounter(len(thresholds))
    for score_row, true_row in zip(scores, truths):
        predicted = [score >= thresholds[i] for i, score in enumerate(score_row)]
        counter.add_batch([predicted], [true_row])
    return counter


def tune_thresholds(
    scores: Sequence[Sequence[float]],
    truths: Sequence[Sequence[bool]],
    schema: LabelSchema,
) -> List[float]:
    """
    Per-label thresholds maximizing that label's F1 on this split.

    Each label is tuned independently, which is valid because the labels are independent
    predictions. A label with no positives keeps its configured threshold: there is nothing
    to maximize, and moving it would be fitting noise.
    """
    tuned: List[float] = []
    for index, name in enumerate(schema.labels):
        column = [row[index] for row in scores]
        actual = [row[index] for row in truths]
        if not any(actual):
            tuned.append(schema.threshold_for(name))
            continue

        best_threshold = schema.threshold_for(name)
        best_f1 = -1.0
        for candidate in _SWEEP_STEPS:
            tp = fp = fn = 0
            for score, truth in zip(column, actual):
                predicted = score >= candidate
                if predicted and truth:
                    tp += 1
                elif predicted:
                    fp += 1
                elif truth:
                    fn += 1
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = f1_from_precision_recall(precision, recall)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = candidate
        logger.info("Tuned %s -> %.2f (F1 %.4f)", name, best_threshold, best_f1)
        tuned.append(best_threshold)
    return tuned


def _build_report(
    req: MetricsRequest,
    schema: LabelSchema,
    counter: MultiLabelCounter,
    thresholds: Sequence[float],
    avg_loss: Optional[float],
    n_samples: int,
) -> MultiLabelClassificationMetricsReport:
    per_label = [
        PerLabelMetrics(
            name=name,
            support=counts.support,
            predicted=counts.predicted,
            true_positives=counts.true_positives,
            false_positives=counts.false_positives,
            false_negatives=counts.false_negatives,
            true_negatives=counts.true_negatives,
            precision=counts.precision,
            recall=counts.recall,
            f1=counts.f1,
            threshold=float(thresholds[index]),
        )
        for index, (name, counts) in enumerate(zip(schema.labels, counter.counts))
    ]

    axis_scores = {}
    for axis_name, axis in schema.axes.items():
        value = counter.axis_macro_f1(schema.labels, axis.labels)
        if value is not None:
            axis_scores[axis_name] = value

    return MultiLabelClassificationMetricsReport(
        model_type=ModelType.IMAGE_CLASSIFICATION,
        framework=FrameworkType.PYTORCH,
        model_path=req.model_path,
        data_dir=req.data_dir,
        n_samples=n_samples,
        label_names=list(schema.labels),
        micro_f1=counter.micro_f1(),
        macro_f1=counter.macro_f1(),
        avg_loss=avg_loss,
        per_label=per_label,
        axis_macro_f1=axis_scores,
        n_labels_with_support=counter.scored_label_count(),
    )


def run_multilabel_metrics(req: MetricsRequest) -> MultiLabelClassificationMetricsReport:
    """
    Score a multi-label checkpoint, optionally retuning the schema's thresholds first.

    Tuning writes back to ``label_schema.json``. It is a post-hoc calibration step on a
    trained model and must never run as part of training.
    """
    if not req.architecture:
        raise ValueError(_("--architecture is required for PyTorch metrics evaluation."))

    # The schema sits beside the splits, so it is one level up from the split being scored.
    schema = load_label_schema(req.data_dir.parent)
    scores, truths, avg_loss, n_samples = _collect_scores(req, schema)

    thresholds = schema.thresholds()
    if req.tune_thresholds:
        thresholds = tune_thresholds(scores, truths, schema)
        updated = LabelSchema(
            labels=list(schema.labels),
            axes=dict(schema.axes),
            default_thresholds={
                name: float(value) for name, value in zip(schema.labels, thresholds)
            },
            default_threshold=schema.default_threshold,
            schema_version=schema.schema_version,
        )
        write_label_schema(req.data_dir.parent, updated)
        schema = updated

    counter = _count_at_thresholds(scores, truths, thresholds)
    return _build_report(req, schema, counter, thresholds, avg_loss, n_samples)


def format_multilabel_report(report: MultiLabelClassificationMetricsReport) -> str:
    """Human-readable block for stdout or logs."""
    lines: List[str] = [
        _("Model: {path}").format(path=report.model_path),
        _("Data: {path}").format(path=report.data_dir),
        _("Framework: {fw}").format(fw=report.framework.value),
        _("Samples: {n}").format(n=report.n_samples),
        _("Micro F1: {v:.4f}").format(v=report.micro_f1),
        _("Macro F1: {v:.4f}").format(v=report.macro_f1),
    ]
    if report.avg_loss is not None:
        lines.append(_("Average loss: {loss:.4f}").format(loss=report.avg_loss))
    lines.append("")
    lines.append(_("Per-label metrics:"))
    header = (
        f"{_('label'):<16}{_('precision'):>10}{_('recall'):>10}{_('f1'):>10}"
        f"{_('support'):>10}{_('predicted'):>11}{_('threshold'):>11}"
    )
    lines.append(header)
    for m in report.per_label:
        lines.append(
            f"{m.name[:16]:<16}{m.precision:>10.4f}{m.recall:>10.4f}{m.f1:>10.4f}"
            f"{m.support:>10d}{m.predicted:>11d}{m.threshold:>11.2f}"
        )
    if report.axis_macro_f1:
        lines.append("")
        lines.append(_("Macro F1 by axis:"))
        for axis_name, value in report.axis_macro_f1.items():
            lines.append(f"  {axis_name}: {value:.4f}")
    if report.n_labels_with_support != len(report.label_names):
        lines.append("")
        lines.append(
            _("Macro averages exclude {n} label(s) with no positives in this split.").format(
                n=len(report.label_names) - report.n_labels_with_support
            )
        )
    return "\n".join(lines)
