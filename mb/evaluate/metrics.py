"""
``mb evaluate metrics`` — public entry points and CLI adapter.

Model-type dispatch lives in :func:`run_metrics`; image-classification backends live under
``mb.evaluate.classification``.
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from mb.evaluate._contracts import ClassificationMetricsReport, MetricsRequest
from mb.models.types import FrameworkType, ModelType
from mb.utils.logging_setup import get_logger
from mb.utils.translations import _

logger = get_logger(__name__)


def run_metrics(request: MetricsRequest) -> ClassificationMetricsReport:
    """
    Run dataset-level metrics for the given :attr:`MetricsRequest.model_type`.

    Today only :class:`~mb.models.types.ModelType.IMAGE_CLASSIFICATION` is implemented.
    """
    mt = request.model_type
    if mt == ModelType.IMAGE_CLASSIFICATION:
        from mb.evaluate.classification.image_metrics import run_image_classification_metrics

        return run_image_classification_metrics(request)
    if mt == ModelType.OBJECT_DETECTION:
        raise NotImplementedError(_("Object detection metrics are not implemented yet."))
    raise ValueError(_("Unsupported model type for metrics: {t}").format(t=mt.value))


def format_classification_report(report: ClassificationMetricsReport) -> str:
    """Human-readable block for stdout or logs."""
    lines: list[str] = [
        _("Model: {path}").format(path=report.model_path),
        _("Data: {path}").format(path=report.data_dir),
        _("Framework: {fw}").format(fw=report.framework.value),
        _("Samples: {n}").format(n=report.n_samples),
        _("Top-1 accuracy: {acc:.2f}%").format(acc=report.accuracy_percent),
    ]
    if report.avg_loss is not None:
        lines.append(_("Average loss: {loss:.4f}").format(loss=report.avg_loss))
    lines.append("")
    lines.append(_("Per-class correct / total:"))
    for name, c, t in zip(report.class_names, report.per_class_correct, report.per_class_total):
        pct = 100.0 * c / max(t, 1)
        lines.append(f"  {name}: {c}/{t} ({pct:.1f}%)")
    lines.append("")
    lines.append(_("Confusion matrix (rows=true class, cols=predicted):"))
    header = " " * 12 + " ".join(f"{n[:8]:>8}" for n in report.class_names)
    lines.append(header)
    for i, row in enumerate(report.confusion_matrix):
        rn = report.class_names[i][:8]
        lines.append(f"{rn:12}" + " ".join(f"{v:8d}" for v in row))
    return "\n".join(lines)


def build_metrics_request(args: Namespace) -> MetricsRequest:
    """Build a :class:`MetricsRequest` from argparse ``metrics`` subcommand namespace."""
    return MetricsRequest(
        model_path=Path(args.model).expanduser().resolve(),
        data_dir=Path(args.data_dir).expanduser().resolve(),
        model_type=ModelType.from_pipeline_value(getattr(args, "model_type", None)),
        framework=FrameworkType.try_from(args.framework) if getattr(args, "framework", None) else None,
        architecture=getattr(args, "architecture", None),
        num_classes=getattr(args, "num_classes", None),
        image_size=int(args.image_size),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        device=getattr(args, "device", None),
    )


def run_evaluate_metrics_cli(args: Namespace) -> int:
    """CLI implementation for ``mb evaluate metrics`` (returns process exit code)."""
    if getattr(args, "dry_run", False):
        if not Path(args.model).expanduser().resolve().is_file():
            logger.error(_("Model file not found: {path}").format(path=args.model))
            return 1
        data_dir = Path(args.data_dir).expanduser().resolve()
        if not data_dir.is_dir():
            logger.error(_("Data directory not found: {path}").format(path=data_dir))
            return 1
        from mb.conversion.converters import detect_model_framework

        fw = detect_model_framework(Path(args.model).expanduser().resolve())
        if fw == "pytorch" and not getattr(args, "architecture", None):
            logger.error(_("--architecture is required for PyTorch models in dry-run validation."))
            return 1
        logger.info(
            _("Dry-run OK: would score model {model} on {data} (framework={fw})").format(
                model=args.model, data=args.data_dir, fw=fw or "?"
            )
        )
        return 0

    req = build_metrics_request(args)
    if not req.model_path.is_file():
        logger.error(_("Model file not found: {path}").format(path=req.model_path))
        return 1
    if not req.data_dir.is_dir():
        logger.error(_("Data directory not found: {path}").format(path=req.data_dir))
        return 1

    try:
        report = run_metrics(req)
    except NotImplementedError as e:
        logger.error(str(e))
        return 1
    except (RuntimeError, ValueError) as e:
        logger.error(str(e))
        return 1
    except ImportError as e:
        logger.error(_("Import error during evaluation: {err}").format(err=e))
        return 1
    except Exception as e:
        logger.error(_("Evaluation failed: {err}").format(err=e), exc_info=args.verbose)
        return 1

    print(format_classification_report(report))
    logger.debug(json.dumps(report.to_jsonable()))
    return 0
