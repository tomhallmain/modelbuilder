"""
``mb evaluate misclassified`` — list samples whose predicted class differs from the folder label.

Only :class:`~mb.models.types.ModelType.IMAGE_CLASSIFICATION` is implemented; other model
types raise :class:`NotImplementedError`. Load or shape failures are reported with guidance
that the checkpoint may not match this subcommand or the dataset layout.
"""

from __future__ import annotations

import csv
import logging
from argparse import Namespace
from pathlib import Path
from typing import Optional

from mb.evaluate._contracts import MisclassifiedListing, MetricsRequest
from mb.evaluate.metrics import build_metrics_request
from mb.models.types import EvaluateSubcommand, ModelType
from mb.utils.translations import _

logger = logging.getLogger(__name__)


def run_misclassified(
    request: MetricsRequest, *, max_report: Optional[int] = None
) -> MisclassifiedListing:
    """
    Run misclassified listing for the given :attr:`MetricsRequest.model_type`.

    Today only :class:`~mb.models.types.ModelType.IMAGE_CLASSIFICATION` is implemented.
    """
    mt = request.model_type
    if mt == ModelType.IMAGE_CLASSIFICATION:
        from mb.evaluate.classification.image_misclassified import (
            run_image_classification_misclassified,
        )

        return run_image_classification_misclassified(request, max_report=max_report)
    if mt == ModelType.OBJECT_DETECTION:
        raise NotImplementedError(_("Object detection misclassified listing is not implemented yet."))
    raise ValueError(_("Unsupported model type for misclassified: {t}").format(t=mt.value))


def format_misclassified_listing(listing: MisclassifiedListing) -> str:
    """Human-readable summary and TSV body (path, true_label, predicted_label, confidence)."""
    lines: list[str] = [
        _("Model: {path}").format(path=listing.model_path),
        _("Data: {path}").format(path=listing.data_dir),
        _("Framework: {fw}").format(fw=listing.framework.value),
        _("Scanned images: {n}").format(n=listing.n_scanned),
        _("Misclassified (total): {m}").format(m=listing.n_misclassified),
        _("Rows printed: {k}").format(k=len(listing.samples)),
    ]
    if len(listing.samples) < listing.n_misclassified:
        lines.append(
            _("(Increase --max-report to print more rows; totals above are for the full split.)")
        )
    lines.append("")
    lines.append("path\ttrue_label\tpredicted_label\tconfidence")
    for s in listing.samples:
        lines.append(f"{s.path}\t{s.true_label}\t{s.predicted_label}\t{s.confidence:.6f}")
    return "\n".join(lines)


def _write_misclassified_csv(path: Path, listing: MisclassifiedListing) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "true_label", "predicted_label", "confidence"])
        for s in listing.samples:
            w.writerow([s.path, s.true_label, s.predicted_label, f"{s.confidence:.6f}"])


def run_evaluate_misclassified_cli(args: Namespace) -> int:
    """CLI implementation for ``mb evaluate misclassified`` (returns process exit code)."""
    sub = EvaluateSubcommand.MISCLASSIFIED.value
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
            _("Dry-run OK: would list misclassified for model {model} on {data} (framework={fw})").format(
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

    max_report = getattr(args, "max_report", None)
    out_path = getattr(args, "output", None)
    if max_report is not None and max_report < 1:
        logger.error(_("--max-report must be at least 1 when provided."))
        return 1

    hint = _(
        "Misclassified evaluation failed for model type {mt} (subcommand {sub}). "
        "This flow only supports image classification checkpoints that match the dataset layout, "
        "--architecture (PyTorch), and class counts. Underlying error: {err}"
    )

    try:
        listing = run_misclassified(req, max_report=max_report)
    except NotImplementedError as e:
        logger.error(str(e))
        return 1
    except (RuntimeError, ValueError) as e:
        logger.error(
            hint.format(mt=req.model_type.value, sub=sub, err=e),
            exc_info=getattr(args, "verbose", False),
        )
        return 1
    except ImportError as e:
        logger.error(_("Import error during misclassified evaluation: {err}").format(err=e))
        return 1
    except Exception as e:
        logger.error(
            hint.format(mt=req.model_type.value, sub=sub, err=e),
            exc_info=getattr(args, "verbose", False),
        )
        return 1

    if out_path:
        _write_misclassified_csv(Path(out_path).expanduser().resolve(), listing)
        logger.info(_("Wrote {n} rows to {path}").format(n=len(listing.samples), path=out_path))
    print(format_misclassified_listing(listing))
    return 0
