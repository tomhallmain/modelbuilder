"""
``mb evaluate compare`` — paired checkpoints on one split.

Contingency counts (both correct, swap wins, both wrong) and raw prediction disagreements
are computed in one pass so upgrade / regression stories stay tied to the same pixels.
"""

from __future__ import annotations

import csv
import json
import logging
from argparse import Namespace
from pathlib import Path
from mb.evaluate._contracts import ClassificationCompareReport, CompareRequest
from mb.models.types import EvaluateSubcommand, FrameworkType, ModelType
from mb.utils.translations import _

logger = logging.getLogger(__name__)


def run_compare(request: CompareRequest) -> ClassificationCompareReport:
    """
    Run paired evaluation for the given :attr:`CompareRequest.model_type`.

    Today only :class:`~mb.models.types.ModelType.IMAGE_CLASSIFICATION` is implemented.
    """
    mt = request.model_type
    if mt == ModelType.IMAGE_CLASSIFICATION:
        from mb.evaluate.classification.image_compare import run_image_classification_compare

        return run_image_classification_compare(request)
    if mt == ModelType.OBJECT_DETECTION:
        raise NotImplementedError(_("Object detection compare is not implemented yet."))
    raise ValueError(_("Unsupported model type for compare: {t}").format(t=mt.value))


def format_classification_compare_report(report: ClassificationCompareReport) -> str:
    """Human-readable summary, contingency table, and optional disagreement rows."""
    n = max(report.n_samples, 1)
    lines: list[str] = [
        _("Model A: {path} ({fw})").format(path=report.model_path_a, fw=report.framework_a.value),
        _("Model B: {path} ({fw})").format(path=report.model_path_b, fw=report.framework_b.value),
        _("Data: {path}").format(path=report.data_dir),
        _("Samples: {n}").format(n=report.n_samples),
        _("Top-1 accuracy A: {acc:.2f}%").format(acc=report.accuracy_a_percent),
        _("Top-1 accuracy B: {acc:.2f}%").format(acc=report.accuracy_b_percent),
    ]
    if report.avg_loss_a is not None:
        lines.append(_("Average loss A: {loss:.4f}").format(loss=report.avg_loss_a))
    if report.avg_loss_b is not None:
        lines.append(_("Average loss B: {loss:.4f}").format(loss=report.avg_loss_b))
    lines.append("")
    lines.append(_("Paired outcomes vs folder label (each row is one image):"))
    lines.append(
        _("  both correct: {n} ({pct:.1f}%)").format(n=report.both_correct, pct=100.0 * report.both_correct / n)
    )
    lines.append(
        _("  only A correct: {n} ({pct:.1f}%)").format(
            n=report.only_a_correct, pct=100.0 * report.only_a_correct / n
        )
    )
    lines.append(
        _("  only B correct: {n} ({pct:.1f}%)").format(
            n=report.only_b_correct, pct=100.0 * report.only_b_correct / n
        )
    )
    lines.append(
        _("  both wrong: {n} ({pct:.1f}%)").format(n=report.both_wrong, pct=100.0 * report.both_wrong / n)
    )
    lines.append("")
    lines.append(
        _("Prediction disagreement (A class ≠ B class): {n} ({pct:.1f}%)").format(
            n=report.pred_disagreement, pct=100.0 * report.pred_disagreement / n
        )
    )
    lines.append(
        _(
            "Note: disagreement counts images where A and B pick different classes; "
            "when both match the folder label, that still counts as both correct above."
        )
    )
    lines.append("")
    lines.append(_("Disagreement rows printed: {k}").format(k=len(report.disagreement_samples)))
    if report.pred_disagreement > len(report.disagreement_samples):
        lines.append(
            _("(Use --max-disagreement-report to print more; counts above are for the full split.)")
        )
    lines.append("")
    lines.append(
        "path\ttrue_label\tpred_a\tpred_b\tconf_a\tconf_b"
    )
    for s in report.disagreement_samples:
        lines.append(
            f"{s.path}\t{s.true_label}\t{s.predicted_a}\t{s.predicted_b}\t"
            f"{s.confidence_a:.6f}\t{s.confidence_b:.6f}"
        )
    return "\n".join(lines)


def build_compare_request(args: Namespace) -> CompareRequest:
    """Build a :class:`CompareRequest` from argparse ``compare`` subcommand namespace."""
    fa = FrameworkType.try_from(args.framework) if getattr(args, "framework", None) else None
    fb_raw = getattr(args, "framework_b", None)
    fb = FrameworkType.try_from(fb_raw) if fb_raw else fa
    arch_b = getattr(args, "architecture_b", None)
    return CompareRequest(
        model_path_a=Path(args.model_a).expanduser().resolve(),
        model_path_b=Path(args.model_b).expanduser().resolve(),
        data_dir=Path(args.data_dir).expanduser().resolve(),
        model_type=ModelType.from_pipeline_value(getattr(args, "model_type", None)),
        framework_a=fa,
        framework_b=fb,
        architecture=getattr(args, "architecture", None),
        architecture_b=arch_b,
        num_classes=getattr(args, "num_classes", None),
        image_size=int(args.image_size),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        device=getattr(args, "device", None),
        max_disagreement_report=getattr(args, "max_disagreement_report", None),
    )


def _write_disagreement_tsv(path: Path, report: ClassificationCompareReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["path", "true_label", "pred_a", "pred_b", "conf_a", "conf_b"])
        for s in report.disagreement_samples:
            w.writerow(
                [
                    s.path,
                    s.true_label,
                    s.predicted_a,
                    s.predicted_b,
                    f"{s.confidence_a:.6f}",
                    f"{s.confidence_b:.6f}",
                ]
            )


def run_evaluate_compare_cli(args: Namespace) -> int:
    """CLI implementation for ``mb evaluate compare`` (returns process exit code)."""
    sub = EvaluateSubcommand.COMPARE.value
    if getattr(args, "dry_run", False):
        ma = Path(args.model_a).expanduser().resolve()
        mb = Path(args.model_b).expanduser().resolve()
        data_dir = Path(args.data_dir).expanduser().resolve()
        if not ma.is_file():
            logger.error(_("Model A file not found: {path}").format(path=ma))
            return 1
        if not mb.is_file():
            logger.error(_("Model B file not found: {path}").format(path=mb))
            return 1
        if not data_dir.is_dir():
            logger.error(_("Data directory not found: {path}").format(path=data_dir))
            return 1
        from mb.conversion.converters import detect_model_framework

        fwa = detect_model_framework(ma)
        fwb = detect_model_framework(mb)
        if fwa == "pytorch" and not getattr(args, "architecture", None):
            logger.error(_("--architecture is required for PyTorch model A in dry-run validation."))
            return 1
        if fwb == "pytorch":
            arch_b = getattr(args, "architecture_b", None) or getattr(args, "architecture", None)
            if not arch_b:
                logger.error(
                    _("Model B is PyTorch: provide --architecture-b or share --architecture in dry-run.")
                )
                return 1
        logger.info(
            _("Dry-run OK: would compare models {a} and {b} on {data} (frameworks {fa}/{fb})").format(
                a=args.model_a, b=args.model_b, data=args.data_dir, fa=fwa or "?", fb=fwb or "?"
            )
        )
        return 0

    max_rep = getattr(args, "max_disagreement_report", None)
    if max_rep is not None and max_rep < 1:
        logger.error(_("--max-disagreement-report must be at least 1 when provided."))
        return 1

    req = build_compare_request(args)
    if not req.model_path_a.is_file():
        logger.error(_("Model A file not found: {path}").format(path=req.model_path_a))
        return 1
    if not req.model_path_b.is_file():
        logger.error(_("Model B file not found: {path}").format(path=req.model_path_b))
        return 1
    if not req.data_dir.is_dir():
        logger.error(_("Data directory not found: {path}").format(path=req.data_dir))
        return 1

    hint = _(
        "Compare evaluation failed for model type {mt} (subcommand {sub}). "
        "Use two checkpoints of the same framework on the same ImageFolder split; "
        "PyTorch needs --architecture (A) and optionally --architecture-b. Underlying error: {err}"
    )

    try:
        report = run_compare(req)
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
        logger.error(_("Import error during compare: {err}").format(err=e))
        return 1
    except Exception as e:
        logger.error(
            hint.format(mt=req.model_type.value, sub=sub, err=e),
            exc_info=getattr(args, "verbose", False),
        )
        return 1

    out_path = getattr(args, "output", None)
    if out_path:
        p = Path(out_path).expanduser().resolve()
        _write_disagreement_tsv(p, report)
        logger.info(_("Wrote {n} disagreement rows to {path}").format(n=len(report.disagreement_samples), path=out_path))

    print(format_classification_compare_report(report))
    logger.debug(json.dumps(report.to_jsonable()))
    return 0
