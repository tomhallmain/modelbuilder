"""
Evaluation, scoring and comparison of text-classification run directories.

:func:`evaluate_text_run` is shared by training (right after calibration) and by
``mb text evaluate``, so a re-evaluation rewrites exactly the files training produced:
``thresholds.json``, ``metrics.json``, ``predictions_test.tsv``, ``review_queue.tsv``,
``unlabeled_scores.tsv``, ``extra_reports.json``, ``predict.py`` and ``MODEL_CARD.md``.

``metrics.json`` holds only values derived from the data and model (no timestamps or
durations), so two runs of a deterministic backend with the same config produce identical
files. Rows from conflict groups (when excluded) and the unlabeled file never enter a
metric; the unlabeled file is only scored for the separate extra report.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

from mb.cancellation import check_cancel_event
from mb.data.text_dataset import (
    TextDataset,
    gold_mask,
    load_text_dataset,
    load_unlabeled,
    read_text_lines,
    verify_text_dataset,
)
from mb.evaluate.classification.text_calibration import Calibrator, load_calibrator
from mb.evaluate.classification.text_metrics import (
    LENGTH_BUCKETS,
    binary_metrics,
    histogram,
    length_bucket_codes,
    recall_slices,
    slice_metrics,
    spearman,
    threshold_for_precision,
    threshold_for_recall,
)
from mb.models.text_backends import TextBackend, load_backend
from mb.models.types import TextThresholdPolicy
from mb.training.text_config import TextRunConfig, parse_text_run_config
from mb.utils.logging_setup import get_logger
from mb.utils.translations import _

logger = get_logger(__name__)

CONFIG_FILE = "config.yaml"
ENVIRONMENT_FILE = "environment.json"
MODEL_DIR = "model"
CALIBRATOR_FILE = "calibrator.json"
THRESHOLDS_FILE = "thresholds.json"
METRICS_FILE = "metrics.json"
PREDICTIONS_FILE = "predictions_test.tsv"
REVIEW_QUEUE_FILE = "review_queue.tsv"
UNLABELED_SCORES_FILE = "unlabeled_scores.tsv"
EXTRA_REPORTS_FILE = "extra_reports.json"
MODEL_CARD_FILE = "MODEL_CARD.md"

# Fallback decision threshold when the configured precision target is unreachable on val.
FALLBACK_THRESHOLD = 0.5

ProgressFn = Callable[[str, Optional[float]], None]


def write_json(path: Path, data: Any) -> None:
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_run_config(run_dir: Path) -> TextRunConfig:
    path = Path(run_dir) / CONFIG_FILE
    if not path.is_file():
        raise FileNotFoundError(_("Not a text-classification run directory (no {file}): {path}").format(
            file=CONFIG_FILE, path=run_dir
        ))
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return parse_text_run_config(raw)


def _sub_progress(progress: Optional[ProgressFn], lo: float, hi: float) -> Optional[ProgressFn]:
    if progress is None:
        return None
    return lambda m, f: progress(m, None if f is None else lo + (hi - lo) * f)


def calibrated_scores(
    backend: TextBackend,
    calibrator: Calibrator,
    texts: Sequence[str],
    *,
    cancel_event: Optional[threading.Event] = None,
    progress: Optional[ProgressFn] = None,
) -> np.ndarray:
    return calibrator.transform(backend.predict_proba(texts, cancel_event=cancel_event, progress=progress))


@dataclass
class Thresholds:
    precision_target: float
    precision_threshold: Optional[float]
    recall_target: float
    recall_threshold: Optional[float]
    kind: TextThresholdPolicy
    decision: float
    fallback: bool

    def to_json(self) -> Dict[str, Any]:
        return {
            "precision_target": {"P": self.precision_target, "threshold": self.precision_threshold},
            "recall_target": {"R": self.recall_target, "threshold": self.recall_threshold},
            "decision": {"kind": self.kind.value, "threshold": self.decision, "fallback": self.fallback},
        }


def choose_thresholds(labels: np.ndarray, probs: np.ndarray, config: TextRunConfig) -> Thresholds:
    """Both named policies on val gold, and the decision threshold the config selects."""
    target = config.threshold_policy.value
    t_p = threshold_for_precision(labels, probs, target)
    t_r = threshold_for_recall(labels, probs, target)
    kind = config.threshold_policy.kind
    chosen = t_p if kind == TextThresholdPolicy.PRECISION_TARGET else t_r
    fallback = chosen is None
    if fallback:
        logger.warning(
            "Threshold policy %s=%.3f is unreachable on val gold; using %.2f",
            kind.value,
            target,
            FALLBACK_THRESHOLD,
        )
    return Thresholds(
        precision_target=target,
        precision_threshold=t_p,
        recall_target=target,
        recall_threshold=t_r,
        kind=kind,
        decision=FALLBACK_THRESHOLD if chosen is None else chosen,
        fallback=fallback,
    )


def _split_report(
    ds: TextDataset,
    idx: np.ndarray,
    probs: np.ndarray,
    gold: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    labels = ds.labels[idx].astype(np.int64)
    out: Dict[str, Any] = {
        "gold": binary_metrics(labels[gold], probs[gold], threshold),
        "full": binary_metrics(labels, probs, threshold),
    }
    slices: Dict[str, Any] = {}
    if ds.tier is not None:
        slices["tier"] = slice_metrics(labels, probs, threshold, [ds.tier.value(i) for i in idx])
    if ds.source is not None:
        slices["source"] = slice_metrics(labels, probs, threshold, [ds.source.value(i) for i in idx])
    if ds.category_hint is not None:
        slices["category_hint"] = recall_slices(
            labels, probs, threshold, [ds.category_hint.value(i) for i in idx]
        )
    codes = length_bucket_codes([ds.texts[i] for i in idx])
    names = [b[0] for b in LENGTH_BUCKETS]
    slices["length"] = slice_metrics(labels, probs, threshold, [names[c] for c in codes])
    out["slices"] = slices
    return out


def _write_tsv(path: Path, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(str(v) for v in r) + "\n")


def _fmt_score(x: float) -> str:
    return f"{float(x):.6f}"


@dataclass
class TextEvaluationResult:
    run_dir: Path
    metrics: Dict[str, Any]
    thresholds: Thresholds


def evaluate_text_run(
    run_dir: Path,
    *,
    data_dir: Optional[Path] = None,
    device: Optional[str] = None,
    backend: Optional[TextBackend] = None,
    calibrator: Optional[Calibrator] = None,
    dataset: Optional[TextDataset] = None,
    val_raw: Optional[np.ndarray] = None,
    cancel_event: Optional[threading.Event] = None,
    progress: Optional[ProgressFn] = None,
) -> TextEvaluationResult:
    """
    Evaluate a run: thresholds on val gold, metrics and slices for val and test (gold and
    full), extra reports, ``predict.py`` and the model card.

    Training passes its in-memory *backend*, *calibrator*, *dataset* and uncalibrated
    *val_raw*; ``mb text evaluate`` passes none of them and everything is reloaded from
    *run_dir* (and *data_dir*, default: the run's own ``config.yaml``).
    """
    from mb.evaluate.classification.text_model_card import write_model_card
    from mb.evaluate.classification.text_predict_script import write_predict_script

    run_dir = Path(run_dir)
    config = load_run_config(run_dir)
    say = progress or (lambda _m, _f: None)

    if dataset is None:
        ddir = Path(data_dir) if data_dir is not None else config.data_dir
        say(_("Verifying dataset…"), None)
        ds = load_text_dataset(
            ddir,
            keep_groups=True,
            reference_score_column=config.reference_score_column,
            cancel_event=cancel_event,
        )
        report = verify_text_dataset(
            ddir,
            dataset=ds,
            unlabeled_file=config.unlabeled_file,
            reference_score_column=config.reference_score_column,
            cancel_event=cancel_event,
        )
        if not report.ok:
            raise ValueError(_("Dataset verification failed:\n{report}").format(report=report.format()))
        ds.drop_groups()
    else:
        ds = dataset
    if backend is None:
        say(_("Loading model…"), None)
        backend = load_backend(run_dir / MODEL_DIR, device or resolve_device(config.device, neural=True))
    if calibrator is None:
        calibrator = load_calibrator(run_dir / CALIBRATOR_FILE)

    excl = config.exclude_conflicts
    val_idx = ds.split_indices("val", exclude_conflicts=excl)
    test_idx = ds.split_indices("test", exclude_conflicts=excl)
    val_texts = [ds.texts[i] for i in val_idx]
    test_texts = [ds.texts[i] for i in test_idx]

    if val_raw is None:
        val_raw = backend.predict_proba(
            val_texts, cancel_event=cancel_event, progress=_sub_progress(say, 0.0, 0.1)
        )
    p_val = calibrator.transform(val_raw)
    say(_("Scoring test…"), 0.1)
    p_test = calibrated_scores(
        backend, calibrator, test_texts, cancel_event=cancel_event, progress=_sub_progress(say, 0.1, 0.25)
    )
    check_cancel_event(cancel_event)

    val_gold = gold_mask(ds, val_idx, config.gold_tiers)
    test_gold = gold_mask(ds, test_idx, config.gold_tiers)
    val_labels = ds.labels[val_idx].astype(np.int64)
    thresholds = choose_thresholds(val_labels[val_gold], p_val[val_gold], config)
    write_json(run_dir / THRESHOLDS_FILE, thresholds.to_json())

    say(_("Computing metrics…"), 0.3)
    metrics: Dict[str, Any] = {
        "decision_threshold": thresholds.decision,
        "excluded_conflict_rows": {
            "val": int(ds.conflict[ds.split.mask(["val"])].sum()) if excl else 0,
            "test": int(ds.conflict[ds.split.mask(["test"])].sum()) if excl else 0,
        },
        "val": _split_report(ds, val_idx, p_val, val_gold, thresholds.decision),
        "test": _split_report(ds, test_idx, p_test, test_gold, thresholds.decision),
    }

    say(_("Counting truncated rows…"), 0.35)
    truncation: Dict[str, Optional[int]] = {}
    for split in ("train", "val", "test"):
        idx = ds.split_indices(split, exclude_conflicts=excl)
        truncation[split] = backend.count_truncated([ds.texts[i] for i in idx])
    if any(v is not None for v in truncation.values()):
        metrics["truncated_rows"] = truncation
        logger.info("Rows truncated to %d tokens: %s", config.max_length, truncation)
    write_json(run_dir / METRICS_FILE, metrics)

    _write_tsv(
        run_dir / PREDICTIONS_FILE,
        ("text", "label", "tier", "score"),
        [
            (ds.texts[i], int(ds.labels[i]), ds.tier_of(i), _fmt_score(s))
            for i, s in zip(test_idx.tolist(), p_test)
        ],
    )

    extra: Dict[str, Any] = {}
    if ds.reference_score is not None:
        ref = ds.reference_score[test_idx]
        test_labels = ds.labels[test_idx]
        extra["reference_spearman_test"] = {
            "all": spearman(p_test, ref),
            "label_0": spearman(p_test[test_labels == 0], ref[test_labels == 0]),
            "label_1": spearman(p_test[test_labels == 1], ref[test_labels == 1]),
        }

    n_queue = config.reports.review_queue_size
    tier = config.unreviewed_tier
    if n_queue > 0 and tier is not None and ds.tier is not None and ds.tier.code_of(tier) is not None:
        say(_("Scoring {tier} rows for the review queue…").format(tier=tier), 0.4)
        _write_review_queue(
            run_dir, ds, config, backend, calibrator, p_val, val_idx, p_test, test_idx, n_queue,
            cancel_event, _sub_progress(say, 0.4, 0.85),
        )

    if config.reports.score_unlabeled:
        unlabeled = load_unlabeled(
            ds.data_dir,
            unlabeled_file=config.unlabeled_file,
            reference_score_column=config.reference_score_column,
        )
        if unlabeled is not None:
            say(_("Scoring unlabeled rows…"), 0.85)
            u_texts, u_ref = unlabeled
            p_u = calibrated_scores(
                backend, calibrator, u_texts, cancel_event=cancel_event, progress=_sub_progress(say, 0.85, 0.97)
            )
            _write_tsv(
                run_dir / UNLABELED_SCORES_FILE,
                ("text", "score"),
                [(t, _fmt_score(s)) for t, s in zip(u_texts, p_u)],
            )
            below = {
                name: (None if t is None else int(np.sum(p_u < t)))
                for name, t in (
                    ("precision_target", thresholds.precision_threshold),
                    ("recall_target", thresholds.recall_threshold),
                    ("decision", thresholds.decision),
                )
            }
            extra["unlabeled"] = {
                "file": config.unlabeled_file,
                "n": len(u_texts),
                "histogram": histogram(p_u),
                "below_threshold": below,
            }
            if u_ref is not None:
                extra["unlabeled"]["reference_spearman"] = spearman(p_u, u_ref)
    write_json(run_dir / EXTRA_REPORTS_FILE, extra)

    env_path = run_dir / ENVIRONMENT_FILE
    env = read_json(env_path) if env_path.is_file() else {}
    write_predict_script(run_dir, type(backend))
    write_model_card(run_dir, config, metrics, thresholds.to_json(), env, extra)
    say(_("Evaluation complete."), 1.0)
    return TextEvaluationResult(run_dir=run_dir, metrics=metrics, thresholds=thresholds)


def _write_review_queue(
    run_dir: Path,
    ds: TextDataset,
    config: TextRunConfig,
    backend: TextBackend,
    calibrator: Calibrator,
    p_val: np.ndarray,
    val_idx: np.ndarray,
    p_test: np.ndarray,
    test_idx: np.ndarray,
    n_queue: int,
    cancel_event: Optional[threading.Event],
    progress: Optional[ProgressFn],
) -> None:
    """
    ``unreviewed_tier`` rows across all splits whose score most disagrees with their label
    (``|score - label|``, largest first): the likeliest label errors.
    """
    unreviewed = ds.tier.mask([config.unreviewed_tier])
    if config.exclude_conflicts:
        unreviewed &= ~ds.conflict
    scores = np.full(len(ds), np.nan)
    scores[val_idx] = p_val
    scores[test_idx] = p_test
    train_unreviewed = np.flatnonzero(unreviewed & ds.split.mask(["train"]))
    if len(train_unreviewed):
        scores[train_unreviewed] = calibrated_scores(
            backend,
            calibrator,
            [ds.texts[i] for i in train_unreviewed],
            cancel_event=cancel_event,
            progress=progress,
        )
    cand = np.flatnonzero(unreviewed)
    disagreement = np.abs(scores[cand] - ds.labels[cand])
    # Largest disagreement first; ties keep dataset order so the file is deterministic.
    order = np.lexsort((cand, -disagreement))[:n_queue]
    top = cand[order]
    ref = ds.reference_score
    _write_tsv(
        run_dir / REVIEW_QUEUE_FILE,
        ("text", "label", "tier", "source", "split", "score", "reference_score"),
        [
            (
                ds.texts[i],
                int(ds.labels[i]),
                ds.tier_of(i),
                ds.source.value(i) if ds.source is not None else "",
                ds.split.value(i),
                _fmt_score(scores[i]),
                "" if ref is None or np.isnan(ref[i]) else f"{float(ref[i]):.4f}",
            )
            for i in top.tolist()
        ],
    )


def resolve_device(requested: Optional[str], *, neural: bool) -> str:
    """The configured device, else ``cuda`` when torch sees a GPU, else ``cpu``."""
    if requested:
        return requested
    if not neural:
        return "cpu"
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


# --- scoring arbitrary lines ---


@dataclass
class ScoreResult:
    n_lines: int
    n_with_tab: int
    output: Path


def score_text_file(
    run_dir: Path,
    input_path: Path,
    output_path: Path,
    *,
    device: Optional[str] = None,
    cancel_event: Optional[threading.Event] = None,
    progress: Optional[ProgressFn] = None,
) -> ScoreResult:
    """
    Score one string per input line; write ``term<TAB>score`` with a header, in input
    order, one output row per input line (empty lines included).
    """
    run_dir = Path(run_dir)
    config = load_run_config(run_dir)
    lines = read_text_lines(Path(input_path))
    backend = load_backend(run_dir / MODEL_DIR, device or resolve_device(config.device, neural=True))
    calibrator = load_calibrator(run_dir / CALIBRATOR_FILE)
    scores = calibrated_scores(backend, calibrator, lines, cancel_event=cancel_event, progress=progress)
    with_tab = sum(1 for t in lines if "\t" in t)
    if with_tab:
        logger.warning(
            "%d input line(s) contain a tab; their output rows have more than two columns", with_tab
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_tsv(output_path, ("term", "score"), [(t, _fmt_score(s)) for t, s in zip(lines, scores)])
    return ScoreResult(n_lines=len(lines), n_with_tab=with_tab, output=output_path)


# --- comparing runs ---


@dataclass
class CompareRow:
    run: str
    backend: str
    model_id: str
    gold_ap: Optional[float]
    full_ap: Optional[float]
    recall_at_p90: Optional[float]
    gold_fpr: Optional[float]
    ece: Optional[float]


def compare_text_runs(run_dirs: Sequence[Path]) -> Tuple[List[CompareRow], List[str]]:
    """One row per run with a ``metrics.json``; also returns the directories skipped."""
    rows: List[CompareRow] = []
    skipped: List[str] = []
    for d in run_dirs:
        d = Path(d)
        mpath = d / METRICS_FILE
        if not mpath.is_file():
            skipped.append(str(d))
            continue
        m = read_json(mpath)
        try:
            cfg = load_run_config(d)
            backend, model_id = cfg.backend.value, cfg.model_id or ""
        except Exception:
            backend, model_id = "?", ""
        test = m.get("test") or {}
        gold, full = test.get("gold") or {}, test.get("full") or {}
        rows.append(
            CompareRow(
                run=d.name,
                backend=backend,
                model_id=model_id,
                gold_ap=gold.get("ap"),
                full_ap=full.get("ap"),
                recall_at_p90=(gold.get("recall_at_precision") or {}).get("0.90"),
                gold_fpr=gold.get("fpr"),
                ece=gold.get("ece"),
            )
        )
    return rows, skipped


def format_compare_table(rows: Sequence[CompareRow]) -> str:
    def f(x: Optional[float]) -> str:
        return "-" if x is None else f"{x:.4f}"

    header = [
        _("run"),
        _("backend"),
        _("model"),
        _("gold AP"),
        _("full AP"),
        _("R@P0.90"),
        _("gold FPR"),
        _("ECE"),
    ]
    body = [
        [r.run, r.backend, r.model_id, f(r.gold_ap), f(r.full_ap), f(r.recall_at_p90), f(r.gold_fpr), f(r.ece)]
        for r in rows
    ]
    widths = [max(len(str(c)) for c in col) for col in zip(header, *body)]
    lines = ["  ".join(str(c).ljust(w) for c, w in zip(header, widths))]
    lines.append("  ".join("-" * w for w in widths))
    lines.extend("  ".join(str(c).ljust(w) for c, w in zip(r, widths)) for r in body)
    return "\n".join(lines)
