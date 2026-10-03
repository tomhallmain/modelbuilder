"""
``MODEL_CARD.md`` for a text-classification run.

The card is a run artifact read alongside the weights, like the TSV and JSON outputs, so its
wording is fixed English rather than gettext-translated: two runs with the same config
must produce comparable cards regardless of the locale they ran under. Dataset-specific
label-policy caveats come from the ``model_card_notes`` config value.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from mb.models.text_backends import get_backend_class
from mb.training.text_config import TextRunConfig

_CARD_FILE = "MODEL_CARD.md"


def _f(x: Optional[float]) -> str:
    return "-" if x is None else f"{x:.4f}"


def _metric_rows(metrics: Dict[str, Any]) -> List[str]:
    rows = [
        "| split | set | n | positives | AP | ROC-AUC | R@P0.90 | R@P0.95 | P@R0.90 | P@R0.95 | Brier | ECE |",
        "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for split in ("val", "test"):
        for which in ("gold", "full"):
            m = (metrics.get(split) or {}).get(which) or {}
            rap = m.get("recall_at_precision") or {}
            par = m.get("precision_at_recall") or {}
            rows.append(
                f"| {split} | {which} | {m.get('n', 0)} | {m.get('n_pos', 0)} | {_f(m.get('ap'))} | "
                f"{_f(m.get('roc_auc'))} | {_f(rap.get('0.90'))} | {_f(rap.get('0.95'))} | "
                f"{_f(par.get('0.90'))} | {_f(par.get('0.95'))} | {_f(m.get('brier'))} | {_f(m.get('ece'))} |"
            )
    return rows


def _confusion_rows(metrics: Dict[str, Any]) -> List[str]:
    rows = ["| set | TN | FP | FN | TP |", "|---|--:|--:|--:|--:|"]
    for which in ("gold", "full"):
        cm = ((metrics.get("test") or {}).get(which) or {}).get("confusion") or {}
        rows.append(f"| test {which} | {cm.get('tn', 0)} | {cm.get('fp', 0)} | {cm.get('fn', 0)} | {cm.get('tp', 0)} |")
    return rows


def _tier_rows(metrics: Dict[str, Any]) -> List[str]:
    tiers = (((metrics.get("test") or {}).get("slices") or {}).get("tier")) or {}
    if not tiers:
        return []
    rows = ["| tier | n | positives | AP | recall | FPR | flag rate |", "|---|--:|--:|--:|--:|--:|--:|"]
    for name, m in tiers.items():
        rows.append(
            f"| {name} | {m.get('n', 0)} | {m.get('n_pos', 0)} | {_f(m.get('ap'))} | "
            f"{_f(m.get('recall'))} | {_f(m.get('fpr'))} | {_f(m.get('flag_rate'))} |"
        )
    return rows


def write_model_card(
    run_dir: Path,
    config: TextRunConfig,
    metrics: Dict[str, Any],
    thresholds: Dict[str, Any],
    env: Dict[str, Any],
    extra: Dict[str, Any],
) -> Path:
    data = env.get("dataset") or {}
    training = env.get("training") or {}
    subsample = training.get("subsample")
    notes = config.model_card_notes.strip()

    lines: List[str] = [
        f"# Model card: {run_dir.name}",
        "",
        "## Model",
        "",
        f"- Backend: `{config.backend.value}`",
        f"- Model: `{config.model_id or '-'}`",
        f"- Seed: {config.seed}",
        f"- Max length: {config.max_length} tokens",
        f"- Calibration: `{config.calibration.value}` (fitted on val)",
        f"- Output: calibrated p(label = 1); flagged when p >= the decision threshold",
        "",
        "## Usage",
        "",
        "`predict.py` in this directory scores strings without `mb` installed. It needs "
        + ", ".join(f"`{r}`" for r in ("numpy",) + tuple(get_backend_class(config.backend).standalone_requirements))
        + ".",
        "",
        "```bash",
        'python predict.py "some string" "another string"     # text, score, flagged',
        "python predict.py --input lines.txt --output scores.tsv",
        "```",
        "",
        "```python",
        "from predict import load",
        "clf = load()                     # run directory = the script's directory",
        'clf.score(["some string"])       # calibrated p(label = 1)',
        'clf.flag(["some string"])        # score >= decision threshold',
        "```",
        "",
        "## Intended use and label policy",
        "",
        notes if notes else "_No label-policy notes were configured (`text_classification.model_card_notes`)._",
        "",
        "## Data",
        "",
        f"- Directory: `{config.data_dir.as_posix()}`",
    ]
    for name, digest in sorted((data.get("sha256") or {}).items()):
        lines.append(f"- sha256 `{name}`: `{digest}`")
    lines += [
        f"- Training rows: {training.get('n_train', '-')} "
        f"({training.get('n_train_pos', '-')} positive)",
        f"- Class weights: {training.get('class_weights', '-')}; tier weights: {dict(config.tier_weight) or 'all 1.0'}",
        f"- Conflict groups excluded from training and metrics: {'yes' if config.exclude_conflicts else 'no'}",
    ]
    if subsample:
        lines.append(
            f"- keep_unreviewed subsampled to {subsample['ratio']}x positives "
            f"(seed {subsample['seed']}): kept {subsample['kept']} of {subsample['available']}"
        )
    else:
        lines.append("- No negative subsampling")
    trunc = metrics.get("truncated_rows")
    if trunc:
        lines.append(f"- Rows truncated at {config.max_length} tokens: {trunc}")
    lines += [
        "",
        "## Thresholds (chosen on val gold)",
        "",
        f"- precision_target P={thresholds['precision_target']['P']}: {_f(thresholds['precision_target']['threshold'])}",
        f"- recall_target R={thresholds['recall_target']['R']}: {_f(thresholds['recall_target']['threshold'])}",
        f"- Decision threshold ({thresholds['decision']['kind']}): {_f(thresholds['decision']['threshold'])}"
        + (" — fallback, target unreachable on val" if thresholds["decision"].get("fallback") else ""),
        "",
        "## Metrics",
        "",
        "Gold = rows in tiers " + ", ".join(f"`{t}`" for t in config.gold_tiers) + " (the headline set). "
        "Full = every evaluated row; where unreviewed negatives are noisy its precision is a lower bound.",
        "",
        *_metric_rows(metrics),
        "",
        "Confusion at the decision threshold:",
        "",
        *_confusion_rows(metrics),
        "",
    ]
    tier_rows = _tier_rows(metrics)
    if tier_rows:
        lines += ["Test by tier (FPR on human-kept negatives is the hard-negative score):", "", *tier_rows, ""]
    if extra.get("unlabeled_cut"):
        below = extra["unlabeled_cut"].get("below_threshold") or {}
        lines += [
            "## Unlabeled cut (not part of any metric)",
            "",
            f"- Lines scored: {extra['unlabeled_cut'].get('n')}; below the decision threshold: {below.get('decision')}",
            "",
        ]
    lines += [
        "## Limits",
        "",
        "- Scores are calibrated on val; they are probabilities only for data resembling this dataset.",
        "- Test metrics come from groups never seen in training (fixed split column).",
        "",
        "## Environment",
        "",
        f"- Python {env.get('python', '-')}, platform {env.get('platform', '-')}",
        f"- Libraries: {env.get('libraries', {})}",
        f"- Device: {env.get('device', '-')}",
        "",
    ]
    path = Path(run_dir) / _CARD_FILE
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
