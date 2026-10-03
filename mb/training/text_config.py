"""
``text_classification`` pipeline section: defaults, strict parsing, CLI/GUI overrides.

The section is merged over :data:`TEXT_CLASSIFICATION_DEFAULTS` like every other pipeline
section, then parsed into a :class:`TextRunConfig`. Unlike the image sections, which fall
back to defaults on a bad value, parsing here is strict: an unknown key or invalid value
is an error, because a silently ignored typo (``tier_weigth``) would change what a run
trains on while its saved ``config.yaml`` claims otherwise. Problems are collected and
raised together so one fix-up pass covers them all.

Imports only the standard library and enums, so :mod:`mb.pipeline_config` can import the
defaults without pulling in numpy or ML frameworks.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from mb.models.types import (
    TextBackendType,
    TextCalibrationMethod,
    TextThresholdPolicy,
)
from mb.utils.translations import _

PIPELINE_SECTION = "text_classification"

CLASS_WEIGHT_NONE = "none"
CLASS_WEIGHT_BALANCED = "balanced"

EARLY_STOPPING_METRICS = ("val_ap",)

TEXT_CLASSIFICATION_DEFAULTS: Dict[str, Any] = {
    # Directory holding dataset.tsv (plus optional manifest.json, label_conflicts.tsv,
    # unlabeled_cut.tsv).
    "data_dir": "text_classifier_training_set",
    # Parent of the per-run output directories.
    "runs_dir": "data/models/text_runs",
    "backend": TextBackendType.get_default().value,
    # null = the backend's default model (none for char_ngram_linear).
    "model_id": None,
    "seed": 13,
    # Token limit for subword models; rows longer than this are counted and reported.
    "max_length": 64,
    # none | balanced | {0: w0, 1: w1}
    "class_weight": CLASS_WEIGHT_BALANCED,
    # Per-tier loss multiplier; tiers not listed get 1.0.
    "tier_weight": {},
    # Drop label_conflicts.tsv groups from training and from every metric.
    "exclude_conflicts": True,
    # null, or train keep_unreviewed rows to keep as a multiple of train positives.
    "subsample_keep_unreviewed": None,
    # Tiers whose rows make up the human-read "gold" evaluation set.
    "gold_tiers": ["reject_reviewed", "keep_reviewed"],
    "optim": {
        # null = the backend's default.
        "lr": None,
        "epochs": None,
        "batch_size": None,
        "warmup_ratio": 0.06,
        "weight_decay": 0.01,
    },
    "early_stopping": {
        "metric": "val_ap",
        "patience": 2,
    },
    # auto | temperature | platt | isotonic
    "calibration": TextCalibrationMethod.AUTO.value,
    # kind picks the decision threshold; both policies are stored, each at `value`.
    "threshold_policy": {
        "kind": TextThresholdPolicy.PRECISION_TARGET.value,
        "value": 0.90,
    },
    "reports": {
        # Highest-scoring keep_unreviewed rows written to review_queue.tsv; 0 disables.
        "review_queue_size": 200,
        # Score unlabeled_cut.tsv (when present) for cut_rescore.tsv.
        "rescore_unlabeled_cut": True,
    },
    # Markdown copied into MODEL_CARD.md under "Intended use and label policy".
    "model_card_notes": "",
    # null = cuda when available, else cpu.
    "device": None,
    # Backend-specific settings; each backend validates its own keys.
    "backend_options": {},
}


class TextConfigError(ValueError):
    """Raised with every problem found in a ``text_classification`` section."""

    def __init__(self, problems: List[str]) -> None:
        self.problems = list(problems)
        super().__init__(
            _("Invalid text_classification config:")
            + "\n"
            + "\n".join(f"  - {p}" for p in self.problems)
        )


@dataclass(frozen=True)
class TextOptimConfig:
    lr: Optional[float]
    epochs: Optional[int]
    batch_size: Optional[int]
    warmup_ratio: float
    weight_decay: float


@dataclass(frozen=True)
class TextEarlyStoppingConfig:
    metric: str
    patience: int


@dataclass(frozen=True)
class TextThresholdConfig:
    kind: TextThresholdPolicy
    value: float


@dataclass(frozen=True)
class TextReportsConfig:
    review_queue_size: int
    rescore_unlabeled_cut: bool


ClassWeight = Union[str, Dict[int, float]]


@dataclass(frozen=True)
class TextRunConfig:
    """A fully validated ``text_classification`` section."""

    data_dir: Path
    runs_dir: Path
    backend: TextBackendType
    model_id: Optional[str]
    seed: int
    max_length: int
    class_weight: ClassWeight
    tier_weight: Dict[str, float]
    exclude_conflicts: bool
    subsample_keep_unreviewed: Optional[float]
    gold_tiers: Tuple[str, ...]
    optim: TextOptimConfig
    early_stopping: TextEarlyStoppingConfig
    calibration: TextCalibrationMethod
    threshold_policy: TextThresholdConfig
    reports: TextReportsConfig
    model_card_notes: str
    device: Optional[str]
    backend_options: Dict[str, Any] = field(default_factory=dict)

    def with_backend_defaults(
        self,
        *,
        model_id: Optional[str],
        lr: Optional[float],
        epochs: Optional[int],
        batch_size: Optional[int],
        backend_options: Dict[str, Any],
    ) -> TextRunConfig:
        """Fill the ``null`` = "backend default" fields so the saved config is explicit."""
        optim = replace(
            self.optim,
            lr=self.optim.lr if self.optim.lr is not None else lr,
            epochs=self.optim.epochs if self.optim.epochs is not None else epochs,
            batch_size=self.optim.batch_size if self.optim.batch_size is not None else batch_size,
        )
        return replace(
            self,
            model_id=self.model_id if self.model_id is not None else model_id,
            optim=optim,
            backend_options=backend_options,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Plain YAML-ready mapping that :func:`parse_text_run_config` accepts back."""
        cw: Any = self.class_weight
        if isinstance(cw, dict):
            cw = {int(k): float(v) for k, v in cw.items()}
        return {
            "data_dir": self.data_dir.as_posix(),
            "runs_dir": self.runs_dir.as_posix(),
            "backend": self.backend.value,
            "model_id": self.model_id,
            "seed": self.seed,
            "max_length": self.max_length,
            "class_weight": cw,
            "tier_weight": dict(self.tier_weight),
            "exclude_conflicts": self.exclude_conflicts,
            "subsample_keep_unreviewed": self.subsample_keep_unreviewed,
            "gold_tiers": list(self.gold_tiers),
            "optim": {
                "lr": self.optim.lr,
                "epochs": self.optim.epochs,
                "batch_size": self.optim.batch_size,
                "warmup_ratio": self.optim.warmup_ratio,
                "weight_decay": self.optim.weight_decay,
            },
            "early_stopping": {
                "metric": self.early_stopping.metric,
                "patience": self.early_stopping.patience,
            },
            "calibration": self.calibration.value,
            "threshold_policy": {
                "kind": self.threshold_policy.kind.value,
                "value": self.threshold_policy.value,
            },
            "reports": {
                "review_queue_size": self.reports.review_queue_size,
                "rescore_unlabeled_cut": self.reports.rescore_unlabeled_cut,
            },
            "model_card_notes": self.model_card_notes,
            "device": self.device,
            "backend_options": copy.deepcopy(self.backend_options),
        }


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


class _Parser:
    """Collects problems while reading one mapping; see :func:`parse_text_run_config`."""

    def __init__(self) -> None:
        self.problems: List[str] = []

    def unknown_keys(self, raw: Mapping[str, Any], allowed: Mapping[str, Any], where: str) -> None:
        for key in raw:
            if key not in allowed:
                self.problems.append(
                    _("Unknown key '{key}' in {where} (allowed: {allowed})").format(
                        key=key, where=where, allowed=", ".join(allowed)
                    )
                )

    def mapping(self, raw: Mapping[str, Any], key: str, where: str) -> Mapping[str, Any]:
        v = raw.get(key)
        if isinstance(v, Mapping):
            return v
        self.problems.append(_("{where}.{key} must be a mapping").format(where=where, key=key))
        return {}

    def path(self, v: Any, name: str) -> Path:
        if isinstance(v, (str, Path)) and str(v).strip():
            return Path(str(v).strip())
        self.problems.append(_("{name} must be a non-empty path").format(name=name))
        return Path(".")

    def int_min(self, v: Any, name: str, minimum: int) -> int:
        if _is_int(v) and v >= minimum:
            return int(v)
        self.problems.append(
            _("{name} must be an integer >= {min} (got {v!r})").format(name=name, min=minimum, v=v)
        )
        return minimum

    def optional_int_min(self, v: Any, name: str, minimum: int) -> Optional[int]:
        return None if v is None else self.int_min(v, name, minimum)

    def float_range(
        self, v: Any, name: str, lo: float, hi: Optional[float], *, lo_inclusive: bool = True
    ) -> float:
        ok = _is_number(v) and (v >= lo if lo_inclusive else v > lo) and (hi is None or v <= hi)
        if ok:
            return float(v)
        bound = ("[" if lo_inclusive else "(") + f"{lo}, " + ("inf)" if hi is None else f"{hi}]")
        self.problems.append(
            _("{name} must be a number in {range} (got {v!r})").format(name=name, range=bound, v=v)
        )
        return lo

    def optional_positive_float(self, v: Any, name: str) -> Optional[float]:
        if v is None:
            return None
        return self.float_range(v, name, 0.0, None, lo_inclusive=False)

    def boolean(self, v: Any, name: str) -> bool:
        if isinstance(v, bool):
            return v
        self.problems.append(_("{name} must be true or false (got {v!r})").format(name=name, v=v))
        return False

    def optional_str(self, v: Any, name: str) -> Optional[str]:
        if v is None:
            return None
        if isinstance(v, str) and v.strip():
            return v.strip()
        self.problems.append(_("{name} must be a non-empty string or null").format(name=name))
        return None


def _parse_class_weight(p: _Parser, v: Any) -> ClassWeight:
    if isinstance(v, str) and v.strip().lower() in (CLASS_WEIGHT_NONE, CLASS_WEIGHT_BALANCED):
        return v.strip().lower()
    if isinstance(v, Mapping):
        out: Dict[int, float] = {}
        problems_before = len(p.problems)
        for k, w in v.items():
            try:
                label = int(str(k).strip())
            except ValueError:
                label = -1
            if label not in (0, 1):
                p.problems.append(
                    _("class_weight keys must be 0 and 1 (got {k!r})").format(k=k)
                )
                continue
            out[label] = p.float_range(w, f"class_weight[{label}]", 0.0, None, lo_inclusive=False)
        if set(out) != {0, 1} and len(p.problems) == problems_before:
            p.problems.append(_("class_weight mapping must give a weight for both 0 and 1"))
        return out
    p.problems.append(
        _("class_weight must be 'none', 'balanced' or {{0: w0, 1: w1}} (got {v!r})").format(v=v)
    )
    return CLASS_WEIGHT_NONE


def _parse_tier_weight(p: _Parser, v: Any) -> Dict[str, float]:
    if v is None:
        return {}
    if not isinstance(v, Mapping):
        p.problems.append(_("tier_weight must be a mapping of tier name to multiplier"))
        return {}
    return {
        str(tier): p.float_range(w, f"tier_weight.{tier}", 0.0, None) for tier, w in v.items()
    }


def _parse_gold_tiers(p: _Parser, v: Any) -> Tuple[str, ...]:
    if isinstance(v, (list, tuple)) and v and all(isinstance(t, str) and t.strip() for t in v):
        return tuple(t.strip() for t in v)
    p.problems.append(_("gold_tiers must be a non-empty list of tier names"))
    return ()


def parse_text_run_config(raw: Mapping[str, Any]) -> TextRunConfig:
    """
    Validate a merged ``text_classification`` mapping.

    Raises:
        TextConfigError: listing every unknown key and invalid value.
    """
    p = _Parser()
    if not isinstance(raw, Mapping):
        raise TextConfigError([_("text_classification must be a mapping")])
    d = TEXT_CLASSIFICATION_DEFAULTS
    p.unknown_keys(raw, d, PIPELINE_SECTION)

    def get(key: str) -> Any:
        return raw.get(key, d[key])

    backend = TextBackendType.try_from(get("backend"))
    if backend is None:
        p.problems.append(
            _("backend must be one of: {choices} (got {v!r})").format(
                choices=", ".join(b.value for b in TextBackendType), v=get("backend")
            )
        )
        backend = TextBackendType.get_default()

    calibration = TextCalibrationMethod.try_from(get("calibration"))
    if calibration is None:
        p.problems.append(
            _("calibration must be one of: {choices} (got {v!r})").format(
                choices=", ".join(c.value for c in TextCalibrationMethod), v=get("calibration")
            )
        )
        calibration = TextCalibrationMethod.AUTO

    optim_raw = p.mapping({"optim": get("optim")}, "optim", PIPELINE_SECTION)
    p.unknown_keys(optim_raw, d["optim"], f"{PIPELINE_SECTION}.optim")
    od = {**d["optim"], **optim_raw}
    optim = TextOptimConfig(
        lr=p.optional_positive_float(od["lr"], "optim.lr"),
        epochs=p.optional_int_min(od["epochs"], "optim.epochs", 1),
        batch_size=p.optional_int_min(od["batch_size"], "optim.batch_size", 1),
        warmup_ratio=p.float_range(od["warmup_ratio"], "optim.warmup_ratio", 0.0, 1.0),
        weight_decay=p.float_range(od["weight_decay"], "optim.weight_decay", 0.0, None),
    )

    es_raw = p.mapping({"early_stopping": get("early_stopping")}, "early_stopping", PIPELINE_SECTION)
    p.unknown_keys(es_raw, d["early_stopping"], f"{PIPELINE_SECTION}.early_stopping")
    ed = {**d["early_stopping"], **es_raw}
    metric = ed["metric"]
    if metric not in EARLY_STOPPING_METRICS:
        p.problems.append(
            _("early_stopping.metric must be one of: {choices} (got {v!r})").format(
                choices=", ".join(EARLY_STOPPING_METRICS), v=metric
            )
        )
        metric = EARLY_STOPPING_METRICS[0]
    early_stopping = TextEarlyStoppingConfig(
        metric=str(metric),
        patience=p.int_min(ed["patience"], "early_stopping.patience", 0),
    )

    tp_raw = p.mapping({"threshold_policy": get("threshold_policy")}, "threshold_policy", PIPELINE_SECTION)
    p.unknown_keys(tp_raw, d["threshold_policy"], f"{PIPELINE_SECTION}.threshold_policy")
    td = {**d["threshold_policy"], **tp_raw}
    kind = TextThresholdPolicy.try_from(td["kind"])
    if kind is None:
        p.problems.append(
            _("threshold_policy.kind must be one of: {choices} (got {v!r})").format(
                choices=", ".join(k.value for k in TextThresholdPolicy), v=td["kind"]
            )
        )
        kind = TextThresholdPolicy.PRECISION_TARGET
    threshold_policy = TextThresholdConfig(
        kind=kind,
        value=p.float_range(td["value"], "threshold_policy.value", 0.0, 1.0, lo_inclusive=False),
    )

    rep_raw = p.mapping({"reports": get("reports")}, "reports", PIPELINE_SECTION)
    p.unknown_keys(rep_raw, d["reports"], f"{PIPELINE_SECTION}.reports")
    rd = {**d["reports"], **rep_raw}
    reports = TextReportsConfig(
        review_queue_size=p.int_min(rd["review_queue_size"], "reports.review_queue_size", 0),
        rescore_unlabeled_cut=p.boolean(rd["rescore_unlabeled_cut"], "reports.rescore_unlabeled_cut"),
    )

    notes = get("model_card_notes")
    if notes is None:
        notes = ""
    if not isinstance(notes, str):
        p.problems.append(_("model_card_notes must be a string"))
        notes = ""

    backend_options = get("backend_options")
    if backend_options is None:
        backend_options = {}
    if not isinstance(backend_options, Mapping):
        p.problems.append(_("backend_options must be a mapping"))
        backend_options = {}

    cfg = TextRunConfig(
        data_dir=p.path(get("data_dir"), "data_dir"),
        runs_dir=p.path(get("runs_dir"), "runs_dir"),
        backend=backend,
        model_id=p.optional_str(get("model_id"), "model_id"),
        seed=p.int_min(get("seed"), "seed", 0),
        max_length=p.int_min(get("max_length"), "max_length", 1),
        class_weight=_parse_class_weight(p, get("class_weight")),
        tier_weight=_parse_tier_weight(p, get("tier_weight")),
        exclude_conflicts=p.boolean(get("exclude_conflicts"), "exclude_conflicts"),
        subsample_keep_unreviewed=p.optional_positive_float(
            get("subsample_keep_unreviewed"), "subsample_keep_unreviewed"
        ),
        gold_tiers=_parse_gold_tiers(p, get("gold_tiers")),
        optim=optim,
        early_stopping=early_stopping,
        calibration=calibration,
        threshold_policy=threshold_policy,
        reports=reports,
        model_card_notes=notes,
        device=p.optional_str(get("device"), "device"),
        backend_options=copy.deepcopy(dict(backend_options)),
    )
    if p.problems:
        raise TextConfigError(p.problems)
    return cfg


def apply_text_overrides(section: Mapping[str, Any], overrides: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Return a copy of *section* with *overrides* applied.

    Override keys are section keys, or ``optim.<key>`` for the optimizer block. ``None``
    values mean "not overridden" and are skipped, matching how CLI flags default.
    """
    out = copy.deepcopy(dict(section))
    for key, value in overrides.items():
        if value is None:
            continue
        if "." in key:
            head, tail = key.split(".", 1)
            sub = out.get(head)
            sub = dict(sub) if isinstance(sub, Mapping) else {}
            sub[tail] = value
            out[head] = sub
        else:
            out[key] = value
    return out


def resolve_text_run_config(pipeline: Any, overrides: Optional[Mapping[str, Any]] = None) -> TextRunConfig:
    """
    Parse the pipeline's ``text_classification`` section with *overrides* applied.

    *pipeline* is a :class:`~mb.pipeline_config.PipelineConfig` (anything with ``get``).
    """
    section = pipeline.get(PIPELINE_SECTION)
    if section is None:
        section = copy.deepcopy(TEXT_CLASSIFICATION_DEFAULTS)
    if not isinstance(section, Mapping):
        raise TextConfigError([_("text_classification must be a mapping")])
    return parse_text_run_config(apply_text_overrides(section, overrides or {}))
