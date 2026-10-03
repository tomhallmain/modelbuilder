"""
Text-classification dataset: reading, integrity checks, training selection and weights.

Layout of a dataset directory (only ``dataset.tsv`` is required)::

    dataset.tsv          text, label (0/1), split (train/val/test); optional tier, source,
                         category_hint, group, stage7_score
    manifest.json        expected sha256/bytes per file and expected counts
    label_conflicts.tsv  ``group`` column: groups whose variants carry both labels
    unlabeled_cut.tsv    ``text`` column (optional stage7_score): unlabeled, never trained on

TSVs are UTF-8 with a header row and no quoting or escaping, so lines are split on ``\\n``
and fields on ``\\t`` only. ``csv`` and ``str.splitlines`` would both be wrong here: texts
include ``"``, ``null``/``NA`` and characters such as U+2028 that ``splitlines`` treats as
line breaks. A trailing ``\\r`` is dropped so files re-saved with CRLF endings (e.g. by a
Windows editor) still read; texts never contain CR.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from mb.cancellation import check_cancel_event
from mb.utils.logging_setup import get_logger
from mb.utils.translations import _

logger = get_logger(__name__)

DATASET_FILE = "dataset.tsv"
MANIFEST_FILE = "manifest.json"
CONFLICTS_FILE = "label_conflicts.tsv"
UNLABELED_CUT_FILE = "unlabeled_cut.tsv"

COL_TEXT = "text"
COL_LABEL = "label"
COL_SPLIT = "split"
COL_TIER = "tier"
COL_SOURCE = "source"
COL_CATEGORY_HINT = "category_hint"
COL_GROUP = "group"
COL_STAGE7 = "stage7_score"

REQUIRED_COLUMNS = (COL_TEXT, COL_LABEL, COL_SPLIT)
SPLITS = ("train", "val", "test")

# Tier whose rows ``subsample_keep_unreviewed`` thins out and the review queue draws from.
KEEP_UNREVIEWED_TIER = "keep_unreviewed"


class TextDatasetError(ValueError):
    """The dataset directory or one of its files cannot be used."""


def _strip_line_end(line: str) -> str:
    if line.endswith("\n"):
        line = line[:-1]
    if line.endswith("\r"):
        line = line[:-1]
    return line


def iter_tsv(path: Path) -> Iterator[Tuple[int, List[str]]]:
    """
    Yield ``(line_number, fields)`` for every line of *path*, header included.

    A trailing newline at end of file does not produce an empty final row.
    """
    with open(path, "r", encoding="utf-8", newline="\n") as f:
        for lineno, line in enumerate(f, start=1):
            yield lineno, _strip_line_end(line).split("\t")


def read_tsv_header(path: Path) -> List[str]:
    for _lineno, fields in iter_tsv(path):
        return fields
    raise TextDatasetError(_("{path} is empty (expected a header row)").format(path=path))


def read_tsv_column(path: Path, column: str) -> List[str]:
    """All values of one column; raises when the column or a field is missing."""
    rows = iter_tsv(path)
    try:
        _lineno, header = next(rows)
    except StopIteration:
        raise TextDatasetError(_("{path} is empty (expected a header row)").format(path=path)) from None
    if column not in header:
        raise TextDatasetError(
            _("{path} has no '{col}' column (header: {header})").format(
                path=path, col=column, header=", ".join(header)
            )
        )
    ci = header.index(column)
    out: List[str] = []
    for lineno, fields in rows:
        if len(fields) != len(header):
            raise TextDatasetError(
                _("{path}:{line}: expected {n} fields, found {m}").format(
                    path=path, line=lineno, n=len(header), m=len(fields)
                )
            )
        out.append(fields[ci])
    return out


def read_text_lines(path: Path) -> List[str]:
    """One string per line (LF or CRLF), in order; a final newline adds no line."""
    with open(path, "r", encoding="utf-8", newline="\n") as f:
        return [_strip_line_end(line) for line in f]


@dataclass
class Categorical:
    """A string column stored as integer codes into ``categories``."""

    codes: np.ndarray
    categories: List[str]

    def value(self, i: int) -> str:
        return self.categories[int(self.codes[i])]

    def mask(self, names: Iterable[str]) -> np.ndarray:
        wanted = {self.categories.index(n) for n in names if n in self.categories}
        if not wanted:
            return np.zeros(len(self.codes), dtype=bool)
        return np.isin(self.codes, np.fromiter(wanted, dtype=self.codes.dtype))

    def code_of(self, name: str) -> Optional[int]:
        return self.categories.index(name) if name in self.categories else None


class _CategoricalBuilder:
    def __init__(self) -> None:
        self._index: Dict[str, int] = {}
        self.categories: List[str] = []
        self.codes: List[int] = []

    def add(self, value: str) -> None:
        code = self._index.get(value)
        if code is None:
            code = len(self.categories)
            self._index[value] = code
            self.categories.append(value)
        self.codes.append(code)

    def build(self) -> Categorical:
        return Categorical(np.asarray(self.codes, dtype=np.int32), list(self.categories))


@dataclass
class TextDataset:
    """All rows of ``dataset.tsv``; per-split views are index arrays into it."""

    data_dir: Path
    texts: List[str]
    labels: np.ndarray
    split: Categorical
    tier: Optional[Categorical] = None
    source: Optional[Categorical] = None
    category_hint: Optional[Categorical] = None
    stage7_score: Optional[np.ndarray] = None
    # True for rows whose group is listed in label_conflicts.tsv.
    conflict: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    # Kept only until verification; see :meth:`drop_groups`.
    groups: Optional[List[str]] = None
    has_group_column: bool = False

    def __len__(self) -> int:
        return len(self.texts)

    def split_indices(self, name: str, *, exclude_conflicts: bool) -> np.ndarray:
        mask = self.split.mask([name])
        if exclude_conflicts:
            mask &= ~self.conflict
        return np.flatnonzero(mask)

    def tier_of(self, i: int) -> str:
        return self.tier.value(i) if self.tier is not None else ""

    def drop_groups(self) -> None:
        """Free the group strings once verification no longer needs them."""
        self.groups = None


def load_conflict_groups(data_dir: Path) -> Set[str]:
    path = Path(data_dir) / CONFLICTS_FILE
    if not path.is_file():
        return set()
    return set(read_tsv_column(path, COL_GROUP))


def load_text_dataset(
    data_dir: Path,
    *,
    keep_groups: bool = True,
    cancel_event=None,
) -> TextDataset:
    """
    Read ``dataset.tsv`` from *data_dir*.

    Raises:
        TextDatasetError: missing file/columns, a malformed row, or an invalid label/split.
    """
    data_dir = Path(data_dir)
    path = data_dir / DATASET_FILE
    if not path.is_file():
        raise TextDatasetError(_("Dataset file not found: {path}").format(path=path))

    rows = iter_tsv(path)
    try:
        _lineno, header = next(rows)
    except StopIteration:
        raise TextDatasetError(_("{path} is empty (expected a header row)").format(path=path)) from None
    missing = [c for c in REQUIRED_COLUMNS if c not in header]
    if missing:
        raise TextDatasetError(
            _("{path} is missing required column(s): {cols}").format(path=path, cols=", ".join(missing))
        )
    col = {name: i for i, name in enumerate(header)}
    i_text, i_label, i_split = col[COL_TEXT], col[COL_LABEL], col[COL_SPLIT]
    i_tier, i_source = col.get(COL_TIER), col.get(COL_SOURCE)
    i_hint, i_group, i_s7 = col.get(COL_CATEGORY_HINT), col.get(COL_GROUP), col.get(COL_STAGE7)

    conflict_groups = load_conflict_groups(data_dir)
    if conflict_groups and i_group is None:
        raise TextDatasetError(
            _("{conflicts} exists but {path} has no '{col}' column to match it against").format(
                conflicts=CONFLICTS_FILE, path=path, col=COL_GROUP
            )
        )

    texts: List[str] = []
    labels: List[int] = []
    groups: List[str] = []
    conflict: List[bool] = []
    s7: List[float] = []
    split_b, tier_b, source_b, hint_b = (
        _CategoricalBuilder(),
        _CategoricalBuilder(),
        _CategoricalBuilder(),
        _CategoricalBuilder(),
    )
    n_fields = len(header)
    for lineno, fields in rows:
        if lineno % 200_000 == 0:
            check_cancel_event(cancel_event)
        if len(fields) != n_fields:
            raise TextDatasetError(
                _("{path}:{line}: expected {n} fields, found {m}").format(
                    path=path, line=lineno, n=n_fields, m=len(fields)
                )
            )
        label = fields[i_label]
        if label == "1":
            labels.append(1)
        elif label == "0":
            labels.append(0)
        else:
            raise TextDatasetError(
                _("{path}:{line}: label must be 0 or 1 (got {v!r})").format(path=path, line=lineno, v=label)
            )
        split = fields[i_split]
        if split not in SPLITS:
            raise TextDatasetError(
                _("{path}:{line}: split must be one of {choices} (got {v!r})").format(
                    path=path, line=lineno, choices=", ".join(SPLITS), v=split
                )
            )
        texts.append(fields[i_text])
        split_b.add(split)
        if i_tier is not None:
            tier_b.add(fields[i_tier])
        if i_source is not None:
            source_b.add(fields[i_source])
        if i_hint is not None:
            hint_b.add(fields[i_hint])
        if i_group is not None:
            g = fields[i_group]
            conflict.append(g in conflict_groups)
            if keep_groups:
                groups.append(g)
        if i_s7 is not None:
            raw = fields[i_s7]
            try:
                s7.append(float(raw) if raw else float("nan"))
            except ValueError:
                raise TextDatasetError(
                    _("{path}:{line}: {col} must be a number or empty (got {v!r})").format(
                        path=path, line=lineno, col=COL_STAGE7, v=raw
                    )
                ) from None

    n = len(texts)
    if n == 0:
        raise TextDatasetError(_("{path} has no data rows").format(path=path))
    ds = TextDataset(
        data_dir=data_dir,
        texts=texts,
        labels=np.asarray(labels, dtype=np.int8),
        split=split_b.build(),
        tier=tier_b.build() if i_tier is not None else None,
        source=source_b.build() if i_source is not None else None,
        category_hint=hint_b.build() if i_hint is not None else None,
        stage7_score=np.asarray(s7, dtype=np.float32) if i_s7 is not None else None,
        conflict=np.asarray(conflict, dtype=bool) if i_group is not None else np.zeros(n, dtype=bool),
        groups=groups if (keep_groups and i_group is not None) else None,
        has_group_column=i_group is not None,
    )
    logger.info("Loaded %d rows from %s (%d in conflict groups)", n, path, int(ds.conflict.sum()))
    return ds


def load_unlabeled_cut(data_dir: Path) -> Optional[Tuple[List[str], Optional[np.ndarray]]]:
    """``(texts, stage7_scores or None)`` from ``unlabeled_cut.tsv``, or ``None`` when absent."""
    path = Path(data_dir) / UNLABELED_CUT_FILE
    if not path.is_file():
        return None
    header = read_tsv_header(path)
    texts = read_tsv_column(path, COL_TEXT)
    scores = None
    if COL_STAGE7 in header:
        raw = read_tsv_column(path, COL_STAGE7)
        scores = np.asarray([float(v) if v else float("nan") for v in raw], dtype=np.float32)
    return texts, scores


# --- verification (integrity checks that must pass before any training) ---


@dataclass
class TextVerifyCheck:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class TextVerifyReport:
    data_dir: Path
    checks: List[TextVerifyCheck] = field(default_factory=list)
    file_sha256: Dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.checks.append(TextVerifyCheck(name, ok, detail))

    def format(self) -> str:
        lines = [_("Dataset: {path}").format(path=self.data_dir)]
        for c in self.checks:
            mark = "ok  " if c.ok else "FAIL"
            lines.append(f"[{mark}] {c.name}" + (f": {c.detail}" if c.detail else ""))
        lines.append(_("Result: {r}").format(r=_("PASS") if self.ok else _("FAIL")))
        return "\n".join(lines)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _count_tsv_rows(path: Path) -> int:
    return sum(1 for _ in iter_tsv(path)) - 1


def _compare_counts(report: TextVerifyReport, name: str, expected: Mapping, actual: Mapping) -> None:
    exp = {str(k): v for k, v in expected.items()}
    act = {str(k): v for k, v in actual.items()}
    diffs = [
        f"{k}: {_('expected')} {exp.get(k, 0)}, {_('found')} {act.get(k, 0)}"
        for k in sorted(set(exp) | set(act))
        if exp.get(k, 0) != act.get(k, 0)
    ]
    report.add(name, not diffs, "; ".join(diffs))


def _value_counts(cat: Categorical, *, skip_empty: bool = False) -> Dict[str, int]:
    counts = np.bincount(cat.codes, minlength=len(cat.categories))
    return {
        name: int(n)
        for name, n in zip(cat.categories, counts)
        if n and not (skip_empty and name == "")
    }


def verify_text_dataset(
    data_dir: Path,
    *,
    dataset: Optional[TextDataset] = None,
    cancel_event=None,
    progress: Optional[Callable[[str], None]] = None,
) -> TextVerifyReport:
    """
    Integrity checks for a dataset directory.

    With ``manifest.json``: every listed file's sha256 and size, then row/label/tier/
    source/category-hint/split/group counts. Always: ``text`` values unique, and each
    ``group`` in exactly one split. Pass an already loaded *dataset* (with groups) to avoid
    reading ``dataset.tsv`` twice.
    """
    data_dir = Path(data_dir)
    report = TextVerifyReport(data_dir=data_dir)
    say = progress or (lambda _m: None)

    manifest: Optional[dict] = None
    manifest_path = data_dir / MANIFEST_FILE
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            report.add(MANIFEST_FILE, False, str(e))
            return report
    else:
        report.add(
            MANIFEST_FILE,
            True,
            _("not present; hash and count checks skipped"),
        )

    if manifest is not None:
        files = manifest.get("files") or {}
        for name, meta in files.items():
            check_cancel_event(cancel_event)
            say(_("Hashing {name}…").format(name=name))
            path = data_dir / name
            if not path.is_file():
                report.add(f"sha256 {name}", False, _("file missing"))
                continue
            digest = sha256_file(path)
            report.file_sha256[name] = digest
            want = str((meta or {}).get("sha256", ""))
            ok = digest == want
            detail = "" if ok else f"{_('expected')} {want}, {_('found')} {digest}"
            want_bytes = (meta or {}).get("bytes")
            if want_bytes is not None and path.stat().st_size != int(want_bytes):
                ok = False
                detail = (detail + "; " if detail else "") + _("size {a} != {b}").format(
                    a=path.stat().st_size, b=want_bytes
                )
            report.add(f"sha256 {name}", ok, detail)
        if not report.ok:
            # Counts from a file whose bytes are wrong say nothing useful.
            return report

    if dataset is None or (dataset.groups is None and dataset.has_group_column):
        say(_("Reading {name}…").format(name=DATASET_FILE))
        try:
            dataset = load_text_dataset(data_dir, keep_groups=True, cancel_event=cancel_event)
        except TextDatasetError as e:
            report.add(DATASET_FILE, False, str(e))
            return report
    ds = dataset
    say(_("Checking counts…"))

    if manifest is not None:
        if "dataset_rows" in manifest:
            report.add(
                _("row count"),
                len(ds) == int(manifest["dataset_rows"]),
                f"{_('expected')} {manifest['dataset_rows']}, {_('found')} {len(ds)}",
            )
        if "labels" in manifest:
            n1 = int(ds.labels.sum())
            _compare_counts(report, _("label counts"), manifest["labels"], {"0": len(ds) - n1, "1": n1})
        if "tiers" in manifest and ds.tier is not None:
            _compare_counts(report, _("tier counts"), manifest["tiers"], _value_counts(ds.tier))
        if "sources" in manifest and ds.source is not None:
            _compare_counts(report, _("source counts"), manifest["sources"], _value_counts(ds.source))
        if "category_hints" in manifest and ds.category_hint is not None:
            _compare_counts(
                report,
                _("category hint counts"),
                manifest["category_hints"],
                _value_counts(ds.category_hint, skip_empty=True),
            )
        if "splits" in manifest:
            actual: Dict[str, Dict[str, int]] = {}
            for s in SPLITS:
                idx = ds.split_indices(s, exclude_conflicts=False)
                n1 = int(ds.labels[idx].sum())
                actual[s] = {"rows": len(idx), "label_1": n1, "label_0": len(idx) - n1}
            flat_exp = {
                f"{s}.{k}": v for s, d in manifest["splits"].items() for k, v in (d or {}).items()
            }
            flat_act = {f"{s}.{k}": v for s, d in actual.items() for k, v in d.items()}
            _compare_counts(report, _("split counts"), flat_exp, flat_act)
        if "groups" in manifest and ds.groups is not None:
            n_groups = len(set(ds.groups))
            report.add(
                _("group count"),
                n_groups == int(manifest["groups"]),
                f"{_('expected')} {manifest['groups']}, {_('found')} {n_groups}",
            )
        if "conflict_groups" in manifest and (data_dir / CONFLICTS_FILE).is_file():
            n_conf = len(load_conflict_groups(data_dir))
            report.add(
                _("conflict group count"),
                n_conf == int(manifest["conflict_groups"]),
                f"{_('expected')} {manifest['conflict_groups']}, {_('found')} {n_conf}",
            )
        if "unlabeled_cut" in manifest and (data_dir / UNLABELED_CUT_FILE).is_file():
            n_cut = _count_tsv_rows(data_dir / UNLABELED_CUT_FILE)
            report.add(
                _("unlabeled cut rows"),
                n_cut == int(manifest["unlabeled_cut"]),
                f"{_('expected')} {manifest['unlabeled_cut']}, {_('found')} {n_cut}",
            )

    check_cancel_event(cancel_event)
    say(_("Checking uniqueness…"))
    seen: Set[str] = set()
    dupes: List[str] = []
    for t in ds.texts:
        if t in seen:
            if len(dupes) < 5:
                dupes.append(t)
            else:
                break
        seen.add(t)
    del seen
    report.add(
        _("text values unique"),
        not dupes,
        _("duplicates include: {d}").format(d=", ".join(repr(t) for t in dupes)) if dupes else "",
    )

    if ds.groups is not None:
        check_cancel_event(cancel_event)
        group_split: Dict[str, int] = {}
        leaks: List[str] = []
        codes = ds.split.codes
        for i, g in enumerate(ds.groups):
            s = int(codes[i])
            prev = group_split.setdefault(g, s)
            if prev != s and len(leaks) < 5:
                leaks.append(g)
        del group_split
        report.add(
            _("each group in one split"),
            not leaks,
            _("groups in several splits include: {g}").format(g=", ".join(repr(g) for g in leaks))
            if leaks
            else "",
        )
    else:
        report.add(_("each group in one split"), True, _("no '{col}' column; skipped").format(col=COL_GROUP))
    return report


# --- training selection and weights ---


@dataclass
class SubsampleInfo:
    """What ``subsample_keep_unreviewed`` did; recorded in the model card."""

    ratio: float
    seed: int
    available: int
    kept: int


def select_training_indices(
    ds: TextDataset,
    *,
    exclude_conflicts: bool,
    subsample_keep_unreviewed: Optional[float],
    seed: int,
) -> Tuple[np.ndarray, Optional[SubsampleInfo]]:
    """
    Train-split rows to fit on: conflicts dropped when asked, then (optionally) all
    positives and other tiers kept with ``keep_unreviewed`` thinned to *ratio* × positives.
    """
    idx = ds.split_indices("train", exclude_conflicts=exclude_conflicts)
    if subsample_keep_unreviewed is None:
        return idx, None
    if ds.tier is None:
        raise TextDatasetError(
            _("subsample_keep_unreviewed needs a '{col}' column in {file}").format(
                col=COL_TIER, file=DATASET_FILE
            )
        )
    code = ds.tier.code_of(KEEP_UNREVIEWED_TIER)
    if code is None:
        raise TextDatasetError(
            _("subsample_keep_unreviewed is set but no rows have tier '{tier}'").format(
                tier=KEEP_UNREVIEWED_TIER
            )
        )
    is_ku = ds.tier.codes[idx] == code
    pool = idx[is_ku]
    n_pos = int(ds.labels[idx].sum())
    n_keep = min(len(pool), int(round(subsample_keep_unreviewed * n_pos)))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(pool, size=n_keep, replace=False) if n_keep < len(pool) else pool
    out = np.sort(np.concatenate([idx[~is_ku], chosen]))
    info = SubsampleInfo(
        ratio=float(subsample_keep_unreviewed), seed=seed, available=len(pool), kept=int(n_keep)
    )
    return out, info


def compute_sample_weights(
    ds: TextDataset,
    idx: np.ndarray,
    *,
    class_weight,
    tier_weight: Mapping[str, float],
) -> Tuple[np.ndarray, Dict[int, float]]:
    """
    Per-row loss weight = class weight × tier weight, for rows *idx*.

    *class_weight* is ``"none"``, ``"balanced"`` (``n / (2 · n_c)`` over *idx*) or
    ``{0: w0, 1: w1}``. Returns the weights and the class weights that were used.
    """
    labels = ds.labels[idx]
    if isinstance(class_weight, Mapping):
        cw = {0: float(class_weight[0]), 1: float(class_weight[1])}
    elif class_weight == "balanced":
        n = len(labels)
        n1 = int(labels.sum())
        n0 = n - n1
        if n0 == 0 or n1 == 0:
            raise TextDatasetError(_("The training rows contain only one label; cannot train."))
        cw = {0: n / (2.0 * n0), 1: n / (2.0 * n1)}
    else:
        cw = {0: 1.0, 1: 1.0}
    w = np.where(labels == 1, cw[1], cw[0]).astype(np.float32)

    if tier_weight:
        if ds.tier is None:
            raise TextDatasetError(
                _("tier_weight is set but {file} has no '{col}' column").format(file=DATASET_FILE, col=COL_TIER)
            )
        unknown = sorted(t for t in tier_weight if t not in ds.tier.categories)
        if unknown:
            raise TextDatasetError(
                _("tier_weight names unknown tier(s): {bad} (known: {known})").format(
                    bad=", ".join(unknown), known=", ".join(ds.tier.categories)
                )
            )
        per_code = np.ones(len(ds.tier.categories), dtype=np.float32)
        for name, mult in tier_weight.items():
            per_code[ds.tier.categories.index(name)] = float(mult)
        w = w * per_code[ds.tier.codes[idx]]
    return w, cw


def gold_mask(ds: TextDataset, idx: np.ndarray, gold_tiers: Sequence[str]) -> np.ndarray:
    """
    Which rows of *idx* belong to the gold set.

    Without a tier column every row counts as gold (there is no other quality signal).
    """
    if ds.tier is None:
        return np.ones(len(idx), dtype=bool)
    unknown = [t for t in gold_tiers if t not in ds.tier.categories]
    if unknown:
        raise TextDatasetError(
            _("gold_tiers names unknown tier(s): {bad} (known: {known})").format(
                bad=", ".join(unknown), known=", ".join(ds.tier.categories)
            )
        )
    return ds.tier.mask(gold_tiers)[idx]
